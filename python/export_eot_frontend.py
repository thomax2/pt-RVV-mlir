#!/usr/bin/env python3
"""Export the fixed-shape GM v33 EOT frontend to PT2 and RAW Torch MLIR."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pt_model.neural_gm_splitcov_v33 import build_default_model_v33

EOT_NAMES = (
    "point_features", "masked_mean", "masked_variance", "masked_max",
    "masked_min", "gru_cell", "gm_parameterize",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_parts(value: Any) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint must be a state_dict or a mapping containing one")
    for config_key in ("model_config", "config", "cfg"):
        config = value.get(config_key)
        if is_dataclass(config):
            metadata = asdict(config)
        elif isinstance(config, Mapping):
            metadata = dict(config)
        if metadata:
            break
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            return candidate, metadata
    if value and all(isinstance(key, str) and isinstance(tensor, torch.Tensor)
                     for key, tensor in value.items()):
        return value, metadata
    raise TypeError("checkpoint has no model_state_dict/state_dict/model tensor mapping")


def _normalize_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state)
    if keys and all(key.startswith("module.") for key in keys):
        print("checkpoint: stripping explicit 'module.' prefix from every key")
        return {key[len("module."):]: value for key, value in state.items()}
    if any(key.startswith("module.") for key in keys):
        raise ValueError("checkpoint mixes prefixed and unprefixed state_dict keys")
    return dict(state)


def _target_name(node: torch.fx.Node) -> str:
    return str(node.target)


def _check_eot_nodes(exported_program: torch.export.ExportedProgram) -> Counter[str]:
    targets = Counter(_target_name(node) for node in exported_program.graph.nodes
                      if node.op == "call_function")
    missing = [name for name in EOT_NAMES
               if not any((f"eot.{name}" in target or f"eot::{name}" in target)
                          for target in targets)]
    if missing:
        raise RuntimeError("strict export expanded or omitted EOT nodes: " + ", ".join(missing))
    return targets


def _import_raw_torch_mlir(exported_program: torch.export.ExportedProgram) -> tuple[str, str]:
    try:
        import torch_mlir
        from torch_mlir import fx
        from torch_mlir.compiler_utils import OutputType
    except ImportError as exc:
        raise RuntimeError(
            "torch-mlir is required to produce gm_frontend_torch.mlir; "
            "set PYTHON to the interpreter used by the matching torch-mlir build") from exc

    # 转换为 MLIR
    module = fx.export_and_import(
        exported_program,
        output_type=OutputType.RAW,
        func_name="gm_frontend",
        enable_graph_printing=True,
    )
    version = getattr(torch_mlir, "__version__", "unknown")
    return str(module), str(version)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--nmax", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict", action="store_true",
                        help="compatibility flag; export is always strict")
    parser.add_argument("--save_reference", action="store_true")
    parser.add_argument("--print_graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not hasattr(torch, "export") or not hasattr(torch.library, "custom_op"):
        raise RuntimeError(
            "GM-EOT export requires a modern PyTorch with torch.export and "
            "torch.library.custom_op")
    from python.eot_frontend_export import EotFrontendExportModule
    if args.device != "cpu":
        raise ValueError("the first EOT deployment exporter supports --device cpu only")
    if args.batch_size != 1:
        raise ValueError("the first EOT deployment ABI requires --batch_size 1")
    if args.nmax < 1:
        raise ValueError("--nmax must be positive")
    checkpoint_path = args.checkpoint.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 从 checkpoint 中提取 state_dict 和模型配置
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, checkpoint_config = _checkpoint_parts(checkpoint)
    if checkpoint_config.get("use_pi_gru", False):
        raise ValueError("checkpoint enables Pi-GRU, unsupported by the first EOT frontend ABI")
    config_keys = {
        "hidden_dim", "state_dim", "front_point_dim", "use_raw_stats",
        "use_pi_gru", "use_c_gru", "n_ref", "c_min", "c_max", "tau_min",
        "tau_max", "alpha_base", "logvar_min", "logvar_max",
    }
    # 根据配置重建模型
    model = build_default_model_v33(**{
        key: value for key, value in checkpoint_config.items() if key in config_keys
    })
    # 加载权重
    model.load_state_dict(_normalize_state_dict(state_dict), strict=True)
    model.eval()
    # 包装为导出友好的接口
    wrapper = EotFrontendExportModule(model).eval()

    # 创建具有代表性的测试输入
    generator = torch.Generator(device="cpu").manual_seed(20260720)
    points = torch.randn(args.batch_size, args.nmax, 2, generator=generator)
    sensor_view = torch.randn(args.batch_size, 2, generator=generator)
    state_feat = torch.randn(args.batch_size, 12, generator=generator)
    mask = torch.ones(args.batch_size, args.nmax, dtype=torch.bool)
    c_hidden_prev = torch.randn(args.batch_size, 128, generator=generator)
    example_inputs = (points, sensor_view, state_feat, mask, c_hidden_prev)

    # 执行模型以获得参考输出（用于后续数值验证）
    with torch.no_grad():
        reference_outputs = wrapper(*example_inputs)

    # 核心导出步骤：使用 torch.export 导出计算图
    # strict=True 要求导出的计算图严格匹配输入输出，禁止隐式的控制流或动态 shape
    exported_program = torch.export.export(wrapper, example_inputs, strict=True)
    targets = _check_eot_nodes(exported_program)
    graph_text = str(exported_program.graph_module.graph)
    if args.print_graph:
        print(graph_text)

    # 标准化模型归档，可被 torch.export.load 加载
    torch.export.save(exported_program, args.output_dir / "gm_frontend.pt2")
    (args.output_dir / "gm_frontend_fx.txt").write_text(graph_text, encoding="utf-8")
    aten_targets = sorted(target for target in targets
                          if "eot." not in target and "eot::" not in target)
    (args.output_dir / "unsupported_aten_report.txt").write_text(
        "ATen/operator inventory before the real Torch-to-TOSA legality run:\n" +
        "\n".join(f"{targets[target]:4d} {target}" for target in aten_targets) + "\n",
        encoding="utf-8")
    if args.save_reference:
        torch.save(example_inputs, args.output_dir / "reference_inputs.pt")
        torch.save(reference_outputs, args.output_dir / "reference_outputs.pt")

    # 原始 Torch MLIR, MLIR 编译流水线的起点
    raw_mlir, torch_mlir_version = _import_raw_torch_mlir(exported_program)
    (args.output_dir / "gm_frontend_torch.mlir").write_text(raw_mlir, encoding="utf-8")
    manifest = {
        "pytorch_version": torch.__version__,
        "torch_mlir_version": torch_mlir_version,
        "llvm_mlir_version": os.environ.get("MLIR_VERSION", "unknown"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_config": asdict(model.cfg),
        "nmax": args.nmax,
        "batch_size": args.batch_size,
        "input_shapes": [list(tensor.shape) for tensor in example_inputs],
        "output_shapes": [list(tensor.shape) for tensor in reference_outputs],
        "custom_op_schema_version": 1,
        "strict_export": True,
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

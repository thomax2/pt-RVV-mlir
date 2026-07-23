#!/usr/bin/env python3
"""Verify eager/PT2 equivalence and the required opaque EOT FX nodes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# torch.export.load resolves custom operator targets through the dispatcher.
# Register the EOT schemas before deserializing the exported program.
from python import eot_export_ops as _eot_export_ops  # noqa: E402,F401

REQUIRED = (
    "point_features", "masked_mean", "masked_variance", "masked_max",
    "masked_min", "gru_cell", "gm_parameterize",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    exported = torch.export.load(args.output_dir / "gm_frontend.pt2")
    graph = str(exported.graph_module.graph)
    missing = [name for name in REQUIRED
               if f"eot.{name}" not in graph and f"eot::{name}" not in graph]
    if missing:
        raise AssertionError("missing opaque EOT FX nodes: " + ", ".join(missing))
    inputs = torch.load(args.output_dir / "reference_inputs.pt", map_location="cpu",
                        weights_only=False)
    expected = torch.load(args.output_dir / "reference_outputs.pt", map_location="cpu",
                          weights_only=False)
    with torch.no_grad():
        actual = exported.module()(*inputs)
    for index, (lhs, rhs) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(lhs, rhs, atol=args.atol, rtol=args.rtol,
                                   msg=lambda message: f"output {index}: {message}")
    print("verified PT2 outputs and all seven opaque EOT nodes")


if __name__ == "__main__":
    main()

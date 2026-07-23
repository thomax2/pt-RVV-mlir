#!/usr/bin/env bash
set -euo pipefail

export_dir=${1:-${EXPORT_DIR:-eot-export-output}}
lowering_dir=${2:-${LOWERING_DIR:-eot-lowering-output}}
python_bin=${PYTHON:-python3}
repo_dir=$(cd "$(dirname "$0")/.." && pwd)

"${python_bin}" "${repo_dir}/python/verify_eot_export.py" --output_dir "${export_dir}"
rg -q 'torch\.operator.*eot|eot::' "${lowering_dir}/01_torch.mlir"
rg -q 'torch\.aten' "${lowering_dir}/01_torch.mlir"
! rg -q 'torch\.operator.*eot|eot::' "${lowering_dir}/02_tosa_custom.mlir"
! rg -q 'torch\.aten' "${lowering_dir}/02_tosa_custom.mlir"
rg -q 'tosa\.custom.*domain_name = "eot"|domain_name = "eot".*tosa\.custom' \
  "${lowering_dir}/02_tosa_custom.mlir"
! rg -q 'tosa\.custom.*domain_name = "eot"|domain_name = "eot".*tosa\.custom' \
  "${lowering_dir}/03_tosa_eot.mlir"
rg -q 'eot\.' "${lowering_dir}/03_tosa_eot.mlir"
! rg -q 'eot\.' "${lowering_dir}/04_standard_tensor.mlir"
rg -q 'linalg\.|tensor\.|arith\.|math\.' "${lowering_dir}/04_standard_tensor.mlir"
! rg -q 'tosa\.' "${lowering_dir}/05_tosa_lowered.mlir"
rg -q 'linalg\.|tensor\.|arith\.|math\.|scf\.' \
  "${lowering_dir}/05_tosa_lowered.mlir"
! rg -q 'tensor\.|tensor<|bufferization\.' \
  "${lowering_dir}/06_bufferized.mlir"
! rg -q 'linalg\.' "${lowering_dir}/07_loops.mlir"
! rg -q 'memref\.(alloc|alloca|dealloc)' \
  "${lowering_dir}/08_workspace.mlir"
! rg -q 'tensor<' "${lowering_dir}/08_workspace.mlir"
! rg -q 'torch\.|tosa\.|eot\.|linalg\.|tensor\.|bufferization\.|scf\.|cf\.|math\.|arith\.|index\.|memref\.|func\.' \
  "${lowering_dir}/09_llvm_dialect.mlir"
rg -q 'llvm\.' "${lowering_dir}/09_llvm_dialect.mlir"
test -s "${lowering_dir}/10_model.ll"
echo "verified EOT export and all lowering stage boundaries"

#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:-${OUTPUT_DIR:-eot-lowering-output}}
python_bin=${PYTHON:-python3}
repo_dir=$(cd "$(dirname "$0")/.." && pwd)

"${python_bin}" "${repo_dir}/python/verify_eot_export.py" --output_dir "${output_dir}"
rg -q 'torch\.operator.*eot|eot::' "${output_dir}/01_torch.mlir"
rg -q 'torch\.aten' "${output_dir}/01_torch.mlir"
! rg -q 'torch\.operator.*eot|eot::' "${output_dir}/02_tosa_custom.mlir"
! rg -q 'torch\.aten' "${output_dir}/02_tosa_custom.mlir"
rg -q 'tosa\.custom.*domain_name = "eot"|domain_name = "eot".*tosa\.custom' \
  "${output_dir}/02_tosa_custom.mlir"
! rg -q 'tosa\.custom.*domain_name = "eot"|domain_name = "eot".*tosa\.custom' \
  "${output_dir}/03_tosa_eot.mlir"
rg -q 'eot\.' "${output_dir}/03_tosa_eot.mlir"
! rg -q 'eot\.' "${output_dir}/04_standard_tensor.mlir"
rg -q 'linalg\.|tensor\.|arith\.|math\.' "${output_dir}/04_standard_tensor.mlir"
! rg -q 'torch\.|tosa\.|eot\.|linalg\.|tensor\.' "${output_dir}/09_llvm_dialect.mlir"
test -s "${output_dir}/10_model.ll"
echo "verified EOT export and all lowering stage boundaries"

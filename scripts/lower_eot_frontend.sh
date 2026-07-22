#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 gm_frontend_torch.mlir [OUTPUT_DIR]" >&2
  exit 2
fi

input=$1
output_dir=${2:-eot-lowering-output}
torch_mlir_build=${TORCH_MLIR_BUILD:?set TORCH_MLIR_BUILD}
mlir_build=${MLIR_BUILD:?set MLIR_BUILD}
project_build=${PROJECT_BUILD:?set PROJECT_BUILD}
torch_mlir_opt=${TORCH_MLIR_OPT:-${torch_mlir_build}/bin/torch-mlir-opt}
npu_opt=${NPU_OPT:-${project_build}/npu-opt}
mlir_translate=${MLIR_TRANSLATE:-${mlir_build}/bin/mlir-translate}
mkdir -p "${output_dir}"
cp "${input}" "${output_dir}/01_torch.mlir"

# The matching torch-mlir build must include the EOT torch.operator pattern in
# its existing Torch-to-TOSA conversion so the same TypeConverter is reused.
"${torch_mlir_opt}" "${output_dir}/01_torch.mlir" \
  --pass-pipeline='builtin.module(torch-backend-to-tosa-backend-pipeline)' \
  -o "${output_dir}/02_tosa_custom.mlir"

"${npu_opt}" "${output_dir}/02_tosa_custom.mlir" \
  --convert-tosa-custom-to-eot \
  -o "${output_dir}/03_tosa_eot.mlir"

"${npu_opt}" "${output_dir}/03_tosa_eot.mlir" \
  --convert-eot-to-standard --canonicalize --cse \
  -o "${output_dir}/04_standard_tensor.mlir"

"${torch_mlir_opt}" "${output_dir}/04_standard_tensor.mlir" \
  --tosa-to-linalg-named --tosa-to-linalg --canonicalize --cse \
  -o "${output_dir}/05_tosa_lowered.mlir"

"${torch_mlir_opt}" "${output_dir}/05_tosa_lowered.mlir" \
  --one-shot-bufferize='bufferize-function-boundaries' \
  -o "${output_dir}/06_bufferized.mlir"

"${torch_mlir_opt}" "${output_dir}/06_bufferized.mlir" \
  --convert-linalg-to-loops \
  -o "${output_dir}/07_loops.mlir"

"${npu_opt}" "${output_dir}/07_loops.mlir" \
  --npu-plan-static-workspace \
  -o "${output_dir}/08_workspace.mlir"

"${torch_mlir_opt}" "${output_dir}/08_workspace.mlir" \
  --expand-strided-metadata --lower-affine --convert-scf-to-cf \
  --convert-math-to-libm --convert-cf-to-llvm --convert-math-to-llvm \
  --convert-arith-to-llvm --convert-index-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  -o "${output_dir}/09_llvm_dialect.mlir"

"${mlir_translate}" --mlir-to-llvmir \
  "${output_dir}/09_llvm_dialect.mlir" -o "${output_dir}/10_model.ll"

cp "${output_dir}/02_tosa_custom.mlir" "${output_dir}/gm_frontend_tosa_custom.mlir"
cp "${output_dir}/03_tosa_eot.mlir" "${output_dir}/gm_frontend_tosa_eot.mlir"
cp "${output_dir}/04_standard_tensor.mlir" "${output_dir}/gm_frontend_standard.mlir"
cp "${output_dir}/06_bufferized.mlir" "${output_dir}/gm_frontend_bufferized.mlir"
cp "${output_dir}/08_workspace.mlir" "${output_dir}/gm_frontend_workspace.mlir"
cp "${output_dir}/09_llvm_dialect.mlir" "${output_dir}/gm_frontend_llvm.mlir"
cp "${output_dir}/10_model.ll" "${output_dir}/gm_frontend.ll"

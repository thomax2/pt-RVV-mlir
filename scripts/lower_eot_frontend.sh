#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 gm_frontend_torch.mlir [OUTPUT_DIR]" >&2
  exit 2
fi

input=$1
output_dir=${2:-eot-lowering-output}
torch_mlir_build=${TORCH_MLIR_BUILD:?set TORCH_MLIR_BUILD}
mlir_build=${MLIR_BUILD:-${torch_mlir_build}}
project_build=${PROJECT_BUILD:?set PROJECT_BUILD}
torch_mlir_opt=${TORCH_MLIR_OPT:-${torch_mlir_build}/bin/torch-mlir-opt}
mlir_opt=${MLIR_OPT:-${mlir_build}/bin/mlir-opt}
eot_opt=${EOT_OPT:-${NPU_OPT:-${project_build}/npu-opt}}
mlir_translate=${MLIR_TRANSLATE:-${mlir_build}/bin/mlir-translate}
target_index_bitwidth=${TARGET_INDEX_BITWIDTH:-32}
llvm_data_layout=${LLVM_DATA_LAYOUT:-}

case "${target_index_bitwidth}" in
  32|64) ;;
  *)
    echo "TARGET_INDEX_BITWIDTH must be 32 or 64" >&2
    exit 2
    ;;
esac

require_no_match() {
  local pattern=$1
  local file=$2
  local description=$3
  if grep -nE "${pattern}" "${file}" >&2; then
    echo "ERROR: ${description}: ${file}" >&2
    exit 1
  fi
}

require_file() {
  local file=$1
  if [[ ! -s "${file}" ]]; then
    echo "ERROR: missing or empty lowering artifact: ${file}" >&2
    exit 1
  fi
}

mkdir -p "${output_dir}"
cp "${input}" "${output_dir}/01_torch.mlir"

# The matching torch-mlir build must include the EOT torch.operator pattern in
# its existing Torch-to-TOSA conversion so the same TypeConverter is reused.
"${torch_mlir_opt}" "${output_dir}/01_torch.mlir" \
  --pass-pipeline='builtin.module(torch-backend-to-tosa-backend-pipeline)' \
  -o "${output_dir}/02_tosa_custom.mlir"

"${eot_opt}" "${output_dir}/02_tosa_custom.mlir" \
  --convert-tosa-custom-to-eot='strict=true' \
  --canonicalize --cse \
  -o "${output_dir}/03_tosa_eot.mlir"
require_no_match 'tosa\.custom' "${output_dir}/03_tosa_eot.mlir" \
  "TOSA custom operations remain after TOSA-custom-to-EOT"

"${eot_opt}" "${output_dir}/03_tosa_eot.mlir" \
  --convert-eot-to-standard --canonicalize --cse \
  -o "${output_dir}/04_standard_tensor.mlir"
require_no_match '(^|[^[:alnum:]_])eot\.' \
  "${output_dir}/04_standard_tensor.mlir" \
  "EOT operations remain after EOT-to-standard"

"${mlir_opt}" "${output_dir}/04_standard_tensor.mlir" \
  --pass-pipeline='builtin.module(func.func(tosa-infer-shapes,tosa-to-scf,tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith{include-apply-rescale=true},canonicalize,cse))' \
  -o "${output_dir}/05_tosa_lowered.mlir"
require_no_match '(^|[^[:alnum:]_])tosa\.' \
  "${output_dir}/05_tosa_lowered.mlir" \
  "TOSA operations or types remain after TOSA lowering"

"${mlir_opt}" "${output_dir}/05_tosa_lowered.mlir" \
  --one-shot-bufferize='bufferize-function-boundaries' \
  --convert-bufferization-to-memref \
  --canonicalize --cse \
  -o "${output_dir}/06_bufferized.mlir"
require_no_match '(^|[^[:alnum:]_])(tensor\.|tensor<|bufferization\.)' \
  "${output_dir}/06_bufferized.mlir" \
  "tensor or bufferization IR remains after bufferization"

"${mlir_opt}" "${output_dir}/06_bufferized.mlir" \
  --convert-linalg-to-loops --canonicalize --cse \
  -o "${output_dir}/07_loops.mlir"
require_no_match '(^|[^[:alnum:]_])linalg\.' \
  "${output_dir}/07_loops.mlir" \
  "Linalg operations remain after loop lowering"

"${eot_opt}" "${output_dir}/07_loops.mlir" \
  --eot-plan-static-workspace \
  -o "${output_dir}/08_workspace.mlir"
require_no_match '(^|[^[:alnum:]_])memref\.(alloc|alloca|dealloc)([^[:alnum:]_]|$)' \
  "${output_dir}/08_workspace.mlir" \
  "workspace planning left allocation operations"
require_no_match '(^|[^[:alnum:]_])tensor<' \
  "${output_dir}/08_workspace.mlir" \
  "tensor types remain before LLVM conversion"

llvm_layout_args=()
if [[ -n "${llvm_data_layout}" ]]; then
  llvm_layout_args+=(
    "--set-llvm-module-datalayout=data-layout=${llvm_data_layout}"
  )
else
  echo "note: LLVM_DATA_LAYOUT is unset; llc/clang must receive the ARM target explicitly" >&2
fi

"${mlir_opt}" "${output_dir}/08_workspace.mlir" \
  "${llvm_layout_args[@]}" \
  --expand-strided-metadata --lower-affine --convert-scf-to-cf \
  --convert-math-to-libm --convert-math-to-llvm \
  --convert-arith-to-llvm="index-bitwidth=${target_index_bitwidth}" \
  --convert-index-to-llvm="index-bitwidth=${target_index_bitwidth}" \
  --finalize-memref-to-llvm="index-bitwidth=${target_index_bitwidth}" \
  --convert-cf-to-llvm="index-bitwidth=${target_index_bitwidth}" \
  --convert-func-to-llvm="index-bitwidth=${target_index_bitwidth}" \
  --reconcile-unrealized-casts \
  -o "${output_dir}/09_llvm_dialect.mlir"
require_no_match '(^|[^[:alnum:]_])(torch|tosa|eot|linalg|tensor|bufferization|scf|cf|math|arith|index|memref|func)\.' \
  "${output_dir}/09_llvm_dialect.mlir" \
  "non-LLVM dialect operations remain after LLVM lowering"

"${mlir_translate}" --mlir-to-llvmir \
  "${output_dir}/09_llvm_dialect.mlir" -o "${output_dir}/10_model.ll"
require_file "${output_dir}/10_model.ll"

cp "${output_dir}/02_tosa_custom.mlir" "${output_dir}/gm_frontend_tosa_custom.mlir"
cp "${output_dir}/03_tosa_eot.mlir" "${output_dir}/gm_frontend_tosa_eot.mlir"
cp "${output_dir}/04_standard_tensor.mlir" "${output_dir}/gm_frontend_standard.mlir"
cp "${output_dir}/06_bufferized.mlir" "${output_dir}/gm_frontend_bufferized.mlir"
cp "${output_dir}/08_workspace.mlir" "${output_dir}/gm_frontend_workspace.mlir"
cp "${output_dir}/09_llvm_dialect.mlir" "${output_dir}/gm_frontend_llvm.mlir"
cp "${output_dir}/10_model.ll" "${output_dir}/gm_frontend.ll"

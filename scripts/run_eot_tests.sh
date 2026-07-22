#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
project_build=${PROJECT_BUILD:-${BUILD_DIR:-build}}
npu_opt=${NPU_OPT:-${project_build}/bin/npu-opt}
filecheck=${FILECHECK:-${project_build}/bin/FileCheck}

for test in "${repo_dir}"/test/Eot/*-parse.mlir; do
  "${npu_opt}" "${test}" | "${filecheck}" "${test}"
done
for test in "${repo_dir}"/test/Eot/invalid-*.mlir; do
  "${npu_opt}" "${test}" --verify-diagnostics --split-input-file
done
for test in "${repo_dir}"/test/Eot/lower-*.mlir; do
  "${npu_opt}" "${test}" --convert-eot-to-standard | "${filecheck}" "${test}"
done
"${npu_opt}" "${repo_dir}/test/Eot/builtin-tensor-isolation.mlir" | \
  "${filecheck}" "${repo_dir}/test/Eot/builtin-tensor-isolation.mlir"
"${npu_opt}" "${repo_dir}/test/Eot/tosa-custom-to-eot.mlir" \
  --convert-tosa-custom-to-eot | "${filecheck}" \
  "${repo_dir}/test/Eot/tosa-custom-to-eot.mlir"
"${npu_opt}" "${repo_dir}/test/Eot/tosa-custom-invalid.mlir" \
  --convert-tosa-custom-to-eot --verify-diagnostics --split-input-file
"${npu_opt}" "${repo_dir}/test/Eot/workspace-plan.mlir" \
  --npu-plan-static-workspace | "${filecheck}" \
  "${repo_dir}/test/Eot/workspace-plan.mlir"
"${npu_opt}" --help | rg -- \
  '--convert-tosa-custom-to-eot|--convert-eot-to-standard|--npu-plan-static-workspace'

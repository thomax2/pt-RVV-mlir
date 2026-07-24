#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
project_build=${PROJECT_BUILD:-${BUILD_DIR:-build}}
eot_opt=${EOT_OPT:-${NPU_OPT:-${project_build}/npu-opt}}
if [[ -n "${FILECHECK:-}" ]]; then
  filecheck=${FILECHECK}
else
  mlir_build=${MLIR_BUILD:?set MLIR_BUILD or FILECHECK}
  filecheck=${mlir_build}/bin/FileCheck
fi

for test in "${repo_dir}"/test/Eot/*-parse.mlir; do
  "${eot_opt}" "${test}" | "${filecheck}" "${test}"
done
for test in "${repo_dir}"/test/Eot/invalid-*.mlir; do
  "${eot_opt}" "${test}" --verify-diagnostics --split-input-file
done
for test in "${repo_dir}"/test/Eot/lower-*.mlir; do
  "${eot_opt}" "${test}" --convert-eot-to-standard | "${filecheck}" "${test}"
done
"${eot_opt}" "${repo_dir}/test/Eot/builtin-tensor-isolation.mlir" | \
  "${filecheck}" "${repo_dir}/test/Eot/builtin-tensor-isolation.mlir"
"${eot_opt}" "${repo_dir}/test/Eot/tosa-custom-to-eot.mlir" \
  --convert-tosa-custom-to-eot | "${filecheck}" \
  "${repo_dir}/test/Eot/tosa-custom-to-eot.mlir"
"${eot_opt}" "${repo_dir}/test/Eot/tosa-custom-invalid.mlir" \
  --convert-tosa-custom-to-eot --verify-diagnostics --split-input-file
"${eot_opt}" "${repo_dir}/test/Eot/workspace-plan.mlir" \
  --eot-plan-static-workspace | "${filecheck}" \
  "${repo_dir}/test/Eot/workspace-plan.mlir"
"${eot_opt}" --help | rg -- \
  '--convert-tosa-custom-to-eot|--convert-eot-to-standard|--eot-plan-static-workspace'

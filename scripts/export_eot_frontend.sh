#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${PYTHON:-python3}
checkpoint=${CHECKPOINT:-}
output_dir=${OUTPUT_DIR:-eot-export-output}
nmax=${NMAX:-64}
batch=${BATCH_SIZE:-1}
random_seed=${RANDOM_SEED:-20260720}

export_args=(
  --output_dir "${output_dir}"
  --nmax "${nmax}"
  --batch_size "${batch}"
  --random_seed "${random_seed}"
  --device cpu
  --strict
  --save_reference
)
if [[ -n "${checkpoint}" ]]; then
  export_args+=(--checkpoint "${checkpoint}")
fi

"${python_bin}" "${repo_dir}/python/export_eot_frontend.py" \
  "${export_args[@]}" "${@}"

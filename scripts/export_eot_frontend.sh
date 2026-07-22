#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${PYTHON:-python3}
checkpoint=${CHECKPOINT:?set CHECKPOINT to the GM v33 checkpoint}
output_dir=${OUTPUT_DIR:-eot-export-output}
nmax=${NMAX:-64}
batch=${BATCH_SIZE:-1}

"${python_bin}" "${repo_dir}/python/export_eot_frontend.py" \
  --checkpoint "${checkpoint}" \
  --output_dir "${output_dir}" \
  --nmax "${nmax}" \
  --batch_size "${batch}" \
  --device cpu \
  --strict \
  --save_reference \
  "${@}"

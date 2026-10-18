#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
device="${1:-cpu}"
[[ "$device" == cpu || "$device" == cuda ]] || { echo 'Usage: cli_smoke.sh [cpu|cuda]' >&2; exit 1; }
embedding_bin="${EMBEDDING_BIN:-/opt/cuwacunu_embedding/build/baseline/embedding}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/output/runs/baseline}"
mkdir -p -- "$run_root"
output="$(mktemp -d "$run_root/smoke-$device-XXXXXX")"
"$embedding_bin" synthetic --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --output "$output/data.pt" --samples 9
"$embedding_bin" train --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --input "$output/data.pt" \
  --checkpoint "$output/model.pt" --steps 8 --device "$device"
"$embedding_bin" train --resume "$output/model.pt" --input "$output/data.pt" \
  --checkpoint "$output/resumed.pt" --steps 2 --device "$device"
"$embedding_bin" embed --checkpoint "$output/resumed.pt" --input "$output/data.pt" \
  --output "$output/embeddings.pt" --batch-size 4 --device "$device"
if [[ "$device" == cuda ]]; then
  # A CUDA-trained checkpoint must also load and serve on CPU.
  "$embedding_bin" embed --checkpoint "$output/resumed.pt" --input "$output/data.pt" \
    --output "$output/embeddings-cpu.pt" --batch-size 4 --device cpu
fi
printf 'PASS: %s CLI train, resume, and embedding export (%s)\n' "$device" "$output"

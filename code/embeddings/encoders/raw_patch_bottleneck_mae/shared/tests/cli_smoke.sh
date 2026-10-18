#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.."
device="${1:-cpu}"
[[ "$device" == cpu || "$device" == cuda ]] || { echo 'Usage: cli_smoke.sh [cpu|cuda]' >&2; exit 1; }
embedding_bin="${RPB_MAE_BIN:-${EMBEDDING_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-mae/embedding_raw_patch_bottleneck_mae}}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/.temp/runs/rpb-mae}"
mkdir -p -- "$run_root"
output="$(mktemp -d "$run_root/smoke-$device-XXXXXX")"
config="${RPB_CONFIG:-code/embeddings/encoders/raw_patch_bottleneck_mae/config/smoke.conf}"
"$embedding_bin" synthetic --config "$config" --output "$output/data.pt" --samples 9
"$embedding_bin" prepare --config "$config" --input "$output/data.pt" --output "$output/scaler.pt"
"$embedding_bin" train --config "$config" --input "$output/data.pt" --scaler "$output/scaler.pt" \
  --checkpoint "$output/model.pt" --steps 4 --checkpoint-every 1 --device "$device"
"$embedding_bin" train --resume "$output/model.pt" --input "$output/data.pt" \
  --checkpoint "$output/resumed.pt" --steps 2 --device "$device"
"$embedding_bin" embed --checkpoint "$output/resumed.pt" --input "$output/data.pt" \
  --output "$output/embeddings.pt" --batch-size 4 --device "$device"
if [[ "$device" == cuda ]]; then
  "$embedding_bin" embed --checkpoint "$output/resumed.pt" --input "$output/data.pt" \
    --output "$output/embeddings-cpu.pt" --batch-size 4 --device cpu
fi
printf 'PASS: RPB-MAE %s raw archive, scaler, train, resume, export (%s)\n' "$device" "$output"

#!/usr/bin/env bash
# Verify a real interrupted process leaves a usable periodic checkpoint.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
embedding_bin="${EMBEDDING_BIN:-/opt/cuwacunu_embedding/build/baseline/embedding}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/embedding-periodic-XXXXXX")"
training_pid=''
cleanup() {
  if [[ -n "$training_pid" ]]; then
    kill -TERM "$training_pid" 2>/dev/null || true
    wait "$training_pid" 2>/dev/null || true
  fi
  rm -rf -- "$test_dir"
}
trap cleanup EXIT
"$embedding_bin" synthetic --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --output "$test_dir/data.pt" --samples 16
"$embedding_bin" train --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --input "$test_dir/data.pt" \
  --checkpoint "$test_dir/interrupted.pt" --checkpoint-every 1 --steps 100000 \
  > "$test_dir/training.log" 2>&1 &
training_pid=$!
ready=false
for ((attempt = 0; attempt < 200; ++attempt)); do
  if [[ -f "$test_dir/interrupted.pt" ]]; then ready=true; break; fi
  if ! kill -0 "$training_pid" 2>/dev/null; then
    cat "$test_dir/training.log" >&2
    echo 'Training exited before a periodic checkpoint was saved.' >&2
    exit 1
  fi
  sleep 0.1
done
[[ "$ready" == true ]] || { echo 'Timed out waiting for a periodic checkpoint.' >&2; exit 1; }
kill -TERM "$training_pid"
wait "$training_pid" 2>/dev/null || true
training_pid=''
"$embedding_bin" train --resume "$test_dir/interrupted.pt" --input "$test_dir/data.pt" \
  --checkpoint "$test_dir/resumed.pt" --steps 1
"$embedding_bin" embed --checkpoint "$test_dir/resumed.pt" --input "$test_dir/data.pt" \
  --output "$test_dir/embeddings.pt"
printf 'PASS: interrupted training preserved a resumable periodic checkpoint\n'

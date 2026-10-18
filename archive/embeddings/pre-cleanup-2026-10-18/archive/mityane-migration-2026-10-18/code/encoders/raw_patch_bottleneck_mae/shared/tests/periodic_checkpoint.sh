#!/usr/bin/env bash
# Check an actual interrupted process preserves an atomic recovery checkpoint.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
embedding_bin="${RPB_MAE_BIN:-${EMBEDDING_BIN:-/opt/cuwacunu_embedding/build/rpb-mae/embedding_raw_patch_bottleneck_mae}}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/rpb-mae-periodic-XXXXXX")"
training_pid=''
cleanup() {
  if [[ -n "$training_pid" ]]; then
    kill -TERM "$training_pid" 2>/dev/null || true
    wait "$training_pid" 2>/dev/null || true
  fi
  rm -rf -- "$test_dir"
}
trap cleanup EXIT
config=code/encoders/raw_patch_bottleneck_mae/config/smoke.conf
"$embedding_bin" synthetic --config "$config" --output "$test_dir/data.pt" --samples 16
"$embedding_bin" train --config "$config" --input "$test_dir/data.pt" \
  --checkpoint "$test_dir/interrupted.pt" --checkpoint-every 1 --steps 100000 --attempt-limit 200000 \
  > "$test_dir/training.log" 2>&1 &
training_pid=$!
ready=false
for ((attempt=0;attempt<300;++attempt)); do
  if [[ -f "$test_dir/interrupted.pt" ]]; then ready=true; break; fi
  if ! kill -0 "$training_pid" 2>/dev/null; then
    cat "$test_dir/training.log" >&2
    echo 'Training exited before saving a periodic checkpoint.' >&2
    exit 1
  fi
  sleep 0.1
done
[[ "$ready" == true ]] || { echo 'Timed out waiting for a periodic checkpoint.' >&2; exit 1; }
kill -TERM "$training_pid"
wait "$training_pid" 2>/dev/null || true
training_pid=''
"$embedding_bin" train --resume "$test_dir/interrupted.pt" --input "$test_dir/data.pt" \
  --checkpoint "$test_dir/resumed.pt" --steps 1 --attempt-limit 100
"$embedding_bin" embed --checkpoint "$test_dir/resumed.pt" --input "$test_dir/data.pt" \
  --output "$test_dir/embeddings.pt"
printf 'PASS: RPB-MAE interrupted training recovered its periodic checkpoint\n'

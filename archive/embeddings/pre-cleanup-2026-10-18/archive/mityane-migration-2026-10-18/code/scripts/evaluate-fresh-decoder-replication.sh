#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run frozen fresh replication in the existing managed container.' >&2; exit 1; }
binary="${FRESH_DECODER_REPLICATION_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_fresh_decoder_replication}"
[[ "$binary" = /* && -x "$binary" ]] || { echo 'An absolute built binary is required.' >&2; exit 1; }
[[ -n "${FRESH_DECODER_REPLICATION_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to bind sources.' >&2; exit 1; }
root="$PWD/output/runs/rpb-fresh-decoder-replication"
helper='code/scripts/prepare-fresh-decoder-replication.py'
card_sha='2236fb794d608c8f7b8bde81814f199abcc3e27c641cfba5902cdc3f4c4a25cf'
[[ "$card_sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'The prospective card is not frozen yet.' >&2; exit 1; }
build_dir="$(dirname -- "$binary")"
[[ "$build_dir" = /opt/cuwacunu_embedding/build/* ]] || { echo 'Build objects must remain in the container filesystem.' >&2; exit 1; }
# task.sh passes its locked descriptor; direct invocation takes the same lock.
if [[ -e /proc/$$/fd/9 ]]; then
  [[ "$(readlink -- /proc/$$/fd/9)" == "$build_dir/.task.lock" ]] || { echo 'Unexpected inherited lock descriptor.' >&2; exit 1; }
else
  exec 9>"$build_dir/.task.lock"
fi
flock -n 9 || { echo 'The build session is active; preserve this binary and retry later.' >&2; exit 1; }
[[ -d "$build_dir/tmp" ]] || { echo 'Use the prepared named build session.' >&2; exit 1; }
export TMPDIR="$build_dir/tmp"
mkdir -p -- "$root"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete the actual CUDA admission first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
capsule="$(mktemp -d "$root/fresh-decoder-replication-XXXXXX")"
printf 'Fresh decoder replication capsule: %s\n' "$capsule"
source_id="$("$binary" --source-id)"
read -r -a sources <<< "$FRESH_DECODER_REPLICATION_SOURCE_INPUTS"
# --plan is a reviewed metadata-only mode: no model, generator or quality input.
"$binary" --plan > "$capsule/recipe-plan.json"
python3 "$helper" freeze-run --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --sources "${sources[@]}" --admission "$admission" \
  --compiled-source-id "$source_id" --binary "$binary"
log_sha="$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)"
"$binary" --output "$capsule/results" --admission-log "$capsule/admission/build-and-tests.log" \
  --admission-sha256 "$log_sha" 2>&1 | tee "$capsule/command.log"
python3 "$helper" finalize --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id"
printf 'Fresh decoder replication complete: %s\n' "$capsule"

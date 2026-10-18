#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run measurement in the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
binary="${FROZEN_AMPLITUDE_TRANSFER_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_frozen_amplitude_transfer}"
[[ "$binary" = /* && -x "$binary" && -n "${FROZEN_AMPLITUDE_TRANSFER_SOURCE_INPUTS:-}" ]] || { echo 'Use the source-bound Make target.' >&2; exit 1; }
root="$PWD/output/runs/rpb-frozen-amplitude-transfer"
helper='code/scripts/prepare-frozen-amplitude-transfer.py'
card_sha='901247488a9588e4b44dd0ae29169bb6c8d59c8af9e3f0b0a84d2568007ed52e'
build_dir="$(dirname -- "$binary")"
[[ "$build_dir" = /opt/cuwacunu_embedding/build/* ]]
if [[ -e /proc/$$/fd/9 ]]; then
  [[ "$(readlink -- /proc/$$/fd/9)" == "$build_dir/.task.lock" ]]
else
  exec 9>"$build_dir/.task.lock"
fi
flock -n 9 || { echo 'Build session active.' >&2; exit 1; }
export TMPDIR="$build_dir/tmp"
[[ -d "$TMPDIR" && -f "$root/admission-approved.path" && -f "$root/reader-approved.paths" ]] || { echo 'Complete CUDA admission and seal the reader first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
mapfile -t reader < "$root/reader-approved.paths"
[[ ${#reader[@]} -eq 3 ]]
capsule="$(mktemp -d "$root/frozen-amplitude-transfer-XXXXXX")"
printf 'Frozen amplitude capsule: %s\n' "$capsule"
source_id="$("$binary" --source-id)"
read -r -a sources <<< "$FROZEN_AMPLITUDE_TRANSFER_SOURCE_INPUTS"
"$binary" --plan > "$capsule/recipe-plan.json"
python3 -B "$helper" freeze-run --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --sources "${sources[@]}" --admission "$admission" \
  --compiled-source-id "$source_id" --binary "$binary" \
  --reader-source "${reader[0]}" --reader-fixtures "${reader[1]}" --reader-seal "${reader[2]}"
log_sha="$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)"
"$binary" --instances "$capsule/instances.tsv" --checksums "$capsule/input-checksums.sha256" \
  --input-root "$PWD" --output "$capsule/results" \
  --admission-log "$capsule/admission/build-and-tests.log" --admission-sha256 "$log_sha" \
  --card "$capsule/source/code/evaluation/cards/frozen_amplitude_transfer_v1.md" --card-sha256 "$card_sha" \
  2>&1 | tee "$capsule/command.log"
python3 -B "$helper" finalize --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id"
printf 'Frozen amplitude complete: %s\n' "$capsule"

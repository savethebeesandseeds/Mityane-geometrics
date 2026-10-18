#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run the closed diagnosis in the managed container.' >&2; exit 1; }
binary="${SAVED_NATIVE_RELIABILITY_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_saved_feature_reliability}"
[[ "$binary" = /* && -x "$binary" && -n "${SAVED_NATIVE_RELIABILITY_SOURCE_INPUTS:-}" ]] || { echo 'Use the source-bound Make target.' >&2; exit 1; }
root="$PWD/output/runs/rpb-saved-native-reliability"
helper='code/scripts/prepare-saved-native-reliability.py'
card_sha='f93898f6dd92869e319b251eabdb466fc9ac57da13e46c53685dcd25f7b38a91'
build_dir="$(dirname -- "$binary")"
[[ "$build_dir" = /opt/cuwacunu_embedding/build/* ]]
if [[ -e /proc/$$/fd/9 ]]; then
  [[ "$(readlink -- /proc/$$/fd/9)" == "$build_dir/.task.lock" ]]
else
  exec 9>"$build_dir/.task.lock"
fi
flock -n 9 || { echo 'The build session is active.' >&2; exit 1; }
export TMPDIR="$build_dir/tmp"
[[ -d "$TMPDIR" && -f "$root/admission-approved.path" && -f "$root/reader-approved.paths" ]] || { echo 'Complete CPU admission and seal the independent reader first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
mapfile -t reader < "$root/reader-approved.paths"
[[ ${#reader[@]} -eq 3 ]]
capsule="$(mktemp -d "$root/saved-native-reliability-XXXXXX")"
printf 'Saved native reliability capsule: %s\n' "$capsule"
source_id="$("$binary" --source-id)"
read -r -a sources <<< "$SAVED_NATIVE_RELIABILITY_SOURCE_INPUTS"
"$binary" --plan > "$capsule/recipe-plan.json"
python3 "$helper" freeze-run --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --sources "${sources[@]}" --admission "$admission" \
  --compiled-source-id "$source_id" --binary "$binary" \
  --reader-source "${reader[0]}" --reader-fixtures "${reader[1]}" --reader-seal "${reader[2]}"
log_sha="$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)"
"$binary" --instances "$capsule/instances.tsv" --checksums "$capsule/input-checksums.sha256" \
  --input-root "$PWD" --output "$capsule/results" \
  --admission-log "$capsule/admission/build-and-tests.log" --admission-sha256 "$log_sha" \
  --card "$capsule/source/code/evaluation/cards/saved_native_reliability_v1.md" --card-sha256 "$card_sha" \
  2>&1 | tee "$capsule/command.log"
python3 "$helper" finalize --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id"
printf 'Saved native reliability complete: %s\n' "$capsule"

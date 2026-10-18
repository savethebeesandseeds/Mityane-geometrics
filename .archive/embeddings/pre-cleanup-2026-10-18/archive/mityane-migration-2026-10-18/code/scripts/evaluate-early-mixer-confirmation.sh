#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
binary="${EARLY_MIXER_CONFIRMATION_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_early_mixer_confirmation}"
[[ "$binary" = /* && -x "$binary" && -n "${EARLY_MIXER_CONFIRMATION_SOURCE_INPUTS:-}" ]] || { echo 'Use the source-bound Make target.' >&2; exit 1; }
root="$PWD/output/runs/rpb-early-mixer-confirmation"
helper='code/scripts/prepare-early-mixer-confirmation.py'
card_sha='98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea'
build_dir="$(dirname -- "$binary")"
[[ "$build_dir" = /opt/cuwacunu_embedding/build/* ]]
if [[ -e /proc/$$/fd/9 ]]; then
  [[ "$(readlink -- /proc/$$/fd/9)" == "$build_dir/.task.lock" ]]
else
  exec 9>"$build_dir/.task.lock"
fi
flock -n 9 || { echo 'Build session active.' >&2; exit 1; }
export TMPDIR="$build_dir/tmp"
[[ -d "$TMPDIR" && -f "$root/admission-approved.path" && -f "$root/reader-approved.paths" ]] || { echo 'Admit CUDA and seal the reader first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
mapfile -t reader < "$root/reader-approved.paths"
[[ ${#reader[@]} -eq 3 ]]
capsule="$(mktemp -d "$root/early-mixer-confirmation-XXXXXX")"
printf 'Early mixer confirmation capsule: %s\n' "$capsule"
source_id="$("$binary" --source-id)"
read -r -a sources <<< "$EARLY_MIXER_CONFIRMATION_SOURCE_INPUTS"
"$binary" --plan > "$capsule/recipe-plan.json"
python3 -B "$helper" freeze-run --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --sources "${sources[@]}" --admission "$admission" \
  --compiled-source-id "$source_id" --binary "$binary" \
  --reader-source "${reader[0]}" --reader-fixtures "${reader[1]}" --reader-seal "${reader[2]}"
log_sha="$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)"
"$binary" --output "$capsule/results" --admission-log "$capsule/admission/build-and-tests.log" \
  --admission-sha256 "$log_sha" --card "$capsule/source/code/evaluation/cards/early_mixer_confirmation_v1.md" \
  --card-sha256 "$card_sha" 2>&1 | tee "$capsule/command.log"
python3 -B "$helper" finalize --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id"
printf 'Early mixer confirmation measurement complete: %s\n' "$capsule"

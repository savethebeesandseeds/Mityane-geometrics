#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
binary="${POOLED_CONTEXT_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_pooled_context}"
[[ "$binary" = /* && -x "$binary" && -n "${POOLED_CONTEXT_SOURCE_INPUTS:-}" ]] || { echo 'Use the source-bound Make target.' >&2; exit 1; }
root="$PWD/output/runs/rpb-pooled-context"
helper='code/scripts/prepare-pooled-context.py'
card_sha='b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7'
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
capsule="$(mktemp -d "$root/pooled-context-XXXXXX")"
printf 'Pooled context capsule: %s\n' "$capsule"
source_id="$("$binary" --source-id)"
read -r -a sources <<< "$POOLED_CONTEXT_SOURCE_INPUTS"
"$binary" --plan > "$capsule/recipe-plan.json"
python3 -B "$helper" freeze-run --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --sources "${sources[@]}" --admission "$admission" \
  --compiled-source-id "$source_id" --binary "$binary" \
  --reader-source "${reader[0]}" --reader-fixtures "${reader[1]}" --reader-seal "${reader[2]}"
log_sha="$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)"
"$binary" --output "$capsule/results" --admission-log "$capsule/admission/build-and-tests.log" \
  --admission-sha256 "$log_sha" --card "$capsule/source/code/evaluation/cards/pooled_context_v1.md" \
  --card-sha256 "$card_sha" 2>&1 | tee "$capsule/command.log"
python3 -B "$helper" finalize --repo-root "$PWD" --target "$capsule" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id"
printf 'Pooled context measurement complete: %s\n' "$capsule"

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the existing managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
binary="${VISIBLE_DIFFERENCE_BIN:-}"
[[ "$binary" == /opt/cuwacunu_embedding/build/rpb-visible-difference/embedding_visible_difference && -x "$binary" ]] || { echo 'Use the source-bound Make target.' >&2; exit 1; }
root="$PWD/output/runs/rpb-visible-difference"
helper='code/evaluation/protocols/visible_difference_v1/prepare_saved_run.py'
build_dir="$(dirname -- "$binary")"
if [[ -e /proc/$$/fd/9 ]]; then
  [[ "$(readlink -- /proc/$$/fd/9)" == "$build_dir/.task.lock" ]]
else
  exec 9>"$build_dir/.task.lock"
fi
flock -n 9 || { echo 'Build session active.' >&2; exit 1; }
export TMPDIR="$build_dir/tmp"
[[ -d "$TMPDIR" && -f "$root/admission-approved.path" && ! -L "$root/admission-approved.path" && -f "$root/reader-approved.path" && ! -L "$root/reader-approved.path" ]] || { echo 'Admit CUDA and freeze the independent reader first.' >&2; exit 1; }
mapfile -t a < "$root/admission-approved.path"
mapfile -t r < "$root/reader-approved.path"
[[ ${#a[@]} -eq 1 && "${a[0]}" = /* && ${#r[@]} -eq 1 && "${r[0]}" = /* ]] || { echo 'One canonical pointer each required.' >&2; exit 1; }
capsule="$(mktemp -d "$root/visible-difference-XXXXXX")"
printf 'Visible difference capsule: %s\n' "$capsule"
python3 -B "$helper" freeze-run --repo-root "$PWD" --target "$capsule" --admission "${a[0]}" --reader-dir "${r[0]}"
log_sha="$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)"
"$binary" --output "$capsule/results" --admission-log "$capsule/admission/build-and-tests.log" --admission-sha256 "$log_sha" \
  --card "$capsule/source/code/evaluation/cards/visible_difference_v1.md" --card-sha256 '75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc' \
  2>&1 | tee "$capsule/command.log"
python3 -B "$helper" finalize --repo-root "$PWD" --target "$capsule"
printf 'Visible difference complete: %s\n' "$capsule"

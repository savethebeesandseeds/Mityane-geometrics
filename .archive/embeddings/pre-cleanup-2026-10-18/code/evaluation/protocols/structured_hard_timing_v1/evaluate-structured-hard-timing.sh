#!/usr/bin/env bash
# One exclusive quality capsule after SOURCE, information, CUDA and reader gates.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
binary="${STRUCTURED_HARD_TIMING_BIN:-/opt/cuwacunu_embedding/build/rpb-structured-hard-timing/embedding_structured_hard_timing}"
[[ "$binary" = /opt/cuwacunu_embedding/build/rpb-structured-hard-timing/embedding_structured_hard_timing && -x "$binary" && -n "${STRUCTURED_HARD_TIMING_SOURCE_INPUTS:-}" ]] || { echo 'Use the source-bound Make target.' >&2; exit 1; }
root="$PWD/output/runs/rpb-structured-hard-timing"
helper='code/evaluation/protocols/structured_hard_timing_v1/prepare_saved_run.py'
card_sha='35f7aa9987f293bafbe735506cc86cd2af02f2ab39bec42dd1dc2d2dedfa758c'
[[ "$card_sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'Freeze the prospective card first.' >&2; exit 1; }
build_dir="$(dirname -- "$binary")"
[[ "$build_dir" = /opt/cuwacunu_embedding/build/rpb-structured-hard-timing ]] || { echo 'Use the named structured timing build.' >&2; exit 1; }
if [[ -e /proc/$$/fd/9 ]]; then
  [[ "$(readlink -- /proc/$$/fd/9)" == "$build_dir/.task.lock" ]]
else
  exec 9>"$build_dir/.task.lock"
fi
flock -n 9 || { echo 'Build session active.' >&2; exit 1; }
export TMPDIR="$build_dir/tmp"
[[ -d "$TMPDIR" && -f "$root/admission-approved.path" && ! -L "$root/admission-approved.path" && -f "$root/reader-approved.path" && ! -L "$root/reader-approved.path" ]] || { echo 'Admit information/CUDA and freeze the reader first.' >&2; exit 1; }
mapfile -t admission_pointer < "$root/admission-approved.path"
mapfile -t reader_pointer < "$root/reader-approved.path"
[[ ${#admission_pointer[@]} -eq 1 && "${admission_pointer[0]}" = /* && ${#reader_pointer[@]} -eq 1 && "${reader_pointer[0]}" = /* ]] || { echo 'Canonical admission and reader bundle paths required.' >&2; exit 1; }
capsule="$(mktemp -d "$root/structured-hard-timing-XXXXXX")"
printf 'Structured hard timing capsule: %s\n' "$capsule"
python3 -B "$helper" freeze-run --repo-root "$PWD" --target "$capsule" \
  --admission "${admission_pointer[0]}" --reader-dir "${reader_pointer[0]}"
log_sha="$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)"
"$binary" --output "$capsule/results" --admission-log "$capsule/admission/build-and-tests.log" \
  --admission-sha256 "$log_sha" --card "$capsule/source/code/evaluation/cards/structured_hard_timing_comparison_v2.md" \
  --card-sha256 "$card_sha" 2>&1 | tee "$capsule/command.log"
python3 -B "$helper" finalize --repo-root "$PWD" --target "$capsule"
printf 'Structured hard timing measurement complete: %s\n' "$capsule"

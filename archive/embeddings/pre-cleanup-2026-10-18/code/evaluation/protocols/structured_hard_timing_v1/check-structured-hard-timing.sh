#!/usr/bin/env bash
# SOURCE admission and actual CUDA engineering gates in the existing container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
root="$PWD/output/runs/rpb-structured-hard-timing"
session='rpb-structured-hard-timing'
protocol_root='code/evaluation/protocols/structured_hard_timing_v1'
helper="$protocol_root/prepare_saved_run.py"
makefile="$protocol_root/Makefile"
card_sha='35f7aa9987f293bafbe735506cc86cd2af02f2ab39bec42dd1dc2d2dedfa758c'
[[ "$card_sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'Freeze the prospective card first.' >&2; exit 1; }
[[ -f "$root/information-admission-approved.path" && ! -L "$root/information-admission-approved.path" ]] || { echo 'Admit the observed-only TEMPO-3 information test first.' >&2; exit 1; }
mapfile -t information_pointer < "$root/information-admission-approved.path"
[[ ${#information_pointer[@]} -eq 1 && "${information_pointer[0]}" = /* ]] || { echo 'One absolute information record required.' >&2; exit 1; }
information="${information_pointer[0]}"
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Structured hard timing admission: %s\n' "$admission"
mapfile -t sources < <(command make -s -f "$makefile" print-structured-hard-timing-sources)
[[ ${#sources[@]} -gt 0 ]] || { echo 'Closed SOURCE matrix required.' >&2; exit 1; }
python3 -B "$helper" self-test
python3 -B "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
# Force actual current object compile records inside task.sh's locked Make call.
# Argument forwarding is literal; no timestamps, objects or prior runs are changed.
make() {
  command make -f code/evaluation/protocols/structured_hard_timing_v1/Makefile \
    -W code/encoders/raw_patch_bottleneck_mae/shared/src/workflow.cpp \
    -W code/encoders/raw_patch_bottleneck_mae/shared/src/learning_curve_adapter.cpp \
    -W code/encoders/raw_patch_bottleneck_mae/shared/src/early_mixer_adapter.cpp \
    -W code/evaluation/protocols/structured_hard_timing_v1/structured_hard_timing_adapter.cpp \
    -W code/evaluation/protocols/structured_hard_timing_v1/paired_timing_run.cpp \
    -W code/evaluation/protocols/structured_hard_timing_v1/structured_hard_timing_main.cpp \
    -W code/evaluation/benchmarks/structured_hard_timing/structured_hard_timing.cpp \
    -W code/evaluation/benchmarks/structured_hard_timing/cross_feature_solvability.cpp \
    "$@"
}
export -f make
bash code/scripts/task.sh "$session" -j2 structured-hard-timing \
  test-rpb-structured-hard-timing-adapter test-rpb-early-mixer-model \
  test-rpb-early-mixer-adapter test-fixed-feature-readouts test-frozen-role-guard \
  2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_structured_hard_timing"
exec 9>"/opt/cuwacunu_embedding/build/$session/.task.lock"
flock -n 9 || { echo 'Build session busy.' >&2; exit 1; }
python3 -B -c 'import json; from pathlib import Path; p=Path("/opt/cuwacunu_embedding/setup/libtorch-installed.json"); print("Container SDK proof: " + json.dumps(json.loads(p.read_text()), sort_keys=True))' | tee -a "$admission/build-and-tests.log"
ldd "$binary" | tee -a "$admission/build-and-tests.log"
if grep -q '/embedding/.external/libtorch\|not found' "$admission/build-and-tests.log"; then
  echo 'Internal LibTorch runtime resolution required; preserve this admission.' >&2
  exit 1
fi
source_id="$("$binary" --source-id)"
python3 -B "$helper" admit --repo-root "$PWD" --target "$admission" \
  --binary "$binary" --compiled-source-id "$source_id" --information "$information"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Structured hard timing admission passed: %s\n' "$admission"

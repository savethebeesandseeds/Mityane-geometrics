#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the existing managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
root="$PWD/output/runs/rpb-visible-difference"
protocol_root='code/evaluation/protocols/visible_difference_v1'
session='rpb-visible-difference'
helper="$protocol_root/prepare_saved_run.py"
card_sha='75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc'
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Visible difference admission: %s\n' "$admission"
mapfile -t sources < <(command make -s -f "$protocol_root/Makefile" print-visible-difference-sources)
[[ ${#sources[@]} -gt 0 ]] || { echo 'Closed source closure required.' >&2; exit 1; }
python3 -B "$helper" self-test
python3 -B "$helper" freeze-admission --repo-root "$PWD" --target "$admission" --card-sha256 "$card_sha" --sources "${sources[@]}"
make() {
  command make -f code/evaluation/protocols/visible_difference_v1/Makefile \
    -W code/encoders/raw_patch_bottleneck_mae/shared/src/workflow.cpp \
    -W code/encoders/raw_patch_bottleneck_mae/shared/src/learning_curve_adapter.cpp \
    -W code/encoders/raw_patch_bottleneck_mae/shared/src/early_mixer_adapter.cpp \
    -W code/encoders/raw_patch_bottleneck_mae/versions/v13/src/visible_difference_adapter.cpp \
    -W code/evaluation/protocols/visible_difference_v1/visible_difference_main.cpp "$@"
}
export -f make
bash code/scripts/task.sh "$session" -j2 visible-difference \
  2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_visible_difference"
# The serialized artificial CPU input fixture uses the exact measured loader
# before any model/adapter CUDA fixture is executed. Its new archives survive.
"$binary" --input-loader-test "$admission/input-loader-fixtures" 2>&1 | tee -a "$admission/build-and-tests.log"
# Required producer objects were freshly compiled above; do not force a second
# record for those same objects while building/running the admission fixtures.
make() { command make -f code/evaluation/protocols/visible_difference_v1/Makefile "$@"; }
export -f make
bash code/scripts/task.sh "$session" -j2 test-visible-difference-model \
  test-visible-difference-adapter test-rpb-early-mixer-adapter test-fixed-feature-readouts test-frozen-role-guard \
  2>&1 | tee -a "$admission/build-and-tests.log"
exec 9>"/opt/cuwacunu_embedding/build/$session/.task.lock"
flock -n 9 || { echo 'Build session busy.' >&2; exit 1; }
python3 -B -c 'import json; from pathlib import Path; p=Path("/opt/cuwacunu_embedding/setup/libtorch-installed.json"); print("Container SDK proof: " + json.dumps(json.loads(p.read_text()),sort_keys=True))' | tee -a "$admission/build-and-tests.log"
ldd "$binary" | tee -a "$admission/build-and-tests.log"
if grep -q '/embedding/.external/libtorch\|not found' "$admission/build-and-tests.log"; then
  echo 'Internal LibTorch runtime required; preserve this admission.' >&2; exit 1
fi
source_id="$("$binary" --source-id)"
python3 -B "$helper" admit --repo-root "$PWD" --target "$admission" --binary "$binary" --compiled-source-id "$source_id"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Visible difference CUDA admission passed: %s\n' "$admission"

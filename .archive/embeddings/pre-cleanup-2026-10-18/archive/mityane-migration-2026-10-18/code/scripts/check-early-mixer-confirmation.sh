#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
root="$PWD/output/runs/rpb-early-mixer-confirmation"
session='rpb-paired-pooling'
helper='code/scripts/prepare-early-mixer-confirmation.py'
card_sha='98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea'
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Early mixer confirmation admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-early-mixer-confirmation-sources)
python3 -B "$helper" self-test
python3 -B "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
bash code/scripts/task.sh "$session" -j2 early-mixer-confirmation \
  test-rpb-early-mixer-confirmation-adapter \
  test-rpb-early-mixer-model test-rpb-early-mixer-adapter test-fixed-feature-readouts \
  test-frozen-role-guard 2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_early_mixer_confirmation"
exec 9>"/opt/cuwacunu_embedding/build/$session/.task.lock"
flock -n 9 || { echo 'Build session busy.' >&2; exit 1; }
# Retain the exact installed bundle proof and actual library resolution with
# the admission. This inspects dependencies without running another model.
python3 -B -c 'import json; from pathlib import Path; p=Path("/opt/cuwacunu_embedding/setup/libtorch-installed.json"); print("Container SDK proof: " + json.dumps(json.loads(p.read_text()), sort_keys=True))' | tee -a "$admission/build-and-tests.log"
ldd "$binary" | tee -a "$admission/build-and-tests.log"
if grep -q '/embedding/.external/libtorch\|not found' "$admission/build-and-tests.log"; then
  echo 'Internal LibTorch runtime resolution required; preserve this admission.' >&2
  exit 1
fi
source_id="$("$binary" --source-id)"
python3 -B "$helper" admit --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id" --binary "$binary"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Early mixer confirmation admission passed: %s\n' "$admission"

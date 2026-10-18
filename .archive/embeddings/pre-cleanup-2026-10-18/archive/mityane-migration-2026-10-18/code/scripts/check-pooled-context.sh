#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
root="$PWD/output/runs/rpb-pooled-context"
session='rpb-paired-pooling'
helper='code/scripts/prepare-pooled-context.py'
card_sha='b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7'
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Pooled context admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-pooled-context-sources)
python3 -B "$helper" self-test
python3 -B "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
bash code/scripts/task.sh "$session" -j2 pooled-context \
  test-rpb-pooled-context-model test-rpb-pooled-context-adapter \
  test-rpb-early-mixer-model test-rpb-early-mixer-adapter test-rpb-matched-target-gain-adapter test-fixed-feature-readouts \
  test-frozen-role-guard 2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_pooled_context"
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
printf 'Pooled context admission passed: %s\n' "$admission"

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
root="$PWD/.temp/embeddings/runs/rpb-matched-target-gain"
session='rpb-paired-pooling'
helper='code/embeddings/scripts/prepare-matched-target-gain.mjs'
card_sha='00d5d84815fd64a3e1787e3f25dddef045b1ca208458c19d5ca4000a5ca04eec'
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Matched target gain admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-matched-target-gain-sources)
bash code/scripts/node.sh --test code/embeddings/tests/preparation-migration.test.mjs
bash code/scripts/node.sh "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
bash code/scripts/task.sh "$session" -j2 matched-target-gain \
  test-rpb-early-mixer-model test-rpb-early-mixer-adapter test-rpb-matched-target-gain-adapter test-fixed-feature-readouts \
  test-frozen-role-guard 2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/mityane-geometrics/embeddings/build/$session/embedding_matched_target_gain"
exec 9>"/opt/mityane-geometrics/embeddings/build/$session/.task.lock"
flock -n 9 || { echo 'Build session busy.' >&2; exit 1; }
# Retain the exact installed bundle proof and actual library resolution with
# the admission. This inspects dependencies without running another model.
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs sdk-proof | tee -a "$admission/build-and-tests.log"
ldd "$binary" | tee -a "$admission/build-and-tests.log"
if grep -q '/embedding/.external/libtorch\|not found' "$admission/build-and-tests.log"; then
  echo 'Internal LibTorch runtime resolution required; preserve this admission.' >&2
  exit 1
fi
source_id="$("$binary" --source-id)"
bash code/scripts/node.sh "$helper" admit --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id" --binary "$binary"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Matched target gain admission passed: %s\n' "$admission"

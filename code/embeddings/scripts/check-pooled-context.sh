#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
root="$PWD/.temp/embeddings/runs/rpb-pooled-context"
session='rpb-paired-pooling'
helper='code/embeddings/scripts/prepare-pooled-context.mjs'
card_sha='b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7'
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Pooled context admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-pooled-context-sources)
bash code/scripts/node.sh --test code/embeddings/tests/preparation-migration.test.mjs
bash code/scripts/node.sh "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
bash code/scripts/task.sh "$session" -j2 pooled-context \
  test-rpb-pooled-context-model test-rpb-pooled-context-adapter \
  test-rpb-early-mixer-model test-rpb-early-mixer-adapter test-rpb-matched-target-gain-adapter test-fixed-feature-readouts \
  test-frozen-role-guard 2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/mityane-geometrics/embeddings/build/$session/embedding_pooled_context"
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
printf 'Pooled context admission passed: %s\n' "$admission"

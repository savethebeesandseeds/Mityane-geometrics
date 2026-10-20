#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run source-bound admission in the managed container.' >&2; exit 1; }
root="$PWD/.temp/embeddings/runs/rpb-saved-native-reliability"
session='rpb-paired-pooling'
helper='code/embeddings/scripts/prepare-saved-native-reliability.mjs'
card_sha='f93898f6dd92869e319b251eabdb466fc9ac57da13e46c53685dcd25f7b38a91'
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Saved native reliability admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-saved-native-reliability-sources)
[[ ${#sources[@]} -gt 0 ]]
bash code/scripts/node.sh --test code/embeddings/tests/preparation-migration.test.mjs
bash code/scripts/node.sh "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
bash code/scripts/task.sh "$session" -j2 saved-native-reliability \
  test-saved-feature-reliability test-frozen-role-guard 2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/mityane-geometrics/embeddings/build/$session/embedding_saved_feature_reliability"
[[ -x "$binary" ]]
exec 9>"/opt/mityane-geometrics/embeddings/build/$session/.task.lock"
flock -n 9 || { echo 'Build session became busy before admission seal.' >&2; exit 1; }
source_id="$("$binary" --source-id)"
bash code/scripts/node.sh "$helper" admit --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id" --binary "$binary"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Saved native reliability admission passed: %s\n' "$admission"

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run source-bound admission in the existing managed container.' >&2; exit 1; }
root="$PWD/output/runs/rpb-fresh-decoder-replication"
session='rpb-paired-pooling'
helper='code/scripts/prepare-fresh-decoder-replication.py'
# Root sets the reviewed final digest before source/admission freeze.
card_sha='2236fb794d608c8f7b8bde81814f199abcc3e27c641cfba5902cdc3f4c4a25cf'
[[ "$card_sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'The prospective card is not frozen yet.' >&2; exit 1; }
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Fresh decoder admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-fresh-decoder-replication-sources)
[[ ${#sources[@]} -gt 0 ]]
python3 "$helper" self-test
python3 "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
bash code/scripts/task.sh "$session" -j2 fresh-decoder-replication \
  test-rpb-frozen-decoder-calibration test-rpb-decoder-calibration test-fixed-feature-readouts \
  2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_fresh_decoder_replication"
[[ -x "$binary" ]]
# Protect the tested binary until admission metadata has been sealed.
exec 9>"/opt/cuwacunu_embedding/build/$session/.task.lock"
flock -n 9 || { echo 'Build session changed or became busy before admission seal.' >&2; exit 1; }
source_id="$("$binary" --source-id)"
python3 "$helper" admit --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id" --binary "$binary"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Fresh decoder admission passed: %s\n' "$admission"

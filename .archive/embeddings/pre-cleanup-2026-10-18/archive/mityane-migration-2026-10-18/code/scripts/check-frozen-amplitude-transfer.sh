#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run admission in the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
root="$PWD/output/runs/rpb-frozen-amplitude-transfer"
session='rpb-paired-pooling'
helper='code/scripts/prepare-frozen-amplitude-transfer.py'
card_sha='901247488a9588e4b44dd0ae29169bb6c8d59c8af9e3f0b0a84d2568007ed52e'
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Frozen amplitude admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-frozen-amplitude-transfer-sources)
[[ ${#sources[@]} -gt 0 ]]
python3 -B "$helper" self-test
python3 -B "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
bash code/scripts/task.sh "$session" -j2 frozen-amplitude-transfer \
  test-rpb-frozen-native-feature test-frozen-amplitude-legacy test-fixed-feature-readouts \
  test-frozen-role-guard 2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_frozen_amplitude_transfer"
[[ -x "$binary" ]]
exec 9>"/opt/cuwacunu_embedding/build/$session/.task.lock"
flock -n 9 || { echo 'Build session busy before admission seal.' >&2; exit 1; }
source_id="$("$binary" --source-id)"
python3 -B "$helper" admit --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id" --binary "$binary"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Frozen amplitude admission passed: %s\n' "$admission"

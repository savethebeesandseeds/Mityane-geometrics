#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Use the managed container.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
root="$PWD/output/runs/rpb-early-mixer-reliability"
session='rpb-paired-pooling'
helper='code/scripts/prepare-early-mixer-reliability.py'
card_sha='a4c3aa956a28e97d39f4b9c181c46b4d7cf5125b27516a5a5a1043b7745718f0'
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Early mixer admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-early-mixer-reliability-sources)
python3 -B "$helper" self-test
python3 -B "$helper" freeze-admission --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --sources "${sources[@]}"
bash code/scripts/task.sh "$session" -j2 early-mixer-reliability \
  test-rpb-early-mixer-model test-rpb-early-mixer-adapter test-fixed-feature-readouts \
  test-frozen-role-guard 2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_early_mixer_reliability"
exec 9>"/opt/cuwacunu_embedding/build/$session/.task.lock"
flock -n 9 || { echo 'Build session busy.' >&2; exit 1; }
source_id="$("$binary" --source-id)"
python3 -B "$helper" admit --repo-root "$PWD" --target "$admission" \
  --card-sha256 "$card_sha" --compiled-source-id "$source_id" --binary "$binary"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Early mixer admission passed: %s\n' "$admission"

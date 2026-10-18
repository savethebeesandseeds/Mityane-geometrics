#!/usr/bin/env bash
# Run one frozen candidate/cohort set in the existing managed container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv && $# -eq 5 ]] || {
  printf '%s\n' 'Usage inside managed container: run_screen.sh BINARY FREEZE CANDIDATE MASTERS NEW_OUTPUT' >&2
  exit 1
}
binary="$1"
freeze="$2"
candidate="$3"
masters="$4"
output="$5"
[[ "$candidate" == v14 || "$candidate" == v15 ]] || exit 1
[[ ! -e "$output" && -d "$(dirname -- "$output")" ]] || exit 1
exec 9>"$(dirname -- "$binary")/.task.lock"
flock -n 9 || { printf '%s\n' 'The named build session is busy.' >&2; exit 1; }
sha256sum -c "$freeze/sources.sha256" >/dev/null
card_sha=$(sha256sum "$freeze/card.md" | cut -d ' ' -f1)
sdk_sha=$(sha256sum "$freeze/sdk-proof.json" | cut -d ' ' -f1)
"$binary" --candidate "$candidate" --masters "$masters" \
  --input-root /embedding/output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS \
  --inputs-sha256 "$freeze/inputs.sha256" --sources-sha256 "$freeze/sources.sha256" \
  --output "$output" --card "$freeze/card.md" --card-sha256 "$card_sha" \
  --sdk-proof "$freeze/sdk-proof.json" --sdk-proof-sha256 "$sdk_sha"
sha256sum -c "$freeze/sources.sha256" >/dev/null
python3 -B code/evaluation/protocols/architecture_screen_v1/check_saved_stdlib.py \
  --run "$output" \
  --input-root /embedding/output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS \
  --output "$output/saved-check.json"

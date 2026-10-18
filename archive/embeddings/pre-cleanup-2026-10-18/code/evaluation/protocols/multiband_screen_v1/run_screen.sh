#!/usr/bin/env bash
# Exactly one new data matrix, then one fixed-budget run for each candidate.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv && $# == 1 ]] || exit 1
admission=$1
[[ "$admission" == /embedding/output/runs/rpb-multiband-screen/admission-* ]]
[[ $(realpath -e -- "$admission") == "$admission" && $(dirname -- "$admission") == /embedding/output/runs/rpb-multiband-screen && ! -L "$admission" ]]
[[ -f "$admission/admission.log" && ! -e "$admission/quality-data" && ! -e "$admission/v18" && ! -e "$admission/v19" ]]
grep -q 'Spectral temporal relation CUDA admission passed' "$admission/v19-test.log"
grep -q 'Multiband screen runner CUDA engineering passed' "$admission/runner-engineering.log"
grep -q 'Multiband quality data engineering passed' "$admission/data-engineering.log"
protocol=code/evaluation/protocols/multiband_screen_v1
build=/opt/cuwacunu_embedding/build/rpb-multiband-screen
exec 9>"$build/.task.lock"
flock -n 9 || { printf '%s\n' 'Named multiband session is busy.' >&2; exit 1; }
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-before-quality.log"
card_sha=$(sha256sum "$admission/freeze/card.md" | cut -d ' ' -f1)
sdk_sha=$(sha256sum "$admission/freeze/sdk-proof.json" | cut -d ' ' -f1)
admit_sha=$(sha256sum "$admission/admission.log" | cut -d ' ' -f1)
"$build/embedding_multiband_quality_data" --output "$admission/quality-data" --card "$admission/freeze/card.md" \
  --card-sha256 "$card_sha" --masters 920903,920904 >"$admission/data-quality.log" 2>&1
python3 -B "$protocol/prepare_screen.py" --output "$admission/freeze" --bind-input-root "$admission/quality-data" >"$admission/input-freeze.log"
for candidate in v18 v19; do
  "$build/embedding_multiband_screen" --candidate "$candidate" --masters 920903,920904 \
    --input-root "$admission/quality-data" --inputs-sha256 "$admission/freeze/inputs.sha256" \
    --sources-sha256 "$admission/freeze/sources.sha256" --output "$admission/$candidate" \
    --card "$admission/freeze/card.md" --card-sha256 "$card_sha" \
    --sdk-proof "$admission/freeze/sdk-proof.json" --sdk-proof-sha256 "$sdk_sha" \
    --admission-log "$admission/admission.log" --admission-sha256 "$admit_sha" >"$admission/$candidate.log" 2>&1
  python3 -B "$protocol/check_saved_stdlib.py" --run "$admission/$candidate" --input-root "$admission/quality-data" \
    --freeze "$admission/freeze" --output "$admission/$candidate-saved-check.json" >"$admission/$candidate-saved-check.log" 2>&1
  printf 'QUALITY_AND_SAVED_CHECK_COMPLETE %s %s\n' "$candidate" "$admission"
done
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-after-quality.log"
printf 'MULTIBAND_SCREEN_COMPLETE %s\n' "$admission"

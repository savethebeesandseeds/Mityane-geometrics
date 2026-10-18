#!/usr/bin/env bash
# One fresh data matrix, then all six predetermined unchanged-v19 trajectories.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv && $# == 1 ]] || exit 1
admission=$1
[[ "$admission" == /embedding/output/runs/rpb-spectral-confirmation/admission-* ]]
[[ $(realpath -e -- "$admission") == "$admission" && $(dirname -- "$admission") == /embedding/output/runs/rpb-spectral-confirmation && ! -L "$admission" ]]
[[ -f "$admission/admission.log" && ! -e "$admission/quality-data" && ! -e "$admission/v19" ]]
grep -q 'Spectral temporal relation CUDA admission passed' "$admission/v19-test.log"
grep -q 'Spectral confirmation runner CUDA engineering passed' "$admission/runner-engineering.log"
grep -q 'Spectral confirmation quality data engineering passed' "$admission/data-engineering.log"
protocol=code/evaluation/protocols/spectral_confirmation_v1
build=/opt/cuwacunu_embedding/build/rpb-spectral-confirmation
exec 9>"$build/.task.lock"
flock -n 9 || { printf '%s\n' 'Named confirmation session is busy.' >&2; exit 1; }
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-before-quality.log"
card_sha=$(sha256sum "$admission/freeze/card.md" | cut -d ' ' -f1)
sdk_sha=$(sha256sum "$admission/freeze/sdk-proof.json" | cut -d ' ' -f1)
admit_sha=$(sha256sum "$admission/admission.log" | cut -d ' ' -f1)
"$build/embedding_spectral_confirmation_quality_data" --output "$admission/quality-data" --card "$admission/freeze/card.md" \
  --card-sha256 "$card_sha" --masters 930905,930906,930907 >"$admission/data-quality.log" 2>&1
python3 -B "$protocol/prepare_confirmation.py" --output "$admission/freeze" --bind-input-root "$admission/quality-data" >"$admission/input-freeze.log"
"$build/embedding_spectral_confirmation" --candidate v19 --masters 930905,930906,930907 \
  --input-root "$admission/quality-data" --inputs-sha256 "$admission/freeze/inputs.sha256" \
  --sources-sha256 "$admission/freeze/sources.sha256" --output "$admission/v19" \
  --card "$admission/freeze/card.md" --card-sha256 "$card_sha" \
  --sdk-proof "$admission/freeze/sdk-proof.json" --sdk-proof-sha256 "$sdk_sha" \
  --admission-log "$admission/admission.log" --admission-sha256 "$admit_sha" >"$admission/v19.log" 2>&1
python3 -B "$protocol/check_saved_stdlib.py" --run "$admission/v19" --input-root "$admission/quality-data" \
  --freeze "$admission/freeze" --output "$admission/v19-saved-check.json" >"$admission/v19-saved-check.log" 2>&1
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-after-quality.log"
printf 'SPECTRAL_CONFIRMATION_COMPLETE %s\n' "$admission"

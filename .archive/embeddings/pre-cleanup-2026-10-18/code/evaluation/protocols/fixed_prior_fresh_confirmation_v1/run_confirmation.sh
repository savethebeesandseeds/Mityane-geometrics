#!/usr/bin/env bash
# One CUDA quality trajectory per declared fresh cohort; saved-only CPU checks.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv && $# -eq 2 ]] || {
  printf '%s\n' 'Usage inside managed container: run_confirmation.sh ADMISSION NEW_OUTPUT' >&2
  exit 1
}
admission="$1"
output="$2"
freeze="$admission/freeze"
build_root=/opt/cuwacunu_embedding/build/rpb-fixed-prior-fresh-confirmation
[[ ! -e "$output" && -d "$(dirname -- "$output")" && -s "$admission/engineering.log" ]] || exit 1
exec 9>"$build_root/.task.lock"
flock -n 9 || { printf '%s\n' 'The named session is busy.' >&2; exit 1; }
sha256sum -c "$freeze/sources.sha256" >/dev/null
card_sha=$(sha256sum "$freeze/card.md" | cut -d ' ' -f1)
sdk_sha=$(sha256sum "$freeze/sdk-proof.json" | cut -d ' ' -f1)
engineering_sha=$(sha256sum "$admission/engineering.log" | cut -d ' ' -f1)
"$build_root/embedding_fixed_prior_fresh_confirmation" \
  --candidate v17 --masters 80787,81888,82989,84090,85191 \
  --input-root "$admission/fresh-data" --inputs-sha256 "$freeze/inputs.sha256" \
  --sources-sha256 "$freeze/sources.sha256" --output "$output" \
  --card "$freeze/card.md" --card-sha256 "$card_sha" \
  --sdk-proof "$freeze/sdk-proof.json" --sdk-proof-sha256 "$sdk_sha" \
  --admission-log "$admission/engineering.log" --admission-sha256 "$engineering_sha"
sha256sum -c "$freeze/sources.sha256" >/dev/null
python3 -B code/evaluation/protocols/fixed_prior_fresh_confirmation_v1/check_saved_stdlib.py \
  --run "$output" --input-root "$admission/fresh-data" --freeze "$freeze" \
  --output "$output/saved-check.json"

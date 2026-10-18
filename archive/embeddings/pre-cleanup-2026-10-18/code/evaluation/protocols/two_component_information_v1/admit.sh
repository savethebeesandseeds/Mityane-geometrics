#!/usr/bin/env bash
# Freeze, build and run data-only admission inside the existing container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv ]] || exit 1
run_root=/embedding/output/runs/two-component-information-v1
mkdir -p "$run_root"
admission=$(mktemp -d "$run_root/admission-XXXXXX")
printf '%s\n' "$admission"
bash code/scripts/task.sh two-component-information-v1 \
  -f code/evaluation/protocols/two_component_information_v1/Makefile \
  print-two-component-information-sources >"$admission/source-list.log"
sed -n '/^Makefile$/,$p' "$admission/source-list.log" >"$admission/source-list.txt"
python3 -B code/evaluation/protocols/two_component_information_v1/prepare.py \
  --source-list "$admission/source-list.txt" --output "$admission/freeze" >"$admission/freeze.log"
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-before.log"
bash code/scripts/task.sh two-component-information-v1 \
  -f code/evaluation/protocols/two_component_information_v1/Makefile -j2 \
  two-component-information two-component-information-tests >"$admission/build.log" 2>&1
exec 9>/opt/cuwacunu_embedding/build/two-component-information-v1/.task.lock
flock -n 9 || { printf '%s\n' 'Named data-only session is busy.' >&2; exit 1; }
/opt/cuwacunu_embedding/build/two-component-information-v1/code/protocols/two_component_information_v1/generator_test >"$admission/generator-test.log" 2>&1
/opt/cuwacunu_embedding/build/two-component-information-v1/code/protocols/two_component_information_v1/information_test >"$admission/information-test.log" 2>&1
card_sha=$(sha256sum "$admission/freeze/card.md" | cut -d ' ' -f1)
sdk_sha=$(sha256sum "$admission/freeze/sdk-proof.json" | cut -d ' ' -f1)
/opt/cuwacunu_embedding/build/two-component-information-v1/embedding_two_component_information \
  --output "$admission/information" --card "$admission/freeze/card.md" \
  --card-sha256 "$card_sha" --sources "$admission/freeze/sources.sha256" \
  --sdk-proof "$admission/freeze/sdk-proof.json" --sdk-sha256 "$sdk_sha" \
  >"$admission/information.log" 2>&1
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-after.log"
python3 -B code/evaluation/protocols/two_component_information_v1/check_saved.py \
  --admission "$admission" >"$admission/saved-check.log" 2>&1
printf 'DATA_ADMISSION_COMPLETE %s\n' "$admission"

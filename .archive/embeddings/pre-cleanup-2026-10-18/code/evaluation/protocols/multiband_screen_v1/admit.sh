#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv ]] || exit 1
run_root=/embedding/output/runs/rpb-multiband-screen
mkdir -p "$run_root"
admission=$(mktemp -d "$run_root/admission-XXXXXX")
printf '%s\n' "$admission"
protocol=code/evaluation/protocols/multiband_screen_v1
bash code/scripts/task.sh rpb-multiband-screen -f "$protocol/Makefile" print-multiband-screen-sources >"$admission/source-list.log"
sed -n '/^Makefile$/,$p' "$admission/source-list.log" >"$admission/source-list.txt"
python3 -B "$protocol/prepare_screen.py" --source-list "$admission/source-list.txt" --output "$admission/freeze" >"$admission/freeze.log"
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-before.log"
bash code/scripts/task.sh rpb-multiband-screen -f "$protocol/Makefile" -j2 \
  multiband-screen multiband-quality-data \
  /opt/cuwacunu_embedding/build/rpb-multiband-screen/code/protocols/multiband_screen_v1/spectral_temporal_relation_test >"$admission/build.log" 2>&1
exec 9>/opt/cuwacunu_embedding/build/rpb-multiband-screen/.task.lock
flock -n 9 || { printf '%s\n' 'Named multiband session is busy.' >&2; exit 1; }
build=/opt/cuwacunu_embedding/build/rpb-multiband-screen
[[ $("$build/embedding_multiband_screen" --source-id) == $(sha256sum "$admission/freeze/sources.sha256" | cut -d ' ' -f1) ]]
[[ $("$build/embedding_multiband_quality_data" --source-id) == $(sha256sum "$admission/freeze/sources.sha256" | cut -d ' ' -f1) ]]
"$build/code/protocols/multiband_screen_v1/spectral_temporal_relation_test" >"$admission/v19-test.log" 2>&1
"$build/embedding_multiband_screen" --engineering "$admission/runner-engineering" >"$admission/runner-engineering.log" 2>&1
"$build/embedding_multiband_quality_data" --engineering "$admission/data-engineering" >"$admission/data-engineering.log" 2>&1
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-after-engineering.log"
cat "$admission/v19-test.log" "$admission/runner-engineering.log" "$admission/data-engineering.log" >"$admission/admission.log"
printf 'CUDA_ADMISSION_COMPLETE %s\n' "$admission"

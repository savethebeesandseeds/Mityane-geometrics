#!/usr/bin/env bash
# Freeze/build/admit one separate design in the existing managed container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv ]] || exit 1
run_root=/embedding/output/runs/rpb-unit-relation-screen
mkdir -p "$run_root"
admission=$(mktemp -d "$run_root/admission-XXXXXX")
printf '%s\n' "$admission"
bash code/scripts/task.sh rpb-unit-relation-screen \
  -f code/evaluation/protocols/unit_relation_screen_v1/Makefile \
  print-unit-relation-screen-sources >"$admission/source-list.log"
sed -n '/^Makefile$/,$p' "$admission/source-list.log" >"$admission/source-list.txt"
python3 -B code/evaluation/protocols/unit_relation_screen_v1/prepare_screen.py \
  --source-list "$admission/source-list.txt" \
  --input-root /embedding/output/runs/rpb-fixed-prior-fresh-confirmation/admission-3CGpxM/fresh-data \
  --masters 80787,84090 --output "$admission/freeze" >"$admission/freeze.log"
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-check-before.log"
bash code/scripts/task.sh rpb-unit-relation-screen \
  -f code/evaluation/protocols/unit_relation_screen_v1/Makefile -j2 \
  unit-relation-screen \
  /opt/cuwacunu_embedding/build/rpb-unit-relation-screen/code/protocols/unit_relation_screen_v1/unit_temporal_relation_test \
  >"$admission/build-tests.log" 2>&1
# Parallel CPU compilation above; CUDA admission programs execute sequentially.
/opt/cuwacunu_embedding/build/rpb-unit-relation-screen/code/protocols/unit_relation_screen_v1/unit_temporal_relation_test \
  >>"$admission/build-tests.log" 2>&1
/opt/cuwacunu_embedding/build/rpb-unit-relation-screen/embedding_unit_relation_screen \
  --engineering "$admission/engineering" >>"$admission/build-tests.log" 2>&1
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-check-after.log"
printf 'ADMISSION_PASSED %s\n' "$admission"

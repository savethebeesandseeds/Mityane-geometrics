#!/usr/bin/env bash
# Freeze/build/admit one separate design in the existing managed container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv ]] || exit 1
run_root=/embedding/output/runs/rpb-partitioned-relation-screen
mkdir -p "$run_root"
admission=$(mktemp -d "$run_root/admission-XXXXXX")
printf '%s\n' "$admission"
bash code/scripts/task.sh rpb-partitioned-relation-screen \
  -f code/evaluation/protocols/partitioned_relation_screen_v1/Makefile \
  print-partitioned-relation-screen-sources >"$admission/source-list.log"
sed -n '/^Makefile$/,$p' "$admission/source-list.log" >"$admission/source-list.txt"
python3 -B code/evaluation/protocols/partitioned_relation_screen_v1/prepare_screen.py \
  --source-list "$admission/source-list.txt" \
  --input-root /embedding/output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS \
  --masters 75272,76373 --output "$admission/freeze" >"$admission/freeze.log"
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-check-before.log"
bash code/scripts/task.sh rpb-partitioned-relation-screen \
  -f code/evaluation/protocols/partitioned_relation_screen_v1/Makefile -j2 \
  partitioned-relation-screen test-partitioned-relation-screen-v16 \
  test-partitioned-relation-screen-runner ENGINEERING_OUTPUT="$admission/engineering" \
  >"$admission/build-tests.log" 2>&1
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-check-after.log"
printf 'ADMISSION_PASSED %s\n' "$admission"

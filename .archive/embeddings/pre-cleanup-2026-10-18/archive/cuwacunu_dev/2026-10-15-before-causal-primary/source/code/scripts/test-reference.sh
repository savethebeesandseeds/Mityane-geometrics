#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: test-reference.sh IMPORT_TEST_BINARY' >&2; exit 1; }
root=output/reference/rpb-v19.alt-01
[[ -d "$root" ]] || { echo 'Reference artifacts missing; see doc/REFERENCE_ARTIFACTS.json.' >&2; exit 1; }
for master in 930905 930906 930907; do
  for task in slow_lag_sign component_balance; do
    cohort="$root/seed-$master-$task"
    "$1" "$cohort/checkpoint.pt" "$cohort/validation.pt" "$cohort/features.pt"
  done
done

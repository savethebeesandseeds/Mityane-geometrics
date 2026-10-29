#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
dashboard_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -f /.dockerenv && $(uname -s) == Linux && $dashboard_root == /mityane/web/dashboard ]] || {
  echo 'Use web/dashboard/scripts/refresh.ps1 on Windows; refresh runs only in mityane-geometrics.' >&2; exit 1;
}
check=()
audit=0
run_tests=0
run_branding=0
for argument in "$@"; do
  case "$argument" in
    --check) check=(--check) ;;
    --audit) audit=1 ;;
    --test) run_tests=1 ;;
    --branding) run_branding=1 ;;
    --help|-h)
      echo 'Usage: bash web/dashboard/scripts/refresh.sh [--check] [--audit] [--test] [--branding]'
      echo 'Publish public/index.html from the Hero JSON and prepared artwork.'
      echo '--audit re-reads evidence files and current receipts. --test runs the focused dashboard tests. --branding prepares artwork.'
      exit 0 ;;
    *) echo "Unknown option: $argument" >&2; exit 2 ;;
  esac
done
node=(bash /mityane/code/scripts/node.sh)
mkdir -p /opt/mityane-geometrics/web/tmp
export TMPDIR=/opt/mityane-geometrics/web/tmp
exec 9>/opt/mityane-geometrics/web/refresh.lock
flock -n 9 || { echo 'Another dashboard refresh is running; retry after it completes.' >&2; exit 1; }
if ((run_branding)); then
  make --no-print-directory -C /mityane dashboard-branding-native
  "${node[@]}" "$dashboard_root/scripts/prepare-branding.mjs" "${check[@]}"
fi
if ((run_tests)); then
  "${node[@]}" --test "$dashboard_root/tests/"*.test.mjs
fi
validate_args=("${check[@]}")
if ((audit)); then validate_args+=(--audit); fi
"${node[@]}" "$dashboard_root/scripts/validate.mjs" "${validate_args[@]}"

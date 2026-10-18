#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
dashboard_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -f /.dockerenv && $(uname -s) == Linux && $dashboard_root == /mityane/web/dashboard ]] || {
  echo 'Use web/dashboard/scripts/refresh.ps1 on Windows; refresh runs only in mityane-geometrics.' >&2; exit 1;
}
check=()
for argument in "$@"; do
  case "$argument" in
    --check) check=(--check) ;;
    --help|-h) echo 'Usage: bash web/dashboard/scripts/refresh.sh [--check]'; exit 0 ;;
    *) echo "Unknown option: $argument" >&2; exit 2 ;;
  esac
done
runtime=/opt/mityane-geometrics/web/tools/node-v22.23.3-linux-x64
[[ -x $runtime/bin/node ]] || { echo 'Run web/dashboard/scripts/setup.sh inside the managed container first.' >&2; exit 1; }
[[ $("$runtime/bin/node" --version) == v22.23.3 ]] || { echo 'Pinned Linux Node v22.23.3 is required.' >&2; exit 1; }
mkdir -p /opt/mityane-geometrics/web/tmp
export TMPDIR=/opt/mityane-geometrics/web/tmp
exec 9>/opt/mityane-geometrics/web/refresh.lock
flock -n 9 || { echo 'Another dashboard refresh is running; retry after it completes.' >&2; exit 1; }
python3 "$dashboard_root/scripts/prepare-branding.py" "${check[@]}"
"$runtime/bin/node" --test "$dashboard_root/tests/"*.test.mjs
"$runtime/bin/node" "$dashboard_root/scripts/validate.mjs" "${check[@]}"

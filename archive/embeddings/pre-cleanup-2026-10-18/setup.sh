#!/usr/bin/env bash
# The component shares the parent project's offline dependency verification.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
exec bash "$project_root/setup.sh" "$@"
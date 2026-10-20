#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Compatibility entry: shared dependency setup owns the pinned Node runtime.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
exec bash "$project_root/code/scripts/setup-node.sh" "$@"

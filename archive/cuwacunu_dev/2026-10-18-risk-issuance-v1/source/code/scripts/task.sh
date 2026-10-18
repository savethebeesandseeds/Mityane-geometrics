#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the documented Linux development container.' >&2; exit 1; }
# Keep bundled cuDNN/LibTorch first, matching the verified inherited runtime.
source /etc/profile.d/embedding.sh
[[ $# -ge 1 ]] || { echo 'Usage: bash code/scripts/task.sh SESSION [make arguments...]' >&2; exit 1; }
session="$1"
shift
[[ "$session" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || { echo 'Invalid session name.' >&2; exit 1; }
build_dir="/opt/cuwacunu/build/$session"
mkdir -p "$build_dir/tmp"
export TMPDIR="$build_dir/tmp"
exec 9>"$build_dir/.task.lock"
flock -n 9 || { echo "Session '$session' is busy." >&2; exit 1; }
make "$@" BUILD_DIR="$build_dir"

#!/usr/bin/env bash
# One project build entry; no environment installation or container lifecycle.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && "$PWD" == /mityane ]] || {
  echo 'Run through the managed mityane-geometrics container.' >&2; exit 1;
}
[[ $# -ge 1 ]] || { echo 'Usage: bash code/scripts/task.sh SESSION [make arguments...]' >&2; exit 1; }
session="$1"
shift
[[ "$session" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
  echo 'Session must be 1-64 lowercase letters, digits, underscores or hyphens, starting with a letter or digit.' >&2; exit 1;
}
build_dir="/opt/mityane-geometrics/embeddings/build/$session"
run_root="$PWD/.temp/embeddings/runs/$session"
mkdir -p -- "$build_dir/tmp"
export TMPDIR="$build_dir/tmp"
exec 9>"$build_dir/.task.lock"
flock -n 9 || { echo "Session '$session' is already active; use another session or wait." >&2; exit 1; }
printf 'Session: %s\nBuild: %s\nRuns: %s\n' "$session" "$build_dir" "$run_root"
make "$@" BUILD_DIR="$build_dir" BIN="$build_dir/embedding" \
  RPB_BIN="$build_dir/embedding_raw_patch_bottleneck_mae" \
  EVALUATION_BIN="$build_dir/embedding_evaluate" \
  HARNESS_BIN="$build_dir/feature_harness" RUN_ROOT="$run_root"

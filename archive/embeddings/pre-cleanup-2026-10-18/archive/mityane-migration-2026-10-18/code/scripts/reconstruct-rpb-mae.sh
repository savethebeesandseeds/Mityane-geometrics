#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run reconstruction diagnostics inside the managed development container.' >&2; exit 1; }
for argument in "$@"; do
  [[ "$argument" != --output && "$argument" != --output=* ]] || {
    echo 'This runner allocates --output; choose its parent with EMBEDDING_RUN_ROOT.' >&2; exit 1;
  }
done
binary="${EVALUATION_BIN:-/opt/cuwacunu_embedding/build/rpb-mae/embedding_evaluate}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/output/runs/rpb-mae}"
mkdir -p -- "$run_root"
run_directory="$(mktemp -d "$run_root/reconstruction-XXXXXX")"
"$binary" reconstruct "$@" --output "$run_directory/results"
printf 'RPB-MAE reconstruction artifacts: %s\n' "$run_directory/results"

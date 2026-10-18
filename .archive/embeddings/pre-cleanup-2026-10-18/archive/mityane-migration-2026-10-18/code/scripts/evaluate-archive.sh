#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run evaluation inside the managed development container.' >&2; exit 1; }
for argument in "$@"; do
  [[ "$argument" != --output && "$argument" != --output=* ]] || {
    echo 'This runner allocates --output; choose its parent with EMBEDDING_RUN_ROOT.' >&2; exit 1;
  }
done
binary="${ARCHIVE_READOUT_BIN:-/opt/cuwacunu_embedding/build/archive-controls/embedding_archive_readout}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/output/runs/archive-controls}"
mkdir -p -- "$run_root"
run_directory="$(mktemp -d "$run_root/archive-readout-XXXXXX")"
"$binary" "$@" --output "$run_directory/results"
printf 'Archive readout artifacts: %s\n' "$run_directory/results"

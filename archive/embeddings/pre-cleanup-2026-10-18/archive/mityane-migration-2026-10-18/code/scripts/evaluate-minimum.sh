#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run evaluation inside the managed development container.' >&2; exit 1; }
select_encoder=true
for argument in "$@"; do
  [[ "$argument" != --output && "$argument" != --output=* ]] || {
    echo 'This runner allocates --output; choose its parent with EMBEDDING_RUN_ROOT.' >&2; exit 1;
  }
  [[ "$argument" != --encoders ]] || select_encoder=false
done
binary="${EMBEDDING_BIN:-/opt/cuwacunu_embedding/build/rpb-mae/feature_harness}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/output/runs/rpb-mae}"
mkdir -p -- "$run_root"
run_directory="$(mktemp -d "$run_root/minimum-XXXXXX")"
selection=()
if [[ "$select_encoder" == true ]]; then selection=(--encoders baseline); fi
"$binary" evaluate "${selection[@]}" "$@" --output "$run_directory/results"
printf 'Minimum evaluation artifacts: %s\n' "$run_directory/results"

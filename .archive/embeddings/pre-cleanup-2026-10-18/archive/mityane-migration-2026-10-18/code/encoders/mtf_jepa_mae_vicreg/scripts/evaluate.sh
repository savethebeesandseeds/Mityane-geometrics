#!/usr/bin/env bash
# Each report gets a new directory so checkpoint and dataset artifacts coexist.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
embedding_bin="${EMBEDDING_BIN:-/opt/cuwacunu_embedding/build/baseline/embedding}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/output/runs/baseline}"
config_args=(--config code/encoders/mtf_jepa_mae_vicreg/config/evaluation.conf)
for argument in "$@"; do
  case "$argument" in
    --config|--checkpoint) config_args=() ;;
    --output) echo 'The runner assigns --output; choose EMBEDDING_RUN_ROOT instead.' >&2; exit 1 ;;
  esac
done
mkdir -p -- "$run_root"
run_dir="$(mktemp -d "$run_root/evaluation-XXXXXX")"
printf 'Evaluation artifacts: %s\n' "$run_dir"
"$embedding_bin" evaluate "${config_args[@]}" --output "$run_dir/report.json" "$@"

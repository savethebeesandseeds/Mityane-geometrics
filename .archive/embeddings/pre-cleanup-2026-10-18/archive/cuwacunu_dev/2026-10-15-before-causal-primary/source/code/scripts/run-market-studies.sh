#!/usr/bin/env bash
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Use the managed cuwacunu container.' >&2; exit 1; }
phase=${1:-all}
[[ $# -le 1 && "$phase" =~ ^(all|continuation|capacity)$ ]] || { echo 'Usage: run-market-studies.sh [all|continuation|capacity]' >&2; exit 1; }
source /etc/profile.d/embedding.sh
build_dir=/opt/cuwacunu/build/market-study
mkdir -p "$build_dir/tmp"
export TMPDIR="$build_dir/tmp"
exec 9>"$build_dir/.study.lock"
flock -n 9 || { echo 'A market study is already running.' >&2; exit 1; }
fingerprint() {
  find code -type f \( -name '*.h' -o -name '*.cpp' \) -print0 |
    sort -z | xargs -0 sha256sum | sha256sum
}
before=$(fingerprint)
bash code/scripts/task.sh market-study -j2 build-market-study build-market-readouts
[[ "$before" == "$(fingerprint)" ]] || { echo 'Source changed during compilation; rebuild from frozen source.' >&2; exit 1; }
batch_log=$(mktemp "$build_dir/study-batch-XXXXXX.txt")
declare -A runs
run_encoder() {
  local mode=$1 profile=$2 config=$3 log run
  log=$(mktemp "$build_dir/${mode}-${profile}-XXXXXX.log")
  printf 'Encoder: %s / %s\n' "$mode" "$profile"
  "$build_dir/market_study" "$config" --mode "$mode" --profile "$profile" | tee "$log"
  run=$(sed -n 's/^RUN_DIRECTORY=//p' "$log")
  [[ -d "$run" && -f "$run/summary.json" ]] || { echo "Incomplete run; preserved log: $log" >&2; exit 1; }
  cp --no-clobber -- "$log" "$run/runner.log"
  runs["$mode-$profile"]=$run
  printf '%s\t%s\t%s\n' "$mode" "$profile" "$run" >> "$batch_log"
}
if [[ "$phase" == all || "$phase" == continuation ]]; then
  # Historical pins from the completed pilot replay and committed readout report.
  # These reused final controls are distinct from its initialized-assets ledger.
  sha256sum --check --strict <<'PARENT_PINS'
7d56bc01cbeb39b28f01c8d82397b7260c3f035f3c8c17c7d3a481536c153715  output/runs/market-history-pilot/run-Ova1QQ/features/trained_v22-train.pt
69eb19669e49ddc1dd183aa5ce835227a615912c4feac3b443e6e22dce77da29  output/runs/market-history-pilot/run-Ova1QQ/features/trained_v22-validation.pt
7fb8f5f8093e63600ac9ff44ae3afdd1c45e621ce838a6e35222a30027c0ccce  output/runs/market-history-pilot/run-Ova1QQ/readouts/report.json
PARENT_PINS
  run_encoder continuation base config/market_studies/continuation.conf
  run=${runs[continuation-base]}
  card=$(sha256sum -- doc/MARKET_TRAINING_CONTINUATION_CARD.md); card=${card:0:64}
  manifest=$(sha256sum -- data/mkt-hist-1/manifest.json); manifest=${manifest:0:64}
  printf 'Fixed CPU readouts: continuation2048; saved controls reused.\n'
  "$build_dir/market_fixed_regression" \
    --train-targets "$run/features/train-targets.pt" --validation-targets "$run/features/validation-targets.pt" \
    --method trained2048 "$run/features/trained_v22-train.pt" "$run/features/trained_v22-validation.pt" \
    --output-directory "$run/readouts" --card-sha256 "$card" --input-manifest-sha256 "$manifest" --dataset-id MKT-HIST-1 > "$run/readout.log"
fi
if [[ "$phase" == all || "$phase" == capacity ]]; then
  for profile in base wide large long; do
    run_encoder capacity "$profile" "config/market_studies/capacity_${profile}.conf"
  done
  shared=/cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1
  base_run=${runs[capacity-base]}
  card=$(sha256sum -- doc/MARKET_CAPACITY_STUDY_CARD.md); card=${card:0:64}
  manifest=$(sha256sum -- data/mkt-hist-1/manifest.json); manifest=${manifest:0:64}
  methods=(--method raw_recent128 "$shared/features/raw_recent128-train.pt" "$shared/features/raw_recent128-validation.pt"
           --method raw_full256 "$shared/features/raw_full256-train.pt" "$shared/features/raw_full256-validation.pt")
  for profile in base wide large long; do
    run=${runs[capacity-$profile]}
    methods+=(--method "${profile}_initialized" "$run/features/initialized_v22-train.pt" "$run/features/initialized_v22-validation.pt"
              --method "${profile}_trained" "$run/features/trained_v22-train.pt" "$run/features/trained_v22-validation.pt")
  done
  printf 'Fixed CPU readouts: shared recent128 targets; each method fit once.\n'
  "$build_dir/market_fixed_regression" \
    --train-targets "$shared/features/train-targets.pt" --validation-targets "$shared/features/validation-targets.pt" \
    "${methods[@]}" --output-directory "$base_run/readouts" --card-sha256 "$card" --input-manifest-sha256 "$manifest" --dataset-id MKT-HIST-2 > "$base_run/readout.log"
fi
printf 'Completed study run list: %s\n' "$batch_log"

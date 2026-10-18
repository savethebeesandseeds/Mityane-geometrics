#!/usr/bin/env bash
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Use the managed cuwacunu container.' >&2; exit 1; }
source /etc/profile.d/embedding.sh
build_dir=/opt/cuwacunu/build/market-relative
fixed=/opt/cuwacunu/build/market-study/market_fixed_regression
mkdir -p "$build_dir/tmp"
export TMPDIR="$build_dir/tmp"
exec 9>"$build_dir/.market-relative-changes.lock"
flock -n 9 || { echo 'The v28 relative-changes study is already running.' >&2; exit 1; }
shopt -s nullglob
existing_runs=(output/runs/market-relative-changes/run-*)
[[ ${#existing_runs[@]} -eq 0 ]] || { echo 'Preserve the existing v28 trajectory; this declared training must not be repeated.' >&2; exit 1; }
[[ "$(sha256sum -- doc/MARKET_RELATIVE_CHANGE_CARD.md | cut -d' ' -f1)" == f4378d4a76185fc17b6c0adbadab5af27accd772ed66e5a8bc628ab7848f94a2 ]] || { echo 'Prospective card changed.' >&2; exit 1; }
fingerprint() {
  {
    find code -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.mk' -o -name '*.sh' \) -print0
    printf '%s\0' Makefile config/market_studies/relative_changes.conf config/market_studies/encoder_relative_changes.conf doc/MARKET_RELATIVE_CHANGE_CARD.md
  } | sort -z | xargs -0 sha256sum | sha256sum
}
check_fixed_evaluator() {
  [[ -x "$fixed" ]] || { echo 'Missing preserved fixed readout executable.' >&2; exit 1; }
  [[ "$(sha256sum -- "$fixed" | cut -d' ' -f1)" == 5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14 ]] || { echo 'Fixed evaluator binary changed.' >&2; exit 1; }
}
check_fixed_evaluator
before=$(fingerprint)
build_log=$(mktemp "$build_dir/relative-changes-build-XXXXXX.log")
bash code/scripts/task.sh market-relative -j2 build-market-relative-changes | tee "$build_log"
[[ "$before" == "$(fingerprint)" ]] || { echo 'Source changed during compilation; preserve logs and inspect before proceeding.' >&2; exit 1; }
check_fixed_evaluator
log=$(mktemp "$build_dir/relative-changes-run-XXXXXX.log")
"$build_dir/market_relative_changes" config/market_studies/relative_changes.conf | tee "$log"
run=$(sed -n 's/^RUN_DIRECTORY=//p' "$log")
[[ -d "$run" && -f "$run/summary.json" ]] || { echo "Incomplete run preserved: $log" >&2; exit 1; }
cp --no-clobber -- "$log" "$run/runner.log"
cp --no-clobber -- "$build_log" "$run/build.log"
[[ "$before" == "$(fingerprint)" ]] || { echo 'Source changed during training; preserve artifacts and stop before readouts.' >&2; exit 1; }
shared=/cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1
card=$(sha256sum -- doc/MARKET_RELATIVE_CHANGE_CARD.md); card=${card:0:64}
manifest=$(sha256sum -- data/mkt-hist-1/manifest.json); manifest=${manifest:0:64}
check_fixed_evaluator
printf 'Fixed CPU readouts: new v28 initial/final native36 only; all saved controls reused.\n'
readout_started_ns=$(date +%s%N)
"$fixed" \
  --train-targets "$shared/features/train-targets.pt" --validation-targets "$shared/features/validation-targets.pt" \
  --method initialized_v28 "$run/features/initialized_v28-train.pt" "$run/features/initialized_v28-validation.pt" \
  --method trained_v28 "$run/features/trained_v28-train.pt" "$run/features/trained_v28-validation.pt" \
  --output-directory "$run/readouts" --card-sha256 "$card" --input-manifest-sha256 "$manifest" --dataset-id MKT-HIST-2 > "$run/readout.log"
readout_finished_ns=$(date +%s%N)
readout_elapsed_ns=$((readout_finished_ns - readout_started_ns))
(
  set -o noclobber
  printf '{"kind":"saved_feature_readout_timing_v1","label":"fixed saved-feature fitting/scoring/I/O","elapsed_ns":%s,"started_epoch_ns":%s,"finished_epoch_ns":%s}\n' \
    "$readout_elapsed_ns" "$readout_started_ns" "$readout_finished_ns" > "$run/readout-timing.json"
)
check_fixed_evaluator
[[ "$before" == "$(fingerprint)" ]] || { echo 'Source changed during fixed readouts; preserve evidence and inspect.' >&2; exit 1; }
printf 'Completed RPB-v28.market-01 relative-changes experiment: %s\n' "$run"



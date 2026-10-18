#!/usr/bin/env bash
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Use the managed cuwacunu container.' >&2; exit 1; }
source /etc/profile.d/embedding.sh
# Reuse only unchanged loader/evaluator objects; the candidate has its own binary.
build_dir=/opt/cuwacunu/build/market-study
mkdir -p "$build_dir/tmp"
export TMPDIR="$build_dir/tmp"
exec 9>"$build_dir/.separate-changes.lock"
flock -n 9 || { echo 'A consecutive-change study is already running.' >&2; exit 1; }
fingerprint() {
  find code -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.mk' \) -print0 |
    sort -z | xargs -0 sha256sum | sha256sum
}
before=$(fingerprint)
build_log=$(mktemp "$build_dir/separate-change-build-XXXXXX.log")
bash code/scripts/task.sh market-study -j2 build-market-separate-changes | tee "$build_log"
[[ "$before" == "$(fingerprint)" ]] || { echo 'Source changed during compilation; preserve logs and rebuild.' >&2; exit 1; }
[[ "$(sha256sum "$build_dir/market_fixed_regression" | cut -d' ' -f1)" == "5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14" ]] || { echo 'Fixed evaluator binary changed.' >&2; exit 1; }
[[ -x "$build_dir/market_fixed_regression" ]] || { echo 'Missing existing fixed readout executable.' >&2; exit 1; }
log=$(mktemp "$build_dir/separate-change-run-XXXXXX.log")
"$build_dir/market_separate_changes" config/market_studies/separate_changes.conf | tee "$log"
run=$(sed -n 's/^RUN_DIRECTORY=//p' "$log")
[[ -d "$run" && -f "$run/summary.json" ]] || { echo "Incomplete run preserved: $log" >&2; exit 1; }
cp --no-clobber -- "$log" "$run/runner.log"
cp --no-clobber -- "$build_log" "$run/build.log"
shared=/cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1
card=$(sha256sum -- doc/MARKET_SEPARATE_CHANGE_CARD.md); card=${card:0:64}
manifest=$(sha256sum -- data/mkt-hist-1/manifest.json); manifest=${manifest:0:64}
printf 'Fixed CPU readouts: new initial/final native40 only; all saved controls reused.\n'
"$build_dir/market_fixed_regression" \
  --train-targets "$shared/features/train-targets.pt" --validation-targets "$shared/features/validation-targets.pt" \
  --method initialized_v24 "$run/features/initialized_v24-train.pt" "$run/features/initialized_v24-validation.pt" \
  --method trained_v24 "$run/features/trained_v24-train.pt" "$run/features/trained_v24-validation.pt" \
  --output-directory "$run/readouts" --card-sha256 "$card" --input-manifest-sha256 "$manifest" --dataset-id MKT-HIST-2 > "$run/readout.log"
printf 'Completed separate-block RPB-v24 variation: %s\n' "$run"

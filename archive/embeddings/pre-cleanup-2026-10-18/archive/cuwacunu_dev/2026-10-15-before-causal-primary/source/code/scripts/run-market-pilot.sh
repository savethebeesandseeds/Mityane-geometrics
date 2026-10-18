#!/usr/bin/env bash
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Use the managed cuwacunu container.' >&2; exit 1; }
[[ $# == 0 || ( $# == 2 && "$1" == '--resume' ) ]] || { echo 'Usage: run-market-pilot.sh [--resume training-N.pt]' >&2; exit 1; }
source /etc/profile.d/embedding.sh
source_fingerprint() {
  find code -type f \( -name '*.h' -o -name '*.cpp' \) -print0 |
    sort -z | xargs -0 sha256sum | sha256sum
}
before_build=$(source_fingerprint)
bash code/scripts/task.sh market-pilot -j1 build-market-pilot build-market-readouts
after_build=$(source_fingerprint)
[[ "$before_build" == "$after_build" ]] || { echo 'Source changed during compilation; rebuild from frozen source before running.' >&2; exit 1; }
build_dir=/opt/cuwacunu/build/market-pilot
export TMPDIR="$build_dir/tmp"
exec 9>"$build_dir/.pilot.lock"
flock -n 9 || { echo 'A market pilot is already running.' >&2; exit 1; }
log=$(mktemp "$build_dir/launch-XXXXXX.log")
"$build_dir/market_history_pilot" config/market_history_pilot.conf "$@" | tee "$log"
run=$(sed -n 's/^RUN_DIRECTORY=//p' "$log")
[[ -d "$run" && -f "$run/summary.json" ]] || { echo "Pilot did not complete; log retained at $log" >&2; exit 1; }
card_hash=$(sha256sum -- "$run/prospective-card.md"); card_hash=${card_hash:0:64}
manifest_hash=$(sha256sum -- "$run/input-manifest.json"); manifest_hash=${manifest_hash:0:64}
"$build_dir/market_fixed_regression" \
  --train-targets "$run/features/train-targets.pt" \
  --validation-targets "$run/features/validation-targets.pt" \
  --method raw_mask "$run/features/raw_mask-train.pt" "$run/features/raw_mask-validation.pt" \
  --method initialized_v22 "$run/features/initialized_v22-train.pt" "$run/features/initialized_v22-validation.pt" \
  --method trained_v22 "$run/features/trained_v22-train.pt" "$run/features/trained_v22-validation.pt" \
  --output-directory "$run/readouts" --card-sha256 "$card_hash" --input-manifest-sha256 "$manifest_hash" > "$run/readout.log"
cp --no-clobber -- "$log" "$run/runner.log"
printf 'Completed pilot and fixed readouts: %s\n' "$run"

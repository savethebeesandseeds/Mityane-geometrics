#!/usr/bin/env bash
# Exercise the current scientific readers against artificial SOURCE only.
# The visible-v2 fixture requires its record beside its own source: use a new
# byte-for-byte copy and the exact pinned archived module profile under /opt.
set -euo pipefail
[[ -f /.dockerenv && "$PWD" == /mityane ]] || {
  echo 'Use container.ps1 and the root reader source-fixture target.' >&2; exit 1;
}
[[ $# == 1 ]] || { echo 'One named /opt fixture directory required.' >&2; exit 1; }
task_tmp=$(realpath -m -- "$1")
case "$task_tmp" in /opt/mityane-geometrics/*) ;; *) exit 1;; esac
mkdir -p -- "$task_tmp"
[[ $(realpath -e -- "$task_tmp") == "$task_tmp" ]]
export TMPDIR="$task_tmp"
work=$(mktemp -d "$task_tmp/reader-source-fixtures.XXXXXX")
reader=code/embeddings/evaluation/protocols/visible_difference_v1/reader-v2
frozen=.archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/protocols/visible_difference_v1/reader-v2
sha256sum -- "$reader/source_fixtures.py" "$reader/validate_visible_difference.py" \
  "$frozen/saved_cpu_math.py" "$frozen/archive_codec.py" "$frozen/legacy_implementation.py" \
  >"$work/input-source.sha256"
for mode in ordinary optimized; do
  destination="$work/$mode/$reader"
  mkdir -p -- "$destination"
  cp -- "$reader/source_fixtures.py" "$reader/validate_visible_difference.py" "$destination/"
  cp -- "$frozen/saved_cpu_math.py" "$frozen/archive_codec.py" "$frozen/legacy_implementation.py" "$destination/"
  cmp -- "$reader/source_fixtures.py" "$destination/source_fixtures.py"
  cmp -- "$reader/validate_visible_difference.py" "$destination/validate_visible_difference.py"
  validator_sha=$(sha256sum -- "$destination/validate_visible_difference.py")
  validator_sha=${validator_sha%% *}
  if [[ "$mode" == ordinary ]]; then
    (
      cd -- "$work/$mode"
      python3 -B code/embeddings/evaluation/protocols/visible_difference_v1/reader-v2/source_fixtures.py \
        --validator-sha256 "$validator_sha" --output "$destination/source-fixtures.json" >"$work/visible-source-fixtures.stdout.json"
    )
  else
    (
      cd -- "$work/$mode"
      python3 -B -O code/embeddings/evaluation/protocols/visible_difference_v1/reader-v2/source_fixtures.py \
        --validator-sha256 "$validator_sha" --output "$destination/source-fixtures.json" >"$work/visible-source-fixtures-optimized.stdout.json"
    )
  fi
done
python3 -B code/embeddings/evaluation/protocols/visible_difference_v1/reader-v2/validate_visible_difference.py --self-test >"$work/visible-self-test.json"
python3 -B -O code/embeddings/evaluation/protocols/visible_difference_v1/reader-v2/validate_visible_difference.py --self-test >"$work/visible-self-test-optimized.json"
python3 -B code/embeddings/evaluation/protocols/structured_hard_timing_v1/reader/validate_structured_hard_timing.py \
  --self-test --fixture-output "$work/structured-source-fixtures.json" >"$work/structured-source-fixtures.stdout.json"
python3 -B -O code/embeddings/evaluation/protocols/structured_hard_timing_v1/reader/validate_structured_hard_timing.py \
  --self-test --fixture-output "$work/structured-source-fixtures-optimized.json" >"$work/structured-source-fixtures-optimized.stdout.json"
sha256sum --check --strict "$work/input-source.sha256"
printf 'Artificial reader receipts retained at %s\n' "$work"

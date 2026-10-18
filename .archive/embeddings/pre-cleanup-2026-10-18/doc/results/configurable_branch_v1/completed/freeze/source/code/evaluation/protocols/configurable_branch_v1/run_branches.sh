#!/usr/bin/env bash
# Saved-coordinate diagnostic; preserve original evidence and named containers.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
[[ -f /.dockerenv ]] || exit 1
protocol=code/evaluation/protocols/configurable_branch_v1
card=code/evaluation/cards/configurable_branch_v1.md
original=/embedding/output/runs/rpb-v20-capacity/admission-0gN4uD
build=/opt/cuwacunu_embedding/build/rpb-v20-branches
run_root=/embedding/output/runs/rpb-v20-branches
mkdir -p -- "$run_root"
capsule=$(mktemp -d "$run_root/diagnostic-XXXXXX")
printf 'BRANCH_DIAGNOSTIC %s\n' "$capsule"
run_timed() {
  local stage=$1
  shift
  TIMEFORMAT='%3R'
  { time "$@" >"$capsule/$stage.log" 2>&1; } 2>"$capsule/$stage.seconds"
}
bash code/scripts/task.sh rpb-v20-branches -j2 configurable-branch >"$capsule/build.log" 2>&1
exec 9>"$build/.task.lock"
flock -n 9 || { printf '%s\n' 'Branch build session is busy.' >&2; exit 1; }
python3 -B "$protocol/capsule.py" freeze "$capsule" --original "$original" --build "$build" >"$capsule/freeze.log"
run_timed audit-test python3 -B "$protocol/audit_test.py"
run_timed audit-test-optimized python3 -B -O "$protocol/audit_test.py"
run_timed engineering "$build/embedding_configurable_branch_readout" --self-test "$capsule/engineering"
(cd -- "$original"; sha256sum -c artifacts.sha256) >"$capsule/original-before-diagnostic.log"
run_timed readout "$build/embedding_configurable_branch_readout" --capsule "$original" --output "$capsule/readouts" \
  --card "/embedding/$card" --sources-sha256 "$capsule/freeze/sources.sha256"
run_timed arithmetic-audit "$build/embedding_configurable_branch_readout" --audit-only "$capsule/readouts" \
  --audit-output "$capsule/arithmetic-audit.json"
run_timed publish python3 -B "$protocol/publish.py" --capsule "$original" --results "$capsule/readouts" \
  --card "$capsule/freeze/card.md" --output-directory "$capsule/published"
(cd -- "$original"; sha256sum -c artifacts.sha256) >"$capsule/original-after-diagnostic.log"
sha256sum -c "$capsule/freeze/sources.sha256" >"$capsule/source-after-diagnostic.log"
sha256sum -c "$capsule/freeze/binaries.sha256" >"$capsule/binary-after-diagnostic.log"
python3 -B "$protocol/capsule.py" finish "$capsule"

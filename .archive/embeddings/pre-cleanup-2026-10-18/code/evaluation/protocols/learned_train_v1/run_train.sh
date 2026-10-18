#!/usr/bin/env bash
set -euo pipefail
cd /embedding
test -f /.dockerenv
session=/opt/cuwacunu_embedding/build/rpb-v20-learned-train
mkdir -p "$session" output/runs/rpb-v20-learned-train
exec 9>"$session/.task.lock"
flock -n 9
capsule=$(mktemp -d output/runs/rpb-v20-learned-train/diagnostic-XXXXXX)
capsule=/embedding/$capsule
printf 'LEARNED_TRAIN_DIAGNOSTIC %s\n' "$capsule"
protocol=code/evaluation/protocols/learned_train_v1
python3 -B "$protocol/analysis.py" --freeze "$capsule" >"$capsule/freeze.log" 2>&1
python3 -B "$protocol/test.py" >"$capsule/tests.log" 2>&1
python3 -O -B "$protocol/test.py" >"$capsule/tests-optimized.log" 2>&1
python3 -B "$protocol/analysis.py" --run "$capsule" >"$capsule/run.log" 2>&1
python3 -B "$protocol/analysis.py" --seal "$capsule"
cat "$capsule/compact.json"

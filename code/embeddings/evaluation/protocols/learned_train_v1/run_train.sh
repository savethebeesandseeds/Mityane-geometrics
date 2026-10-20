#!/usr/bin/env bash
set -euo pipefail
cd /mityane
test -f /.dockerenv
session=/opt/mityane-geometrics/embeddings/build/rpb-v20-learned-train
mkdir -p "$session" .temp/embeddings/runs/rpb-v20-learned-train
exec 9>"$session/.task.lock"
flock -n 9
capsule=$(mktemp -d .temp/embeddings/runs/rpb-v20-learned-train/diagnostic-XXXXXX)
capsule=/mityane/$capsule
printf 'LEARNED_TRAIN_DIAGNOSTIC %s\n' "$capsule"
protocol=code/embeddings/evaluation/protocols/learned_train_v1
python3 -B "$protocol/analysis.py" --freeze "$capsule" >"$capsule/freeze.log" 2>&1
python3 -B "$protocol/test.py" >"$capsule/tests.log" 2>&1
python3 -O -B "$protocol/test.py" >"$capsule/tests-optimized.log" 2>&1
python3 -B "$protocol/analysis.py" --run "$capsule" >"$capsule/run.log" 2>&1
python3 -B "$protocol/analysis.py" --seal "$capsule"
cat "$capsule/compact.json"

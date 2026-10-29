#!/bin/bash
# Fresh January timeframe screen. Development only. Does not promote.
set -euo pipefail
ROOT=/mityane/.temp/embeddings/runs/timeframe-input-2026-10-29
BUILD=/opt/mityane-geometrics/embeddings/build/timeframe-input-2026-10-29
DATA=$ROOT/data/log/calibration-TRAIN.pt
CPB_SETTINGS=/mityane/code/data/market/experiments/timeframe_completed_bar_v1/cpb.conf
RPB_SETTINGS=/mityane/code/data/market/experiments/timeframe_completed_bar_v1/rpb.conf
LOG=$ROOT/driver.log
mkdir -p "$ROOT/cpb" "$ROOT/rpb"
exec >>"$LOG" 2>&1
echo "TIMEFRAME_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
START=$(cut -d. -f1 /proc/uptime)
health() {
  python3 - "$1" <<'PY'
import json, math, sys
path = sys.argv[1]
bad = False
last = None
with open(path) as handle:
    for line in handle:
        row = json.loads(line)
        if not row.get("updated"):
            continue
        loss = float(row["loss"])
        grad = float(row["gradient_norm"])
        last = (loss, grad, row["completed_updates"])
        if not math.isfinite(loss) or not math.isfinite(grad) or loss > 100 or grad > 1e6:
            print("BAD", loss, grad, flush=True)
            bad = True
            break
if bad:
    sys.exit(2)
if last:
    print("HEALTH", last[2], last[0], last[1], flush=True)
PY
}
run_one() {
  local family="$1" seed="$2"
  local now elapsed
  now=$(cut -d. -f1 /proc/uptime)
  elapsed=$((now - START))
  if [ "$elapsed" -lt 0 ]; then elapsed=0; fi
  if [ "$elapsed" -gt 28800 ]; then
    echo "TIMEFRAME_FAILED budget ${elapsed}s"
    exit 3
  fi
  local out="$ROOT/$family/$seed"
  if [ -d "$out" ]; then
    echo "TIMEFRAME_FAILED output exists $out"
    exit 4
  fi
  echo "SEED_START $family $seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local status=0
  if [ "$family" = cpb ]; then
    "$BUILD/cpb_family_train" --settings "$CPB_SETTINGS" --dataset "$DATA" --output "$out" \
      --steps 2048 --max-attempts 4096 --batch-size 8 --seed "$seed" \
      --row-seed $((seed + 1000)) --mask-seed $((seed + 2000)) --context-seed $((seed + 3000)) || status=$?
  else
    "$BUILD/rpb_family_train" --settings "$RPB_SETTINGS" --dataset "$DATA" --output "$out" \
      --steps 2048 --max-attempts 4096 --batch-size 8 --initialization-seed "$seed" \
      --row-seed $((seed + 1000)) --mask-seed $((seed + 2000)) --context-seed $((seed + 3000)) || status=$?
  fi
  if [ "$status" -ne 0 ]; then
    echo "TIMEFRAME_FAILED train $family $seed status $status"
    exit 5
  fi
  if ! health "$out/steps.jsonl"; then
    echo "TIMEFRAME_FAILED health $family $seed"
    exit 2
  fi
  echo "SEED_DONE $family $seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
for seed in 948201 948202 948203 948204 948205 948206 948207 948208 948209; do
  run_one cpb "$seed"
done
for seed in 948201 948202 948203 948204 948205 948206 948207 948208 948209; do
  run_one rpb "$seed"
done
echo "TIMEFRAME_TRAIN_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCORE=$ROOT/scores
mkdir -p "$SCORE"
for family in cpb rpb; do
  for seed in 948201 948202 948203 948204 948205 948206 948207 948208 948209; do
    settings=$CPB_SETTINGS
    if [ "$family" = rpb ]; then settings=$RPB_SETTINGS; fi
    "$BUILD/timeframe_score" --family "$family" --settings "$settings" \
      --checkpoint "$ROOT/$family/$seed/inference-final.pt" \
      --features-train "$ROOT/data/log/probe-TRAIN.pt" \
      --features-validation "$ROOT/data/log/probe-VALIDATION.pt" \
      --targets-train "$ROOT/data/raw/probe-TRAIN.pt" \
      --targets-validation "$ROOT/data/raw/probe-VALIDATION.pt" \
      --output "$SCORE/$family-$seed.json" --expected-updates 2048
  done
done
python3 - "$SCORE" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
names = ["past_volatility", "past_trend", "past_activity", "past_relation"]
print("TIMEFRAME_SCORES")
for family in ("cpb", "rpb"):
    rows = []
    for path in sorted(root.glob(f"{family}-*.json")):
        rows.append(json.loads(path.read_text()))
    print(family, "seeds", len(rows))
    by_task = {name: [] for name in names}
    for row in rows:
        for task in row["tasks"]:
            by_task[task["task"]].append(task)
    for name in names:
        tasks = by_task[name]
        scored = [task for task in tasks if task["scored"]]
        coverage = tasks[0]["coverage"] if tasks else None
        eligible = tasks[0]["validation_eligible"] if tasks else None
        total = tasks[0]["validation_rows"] if tasks else None
        if not scored:
            print(f"  {name} abstained coverage={coverage} eligible={eligible}/{total}")
            continue
        values = sorted(task["r2"] for task in scored)
        mid = values[len(values)//2]
        print(f"  {name} median_r2={mid:.6f} n={len(values)} min={values[0]:.6f} max={values[-1]:.6f} coverage={coverage} eligible={eligible}/{total}")
PY
echo "TIMEFRAME_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

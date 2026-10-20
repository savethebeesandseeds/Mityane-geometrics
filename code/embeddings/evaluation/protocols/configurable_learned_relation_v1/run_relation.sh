#!/usr/bin/env bash
# Additive candidate experiment in the existing managed container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
[[ -f /.dockerenv && "$PWD" == /mityane ]] || exit 1
protocol=code/embeddings/evaluation/protocols/configurable_learned_relation_v1
config=code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v21/config/learned_ordered_relation.conf
build=/opt/mityane-geometrics/embeddings/build/rpb-v21-learned-relation
run_root=/mityane/.temp/embeddings/runs/rpb-v21-learned-relation
original=/mityane/code/embeddings/output/runs/rpb-v20-capacity/admission-0gN4uD
branch=/mityane/code/embeddings/output/runs/rpb-v20-branches/diagnostic-YA3Nws
[[ $# -ge 1 && $# -le 2 && ( "$1" == admit || "$1" == run ) ]] || {
  printf '%s\n' 'Usage: run_relation.sh admit | run ADMITTED_CAPSULE' >&2; exit 1;
}
mode=$1
run_timed() {
  local stage=$1
  shift
  TIMEFORMAT='%3R'
  { time "$@" >"$capsule/$stage.log" 2>&1; } 2>"$capsule/$stage.seconds"
}
lock_session() {
  exec 9>"$build/.task.lock"
  flock -n 9 || { printf '%s\n' 'Relation build session is busy.' >&2; exit 1; }
}
if [[ "$mode" == admit ]]; then
  [[ $# == 1 ]]
  mkdir -p -- "$run_root"
  capsule=$(mktemp -d "$run_root/admission-XXXXXX")
  printf 'LEARNED_RELATION_ADMISSION %s\n' "$capsule"
  bash code/scripts/task.sh rpb-v21-learned-relation -j2 configurable-learned-relation rpb-v21 >"$capsule/build.log" 2>&1
  lock_session
  bash code/scripts/node.sh "$protocol/capsule.mjs" freeze "$capsule" --build "$build" --original "$original" --branch "$branch" >"$capsule/freeze.log"
  run_timed gate-test python3 -B "$protocol/gate_test.py"
  run_timed gate-test-optimized python3 -B -O "$protocol/gate_test.py"
  run_timed publisher-test python3 -B "$protocol/publish_test.py"
  run_timed publisher-test-optimized python3 -B -O "$protocol/publish_test.py"
  bash code/scripts/node.sh code/scripts/verify-libtorch.mjs --check >"$capsule/sdk-proof.json"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader >"$capsule/gpu-proof.csv"
  g++ --version >"$capsule/compiler-proof.txt"
  run_timed model-engineering "$build/code/tests/encoders/raw_patch_bottleneck_mae/learned_ordered_relation/learned_ordered_relation_test" "$config"
  run_timed runner-engineering "$build/embedding_configurable_learned_relation" --engineering "$capsule/runner-engineering" --config "$config"
  run_timed readout-engineering "$build/embedding_configurable_learned_relation_readout" --self-test "$capsule/readout-engineering"
  sha256sum -c "$capsule/freeze/sources.sha256" >"$capsule/source-after-engineering.log"
  sha256sum -c "$capsule/freeze/binaries.sha256" >"$capsule/binary-after-engineering.log"
  source_sha=$(sha256sum "$capsule/freeze/sources.sha256" | cut -d ' ' -f1)
  card_sha=$(sha256sum "$capsule/freeze/card.md" | cut -d ' ' -f1)
  printf '{"protocol":"configurable-learned-relation-v1","status":"admitted","quality_encoder_runs":0,"source_fingerprint":"%s","card_sha256":"%s"}\n' "$source_sha" "$card_sha" >"$capsule/admission.json"
  bash code/scripts/node.sh "$protocol/capsule.mjs" bind-quality "$capsule" >"$capsule/quality-admission.log"
  bash code/scripts/node.sh "$protocol/capsule.mjs" verify-admission "$capsule" >"$capsule/admission-check.log"
  printf 'LEARNED_RELATION_ADMITTED %s\n' "$capsule"
  exit 0
fi
[[ $# == 2 ]]
capsule=$2
[[ "$capsule" == /mityane/.temp/embeddings/runs/rpb-v21-learned-relation/admission-* && $(realpath -e -- "$capsule") == "$capsule" && ! -L "$capsule" ]]
[[ -f "$capsule/admission.json" && ! -e "$capsule/candidates" && ! -e "$capsule/screen" && ! -e "$capsule/complete.json" ]]
lock_session
bash code/scripts/node.sh "$protocol/capsule.mjs" verify-admission "$capsule" >"$capsule/admission-before-quality.log"
sha256sum -c "$capsule/freeze/sources.sha256" >"$capsule/source-before-quality.log"
sha256sum -c "$capsule/freeze/binaries.sha256" >"$capsule/binary-before-quality.log"
(cd -- "$original"; sha256sum -c artifacts.sha256) >"$capsule/original-before-quality.log"
(cd -- "$branch"; sha256sum -c artifacts.sha256) >"$capsule/branch-before-quality.log"
source_sha=$(sha256sum "$capsule/freeze/sources.sha256" | cut -d ' ' -f1)
card_sha=$(sha256sum "$capsule/freeze/card.md" | cut -d ' ' -f1)
mkdir -- "$capsule/candidates"
train_master() {
  local master=$1
  printf 'CUDA_LEARNED_RELATION master=%s\n' "$master"
  run_timed "candidate-master-$master" "$build/embedding_configurable_learned_relation" \
    --data-root "$original/quality-data" --inputs-sha256 "$capsule/freeze/inputs.sha256" \
    --original-capsule "$original" --output "$capsule/candidates/master-$master" \
    --profile A --master "$master" --config "$capsule/freeze/config.conf" \
    --card "$capsule/freeze/card.md" --card-sha256 "$card_sha" --source-id "$source_sha"
}
evaluate_stage() {
  local stage=$1
  mkdir -- "$capsule/$stage"
  run_timed "$stage-readout" "$build/embedding_configurable_learned_relation_readout" \
    --original "$original" --branch "$branch" --producer-root "$capsule" \
    --output "$capsule/$stage/readouts" --stage "$stage" --card "$capsule/freeze/card.md" \
    --sources-sha256 "$capsule/freeze/sources.sha256"
  run_timed "$stage-publish" python3 -B "$protocol/publish.py" --capsule "$capsule" \
    --readouts "$capsule/$stage/readouts" --output-directory "$capsule/$stage/published"
}
train_master 941101
train_master 941102
evaluate_stage screen
if bash code/scripts/node.sh code/embeddings/evaluation/protocols/node/v1/gate_status.mjs "$capsule/screen/published/summary.json"; then
  bash code/scripts/node.sh "$protocol/capsule.mjs" admit-third "$capsule" >"$capsule/third-admission.log"
  train_master 941103
  evaluate_stage third
else
  printf '%s\n' 'The prospective joint screen failed; retain both cohorts and stop this frozen candidate.'
fi
(cd -- "$original"; sha256sum -c artifacts.sha256) >"$capsule/original-after-quality.log"
(cd -- "$branch"; sha256sum -c artifacts.sha256) >"$capsule/branch-after-quality.log"
sha256sum -c "$capsule/freeze/sources.sha256" >"$capsule/source-after-quality.log"
sha256sum -c "$capsule/freeze/binaries.sha256" >"$capsule/binary-after-quality.log"
bash code/scripts/node.sh "$protocol/capsule.mjs" finish "$capsule"

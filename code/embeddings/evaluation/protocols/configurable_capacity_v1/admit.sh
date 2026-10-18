#!/usr/bin/env bash
# New output capsule for every attempt; never overwrite engineering or quality evidence.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
[[ -f /.dockerenv ]] || exit 1
protocol=code/embeddings/evaluation/protocols/configurable_capacity_v1
build=/opt/cuwacunu_embedding/build/rpb-v20-capacity
run_root=/embedding/output/runs/rpb-v20-capacity
config=code/embeddings/encoders/raw_patch_bottleneck_mae/config/configurable_spectral.conf
mkdir -p -- "$run_root"
capsule=$(mktemp -d "$run_root/admission-XXXXXX")
printf 'CAPACITY_ADMISSION %s\n' "$capsule"
bash code/scripts/task.sh rpb-v20-capacity -j2 configurable-capacity >"$capsule/build.log" 2>&1
exec 9>"$build/.task.lock"
flock -n 9 || { printf '%s\n' 'Capacity build session is busy.' >&2; exit 1; }
python3 -B "$protocol/capsule.py" freeze "$capsule" --build "$build" >"$capsule/freeze.log"
python3 -B "$protocol/capsule_runtime_test.py" >"$capsule/runtime-test.log" 2>&1
python3 -B -O "$protocol/capsule_runtime_test.py" >"$capsule/runtime-test-optimized.log" 2>&1
python3 -B "$protocol/evaluation_audit_test.py" >"$capsule/evaluation-audit-test.log" 2>&1
python3 -B -O "$protocol/evaluation_audit_test.py" >"$capsule/evaluation-audit-test-optimized.log" 2>&1
python3 -B code/scripts/install-libtorch.py --check >"$capsule/sdk-proof.json"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader >"$capsule/gpu-proof.csv"
g++ --version >"$capsule/compiler-proof.txt"
"$build/code/protocols/configurable_capacity_v1/generator_test" >"$capsule/generator-test.log" 2>&1
"$build/embedding_configurable_capacity_data" --engineering "$capsule/data-engineering" >"$capsule/data-engineering.log" 2>&1
"$build/embedding_configurable_capacity" --engineering "$capsule/runner-engineering" --config "$config" >"$capsule/runner-engineering.log" 2>&1
"$build/embedding_configurable_capacity_readout" --self-test "$capsule/readout-engineering" >"$capsule/readout-engineering.log" 2>&1
sha256sum -c "$capsule/freeze/sources.sha256" >"$capsule/source-after-engineering.log"
sha256sum -c "$capsule/freeze/binaries.sha256" >"$capsule/binary-after-engineering.log"
printf '{"protocol":"configurable-capacity-v1","status":"admitted","quality_encoder_runs":0}\n' >"$capsule/admission.json"
printf 'CAPACITY_ADMITTED %s\n' "$capsule"

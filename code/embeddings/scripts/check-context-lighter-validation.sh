#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'Lighter context admission takes no overrides.' >&2; exit 1; }
session='rpb-paired-pooling'
root="$PWD/output/runs/rpb-context-lighter-validation"
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Lighter context admission: %s\n' "$admission"
mapfile -t production < <(make -s print-context-lighter-sources)
[[ ${#production[@]} -gt 0 ]] || { echo 'Production source list missing.' >&2; exit 1; }
sources=("${production[@]}"
    code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/tests/context_lighter_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v6/tests/context_deletion_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/context_deletion_adapter_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/context_replication_adapter_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/context_replay_adapter_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/training_policy_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/rpb_test_support.h
    code/shared/representation/tests/shared_test_support.h
    code/shared/representation/tests/native_curve_test.cpp
    code/shared/representation/tests/paired_pooling_test.cpp
    code/shared/representation/tests/archive_readout_test.cpp)
mkdir -- "$admission/source"
cp --parents -- "${sources[@]}" "$admission/source/"
sha256sum -- "${production[@]}" > "$admission/production-source-inputs.sha256"
sha256sum -- "${sources[@]}" > "$admission/admission-source-inputs.sha256"
bash code/scripts/task.sh "$session" -j2 -W Makefile -W code/shared/representation/src/paired_pooling.cpp \
    context-lighter-validation test-rpb-context-lighter test-rpb-context-deletion \
    test-rpb-context-replication test-rpb-context-replay test-rpb-training-policy \
    test-native-curve test-paired-pooling test-archive-readout \
    2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/mityane-geometrics/embeddings/build/$session/embedding_context_lighter_validation"
"$binary" --phase gate --output "$admission/cuda-gate" 2>&1 | tee "$admission/cuda-gate.log"
sha256sum -c "$admission/admission-source-inputs.sha256" > "$admission/source-preserved-after.txt"
(cd -- "$admission/source"; sha256sum -c "$admission/admission-source-inputs.sha256") > "$admission/copied-source-preserved-after.txt"
source_id="$("$binary" --source-id)"
[[ "$source_id" == "$(sha256sum "$admission/production-source-inputs.sha256" | cut -d ' ' -f 1)" ]] || {
    echo 'Admission binary differs from preserved production source.' >&2; exit 1;
}
printf '%s\n' "$source_id" > "$admission/source-fingerprint.txt"
python3 - "$admission" "$source_id" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
record = dict(protocol='context-lighter-validation-v1', status='passed', source_fingerprint=sys.argv[2],
    source_preserved=True,
    gates=['actual CUDA .15 training and native32 serving', 'strict declared policy and full initialization pairing',
           'original query/loss support and deterministic repaired context', 'legacy disabled/.30 defaults and replay',
           'tagged ordinary resume rejection', 'development-only completion and default paired protocols',
           'fixed original-TRAIN heads reused across validation views'],
    log_sha256=hashlib.sha256((root/'build-and-tests.log').read_bytes()).hexdigest(),
    cuda_gate_log_sha256=hashlib.sha256((root/'cuda-gate.log').read_bytes()).hexdigest(),
    admission_source_manifest_sha256=hashlib.sha256((root/'admission-source-inputs.sha256').read_bytes()).hexdigest())
with (root/'passed.json').open('x', encoding='utf-8') as output:
    json.dump(record, output, indent=2); output.write('\n')
PY
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Lighter context admission passed: %s\n' "$admission"

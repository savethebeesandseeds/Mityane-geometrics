#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'Context optimization admission takes no overrides.' >&2; exit 1; }
session='rpb-paired-pooling'
root="$PWD/.temp/embeddings/runs/rpb-context-optimization-validation"
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Context optimization admission evidence: %s\n' "$admission"
mapfile -t production < <(make -s print-context-optimization-sources)
[[ ${#production[@]} -gt 0 ]] || { echo 'Production source list missing.' >&2; exit 1; }
sources=("${production[@]}"
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/context_replay_adapter_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v6/tests/context_deletion_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/context_deletion_adapter_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/training_policy_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/rpb_test_support.h
    code/shared/representation/tests/shared_test_support.h
    code/shared/representation/tests/archive_readout_test.cpp
    code/shared/representation/tests/feature_harness_test.cpp
    code/shared/representation/tests/feature_stress_test.cpp
    code/shared/representation/tests/paired_pooling_test.cpp)
mkdir -- "$admission/source"
cp --parents -- "${sources[@]}" "$admission/source/"
sha256sum -- "${production[@]}" > "$admission/production-source-inputs.sha256"
sha256sum -- "${sources[@]}" > "$admission/admission-source-inputs.sha256"
bash code/scripts/task.sh "$session" -j2 -W Makefile -W code/shared/representation/src/paired_pooling.cpp \
    context-optimization-validation test-rpb-context-replay test-rpb-context-deletion \
    test-rpb-training-policy test-archive-readout test-feature-harness test-feature-stress test-paired-pooling \
    2>&1 | tee "$admission/build-and-tests.log"
sha256sum -c "$admission/admission-source-inputs.sha256" > "$admission/source-preserved-after.txt"
(cd -- "$admission/source"; sha256sum -c "$admission/admission-source-inputs.sha256") > "$admission/copied-source-preserved-after.txt"
binary="/opt/mityane-geometrics/embeddings/build/$session/embedding_context_optimization_validation"
source_id="$("$binary" --source-id)"
[[ "$source_id" == "$(sha256sum "$admission/production-source-inputs.sha256" | cut -d ' ' -f 1)" ]] || {
    echo 'Admission binary and frozen production sources differ.' >&2; exit 1;
}
printf '%s\n' "$source_id" > "$admission/source-fingerprint.txt"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs pass-admission context-optimization-validation-v1 "$admission" "$source_id"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Context optimization admission passed: %s\n' "$admission"

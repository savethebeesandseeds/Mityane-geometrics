#!/usr/bin/env bash
# Coordinated admission in the existing container, preserving its exact sources
# and both the enabled-context CUDA contract and ordinary checkpoint checks.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'Context admission takes no recipe overrides.' >&2; exit 1; }
session='rpb-paired-pooling'
root="$PWD/.temp/embeddings/runs/rpb-context-deletion"
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Context admission evidence: %s\n' "$admission"
mapfile -t production < <(make -s print-context-deletion-sources)
[[ ${#production[@]} -gt 0 ]] || { echo 'Production source list missing.' >&2; exit 1; }
sources=("${production[@]}"
    code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v6/tests/context_deletion_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/context_deletion_adapter_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/training_policy_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/preprocessing_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/numerics_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/masking_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/model_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/workflow_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/legacy_checkpoint_parity_test.cpp
    code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/rpb_test_support.h
    code/shared/representation/tests/shared_test_support.h
    code/shared/representation/tests/paired_pooling_test.cpp)
mkdir -- "$admission/source"
cp --parents -- "${sources[@]}" "$admission/source/"
sha256sum -- "${production[@]}" > "$admission/production-source-inputs.sha256"
sha256sum -- "${sources[@]}" > "$admission/admission-source-inputs.sha256"
bash code/scripts/task.sh "$session" -j2 -W Makefile -W code/shared/representation/src/paired_pooling.cpp \
    context-deletion test-rpb-context-deletion test-rpb-training-policy \
    test-paired-pooling test-rpb-mae test-rpb-legacy-checkpoint \
    2>&1 | tee "$admission/build-and-tests.log"
sha256sum -c "$admission/admission-source-inputs.sha256" > "$admission/source-preserved-after.txt"
(cd -- "$admission/source"; sha256sum -c "$admission/admission-source-inputs.sha256") > "$admission/copied-source-preserved-after.txt"
binary="/opt/mityane-geometrics/embeddings/build/$session/embedding_context_deletion"
source_id="$("$binary" --source-id)"
[[ "$source_id" == "$(sha256sum "$admission/production-source-inputs.sha256" | cut -d ' ' -f 1)" ]] || {
    echo 'Admission binary and preserved production sources differ.' >&2; exit 1;
}
printf '%s\n' "$source_id" > "$admission/source-fingerprint.txt"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs pass-admission context-deletion-v1 "$admission" "$source_id"
# Advance only this evidence pointer after every declared check succeeded.
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Context admission passed: %s\n' "$admission"

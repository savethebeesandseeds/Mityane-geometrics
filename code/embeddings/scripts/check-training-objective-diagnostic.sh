#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run frozen admission inside the managed container.' >&2; exit 1; }
session='rpb-paired-pooling'
root="$PWD/.temp/embeddings/runs/rpb-training-objective-diagnostic"
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'TRAIN objective admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-training-objective-sources)
[[ ${#sources[@]} -gt 0 ]] || exit 1
mkdir -- "$admission/source"
cp --parents -- "${sources[@]}" "$admission/source/"
sha256sum -- "${sources[@]}" > "$admission/source-inputs.sha256"
bash code/scripts/task.sh "$session" -j2 -W Makefile \
  training-objective-diagnostic test-rpb-training-objective-diagnostic \
  2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/mityane-geometrics/embeddings/build/$session/embedding_training_objective_diagnostic"
"$binary" --metadata-self-test 2>&1 | tee "$admission/metadata-gate.log"
sha256sum -c "$admission/source-inputs.sha256" > "$admission/source-preserved-after.txt"
(cd -- "$admission/source"; sha256sum -c "$admission/source-inputs.sha256") > "$admission/copied-source-preserved-after.txt"
source_id="$("$binary" --source-id)"
[[ "$source_id" == "$(sha256sum "$admission/source-inputs.sha256" | cut -d ' ' -f 1)" ]] || exit 1
printf '%s\n' "$source_id" > "$admission/source-fingerprint.txt"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs pass-admission training-objective-diagnostic-v1 "$admission" "$source_id"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'TRAIN objective admission passed: %s\n' "$admission"

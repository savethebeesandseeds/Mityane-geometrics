#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run the frozen diagnostic inside the managed container.' >&2; exit 1; }
binary="${TRAINING_OBJECTIVE_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-paired-pooling/embedding_training_objective_diagnostic}"
root="$PWD/.temp/embeddings/runs/rpb-training-objective-diagnostic"
[[ -n "${TRAINING_OBJECTIVE_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze source inputs.' >&2; exit 1; }
mkdir -p -- "$root"
capsule="$(mktemp -d "$root/training-objective-XXXXXX")"
printf 'TRAIN objective capsule: %s\n' "$capsule"
read -r -a sources <<< "$TRAINING_OBJECTIVE_SOURCE_INPUTS"
mkdir -- "$capsule/source"
cp --parents -- "${sources[@]}" "$capsule/source/"
sha256sum -- "${sources[@]}" > "$capsule/source-inputs.sha256"
source_id="$(sha256sum "$capsule/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled/preserved source mismatch.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$capsule/source-fingerprint.txt"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete actual CUDA admission first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs verify-admission training-objective-diagnostic-v1 "$admission" "$root/admission" "$source_id"
cp -a -- "$admission" "$capsule/admission"
card="$PWD/code/embeddings/evaluation/cards/training_objective_diagnostic_v1.md"
card_sha="$(sha256sum "$card" | cut -d ' ' -f 1)"
bash code/scripts/node.sh code/embeddings/scripts/prepare-training-objective-inputs.mjs --freeze --repo-root "$PWD" --historical-parent-root "${MITYANE_HISTORICAL_PARENT_ROOT:?Set the explicit retained parent root}" \
  --output "$capsule/inputs" --frozen-card "$card" --frozen-card-sha256 "$card_sha" > "$capsule/input-freeze.json"
tsv="$capsule/inputs/instances.tsv"
manifest="$capsule/inputs/inputs.sha256"
"$binary" --repo-root "$PWD" --output "$capsule/results" \
  --instances "$tsv" --instances-sha256 "$(sha256sum "$tsv" | cut -d ' ' -f 1)" \
  --input-manifest "$manifest" --input-manifest-sha256 "$(sha256sum "$manifest" | cut -d ' ' -f 1)" \
  --admission-log "$capsule/admission/build-and-tests.log" \
  --admission-log-sha256 "$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)" \
  2>&1 | tee "$capsule/command.log"
sha256sum -c "$manifest" > "$capsule/inputs-preserved-after.txt"
sha256sum -c "$capsule/source-inputs.sha256" > "$capsule/source-preserved-after.txt"
(cd -- "$capsule/source"; sha256sum -c "$capsule/source-inputs.sha256") > "$capsule/copied-source-preserved-after.txt"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs inventory training-objective-diagnostic-v1 "$capsule"
printf 'TRAIN objective diagnostic artifacts: %s\n' "$capsule/results"

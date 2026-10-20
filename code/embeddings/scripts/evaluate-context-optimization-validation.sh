#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'context-optimization-validation-v1 has a fixed recipe; no runner overrides.' >&2; exit 1; }
binary="${CONTEXT_OPTIMIZATION_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-paired-pooling/embedding_context_optimization_validation}"
run_root="$PWD/.temp/embeddings/runs/rpb-context-optimization-validation"
reference="${MITYANE_HISTORICAL_PARENT_ROOT:?Bind the retained historical parent root}/output/runs/rpb-context-deletion/context-deletion-KTbte3"
inventory_sha='62415443de5db9ebe7509d31d79b3bbab8d95efa3bf72b7afde3466b1ae021dd'
[[ "$(sha256sum "$reference/artifact-integrity.json" | cut -d ' ' -f 1)" == "$inventory_sha" ]] || { echo 'Exact parent inventory differs.' >&2; exit 1; }
[[ -n "${CONTEXT_OPTIMIZATION_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze sources.' >&2; exit 1; }
mkdir -p -- "$run_root"
run_directory="$(mktemp -d "$run_root/context-optimization-validation-XXXXXX")"
printf 'Optimization validation capsule: %s\n' "$run_directory"
read -r -a source_inputs <<< "$CONTEXT_OPTIMIZATION_SOURCE_INPUTS"
mkdir -- "$run_directory/source"
cp --parents -- "${source_inputs[@]}" "$run_directory/source/"
sha256sum -- "${source_inputs[@]}" > "$run_directory/source-inputs.sha256"
source_id="$(sha256sum "$run_directory/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source differs from snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$run_directory/source-fingerprint.txt"
[[ -f "$run_root/admission-approved.path" ]] || { echo 'Run check-context-optimization-validation.sh first.' >&2; exit 1; }
admission="$(cat "$run_root/admission-approved.path")"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs verify-admission context-optimization-validation-v1 "$admission" "$run_root/admission" "$source_id"
cp -a -- "$admission" "$run_directory/admission"
cp -- "$reference/artifact-integrity.json" "$run_directory/reference-artifact-integrity.json"
# Inventory metadata only. The allowlist opens no TEST/stress/report payloads.
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs retained-inputs "$run_directory/reference-artifact-integrity.json" "$run_directory/retained-inputs.sha256" candidate-512
(cd -- "$reference" && sha256sum -c "$run_directory/retained-inputs.sha256") > "$run_directory/retained-before.txt"
"$binary" --output "$run_directory/results" --retained-hashes "$run_directory/retained-inputs.sha256" 2>&1 | tee "$run_directory/command.log"
(cd -- "$reference" && sha256sum -c "$run_directory/retained-inputs.sha256") > "$run_directory/retained-after.txt"
[[ "$(sha256sum "$reference/artifact-integrity.json" | cut -d ' ' -f 1)" == "$inventory_sha" ]] || { echo 'Parent inventory changed.' >&2; exit 1; }
printf 'Optimization validation artifacts: %s\n' "$run_directory/results"
sha256sum -c "$run_directory/source-inputs.sha256" > "$run_directory/source-preserved-after.txt"
(cd -- "$run_directory/source"; sha256sum -c "$run_directory/source-inputs.sha256") > "$run_directory/copied-source-preserved-after.txt"

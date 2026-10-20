#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'optimization-validation-v1 has a fixed recipe; no runner overrides.' >&2; exit 1; }
binary="${OPTIMIZATION_DIAGNOSTIC_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-paired-pooling/embedding_optimization_diagnostic}"
run_root="$PWD/.temp/embeddings/runs/rpb-optimization-validation"
reference="${MITYANE_HISTORICAL_PARENT_ROOT:?Bind the retained historical parent root}/output/runs/rpb-paired-pooling/paired-pooling-BFFPX5"
inventory_sha='610976e666468ac2f80404bd56884e36f9275645305e83a9d0e50b762e702be3'
[[ "$(sha256sum "$reference/artifact-integrity.json" | cut -d ' ' -f 1)" == "$inventory_sha" ]] || { echo 'Exact parent inventory differs.' >&2; exit 1; }
[[ -n "${OPTIMIZATION_DIAGNOSTIC_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze sources.' >&2; exit 1; }
mkdir -p -- "$run_root"
run_directory="$(mktemp -d "$run_root/optimization-validation-XXXXXX")"
printf 'Optimization validation capsule: %s\n' "$run_directory"
read -r -a source_inputs <<< "$OPTIMIZATION_DIAGNOSTIC_SOURCE_INPUTS"
mkdir -- "$run_directory/source"
cp --parents -- "${source_inputs[@]}" "$run_directory/source/"
sha256sum -- "${source_inputs[@]}" > "$run_directory/source-inputs.sha256"
source_id="$(sha256sum "$run_directory/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source differs from snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$run_directory/source-fingerprint.txt"
cp -- "$reference/artifact-integrity.json" "$run_directory/reference-artifact-integrity.json"
# Inventory metadata only. The allowlist opens no TEST/stress/report payloads.
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs retained-inputs "$run_directory/reference-artifact-integrity.json" "$run_directory/retained-inputs.sha256" candidate-512
(cd -- "$reference" && sha256sum -c "$run_directory/retained-inputs.sha256") > "$run_directory/retained-before.txt"
"$binary" --output "$run_directory/results" --retained-hashes "$run_directory/retained-inputs.sha256" 2>&1 | tee "$run_directory/command.log"
(cd -- "$reference" && sha256sum -c "$run_directory/retained-inputs.sha256") > "$run_directory/retained-after.txt"
[[ "$(sha256sum "$reference/artifact-integrity.json" | cut -d ' ' -f 1)" == "$inventory_sha" ]] || { echo 'Parent inventory changed.' >&2; exit 1; }
printf 'Optimization validation artifacts: %s\n' "$run_directory/results"

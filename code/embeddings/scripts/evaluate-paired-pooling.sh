#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'paired-pooling-v1 has a fixed recipe; no runner overrides.' >&2; exit 1; }
binary="${PAIRED_POOLING_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-paired-pooling/embedding_paired_pooling}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/.temp/runs/rpb-paired-pooling}"
reference="${MITYANE_HISTORICAL_PARENT_ROOT:?Bind the retained historical parent root}/output/runs/rpb-native-curve/native-curve-5G8O5c"
inventory_sha='c9922d3c817630da3d8609b7ba8d2b47cab7434a17fe28bf5ae272a841da0d95'
[[ "$(sha256sum "$reference/artifact-integrity.json" | cut -d ' ' -f 1)" == "$inventory_sha" ]] || { echo 'Retained reference inventory identity differs.' >&2; exit 1; }
mkdir -p -- "$run_root"
run_directory="$(mktemp -d "$run_root/paired-pooling-XXXXXX")"
printf 'Paired pooling capsule: %s\n' "$run_directory"
[[ -n "${PAIRED_POOLING_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze source inputs.' >&2; exit 1; }
read -r -a source_inputs <<< "$PAIRED_POOLING_SOURCE_INPUTS"
mkdir -- "$run_directory/source"
cp --parents -- "${source_inputs[@]}" "$run_directory/source/"
sha256sum -- "${source_inputs[@]}" > "$run_directory/source-inputs.sha256"
source_id="$(sha256sum "$run_directory/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source differs from preserved snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$run_directory/source-fingerprint.txt"
cp -- "$reference/artifact-integrity.json" "$run_directory/reference-artifact-integrity.json"
# Read only inventory metadata; do not open prior TEST/stress inputs. The C++
# engine verifies every declared retained input against these frozen SHA256s.
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs retained-inputs "$run_directory/reference-artifact-integrity.json" "$run_directory/retained-inputs.sha256" native
"$binary" --output "$run_directory/results" --retained-hashes "$run_directory/retained-inputs.sha256"
printf 'Paired pooling artifacts: %s\n' "$run_directory/results"

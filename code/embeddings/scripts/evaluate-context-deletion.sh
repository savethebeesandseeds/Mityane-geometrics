#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'context-deletion-v1 has a fixed recipe; no runner overrides.' >&2; exit 1; }
binary="${CONTEXT_DELETION_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-context-deletion/embedding_context_deletion}"
run_root="$PWD/.temp/embeddings/runs/rpb-context-deletion"
reference="${MITYANE_HISTORICAL_PARENT_ROOT:?Bind the retained historical parent root}/output/runs/rpb-native-curve/native-curve-5G8O5c"
inventory_sha='c9922d3c817630da3d8609b7ba8d2b47cab7434a17fe28bf5ae272a841da0d95'
[[ "$(sha256sum "$reference/artifact-integrity.json" | cut -d ' ' -f 1)" == "$inventory_sha" ]] || { echo 'Retained reference inventory identity differs.' >&2; exit 1; }
mkdir -p -- "$run_root"
run_directory="$(mktemp -d "$run_root/context-deletion-XXXXXX")"
printf 'Context deletion capsule: %s\n' "$run_directory"
[[ -n "${CONTEXT_DELETION_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze source inputs.' >&2; exit 1; }
read -r -a source_inputs <<< "$CONTEXT_DELETION_SOURCE_INPUTS"
mkdir -- "$run_directory/source"
cp --parents -- "${source_inputs[@]}" "$run_directory/source/"
sha256sum -- "${source_inputs[@]}" > "$run_directory/source-inputs.sha256"
source_id="$(sha256sum "$run_directory/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source differs from preserved snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$run_directory/source-fingerprint.txt"
[[ -f "$run_root/admission-approved.path" ]] || { echo 'Run check-context-deletion.sh before measurement.' >&2; exit 1; }
admission="$(cat "$run_root/admission-approved.path")"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs verify-admission context-deletion-v1 "$admission" "$run_root/admission" "$source_id"
cp -a -- "$admission" "$run_directory/admission"
cp -- "$reference/artifact-integrity.json" "$run_directory/reference-artifact-integrity.json"
# Read only inventory metadata; do not open prior TEST/stress inputs. The C++
# engine verifies every declared retained input against these frozen SHA256s.
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs retained-inputs "$run_directory/reference-artifact-integrity.json" "$run_directory/retained-inputs.sha256" native
"$binary" --output "$run_directory/results" --retained-hashes "$run_directory/retained-inputs.sha256" \
    --context-training-log "$run_directory/admission/build-and-tests.log" \
    --context-training-log-sha256 "$(sha256sum "$run_directory/admission/build-and-tests.log" | cut -d ' ' -f 1)"
printf 'Context deletion artifacts: %s\n' "$run_directory/results"

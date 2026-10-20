#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'context-lighter-validation-v1 has a frozen recipe; no overrides.' >&2; exit 1; }
binary="${CONTEXT_LIGHTER_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-paired-pooling/embedding_context_lighter_validation}"
root="$PWD/.temp/embeddings/runs/rpb-context-lighter-validation"
parent="${MITYANE_HISTORICAL_PARENT_ROOT:?Bind the retained historical parent root}/output/runs/rpb-context-replication/context-replication-JjNEUc"
[[ -n "${CONTEXT_LIGHTER_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze source inputs.' >&2; exit 1; }
mkdir -p -- "$root"
capsule="$(mktemp -d "$root/lighter-validation-XXXXXX")"
printf 'Lighter context validation capsule: %s\n' "$capsule"
read -r -a sources <<< "$CONTEXT_LIGHTER_SOURCE_INPUTS"
mkdir -- "$capsule/source"
cp --parents -- "${sources[@]}" "$capsule/source/"
sha256sum -- "${sources[@]}" > "$capsule/source-inputs.sha256"
source_id="$(sha256sum "$capsule/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source differs from preserved snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$capsule/source-fingerprint.txt"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete check-context-lighter-validation.sh first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs verify-admission context-lighter-validation-v1 "$admission" "$root/admission" "$source_id"
cp -a -- "$admission" "$capsule/admission"
admission_log="$capsule/admission/build-and-tests.log"
admission_log_sha="$(sha256sum "$admission_log" | cut -d ' ' -f 1)"
# This command emits only compiled role metadata. The exact list is frozen
# before opening or hashing any permitted parent payload.
"$binary" --parent-roles > "$capsule/parent-role-plan.json"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs freeze-parents context-lighter-validation-v1 "${MITYANE_HISTORICAL_PARENT_ROOT:?Bind the preserved output/runs physical root explicitly}" "$capsule"
(cd -- "$parent"; sha256sum -c "$capsule/parent-inputs.sha256") > "$capsule/parent-preserved-before.txt"
"$binary" --phase measure --output "$capsule/results" \
    --retained-hashes "$capsule/parent-inputs.sha256" \
    --admission-log "$admission_log" --admission-log-sha256 "$admission_log_sha" \
    2>&1 | tee "$capsule/command.log"
(cd -- "$parent"; sha256sum -c "$capsule/parent-inputs.sha256") > "$capsule/parent-preserved-after.txt"
sha256sum -c "$capsule/source-inputs.sha256" > "$capsule/source-preserved-after.txt"
(cd -- "$capsule/source"; sha256sum -c "$capsule/source-inputs.sha256") > "$capsule/copied-source-preserved-after.txt"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs inventory context-lighter-validation-v1 "$capsule"
printf 'Lighter context validation artifacts: %s\n' "$capsule/results"

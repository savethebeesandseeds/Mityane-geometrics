#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'context-replication-v1 has a frozen recipe; no overrides.' >&2; exit 1; }
binary="${CONTEXT_REPLICATION_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-paired-pooling/embedding_context_replication}"
root="$PWD/.temp/embeddings/runs/rpb-context-replication"
[[ -n "${CONTEXT_REPLICATION_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze source inputs.' >&2; exit 1; }
mkdir -p -- "$root"
capsule="$(mktemp -d "$root/context-replication-XXXXXX")"
printf 'Context replication capsule: %s\n' "$capsule"
read -r -a sources <<< "$CONTEXT_REPLICATION_SOURCE_INPUTS"
mkdir -- "$capsule/source"
cp --parents -- "${sources[@]}" "$capsule/source/"
sha256sum -- "${sources[@]}" > "$capsule/source-inputs.sha256"
source_id="$(sha256sum "$capsule/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source differs from preserved snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$capsule/source-fingerprint.txt"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete check-context-replication.sh first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs verify-admission context-replication-v1 "$admission" "$root/admission" "$source_id"
cp -a -- "$admission" "$capsule/admission"
admission_log="$capsule/admission/build-and-tests.log"
admission_log_sha="$(sha256sum "$admission_log" | cut -d ' ' -f 1)"
"$binary" --phase prepare --output "$capsule/reference" \
    --admission-log "$admission_log" --admission-log-sha256 "$admission_log_sha" \
    2>&1 | tee "$capsule/reference-command.log"
# This new preparation contains TRAIN/VALIDATION only. Freeze every file before
# any candidate fit or fresh TEST generation; no historical payload is read.
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs freeze-reference "$capsule/reference" "$capsule/reference-inputs.sha256" "$capsule/reference-inputs.json"
(cd -- "$capsule/reference"; sha256sum -c "$capsule/reference-inputs.sha256") > "$capsule/reference-preserved-before.txt"
"$binary" --phase compare --output "$capsule/results" --reference "$capsule/reference" \
    --retained-hashes "$capsule/reference-inputs.sha256" --admission-log "$admission_log" \
    --admission-log-sha256 "$admission_log_sha" 2>&1 | tee "$capsule/command.log"
(cd -- "$capsule/reference"; sha256sum -c "$capsule/reference-inputs.sha256") > "$capsule/reference-preserved-after.txt"
sha256sum -c "$capsule/source-inputs.sha256" > "$capsule/source-preserved-after.txt"
(cd -- "$capsule/source"; sha256sum -c "$capsule/source-inputs.sha256") > "$capsule/copied-source-preserved-after.txt"
printf 'Context replication artifacts: %s\n' "$capsule/results"

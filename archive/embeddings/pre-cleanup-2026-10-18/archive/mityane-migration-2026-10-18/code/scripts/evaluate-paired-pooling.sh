#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'paired-pooling-v1 has a fixed recipe; no runner overrides.' >&2; exit 1; }
binary="${PAIRED_POOLING_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_paired_pooling}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/output/runs/rpb-paired-pooling}"
reference="$PWD/output/runs/rpb-native-curve/native-curve-5G8O5c"
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
python3 - "$run_directory/reference-artifact-integrity.json" "$run_directory/retained-inputs.sha256" <<'PY'
import json, sys
from pathlib import PurePosixPath
with open(sys.argv[1], encoding='utf-8') as stream:
    inventory = json.load(stream)
with open(sys.argv[2], 'x', encoding='utf-8', newline='\n') as target:
    for entry in inventory['files']:
        path = PurePosixPath(entry['path'])
        parts = path.parts
        if str(path) in {'results/native-curve-card.json', 'results/selection.json'}:
            target.write(f"{entry['sha256']}  {path}\n")
            continue
        if len(parts) < 3 or parts[0] != 'results' or parts[1] not in {'seed-3101-lag_sign', 'seed-3202-lag_sign', 'seed-3303-lag_sign'}:
            continue
        if any('testing' in part or 'stress' in part or 'selected-test' in part for part in parts):
            continue
        if any(part in {'milestone-128', 'milestone-2048'} for part in parts):
            continue
        target.write(f"{entry['sha256']}  {path}\n")
PY
"$binary" --output "$run_directory/results" --retained-hashes "$run_directory/retained-inputs.sha256"
printf 'Paired pooling artifacts: %s\n' "$run_directory/results"

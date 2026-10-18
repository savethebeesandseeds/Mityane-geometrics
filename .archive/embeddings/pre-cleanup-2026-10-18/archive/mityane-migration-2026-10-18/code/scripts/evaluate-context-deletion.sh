#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'context-deletion-v1 has a fixed recipe; no runner overrides.' >&2; exit 1; }
binary="${CONTEXT_DELETION_BIN:-/opt/cuwacunu_embedding/build/rpb-context-deletion/embedding_context_deletion}"
run_root="$PWD/output/runs/rpb-context-deletion"
reference="$PWD/output/runs/rpb-native-curve/native-curve-5G8O5c"
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
python3 - "$admission" "$run_root/admission" "$source_id" <<'PY'
import hashlib, json, sys
from pathlib import Path
admission = Path(sys.argv[1]).resolve(strict=True)
assert admission.parent == Path(sys.argv[2]).resolve(strict=True) and admission.name.startswith('admission-')
record = json.loads((admission/'passed.json').read_text())
assert record['protocol'] == 'context-deletion-v1' and record['status'] == 'passed'
assert record['source_fingerprint'] == sys.argv[3] and record['source_preserved'] is True
assert hashlib.sha256((admission/'build-and-tests.log').read_bytes()).hexdigest() == record['log_sha256']
assert 'RPB context-deletion CUDA/default-parity tests passed' in (admission/'build-and-tests.log').read_text()
assert hashlib.sha256((admission/'admission-source-inputs.sha256').read_bytes()).hexdigest() == record['admission_source_manifest_sha256']
assert hashlib.sha256((admission/'production-source-inputs.sha256').read_bytes()).hexdigest() == sys.argv[3]
for line in (admission/'admission-source-inputs.sha256').read_text().splitlines():
    expected, relative = line.split('  ', 1)
    source = (admission/'source'/relative).resolve(strict=True)
    assert source.is_relative_to((admission/'source').resolve(strict=True))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
PY
cp -a -- "$admission" "$run_directory/admission"
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
"$binary" --output "$run_directory/results" --retained-hashes "$run_directory/retained-inputs.sha256" \
    --context-training-log "$run_directory/admission/build-and-tests.log" \
    --context-training-log-sha256 "$(sha256sum "$run_directory/admission/build-and-tests.log" | cut -d ' ' -f 1)"
printf 'Context deletion artifacts: %s\n' "$run_directory/results"

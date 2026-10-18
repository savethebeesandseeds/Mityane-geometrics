#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'context-optimization-validation-v1 has a fixed recipe; no runner overrides.' >&2; exit 1; }
binary="${CONTEXT_OPTIMIZATION_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_context_optimization_validation}"
run_root="$PWD/output/runs/rpb-context-optimization-validation"
reference="$PWD/output/runs/rpb-context-deletion/context-deletion-KTbte3"
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
python3 - "$admission" "$run_root/admission" "$source_id" <<'PY'
import hashlib, json, sys
from pathlib import Path
admission = Path(sys.argv[1]).resolve(strict=True)
assert admission.parent == Path(sys.argv[2]).resolve(strict=True) and admission.name.startswith('admission-')
record = json.loads((admission/'passed.json').read_text())
assert record['protocol'] == 'context-optimization-validation-v1' and record['status'] == 'passed'
assert record['source_fingerprint'] == sys.argv[3] and record['source_preserved'] is True
assert hashlib.sha256((admission/'build-and-tests.log').read_bytes()).hexdigest() == record['log_sha256']
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
# Inventory metadata only. The allowlist opens no TEST/stress/report payloads.
python3 - "$run_directory/reference-artifact-integrity.json" "$run_directory/retained-inputs.sha256" <<'PY'
import json, sys
from pathlib import PurePosixPath
allowed = {'results/validation-report.json'}
for master in (3101, 3202, 3303):
    base = f'results/seed-{master}-lag_sign'
    allowed.update(f'{base}/{name}' for name in (
        'controlled-training.pt', 'controlled-validation.pt',
        'candidate-training-reconstruction.pt', 'candidate-validation-reconstruction.pt'))
    point = f'{base}/candidate-milestone-512'
    allowed.update(f'{point}/{name}' for name in (
        'checkpoint.pt', 'checkpoint.pt.audit.pt', 'checkpoint.pt.scaler.pt',
        'checkpoint.pt.training-raw.pt', 'native-training.pt', 'native-validation.pt', 'point.json'))
    for repetition in ('rep-1', 'rep-2', 'rep-3'):
        allowed.update(f'{point}/{repetition}/{name}' for name in ('fit.pt', 'validation-predictions.pt'))
inventory = json.load(open(sys.argv[1], encoding='utf-8'))
entries = {entry['path']: entry for entry in inventory['files']}
if not allowed <= entries.keys():
    raise SystemExit(f'Missing declared retained inputs: {sorted(allowed-entries.keys())}')
with open(sys.argv[2], 'x', encoding='utf-8', newline='\n') as target:
    for path in sorted(allowed):
        assert PurePosixPath(path).is_relative_to('results')
        target.write(f"{entries[path]['sha256']}  {path}\n")
PY
(cd -- "$reference" && sha256sum -c "$run_directory/retained-inputs.sha256") > "$run_directory/retained-before.txt"
"$binary" --output "$run_directory/results" --retained-hashes "$run_directory/retained-inputs.sha256" 2>&1 | tee "$run_directory/command.log"
(cd -- "$reference" && sha256sum -c "$run_directory/retained-inputs.sha256") > "$run_directory/retained-after.txt"
[[ "$(sha256sum "$reference/artifact-integrity.json" | cut -d ' ' -f 1)" == "$inventory_sha" ]] || { echo 'Parent inventory changed.' >&2; exit 1; }
printf 'Optimization validation artifacts: %s\n' "$run_directory/results"
sha256sum -c "$run_directory/source-inputs.sha256" > "$run_directory/source-preserved-after.txt"
(cd -- "$run_directory/source"; sha256sum -c "$run_directory/source-inputs.sha256") > "$run_directory/copied-source-preserved-after.txt"

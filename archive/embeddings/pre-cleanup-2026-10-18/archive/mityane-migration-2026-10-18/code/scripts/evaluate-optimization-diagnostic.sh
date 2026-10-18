#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'optimization-validation-v1 has a fixed recipe; no runner overrides.' >&2; exit 1; }
binary="${OPTIMIZATION_DIAGNOSTIC_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_optimization_diagnostic}"
run_root="$PWD/output/runs/rpb-optimization-validation"
reference="$PWD/output/runs/rpb-paired-pooling/paired-pooling-BFFPX5"
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

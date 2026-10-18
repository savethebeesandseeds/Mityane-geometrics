#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run the frozen diagnostic inside the managed container.' >&2; exit 1; }
binary="${TRAINING_OBJECTIVE_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_training_objective_diagnostic}"
root="$PWD/output/runs/rpb-training-objective-diagnostic"
[[ -n "${TRAINING_OBJECTIVE_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze source inputs.' >&2; exit 1; }
mkdir -p -- "$root"
capsule="$(mktemp -d "$root/training-objective-XXXXXX")"
printf 'TRAIN objective capsule: %s\n' "$capsule"
read -r -a sources <<< "$TRAINING_OBJECTIVE_SOURCE_INPUTS"
mkdir -- "$capsule/source"
cp --parents -- "${sources[@]}" "$capsule/source/"
sha256sum -- "${sources[@]}" > "$capsule/source-inputs.sha256"
source_id="$(sha256sum "$capsule/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled/preserved source mismatch.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$capsule/source-fingerprint.txt"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete actual CUDA admission first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
python3 - "$admission" "$root/admission" "$source_id" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
assert root.parent==Path(sys.argv[2]).resolve(strict=True) and root.name.startswith('admission-')
record=json.loads((root/'passed.json').read_text())
assert record['protocol']=='training-objective-diagnostic-v1' and record['status']=='passed'
assert record['source_fingerprint']==sys.argv[3] and record['source_preserved'] is True
assert record['payload_files_read']==record['encoder_updates']==record['head_fits']==0
for name,digest in record['logs'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256((root/'source-inputs.sha256').read_bytes()).hexdigest()==sys.argv[3]
for line in (root/'source-inputs.sha256').read_text().splitlines():
    digest,relative=line.split('  ',1)
    path=(root/'source'/relative).resolve(strict=True)
    assert path.is_relative_to((root/'source').resolve(strict=True))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
PY
cp -a -- "$admission" "$capsule/admission"
card="$PWD/code/evaluation/cards/training_objective_diagnostic_v1.md"
card_sha="$(sha256sum "$card" | cut -d ' ' -f 1)"
python3 code/scripts/prepare-training-objective-inputs.py --freeze --repo-root "$PWD" \
  --output "$capsule/inputs" --frozen-card "$card" --frozen-card-sha256 "$card_sha" > "$capsule/input-freeze.json"
tsv="$capsule/inputs/instances.tsv"
manifest="$capsule/inputs/inputs.sha256"
"$binary" --repo-root "$PWD" --output "$capsule/results" \
  --instances "$tsv" --instances-sha256 "$(sha256sum "$tsv" | cut -d ' ' -f 1)" \
  --input-manifest "$manifest" --input-manifest-sha256 "$(sha256sum "$manifest" | cut -d ' ' -f 1)" \
  --admission-log "$capsule/admission/build-and-tests.log" \
  --admission-log-sha256 "$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)" \
  2>&1 | tee "$capsule/command.log"
sha256sum -c "$manifest" > "$capsule/inputs-preserved-after.txt"
sha256sum -c "$capsule/source-inputs.sha256" > "$capsule/source-preserved-after.txt"
(cd -- "$capsule/source"; sha256sum -c "$capsule/source-inputs.sha256") > "$capsule/copied-source-preserved-after.txt"
python3 - "$capsule" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
rows=[]
for path in sorted(root.rglob('*')):
    if path.is_file():
        assert not path.is_symlink()
        data=path.read_bytes()
        rows.append(dict(path=path.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
record=dict(schema_version=1,protocol='training-objective-diagnostic-v1',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checksum_algorithm='sha256-file-bytes',
    inventory_excludes_itself=True,files=rows,file_count=len(rows),total_bytes=sum(x['bytes'] for x in rows))
with (root/'artifact-integrity.json').open('x',encoding='utf-8') as out:
    json.dump(record,out,indent=2);out.write('\n')
PY
printf 'TRAIN objective diagnostic artifacts: %s\n' "$capsule/results"

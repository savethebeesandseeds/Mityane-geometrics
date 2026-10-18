#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run frozen measurement inside the managed container.' >&2; exit 1; }
binary="${V7_DECODER_CALIBRATION_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_v7_decoder_calibration}"
root="$PWD/output/runs/rpb-v7-decoder-calibration"
[[ -n "${V7_DECODER_CALIBRATION_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to bind sources.' >&2; exit 1; }
mkdir -p -- "$root"
capsule="$(mktemp -d "$root/decoder-calibration-XXXXXX")"
printf 'Decoder calibration capsule: %s\n' "$capsule"
read -r -a sources <<< "$V7_DECODER_CALIBRATION_SOURCE_INPUTS"
mkdir -- "$capsule/source"
cp --parents -- "${sources[@]}" "$capsule/source/"
sha256sum -- "${sources[@]}" > "$capsule/source-inputs.sha256"
source_id="$(sha256sum "$capsule/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]]
printf '%s\n' "$source_id" > "$capsule/source-fingerprint.txt"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete actual decoder CUDA admission first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
python3 - "$admission" "$root/admission" "$source_id" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
assert root.parent==Path(sys.argv[2]).resolve(strict=True) and root.name.startswith('admission-')
record=json.loads((root/'passed.json').read_text())
assert record['protocol']=='v7-decoder-calibration-v1' and record['status']=='passed' and record['source_preserved'] is True
assert record['human_card_sha256']=='5ae05a3b5ec253d1743842f82fc28b45e0f20c8c84da1b58bf4ab9c514d292e9'
assert record['source_fingerprint']==sys.argv[3] and record['quality_payloads_read']==record['head_fits']==0
assert hashlib.sha256((root/'source-inputs.sha256').read_bytes()).hexdigest()==sys.argv[3]
assert hashlib.sha256((root/'build-and-tests.log').read_bytes()).hexdigest()==record['log_sha256']
for row in (root/'source-inputs.sha256').read_text().splitlines():
  sha,relative=row.split('  ',1); lexical=root/'source'/relative
  path=lexical.resolve(strict=True)
  assert path==lexical and path.is_relative_to(root/'source') and not path.is_symlink()
  assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
PY
cp -a -- "$admission" "$capsule/admission"
"$binary" --parent-roles > "$capsule/parent-role-plan.json"
python3 code/scripts/prepare-v7-decoder-inputs.py --repo-root "$PWD" --capsule "$capsule"
"$binary" --output "$capsule/results" --parent-input-manifest "$capsule/parent-inputs.sha256" \
  --admission-log "$capsule/admission/build-and-tests.log" \
  --admission-sha256 "$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)" \
  2>&1 | tee "$capsule/command.log"
python3 code/scripts/prepare-v7-decoder-inputs.py --repo-root "$PWD" --capsule "$capsule" --verify > "$capsule/inputs-preserved-after.txt"
sha256sum -c "$capsule/source-inputs.sha256" > "$capsule/source-preserved-after.txt"
(cd -- "$capsule/source"; sha256sum -c "$capsule/source-inputs.sha256") > "$capsule/copied-source-preserved-after.txt"
python3 - "$capsule" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files=[]
for path in sorted(root.rglob('*')):
  if path.is_file():
    assert not path.is_symlink();content=path.read_bytes()
    files.append(dict(path=path.relative_to(root).as_posix(),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
record=dict(schema_version=1,protocol='v7-decoder-calibration-v1',
  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checksum_algorithm='sha256-file-bytes',
  inventory_excludes_itself=True,files=files,file_count=len(files),total_bytes=sum(x['bytes'] for x in files))
with (root/'artifact-integrity.json').open('x',encoding='utf-8') as out:
  json.dump(record,out,indent=2);out.write('\n')
PY
printf 'Decoder calibration complete: %s\n' "$capsule"

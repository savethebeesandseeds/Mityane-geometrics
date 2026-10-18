#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run frozen admission inside the managed container.' >&2; exit 1; }
root="$PWD/output/runs/rpb-v7-decoder-calibration"
session='rpb-paired-pooling'
python3 - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('code/evaluation/cards/v7_decoder_calibration_v1.md').read_bytes()).hexdigest()=='5ae05a3b5ec253d1743842f82fc28b45e0f20c8c84da1b58bf4ab9c514d292e9'
PY
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Decoder admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-v7-decoder-calibration-sources)
[[ ${#sources[@]} -gt 0 ]]
mkdir -- "$admission/source"
cp --parents -- "${sources[@]}" "$admission/source/"
sha256sum -- "${sources[@]}" > "$admission/source-inputs.sha256"
source_id="$(sha256sum "$admission/source-inputs.sha256" | cut -d ' ' -f 1)"
printf '%s\n' "$source_id" > "$admission/source-fingerprint.txt"
bash code/scripts/task.sh "$session" -j2 v7-decoder-calibration test-frozen-role-guard test-rpb-decoder-calibration \
  2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_v7_decoder_calibration"
[[ "$("$binary" --source-id)" == "$source_id" ]]
sha256sum -c "$admission/source-inputs.sha256" > "$admission/source-preserved-after.txt"
(cd -- "$admission/source"; sha256sum -c "$admission/source-inputs.sha256") > "$admission/copied-source-preserved-after.txt"
python3 - "$admission" "$source_id" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); log=(root/'build-and-tests.log').read_text()
assert 'V7 decoder calibration CUDA admission passed' in log and sys.argv[2] in log
assert 'Frozen role guard checks passed' in log
record=dict(protocol='v7-decoder-calibration-v1',status='passed',source_fingerprint=sys.argv[2],
  human_card_sha256='5ae05a3b5ec253d1743842f82fc28b45e0f20c8c84da1b58bf4ab9c514d292e9',
  source_preserved=True,quality_payloads_read=0,new_quality_encoder_updates=0,head_fits=0,
  log_sha256=hashlib.sha256((root/'build-and-tests.log').read_bytes()).hexdigest(),
  gates=['closed-role admission before hashes; redirects and mutation rejection',
         'actual CUDA decoder-only finite gradients and updates; encoder/native/scaler exact',
         'original Q and hidden-value isolation; direct/split AdamW parity',
         'typed immutable snapshots and parent/data/policy/seed/overwrite rejection'])
with (root/'passed.json').open('x',encoding='utf-8') as out:
  json.dump(record,out,indent=2);out.write('\n')
PY
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Decoder admission passed: %s\n' "$admission"

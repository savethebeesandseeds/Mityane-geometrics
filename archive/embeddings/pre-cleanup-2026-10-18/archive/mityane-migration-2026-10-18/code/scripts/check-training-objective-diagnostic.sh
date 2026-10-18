#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run frozen admission inside the managed container.' >&2; exit 1; }
session='rpb-paired-pooling'
root="$PWD/output/runs/rpb-training-objective-diagnostic"
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'TRAIN objective admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-training-objective-sources)
[[ ${#sources[@]} -gt 0 ]] || exit 1
mkdir -- "$admission/source"
cp --parents -- "${sources[@]}" "$admission/source/"
sha256sum -- "${sources[@]}" > "$admission/source-inputs.sha256"
bash code/scripts/task.sh "$session" -j2 -W Makefile \
  training-objective-diagnostic test-rpb-training-objective-diagnostic \
  2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/cuwacunu_embedding/build/$session/embedding_training_objective_diagnostic"
"$binary" --metadata-self-test 2>&1 | tee "$admission/metadata-gate.log"
sha256sum -c "$admission/source-inputs.sha256" > "$admission/source-preserved-after.txt"
(cd -- "$admission/source"; sha256sum -c "$admission/source-inputs.sha256") > "$admission/copied-source-preserved-after.txt"
source_id="$("$binary" --source-id)"
[[ "$source_id" == "$(sha256sum "$admission/source-inputs.sha256" | cut -d ' ' -f 1)" ]] || exit 1
printf '%s\n' "$source_id" > "$admission/source-fingerprint.txt"
python3 - "$admission" "$source_id" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
assert 'TRAIN objective CUDA admission passed' in (root/'build-and-tests.log').read_text()
record = dict(protocol='training-objective-diagnostic-v1', status='passed', source_fingerprint=sys.argv[2],
    source_preserved=True, payload_files_read=0, encoder_updates=0, head_fits=0,
    gates=['actual CUDA parameters, input, loss and gradients', 'zero weight and buffer updates',
           'original support and hierarchical pair-Huber math', 'both frozen Ridge affine maps and descent sign',
           'real encoder gradients separate from frozen latent leaf', 'explicit195 TRAIN role metadata and SHA256'],
    logs={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
          for name in ('build-and-tests.log','metadata-gate.log','source-inputs.sha256')})
with (root/'passed.json').open('x', encoding='utf-8') as out:
    json.dump(record,out,indent=2);out.write('\n')
PY
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'TRAIN objective admission passed: %s\n' "$admission"

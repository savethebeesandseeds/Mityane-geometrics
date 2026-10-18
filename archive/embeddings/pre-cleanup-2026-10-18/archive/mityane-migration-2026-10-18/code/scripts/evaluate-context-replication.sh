#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'context-replication-v1 has a frozen recipe; no overrides.' >&2; exit 1; }
binary="${CONTEXT_REPLICATION_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_context_replication}"
root="$PWD/output/runs/rpb-context-replication"
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
python3 - "$admission" "$root/admission" "$source_id" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
assert root.parent == Path(sys.argv[2]).resolve(strict=True) and root.name.startswith('admission-')
record = json.loads((root/'passed.json').read_text())
assert record['protocol'] == 'context-replication-v1' and record['status'] == 'passed'
assert record['source_fingerprint'] == sys.argv[3] and record['source_preserved'] is True
for name, key in [('build-and-tests.log','log_sha256'), ('cuda-gate.log','cuda_gate_log_sha256'),
                  ('admission-source-inputs.sha256','admission_source_manifest_sha256')]:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == record[key]
assert hashlib.sha256((root/'production-source-inputs.sha256').read_bytes()).hexdigest() == sys.argv[3]
for line in (root/'admission-source-inputs.sha256').read_text().splitlines():
    expected, relative = line.split('  ', 1)
    source = (root/'source'/relative).resolve(strict=True)
    assert source.is_relative_to((root/'source').resolve(strict=True))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
PY
cp -a -- "$admission" "$capsule/admission"
admission_log="$capsule/admission/build-and-tests.log"
admission_log_sha="$(sha256sum "$admission_log" | cut -d ' ' -f 1)"
"$binary" --phase prepare --output "$capsule/reference" \
    --admission-log "$admission_log" --admission-log-sha256 "$admission_log_sha" \
    2>&1 | tee "$capsule/reference-command.log"
# This new preparation contains TRAIN/VALIDATION only. Freeze every file before
# any candidate fit or fresh TEST generation; no historical payload is read.
python3 - "$capsule/reference" "$capsule/reference-inputs.sha256" "$capsule/reference-inputs.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
card = json.loads((root/'native-development-card.json').read_text())
completion = json.loads((root/'development-complete.json').read_text())
assert card['protocol'] == 'native-development-v1' and card['test_pairs'] == 0
assert card['development_only'] is True and card['selection_performed'] is False
assert card['testing_generated'] is False and card['stress_evaluated'] is False
assert completion['protocol'] == 'native-development-v1'
assert completion['all_declared_milestones_complete'] is True
assert completion['all_retained_witnesses_exact'] is True and completion['controls_immutable'] is True
assert completion['selection_performed'] is False and completion['testing_generated'] is False
assert completion['stress_evaluated'] is False and completion['cohorts'] == 5 and completion['checked_points'] == 10
assert completion['milestones'] == [0,512] and card['masters'] == [4404,5505,6606,7707,8808]
assert not any('testing' in p.name or 'stress' in p.name or p.name == 'selection.json' for p in root.rglob('*'))
records = []
with Path(sys.argv[2]).open('x', encoding='utf-8', newline='\n') as output:
    for path in sorted(root.rglob('*')):
        if path.is_file():
            content = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256(content).hexdigest()
            output.write(f'{digest}  {relative}\n')
            records.append(dict(path=relative, bytes=len(content), sha256=digest))
with Path(sys.argv[3]).open('x', encoding='utf-8', newline='\n') as output:
    json.dump(dict(protocol='context-replication-v1', role='reference-preparation',
                   checksum_algorithm='sha256-file-bytes', files=records,
                   frozen_before_candidate_and_testing=True,
                   manifest_sha256=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()), output, indent=2)
    output.write('\n')
PY
(cd -- "$capsule/reference"; sha256sum -c "$capsule/reference-inputs.sha256") > "$capsule/reference-preserved-before.txt"
"$binary" --phase compare --output "$capsule/results" --reference "$capsule/reference" \
    --retained-hashes "$capsule/reference-inputs.sha256" --admission-log "$admission_log" \
    --admission-log-sha256 "$admission_log_sha" 2>&1 | tee "$capsule/command.log"
(cd -- "$capsule/reference"; sha256sum -c "$capsule/reference-inputs.sha256") > "$capsule/reference-preserved-after.txt"
sha256sum -c "$capsule/source-inputs.sha256" > "$capsule/source-preserved-after.txt"
(cd -- "$capsule/source"; sha256sum -c "$capsule/source-inputs.sha256") > "$capsule/copied-source-preserved-after.txt"
printf 'Context replication artifacts: %s\n' "$capsule/results"

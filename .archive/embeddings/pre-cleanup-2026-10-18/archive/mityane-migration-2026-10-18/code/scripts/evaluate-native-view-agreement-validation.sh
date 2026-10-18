#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'native-view-agreement-validation-v1 has a frozen recipe; no overrides.' >&2; exit 1; }
binary="${NATIVE_VIEW_AGREEMENT_VALIDATION_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_native_view_agreement_validation}"
root="$PWD/output/runs/rpb-native-view-agreement-validation"
[[ -n "${NATIVE_VIEW_AGREEMENT_VALIDATION_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze source inputs.' >&2; exit 1; }
python3 - <<'PY'
import hashlib
from pathlib import Path
card = Path('code/evaluation/cards/native_view_agreement_validation_v1.md')
assert hashlib.sha256(card.read_bytes()).hexdigest() == '1f33eff0e6ca3e3241fd972fdc855900c535defa330f4eaec7effdf0c90d1c2e'
PY
mkdir -p -- "$root"
capsule="$(mktemp -d "$root/native-view-agreement-validation-XXXXXX")"
printf 'Native view agreement validation capsule: %s\n' "$capsule"
read -r -a sources <<< "$NATIVE_VIEW_AGREEMENT_VALIDATION_SOURCE_INPUTS"
mkdir -- "$capsule/source"
cp --parents -- "${sources[@]}" "$capsule/source/"
sha256sum -- "${sources[@]}" > "$capsule/source-inputs.sha256"
source_id="$(sha256sum "$capsule/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source differs from preserved snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$capsule/source-fingerprint.txt"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete check-native-view-agreement-validation.sh first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
python3 - "$admission" "$root/admission" "$source_id" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
assert root.parent == Path(sys.argv[2]).resolve(strict=True) and root.name.startswith('admission-')
record = json.loads((root/'passed.json').read_text())
assert record['protocol'] == 'native-view-agreement-validation-v1' and record['status'] == 'passed'
assert record['human_card_sha256'] == '1f33eff0e6ca3e3241fd972fdc855900c535defa330f4eaec7effdf0c90d1c2e'
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
# This command emits compiled role metadata only, before any parent payload.
"$binary" --parent-roles > "$capsule/parent-role-plan.json"
python3 - "$PWD" "$capsule" <<'PY'
import hashlib, json, sys
from pathlib import Path, PurePosixPath
repo = Path(sys.argv[1]).resolve(strict=True)
capsule = Path(sys.argv[2]).resolve(strict=True)
plan = json.loads((capsule/'parent-role-plan.json').read_text())
assert plan['protocol'] == 'native-view-agreement-validation-v1' and plan['payload_access'] is False
assert plan['masters'] == [4404,5505,6606,7707,8808]
assert plan['testing_accessed'] is False and plan['stress_accessed'] is False
expected_parents = {
    'v4': ('output/runs/rpb-context-replication/context-replication-JjNEUc',
           '13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9'),
    'v7': ('output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2',
           'd4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821')}
assert len(plan['parents']) == len(expected_parents)
parents = {}; inventories = {}; parent_records = []
for item in plan['parents']:
    identity = item['id']; relative, expected_inventory = expected_parents[identity]
    assert identity not in parents and item['capsule'] == relative and item['inventory_sha256'] == expected_inventory
    path = (repo/relative).resolve(strict=True)
    assert path.is_relative_to(repo) and not (repo/relative).is_symlink()
    inventory_bytes = (path/'artifact-integrity.json').read_bytes()
    assert hashlib.sha256(inventory_bytes).hexdigest() == expected_inventory
    inventory = json.loads(inventory_bytes)
    assert inventory['inventory_excludes_itself'] is True
    assert inventory['protocol'] == ('context-replication-v1' if identity == 'v4' else 'context-lighter-validation-v1')
    indexed = {entry['path']:entry for entry in inventory['files']}
    assert len(indexed) == len(inventory['files'])
    parents[identity] = path; inventories[identity] = indexed
    parent_records.append(dict(item, absolute_path=str(path)))
roles = plan['inputs']
assert len(roles) == 326 and len({(item['parent_id'],item['path']) for item in roles}) == 326
assert sum(item['parent_id']=='v4' for item in roles) == 158
assert sum(item['parent_id']=='v7' for item in roles) == 168
closed_roles = set()
for identity in ('v4','v7'):
    prefix = 'reference' if identity=='v4' else 'results/candidate-development'
    for file in ('native-development-card.json','development-complete.json','validation-report.json'):
        closed_roles.add((identity,f'{prefix}/{file}'))
    for master in plan['masters']:
        base=f'{prefix}/seed-{master}-lag_sign'
        for file in ('controlled-training.pt','controlled-validation.pt','development-manifest.json','trainer-audit.json'):
            closed_roles.add((identity,f'{base}/{file}'))
        for budget in (0,512):
            point=f'{base}/milestone-{budget}'
            for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt'):
                closed_roles.add((identity,f'{point}/checkpoint.pt{suffix}'))
            for file in ('native-training.pt','native-validation.pt','point.json','training-reconstruction.pt','validation-reconstruction.pt'):
                closed_roles.add((identity,f'{point}/{file}'))
            if budget==512:
                for rep in ('rep-1','rep-2','rep-3'):
                    for file in ('fit.pt','training-predictions.pt','validation-predictions.pt'):
                        closed_roles.add((identity,f'{point}/{rep}/{file}'))
        if identity=='v7':
            for suffix in ('.pt','.json'):
                closed_roles.add((identity,f'results/validation-views/seed-{master}-lag_sign/validation-dropout-030{suffix}'))
assert len(closed_roles)==326 and {(item['parent_id'],item['path']) for item in roles}==closed_roles
# Validate every role and redirect before the first permitted payload read/hash.
pending = []; physical = set()
for role in sorted(roles, key=lambda item:(item['parent_id'],item['path'])):
    identity = role['parent_id']; relative = PurePosixPath(role['path'])
    assert not relative.is_absolute() and '..' not in relative.parts and str(relative)==role['path']
    assert not any('testing' in part or 'stress' in part or part=='report.json' for part in relative.parts)
    assert role['tag'] == ('RPB-v4' if identity=='v4' else 'RPB-v7') and role['role']
    expected_master = next((master for master in plan['masters'] if f'/seed-{master}-lag_sign/' in role['path']),None)
    expected_budget = 0 if '/milestone-0/' in role['path'] else (512 if '/milestone-512/' in role['path'] else None)
    assert role['master_seed']==expected_master and role['budget']==expected_budget
    assert expected_master is not None or role['role']=='development-metadata'
    key=identity+'/'+role['path']; assert role.get('manifest_path',key)==key
    lexical = parents[identity]/relative; assert not lexical.is_symlink()
    path = lexical.resolve(strict=True)
    assert path == lexical and path.is_relative_to(parents[identity]) and path.is_file() and path not in physical
    physical.add(path)
    original = inventories[identity][role['path']]
    assert isinstance(original['bytes'],int) and original['bytes']>=0
    assert len(original['sha256'])==64 and set(original['sha256'])<=set('0123456789abcdef')
    pending.append((role,key,path,original))
records = []
with (capsule/'parent-inputs.sha256').open('x', encoding='utf-8', newline='\n') as output:
    for role,key,path,original in pending:
        content = path.read_bytes(); digest = hashlib.sha256(content).hexdigest()
        assert len(content)==original['bytes'] and digest==original['sha256']
        output.write(f'{digest}  {key}\n')
        records.append(dict(role, manifest_path=key, absolute_path=str(path), bytes=len(content), sha256=digest))
record = dict(protocol='native-view-agreement-validation-v1', role='known-development-parent-inputs',
              checksum_algorithm='sha256-file-bytes', frozen_before_new_fitting=True,
              testing_accessed=False, stress_accessed=False, parents=parent_records, inputs=records,
              manifest_sha256=hashlib.sha256((capsule/'parent-inputs.sha256').read_bytes()).hexdigest(),
              role_plan_sha256=hashlib.sha256((capsule/'parent-role-plan.json').read_bytes()).hexdigest())
with (capsule/'parent-inputs.json').open('x', encoding='utf-8', newline='\n') as output:
    json.dump(record, output, indent=2); output.write('\n')
PY
"$binary" --phase measure --output "$capsule/results" \
    --parent-input-manifest "$capsule/parent-inputs.sha256" \
    --admission-log "$admission_log" --admission-sha256 "$admission_log_sha" \
    2>&1 | tee "$capsule/command.log"
python3 - "$capsule" <<'PY'
import hashlib, json, sys
from pathlib import Path
capsule = Path(sys.argv[1])
record = json.loads((capsule/'parent-inputs.json').read_text())
for item in record['inputs']:
    content = Path(item['absolute_path']).read_bytes()
    assert len(content)==item['bytes'] and hashlib.sha256(content).hexdigest()==item['sha256']
with (capsule/'parent-preserved-after.json').open('x',encoding='utf-8') as output:
    json.dump(dict(protocol='native-view-agreement-validation-v1',status='passed',
                   parent_inputs=len(record['inputs']),all_allowed_bytes_preserved=True),output,indent=2)
    output.write('\n')
PY
sha256sum -c "$capsule/source-inputs.sha256" > "$capsule/source-preserved-after.txt"
(cd -- "$capsule/source"; sha256sum -c "$capsule/source-inputs.sha256") > "$capsule/copied-source-preserved-after.txt"
python3 - "$capsule" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
records = []
for path in sorted(root.rglob('*')):
    if path.is_file():
        assert not path.is_symlink()
        content = path.read_bytes()
        records.append(dict(path=path.relative_to(root).as_posix(), bytes=len(content),
                            sha256=hashlib.sha256(content).hexdigest()))
record = dict(schema_version=1, protocol='native-view-agreement-validation-v1',
              created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              checksum_algorithm='sha256-file-bytes', inventory_excludes_itself=True,
              files=records, file_count=len(records), total_bytes=sum(item['bytes'] for item in records))
with (root/'artifact-integrity.json').open('x', encoding='utf-8', newline='\n') as output:
    json.dump(record, output, indent=2); output.write('\n')
PY
printf 'Native view agreement validation artifacts: %s\n' "$capsule/results"

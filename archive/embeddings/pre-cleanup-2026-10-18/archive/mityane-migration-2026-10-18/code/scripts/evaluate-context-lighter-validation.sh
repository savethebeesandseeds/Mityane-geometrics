#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'context-lighter-validation-v1 has a frozen recipe; no overrides.' >&2; exit 1; }
binary="${CONTEXT_LIGHTER_BIN:-/opt/cuwacunu_embedding/build/rpb-paired-pooling/embedding_context_lighter_validation}"
root="$PWD/output/runs/rpb-context-lighter-validation"
parent="$PWD/output/runs/rpb-context-replication/context-replication-JjNEUc"
[[ -n "${CONTEXT_LIGHTER_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to freeze source inputs.' >&2; exit 1; }
mkdir -p -- "$root"
capsule="$(mktemp -d "$root/lighter-validation-XXXXXX")"
printf 'Lighter context validation capsule: %s\n' "$capsule"
read -r -a sources <<< "$CONTEXT_LIGHTER_SOURCE_INPUTS"
mkdir -- "$capsule/source"
cp --parents -- "${sources[@]}" "$capsule/source/"
sha256sum -- "${sources[@]}" > "$capsule/source-inputs.sha256"
source_id="$(sha256sum "$capsule/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source differs from preserved snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$capsule/source-fingerprint.txt"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete check-context-lighter-validation.sh first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
python3 - "$admission" "$root/admission" "$source_id" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
assert root.parent == Path(sys.argv[2]).resolve(strict=True) and root.name.startswith('admission-')
record = json.loads((root/'passed.json').read_text())
assert record['protocol'] == 'context-lighter-validation-v1' and record['status'] == 'passed'
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
# This command emits only compiled role metadata. The exact list is frozen
# before opening or hashing any permitted parent payload.
"$binary" --parent-roles > "$capsule/parent-role-plan.json"
python3 - "$parent" "$capsule" <<'PY'
import hashlib, json, sys
from pathlib import Path, PurePosixPath
parent = Path(sys.argv[1]).resolve(strict=True)
capsule = Path(sys.argv[2]).resolve(strict=True)
inventory_path = parent/'artifact-integrity.json'
inventory_bytes = inventory_path.read_bytes()
inventory_sha = hashlib.sha256(inventory_bytes).hexdigest()
assert inventory_sha == '13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9'
inventory = json.loads(inventory_bytes)
assert inventory['protocol'] == 'context-replication-v1' and inventory['inventory_excludes_itself'] is True
indexed = {item['path']: item for item in inventory['files']}
assert len(indexed) == len(inventory['files'])
plan = json.loads((capsule/'parent-role-plan.json').read_text())
assert plan['protocol'] == 'context-lighter-validation-v1'
assert plan['parent_capsule'] == 'output/runs/rpb-context-replication/context-replication-JjNEUc'
assert plan['payload_access'] is False
assert plan['masters'] == [4404,5505,6606,7707,8808]
assert plan['testing_accessed'] is False and plan['stress_accessed'] is False
roles = plan['inputs']
assert roles and len({item['path'] for item in roles}) == len(roles)
records = []
with (capsule/'parent-inputs.sha256').open('x', encoding='utf-8', newline='\n') as output:
    for role in sorted(roles, key=lambda item: item['path']):
        relative = PurePosixPath(role['path'])
        assert not relative.is_absolute() and '..' not in relative.parts and str(relative) == role['path']
        assert relative.parts[0] in ('reference','results')
        assert not any('testing' in part or 'stress' in part or part == 'report.json' for part in relative.parts)
        assert role['path'] in indexed and role['role'] and role['tag'] in ('RPB-v4','RPB-v6')
        assert role['master_seed'] in plan['masters'] or (role['master_seed'] is None and
            role['role'] == 'development-metadata' and role['tag'] == 'RPB-v4')
        assert role['budget'] in (0,512,None)
        lexical = parent/relative
        assert not lexical.is_symlink()
        path = lexical.resolve(strict=True)
        assert path == lexical and path.is_relative_to(parent) and path.is_file()
        original = indexed[role['path']]
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        assert len(content) == original['bytes'] and digest == original['sha256']
        output.write(f'{digest}  {role["path"]}\n')
        records.append(dict(role, absolute_path=str(path), bytes=len(content), sha256=digest))
record = dict(protocol='context-lighter-validation-v1', role='known-development-parent-inputs',
              checksum_algorithm='sha256-file-bytes', parent=str(parent),
              parent_inventory_sha256=inventory_sha,
              parent_source_fingerprint='9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828',
              parent_training_producer='2fff50c48605ee21dbbfd28d6989bdcf6b5cd057785c2acb19419571af48b895',
              parent_core_writer='587f2423758c3e70c2c7665d9e7b59bfe10af4c7f9aaba23f5318b3c55b13bca',
              frozen_before_new_fitting=True, testing_accessed=False, stress_accessed=False,
              manifest_sha256=hashlib.sha256((capsule/'parent-inputs.sha256').read_bytes()).hexdigest(),
              role_plan_sha256=hashlib.sha256((capsule/'parent-role-plan.json').read_bytes()).hexdigest(), inputs=records)
with (capsule/'parent-inputs.json').open('x', encoding='utf-8', newline='\n') as output:
    json.dump(record, output, indent=2); output.write('\n')
PY
(cd -- "$parent"; sha256sum -c "$capsule/parent-inputs.sha256") > "$capsule/parent-preserved-before.txt"
"$binary" --phase measure --output "$capsule/results" \
    --retained-hashes "$capsule/parent-inputs.sha256" \
    --admission-log "$admission_log" --admission-log-sha256 "$admission_log_sha" \
    2>&1 | tee "$capsule/command.log"
(cd -- "$parent"; sha256sum -c "$capsule/parent-inputs.sha256") > "$capsule/parent-preserved-after.txt"
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
record = dict(schema_version=1, protocol='context-lighter-validation-v1',
              created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              checksum_algorithm='sha256-file-bytes', inventory_excludes_itself=True,
              files=records, file_count=len(records), total_bytes=sum(item['bytes'] for item in records))
with (root/'artifact-integrity.json').open('x', encoding='utf-8', newline='\n') as output:
    json.dump(record, output, indent=2); output.write('\n')
PY
printf 'Lighter context validation artifacts: %s\n' "$capsule/results"

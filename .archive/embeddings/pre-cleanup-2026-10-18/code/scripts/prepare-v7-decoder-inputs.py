#!/usr/bin/env python3
"""Closed85 legal inputs for v7-decoder-calibration-v1; no tensor decoding."""
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = 'v7-decoder-calibration-v1'
MASTERS = (4404, 5505, 6606, 7707, 8808)
PARENTS = {
    'v4': ('output/runs/rpb-context-replication/context-replication-JjNEUc',
           '13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9', 'reference'),
    'v7': ('output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2',
           'd4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821', 'results/candidate-development')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template():
    roles = {}
    for identity, (root, _, prefix) in PARENTS.items():
        names = [f'{prefix}/{name}' for name in ('native-development-card.json', 'validation-report.json')]
        if identity == 'v7':
            names.append(f'{prefix}/development-complete.json')
        for master in MASTERS:
            base = f'{prefix}/seed-{master}-lag_sign'
            point = f'{base}/milestone-512'
            if identity == 'v7':
                names.extend(f'{base}/{name}' for name in ('controlled-training.pt', 'controlled-validation.pt',
                    'development-manifest.json', 'trainer-audit.json'))
                names.extend(f'{point}/{name}' for name in ('checkpoint.pt', 'checkpoint.pt.audit.pt',
                    'checkpoint.pt.scaler.pt', 'checkpoint.pt.training-raw.pt', 'native-training.pt', 'native-validation.pt'))
            names.extend(f'{point}/{name}' for name in ('point.json', 'training-reconstruction.pt', 'validation-reconstruction.pt'))
        for name in names:
            roles[f'{identity}/{name}'] = f'{root}/{name}'
    assert len(roles) == 85
    return roles


def freeze(repo, capsule):
    plan = json.loads((capsule/'parent-role-plan.json').read_text())
    assert plan['protocol'] == PROTOCOL and plan['payload_access'] is False
    assert plan['testing_accessed'] is False and plan['stress_accessed'] is False
    roles = plan['inputs']
    expected = template()
    assert len(roles) == 85 and len({item['manifest_path'] for item in roles}) == 85
    assert {item['manifest_path']: item['relative_path'] for item in roles} == expected
    inventories = {}
    for identity, (relative, inventory_sha, _) in PARENTS.items():
        lexical = repo/relative
        assert not lexical.is_symlink() and lexical.resolve(strict=True) == lexical
        content = (lexical/'artifact-integrity.json').read_bytes()  # Inventory metadata only.
        assert hashlib.sha256(content).hexdigest() == inventory_sha
        inventory = json.loads(content)
        assert inventory['inventory_excludes_itself'] is True
        index = {row['path']: row for row in inventory['files']}
        assert len(index) == len(inventory['files'])
        inventories[identity] = index
    pending = []
    physical = set()
    # Complete every key, path, redirect, inventory entry before first payload hash.
    for item in sorted(roles, key=lambda x: x['manifest_path']):
        key = item['manifest_path']
        identity, relative = key.split('/', 1)
        lexical = repo/expected[key]
        assert not lexical.is_symlink() and lexical.resolve(strict=True) == lexical
        assert lexical.is_file() and lexical.is_relative_to(repo) and lexical not in physical
        physical.add(lexical)
        original = inventories[identity][relative]
        assert isinstance(original['bytes'], int) and original['bytes'] >= 0
        assert len(original['sha256']) == 64 and set(original['sha256']) <= set('0123456789abcdef')
        pending.append((item, lexical, original))
    records = []
    with (capsule/'parent-inputs.sha256').open('x', encoding='utf-8', newline='\n') as out:
        for item, path, original in pending:
            content = path.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            assert len(content) == original['bytes'] and sha == original['sha256']
            out.write(f"{sha}  {item['manifest_path']}\n")
            records.append(dict(item, absolute_path=str(path), bytes=len(content), sha256=sha))
    record = dict(protocol=PROTOCOL, checksum_algorithm='sha256-file-bytes', closed_roles=True,
        frozen_before_new_fitting=True, testing_accessed=False, stress_accessed=False,
        parents={identity: dict(capsule=value[0], inventory_sha256=value[1]) for identity,value in PARENTS.items()},
        inputs=records, manifest_sha256=digest(capsule/'parent-inputs.sha256'),
        role_plan_sha256=digest(capsule/'parent-role-plan.json'))
    with (capsule/'parent-inputs.json').open('x', encoding='utf-8') as out:
        json.dump(record, out, indent=2); out.write('\n')


def verify(repo, capsule):
    record = json.loads((capsule/'parent-inputs.json').read_text())
    assert record['protocol'] == PROTOCOL and record['closed_roles'] is True
    assert record['manifest_sha256'] == digest(capsule/'parent-inputs.sha256')
    assert record['role_plan_sha256'] == digest(capsule/'parent-role-plan.json')
    expected = template()
    assert len(record['inputs']) == 85
    assert {row['manifest_path']: row['relative_path'] for row in record['inputs']} == expected
    for row in record['inputs']:
        path = repo/expected[row['manifest_path']]
        assert path.resolve(strict=True) == path and str(path) == row['absolute_path'] and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
    print('All85 legal parent inputs preserved; TEST/stress closed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    repo = args.repo_root.resolve(strict=True)
    capsule = args.capsule.resolve(strict=True)
    assert capsule.is_relative_to(repo/'output/runs/rpb-v7-decoder-calibration')
    (verify if args.verify else freeze)(repo, capsule)

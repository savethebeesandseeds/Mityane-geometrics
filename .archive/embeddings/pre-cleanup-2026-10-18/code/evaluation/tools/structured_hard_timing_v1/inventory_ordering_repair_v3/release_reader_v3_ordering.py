#!/usr/bin/env python3
"""One flags-only copy, used only after ROOT authorizes a completed inventory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path('/embedding/output/runs/rpb-structured-hard-timing')
ORIGINAL = ROOT/'audit-tools/independent-structured-hard-timing-v2-prequality'
ENTRY = 'validate_structured_hard_timing.py'
PINS = {
    ENTRY: '2143d8d01b8a549e3a361d0b12b97c53abc87c57ca57714e68f890d6390a7708',
    'source-fixtures.json': 'f8957e3045149c8662cac2357c65b4998324e1010b01b9717f742bf07c48c2c6',
    'reader-seal.json': '47852a555d91addc3d7b3cd545cd6aa107d6ac4339ece80e202b2ab5b00e38b0',
    'archive_codec.py': '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d',
    'legacy_implementation.py': 'f0c64b9fe67a90a9d95196433e05dec527515778d1635bdc909d1703fa278b68',
    'saved_cpu_math.py': '1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9',
}

CORRECTED = ROOT/'audit-tools/independent-structured-hard-timing-v3-ordering-repair'
REPAIR_PROOF = ROOT/'source-review/inventory-ordering-repair-v3-proof.json'
REPAIR_PROOF_SHA = '2a3b052f1282275c7d0b8c5db2961cbf3c8b3722db4b853cb1378342c126dfc9'
CORRECTED_PINS = dict(PINS, **{
    ENTRY: '283ce3ec28c133c0d4053f4a49f08b134f51e6d3efa569c0a2199b942a3a5e7f',
    'source-fixtures.json': '980f2562015465447ab9bd8c9005dffdfaa5c3f1caedc6c02914c704f575ff1e',
    'reader-seal.json': 'f28e9e4edeb6980de5a30bdbe14899444b32f16a2814023b1f31aca837834f7a',
})

def inventory_names(rows):
    names=[row['path'] for row in rows]
    assert all(isinstance(n,str) and str(PurePosixPath(n))==n and not n.startswith('/') and '\\' not in n and
               all(x not in ('','.','..') for x in n.split('/')) for n in names)
    assert len(names)==len(set(names)) and names==sorted(names,key=PurePosixPath)
    return names

def sha(body):
    return hashlib.sha256(body).hexdigest()

def admit(paths):
    identities = set()
    for path in paths:
        assert path.is_absolute() and path.resolve(strict=True) == path
        assert all(not p.is_symlink() for p in (path, *path.parents))
        info = path.stat()
        assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        assert (info.st_dev, info.st_ino) not in identities
        identities.add((info.st_dev, info.st_ino))

def release_bytes(body):
    changed = body
    for name in (b'REVIEWED_SCHEMA', b'MEASURED_IMPLEMENTATION'):
        false = rb'(?m)^'+name+rb' = False$'
        true = rb'(?m)^'+name+rb' = True$'
        assert len(re.findall(false, changed)) == 1 and not re.findall(true, changed)
        changed = re.sub(false, name+b' = True', changed)
    restored = changed
    for name in (b'REVIEWED_SCHEMA', b'MEASURED_IMPLEMENTATION'):
        restored = re.sub(rb'(?m)^'+name+rb' = True$', name+b' = False', restored)
    assert restored == body
    return changed

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--capsule')
    parser.add_argument('--inventory-sha256')
    args = parser.parse_args()
    assert Path('/.dockerenv').exists()
    if args.self_test:
        good = b'REVIEWED_SCHEMA = False\nMEASURED_IMPLEMENTATION = False\nx = 1\n'
        assert release_bytes(good) == good.replace(b' = False', b' = True')
        failures = 0
        for bad in (good+good, good.replace(b'REVIEWED_SCHEMA = False', b'REVIEWED_SCHEMA = True'),
                    good.replace(b'MEASURED_IMPLEMENTATION = False', b'MEASURED_IMPLEMENTATION=False')):
            try:
                release_bytes(bad)
            except AssertionError:
                failures += 1
            else:
                raise AssertionError('invalid release source accepted')
        good_names=['a/z.txt','a-y.txt','source/code/x.cpp','source-fingerprint.txt']
        assert inventory_names([{'path':n} for n in good_names])==good_names
        for names in (sorted(good_names),list(reversed(good_names)),good_names+[good_names[0]],['a/../x'],['/absolute']):
            try:inventory_names([{'path':n} for n in names])
            except AssertionError:failures+=1
            else:raise AssertionError('invalid inventory matrix accepted')
        print(json.dumps({'status':'passed','negative_fixtures':failures,'quality_reads':0,
                          'archive_reads':0,'releases':0}))
        return
    assert args.capsule and re.fullmatch('[0-9a-f]{64}', args.inventory_sha256 or '')
    capsule = Path(args.capsule)
    assert capsule.parent == ROOT and re.fullmatch('structured-hard-timing-[A-Za-z0-9]{6}', capsule.name)
    originals = [ORIGINAL/name for name in sorted(PINS)]
    captures = [capsule/'independent-reader'/name for name in sorted(PINS)]
    inventory_path = capsule/'artifact-integrity.json'
    complete_path = capsule/'results/complete.json'
    corrected_paths = [CORRECTED/name for name in sorted(CORRECTED_PINS)]
    paths = originals + captures + corrected_paths + [inventory_path, complete_path, REPAIR_PROOF]
    admit(paths)  # Complete bounded metadata/SOURCE matrix precedes every body/hash.
    before = {str(path): path.read_bytes() for path in paths}
    assert sha(before[str(inventory_path)]) == args.inventory_sha256
    inventory = json.loads(before[str(inventory_path)])
    assert inventory['protocol'] == 'structured-hard-timing-comparison-v2' and inventory['status'] == 'complete'
    rows = inventory['files']
    names = inventory_names(rows)
    index = {row['path']: row for row in rows}
    for path in captures + [complete_path]:
        row = index[path.relative_to(capsule).as_posix()]
        assert row['bytes'] == len(before[str(path)]) and row['sha256'] == sha(before[str(path)])
    for name, original, captured in zip(sorted(PINS), originals, captures):
        assert sha(before[str(original)]) == PINS[name] and before[str(original)] == before[str(captured)]
    seal = json.loads(before[str(ORIGINAL/'reader-seal.json')])
    assert seal['status'] == 'passed' and seal['measured_execution_enabled'] is False
    assert [r['path'] for r in seal['files']] == sorted(set(PINS)-{'reader-seal.json'})
    for row in seal['files']:
        body = before[str(ORIGINAL/row['path'])]
        assert row['bytes'] == len(body) and row['sha256'] == sha(body)
    complete = json.loads(before[str(complete_path)])
    assert complete['status'] == 'complete' and complete['protocol'] == inventory['protocol']
    assert complete['encoder_trajectories'] == 10 and complete['encoder_updates_each'] == 512
    assert complete['retained_points'] == 20 and complete['sampled_rows'] == 40960 and complete['skipped_attempts'] == 0
    assert sha(before[str(REPAIR_PROOF)]) == REPAIR_PROOF_SHA
    repair = json.loads(before[str(REPAIR_PROOF)])
    assert repair['status']=='passed' and repair['exact_reverse_byte_proof'] is True
    assert repair['original_source_sha256']==PINS[ENTRY] and repair['corrected_false_source_sha256']==CORRECTED_PINS[ENTRY]
    assert repair['numerical_changes']==repair['schema_or_metric_changes']==repair['archive_reads']==repair['quality_reads']==0
    for path in corrected_paths:
        assert sha(before[str(path)])==CORRECTED_PINS[path.name]
    corrected_seal=json.loads(before[str(CORRECTED/'reader-seal.json')])
    assert corrected_seal['status']=='passed' and corrected_seal['measured_execution_enabled'] is False
    assert corrected_seal['source_sha256']==CORRECTED_PINS[ENTRY] and corrected_seal['fixtures_sha256']==CORRECTED_PINS['source-fixtures.json']
    assert [r['path'] for r in corrected_seal['files']]==sorted(set(CORRECTED_PINS)-{'reader-seal.json'})
    for row in corrected_seal['files']:
        payload=before[str(CORRECTED/row['path'])]
        assert row['bytes']==len(payload) and row['sha256']==sha(payload)
    restored=re.sub(rb'(?ms)^[ ]*# BEGIN_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR\n.*?^[ ]*# END_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR\n',b'',before[str(CORRECTED/ENTRY)])
    for pair in repair['replacement_pairs']:
        old,new=pair['original'].encode(),pair['corrected'].encode()
        assert restored.count(new)==1
        restored=restored.replace(new,old,1)
    assert restored==before[str(ORIGINAL/ENTRY)]
    released = release_bytes(before[str(CORRECTED/ENTRY)])
    leaf = ROOT/'audit-tools'/('independent-structured-hard-timing-released-'+capsule.name[-6:]+'-v3')
    assert leaf.parent.resolve(strict=True) == leaf.parent and all(not p.is_symlink() for p in leaf.parents)
    assert not leaf.exists() and not leaf.is_symlink()
    leaf.mkdir()
    for name in sorted(PINS):
        with (leaf/name).open('xb') as out:
            out.write(released if name == ENTRY else before[str(CORRECTED/name)])
    proof = {'status':'passed','capsule':str(capsule),'inventory_sha256':args.inventory_sha256,
             'prequality_source_sha256':PINS[ENTRY],'released_source_sha256':sha(released),
             'fixtures_sha256':PINS['source-fixtures.json'],'seal_sha256':PINS['reader-seal.json'],
             'exact_two_line_reverse_byte_proof':True,'numerical_changes':0,'archive_reads':0,
             'corrected_false_source_sha256':CORRECTED_PINS[ENTRY],'corrected_fixtures_sha256':CORRECTED_PINS['source-fixtures.json'],
             'corrected_seal_sha256':CORRECTED_PINS['reader-seal.json'],'metadata_repair_proof_sha256':REPAIR_PROOF_SHA,
             'metadata_repair_reverse_to_original':True,
             'original_bundle_preserved':True,'release_utility_sha256':sha(Path(__file__).read_bytes())}
    for path in paths:
        assert path.read_bytes() == before[str(path)]
    with (leaf/'release-proof.json').open('x', encoding='utf-8') as out:
        json.dump(proof,out,indent=2,allow_nan=False)
        out.write('\n')
    print(json.dumps(proof | {'entry':str(leaf/ENTRY),'release_proof_sha256':sha((leaf/'release-proof.json').read_bytes())}))

if __name__ == '__main__':
    main()

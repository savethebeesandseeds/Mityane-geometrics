#!/usr/bin/env python3
"""Derive the one-copy release command for the reviewed inventory repair."""
import hashlib
from pathlib import Path

ROOT=Path('/embedding/output/runs/rpb-structured-hard-timing')
source=ROOT/'audit-tools/release_reader_v2.py'
original=source.read_bytes()
assert hashlib.sha256(original).hexdigest()=='e15c0e2a9c8731a1a9d0dd72f8897c354173670a0d07097f172e3dbba591f11e'
body=original.decode()
body=body.replace('from pathlib import Path\n','from pathlib import Path, PurePosixPath\n',1)
insert="""CORRECTED = ROOT/'audit-tools/independent-structured-hard-timing-v3-ordering-repair'
REPAIR_PROOF = ROOT/'source-review/inventory-ordering-repair-v3-proof.json'
REPAIR_PROOF_SHA = '2a3b052f1282275c7d0b8c5db2961cbf3c8b3722db4b853cb1378342c126dfc9'
CORRECTED_PINS = dict(PINS, **{
    ENTRY: '283ce3ec28c133c0d4053f4a49f08b134f51e6d3efa569c0a2199b942a3a5e7f',
    'source-fixtures.json': '980f2562015465447ab9bd8c9005dffdfaa5c3f1caedc6c02914c704f575ff1e',
    'reader-seal.json': 'f28e9e4edeb6980de5a30bdbe14899444b32f16a2814023b1f31aca837834f7a',
})

def inventory_names(rows):
    names=[row['path'] for row in rows]
    assert all(isinstance(n,str) and str(PurePosixPath(n))==n and not n.startswith('/') and '\\\\' not in n and
               all(x not in ('','.','..') for x in n.split('/')) for n in names)
    assert len(names)==len(set(names)) and names==sorted(names,key=PurePosixPath)
    return names

"""
body=body.replace('def sha(body):',insert+'def sha(body):',1)
body=body.replace("    paths = originals + captures + [inventory_path, complete_path]\n",
                  "    corrected_paths = [CORRECTED/name for name in sorted(CORRECTED_PINS)]\n    paths = originals + captures + corrected_paths + [inventory_path, complete_path, REPAIR_PROOF]\n",1)
old="    names = [row['path'] for row in rows]\n    assert names == sorted(set(names)) and all(isinstance(n,str) and not n.startswith('/') and '\\\\' not in n and\n                                               all(x not in ('','.','..') for x in n.split('/')) for n in names)\n"
assert body.count(old)==1
body=body.replace(old,"    names = inventory_names(rows)\n",1)
repair_checks="""    assert sha(before[str(REPAIR_PROOF)]) == REPAIR_PROOF_SHA
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
    restored=re.sub(rb'(?ms)^[ ]*# BEGIN_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR\\n.*?^[ ]*# END_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR\\n',b'',before[str(CORRECTED/ENTRY)])
    for pair in repair['replacement_pairs']:
        old,new=pair['original'].encode(),pair['corrected'].encode()
        assert restored.count(new)==1
        restored=restored.replace(new,old,1)
    assert restored==before[str(ORIGINAL/ENTRY)]
"""
needle="    released = release_bytes(before[str(ORIGINAL/ENTRY)])\n"
assert body.count(needle)==1
body=body.replace(needle,repair_checks+"    released = release_bytes(before[str(CORRECTED/ENTRY)])\n",1)
body=body.replace("+capsule.name[-6:]+'-v2')","+capsule.name[-6:]+'-v3')",1)
body=body.replace("out.write(released if name == ENTRY else before[str(ORIGINAL/name)])","out.write(released if name == ENTRY else before[str(CORRECTED/name)])",1)
body=body.replace("'exact_two_line_reverse_byte_proof':True,'numerical_changes':0,'archive_reads':0,",
                  "'exact_two_line_reverse_byte_proof':True,'numerical_changes':0,'archive_reads':0,\n             'corrected_false_source_sha256':CORRECTED_PINS[ENTRY],'corrected_fixtures_sha256':CORRECTED_PINS['source-fixtures.json'],\n             'corrected_seal_sha256':CORRECTED_PINS['reader-seal.json'],'metadata_repair_proof_sha256':REPAIR_PROOF_SHA,\n             'metadata_repair_reverse_to_original':True,",1)
test="""        good_names=['a/z.txt','a-y.txt','source/code/x.cpp','source-fingerprint.txt']
        assert inventory_names([{'path':n} for n in good_names])==good_names
        for names in (sorted(good_names),list(reversed(good_names)),good_names+[good_names[0]],['a/../x'],['/absolute']):
            try:inventory_names([{'path':n} for n in names])
            except AssertionError:failures+=1
            else:raise AssertionError('invalid inventory matrix accepted')
"""
body=body.replace("        print(json.dumps({'status':'passed','negative_fixtures':failures",test+"        print(json.dumps({'status':'passed','negative_fixtures':failures",1)
target=ROOT/'audit-tools/release_reader_v3_ordering.py'
with target.open('xb') as out:out.write(body.encode())
assert source.read_bytes()==original
print(str(target)+' '+hashlib.sha256(target.read_bytes()).hexdigest())

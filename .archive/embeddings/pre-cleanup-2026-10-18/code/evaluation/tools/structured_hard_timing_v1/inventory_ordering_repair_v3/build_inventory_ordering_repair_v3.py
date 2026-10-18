#!/usr/bin/env python3
"""Derive a blocked metadata-order repair from the immutable prequality reader."""
import ast
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path('/embedding/output/runs/rpb-structured-hard-timing')
ORIGINAL = ROOT/'audit-tools/independent-structured-hard-timing-v2-prequality'
DEST = ROOT/'audit-tools/independent-structured-hard-timing-v3-ordering-repair'
ENTRY = 'validate_structured_hard_timing.py'
SOURCE_SHA = '2143d8d01b8a549e3a361d0b12b97c53abc87c57ca57714e68f890d6390a7708'
FIXTURE_SHA = 'f8957e3045149c8662cac2357c65b4998324e1010b01b9717f742bf07c48c2c6'
SEAL_SHA = '47852a555d91addc3d7b3cd545cd6aa107d6ac4339ece80e202b2ab5b00e38b0'
START = '# BEGIN_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR'
END = '# END_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR'

def sha(body):
    return hashlib.sha256(body).hexdigest()

assert Path('/.dockerenv').exists()
original_files = {p.name:p.read_bytes() for p in ORIGINAL.iterdir()}
assert sha(original_files[ENTRY]) == SOURCE_SHA
assert sha(original_files['source-fixtures.json']) == FIXTURE_SHA
assert sha(original_files['reader-seal.json']) == SEAL_SHA
before = original_files[ENTRY].decode()
assert '\r' not in before and not DEST.exists()
old_matrix = "    rows = inventory['files']; names = [relative_name(x['path']) for x in rows]\n    check(names==sorted(set(names)) and 'artifact-integrity.json' not in names,'whole sorted inventory matrix')\n"
new_matrix = "    rows = inventory['files']; names = capsule_inventory_names(rows)\n    check('artifact-integrity.json' not in names,'inventory excludes itself')\n"
old_actual = "    actual_names = sorted(p.relative_to(capsule).as_posix() for p in capsule.rglob('*') if p.is_file() and p!=inventory_path)\n"
new_actual = "    actual_names = sorted((p.relative_to(capsule).as_posix() for p in capsule.rglob('*') if p.is_file() and p!=inventory_path),key=PurePosixPath)\n"
old_return = "            'fixtures_sha256':seal['fixtures_sha256'],'two_flag_reverse_byte_proof':True,'modules':MODULE_PINS}\n"
new_return = "            'fixtures_sha256':seal['fixtures_sha256'],'two_flag_reverse_byte_proof':True,'modules':MODULE_PINS,\n            'inventory_ordering_metadata_repair':{'order':'PurePosixPath component order','original_prequality_source_sha256':'"+SOURCE_SHA+"','full_reverse_to_original':True,'numerical_changes':0}}\n"
patches = [(old_matrix,new_matrix),(old_actual,new_actual),(old_return,new_return)]
after = before
for old,new in patches:
    assert after.count(old) == 1
    after = after.replace(old,new,1)
helper = START+"\ndef capsule_inventory_names(rows):\n    names=[relative_name(row['path']) for row in rows]\n    check(len(names)==len(set(names)) and names==sorted(names,key=PurePosixPath),'unique normalized component-ordered capsule inventory')\n    return names\n"+END+"\n"
assert after.count('def admit_capsule(capsule):') == 1
after = after.replace('def admit_capsule(capsule):',helper+'def admit_capsule(capsule):',1)
normalizer = "    "+START+"\n"
normalizer += "    normalized,metadata_blocks=re.subn(rb'(?ms)^[ ]*# BEGIN_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR\\n.*?^[ ]*# END_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR\\n',b'',normalized)\n"
normalizer += "    check(metadata_blocks==3,'exact three reviewed metadata repair blocks')\n"
normalizer += "    for original_line,repaired_line in "+repr(patches)+":\n"
normalizer += "        check(normalized.count(repaired_line.encode())==1,'one exact reviewed inventory metadata replacement')\n"
normalizer += "        normalized=normalized.replace(repaired_line.encode(),original_line.encode(),1)\n"
normalizer += "    "+END+"\n"
needle = "    check(normalized==before,'full two-line reverse byte proof, no numeric change')\n"
assert after.count(needle) == 1
after = after.replace(needle,normalizer+needle,1)
fixture = "    "+START+"\n"
fixture += "    good_names=['a/z.txt','a-y.txt','source/code/x.cpp','source-fingerprint.txt']\n"
fixture += "    check(capsule_inventory_names([{'path':name} for name in good_names])==good_names,'folder-prefix and hyphen-file component order')\n"
fixture += "    for bad_names in (sorted(good_names),list(reversed(good_names)),good_names+[good_names[0]],['a/../x'],['/absolute'],['a\\\\b']):\n"
fixture += "        try:capsule_inventory_names([{'path':name} for name in bad_names])\n"
fixture += "        except AssertionError:negatives+=1\n"
fixture += "        else:raise AssertionError('invalid capsule ordering/name accepted')\n"
fixture += "    "+END+"\n"
assert after.count('    negatives=0\n') == 1
after = after.replace('    negatives=0\n','    negatives=0\n'+fixture,1)
ast.parse(after)
restored = __import__('re').sub(r'(?ms)^[ ]*# BEGIN_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR\n.*?^[ ]*# END_CAPSULE_COMPONENT_ORDER_METADATA_REPAIR\n','',after)
for old,new in patches:
    assert restored.count(new) == 1
    restored = restored.replace(new,old,1)
assert restored == before
old_functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(before).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new_functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(after).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
changed=[name for name in old_functions if old_functions[name]!=new_functions[name]]
assert changed == ['admit_capsule','release_binding','self_test']
DEST.mkdir()
for name in ('archive_codec.py','legacy_implementation.py','saved_cpu_math.py'):
    (DEST/name).write_bytes(original_files[name])
(DEST/ENTRY).write_bytes(after.encode())
subprocess.run([sys.executable,'-B',str(DEST/ENTRY),'--self-test','--fixture-output',str(DEST/'source-fixtures.json')],check=True)
fixture_record=json.loads((DEST/'source-fixtures.json').read_text())
assert fixture_record['status']=='passed' and fixture_record['archive_reads']==fixture_record['quality_reads']==0
assert fixture_record['measured_execution_enabled'] is False
records=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(DEST.iterdir())]
seal={'status':'passed','protocol':'structured-hard-timing-comparison-v2','source_sha256':sha(after.encode()),
      'fixtures_sha256':sha((DEST/'source-fixtures.json').read_bytes()),'files':records,
      'measured_execution_enabled':False,'archive_reads':0,'quality_reads':0,
      'card_sha256':'35f7aa9987f293bafbe735506cc86cd2af02f2ab39bec42dd1dc2d2dedfa758c',
      'checks':fixture_record['checks'],'negative_fixtures':fixture_record['negative_fixtures'],
      'repair':'capsule inventory component ordering only; original captured prequality seal preserved',
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (DEST/'reader-seal.json').open('x',encoding='utf-8') as out:json.dump(seal,out,indent=2);out.write('\n')
diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='sealed-v2',tofile='blocked-v3-ordering-repair'))
proof={'status':'passed','original_source_sha256':SOURCE_SHA,'original_fixtures_sha256':FIXTURE_SHA,'original_seal_sha256':SEAL_SHA,
       'corrected_false_source_sha256':sha(after.encode()),'fixtures_sha256':seal['fixtures_sha256'],
       'seal_sha256':sha((DEST/'reader-seal.json').read_bytes()),'exact_reverse_byte_proof':True,
       'changed_existing_functions':changed,'added_function':'capsule_inventory_names','unchanged_existing_functions':len(old_functions)-len(changed),
       'numerical_changes':0,'schema_or_metric_changes':0,'archive_reads':0,'quality_reads':0,
       'ordering':'normalized relative names sorted by PurePosixPath; source manifests/seals retain string order',
       'replacement_pairs':[{'original':old,'corrected':new} for old,new in patches],'diff':diff}
proof_path=ROOT/'source-review/inventory-ordering-repair-v3-proof.json'
with proof_path.open('x',encoding='utf-8') as out:json.dump(proof,out,indent=2);out.write('\n')
for name,body in original_files.items():assert (ORIGINAL/name).read_bytes()==body
print(json.dumps({'status':'passed','bundle':str(DEST),'source_sha256':seal['source_sha256'],'fixtures_sha256':seal['fixtures_sha256'],
                  'seal_sha256':proof['seal_sha256'],'proof_sha256':sha(proof_path.read_bytes()),'checks':seal['checks'],
                  'negative_fixtures':seal['negative_fixtures'],'both_flags':False,'archive_reads':0,'quality_reads':0}))

#!/usr/bin/env python3
"""Freeze exact TRAIN-only head roles from the already sealed capsule inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path('/embedding')
CAPSULE = ROOT/'output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS'
INVENTORY_SHA = '9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa'
SUMMARY_SHA = '1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4'
MASTERS = [75272,76373,77474,78575,79676]
METHODS = ['raw','native_late','native_early']
REPS = [2701,2802,2903]

def expected_roles():
    result = []
    for master in MASTERS:
        base = f'results/seed-{master}-lag_sign'
        result.append({'path':base+'/controlled-training.pt','role':'controlled_training','master':master})
        for method in METHODS:
            path = base+'/lag_sign/readouts/'+method
            result.append({'path':path+'/training-features.pt','role':'features','master':master,'method':method})
            for rep in REPS:
                for role,name in [('fit','fit.pt'),('predictions','training-predictions.pt')]:
                    result.append({'path':path+f'/rep-{rep}/'+name,'role':role,'master':master,'method':method,'repetition':rep})
    return sorted(result,key=lambda x:x['path'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    assert Path('/.dockerenv').is_file()
    inventory=CAPSULE/'artifact-integrity.json'; s=inventory.stat()
    assert inventory.resolve(strict=True)==inventory and all(not q.is_symlink() for q in (inventory,*inventory.parents))
    assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
    output=Path(a.output)
    assert output.parent==ROOT/'code/evaluation/tools/structured_hard_timing_train_v1' and output.name=='head_roles.json'
    assert not output.exists() and output.parent.resolve(strict=True)==output.parent
    body=inventory.read_bytes();assert hashlib.sha256(body).hexdigest()==INVENTORY_SHA
    entries=json.loads(body);assert entries['status']=='complete' and entries['protocol']=='structured-hard-timing-comparison-v2'
    records={x['path']:x for x in entries['files']}
    roles=[]
    for role in expected_roles():
        item=records[role['path']]
        assert item['bytes']>0 and len(item['sha256'])==64
        roles.append({**role,'bytes':item['bytes'],'sha256':item['sha256']})
    assert len(roles)==110 and len({x['path'] for x in roles})==110
    manifest={'protocol':'structured-hard-timing-train-diagnostic-v1','component':'fixed_head_train',
              'capsule':str(CAPSULE),'inventory_sha256':INVENTORY_SHA,
              'validation_metadata':'doc/results/structured_hard_timing_comparison_v2.json',
              'validation_metadata_sha256':SUMMARY_SHA,'master_seeds':MASTERS,'methods':METHODS,
              'repetitions':REPS,'unique_train_archive_roles':110,'rows_each':256,'source_pairs_each':128,
              'role_records':roles,'VAL_tensor_roles':0,'model_checkpoint_roles':0,'head_refits':0,'PCA_fits':0}
    with output.open('x',encoding='utf-8',newline='\n') as out: out.write(json.dumps(manifest,indent=2)+'\n')
    assert inventory.read_bytes()==body
    print(json.dumps({'status':'passed','roles':len(roles),'manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                      'archive_opens':0,'output':str(output)}))

if __name__=='__main__':main()

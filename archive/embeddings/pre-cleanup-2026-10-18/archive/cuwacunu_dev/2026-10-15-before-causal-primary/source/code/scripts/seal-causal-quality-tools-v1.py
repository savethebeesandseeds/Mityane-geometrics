#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Seal compact metadata-only proof audit/report/guard sources before quality."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/cuwacunu')
EXPECTED={
 'code/scripts/audit-causal-patch-bottleneck-mae-v1.py':'7dc467e7a198f0c9674f123fc541e317da64891c8b8e498c0ed6df5b234ab896',
 'code/scripts/report-causal-patch-bottleneck-mae-v1.py':'1114a7fe95e1c8af48ca3449370d5b3c5650e46ff76945115012df574bc15dd9',
 'code/scripts/quality-causal-patch-bottleneck-mae-v1.py':'4272284c384dc2af2e029e39d0ca73f6fb8c0592208c44df811aec71b7cdc812',
}
def require(value,why):
    if not value: raise RuntimeError(why)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p): return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engineering',type=Path,required=True)
    a=p.parse_args()
    require(Path('/.dockerenv').exists() and Path.cwd()==ROOT,'Existing container only')
    leaf=a.engineering.resolve(strict=True)
    require(leaf.is_relative_to(ROOT/'output/engineering/causal-patch-bottleneck-mae-v1'),'New engineering leaf required')
    parent=ROOT/'output/runs/causal-patch-bottleneck-mae-v1'
    require(not list(parent.glob('run-*')) and not (parent/'quality-card-claim.json').exists(),'Quality must not have started')
    files=[]
    for name,digest in EXPECTED.items():
        path=ROOT/name
        require(sha(path)==digest,'Final peer-reviewed tool hash: '+name)
        files.append(path)
    files.append(Path(__file__).resolve())
    capture=leaf/'quality-tools-source'
    require(not capture.exists(),'Preserve prior tool seal')
    capture.mkdir()
    for path in files:
        target=capture/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f: f.write(path.read_bytes())
        require(sha(target)==sha(path),'Exact compact tool copy')
    receipt=ROOT/'output/engineering/location-scale-transfer-build/attempt-01/build-receipt.json'
    old=json.loads(receipt.read_text())['compiled_repository_dependencies']
    prefix=ROOT/'code/evaluation/market_synthetic_transfer_v1/deps'
    old_files=[Path(name) for name in old if Path(name).is_relative_to(prefix)]
    require(len(old_files)==8,'Eight unchanged head dependency sources')
    for path in old_files:
        require(sha(path)==old[str(path)]['sha256'],'Original fixed-head source hash')
    files+=old_files
    for stem in ('data','feature_harness','fixed_feature_readouts'):
        files.append(Path('/opt/cuwacunu/build/market-location-scale-transfer')/f'market_location_scale_synthetic_transfer_{stem}.d')
    with (leaf/'quality-tools-admission.json').open('x',encoding='utf-8') as f:
        json.dump({'kind':'causal_patch_bottleneck_mae_v1_quality_tools_admission','status':'passed',
          'sources':[pin(path) for path in files], 'source_capsule':str(capture),
          'admitted_before_quality':True,'model_calls':0,'tensor_deserializations':0},f,indent=2)
        f.write('\n')
    print('QUALITY_TOOLS_ADMITTED='+str(leaf/'quality-tools-admission.json'))
if __name__=='__main__': main()

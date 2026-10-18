#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Once-only independent saved-TRAIN audit recorder; no producer imports."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/opt/cuwacunu/research/causal-v9-v11-train-diagnosis-audit/attempt-001')


def pin(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink():raise RuntimeError('Regular nonlinked metadata file required')
    data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--audit-card-sha256',required=True)
    parser.add_argument('--stage-sha256',required=True)
    args=parser.parse_args()
    root=args.root.resolve(strict=True)
    assert sys.platform.startswith('linux') and Path('/.dockerenv').is_file() and root==ROOT
    source=root/'source'
    assert pin(root/'source-stage.json')['sha256']==args.stage_sha256,'External exact Stage before JSON parse'
    stage=read(root/'source-stage.json')
    assert pin(source/'CAPSULE_MANIFEST.json')['sha256']==stage['manifest_sha256'],'Exact capsule before JSON parse'
    assert pin(source/'audit-card.json')['sha256']==args.audit_card_sha256,'External exact audit card before JSON parse'
    capsule=read(source/'CAPSULE_MANIFEST.json')
    assert stage['kind']=='cpb_v9_v11_TRAIN_independent_audit_source_stage_v1'
    assert stage['status']=='passed_AST_only' and stage['module_executions']==0
    assert stage['files']>0 and stage['AST_files']==4 and stage['data_guard_occurrences']>=72
    assert stage['audit_card_sha256']==capsule['audit_card_sha256']==args.audit_card_sha256
    assert stage['manifest_sha256']==pin(source/'CAPSULE_MANIFEST.json')['sha256']
    assert stage['source_zip_sha256']==pin(root/'source.zip')['sha256'] and len(capsule['files'])==stage['files']
    assert len({row['file'] for row in capsule['files']})==stage['files']
    before=[]
    for row in capsule['files']:
        relative=Path(row['file']);assert not relative.is_absolute() and '..' not in relative.parts
        actual=pin(source/relative);assert all(actual[key]==row[key] for key in ('bytes','sha256'))
        before.append(actual)
    card=read(source/'audit-card.json')
    assert card['kind']=='cpb_v9_v11_TRAIN_diagnosis_independent_audit_card_v1' and card['execution_admitted'] is True
    assert len(card['auditor_sources'])==5 and stage['files']==len(card['auditor_sources'])+5
    assert stage['main_input_binding_counts']==card['main_input_binding_counts']
    assert stage['data_guard_occurrences']==card['main_input_binding_counts']['occurrences']
    assert stage['unique_container_physical_files']==card['main_input_binding_counts']['unique_container_paths']
    assert card['execution_wrapper']['sha256']==pin(__file__)['sha256']
    assert card['main_card']['sha256']==pin(source/'MAIN_CARD.json')['sha256']==stage['main_card_sha256']
    assert card['root_supplied_main_card_sha256']==stage['main_card_sha256']
    assert stage['main_diagnosis_sha256']==pin(source/'diagnosis.json')['sha256']
    assert stage['execution_wrapper_sha256']==pin(__file__)['sha256']
    assert all(card['main_diagnosis'][key]==pin(source/'diagnosis.json')[key] for key in ('bytes','sha256'))
    output=root/'run-001';assert not output.exists();output.mkdir()
    command=[sys.executable,str(source/'audit.py'),'--card',str(source/'MAIN_CARD.json'),
             '--diagnosis',str(source/'diagnosis.json'),'--audit-card',str(source/'audit-card.json'),
             '--audit-card-sha256',args.audit_card_sha256,'--output',str(output/'audit.json')]
    write(output/'invocation.json',dict(argv=command,source_before=before,wrapper=pin(__file__),
          audit_card=pin(source/'audit-card.json'),externally_reviewed_audit_card_sha256=args.audit_card_sha256,
          externally_reviewed_stage_sha256=args.stage_sha256,model_calls=0,head_fits=0,training_updates=0))
    with Path('/opt/cuwacunu/research/causal-v9-v11-neutral-TRAIN-audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat();started=time.monotonic()
        with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
            finished=subprocess.run(command,stdout=stdout,stderr=stderr,check=False,
                                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        elapsed=time.monotonic()-started
    after=[pin(source/row['file']) for row in capsule['files']]
    result=read(output/'audit.json') if (output/'audit.json').is_file() else None
    status='passed' if finished.returncode==0 and result and result['status']=='passed' and not result['mismatches'] and before==after else 'failed'
    execution=dict(kind='cpb_v9_v11_TRAIN_independent_audit_execution_v1',status=status,exit_code=finished.returncode,
        argv=command,started_utc=utc,elapsed_seconds=elapsed,source_before=before,source_after=after,
        wrapper=pin(__file__),audit_card=pin(source/'audit-card.json'),stdout=pin(output/'stdout.log'),
        externally_reviewed_audit_card_sha256=args.audit_card_sha256,externally_reviewed_stage_sha256=args.stage_sha256,
        stderr=pin(output/'stderr.log'),model_calls=0,head_fits=0,training_updates=0)
    if result is not None:execution['result']=pin(output/'audit.json')
    write(output/'execution.json',execution)
    inventory=[dict(relative=p.relative_to(root).as_posix(),**pin(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    write(root/'mirror-inventory.json',dict(kind='cpb_v9_v11_TRAIN_independent_audit_inventory_v1',status=status,
        externally_reviewed_audit_card_sha256=args.audit_card_sha256,externally_reviewed_stage_sha256=args.stage_sha256,files=inventory))
    print(json.dumps(dict(status=status,elapsed_seconds=elapsed,result=execution.get('result'))))
    return 0 if status=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

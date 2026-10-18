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

ROOT=Path('/opt/cuwacunu/research/causal-v6-v7-train-diagnosis-audit/attempt-001')


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
    root=parser.parse_args().root.resolve(strict=True)
    assert sys.platform.startswith('linux') and Path('/.dockerenv').is_file() and root==ROOT
    source=root/'source';stage=read(root/'source-stage.json');capsule=read(source/'CAPSULE_MANIFEST.json')
    assert stage['status']=='passed_AST_only' and stage['module_executions']==0
    assert stage['manifest_sha256']==pin(source/'CAPSULE_MANIFEST.json')['sha256']
    assert stage['source_zip_sha256']==pin(root/'source.zip')['sha256'] and len(capsule['files'])==10
    before=[]
    for row in capsule['files']:
        relative=Path(row['file']);assert not relative.is_absolute() and '..' not in relative.parts
        actual=pin(source/relative);assert all(actual[key]==row[key] for key in ('bytes','sha256'))
        before.append(actual)
    card=read(source/'audit-card.json')
    assert card['kind']=='cpb_v6_v7_TRAIN_diagnosis_independent_audit_card_v1'
    assert card['execution_wrapper']['sha256']==pin(__file__)['sha256']
    assert card['main_card']['sha256']==pin(source/'MAIN_CARD.json')['sha256']=='6067c3f5128991619e72197dbc5b8ea3f44f6e7e4186a56f4962f70c43b03304'
    assert all(card['main_diagnosis'][key]==pin(source/'diagnosis.json')[key] for key in ('bytes','sha256'))
    output=root/'run-001';assert not output.exists();output.mkdir()
    command=[sys.executable,str(source/'audit.py'),'--card',str(source/'MAIN_CARD.json'),
             '--diagnosis',str(source/'diagnosis.json'),'--audit-card',str(source/'audit-card.json'),
             '--output',str(output/'audit.json')]
    write(output/'invocation.json',dict(argv=command,source_before=before,wrapper=pin(__file__),
          audit_card=pin(source/'audit-card.json'),model_calls=0,head_fits=0,training_updates=0))
    with Path('/opt/cuwacunu/research/causal-v6-v7-neutral-TRAIN-audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat();started=time.monotonic()
        with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
            finished=subprocess.run(command,stdout=stdout,stderr=stderr,check=False,
                                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        elapsed=time.monotonic()-started
    after=[pin(source/row['file']) for row in capsule['files']]
    result=read(output/'audit.json') if (output/'audit.json').is_file() else None
    status='passed' if finished.returncode==0 and result and result['status']=='passed' and not result['mismatches'] and before==after else 'failed'
    execution=dict(kind='cpb_v6_v7_TRAIN_independent_audit_execution_v1',status=status,exit_code=finished.returncode,
        argv=command,started_utc=utc,elapsed_seconds=elapsed,source_before=before,source_after=after,
        wrapper=pin(__file__),audit_card=pin(source/'audit-card.json'),stdout=pin(output/'stdout.log'),
        stderr=pin(output/'stderr.log'),model_calls=0,head_fits=0,training_updates=0)
    if result is not None:execution['result']=pin(output/'audit.json')
    write(output/'execution.json',execution)
    inventory=[dict(relative=p.relative_to(root).as_posix(),**pin(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    write(root/'mirror-inventory.json',dict(kind='cpb_v6_v7_TRAIN_independent_audit_inventory_v1',status=status,files=inventory))
    print(json.dumps(dict(status=status,elapsed_seconds=elapsed,result=execution.get('result'))))
    return 0 if status=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

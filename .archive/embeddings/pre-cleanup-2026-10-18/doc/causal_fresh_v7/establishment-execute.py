#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Once-only metadata-join recorder; never imports scientific producer code."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time


def pin(path):
    path=Path(path).resolve(strict=True)
    data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve(strict=True)
    assert Path('/.dockerenv').is_file() and str(root)=='/opt/cuwacunu/research/causal-fresh-v7-establishment/attempt-001'
    source=root/'source';stage=read(root/'source-stage.json');capsule=read(source/'CAPSULE_MANIFEST.json')
    assert stage['status']=='passed_AST_only' and stage['module_executions']==0
    assert stage['manifest_sha256']==pin(source/'CAPSULE_MANIFEST.json')['sha256']
    assert stage['source_zip_sha256']==pin(root/'source.zip')['sha256']
    before=[]
    for item in capsule['files']:
        value=pin(source/item['file'])
        assert value['bytes']==item['bytes'] and value['sha256']==item['sha256']
        before.append(value)
    declaration=read(source/'input-manifest.json')
    assert declaration['execution_wrapper']['sha256']==pin(__file__)['sha256']
    assert len(declaration['inputs'])==8 and {r['id'] for r in declaration['inputs']}=={
        'comparison','table_rows','quality_audit','training_audit','complete_mirror','card','quality_receipt','pairings'}
    output=root/'run-001';assert not output.exists();output.mkdir()
    command=[sys.executable,str(source/'establish.py'),'--manifest',str(source/'input-manifest.json'),
             '--output',str(output/'final-establishment.json')]
    write(output/'invocation.json',dict(argv=command,source_before=before,manifest=pin(source/'input-manifest.json'),
          wrapper=pin(__file__),model_calls=0,head_fits=0,training_updates=0))
    with Path('/opt/cuwacunu/research/cpb-v7-metadata-establishment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
            finished=subprocess.run(command,stdout=stdout,stderr=stderr,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=False)
        elapsed=time.monotonic()-started
    after=[pin(source/item['file']) for item in capsule['files']]
    result=read(output/'final-establishment.json') if (output/'final-establishment.json').is_file() else None
    status='passed' if finished.returncode==0 and result and result['status']=='passed' and before==after else 'failed'
    execution=dict(kind='cpb_v7_metadata_establishment_execution_v1',status=status,exit_code=finished.returncode,
        argv=command,started_utc=utc,elapsed_seconds=elapsed,source_before=before,source_after=after,
        stdout=pin(output/'stdout.log'),stderr=pin(output/'stderr.log'),wrapper=pin(__file__),
        manifest=pin(source/'input-manifest.json'),model_calls=0,head_fits=0,training_updates=0)
    if result is not None:execution['result']=pin(output/'final-establishment.json')
    write(output/'execution.json',execution)
    inventory=[dict(relative=p.relative_to(root).as_posix(),**pin(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    write(root/'mirror-inventory.json',dict(kind='cpb_v7_metadata_establishment_inventory_v1',status=status,files=inventory))
    print(json.dumps(dict(status=status,elapsed_seconds=elapsed,result=execution.get('result'))))
    return 0 if status=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())

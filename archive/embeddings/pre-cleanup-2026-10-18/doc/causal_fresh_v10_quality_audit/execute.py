#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""One declared independent audit invocation; recorder only, no model imports."""
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


def pin(path):
    path=Path(path).resolve(strict=True)
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--kind',choices=('quality',),required=True)
    parser.add_argument('--audit-card-sha256',required=True)
    parser.add_argument('--stage-sha256',required=True)
    args=parser.parse_args()
    assert Path('/.dockerenv').is_file()
    root=args.root.resolve(strict=True);source=root/'source'
    assert str(root)==f'/opt/cuwacunu/research/causal-fresh-v10-{args.kind}-audit/attempt-001'
    assert pin(root/'source-stage.json')['sha256']==args.stage_sha256,'External Stage hash before parse'
    assert pin(source/'audit-card.json')['sha256']==args.audit_card_sha256,'External card hash before parse'
    stage=read(root/'source-stage.json')
    assert stage['kind']=='cpb_v10_quality_audit_source_stage_v1' and stage['status']=='passed' and stage['numeric_archive_reads']==0 and stage['source_root']==str(source)
    assert stage['source_manifest_sha256']==pin(source/'SOURCE_MANIFEST.json')['sha256']
    assert stage['source_zip_sha256']==pin(root/'source.zip')['sha256']
    assert stage['audit_card_sha256']==args.audit_card_sha256
    manifest=read(source/'SOURCE_MANIFEST.json')
    assert manifest['container_root']==str(root) and manifest['audit_card_sha256']==args.audit_card_sha256
    before=[]
    for entry in manifest['files']:
        actual=pin(source/entry['file'])
        assert actual['bytes']==entry['bytes'] and actual['sha256']==entry['sha256']
        before.append(actual)
    card=read(source/'audit-card.json')
    assert card['stage']=='declared_after_static_review_before_any_saved_numeric_audit' and card['execution_pending']
    assert card['execution_wrapper']['sha256']==pin(__file__)['sha256']
    assert card['model']=='CPB-v10.market-01' and stage['files']==7 and stage['python_AST_files']==3
    assert card['audit_attempt']=='attempt-001' and card['report_only_recovery_required'] is False
    assert set(card['expected_main_paths'])=={'training','quality','card'}
    main_label=card['main_attempt']
    assert main_label.startswith('attempt-') and len(main_label)==11 and main_label[8:].isdigit()
    attempt=Path('/opt/cuwacunu/research/causal-fresh-v10')/main_label
    main_source=attempt/'source';training=attempt/'run-001';quality=attempt/'quality-001'
    assert card['expected_main_paths']=={'training':str(training),'quality':str(quality),
                                       'card':str(main_source/'doc/CAUSAL_FRESH_V10_CARD.json')}
    assert pin(main_source/'CAPSULE_MANIFEST.json')['sha256']==card['main_source_manifest_sha256']
    assert pin(attempt/'source.zip')['sha256']==card['main_source_zip_sha256']
    required_stage='quality'
    assert Path(card['completed_quality_stage']['path']).is_absolute()
    assert Path(card['completed_quality_stage']['path']).resolve(strict=True)==(attempt/'quality-stage.json').resolve(strict=True)
    assert all(pin(attempt/'quality-stage.json')[key]==card['completed_quality_stage'][key] for key in ('bytes','sha256'))
    assert all(pin(quality/'quality-receipt.json')[key]==card['completed_quality_receipt'][key] for key in ('bytes','sha256')), 'Opaque durable receipt label is not a runtime filesystem path'
    completed=read(attempt/'quality-stage.json')
    assert completed['kind']=='causal_fresh_v10_exclusive_stage_execution_v1' and completed['status']=='passed' and completed['exit_code']==0
    quality_receipt=read(quality/'quality-receipt.json')
    assert quality_receipt['status']=='passed' and quality_receipt['model']=='CPB-v10.market-01'
    assert quality_receipt['card_sha256']==card['main_card']['sha256']
    scientific=('extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads')
    assert set(quality_receipt['stage_seconds'])==set(scientific), 'Receipt has five scientific stages before aggregation'
    for name in (*scientific,'report','comparison'):
        local=read(quality/(name+'-stage.json'))
        assert local['stage']==name and local['exit_code']==0
    assert not {'reporting_recovery_card','reporting_recovery_receipt'}.intersection(card)
    main_inputs=[main_source/'CAPSULE_MANIFEST.json',attempt/'source.zip',attempt/'quality-stage.json',quality/'quality-receipt.json']
    main_before=[pin(path) for path in main_inputs]
    summary=read(training/'training-summary.json')
    assert summary['status']=='passed' and summary['model']=='CPB-v10.market-01' and summary['new_causal_updates']==2048
    output=root/'run-001'
    assert not output.exists()
    output.mkdir()
    command=[sys.executable,str(source/'audit.py'),'--card',str(main_source/'doc/CAUSAL_FRESH_V10_CARD.json'),
                 '--input-manifest',str(main_source/'doc/causal_fresh_v10/input-manifest.json'),
                 '--raw-controls',str(main_source/'doc/causal_fresh_v10/raw-controls.json'),
                 '--training',str(training),'--run',str(quality),'--report',str(quality/'report'),
                 '--quality-receipt',str(quality/'quality-receipt.json'),
                 '--comparison',str(quality/'comparison'),'--audit-card',str(source/'audit-card.json')]
    command+=['--output',str(output/'raw-result.json')]
    write(output/'invocation.json',{'kind':'cpb_v10_independent_audit_invocation_v1','argv':command,
          'main_stage':pin(attempt/(required_stage+'-stage.json')),'audit_card':pin(source/'audit-card.json'),
          'source_stage':pin(root/'source-stage.json'),'source_manifest':pin(source/'SOURCE_MANIFEST.json'),
          'completed_quality_receipt':pin(quality/'quality-receipt.json'),'main_inputs_before':main_before,
          'wrapper':pin(__file__),'source_before':before,'model_calls':0,'head_fits':0,'training_updates':0})
    lock_path=Path('/opt/cuwacunu/research')/f'cpb-v10-independent-{args.kind}-audit.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic();start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
            finished=subprocess.run(command,stdout=stdout,stderr=stderr,env=env,check=False)
        elapsed=time.monotonic()-started
    after=[pin(source/e['file']) for e in manifest['files']]
    main_after=[pin(path) for path in main_inputs]
    result=read(output/'raw-result.json') if (output/'raw-result.json').is_file() else None
    status='passed' if (finished.returncode==0 and result and result['status']=='passed' and before==after and main_before==main_after
                        and stage['source_manifest_sha256']==pin(source/'SOURCE_MANIFEST.json')['sha256']) else 'failed'
    execution={'kind':f'causal_fresh_v10_{args.kind}_audit_execution_v1','status':status,'exit_code':finished.returncode,
               'argv':command,'started_utc':start_utc,'elapsed_seconds':elapsed,'source_before':before,'source_after':after,
               'stdout':pin(output/'stdout.log'),'stderr':pin(output/'stderr.log'),'wrapper':pin(__file__),
               'audit_card':pin(source/'audit-card.json'),'model_calls':0,'head_fits':0,'training_updates':0}
    execution['main_inputs_before']=main_before;execution['main_inputs_after']=main_after
    execution['source_stage']=pin(root/'source-stage.json')
    execution['source_manifest']=pin(source/'SOURCE_MANIFEST.json')
    if result is not None:execution['raw_result']=pin(output/'raw-result.json')
    write(output/'execution.json',execution)
    inventory=[]
    for file in sorted(root.rglob('*')):
        if file.is_file():inventory.append({'relative':file.relative_to(root).as_posix(),**pin(file)})
    write(root/'mirror-inventory.json',{'kind':'cpb_v10_audit_complete_file_inventory_v1','status':status,'files':inventory})
    print(json.dumps({'status':status,'kind':args.kind,'elapsed_seconds':elapsed,'output':str(output),
                      'checks':result.get('checks') if result else None}))
    return 0 if status=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())

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
    parser.add_argument('--workflow-source-review-sha256',required=True)
    args=parser.parse_args()
    assert Path('/.dockerenv').is_file()
    root=args.root.resolve(strict=True);source=root/'source'
    assert str(root)==f'/opt/cuwacunu/research/causal-fresh-v14-{args.kind}-audit/attempt-001'
    assert pin(root/'source-stage.json')['sha256']==args.stage_sha256,'External Stage hash before parse'
    assert pin(source/'audit-card.json')['sha256']==args.audit_card_sha256,'External card hash before parse'
    assert pin(source/'workflow-review.json')['sha256']==args.workflow_source_review_sha256,'External independent workflow review before parse'
    stage=read(root/'source-stage.json')
    assert stage['kind']=='cpb_v14_quality_audit_source_stage_v1' and stage['status']=='passed' and stage['numeric_archive_reads']==0 and stage['source_root']==str(source)
    assert stage['source_manifest_sha256']==pin(source/'SOURCE_MANIFEST.json')['sha256']
    assert stage['source_zip_sha256']==pin(root/'source.zip')['sha256']
    assert stage['audit_card_sha256']==args.audit_card_sha256
    assert stage['workflow_source_review_sha256']==args.workflow_source_review_sha256
    manifest=read(source/'SOURCE_MANIFEST.json')
    assert manifest['container_root']==str(root) and manifest['audit_card_sha256']==args.audit_card_sha256
    before=[]
    for entry in manifest['files']:
        actual=pin(source/entry['file'])
        assert actual['bytes']==entry['bytes'] and actual['sha256']==entry['sha256']
        before.append(actual)
    card=read(source/'audit-card.json')
    assert card['stage']=='declared_after_static_review_before_any_saved_numeric_audit' and card['execution_pending'] is True
    assert card['execution_wrapper']['sha256']==pin(__file__)['sha256']
    assert card['model']=='CPB-v14.market-01'
    assert card['temporal_pooling_geometry']==[4,16,32,8] and card['temporal_pooling_trailing_lengths']==[4,8,12,16]
    assert 'temporal_pooling_partition' not in card
    assert card['temporal_fixture_kernel_calls']==53 and card['engineering_typed_negative_cases']==821
    assert card['engineering_full_invalid_archive_files']==817 and card['engineering_direct_loader_requests']==4
    assert card['engineering_compact_archive_files']==card['engineering_restored_target_controls']==0
    assert card['engineering_legacy_extra_partition_files']==5
    assert card['actual_step_local_weight_fields']==['temporal_pool_local_weights','temporal_contextual_pool_local_weights']
    assert card['actual_step_local_weight_shape']==['R',4,'T']
    assert stage['files']==len(manifest['files'])==card['staging_transport']['payload_count']
    assert stage['ZIP_members']==len(manifest['files'])+1==card['staging_transport']['ZIP_member_count']
    assert stage['python_AST_files']==len(card['staging_transport']['python_AST_files'])==card['staging_transport']['python_AST_count']
    assert {entry['file'] for entry in manifest['files']}==set(card['staging_transport']['payload_files'])
    workflow=read(source/'workflow-review.json')
    assert workflow['kind']=='cpb_v14_quality_audit_workflow_independent_static_review_v1' and workflow['status']=='passed'
    assert workflow['source_only'] is True and workflow['independent_of_author'] is True and workflow['scientific_execution_admitted'] is False
    assert card['independent_workflow_source_review']['sha256']==args.workflow_source_review_sha256==manifest['workflow_source_review_sha256']
    assert pin(source/'workflow-provenance.json')['sha256']==card['helpers_provenance']['sha256']
    provenance=read(source/'workflow-provenance.json')
    expected_root104=provenance['scientific_source_authorities']['combined_root104']
    assert card['independent_source_review']==expected_root104, 'Actual revised root104closure002 card/provenance binding'
    workflow_prefix='doc/causal_fresh_v14_quality_audit/'
    expected_workflow={workflow_prefix+name for name in ('prepare-card.ps1','stage.ps1','execute.py','mirror.ps1','provenance.json')}
    assert len(workflow['reviewed_sources'])==len(expected_workflow) and {p['path'] for p in workflow['reviewed_sources']}==expected_workflow
    assert provenance['kind']=='cpb_v14_owned_quality_audit_orchestration_source_v1'
    assert len(provenance['current_sources'])==len(expected_workflow)-1
    assert {entry['path'] for entry in provenance['current_sources']}==expected_workflow-{workflow_prefix+'provenance.json'}
    for entry in provenance['current_sources']:
        matched=[p for p in workflow['reviewed_sources'] if p['path']==entry['path']]
        assert len(matched)==1 and all(matched[0][k]==entry[k] for k in ('bytes','sha256'))
    own_review=[p for p in workflow['reviewed_sources'] if p['path']==workflow_prefix+'execute.py']
    provenance_review=[p for p in workflow['reviewed_sources'] if p['path']==workflow_prefix+'provenance.json']
    assert len(own_review)==len(provenance_review)==1
    assert all(own_review[0][k]==pin(__file__)[k] for k in ('bytes','sha256'))
    assert all(provenance_review[0][k]==pin(source/'workflow-provenance.json')[k] for k in ('bytes','sha256'))
    assert pin(source/'scientific-source-review.json')['sha256']==card['independent_source_review']['sha256']==manifest['scientific_source_review_sha256']==stage['scientific_source_review_sha256']
    assert pin(source/'quality-source-review.json')['sha256']==card['independent_quality_source_review']['sha256']==manifest['quality_source_review_sha256']==stage['quality_source_review_sha256']
    assert pin(source/'evaluation-source-review.json')['sha256']==card['independent_evaluation_source_review']['sha256']==manifest['evaluation_source_review_sha256']==stage['evaluation_source_review_sha256']
    assert stage['actual_card_peer_review']==manifest['actual_card_peer_review']
    scientific_review=read(source/'scientific-source-review.json')
    quality_review=read(source/'quality-source-review.json')
    evaluation_review=read(source/'evaluation-source-review.json')
    assert scientific_review['kind']=='cpb_v14_complete_104_owned_source_root_closure_v1' and scientific_review['status']=='passed' and len(scientific_review['reviewed_owned_files'])==104
    assert quality_review['kind']=='cpb_v14_independent_quality_auditor_source_review_v1' and quality_review['status']=='passed' and quality_review['independent_static_source_review'] is True
    assert len(quality_review['reviewed_owned_files'])==4
    assert evaluation_review['kind']=='cpb_v14_training_evaluation_and_training_auditor_independent_static_review_v1' and evaluation_review['status']=='passed' and evaluation_review['independent_of_author'] is True
    assert len(evaluation_review['reviewed_owned_files'])==32
    for authority in (card['independent_quality_source_review'],card['independent_evaluation_source_review']):
        matched=[p for p in scientific_review['independent_reviews'].values() if p['path']==authority['path']]
        assert len(matched)==1 and all(matched[0][k]==authority[k] for k in ('bytes','sha256'))
    quality_sources=[p for p in quality_review['reviewed_owned_files'] if p['path'].startswith('code/research/causal_fresh_v14_quality_audit/')]
    assert len(quality_sources)==len(card['auditor_sources'])==4
    for entry in quality_sources:
        local=pin(source/Path(entry['path']).name)
        assert all(local[k]==entry[k] for k in ('bytes','sha256'))
    assert card['audit_attempt']=='attempt-001' and card['report_only_recovery_required'] is False
    assert set(card['expected_main_paths'])=={'training','quality','card'}
    main_label=card['main_attempt']
    assert main_label.startswith('attempt-') and len(main_label)==11 and main_label[8:].isdigit()
    attempt=Path('/opt/cuwacunu/research/causal-fresh-v14')/main_label
    main_source=attempt/'source';training=attempt/'run-001';quality=attempt/'quality-001'
    assert card['expected_main_paths']=={'training':str(training),'quality':str(quality),
                                       'card':str(main_source/'doc/CAUSAL_FRESH_V14_CARD.json')}
    assert pin(main_source/'CAPSULE_MANIFEST.json')['sha256']==card['main_source_manifest_sha256']
    assert pin(attempt/'source.zip')['sha256']==card['main_source_zip_sha256']
    actual_main_card=main_source/'doc/CAUSAL_FRESH_V14_CARD.json'
    assert all(pin(actual_main_card)[k]==card['main_card'][k] for k in ('bytes','sha256'))
    main_card=read(actual_main_card)
    assert main_card['model']=='CPB-v14' and 'DRAFT' not in main_card['stage'] and main_card['frozen_utc'] and main_card['execution_admitted'] is True
    main_manifest=read(main_source/'CAPSULE_MANIFEST.json')
    assert len(main_manifest['files'])==card['main_source_files']==main_card['source_counts']['capsule_sources']>0
    assert main_card['source_counts']['binaries']==7 and main_card['source_counts']['translation_units']==11
    evaluation_sources=[p for p in evaluation_review['reviewed_owned_files'] if p['path'].startswith('code/research/causal_fresh_v14/evaluation/')]
    assert len(evaluation_sources)==20
    for entry in evaluation_sources:
        matches=[p for p in main_manifest['files'] if p['path']==entry['path']]
        assert len(matches)==1 and all(matches[0][k]==entry[k] for k in ('bytes','sha256'))
        assert all(pin(main_source/entry['path'])[k]==entry[k] for k in ('bytes','sha256'))
    assert all(pin(main_source/'doc/causal_fresh_v14/input-manifest.json')[k]==card['input_manifest'][k] for k in ('bytes','sha256'))
    required_stage='quality'
    assert Path(card['completed_quality_stage']['path']).is_absolute()
    assert Path(card['completed_quality_stage']['path']).resolve(strict=True)==(attempt/'quality-stage.json').resolve(strict=True)
    assert all(pin(attempt/'quality-stage.json')[key]==card['completed_quality_stage'][key] for key in ('bytes','sha256'))
    assert all(pin(quality/'quality-receipt.json')[key]==card['completed_quality_receipt'][key] for key in ('bytes','sha256')), 'Opaque durable receipt label is not a runtime filesystem path'
    completed=read(attempt/'quality-stage.json')
    assert completed['kind']=='causal_fresh_v14_exclusive_stage_execution_v1' and completed['status']=='passed' and completed['exit_code']==0
    quality_receipt=read(quality/'quality-receipt.json')
    assert quality_receipt['status']=='passed' and quality_receipt['model']=='CPB-v14.market-01'
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
    assert summary['status']=='passed' and summary['model']=='CPB-v14.market-01' and summary['new_causal_updates']==2048
    output=root/'run-001'
    assert not output.exists()
    output.mkdir()
    command=[sys.executable,str(source/'audit.py'),'--card',str(main_source/'doc/CAUSAL_FRESH_V14_CARD.json'),
                 '--input-manifest',str(main_source/'doc/causal_fresh_v14/input-manifest.json'),
                 '--raw-controls',str(main_source/'doc/causal_fresh_v14/raw-controls.json'),
                 '--training',str(training),'--run',str(quality),'--report',str(quality/'report'),
                 '--quality-receipt',str(quality/'quality-receipt.json'),
                 '--comparison',str(quality/'comparison'),'--audit-card',str(source/'audit-card.json')]
    command+=['--output',str(output/'raw-result.json')]
    write(output/'invocation.json',{'kind':'cpb_v14_independent_audit_invocation_v1','argv':command,
          'main_stage':pin(attempt/(required_stage+'-stage.json')),'audit_card':pin(source/'audit-card.json'),
          'source_stage':pin(root/'source-stage.json'),'source_manifest':pin(source/'SOURCE_MANIFEST.json'),
          'completed_quality_receipt':pin(quality/'quality-receipt.json'),'main_inputs_before':main_before,
          'wrapper':pin(__file__),'source_before':before,'model_calls':0,'head_fits':0,'training_updates':0})
    lock_path=Path('/opt/cuwacunu/research')/f'cpb-v14-independent-{args.kind}-audit.lock'
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
    execution={'kind':f'causal_fresh_v14_{args.kind}_audit_execution_v1','status':status,'exit_code':finished.returncode,
               'argv':command,'started_utc':start_utc,'elapsed_seconds':elapsed,'source_before':before,'source_after':after,
               'stdout':pin(output/'stdout.log'),'stderr':pin(output/'stderr.log'),'wrapper':pin(__file__),
               'audit_card':pin(source/'audit-card.json'),'model_calls':0,'head_fits':0,'training_updates':0}
    execution['main_inputs_before']=main_before;execution['main_inputs_after']=main_after
    execution['source_stage']=pin(root/'source-stage.json')
    execution['source_manifest']=pin(source/'SOURCE_MANIFEST.json')
    execution['independent_workflow_source_review']=pin(source/'workflow-review.json')
    execution['independent_scientific_source_review']=pin(source/'scientific-source-review.json')
    execution['independent_quality_source_review']=pin(source/'quality-source-review.json')
    execution['independent_evaluation_source_review']=pin(source/'evaluation-source-review.json')
    execution['actual_card_peer_review']=stage['actual_card_peer_review']
    if result is not None:execution['raw_result']=pin(output/'raw-result.json')
    write(output/'execution.json',execution)
    inventory=[]
    for file in sorted(root.rglob('*')):
        if file.is_file():inventory.append({'relative':file.relative_to(root).as_posix(),**pin(file)})
    write(root/'mirror-inventory.json',{'kind':'cpb_v14_audit_complete_file_inventory_v1','status':status,'files':inventory})
    print(json.dumps({'status':status,'kind':args.kind,'elapsed_seconds':elapsed,'output':str(output),
                      'checks':result.get('checks') if result else None}))
    return 0 if status=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())

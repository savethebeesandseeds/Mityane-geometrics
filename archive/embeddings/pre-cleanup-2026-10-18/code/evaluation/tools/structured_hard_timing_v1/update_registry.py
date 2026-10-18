#!/usr/bin/env python3
"""Append two audited TEMPO-3 groups; preserve all previous registry objects."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path('/embedding')
REGISTRY = ROOT/'doc/embedding_versions.json'
REPORT = ROOT/'code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_DIAGNOSTIC.md'
SUMMARY = ROOT/'doc/results/structured_hard_timing_comparison_v2.json'
RUN_ROOT = ROOT/'output/runs/rpb-structured-hard-timing'
PROTOCOL = 'structured-hard-timing-comparison-v2'
MASTERS = [75272,76373,77474,78575,79676]
TAGS = ['RPB-v7.alt-05','RPB-v10.alt-05']

def digest(body):
    return hashlib.sha256(body).hexdigest()

def admit(paths):
    identities = set()
    for path in paths:
        assert path.is_absolute() and path.resolve(strict=True) == path and all(not p.is_symlink() for p in (path,*path.parents))
        s = path.stat()
        assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and (s.st_dev,s.st_ino) not in identities
        identities.add((s.st_dev,s.st_ino))

def updated(registry, summary, report_sha, summary_sha, audit_sha):
    assert summary['protocol'] == PROTOCOL and summary['dataset_id'] == 'TEMPO-3'
    audit = summary['audit']; source = audit['source']; complete = summary['complete']
    assert audit['status'] == 'passed' and audit['protocol'] == PROTOCOL
    assert summary['metadata_bindings']['audit']['sha256'] == audit_sha
    assert [c['timing_master'] for c in summary['cohorts']] == MASTERS
    assert source['source_fingerprint'] == '9cd4af69f71b1cdc0368987185aa836040d349ec7936b2ef6b11e1838c2b6eb8'
    assert source['inventory_sha256'] == '9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa'
    assert source['human_card_sha256'] == '35f7aa9987f293bafbe735506cc86cd2af02f2ab39bec42dd1dc2d2dedfa758c'
    assert source['information_sha256'] == '6c5f295c7c3956d5ede11a8f1bc28a8402031ee1946c769f876ce9e457aee8b8'
    assert summary['designed_complexity_level'] == 4 and summary['complexity_scale_max'] == 5
    capsule = Path(source['capsule'])
    assert capsule == RUN_ROOT/'structured-hard-timing-xrZMAS'
    assert complete['encoder_updates_each'] == 512 and complete['encoder_trajectories'] == 10
    assert complete['sampled_rows'] == 40960 and complete['skipped_attempts'] == 0
    assert all(complete[k] is False for k in ('testing_accessed','stress_accessed','promotion','selection'))
    assert registry['working_encoder_tag'] == registry['working_instance_bundle'] == 'RPB-v7'
    assert registry['active_research_tag'] == 'RPB-v4'
    assert 'structured_hard_timing_diagnostic' not in registry and not set(TAGS)&{b['id'] for b in registry['instance_bundles']}
    report_path = REPORT.relative_to(ROOT).as_posix(); summary_path = SUMMARY.relative_to(ROOT).as_posix()
    capsule_path = capsule.relative_to(ROOT).as_posix()
    result = copy.deepcopy(registry)
    for tag, role, method, base_tag in zip(TAGS,('late','early'),('native_late','native_early'),('RPB-v7','RPB-v10')):
        intact = next(x for x in summary['quality']['validation_intact'] if x['method']==method)
        deleted = next(x for x in summary['quality']['validation_deleted'] if x['method']==method)
        percentages = lambda item,key: None if item[key] is None else 100*item[key]
        result['instance_bundles'].append({'id':tag,'encoder_tag':base_tag,
            'status':'measured_fresh_structured_hard_timing_no_promotion_no_reference_replacement',
            'description':'Fresh '+role+' mixer on TEMPO-3; unchanged architecture and fixed512 recipe',
            'master_seeds':MASTERS,'dataset_codename':'TEMPO-3','dataset_recipe_id':'structured-hard-timing-v1',
            'designed_complexity_level':4,'complexity_scale_max':5,'task':'lag_sign','encoder_updates_each':512,
            'retained_points_each':[0,512],'native_size':32,'registered_parameters':225805,'batch_size':8,
            'context_deletion_rate':.15,'checkpoint_path_pattern':capsule_path+'/results/seed-{master}-lag_sign/'+role+'/point-{updates}/checkpoint.pt',
            'capsule':capsule_path,'artifact_inventory_sha256':source['inventory_sha256'],'measured_record':report_path,
            'durable_summary':summary_path,'intact_linear_accuracy_percent':percentages(intact,'ridge_mean'),
            'intact_neural_accuracy_percent':percentages(intact,'tiny_secondary_mean'),
            'deleted_linear_accuracy_percent':percentages(deleted,'ridge_mean'),
            'deleted_neural_accuracy_percent':percentages(deleted,'tiny_secondary_mean'),
            'intact_coverage_percent':percentages(intact,'coverage_mean'),'deleted_coverage_percent':percentages(deleted,'coverage_mean'),
            'promotion':False,'replacement_by_new_training':'forbidden; this fresh dataset cohort does not replace original saved models'})
    diagnostic = {'protocol':PROTOCOL,'dataset_codename':'TEMPO-3','dataset_recipe_id':'structured-hard-timing-v1',
        'designed_complexity_level':4,'complexity_scale_max':5,'master_seeds':MASTERS,'instance_groups':TAGS,
        'measured_capsule':capsule_path,'artifact_inventory_sha256':source['inventory_sha256'],
        'source_fingerprint':source['source_fingerprint'],'card':'code/evaluation/cards/structured_hard_timing_comparison_v2.md',
        'card_sha256':source['human_card_sha256'],'audit_reader_sha256':source['reader_sha256'],
        'audit_validation_sha256':audit_sha,'audit_checks':audit['checks'],'audit_archive_decodes':audit['archive_decodes'],
        'actual_admission_sha256':source['admission_sha256'],'actual_admission_log_sha256':source['admission_log_sha256'],
        'information_admission_sha256':source['information_sha256'],'measured_record':report_path,'measured_record_sha256':report_sha,
        'durable_summary':summary_path,'durable_summary_sha256':summary_sha,'quality':copy.deepcopy(summary['quality']),
        'training':[{k:copy.deepcopy(v) for k,v in row.items() if k != 'per_master'} for row in summary['training']],
        'analytic':copy.deepcopy(summary['analytic']),
        'information_admission_attempts':copy.deepcopy(summary['information_admission_attempts']),
        'complete':copy.deepcopy(complete),'independent_audit_limits':copy.deepcopy(audit['limits']),
        'independent_source_binding':copy.deepcopy(source),'costs_mean_per_cohort_seconds':copy.deepcopy(summary['costs_mean_per_cohort_seconds']),
        'current_continuation':'doc/CONTINUATION_2026-10-09_AFTER_STRUCTURED_HARD_TIMING.md',
        'next_direction':'proposed saved-TRAIN fit/margin diagnosis and same-bank8tick reconstruction timing check; no fits or model updates',
        'testing_accessed':False,'stress_accessed':False,'selection':False,'promotion':False,'existing_groups_replaced':False}
    result['structured_hard_timing_diagnostic'] = diagnostic
    result['latest_completed_diagnostic'] = {'protocol':PROTOCOL,'instance_groups':TAGS,'measured_record':report_path,
        'durable_summary':summary_path,'measured_record_sha256':report_sha,'durable_summary_sha256':summary_sha,
        'audit_validation_sha256':audit_sha,'audit_checks':audit['checks'],'dataset_codename':'TEMPO-3',
        'designed_complexity_level':4,'promotion':False,'existing_groups_replaced':False}
    result['last_updated_date'] = '2026-10-09'
    assert result['versions'] == registry['versions'] and result['instance_bundles'][:-2] == registry['instance_bundles']
    assert all(result[k]==v for k,v in registry.items() if k not in ('instance_bundles','latest_completed_diagnostic','last_updated_date'))
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('summary-sha256','report-sha256','audit-sha256','proof-directory'): p.add_argument('--'+name,required=True)
    a = p.parse_args(); assert Path('/.dockerenv').is_file()
    paths = [REGISTRY,SUMMARY,REPORT]; admit(paths)
    bodies = [path.read_bytes() for path in paths]
    assert digest(bodies[1]) == a.summary_sha256 and digest(bodies[2]) == a.report_sha256
    previous, summary = json.loads(bodies[0]),json.loads(bodies[1])
    result = updated(previous,summary,a.report_sha256,a.summary_sha256,a.audit_sha256)
    proof_dir = Path(a.proof_directory)
    assert proof_dir.parent == RUN_ROOT/'report-tools' and proof_dir.name.startswith('registry-qa-')
    assert proof_dir.parent.resolve(strict=True)==proof_dir.parent and all(not p.is_symlink() for p in proof_dir.parents)
    assert not proof_dir.exists() and not proof_dir.is_symlink()
    assert [p.read_bytes() for p in paths] == bodies
    proof_dir.mkdir()
    with (proof_dir/'registry-before.json').open('xb') as out: out.write(bodies[0])
    REGISTRY.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    reread = json.loads(REGISTRY.read_text()); assert reread == result
    assert SUMMARY.read_bytes()==bodies[1] and REPORT.read_bytes()==bodies[2]
    proof = {'status':'passed','registry_before_sha256':digest(bodies[0]),'registry_after_sha256':digest(REGISTRY.read_bytes()),
        'old_designs_preserved':len(previous['versions']),'old_instance_groups_preserved':len(previous['instance_bundles']),
        'historical_top_level_objects_preserved':len([k for k in previous if k not in ('instance_bundles','latest_completed_diagnostic','last_updated_date')]),
        'added_instance_groups':TAGS,'added_diagnostic':PROTOCOL,'promotion':False,
        'report_sha256':a.report_sha256,'summary_sha256':a.summary_sha256,'audit_sha256':a.audit_sha256,
        'tensor_reads_or_numerical_reruns':0,'original_groups_replaced':False}
    with (proof_dir/'preservation-proof.json').open('x',encoding='utf-8',newline='\n') as out: out.write(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))

if __name__=='__main__': main()

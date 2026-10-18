#!/usr/bin/env python3
"""Publish checked metadata by task; never decode tensors or execute models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path('/embedding')
TAGS = {'v18':'RPB-v18.alt-01', 'v19':'RPB-v19'}
TASKS = {'slow_lag_sign':('TEMPO-4','slow-component lead/lag in a coherent two-rhythm mixture'),
         'component_balance':('AMP-2','relative slow/fast strength in a coherent two-rhythm mixture')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return None if any(v is None for v in values) else statistics.mean(values)


def fmt(v, scale=1, digits=2):
    return '—' if v is None else f'{v*scale:.{digits}f}'


def rows(cohorts, updates, view):
    result = []
    for c in cohorts:
        point = next(p for p in c['points'] if p['updates'] == updates)
        method, = point['readouts']['methods']
        reps = method['repetitions']
        measured = method['status'] == 'measured'
        result.append(dict(master=c['master'],
            ridge=mean(r[view]['ridge']['accuracy'] for r in reps) if measured else None,
            tiny_secondary=mean(r[view]['tiny_secondary']['accuracy'] for r in reps) if measured else None,
            coverage=mean(r[view]['ridge']['coverage'] for r in reps) if measured else method[view+'_population']['coverage'],
            fit_status=method['status'], reason=method['reason'],
            repetitions=[{k:r[k] for k in ('id','actual_probe_seed_decimal',view) if k in r} for r in reps]))
    return result


def aggregate(tag, records):
    return dict(label=tag, size=32, evidence='new', per_master=records,
                ridge_mean=mean(r['ridge'] for r in records),
                tiny_secondary_mean=mean(r['tiny_secondary'] for r in records),
                coverage_mean=mean(r['coverage'] for r in records),
                ridge_worst=min(r['ridge'] for r in records) if all(r['ridge'] is not None for r in records) else None,
                tiny_secondary_worst=min(r['tiny_secondary'] for r in records) if all(r['tiny_secondary'] is not None for r in records) else None)


def table(records):
    return ['| Method | Size | Linear head % | Neural head % | Coverage % |',
            '| --- | ---: | ---: | ---: | ---: |'] + [
        f"| {r['label']} | {r['size']} | {fmt(r['ridge_mean'],100)} | {fmt(r['tiny_secondary_mean'],100)} | {fmt(r['coverage_mean'],100)} |" for r in records]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--admission', type=Path, required=True)
    a = p.parse_args()
    capsule = a.admission
    freeze = json.loads((capsule/'freeze/freeze.json').read_text())
    registry_path = ROOT/'doc/embedding_versions.json'
    original_registry = json.loads(registry_path.read_text())
    if any(v['tag'] == 'RPB-v19' for v in original_registry['versions']) or any(v['id'] == 'RPB-v18.alt-01' for v in original_registry['instance_bundles']):
        raise ValueError('new additive identities required; never replace a registry entry')
    reports, proofs, inputs = {}, {}, []
    for candidate, tag in TAGS.items():
        run = capsule/candidate
        report = json.loads((run/'report.json').read_text())
        proof = json.loads((capsule/(candidate+'-saved-check.json')).read_text())
        if (proof['status'] != 'passed' or proof['report_sha256'] != sha(run/'report.json')
                or report['protocol'] != 'multiband-screen-v1' or report['candidate'] != candidate
                or report['source_fingerprint'] != freeze['source_fingerprint']
                or report['human_card_sha256'] != freeze['card_sha256'] or report['inputs_sha256'] != freeze['inputs_sha256']):
            raise ValueError('passed immutable report/reader/source/card/input association required')
        if [(c['master'],c['task']) for c in report['cohorts']] != [(m,t) for m in (920903,920904) for t in TASKS]:
            raise ValueError('all four ordered cohorts required')
        if any(c['model_tag'] != tag for c in report['cohorts']):
            raise ValueError('exact candidate tags required')
        reports[candidate], proofs[candidate] = report, proof
        inputs.append(dict(candidate=candidate,tag=tag,run=str(run),report_sha256=sha(run/'report.json'),
                           saved_check_sha256=sha(capsule/(candidate+'-saved-check.json')),saved_check=proof,
                           counts=report['counts'],cohorts=[{
                               k:({pk:pv for pk,pv in v.items() if pk != 'losses'} if k == 'progress' else v)
                               for k,v in c.items()} for c in report['cohorts']]))
    for line in (capsule/'freeze/sources.sha256').read_text().splitlines():
        digest, name = line.split('  ',1)
        if sha(ROOT/name) != digest or sha(capsule/'freeze/SOURCE'/name) != digest:
            raise ValueError('frozen source changed: '+name)
    meta = capsule/'METADATA_SOURCE'
    meta.mkdir(exist_ok=False)
    path = ROOT/'code/evaluation/protocols/multiband_screen_v1/publish.py'
    target = meta/path.relative_to(ROOT)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,target)
    manifest = capsule/'metadata-sources.sha256'
    manifest.write_text(f'{sha(path)}  {path.relative_to(ROOT).as_posix()}\n')
    shutil.copyfile(registry_path,capsule/'registry-before.json')
    inventory = capsule/'artifact-inventory.sha256'
    inventory.write_text(''.join(f'{sha(f)}  {f.relative_to(capsule).as_posix()}\n'
                                 for f in sorted(capsule.rglob('*')) if f.is_file() and f != inventory))
    summary = dict(protocol='multiband-screen-v1',data_recipe='two-component-v1',
                   masters=[920903,920904],datasets={},inputs=inputs,freeze=freeze,
                   promotion=False,old_instances_replaced=False,testing_accessed=False,
                   tasks_averaged=False, captured_evidence=dict(admission=str(capsule),
                       metadata_sources_manifest_sha256=sha(manifest),artifact_inventory_sha256=sha(inventory)))
    lines = ['# Multiband representation screen', '',
        'Two new cohorts per task, TRAIN256/128 source pairs and VALIDATION128/64 pairs.',
        'Each candidate ran once per task/cohort on CUDA for512 updates; initial0 controls',
        'are retained. Native32 and fixed Ridge1/tanh16 heads; no post-encoder PCA.', '',
        '**RPB-v18.alt-01**: new instances of the unchanged unit temporal-relation design.',
        '**RPB-v19**: same learned shape20 and fixed12 allocation, with mask-aware generic',
        'low/high harmonic complex relations and joint per-pair normalization.',
        'Both have226,877 registered/226,445 trainable/432 frozen parameters.', '',
        'Initial success credits the architectural prior; learned shape requires gains',
        'over that instance’s own initial control. No old model or head was rerun.', '']
    for task,(dataset, description) in TASKS.items():
        selected = {candidate:[c for c in report['cohorts'] if c['task'] == task]
                    for candidate,report in reports.items()}
        item = dict(task=task, dataset_id=dataset, recipe_id='two-component-v1', designed_complexity_level=5,
                    complexity_scale_max=5,quality={},untrained={},training=[])
        summary['datasets'][dataset] = item
        for view in ('validation_intact','validation_deleted'):
            for updates,key in ((512,'quality'),(0,'untrained')):
                records = [aggregate(TAGS[candidate],rows(cohorts,updates,view)) for candidate,cohorts in selected.items()]
                item[key][view] = records
                lines += [f'Dataset: **{dataset}** · {description} · **complexity5/5** · {view} · updates{updates}.','']+table(records)+['']
        for candidate,cohorts in selected.items():
            records = []
            for c in cohorts:
                progress_path = capsule/candidate/f"seed-{c['master']}-{task}/progress.json"
                records.append(dict(master=c['master'],training_reconstruction=c['training_reconstruction'],
                    validation_reconstruction=c['validation_reconstruction'],costs=c['costs'],
                    progress={k:v for k,v in c['progress'].items() if k != 'losses'},
                    full_progress_artifact=str(progress_path),full_progress_sha256=sha(progress_path)))
            item['training'].append(dict(label=TAGS[candidate],updates=512,
                train_error=mean(c['training_reconstruction']['standardized_mae'] for c in cohorts),
                validation_error=mean(c['validation_reconstruction']['standardized_mae'] for c in cohorts),
                GPU_training_seconds=mean(c['progress']['training_seconds'] for c in cohorts),per_master=records))
        lines += [f'Dataset: **{dataset}** · {description} · **complexity5/5** · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.','',
            '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |',
            '| --- | ---: | ---: | ---: | ---: |']
        for r in item['training']:
            lines += [f"| {r['label']} |512| {fmt(r['train_error'],digits=6)} | {fmt(r['validation_error'],digits=6)} | {fmt(r['GPU_training_seconds'])} |"]
        lines += ['']
    conditions = []
    for dataset,item in summary['datasets'].items():
        for view,records in item['quality'].items():
            v19 = next(r for r in records if r['label'] == 'RPB-v19')
            for r in v19['per_master']:
                conditions.append(dict(dataset_id=dataset,view=view,master=r['master'],
                    scope='both heads >=75%; coverage100%',ridge=r['ridge'],tiny_secondary=r['tiny_secondary'],
                    coverage=r['coverage'],passed=r['ridge'] is not None and r['tiny_secondary'] is not None
                    and min(r['ridge'],r['tiny_secondary']) >= .75 and r['coverage'] == 1))
    for dataset,required in (('TEMPO-4',.05),('AMP-2',0)):
        control,candidate = summary['datasets'][dataset]['quality']['validation_deleted']
        gain = None if control['ridge_mean'] is None or candidate['ridge_mean'] is None else candidate['ridge_mean']-control['ridge_mean']
        conditions.append(dict(dataset_id=dataset,scope='mean deleted Linear gain',control=control['ridge_mean'],
            candidate=candidate['ridge_mean'],gain=gain,required_gain=required,passed=gain is not None and gain+1e-15 >= required))
    passed = all(c['passed'] for c in conditions)
    summary['continuation_gate'] = dict(passed=passed,conditions=conditions,
        action='separate prospective confirmation required; no further seeds allocated' if passed else 'stop this recipe expansion; no frequency/head/mask rescue')
    summary['counts'] = {k:sum(report['counts'][k] for report in reports.values()) for k in
        ('encoder_trajectories','sampled_rows','retained_points','head_pipelines','individual_heads','native_exports',
         'query_writer_calls','necessary_query_forwards','generator_calls','information_fit_calls','PCA_fits',
         'old_encoder_or_head_refits','CPU_encoder_calls','skipped_attempts')}
    summary['counts']['encoder_updates'] = 8*512
    summary['counts']['quality_data_generator_calls'] = 4
    summary['counts']['saved_check_CPU_archives_decoded'] = sum(p['CPU_archive_decodes'] for p in proofs.values())
    summary['counts']['saved_check_checks'] = sum(p['checks'] for p in proofs.values())
    lines += ['The frozen joint continuation rule **'+('passed' if passed else 'failed')+'**.',
        summary['continuation_gate']['action']+'.', '',
        'Per-cohort results, initial controls, populations, source-group intervals, costs',
        'and every gate condition are retained in the [durable JSON](results/multiband_screen_v1.json).',
        'The [prospective card](../code/evaluation/cards/multiband_screen_v1.md) fixes the',
        'decision before quality. Both datasets remain separate; complexity is a designed',
        'ordinal level, not an accuracy-derived score or a claim of equal task difficulty.', '',
        'All12 fixed values per row are byte-identical at0/512 in both designs.',
        'CPU saved-only verification replays predictions from retained fitted heads and',
        'query arithmetic, with zero model execution, head fits or frequency searches.',
        'Full artifacts remain in `'+str(capsule.relative_to(ROOT))+'`.',
        'Source, card, SDK, input and artifact hashes are retained in the JSON.', '',
        'TEMPO-3 RPB-v18, formal RPB-v4, original RPB-v7 and all older results remain',
        'preserved. No TEST, stress evaluation, default change or reference promotion.', '']
    output = ROOT/'doc/results/multiband_screen_v1.json'
    report_path = ROOT/'doc/MULTIBAND_SCREEN_V1.md'
    with output.open('x') as f:
        f.write(json.dumps(summary,indent=2)+'\n')
    with report_path.open('x') as f:
        f.write('\n'.join(lines))
    evidence = dict(protocol=summary['protocol'],datasets=['TEMPO-4','AMP-2'],master_seeds=[920903,920904],
                    encoder_updates_each=512,native_global_width=32,measured_record='doc/MULTIBAND_SCREEN_V1.md',
                    durable_summary='doc/results/multiband_screen_v1.json',durable_summary_sha256=sha(output),
                    artifact_inventory_sha256=sha(inventory),capsule=str(capsule),promotion=False,
                    source_fingerprint=freeze['source_fingerprint'],card_sha256=freeze['card_sha256'])
    registry = json.loads(json.dumps(original_registry))
    registry['versions'].append(dict(evidence,tag='RPB-v19',encoder='raw_patch_bottleneck_mae',
        description='Learned shape with fixed generic low/high complex spectral relations',
        status='measured_two_task_screen_not_promoted',
        architecture_id='early-native20-shape-plus12-fixed-two-band-complex-relations-v1',
        parameter_count_at_C3_H32_F3=226877,trainable_parameter_values=226445,frozen_parameter_values=432,
        continuation_gate_passed=passed,task_results=summary['datasets']))
    registry['instance_bundles'].append(dict(evidence,id='RPB-v18.alt-01',encoder_tag='RPB-v18',
        status='separate_two_component_matched_control_no_reference_replacement',
        description='New unchanged v18 instances for separate two-component timing and balance tasks',
        checkpoint_path_pattern=str(capsule/'v18/seed-{master}-{task}/point-512/checkpoint.pt')))
    registry['multiband_screen'] = dict(evidence,continuation_gate=summary['continuation_gate'],counts=summary['counts'])
    if registry['versions'][:-1] != original_registry['versions'] or registry['instance_bundles'][:-1] != original_registry['instance_bundles']:
        raise ValueError('historical registry entries changed')
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(dict(summary_sha256=sha(output),report_sha256=sha(report_path),continuation_gate_passed=passed)))


if __name__ == '__main__':
    main()

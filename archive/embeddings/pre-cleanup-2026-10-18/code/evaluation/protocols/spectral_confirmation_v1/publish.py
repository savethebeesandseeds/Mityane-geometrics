#!/usr/bin/env python3
"""Publish checked metadata by task; never decode tensors or execute models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path('/embedding')
TAGS = {'v19':'RPB-v19.alt-01'}
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
    run_root = ROOT/'output/runs/rpb-spectral-confirmation'
    if (capsule.resolve(strict=True) != capsule or capsule.parent != run_root
            or not capsule.name.startswith('admission-') or capsule.is_symlink()):
        raise ValueError('direct confirmation admission required')
    freeze = json.loads((capsule/'freeze/freeze.json').read_text())
    registry_path = ROOT/'doc/embedding_versions.json'
    original = json.loads(registry_path.read_text())
    if (any(v['id'] == 'RPB-v19.alt-01' for v in original['instance_bundles'])
            or not any(v['tag'] == 'RPB-v19' for v in original['versions'])
            or 'spectral_confirmation' in original):
        raise ValueError('existing design and new additive bundle required')
    run = capsule/'v19'
    report = json.loads((run/'report.json').read_text())
    proof_path = capsule/'v19-saved-check.json'
    proof = json.loads(proof_path.read_text())
    if (proof['status'] != 'passed' or proof['report_sha256'] != sha(run/'report.json')
            or proof['reader_sha256'] != sha(Path(__file__).parent/'check_saved_stdlib.py')
            or report['protocol'] != 'spectral-confirmation-v1' or report['candidate'] != 'v19'
            or report['source_fingerprint'] != freeze['source_fingerprint']
            or report['human_card_sha256'] != freeze['card_sha256']
            or report['inputs_sha256'] != freeze['inputs_sha256']):
        raise ValueError('passed immutable report/reader/source/card/input association required')
    masters = [930905,930906,930907]
    tag = TAGS['v19']
    if ([(c['master'],c['task']) for c in report['cohorts']] != [(m,t) for m in masters for t in TASKS]
            or any(c['model_tag'] != tag for c in report['cohorts'])):
        raise ValueError('all six ordered unchanged-v19 cohorts required')
    for line in (capsule/'freeze/sources.sha256').read_text().splitlines():
        digest, name = line.split('  ',1)
        if sha(ROOT/name) != digest or sha(capsule/'freeze/SOURCE'/name) != digest:
            raise ValueError('frozen source changed: '+name)
    meta = capsule/'METADATA_SOURCE'
    meta.mkdir(exist_ok=False)
    path = Path(__file__)
    target = meta/path.relative_to(ROOT)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,target)
    manifest = capsule/'metadata-sources.sha256'
    manifest.write_text(f'{sha(path)}  {path.relative_to(ROOT).as_posix()}\n')
    shutil.copyfile(registry_path,capsule/'registry-before.json')
    inventory = capsule/'artifact-inventory.sha256'
    inventory.write_text(''.join(f'{sha(f)}  {f.relative_to(capsule).as_posix()}\n'
                                 for f in sorted(capsule.rglob('*')) if f.is_file() and f != inventory))
    input_record = dict(candidate='v19',tag=tag,run=str(run),report_sha256=sha(run/'report.json'),
        saved_check_sha256=sha(proof_path),saved_check=proof,counts=report['counts'],
        cohorts=[{k:({pk:pv for pk,pv in v.items() if pk != 'losses'} if k == 'progress' else v)
                  for k,v in c.items()} for c in report['cohorts']])
    summary = dict(protocol='spectral-confirmation-v1',data_recipe='two-component-v1',
        masters=masters,datasets={},inputs=[input_record],freeze=freeze,
        promotion=False,old_instances_replaced=False,testing_accessed=False,tasks_averaged=False,
        captured_evidence=dict(admission=str(capsule),
            metadata_sources_manifest_sha256=sha(manifest),artifact_inventory_sha256=sha(inventory)))
    lines = ['# Fresh spectral representation confirmation', '',
        'Three fresh cohorts per task; TRAIN256/128 source pairs and VALIDATION128/64 pairs.',
        'Each instance ran once on CUDA for512 updates. Initial0 controls, native32,',
        'fixed Ridge1/tanh16 heads and original masked waveform queries are retained.', '',
        '**RPB-v19.alt-01**: new instances of unchanged RPB-v19, with20 learned shape',
        'coordinates and12 fixed generic low/high complex spectral relations.',
        '226,877 registered/226,445 trainable/432 frozen parameters.', '',
        'This tests repeatability on fresh sources. Strong initial results credit the',
        'fixed prior; trained gains over those controls are a separate learning question.',
        'No old encoder/head, raw baseline or information-frequency search was rerun.', '']
    conditions = []
    for task,(dataset,description) in TASKS.items():
        cohorts = [c for c in report['cohorts'] if c['task'] == task]
        item = dict(task=task,dataset_id=dataset,recipe_id='two-component-v1',
            designed_complexity_level=5,complexity_scale_max=5,quality={},untrained={},training=[])
        summary['datasets'][dataset] = item
        for view in ('validation_intact','validation_deleted'):
            for updates,key in ((512,'quality'),(0,'untrained')):
                record = aggregate(tag,rows(cohorts,updates,view))
                item[key][view] = [record]
                lines += [f'Dataset: **{dataset}** · {description} · **complexity5/5** · {view} · updates{updates}.','']+table([record])+['']
                if updates == 512:
                    for r in record['per_master']:
                        conditions.append(dict(dataset_id=dataset,view=view,master=r['master'],
                            scope='each head mean over three repetitions >=75%; coverage100%',
                            ridge=r['ridge'],tiny_secondary=r['tiny_secondary'],coverage=r['coverage'],
                            passed=r['ridge'] is not None and r['tiny_secondary'] is not None
                            and min(r['ridge'],r['tiny_secondary']) >= .75 and r['coverage'] == 1))
        per_master = []
        for c in cohorts:
            progress_path = run/f"seed-{c['master']}-{task}/progress.json"
            per_master.append(dict(master=c['master'],training_reconstruction=c['training_reconstruction'],
                validation_reconstruction=c['validation_reconstruction'],costs=c['costs'],
                progress={k:v for k,v in c['progress'].items() if k != 'losses'},
                full_progress_artifact=str(progress_path),full_progress_sha256=sha(progress_path)))
        training = dict(label=tag,updates=512,
            train_error=mean(c['training_reconstruction']['standardized_mae'] for c in cohorts),
            validation_error=mean(c['validation_reconstruction']['standardized_mae'] for c in cohorts),
            GPU_training_seconds=mean(c['progress']['training_seconds'] for c in cohorts),per_master=per_master)
        item['training'] = [training]
        lines += [f'Dataset: **{dataset}** · {description} · **complexity5/5** · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.','',
            '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |',
            '| --- | ---: | ---: | ---: | ---: |',
            f"| {tag} |512| {fmt(training['train_error'],digits=6)} | {fmt(training['validation_error'],digits=6)} | {fmt(training['GPU_training_seconds'])} |",'']
    if len(conditions) != 12 or conditions != proof['continuation_gate']['conditions']:
        raise ValueError('independent saved checker and publisher must agree on all12 conditions')
    passed = all(c['passed'] for c in conditions)
    summary['continuation_gate'] = dict(passed=passed,conditions=conditions,
        action='confirmation complete; a learned-path change requires its own prospective card' if passed
        else 'stop this confirmation recipe; retain every result, no frequency/head/mask rescue')
    if proof['continuation_gate']['passed'] != passed:
        raise ValueError('saved checker gate disagreement')
    summary['counts'] = dict(report['counts'],encoder_updates=6*512,quality_data_generator_calls=6,
        saved_check_CPU_archives_decoded=proof['CPU_archive_decodes'],saved_check_checks=proof['checks'])
    lines += ['The frozen confirmation gate **'+('passed' if passed else 'failed')+'** across all12 conditions.',
        summary['continuation_gate']['action']+'.','',
        'Per-cohort results, initial controls, support populations, source-group intervals,',
        'costs and gates are in the [durable JSON](results/spectral_confirmation_v1.json).',
        'The [prospective card](../code/evaluation/cards/spectral_confirmation_v1.md)',
        'fixed all three fresh masters and the gate before generation. No additional',
        'quality seeds are allocated. All12 fixed values are byte-identical at0/512.',
        'Saved-only verification performs zero encoder forwards, updates or head fits.',
        'Full artifacts: `'+str(capsule.relative_to(ROOT))+'`.','',
        'The original v19 screen, TEMPO-3 v18, formal v4, original v7, older evidence',
        'and defaults remain preserved. No TEST, stress or promotion.', '']
    output = ROOT/'doc/results/spectral_confirmation_v1.json'
    report_path = ROOT/'doc/SPECTRAL_CONFIRMATION_V1.md'
    with output.open('x') as f:
        f.write(json.dumps(summary,indent=2)+'\n')
    with report_path.open('x') as f:
        f.write('\n'.join(lines))
    evidence = dict(protocol=summary['protocol'],datasets=['TEMPO-4','AMP-2'],master_seeds=masters,
        encoder_updates_each=512,native_global_width=32,measured_record='doc/SPECTRAL_CONFIRMATION_V1.md',
        durable_summary='doc/results/spectral_confirmation_v1.json',durable_summary_sha256=sha(output),
        artifact_inventory_sha256=sha(inventory),capsule=str(capsule),promotion=False,
        source_fingerprint=freeze['source_fingerprint'],card_sha256=freeze['card_sha256'])
    registry = json.loads(json.dumps(original))
    registry['instance_bundles'].append(dict(evidence,id=tag,encoder_tag='RPB-v19',
        status='fresh_three_cohort_per_task_confirmation_not_promoted',
        description='Unchanged spectral v19 on three fresh source cohorts for each two-rhythm task',
        checkpoint_path_pattern=str(run/'seed-{master}-{task}/point-512/checkpoint.pt'),
        continuation_gate_passed=passed,task_results=summary['datasets']))
    registry['spectral_confirmation'] = dict(evidence,continuation_gate=summary['continuation_gate'],counts=summary['counts'])
    if registry['versions'] != original['versions'] or registry['instance_bundles'][:-1] != original['instance_bundles']:
        raise ValueError('historical registry entries changed')
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(dict(summary_sha256=sha(output),report_sha256=sha(report_path),continuation_gate_passed=passed)))


if __name__ == '__main__':
    main()


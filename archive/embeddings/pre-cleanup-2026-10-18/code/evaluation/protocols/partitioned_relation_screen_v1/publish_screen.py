#!/usr/bin/env python3
"""Summarize checked architecture screens from metadata; no tensor/model work."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path('/embedding')
SCREEN = {75272, 76373}
LEGEND = '**TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mean(values):
    values = list(values)
    return None if any(x is None for x in values) else statistics.mean(values)


def display(value, scale=1, digits=2):
    return '—' if value is None else f'{scale*value:.{digits}f}'


def rows_for(cohorts, updates, view):
    rows = []
    for cohort in cohorts:
        point = next(p for p in cohort['points'] if p['updates'] == updates)
        method, = point['readouts']['methods']
        reps = method['repetitions']
        supported = method['status'] == 'measured'
        rows.append({'master': cohort['master'],
                     'ridge': mean([r[view]['ridge']['accuracy'] for r in reps]) if supported else None,
                     'tiny_secondary': mean([r[view]['tiny_secondary']['accuracy'] for r in reps]) if supported else None,
                     'coverage': mean([r[view]['ridge']['coverage'] for r in reps]) if supported else method[view+'_population']['coverage'],
                     'fit_status': method['status'], 'reason': method['reason'],
                     'repetitions': [{key: r[key] for key in ('id', 'actual_probe_seed_decimal', view) if key in r} for r in reps]})
    return rows


def aggregate(label, rows, size=32, reused=False):
    return {'label': label, 'size': size, 'evidence': 'reused' if reused else 'new',
            'per_master': rows, 'ridge_mean': mean([r['ridge'] for r in rows]),
            'tiny_secondary_mean': mean([r['tiny_secondary'] for r in rows]),
            'coverage_mean': mean([r['coverage'] for r in rows]),
            'ridge_worst': min(r['ridge'] for r in rows) if all(r['ridge'] is not None for r in rows) else None,
            'tiny_secondary_worst': min(r['tiny_secondary'] for r in rows) if all(r['tiny_secondary'] is not None for r in rows) else None}


def table(rows):
    lines = ['| Method | Size | Linear head % | Neural head % | Coverage % |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['label']} | {r['size']} | {display(r['ridge_mean'],100)} | {display(r['tiny_secondary_mean'],100)} | {display(r['coverage_mean'],100,0)} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, type=Path)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    candidates, inputs, run_by_cohort = {}, [], {}
    for run in args.run:
        report = json.loads((run / 'report.json').read_text())
        checked = json.loads((run / 'saved-check.json').read_text())
        if checked['status'] != 'passed' or checked['report_sha256'] != sha(run / 'report.json'):
            raise ValueError('passed saved-evidence check for exact producer report required')
        for c in report['cohorts']:
            tag = c['model_tag']
            if any(x['master'] == c['master'] for x in candidates.get(tag, [])):
                raise ValueError('duplicate candidate/cohort')
            candidates.setdefault(tag, []).append(c)
            run_by_cohort[(tag, c['master'])] = run
        inputs.append({'run': str(run), 'report_sha256': sha(run / 'report.json'),
                       'saved_check_sha256': sha(run / 'saved-check.json'),
                       'source_fingerprint': report['source_fingerprint'],
                       'saved_check': checked})
    parent_path = ROOT / 'doc/results/structured_hard_timing_comparison_v2.json'
    parent = json.loads(parent_path.read_text())
    if sha(parent_path) != '1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4':
        raise ValueError('protected baseline metadata changed')
    summary = {'protocol': 'partitioned-relation-screen-v1', 'dataset_id': 'TEMPO-3', 'task': 'lag_sign',
               'recipe_id': 'structured-hard-timing-v1', 'designed_complexity_level': 4,
               'complexity_scale_max': 5, 'screen_masters': sorted(SCREEN), 'inputs': inputs,
               'promotion': False, 'old_instances_replaced': False, 'testing_accessed': False,
               'stages': {}, 'advancement': {}}
    lines = ['# Partitioned temporal relation screen', '',
             'New CUDA architecture/objective runs; known TRAIN/VALIDATION development data.',
             'TRAIN256 examples/128 pairs and VALIDATION128/64 pairs per run. B8,512 updates,',
             'native32; fixed Ridge1 and tanh16/Adam .01/100, three head repetitions.',
             'No PCA after either encoder. Existing baselines reuse saved metadata.', '',
             '**RPB-v16** dedicates20 native coordinates to learned waveform shape and12',
             'to grouped antisymmetric timing relations. Total226,877 parameters.', '',
             'Untrained scores measure the built-in prior; they are not evidence of learning.', '']
    for tag, cohorts in candidates.items():
        screen = [c for c in cohorts if c['master'] in SCREEN]
        if {c['master'] for c in screen} != SCREEN:
            raise ValueError('both declared screen cohorts required')
        conditions = []
        for c in screen:
            for view in ('validation_intact', 'validation_deleted'):
                r, = rows_for([c], 512, view)
                conditions.append({'master': c['master'], 'view': view,
                                   'ridge': r['ridge'], 'tiny_secondary': r['tiny_secondary'],
                                   'coverage': r['coverage'],
                                   'passed': r['ridge'] is not None and r['tiny_secondary'] is not None
                                   and min(r['ridge'], r['tiny_secondary']) >= .75 and r['coverage'] == 1})
        summary['advancement'][tag] = {'passed': all(x['passed'] for x in conditions), 'conditions': conditions}
    for stage in ('screen', 'all_known'):
        selected = {tag: [c for c in cohorts if stage != 'screen' or c['master'] in SCREEN]
                    for tag, cohorts in candidates.items()
                    if stage == 'screen' or len(cohorts) == 5}
        if not selected:
            continue
        masters = sorted({c['master'] for cs in selected.values() for c in cs})
        stage_data = {'masters': masters, 'quality': {}, 'untrained': {}, 'training': []}
        lines.extend([f'## {"Two-cohort screen" if stage == "screen" else "Five known cohorts"}', '',
                      f'Masters: {", ".join(map(str, masters))}.', ''])
        for view in ('validation_intact', 'validation_deleted'):
            quality = []
            for old in parent['quality'][view]:
                if old['method'] in ('raw', 'pca_only', 'native_early'):
                    old_rows = [x for x in old['per_master'] if x['master'] in masters]
                    quality.append(aggregate(old['label'], old_rows, old['size'], True))
            quality.extend(aggregate(tag, rows_for(cs, 512, view)) for tag, cs in selected.items())
            initial = [aggregate(tag, rows_for(cs, 0, view)) for tag, cs in selected.items()]
            stage_data['quality'][view] = quality
            stage_data['untrained'][view] = initial
            view_label = 'intact VALIDATION' if view == 'validation_intact' else 'VALIDATION with extra30% deletion'
            lines.extend([f'Dataset: {LEGEND} · {view_label} · trained512.', '', table(quality), '',
                          f'Dataset: {LEGEND} · {view_label} · untrained0.', '', table(initial), ''])
        for tag, cs in selected.items():
            stage_data['training'].append({'label': tag, 'updates': 512,
                'train_error': mean([c['training_reconstruction']['standardized_mae'] for c in cs]),
                'validation_error': mean([c['validation_reconstruction']['standardized_mae'] for c in cs]),
                'GPU_training_seconds': mean([c['progress']['training_seconds'] for c in cs]),
                'per_master': [dict({key: c[key] for key in ('master', 'training_reconstruction', 'validation_reconstruction', 'costs')},
                    progress={key: value for key, value in c['progress'].items() if key != 'losses'},
                    full_progress_artifact=str(run_by_cohort[(tag, c['master'])] / f"seed-{c['master']}/progress.json"),
                    full_progress_sha256=sha(run_by_cohort[(tag, c['master'])] / f"seed-{c['master']}/progress.json")) for c in cs]})
        lines.extend([f'Dataset: {LEGEND} · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.', '',
                      '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |',
                      '| --- | ---: | ---: | ---: | ---: |'])
        for r in stage_data['training']:
            lines.append(f"| {r['label']} |512| {display(r['train_error'],digits=6)} | {display(r['validation_error'],digits=6)} | {display(r['GPU_training_seconds'])} |")
        lines.append('')
        summary['stages'][stage] = stage_data
    lines.extend(['The screen gate is75% in both fixed heads and both views on each of the',
                  'two starting cohorts, with100% coverage. Retain all failures and initial',
                  'scores. Additional known cohorts are development checks after screening,',
                  'not fresh-data confirmation. No automatic promotion or reference replacement.', ''])
    for tag, gate in summary['advancement'].items():
        lines.append(f"- {tag}: {'passes' if gate['passed'] else 'fails'} the predeclared continuation gate.")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open('x') as stream:
        stream.write(json.dumps(summary, indent=2) + '\n')
    with args.report.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'summary_sha256': sha(args.summary), 'report_sha256': sha(args.report),
                      'advancement': {k: v['passed'] for k, v in summary['advancement'].items()}}))


if __name__ == '__main__':
    main()

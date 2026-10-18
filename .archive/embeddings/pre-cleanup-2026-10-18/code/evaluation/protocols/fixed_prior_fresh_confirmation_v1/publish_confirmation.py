#!/usr/bin/env python3
"""Summarize checked architecture screens from metadata; no tensor/model work."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path('/embedding')
MASTERS = [80787, 81888, 82989, 84090, 85191]
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
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--admission', required=True, type=Path)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text())
    checked = json.loads((args.run / 'saved-check.json').read_text())
    freeze = json.loads((args.admission / 'freeze/freeze.json').read_text())
    if report['protocol'] != 'fixed-prior-fresh-confirmation-v1' or report['masters'] != MASTERS:
        raise ValueError('complete fresh confirmation required')
    if checked['status'] != 'passed' or checked['report_sha256'] != sha(args.run / 'report.json'):
        raise ValueError('passed saved checks for the exact producer report required')
    if freeze['source_fingerprint'] != report['source_fingerprint'] or freeze['inputs_sha256'] != report['inputs_sha256']:
        raise ValueError('captured admission/data freeze must match the quality run')
    cohorts = report['cohorts']
    if [c['master'] for c in cohorts] != MASTERS or any(c['model_tag'] != 'RPB-v17.alt-01' for c in cohorts):
        raise ValueError('all declared instances must be retained')
    summary = {'protocol': report['protocol'], 'instance_bundle': 'RPB-v17.alt-01', 'model_design': 'RPB-v17',
               'dataset_id': 'TEMPO-3', 'recipe_id': 'structured-hard-timing-v1',
               'designed_complexity_level': 4, 'complexity_scale_max': 5, 'masters': MASTERS,
               'input': {'run': str(args.run), 'report_sha256': sha(args.run / 'report.json'),
                         'saved_check_sha256': sha(args.run / 'saved-check.json'),
                         'saved_check': checked, 'source_fingerprint': report['source_fingerprint'],
                         'card_sha256': report['human_card_sha256'], 'inputs_sha256': report['inputs_sha256'],
                         'sdk_proof_sha256': report['sdk_proof_sha256'],
                         'admission': str(args.admission),
                         'engineering_log_sha256': sha(args.admission / 'engineering.log'),
                         'data_report_sha256': sha(args.admission / 'fresh-data/data-report.json'),
                         'freeze_json_sha256': sha(args.admission / 'freeze/freeze.json'),
                         'compiled_source_files': freeze['source_files'],
                         'metadata_sources_manifest_sha256': sha(args.admission / 'metadata-sources.sha256'),
                         'artifact_inventory_sha256': sha(args.admission / 'artifact-inventory.sha256')},
               'quality': {}, 'untrained': {}, 'confirmation': {},
               'execution_counts': report['counts'], 'promotion': False, 'old_instances_replaced': False,
               'testing_accessed': False, 'stress_accessed': False, 'selection': False}
    lines = ['# RPB-v17 fresh-source confirmation', '',
             'New **RPB-v17.alt-01** instances reuse the exact RPB-v17 design:20 learned',
             'shape coordinates and12 fixed generic temporal-relation coordinates, native32.',
             'TRAIN256/VALIDATION128 per cohort, CUDA B8/512 updates; original .15 context',
             'deletion and waveform Huber1. Ridge1; tanh16/Adam .01/100, repetitions2701/2802/2903.',
             'Five new source-disjoint cohorts run once, with all results retained.', '',
             f'Masters: {", ".join(map(str, MASTERS))}.', '']
    continuity, capability = [], []
    for view in ('validation_intact', 'validation_deleted'):
        trained = aggregate('RPB-v17.alt-01', rows_for(cohorts, 512, view))
        untrained = aggregate('RPB-v17.alt-01', rows_for(cohorts, 0, view))
        summary['quality'][view] = trained
        summary['untrained'][view] = untrained
        title = 'intact VALIDATION' if view.endswith('intact') else 'VALIDATION with extra30% deletion'
        lines.extend([f'Dataset: {LEGEND} · {title} · trained512.', '', table([trained]), '',
                      f'Dataset: {LEGEND} · {title} · untrained0; heads still fit TRAIN.', '', table([untrained]), '',
                      f"Worst cohort Linear/Neural: {display(trained['ridge_worst'],100)}%/"
                      f"{display(trained['tiny_secondary_worst'],100)}%.", ''])
        for row in trained['per_master']:
            supported = row['ridge'] is not None and row['tiny_secondary'] is not None
            base = {'master': row['master'], 'view': view, 'ridge': row['ridge'],
                    'tiny_secondary': row['tiny_secondary'], 'coverage': row['coverage']}
            continuity.append(dict(base, passed=supported and min(row['ridge'], row['tiny_secondary']) >= .75
                                   and row['coverage'] == 1))
            linear_threshold = .95 if view.endswith('intact') else .75
            capability.append(dict(base, linear_threshold=linear_threshold, neural_threshold=.95,
                                   passed=supported and row['ridge'] >= linear_threshold
                                   and row['tiny_secondary'] >= .95 and row['coverage'] == 1))
    summary['confirmation'] = {
        'continuity': {'passed': all(c['passed'] for c in continuity), 'conditions': continuity},
        'strong_capability': {'passed': all(c['passed'] for c in capability), 'conditions': capability},
        'all_five_retained': True, 'intermediate_score_selection': False}
    training = {'label': 'RPB-v17.alt-01', 'updates': 512,
                'train_error': mean(c['training_reconstruction']['standardized_mae'] for c in cohorts),
                'validation_error': mean(c['validation_reconstruction']['standardized_mae'] for c in cohorts),
                'GPU_training_seconds': mean(c['progress']['training_seconds'] for c in cohorts),
                'per_master': [dict({k: c[k] for k in ('master', 'training_reconstruction', 'validation_reconstruction', 'costs')},
                    progress={k: v for k, v in c['progress'].items() if k != 'losses'},
                    full_progress_artifact=str(args.run / f"seed-{c['master']}/progress.json"),
                    full_progress_sha256=sha(args.run / f"seed-{c['master']}/progress.json")) for c in cohorts]}
    summary['training'] = training
    lines.extend([f'Dataset: {LEGEND} · fixed hidden-target TRAIN/intact VALIDATION queries; standardized MAE.', '',
                  '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |',
                  '| --- | ---: | ---: | ---: | ---: |',
                  f"| RPB-v17.alt-01 |512| {display(training['train_error'],digits=6)} |"
                  f" {display(training['validation_error'],digits=6)} | {display(training['GPU_training_seconds'])} |", '',
                  'Time is synchronized training-loop wall time, including CPU batch preparation/transfers;',
                  'it excludes native extraction, head fitting, queries and checkpoint I/O.', '',
                  'The primary continuity rule is75% for both heads in both views on EACH cohort;',
                  'the secondary strong-capability target is95% intact for both heads,95% deleted Neural',
                  'and75% deleted Linear, with100% coverage. Both were fixed before quality measurement.',
                  f"Continuity: **{'passes' if summary['confirmation']['continuity']['passed'] else 'fails'}**.",
                  f"Strong capability: **{'passes' if summary['confirmation']['strong_capability']['passed'] else 'fails'}**.", '',
                  'Credit the fixed12-coordinate prior when the untrained encoder already solves timing.',
                  'The learned20-coordinate block and decoder still train, but timing accuracy alone does',
                  'not establish learned improvement or broad representation quality. This is same-law',
                  'fresh-source replication, not a new complexity level or automatic default promotion.', '',
                  'Historical development scores are preserved in the milestone note. No older encoder',
                  'or matched baseline was refitted here, and no PCA follows the encoder.', '',
                  'Next: a separately preregistered coherent multi-component timing/shape dataset,',
                  'with its new codename, numeric complexity and fixed tasks before measurement.', ''])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open('x') as stream:
        stream.write(json.dumps(summary, indent=2) + '\n')
    with args.report.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'summary_sha256': sha(args.summary), 'report_sha256': sha(args.report),
                      'confirmation': {k: v['passed'] for k, v in summary['confirmation'].items() if isinstance(v, dict)}}))


if __name__ == '__main__':
    main()

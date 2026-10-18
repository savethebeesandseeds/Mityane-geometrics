#!/usr/bin/env python3
"""Publish exactly three explicitly pinned JSON metadata inputs after a passed audit.

No archive, model, optimizer, head, PCA, bootstrap or audit is executed.
Self-test constructs artificial metadata only. Completed input access requires
the caller's explicit report/audit/parent-summary SHA pins.
"""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
PROTOCOL = 'visible-difference-v1'
CARD_SHA = '75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc'
PARENT_SHA = '1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4'
MASTERS = [75272, 76373, 77474, 78575, 79676]
REPS = [2701, 2802, 2903]
OLD_METHODS = ['raw', 'mask_metadata', 'pca_only', 'untrained_late', 'untrained_early', 'native_late', 'native_early']
METHOD = 'native_visible_difference'
WIDTHS = [576, 288, 32, 32, 32, 32, 32, 32]
LABELS = ['Raw data — no encoder', 'Mask metadata — no encoder', 'PCA only — no encoder',
          'Untrained late mixer', 'Untrained early mixer', 'RPB-v7.alt-05', 'RPB-v10.alt-05', 'RPB-v13']
VIEWS = ['training', 'validation_intact', 'validation_deleted']
LEGEND = '**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps'
CHECKS = 0


def require(ok, why):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise ValueError(why)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hex_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def near(a, b):
    require(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b)
            and abs(a - b) <= 2e-9 + 2e-9 * abs(b), 'audited metadata numeric association')


def optional_near(a, b):
    require((a is None) == (b is None), 'undefined score retained')
    if a is not None:
        near(a, b)


def mean(values):
    require(bool(values), 'nonempty declared aggregate')
    return None if any(x is None for x in values) else sum(values) / len(values)


def stats(values, prefix):
    defined = [x for x in values if x is not None]
    return {prefix + '_mean': mean(values), prefix + '_worst': min(defined) if len(defined) == len(values) else None,
            prefix + '_range': [min(defined), max(defined)] if len(defined) == len(values) else None,
            prefix + '_defined_cohorts': len(defined)}


def score(value, total):
    require(all(type(value[k]) is int for k in ('total', 'valid', 'correct')) and value['total'] == total and
            0 <= value['correct'] <= value['valid'] <= total, 'exact score count universe')
    result = Fraction(value['correct'], value['valid']) if value['valid'] else None
    optional_near(value['accuracy'], float(result) if result is not None else None)
    near(value['coverage'], value['valid'] / total)
    if 'full_population_correctness' in value:
        near(value['full_population_correctness'], value['correct'] / total)
    return result


def score_association(producer, audited, total):
    score(producer, total); score(audited, total)
    require(all(producer[k] == audited[k] for k in ('total', 'valid', 'correct')), 'exact saved replay score counts')
    for key in ('accuracy', 'coverage'):
        optional_near(producer[key], audited[key])


def method_summary(method, view, master):
    total = 256 if view == 'training' else 128
    pop = method[view + '_population']
    require(pop['total_rows'] == total and type(pop['valid_rows']) is int and 0 <= pop['valid_rows'] <= total,
            'whole fitting/scoring population retained')
    near(pop['coverage'], pop['valid_rows'] / total)
    require([r['id'] for r in method['repetitions']] == ['rep-' + str(r) for r in REPS], 'three fixed repetitions')
    require(method['status'] in ('measured', 'unsupported_fit'), 'explicit fitting disposition')
    values = {h: [] for h in ('ridge', 'tiny_secondary')}
    for rep in method['repetitions']:
        require(rep['status'] == method['status'], 'consistent fixed-head disposition')
        if method['status'] == 'unsupported_fit':
            require(view not in rep and bool(method['reason']), 'unsupported fit without invented score')
            for h in values:
                values[h].append(None)
        else:
            require(rep[view]['population'] == pop, 'saved repetition population association')
            for h in values:
                require(rep[view][h]['valid'] == pop['valid_rows'], 'score support is whole population')
                values[h].append(score(rep[view][h], total))
    return {'master': master, **{h: float(mean(v)) if mean(v) is not None else None for h, v in values.items()},
            'coverage': pop['valid_rows'] / total, 'valid_rows': pop['valid_rows'], 'total_rows': total,
            'fit_status': method['status'], 'reason': method['reason']}


def method_from(cohort, method):
    require(len(cohort['tasks']) == 1 and cohort['tasks'][0]['task'] == 'lag_sign', 'single timing task')
    methods = cohort['tasks'][0]['readouts']['methods']
    require(len([m for m in methods if m['method'] == method]) == 1, 'unique method in cohort')
    return next(m for m in methods if m['method'] == method)


def row(method, label, width, cohorts, view, evidence):
    rows = [method_summary(method_from(c, method), view, c['timing_master']) for c in cohorts]
    result = {'method': method, 'label': label, 'size': width, 'per_master': rows,
              'dataset_id': 'TEMPO-3', 'task': 'lag_sign', 'recipe_id': 'structured-hard-timing-v1',
              'designed_complexity_level': 4, 'complexity_scale_max': 5, 'view': view, 'evidence': evidence}
    for key in ('ridge', 'tiny_secondary', 'coverage'):
        result.update(stats([r[key] for r in rows], key))
    result.update(linear_accuracy_percent=None if result['ridge_mean'] is None else 100 * result['ridge_mean'],
                  neural_accuracy_percent=None if result['tiny_secondary_mean'] is None else 100 * result['tiny_secondary_mean'],
                  coverage_percent=None if result['coverage_mean'] is None else 100 * result['coverage_mean'])
    return result


def validate(audit, report, parent):
    require(audit['status'] == 'passed' and audit['protocol'] == report['protocol'] == PROTOCOL,
            'passed candidate saved-evidence audit')
    require(report['dataset_id'] == parent['dataset_id'] == 'TEMPO-3' and report['task'] == parent['task'] == 'lag_sign'
            and report['recipe_id'] == parent['recipe_id'] == 'structured-hard-timing-v1', 'unchanged retained dataset')
    require(parent['protocol'] == 'structured-hard-timing-comparison-v2' and
            report['parent_summary_sha256'] == audit['validated_metadata']['parent_summary_sha256'] == PARENT_SHA,
            'exact retained passed comparison')
    require(audit['card_sha256'] == audit['source']['card_sha256'] == CARD_SHA and
            audit['source_sha256'] == audit['source']['reader_sha256'] and
            audit['inventory_sha256'] == audit['source']['inventory_sha256'] and
            audit['source']['source_fingerprint'] == report['source_fingerprint'], 'reader/card/producer associations')
    require(all(hex_sha(v) for v in (audit['source_sha256'], audit['inventory_sha256'], report['source_fingerprint'])),
            'exact completed reader/inventory/source SHA pins')
    require(type(audit['checks']) is int and audit['checks'] > 0 and type(audit['archive_decodes']) is int and
            audit['archive_decodes'] > 0, 'actual completed saved arithmetic')
    require([c['timing_master'] for c in report['cohorts']] == [c['timing_master'] for c in audit['per_master']] ==
            [c['timing_master'] for c in parent['cohorts']] == MASTERS, 'all five declared cohorts retained')
    complete = audit['complete']
    require(complete['status'] == 'complete' and complete['protocol'] == PROTOCOL, 'passed complete metadata')
    for key, value in {'encoder_trajectories': 5, 'encoder_updates_each': 512, 'sampled_rows': 20480,
        'retained_points': 10, 'cohorts': 5, 'skipped_attempts': 0, 'checkpoint_roles_per_point': 6,
        'planned_pipelines': 15, 'planned_heads': 30, 'retained_payload_roles': 130, 'initial_parity_exports': 15,
        'quality_native_exports': 15, 'full_native_export_calls': 30, 'query_writer_calls': 10,
        'necessary_query_forwards': 40, 'baseline_or_control_refits': 0, 'parent_model_forwards': 0,
        'generator_calls': 0, 'PCA_fits': 0, 'extra_decoder_calibration_updates': 0}.items():
        require(type(complete[key]) is int and complete[key] == value, 'exact workload ' + key)
    for record in (report, complete):
        require(all(record[k] is False for k in ('selection', 'promotion', 'testing_accessed', 'stress_accessed')),
                'no selection/promotion/TEST/stress')
    require(all(audit['limits'][k] in (0, False) for k in ('model_forward', 'encoder_updates', 'head_refits',
            'PCA_fits', 'CUDA_checkpoint_body_decodes')) and audit['limits']['all_inputs_preserved'] is True,
            'saved-only audit boundary and preservation')
    fits = 0
    for cohort, checked in zip(report['cohorts'], audit['per_master']):
        require(len(cohort['encoder_points']) == 1 and cohort['encoder_points'][0]['model_tag'] == checked['model_tag'] == 'RPB-v13'
                and checked['initialization_exact'] is True, 'one exactly initialized candidate')
        point = cohort['encoder_points'][0]
        require(point['encoder_progress'] == checked['encoder_progress'] and cohort['costs'] == checked['costs'],
                'all producer traces and cost scopes retained verbatim')
        require(point['encoder_progress']['attempted'] == point['encoder_progress']['completed'] == 512 and
                point['encoder_progress']['sampled_rows'] == 4096 and point['encoder_progress']['parameter_count'] == 228877,
                'candidate budget/capacity')
        near(point['encoder_progress']['training_seconds'], checked['training_seconds'])
        for v in ('training', 'validation'):
            declared = point[v + '_reconstruction']; replay = checked['query'][v]
            for k, r in (('standardized_mae', 'mae'), ('standardized_huber', 'huber')):
                optional_near(declared[k], replay[r])
            require(all(declared[k] == replay[k] for k in ('valid_examples', 'total_examples', 'valid_target_cells',
                                                          'requested_observed_target_cells')), 'query original population')
        method = method_from(cohort, METHOD)
        require(len(cohort['tasks'][0]['readouts']['methods']) == 1 and method['size'] == 32,
                'candidate-only fitting method')
        require([r['repetition'] for r in checked['readouts']] == REPS, 'all audited repetitions')
        fits += 3 * int(method['status'] == 'measured')
        for producer, replay in zip(method['repetitions'], checked['readouts']):
            require(producer['id'] == 'rep-' + str(replay['repetition']) and producer['status'] == replay['status'] == method['status'],
                    'producer/replayed fixed head status')
            if method['status'] == 'unsupported_fit':
                require(replay['views'] == {}, 'undefined fit retained')
                continue
            require(set(replay['views']) == set(VIEWS), 'all audited scoring views')
            for view in VIEWS:
                for head in ('ridge', 'tiny_secondary'):
                    score_association(producer[view][head], replay['views'][view][head], 256 if view == 'training' else 128)
        require(cohort['tasks'][0]['reused_paired_comparisons'] == checked['paired'], 'full audited paired intervals preserved')
        require([(p['reference'], p['repetition'], p['view']) for p in checked['paired']] ==
                [(c, 'rep-' + str(r), v) for c in ('native_early', 'native_late') for r in REPS for v in VIEWS[1:]],
                'all two-control paired repetitions and views')
    require(complete['head_pipelines'] == fits and complete['individual_heads'] == 2 * fits,
            'actual fitting counts include explicit unsupported fits')
    for cohort in parent['cohorts']:
        require([m['method'] for m in cohort['tasks'][0]['readouts']['methods']] == OLD_METHODS,
                'all seven original control/baseline methods')


def summarize(audit, report, parent):
    quality = {}
    for view in VIEWS:
        quality[view] = [row(m, LABELS[i], WIDTHS[i], parent['cohorts'], view,
                            'reused audited parent; no new encoder export or head fit') for i, m in enumerate(OLD_METHODS)]
        quality[view].append(row(METHOD, LABELS[-1], 32, report['cohorts'], view,
                                 'new candidate-only CUDA export and saved-head arithmetic audit'))
        if view != 'training':
            require([r['method'] for r in parent['quality'][view]] == OLD_METHODS, 'old durable row order')
            for regenerated, original in zip(quality[view], parent['quality'][view]):
                require(regenerated['per_master'] == original['per_master'], 'reused whole durable cohort rows')
                for key in ('ridge_mean', 'tiny_secondary_mean', 'coverage_mean'):
                    optional_near(regenerated[key], original[key])
    training = copy.deepcopy(parent['training'])
    require([r['model_tag'] for r in training] == ['RPB-v7.alt-05', 'RPB-v10.alt-05'], 'two original reconstruction controls')
    for item in training:
        item['evidence'] = 'reused audited parent; encoder/query not rerun'
    points = [c['encoder_points'][0] for c in report['cohorts']]
    training.append({'model_tag': 'RPB-v13', 'updates': 512,
                     'train_mae': mean([a['query']['training']['mae'] for a in audit['per_master']]),
                     'validation_mae': mean([a['query']['validation']['mae'] for a in audit['per_master']]),
                     'training_loop_seconds': mean([a['training_seconds'] for a in audit['per_master']]),
                     'per_master': copy.deepcopy(points), 'evidence': 'new candidate; original query/target/support unchanged'})
    for item in training:
        item['parameter_count'] = 228877 if item['model_tag'] == 'RPB-v13' else 225805
    cost_names = list(report['cohorts'][0]['costs'])
    require(all(set(c['costs']) == set(cost_names) for c in report['cohorts']), 'all stage cost scopes')
    costs = {key: mean([c['costs'][key] for c in report['cohorts']]) for key in cost_names}
    return {'protocol': PROTOCOL, 'dataset_id': 'TEMPO-3', 'task': 'lag_sign', 'recipe_id': 'structured-hard-timing-v1',
            'designed_complexity_level': 4, 'complexity_scale_max': 5, 'quality': quality, 'training': training,
            'costs_mean_per_cohort_seconds': costs, 'complete': copy.deepcopy(audit['complete']),
            'cohorts': copy.deepcopy(report['cohorts']), 'audit_record': copy.deepcopy(audit),
            'retained_parent': copy.deepcopy(parent), 'limits': copy.deepcopy(audit['limits']),
            'selection': False, 'promotion': False,
            'cost_scope': 'training_loop_seconds includes CUDA updates plus synchronized scalar/counter trace work; full CPU live state/checkpoint capture is in binding_checkpoint_and_snapshot_IO_seconds',
            'comparison_scope': 'same saved TEMPO-3 observations, masks, labels, initial common state and original TRAIN scaler; route plus 3072 trainable parameters changed; descriptive comparison only'}


def number(value, percentage=False, digits=2):
    return '—' if value is None else f'{value * (100 if percentage else 1):.{digits}f}'


def interval(value):
    return number(value['estimate'], True) + ' [' + number(value['lower'], True) + ', ' + number(value['upper'], True) + ']'


def markdown(summary):
    lines = ['# Visible first difference diagnostic', '',
             'RPB-v13 adds a zero-initialized, bias-free projection of currently visible adjacent backward differences to the early mixer. The original 225,805 parameter values, buffers and TRAIN scaler matched the retained initial model before AdamW; the new 3,072 values increase total capacity to 228,877.', '',
             'Five candidate trajectories used 512 updates and the same retained TEMPO-3 data, masks, training streams, query targets and fixed heads. All seven parent baseline/control rows are reused from their passed audit. The candidate initial features matched saved untrained early features in all three views before each cohort’s training and head fitting.', '',
             'Timing task: 256 TRAIN rows / 128 source pairs and 128 VALIDATION rows / 64 source pairs per master, CUDA batch size 8. Each fixed head uses repetitions 2701/2802/2903: Ridge penalty 1, or Tiny tanh width 16 trained with Adam learning rate 0.01 for 100 steps. The native embedding width is 32, with no PCA afterward.', '',
             'This is a descriptive comparison. The input route and capacity changed together, so results do not isolate a causal effect of differences. No TEST, stress evaluation, model selection or promotion occurred.', '']
    for view in VIEWS:
        lines += ['## ' + {'training': 'TRAIN fixed-head quality', 'validation_intact': 'Intact validation', 'validation_deleted': 'Validation with extra 30% coordinate deletion'}[view], '', LEGEND,
                  '', '| Method | Size | Linear head % | Neural head % | Coverage % |', '|---|---:|---:|---:|---:|']
        for item in summary['quality'][view]:
            lines.append('| ' + item['label'] + ' | ' + str(item['size']) + ' | ' + number(item['ridge_mean'], True) + ' | ' + number(item['tiny_secondary_mean'], True) + ' | ' + number(item['coverage_mean'], True) + ' |')
        lines += ['', 'Percent accuracy is conditional on supported rows. Means weight all five cohorts equally, then all three fixed repetitions equally; any undefined required score keeps its aggregate undefined. Worst-cohort values and every population, head and interval are retained in the JSON.', '']
    lines += ['## Candidate cohort spread', '', LEGEND, '',
              '| Master | TRAIN linear / neural | Intact linear / neural | Deleted linear / neural | Coverage TRAIN / intact / deleted |',
              '|---|---:|---:|---:|---:|']
    for i, master in enumerate(MASTERS):
        rows = [summary['quality'][v][-1]['per_master'][i] for v in VIEWS]
        lines.append('| ' + str(master) + ' | ' + ' | '.join(number(r['ridge'], True) + ' / ' + number(r['tiny_secondary'], True) for r in rows) + ' | ' + ' / '.join(number(r['coverage'], True) for r in rows) + ' |')
    lines += ['', '## Reconstruction and training cost', '', LEGEND, '',
              '| Encoder | Updates | Train error | Validation error | GPU training seconds |',
              '|---|---:|---:|---:|---:|']
    for item in summary['training']:
        lines.append('| ' + item['model_tag'] + ' | 512 | ' + number(item['train_mae'], digits=6) + ' | ' + number(item['validation_mae'], digits=6) + ' | ' + number(item['training_loop_seconds'], digits=3) + ' |')
    lines += ['', 'RPB-v7.alt-05 and RPB-v10.alt-05 each have 225,805 parameters; their reconstruction errors and timers are reused without encoder/query execution. RPB-v13 has 228,877 parameters and is newly measured here.', '',
              'MAE uses the same original frozen TRAIN scaler and original masked query targets. The GPU training seconds column reports the synchronized loop timer, including CUDA work plus scalar/counter trace work. Full CPU live model/AdamW state capture and checkpoint work belong to the separately measured binding/checkpoint stage; this is not a pure GPU kernel timer.', '',
              LEGEND, '', '| Master | Candidate TRAIN MAE | Candidate VAL MAE | Candidate loop seconds |', '|---|---:|---:|---:|']
    for item in summary['audit_record']['per_master']:
        lines.append('| ' + str(item['timing_master']) + ' | ' + number(item['query']['training']['mae'], digits=6) + ' | ' + number(item['query']['validation']['mae'], digits=6) + ' | ' + number(item['training_seconds'], digits=3) + ' |')
    lines += ['', LEGEND, '', '| New candidate stage | Mean seconds per cohort |', '|---|---:|']
    for key, value in summary['costs_mean_per_cohort_seconds'].items():
        lines.append('| ' + key + ' | ' + number(value, digits=3) + ' |')
    for head, title in (('ridge', 'Linear'), ('tiny_secondary', 'Neural secondary')):
        lines += ['', '## ' + title + ' matched validation effects', '', LEGEND, '',
                  'Candidate minus retained control, in percentage points. Brackets are saved 95% source-group intervals conditional on each fitted head; intervals were copied after the audit, without new resampling.', '',
                  '| Master | Control | Repetition | View | Common rows | Effect [95% interval] |', '|---|---|---:|---|---:|---:|']
        for cohort in summary['audit_record']['per_master']:
            for pair in cohort['paired']:
                effect = interval(pair[head]) if head in pair else '— (' + pair['status'] + ')'
                lines.append('| ' + str(cohort['timing_master']) + ' | ' + pair['reference'] + ' | ' + pair['repetition'] + ' | ' + pair['view'] + ' | ' + str(pair.get('common_valid_rows', '—')) + ' | ' + effect + ' |')
    a = summary['audit_record']; b = summary['metadata_bindings']
    lines += ['', '## Evidence', '',
              'The sole saved-arithmetic audit passed ' + str(a['checks']) + ' checks and decoded ' + str(a['archive_decodes']) + ' CPU witness archives in ' + number(a['elapsed_seconds'], digits=3) + ' seconds. Ordinary CUDA checkpoint bodies were bound by bytes and admission; they were not decoded or run by the CPU reader.', '',
              'The candidate produced 15 quality exports plus 15 initial-parity exports, 10 query writers / 40 masked forwards, and ' + str(a['complete']['head_pipelines']) + ' supported pipelines / ' + str(a['complete']['individual_heads']) + ' individual heads. No baseline, control or untrained head was refitted, and no parent encoder was re-exported.', '',
              'Full five-cohort objective traces, reconstruction populations, individual intervals, matched pairs and retained parent metadata are saved in [the durable JSON](../../../doc/results/visible_difference_v1.json).', '',
              '- Card SHA: `' + CARD_SHA + '`',
              '- Producer SOURCE: `' + a['source']['source_fingerprint'] + '`',
              '- Reader SHA: `' + a['source_sha256'] + '`',
              '- Completed inventory SHA: `' + a['inventory_sha256'] + '`']
    for role in ('report', 'audit', 'parent_summary'):
        lines.append('- ' + role + ' SHA: `' + b[role]['sha256'] + '`')
    return '\n'.join(lines) + '\n'


def admit(paths):
    seen = set()
    for path in paths:
        require(path.is_absolute() and path.resolve(strict=True) == path and
                all(not p.is_symlink() for p in (path, *path.parents)), 'direct canonical metadata role')
        st = path.stat()
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and (st.st_dev, st.st_ino) not in seen,
                'whole unique non-aliased metadata matrix before bytes')
        seen.add((st.st_dev, st.st_ino))


def artificial_case(unsupported=False, null_view=False):
    def method(name, width):
        value = {'method': name, 'size': width, 'status': 'unsupported_fit' if unsupported and name == METHOD else 'measured',
                 'reason': 'synthetic unsupported TRAIN fit' if unsupported and name == METHOD else '', 'repetitions': []}
        for view in VIEWS:
            total = 256 if view == 'training' else 128
            valid = 0 if null_view and name == METHOD and view == 'validation_deleted' else total
            value[view + '_population'] = {'total_rows': total, 'valid_rows': valid, 'coverage': valid / total}
        for rep in REPS:
            record = {'id': 'rep-' + str(rep), 'status': value['status']}
            if value['status'] == 'measured':
                for view in VIEWS:
                    pop = value[view + '_population']; valid, total = pop['valid_rows'], pop['total_rows']
                    s = {'total': total, 'valid': valid, 'correct': valid // 2,
                         'coverage': valid / total, 'accuracy': .5 if valid else None,
                         'full_population_correctness': valid / (2 * total)}
                    record[view] = {'population': copy.deepcopy(pop), 'ridge': copy.deepcopy(s), 'tiny_secondary': copy.deepcopy(s)}
            value['repetitions'].append(record)
        return value
    parent = {'protocol': 'structured-hard-timing-comparison-v2', 'dataset_id': 'TEMPO-3', 'task': 'lag_sign',
              'recipe_id': 'structured-hard-timing-v1', 'cohorts': [], 'quality': {}, 'training': []}
    report = {'protocol': PROTOCOL, 'dataset_id': 'TEMPO-3', 'task': 'lag_sign', 'recipe_id': 'structured-hard-timing-v1',
              'source_fingerprint': 'b' * 64, 'parent_summary_sha256': PARENT_SHA, 'cohorts': [],
              'selection': False, 'promotion': False, 'testing_accessed': False, 'stress_accessed': False}
    audit = {'status': 'passed', 'protocol': PROTOCOL, 'source_sha256': 'a' * 64, 'inventory_sha256': 'c' * 64,
             'card_sha256': CARD_SHA, 'checks': 100, 'archive_decodes': 100, 'elapsed_seconds': 1,
             'source': {'reader_sha256': 'a' * 64, 'inventory_sha256': 'c' * 64, 'card_sha256': CARD_SHA, 'source_fingerprint': 'b' * 64},
             'validated_metadata': {'parent_summary_sha256': PARENT_SHA}, 'per_master': [],
             'limits': {'model_forward': False, 'encoder_updates': 0, 'head_refits': 0, 'PCA_fits': 0,
                        'CUDA_checkpoint_body_decodes': 0, 'all_inputs_preserved': True}}
    audit['complete'] = {'status': 'complete', 'protocol': PROTOCOL, 'encoder_trajectories': 5, 'encoder_updates_each': 512,
        'sampled_rows': 20480, 'retained_points': 10, 'cohorts': 5, 'skipped_attempts': 0, 'checkpoint_roles_per_point': 6,
        'planned_pipelines': 15, 'planned_heads': 30, 'retained_payload_roles': 130, 'initial_parity_exports': 15,
        'quality_native_exports': 15, 'full_native_export_calls': 30, 'query_writer_calls': 10, 'necessary_query_forwards': 40,
        'baseline_or_control_refits': 0, 'parent_model_forwards': 0, 'generator_calls': 0, 'PCA_fits': 0,
        'extra_decoder_calibration_updates': 0, 'selection': False, 'promotion': False, 'testing_accessed': False,
        'stress_accessed': False, 'head_pipelines': 0 if unsupported else 15, 'individual_heads': 0 if unsupported else 30}
    for master in MASTERS:
        parent['cohorts'].append({'timing_master': master, 'tasks': [{'task': 'lag_sign',
            'readouts': {'methods': [method(m, w) for m, w in zip(OLD_METHODS, WIDTHS)]}}]})
        native = method(METHOD, 32)
        pairs = [{'reference': control, 'repetition': 'rep-' + str(rep), 'view': view,
                  'status': 'unsupported_fit' if unsupported else 'measured', 'common_valid_rows': 128,
                  'ridge': {'estimate': 0., 'lower': -.1, 'upper': .1}, 'tiny_secondary': {'estimate': 0., 'lower': -.1, 'upper': .1}}
                 for control in ('native_early', 'native_late') for rep in REPS for view in VIEWS[1:]]
        progress = {'attempted': 512, 'completed': 512, 'sampled_rows': 4096, 'parameter_count': 228877,
                    'training_seconds': 1., 'losses': [[i, i, 1, .5, .1] for i in range(1, 513)]}
        point = {'model_tag': 'RPB-v13', 'encoder_progress': progress}
        queries = {}
        for view, total in (('training', 256), ('validation', 128)):
            point[view + '_reconstruction'] = {'standardized_mae': .25, 'standardized_huber': .1,
                'valid_examples': total, 'total_examples': total, 'valid_target_cells': total,
                'requested_observed_target_cells': total}
            queries[view] = {'mae': .25, 'huber': .1, 'valid_examples': total, 'total_examples': total,
                'valid_target_cells': total, 'requested_observed_target_cells': total}
        report['cohorts'].append({'timing_master': master, 'encoder_points': [point], 'costs': {'synthetic_stage_seconds': 1.},
            'tasks': [{'task': 'lag_sign', 'readouts': {'methods': [native]}, 'reused_paired_comparisons': pairs}]})
        replay = [{'repetition': rep, 'status': native['status'], 'views': {} if unsupported else {
            view: {head: copy.deepcopy(native['repetitions'][i][view][head]) for head in ('ridge', 'tiny_secondary')}
            for view in VIEWS}} for i, rep in enumerate(REPS)]
        audit['per_master'].append({'timing_master': master, 'model_tag': 'RPB-v13', 'initialization_exact': True,
            'encoder_progress': copy.deepcopy(progress), 'costs': {'synthetic_stage_seconds': 1.}, 'training_seconds': 1.,
            'query': queries, 'readouts': replay, 'paired': copy.deepcopy(pairs)})
    for view in VIEWS[1:]:
        parent['quality'][view] = [row(m, LABELS[i], WIDTHS[i], parent['cohorts'], view, 'synthetic') for i, m in enumerate(OLD_METHODS)]
    parent['training'] = [{'model_tag': tag, 'updates': 512, 'train_mae': .3, 'validation_mae': .4,
                           'training_loop_seconds': 1., 'per_master': []} for tag in ('RPB-v7.alt-05', 'RPB-v10.alt-05')]
    return audit, report, parent


def self_test():
    negatives = 0
    for options in ({}, {'unsupported': True}, {'null_view': True}):
        case = artificial_case(**options)
        validate(*case); result = summarize(*case)
        result['metadata_bindings'] = {r: {'sha256': 'a' * 64} for r in ('report', 'audit', 'parent_summary')}
        text = markdown(result)
        require(all(len(result['quality'][v]) == 8 for v in VIEWS) and len(result['training']) == 3,
                'all eight rows and three reconstruction rows retained')
        require(text.count(LEGEND) == 9 and all(str(m) in text for m in MASTERS), 'table legends and all masters')
        if options:
            require(result['quality']['validation_deleted'][-1]['ridge_mean'] is None, 'undefined aggregate never dropped')
    changes = [lambda a, r, p: a.update(status='failed'), lambda a, r, p: r.update(parent_summary_sha256='0' * 64),
        lambda a, r, p: r['cohorts'].reverse(), lambda a, r, p: a['per_master'].pop(),
        lambda a, r, p: a['complete'].update(full_native_export_calls=15),
        lambda a, r, p: a['per_master'][0]['readouts'][0]['views']['training']['ridge'].update(correct=129),
        lambda a, r, p: a['per_master'][0]['query']['training'].update(mae=.9),
        lambda a, r, p: a['per_master'][0]['encoder_progress'].update(sampled_rows=0),
        lambda a, r, p: a['per_master'][0]['paired'][0].update(reference='native_late'),
        lambda a, r, p: r['cohorts'][0]['tasks'][0]['readouts']['methods'][0]['repetitions'].pop(),
        lambda a, r, p: p['cohorts'][0]['tasks'][0]['readouts']['methods'].pop(),
        lambda a, r, p: r.update(promotion=True)]
    for change in changes:
        case = artificial_case(); change(*case)
        try:
            validate(*case); summarize(*case)
        except (ValueError, KeyError, IndexError):
            negatives += 1
        else:
            raise AssertionError('synthetic corruption must reject')
    require(negatives == 12 and mean([.5, None, .5]) is None, 'all synthetic negatives and null law')
    return {'status': 'passed', 'checks': CHECKS, 'negative_fixtures': negatives,
            'actual_metadata_reads': 0, 'actual_archive_reads': 0, 'model_or_head_fits': 0, 'bootstrap_resampling': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    for role in ('report', 'audit', 'parent-summary'):
        parser.add_argument('--' + role)
        parser.add_argument('--' + role + '-sha256')
    parser.add_argument('--source-sha256')
    parser.add_argument('--summary-output')
    parser.add_argument('--report-output')
    parser.add_argument('--proof-output')
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'existing managed container only')
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
        return
    require(all(getattr(args, name) for name in ('report', 'audit', 'parent_summary', 'report_sha256', 'audit_sha256',
        'parent_summary_sha256', 'source_sha256', 'summary_output', 'report_output', 'proof_output')),
        'explicit completed metadata and exclusive outputs')
    roles = {role: Path(getattr(args, role)) for role in ('report', 'audit', 'parent_summary')}
    source = Path(__file__).resolve()
    admit([*roles.values(), source])
    pins = {role: getattr(args, role + '_sha256') for role in roles}
    require(all(hex_sha(p) for p in [*pins.values(), args.source_sha256]) and pins['parent_summary'] == PARENT_SHA,
            'explicit reviewed source and exact retained metadata pins')
    require(digest(source) == args.source_sha256 and all(digest(p) == pins[r] for r, p in roles.items()),
            'all metadata bytes pinned before JSON reads')
    values = {role: json.loads(path.read_text()) for role, path in roles.items()}
    audit, report, parent = values['audit'], values['report'], values['parent_summary']
    require(audit['validated_metadata']['report_sha256'] == pins['report'], 'producer report authenticated by passed audit')
    validate(audit, report, parent)
    summary = summarize(audit, report, parent)
    summary['metadata_bindings'] = {role: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': pins[role]} for role, path in roles.items()}
    summary['publisher_sha256'] = args.source_sha256
    text = markdown(summary)
    payload = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    outputs = [Path(args.summary_output), Path(args.report_output), Path(args.proof_output)]
    require(outputs[0].name == 'visible_difference_v1.json' and outputs[1].name == 'VISIBLE_DIFFERENCE_DIAGNOSTIC.md',
            'exact durable filenames')
    require(len(set(outputs)) == 3 and all(p.is_absolute() and p.parent.resolve(strict=True) == p.parent and
            all(not q.is_symlink() for q in (p.parent, *p.parent.parents)) and not p.exists() and not p.is_symlink() for p in outputs),
            'exclusive canonical new outputs')
    require(all(digest(path) == pins[role] for role, path in roles.items()) and digest(source) == args.source_sha256,
            'all inputs preserved through metadata QA')
    for path, body in zip(outputs[:2], (payload, text)):
        with path.open('x', encoding='utf-8', newline='\n') as target:
            target.write(body)
    require(all(digest(path) == pins[role] for role, path in roles.items()) and digest(source) == args.source_sha256,
            'all inputs preserved after exact publication')
    proof = {'status': 'passed', 'protocol': PROTOCOL, 'checks': CHECKS, 'metadata_inputs': summary['metadata_bindings'],
             'publisher_sha256': args.source_sha256,
             'outputs': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in outputs[:2]],
             'archives_opened': 0, 'model_or_head_fits': 0, 'bootstrap_resampling': 0, 'audit_reruns': 0,
             'all_inputs_preserved': True}
    with outputs[2].open('x', encoding='utf-8', newline='\n') as target:
        target.write(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    print(json.dumps(proof, allow_nan=False))


if __name__ == '__main__':
    main()

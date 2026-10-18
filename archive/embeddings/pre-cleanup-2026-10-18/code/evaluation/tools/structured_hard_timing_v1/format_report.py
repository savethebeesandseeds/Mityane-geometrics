#!/usr/bin/env python3
"""Format five explicitly hash-bound JSON metadata roles after one passed audit.

No archive, encoder, head, PCA, bootstrap or audit is executed. SOURCE fixtures
are artificial. The caller supplies completed metadata only after its audit.
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
PROTOCOL = 'structured-hard-timing-comparison-v2'
MASTERS = [75272, 76373, 77474, 78575, 79676]
REPETITIONS = ['rep-2701', 'rep-2802', 'rep-2903']
METHODS = ['raw', 'mask_metadata', 'pca_only', 'untrained_late', 'untrained_early', 'native_late', 'native_early']
WIDTHS = [576, 288, 32, 32, 32, 32, 32]
LABELS = ['Raw data — no encoder', 'Mask metadata — no encoder', 'PCA only — no encoder',
          'Untrained late mixer', 'Untrained early mixer', 'RPB-v7.alt-05', 'RPB-v10.alt-05']
VIEWS = ['validation_intact', 'validation_deleted']
LEGEND = '**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps'
RUN_ROOT = Path('/embedding/output/runs/rpb-structured-hard-timing')
CARD_SHA = '35f7aa9987f293bafbe735506cc86cd2af02f2ab39bec42dd1dc2d2dedfa758c'
INFO_PINS = ['404ec6e0ef2134bf5a99c483358fa2c8f1c95b5f2ae2bb4237764fa7ac45a1e6',
             '6c5f295c7c3956d5ede11a8f1bc28a8402031ee1946c769f876ce9e457aee8b8']
ROLES = ('audit', 'experiment', 'complete', 'information_v1', 'information_v2')
COUNTS = {'cohorts': 5, 'tasks_each': 1, 'encoder_trajectories': 10, 'encoder_updates_each': 512,
          'attempt_limit': 1024, 'retained_points': 20, 'skipped_attempts': 0, 'sampled_rows': 40960,
          'planned_pipelines': 105, 'planned_heads': 210, 'unique_quality_native_exports': 60,
          'initial_counterpart_exports': 0, 'full_native_export_calls': 60, 'query_evaluation_calls': 20,
          'necessary_query_forwards': 80, 'analytic_solver_fits': 0, 'analytic_individual_heads': 0,
          'analytic_surface_evaluations': 15, 'driver_raw_outer_fits': 5, 'extra_decoder_calibration_updates': 0}
CHECKS = 0


def require(ok, why):
    global CHECKS
    if not ok:
        raise ValueError(why)
    CHECKS += 1


def near(a, b):
    require(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and
            abs(a - b) <= 2e-9 + 2e-9 * abs(b),
            'metadata arithmetic association')


def optional_near(a, b):
    require((a is None) == (b is None), 'undefined metadata retained')
    if a is not None:
        near(a, b)


def audit_score(declared, checked):
    require(all(declared[k] == checked[k] for k in ('total', 'valid', 'correct')), 'audited exact score counts')
    for key in ('accuracy', 'coverage'):
        optional_near(declared[key], checked[key])


def audit_interval(declared, checked):
    require(declared['replicates'] == 1000 and declared['confidence'] == .95 and
            declared['source_groups'] == checked['source_groups'], 'audited conditional interval population')
    for key in ('estimate', 'lower', 'upper'):
        optional_near(declared[key], checked[key])


def validate_information(records):
    rows = []
    for index, record in enumerate(records):
        version = index + 1
        receipt = record['information_receipt']
        info = receipt['information']
        require(record['source_record_sha256'] == INFO_PINS[index] and
                record['protocol'] == info['protocol'] == f'structured-hard-timing-information-v{version}',
                'preserved separate information attempts')
        require(record['dataset_id'] == info['dataset_id'] == 'TEMPO-3' and
                record['recipe_id'] == 'structured-hard-timing-v1', 'unchanged information dataset')
        require(receipt['archive_reads'] == receipt['encoder_or_head_fits'] == 0 and
                receipt['quality_generated'] is False and info['TEST_generated'] is False,
                'data-only information scope')
        require(receipt['passed'] is bool(index) and info['passed'] is bool(index), 'original failure and later pass retained')
        require([c['seed'] for c in info['seeds']] == ([80383, 81484] if index == 0 else [82585, 83686]),
                'separate fresh engineering masters')
        for cohort in info['seeds']:
            require(cohort['training_source_pairs'] == cohort['validation_source_pairs'] == 512, 'information split budget')
            for view in ('intact', 'deleted'):
                value = cohort[view]
                accuracy = score(value, 1024)
                accuracy_min, coverage_min = ((.99, .99) if view == 'intact' else (.98, .95))
                require(value['passed'] == (accuracy is not None and accuracy >= accuracy_min and value['coverage'] >= coverage_min),
                        'unchanged independent information thresholds')
                rows.append({'version': version, 'rule': 'same-feature v1' if not index else 'cross-feature v2',
                             'master': cohort['seed'], 'view': view, **copy.deepcopy(value)})
    return rows


def mean(values):
    return None if any(x is None for x in values) else sum(values) / len(values)


def number(value, percentage=False, digits=2):
    return '—' if value is None else f'{float(value) * (100 if percentage else 1):.{digits}f}'


def interval_text(value):
    return number(value['estimate'], True) + ' [' + number(value['lower'], True) + ', ' + number(value['upper'], True) + ']'


def score(value, total):
    require(value['total'] == total and all(type(value[k]) is int for k in ('total', 'valid', 'correct')),
            'declared score integer population')
    require(0 <= value['correct'] <= value['valid'] <= total, 'score counts')
    accuracy = Fraction(value['correct'], value['valid']) if value['valid'] else None
    require((value['accuracy'] is None) == (accuracy is None), 'undefined accuracy retained')
    if accuracy is not None:
        near(value['accuracy'], float(accuracy))
    near(value['coverage'], value['valid'] / total)
    near(value['full_population_correctness'], value['correct'] / total)
    return accuracy


def cohort_method(method, view):
    population = method[view + '_population']
    require(population['total_rows'] == 128 and 0 <= population['valid_rows'] <= 128,
            'whole validation population')
    near(population['coverage'], population['valid_rows'] / 128)
    require([r['id'] for r in method['repetitions']] == REPETITIONS, 'all three fixed repetitions')
    values = {head: [] for head in ('ridge', 'tiny_secondary')}
    for repetition in method['repetitions']:
        require(repetition['status'] == method['status'], 'method/repetition fit status')
        if method['status'] == 'unsupported_fit':
            require(view not in repetition and method['reason'], 'unsupported TRAIN fit is named')
            for head in values:
                values[head].append(None)
            continue
        require(method['status'] == 'measured', 'closed method status')
        for head in values:
            values[head].append(score(repetition[view][head], 128))
        require(repetition[view]['population'] == population, 'score support association')
    return {head: mean(v) for head, v in values.items()} | {
        'coverage': Fraction(population['valid_rows'], 128),
        'valid_rows': population['valid_rows'], 'total_rows': 128,
        'fit_status': method['status'], 'reason': method['reason']}


def validate(audit, experiment, complete, information_v1, information_v2):
    require(audit['status'] == 'passed' and audit['protocol'] == experiment['protocol'] == complete['protocol'] == PROTOCOL,
            'one passed protocol')
    require(audit['dataset'] == {'codename':'TEMPO-3','recipe':'structured-hard-timing-v1',
            'designed_complexity_level':4,'maximum':5,'rule_id':'observed-cross-feature-determinant-v2'}, 'audited fixed dataset legend')
    require(experiment['dataset_id'] == 'TEMPO-3' and experiment['designed_complexity_level'] == 4 and
            experiment['complexity_scale_max'] == 5 and experiment['complexity_is_designed_not_accuracy_derived'] is True,
            'fixed ordinal dataset identity')
    require(complete['dataset_codename'] == 'TEMPO-3' and complete['status'] == 'complete', 'completed dataset')
    require(experiment['information_rule_id'] == complete['information_rule_id'] == 'observed-cross-feature-determinant-v2',
            'fixed cross-feature information rule')
    source = audit['source']
    require(source['source_fingerprint'] == experiment['source_fingerprint'] and source['human_card_sha256'] == CARD_SHA and
            source['information_sha256'] == INFO_PINS[1], 'same producer, frozen card and passed information binding')
    require(all(type(source[k]) is str and len(source[k]) == 64 and all(c in '0123456789abcdef' for c in source[k])
                for k in ('source_fingerprint', 'reader_sha256', 'inventory_sha256', 'admission_log_sha256')),
            'exact source/reader/inventory/admission hashes')
    require(type(audit['checks']) is int and audit['checks'] > 0 and type(audit['archive_decodes']) is int and
            audit['archive_decodes'] > 0, 'one completed independent archive audit')
    validate_information((information_v1, information_v2))
    require([c['timing_master'] for c in experiment['cohorts']] == MASTERS and
            [c['timing_master'] for c in audit['per_master']] == MASTERS, 'all five ordered independent masters')
    for name, expected in COUNTS.items():
        require(type(complete[name]) is int and complete[name] == expected, 'fixed completion count: ' + name)
    for name in ('CPU_encoder_training', 'CPU_encoder_forward', 'testing_accessed', 'stress_accessed', 'selection', 'promotion'):
        require(complete[name] is False, 'zero forbidden scope: ' + name)
    require(complete['initial_shared_state_exact_before_training_and_heads'] is True and complete['separate_initial_controls'] == 2,
            'two independently served initial controls and paired state')
    pipelines = 0
    for cohort, verified in zip(experiment['cohorts'], audit['per_master']):
        require(len(cohort['tasks']) == 1 and cohort['tasks'][0]['task'] == 'lag_sign', 'timing task only')
        task = cohort['tasks'][0]
        methods = task['readouts']['methods']
        require([m['method'] for m in methods] == METHODS and [m['size'] for m in methods] == WIDTHS,
                'all seven representations/native sizes')
        require(len(verified['tasks']) == 1 and verified['tasks'][0]['task'] == 'lag_sign', 'one audited timing task')
        checked = verified['tasks'][0]
        require(checked['data_master'] == cohort['timing_master'] and checked['pairs'] == task['readouts']['pairs'],
                'whole original paired records retained')
        require([m['method'] for m in checked['methods']] == METHODS, 'audited seven-method order')
        for declared, replayed in zip(methods, checked['methods']):
            require((declared['method'], declared['size'], declared['status']) ==
                    (replayed['method'], replayed['size'], replayed['status']), 'audited method identity/status')
            require([r['repetition'] for r in replayed['repetitions']] ==
                    ([] if declared['status'] == 'unsupported_fit' else REPETITIONS), 'all supported replayed repetitions')
            for saved_rep, checked_rep in zip(declared['repetitions'], replayed['repetitions']):
                require(saved_rep['id'] == checked_rep['repetition'] and saved_rep['actual_probe_seed_decimal'] ==
                        checked_rep['actual_probe_seed_decimal'], 'fixed actual head seed association')
                for view in ('training', *VIEWS):
                    for head in ('ridge', 'tiny_secondary'):
                        audit_score(saved_rep[view][head], checked_rep[view][head])
                        if view != 'training':
                            field = 'ridge_grouped_interval' if head == 'ridge' else 'tiny_grouped_interval'
                            audit_interval(saved_rep[view][field], checked_rep[view][head]['grouped_interval'])
        analytic = checked['analytic_solvability']
        require(analytic == task['analytic_solvability'], 'whole audited analytic record association')
        require(len(cohort['encoder_points']) == 2 and [p['placement'] for p in cohort['encoder_points']] == [0, 1],
                'both architectures retained')
        require(len(verified['encoders']) == 2, 'two audited encoder records')
        for point, replayed in zip(cohort['encoder_points'], verified['encoders']):
            progress = point['encoder_progress']
            require(point['model_tag'] == replayed['tag'] and point['placement'] == replayed['placement'] and
                    replayed['budget'] == 512, 'audited architecture/budget association')
            require(progress['attempted'] == progress['completed'] == 512 and progress['sampled_rows'] == 4096 and
                    progress['parameter_count'] == progress['cuda_parameter_count'] == 225805 and len(progress['losses']) == 512,
                    'complete CUDA trajectory and trace')
            require(progress['last_input_cuda'] is True and progress['last_loss_cuda'] is True and
                    progress['finite_gradients'] is True and progress['weights_changed'] is True,
                    'actual CUDA weight updates')
            require(all(len(row) == 5 and row[0] == row[1] == i + 1 and row[2] > 0 and
                        all(type(v) in (int, float) and math.isfinite(v) for v in row)
                        for i, row in enumerate(progress['losses'])), 'complete finite per-update trace')
            require(type(progress['training_seconds']) in (int, float) and math.isfinite(progress['training_seconds']) and
                    progress['training_seconds'] >= 0, 'descriptive loop timer')
            near(progress['training_seconds'], replayed['training_seconds'])
            require(all(progress[k] == replayed[k] for k in ('attempted', 'completed', 'sampled_rows')), 'audited update counters')
            for name, key in (('training', 'training_reconstruction'), ('validation', 'validation_reconstruction')):
                optional_near(point[key]['standardized_mae'], replayed['fixed_query'][name]['standardized_mae'])
        for method in methods:
            pipelines += sum(r['status'] == 'measured' for r in method['repetitions'])
            for view in VIEWS:
                cohort_method(method, view)
        for view, total in [('training', 256), ('validation_intact', 128), ('validation_deleted', 128)]:
            result = task['analytic_solvability'][view]
            require(result['fitted_heads'] == 0 and result['observed_only'] is True and result['total_examples'] == total,
                    'analytic information diagnostic has no fitted heads')
            score({'total': total, 'valid': result['valid_examples'], 'correct': result['correct_supported'],
                   'accuracy': result['accuracy'], 'coverage': result['coverage'],
                   'full_population_correctness': result['full_population_correctness']}, total)
    require(complete['head_pipelines'] == pipelines and complete['individual_heads'] == 2 * pipelines,
            'actual supported fits retained, not silently filled')
    helper_fits = sum(m['outer_train_normalizer_fits'] for c in experiment['cohorts'] for m in c['tasks'][0]['readouts']['methods'])
    require(complete['helper_outer_train_fits'] == helper_fits <= 25, 'one helper outer map per supported unprepared method')
    return pipelines


def summarize(audit, experiment, complete, information_v1, information_v2):
    validate(audit, experiment, complete, information_v1, information_v2)
    quality = {}
    for view in VIEWS:
        panel = []
        for index, method_name in enumerate(METHODS):
            rows = [cohort_method(c['tasks'][0]['readouts']['methods'][index], view) for c in experiment['cohorts']]
            item = {'method': method_name, 'label': LABELS[index], 'size': WIDTHS[index],
                    'per_master': [{'master': m, **{k: float(v) if isinstance(v, Fraction) else v for k, v in row.items()}}
                                   for m, row in zip(MASTERS, rows)]}
            for head in ('ridge', 'tiny_secondary', 'coverage'):
                values = [r[head] for r in rows]
                item[head + '_mean'] = None if any(x is None for x in values) else float(mean(values))
                item[head + '_worst'] = None if any(x is None for x in values) else float(min(values))
                item[head + '_range'] = None if any(x is None for x in values) else [float(min(values)), float(max(values))]
                item[head + '_defined_cohorts'] = sum(x is not None for x in values)
            panel.append(item)
        quality[view] = panel
    training = []
    for role in range(2):
        points = [c['encoder_points'][role] for c in experiment['cohorts']]
        training.append({'model_tag': ['RPB-v7.alt-05', 'RPB-v10.alt-05'][role], 'updates': 512,
                         'train_mae': mean([p['training_reconstruction']['standardized_mae'] for p in points]),
                         'validation_mae': mean([p['validation_reconstruction']['standardized_mae'] for p in points]),
                         'training_loop_seconds': mean([p['encoder_progress']['training_seconds'] for p in points]),
                         'per_master': copy.deepcopy(points)})
    analytic = []
    for view in ('training', *VIEWS):
        values = [c['tasks'][0]['analytic_solvability'][view] for c in experiment['cohorts']]
        accuracy = mean([Fraction(v['correct_supported'],v['valid_examples']) if v['valid_examples'] else None for v in values])
        analytic.append({'view': view, 'conditional_accuracy_mean': None if accuracy is None else float(accuracy),
                         'coverage_mean': float(mean([Fraction(v['valid_examples'],v['total_examples']) for v in values])),
                         'full_population_correctness_mean': float(mean([Fraction(v['correct_supported'],v['total_examples']) for v in values])),
                         'valid_counts': [v['valid_examples'] for v in values], 'total_each': values[0]['total_examples'],
                         'per_master': copy.deepcopy(values)})
    costs = {key: mean([c['costs'][key] for c in experiment['cohorts']]) for key in experiment['cohorts'][0]['costs']}
    return {'protocol': PROTOCOL, 'dataset_id': 'TEMPO-3', 'task': 'lag_sign', 'recipe_id': 'structured-hard-timing-v1',
            'designed_complexity_level': 4, 'complexity_scale_max': 5, 'quality': quality, 'training': training,
            'analytic': analytic, 'information_admission_attempts': [copy.deepcopy(information_v1), copy.deepcopy(information_v2)],
            'information_admission_rows': validate_information((information_v1, information_v2)),
            'costs_mean_per_cohort_seconds': costs, 'complete': copy.deepcopy(complete),
            'cohorts': copy.deepcopy(experiment['cohorts']), 'audit': copy.deepcopy(audit),
            'limits': {'complexity_is_ordinal_designed': True, 'promotion': False,
                       'quality_means_equal_master_then_equal_head_repetition': True,
                       'undefined_cohorts_not_dropped': True, 'interval_bounds_not_averaged': True,
                       'metadata_only_formatter': True, 'model_head_PCA_bootstrap_or_audit_execution': False}}


def table(lines, view, headers, rows):
    lines.extend(['', 'Dataset: ' + LEGEND + ' · view: ' + view + '.', '',
                  '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)


def markdown(summary):
    lines = ['# TEMPO-3 structured harder timing comparison', '',
             'New measurements on a separately specified timing recipe. Designed complexity 4/5 is ordinal, fixed before measurement; it is not an accuracy score.', '',
             'Each of five paired masters has 256 TRAIN examples from 128 independent sources and 128 VALIDATION examples from 64 separate sources. Ten fresh CUDA trajectories complete 512 updates at batch 8 with no skips (16 equivalent presentations). Native exports are 32 numbers; each design has 225,805 parameters. No TEST/stress or architecture promotion.', '',
             'The late RPB-v7.alt-05 mixer follows the temporal pass. RPB-v10.alt-05 mixes before it and retains an independent unmixed local temporal pass. Equal parameters do not imply equal compute. These fresh harder-data results do not change original saved v7 weights.', '',
             'Ridge penalty 1 and tanh16/Adam .01/100 updates use repetitions 2701/2802/2903, fitted once on TRAIN and retained for both VALIDATION views. Raw/PCA share one TRAIN outer map per cohort; encoder features have no PCA. Raw/mask heads contain 1,154/578 linear and 9,266/4,658 neural parameters; native/PCA32 heads contain 66/562.', '',
             'Tables use equal-master means after averaging all three fixed head repetitions. Every cohort, unsupported fit, conditional source-group interval and paired common-support record remains in the JSON. Undefined cohorts keep means undefined; they are not removed. Conditional interval bounds are never averaged into retraining uncertainty.']
    for view in VIEWS:
        panel = summary['quality'][view]
        table(lines, view.replace('_', ' '), ['Method', 'Size', 'Linear head %', 'Neural head %', 'Coverage %'],
              [[row['label'], row['size'], number(row['ridge_mean'], True), number(row['tiny_secondary_mean'], True), number(row['coverage_mean'], True)] for row in panel])
        table(lines, view.replace('_', ' ') + ' · variation across five retrainings',
              ['Encoder', 'Mean linear %', 'Worst linear %', 'Linear range %', 'Mean neural %', 'Worst neural %', 'Neural range %', 'Coverage %'],
              [[row['label'], number(row['ridge_mean'], True), number(row['ridge_worst'], True),
                '—' if row['ridge_range'] is None else '–'.join(number(x, True) for x in row['ridge_range']),
                number(row['tiny_secondary_mean'], True), number(row['tiny_secondary_worst'], True),
                '—' if row['tiny_secondary_range'] is None else '–'.join(number(x, True) for x in row['tiny_secondary_range']),
                number(row['coverage_mean'], True)] for row in panel[-2:]])
        late, early = panel[-2:]
        table(lines, view.replace('_', ' ') + ' · all paired cohorts',
              ['Master', 'RPB-v7.alt-05 linear %', 'RPB-v10.alt-05 linear %', 'Difference pp', 'Late/early coverage %'],
              [[m, number(a['ridge'], True), number(b['ridge'], True),
                number(None if a['ridge'] is None or b['ridge'] is None else b['ridge'] - a['ridge'], True),
                number(a['coverage'], True) + '/' + number(b['coverage'], True)]
               for m, a, b in zip(MASTERS, late['per_master'], early['per_master'])])
        table(lines, view.replace('_', ' ') + ' · all paired cohorts · fixed neural head',
              ['Master', 'RPB-v7.alt-05 neural %', 'RPB-v10.alt-05 neural %', 'Difference pp', 'Late/early coverage %'],
              [[m, number(a['tiny_secondary'], True), number(b['tiny_secondary'], True),
                number(None if a['tiny_secondary'] is None or b['tiny_secondary'] is None else b['tiny_secondary'] - a['tiny_secondary'], True),
                number(a['coverage'], True) + '/' + number(b['coverage'], True)]
               for m, a, b in zip(MASTERS, late['per_master'], early['per_master'])])
        lines.extend(['', 'Per-cohort differences above use each method’s own support. The saved paired intervals use exact common support; both populations and every repetition are retained in JSON.'])
        paired_rows = []
        for cohort in summary['cohorts']:
            for pair in cohort['tasks'][0]['readouts']['pairs']:
                if pair['view'] != view:
                    continue
                if pair['status'] == 'unsupported_fit':
                    paired_rows.append([cohort['timing_master'], pair['repetition'], '—', '—', '—'])
                else:
                    paired_rows.append([cohort['timing_master'], pair['repetition'], number(pair['common_population']['coverage'], True),
                                        interval_text(pair['ridge']), interval_text(pair['tiny_secondary'])])
        table(lines, view.replace('_', ' ') + ' · within-master conditional paired uncertainty',
              ['Master', 'Head repetition', 'Common coverage %', 'Linear effect [95% interval] pp', 'Neural effect [95% interval] pp'], paired_rows)
        lines.extend(['', 'Effects are early minus late on common source support. These source-group percentile intervals hold each saved encoder/readout fixed; they do not quantify encoder retraining uncertainty. Every fixed head repetition is shown.'])
    table(lines, 'original whole-patch TRAIN and intact VALIDATION queries',
          ['Encoder', 'Updates', 'Train error ↓', 'Validation error ↓', 'GPU training seconds'],
          [[r['model_tag'], r['updates'], number(r['train_mae'], digits=6), number(r['validation_mae'], digits=6),
            number(r['training_loop_seconds'])] for r in summary['training']])
    lines.extend(['', 'Errors are standardized MAE on fixed hidden-target queries, distinct from the random-mask Huber trace and classifier accuracy. The synchronized training-loop timer includes CPU trace capture and excludes checkpoint writing, feature/query extraction and CPU head work; it is not pure GPU kernel time. Ordinary updates jointly train the decoder; extra decoder-only calibration is zero. Every 512-entry loss/gradient trace is retained.'])
    table(lines, 'observed-only analytic information diagnostic',
          ['View', 'Conditional accuracy %', 'Coverage %', 'Full-population correctness %', 'Valid rows per master / total'],
          [[r['view'], number(r['conditional_accuracy_mean'], True), number(r['coverage_mean'], True),
            number(r['full_population_correctness_mean'], True), '/'.join(map(str, r['valid_counts'])) + ' / ' + str(r['total_each'])]
           for r in summary['analytic']])
    lines.extend(['', 'The analytic rule uses legal observed values/masks only, cancels constant offsets and has zero fitted heads. It abstains below four supported centres or at zero margin. It is a declared information check, not Bayes optimality or a classifier head result.'])
    for version in (1, 2):
        table(lines, f'premodel information admission v{version} · fresh engineering cohorts',
              ['Rule', 'Master', 'View', 'Valid / total', 'Conditional accuracy %', 'Coverage %', 'Gate'],
              [[r['rule'], r['master'], r['view'], f"{r['valid']}/{r['total']}", number(r['accuracy'], True),
                number(r['coverage'], True, 4), 'PASS' if r['passed'] else 'FAIL']
               for r in summary['information_admission_rows'] if r['version'] == version])
    lines.extend(['', 'The same-feature v1 rule failed its predeclared deleted-coverage gate. The separately specified cross-feature v2 rule passed unchanged accuracy/coverage thresholds on fresh seeds. Both use the identical TEMPO-3 generator; this is not a paired change estimate or encoder-quality result. The original failed receipt/report/JSON remain preserved.'])
    table(lines, 'whole experiment · descriptive mean per cohort', ['Cost scope', 'Seconds'],
          [[name.replace('_', ' '), number(value)] for name, value in summary['costs_mean_per_cohort_seconds'].items()])
    lines.extend(['', 'Mixed scopes include transfers, verification and/or I/O as their names state. CPU head scope includes fixed fit, prediction, bootstrap and asset writing. Audit cost is separate. Retain all five cohorts when interpreting means and worst-cohort reliability; no automatic numerical promotion rule or best-seed selection applies.', '',
                  'Provenance, complete counters, all analytic counts, head predictions summarized as metadata, marginal/paired intervals, original-query summaries and traces are retained in `structured_hard_timing_comparison_v2.json`. The formatter reads JSON metadata only; it runs no encoder, head, PCA, bootstrap or independent audit.', '',
                  'Audit: ' + str(summary['audit']['checks']) + ' checks / ' + str(summary['audit']['archive_decodes']) +
                  ' saved archive decodes. Source `' + summary['audit']['source']['source_fingerprint'] + '`, reader `' +
                  summary['audit']['source']['reader_sha256'] + '`, inventory `' + summary['audit']['source']['inventory_sha256'] + '`.', ''])
    return '\n'.join(lines)


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve(strict=True) == p and p.suffix == '.json', 'direct absolute JSON metadata')
    require(all(not n.is_symlink() for n in (p, *p.parents)), 'no metadata symlink ancestors')
    s = p.stat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'regular unaliased metadata')
    return p


def load_inputs(args):
    paths = [canonical(getattr(args, name)) for name in ROLES]
    require(len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) == 5, 'whole five-role metadata matrix before hashes')
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    require(before == [getattr(args, n + '_sha256') for n in ROLES], 'exact root input pins')
    values = [json.loads(p.read_text()) for p in paths]
    binding = values[0]['validated_metadata']
    require(binding['report_sha256'] == before[1] and binding['complete_sha256'] == before[2], 'audit binds these exact producer JSONs')
    capsule = Path(values[0]['source']['capsule'])
    require(capsule.parent == RUN_ROOT and capsule.name.startswith('structured-hard-timing-') and
            paths[1] == capsule/'results/report.json' and paths[2] == capsule/'results/complete.json', 'same completed capsule routes')
    require(paths[3:] == [Path('/embedding/doc/results/structured_hard_timing_information_v1.json'),
                         Path('/embedding/doc/results/structured_hard_timing_information_v2.json')], 'two explicit durable information metadata roles')
    return paths, before, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    for role in ROLES:
        parser.add_argument('--' + role.replace('_', '-'))
        parser.add_argument('--' + role.replace('_', '-') + '-sha256')
    parser.add_argument('--output')
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'managed container only')
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    require(all(getattr(args, n) for n in (*ROLES, *(r + '_sha256' for r in ROLES), 'output')),
            'all exact metadata roles/pins and exclusive output required')
    paths, before, values = load_inputs(args)
    summary = summarize(*values)
    summary['metadata_bindings'] = {role: {'path': str(path), 'sha256': digest}
                                    for role, path, digest in zip(ROLES, paths, before)}
    summary['formatter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == RUN_ROOT / 'report-tools' and output.name.startswith('emitted-') and
            output.parent.resolve(strict=True) == output.parent and not output.exists() and not output.is_symlink(), 'exclusive report leaf')
    require(all(not p.is_symlink() for p in output.parents), 'canonical output ancestors')
    report = markdown(summary)
    require([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before, 'metadata preserved before emission')
    output.mkdir()
    for name, body in [('STRUCTURED_HARD_TIMING_COMPARISON_DIAGNOSTIC.md', report),
                       ('structured_hard_timing_comparison_v2.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')]:
        with (output / name).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(body)
    require([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before, 'metadata preserved after emission')
    print(json.dumps({'status': 'passed', 'checks': CHECKS, 'output': str(output), 'metadata_roles': 5,
                      'archive_reads': 0, 'model_or_head_execution': False,
                      'files': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir())]}))


def artificial_metadata():
    """Literal metadata fixtures; no tensor, model, fit, bootstrap or path reads."""
    def counts(total, valid, correct):
        return {'total': total, 'valid': valid, 'correct': correct, 'accuracy': correct / valid if valid else None,
                'coverage': valid / total, 'full_population_correctness': correct / total}
    def population(total, valid):
        return {'total_rows': total, 'valid_rows': valid, 'coverage': valid/total,
                'class_valid_rows': [valid//2, valid-valid//2], 'total_source_groups': total//2, 'valid_source_groups': (valid+1)//2}
    def interval(value):
        return {'replicates':1000, 'confidence':.95, 'source_groups':64 if value is not None else 0,
                'estimate':value, 'lower':value, 'upper':value}
    complete = dict(COUNTS, protocol=PROTOCOL, dataset_codename='TEMPO-3', status='complete',
        head_pipelines=105, individual_heads=210, helper_outer_train_fits=25,
        initial_shared_state_exact_before_training_and_heads=True, separate_initial_controls=2,
        information_rule_id='observed-cross-feature-determinant-v2')
    for name in ('CPU_encoder_training','CPU_encoder_forward','testing_accessed','stress_accessed','selection','promotion'):
        complete[name] = False
    experiment = {'protocol':PROTOCOL,'dataset_id':'TEMPO-3','designed_complexity_level':4,'complexity_scale_max':5,
        'complexity_is_designed_not_accuracy_derived':True,'information_rule_id':'observed-cross-feature-determinant-v2',
        'source_fingerprint':'a'*64,'cohorts':[]}
    audit = {'status':'passed','protocol':PROTOCOL,'checks':123,'archive_decodes':456,'per_master':[],
        'dataset':{'codename':'TEMPO-3','recipe':'structured-hard-timing-v1','designed_complexity_level':4,'maximum':5,
                   'rule_id':'observed-cross-feature-determinant-v2'},
        'source':{'source_fingerprint':'a'*64,'human_card_sha256':CARD_SHA,'information_sha256':INFO_PINS[1],
                  'reader_sha256':'b'*64,'inventory_sha256':'c'*64,'admission_log_sha256':'d'*64},
        'validated_metadata':{'report_sha256':'e'*64,'complete_sha256':'f'*64}}
    for ci, master in enumerate(MASTERS):
        methods, checked_methods = [], []
        for mi, (method, width) in enumerate(zip(METHODS, WIDTHS)):
            item = {'method':method,'size':width,'status':'measured','reason':'',
                    'outer_train_normalizer_fits':int(mi not in (0,2)), 'repetitions':[]}
            checked = {'method':method,'size':width,'status':'measured','repetitions':[]}
            for view in ('training', *VIEWS):
                total = 256 if view == 'training' else 128
                item[view+'_population'] = population(total,total)
            for ri, rep in enumerate(REPETITIONS):
                record = {'id':rep,'status':'measured','actual_probe_seed_decimal':str(1000+width+ri)}
                replay = {'repetition':rep,'actual_probe_seed_decimal':record['actual_probe_seed_decimal']}
                for view in ('training',*VIEWS):
                    total = 256 if view == 'training' else 128
                    correct = min(total, 100+mi+ci+ri) if mi < 5 else (80+ci+ri if mi == 5 else 90+ci+ri)
                    result = counts(total,total,correct)
                    record[view] = {'population':item[view+'_population'],'ridge':result,'tiny_secondary':copy.deepcopy(result)}
                    replay[view] = {'ridge':{k:v for k,v in result.items() if k!='full_population_correctness'},
                                    'tiny_secondary':{k:v for k,v in result.items() if k!='full_population_correctness'}}
                    if view != 'training':
                        for head, field in [('ridge','ridge_grouped_interval'),('tiny_secondary','tiny_grouped_interval')]:
                            record[view][field] = interval(result['accuracy'])
                            replay[view][head]['grouped_interval'] = interval(result['accuracy'])
                item['repetitions'].append(record); checked['repetitions'].append(replay)
            methods.append(item); checked_methods.append(checked)
        analytic = {}
        for view in ('training',*VIEWS):
            total = 256 if view == 'training' else 128
            valid = total if view != 'validation_deleted' else 120+ci
            analytic[view] = {'fitted_heads':0,'observed_only':True,'total_examples':total,'valid_examples':valid,
                'correct_supported':valid,'accuracy':1.,'coverage':valid/total,'full_population_correctness':valid/total}
        pairs = [{'id':'native_early_minus_native_late','reference':'native_late','candidate':'native_early',
                  'repetition':rep,'view':view,'status':'measured','common_population':population(128,128),
                  'ridge':interval(10/128),'tiny_secondary':interval(10/128)}
                 for rep in REPETITIONS for view in VIEWS]
        task = {'task':'lag_sign','readouts':{'methods':methods,'pairs':pairs},'analytic_solvability':analytic}
        points, encoders = [], []
        for placement, tag in enumerate(('RPB-v7.alt-05','RPB-v10.alt-05')):
            progress = {'attempted':512,'completed':512,'sampled_rows':4096,'parameter_count':225805,'cuda_parameter_count':225805,
                'last_input_cuda':True,'last_loss_cuda':True,'finite_gradients':True,'weights_changed':True,
                'training_seconds':10.+ci+placement,'losses':[[i,i,8,.1,.2] for i in range(1,513)]}
            query = {'standardized_mae':.07-.01*placement}
            points.append({'model_tag':tag,'placement':placement,'encoder_progress':progress,
                           'training_reconstruction':query,'validation_reconstruction':copy.deepcopy(query)})
            encoders.append({'tag':tag,'placement':placement,'budget':512,'attempted':512,'completed':512,'sampled_rows':4096,
                'training_seconds':progress['training_seconds'],'fixed_query':{'training':query,'validation':copy.deepcopy(query)}})
        costs = {'generation_and_observation_io_seconds':2.,'binding_and_checkpoint_io_seconds':3.,
            'CUDA_query_transfer_verification_io_seconds':4.,'CUDA_native_transfer_verification_seconds':5.,
            'CPU_baseline_preparation_io_seconds':6.,'CPU_head_bootstrap_io_seconds':7.,'CPU_analytic_observed_only_and_asset_IO_seconds':8.}
        experiment['cohorts'].append({'timing_master':master,'encoder_points':points,'tasks':[task],'costs':costs})
        audit['per_master'].append({'timing_master':master,'encoders':encoders,'tasks':[{'task':'lag_sign','data_master':master,
            'methods':checked_methods,'pairs':copy.deepcopy(pairs),'analytic_solvability':copy.deepcopy(analytic)}],'costs':costs})
    information = []
    for index in (0,1):
        seeds = [80383,81484] if not index else [82585,83686]
        valid = [950,900] if not index else [1020,1024]
        cohorts = []
        for master, retained in zip(seeds,valid):
            intact = dict(counts(1024,1024,1024),passed=True)
            deleted = dict(counts(1024,retained,retained),passed=bool(index))
            cohorts.append({'seed':master,'training_source_pairs':512,'validation_source_pairs':512,'intact':intact,'deleted':deleted})
        info = {'protocol':f'structured-hard-timing-information-v{index+1}','dataset_id':'TEMPO-3','TEST_generated':False,
                'passed':bool(index),'seeds':cohorts}
        receipt = {'archive_reads':0,'encoder_or_head_fits':0,'quality_generated':False,'passed':bool(index),'information':info}
        information.append({'protocol':info['protocol'],'dataset_id':'TEMPO-3','recipe_id':'structured-hard-timing-v1',
            'source_record_sha256':INFO_PINS[index],'information_receipt':receipt})
    return audit, experiment, complete, *information


def self_test():
    fixture = artificial_metadata()
    result = summarize(*fixture)
    require(result['quality']['validation_intact'][-2]['ridge_mean'] == 83/128, 'exact equal-master/equal-repetition mean fixture')
    require(result['quality']['validation_deleted'][-1]['ridge_range'] == [91/128,95/128], 'all five cohort range fixture')
    require(result['training'][0]['training_loop_seconds'] == 12., 'descriptive mean timer fixture')
    report = markdown(result)
    require(report.count('Dataset: **TEMPO-3**') == sum(line.startswith('| --- ') for line in report.splitlines()),
            'every table has its own dataset legend')
    require('same-feature v1' in report and 'cross-feature v2' in report and '973' not in report,
            'separate premodel admissions present without classifier-column substitution')
    unsupported = copy.deepcopy(fixture)
    for cohort, checked in zip(unsupported[1]['cohorts'],unsupported[0]['per_master']):
        saved = cohort['tasks'][0]['readouts']['methods'][2]; replay = checked['tasks'][0]['methods'][2]
        saved['status'] = replay['status'] = 'unsupported_fit'; saved['reason'] = 'artificial TRAIN rank failure'
        saved['repetitions'] = [{'id':rep,'status':'unsupported_fit'} for rep in REPETITIONS]; replay['repetitions'] = []
    unsupported[2]['head_pipelines'] = 90; unsupported[2]['individual_heads'] = 180
    reduced = summarize(*unsupported)
    require(reduced['quality'][VIEWS[0]][2]['ridge_mean'] is None and
            reduced['quality'][VIEWS[0]][2]['ridge_defined_cohorts'] == 0, 'unsupported TRAIN fits remain undefined')
    zero = copy.deepcopy(fixture)
    saved = zero[1]['cohorts'][0]['tasks'][0]['readouts']['methods'][-1]
    checked = zero[0]['per_master'][0]['tasks'][0]['methods'][-1]
    saved[VIEWS[1]+'_population'] = dict(saved[VIEWS[1]+'_population'],valid_rows=0,coverage=0.)
    for rep, replay in zip(saved['repetitions'],checked['repetitions']):
        rep[VIEWS[1]]['population'] = saved[VIEWS[1]+'_population']
        for head, field in [('ridge','ridge_grouped_interval'),('tiny_secondary','tiny_grouped_interval')]:
            rep[VIEWS[1]][head].update(valid=0,correct=0,accuracy=None,coverage=0.,full_population_correctness=0.)
            replay[VIEWS[1]][head].update(valid=0,correct=0,accuracy=None,coverage=0.,grouped_interval={'source_groups':0,'estimate':None,'lower':None,'upper':None})
            rep[VIEWS[1]][field].update(source_groups=0,estimate=None,lower=None,upper=None)
    abstained = summarize(*zero)
    require(abstained['quality'][VIEWS[1]][-1]['ridge_mean'] is None and
            abstained['quality'][VIEWS[1]][-1]['ridge_defined_cohorts'] == 4, 'one zero-support cohort never dropped')
    query_null = copy.deepcopy(fixture)
    query_null[1]['cohorts'][0]['encoder_points'][0]['training_reconstruction']['standardized_mae'] = None
    query_null[0]['per_master'][0]['encoders'][0]['fixed_query']['training']['standardized_mae'] = None
    require(summarize(*query_null)['training'][0]['train_mae'] is None, 'undefined query support does not invent a mean')
    negatives = [
        lambda x: x[0].update(status='failed'), lambda x:x[1].update(protocol='wrong'),
        lambda x:x[1].update(dataset_id='TEMPO-2'), lambda x:x[1].update(source_fingerprint='f'*64),
        lambda x:x[0]['source'].update(human_card_sha256='0'*64),
        lambda x:x[1]['cohorts'].reverse(), lambda x:x[2].update(full_native_export_calls=59),
        lambda x:x[2].update(individual_heads=209), lambda x:x[2].update(CPU_encoder_forward=True),
        lambda x:x[2].update(helper_outer_train_fits=24),
        lambda x:x[1]['cohorts'][0]['tasks'][0]['readouts']['methods'].reverse(),
        lambda x:x[0]['per_master'][0]['tasks'][0]['pairs'].clear(),
        lambda x:x[0]['per_master'][0]['encoders'][0].update(training_seconds=999.),
        lambda x:x[1]['cohorts'][0]['encoder_points'][0]['encoder_progress']['losses'][0].__setitem__(0,2),
        lambda x:x[1]['cohorts'][0]['tasks'][0]['readouts']['methods'][0]['repetitions'][0][VIEWS[0]]['ridge'].update(correct=0),
        lambda x:x[1]['cohorts'][0]['tasks'][0]['readouts']['methods'][0]['repetitions'][0][VIEWS[0]]['ridge_grouped_interval'].update(lower=0.),
        lambda x:x[4]['information_receipt']['information']['seeds'][0]['deleted'].update(coverage=.5),
        lambda x:x[3].update(source_record_sha256='0'*64)]
    for mutation in negatives:
        bad = copy.deepcopy(fixture); mutation(bad)
        try: summarize(*bad)
        except (ValueError, KeyError, TypeError): pass
        else: raise ValueError('negative artificial metadata fixture accepted')
    return {'status':'passed','checks':CHECKS,'negative_cases':len(negatives),'real_metadata_reads':0,
            'archive_reads':0,'model_or_head_execution':False,'SOURCE_only':True}


if __name__ == '__main__':
    main()

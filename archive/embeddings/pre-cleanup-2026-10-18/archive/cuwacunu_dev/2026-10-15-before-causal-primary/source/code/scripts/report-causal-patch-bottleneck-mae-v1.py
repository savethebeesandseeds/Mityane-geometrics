#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Report admitted frozen CPB-v1 proof and saved fixed-head scalars; stdlib only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path('/cuwacunu')
PROTOCOL = 'causal-patch-bottleneck-mae-v1'
TAG = 'CPB-v1.market-01'
CARD = '3b059b84a60a7d193ffcb034480f54fa5aba532ed6cb6b541fa620c4ab584d5f'
ADDENDUM = 'ae0a8adae317a7b2daee3f58a645c9000faab130cde17ba6662a66fae06047dd'
SOURCE = 'c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb'
FIXED = '5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14'
MARKET_MANIFEST = '828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8'
SYNTHETIC_MANIFEST = 'fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb'
OLD_MARKET = 'b73008d6eb2796570de60914365582a916237348958a54156f9745b2666cb46b'
OLD_SYNTHETIC = '2e1a3561127f4fe01b1f57834f10c0b90af671fc568b76fc9b27fe66e5e008cf'
REPS = (2701, 2802, 2903)
SEEDS = (13088411117524848790, 10018485049408248344, 9083751155097190455)
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
MODES = ('synthetic_train', 'market_scaler')
MASTERS = (941101, 941102, 941103)
VIEWS = ('validation_intact', 'validation_deleted')
HEADS = ('ridge', 'tiny_secondary')
MARKET_FLOORS = {'past_volatility': (.6781069298378846, .3926141918082198),
                 'past_trend': (.6549112553681312, .1342955201053261),
                 'past_activity': (.7514296819362523, .5631529010630465),
                 'past_relation': (.3406853885730756, .3495834881128966)}
SYNTHETIC_FLOORS = {'synthetic_train': {'validation_intact': (.9895833333333334, .9774305555555556),
                                     'validation_deleted': (.93359375, .9253472222222222)},
                    'market_scaler': {'validation_intact': (1., .9952256944444444),
                                      'validation_deleted': (.9609375, .9505208333333334)}}
DEPS = {
    'include/embedding/shared/types.h': '3c7987b9df5d936e009da619474af7ec08630083721281efb5523492200e2371',
    'include/embedding/shared/data.h': 'b9a7b80f1df93f541d05453ebc6e6abde5de1a2144f8d5e03ad5f32a068cb92e',
    'include/embedding/shared/evaluation.h': '63b7064fec06d3e33ccca7e32b20ed604c7e97550a79117ff9bbf6b93283a270',
    'include/embedding/shared/feature_harness.h': '0823c7a8e72c55e9907a4d89b076393d6356cb495122c23c78a3086ba9062657',
    'include/embedding/shared/fixed_feature_readouts.h': 'aea837bf3129044e2921695c4770082f07ae2f4ec7508b1ffeef535dfd52ce79',
    'src/data.cpp': '2a5b1b19baa0f67fac9020d91cd2883a35f7448a3e2fedb3d430b5928800043c',
    'src/feature_harness.cpp': '8cb2aaa6c2b0d6582441fe51fe04445479e15824a44003cf59b97a8f8a67abbc',
    'src/fixed_feature_readouts.cpp': '1de2800e99c52a3df6b2bc168cd7b4c900888162e0056d1c681c2e74c3396b2c',
}
MKT = ('Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute history, recent128 '
       'past-window targets · **complexity4/5** · known chronological VALIDATION32, TRAIN112.')
T5 = ('Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features '
      'and structured distractors · **complexity5/5**.')
CHECKS = 0


def require(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def path_at(name):
    p = Path(name)
    return p if p.is_absolute() else ROOT / p


def asset(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def dump(path, value):
    save(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def verify(pins):
    pairs = ((pin['path'], pin) for pin in pins) if isinstance(pins, list) else pins.items()
    for name, pin in pairs:
        path = path_at(name)
        require(path.is_file() and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                'Protected file changed: ' + name)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual, expected, why):
    require(finite(actual) and finite(expected) and abs(actual - expected) <= 1e-12, why)


def stream_seed(repetition, width):
    mask = (1 << 64) - 1
    value = (repetition + 0x9e3779b97f4a7c15 * width) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return str(value ^ (value >> 31))


def named(value):
    result = 14695981039346656037
    for byte in value.encode('ascii'):
        result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return result


def validate_proof(path):
    proof = load(path)
    require(proof['kind'] == 'causal_patch_bottleneck_mae_v1_cuda_proof_v1' and
            proof['status'] == 'passed' and proof['model'] == 'CPB-v1' and
            proof['source_checkpoint_sha256'] == SOURCE and proof['device'] == 'cuda:0', 'Admitted exact causal CUDA proof')
    for key, expected in {'source_updates': 2048, 'causal_updates': 0, 'encoder_updates': 0,
                          'source_parameter_tensors_exact': 76, 'source_parameter_count_exact': 297560,
                          'all_cutoffs_tested': 128, 'suffix_intervention_variants': 9, 'suffix_layouts': 2,
                          'suffix_invariance_cases': 2350, 'bounded_witness_cases': 20,
                          'rolling_history_length_tested': 193, 'mixed_cutoff_plans': 5,
                          'internal_temporal_cases': 30, 'CPU_encoder_calls': 0, 'head_fits': 0, 'decoder_calls': 0}.items():
        require(proof[key] == expected, 'Causal proof scope/count: ' + key)
    require(proof['cutoff_indices'] == list(range(128)) and proof['suffix_invariance_cases'] > 0 and
            proof['typed_negative_cases'] > 0 and proof['source_tensor_leaves_exact'] > 0 and
            proof['CUDA_encoder_calls'] > 0 and proof['checks'] > 0, 'Actual full-cutoff/nonvacuous proof accounting')
    require(proof['model_dtype'] == 'float32' and proof['checkpoint'] == 'inference.pt' and
            proof['causal_policy'] == 'crop-before-validation;rolling-last-H;masked-left-pad;original-patch-end-causal-attention;prefix-only-banks-pooling-v1',
            'Exact public causal dtype/policy/artifact')
    require(all(finite(proof[key]) and proof[key] >= 0 for key in
                ('import_seconds_including_state_and_io', 'proof_seconds_including_state_checks',
                 'artifact_and_typed_check_seconds')), 'Actual separate proof timing fields')
    for key in ('weights_imported_exact', 'typed_save_load_exact', 'ordinary_loader_rejects_RPB',
                'future_values_and_masks_invariant', 'observed_future_NaN_Inf_overflow_ignored',
                'hidden_prefix_poison_ignored', 'observed_prefix_poison_rejected', 'both_temporal_paths_causal',
                'legal_past_value_positive_control', 'legal_past_mask_positive_control', 'all_state_exact', 'undefined_gradients'):
        require(proof[key] is True, 'Required causal proof: ' + key)
    for name in ('inference.pt', 'proof-witnesses.json', 'frozen-state.json', 'import-parity.json'):
        require((path.parent / name).is_file(), 'Necessary compact proof artifact: ' + name)
    return proof


def market_rows(run):
    path = run / 'market-readouts/report.json'
    report = load(path)
    require(report['kind'] == 'fixed_past_window_regression_v1' and report['dataset_id'] == 'MKT-HIST-2' and
            report['card_sha256'] == CARD and report['input_manifest_sha256'] == MARKET_MANIFEST and
            (report['training_rows'], report['validation_rows']) == (112, 32), 'Market evaluator identity/split')
    expected = {'ridge_penalty': 1, 'neural_hidden': 16, 'neural_learning_rate': .01, 'neural_updates': 100,
                'outputs': 4, 'loss': 'masked TRAIN-target-standardized MSE', 'device': 'CPU (frozen readouts only)'}
    require(all(report['heads'][key] == value for key, value in expected.items()), 'Unchanged market head recipe')
    require(len(report['methods']) == 1 and report['methods'][0]['name'] == 'cpb_v1', 'Only new causal market features scored')
    method = report['methods'][0]
    require((method['size'], method['ridge_parameters'], method['neural_parameters'], method['ridge_fits'],
             method['ridge_task_fits'], method['neural_fits'], method['training_feature_rows'], method['feature_training_supported']) ==
            (36, 148, 660, 1, 4, 3, 112, True), 'Fixed D36 market fit budget/support')
    for key, role in (('training', 'train'), ('validation', 'validation')):
        require(path_at(method[key + '_path']).resolve() == run / 'features/market' / ('cpb_v1-' + role + '.pt'),
                'New market head input path')
    parent = load(ROOT / 'output/runs/market-capacity-study/run-base-gryiAV/readouts/report.json')['methods'][0]['repetitions'][0]['targets']
    require(tuple(x['repetition'] for x in method['repetitions']) == REPS and
            tuple(x['actual_seed'] for x in method['repetitions']) == SEEDS, 'D36 fixed market seeds')
    coverage = True
    for index, rep in enumerate(method['repetitions']):
        require(rep['neural_updates_completed'] == 100 and rep['ridge_fit_reused'] == (index != 0), 'Market fit reuse/updates')
        require(tuple(x['name'] for x in rep['targets']) == TASKS, 'Same four market targets')
        for target, old in zip(rep['targets'], parent):
            require(target['train_target_eligible'] == 112 and target['train_target_supported'] and
                    target['method_training_supported'] and all(target[k] == old[k] for k in
                    ('name', 'unit', 'train_target_mean', 'train_target_scale')), 'Exact TRAIN market target support/scaler')
            for head in ('ridge', 'neural'):
                m = target[head]
                require(m['total'] == m['target_eligible'] == 32 and 0 < m['valid'] <= 32 and
                        all(finite(m[k]) for k in ('r2', 'mae', 'rmse')), 'Finite market support/physical scores')
                close(m['coverage'], m['valid'] / 32, 'Market coverage arithmetic')
                coverage &= m['valid'] == 32
    rows = []
    for index, task in enumerate(TASKS):
        targets = [x['targets'][index] for x in method['repetitions']]
        ridge = targets[0]['ridge']
        require(all(x['ridge'] == ridge for x in targets), 'One ridge pipeline reused across neural repetitions')
        row = {'dataset_id': 'MKT-HIST-2', 'task': task, 'recipe_id': PROTOCOL, 'designed_complexity_level': 4,
               'complexity_scale_max': 5, 'evidence_stage': 'known-source development; zero-update causal conversion',
               'view': 'intact chronological VALIDATION', 'method': TAG, 'source_method': 'cpb_v1', 'size': 36,
               'reused': False, 'new_head_measurement': True, 'linear_r2': ridge['r2'], 'linear_mae': ridge['mae'],
               'linear_rmse': ridge['rmse'], 'unit': targets[0]['unit'], 'valid': ridge['valid'], 'total': 32,
               'coverage': min(x[h]['coverage'] for x in targets for h in ('ridge', 'neural')),
               'report': asset(path), 'repetitions': [{'repetition': r['repetition'], 'actual_seed': r['actual_seed'],
               'neural_updates_completed': r['neural_updates_completed'], 'target': r['targets'][index]} for r in method['repetitions']]}
        for metric in ('r2', 'mae', 'rmse'):
            row['neural_' + metric + '_each'] = [x['neural'][metric] for x in targets]
            row['neural_mean_' + metric] = statistics.mean(row['neural_' + metric + '_each'])
        rows.append(row)
    return rows, coverage


def market_decisions(old, new, coverage):
    lookup = {(x['method'], x['task']): x for x in old + new}
    trend = lookup[TAG, 'past_trend']
    original, strict = [], []
    for key in ('linear_r2', 'neural_mean_r2'):
        value = trend[key]
        original.append({'criterion': 'positive_' + key + '_trend', 'candidate': value, 'floor': 0., 'passed': value > 0.})
    for task in ('past_volatility', 'past_activity', 'past_relation'):
        for key in ('linear_r2', 'neural_mean_r2'):
            base, value = lookup['RPB-v22.market-02', task][key], lookup[TAG, task][key]
            original.append({'criterion': task + '_' + key + '_v22_retention', 'candidate': value,
                             'baseline': base, 'floor': base - .05, 'passed': value + 1e-12 >= base - .05})
    base = lookup['RPB-v23.market-01', 'past_trend']['linear_r2']
    require(base == .3871309972495327, 'Saved original ninth trend reference')
    original.append({'criterion': 'linear_trend_exceeds_saved_v23', 'candidate': trend['linear_r2'],
                     'floor': base, 'passed': trend['linear_r2'] + 1e-12 > base})
    for task in TASKS:
        for key, floor in zip(('linear_r2', 'neural_mean_r2'), MARKET_FLOORS[task]):
            value = lookup[TAG, task][key]
            strict.append({'criterion': task + '_' + key + '_native36_guard', 'candidate': value,
                           'floor': floor, 'passed': value + 1e-12 >= floor})
    individual = [{'repetition': rep, 'candidate': value, 'floor': 0., 'passed': value > 0.}
                  for rep, value in zip(REPS, trend['neural_r2_each'])]
    return {'original_criteria': original, 'strict_native36_criteria': strict, 'individual_neural_trend_criteria': individual,
            'original_passed': sum(x['passed'] for x in original), 'original_total': 9,
            'strict_passed': sum(x['passed'] for x in strict), 'strict_total': 8, 'full_coverage': coverage,
            'all_three_neural_trend_positive': all(x['passed'] for x in individual),
            'full_market_comparison_gate_passed': coverage and all(x['passed'] for x in original + strict + individual)}


def validate_classification(report, master):
    require(report['protocol'] == 'fixed-feature-readouts-v1' and int(report['master_seed']) == master,
            'Unchanged synthetic evaluator/cohort')
    for key, value in {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100, 'tiny_learning_rate': .01,
                       'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0,
                       'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods'}.items():
        require(report['recipe'][key] == value, 'Unchanged synthetic head recipe: ' + key)


def validate_classification_method(method, width, master):
    require(method['size'] == width and method['status'] == 'measured' and
            method['inputs_train_prepared'] is False and method['outer_train_normalizer_fits'] == 1,
            'Direct native width/TRAIN-only classification scaling')
    require(len(method['repetitions']) == 3, 'Three fixed classification fits')
    for rep, repetition in zip(method['repetitions'], REPS):
        require(rep['id'] == 'rep-' + str(repetition) and rep['status'] == 'measured' and
                rep['actual_probe_seed_decimal'] == stream_seed(repetition, width) and
                rep['ridge_parameters'] == 2 * width + 2 and rep['neural_parameters'] == 16 * (width + 3) + 2,
                'Classification fit/width-derived seed/head parameters')
        for view in ('training',) + VIEWS:
            value = rep[view]
            rows, groups = (512, 256) if view == 'training' else (256, 128)
            population = value['population']
            require(population['total_rows'] == rows and population['total_source_groups'] == groups and
                    0 < population['valid_rows'] <= rows, 'Classification source/row population')
            close(population['coverage'], population['valid_rows'] / rows, 'Classification population coverage')
            for head in HEADS:
                score = value[head]
                require(score['total'] == rows and score['valid'] == population['valid_rows'] and
                        isinstance(score['correct'], int) and 0 <= score['correct'] <= score['valid'], 'Correct/valid counts')
                close(score['accuracy'], score['correct'] / score['valid'], 'Classification saved accuracy')
                close(score['coverage'], score['valid'] / rows, 'Classification head coverage')
                close(score['full_population_correctness'], score['correct'] / rows, 'Full-population correctness')
            if view in VIEWS:
                if method['method'] in MODES:
                    close_seed = stream_seed(master, named(method['method'] + '/' + rep['id'] + '/' + view.replace('_', '-')))
                    require(value['bootstrap_seed_decimal'] == close_seed, 'Unchanged source-group bootstrap seed')
                for head, key in (('ridge', 'ridge_grouped_interval'), ('tiny_secondary', 'tiny_grouped_interval')):
                    ci = value[key]
                    require(ci['replicates'] == 1000 and ci['confidence'] == .95 and ci['source_groups'] == groups and
                            ci['method'] == 'source-group percentile bootstrap; within-master conditional on fitted readout' and
                            0 <= ci['lower'] <= ci['upper'] <= 1, 'Conditional source-group interval recipe')
                    close(ci['estimate'], value[head]['accuracy'], 'Conditional interval estimate')


def aggregate(methods):
    result = {}
    for view in VIEWS:
        result[view] = {}
        for head in HEADS:
            means = [statistics.mean(r[view][head]['accuracy'] for r in m['repetitions']) for m in methods]
            result[view][head] = {'mean': statistics.mean(means), 'master_means': means, 'minimum': min(means), 'maximum': max(means)}
        means = [statistics.mean(r[view]['population']['coverage'] for r in m['repetitions']) for m in methods]
        result[view]['coverage'] = {'mean': statistics.mean(means), 'master_means': means}
    return result


def synthetic_rows(run, old):
    new_reports, old_reports, reference_reports = [], old['all_saved_repetitions_and_intervals']['new'], old['all_saved_repetitions_and_intervals']['reused_reference']
    require(len(old_reports) == len(reference_reports) == 3, 'Every old synthetic cohort retained')
    methods, previous = {m: [] for m in MODES}, {m: [] for m in MODES}
    refs = []
    coverage = True
    for master, older, reference in zip(MASTERS, old_reports, reference_reports):
        report = load(run / 'synthetic-readouts' / ('master-' + str(master)) / 'report.json')
        new_reports.append(report)
        for value in (report, older, reference):
            validate_classification(value, master)
        require(len(report['methods']) == 2 and {x['method'] for x in report['methods']} == set(MODES) and not report['pairs'],
                'Exactly two declared CPB scaler policies')
        require(report['fit_counts'] == {'outer_train_normalizer_fits': 2, 'ridge_fits': 6, 'tiny_fits': 6,
                                        'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'Fixed per-cohort fit budget')
        ref = next(x for x in reference['methods'] if x['method'] == 'native_512')
        validate_classification_method(ref, 32, master)
        refs.append(ref)
        for mode in MODES:
            method = next(x for x in report['methods'] if x['method'] == mode)
            older_method = next(x for x in older['methods'] if x['method'] == mode)
            validate_classification_method(method, 36, master)
            validate_classification_method(older_method, 36, master)
            methods[mode].append(method)
            previous[mode].append(older_method)
            counts = load(run / 'synthetic-readouts' / ('master-' + str(master)) / mode / 'fit-counts.json')
            require(all(counts[k] == v for k, v in {'outer_train_normalizer_fits': 1, 'ridge_fits': 3,
                                                   'tiny_fits': 3, 'validation_fits': 0}.items()), 'Saved per-policy fit counts')
            for rep in method['repetitions']:
                for view in ('training',) + VIEWS:
                    value = rep[view]
                    coverage &= (value['population']['valid_rows'] == value['population']['total_rows'] and
                                 value['population']['valid_source_groups'] == value['population']['total_source_groups'] and
                                 all(value[h]['coverage'] == 1 for h in HEADS))
                leaf = run / 'synthetic-readouts' / ('master-' + str(master)) / mode / rep['id']
                for name in ('fit.pt', 'training-predictions.pt', 'validation-intact-predictions.pt', 'validation-deleted-predictions.pt'):
                    require((leaf / name).is_file(), 'Saved new synthetic head/prediction artifact')
    scores = {m: aggregate(methods[m]) for m in MODES}
    old_scores = {m: aggregate(previous[m]) for m in MODES}
    reference = aggregate(refs)
    gates = []
    for mode in MODES:
        for view in VIEWS:
            for head, floor in zip(HEADS, SYNTHETIC_FLOORS[mode][view]):
                close(old_scores[mode][view][head]['mean'], old['new_scores'][mode][view][head]['mean'], 'Saved native36 aggregation')
                close(old_scores[mode][view][head]['mean'], floor, 'Frozen native36 floor')
                value = scores[mode][view][head]['mean']
                gates.append({'mode': mode, 'view': view, 'head': head, 'mean': value, 'floor': floor,
                              'difference': value - floor, 'passed': value + 1e-12 >= floor, 'arithmetic_tolerance': 1e-12})
    panels = []
    for mode in MODES:
        for view in VIEWS:
            panels.append({'dataset_id': 'TEMPO-5', 'task': 'lag_sign', 'recipe_id': 'tempo-5-slow-fast-nine-feature-v1',
                           'designed_complexity_level': 5, 'complexity_scale_max': 5, 'view': view, 'input_policy': mode,
                           'head_fitting_dataset_id': 'TEMPO-5', 'scoring_dataset_id': 'TEMPO-5',
                           'rows': [{'method': tag, 'size': width, 'evidence': use, 'encoder_fitting_dataset_id': fitting,
                                     'linear_head_accuracy': data[view]['ridge']['mean'], 'neural_head_accuracy': data[view]['tiny_secondary']['mean'],
                                     'coverage': data[view]['coverage']['mean']} for tag, width, use, fitting, data in
                                    (('RPB-v22.alt-01', 32, 'reused synthetic-trained control', 'TEMPO-5', reference),
                                     ('RPB-v24.market-01.alt-01', 36, 'reused native36 transfer', 'MKT-HIST-2', old_scores[mode]),
                                     (TAG, 36, 'new zero-update causal execution', 'MKT-HIST-2', scores[mode]))]})
    return {'scores': scores, 'native36_scores': old_scores, 'synthetic_reference_scores': reference, 'panels': panels,
            'retention_conditions': gates, 'retention_passed': sum(x['passed'] for x in gates), 'retention_total': 8,
            'full_coverage': coverage, 'full_synthetic_comparison_gate_passed': coverage and all(x['passed'] for x in gates),
            'all_saved_repetitions_and_intervals': {'new': new_reports, 'native36': old_reports, 'synthetic_reference': reference_reports}}, methods, previous, refs


def add_table(lines, legend, headers, rows):
    lines += [legend, '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines += ['| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |' for row in rows]
    lines += ['']


def pct(value):
    return f'{100 * value:.2f}'


def repetitions(lines, title, methods):
    rows = []
    for master, method in zip(MASTERS, methods):
        for rep in method['repetitions']:
            for view in VIEWS:
                value = rep[view]
                l, n = value['ridge_grouped_interval'], value['tiny_grouped_interval']
                rows.append((master, rep['id'][4:], view, f"{pct(l['estimate'])} [{pct(l['lower'])},{pct(l['upper'])}]",
                             f"{pct(n['estimate'])} [{pct(n['lower'])},{pct(n['upper'])}]", pct(value['population']['coverage'])))
    lines += [title, '']
    add_table(lines, T5 + ' Known intact/deleted VALIDATION; every saved fit.',
              ('Cohort', 'Fit', 'View', 'Linear % [95% interval]', 'Neural % [95% interval]', 'Coverage %'), rows)


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--proof-receipt', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('doc/results/causal_patch_bottleneck_mae_v1'))
    args = p.parse_args()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Existing managed cuwacunu container only')
    run = args.run.resolve(strict=True)
    proof_path = args.proof_receipt.resolve(strict=True)
    require(run.is_relative_to(ROOT / 'output/runs/causal-patch-bottleneck-mae-v1') and run.name.startswith('run-'), 'Declared one-run root')
    durable = args.output_dir.resolve()
    require(durable.is_relative_to(ROOT / 'doc/results'), 'Durable evidence location')
    durable.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md'
    for path in (report_path, durable / 'completion.json', durable / 'table_rows.json', durable / 'asset_hashes.json',
                 durable / 'durable_files.json', durable / 'report-capture'):
        require(not path.exists(), 'Preserve existing report attempt: ' + str(path))
    card = ROOT / 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_CARD.md'
    addendum = ROOT / 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_LAYOUT_ADDENDUM.md'
    old_market_path = ROOT / 'doc/results/market_location_scale_decoder_v1/table_rows.json'
    old_synthetic_path = ROOT / 'doc/results/market_synthetic_transfer_v1/table_rows.json'
    for path, digest in ((card, CARD), (addendum, ADDENDUM), (old_market_path, OLD_MARKET), (old_synthetic_path, OLD_SYNTHETIC),
                         (Path('/opt/cuwacunu/build/market-study/market_fixed_regression'), FIXED),
                         (ROOT / 'output/inputs/market-synthetic-transfer-v1/manifest.json', SYNTHETIC_MANIFEST)):
        require(sha(path) == digest, 'Prospective/evaluator/reference identity: ' + str(path))
    preflight = load(durable / 'preflight.json')
    require(preflight['status'] == 'passed' and preflight['checkpoints_verified'] == 55 and
            preflight['portable_pairs_verified'] == 99 and preflight['frozen_document_bindings'] == 44,
            'Bounded old checkpoint/source/result preservation')
    proof = validate_proof(proof_path)
    checkpoint = proof_path.parent / 'inference.pt'
    execution = load(run / 'verification.json')
    audit_path = path_at(execution['independent_audit_path'])
    audit = load(audit_path)
    admission_path = path_at(execution['admission_path'])
    admission = load(admission_path)
    engineering = proof_path.parent.parent
    require(admission_path.resolve() == engineering / 'admission.json' and
            execution['status'] == audit['status'] == 'passed' and
            execution['kind'] == 'causal_patch_bottleneck_mae_v1_verification_v1' and
            admission['status'] == 'built_pending_source_review_and_cuda_proof', 'Build/proof/execution audit stages')
    require(load(engineering / 'source-review.json')['status'] ==
            load(engineering / 'proof-stage.json')['status'] == 'passed', 'Independent source and actual proof admission')
    require(path_at(execution['proof_path']).resolve() == proof_path and execution['proof_sha256'] == sha(proof_path) and
            path_at(execution['causal_checkpoint_path']).resolve() == checkpoint and
            execution['causal_checkpoint_sha256'] == sha(checkpoint), 'Quality uses exactly the admitted proof/import artifact')
    require(execution['admission_sha256'] == sha(admission_path) and execution['independent_audit_sha256'] == sha(audit_path),
            'Exact independent verification/admission bindings')
    for field in ('source', 'pins', 'binaries', 'reused_fixed_head_objects'):
        verify(admission[field])
    manifest_pin = execution['source']
    require(sha(path_at(manifest_pin['path'])) == manifest_pin['sha256'] and
            load(path_at(manifest_pin['path'])) == admission['source'], 'Exact compiled source manifest')
    capsule = path_at(execution['source_capsule'])
    for pin in admission['source']:
        original = path_at(pin['path'])
        preserved = capsule / original.relative_to(ROOT)
        require(preserved.stat().st_size == pin['bytes'] and sha(preserved) == pin['sha256'],
                'Once-captured version/compiler source identity')
    for pins in admission['dependencies'].values():
        verify(pins)
    tools_path = engineering / 'quality-tools-admission.json'
    tools = load(tools_path)
    require(tools['status'] == 'passed', 'Prospective saved-quality reporting/audit tools admission')
    verify(tools['sources'])
    require(any(path_at(pin['path']).resolve() == Path(__file__).resolve() for pin in tools['sources']),
            'Reporter source admitted before quality scores')
    require(execution['preservation_after']['status'] == 'passed', 'Preexisting artifacts preserved')
    preservation = path_at(execution['preservation_after']['manifest_path'])
    require(sha(preservation) == execution['preservation_after']['manifest_sha256'], 'Preservation manifest binding')
    summary = load(run / 'features/summary.json')
    require(summary['protocol'] == PROTOCOL and summary['model'] == TAG and summary['device'] == 'cuda:0' and
            summary['batch_size'] == 8 and summary['native_width'] == 36 and summary['full_CUDA_encode_calls'] == 786 and
            summary['synthetic_rows'] == 6144, 'Single new causal extraction scope/calls')
    for key in ('encoder_updates', 'decoder_calls', 'old_encoder_calls', 'new_input_scaler_fits', 'new_bank_normalizer_fits'):
        require(summary[key] == 0, 'Forbidden new extraction work: ' + key)
    for key in ('all_parameters_buffers_normalizer_scaler_unchanged', 'all_gradients_undefined',
                'no_labels_in_encoder', 'all_outputs_at_last_observed_grid_timestamp'):
        require(summary[key] is True, 'Frozen causal extraction invariant: ' + key)
    for role, rows, calls in (('train', 112, 14), ('validation', 32, 4)):
        value = summary['market'][role]
        require(value['rows'] == value['valid'] == rows and value['full_CUDA_encode_calls'] == calls, 'Market endpoint export/support')
    require(len(summary['synthetic']) == 6 and {(x['mode'], x['master']) for x in summary['synthetic']} ==
            {(mode, master) for mode in MODES for master in MASTERS}, 'Both synthetic policies/all cohorts')
    for cohort in summary['synthetic']:
        require(len(cohort['roles']) == 3 and {x['role'] for x in cohort['roles']} == {'training', 'validation', 'deleted'},
                'All synthetic roles')
        for role in cohort['roles']:
            rows = 512 if role['role'] == 'training' else 256
            require(role['capture']['rows'] == role['capture']['valid'] == rows and
                    role['capture']['full_CUDA_encode_calls'] == rows // 8, 'Synthetic export/support/call budget')
    old_market = load(old_market_path)
    require(len(old_market['rows']) == 96, 'All frozen historical market task rows preserved')
    market, market_coverage = market_rows(run)
    market_gates = market_decisions(old_market['rows'], market, market_coverage)
    synthetic, methods, previous, refs = synthetic_rows(run, load(old_synthetic_path))
    reader = load(run / 'synthetic-readouts/reader-summary.json')
    require(reader['model'] == TAG and reader['protocol'] == PROTOCOL and reader['native_width'] == 36 and
            reader['primary_input_mode'] == MODES[0] and reader['secondary_input_mode'] == MODES[1], 'CPB fixed synthetic reader')
    for key in ('encoder_calls', 'encoder_updates', 'old_encoder_calls', 'old_head_fits', 'post_encoder_pca_fits'):
        require(reader[key] == 0, 'Forbidden reader work: ' + key)
    dependency_pins = []
    for relative, digest in DEPS.items():
        path = ROOT / 'code/evaluation/market_synthetic_transfer_v1/deps' / relative
        require(sha(path) == digest, 'Byte-identical classifier algorithm: ' + relative)
        dependency_pins.append(asset(path))
    # The mandatory causal proof is a correctness gate. Quality comparisons are separate and descriptive.
    table = {'kind': 'causal_patch_bottleneck_mae_v1_results', 'protocol': PROTOCOL, 'status': 'complete',
             'model': TAG, 'causality_correctness_gate_passed': True, 'market_rows': old_market['rows'] + market,
             'historical_market_rows_reused': 96, 'new_market_rows': 4, 'historical_market_source': asset(old_market_path),
             'market_comparisons': market_gates, 'synthetic': synthetic, 'proof': proof,
             'new_encoder_training_updates': 0, 'new_scaler_or_bank_fits': 0, 'primary_changed': False,
             'balanced_selection_changed': False, 'comparison_absolute_arithmetic_tolerance': 1e-12}
    dump(durable / 'table_rows.json', table)
    version_root = ROOT / 'code/encoders/causal_patch_bottleneck_mae/versions/v1'
    version_assets = [asset(path) for path in sorted(version_root.rglob('*')) if path.is_file()]
    require(version_assets, 'Physically version-owned causal implementation')
    run_assets = [asset(path) for path in sorted(run.rglob('*')) if path.is_file()]
    proof_assets = [asset(path) for path in sorted(proof_path.parent.rglob('*')) if path.is_file()]
    dump(durable / 'asset_hashes.json', {'kind': 'causal_patch_bottleneck_mae_v1_report_assets', 'run_assets': run_assets,
              'proof_assets': proof_assets, 'version_owned_source_assets': version_assets,
              'admitted_source_assets': admission['source'], 'admitted_compiler_dependencies': admission['dependencies'],
              'admitted_binaries': admission['binaries'], 'quality_tools_admission': asset(tools_path),
              'quality_tools_sources': tools['sources'], 'eight_fixed_dependencies': dependency_pins,
              'old_market_reference': asset(old_market_path), 'old_synthetic_reference': asset(old_synthetic_path),
              'card': asset(card), 'layout_addendum': asset(addendum), 'preflight': asset(durable / 'preflight.json'),
              'checkpoint': asset(checkpoint), 'reporter': asset(Path(__file__).resolve()),
              'no_old_capsule_copy': True, 'large_arrays_and_tensors_local_only': True})
    capture = durable / 'report-capture'
    capture.mkdir()
    # Only compact new receipts/source; the once-admitted compiler capsule is bound by hash, not recopied.
    compact = [card, addendum, Path(__file__).resolve(), proof_path, audit_path, run / 'verification.json',
               run / 'features/summary.json', run / 'market-readouts/report.json',
               run / 'synthetic-readouts/reader-summary.json', admission_path, tools_path,
               engineering / 'source-review.json', engineering / 'proof-stage.json']
    for path in compact:
        require(path.stat().st_size <= 1024 * 1024, 'Compact receipt/source size bound')
        dest = capture / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(path.read_bytes())
        require(sha(dest) == sha(path), 'Compact captured receipt/source identity')
    lines = ['# CPB-v1: causal conversion and fixed-head results', '',
             'Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.', '',
             '**CPB-v1 passes the mandatory cutoff/suffix causality proof.** Its market and synthetic quality are reported separately below. '
             'No encoder training or input/bank normalization fit occurred. Fixed-head normalization fits retain their separately declared budgets. '
             'RPB-v22 stays primary and native36 stays balanced; this report changes no live pointer.', '',
             '## What changed and what the proof establishes', '',
             'CPB-v1.market-01 imports exact saved V30 weights/scaler/normalizer as data into an independently owned causal family. '
             'Its only public version entry is under `code/encoders/causal_patch_bottleneck_mae/versions/v1/include/`; that physical version owns all include/config/tests/build/provenance dependencies. '
             'There is no raw-family implementation forwarding or mutable shared encoder implementation across causal versions. The exact dependency/provenance admission is bound in the asset inventory.', '',
             'The API crops future values AND masks away before finiteness checks, F32 rounding, scaling, token/support construction or bank statistics. '
             'It retains rolling lastH128 ending at each explicit cutoff, left-pads unavailable past with hidden zeros, uses original-patch-end causal temporal attention plus padding masks, '
             'and computes every sequence z_t through its own cutoff path. Channel mixing remains same-time and normalization is over feature width. '
             'Full-window decoder helpers are not presented as an autoregressive forecast.', '',
             f"The CUDA proof tests all128 cutoff indices, nine suffix variants and two semantic layouts, including{proof['suffix_invariance_cases']} actual suffix cases, longer193 histories, "
             'five mixed-cutoff plans and all eight P8 positions. Future observed NaN/Inf/F32-overflow and altered masks are ignored; observed poison inside the retained prefix rejects and hidden prefix storage poison is ignored. '
             'Legal past-value and past-mask positive controls change features/support, preventing a constant-output proof. '
             'Thirty internal temporal cases cover both temporal paths at completed original-patch boundaries; same-cutoff suffix invariance is exact, while separate cross-layout comparisons use the declared1e-5 tolerance.', '',
             f"All{proof['source_parameter_tensors_exact']} source parameter tensors/{proof['source_parameter_count_exact']} parameter values and{proof['source_tensor_leaves_exact']} tensor leaves import exactly. "
             'Typed save/load is exact; ordinary causal loaders reject RPB kinds and malformed causal identities. Model/scaler/normalizer state remains exact, gradients undefined and encoder updates0. '
             f"Engineering performs{proof['CUDA_encoder_calls']} successful CUDA encode/trace calls and{proof['public_API_attempts']} public API attempts, including declared rejection cases. "
             'These correctness calls are separate from786 quality exports. The proof covers this admitted implementation, geometry and deterministic intervention suite; it is not a future-prediction accuracy claim.', '',
             'Causal attention intentionally changes execution, so no equality of old RPB native32/36 outputs is promised. Score differences under the joint conversion do not measure an amount of future leakage. '
             'All old results keep their historical-window semantics and are not relabeled causal.', '',
             '## Market information retention', '',
             'MKT-HIST-2 keeps112 TRAIN/32 known chronological VALIDATION probes and the same recent128 historical targets/scaler. '
             'One frozen causal export uses CUDA:0 RTX A2000 8GB, batch8,14 TRAIN+4 VALIDATION calls. '
             'TRAIN-only ridge1/four targets and neural16-tanh/Adam.01/100 updates retain repetitions2701/2802/2903 and the exact D36 seeds. '
             'D36 regression heads have148/660 parameters; one ridge pipeline/four target fits and three neural fits are new. Native embeddings have no PCA afterward. '
             'Older controls are saved-score reuse, with no old encoder/head replay. These remain past-window information-retention targets, not forecasting.']
    display = ('RPB-v22.market-02', 'RPB-v24.market-01', 'RPB-v24.market-01.alt-01', 'RPB-v25.market-01', 'RPB-v26.market-01', 'RPB-v30.market-01', TAG)
    lookup = {(x['method'], x['task']): x for x in old_market['rows'] + market}
    for task in TASKS:
        lines += ['', '### ' + task.replace('past_', '').title(), '']
        rows = [(name, lookup[name, task]['size'], f"{lookup[name, task]['linear_r2']:.6f}",
                 f"{lookup[name, task]['neural_mean_r2']:.6f}", pct(lookup[name, task]['coverage'])) for name in display]
        add_table(lines, MKT, ('Method', 'Size', 'Linear head R²', 'Neural head R²', 'Coverage %'), rows)
        row = lookup[TAG, task]
        lines += ['CPB neural R² by2701/2802/2903: ' + ', '.join(repr(x) for x in row['neural_r2_each']) +
                  f". Physical linear MAE/RMSE:{row['linear_mae']!r}/{row['linear_rmse']!r} {row['unit']}; "
                  f"mean neural MAE/RMSE:{row['neural_mean_mae']!r}/{row['neural_mean_rmse']!r}. All individual physical errors remain in JSON.", '']
    lines += [f"Original comparisons pass{market_gates['original_passed']}/9; strict native36 floors{market_gates['strict_passed']}/8; "
              f"positive neural trend fits{sum(x['passed'] for x in market_gates['individual_neural_trend_criteria'])}/3; "
              f"coverage is{'100%' if market_coverage else 'incomplete'}. These fixed comparisons do not relax the mandatory causality requirement. "
              'All full-precision decisions, including the original ninth saved-v23 trend comparison, use unchanged arithmetic tolerance1e-12.', '',
              '## Frozen hard-signal transfer', '',
              'TEMPO-5 cohorts941101/941102/941103 retain512 TRAIN/256 paired source groups and256 known VALIDATION/128 disjoint groups each. '
              'The exact saved intact and additional30%-erased masks and both scaler policies are fixed. Three cohorts are data cohorts; no new encoder trajectories were trained. '
              'The relative-bank normalizer remains its frozen market fit. The parent path rounds rawF64 to rawF32 before CPU-F64 scaling, unlike the old native36 synthetic rounding order. '
              'No favorable precision/scaler arm is selected. Saved v22 controls retain their own synthetic input pipeline and are contextual in both panels; native36 is the matched-policy retention reference.', '',
              'Six TRAIN outer-normalizer fits,18 ridge and18 neural fits are new. Binary D36 classifiers have74/626 parameters with the same width-derived seeds; D32 controls retain66/562. '
              'Each fit scores TRAIN/intact/deleted unchanged. All786 quality calls comprise18 market plus768 synthetic calls, with no decoder, CPU encode or encoder update. '
              'The interrupted V30 transfer remains extraction-only/unscored and is not used as a synthetic quality reference.', '']
    for panel in synthetic['panels']:
        lines += ['Policy `' + panel['input_policy'] + '`, view `' + panel['view'] + '`; equal three-cohort means and all neural repetitions.', '']
        add_table(lines, T5 + ' Known VALIDATION, exact saved masks.',
                  ('Method', 'Size', 'Linear head %', 'Neural head %', 'Coverage %'),
                  [(x['method'], x['size'], pct(x['linear_head_accuracy']), pct(x['neural_head_accuracy']), pct(x['coverage'])) for x in panel['rows']])
    add_table(lines, T5 + ' Both policies/intact-deleted known VALIDATION; fixed native36 floors.',
              ('Policy', 'View', 'Head', 'Exact mean', 'Frozen floor', 'Pass'),
              [(x['mode'], x['view'], x['head'], repr(x['mean']), repr(x['floor']), str(x['passed'])) for x in synthetic['retention_conditions']])
    lines += [f"Synthetic comparisons pass{synthetic['retention_passed']}/8; full role/cohort/policy/head coverage is{synthetic['full_coverage']}. "
              'There is no post-hoc individual-cohort floor. The old primary intact linear mean differs by one ULP between aggregation orders0.9895833333333333/0.9895833333333334; '
              'use the fixed floor and arithmetic tolerance only.', '', '## Every cohort, repetition and conditional interval', '',
              'Intervals retain95% source-group percentile bootstrap1000, conditional on each fitted head. They do not quantify independent encoder-training or unseen-domain uncertainty. '
              'Per-row predictions remain ignored tensors. The stdlib reporter/audit checks saved scalar arithmetic; full prediction correctness/bootstrap quantiles rely on the byte-identical eight-file evaluator and hash bindings, without tensor replay.', '']
    for mode in MODES:
        repetitions(lines, TAG + ' — ' + mode + ' — new causal scoring', methods[mode])
        repetitions(lines, 'RPB-v24.market-01.alt-01 — ' + mode + ' — reused transfer', previous[mode])
    repetitions(lines, 'RPB-v22.alt-01 — reused synthetic-trained reference', refs)
    lines += ['## Cost, provenance and next direction', '',
              '**Zero new encoder training updates.** No causal reconstruction query was measured; inherited V30 MAE/NLL is not a new causal result. '
              'Engineering proof, frozen API extraction/input I/O/state checks, fixed-head fitting/scoring/output I/O, saved arithmetic and reporting are separate costs.', '']
    add_table(lines, MKT + ' Fit/score: TEMPO-5 complexity5/5 slow timing as separately identified above; frozen stages only.',
              ('Stage', 'Seconds'), [('import/state/source-parity I/O', proof['import_seconds_including_state_and_io']),
              ('causal intervention proof/state checks', proof['proof_seconds_including_state_checks']),
              ('artifact/typed checks', proof['artifact_and_typed_check_seconds']),
              ('full extraction process wall, including initial checkpoint/cache loading', execution['stage_seconds']['extraction']),
              ('quality export segment, excluding initial checkpoint/cache loading', summary['elapsed_seconds_including_input_io_and_state_checks']),
              ('market fixed heads/scoring/output I/O wall', execution['stage_seconds']['market_heads']),
              ('synthetic fixed heads/scoring/output I/O', reader['head_fitting_and_scoring_seconds_including_io'])])
    lines += ['All55 old checkpoint pins/99 portable pairs/44 frozen-document bindings and the23-file interrupted V30 extraction were verified before work. '
              'The completed77,831-file historical scan and additive totals correction are bound through preflight; no20GB rescan or recopy of old capsules occurred. '
              'New source/compiler closure is independently owned under physical versions/v1, mirrored and dependency-checked before quality. '
              'Necessary source/import/proof/quality assets are hash-bound; tensors and bounded raw/native intervention arrays remain local/ignored.', '',
              'The metadata indexes under raw_patch_bottleneck_mae/versions and causal_patch_bottleneck_mae/versions distinguish architecture, instance and checkpoint. '
              'Historical definition gaps remain explicit; existing source/results were not moved or retroactively relabeled. Preserve every old/new reference and failed attempt.', '',
              'This report changes neither primary RPB-v22 nor balanced native36 selection. Interpret the causal conversion through both fixed-head result families. '
              'Any causal training, fresh chronology or forecasting study needs a separate prospective protocol. No primary promotion, commit, push or publication follows.']
    save(report_path, '\n'.join(lines).rstrip() + '\n')
    completion = {'kind': 'causal_patch_bottleneck_mae_v1_completion', 'protocol': PROTOCOL, 'status': 'passed',
                  'model': TAG, 'causal_proof_passed': True, 'market_comparisons': market_gates,
                  'synthetic_retention_passed': synthetic['retention_passed'], 'synthetic_retention_total': 8,
                  'synthetic_full_coverage': synthetic['full_coverage'],
                  'synthetic_comparison_gate_passed': synthetic['full_synthetic_comparison_gate_passed'],
                  'new_encoder_training_updates': 0, 'new_scaler_or_bank_fits': 0, 'full_quality_CUDA_calls': 786,
                  'engineering_CUDA_calls': proof['CUDA_encoder_calls'], 'market_new_ridge_pipelines': 1,
                  'market_new_ridge_target_fits': 4, 'market_new_neural_fits': 3, 'synthetic_new_outer_normalizers': 6,
                  'synthetic_new_ridge_fits': 18, 'synthetic_new_neural_fits': 18, 'old_encoder_calls': 0, 'old_head_fits': 0,
                  'historical_market_rows_preserved': 96, 'new_market_task_rows': 4, 'synthetic_panels': 4,
                  'card_sha256': CARD, 'layout_addendum_sha256': ADDENDUM, 'source_checkpoint_sha256': SOURCE,
                  'causal_checkpoint': asset(checkpoint), 'proof_receipt': asset(proof_path),
                  'independent_audit_sha256': sha(audit_path), 'wrapper_verification_sha256': sha(run / 'verification.json'),
                  'admission_sha256': sha(admission_path), 'report_sha256': sha(report_path),
                  'quality_tools_admission_sha256': sha(tools_path), 'prequality_proof_audit_only': True,
                  'postquality_saved_scalar_arithmetic_checked_by_reporter': True,
                  'table_rows_sha256': sha(durable / 'table_rows.json'), 'asset_hashes_sha256': sha(durable / 'asset_hashes.json'),
                  'reporter_sha256': sha(Path(__file__).resolve()), 'reporter_scalar_checks': CHECKS,
                  'reporting_and_hash_capture_seconds': time.monotonic() - started, 'tensor_deserializations_by_reporter': 0,
                  'encoder_or_head_calls_by_reporter': 0, 'primary_changed': False, 'balanced_selection_changed': False}
    dump(durable / 'completion.json', completion)
    verify(admission['source'])
    verify(tools['sources'])
    members = [{'path': str(path.relative_to(durable)), 'sha256': sha(path), 'bytes': path.stat().st_size}
               for path in sorted(durable.rglob('*')) if path.is_file()]
    dump(durable / 'durable_files.json', members)
    print(json.dumps(completion, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

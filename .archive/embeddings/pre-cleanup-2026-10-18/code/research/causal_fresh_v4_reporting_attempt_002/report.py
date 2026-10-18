#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Describe fresh CPB-v4 initial/final saved quality; no pretrained quality targets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

TAG = 'CPB-v4.market-01'
INIT = 'CPB-v4.market-01.init'
CARD_SHA = '114026b2e6d11d84aee1855a8af673d84e782ab057eb45d398a5d66fa94b1dce'
INPUT_SHA = 'a6bff90ffc9dd917c6a9c446659c52e0c76b5fb34a9f67201788eb3168948eb9'
GRADIENT_POLICY = 'variance_input_gradient_stop-v1'
REPS = (2701, 2802, 2903)
SEEDS = (13088411117524848790, 10018485049408248344, 9083751155097190455)
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
MASTERS = (941101, 941102, 941103)
MODES = ('synthetic_train', 'market_scaler')
VIEWS = ('validation_intact', 'validation_deleted')
HEADS = ('ridge', 'tiny_secondary')
STAGES = ('initial', 'final')
MKT = ('Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · '
       '**complexity 4/5** · TRAIN112 / known chronological VALIDATION32.')
T5 = ('Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features '
      'and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256.')


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify_pin(binding, origin):
    path = Path(binding['path'])
    if not path.is_absolute():
        path = Path(origin) / path
    require(path.is_file() and path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'],
            'Bound artifact changed: ' + str(path))
    return path.resolve(strict=True)


def close(a, b, why):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12, why)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)


def dump(path, value):
    write_new(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def stream_seed(rep, width):
    mask = (1 << 64) - 1
    value = (rep + 0x9e3779b97f4a7c15 * width) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return str(value ^ (value >> 31))


def market_rows(path, card_sha, input_manifest_sha, checkpoint_stage):
    report = load(path)
    require(report['kind'] == 'fixed_past_window_regression_v1' and
            report['dataset_id'] == 'MKT-HIST-2' and
            report['card_sha256'] == card_sha and report['input_manifest_sha256'] == input_manifest_sha and
            (report['training_rows'], report['validation_rows']) == (112, 32), 'Fixed market population/card identity')
    expected = {'ridge_penalty': 1, 'neural_hidden': 16, 'neural_learning_rate': .01,
                'neural_updates': 100, 'outputs': 4, 'loss': 'masked TRAIN-target-standardized MSE',
                'device': 'CPU (frozen readouts only)'}
    require(all(report['heads'][k] == v for k, v in expected.items()), 'Fixed market recipe')
    require(len(report['methods']) == 1 and report['methods'][0]['name'] == 'cpb_v4', 'Only this saved fresh checkpoint market heads')
    method = report['methods'][0]
    require(tuple(method[k] for k in ('size', 'ridge_parameters', 'neural_parameters', 'ridge_fits',
                                     'ridge_task_fits', 'neural_fits', 'training_feature_rows')) ==
            (36, 148, 660, 1, 4, 3, 112) and method['feature_training_supported'], 'D36 market head dimensions/budgets')
    require(tuple(r['repetition'] for r in method['repetitions']) == REPS and
            tuple(r['actual_seed'] for r in method['repetitions']) == SEEDS, 'Unchanged D36 market seeds')
    rows = []
    for i, task in enumerate(TASKS):
        targets = [r['targets'][i] for r in method['repetitions']]
        ridge = targets[0]['ridge']
        require(all(t['name'] == task and t['ridge'] == ridge and t['train_target_eligible'] == 112 and
                    t['train_target_supported'] and t['method_training_supported'] for t in targets), 'Market task support/ridge reuse')
        for j, rep in enumerate(method['repetitions']):
            require(rep['neural_updates_completed'] == 100 and rep['ridge_fit_reused'] == (j != 0), 'Market fixed updates/reuse')
        for t in targets:
            for head in ('ridge', 'neural'):
                s = t[head]
                require(s['total'] == s['target_eligible'] == 32 and 0 < s['valid'] <= 32 and
                        all(math.isfinite(s[k]) for k in ('r2', 'mae', 'rmse')), 'Market physical metrics/support')
                close(s['coverage'], s['valid'] / 32, 'Market coverage')
        row = {'dataset_id': 'MKT-HIST-2', 'task': task, 'recipe_id': 'causal-fresh-v4',
               'designed_complexity_level': 4, 'complexity_scale_max': 5,
               'view': 'intact chronological VALIDATION', 'evidence_stage': 'known-source development',
               'method': INIT if checkpoint_stage == 'initial' else TAG, 'checkpoint_stage': checkpoint_stage, 'size': 36, 'unit': targets[0]['unit'], 'linear_r2': ridge['r2'],
               'linear_mae': ridge['mae'], 'linear_rmse': ridge['rmse'],
               'valid': ridge['valid'], 'total': 32,
               'coverage': min(t[h]['coverage'] for t in targets for h in ('ridge', 'neural')),
               'train_target_mean': targets[0]['train_target_mean'], 'train_target_scale': targets[0]['train_target_scale'],
               'repetitions': [{'repetition': r['repetition'], 'actual_seed': r['actual_seed'], 'target': r['targets'][i]}
                               for r in method['repetitions']], 'report': pin(path), 'reused': False}
        for metric in ('r2', 'mae', 'rmse'):
            row['neural_' + metric + '_each'] = [t['neural'][metric] for t in targets]
            row['neural_mean_' + metric] = statistics.mean(row['neural_' + metric + '_each'])
        rows.append(row)
    return rows, report


def validate_classification(report, master):
    require(report['protocol'] == 'fixed-feature-readouts-v1' and int(report['master_seed']) == master, 'Fixed classification protocol/cohort')
    for key, value in {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100, 'tiny_learning_rate': .01,
                       'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods',
                       'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}.items():
        require(report['recipe'][key] == value, 'Fixed classification recipe: ' + key)
    require(report['fit_counts'] == {'outer_train_normalizer_fits': 2, 'ridge_fits': 6, 'tiny_fits': 6,
                                    'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'Per-cohort fixed head fits')
    require(len(report['methods']) == 2 and {m['method'] for m in report['methods']} == set(MODES) and not report['pairs'], 'Only declared scaler policies')
    for method in report['methods']:
        require(method['size'] == 36 and method['status'] == 'measured' and not method['inputs_train_prepared'] and
                method['outer_train_normalizer_fits'] == 1 and len(method['repetitions']) == 3, 'D36 native classification inputs')
        for rep, repetition in zip(method['repetitions'], REPS):
            require(rep['id'] == 'rep-' + str(repetition) and rep['status'] == 'measured' and
                    rep['actual_probe_seed_decimal'] == stream_seed(repetition, 36) and
                    rep['ridge_parameters'] == 74 and rep['neural_parameters'] == 626, 'D36 classification seeds/head sizes')
            for view in ('training',) + VIEWS:
                score = rep[view]
                n, groups = (512, 256) if view == 'training' else (256, 128)
                population = score['population']
                require(population['total_rows'] == n and population['total_source_groups'] == groups and
                        0 < population['valid_rows'] <= n, 'Fixed classification rows/source groups')
                close(population['coverage'], population['valid_rows'] / n, 'Synthetic population coverage')
                for head in HEADS:
                    s = score[head]
                    require(s['total'] == n and s['valid'] == population['valid_rows'] and
                            0 <= s['correct'] <= s['valid'], 'Synthetic correct/valid counts')
                    close(s['accuracy'], s['correct'] / s['valid'], 'Synthetic accuracy')
                    close(s['coverage'], s['valid'] / n, 'Synthetic coverage')
                    close(s['full_population_correctness'], s['correct'] / n, 'Synthetic full-population correctness')
                if view in VIEWS:
                    for key, head in (('ridge_grouped_interval', 'ridge'), ('tiny_grouped_interval', 'tiny_secondary')):
                        ci = score[key]
                        require(ci['replicates'] == 1000 and ci['confidence'] == .95 and ci['source_groups'] == groups and
                                0 <= ci['lower'] <= ci['upper'] <= 1, 'Fixed conditional source-group interval')
                        close(ci['estimate'], score[head]['accuracy'], 'Interval point estimate')


def aggregate_classification(reports):
    scores = {}
    for mode in MODES:
        methods = [next(m for m in report['methods'] if m['method'] == mode) for report in reports]
        scores[mode] = {}
        for view in VIEWS:
            scores[mode][view] = {}
            for head in HEADS:
                means = [statistics.mean(r[view][head]['accuracy'] for r in m['repetitions']) for m in methods]
                scores[mode][view][head] = {'mean': statistics.mean(means), 'master_means': means,
                                          'minimum': min(means), 'maximum': max(means)}
            scores[mode][view]['coverage'] = {'mean': statistics.mean(r[view]['population']['coverage']
                for m in methods for r in m['repetitions'])}
    return scores


def table(lines, legend, columns, rows):
    lines.extend([legend, '', '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |'])
    lines.extend('| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |' for row in rows)
    lines.append('')


def saved_causal_proof(binding, checkpoint, updates, origin):
    path = verify_pin(binding, origin)
    proof = load(path)
    expected = {'kind': 'cpb_v4_saved_checkpoint_causal_proof_v1', 'status': 'passed', 'model': 'CPB-v4',
                'gradient_policy': GRADIENT_POLICY,
                'source_updates': 0, 'causal_updates': updates, 'initialization_seed': 943001,
                'pretrained_ancestor': 'none', 'new_training_updates': 0, 'all_cutoffs': 128,
                'suffix_cases': 2350, 'internal_temporal_cases': 30, 'state_unchanged': True,
                'future_values_and_masks_invariant': True, 'CPU_encoder_calls': 0, 'head_fits': 0}
    require(all(proof[k] == v for k, v in expected.items()) and proof['CUDA_encoder_calls'] > 0 and proof['checks'] > 0,
            'Actual fresh saved initial/final full causal proof')
    require(Path(proof['checkpoint']).resolve(strict=True) == checkpoint and proof['checkpoint_sha256'] == sha(checkpoint),
            'Proof binds exactly the scored fresh checkpoint')
    witness = path.parent / (('initial0' if updates == 0 else 'final2048') + '-causality.json')
    detail = load(witness)
    require(detail['status'] == 'passed' and detail['all_cutoffs'] == 128 and detail['suffix_cases'] == 2350 and
            detail['internal_temporal_cases'] == 30 and detail['state_unchanged'], 'Saved bounded causal witness')
    return {'proof': pin(path), 'witness': pin(witness), 'checks': proof['checks'], 'CUDA_encoder_calls': proof['CUDA_encoder_calls']}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'training', 'card', 'input-manifest', 'raw-controls', 'quality-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'Existing managed container only')
    run, training = args.run.resolve(strict=True), args.training.resolve(strict=True)
    require(sha(args.card) == CARD_SHA and sha(args.input_manifest) == INPUT_SHA, 'Frozen prospective card/input manifest')
    card, inputs = load(args.card), load(args.input_manifest)
    require(card['kind'] == 'causal_fresh_v4_prospective_card_v1' and card['instance'] == TAG and
            card['initial_quality_label'] == INIT and not card['historical_reference_is_quality_gate'] and
            card['mechanism']['source_updates'] == 0 and not card['mechanism']['pretrained_model_inputs'] and
            not card['mechanism']['teacher_model_inputs'] and not card['mechanism']['checkpoint_import_for_initialization'] and
            not card['quality']['historical_quality_floors'] and not card['quality']['initial_quality_reused'],
            'Fresh ancestry, initial measurement and absence of historical quality targets')
    require(card['criteria']['establishment'].startswith('All existing trust/coverage admission') and
            card['baseline'] == 'CPB-v3.market-01' and card['baseline_remains_selected'] and
            card['mechanism']['kind'] == GRADIENT_POLICY and
            not card['criteria']['rescue_allowed'] and not card['criteria']['automatic_application_promotion'],
            'Trust establishment separated from descriptive quality')
    require(card['input_manifest']['sha256'] == INPUT_SHA and inputs['model'] == TAG and
            tuple(card['fixed_heads']['market']['neural_repetitions']) == REPS and
            tuple(int(x) for x in card['fixed_heads']['market']['dimension36_seeds']) == SEEDS,
            'Fixed populations, input manifest and D36 seeds')
    # V4 prospectively reuses the exact V3 encoder-free raw-control snapshot.
    require(sha(args.raw_controls) == 'c37d484cde4ddf9ae551b35aad8b2595865e109d566dab2ba7c1872e814e8077',
            'Exact preserved V3 raw-control snapshot; no new raw fits or relabeling')
    raw = load(args.raw_controls)
    require(raw['kind'] == 'causal_fresh_v3_raw_controls_v1' and raw['derived_without_new_fits'] and
            not raw['encoder_rows_included'] and not raw['quality_gates_included'] and
            raw['source'] == card['quality']['raw_controls_reused'] and raw['source'] in inputs['files'] and len(raw['rows']) == 8,
            'Only declared source-pinned encoder-free raw controls')
    require({(r['source_method'], r['task']) for r in raw['rows']} ==
            {(m, t) for m in ('raw_recent128', 'raw_full256') for t in TASKS}, 'Complete four-task raw controls only')
    for row in raw['rows']:
        require(row['method'].startswith('Raw data — no encoder') and row['coverage'] == 1 and row['total'] == row['valid'] == 32,
                'Raw controls retain their original population and encoder-free identity')
    quality = load(args.quality_receipt)
    require(quality['status'] == 'passed' and quality['protocol'] == 'causal-fresh-v4' and quality['model'] == TAG and
            quality['card_sha256'] == CARD_SHA, 'Completed admitted fresh initial/final quality execution')
    build_path = verify_pin(quality['build_admission'], args.quality_receipt.parent)
    build = load(build_path)
    require(build['status'] == 'passed' and build['card_sha256'] == CARD_SHA and len(build['binaries']) == 6,
            'Same source-admitted compiler/binary closure')
    admitted_binaries = {verify_pin(binary, build_path.parent) for binary in build['binaries']}
    gradient_path = verify_pin(quality['variance_gradient_proof'], args.quality_receipt.parent)
    gradient = load(gradient_path)
    expected_gradient = {'status': 'passed', 'model': 'CPB-v4', 'gradient_policy': GRADIENT_POLICY,
                         'actual_CUDA': True, 'native_variance_gradient_blocked': True,
                         'mean_path_live': True, 'variance_head_learning': True,
                         'forward_variance_input_dependence': True, 'manual_partial_gradient_parity': True,
                         'nonvacuous_nonzero_fixture': True}
    require(all(gradient[k] == value for k, value in expected_gradient.items()),
            'Nonvacuous actual CUDA variance gradient-isolation mechanism proof')
    gradient_binary = verify_pin(gradient['binary'], gradient_path.parent)
    gradient_log = verify_pin(gradient['log'], gradient_path.parent)
    require(gradient_binary in admitted_binaries and
            'PASS CPB-v4 variance_input_gradient_stop-v1:' in gradient_log.read_text(encoding='utf-8'),
            'Mechanism receipt binds admitted binary and observed passing test output')
    engineering_path = verify_pin(quality['engineering_proof'], args.quality_receipt.parent)
    engineering = load(engineering_path)
    expected_engineering = {'kind': 'cpb_v4_fresh_cuda_engineering_v1', 'status': 'passed', 'source_updates': 0,
        'gradient_policy': GRADIENT_POLICY,
        'initialization_seed': 943001, 'pretrained_ancestor': 'none', 'old_checkpoints_opened': 0,
        'all_initial_tensors_equal_seeded_constructor': True, 'first_typed_artifact_exact': True,
        'scaler_and_bank_fitted_fixture_TRAIN_only': True, 'all_cutoffs_per_stage': 128,
        'suffix_cases_per_stage': 2350, 'internal_temporal_cases_per_stage': 30,
        'causal_updates_primary_fixture': 2, 'resume_witness_updates': 1, 'quality_updates': 0,
        'typed_resume_next_loss_model_moments_bitwise_exact': True, 'CPU_encoder_calls': 0, 'head_fits': 0}
    require(all(engineering[k] == v for k, v in expected_engineering.items()) and
            engineering['stages'] == ['initialized', 'trained'] and engineering['typed_negative_cases'] >= 30 and
            engineering['CUDA_encoder_calls'] > 0 and engineering['checks'] > 0 and
            all(v['nonzero_gradient_parameter_events'] > 0 and v['weight_movement_parameter_events'] > 0
                for v in engineering['actual_cuda_learning'].values()), 'Full fresh CUDA/causality/typed refusal-resume engineering admission')
    checkpoints = {s: verify_pin(quality[s + '_checkpoint'], args.quality_receipt.parent) for s in STAGES}
    require(checkpoints['initial'] == training / 'checkpoints/inference-0.pt' and
            checkpoints['final'] == training / 'checkpoints/inference-2048.pt', 'Only initial0/final2048 selected')
    proofs = {'initial': saved_causal_proof(quality['initial_saved_proof'], checkpoints['initial'], 0, args.quality_receipt.parent),
              'final': saved_causal_proof(quality['post_training_proof'], checkpoints['final'], 2048, args.quality_receipt.parent)}
    initialization_path = verify_pin(quality['fresh_initialization'], args.quality_receipt.parent)
    require(initialization_path == training / 'fresh-initialization.json', 'Actual main trajectory initialization receipt')
    initialization = load(initialization_path)
    expected_init = {'kind': 'cpb_v4_fresh_initialization_v1', 'status': 'passed',
                     'initialization_kind': 'fresh-seeded-constructor-no-import-v1', 'initialization_seed': 943001,
                     'pretrained_ancestor': 'none', 'inherited_learned_parameters': 0,
                     'fresh_learned_parameter_values': 293672, 'analytic_fixed_prior_values': 3888,
                     'source_updates': 0, 'causal_updates': 0, 'old_model_artifacts_opened': 0,
                     'old_normalization_artifacts_opened': 0, 'all_initial_parameters_equal_direct_seeded_constructor': True,
                     'parameter_tensors_checked': 76, 'parameter_values_checked': 297560}
    require(all(initialization[k] == v for k, v in expected_init.items()) and
            initialization['initial_state_identity'] == initialization['direct_constructor_state_identity'],
            'Every learned parameter is fresh and matches an independent seeded constructor')
    require(verify_pin(initialization['initial_checkpoint'], training) == checkpoints['initial'], 'Fresh initialization binds measured initial checkpoint')
    calibration = load(training / 'calibration/summary.json')
    expected_calibration = {'kind': 'cpb_v4_fresh_TRAIN_calibration_v1', 'status': 'passed', 'TRAIN_anchors': 4096,
                            'history': 128, 'input_scaler_fits': 1, 'bank_normalizer_fits': 1,
                            'validation_rows_used': 0, 'test_rows_used': 0, 'encoder_calls': 0,
                            'old_normalization_artifacts_opened': 0}
    require(all(calibration[k] == v for k, v in expected_calibration.items()), 'New market statistics fitted on TRAIN only')
    for artifact in calibration['artifacts']:
        verify_pin({'path': artifact['file'], 'bytes': artifact['bytes'], 'sha256': artifact['sha256']}, training / 'calibration')
    require(calibration['input_scaler_identity'] == initialization['fresh_input_scaler_identity'] and
            calibration['bank_identity'] == initialization['fresh_bank_identity'], 'Fresh calibration used by the actual model')
    train_summary = load(training / 'training-summary.json')
    require(train_summary['kind'] == 'causal_fresh_v4_training_summary_v1' and train_summary['status'] == 'passed' and
            train_summary['gradient_policy'] == GRADIENT_POLICY and
            train_summary['encoder_gradient_kind'] == 'partial-NLL-gradient;live-mean-residual;variance-input-stopped' and
            train_summary['conditional_variance_head_trainable'] is True and
            train_summary['model'] == TAG and train_summary['source_updates'] == 0 and train_summary['new_causal_updates'] == 2048 and
            train_summary['batch_size'] == 8 and train_summary['TRAIN_windows'] == 4096 and train_summary['device'] == 'cuda:0' and
            train_summary['typed_resume_512_exact'] and train_summary['fixed_prior_scaler_normalizer_unchanged'] and
            train_summary['new_scaler_fits'] == train_summary['new_normalizer_fits'] == 1 and
            not train_summary['saved_quality_features_reused'], 'One fresh causal CUDA trajectory, fresh statistics and typed resume')
    require(all(train_summary[k] == 0 for k in ('old_model_artifacts_opened', 'old_normalization_artifacts_opened',
                'calibration_encoder_calls', 'calibration_validation_rows', 'CPU_encoder_calls', 'old_encoder_calls',
                'head_fits', 'training_replays')), 'No old-state reuse, held-out calibration, old execution or training replay')
    require(all(g['nonzero_gradient_tensors'] > 0 and g['moved_tensors'] > 0 for g in train_summary['parameter_groups']),
            'Actual CUDA gradient and movement in every recorded learned block')
    resume = load(training / 'typed-resume-512.json')
    require(resume['status'] == 'passed' and resume['completed_updates'] == 512 and resume['same_parameters_buffers_moments_steps_order'] and
            resume['same_attempt_counter_and_three_seeds'] and resume['continued_actual_trajectory_from_saved_state'] and
            resume['repeated_training_updates'] == 0, 'Exact typed continuation in the actual trajectory')
    queries = {s: load(training / ('query-' + s + '.json')) for s in STAGES}
    for stage in STAGES:
        require(train_summary['fixed_query_' + stage] == queries[stage], 'Saved query measurements copied exactly')
        for role, n in (('train', 112), ('validation', 32)):
            q = queries[stage][role]
            require(q['valid_rows'] == n and len(q['row_mae']) == len(q['row_nll']) == len(q['row_target_cells']) == n,
                    'Unchanged fixed-query population')
            close(q['standardized_fixed_query_MAE'], statistics.mean(q['row_mae']), 'Standardized query MAE aggregation')
            close(q['Gaussian_NLL_F64'], statistics.mean(q['row_nll']), 'Separate Gaussian NLL aggregation')
            require(sum(q['row_target_cells']) == q['target_cells'], 'Fixed-query target support')
    all_extraction = load(run / 'features/extraction-summary.json')
    require(all_extraction['status'] == 'passed' and all_extraction['full_CUDA_encode_calls'] == 1572 and
            all_extraction['new_synthetic_TRAIN_input_scaler_fits'] == 3 and
            all_extraction['new_market_input_scaler_fits'] == all_extraction['new_bank_normalizer_fits'] == 0 and
            all_extraction['old_encoder_calls'] == all_extraction['old_model_or_scaler_archives_opened'] == 0,
            'Both new checkpoint exports and three fresh shared synthetic TRAIN scaler fits only')
    measurements, classification, classification_reports, extraction = {}, {}, {}, {}
    full_coverage = True
    for stage in STAGES:
        stage_run = run / stage
        rows, market_report = market_rows(stage_run / 'market-readouts/report.json', CARD_SHA, INPUT_SHA, stage)
        measurements[stage] = rows
        full_coverage &= all(r['coverage'] == 1 and r['valid'] == r['total'] == 32 for r in rows)
        for key, role in (('training', 'train'), ('validation', 'validation')):
            require(Path(market_report['methods'][0][key + '_path']).resolve(strict=True) ==
                    run / 'features' / stage / 'market' / ('cpb_v4-' + role + '.pt'), 'This checkpoint stage supplies its own market heads')
        reports = [load(stage_run / 'synthetic-readouts' / ('master-' + str(m)) / 'report.json') for m in MASTERS]
        for report, master in zip(reports, MASTERS):
            validate_classification(report, master)
        classification_reports[stage] = reports
        classification[stage] = aggregate_classification(reports)
        full_coverage &= all(rep[view]['population']['valid_rows'] == rep[view]['population']['total_rows'] and
                             rep[view]['population']['valid_source_groups'] == rep[view]['population']['total_source_groups']
                             for report in reports for method in report['methods'] for rep in method['repetitions']
                             for view in ('training',) + VIEWS)
        reader = load(stage_run / 'synthetic-readouts/reader-summary.json')
        require(reader['model'] == TAG and reader['checkpoint_stage'] == stage and reader['native_width'] == 36 and
                all(reader[k] == 0 for k in ('encoder_calls', 'encoder_updates', 'old_encoder_calls', 'old_head_fits', 'post_encoder_pca_fits')),
                'New saved initial/final feature-only classification fitting')
        e = load(run / 'features' / stage / 'summary.json')
        require(e['model'] == TAG and e['checkpoint_stage'] == stage and e['source_updates'] == 0 and
                e['causal_updates'] == (0 if stage == 'initial' else 2048) and
                e['full_CUDA_encode_calls'] == 786 and e['synthetic_rows'] == 6144 and
                e['market']['train']['full_CUDA_encode_calls'] == 14 and e['market']['validation']['full_CUDA_encode_calls'] == 4 and
                e['all_parameters_buffers_normalizer_scaler_unchanged'] and e['all_gradients_undefined'] and e['no_labels_in_encoder'],
                'One immutable full API extraction per new checkpoint')
        require(Path(e['checkpoint']).resolve(strict=True) == checkpoints[stage] and
                e['market_scaler_identity'] == calibration['input_scaler_identity'] and e['bank_normalizer_identity'] == calibration['bank_identity'],
                'Fresh checkpoint/statistics identity across export stages')
        extraction[stage] = e
    for mode in MODES:
        for master in MASTERS:
            identities = [next(x['scaler_identity'] for x in extraction[s]['synthetic'] if x['mode'] == mode and x['master'] == master) for s in STAGES]
            require(identities[0] == identities[1], 'Same fresh fitted external scaler for initial/final')
    market_changes, synthetic_changes, panels = [], [], []
    for before, after in zip(measurements['initial'], measurements['final']):
        require(before['task'] == after['task'] and before['train_target_mean'] == after['train_target_mean'] and
                before['train_target_scale'] == after['train_target_scale'], 'Same physical target scaling in initial/final heads')
        market_changes.append({'task': after['task'], **{k: {'initial': before[k], 'final': after[k], 'difference': after[k]-before[k]}
            for k in ('linear_r2', 'neural_mean_r2', 'linear_mae', 'neural_mean_mae')}})
    for mode in MODES:
        for view in VIEWS:
            rows = []
            for stage, label in (('initial', INIT), ('final', TAG)):
                scores = classification[stage][mode][view]
                rows.append({'method': label, 'size': 36, 'checkpoint_stage': stage, 'reused': False,
                    'linear_head_accuracy': scores['ridge']['mean'], 'neural_head_accuracy': scores['tiny_secondary']['mean'],
                    'coverage': scores['coverage']['mean']})
            panels.append({'dataset_id': 'TEMPO-5', 'task': 'lag_sign', 'recipe_id': 'tempo-5-slow-fast-nine-feature-v1',
                'designed_complexity_level': 5, 'complexity_scale_max': 5, 'view': view, 'input_policy': mode,
                'head_fitting_dataset_id': 'TEMPO-5', 'scoring_dataset_id': 'TEMPO-5', 'rows': rows})
            for head in HEADS:
                before, after = (classification[s][mode][view][head]['mean'] for s in STAGES)
                synthetic_changes.append({'mode': mode, 'view': view, 'head': head, 'initial': before, 'final': after, 'difference': after-before})
    result = {'kind': 'causal_fresh_v4_results_v1', 'status': 'complete', 'model': TAG, 'initial_label': INIT,
        'source_updates': 0, 'initial_quality_reused': False, 'old_encoder_calls': 0, 'historical_encoder_quality_targets_used': False,
        'trust_protocol_admission_passed': True, 'full_coverage': full_coverage, 'clean_causal_candidate_measured': full_coverage,
        'clean_baseline': 'CPB-v3.market-01', 'clean_baseline_unchanged': True, 'gradient_policy': GRADIENT_POLICY,
        'quality_interpretation': 'Descriptive initial-to-final per-task measures; establishes no quality winner',
        'market_rows': raw['rows'] + measurements['initial'] + measurements['final'], 'new_market_rows': 8,
        'synthetic': {'scores': classification, 'panels': panels, 'all_saved_repetitions_and_intervals': classification_reports},
        'descriptive_changes': {'market': market_changes, 'synthetic': synthetic_changes},
        'training': train_summary, 'fixed_queries': queries, 'proofs': proofs, 'fresh_initialization': initialization,
        'calibration': calibration, 'post_encoder_pca_fits': 0, 'application_changed': False, 'overall_winner_declared': False,
        'card': pin(args.card), 'input_manifest': pin(args.input_manifest), 'raw_controls': pin(args.raw_controls),
        'quality_execution_receipt': pin(args.quality_receipt), 'variance_gradient_proof': pin(gradient_path)}
    output = args.output.resolve()
    require(not output.exists(), 'Preserve all report attempts; exclusive new output required')
    output.mkdir(parents=True)
    dump(output / 'table_rows.json', result)
    assets = {'kind': 'causal_fresh_v4_report_assets_v1', 'run_assets': [pin(p) for p in sorted(run.rglob('*')) if p.is_file()],
              'training_summary': pin(training / 'training-summary.json'), 'initialization': pin(initialization_path),
              'calibration': pin(training / 'calibration/summary.json'), 'typed_resume': pin(training / 'typed-resume-512.json'),
              'query_initial': pin(training / 'query-initial.json'), 'query_final': pin(training / 'query-final.json'),
              'proofs': proofs, 'engineering_proof': pin(engineering_path), 'build_admission': pin(build_path),
              'variance_gradient_proof': pin(gradient_path),
              'reporter': pin(__file__), 'no_old_model_or_tensor_copies': True}
    dump(output / 'asset_hashes.json', assets)
    lines = ['# CPB-v4: fresh causal training and fixed-head measurements', '',
        'CPB-v4 starts every learned tensor from its declared fresh constructor, with source updates0 and '
        'new TRAIN-only input/bank statistics. Initial0 and final2048 embeddings and heads are newly measured. '
        'Pretrained historical encoders supply no quality target, floor, teacher, weights or fitted statistics.', '',
        'Trust admission covers fresh ancestry, actual CUDA learning, typed resume and saved initial/final causal proofs. '
        'Complete finite fixed-head results with full coverage establish a measured clean causal candidate; CPB-v3 remains the clean baseline. '
        'The separate clean-baseline comparison evaluates all 22 frozen advancement conditions. No overall winner or automatic promotion is declared.', '',
        'The sole controlled change stops the variance input gradient at change4. The conditional variance head still learns and its '
        'forward dependence is preserved; mean/residual gradients remain attached. Gaussian NLL values use the unchanged formula, '
        'but encoder updates use its mean-path partial gradient rather than the full joint NLL gradient.', '',
        'One CUDA:0 trajectory uses2048 updates/B8. Four market outputs use fixed ridge1 and tanh16/Adam0.01/100 updates, '
        'TRAIN112 / known chronological VALIDATION32. Repetitions2701/2802/2903 retain the exact D36 seeds, '
        'with148/660 head parameters. Both stages fit their own heads; there is no initial score reuse or post-encoder PCA. '
        'Raw controls are reused encoder-free results at their original larger input sizes. Their original head sizes are '
        '(D+1)×4 linear and16×(D+1)+68 neural. These targets measure retained past-window information, not forecasting.', '']
    for task in TASKS:
        rows = [r for r in result['market_rows'] if r['task'] == task]
        lines.extend(['## ' + task.removeprefix('past_').capitalize(), ''])
        table(lines, MKT + ' Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.',
              ('Method', 'Size', 'Linear R²', 'Neural mean R²', 'Coverage %'),
              [(r['method'], r['size'], format(r['linear_r2'], '.8f'), format(r['neural_mean_r2'], '.8f'), format(100*r['coverage'], '.2f')) for r in rows])
        table(lines, MKT + ' Physical prediction errors; units: ' + rows[-1]['unit'] + '.',
              ('Method', 'Linear MAE', 'Linear RMSE', 'Neural mean MAE', 'Neural mean RMSE'),
              [(r['method'], *(format(r[k], '.17g') for k in ('linear_mae', 'linear_rmse', 'neural_mean_mae', 'neural_mean_rmse'))) for r in rows])
        for stage in STAGES:
            row = next(r for r in measurements[stage] if r['task'] == task)
            lines.extend([row['method'] + ' neural R² for repetitions2701/2802/2903: ' + ', '.join(format(x, '.17g') for x in row['neural_r2_each']) + '.', ''])
    lines.extend(['## Hard timing information', '',
        'TEMPO-5 has cohorts941101/941102/941103, each512 TRAIN examples from256 paired sources and256 VALIDATION '
        'examples from128 distinct paired sources. Both stages use newly fitted three-seed fixed heads per policy/cohort. '
        'Classification heads contain74/626 parameters. Synthetic TRAIN input statistics are freshly fitted once per cohort '
        'and shared by initial/final; the secondary policy uses this run’s fresh market TRAIN scaler. No synthetic encoder training occurs.', ''])
    for panel in panels:
        table(lines, T5 + ' ' + panel['input_policy'] + ', ' + panel['view'] + '.',
              ('Method', 'Size', 'Linear head %', 'Neural head %', 'Coverage %'),
              [(r['method'], r['size'], *(format(100*r[k], '.4f') for k in ('linear_head_accuracy', 'neural_head_accuracy', 'coverage'))) for r in panel['rows']])
    for stage in STAGES:
        for mode in MODES:
            rows = []
            for master, report in zip(MASTERS, classification_reports[stage]):
                method = next(m for m in report['methods'] if m['method'] == mode)
                for rep in method['repetitions']:
                    for view in VIEWS:
                        value = rep[view]
                        rows.append((master, rep['id'], view, format(value['ridge']['accuracy'], '.17g'),
                                     format(value['tiny_secondary']['accuracy'], '.17g'), format(value['population']['coverage'], '.17g')))
            table(lines, T5 + ' ' + stage + ', ' + mode + '; every intact/deleted VALIDATION repetition; exact accuracy fractions.',
                  ('Cohort', 'Fit', 'View', 'Linear accuracy', 'Neural accuracy', 'Coverage'), rows)
    lines.extend(['## Training, fixed queries and separate cost', '',
        'Errors use the immutable saved hidden reconstruction queries with hierarchical standardized MAE. '
        'Gaussian NLL is reported separately and can be negative. The synchronized step timer covers forward/NLL/backward/clip/AdamW '
        'and in-step checks; loading, masking, checkpoint/resume I/O, evidence, extraction and readout fitting remain separate.', ''])
    table(lines, MKT + ' Fixed saved TRAIN/VALIDATION query; standardized MAE, source updates0.',
          ('Encoder', 'Updates', 'Train error ↓', 'Validation error ↓', 'GPU training seconds'),
          [(INIT if s == 'initial' else TAG, 0 if s == 'initial' else 2048,
            format(queries[s]['train']['standardized_fixed_query_MAE'], '.17g'), format(queries[s]['validation']['standardized_fixed_query_MAE'], '.17g'),
            0 if s == 'initial' else format(train_summary['GPU_step_seconds_synchronized'], '.17g')) for s in STAGES])
    table(lines, MKT + ' Same saved fixed queries; Gaussian waveform NLL, distinct from MAE.',
          ('Encoder', 'Updates', 'TRAIN Gaussian NLL', 'VALIDATION Gaussian NLL'),
          [(INIT if s == 'initial' else TAG, 0 if s == 'initial' else 2048,
            format(queries[s]['train']['Gaussian_NLL_F64'], '.17g'), format(queries[s]['validation']['Gaussian_NLL_F64'], '.17g')) for s in STAGES])
    table(lines, MKT + ' and ' + T5 + ' One-run descriptive cost; components are not interchangeable.',
          ('Stage', 'Seconds'), [('Fresh market TRAIN calibration including input/witness I/O', format(train_summary['calibration_seconds'], '.17g')),
                               ('Initial+final export including fresh synthetic statistics and input/state I/O', format(quality['stage_seconds']['extraction'], '.17g'))] +
          [(s + ' market CPU fixed heads and I/O', format(quality['stage_seconds'][s + '_market_heads'], '.17g')) for s in STAGES] +
          [(s + ' synthetic CPU fixed heads/scoring and I/O', format(quality['stage_seconds'][s + '_synthetic_heads'], '.17g')) for s in STAGES])
    lines.extend(['## Interpretation and next direction', '',
        'Each initial-to-final task/head difference is recorded at full precision in `table_rows.json`. '
        'The frozen protocol measures whether fresh causal training changes useful information; it does not require retention '
        'of any pretrained encoder’s scores. Mixed or weak quality remains a valid reported candidate, with no head or checkpoint rescue. '
        'The frozen 22-condition comparison against clean CPB-v3 is reported separately; CPB-v3 remains selected. '
        'Keep this run and every earlier artifact; use a separately declared mechanism for any later improvement. '
        'The application selection is unchanged and no overall quality winner is declared.', ''])
    write_new(output / 'RESULTS.md', '\n'.join(lines) + '\n')
    dump(output / 'completion.json', {'kind': 'causal_fresh_v4_report_completion_v1', 'status': 'complete', 'model': TAG,
        'trust_protocol_admission_passed': True, 'full_coverage': full_coverage, 'clean_causal_candidate_measured': full_coverage,
        'clean_baseline': 'CPB-v3.market-01', 'clean_baseline_unchanged': True, 'gradient_policy': GRADIENT_POLICY,
        'historical_encoder_quality_targets_used': False, 'overall_winner_declared': False, 'new_market_rows': 8, 'synthetic_panels': 4,
        'reporting_seconds': time.monotonic()-started, 'report': pin(output / 'RESULTS.md'),
        'tables': pin(output / 'table_rows.json'), 'assets': pin(output / 'asset_hashes.json')})


if __name__ == '__main__':
    main()



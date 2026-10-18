#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Describe fresh CPB-v12 initial/final saved quality; no pretrained quality targets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

TAG = 'CPB-v12.market-01'
INIT = 'CPB-v12.market-01.init'
# The admitted source manifest binds both this reporter and its prospective card.
# Read the card identity at runtime to avoid a source/card hash cycle.
CARD_SHA = None
INPUT_SHA = None
OBJECTIVE_POLICY = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
GRADIENT_POLICY = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
ENCODER_GRADIENT_KIND = 'Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1'
DIRECT_GRADIENT_ROUTES = 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
OBJECTIVE_FIELDS = {'objective_policy': OBJECTIVE_POLICY, 'gradient_policy': GRADIENT_POLICY,
                    'mean_huber_delta': 1., 'loss_coefficients': [1., 1.],
                    'variance_error_detached': True, 'reporting_nll_backpropagated': False,
                    'direct_gradient_routes': DIRECT_GRADIENT_ROUTES}
ESTABLISHMENT_CONTRACT = ('Full trust including exact initial native20:40 DATA and whole-statistic pairings against saved V11, '
                          'and independent training/quality audits, establishes a measured candidate. Advancement additionally '
                          'requires full coverage and all22 declared numeric conditions. Winner and application '
                          'selection remain separate and false.')
TEMPORAL_POLICY = 'original-position-four-bin-attention-mean4-uniform-centered-variance4-v1'
TEMPORAL_PARTITION = [4, 16, 32, 4, 8]
TEMPORAL_FIELDS = {'temporal_pooling_policy': TEMPORAL_POLICY,
                   'temporal_pooling_partition': TEMPORAL_PARTITION,
                   'temporal_pooling_mix': [0.5, 0.5],
                   'empty_local_bin_policy': 'zero-whole-export-slice-after-bias-v1',
                   'temporal_moment_policy': 'componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1',
                   'temporal_moment_partition': [4, 4],
                   'temporal_export_layout': 'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'}
GAUGE_PARAMETER = 'core.encoder.level.base.backbone.pool_score.bias'
LEARNING_FIELDS = {'registered_learned_tensors': 76,
                   'mandatory_identifiable_learned_tensors': 75,
                   'data_gradient_gauge_exemption': GAUGE_PARAMETER}
POSTMEASUREMENT_TRUST_PENDING = {'initial_native20_40_and_statistics_pairings': 'pending',
                               'independent_training_audit': 'pending',
                               'independent_quality_audit': 'pending'}
REPS = (2701, 2802, 2903)
SEEDS = (6049515622977958931, 8262941649267994858, 2095371932330426401)
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


def objective_contract(record, label):
    require(all(record.get(key) == value for key, value in OBJECTIVE_FIELDS.items()),
            'Exact fixed Huber1 plus detached-error variance-calibration objective contract: '+label)


def temporal_contract(record, label):
    require(all(record.get(key) == value for key, value in TEMPORAL_FIELDS.items()),
            'Exact original-position four-bin temporal pooling contract: '+label)


def learning_contract(record, label):
    require(all(record.get(key) == value for key, value in LEARNING_FIELDS.items()),
            '75 identifiable learned tensors and the single registered softmax-gauge exemption: '+label)


def temporal_learning(summary, origin):
    path = verify_pin(summary['temporal_pooling_learning'], origin)
    value = load(path)
    temporal_contract(value, 'actual temporal learning receipt')
    require(value['kind'] == 'cpb_v12_actual_temporal_pooling_learning_v1' and value['status'] == 'passed' and
            value['updates'] == 2048 and value['group_rows'] == 8192 and value['extra_encoder_calls'] == 0 and
            value['weight_decay_counted_as_data_gradient'] is False and
            value['movement_includes_AdamW_weight_decay'] is True,
            'Main-trajectory temporal evidence from existing steps; decay is not data-learning')
    groups = value['groups']
    require(len(groups) == 4 and [g['bin'] for g in groups] == list(range(4)), 'Four ordered temporal groups')
    for index, group in enumerate(groups):
        require(group['export_row_begin'] == 8*index and group['export_row_end'] == 8*(index+1) and
                group['position_begin'] == 4*index and group['position_end'] == 4*(index+1),
                'Original-position/unchanged export-row partition')
        require(all(isinstance(group[key], int) and 0 < group[key] <= 2048 for key in
                    ('weight_nonzero_data_gradient_steps','bias_nonzero_data_gradient_steps',
                     'weight_movement_steps','bias_movement_steps',
                     'position_centered_data_gradient_steps','position_centered_movement_steps')) and
                group['final_weight_changed'] is True and group['final_bias_changed'] is True and
                math.isfinite(group['final_position_centered_change_l2']) and
                group['final_position_centered_change_l2'] > 0,
                'All four row groups and within-bin centered position contrasts actually learn')
    gauge = value['gauge']
    require(gauge['name'] == GAUGE_PARAMETER and gauge['data_gradient_requirement_exempt'] is True and
            gauge['registered_in_same_AdamW'] is True and gauge['movement_is_not_data_learning_proof'] is True,
            'Only scalar softmax gauge exempt; registered and observed without a decay-learning claim')
    between = value['between_bin_positions']
    require(between['basis'] == 'hadamard-full16-signs-div4-v1' and
            between['basis_signs'] == [[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]] and
            between['full_position_divisor'] == between['position_bin_width'] == 4 and
            between['rows'] == 2048 and
            len(between['gradient_steps']) == len(between['movement_steps']) == 3 and
            all(isinstance(n, int) and 0 < n <= 2048 for n in between['gradient_steps']+between['movement_steps']) and
            len(between['final_contrast_changes']) == 3 and
            all(math.isfinite(v) for v in between['final_contrast_changes']) and
            math.isfinite(between['final_projection_change_l2']) and between['final_projection_change_l2'] > 0 and
            math.isclose(sum(v*v for v in between['final_contrast_changes']),
                         between['final_projection_change_l2']**2, rel_tol=1e-10, abs_tol=1e-12) and
            between['gradient_definition'] == 'observed-post-globalclip-data-gradient' and
            between['movement_includes_AdamW_weight_decay'] is True and
            between['weight_decay_counted_as_data_learning'] is False and between['extra_encoder_calls'] == 0,
            'Three fixed full-position Hadamard data-gradient/movement directions and final between-bin change')
    require(value['moment_subgroup_rows'] == 16384 and len(value['moment_subgroups']) == 8,
            'Eight mean/centered-variance subgroups,2048 observations each')
    for index, group in enumerate(value['moment_subgroups']):
        bin_index, half = divmod(index, 2)
        require(group['bin'] == bin_index and group['moment'] == ('mean' if half == 0 else 'centered_variance') and
                group['export_row_begin'] == 8*bin_index+4*half and
                group['export_row_end'] == 8*bin_index+4*(half+1) and
                all(isinstance(group[key], int) and 0 < group[key] <= 2048 for key in
                    ('weight_nonzero_data_gradient_steps','bias_nonzero_data_gradient_steps',
                     'weight_movement_steps','bias_movement_steps')) and
                group['final_weight_changed'] is True and group['final_bias_changed'] is True,
                'Both four-row moment halves have observed weight/bias data gradients and movement; no per-scalar gate')
    require(len(value['artifacts']) == 4, 'Group8192,gauge2048,between-bin2048 and moment-subgroup16384 CSV artifacts')
    artifacts = [pin(verify_pin(binding, path.parent)) for binding in value['artifacts']]
    require(len({artifact['path'] for artifact in artifacts}) == 4 and
            {Path(artifact['path']).name for artifact in artifacts} == {
                'temporal-projection-learning.csv', 'temporal-pool-score-bias-learning.csv',
                'temporal-between-bin-position-learning.csv','temporal-moment-subgroup-learning.csv'},
            'Distinct declared temporal/gauge/between/moment evidence artifacts')
    require(summary['all_eight_temporal_moment_subgroups_learned'] is True and
            summary['all_four_temporal_export_row_groups_learned'] is True and
            summary['all_four_pool_position_contrasts_learned'] is True and
            summary['all_three_between_bin_position_directions_learned'] is True and
            summary['final_between_bin_position_projection_changed'] is True and
            summary['all_mandatory_identifiable_tensors_learned'] is True and
            summary['mandatory_identifiable_tensor_value_capacity'] == 294751 and
            summary['mandatory_identifiable_count_semantics'] == 'tensor numel capacity, not every-scalar gradient/movement proof' and
            summary['per_scalar_nonzero_gradient_proof_claimed'] is False and
            294751 <= summary['learned_tensor_value_capacity_with_recorded_movement_events'] <= 294752 and
            summary['recorded_movement_capacity_semantics'] == 'sum of numel of learned tensors with at least one recorded movement event; not a count of individually moved scalar values',
            'Main summary identifies75 mandatory tensors and294751 tensor-value capacity; no individual-scalar proof')
    return {'receipt': pin(path), 'artifacts': artifacts}


def temporal_proof(binding, origin, admitted_binaries):
    path = verify_pin(binding, origin)
    value = load(path)
    temporal_contract(value, 'separate temporal mechanism fixture')
    expected = {'status': 'passed', 'model': 'CPB-v12', 'actual_CUDA': True,
                'original_position_membership': True, 'both_temporal_paths': True,
                'empty_bins_finite_zero': True, 'true_per_bin_trace_weights': True,
                'manual_pooling_and_adjoint_parity': True,
                'four_export_slices_gradient_and_movement': True,
                'centered_position_contrasts_gradient_and_movement': True,
                'common_softmax_gauge_invariance': True,
                'shared_global_half_mixture': True, 'relative_bin_global_sensitivity': True,
                'three_between_bin_contrasts_gradient_and_movement': True,
                'mean4_centered_variance4_export_layout': True,
                'attention_mean_and_uniform_dispersion_full_expression_adjoints': True, 'uniform_dispersion_own_center': True, 'uniform_dispersion_score_independent': True, 'uniform_dispersion_hidden_gradient_live': True, 'attention_mean_score_gradient_live': True, 'dual_support_actual_trace': True,
                'independent_full_expression_F64_finite_differences': True,
                'actual_F32_kernel_finite_differences': True,
                'equal_mean_different_variance': True,
                'present_constant_variance_dtype_tolerance': True,
                'eight_moment_subgroups_gradient_and_movement': True,
                'backbone_encode_calls': 1, 'public_encoder_calls': 0,
                'direct_pool_kernel_calls': 47, 'head_fits': 0, 'synthetic_optimizer_steps': 1,
                'manual_F32_absolute_tolerance': 3e-5, 'manual_F32_relative_tolerance': 3e-5}
    require(all(value.get(key) == expected_value for key, expected_value in expected.items()),
            'Nonvacuous actual CUDA original-position/row-group temporal fixture')
    binary, log = (verify_pin(value[key], path.parent) for key in ('binary','log'))
    require(binary in admitted_binaries and all(token in log.read_text(encoding='utf-8') for token in
            ('PASS CPB-v12 original-position-four-bin-attention-mean4-uniform-centered-variance4-v1:',
             'manual attention-mean and uniform-dispersion full-expression parity',
             'independent full-expression F64 finite differences', 'actual CUDA F32 kernel finite differences',
             'four mean4 and variance4 projection gradient and movement',
             'present constant variance dtype tolerance', 'equal mean different spread',
             'three between-bin Hadamard contrast gradient and movement', 'relative-bin global sensitivity',
             'empty-bin finite zero', 'common softmax gauge invariance', 'synthetic optimizer steps=1',
             'backbone_encode_calls=1', 'public_encoder_calls=0', 'direct_pool_kernel_calls=47', 'head_fits=0')),
            'Temporal proof binds the admitted seventh binary and observed passing output')
    return {'proof': pin(path), 'binary': pin(binary), 'log': pin(log)}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def configure_card(path, input_path=None):
    global CARD_SHA, INPUT_SHA
    card = load(path)
    require(card['kind'] == 'causal_fresh_v12_prospective_card_v1' and card['instance'] == TAG and
            card['protocol'] == 'causal-fresh-v12' and 'DRAFT' not in card['stage'],
            'Declared V12 card; draft declarations cannot execute quality')
    validate_card_contract(card)
    CARD_SHA, INPUT_SHA = sha(path), card['input_manifest']['sha256']
    require(len(CARD_SHA) == len(INPUT_SHA) == 64, 'SHA256 declaration identities')
    if input_path is not None:
        require(sha(input_path) == INPUT_SHA and Path(input_path).stat().st_size == card['input_manifest']['bytes'],
                'Card-bound unchanged input manifest')
    return card


def validate_card_contract(card):
    """Metadata-only admission, also called before extraction/readout fitting by the quality driver.

    A source-only preflight can compare ESTABLISHMENT_CONTRACT and these literal
    metadata fields without importing this module or performing metric arithmetic.
    """
    require(card['criteria']['establishment'] == ESTABLISHMENT_CONTRACT and
            card['baseline'] == 'CPB-v3.market-01' and card['baseline_remains_selected'] and
            card['mechanism']['kind'] == TEMPORAL_POLICY and
            card['mechanism']['objective_policy'] == OBJECTIVE_POLICY and
            card['mechanism']['gradient_policy'] == GRADIENT_POLICY and
            card['mechanism']['direct_gradient_routes'] == DIRECT_GRADIENT_ROUTES and
            all(card['mechanism'][key] == value for key, value in TEMPORAL_FIELDS.items()) and
            card['training']['objective_policy'] == OBJECTIVE_POLICY and
            card['training']['mean_huber_delta'] == 1. and
            card['training']['loss_coefficients'] == [1., 1.] and
            card['training']['variance_error_detached'] is True and
            card['training']['reporting_nll_backpropagated'] is False and
            not card['criteria']['rescue_allowed'] and not card['criteria']['automatic_application_promotion'],
            'Exact prospective objective/pooling/establishment metadata contract before scientific quality stages')


def admitted_source(build, build_path, path):
    manifest_path = verify_pin(build['source_manifest'], build_path.parent)
    manifest = load(manifest_path)
    relative = Path(path).resolve(strict=True).relative_to(manifest_path.parent.resolve(strict=True)).as_posix()
    matches = [entry for entry in manifest['files'] if entry['path'] == relative]
    require(len(matches) == 1 and matches[0]['bytes'] == Path(path).stat().st_size and
            matches[0]['sha256'] == sha(path), 'Exact source-admitted declaration/reporter bytes: '+relative)


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
    require(len(report['methods']) == 1 and report['methods'][0]['name'] == 'cpb_v12', 'Only this saved fresh checkpoint market heads')
    method = report['methods'][0]
    require(tuple(method[k] for k in ('size', 'ridge_parameters', 'neural_parameters', 'ridge_fits',
                                     'ridge_task_fits', 'neural_fits', 'training_feature_rows')) ==
            (40, 164, 724, 1, 4, 3, 112) and method['feature_training_supported'], 'D40 market head dimensions/budgets')
    require(tuple(r['repetition'] for r in method['repetitions']) == REPS and
            tuple(r['actual_seed'] for r in method['repetitions']) == SEEDS, 'Unchanged D40 market seeds')
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
        row = {'dataset_id': 'MKT-HIST-2', 'task': task, 'recipe_id': 'causal-fresh-v12',
               'designed_complexity_level': 4, 'complexity_scale_max': 5,
               'view': 'intact chronological VALIDATION', 'evidence_stage': 'known-source development',
               'method': INIT if checkpoint_stage == 'initial' else TAG, 'checkpoint_stage': checkpoint_stage, 'size': 40, 'unit': targets[0]['unit'], 'linear_r2': ridge['r2'],
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
        require(method['size'] == 40 and method['status'] == 'measured' and not method['inputs_train_prepared'] and
                method['outer_train_normalizer_fits'] == 1 and len(method['repetitions']) == 3, 'D40 native classification inputs')
        for rep, repetition in zip(method['repetitions'], REPS):
            require(rep['id'] == 'rep-' + str(repetition) and rep['status'] == 'measured' and
                    rep['actual_probe_seed_decimal'] == stream_seed(repetition, 40) and
                    rep['ridge_parameters'] == 82 and rep['neural_parameters'] == 690, 'D40 classification seeds/head sizes')
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
    objective_contract(proof, 'saved checkpoint causal proof')
    temporal_contract(proof, 'saved checkpoint causal proof')
    expected = {'kind': 'cpb_v12_saved_checkpoint_causal_proof_v1', 'status': 'passed', 'model': 'CPB-v12',
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
    card = configure_card(args.card, args.input_manifest)
    inputs = load(args.input_manifest)
    require(card['kind'] == 'causal_fresh_v12_prospective_card_v1' and card['instance'] == TAG and
            card['initial_quality_label'] == INIT and not card['historical_reference_is_quality_gate'] and
            card['mechanism']['source_updates'] == 0 and not card['mechanism']['pretrained_model_inputs'] and
            not card['mechanism']['teacher_model_inputs'] and not card['mechanism']['checkpoint_import_for_initialization'] and
            not card['quality']['historical_quality_floors'] and not card['quality']['initial_quality_reused'],
            'Fresh ancestry, initial measurement and absence of historical quality targets')
    validate_card_contract(card)
    require(card['input_manifest']['sha256'] == INPUT_SHA and inputs['model'] == TAG and
            tuple(card['fixed_heads']['market']['neural_repetitions']) == REPS and
            tuple(int(x) for x in card['fixed_heads']['market']['dimension40_seeds']) == SEEDS,
            'Fixed populations, input manifest and D40 seeds')
    # This candidate reuses the exact V3 encoder-free raw-control snapshot.
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
    require(quality['status'] == 'passed' and quality['protocol'] == 'causal-fresh-v12' and quality['model'] == TAG and
            quality['card_sha256'] == CARD_SHA, 'Completed admitted fresh initial/final quality execution')
    build_path = verify_pin(quality['build_admission'], args.quality_receipt.parent)
    build = load(build_path)
    require(build['status'] == 'passed' and build['card_sha256'] == CARD_SHA and len(build['binaries']) == 7,
            'Same source-admitted compiler/binary closure')
    admitted_source(build, build_path, args.card)
    admitted_source(build, build_path, __file__)
    admitted_binaries = {verify_pin(binary, build_path.parent) for binary in build['binaries']}
    temporal_fixture = temporal_proof(quality['temporal_pooling_proof'], args.quality_receipt.parent, admitted_binaries)
    gradient_path = verify_pin(quality['variance_gradient_proof'], args.quality_receipt.parent)
    gradient = load(gradient_path)
    objective_contract(gradient, 'nonzero CUDA objective fixture')
    expected_gradient = {'status': 'passed', 'model': 'CPB-v12', 'gradient_policy': GRADIENT_POLICY,
                         'actual_CUDA': True, 'mean_native_variance_gradient_blocked': True, 'uncertainty_native_variance_gradient_live': True, 'uncertainty_projection_learning': True,
                         'mean_path_live': True, 'variance_head_learning': True,
                         'forward_variance_input_dependence': True, 'manual_composite_gradient_parity': True,
                         'nonvacuous_nonzero_fixture': True,
                         'direct_NLL_mean_gradient_zero': True, 'huber_mean_gradient_live': True,
                         'direct_Huber_uncertainty_gradient_zero': True, 'shared_clip_coupling_witness': True}
    require(all(gradient[k] == value for k, value in expected_gradient.items()),
            'Nonvacuous actual CUDA composite detached-variance gradient-isolation mechanism proof')
    gradient_binary = verify_pin(gradient['binary'], gradient_path.parent)
    gradient_log = verify_pin(gradient['log'], gradient_path.parent)
    require(gradient_binary in admitted_binaries and
            'PASS CPB-v12 huber1-mean-plus-detached-error-gaussian-variance-v1:' in gradient_log.read_text(encoding='utf-8'),
            'Mechanism receipt binds admitted binary and observed passing test output')
    engineering_path = verify_pin(quality['engineering_proof'], args.quality_receipt.parent)
    engineering = load(engineering_path)
    objective_contract(engineering, 'fresh CUDA engineering proof')
    temporal_contract(engineering, 'fresh CUDA engineering proof')
    learning_contract(engineering, 'fresh CUDA engineering proof')
    expected_engineering = {'kind': 'cpb_v12_fresh_cuda_engineering_v1', 'status': 'passed', 'source_updates': 0,
        'gradient_policy': GRADIENT_POLICY,
        'initialization_seed': 943001, 'pretrained_ancestor': 'none', 'old_checkpoints_opened': 0,
        'all_initial_tensors_equal_seeded_constructor': True, 'first_typed_artifact_exact': True,
        'all_four_temporal_export_row_groups_learned': True,
        'all_eight_temporal_moment_subgroups_learned': True,
        'all_four_pool_position_contrasts_learned': True,
        'all_three_between_bin_position_directions_learned': True,
        'final_between_bin_position_projection_changed': True,
        'weight_decay_counted_as_data_gradient': False,
        'scaler_and_bank_fitted_fixture_TRAIN_only': True, 'all_cutoffs_per_stage': 128,
        'suffix_cases_per_stage': 2350, 'internal_temporal_cases_per_stage': 30,
        'causal_updates_primary_fixture': 2, 'resume_witness_updates': 1, 'quality_updates': 0,
        'typed_resume_next_loss_model_moments_bitwise_exact': True, 'CPU_encoder_calls': 0, 'head_fits': 0}
    require(all(engineering[k] == v for k, v in expected_engineering.items()) and
            engineering['stages'] == ['initialized', 'trained'] and engineering['typed_negative_cases'] >= 30 and
            engineering['CUDA_encoder_calls'] > 0 and engineering['checks'] > 0 and
            all(v['nonzero_gradient_parameter_events'] > 0 and v['weight_movement_parameter_events'] > 0
                for v in engineering['actual_cuda_learning'].values()), 'Full fresh CUDA/causality/typed refusal-resume engineering admission')
    require({f'temporal_export_bin_{index}' for index in range(4)} |
            {f'temporal_pool_mean_bin_{index}' for index in range(4)} |
            {f'temporal_pool_centered_variance_bin_{index}' for index in range(4)} |
            {f'temporal_position_bin_{index}' for index in range(4)} |
            {f'temporal_between_direction_{index}' for index in range(3)} <= set(engineering['actual_cuda_learning']),
            'Engineering includes eight moment subgroups,four export-row/within-bin and three between-bin groups')
    checkpoints = {s: verify_pin(quality[s + '_checkpoint'], args.quality_receipt.parent) for s in STAGES}
    require(checkpoints['initial'] == training / 'checkpoints/inference-0.pt' and
            checkpoints['final'] == training / 'checkpoints/inference-2048.pt', 'Only initial0/final2048 selected')
    proofs = {'initial': saved_causal_proof(quality['initial_saved_proof'], checkpoints['initial'], 0, args.quality_receipt.parent),
              'final': saved_causal_proof(quality['post_training_proof'], checkpoints['final'], 2048, args.quality_receipt.parent)}
    initialization_path = verify_pin(quality['fresh_initialization'], args.quality_receipt.parent)
    require(initialization_path == training / 'fresh-initialization.json', 'Actual main trajectory initialization receipt')
    initialization = load(initialization_path)
    require('gradient_policy' not in initialization and
            all(initialization.get(k) == v for k, v in OBJECTIVE_FIELDS.items() if k != 'gradient_policy'),
            'Fresh declaration intentionally has no gradient-policy selector; objective fields remain exact')
    temporal_contract(initialization, 'actual fresh initialization')
    learning_contract(initialization, 'actual fresh initialization')
    expected_init = {'kind': 'cpb_v12_fresh_initialization_v1', 'status': 'passed',
                     'initialization_kind': 'fresh-seeded-constructor-no-import-v1', 'initialization_seed': 943001,
                     'pretrained_ancestor': 'none', 'inherited_learned_parameters': 0,
                     'fresh_learned_parameter_values': 294752, 'analytic_fixed_prior_values': 3888,
                     'source_updates': 0, 'causal_updates': 0, 'old_model_artifacts_opened': 0,
                     'old_normalization_artifacts_opened': 0, 'all_initial_parameters_equal_direct_seeded_constructor': True,
                     'parameter_tensors_checked': 77, 'parameter_values_checked': 298640}
    require(all(initialization[k] == v for k, v in expected_init.items()) and
            initialization['initial_state_identity'] == initialization['direct_constructor_state_identity'],
            'Every learned parameter is fresh and matches an independent seeded constructor')
    require(verify_pin(initialization['initial_checkpoint'], training) == checkpoints['initial'], 'Fresh initialization binds measured initial checkpoint')
    calibration = load(training / 'calibration/summary.json')
    expected_calibration = {'kind': 'cpb_v12_fresh_TRAIN_calibration_v1', 'status': 'passed', 'TRAIN_anchors': 4096,
                            'history': 128, 'input_scaler_fits': 1, 'bank_normalizer_fits': 1,
                            'validation_rows_used': 0, 'test_rows_used': 0, 'encoder_calls': 0,
                            'old_normalization_artifacts_opened': 0}
    require(all(calibration[k] == v for k, v in expected_calibration.items()), 'New market statistics fitted on TRAIN only')
    for artifact in calibration['artifacts']:
        verify_pin({'path': artifact['file'], 'bytes': artifact['bytes'], 'sha256': artifact['sha256']}, training / 'calibration')
    require(calibration['input_scaler_identity'] == initialization['fresh_input_scaler_identity'] and
            calibration['bank_identity'] == initialization['fresh_bank_identity'], 'Fresh calibration used by the actual model')
    train_summary = load(training / 'training-summary.json')
    objective_contract(train_summary, 'actual main training trajectory')
    temporal_contract(train_summary, 'actual main training trajectory')
    learning_contract(train_summary, 'actual main training trajectory')
    require(train_summary['kind'] == 'causal_fresh_v12_training_summary_v1' and train_summary['status'] == 'passed' and
            train_summary['gradient_policy'] == GRADIENT_POLICY and
            train_summary['encoder_gradient_kind'] == ENCODER_GRADIENT_KIND and
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
    temporal_evidence = temporal_learning(train_summary, training)
    require(train_summary['backward_tensor'] == 'objective.total_loss', 'Only the declared composite objective backpropagates')
    for position in ('first', 'last'):
        huber, joint, total, reporting = (train_summary[position+'_'+key] for key in
            ('mean_Huber_F64','variance_NLL_F64','total_objective_F64','training_NLL'))
        require(math.isfinite(huber) and huber >= 0 and all(math.isfinite(x) for x in (joint,total,reporting)),
                'Finite first/last objective components; negative variance-NLL/total values permitted')
        close(total, huber+joint, 'First/last optimized objective sum')
        close(joint, reporting, 'Detached-error variance objective and detached reporting copy have equal forward values')
    component_path = training/'optimization-component-witnesses.json'
    component_witnesses = load(component_path)
    require(component_witnesses == train_summary['optimization_component_witnesses'] and
            [w['update'] for w in component_witnesses] == [1,2048] and
            all(w['pre_step_forward'] and w['extra_encoder_calls'] == 0 for w in component_witnesses),
            'First/last component witnesses come from the actual existing pre-step forwards')
    for witness in component_witnesses:
        verify_pin(witness, training)
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
            for field, row_field in (('mean_Huber_F64','row_mean_huber'),
                                     ('variance_NLL_F64','row_variance_nll'),
                                     ('total_objective_F64','row_total_objective')):
                require(len(q[row_field]) == n and all(math.isfinite(x) for x in q[row_field]),
                        'Complete finite per-row query objective components')
                close(q[field], statistics.mean(q[row_field]), 'Query objective-component aggregation')
            require(all(x >= 0 for x in q['row_mean_huber']), 'Huber component is nonnegative')
            close(q['total_objective_F64'], q['mean_Huber_F64']+q['variance_NLL_F64'], 'Query objective component sum')
            close(q['variance_NLL_F64'], q['Gaussian_NLL_F64'], 'Query detached-error variance-NLL and reporting-copy forward values')
            for huber, joint, total, reporting in zip(q['row_mean_huber'],q['row_variance_nll'],q['row_total_objective'],q['row_nll']):
                close(total,huber+joint,'Per-row query objective component sum')
                close(joint,reporting,'Per-row variance-NLL and detached reporting-copy values')
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
                    run / 'features' / stage / 'market' / ('cpb_v12-' + role + '.pt'), 'This checkpoint stage supplies its own market heads')
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
        require(reader['model'] == TAG and reader['checkpoint_stage'] == stage and reader['native_width'] == 40 and
                all(reader[k] == 0 for k in ('encoder_calls', 'encoder_updates', 'old_encoder_calls', 'old_head_fits', 'post_encoder_pca_fits')),
                'New saved initial/final feature-only classification fitting')
        e = load(run / 'features' / stage / 'summary.json')
        temporal_contract(e, 'saved initial/final extraction stage')
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
                rows.append({'method': label, 'size': 40, 'checkpoint_stage': stage, 'reused': False,
                    'linear_head_accuracy': scores['ridge']['mean'], 'neural_head_accuracy': scores['tiny_secondary']['mean'],
                    'coverage': scores['coverage']['mean']})
            panels.append({'dataset_id': 'TEMPO-5', 'task': 'lag_sign', 'recipe_id': 'tempo-5-slow-fast-nine-feature-v1',
                'designed_complexity_level': 5, 'complexity_scale_max': 5, 'view': view, 'input_policy': mode,
                'head_fitting_dataset_id': 'TEMPO-5', 'scoring_dataset_id': 'TEMPO-5', 'rows': rows})
            for head in HEADS:
                before, after = (classification[s][mode][view][head]['mean'] for s in STAGES)
                synthetic_changes.append({'mode': mode, 'view': view, 'head': head, 'initial': before, 'final': after, 'difference': after-before})
    result = {'kind': 'causal_fresh_v12_results_v1', 'status': 'complete', 'model': TAG, 'initial_label': INIT,
        'source_updates': 0, 'initial_quality_reused': False, 'old_encoder_calls': 0, 'historical_encoder_quality_targets_used': False,
        'runtime_trust_admission_passed': True, 'trust_protocol_admission_passed': False,
        'mandatory_postmeasurement_trust': POSTMEASUREMENT_TRUST_PENDING,
        'overall_establishment_passed': False,
        'full_coverage': full_coverage, 'clean_causal_candidate_measured': full_coverage,
        'clean_baseline': 'CPB-v3.market-01', 'clean_baseline_unchanged': True, 'scoped_reference': 'CPB-v11.market-01', 'scoped_reference_role': 'latest measured control,not best/winner; source-only parent', 'scientific_recipe_parent': 'CPB-v11.market-01', 'scientific_recipe_parent_is_source_only': True, 'pairing_reference': 'CPB-v11.market-01 saved initial DATA/statistics only', **OBJECTIVE_FIELDS, **TEMPORAL_FIELDS, **LEARNING_FIELDS,
        'quality_interpretation': 'Descriptive initial-to-final per-task measures; establishes no quality winner',
        'market_rows': raw['rows'] + measurements['initial'] + measurements['final'], 'new_market_rows': 8,
        'synthetic': {'scores': classification, 'panels': panels, 'all_saved_repetitions_and_intervals': classification_reports},
        'descriptive_changes': {'market': market_changes, 'synthetic': synthetic_changes},
        'training': train_summary, 'fixed_queries': queries, 'proofs': proofs, 'fresh_initialization': initialization,
        'optimization_component_witnesses': component_witnesses, 'temporal_learning': temporal_evidence,
        'calibration': calibration, 'post_encoder_pca_fits': 0, 'application_changed': False, 'overall_winner_declared': False,
        'card': pin(args.card), 'input_manifest': pin(args.input_manifest), 'raw_controls': pin(args.raw_controls),
        'quality_execution_receipt': pin(args.quality_receipt), 'variance_gradient_proof': pin(gradient_path),
        'temporal_pooling_proof': temporal_fixture['proof'], 'temporal_mechanism_bindings': temporal_fixture}
    output = args.output.resolve()
    require(not output.exists(), 'Preserve all report attempts; exclusive new output required')
    output.mkdir(parents=True)
    dump(output / 'table_rows.json', result)
    assets = {'kind': 'causal_fresh_v12_report_assets_v1', 'run_assets': [pin(p) for p in sorted(run.rglob('*')) if p.is_file()],
              'training_summary': pin(training / 'training-summary.json'), 'initialization': pin(initialization_path),
              'calibration': pin(training / 'calibration/summary.json'), 'typed_resume': pin(training / 'typed-resume-512.json'),
              'query_initial': pin(training / 'query-initial.json'), 'query_final': pin(training / 'query-final.json'),
              'optimization_component_witness_receipt': pin(component_path),
              'optimization_component_witnesses': component_witnesses,
              'proofs': proofs, 'engineering_proof': pin(engineering_path), 'build_admission': pin(build_path),
              'variance_gradient_proof': pin(gradient_path),
              'temporal_pooling_proof': temporal_fixture, 'temporal_learning': temporal_evidence,
              'reporter': pin(__file__), 'no_old_model_or_tensor_copies': True}
    dump(output / 'asset_hashes.json', assets)
    lines = ['# CPB-v12: fresh causal training and fixed-head measurements', '',
        'CPB-v12 starts every learned tensor from its declared fresh constructor, with source updates0 and '
        'new TRAIN-only input/bank statistics. Initial0 and final2048 embeddings and heads are newly measured. '
        'Pretrained historical encoders supply no quality target, floor, teacher, weights or fitted statistics.', '',
        'Runtime admission covers fresh ancestry, actual CUDA learning, typed resume and saved initial/final causal proofs. '
        'Complete finite fixed-head results describe a measured clean causal candidate; CPB-v3 remains the clean baseline. '
        'The mandatory initial native20:40 DATA/five whole-statistic pairings and independent training/quality audits remain pending in this producer. '
        'Overall protocol trust and establishment remain false/pending until those are independently bound. '
        'The separate comparison evaluates the 22 numeric conditions and coverage independently. No overall winner or automatic promotion is declared.', '',
        'V12 copies the V11 SOURCE shape with all76 learned tensors and TRAIN statistics fresh. Each current-cutoff original bin keeps its first4 attention-mean exports. The last4 rows now project uniform-centered componentwise variance using fixed half-local/half-global weights over retained ORIGINAL patch support and its OWN live two-sum uniform center. Attention mean and Transformer attention remain live; dispersion has zero direct score/position derivative. Empty local bins zero the whole8-row slice after bias. '
        'The intermediate32-coordinate summary contains16 mean-derived and16 variance-derived rows,then enters the unchangedglobalMLP to nativecontent20. Existing native40 remains content20+literal spectral12+mean4+uncertainty4;the centered pool variance is not decoder conditional variance or appended native uncertainty coordinates. '
        'Parameter shapes/registration/draw order/native40/fixed D40 heads retain the V11 declaration; constructor parity needs its own saved proof. Routing, scales, FLOPs, shared clipping and optimizer trajectories can change. Initial native20:40 and five whole TRAIN-statistic archives pair against saved V11 DATA; content0:20 may change. Both initial/final heads are newly fitted; no old weights or fitted statistics are reused. '
        'Unit Huber(delta1) mean loss plus detached-error Gaussian variance calibration restores the V9 objective. Gaussian error has no direct mean-loss route,while sharedglobalclip1/singleAdamW and future residual changes still couple trajectories. The separate reporting NLL copy stays detached;only objective.total_loss backpropagates. Negative NLL/total values are possible. '
        '75 identifiable tensors require actual data-gradient/movement events;294751 is tensor-value capacity,not every-scalar proof. The sole common pool-score bias gauge remains registered and observed. Four whole8-row groups,eight separate mean4/centeredvariance4 subgroups,four centered position groups and three Hadamard directions require their declared events/finalchanges,not every scalar or update. Weightdecay is not data-learning. '
        'V11 is the latest measured historical control,not a best-model/winner claim. All scoped leaders remain metric-specific. No validation-error cause,generalization or improvement is established by this architectural hypothesis. Complexity4/5 and5/5 are ordinal recipes,not accuracy or observed difficulty.', '',
        'One CUDA:0 trajectory uses2048 updates/B8. Four market outputs use fixed ridge1 and tanh16/Adam0.01/100 updates, '
        'TRAIN112 / known chronological VALIDATION32. Repetitions2701/2802/2903 retain the exact D40 seeds, '
        'with164/724 head parameters. Both stages fit their own heads; there is no initial score reuse or post-encoder PCA. '
        'Raw controls are reused encoder-free results at their original larger input sizes. Their original head sizes are '
        '(D+1)×4 linear and16×(D+1)+68 neural. These targets measure retained past-window information, not forecasting.', '']
    for task in TASKS:
        rows = [r for r in result['market_rows'] if r['task'] == task]
        lines.extend(['## ' + task.removeprefix('past_').capitalize(), ''])
        table(lines, MKT + ' Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.',
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
        'Classification heads contain82/690 parameters. Synthetic TRAIN input statistics are freshly fitted once per cohort '
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
        'Gaussian NLL is an unchanged reporting metric and can be negative; the optimized total is Huber1 plus detached-error Gaussian variance calibration. The synchronized step timer covers forward/objective/backward/clip/AdamW '
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
    table(lines, MKT + ' Same fixed query masks; descriptive objective components, not additional advancement criteria. Detached-error variance NLL and total may be negative.',
          ('Encoder','Split','Mean Huber1','Variance NLL','Total objective'),
          [(INIT if s=='initial' else TAG,role,*(format(queries[s][role][key],'.17g') for key in
            ('mean_Huber_F64','variance_NLL_F64','total_objective_F64'))) for s in STAGES for role in ('train','validation')])
    table(lines, MKT + ' and ' + T5 + ' One-run descriptive cost; components are not interchangeable.',
          ('Stage', 'Seconds'), [('Fresh market TRAIN calibration including input/witness I/O', format(train_summary['calibration_seconds'], '.17g')),
                               ('Initial+final export including fresh synthetic statistics and input/state I/O', format(quality['stage_seconds']['extraction'], '.17g'))] +
          [(s + ' market CPU fixed heads and I/O', format(quality['stage_seconds'][s + '_market_heads'], '.17g')) for s in STAGES] +
          [(s + ' synthetic CPU fixed heads/scoring and I/O', format(quality['stage_seconds'][s + '_synthetic_heads'], '.17g')) for s in STAGES])
    lines.extend(['## Interpretation and next direction', '',
        'Each initial-to-final task/head difference is recorded at full precision in `table_rows.json`. '
        'The frozen protocol measures whether fresh causal training changes useful information; it does not require retention '
        'of any pretrained encoder’s scores. Mixed or weak quality remains a valid reported candidate, with no head or checkpoint rescue. '
        'The frozen 22-condition comparison protects all clean V3/V4/V5/V6/V7/V8/V9/V10/V11 final scopes separately; CPB-v3 remains selected. '
        'Keep this run and every earlier artifact; use a separately declared mechanism for any later improvement. '
        'The application selection is unchanged and no overall quality winner is declared.', ''])
    write_new(output / 'RESULTS.md', '\n'.join(lines) + '\n')
    dump(output / 'completion.json', {'kind': 'causal_fresh_v12_report_completion_v1', 'status': 'complete', 'model': TAG,
        'runtime_trust_admission_passed': True, 'trust_protocol_admission_passed': False,
        'mandatory_postmeasurement_trust': POSTMEASUREMENT_TRUST_PENDING,
        'overall_establishment_passed': False,
        'full_coverage': full_coverage, 'clean_causal_candidate_measured': full_coverage,
        'clean_baseline': 'CPB-v3.market-01', 'clean_baseline_unchanged': True, 'scoped_reference': 'CPB-v11.market-01', 'scoped_reference_role': 'latest measured control,not best/winner; source-only parent', 'scientific_recipe_parent': 'CPB-v11.market-01', 'scientific_recipe_parent_is_source_only': True, 'pairing_reference': 'CPB-v11.market-01 saved initial DATA/statistics only', **OBJECTIVE_FIELDS, **TEMPORAL_FIELDS, **LEARNING_FIELDS,
        'historical_encoder_quality_targets_used': False, 'overall_winner_declared': False, 'new_market_rows': 8, 'synthetic_panels': 4,
        'reporting_seconds': time.monotonic()-started, 'report': pin(output / 'RESULTS.md'),
        'tables': pin(output / 'table_rows.json'), 'assets': pin(output / 'asset_hashes.json')})


if __name__ == '__main__':
    main()



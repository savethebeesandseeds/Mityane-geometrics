#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Report one admitted CPB-v2 final quality pass using saved features/readouts only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

TAG = 'CPB-v2.market-01'
BASE = 'CPB-v1.market-01'
BASE_TABLE_SHA = '905c2fcd69e8c0d513db589213a5c1e4da67b51407d2fcea0bcfcf389f61c6f6'
BASE_CHECKPOINT_SHA = '54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895'
CARD_SHA = '30aa70ce6f3999d9f6a991c1b298516db41caf3db276cacb75d91c9d3dcba222'
MARKET_MANIFEST_SHA = '828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8'
REPS = (2701, 2802, 2903)
SEEDS = (13088411117524848790, 10018485049408248344, 9083751155097190455)
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
MASTERS = (941101, 941102, 941103)
MODES = ('synthetic_train', 'market_scaler')
VIEWS = ('validation_intact', 'validation_deleted')
HEADS = ('ridge', 'tiny_secondary')
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


def market_rows(path, card_sha, method_name):
    report = load(path)
    require(report['kind'] == 'fixed_past_window_regression_v1' and
            report['dataset_id'] == 'MKT-HIST-2' and
            report['card_sha256'] == card_sha and report['input_manifest_sha256'] == MARKET_MANIFEST_SHA and
            (report['training_rows'], report['validation_rows']) == (112, 32), 'Fixed market population/card identity')
    expected = {'ridge_penalty': 1, 'neural_hidden': 16, 'neural_learning_rate': .01,
                'neural_updates': 100, 'outputs': 4, 'loss': 'masked TRAIN-target-standardized MSE',
                'device': 'CPU (frozen readouts only)'}
    require(all(report['heads'][k] == v for k, v in expected.items()), 'Fixed market recipe')
    require(len(report['methods']) == 1 and report['methods'][0]['name'] == method_name, 'Only final candidate market heads')
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
        row = {'dataset_id': 'MKT-HIST-2', 'task': task, 'recipe_id': 'causal-finetune-v2',
               'designed_complexity_level': 4, 'complexity_scale_max': 5,
               'view': 'intact chronological VALIDATION', 'evidence_stage': 'known-source development',
               'method': TAG, 'size': 36, 'unit': targets[0]['unit'], 'linear_r2': ridge['r2'],
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


def decisions(card, baseline_table, new_market, new_synthetic, all_synthetic_populations_full):
    criteria = card['criteria']
    tolerance = criteria['arithmetic_absolute_tolerance']
    require(tolerance == 1e-12 and criteria['all_three_neural_trend_positive'] and
            not criteria['posthoc_rescue'] and criteria['checkpoint_selection'] == 'final2048 only',
            'Prospective unchanged gate/selection policy')
    old = {(x['method'], x['task']): x for x in baseline_table['market_rows']}
    new = {x['task']: x for x in new_market}
    def market_value(name):
        if name == 'positive_linear_r2_trend' or name == 'linear_trend_exceeds_saved_v23':
            return new['past_trend']['linear_r2']
        if name == 'positive_neural_mean_r2_trend':
            return new['past_trend']['neural_mean_r2']
        task = next((t for t in TASKS if name.startswith(t + '_')), None)
        require(task is not None, 'Unknown prospective market criterion: ' + name)
        key = name[len(task)+1:]
        for suffix in ('_v22_retention', '_native36_guard'):
            if key.endswith(suffix):
                key = key[:-len(suffix)]
        require(key in ('linear_r2', 'neural_mean_r2'), 'Unknown market metric: ' + key)
        return new[task][key]
    original, strict, preservation, synthetic = [], [], [], []
    for declared in criteria['original9']:
        name, floor = declared['criterion'], declared['floor']
        value = market_value(name)
        positive = name.startswith('positive_') or name == 'linear_trend_exceeds_saved_v23'
        original.append({**declared, 'candidate': value, 'passed': value > floor if positive else value+tolerance >= floor})
    for declared in criteria['strict_native36_8']:
        value = market_value(declared['criterion'])
        strict.append({**declared, 'candidate': value, 'passed': value+tolerance >= declared['floor']})
    require(len(original) == 9 and len(strict) == 8, 'Complete declared nine/eight gate populations')
    coverage = all(x['coverage'] == 1 and x['valid'] == x['total'] == 32 for x in new_market)
    for declared in criteria['causal_baseline_market']:
        task = declared['task']
        for key, floor_name in (('linear_r2', 'linear_floor'), ('neural_mean_r2', 'neural_mean_floor')):
            floor, value = declared[floor_name], new[task][key]
            close(floor, old[BASE, task][key], 'Declared causal market floor matches immutable baseline')
            preservation.append({'criterion': task+'_'+key, 'floor': floor, 'candidate': value,
                                 'passed': value+tolerance >= floor})
        require(declared['baseline_neural_repetitions'] == old[BASE, task]['neural_r2_each'], 'Declared baseline individual neural values')
    require(len(preservation) == 8, 'All causal baseline market scores protected')
    robust = [{'repetition': rep, 'candidate': value, 'passed': value > 0}
              for rep, value in zip(REPS, new['past_trend']['neural_r2_each'])]
    improvement_floor = criteria['strictly_improve_mean_neural_trend']
    close(improvement_floor, old[BASE, 'past_trend']['neural_mean_r2'], 'Strict trend baseline value')
    improvement = new['past_trend']['neural_mean_r2'] > improvement_floor
    synthetic_coverage = all_synthetic_populations_full
    for declared in criteria['synthetic_max_baseline_and_native36']:
        mode, view, head = (declared[k] for k in ('mode', 'view', 'head'))
        require(mode in MODES and view in VIEWS and head in HEADS, 'Declared synthetic panel/head')
        close(declared['baseline'], baseline_table['synthetic']['scores'][mode][view][head]['mean'], 'Declared synthetic causal floor')
        close(declared['floor'], max(declared['baseline'], declared['native36_floor']), 'Prospective stronger floor')
        value = new_synthetic[mode][view][head]['mean']
        synthetic_coverage &= new_synthetic[mode][view]['coverage']['mean'] == declared['coverage_floor'] == 1
        synthetic.append({**declared, 'candidate': value, 'passed': value+tolerance >= declared['floor']})
    require(len(synthetic) == 8, 'All eight synthetic head/policy/view floors')
    groups = {'original_market': original, 'strict_native36_market': strict,
              'causal_baseline_market': preservation, 'individual_neural_trend': robust, 'synthetic_retention': synthetic}
    return {'conditions': groups, 'counts': {k: {'passed': sum(x['passed'] for x in v), 'total': len(v)} for k, v in groups.items()},
            'neural_trend_mean_strict_improvement': improvement,
            'full_market_coverage': coverage, 'full_synthetic_coverage': synthetic_coverage,
            'overall_gate_passed': coverage and synthetic_coverage and improvement and
                all(x['passed'] for group in groups.values() for x in group), 'absolute_tolerance': tolerance}


def table(lines, legend, columns, rows):
    lines.extend([legend, '', '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |'])
    lines.extend('| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |' for row in rows)
    lines.append('')


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'training', 'data-root', 'card', 'initial-parity', 'quality-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'Existing managed container only')
    run, training, data = (p.resolve(strict=True) for p in (args.run, args.training, args.data_root))
    baseline_path = data / 'doc/results/causal_patch_bottleneck_mae_v1/table_rows.json'
    require(sha(baseline_path) == BASE_TABLE_SHA, 'Immutable causal baseline result')
    baseline = load(baseline_path)
    parity, quality = load(args.initial_parity), load(args.quality_receipt)
    require(parity['status'] == 'passed' and quality['status'] == 'passed', 'Passed initial parity and admitted one-pass quality receipts')
    require(quality['protocol'] == 'causal-finetune-v2' and quality['model'] == TAG, 'Admitted final quality protocol/model')
    final_checkpoint = verify_pin(quality['final_checkpoint'], args.quality_receipt.parent)
    require(final_checkpoint == training / 'checkpoints/inference-2048.pt', 'Quality scores the declared final checkpoint')
    post_proof_path = verify_pin(quality['post_training_proof'], args.quality_receipt.parent)
    post_proof = load(post_proof_path)
    require(post_proof['kind'] == 'cpb_v2_final2048_causal_proof_v1' and post_proof['status'] == 'passed' and
            post_proof['model'] == 'CPB-v2' and post_proof['baseline_checkpoint_sha256'] == BASE_CHECKPOINT_SHA and
            post_proof['source_updates'] == post_proof['causal_updates'] == 2048 and post_proof['new_training_updates'] == 0 and
            post_proof['all_cutoffs'] == 128 and post_proof['suffix_cases'] == 2350 and
            post_proof['internal_temporal_cases'] == 30 and post_proof['state_unchanged'] and
            post_proof['future_values_and_masks_invariant'] and post_proof['CPU_encoder_calls'] == post_proof['head_fits'] == 0 and
            post_proof['CUDA_encoder_calls'] > 0 and post_proof['checks'] > 0,
            'Actual final2048 causal proof with no new training or head fits')
    require(Path(post_proof['final_checkpoint']).resolve(strict=True) == final_checkpoint and
            post_proof['final_checkpoint_sha256'] == quality['final_checkpoint']['sha256'], 'Proof and quality use the same final bytes')
    require(verify_pin(quality['initial_parity'], args.quality_receipt.parent) == args.initial_parity.resolve(strict=True),
            'Quality receipt binds this exact initial-parity evidence')
    require(parity['baseline_checkpoint_sha256'] == BASE_CHECKPOINT_SHA and parity['all_initial_weights_exact'] and
            parity['all_initial_scaler_bank_exact'] and parity['initial_feature_witness_exact'] and
            parity['initial_saved_features_reused'], 'Exact frozen baseline initialization and saved initial feature reuse')
    verify_pin(parity['initial_checkpoint'], args.initial_parity.parent)
    require(parity['rows_checked'] == 144 and parity['native_coordinates_checked'] == 5184 and
            parity['CUDA_parity_calls'] == 18 and parity['new_initial_head_fits'] == parity['old_encoder_calls'] == 0,
            'All saved initial market rows/coordinates compared without baseline replay/refits')
    require(len(parity['witnesses']) == 2, 'Both initial TRAIN/VALIDATION witnesses')
    for witness in parity['witnesses']:
        verify_pin(witness, args.initial_parity.parent)
    verify_pin(parity['feature_witness'], args.initial_parity.parent)
    # The prospective card is hash-bound to the new market fixed-head report.
    card_sha = sha(args.card)
    require(card_sha == CARD_SHA, 'Frozen prospective machine-card identity')
    card = load(args.card)
    require(card['kind'] == 'causal_encoder_finetune_v2_prospective_card' and card['instance'] == TAG and
            card['baseline'] == BASE and card['geometry']['native_width'] == 36 and
            card['training']['updates'] == 2048 and card['training']['batch_size'] == 8 and
            tuple(card['fixed_heads']['market']['neural_repetitions']) == REPS and
            tuple(int(x) for x in card['fixed_heads']['market']['dimension36_seeds']) == SEEDS,
            'Declared model, budget, geometry and head seeds')
    new_market, market_report = market_rows(run / 'market-readouts/report.json', card_sha, 'cpb_v2')
    old_market_report = load(data / 'output/runs/causal-patch-bottleneck-mae-v1/run-001/market-readouts/report.json')
    for new_rep, old_rep in zip(market_report['methods'][0]['repetitions'], old_market_report['methods'][0]['repetitions']):
        for new_target, old_target in zip(new_rep['targets'], old_rep['targets']):
            require(all(new_target[k] == old_target[k] for k in ('name', 'unit', 'train_target_mean', 'train_target_scale')),
                    'Exact saved TRAIN target scalers')
    new_reports = [load(run / 'synthetic-readouts' / ('master-' + str(m)) / 'report.json') for m in MASTERS]
    for report, master in zip(new_reports, MASTERS):
        validate_classification(report, master)
    all_synthetic_populations_full = all(
        rep[view]['population']['valid_rows'] == rep[view]['population']['total_rows'] and
        rep[view]['population']['valid_source_groups'] == rep[view]['population']['total_source_groups']
        for report in new_reports for method in report['methods'] for rep in method['repetitions']
        for view in ('training',) + VIEWS)
    scores = aggregate_classification(new_reports)
    reader = load(run / 'synthetic-readouts/reader-summary.json')
    require(reader['model'] == TAG and reader['native_width'] == 36, 'Final candidate synthetic reader identity')
    require(all(reader[k] == 0 for k in ('encoder_calls', 'encoder_updates', 'old_encoder_calls', 'old_head_fits', 'post_encoder_pca_fits')),
            'Saved feature reader has no forbidden encoder work')
    extraction = load(run / 'features/summary.json')
    require(extraction['model'] == TAG and extraction['device'] == 'cuda:0' and
            extraction['full_CUDA_encode_calls'] == 786 and extraction['synthetic_rows'] == 6144 and
            extraction['market']['train']['full_CUDA_encode_calls'] == 14 and
            extraction['market']['validation']['full_CUDA_encode_calls'] == 4 and
            extraction['all_parameters_buffers_normalizer_scaler_unchanged'] and extraction['all_gradients_undefined'],
            'One final frozen CUDA extraction pass with exact populations')
    require(all(extraction[k] == 0 for k in ('encoder_updates', 'decoder_calls', 'old_encoder_calls', 'new_input_scaler_fits', 'new_bank_normalizer_fits')),
            'Extraction fits no encoder/scaler/normalizer')
    train_summary = load(training / 'training-summary.json')
    query_initial, query_final = load(training / 'query-initial.json'), load(training / 'query-final.json')
    require(train_summary['status'] == 'passed' and train_summary['model'] == TAG and
            train_summary['source_updates'] == train_summary['new_causal_updates'] == 2048 and
            train_summary['batch_size'] == 8 and train_summary['TRAIN_windows'] == 4096 and
            train_summary['device'] == 'cuda:0' and train_summary['typed_resume_512_exact'] and
            train_summary['fixed_prior_scaler_normalizer_unchanged'], 'Actual admitted CUDA trajectory/resume')
    require(all(train_summary[k] == 0 for k in ('new_scaler_fits', 'new_normalizer_fits', 'CPU_encoder_calls',
                                               'old_encoder_calls', 'head_fits', 'training_replays')), 'No forbidden training work')
    require(train_summary['fixed_query_initial'] == query_initial and train_summary['fixed_query_final'] == query_final,
            'Training summary exactly retains saved query measurements')
    for query in (query_initial, query_final):
        for role, count in (('train', 112), ('validation', 32)):
            value = query[role]
            require(value['valid_rows'] == count and len(value['row_mae']) == len(value['row_nll']) == count and
                    len(value['row_target_cells']) == count and value['target_cells'] > 0,
                    'Fixed standardized query support/rows')
            close(value['standardized_fixed_query_MAE'], statistics.mean(value['row_mae']), 'Fixed-query MAE aggregation')
            close(value['Gaussian_NLL_F64'], statistics.mean(value['row_nll']), 'Separate fixed-query NLL aggregation')
            require(sum(value['row_target_cells']) == value['target_cells'], 'Fixed-query target-cell accounting')
    gates = decisions(card, baseline, new_market, scores, all_synthetic_populations_full)
    old_rows = [dict(x, reused=True, new_head_measurement=False) for x in baseline['market_rows']]
    initial_rows = [dict(x, method=TAG, reused=True, checkpoint_stage='initial',
                         initialization_reuse_via_verified_native_parity=True) for x in old_rows if x['method'] == BASE]
    panels = []
    for mode in MODES:
        for view in VIEWS:
            old_panel = next(x for x in baseline['synthetic']['panels'] if x['input_policy'] == mode and x['view'] == view)
            rows = [dict(x, evidence='reused saved baseline/reference') for x in old_panel['rows']]
            rows.append({'method': TAG, 'size': 36, 'evidence': 'new final saved-feature fixed-head fits',
                         'encoder_fitting_dataset_id': 'MKT-HIST-2',
                         'linear_head_accuracy': scores[mode][view]['ridge']['mean'],
                         'neural_head_accuracy': scores[mode][view]['tiny_secondary']['mean'],
                         'coverage': scores[mode][view]['coverage']['mean']})
            panels.append({**{k: v for k, v in old_panel.items() if k != 'rows'}, 'rows': rows})
    result = {'kind': 'causal_finetune_v2_results_v1', 'status': 'complete', 'model': TAG, 'baseline': BASE,
              'market_rows': old_rows + new_market, 'new_market_rows': 4, 'initial_market_quality_rows_reused': initial_rows,
              'synthetic': {'scores': scores, 'panels': panels, 'all_final_repetitions_and_intervals': new_reports,
                            'all_baseline_repetitions_and_intervals': baseline['synthetic']['all_saved_repetitions_and_intervals']},
              'gates': gates, 'training': train_summary, 'query_initial': query_initial, 'query_final': query_final,
              'initial_quality_reused': True, 'initial_quality_reused_from': BASE, 'initial_head_refits': 0, 'old_encoder_calls': 0,
              'post_encoder_pca_fits': 0, 'primary_changed': False, 'baseline_replaced': False,
              'card': pin(args.card), 'initial_parity': pin(args.initial_parity), 'quality_receipt': pin(args.quality_receipt)}
    output = args.output.resolve()
    require(not output.exists(), 'Exclusive new report directory; preserve every earlier attempt')
    output.mkdir(parents=True)
    dump(output / 'table_rows.json', result)
    assets = [pin(p) for p in sorted(run.rglob('*')) if p.is_file()]
    dump(output / 'asset_hashes.json', {'kind': 'causal_finetune_v2_report_assets_v1', 'run_assets': assets,
        'baseline': pin(baseline_path), 'training_summary': pin(training / 'training-summary.json'),
        'query_initial': pin(training / 'query-initial.json'), 'query_final': pin(training / 'query-final.json'),
        'reporter': pin(Path(__file__)), 'post_training_proof': pin(post_proof_path),
        'post_training_causal_witness': pin(post_proof_path.parent / 'final2048-causality.json'),
        'no_baseline_tensor_copies': True})
    lines = ['# CPB-v2: causal fine-tuning and fixed-head results', '',
             '**CPB-v1.market-01 remains the frozen application baseline.** CPB-v2 is a separately named research candidate. '
             'The baseline initial embeddings and head scores are reused after exact initial parity, with zero initial refits. '
             'Only the final checkpoint receives one new quality extraction and fixed-head pass.', '',
             'Market fitting: TRAIN112 / known VALIDATION32, CUDA:0 batch8, native36; '
             'one ridge pipeline/four target fits and three tanh16/Adam0.01/100-update neural fits. '
             'Repetitions2701/2802/2903 retain exact D36 seeds; regression heads148/660 parameters. '
             'The targets measure retained past-window information, not forecasting.', '']
    display = ('Raw data — no encoder (recent128)', 'RPB-v24.market-01.alt-01', BASE, TAG)
    rows_by_key = {(r['method'], r['task']): r for r in result['market_rows']}
    for task in TASKS:
        rows = [rows_by_key[m, task] for m in display if (m, task) in rows_by_key]
        lines.extend(['## ' + task.replace('past_', '').capitalize(), ''])
        table(lines, MKT + ' Intact chronological VALIDATION; R² is dimensionless.',
              ('Method', 'Size', 'Linear R²', 'Neural mean R²', 'Coverage %'),
              [(r['method'], r['size'], format(r['linear_r2'], '.8f'), format(r['neural_mean_r2'], '.8f'), format(100*r['coverage'], '.2f')) for r in rows])
        candidate = next(x for x in new_market if x['task'] == task)
        table(lines, MKT + ' Physical prediction errors; units: ' + candidate['unit'] + '.',
              ('Method', 'Linear MAE', 'Linear RMSE', 'Neural mean MAE', 'Neural mean RMSE'),
              [(r['method'], *(format(r[k], '.17g') for k in ('linear_mae', 'linear_rmse', 'neural_mean_mae', 'neural_mean_rmse'))) for r in rows])
        lines.extend(['Every final neural R², repetitions2701/2802/2903: ' +
                      ', '.join(format(x, '.17g') for x in candidate['neural_r2_each']) + '.', ''])
    lines.extend(['## Hard timing retention', '',
                  'TEMPO-5 uses cohorts941101/941102/941103, each512 TRAIN variants from256 sources and256 VALIDATION variants '
                  'from128 distinct sources. Both saved input-scaler policies are frozen. Ridge1 and neural tanh16/Adam0.01/100 '
                  'retain three D36 seeds per cohort/policy; classification heads74/626 parameters. No synthetic encoder training occurs.', ''])
    for panel in panels:
        table(lines, T5 + ' ' + panel['input_policy'] + ', ' + panel['view'] + '.',
              ('Method', 'Size', 'Linear head %', 'Neural head %', 'Coverage %'),
              [(r['method'], r['size'], *(format(100*r[k], '.4f') for k in ('linear_head_accuracy', 'neural_head_accuracy', 'coverage'))) for r in panel['rows']])
    lines.extend(['## Every final synthetic repetition', ''])
    for mode in MODES:
        repetitions = []
        for master, report in zip(MASTERS, new_reports):
            method = next(m for m in report['methods'] if m['method'] == mode)
            for rep in method['repetitions']:
                for view in VIEWS:
                    value = rep[view]
                    repetitions.append((master, rep['id'], view,
                        format(value['ridge']['accuracy'], '.17g'), format(value['tiny_secondary']['accuracy'], '.17g'),
                        format(value['population']['coverage'], '.17g')))
        table(lines, T5 + ' ' + mode + ', intact/deleted VALIDATION; exact accuracy fractions.',
              ('Cohort', 'Fit', 'View', 'Linear accuracy', 'Neural accuracy', 'Coverage'), repetitions)
    lines.extend(['## Training, fixed queries and cost', '',
                  'The complete training/query measurements, including standardized fixed-query MAE and separate Gaussian NLL, '
                  'are retained without substitution in `table_rows.json`. Per-update trajectories stay in the training artifacts. '
                  'Encoder training time, frozen extraction time and CPU head fitting/scoring time are reported separately. '
                  'Initial/final rows count new causal updates; inherited source updates2048 are separate. '
                  'The synchronized GPU step measurement is wall time around forward/NLL/backward/clip/AdamW and includes '
                  'in-step contract checks; loading, masks, checkpoints and evidence logging are excluded.', ''])
    table(lines, MKT + ' Unchanged fixed hidden-target query; errors are standardized MAE.',
          ('Encoder', 'Updates', 'Train error ↓', 'Validation error ↓', 'GPU training seconds'),
          [(TAG, 0, format(query_initial['train']['standardized_fixed_query_MAE'], '.17g'),
            format(query_initial['validation']['standardized_fixed_query_MAE'], '.17g'), 0),
           (TAG, 2048, format(query_final['train']['standardized_fixed_query_MAE'], '.17g'),
            format(query_final['validation']['standardized_fixed_query_MAE'], '.17g'),
            format(train_summary['GPU_step_seconds_synchronized'], '.17g'))])
    table(lines, MKT + ' Same saved query; Gaussian waveform NLL is distinct from MAE and can be negative.',
          ('Encoder', 'Updates', 'TRAIN Gaussian NLL', 'VALIDATION Gaussian NLL'),
          [(TAG, 0, format(query_initial['train']['Gaussian_NLL_F64'], '.17g'),
            format(query_initial['validation']['Gaussian_NLL_F64'], '.17g')),
           (TAG, 2048, format(query_final['train']['Gaussian_NLL_F64'], '.17g'),
            format(query_final['validation']['Gaussian_NLL_F64'], '.17g'))])
    table(lines, MKT + ' and ' + T5 + ' Final frozen extraction and head-scoring cost; one run, descriptive timing.',
          ('Stage', 'Seconds'), [('Final CUDA extraction, full API including input/state checks', format(extraction['elapsed_seconds_including_input_io_and_state_checks'], '.17g')),
                               ('Market CPU heads/scoring including process and I/O', format(quality['stage_seconds']['market_heads'], '.17g')),
                               ('Synthetic CPU heads/scoring including I/O', format(reader['head_fitting_and_scoring_seconds_including_io'], '.17g'))])
    lines.extend(['## Fixed prospective gates', '', '```json', json.dumps(gates, indent=2, allow_nan=False), '```', '',
                  '## What changed and next direction', '',
                  'The candidate performs separately declared causal fine-tuning from the preserved saved baseline. '
                  'Its native size, input populations, scalers and fixed-head recipes remain fixed. All eight market baseline '
                  'floors and the stronger of each causal/native36 synthetic floor are reported, alongside the historical gates. '
                  'No application pointer or historical leader is changed by this report. '
                  'A failed frozen recipe is retained and stopped; a later mechanism requires a separately declared version/card.', ''])
    write_new(output / 'RESULTS.md', '\n'.join(lines) + '\n')
    dump(output / 'completion.json', {'kind': 'causal_finetune_v2_report_completion_v1', 'status': 'complete',
        'model': TAG, 'overall_gate_passed': gates['overall_gate_passed'], 'baseline_replaced': False,
        'new_market_task_rows': 4, 'synthetic_panels': 4, 'reporting_seconds': time.monotonic()-started,
        'report': pin(output / 'RESULTS.md'), 'tables': pin(output / 'table_rows.json'), 'assets': pin(output / 'asset_hashes.json')})


if __name__ == '__main__':
    main()

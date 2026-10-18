#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independently check saved v30 scalar witnesses; no models, tensors or fits."""
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT = Path('/cuwacunu')
CHECKS = 0
ERRORS = {}


def require(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text())


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def near(actual, expected, label, tolerance=1e-10):
    difference = abs(actual - expected) / (1 + abs(expected))
    ERRORS[label] = max(ERRORS.get(label, 0.), difference)
    require(math.isfinite(actual) and difference <= tolerance,
            'Independent arithmetic mismatch: ' + label)


def counter(base, attempt):
    mask = (1 << 64) - 1
    x = (base + 0x9e3779b97f4a7c15 * (attempt + 1)) & mask
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
    return (x ^ (x >> 31)) & ((1 << 63) - 1)


def matrix(value, rows, width, name):
    require(len(value) == rows and all(len(row) == width for row in value),
            'Witness shape: ' + name)


def calibration(run, summary):
    reuse = load(run / 'calibration-reuse.json')
    require(reuse['new_bank_normalizer_fits'] == reuse['new_calibration_encoder_calls'] ==
            reuse['new_calibration_encoder_updates'] == 0 and reuse['TRAIN_anchors'] == 4096,
            'No new calibration fit/encoder execution')
    require(reuse['archive_sha256'] == sha(run / 'bank-standardizer.pt') ==
            'c3e4f9e850ebbb355ae58e8b434173e4db0d1ff98055c844af4fe16ba7879fbc' and
            reuse['statistics_sha256'] == sha(run / 'bank-standardizer.json') ==
            '782128503036956beb1ec87e39ceb3c502395f00f1d0e8a6138e9d8e52d933ee',
            'Exact reused frozen bank archive/statistics')
    old = ROOT / 'output/runs/market-relative-changes/run-vaPC4X'
    for filename, key, expected in (
            ('raw-relative-bank-witness.json', 'original_raw_bank_witness_sha256',
             '8abc4845192236def58b2727b02d60e816ecca21c501c4f742a1776b23273fa8'),
            ('raw-loader-record-witness.json', 'original_record_witness_sha256',
             '804606752fd991e693cc558808dfab53ebd5a741891559e1ecc41dabdf4969cc')):
        require(reuse[key] == sha(old / filename) == expected, 'Bound original raw precision witness')
    require(reuse['source_run'].rstrip('/').endswith('market-relative-changes/run-vaPC4X') and
            reuse['original_audit_sha256'] == sha(old / 'independent-audit.json') ==
            'b8278365f60f95eb1a611b8ed34fe4a144a77be59de686e1891eda9e30b25d97' and
            reuse['original_audit_checks'] == 1549413 and
            load(old / 'independent-audit.json')['status'] == 'passed', 'Verified original bank arithmetic bound')
    normalizer = load(run / 'bank-standardizer.json')
    require(reuse['identity'] == normalizer['identity'] == summary['bank_normalizer_identity'] ==
            'rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476',
            'Unchanged normalizer identity')
    first, end = 1704067259999+255*60000, 1705881599999
    available = (end-first)//60000+1
    anchors = [first+(j*(available-1)//4095)*60000 for j in range(4096)]
    require(reuse['TRAIN_anchor_ids'] == anchors, 'Unchanged legal uniform TRAIN anchor plan')
    require(not (run / 'calibration-bank.pt').exists() and
            not (run / 'calibration-witness.json').exists(), 'No new calibration population generated')
    return normalizer, set(anchors)


def trace(run, summary, anchors):
    with (run / 'training.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == summary['attempted_batches'] <= 4096, 'Trace attempts and skip cap')
    flags = ('projection_gradient_nonzero', 'wave_decoder_gradient_nonzero',
             'variance_matrix_gradient_nonzero', 'variance_bias_gradient_nonzero',
             'projection_weight_changed', 'wave_decoder_weight_changed',
             'variance_matrix_weight_changed', 'variance_bias_weight_changed')
    totals = {key: 0 for key in ('eligible_examples', 'target_cells', *flags)}
    active, updates = [], 0
    for attempt, row in enumerate(rows):
        require(int(row['attempt']) == attempt and
                int(row['row_seed']) == counter(942002, attempt) and
                int(row['mask_seed']) == counter(942003, attempt), 'Absolute counter streams')
        selected = [int(value) for value in row['anchors'].split(';')]
        require(len(selected) == 8 and all(value in anchors for value in selected),
                'B8 legal TRAIN presentations')
        if not row['loss']:
            require(int(row['update']) == updates and all(int(row[key]) == 0 for key in totals),
                    'Skipped batch preserves counters with no targets/updates')
            continue
        active.append(row)
        updates += 1
        require(int(row['update']) == updates and math.isfinite(float(row['loss'])),
                'Finite ordered Gaussian NLL, including legitimate negative values')
        require(0 < int(row['eligible_examples']) <= 8 and
                0 < int(row['target_cells']) <= 8 * 3 * 128 * 9, 'Target eligibility')
        for key in totals:
            value = int(row[key])
            if key in flags:
                require(value in (0, 1), 'Boolean CUDA gradient/update trace flag')
            totals[key] += value
        if updates <= 2:
            require(int(row['projection_gradient_nonzero']) == updates - 1 and
                    all(row[key] == '1' for key in flags if key not in
                        ('projection_gradient_nonzero', 'projection_weight_changed')),
                    'First mean/variance matrix/bias learning; then projection learning')
            if updates == 2:
                require(row['projection_weight_changed'] == '1', 'Second projection update')
        minimum, maximum, mean = [float(row[key]) for key in
                                  ('variance_min', 'variance_max', 'variance_mean')]
        require(all(math.isfinite(value) for value in (minimum, maximum, mean)) and
                f32(1e-6) <= minimum and minimum-1e-12*(1+maximum) <= mean <= maximum+1e-12*(1+maximum),
                'Finite positive variance statistics')
        for key in ('linear_score_abs_mean', 'variance_bias_abs_mean'):
            require(math.isfinite(float(row[key])) and float(row[key]) >= 0.,
                    'Separate finite change-dependent and bias score magnitudes')
        if updates == 1:
            require(minimum == maximum == mean == 1. and
                    float(row['linear_score_abs_mean']) == float(row['variance_bias_abs_mean']) == 0.,
                    'Actual preupdate variance is bitwise one with zero scores')
    require(updates == summary['completed_updates'] == 2048 and
            summary['window_presentations'] == 8 * len(rows), 'Exactly one declared budget')
    require(float(active[0]['loss']) == summary['first_loss'] and
            float(active[-1]['loss']) == summary['last_loss'], 'First/final NLL receipts')
    mapping = {'eligible_examples': 'wave_eligible_exposures', 'target_cells': 'target_cells',
               'projection_gradient_nonzero': 'nonzero_change_gradient_updates',
               'wave_decoder_gradient_nonzero': 'nonzero_decoder_gradient_updates',
               'projection_weight_changed': 'projection_weight_updates',
               'wave_decoder_weight_changed': 'decoder_weight_updates',
               'variance_matrix_gradient_nonzero': 'nonzero_variance_matrix_gradient_updates',
               'variance_bias_gradient_nonzero': 'nonzero_variance_bias_gradient_updates',
               'variance_matrix_weight_changed': 'variance_matrix_weight_updates',
               'variance_bias_weight_changed': 'variance_bias_weight_updates'}
    for key, field in mapping.items():
        require(totals[key] == summary[field], 'Trace/summary: ' + field)
    require(all(totals[key] > 0 for key in flags), 'All four branch blocks learn on CUDA')
    return active

def projection(run, summary, normalizer):
    previous = {}
    for method in ('initialized_v30', 'trained_v30'):
        for role, rows in (('train', 112), ('validation', 32)):
            witness = load(run / 'features' / (method + '-' + role + '-conditioning-witness.json'))
            require(witness['method'] == method and witness['role'] == role and
                    witness['rows'] == rows and witness['width'] == 270 and
                    witness['native_width'] == 36 and witness['exact_projected_suffix'] is True,
                    'Saved native/projection witness identity')
            require(witness['normalizer_identity'] == summary['bank_normalizer_identity'] and
                    witness['input_scaler_identity'] == summary['scaler_identity'],
                    'Frozen inference normalizer binding')
            for key, width in (('values', 270), ('valid', 270), ('conditioned', 270), ('native', 36)):
                matrix(witness[key], rows, width, key)
            matrix(witness['projection'], 4, 270, 'projection')
            require(len(witness['row_ids']) == len(set(witness['row_ids'])) == rows and
                    len(witness['sample_valid']) == rows, 'Readout row association')
            for row in range(rows):
                computed = []
                for column in range(270):
                    value = witness['values'][row][column]
                    valid = witness['valid'][row][column]
                    require(valid in (0, 1) and math.isfinite(value) and f32(value) == value,
                            'Finite F32 feature-bank witness')
                    supported = normalizer['support'][column]
                    expected = f32(f32(value - f32(normalizer['mean'][column])) /
                                   f32(normalizer['scale'][column])) if valid and supported else 0.
                    actual = witness['conditioned'][row][column]
                    near(actual, expected, 'F32_conditioning', 3e-7)
                    if not valid or not supported:
                        require(actual == 0., 'Literal zero for unsupported/hidden entries')
                    computed.append(expected)
                for coordinate, weights in enumerate(witness['projection']):
                    terms = [value * weight for value, weight in zip(computed, weights)]
                    expected = math.fsum(terms) if witness['sample_valid'][row] else 0.
                    actual = witness['native'][row][32 + coordinate]
                    difference = abs(actual - expected)
                    # Standard F32 dot-product forward-error bound, separate
                    # from the 1e-12 quality threshold. No numerical rescue.
                    gamma = (270 * 2**-24) / (1 - 270 * 2**-24)
                    bound = gamma * math.fsum(abs(value) for value in terms) + 3e-7
                    ERRORS['projection_absolute'] = max(ERRORS.get('projection_absolute', 0.), difference)
                    require(math.isfinite(actual) and difference <= bound,
                            'Independent bank/projection/native arithmetic')
            if method == 'initialized_v30':
                previous[role] = witness
            else:
                old = previous[role]
                require(old['row_ids'] == witness['row_ids'] and old['sample_valid'] == witness['sample_valid'],
                        'Initial/final exact row/valid association')
                require(all(a[:32] == b[:32] for a, b in zip(old['native'], witness['native'])),
                        'Exact protected parent32 from independent saved JSON')
                require(old['values'] == witness['values'] and old['valid'] == witness['valid'] and
                        old['conditioned'] == witness['conditioned'], 'Immutable conditioned input banks')


def initial_reuse(run):
    value = load(run / 'initial-scoring-reuse.json')
    old = ROOT / 'output/runs/market-relative-changes/run-vaPC4X'
    require(value['full_initial36_equal'] is True and value['new_initial_head_fits'] == 0 and
            value['querymask_target_parity'] and value['query_mask_parity'] and value['target_bank_parity'],
            'Full initial-bank parity before reused scoring')
    require(value['old_report_sha256'] == sha(old / 'readouts/report.json') ==
            '0f9199d88e7d3a90039d8a555873fb74399b8b79f6505798a58c3fcc5bae4156',
            'Exact reused initialized fixed-head source')
    for role, rows, expected in (
            ('train', 112, '953c9967515f2de6c709e1651fab7a0b464410f1f62bd25e06c2a534d72cc593'),
            ('validation', 32, 'a774470823dc2d0aad0e5f622fee4acd52383f05aabf11e193071d88f5b8aea0')):
        require(value['source_feature_'+role+'_sha256'] ==
                sha(old / 'features' / ('initialized_v28-'+role+'.pt')) == expected,
                'Exact reused initial feature archive binding')
        previous = load(old / 'features' / ('initialized_v28-'+role+'-conditioning-witness.json'))
        current = load(run / 'features' / ('initialized_v30-'+role+'-conditioning-witness.json'))
        require(len(current['native']) == rows and
                all(previous[key] == current[key] for key in
                    ('row_ids', 'sample_valid', 'native', 'values', 'valid', 'conditioned')),
                'Independent complete initial36/mask/row/bank JSON parity')
    report = load(run / 'readouts/report.json')
    require([item['name'] for item in report['methods']] == ['trained_v30'],
            'Only new final embeddings receive new fixed-head fits')
    method = report['methods'][0]
    require((method['size'], method['ridge_fits'], method['ridge_task_fits'], method['neural_fits'],
             method['ridge_parameters'], method['neural_parameters']) == (36, 1, 4, 3, 148, 660),
            'Final-only fixed head counts/width/cost')


def distribution(witness, initial, label):
    require(witness['kind'] == 'market_location_scale_decoder_query_witness_v1' and
            (witness['B'], witness['C'], witness['H'], witness['F']) == (8, 3, 128, 9) and
            witness['channel_ids_layout'] == 'C' and witness['channel_ids'] == [0, 1, 2] and
            witness['distribution_dtype'] == 'float32' and witness['exact_parent_detached'] is True,
            'Actual CUDA distribution witness contract: ' + label)
    for key, width in (('native', 36), ('L', 32), ('z', 4), ('parent_mean', 3456),
                       ('canonical_residual', 3456), ('mean', 3456), ('variance', 3456),
                       ('linear_score', 27), ('score', 27), ('u', 27), ('variance_CF', 27)):
        matrix(witness[key], 8, width, key)
    matrix(witness['D_mu'], 3456, 4, 'D_mu')
    matrix(witness['A_sigma'], 27, 4, 'A_sigma')
    require(len(witness['b_sigma']) == 27, 'Variance bias shape')
    parameters = [value for key in ('D_mu', 'A_sigma') for row in witness[key] for value in row]
    parameters += witness['b_sigma']
    require(all(math.isfinite(value) and f32(value) == value for value in parameters),
            'Finite exact F32 distribution parameters')
    if initial:
        require(all(value == 0. for value in parameters), 'Literal-zero initial location/scale parameters')
    eps32, c32, coefficient32, u0 = [witness[key] for key in ('eps32', 'c32', 'coefficient32', 'u0')]
    require(eps32 == f32(1e-6) and c32 == f32(0.5413232726357509) and
            coefficient32 == f32(1.-eps32) and math.isfinite(u0) and u0 > 0. and f32(u0) == u0,
            'Fixed actual CUDA F32 link constants')
    near(u0, math.log1p(math.exp(c32)), 'softplus_baseline', 3e-7)
    gamma4 = 4*2**-24/(1-4*2**-24)
    for row in range(8):
        native, parent, z = witness['native'][row], witness['L'][row], witness['z'][row]
        require(native[:32] == parent and native[32:] == z and
                all(math.isfinite(value) and f32(value) == value for value in native),
                'Exact same-forward native split')
        for cf, weights in enumerate(witness['A_sigma']):
            terms = [a*value for a, value in zip(weights, z)]
            expected = math.fsum(terms); linear = witness['linear_score'][row][cf]
            require(math.isfinite(linear) and f32(linear) == linear and
                    abs(linear-expected) <= gamma4*math.fsum(abs(term) for term in terms)+3e-7,
                    'Independent F32 change-to-variance dot')
            score = witness['score'][row][cf]
            require(score == f32(linear+witness['b_sigma'][cf]), 'Exact F32 score plus bias')
            shifted = f32(score+c32)
            expected_u = shifted if shifted > 20. else math.log1p(math.exp(shifted))
            u = witness['u'][row][cf]
            require(math.isfinite(u) and u >= 0. and f32(u) == u, 'Finite stable F32 softplus')
            near(u, expected_u, 'softplus_value', 3e-7)
            expected_variance = f32(eps32+f32(coefficient32*f32(u/u0)))
            variance = witness['variance_CF'][row][cf]
            require(variance == expected_variance and math.isfinite(variance) and variance >= eps32,
                    'Exact F32 divide then multiply then add anchored link')
            if initial:
                require(linear == score == 0. and u == u0 and variance == 1.,
                        'Bitwise unit initial variance from actual CUDA intermediates')
        for index, weights in enumerate(witness['D_mu']):
            terms = [a*value for a, value in zip(weights, z)]
            expected = math.fsum(terms); residual = witness['canonical_residual'][row][index]
            require(math.isfinite(residual) and f32(residual) == residual and
                    abs(residual-expected) <= gamma4*math.fsum(abs(term) for term in terms)+3e-7,
                    'Independent F32 change-to-mean residual dot')
            base = witness['parent_mean'][row][index]
            mean = witness['mean'][row][index]
            require(math.isfinite(base) and f32(base) == base and mean == f32(base+residual),
                    'Exact F32 frozen parent plus semantic residual')
            cf = (index//(128*9))*9+index%9
            require(witness['variance'][row][index] == witness['variance_CF'][row][cf],
                    'Exact canonical CF broadcast and semantic gather')
            if initial:
                require(residual == 0. and mean == base, 'Exact initial parent mean')


def decoder(run):
    saved_weights = {}
    for method in ('initialized_v30', 'trained_v30'):
        for role in ('train', 'validation'):
            witness = load(run / 'features' / (method+'-'+role+'-decoder-witness.json'))
            require(witness['method'] == method and witness['role'] == role, 'Decoder method/role')
            distribution(witness, method == 'initialized_v30', method+'-'+role)
            feature = load(run / 'features' / (method+'-'+role+'-conditioning-witness.json'))
            require(witness['row_ids'] == feature['row_ids'][:8], 'Same-query row association')
            weights = witness['D_mu'], witness['A_sigma'], witness['b_sigma']
            if role == 'train':
                saved_weights[method] = weights
            else:
                require(saved_weights[method] == weights, 'Identical frozen distribution parameters across roles')


def training_losses(run, active):
    for number in (1, 2):
        value = load(run / ('training-loss-'+str(number)+'-witness.json'))
        row = active[number-1]
        require(value['kind'] == 'market_location_scale_decoder_training_loss_witness_v1' and
                value['attempt'] == int(row['attempt']) and value['updates_before_forward'] == number-1 and
                value['target_dtype'] == 'float32' and value['loss_dtype'] == 'float64',
                'Actual first-two-forward NLL witness identity')
        d = value['distribution']
        require(d['method'] == 'training_update_'+str(number-1) and d['role'] == 'train' and
                d['row_ids'] == [int(item) for item in row['anchors'].split(';')],
                'NLL forward tied to actual TRAIN batch and counters')
        distribution(d, number == 1, 'training-'+str(number))
        if number == 2:
            require(all(any(item != 0. for record in d[key] for item in record) for key in ('D_mu', 'A_sigma'))
                    and any(item != 0. for item in d['b_sigma']), 'First mean/variance updates visible on next actual forward')
        for key in ('targets', 'target_mask', 'visible', 'cell_nll'):
            matrix(value[key], 8, 3456, key)
        for key in ('eligible_channels', 'target_counts', 'channel_loss'):
            matrix(value[key], 8, 3, key)
        require(len(value['eligible_examples']) == len(value['example_loss']) == 8,
                'NLL example support/aggregation shape')
        all_examples, total_cells, total_channels = [], 0, 0
        for b in range(8):
            channel_values, eligible_count = [], 0
            for channel in range(3):
                cells = []
                for offset in range(128*9):
                    index = channel*128*9+offset
                    supported, visible = value['target_mask'][b][index], value['visible'][b][index]
                    require(supported in (0, 1) and visible in (0, 1) and not (supported and visible),
                            'Disjoint query/context visible and target support')
                    target = value['targets'][b][index]
                    require(math.isfinite(target) and f32(target) == target, 'Finite sanitized F32 target')
                    actual = value['cell_nll'][b][index]
                    if supported:
                        mean, variance = d['mean'][b][index], d['variance'][b][index]
                        expected = .5*((mean-target)**2/variance+math.log(variance))
                        near(actual, expected, 'F64_Gaussian_cell_NLL', 1e-10)
                        cells.append(expected)
                    else:
                        require(target == actual == 0., 'Non-target storage sanitized and cell NLL exactly zero')
                count = len(cells); eligible = value['eligible_channels'][b][channel]
                require(value['target_counts'][b][channel] == count and eligible == int(count > 0),
                        'Exact target counts and channel eligibility')
                expected = math.fsum(cells)/count if count else 0.
                near(value['channel_loss'][b][channel], expected, 'F64_channel_mean_NLL', 1e-10)
                if count:
                    channel_values.append(expected);eligible_count += 1
                total_cells += count
            expected = math.fsum(channel_values)/eligible_count if eligible_count else 0.
            near(value['example_loss'][b], expected, 'F64_example_mean_NLL', 1e-10)
            require(value['eligible_examples'][b] == int(eligible_count > 0), 'Exact example eligibility')
            if eligible_count:
                all_examples.append(expected)
            total_channels += eligible_count
        loss = math.fsum(all_examples)/len(all_examples) if all_examples else 0.
        near(value['loss'], loss, 'F64_hierarchical_Gaussian_NLL', 1e-10)
        require(value['loss'] == float(row['loss']) and value['target_cell_count'] == total_cells == int(row['target_cells'])
                and value['eligible_channel_count'] == total_channels
                and value['eligible_example_count'] == len(all_examples) == int(row['eligible_examples']),
                'Actual training loss/counters equal independently recomputed same-forward hierarchy')

def query(capture, rows):
    require(capture['total_rows'] == capture['eligible_rows'] == rows and
            len(capture['row_mae']) == len(capture['row_target_cells']) ==
            len(capture['row_eligible_channels']) == rows and capture['CPU_encoder_calls'] == 0,
            'Query population and CUDA-only calls')
    require(sum(capture['row_target_cells']) == capture['target_cells'], 'Query target-cell sum')
    near(capture['mae_train_scaled'], math.fsum(capture['row_mae']) / rows,
         'query_MAE_aggregation', 3e-7)
    require(len(capture['row_nll64']) == len(capture['row_variance_mean']) == rows and
            all(math.isfinite(value) for value in capture['row_nll64']) and
            all(math.isfinite(value) and value >= f32(1e-6) for value in capture['row_variance_mean']),
            'Finite per-row Gaussian NLL and positive variance summaries')
    near(capture['gaussian_nll64'], math.fsum(capture['row_nll64'])/rows,
         'query_NLL_aggregation', 1e-10)
    near(capture['variance_mean'], math.fsum(capture['row_variance_mean'])/rows,
         'query_variance_mean_aggregation', 1e-10)
    require(capture['variance_statistics_population'] == 'all_query_cells' and
            math.isfinite(capture['variance_min']) and math.isfinite(capture['variance_max']) and
            f32(1e-6) <= capture['variance_min'] <= capture['variance_mean'] <= capture['variance_max'],
            'Declared positive finite query-variance population')


def main():
    started = time.monotonic()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT,
            'Run inside existing managed cuwacunu container')
    require(len(sys.argv) == 2, 'Usage: audit-market-location-scale-decoder.py RUN')
    run = Path(sys.argv[1]).resolve()
    require(run.is_relative_to(ROOT / 'output/runs/market-location-scale-decoder') and
            run.name.startswith('run-'), 'Correct candidate run')
    output = run / 'independent-audit.json'
    require(not output.exists(), 'Preserve completed audit')
    before = {str(path.relative_to(ROOT)): sha(path) for path in sorted(run.rglob('*'))
              if path.is_file() and path.name != 'audit.log'}
    summary = load(run / 'summary.json')
    require(summary['instance'] == 'RPB-v30.market-01' and summary['parent_updates'] == 2048 and
            (summary['D'], summary['parameters'], summary['optimizer_updated_parameters'],
             summary['protected_parameters']) == (36, 297560, 15039, 282521), 'Model/budget contract')
    for key in ('cuda_learning_verified', 'initial_prefix_matches_saved_trained_v22',
                'final_prefix_matches_saved_trained_v22', 'protected_level_exact',
                'initial_weights_match_saved_v24_init', 'initial_all36_equality_to_v24_not_expected'):
        require(summary[key] is True, key)
    require(summary['CPU_encoder_calls'] == summary['protected_optimizer_states'] ==
            summary['bank_normalizer_encoder_calls'] == summary['bank_normalizer_validation_anchors'] == 0,
            'GPU learning and protected/calibration contract')
    require(summary['initial_all36_equal_to_v28'] is True and
            summary['new_bank_normalizer_fits'] == summary['new_initial_head_fits'] == 0,
            'Reused initialization/calibration scoring contract')
    require((summary['variance_parameters'], summary['variance_matrix_parameters'],
             summary['variance_bias_parameters']) == (135, 108, 27) and
            summary['first_projection_gradient_zero'] is True and
            summary['first_variance_matrix_and_bias_learn'] is True and summary['initial_unit_variance'] is True and
            (summary['new_final_ridge_pipelines'], summary['new_final_ridge_task_fits'],
             summary['new_final_neural_fits']) == (1, 4, 3), 'Declared location-scale and final-only head costs')
    require(summary['loss_recipe'] == 'hierarchical_diagonal_Gaussian_NLL_without_constant_v1' and
            summary['loss_dtype'] == 'float64' and summary['distribution_dtype'] == 'float32' and
            summary['variance_epsilon'] == 1e-6 and summary['variance_c'] == 0.5413232726357509 and
            summary['softplus_beta'] == 1 and summary['softplus_threshold'] == 20 and
            summary['variance_link_operation_order'] == 'divide,multiply,add' and
            summary['target_sanitization'] == 'outside-mask0 before float64 arithmetic' and
            summary['training_loss_witnesses'] == 2, 'Frozen likelihood/link precision and operation recipe')
    for name in ('initialization-parity.json', 'final-prefix-parity.json'):
        value = load(run / name)
        require(value['saved_v22_train_and_validation_prefix32_equal'] and
                value['old_encoder_replay_calls'] == value['old_head_refits'] == 0, 'Exact saved prefix parity')
    normalizer, anchors = calibration(run, summary)
    active = trace(run, summary, anchors)
    projection(run, summary, normalizer)
    initial_reuse(run)
    decoder(run)
    training_losses(run, active)
    raw = load(run / 'raw-loading-provenance.json')
    require(raw['loader_source_sha256'] == sha(ROOT /
            'code/training/include/cuwacunu/training/market_raw_f64_data.h') ==
            '4165ee929445d0bce9cac2191b63bdde72555315fbb6d0e6bd1b022c87f3b4a6' and
            raw['genuine_F64_differs_from_F32_count'] > 0 and raw['load_calls'] > 0 and
            raw['original_record_to_F64_exact'] and raw['raw_F64_to_legacy_F32_exact'] and
            raw['historical_parent_preprocessing_exact'] and
            raw['old_encoder_calls'] == raw['cache_mutations'] == 0,
            'Actual new batches preserve genuine raw source and unchanged legacy parent preprocessing')
    initial = load(run / 'initial-query.json')
    near(initial['train']['mae_train_scaled'], .27862535204206196,
         'initial_exact_parent_query_TRAIN', 1e-12)
    near(initial['validation']['mae_train_scaled'], .39052824676036835,
         'initial_exact_parent_query_VALIDATION', 1e-12)
    for captures in (initial, summary['trained_query']):
        query(captures['train'], 112)
        query(captures['validation'], 32)
    for role in ('train', 'validation'):
        require(initial[role]['variance_min'] == initial[role]['variance_max'] ==
                initial[role]['variance_mean'] == 1. and
                all(value == 1. for value in initial[role]['row_variance_mean']),
                'Initial unit variance on all actual query rows')
    require(sha(ROOT / 'doc/MARKET_LOCATION_SCALE_DECODER_CARD.md') == summary['card_sha256'] ==
            'f620298e9ecf9f73a32ae61737b67295f8dcde709416aa3d6ed16feee613e45c', 'Sealed card')
    require(sha(Path('/opt/cuwacunu/build/market-study/market_fixed_regression')) ==
            '5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14', 'Fixed evaluator')
    for name, expected in before.items():
        require(sha(ROOT / name) == expected, 'Saved audit input changed')
    result = dict(kind='market_location_scale_decoder_independent_arithmetic_audit_v1',
                  status='passed', checks=CHECKS, maximum_errors=ERRORS,
                  bank_calibration_reused_without_new_fit=True, conditioning_normalized_tolerance=3e-7,
                  original_bank_arithmetic_checks_bound=1549413,
                  projection_tolerance='F32 gamma270*sum(abs(products))+3e-7',
                  variance_dot_tolerance='F32 gamma4*sum(abs(products))+3e-7',
                  softplus_normalized_tolerance=3e-7,
                  residual_tolerance='F32 gamma4*sum(abs(products))+3e-7',
                  variance_link_and_parent_add_tolerance='Exact scalar F32 results from saved intermediates',
                  gaussian_nll_and_hierarchy_normalized_tolerance=1e-10,
                  same_forward_training_loss_witnesses_checked=2,
                  quality_gate_tolerance=1e-12, input_sha256=before,
                  tensor_archives_deserialized=0, new_encoder_calls=0, new_encoder_updates=0,
                  head_or_scaler_fits=0, elapsed_seconds=time.monotonic() - started)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'checks', 'maximum_errors', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()

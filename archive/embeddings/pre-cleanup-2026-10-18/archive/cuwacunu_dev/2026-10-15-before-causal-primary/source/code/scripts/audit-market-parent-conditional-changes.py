#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independently check saved v27 scalar witnesses; no models, tensors or fits."""
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


def cholesky(gram):
    width = len(gram)
    lower = [[0.] * width for _ in range(width)]
    for i in range(width):
        for j in range(i + 1):
            value = gram[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                require(value > 0. and math.isfinite(value), 'Independent SPD Cholesky pivot')
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def solve(lower, rhs):
    width = len(rhs)
    intermediate, result = [0.] * width, [0.] * width
    for i in range(width):
        intermediate[i] = (rhs[i] - math.fsum(lower[i][k] * intermediate[k] for k in range(i))) / lower[i][i]
    for i in range(width - 1, -1, -1):
        result[i] = (intermediate[i] - math.fsum(lower[k][i] * result[k] for k in range(i + 1, width))) / lower[i][i]
    return result


def calibration(run, summary):
    original = ROOT / 'output/runs/market-conditioned-changes/run-RjTTZZ'
    normalizer = load(run / 'bank-standardizer.json')
    require(sha(run / 'bank-standardizer.json') == sha(original / 'bank-standardizer.json') ==
            '33621009dc68cf2f3e53f39382925f98d2d32e2ba9f61eccb2fbc2b084eed0cc', 'Exact reused normalizer metadata')
    require(sha(run / 'bank-standardizer.pt') == summary['bank_normalizer_sha256'] ==
            '15e11373b14fe438fe920ccf13a72b86aa4719ea1c17dee8422249db1c185535', 'Exact reused normalizer archive')
    old = load(original / 'calibration-witness.json')
    require(sha(original / 'calibration-witness.json') ==
            'add7f7ed8826fdb7ef89b68155bfc25c570f1827c76aead07bc71e71ab523e52', 'Audited old bank witness')
    parent = load(run / 'calibration-parent-witness.json')
    witness = load(run / 'calibration-map-witness.json')
    fitted = load(run / 'conditional-map.json')
    compact = load(run / 'conditional-map-summary.json')
    reuse = load(run / 'calibration-reuse.json')
    require(parent['rows'] == witness['rows'] == 4096 and parent['width'] == witness['parent_width'] == 32 and
            witness['bank_width'] == 270 and parent['batch_size'] == witness['batch_size'] == 8,
            'Exact new calibration geometry')
    require(parent['CUDA_encoder_calls'] == summary['parent_calibration_CUDA_encoder_calls'] == 512 and
            parent['CPU_encoder_calls'] == parent['encoder_updates'] == summary['parent_calibration_CPU_encoder_calls'] ==
            summary['parent_calibration_encoder_updates'] == summary['new_bank_normalizer_fits'] ==
            reuse['new_bank_normalizer_fits'] == reuse['new_bank_encoder_calls'] == 0,
            'Explicit new frozen CUDA features, zero updates or bank refits')
    require(parent['row_ids'] == witness['row_ids'] == old['row_ids'] and
            len(set(parent['row_ids'])) == 4096 and parent['values'] == witness['parent'] and
            parent['sample_valid'] == witness['sample_valid'], 'Exact parent/bank calibration row association')
    require(old['values'] == witness['values'] and old['valid'] == witness['valid'], 'Reused raw F32 bank exactly')
    require(reuse['normalizer_identity'] == fitted['normalizer_identity'] == witness['normalizer_identity'] ==
            summary['bank_normalizer_identity'] == 'rpb-bank-standardizer-v1-fnv1a64-59b99cdeeeadc038',
            'Immutable bank normalizer identity')
    require(fitted['identity'] == compact['identity'] == witness['conditional_map_identity'] ==
            summary['conditional_map_identity'], 'New conditional map identity')
    require(fitted['source_identity'] == summary['conditional_map_source_identity'] and
            fitted['parent_identity'] == parent['parent_identity'] ==
            'e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c', 'Map calibration source/parent binding')
    require(fitted['input_scaler_identity'] == parent['input_scaler_identity'] == summary['scaler_identity'] ==
            'rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969', 'Same input scaler')
    require(fitted['penalty'] == fitted['ridge'] == 1. and fitted['intercept_penalized'] is False and
            fitted['minimum_count'] == fitted['minimum_support'] == 2 and fitted['parent_scale_floor'] == fitted['floor'] == 1e-6 and
            fitted['gram_is_unpenalized'] is True and fitted['validation_anchors'] == compact['validation_anchors'] == 0 and
            fitted['coefficient_count'] == summary['conditional_map_coefficient_count'] == 8910,
            'Prospective solver/regularization/support/size contract')
    matrix(parent['values'], 4096, 32, 'parent calibration')
    matrix(witness['values'], 4096, 270, 'raw bank')
    matrix(witness['valid'], 4096, 270, 'bank validity')
    matrix(witness['conditioned'], 4096, 270, 'conditioned bank')
    matrix(fitted['coefficients'], 33, 270, 'conditional coefficients')
    matrix(fitted['rhs'], 270, 33, 'conditional rhs')
    matrix(fitted['normal_equation_residual'], 270, 33, 'normal-equation witness')
    require(len(fitted['gram']) == len(fitted['counts']) == len(fitted['support']) == len(fitted['target_sumsq']) == 270,
            'Map sufficient-statistic width')
    sample_valid = parent['sample_valid']
    require(all(value in (0, 1) for value in sample_valid) and sum(sample_valid) >= 2, 'Sample-valid calibration support')
    valid_rows = [index for index, flag in enumerate(sample_valid) if flag]
    for column in range(32):
        values = [parent['values'][row][column] for row in valid_rows]
        require(all(math.isfinite(value) and f32(value) == value for value in values), 'F32 parent primitive')
        total, squared = math.fsum(values), math.fsum(value * value for value in values)
        mean = total / len(values)
        scale = max(math.sqrt(max(squared / len(values) - mean * mean, 0.)), 1e-6)
        require(fitted['parent_counts'][column] == len(values), 'Parent statistic count')
        for key, expected in (('parent_sum', total), ('parent_sumsq', squared), ('parent_mean', mean), ('parent_scale', scale)):
            near(fitted[key][column], expected, key)
    design = [[(parent['values'][row][j] - fitted['parent_mean'][j]) / fitted['parent_scale'][j]
               for j in range(32)] + [1.] for row in range(4096)]
    for row in range(4096):
        for column in range(270):
            value, valid = witness['values'][row][column], witness['valid'][row][column]
            require(valid in (0, 1) and f32(value) == value and math.isfinite(value), 'Original F32 bank primitive')
            expected = f32(f32(value - f32(normalizer['mean'][column])) / f32(normalizer['scale'][column])) \
                if valid and normalizer['support'][column] else 0.
            near(witness['conditioned'][row][column], expected, 'calibration_F32_conditioning', 3e-7)
    # Cache only identical coordinate supports, not fitted answers. All Gram,
    # rhs, Cholesky and coefficient arithmetic below is independent scalar F64.
    cache = {}
    for column in range(270):
        selected = tuple(row for row in valid_rows if witness['valid'][row][column] and normalizer['support'][column])
        supported = len(selected) >= 2
        require(fitted['counts'][column] == len(selected) and fitted['support'][column] == int(supported), 'Map support count')
        matrix(fitted['gram'][column], 33, 33, 'coordinate Gram')
        if not supported:
            require(all(value == 0. for row in fitted['gram'][column] for value in row) and
                    all(value == 0. for value in fitted['rhs'][column]) and
                    all(fitted['coefficients'][j][column] == 0. for j in range(33)) and fitted['target_sumsq'][column] == 0.,
                    'Unsupported operator is literal zero')
            continue
        if selected not in cache:
            columns = [[design[row][j] for row in selected] for j in range(33)]
            gram = [[0.] * 33 for _ in range(33)]
            for i in range(33):
                for j in range(i + 1):
                    gram[i][j] = gram[j][i] = math.fsum(a * b for a, b in zip(columns[i], columns[j]))
            regularized = [[gram[i][j] + (1. if i == j and i < 32 else 0.) for j in range(33)] for i in range(33)]
            cache[selected] = columns, gram, cholesky(regularized)
        columns, gram, lower = cache[selected]
        target = [witness['conditioned'][row][column] for row in selected]
        rhs = [math.fsum(a * b for a, b in zip(values, target)) for values in columns]
        answer = solve(lower, rhs)
        for i in range(33):
            for j in range(33):
                near(fitted['gram'][column][i][j], gram[i][j], 'map_unpenalized_Gram')
            near(fitted['rhs'][column][i], rhs[i], 'map_rhs')
            near(fitted['coefficients'][i][column], answer[i], 'map_independent_coefficient', 1e-8)
            residual = math.fsum(fitted['gram'][column][i][j] * fitted['coefficients'][j][column] for j in range(33)) \
                + (fitted['coefficients'][i][column] if i < 32 else 0.) - fitted['rhs'][column][i]
            near(fitted['normal_equation_residual'][column][i] / (1 + abs(fitted['rhs'][column][i])),
                 residual / (1 + abs(fitted['rhs'][column][i])), 'normal_equation_witness_normalized', 1e-12)
            require(abs(residual) / (1 + abs(fitted['rhs'][column][i])) <= 1e-10, 'Accurate regularized normal equations')
        near(fitted['target_sumsq'][column], math.fsum(value * value for value in target), 'map_target_sumsq')
    require(sha(run / 'conditional-map.pt') == compact['archive_sha256'] == summary['conditional_map_sha256'], 'Immutable map archive hash')
    return normalizer, set(parent['row_ids']), fitted


def projection(run, summary, normalizer, fitted):
    previous = {}
    for method in ('initialized_v27', 'trained_v27'):
        for role, rows in (('train', 112), ('validation', 32)):
            witness = load(run / 'features' / (method + '-' + role + '-conditioning-witness.json'))
            require(witness['method'] == method and witness['role'] == role and witness['rows'] == rows and
                    witness['normalizer_identity'] == summary['bank_normalizer_identity'] and
                    witness['conditional_map_identity'] == summary['conditional_map_identity'] and
                    witness['exact_projected_suffix'] is True, 'Runtime frozen identity bindings')
            for key, width in (('values', 270), ('valid', 270), ('conditioned', 270), ('corrected', 270), ('parent', 32), ('native', 36)):
                matrix(witness[key], rows, width, key)
            matrix(witness['projection'], 4, 270, 'projection')
            require(len(witness['row_ids']) == len(set(witness['row_ids'])) == rows and len(witness['sample_valid']) == rows,
                    'Runtime row association')
            for row in range(rows):
                require(witness['parent'][row] == witness['native'][row][:32], 'Exact direct parent prefix')
                design = [f32(f32(witness['parent'][row][j] - f32(fitted['parent_mean'][j])) / f32(fitted['parent_scale'][j]))
                          for j in range(32)] + [1.]
                residuals = []
                for column in range(270):
                    valid = witness['valid'][row][column]
                    value = witness['values'][row][column]
                    expected_bank = f32(f32(value - f32(normalizer['mean'][column])) / f32(normalizer['scale'][column])) \
                        if valid and normalizer['support'][column] else 0.
                    near(witness['conditioned'][row][column], expected_bank, 'runtime_F32_conditioning', 3e-7)
                    eligible = valid and normalizer['support'][column] and fitted['support'][column] and witness['sample_valid'][row]
                    terms = [design[j] * f32(fitted['coefficients'][j][column]) for j in range(33)]
                    expected = expected_bank - math.fsum(terms) if eligible else 0.
                    actual = witness['corrected'][row][column]
                    gamma = (33 * 2**-24) / (1 - 33 * 2**-24)
                    bound = gamma * math.fsum(abs(term) for term in terms) + 3e-7 * (1 + abs(expected_bank) + abs(expected))
                    error = abs(actual - expected)
                    ERRORS['runtime_correction_absolute'] = max(ERRORS.get('runtime_correction_absolute', 0.), error)
                    require(math.isfinite(actual) and error <= bound, 'Independent conditional correction')
                    if not eligible:
                        require(actual == 0., 'Literal zero unsupported/invalid correction')
                    residuals.append(actual)
                for coordinate, weights in enumerate(witness['projection']):
                    terms = [value * weight for value, weight in zip(residuals, weights)]
                    expected = math.fsum(terms) if witness['sample_valid'][row] else 0.
                    actual = witness['native'][row][32 + coordinate]
                    gamma = (270 * 2**-24) / (1 - 270 * 2**-24)
                    error = abs(actual - expected)
                    ERRORS['projection_absolute'] = max(ERRORS.get('projection_absolute', 0.), error)
                    require(math.isfinite(actual) and error <= gamma * math.fsum(abs(value) for value in terms) + 3e-7,
                            'Independent corrected-bank/native projection')
            if method == 'initialized_v27':
                previous[role] = witness
            else:
                old = previous[role]
                for key in ('row_ids', 'sample_valid', 'parent', 'values', 'valid', 'conditioned', 'corrected'):
                    require(old[key] == witness[key], 'Frozen initial/final bank/prefix association: ' + key)


def trace(run, summary, anchors):
    with (run / 'training.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == summary['attempted_batches'] <= 4096,
            'Trace attempts and skip cap')
    totals = {key: 0 for key in ('eligible_examples', 'target_cells',
              'projection_gradient_nonzero', 'wave_decoder_gradient_nonzero',
              'projection_weight_changed', 'wave_decoder_weight_changed')}
    updates = 0
    active = []
    for attempt, row in enumerate(rows):
        require(int(row['attempt']) == attempt and
                int(row['row_seed']) == counter(942002, attempt) and
                int(row['mask_seed']) == counter(942003, attempt), 'Absolute counter streams')
        selected = [int(value) for value in row['anchors'].split(';')]
        require(len(selected) == 8 and all(value in anchors for value in selected),
                'B8 legal TRAIN presentations')
        if not row['loss']:
            require(int(row['update']) == updates, 'Skip preserves update count')
            continue
        active.append(row)
        updates += 1
        require(int(row['update']) == updates and math.isfinite(float(row['loss'])) and
                float(row['loss']) >= 0., 'Finite ordered waveform objective')
        require(0 < int(row['eligible_examples']) <= 8 and
                0 < int(row['target_cells']) <= 8 * 3 * 128 * 9, 'Waveform target eligibility')
        for key in totals:
            value = int(row[key])
            if 'nonzero' in key or 'changed' in key:
                require(value in (0, 1), 'Boolean CUDA trace flag')
            totals[key] += value
        if updates <= 2:
            require(int(row['projection_gradient_nonzero']) == updates - 1 and
                    row['wave_decoder_gradient_nonzero'] == row['wave_decoder_weight_changed'] == '1',
                    'Zero decoder then projection learning')
            if updates == 2:
                require(row['projection_weight_changed'] == '1', 'Second-step projection update')
    require(updates == summary['completed_updates'] == 2048 and
            summary['window_presentations'] == 8 * len(rows), 'Exactly one declared budget')
    require(float(active[0]['loss']) == summary['first_loss'] and
            float(active[-1]['loss']) == summary['last_loss'], 'First/final losses')
    mapping = {'eligible_examples': 'wave_eligible_exposures', 'target_cells': 'target_cells',
               'projection_gradient_nonzero': 'nonzero_change_gradient_updates',
               'wave_decoder_gradient_nonzero': 'nonzero_decoder_gradient_updates',
               'projection_weight_changed': 'projection_weight_updates',
               'wave_decoder_weight_changed': 'decoder_weight_updates'}
    for key, field in mapping.items():
        # Earlier waveform producers expose eligible populations only in trace.
        if field in summary:
            require(totals[key] == summary[field], 'Trace/summary: ' + field)


def query(capture, rows):
    require(capture['total_rows'] == capture['eligible_rows'] == rows and
            len(capture['row_mae']) == len(capture['row_target_cells']) ==
            len(capture['row_eligible_channels']) == rows and capture['CPU_encoder_calls'] == 0,
            'Query population and CUDA-only calls')
    require(sum(capture['row_target_cells']) == capture['target_cells'], 'Query target-cell sum')
    near(capture['mae_train_scaled'], math.fsum(capture['row_mae']) / rows,
         'query_MAE_aggregation', 3e-7)


def main():
    started = time.monotonic()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT,
            'Run inside existing managed cuwacunu container')
    require(len(sys.argv) == 2, 'Usage: audit-market-parent-conditional-changes.py RUN')
    run = Path(sys.argv[1]).resolve()
    require(run.is_relative_to(ROOT / 'output/runs/market-parent-conditional-changes') and
            run.name.startswith('run-'), 'Correct candidate run')
    output = run / 'independent-audit.json'
    require(not output.exists(), 'Preserve completed audit')
    before = {str(path.relative_to(ROOT)): sha(path) for path in sorted(run.rglob('*'))
              if path.is_file() and path.name != 'audit.log'}
    summary = load(run / 'summary.json')
    require(summary['instance'] == 'RPB-v27.market-01' and summary['parent_updates'] == 2048 and
            (summary['D'], summary['parameters'], summary['optimizer_updated_parameters'],
             summary['protected_parameters']) == (36, 297425, 14904, 282521), 'Model/budget contract')
    for key in ('cuda_learning_verified', 'initial_prefix_matches_saved_trained_v22',
                'final_prefix_matches_saved_trained_v22', 'protected_level_exact',
                'initial_weights_match_saved_v24_init', 'initial_all36_equality_to_v24_not_expected'):
        require(summary[key] is True, key)
    require(summary['CPU_encoder_calls'] == summary['protected_optimizer_states'] ==
            summary['bank_normalizer_encoder_calls'] == summary['bank_normalizer_validation_anchors'] == 0,
            'GPU learning and protected/calibration contract')
    for name in ('initialization-parity.json', 'final-prefix-parity.json'):
        value = load(run / name)
        require(value['saved_v22_train_and_validation_prefix32_equal'] and
                value['old_encoder_replay_calls'] == value['old_head_refits'] == 0, 'Exact saved prefix parity')
    normalizer, anchors, fitted = calibration(run, summary)
    trace(run, summary, anchors)
    projection(run, summary, normalizer, fitted)
    initial = load(run / 'initial-query.json')
    near(initial['train']['mae_train_scaled'], .27862535204206196,
         'initial_exact_parent_query_TRAIN', 1e-12)
    near(initial['validation']['mae_train_scaled'], .39052824676036835,
         'initial_exact_parent_query_VALIDATION', 1e-12)
    for captures in (initial, summary['trained_query']):
        query(captures['train'], 112)
        query(captures['validation'], 32)
    require(sha(ROOT / 'doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md') == summary['card_sha256'] ==
            '6e24e48bfdd05148f99446cf1421a713a7736d522f6c00159ba156e43b39c29b', 'Sealed card')
    require(sha(Path('/opt/cuwacunu/build/market-study/market_fixed_regression')) ==
            '5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14', 'Fixed evaluator')
    for name, expected in before.items():
        require(sha(ROOT / name) == expected, 'Saved audit input changed')
    result = dict(kind='market_parent_conditional_changes_independent_arithmetic_audit_v1',
                  status='passed', checks=CHECKS, maximum_errors=ERRORS,
                  calibration_normalized_tolerance=1e-10, independent_coefficient_normalized_tolerance=1e-8,
                  conditioning_normalized_tolerance=3e-7, regularized_normal_equation_tolerance=1e-10,
                  projection_tolerance='F32 gamma270*sum(abs(products))+3e-7',
                  quality_gate_tolerance=1e-12, input_sha256=before,
                  tensor_archives_deserialized=0, new_encoder_calls=0, new_encoder_updates=0,
                  head_or_scaler_fits=0, elapsed_seconds=time.monotonic() - started)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'checks', 'maximum_errors', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independently check saved v29 scalar witnesses; no models, tensors or fits."""
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
    require(len(rows) == summary['attempted_batches'] <= 4096,
            'Trace attempts and skip cap')
    totals = {key: 0 for key in ('eligible_examples', 'target_cells',
              'projection_gradient_nonzero', 'wave_decoder_gradient_nonzero',
              'projection_weight_changed', 'wave_decoder_weight_changed',
              'gain_gradient_nonzero', 'gain_weight_changed')}
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
                require(row['projection_weight_changed'] == row['gain_gradient_nonzero'] ==
                        row['gain_weight_changed'] == '1', 'Second-step projection and gain learning')
            else:
                require(row['gain_gradient_nonzero'] == row['gain_weight_changed'] == '0',
                        'Zero first-step gain gradient and unchanged zero gain matrix')
        minimum, maximum, mean = [float(row[key]) for key in ('gain_min', 'gain_max', 'gain_mean')]
        roundoff = 1e-6*(1.+max(abs(minimum), abs(maximum)))
        require(all(math.isfinite(value) for value in (minimum, maximum, mean)) and
                minimum-roundoff <= mean <= maximum+roundoff, 'Finite unconstrained gain statistics')
    require(updates == summary['completed_updates'] == 2048 and
            summary['window_presentations'] == 8 * len(rows), 'Exactly one declared budget')
    require(float(active[0]['loss']) == summary['first_loss'] and
            float(active[-1]['loss']) == summary['last_loss'], 'First/final losses')
    mapping = {'eligible_examples': 'wave_eligible_exposures', 'target_cells': 'target_cells',
               'projection_gradient_nonzero': 'nonzero_change_gradient_updates',
               'wave_decoder_gradient_nonzero': 'nonzero_decoder_gradient_updates',
               'projection_weight_changed': 'projection_weight_updates',
               'wave_decoder_weight_changed': 'decoder_weight_updates',
               'gain_gradient_nonzero': 'nonzero_gain_gradient_updates',
               'gain_weight_changed': 'gain_weight_updates'}
    for key, field in mapping.items():
        # Earlier waveform producers expose eligible populations only in trace.
        if field in summary:
            require(totals[key] == summary[field], 'Trace/summary: ' + field)
    require(summary['nonzero_gain_gradient_updates'] == totals['gain_gradient_nonzero'] > 0 and
            summary['gain_weight_updates'] == totals['gain_weight_changed'] > 0,
            'Saved trace and summary prove real CUDA gain learning')


def projection(run, summary, normalizer):
    previous = {}
    for method in ('initialized_v29', 'trained_v29'):
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
            if method == 'initialized_v29':
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
        current = load(run / 'features' / ('initialized_v29-'+role+'-conditioning-witness.json'))
        require(len(current['native']) == rows and
                all(previous[key] == current[key] for key in
                    ('row_ids', 'sample_valid', 'native', 'values', 'valid', 'conditioned')),
                'Independent complete initial36/mask/row/bank JSON parity')
    report = load(run / 'readouts/report.json')
    require([item['name'] for item in report['methods']] == ['trained_v29'],
            'Only new final embeddings receive new fixed-head fits')
    method = report['methods'][0]
    require((method['size'], method['ridge_fits'], method['ridge_task_fits'], method['neural_fits'],
             method['ridge_parameters'], method['neural_parameters']) == (36, 1, 4, 3, 148, 660),
            'Final-only fixed head counts/width/cost')


def decoder(run):
    saved_weights = {}
    eps = 2**-24
    for method in ('initialized_v29', 'trained_v29'):
        for role in ('train', 'validation'):
            witness = load(run / 'features' / (method+'-'+role+'-decoder-witness.json'))
            require(witness['kind'] == 'market_parent_modulated_decoder_query_witness_v1' and
                    (witness['B'], witness['C'], witness['H'], witness['F']) == (8, 3, 128, 9) and
                    witness['method'] == method and witness['role'] == role and
                    witness['channel_ids_layout'] == 'C' and witness['channel_ids'] == [0, 1, 2] and
                    witness['gain_dtype'] == 'float32' and witness['exact_parent_detached'] is True,
                    'Actual same-query decoder witness contract')
            for key, width in (('native', 36), ('L', 32), ('parent_prediction', 3456),
                               ('canonical_base_residual', 3456), ('gain', 27),
                               ('scaled_residual', 3456), ('prediction', 3456)):
                matrix(witness[key], 8, width, 'decoder_'+key)
            matrix(witness['D'], 3456, 4, 'decoder_D')
            matrix(witness['A'], 27, 32, 'decoder_A')
            feature = load(run / 'features' / (method+'-'+role+'-conditioning-witness.json'))
            require(witness['row_ids'] == feature['row_ids'][:8], 'Decoder query association')
            if role == 'train':
                saved_weights[method] = witness['D'], witness['A']
            else:
                require(saved_weights[method] == (witness['D'], witness['A']),
                        'Same decoder/gain tensors across frozen role captures')
            initial = method == 'initialized_v29'
            require(all(math.isfinite(value) and f32(value) == value
                        for key in ('D', 'A') for row in witness[key] for value in row),
                    'Finite exact model-F32 decoder/gain parameters')
            if initial:
                require(all(value == 0. for key in ('D', 'A') for row in witness[key] for value in row),
                        'Literal-zero initial waveform decoder and gain matrix')
            else:
                require(any(value != 0. for row in witness['A'] for value in row) and
                        any(value != 0. for row in witness['D'] for value in row),
                        'Saved learned gain and residual decoder are nonzero')
            for row in range(8):
                native, parent = witness['native'][row], witness['L'][row]
                require(native[:32] == parent and all(math.isfinite(value) and f32(value) == value
                        for value in native), 'Detached L is exactly same-forward native prefix')
                for cf, weights in enumerate(witness['A']):
                    terms = [a*l for a, l in zip(weights, parent)]
                    expected = 1.+math.fsum(terms)
                    actual = witness['gain'][row][cf]
                    bound = (32*eps/(1-32*eps))*math.fsum(abs(term) for term in terms)
                    bound += eps*(1.+abs(expected))+3e-7
                    error = abs(actual-expected)
                    ERRORS['gain_absolute'] = max(ERRORS.get('gain_absolute', 0.), error)
                    require(math.isfinite(actual) and f32(actual) == actual and error <= bound,
                            'Independent F32 parent-to-gain dot/add arithmetic')
                    if initial:
                        require(actual == 1., 'Exact unit initial gain')
                for index, weights in enumerate(witness['D']):
                    terms = [a*z for a, z in zip(weights, native[32:])]
                    expected = math.fsum(terms)
                    actual = witness['canonical_base_residual'][row][index]
                    bound = (4*eps/(1-4*eps))*math.fsum(abs(term) for term in terms)+3e-7
                    error = abs(actual-expected)
                    ERRORS['residual_absolute'] = max(ERRORS.get('residual_absolute', 0.), error)
                    require(math.isfinite(actual) and f32(actual) == actual and error <= bound,
                            'Independent F32 change4-to-waveform dot arithmetic')
                    cf = (index//(128*9))*9+index%9
                    gain = witness['gain'][row][cf]
                    scaled = witness['scaled_residual'][row][index]
                    require(scaled == f32(actual*gain), 'Exact scalar F32 canonical gain broadcast/product')
                    prediction = witness['prediction'][row][index]
                    base = witness['parent_prediction'][row][index]
                    require(math.isfinite(base) and f32(base) == base and
                            prediction == f32(base+scaled), 'Exact F32 frozen-parent plus gathered residual')
                    if initial:
                        require(actual == scaled == 0. and prediction == base,
                                'Exact zero initial residual and protected parent reconstruction')


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
    require(len(sys.argv) == 2, 'Usage: audit-market-parent-modulated-decoder.py RUN')
    run = Path(sys.argv[1]).resolve()
    require(run.is_relative_to(ROOT / 'output/runs/market-parent-modulated-decoder') and
            run.name.startswith('run-'), 'Correct candidate run')
    output = run / 'independent-audit.json'
    require(not output.exists(), 'Preserve completed audit')
    before = {str(path.relative_to(ROOT)): sha(path) for path in sorted(run.rglob('*'))
              if path.is_file() and path.name != 'audit.log'}
    summary = load(run / 'summary.json')
    require(summary['instance'] == 'RPB-v29.market-01' and summary['parent_updates'] == 2048 and
            (summary['D'], summary['parameters'], summary['optimizer_updated_parameters'],
             summary['protected_parameters']) == (36, 298289, 15768, 282521), 'Model/budget contract')
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
    require(summary['gain_parameters'] == 864 and summary['first_gain_gradient_zero'] is True and
            summary['first_gain_weights_zero'] is True and
            (summary['new_final_ridge_pipelines'], summary['new_final_ridge_task_fits'],
             summary['new_final_neural_fits']) == (1, 4, 3), 'Declared new gain and final-only head costs')
    for name in ('initialization-parity.json', 'final-prefix-parity.json'):
        value = load(run / name)
        require(value['saved_v22_train_and_validation_prefix32_equal'] and
                value['old_encoder_replay_calls'] == value['old_head_refits'] == 0, 'Exact saved prefix parity')
    normalizer, anchors = calibration(run, summary)
    trace(run, summary, anchors)
    projection(run, summary, normalizer)
    initial_reuse(run)
    decoder(run)
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
    require(sha(ROOT / 'doc/MARKET_PARENT_MODULATED_DECODER_CARD.md') == summary['card_sha256'] ==
            'fd900adeaf93fef6258ca2bfe344cd1dbb69ffa6cc03e00f697e2000b455f666', 'Sealed card')
    require(sha(Path('/opt/cuwacunu/build/market-study/market_fixed_regression')) ==
            '5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14', 'Fixed evaluator')
    for name, expected in before.items():
        require(sha(ROOT / name) == expected, 'Saved audit input changed')
    result = dict(kind='market_parent_modulated_decoder_independent_arithmetic_audit_v1',
                  status='passed', checks=CHECKS, maximum_errors=ERRORS,
                  bank_calibration_reused_without_new_fit=True, conditioning_normalized_tolerance=3e-7,
                  original_bank_arithmetic_checks_bound=1549413,
                  projection_tolerance='F32 gamma270*sum(abs(products))+3e-7',
                  gain_tolerance='F32 gamma32*sum(abs(products))+eps*(1+abs(1+dot))+3e-7',
                  residual_tolerance='F32 gamma4*sum(abs(products))+3e-7',
                  product_and_parent_add_tolerance='Exact scalar F32 results from saved intermediates',
                  quality_gate_tolerance=1e-12, input_sha256=before,
                  tensor_archives_deserialized=0, new_encoder_calls=0, new_encoder_updates=0,
                  head_or_scaler_fits=0, elapsed_seconds=time.monotonic() - started)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'checks', 'maximum_errors', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()

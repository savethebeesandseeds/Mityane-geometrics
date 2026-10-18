#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independently check saved v28 scalar witnesses; no models, tensors or fits."""
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


def relative_bank(run):
    witness = load(run / 'raw-relative-bank-witness.json')
    require(witness['raw_source_dtype'] == 'float64' and
            witness['parent_rounding_dtype'] == 'float32', 'Two precision paths')
    require(witness['increment_recipe'] ==
            'cuda-f64-maxabs-scaled-symmetric-relative;u64-to-f32-to-f64;visible-adjacent-zero-safe-v1',
            'Exact frozen relative increment recipe')
    require(len(witness['cases']) == 2, 'Natural and final-masked raw witnesses')
    require({case['name'] for case in witness['cases']} == {'natural_TRAIN', 'masked_TRAIN'},
            'Required raw witness populations')
    calibration_rows = load(run / 'calibration-witness.json')['row_ids'][:8]
    with (run / 'training.csv').open() as stream:
        first_update = next(row for row in csv.DictReader(stream) if row['loss'])
        training_rows = [int(value) for value in first_update['anchors'].split(';')]
    for case in witness['cases']:
        B, C, H, F = [case[key] for key in ('B', 'C', 'H', 'F')]
        require((B, C, H, F) == (8, 3, 128, 9) and
                case['channel_ids'] == [0, 1, 2], 'Canonical raw witness geometry')
        require(case['row_ids'] == (calibration_rows if case['name'] == 'natural_TRAIN' else training_rows),
                'Natural calibration and actual first-training witness row association')
        for key, width in (('raw', C*H*F), ('observed', C*H*F),
                           ('increments', C*(H-1)*F), ('adjacent_valid', C*(H-1)*F), ('counts', C*F),
                           ('values', C*F*10), ('valid', C*F*10)):
            matrix(case[key], B, width, 'raw_' + key)
        for row in range(B):
            raw, observed = case['raw'][row], case['observed'][row]
            for value, flag in zip(raw, observed):
                require(math.isfinite(value) and flag in (0, 1), 'Finite raw/mask')
            for c in range(C):
                for f in range(F):
                    increments, support = [], []
                    for t in range(1, H):
                        j = (c*H+t)*F+f
                        active = bool(observed[j] and observed[j-F])
                        a, b = (raw[j], raw[j-F]) if active else (0., 0.)
                        maximum = max(abs(a), abs(b))
                        divisor = maximum if maximum > 0. else 1.
                        aa, bb = a/divisor, b/divisor
                        denominator = abs(aa)+abs(bb)
                        expected = f32(2.*(aa-bb)/(denominator if denominator > 0. else 1.))
                        actual = case['increments'][row][(c*(H-1)+t-1)*F+f]
                        require(case['adjacent_valid'][row][(c*(H-1)+t-1)*F+f] == int(active),
                                'Actual endpoint-derived pair support')
                        near(actual, expected, 'relative_increment_F32', 3e-7)
                        require(f32(actual) == actual and (active or actual == 0.),
                                'F32 increment and no gap bridging')
                        increments.append(actual)
                        support.append(active)
                    count = sum(support)
                    require(case['counts'][row][c*F+f] == count, 'Adjacent pair count')
                    denominator = max(count, 1)
                    energy = math.fsum(value*value for value in increments)/denominator
                    rms = math.sqrt(max(energy, 1e-12))
                    actual_rms = rms if energy > 0. else 0.
                    active = count >= 2
                    moments = [math.fsum(increments)/denominator if active else 0.,
                               actual_rms if active else 0.]
                    validity = [active, active]
                    for q in range(4):
                        begin, end = q*(H-1)//4, (q+1)*(H-1)//4
                        n = sum(support[begin:end])
                        valid = active and n >= 2
                        part = increments[begin:end]
                        mean = math.fsum(part)/max(n, 1)
                        part_energy = math.fsum(value*value for value in part)/max(n, 1)
                        part_rms = math.sqrt(max(part_energy, 1e-12)) if part_energy > 0. else 0.
                        moments.extend([mean/rms if valid else 0., part_rms/rms if valid else 0.])
                        validity.extend([valid, valid])
                    for index, (value, flag) in enumerate(zip(moments, validity)):
                        column = (c*F+f)*10+index
                        require(case['valid'][row][column] == int(flag), 'Moment validity')
                        near(case['values'][row][column], f32(value), 'relative_bank_moments_F64', 3e-7)
    provenance = load(run / 'raw-loading-provenance.json')
    for field in ('original_record_to_F64_exact', 'raw_F64_to_legacy_F32_exact',
                  'historical_parent_preprocessing_exact'):
        require(provenance[field] is True, 'Genuine raw loader/parent witness: ' + field)
    require(provenance['genuine_F64_differs_from_F32_count'] > 0,
            'Raw precision retained before rounding')
    records = load(run / 'raw-loader-record-witness.json')
    require(records['record_size'] == 88 and records['record_struct_format'] == '<q5dq4d'
            and records['endian'] == 'little' and records['raw_feature_field_indexes'] ==
            [1, 2, 3, 4, 5, 7, 8, 9, 10], 'Original packed-record precision/layout')
    natural = next(case for case in witness['cases'] if case['name'] == 'natural_TRAIN')
    require(records['row_ids'] == natural['row_ids'] and len(records['records']) == 8*3*128,
            'Bounded original record association')
    handles, sources = {}, {}
    loaded_pins = dict(line.rstrip('\n').split('  ', 1)[::-1]
                       for line in (run / 'loaded-inputs.sha256').read_text().splitlines() if line)
    bounded_precision_differences = 0
    try:
        for source in records['sources']:
            path = Path(source['path'])
            if not path.is_absolute():
                path = ROOT / path
            require(path.resolve().is_relative_to(ROOT / 'data/mkt-hist-1') and
                    path.stat().st_size == source['bytes'] and sha(path) == source['sha256'],
                    'Read-only pinned original record file')
            require(loaded_pins.get(str(path)) == source['sha256'],
                    'Original record source bound to loaded input manifest')
            channel = source['channel']
            require(channel in (0, 1, 2) and channel not in handles, 'Unique original channel source')
            handles[channel] = path.open('rb')
            sources[channel] = source
        seen = set()
        for record in records['records']:
            b, c, index = [record[key] for key in ('batch', 'channel', 'index')]
            packed = bytes.fromhex(record['packed_bytes'])
            require(0 <= b < 8 and c in handles and len(packed) == 88 and
                    record['byte_offset'] == index*88, 'Original record bounds/offset')
            handles[c].seek(index*88)
            require(handles[c].read(88) == packed, 'Saved original bytes equal pinned mmap bytes')
            fields = struct.unpack('<q5dq4d', packed)
            require(record['record_valid'] == (fields[0] != -(1 << 63)),
                    'Independent original record validity')
            t = 127-(natural['row_ids'][b]-fields[6])//60000
            require(fields[6] == record['timestamp'] and 0 <= t < 128 and
                    natural['row_ids'][b]-fields[6] == (127-t)*60000 and
                    (b, c, t) not in seen, 'Exact past timestamp and unique raw record')
            seen.add((b, c, t))
            for f, field in enumerate(records['raw_feature_field_indexes']):
                j = (c*128+t)*9+f
                value, observed = natural['raw'][b][j], natural['observed'][b][j]
                expected = fields[field] if record['record_valid'] and math.isfinite(fields[field]) else 0.
                require(struct.pack('<d', value) == struct.pack('<d', expected) and
                        observed == int(record['record_valid'] and math.isfinite(fields[field])),
                        'Independent original F64 record-to-raw value/mask')
                if observed and value != f32(value):
                    bounded_precision_differences += 1
        require(len(seen) == 8*3*128, 'Complete bounded original record witness')
    finally:
        for handle in handles.values():
            handle.close()
    require(provenance['cache_mutations'] == provenance['old_encoder_calls'] == 0 and
            provenance['parent_preprocessing_checks'] == 2 and
            [{key: source[key] for key in ('channel', 'path', 'sha256', 'bytes')}
             for source in provenance['sources']] == records['sources'],
            'Read-only actual source provenance')
    require(provenance['loader_source_sha256'] == sha(ROOT /
            'code/training/include/cuwacunu/training/market_raw_f64_data.h'), 'Actual loader source binding')
    require(0 < bounded_precision_differences <= provenance['genuine_F64_differs_from_F32_count'],
            'Independent original precision differences present before F32 rounding')


def calibration(run, summary):
    witness = load(run / 'calibration-witness.json')
    normalizer = load(run / 'bank-standardizer.json')
    require(normalizer['kind'] == 'market_relative_changes_TRAIN_fit_v1' and
            normalizer['bank_family'] == 'raw-symmetric-relative-f64-rounded32-global-quarter-v1' and
            normalizer['raw_contract'] == 'genuine-raw-f64;parent-cpu-raw-f32-before-scaler-v1' and
            normalizer['increment_recipe'] ==
            'cuda-f64-maxabs-scaled-symmetric-relative;u64-to-f32-to-f64;visible-adjacent-zero-safe-v1' and
            normalizer['source_identity'].startswith('RPB-v28;raw-symmetric-relative-f64-v1;'),
            'New relative bank family/raw arithmetic/source binding')
    require(witness['rows'] == normalizer['TRAIN_anchors'] == 4096 and
            witness['width'] == 270 and witness['batch_size'] == 8 and
            witness['bank_dtype'] == 'float32', 'Exact calibration population')
    require(normalizer['encoder_calls'] == normalizer['validation_anchors'] == 0,
            'Natural TRAIN-only calibration without encoder calls')
    require(len(witness['row_ids']) == len(set(witness['row_ids'])) == 4096,
            'Unique calibration anchors')
    require(witness['source_identity'] == normalizer['source_identity'] ==
            summary['bank_normalizer_source_identity'], 'Calibration source binding')
    require(witness['input_scaler_identity'] == normalizer['input_scaler_identity'] ==
            summary['scaler_identity'] == 'rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969',
            'Unchanged input scaler')
    values, valid = witness['values'], witness['valid']
    matrix(values, 4096, 270, 'calibration values')
    matrix(valid, 4096, 270, 'calibration masks')
    for key in ('counts', 'sum', 'sumsq', 'mean', 'scale', 'support'):
        require(len(normalizer[key]) == 270, 'Normalizer width: ' + key)
    counts = [0] * 270
    sums, squares = [0.] * 270, [0.] * 270
    for row, mask in zip(values, valid):
        for index, (value, observed) in enumerate(zip(row, mask)):
            require(observed in (0, 1) and math.isfinite(value) and f32(value) == value,
                    'Exact finite F32 bank/mask witness')
            if observed:
                counts[index] += 1
                sums[index] += value
                squares[index] += value * value
            else:
                require(value == 0., 'Invalid bank entry must be zero')
    for index in range(270):
        count = counts[index]
        supported = count >= 2
        mean = sums[index] / count if supported else 0.
        variance = max(squares[index] / count - mean * mean, 0.) if supported else 0.
        scale = max(math.sqrt(variance), 1e-6) if supported else 1.
        require(normalizer['counts'][index] == count and
                normalizer['support'][index] == int(supported), 'Exact support/count')
        for key, expected in (('sum', sums[index]), ('sumsq', squares[index]),
                              ('mean', mean), ('scale', scale)):
            near(normalizer[key][index], expected, 'normalizer_' + key)
    require(sha(run / 'bank-standardizer.pt') == normalizer['archive_sha256'] ==
            summary['bank_normalizer_sha256'], 'Frozen normalizer archive hash')
    return normalizer, set(witness['row_ids'])


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


def projection(run, summary, normalizer):
    previous = {}
    for method in ('initialized_v28', 'trained_v28'):
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
            if method == 'initialized_v28':
                previous[role] = witness
            else:
                old = previous[role]
                require(old['row_ids'] == witness['row_ids'] and old['sample_valid'] == witness['sample_valid'],
                        'Initial/final exact row/valid association')
                require(all(a[:32] == b[:32] for a, b in zip(old['native'], witness['native'])),
                        'Exact protected parent32 from independent saved JSON')
                require(old['values'] == witness['values'] and old['valid'] == witness['valid'] and
                        old['conditioned'] == witness['conditioned'], 'Immutable conditioned input banks')


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
    require(len(sys.argv) == 2, 'Usage: audit-market-relative-changes.py RUN')
    run = Path(sys.argv[1]).resolve()
    require(run.is_relative_to(ROOT / 'output/runs/market-relative-changes') and
            run.name.startswith('run-'), 'Correct candidate run')
    output = run / 'independent-audit.json'
    require(not output.exists(), 'Preserve completed audit')
    before = {str(path.relative_to(ROOT)): sha(path) for path in sorted(run.rglob('*'))
              if path.is_file() and path.name != 'audit.log'}
    summary = load(run / 'summary.json')
    require(summary['instance'] == 'RPB-v28.market-01' and summary['parent_updates'] == 2048 and
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
    relative_bank(run)
    normalizer, anchors = calibration(run, summary)
    trace(run, summary, anchors)
    projection(run, summary, normalizer)
    initial = load(run / 'initial-query.json')
    near(initial['train']['mae_train_scaled'], .27862535204206196,
         'initial_exact_parent_query_TRAIN', 1e-12)
    near(initial['validation']['mae_train_scaled'], .39052824676036835,
         'initial_exact_parent_query_VALIDATION', 1e-12)
    for captures in (initial, summary['trained_query']):
        query(captures['train'], 112)
        query(captures['validation'], 32)
    require(sha(ROOT / 'doc/MARKET_RELATIVE_CHANGE_CARD.md') == summary['card_sha256'] ==
            'f4378d4a76185fc17b6c0adbadab5af27accd772ed66e5a8bc628ab7848f94a2', 'Sealed card')
    require(sha(Path('/opt/cuwacunu/build/market-study/market_fixed_regression')) ==
            '5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14', 'Fixed evaluator')
    for name, expected in before.items():
        require(sha(ROOT / name) == expected, 'Saved audit input changed')
    result = dict(kind='market_relative_changes_independent_arithmetic_audit_v1',
                  status='passed', checks=CHECKS, maximum_errors=ERRORS,
                  calibration_normalized_tolerance=1e-10, conditioning_normalized_tolerance=3e-7,
                  relative_increment_and_moment_normalized_tolerance=3e-7,
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

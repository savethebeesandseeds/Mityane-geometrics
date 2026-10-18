#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent scalar audit of local TRAIN JSON witnesses; no tensor loader."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time

ROOT = Path('/cuwacunu')
PROTOCOL = 'market-change-scale-diagnosis-v1'
CARD = ROOT / 'doc/MARKET_CHANGE_SCALE_DIAGNOSIS_CARD.md'
ADDENDUM = ROOT / 'doc/MARKET_CHANGE_SCALE_DIAGNOSIS_ADDENDUM.md'
MANIFEST = ROOT / 'doc/results/market_change_scale_diagnosis_v1/input-manifest.json'
EXPECTED = {
    CARD: '848c5dbf3b6dbce0450c4287c3bd30d0d0b7978ea5571a925d86f492e11d0260',
    ADDENDUM: '19a3dc4455a79d40ebd7e6086be1ad44d2d077a26763d2f3826557c3b789fa8c',
    MANIFEST: '241625e6815fa32175f082e4ffc73d460eec9b050de3d2f750649186b10488e3',
}
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
CONTEXTS = (('all112', 0, 112), ('first56', 0, 56), ('last56', 56, 112))
TOLERANCE = 1e-10
CHECKS = 0
MAX_ERROR = 0.0
MAX_ERRORS = {}


def require(value, label):
    global CHECKS
    CHECKS += 1
    if not value:
        raise RuntimeError(label)


def near(actual, expected, label, tolerance=TOLERANCE, category='arithmetic'):
    global MAX_ERROR
    if actual is None or expected is None:
        require(actual is None and expected is None, label + ': unsupported differs')
        return
    require(isinstance(actual, (int, float)) and isinstance(expected, (int, float)), label + ': numeric required')
    require(math.isfinite(actual) and math.isfinite(expected), label + ': finite required')
    error = abs(actual - expected) / (1.0 + abs(expected))
    MAX_ERROR = max(MAX_ERROR, error)
    MAX_ERRORS[category] = max(MAX_ERRORS.get(category, 0.0), error)
    require(error <= tolerance, f'{label}: {actual!r} != {expected!r}; normalized error {error!r}')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def invalid(value):
        raise RuntimeError('Nonfinite JSON constant: ' + value)
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=invalid)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def mean(values):
    return math.fsum(values) / len(values)


def dot(left, right):
    require(len(left) == len(right), 'dot shape')
    return math.fsum(a * b for a, b in zip(left, right))


def centered(values):
    average = mean(values)
    return [value - average for value in values]


def variance(values):
    values = centered(values)
    return math.fsum(value * value for value in values) / len(values)


def correlation(left, right):
    require(len(left) == len(right), 'correlation shape')
    if len(left) < 2:
        return None
    x, y = centered(left), centered(right)
    xx, yy = dot(x, x), dot(y, y)
    if xx == 0.0 or yy == 0.0:
        return None
    return max(-1.0, min(1.0, dot(x, y) / math.sqrt(xx * yy)))


def moment(values):
    count = len(values)
    if count < 2:
        return {'count': count, 'mean': None, 'rms': None, 'sd': None,
                'variance': None, 'drift_share': None}
    average = mean(values)
    square = math.fsum(value * value for value in values) / count
    var = math.fsum((value - average) ** 2 for value in values) / count
    rms = math.sqrt(square)
    sd = math.sqrt(var)
    near(square, var + average * average, 'unfloored RMS/variance/mean identity', category='moment_identity')
    return {'count': count, 'mean': average, 'rms': rms, 'sd': sd,
            'variance': var, 'drift_share': average * average / square if square > 0.0 else None}


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None,
                'std': None, 'rms': None}
    return {'count': len(values), 'mean': mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'std': math.sqrt(variance(values)),
            'rms': math.sqrt(math.fsum(value * value for value in values) / len(values))}


def symmetric_relative(previous, current):
    scale = max(abs(previous), abs(current))
    if scale == 0.0:
        return 0.0
    left, right = previous / scale, current / scale
    result = 2.0 * (right - left) / (abs(left) + abs(right))
    require(math.isfinite(result) and abs(result) <= 2.0 + 1e-15, 'bounded symmetric fractional change')
    return result


def compare_tree(actual, expected, label, category='arithmetic'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': object required')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            compare_tree(actual[key], value, label + '.' + key, category)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list shape')
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_tree(left, right, f'{label}[{index}]', category)
    elif isinstance(expected, bool) or isinstance(expected, str):
        require(actual == expected, label + ': exact value')
    else:
        near(actual, expected, label, category=category)


def score(prediction, targets, prediction_valid, target_valid):
    require(len(prediction) == len(targets) == len(prediction_valid) == len(target_valid), 'score row shapes')
    eligible = sum(bool(value) for value in target_valid)
    pairs = [(p, t) for p, t, pv, tv in zip(prediction, targets, prediction_valid, target_valid) if pv and tv]
    if not pairs:
        return {'total': len(targets), 'target_eligible': eligible, 'valid': 0,
                'coverage': 0.0, 'mae': None, 'rmse': None, 'r2': None}
    error = [p - t for p, t in pairs]
    y = [t for _, t in pairs]
    sse = math.fsum(value * value for value in error)
    average = mean(y)
    sst = math.fsum((value - average) ** 2 for value in y)
    return {'total': len(targets), 'target_eligible': eligible, 'valid': len(pairs),
            'coverage': len(pairs) / eligible if eligible else 0.0,
            'mae': mean([abs(value) for value in error]), 'rmse': math.sqrt(sse / len(pairs)),
            'r2': 1.0 - sse / sst if sst > 0.0 else None}


def stat_vector(values, bank=False):
    result = moment(values)
    out = [result['count'], result['mean'], result['rms'], result['sd'], result['drift_share']]
    if bank:
        rms = result['rms']
        out += [None if rms is None else (max(rms, 1e-6) if rms > 0.0 else 0.0),
                int(rms is not None and 0.0 < rms < 1e-6)]
    return out


def scaler_identity(scaler):
    value = 14695981039346656037
    def add(data):
        nonlocal value
        for byte in data:
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    def tensor(data, shape, dtype, format):
        add(struct.pack('<q', dtype))
        for dimension in shape:
            add(struct.pack('<q', dimension))
        def flatten(items):
            for item in items:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item
        for item in flatten(data):
            add(struct.pack(format, item))
    tensor(scaler['mean'], [3, 9], 7, '<d')
    tensor(scaler['scale'], [3, 9], 7, '<d')
    tensor(scaler['count'], [3, 9], 4, '<q')
    tensor(scaler['channel_ids'], [3], 4, '<q')
    tensor(scaler['floor_applied'], [3, 9], 11, '<?')
    add(struct.pack('<d', scaler['scale_floor']))
    return f'rpb-scaler-v1-fnv1a64-{value:016x}'


def movement_targets(raw, observed):
    means, deviations, counts = [], [], []
    for channel in range(3):
        returns = [math.log(raw[channel][t][3]) - math.log(raw[channel][t - 1][3])
                   for t in range(1, 128)
                   if observed[channel][t][3] and observed[channel][t - 1][3]
                   and raw[channel][t][3] > 0.0 and raw[channel][t - 1][3] > 0.0]
        counts.append(len(returns))
        means.append(mean(returns) if returns else 0.0)
        deviations.append(math.sqrt(variance(returns)) if returns else 0.0)
    valid = all(count >= 8 for count in counts)
    return ([mean(deviations), mean(means)] if valid else [0.0, 0.0]), [int(valid), int(valid)]


def recompute_witness(witness):
    scaled, observed, scaler = witness['scaled_values'], witness['observed'], witness['scaler']
    require(len(scaled) == len(observed) == 112, '112 TRAIN windows')
    require(scaler['channel_ids'] == [0, 1, 2], 'exact canonical channel IDs')
    near(scaler['scale_floor'], 1e-6, 'saved scaler floor', tolerance=0.0)
    require(scaler_identity(scaler) == scaler['identity'] ==
            'rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969', 'exact FrozenScaler content identity')
    scaled_moments, symmetric_moments, close_moments, close_levels = [], [], [], []
    bank_values, bank_valid, targets64, targets32, target_valid = [], [], [], [], []
    roundtrip_error = 0.0
    for row in range(112):
        require(len(scaled[row]) == len(observed[row]) == 3, 'C3 geometry')
        raw, raw32, model = [], [], []
        row_scaled, row_symmetric, row_close, row_levels = [], [], [], []
        row_bank, row_bank_valid = [], []
        for channel in range(3):
            require(len(scaled[row][channel]) == len(observed[row][channel]) == 128, 'H128 geometry')
            channel_raw, channel_raw32, channel_model = [], [], []
            for tick in range(128):
                require(len(scaled[row][channel][tick]) == len(observed[row][channel][tick]) == 9, 'F9 geometry')
                line_raw, line_raw32, line_model = [], [], []
                for feature in range(9):
                    mask = observed[row][channel][tick][feature]
                    require(mask in (0, 1), 'Boolean observation support')
                    z = scaled[row][channel][tick][feature]
                    require(math.isfinite(z), 'finite witness normalized storage')
                    mu, sigma = scaler['mean'][channel][feature], scaler['scale'][channel][feature]
                    require(math.isfinite(mu) and math.isfinite(sigma) and sigma >= 1e-6 and
                            scaler['count'][channel][feature] > 0, 'saved scaler statistics')
                    x = z * sigma + mu if mask else 0.0
                    line_raw.append(x)
                    line_raw32.append(f32(x))
                    line_model.append(f32(z) if mask else 0.0)
                    if mask:
                        recovered = (x - mu) / sigma
                        roundtrip_error = max(roundtrip_error, abs(recovered - z) / (1.0 + abs(z)))
                channel_raw.append(line_raw)
                channel_raw32.append(line_raw32)
                channel_model.append(line_model)
            raw.append(channel_raw)
            raw32.append(channel_raw32)
            model.append(channel_model)
        for channel in range(3):
            channel_scaled, channel_symmetric = [], []
            for feature in range(9):
                blocks, symmetric_blocks = [], []
                for partition in range(5):
                    begin, end = (0, 127) if partition == 0 else ((partition - 1) * 127 // 4, partition * 127 // 4)
                    differences, relative = [], []
                    for endpoint in range(begin, end):
                        if observed[row][channel][endpoint][feature] and observed[row][channel][endpoint + 1][feature]:
                            differences.append(model[channel][endpoint + 1][feature] - model[channel][endpoint][feature])
                            relative.append(symmetric_relative(raw[channel][endpoint][feature], raw[channel][endpoint + 1][feature]))
                    blocks.append(stat_vector(differences, bank=True))
                    symmetric_blocks.append(stat_vector(relative))
                channel_scaled.append(blocks)
                channel_symmetric.append(symmetric_blocks)
                global_block = blocks[0]
                eligible = global_block[0] >= 2
                bank = [global_block[1], global_block[5]] if eligible else [0.0, 0.0]
                support = [int(eligible), int(eligible)]
                denominator = max(global_block[5], 1e-6) if eligible else 1e-6
                for quarter in range(1, 5):
                    active = eligible and blocks[quarter][0] >= 2
                    bank += [blocks[quarter][1] / denominator, blocks[quarter][5] / denominator] if active else [0.0, 0.0]
                    support += [int(active), int(active)]
                row_bank += [f32(value) for value in bank]
                row_bank_valid += support
            row_scaled.append(channel_scaled)
            row_symmetric.append(channel_symmetric)
            levels = [raw[channel][tick][3] for tick in range(128)
                      if observed[row][channel][tick][3] and raw[channel][tick][3] > 0.0]
            row_levels.append([len(levels), mean(levels) if levels else None,
                               statistics.median(levels) if levels else None])
            close_blocks = []
            for partition in range(5):
                begin, end = (0, 127) if partition == 0 else ((partition - 1) * 127 // 4, partition * 127 // 4)
                values = [[], [], []]
                for endpoint in range(begin, end):
                    a, b = raw[channel][endpoint][3], raw[channel][endpoint + 1][3]
                    if (observed[row][channel][endpoint][3] and observed[row][channel][endpoint + 1][3]
                            and a > 0.0 and b > 0.0):
                        values[0].append(b - a)
                        values[1].append((b - a) / a)
                        values[2].append(math.log(b) - math.log(a))
                close_blocks.append([stat_vector(value) for value in values])
            row_close.append(close_blocks)
        target64, valid64 = movement_targets(raw, observed[row])
        target32, valid32 = movement_targets(raw32, observed[row])
        require(valid64 == valid32, 'raw64/raw32 target support')
        targets64.append(target64)
        targets32.append(target32)
        target_valid.append(valid32)
        scaled_moments.append(row_scaled)
        symmetric_moments.append(row_symmetric)
        close_moments.append(row_close)
        close_levels.append(row_levels)
        bank_values.append(row_bank)
        bank_valid.append(row_bank_valid)
    derived = {'scaled_moments': scaled_moments, 'symmetric_moments': symmetric_moments,
               'close_moments': close_moments, 'close_levels': close_levels,
               'bank_values': bank_values, 'bank_valid': bank_valid,
               'reconstructed_targets_64': targets64, 'reconstructed_targets_32': targets32,
               'reconstructed_target_valid': target_valid}
    for key, value in derived.items():
        compare_tree(witness['derived'][key], value, 'witness.derived.' + key, category=key)
    near(witness['derived']['maximum_normalized_errors']['inverse_roundtrip'], roundtrip_error,
         'inverse roundtrip error', category='raw_roundtrip')
    saved = witness['targets']
    require(len(saved['values']) == len(saved['valid']) == len(saved['row_ids']) == len(saved['group_ids']) == 112, 'saved target association shapes')
    for mode, reconstructed in (('64', targets64), ('32', targets32)):
        maximum = 0.0
        for row in range(112):
            for task in range(2):
                require(bool(saved['valid'][row][task]) == bool(target_valid[row][task]), 'saved reconstructed target support')
                if target_valid[row][task]:
                    near(reconstructed[row][task], saved['values'][row][task],
                         'original saved movement target from raw' + mode, category='target_reconstruction_' + mode)
                    maximum = max(maximum, abs(reconstructed[row][task] - saved['values'][row][task]) /
                                  (1.0 + abs(saved['values'][row][task])))
        near(witness['derived']['maximum_normalized_errors']['target' + mode], maximum,
             'target discrepancy raw' + mode)
    return derived


def recompute_methods(witness):
    methods, saved = witness['methods'], witness['targets']
    require([method['model'] for method in methods] ==
            ['RPB-v22.market-02', 'RPB-v24.market-01.alt-01', 'RPB-v25.market-01'], 'three exact saved model labels')
    computed = {}
    for method in methods:
        width = method['width']
        require(width == (32 if method['model'] == 'RPB-v22.market-02' else 36), 'registered native dimension')
        require(len(method['raw_features']) == len(method['feature_valid']) == 112, 'feature row geometry')
        require(len(method['feature_mean']) == len(method['feature_scale']) == len(method['ridge_weights']) == width, 'fit width geometry')
        rows = [row for row in range(112) if method['feature_valid'][row]]
        for coordinate in range(width):
            values = [method['raw_features'][row][coordinate] for row in rows]
            near(method['feature_mean'][coordinate], mean(values), 'existing feature mean', category='fit_normalizers')
            near(method['feature_scale'][coordinate], max(math.sqrt(variance(values)), 1e-8), 'existing feature scale', category='fit_normalizers')
        for task in range(4):
            values = [saved['values'][row][task] for row in range(112) if saved['valid'][row][task]]
            near(method['target_mean'][task], mean(values), 'existing target mean', category='fit_normalizers')
            near(method['target_scale'][task], max(math.sqrt(variance(values)), 1e-8), 'existing target scale', category='fit_normalizers')
        prefix, suffix, prediction = [], [], []
        intercept = [method['ridge_intercept'][task] * method['target_scale'][task] +
                     method['target_mean'][task] for task in range(4)]
        for row in range(112):
            require(len(method['raw_features'][row]) == width, 'feature width')
            standardized = [(method['raw_features'][row][j] - method['feature_mean'][j]) /
                            method['feature_scale'][j] for j in range(width)]
            p = [math.fsum(standardized[j] * method['ridge_weights'][j][task] for j in range(32)) *
                 method['target_scale'][task] for task in range(4)]
            s = [math.fsum(standardized[j] * method['ridge_weights'][j][task] for j in range(32, width)) *
                 method['target_scale'][task] for task in range(4)]
            full = [p[task] + s[task] + intercept[task] for task in range(4)]
            for task in range(4):
                near(full[task], method['saved_ridge_predictions'][row][task], 'existing ridge prediction', category='ridge_predictions')
                actual_standardized = (method['saved_ridge_predictions'][row][task] - method['target_mean'][task]) / method['target_scale'][task]
                expected_standardized = math.fsum(standardized[j] * method['ridge_weights'][j][task] for j in range(width)) + method['ridge_intercept'][task]
                near(actual_standardized, expected_standardized, 'existing standardized ridge prediction', category='ridge_predictions_standardized')
            prefix.append(p)
            suffix.append(s)
            prediction.append(full)
        computed[method['model']] = {'prefix': prefix, 'suffix': suffix, 'prediction': prediction, 'intercept': intercept}
    parent = computed['RPB-v22.market-02']
    for method in methods:
        result = computed[method['model']]
        result['prefix_shift'] = [[result['prefix'][row][task] - parent['prefix'][row][task]
                                   for task in range(4)] for row in range(112)]
        result['prediction_delta'] = [[result['prediction'][row][task] - parent['prediction'][row][task]
                                       for task in range(4)] for row in range(112)]
        for key in ('prefix', 'suffix', 'prediction', 'prefix_shift', 'prediction_delta'):
            compare_tree(method['derived'][key], result[key], method['model'] + '.' + key, category='ridge_decomposition')
        compare_tree(method['derived']['intercept'], result['intercept'], method['model'] + '.intercept', category='ridge_decomposition')
        for row in range(112):
            require(method['raw_features'][row][:32] == methods[0]['raw_features'][row][:32], 'exact saved prefix32')
    residual = [[saved['values'][row][task] - parent['prediction'][row][task] for task in range(4)] for row in range(112)]
    compare_tree(witness['derived']['parent_residual'], residual, 'parent residual', category='ridge_decomposition')
    return computed, residual


METRIC_NAMES = ('level_mean', 'level_median', 'scaled_mean', 'scaled_rms', 'scaled_sd',
                'scaled_drift_share', 'raw_mean', 'raw_rms', 'raw_sd', 'raw_drift_share',
                'relative_rms', 'relative_sd', 'log_rms', 'log_sd', 'symmetric_rms', 'symmetric_sd')
METRIC_PAIRS = (('level_mean', 'scaled_rms'), ('level_mean', 'scaled_sd'),
                ('level_mean', 'raw_rms'), ('level_mean', 'raw_sd'),
                ('level_mean', 'relative_rms'), ('level_mean', 'relative_sd'),
                ('level_mean', 'log_rms'), ('level_mean', 'log_sd'),
                ('scaled_rms', 'scaled_sd'), ('scaled_rms', 'scaled_drift_share'),
                ('scaled_rms', 'log_sd'), ('scaled_rms', 'symmetric_sd'),
                ('scaled_sd', 'log_sd'), ('scaled_sd', 'symmetric_sd'),
                ('raw_rms', 'log_sd'), ('raw_rms', 'raw_sd'),
                ('relative_sd', 'log_sd'), ('symmetric_sd', 'log_sd'))


def recompute_metrics(witness, derived):
    vectors = {}
    for channel in range(3):
        for name in METRIC_NAMES:
            values = []
            for row in range(112):
                scaled = derived['scaled_moments'][row][channel][3][0]
                raw, relative, log = derived['close_moments'][row][channel][0]
                symmetric = derived['symmetric_moments'][row][channel][3][0]
                levels = derived['close_levels'][row][channel]
                mapping = {'level_mean': levels[1], 'level_median': levels[2],
                           'scaled_mean': scaled[1], 'scaled_rms': scaled[2], 'scaled_sd': scaled[3],
                           'scaled_drift_share': scaled[4], 'raw_mean': raw[1], 'raw_rms': raw[2],
                           'raw_sd': raw[3], 'raw_drift_share': raw[4], 'relative_rms': relative[2],
                           'relative_sd': relative[3], 'log_rms': log[2], 'log_sd': log[3],
                           'symmetric_rms': symmetric[2], 'symmetric_sd': symmetric[3]}
                values.append(mapping[name])
            vectors[f'c{channel}.{name}'] = values
    for name in METRIC_NAMES:
        values = []
        for row in range(112):
            channels = [vectors[f'c{channel}.{name}'][row] for channel in range(3)]
            values.append(mean(channels) if all(value is not None for value in channels) else None)
        vectors['aggregate.' + name] = values
    actual = witness['derived']['metric_vectors']
    require(len(actual) == len(vectors) == 64, 'all channel and equal-channel metric vectors')
    require(len({value['name'] for value in actual}) == 64, 'unique metric-vector names')
    for value in actual:
        require(value['name'] in vectors, 'predeclared metric-vector name')
        compare_tree(value['values'], vectors[value['name']], 'metric.' + value['name'], category='metric_vectors')
    return vectors


def association(left, right):
    pairs = [(x, y) for x, y in zip(left, right) if x is not None and y is not None]
    positive = [(math.log(x), math.log(y)) for x, y in pairs if x > 0.0 and y > 0.0]
    return {'count': len(pairs), 'pearson': correlation([x for x, _ in pairs], [y for _, y in pairs]),
            'positive_log_count': len(positive),
            'positive_log_pearson': correlation([x for x, _ in positive], [y for _, y in positive])}


def group_indices(group):
    if group == 'total':
        return list(range(270))
    if group == 'global_mean':
        return [j for j in range(270) if j % 10 == 0]
    if group == 'global_rms':
        return [j for j in range(270) if j % 10 == 1]
    if group == 'global_slots':
        return [j for j in range(270) if j % 10 < 2]
    if group == 'quarter_slots':
        return [j for j in range(270) if j % 10 >= 2]
    if group.startswith('quarter_') and group[-1:] in '0123':
        quarter = int(group[-1:])
        return [j for j in range(270) if 2 + quarter * 2 <= j % 10 < 4 + quarter * 2]
    if group.startswith('channel_') and group[-1:] in '012':
        return [j for j in range(270) if j // 90 == int(group[-1:])]
    if group.startswith('feature_') and group[-1:] in '012345678':
        return [j for j in range(270) if (j // 10) % 9 == int(group[-1:])]
    if group == 'close_slots':
        return [j for j in range(270) if (j // 10) % 9 == 3]
    raise RuntimeError('Unknown predeclared bank group: ' + group)


def audit_report(report, witness, derived, metrics, methods, residual):
    contexts = {name: (start, end) for name, start, end in CONTEXTS}
    families = {'scaled', 'symmetric', 'close_raw', 'close_proportional', 'close_log'}
    seen = set()
    for entry in report['moment_summaries']:
        family, channel, feature, partition = entry['family'], entry['channel'], entry['feature'], entry['partition']
        require(family in families and 0 <= channel < 3 and 0 <= partition < 5 and
                0 <= feature < 9, 'moment summary coordinate')
        require((family, channel, feature, partition) not in seen, 'unique moment summary')
        seen.add((family, channel, feature, partition))
        if family in ('scaled', 'symmetric'):
            blocks = [derived[family + '_moments'][row][channel][feature][partition] for row in range(112)]
        else:
            require(feature == 3, 'close summary feature3')
            transform = {'close_raw': 0, 'close_proportional': 1, 'close_log': 2}[family]
            blocks = [derived['close_moments'][row][channel][partition][transform] for row in range(112)]
        require(len(entry['contexts']) == 3 and {value['context'] for value in entry['contexts']} == set(contexts),
                'all fixed moment contexts')
        for context in entry['contexts']:
            start, end = contexts[context['context']]
            for name, index in (('count', 0), ('mean', 1), ('rms', 2), ('sd', 3), ('drift_share', 4)):
                compare_tree(context['metrics'][name], describe([value[index] for value in blocks[start:end]]),
                             'summary.' + family + '.' + name, category='moment_summaries')
            if family == 'scaled':
                compare_tree(context['metrics']['bank_rms'], describe([value[5] for value in blocks[start:end]]),
                             'summary.bank_rms', category='moment_summaries')
                near(context['floor_active_rows'], sum(value[6] for value in blocks[start:end]), 'floor-active count', tolerance=0.0)
    require(len(seen) == (2 * 3 * 9 * 5 + 3 * 3 * 5), 'all 315 channel/feature/partition moment summaries')
    require(len(report['close_level_summaries']) == 9, 'all close level contexts')
    require({(value['channel'], value['context']) for value in report['close_level_summaries']} ==
            {(channel, name) for channel in range(3) for name in contexts}, 'exact close-level summary inventory')
    for entry in report['close_level_summaries']:
        start, end = contexts[entry['context']]
        for name, index in (('mean_level', 1), ('median_level', 2)):
            compare_tree(entry[name], describe([derived['close_levels'][row][entry['channel']][index] for row in range(start, end)]),
                         'close_level.' + name, category='close_levels_summary')
    require(len(report['bank_balance']) == 3, 'all bank-balance contexts')
    require({value['context'] for value in report['bank_balance']} == set(contexts), 'exact bank-balance contexts')
    for entry in report['bank_balance']:
        start, end = contexts[entry['context']]
        require(len(entry['coordinates']) == 270, 'all 270 bank coordinates')
        coordinates = []
        for coordinate in range(270):
            values = [derived['bank_values'][row][coordinate] for row in range(start, end)
                      if derived['bank_valid'][row][coordinate]]
            square = math.fsum(value * value for value in values)
            center_square = math.fsum(value * value for value in centered(values)) if values else 0.0
            count = len(values)
            expected = {'coordinate': coordinate, 'channel': coordinate // 90,
                        'feature': (coordinate // 10) % 9, 'moment': coordinate % 10, 'count': count,
                        'mean': mean(values) if count else None,
                        'std': math.sqrt(center_square / count) if count else None,
                        'rms': math.sqrt(square / count) if count else None,
                        'sum_squares': square, 'centered_sum_squares': center_square,
                        'mean_square': square / count if count else None,
                        'variance': center_square / count if count else None}
            compare_tree(entry['coordinates'][coordinate], expected, 'bank.coordinate', category='bank_balance')
            coordinates.append(expected)
        total_square = math.fsum(value['sum_squares'] for value in coordinates)
        total_variance = math.fsum(value['variance'] for value in coordinates if value['variance'] is not None)
        names = {'total', 'global_mean', 'global_rms', 'global_slots', 'quarter_slots', 'close_slots'}
        names.update('quarter_' + str(j) for j in range(4))
        names.update('channel_' + str(j) for j in range(3))
        names.update('feature_' + str(j) for j in range(9))
        require(len(entry['groups']) == 22 and {value['group'] for value in entry['groups']} == names,
                'all 22 predeclared bank groups')
        for group in entry['groups']:
            members = [coordinates[j] for j in group_indices(group['group'])]
            square = math.fsum(value['sum_squares'] for value in members)
            var = math.fsum(value['variance'] for value in members if value['variance'] is not None)
            expected = {'sum_squares': square,
                        'centered_sum_squares': math.fsum(value['centered_sum_squares'] for value in members),
                        'mean_square_sum': math.fsum(value['mean_square'] for value in members if value['mean_square'] is not None),
                        'variance_sum': var,
                        'raw_squared_share': square / total_square if total_square > 0.0 else None,
                        'centered_variance_share': var / total_variance if total_variance > 0.0 else None}
            compare_tree(group, expected, 'bank.group.' + group['group'], category='bank_group_balance')
    require(len(report['metric_associations']) == 216, '18 planned associations per channel/aggregate and context')
    seen = set()
    for entry in report['metric_associations']:
        require(entry['left'] in metrics and entry['right'] in metrics, 'metric association names')
        key = (entry['context'], entry['left'], entry['right'])
        require(key not in seen, 'unique metric association')
        seen.add(key)
        start, end = contexts[entry['context']]
        compare_tree(entry, association(metrics[entry['left']][start:end], metrics[entry['right']][start:end]),
                     'metric association', category='metric_associations')
    require(seen == {(context, prefix + left, prefix + right) for context in contexts
                     for prefix in ('c0.', 'c1.', 'c2.', 'aggregate.') for left, right in METRIC_PAIRS},
            'exact prospectively fixed metric association population')
    require(len(report['methods']) == 3, 'three methods in report')
    require({value['model'] for value in report['methods']} == set(methods), 'exact report method inventory')
    saved = witness['targets']
    for entry in report['methods']:
        model = entry['model']
        require(model in methods, 'saved method in report')
        source = next(value for value in witness['methods'] if value['model'] == model)
        require(entry['width'] == source['width'], 'reported method width')
        result = methods[model]
        require(len(entry['contexts']) == 3 and {value['context'] for value in entry['contexts']} == set(contexts),
                'all fixed method contexts')
        for context in entry['contexts']:
            start, end = contexts[context['context']]
            require([value['task'] for value in context['tasks']] == list(TASKS), 'four separate task rows')
            for task, value in enumerate(context['tasks']):
                prediction = [result['prediction'][row][task] for row in range(start, end)]
                target = [saved['values'][row][task] for row in range(start, end)]
                pv = [source['prediction_valid'][row][task] for row in range(start, end)]
                tv = [saved['valid'][row][task] for row in range(start, end)]
                expected_score = score(prediction, target, pv, tv)
                compare_tree(value['linear'], {key: expected_score[key] for key in
                             ('total', 'valid', 'coverage', 'r2', 'mae', 'rmse')}, model + '.score', category='ridge_scores')
                selected = [row for row in range(start, end) if source['prediction_valid'][row][task] and saved['valid'][row][task]]
                components = {name: [result[name][row][task] for row in selected]
                              for name in ('prefix', 'suffix', 'prefix_shift', 'prediction_delta')}
                components['parent_residual'] = [residual[row][task] for row in selected]
                components['intercept'] = [result['intercept'][task] for _ in selected]
                for name, values in components.items():
                    compare_tree(value['components'][name], describe(values), model + '.component.' + name, category='component_summaries')
                require(len(value['associations']) == 80 and
                        {(item['component'], item['metric']) for item in value['associations']} ==
                        {(name, 'aggregate.' + metric) for name in components if name != 'intercept' for metric in METRIC_NAMES},
                        'all 80 fixed component associations per task/context')
                for item in value['associations']:
                    require(item['component'] in components and item['metric'].startswith('aggregate.') and item['metric'] in metrics,
                            'predeclared component/aggregate association')
                    expected = association(components[item['component']], [metrics[item['metric']][row] for row in selected])
                    compare_tree(item, expected, model + '.component_association', category='component_associations')


def main():
    started = time.monotonic()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT,
            'Run in the existing managed cuwacunu container')
    require(len(sys.argv) == 2, 'Usage: audit-market-change-scale.py RUN_DIRECTORY')
    run = Path(sys.argv[1]).resolve()
    require(run.parent == ROOT / 'output/runs/market-change-scale-diagnosis' and
            run.name.startswith('run-'), 'Exact diagnostic run root required')
    output = run / 'independent-audit.json'
    require(not output.exists(), 'Preserve the previous arithmetic audit')
    for path, expected in EXPECTED.items():
        require(sha(path) == expected, 'Prospective declaration hash: ' + str(path))
    manifest = load(MANIFEST)
    require(manifest['protocol'] == PROTOCOL and len(manifest['members']) == 12 and
            manifest['card_sha256'] == EXPECTED[CARD], 'Exact 12-input declaration')
    expected_inputs = {}
    for member in manifest['members']:
        path = (ROOT / member['path']).resolve()
        require(path.is_relative_to(ROOT) and path.suffix == '.pt' and
                member['path'] not in expected_inputs, 'Unique contained archive allowlist')
        require(path.stat().st_size == member['bytes'] and sha(path) == member['sha256'],
                'Allowlisted TRAIN input hash/size: ' + member['path'])
        expected_inputs[member['path']] = member
    preflight = ROOT / 'doc/results/market_change_scale_diagnosis_v1/preflight.json'
    require(sha(preflight) == manifest['preflight_sha256'], 'Sealed preflight unchanged')
    reader = run / 'reader'
    report_path, witness_path, opened_path = [reader / name for name in
                                           ('diagnosis.json', 'numerical-witness.json', 'opened-inputs.json')]
    json_paths = (report_path, witness_path, opened_path)
    json_before = {str(path.relative_to(ROOT)): sha(path) for path in json_paths}
    source_paths = [CARD, ADDENDUM, MANIFEST, preflight, Path(__file__).resolve(),
                    ROOT / 'code/evaluation/market_change_scale_diagnosis_v1/main.cpp',
                    ROOT / 'code/evaluation/market_change_scale_diagnosis_v1/diagnosis.mk',
                    ROOT / 'code/scripts/run-market-change-scale-diagnosis.py',
                    ROOT / 'code/scripts/report-market-change-scale.py', ROOT / 'Makefile',
                    Path('/opt/cuwacunu/build/market-scale-diagnosis/market_change_scale_diagnosis')]
    source_before = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
                     for path in source_paths}
    report, witness, opened = load(report_path), load(witness_path), load(opened_path)
    for value in (report, witness):
        require(value['protocol'] == PROTOCOL, 'Exact reader/witness protocol')
        for key, path in (('card_sha256', CARD), ('addendum_sha256', ADDENDUM), ('input_manifest_sha256', MANIFEST)):
            require(value['pins'][key] == EXPECTED[path], 'Reader declaration pin: ' + key)
        for key, path in (('source_sha256', source_paths[5]), ('makefile_sha256', source_paths[6]),
                          ('binary_sha256', source_paths[-1])):
            require(value['pins'][key] == sha(path), 'Reader source/binary pin: ' + key)
    require(report['admission_checks']['status'] == 'passed' and
            report['admission_checks']['opened_archives'] == 12 and
            report['admission_checks']['exact_prefix32'] is True and
            report['admission_checks']['exact_row_group_target_support'] is True,
            'Reader exact admission checks')
    require(all(report[key] == 0 for key in ('encoder_calls', 'encoder_updates', 'head_fits',
                                            'scaler_fits', 'held_out_tensor_inputs')), 'Zero unpermitted operations')
    require(report['numerical_admission_tolerance'] == TOLERANCE, 'Unchanged numerical tolerance')
    require(opened == report['opened_inputs'] and len(opened) == 12, 'Actual 12-archive loader ledger')
    require({entry['path'] for entry in opened} == set(expected_inputs), 'Loader matches allowlist exactly')
    for entry in opened:
        expected = expected_inputs[entry['path']]
        require(entry['sha256'] == expected['sha256'] and entry['bytes'] == expected['bytes'],
                'Loader hash/size entry')
    derived = recompute_witness(witness)
    methods, residual = recompute_methods(witness)
    metrics = recompute_metrics(witness, derived)
    audit_report(report, witness, derived, metrics, methods, residual)
    for category, error in report['maximum_normalized_errors'].items():
        require(isinstance(error, (int, float)) and math.isfinite(error) and 0.0 <= error <= TOLERANCE,
                'Reported admission bound: ' + category)
    require(json_before == {str(path.relative_to(ROOT)): sha(path) for path in json_paths},
            'All numerical JSON inputs unchanged')
    for name, expected in source_before.items():
        path = ROOT / name if not Path(name).is_absolute() else Path(name)
        require(sha(path) == expected, 'Source/declaration/binary unchanged: ' + name)
    for name, member in expected_inputs.items():
        path = ROOT / name
        require(path.stat().st_size == member['bytes'] and sha(path) == member['sha256'],
                'Allowlisted TRAIN archive unchanged after scalar audit: ' + name)
    result = {'kind': 'market_change_scale_independent_arithmetic_audit_v1', 'protocol': PROTOCOL,
              'status': 'passed', 'checks': CHECKS, 'maximum_normalized_error': MAX_ERROR,
              'maximum_normalized_error_by_category': MAX_ERRORS,
              'normalized_absolute_tolerance': TOLERANCE, 'input_archive_count': 12,
              'input_sha256': {name: value['sha256'] for name, value in expected_inputs.items()},
              'json_input_sha256': json_before, 'source_sha256': source_before,
              'card_sha256': EXPECTED[CARD], 'addendum_sha256': EXPECTED[ADDENDUM],
              'input_manifest_sha256': EXPECTED[MANIFEST],
              'witness': str(witness_path.relative_to(ROOT)), 'witness_sha256': json_before[str(witness_path.relative_to(ROOT))],
              'opened_inputs_sha256': json_before[str(opened_path.relative_to(ROOT))],
              'new_encoder_calls': 0, 'new_encoder_updates': 0, 'head_or_scaler_fits': 0,
              'tensor_archives_deserialized_by_this_audit': 0,
              'elapsed_seconds': time.monotonic() - started}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

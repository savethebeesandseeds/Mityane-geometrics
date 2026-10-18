#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Audit saved V30 transfer scalars with stdlib; no tensor/model loads or fits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time

ROOT = Path('/cuwacunu')
PROTOCOL = 'market-location-scale-synthetic-transfer-v1'
CARD = 'c3cfe0a04b174734599b2b01363c90c3f9b81fd5964fb30ffd4ce8a21976260e'
CHECKPOINT = 'c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb'
SETTINGS = '43320396447ca80be5fde4324d17c3a9ab129e116bf184455fcb6261aeb6923c'
MANIFEST = 'fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb'
NORMALIZER = 'rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476'
MASTERS = (941101, 941102, 941103)
MODES = ('synthetic_train', 'market_scaler')
ROLES = ('training', 'validation', 'deleted')
VIEWS = ('validation_intact', 'validation_deleted')
HEADS = ('ridge', 'tiny_secondary')
REPETITIONS = (2701, 2802, 2903)
SCALERS = {941101: 'rpb-scaler-v1-fnv1a64-ac9d9673f1e24751',
           941102: 'rpb-scaler-v1-fnv1a64-5f51eb3918f161a4',
           941103: 'rpb-scaler-v1-fnv1a64-3a642694962518e9'}
MARKET_SCALER = 'rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969'
FLOORS = {'synthetic_train': {'validation_intact': (0.9895833333333334, 0.9774305555555556),
                             'validation_deleted': (0.93359375, 0.9253472222222222)},
          'market_scaler': {'validation_intact': (1., 0.9952256944444444),
                            'validation_deleted': (0.9609375, 0.9505208333333334)}}
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
CHECKS = 0
MAX_ERRORS = {}


def require(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual, expected, label, tolerance=1e-12):
    require(finite(actual) and finite(expected), 'Finite scalar: ' + label)
    error = abs(actual - expected)
    MAX_ERRORS[label] = max(MAX_ERRORS.get(label, 0.), error)
    require(error <= tolerance, 'Scalar arithmetic: ' + label)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bits32(value):
    return struct.pack('<f', value)


def exact32(actual, expected, label):
    require(finite(actual) and finite(expected) and f32(actual) == actual and
            bits32(actual) == bits32(expected), 'Exact F32: ' + label)


def near32(actual, expected, label):
    require(finite(actual) and finite(expected) and f32(actual) == actual, 'Finite F32: ' + label)
    word = struct.unpack('<I', bits32(abs(expected)))[0]
    ulp = (struct.unpack('<f', struct.pack('<I', word + 1))[0] - abs(expected)
           if word < 0x7f7fffff else abs(expected) - struct.unpack('<f', struct.pack('<I', word - 1))[0])
    error = abs(actual - expected)
    MAX_ERRORS[label] = max(MAX_ERRORS.get(label, 0.), error)
    require(error <= 2 * ulp, 'F32 scalar arithmetic exceeds two ULP: ' + label)


def stream_seed(seed, stream):
    mask = (1 << 64) - 1
    value = (seed + 0x9e3779b97f4a7c15 * stream) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return value ^ (value >> 31)


def named(value):
    result = 14695981039346656037
    for byte in value.encode('ascii'):
        result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return result


def matrix(value, rows, width, label):
    require(len(value) == rows and all(len(row) == width for row in value), 'Matrix shape: ' + label)


def policy_scaler(mode, master):
    return SCALERS[master] if mode == 'synthetic_train' else MARKET_SCALER


def pins(run):
    card = ROOT / 'doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md'
    settings = ROOT / 'config/market_studies/encoder_location_scale_decoder.conf'
    checkpoint = ROOT / 'output/runs/market-location-scale-decoder/run-MAmvTJ/checkpoints/inference-2048.pt'
    inputs = ROOT / 'output/inputs/market-synthetic-transfer-v1'
    for path, digest in ((card, CARD), (settings, SETTINGS), (checkpoint, CHECKPOINT),
                         (inputs / 'manifest.json', MANIFEST)):
        require(sha(path) == digest, 'Frozen prospective pin: ' + str(path))
    manifest = load(inputs / 'manifest.json')
    require(len(manifest['members']) == 18 and tuple(manifest['masters']) == MASTERS, 'Exact staged input family')
    for member in manifest['members']:
        path = inputs / member['path']
        require(path.resolve().is_relative_to(inputs) and path.stat().st_size == member['bytes'] and
                sha(path) == member['sha256'], 'Exact original staged member')
    base = ROOT / 'code/evaluation/market_synthetic_transfer_v1/deps'
    for relative, digest in DEPS.items():
        require(sha(base / relative) == digest, 'Unchanged fixed algorithm dependency: ' + relative)
    for filename in ('sources-before.json', 'features-before-heads.json'):
        saved = load(run / filename)
        require(saved, 'Nonempty saved provenance')
        for name, pin in saved.items():
            path = Path(name)
            require(path.is_file() and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                    'Unchanged admitted source/feature: ' + name)


def receipts(features, summary):
    data = load(features / 'preprocessing-receipts.json')
    values = data['receipts']
    require(len(values) == 768, 'All per-batch preprocessing receipts')
    expected = {(mode, master, role, offset) for mode in MODES for master in MASTERS
                for role in ROLES for offset in range(0, 512 if role == 'training' else 256, 8)}
    seen = set()
    lookup = {}
    for value in values:
        key = (value['mode'], value['master'], value['role'], value['offset'])
        require(key in expected and key not in seen, 'Unique exact batch plan')
        seen.add(key)
        lookup[key] = value
        require(value['rows'] == 8 and 0 <= value['observed_cells'] <= 27648, 'Receipt B8 observed population')
        for count, maximum in (('raw_rounding_difference_cells', 'max_raw_rounding_delta'),
                               ('normalized_difference_cells', 'max_normalized_delta')):
            require(isinstance(value[count], int) and 0 <= value[count] <= value['observed_cells'], 'Receipt difference count')
            require(finite(value[maximum]) and value[maximum] >= 0 and
                    ((value[count] == 0) == (value[maximum] == 0)), 'Receipt finite difference maximum')
        require(value['hidden_zero'] is True and value['mask_metadata_exact'] is True and
                value['poison_checks'] == (value['offset'] == 0), 'Batch mask/metadata/poison checks')
    require(seen == expected, 'Complete batch receipt coverage')
    require(summary['full_model_encode_calls'] == summary['total_CUDA_encoder_calls'] == 768 and
            summary['total_rows'] == 6144 and summary['parent_correctness_calls'] == 6 and
            summary['parent_correctness_rows'] == 48 and summary['total_parent_forwards'] == 774,
            'Exact full/parent extraction call accounting')
    runs = summary['runs']
    require(len(runs) == 6 and {(v['mode'], v['master']) for v in runs} ==
            {(mode, master) for mode in MODES for master in MASTERS}, 'Both policies and all cohorts')
    for run in runs:
        require(run['scaler_identity'] == policy_scaler(run['mode'], run['master']), 'Fixed policy scaler')
        require(len(run['roles']) == 3 and {v['role'] for v in run['roles']} == set(ROLES), 'All extraction roles')
        for role in run['roles']:
            rows = 512 if role['role'] == 'training' else 256
            require(role['rows'] == rows and 0 <= role['valid'] <= rows and
                    role['CUDA_encoder_calls'] == rows // 8, 'Role rows/full encode calls')
            entries = [v for (mode, master, name, _), v in lookup.items()
                       if (mode, master, name) == (run['mode'], run['master'], role['role'])]
            require(sum(27648 - v['observed_cells'] for v in entries) == role['masked_cells'], 'Role mask total')
    return lookup


def flatten(value):
    if isinstance(value, list):
        return [element for item in value for element in flatten(item)]
    return [value]


def preprocessing(features, lookup, external_scalers):
    witness = load(features / 'preprocessing-witnesses.json')
    require(len(witness['cases']) == 18, 'All bounded first-role preprocessing witnesses')
    seen = set()
    cells_checked = 0
    for case in witness['cases']:
        key = (case['mode'], case['master'], case['role'], case['offset'])
        require(key in lookup and key[-1] == 0 and key not in seen, 'Unique first-batch witness')
        seen.add(key)
        require((case['B'], case['C'], case['H'], case['F']) == (8, 3, 128, 9), 'Exact bounded witness geometry')
        require(case['scaler_identity'] == policy_scaler(key[0], key[1]) and
                case['channel_ids'] == [0, 1, 2] and case['endpoints'] == [7620000] * 8 and
                case['sampling_interval'] == 60000, 'Witness scaler/semantic metadata')
        matrix(case['mean64'], 3, 9, 'scaler means')
        matrix(case['scale64'], 3, 9, 'scaler scales')
        external = external_scalers[key[:2]]
        require(flatten(case['mean64']) == flatten(external['mean']) and
                flatten(case['scale64']) == flatten(external['scale']), 'Witness uses exact policy scaler tensors')
        require(all(finite(x) for row in case['mean64'] for x in row) and
                all(finite(x) and x > 0 for row in case['scale64'] for x in row), 'Finite scaler64')
        mask = case['observed_flat']
        require(len(mask) == 27648 and all(isinstance(x, bool) for x in mask), 'Full original-time witness mask')
        require(sum(mask) == lookup[key]['observed_cells'], 'Witness observed count equals full receipt')
        selected = {((b * 3 + c) * 128 + t) * 9 + f for b in range(8) for c in range(3)
                    for t in (0, 127) for f in range(9)}
        selected.update([i for i, observed in enumerate(mask) if not observed][:32])
        require(len(case['cells']) == len(selected) and {v['flat'] for v in case['cells']} == selected,
                'Exact declared 432 endpoint plus first32 hidden-cell selector')
        round_differences = normalized_differences = 0
        max_round = max_normalized = 0.
        for cell in case['cells']:
            flat = cell['flat']
            b, c, t, f = flat // 3456, (flat // 1152) % 3, (flat // 9) % 128, flat % 9
            require((cell['b'], cell['c'], cell['t'], cell['f']) == (b, c, t, f) and
                    cell['observed'] is mask[flat], 'Original B/C/H/F cell association')
            observed = mask[flat]
            raw = cell['raw64']
            if observed:
                require(finite(raw), 'Observed genuine finite raw64')
                rounded = f32(raw)
                legacy = f32((raw - case['mean64'][c][f]) / case['scale64'][c][f])
                current = f32((rounded - case['mean64'][c][f]) / case['scale64'][c][f])
                require(math.isfinite(rounded), 'Observed rawF32 finite')
                exact32(cell['rounded32'], rounded, 'observed_raw_rounding')
                near32(cell['legacy_scaled32'], legacy, 'legacy_CPUF64_then_F32')
                near32(cell['v30_scaled32'], current, 'V30_rawF32_CPUF64_then_F32')
                delta = abs(rounded - raw)
                # Both actual normalized F32 values are promoted before taking
                # the difference, matching the declared producer receipt.
                normalized_delta = abs(cell['v30_scaled32'] - cell['legacy_scaled32'])
                round_differences += delta != 0
                normalized_differences += normalized_delta != 0
                max_round, max_normalized = max(max_round, delta), max(max_normalized, normalized_delta)
            else:
                require(finite(raw) or raw in ('nan', 'inf', '-inf', 'NaN', 'Infinity', '+Infinity', '-Infinity'), 'Explicit hidden raw nonfinite encoding')
                for field in ('rounded32', 'legacy_scaled32', 'v30_scaled32'):
                    exact32(cell[field], 0., 'sanitized_hidden_positive_zero')
            cells_checked += 1
        receipt = lookup[key]
        require(round_differences <= receipt['raw_rounding_difference_cells'] and
                normalized_differences <= receipt['normalized_difference_cells'] and
                max_round <= receipt['max_raw_rounding_delta'] and
                max_normalized <= receipt['max_normalized_delta'], 'Bounded witness within whole-batch differences')
        for field in ('poison_parent_exact', 'poison_bank_values_exact', 'poison_bank_valid_exact', 'poison_conditioned_exact'):
            require(case[field] is True, 'Hidden poison preprocessing-only invariant: ' + field)
    require(seen == {key for key in lookup if key[-1] == 0}, 'All roles/policies/cohorts witnessed')
    return cells_checked


def parents(features):
    saved = load(features / 'parent-parity.json')
    values = saved['cases']
    require(len(values) == 6, 'Exactly six same-checkpoint parent-only correctness calls')
    seen = set()
    for value in values:
        key = (value['mode'], value['master'])
        require(key not in seen and key in {(mode, master) for mode in MODES for master in MASTERS}, 'Unique parent policy/cohort')
        seen.add(key)
        require(value['role'] == 'training' and value['offset'] == 0 and value['rows'] == 8 and
                value['exact_equal'] is True, 'First TRAIN B8 actual same-input parent parity')
        matrix(value['full_prefix32'], 8, 32, 'full prefix')
        matrix(value['parent_native32'], 8, 32, 'same checkpoint parent')
        for actual, expected in zip(value['full_prefix32'], value['parent_native32']):
            for a, e in zip(actual, expected):
                exact32(a, e, 'same_input_parent_prefix')


def frozen_state(features, summary):
    state = load(features / 'frozen-state.json')
    initial = state['initial']
    parameters, buffers = initial['parameters'], initial['buffers']
    require(len(parameters) == summary['parameter_tensor_count'] == 76 and
            len(buffers) == summary['buffer_tensor_count'] == 0, 'Complete registered model state inventory')
    require(len({value['name'] for value in parameters}) == len(parameters), 'Unique model parameter names')
    total = 0
    for value in parameters:
        require(value['shape'] and all(isinstance(x, int) and x > 0 for x in value['shape']) and
                value['device'] == 'cuda:0' and isinstance(value['requires_grad'], bool) and
                value['gradient_defined'] is False, 'Parameter shape/device/gradient snapshot')
        require(isinstance(value['dtype'], str) and value['dtype'] in ('Float', 'Float32', 'float32', 'torch.float32'), 'CUDA F32 parameter dtype')
        digest = value['content_fnv1a64']
        require(isinstance(digest, str) and len(digest) == 16 and all(c in '0123456789abcdef' for c in digest), 'Lossless tensor-byte FNV64 text')
        require(int(digest, 16) < (1 << 64), 'Tensor-byte FNV64 width')
        total += math.prod(value['shape'])
    require(total == 297560, 'All trained inference parameters preserved including unused decoder')
    normalizer = initial['normalizer']
    old_path = ROOT / 'output/runs/market-relative-changes/run-vaPC4X/bank-standardizer.json'
    require(sha(old_path) == '782128503036956beb1ec87e39ceb3c502395f00f1d0e8a6138e9d8e52d933ee', 'Frozen original normalizer statistics')
    old = load(old_path)
    for key in ('identity', 'counts', 'sum', 'sumsq', 'mean', 'scale', 'support', 'input_geometry',
                'sampling_interval', 'floor', 'source_identity', 'input_scaler_identity', 'layout_identity',
                'recipe_identity', 'bank_family', 'raw_contract', 'increment_recipe'):
        require(normalizer[key] == old[key], 'Exact original unregistered normalizer tensor/metadata: ' + key)
    require(normalizer['identity'] == summary['bank_normalizer_identity'] == NORMALIZER and
            normalizer['minimum_support'] == 2 and normalizer['frozen'] is True and
            normalizer['input_scaler_identity'] == MARKET_SCALER, 'Normalizer remains bound to market calibration')
    for key in ('counts', 'sum', 'sumsq', 'mean', 'scale', 'support'):
        require(len(normalizer[key]) == 270, 'All normalizer coordinates: ' + key)
    checkpoint_scaler = initial['checkpoint_scaler']
    require(checkpoint_scaler['identity'] == summary['checkpoint_scaler_identity'] == MARKET_SCALER,
            'Checkpoint scaler never replaced by policy scaler')
    checks = state['checks']
    expected = {(mode, master, role) for mode in MODES for master in MASTERS for role in ROLES}
    seen = set()
    external_by_policy = {}
    require(len(state['external_initial']) == 6, 'All policy scalers captured before extraction')
    for entry in state['external_initial']:
        policy = (entry['mode'], entry['master'])
        require(policy in {(mode, master) for mode in MODES for master in MASTERS} and
                policy not in external_by_policy and
                entry['external_scaler']['identity'] == policy_scaler(*policy), 'Unique exact initial policy scaler')
        external_by_policy[policy] = entry['external_scaler']
    exits = 0
    for check in checks:
        require(check['exact_equal'] is True and check['parameters'] == parameters and
                check['buffers'] == buffers and check['normalizer'] == normalizer and
                check['checkpoint_scaler'] == checkpoint_scaler, 'Exact all-tensor and unregistered state after each role/exit')
        if check['role'] == 'exit':
            exits += 1
            continue
        key = (check['mode'], check['master'], check['role'])
        require(key in expected and key not in seen, 'Every role state comparison once')
        seen.add(key)
        external = check['external_scaler']
        policy = key[:2]
        require(external['identity'] == policy_scaler(*policy), 'Exact external scaler policy identity')
        require(external == external_by_policy[policy], 'External scaler full tensors unchanged from before extraction')
        if key[0] == 'market_scaler':
            require(external == checkpoint_scaler, 'Market policy uses original checkpoint scaler tensors')
    require(seen == expected and exits == 1 and len(checks) == 19, 'All eighteen role and one exit state checks')
    return len(checks), external_by_policy


def saved_prefixes(features):
    cases = load(features / 'saved-prefix-comparison.json')['cases']
    require(len(cases) == 18, 'All descriptive historical prefix comparisons')
    seen = set()
    for case in cases:
        key = (case['mode'], case['master'], case['role'])
        require(key in {(mode, master, role) for mode in MODES for master in MASTERS for role in ROLES}
                and key not in seen, 'Unique descriptive historical comparison')
        seen.add(key)
        rows = 512 if key[2] == 'training' else 256
        require(case['rows'] == rows and case['row_associations_exact'] is True and
                isinstance(case['validity_exact'], bool) and case['descriptive_only'] is True,
                'Historical comparison scoped to identical saved row associations')
        require(0 <= case['changed_rows'] <= rows and 0 <= case['changed_cells'] <= 32 * rows and
                case['changed_rows'] <= case['changed_cells'] <= 32 * case['changed_rows'] and
                finite(case['max_absolute_delta']) and case['max_absolute_delta'] >= 0 and
                case['exact_equal'] == (case['changed_cells'] == 0) == (case['max_absolute_delta'] == 0),
                'Descriptive change count/max consistency without parity gate')
        current = features / key[0] / ('master-' + str(key[1])) / 'features.pt'
        old = ROOT / 'output/runs/market-synthetic-transfer/run-oujlisd9/features' / key[0] / ('master-' + str(key[1])) / 'features.pt'
        require(sha(current) == case['new_features_sha256'] and sha(old) == case['old_features_sha256'], 'Bound exact old/new descriptive feature archives')


def population(value, rows, groups):
    require(value['total_rows'] == rows and value['total_source_groups'] == groups and
            0 <= value['valid_rows'] <= rows and 0 <= value['valid_source_groups'] <= groups,
            'Fixed head source population')
    require(sum(value['class_valid_rows']) == value['valid_rows'] and len(value['class_valid_rows']) == 2,
            'Binary class support')
    close(value['coverage'], value['valid_rows'] / rows, 'population_coverage')


def readouts(run):
    summary = load(run / 'readouts/reader-summary.json')
    require(summary['protocol'] == PROTOCOL and summary['model'] == 'RPB-v30.market-01' and
            summary['native_width'] == 36 and summary['designed_complexity_level'] == 5 and
            summary['primary_input_mode'] == MODES[0] and summary['secondary_input_mode'] == MODES[1], 'Frozen readout scope')
    for field in ('encoder_calls', 'encoder_updates', 'old_encoder_calls', 'old_head_fits', 'post_encoder_pca_fits'):
        require(summary[field] == 0, 'Readout performs no encoder/old fit/PCA: ' + field)
    aggregates = {mode: {view: {head: [] for head in HEADS} for view in VIEWS} for mode in MODES}
    full_coverage = True
    totals = {'outer_train_normalizer_fits': 0, 'ridge_fits': 0, 'tiny_fits': 0}
    require(len(summary['cohorts']) == 3 and {v['master_seed'] for v in summary['cohorts']} == set(MASTERS), 'All reader cohorts')
    for master in MASTERS:
        base = run / 'readouts' / ('master-' + str(master))
        report = load(base / 'report.json')
        require(report['protocol'] == 'fixed-feature-readouts-v1' and report['master_seed'] == str(master), 'Frozen head protocol/master')
        for name, expected in {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100, 'tiny_learning_rate': .01,
                               'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0,
                               'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods'}.items():
            require(report['recipe'][name] == expected, 'Fixed head recipe: ' + name)
        require(len(report['methods']) == 2 and {v['method'] for v in report['methods']} == set(MODES) and report['pairs'] == [], 'Both policy methods without new comparisons')
        for method in report['methods']:
            mode = method['method']
            require(method['size'] == 36 and method['status'] == 'measured' and
                    method['inputs_train_prepared'] is False and method['outer_train_normalizer_fits'] == 1,
                    'TRAIN-only outer normalizer and native width')
            require(len(method['repetitions']) == 3, 'All fixed head repetitions')
            for rep, repetition in zip(method['repetitions'], REPETITIONS):
                require(rep['id'] == 'rep-' + str(repetition) and rep['status'] == 'measured' and
                        rep['actual_probe_seed_decimal'] == str(stream_seed(repetition, 36)) and
                        rep['ridge_parameters'] == 74 and rep['neural_parameters'] == 626, 'D36 fixed seed/head parameter counts')
                for view in ('training',) + VIEWS:
                    rows, groups = (512, 256) if view == 'training' else (256, 128)
                    value = rep[view]
                    population(value['population'], rows, groups)
                    full_coverage &= value['population']['valid_rows'] == rows
                    for head in HEADS:
                        score = value[head]
                        require(score['total'] == rows and score['valid'] == value['population']['valid_rows'] and
                                isinstance(score['correct'], int) and 0 <= score['correct'] <= score['valid'], 'Correct/valid count arithmetic')
                        close(score['coverage'], score['valid'] / rows, 'head_coverage')
                        require(score['valid'] > 0, 'Nonempty measured head support')
                        close(score['accuracy'], score['correct'] / score['valid'], 'head_accuracy')
                        close(score['full_population_correctness'], score['correct'] / rows, 'full_population_correctness')
                        if view in VIEWS:
                            aggregates[mode][view][head].append(score['accuracy'])
                    if view in VIEWS:
                        require(value['bootstrap_seed_decimal'] == str(stream_seed(master, named(mode + '/' + rep['id'] + '/' + view.replace('_', '-')))), 'Original policy/mode grouped bootstrap seed')
                        for head, field in (('ridge', 'ridge_grouped_interval'), ('tiny_secondary', 'tiny_grouped_interval')):
                            interval = value[field]
                            require(interval['method'] == 'source-group percentile bootstrap; within-master conditional on fitted readout' and
                                    interval['replicates'] == 1000 and interval['confidence'] == .95 and interval['source_groups'] == groups,
                                    'Fixed conditional group interval recipe')
                            close(interval['estimate'], value[head]['accuracy'], 'interval_estimate')
                            require(0 <= interval['lower'] <= interval['upper'] <= 1, 'Finite ordered probability interval')
                require(rep['fit_artifact'] == mode + '/' + rep['id'] + '/fit.pt' and (base / rep['fit_artifact']).is_file(), 'Saved new fitted head artifact')
            counts = load(base / mode / 'fit-counts.json')
            for name, expected in {'outer_train_normalizer_fits': 1, 'ridge_fits': 3, 'tiny_fits': 3, 'validation_fits': 0}.items():
                require(counts[name] == expected, 'Saved policy fit count')
                if name in totals:
                    totals[name] += counts[name]
        for name, expected in {'outer_train_normalizer_fits': 2, 'ridge_fits': 6, 'tiny_fits': 6,
                               'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}.items():
            require(report['fit_counts'][name] == expected, 'Per-cohort aggregate fit count')
    require(totals == {'outer_train_normalizer_fits': 6, 'ridge_fits': 18, 'tiny_fits': 18}, 'Total fixed fit budget')
    gates = []
    for mode in MODES:
        for view in VIEWS:
            for head, floor in zip(HEADS, FLOORS[mode][view]):
                values = aggregates[mode][view][head]
                require(len(values) == 9, 'Three equal cohorts times three fits')
                cohort_means = [statistics.mean(values[index:index + 3]) for index in range(0, 9, 3)]
                mean = statistics.mean(cohort_means)
                gates.append({'mode': mode, 'view': view, 'head': head, 'cohort_means': cohort_means,
                              'mean': mean, 'floor': floor, 'arithmetic_tolerance': 1e-12,
                              'passed': mean + 1e-12 >= floor})
    return totals, full_coverage, gates


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Use existing cuwacunu container only')
    run = args.run.resolve(strict=True)
    require(run.is_relative_to(ROOT / 'output/runs/market-location-scale-synthetic-transfer'), 'Declared transfer run path')
    require(not args.output.exists(), 'Preserve existing audit output')
    pins(run)
    features = run / 'features'
    summary = load(features / 'summary.json')
    require(summary['protocol'] == PROTOCOL and summary['model'] == 'RPB-v30.market-01' and
            summary['checkpoint_sha256'] == CHECKPOINT and summary['settings_sha256'] == SETTINGS and
            summary['card_sha256'] == CARD and summary['native_width'] == 36 and summary['batch_size'] == 8 and
            summary['device'] == 'cuda:0' and summary['endpoint'] == 7620000 and
            summary['sampling_interval'] == 60000 and summary['resampling'] is False, 'Exact frozen extraction geometry and state pins')
    for field in ('encoder_updates', 'decoder_calls', 'head_calls', 'CPU_encoder_calls', 'scaler_refits'):
        require(summary[field] == 0, 'Zero forbidden extraction operation: ' + field)
    for field in ('all_parameters_and_buffers_bitwise_unchanged', 'all_parameter_gradients_undefined',
                  'masked_sentinel_input_invariance', 'source_hashes_unchanged', 'normalizer_unchanged',
                  'scalers_unchanged', 'same_preprocessing_parent_prefix_exact'):
        require(summary[field] is True, 'Frozen preprocessing/state invariant: ' + field)
    require(summary['labels_passed_to_encoder'] is False, 'Labels excluded from encoder')
    for field in ('bank_normalizer_fits', 'calibration_encoder_calls', 'calibration_encoder_updates', 'old_encoder_calls'):
        require(summary[field] == 0, 'No new calibration operations: ' + field)
    for field in ('full_API_inference_seconds', 'external_preprocessing_seconds', 'parent_correctness_seconds',
                  'feature_and_receipt_io_seconds', 'state_verification_seconds'):
        require(finite(summary[field]) and summary[field] >= 0, 'Separated finite nonnegative timing: ' + field)
    lookup = receipts(features, summary)
    state_checks, external_scalers = frozen_state(features, summary)
    cells = preprocessing(features, lookup, external_scalers)
    parents(features)
    saved_prefixes(features)
    require((features / 'sources-before.sha256').read_bytes() == (features / 'sources-after.sha256').read_bytes(), 'Extractor source and input hash ledgers unchanged')
    require(summary['preprocessing_receipt_count'] == 768 and summary['preprocessing_witness_case_count'] == 18 and
            summary['hidden_poison_preprocessing_checks'] == 18, 'Independent witness/receipt count summary')
    totals, coverage, gates = readouts(run)
    result = {'kind': 'market_location_scale_synthetic_transfer_independent_audit_v1',
              'protocol': PROTOCOL, 'status': 'passed', 'checks': CHECKS,
              'card_sha256': CARD, 'checkpoint_sha256': CHECKPOINT, 'staged_manifest_sha256': MANIFEST,
              'auditor': record(Path(__file__)), 'summary': record(features / 'summary.json'),
              'preprocessing_receipts': 768, 'bounded_preprocessing_cases': 18, 'bounded_cells_checked': cells,
              'parent_prefix_cases': 6, 'exact_parent_prefix_cells': 1536,
              'complete_state_checks': state_checks,
              'descriptive_historical_prefix_comparisons': 18,
              'full_cuda_encode_calls': 768, 'same_checkpoint_parent_correctness_calls': 6,
              'total_parent_forwards': 774, 'encoder_updates': 0, 'decoder_calls': 0,
              'new_calibration_fits': 0, 'auditor_encoder_calls': 0, 'auditor_head_fits': 0,
              'head_fits': totals, 'bank_normalizer_identity': NORMALIZER,
              'floating_point_rule': 'Raw and same-input prefix F32 exact; bounded normalized CPUF64 arithmetic at most two F32 ULP; normalized-input difference magnitudes subtract actual F32 values after promotion to F64; score/gate arithmetic absolute tolerance1e-12',
              'maximum_absolute_errors': MAX_ERRORS, 'full_coverage': coverage,
              'retention_conditions': gates, 'all_eight_retention_conditions_passed': all(gate['passed'] for gate in gates),
              'balanced_transfer_gate_passed': coverage and all(gate['passed'] for gate in gates),
              'limitations': 'No tensor deserialization, prediction replay or fitting. Counts/accuracy/means and bootstrap recipe/seeds are independently checked; full prediction correctness and bootstrap quantiles rely on the byte-identical frozen eight-file evaluator and saved artifact/provenance bindings.',
              'seconds': time.monotonic() - started}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('LOCATION_SCALE_TRANSFER_INDEPENDENT_AUDIT_PASSED checks=' + str(CHECKS), flush=True)


if __name__ == '__main__':
    main()

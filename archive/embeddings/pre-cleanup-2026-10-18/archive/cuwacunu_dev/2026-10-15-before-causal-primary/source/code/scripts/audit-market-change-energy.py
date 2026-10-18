#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Saved-only independent v25 arithmetic/stream/normalizer verification."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path('/cuwacunu')
CHECKS = 0
MAX_LOSS_ERROR = 0.0


def require(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counter(base, attempt):
    mask = (1 << 64) - 1
    x = (base + 0x9e3779b97f4a7c15 * (attempt + 1)) & mask
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
    return (x ^ (x >> 31)) & ((1 << 63) - 1)


def close_float32(actual, expected, label):
    global MAX_LOSS_ERROR
    difference = abs(actual - expected)
    MAX_LOSS_ERROR = max(MAX_LOSS_ERROR, difference)
    # Float32 arithmetic witness, independent of the 1e-12 quality-gate tolerance.
    require(math.isfinite(actual) and difference <= 4e-7 * (1 + abs(expected)), label)


def main():
    started = time.monotonic()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT,
            'Run inside managed cuwacunu container')
    require(len(sys.argv) == 2, 'Usage: audit-market-change-energy.py RUN_DIRECTORY')
    run = Path(sys.argv[1]).resolve()
    require(run.is_relative_to(ROOT / 'output/runs/market-change-energy') and
            run.name.startswith('run-'), 'Wrong candidate run path')
    output = run / 'independent-audit.json'
    require(not output.exists(), 'Preserve existing audit')
    source = [run / name for name in ('summary.json', 'training.csv',
              'moment-rms-scaler.json', 'initial-query.json',
              'initialization-parity.json', 'final-prefix-parity.json',
              'moment-rms-scaler.pt')]
    before = {str(p.relative_to(ROOT)): sha(p) for p in source}
    summary = json.loads(source[0].read_text())
    rows = list(csv.DictReader(source[1].open()))
    normalizer = json.loads(source[2].read_text())
    require(summary['instance'] == 'RPB-v25.market-01' and
            summary['completed_updates'] == 2048 and summary['parent_updates'] == 2048,
            'Exact instance and separate budgets')
    require(summary['D'] == 36 and summary['parameters'] == 298505 and
            summary['optimizer_updated_parameters'] == 15984 and
            summary['protected_parameters'] == 282521, 'Parameter/export contract')
    require(len(rows) == summary['attempted_batches'] <= 4096, 'Trace attempts')
    total_update = 0
    sums = {key: 0 for key in ('target_cells', 'eligible_examples',
            'aux_eligible_examples', 'signed_cells', 'energy_cells',
            'projection_gradient_nonzero', 'wave_decoder_gradient_nonzero',
            'moment_decoder_gradient_nonzero', 'projection_weight_changed',
            'wave_decoder_weight_changed', 'moment_decoder_weight_changed')}
    for attempt, row in enumerate(rows):
        require(int(row['attempt']) == attempt, 'Ordered attempt counter')
        require(int(row['row_seed']) == counter(942002, attempt) and
                int(row['mask_seed']) == counter(942003, attempt), 'Counter streams')
        require(len(row['anchors'].split(';')) == 8, 'B8 presentations')
        if not row['loss']:
            require(int(row['update']) == total_update, 'Skipped update unchanged')
            continue
        total_update += 1
        require(int(row['update']) == total_update, 'Ordered update counter')
        loss, wave, aux, signed, energy = [float(row[k]) for k in
                                         ('loss', 'wave_loss', 'aux_loss',
                                          'signed_loss', 'energy_loss')]
        require(all(v >= 0 and math.isfinite(v) for v in
                    (loss, wave, aux, signed, energy)), 'Finite nonnegative components')
        close_float32(aux, .5 * signed + .5 * energy, 'Balanced component loss')
        close_float32(loss, wave + .1 * aux, 'Total loss coefficient')
        require(0 < int(row['eligible_examples']) <= 8 and
                0 <= int(row['aux_eligible_examples']) <= 8, 'Eligibility population')
        require(0 < int(row['target_cells']) <= 8 * 3 * 128 * 9 and
                0 <= int(row['signed_cells']) <= 8 * 135 and
                0 <= int(row['energy_cells']) <= 8 * 135, 'Target cell bounds')
        for key in ('projection_gradient_nonzero', 'wave_decoder_gradient_nonzero',
                    'moment_decoder_gradient_nonzero', 'projection_weight_changed',
                    'wave_decoder_weight_changed', 'moment_decoder_weight_changed'):
            require(row[key] in ('0', '1'), 'Boolean trace flag: ' + key)
        for key in sums:
            sums[key] += int(row[key])
        if total_update <= 2:
            require(int(row['projection_gradient_nonzero']) == total_update - 1,
                    'Zero-decoder first step / projection second step')
            for key in ('wave_decoder_gradient_nonzero', 'moment_decoder_gradient_nonzero',
                        'wave_decoder_weight_changed', 'moment_decoder_weight_changed'):
                require(row[key] == '1', 'Both decoders learn from first update')
            if total_update == 2:
                require(row['projection_weight_changed'] == '1', 'Projection learning')
    require(total_update == 2048, 'Exactly 2048 new updates')
    active_rows = [row for row in rows if row['loss']]
    require(float(active_rows[0]['loss']) == summary['first_loss'], 'First loss summary')
    for key in ('loss', 'wave_loss', 'aux_loss', 'signed_loss', 'energy_loss'):
        require(float(active_rows[-1][key]) == summary['last_' + key],
                'Final component summary: ' + key)
    mapping = {'target_cells': 'target_cells', 'eligible_examples': 'wave_eligible_exposures',
               'aux_eligible_examples': 'aux_eligible_exposures',
               'signed_cells': 'signed_target_cells', 'energy_cells': 'energy_target_cells',
               'projection_gradient_nonzero': 'nonzero_change_gradient_updates',
               'wave_decoder_gradient_nonzero': 'nonzero_decoder_gradient_updates',
               'moment_decoder_gradient_nonzero': 'nonzero_moment_gradient_updates',
               'projection_weight_changed': 'projection_weight_updates',
               'wave_decoder_weight_changed': 'decoder_weight_updates',
               'moment_decoder_weight_changed': 'moment_weight_updates'}
    for key, name in mapping.items():
        require(sums[key] == summary[name], 'Trace/summary aggregate: ' + key)
    require(normalizer['TRAIN_anchors'] == 4096 and
            normalizer['validation_anchors'] == normalizer['encoder_calls'] == 0,
            'TRAIN-only bank normalization')
    require(all(len(normalizer[key]) == 270 for key in
                ('counts', 'sumsq', 'scales', 'support')), 'Moment width')
    for count, square, scale, supported in zip(normalizer['counts'], normalizer['sumsq'],
                                               normalizer['scales'], normalizer['support']):
        require(0 <= count <= 4096 and math.isfinite(square) and square >= 0,
                'RMS statistics domain')
        require(bool(supported) == (count > 0), 'RMS support')
        require(supported in (0, 1) and (count != 0 or square == 0), 'RMS empty support')
        expected = max(math.sqrt(square / count), 1e-6) if count else 1.0
        require(abs(scale - expected) <= 2e-14 * (1 + expected), 'RMS arithmetic')
    require(sha(run / 'moment-rms-scaler.pt') == normalizer['archive_sha256'] ==
            summary['moment_normalizer_sha256'], 'Immutable RMS archive hash')
    for name in ('initialization-parity.json', 'final-prefix-parity.json'):
        value = json.loads((run / name).read_text())
        require(value['saved_v22_train_and_validation_prefix32_equal'] and
                value['old_encoder_replay_calls'] == value['old_head_refits'] == 0,
                'Saved exact prefix parity')
        if name == 'initialization-parity.json':
            require(value['saved_v24_initial_train_and_validation_all36_equal'] is True,
                    'Exact saved native36 initialization')
    for key in ('cuda_learning_verified', 'initial_prefix_matches_saved_trained_v22',
                'initial_all36_match_saved_v24_init', 'final_prefix_matches_saved_trained_v22',
                'protected_level_exact'):
        require(summary[key] is True, key)
    require(summary['CPU_encoder_calls'] == summary['protected_optimizer_states'] == 0,
            'GPU-only new encoder and protected optimizer')
    fixed = Path('/opt/cuwacunu/build/market-study/market_fixed_regression')
    require(sha(fixed) == '5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14',
            'Unchanged fixed head executable')
    require(sha(ROOT / 'doc/MARKET_CHANGE_ENERGY_CARD.md') ==
            summary['card_sha256'] ==
            'c67e5c37d56e5bc2801a9d32152589e3067ae11f8b18c8c804418b8cca807938',
            'Prospective card immutable')
    require(before == {str(p.relative_to(ROOT)): sha(p) for p in source},
            'Saved audit inputs unchanged')
    result = {'kind': 'market_change_energy_independent_arithmetic_audit_v1',
              'status': 'passed', 'checks': CHECKS,
              'maximum_absolute_loss_component_difference': MAX_LOSS_ERROR,
              'loss_arithmetic_tolerance': '4e-7*(1+abs(expected)); float32 only',
              'quality_gate_tolerance': 1e-12,
              'input_sha256': before, 'new_encoder_calls': 0,
              'new_encoder_updates': 0, 'head_or_scaler_fits': 0,
              'elapsed_seconds': time.monotonic() - started}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

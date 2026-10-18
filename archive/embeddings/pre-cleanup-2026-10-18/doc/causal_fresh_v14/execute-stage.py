# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
"""Record one exclusive container command; never retry or overwrite its output."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--name', required=True)
p.add_argument('--gradient-proof', action='store_true')
p.add_argument('--temporal-proof', action='store_true')
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
if not Path('/.dockerenv').is_file():
    raise RuntimeError('Existing project container required')
if a.gradient_proof and a.temporal_proof:
    raise RuntimeError('Separate objective and temporal fixture invocations required')
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command or not a.name.replace('-', '').replace('_', '').isalnum():
    raise RuntimeError('Explicit command and simple stage name required')
log = a.root / (a.name + '.log')
receipt = a.root / (a.name + '-stage.json')
if log.exists() or receipt.exists():
    raise RuntimeError('Preserve the existing stage; refuse replay')
if ((a.gradient_proof and (a.root / 'variance-gradient-proof.json').exists()) or
        (a.temporal_proof and (a.root / 'temporal-pooling-proof.json').exists())):
    raise RuntimeError('Preserve the existing mechanism proof; refuse a repeated fixture')


def pin(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


start = time.monotonic()
with log.open('x') as out:
    process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
result = {
    'kind': 'causal_fresh_v14_exclusive_stage_execution_v1',
    'status': 'passed' if process.returncode == 0 else 'failed',
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'argv': command, 'exit_code': process.returncode,
    'elapsed_seconds': time.monotonic() - start,
    'binary': pin(command[0]), 'log': pin(log),
}
with receipt.open('x') as out:
    json.dump(result, out, indent=2)
    out.write('\n')
print(json.dumps(result), flush=True)
if process.returncode:
    raise SystemExit(process.returncode)
if a.gradient_proof:
    observed = log.read_text()
    if (Path(command[0]).name != 'test_cpb_v14_variance_gradient'
            or 'PASS CPB-v14 huber1-mean-plus-detached-error-gaussian-variance-v1:' not in observed
            or 'manual composite-objective native40 gradient parity' not in observed
            or 'detached-error NLL mean gradient zero' not in observed
            or 'synthetic optimizer steps=1' not in observed):
        raise RuntimeError('Actual dedicated CUDA test PASS output required')
    proof = {
        'status': 'passed', 'model': 'CPB-v14',
        'gradient_policy': 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1', 'actual_CUDA': True,
        'objective_policy': 'huber1-mean-plus-detached-error-gaussian-variance-v1',
        'mean_huber_delta': 1, 'loss_coefficients': [1, 1],
        'direct_gradient_routes': 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1',
        'variance_error_detached': True, 'reporting_nll_backpropagated': False,
        'direct_NLL_mean_gradient_zero': True, 'huber_mean_gradient_live': True,
        'direct_Huber_uncertainty_gradient_zero': True,
        'shared_clip_coupling_witness': True,
        'mean_native_variance_gradient_blocked': True, 'uncertainty_native_variance_gradient_live': True, 'uncertainty_projection_learning': True, 'mean_path_live': True,
        'variance_head_learning': True, 'forward_variance_input_dependence': True,
        'manual_composite_gradient_parity': True, 'nonvacuous_nonzero_fixture': True,
        'binary': result['binary'], 'log': result['log'],
        'execution': pin(receipt),
    }
    with (a.root / 'variance-gradient-proof.json').open('x') as out:
        json.dump(proof, out, indent=2)
        out.write('\n')

if a.temporal_proof:
    # Contract originates in independently reviewed actual fixture source and is
    # frozen by the external input plan/card before this binary can execute.
    source = a.root / 'source'
    capsule = json.loads((source / 'CAPSULE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    card_path = source / 'doc/CAUSAL_FRESH_V14_CARD.json'
    if pin(card_path)['sha256'] != capsule['card_sha256']:
        raise RuntimeError('Exact frozen card before mechanism proof construction')
    card = json.loads(card_path.read_text(encoding='utf-8-sig'))
    if card['stage'].startswith('DRAFT') or card['execution_admitted'] is not True:
        raise RuntimeError('Separately frozen/admitted source-derived contract required')
    contract = card['temporal_pooling_proof_contract']
    expected = {row['path']: row for row in capsule['files']}
    fixture = contract['fixture_source']
    scientific_review = contract['source_review']
    if scientific_review != card['independent_training_auditor_source_review']:
        raise RuntimeError('Independent scientific review must bind actual fixture contract')
    for binding in (fixture, scientific_review):
        actual = pin(source / binding['path'])
        if binding['path'] not in expected or any(actual[key] != binding[key]
                or expected[binding['path']][key] != binding[key] for key in ('bytes','sha256')):
            raise RuntimeError('Actual reviewed fixture/review bytes required')
    observed = log.read_text(encoding='utf-8')
    tokens = [contract['PASS_prefix']] + contract['log_tokens']
    if Path(command[0]).name != contract['binary'] or not tokens or not all(
            isinstance(token,str) and token and token in observed for token in tokens):
        raise RuntimeError('Dedicated actual CUDA fixture PASS/tokens required')
    literal_flags = ('mean4_regularized_stddev4_export_layout','full_attention_dependent_regularized_moment_adjoints','live_dispersion_score_gradient','raw_variance_root_dispersion_trace','zero_tiny_large_analytic_dispersion_adjoints','supported_nonfinite_and_variance_overflow_refused','absent_local_huge_singleton_sanitized','captured_F32_residual_correction','raw_root_absent_offset_exception','nested_trailing_supports_fixed_edge16','per_window_local_weights_captured','longest_local_equals_global_two_sum','packed_padding_membership_invariance',)
    tokens += [key + '=true' for key in literal_flags]
    if not all(token in observed for token in tokens):
        raise RuntimeError('Actual retained regularized-moment fixture flags and all new declared V14 tokens required')
    proof = dict(contract['required_proof_fields'])
    if (proof['model'] != 'CPB-v14' or proof['actual_CUDA'] is not True
            or proof['temporal_dispersion_transform_policy'] !=
            'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'
            or proof['temporal_dispersion_regularizer'] != 0.0009765625
            or any('uniform' in key or 'dual_support' in key for key in proof)):
        raise RuntimeError('Only reviewed V14 nested-support attention-coupled transform proof permitted')
    if (not all(proof.get(key) is True for key in literal_flags)
            or proof.get('temporal_pooling_geometry') != [4,16,32,8]
            or proof.get('temporal_pooling_trailing_lengths') != [4,8,12,16]
            or 'temporal_pooling_partition' in proof
            or proof['transform_derivative_upper_bound'] != 16):
        raise RuntimeError('Exact new geometry/trailing lengths and retained transform bound required')
    for key in ('direct_pool_kernel_calls','F64_finite_difference_coordinates',
                'F32_finite_difference_coordinates','isolated_autograd_grad_calls'):
        value = proof.get(key)
        if type(value) is not int or value <= 0 or f'{key}={value}' not in observed:
            raise RuntimeError('Actual independently source-derived count must match emitted stdout: '+key)
    expected_numerics = {
        'backbone_encode_calls': 1, 'public_encoder_calls': 0,
        'head_fits': 0, 'synthetic_optimizer_steps': 1,
        'manual_F32_absolute_tolerance': 3e-5, 'manual_F32_relative_tolerance': 3e-5,
        'F64_finite_difference_step': 1e-5,
        'F64_finite_difference_absolute_tolerance': 2e-8,
        'F64_finite_difference_relative_tolerance': 2e-8,
        'F32_finite_difference_step': 2e-3,
        'F32_finite_difference_absolute_tolerance': 2e-3,
        'F32_finite_difference_relative_tolerance': 2e-3,
        'softmax_gauge_tolerance': 2e-6,
    }
    if any(proof.get(key) != value for key,value in expected_numerics.items()):
        raise RuntimeError('Actual source-derived fixture tolerance/count mismatch')
    proof.update(kind='cpb_v14_temporal_pooling_mechanism_proof_v1',status='passed',
                 binary=result['binary'],log=result['log'],execution=pin(receipt),
                 frozen_card=pin(card_path),fixture_source=fixture,
                 independent_scientific_source_review=scientific_review)
    with (a.root / 'temporal-pooling-proof.json').open('x') as out:
        json.dump(proof, out, indent=2)
        out.write('\n')

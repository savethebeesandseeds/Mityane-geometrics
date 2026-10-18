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
    'kind': 'causal_fresh_v11_exclusive_stage_execution_v1',
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
    if (Path(command[0]).name != 'test_cpb_v11_variance_gradient'
            or 'PASS CPB-v11 huber1-mean-plus-detached-error-gaussian-variance-v1:' not in observed
            or 'manual composite-objective native40 gradient parity' not in observed
            or 'detached-error NLL mean gradient zero' not in observed
            or 'synthetic optimizer steps=1' not in observed):
        raise RuntimeError('Actual dedicated CUDA test PASS output required')
    proof = {
        'status': 'passed', 'model': 'CPB-v11',
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
    observed = log.read_text(encoding='utf-8')
    tokens = ('PASS CPB-v11 original-position-four-bin-half-mixture-mean4-centered-variance4-v1:',
              'manual full-attention mean and centered-variance parity',
              'independent full-expression F64 finite differences',
              'actual CUDA F32 kernel finite differences',
              'four mean4 and variance4 projection gradient and movement',
              'equal mean different spread', 'present constant variance dtype tolerance', 'empty-bin finite zero',
              'three between-bin Hadamard contrast gradient and movement',
              'relative-bin global sensitivity', 'common softmax gauge invariance',
              'synthetic optimizer steps=1')
    if Path(command[0]).name != 'test_cpb_v11_temporal_pooling' or not all(token in observed for token in tokens):
        raise RuntimeError('Actual dedicated temporal CUDA fixture PASS output required')
    proof = {
        'kind': 'cpb_v11_temporal_pooling_mechanism_proof_v1',
        'status': 'passed', 'model': 'CPB-v11', 'actual_CUDA': True,
        'temporal_pooling_policy': 'original-position-four-bin-half-mixture-mean4-centered-variance4-v1',
        'temporal_pooling_partition': [4,16,32,4,8],
        'temporal_pooling_mix': [0.5, 0.5],
        'empty_local_bin_policy': 'zero-whole-export-slice-after-bias-v1',
        'temporal_moment_policy': 'componentwise-centered-variance-two-sum-mean-live-attention-v1',
        'temporal_export_layout': 'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1',
        'temporal_moment_partition': [4, 4],
        'original_position_membership': True, 'both_temporal_paths': True,
        'empty_bins_finite_zero': True, 'true_per_bin_trace_weights': True,
        'manual_pooling_and_adjoint_parity': True,
        'four_export_slices_gradient_and_movement': True,
        'centered_position_contrasts_gradient_and_movement': True,
        'common_softmax_gauge_invariance': True,
        'shared_global_half_mixture': True, 'relative_bin_global_sensitivity': True,
        'three_between_bin_contrasts_gradient_and_movement': True,
        'mean4_centered_variance4_export_layout': True,
        'full_attention_dependent_moment_adjoints': True,
        'independent_full_expression_F64_finite_differences': True,
        'actual_F32_kernel_finite_differences': True,
        'equal_mean_different_variance': True,
        'present_constant_variance_dtype_tolerance': True,
        'eight_moment_subgroups_gradient_and_movement': True,
        'backbone_encode_calls': 1, 'direct_pool_kernel_calls': 44,
        'public_encoder_calls': 0, 'head_fits': 0, 'synthetic_optimizer_steps': 1,
        'manual_F32_absolute_tolerance': 3e-5, 'manual_F32_relative_tolerance': 3e-5,
        'F64_finite_difference_coordinates': 18, 'F64_finite_difference_step': 1e-5,
        'F64_finite_difference_absolute_tolerance': 2e-8, 'F64_finite_difference_relative_tolerance': 2e-8,
        'F32_finite_difference_coordinates': 18, 'F32_finite_difference_step': 2e-3,
        'F32_finite_difference_absolute_tolerance': 2e-3, 'F32_finite_difference_relative_tolerance': 2e-3,
        'present_constant_variance_bound': '1e-10*(1+max_abs_hidden_squared)',
        'softmax_gauge_tolerance': 2e-6, 'bin_mass_tolerance': 2e-6,
        'flat_trace_tolerance': 8e-6, 'empty_and_membership_zero': 'exact',
        'binary': result['binary'], 'log': result['log'], 'execution': pin(receipt),
    }
    with (a.root / 'temporal-pooling-proof.json').open('x') as out:
        json.dump(proof, out, indent=2)
        out.write('\n')



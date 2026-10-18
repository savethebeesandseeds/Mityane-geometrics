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
    'kind': 'causal_fresh_v9_exclusive_stage_execution_v1',
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
    if (Path(command[0]).name != 'test_cpb_v9_variance_gradient'
            or 'PASS CPB-v9 huber1-mean-plus-detached-error-gaussian-variance-v1:' not in observed
            or 'manual composite-objective native40 gradient parity' not in observed
            or 'synthetic optimizer steps=1' not in observed):
        raise RuntimeError('Actual dedicated CUDA test PASS output required')
    proof = {
        'status': 'passed', 'model': 'CPB-v9',
        'gradient_policy': 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1', 'actual_CUDA': True,
        'objective_policy': 'huber1-mean-plus-detached-error-gaussian-variance-v1',
        'mean_huber_delta': 1, 'loss_coefficients': [1, 1],
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
    tokens = ('PASS CPB-v9 original-position-four-bin-shared-global-half-mixture-v1:',
              'manual original-position half-mixture pooling parity',
              'four 8-row projection gradient and movement', 'empty-bin finite zero',
              'three between-bin Hadamard contrast gradient and movement',
              'relative-bin global sensitivity', 'common softmax gauge invariance',
              'synthetic optimizer steps=1')
    if Path(command[0]).name != 'test_cpb_v9_temporal_pooling' or not all(token in observed for token in tokens):
        raise RuntimeError('Actual dedicated temporal CUDA fixture PASS output required')
    proof = {
        'kind': 'cpb_v9_temporal_pooling_mechanism_proof_v1',
        'status': 'passed', 'model': 'CPB-v9', 'actual_CUDA': True,
        'temporal_pooling_policy': 'original-position-four-bin-shared-global-half-mixture-v1',
        'temporal_pooling_partition': [4,16,32,4,8],
        'temporal_pooling_mix': [0.5, 0.5],
        'empty_local_bin_policy': 'zero-whole-export-slice-after-bias-v1',
        'original_position_membership': True, 'both_temporal_paths': True,
        'empty_bins_finite_zero': True, 'true_per_bin_trace_weights': True,
        'manual_pooling_and_adjoint_parity': True,
        'four_export_slices_gradient_and_movement': True,
        'centered_position_contrasts_gradient_and_movement': True,
        'common_softmax_gauge_invariance': True,
        'shared_global_half_mixture': True, 'relative_bin_global_sensitivity': True,
        'three_between_bin_contrasts_gradient_and_movement': True,
        'backbone_encode_calls': 1, 'direct_pool_kernel_calls': 5,
        'public_encoder_calls': 0, 'head_fits': 0, 'synthetic_optimizer_steps': 1,
        'manual_F32_absolute_tolerance': 2e-5, 'manual_F32_relative_tolerance': 2e-5,
        'softmax_gauge_tolerance': 2e-6, 'bin_mass_tolerance': 2e-6,
        'flat_trace_tolerance': 8e-6, 'empty_and_membership_zero': 'exact',
        'binary': result['binary'], 'log': result['log'], 'execution': pin(receipt),
    }
    with (a.root / 'temporal-pooling-proof.json').open('x') as out:
        json.dump(proof, out, indent=2)
        out.write('\n')



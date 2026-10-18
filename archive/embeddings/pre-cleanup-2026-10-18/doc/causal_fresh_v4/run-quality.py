# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
"""Owned v3-derived one-pass v4 quality driver; require dedicated gradient proof first."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
for name in ('build', 'source', 'training', 'engineering-proof', 'initial-proof', 'final-proof',
             'admission', 'gradient-proof', 'data-root', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def pin(path):
    content = path.read_bytes()
    return {'path': str(path), 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}


admission = read(args.admission)
require(admission['status'] == 'passed'
        and admission['card_sha256'] == '114026b2e6d11d84aee1855a8af673d84e782ab057eb45d398a5d66fa94b1dce'
        and admission['gradient_policy'] == 'variance_input_gradient_stop-v1'
        and len(admission['binaries']) == 6, 'Six-binary frozen v4 build admission required')
for binary in admission['binaries']:
    require(pin(Path(binary['path'])) == binary, 'Admitted binary changed')
gradient = read(args.gradient_proof)
expected_gradient = {
    'status': 'passed', 'model': 'CPB-v4',
    'gradient_policy': 'variance_input_gradient_stop-v1', 'actual_CUDA': True,
    'native_variance_gradient_blocked': True, 'mean_path_live': True,
    'variance_head_learning': True, 'forward_variance_input_dependence': True,
    'manual_partial_gradient_parity': True, 'nonvacuous_nonzero_fixture': True,
}
for key, expected in expected_gradient.items():
    require(gradient.get(key) == expected, 'Dedicated CUDA mechanism proof: ' + key)
gradient_binary = next((item for item in admission['binaries']
                        if Path(item['path']).name == 'test_cpb_v4_variance_gradient'), None)
require(gradient_binary is not None and gradient['binary'] == gradient_binary
        and pin(Path(gradient['binary']['path'])) == gradient['binary'],
        'Mechanism proof must bind admitted dedicated binary')
require(pin(Path(gradient['log']['path'])) == gradient['log'],
        'Observed dedicated mechanism stdout/log must remain exact')
require('PASS CPB-v4 variance_input_gradient_stop-v1:' in
        Path(gradient['log']['path']).read_text(encoding='utf-8'),
        'Observed mechanism output must contain the exact dedicated PASS prefix')
engineering = read(args.engineering_proof)
require(engineering['status'] == 'passed' and engineering['source_updates'] == 0
        and engineering['gradient_policy'] == 'variance_input_gradient_stop-v1'
        and engineering['old_checkpoints_opened'] == 0
        and engineering['all_initial_tensors_equal_seeded_constructor']
        and engineering['first_typed_artifact_exact']
        and engineering['typed_resume_next_loss_model_moments_bitwise_exact']
        and engineering['typed_negative_cases'] >= 30,
        'Fresh CUDA engineering, causal matrices and typed refusal proof required')
training = read(args.training / 'training-summary.json')
require(training['status'] == 'passed' and training['new_causal_updates'] == 2048
        and training['source_updates'] == 0
        and training['gradient_policy'] == 'variance_input_gradient_stop-v1',
        'One completed fresh 2048-update trajectory required')
require(read(args.training / 'typed-resume-512.json')['status'] == 'passed',
        'Actual trajectory typed resume required')
fresh_path = args.training / 'fresh-initialization.json'
fresh = read(fresh_path)
require(fresh['status'] == 'passed' and fresh['source_updates'] == 0
        and fresh['causal_updates'] == 0
        and fresh['inherited_learned_parameters'] == 0
        and fresh['all_initial_parameters_equal_direct_seeded_constructor'],
        'Fresh direct initialization, not inherited learned weights, required')
initial_path = args.training / 'checkpoints/inference-0.pt'
final_path = args.training / 'checkpoints/inference-2048.pt'
for path, proof_path, updates in ((initial_path, args.initial_proof, 0),
                                  (final_path, args.final_proof, 2048)):
    proof = read(proof_path)
    require(proof['status'] == 'passed' and proof['source_updates'] == 0
            and proof['gradient_policy'] == 'variance_input_gradient_stop-v1'
            and proof['causal_updates'] == updates
            and proof['new_training_updates'] == 0
            and proof['future_values_and_masks_invariant']
            and proof['checkpoint_sha256'] == pin(path)['sha256'],
            'Saved initial/final checkpoint causal proof must bind scored bytes')

# Full proof-before-quality admission duplicates the reporter's trust predicates.
# No extraction, initial/final head fit or output leaf exists before these pass.
def fields(actual, expected, message):
    require(all(actual.get(key) == value for key, value in expected.items()), message)


def verified_artifact(binding, origin):
    path = Path(binding['path'])
    path = (path if path.is_absolute() else origin / path).resolve(strict=True)
    actual = pin(path)
    require(actual['bytes'] == binding['bytes'] and actual['sha256'] == binding['sha256'],
            'Exact saved artifact required: ' + str(path))
    return path


fields(engineering, {
    'kind': 'cpb_v4_fresh_cuda_engineering_v1', 'status': 'passed',
    'gradient_policy': 'variance_input_gradient_stop-v1', 'source_updates': 0,
    'initialization_seed': 943001, 'pretrained_ancestor': 'none',
    'old_checkpoints_opened': 0, 'all_initial_tensors_equal_seeded_constructor': True,
    'first_typed_artifact_exact': True, 'scaler_and_bank_fitted_fixture_TRAIN_only': True,
    'all_cutoffs_per_stage': 128, 'suffix_cases_per_stage': 2350,
    'internal_temporal_cases_per_stage': 30, 'causal_updates_primary_fixture': 2,
    'resume_witness_updates': 1, 'quality_updates': 0,
    'typed_resume_next_loss_model_moments_bitwise_exact': True,
    'CPU_encoder_calls': 0, 'head_fits': 0,
}, 'Complete fresh CUDA/causality/typed refusal-resume engineering proof required')
require(engineering['stages'] == ['initialized', 'trained']
        and engineering['typed_negative_cases'] >= 30
        and engineering['CUDA_encoder_calls'] > 0 and engineering['checks'] > 0
        and engineering['actual_cuda_learning']
        and all(value['nonzero_gradient_parameter_events'] > 0
                and value['weight_movement_parameter_events'] > 0
                for value in engineering['actual_cuda_learning'].values()),
        'Every engineering stage and learned category must have actual CUDA evidence')

for checkpoint, proof_path, updates in ((initial_path, args.initial_proof, 0),
                                        (final_path, args.final_proof, 2048)):
    proof = read(proof_path)
    fields(proof, {
        'kind': 'cpb_v4_saved_checkpoint_causal_proof_v1', 'status': 'passed',
        'model': 'CPB-v4', 'gradient_policy': 'variance_input_gradient_stop-v1',
        'source_updates': 0, 'causal_updates': updates, 'initialization_seed': 943001,
        'pretrained_ancestor': 'none', 'new_training_updates': 0,
        'all_cutoffs': 128, 'suffix_cases': 2350, 'internal_temporal_cases': 30,
        'state_unchanged': True, 'future_values_and_masks_invariant': True,
        'CPU_encoder_calls': 0, 'head_fits': 0,
    }, 'Full actual saved-checkpoint causal proof required')
    require(proof['CUDA_encoder_calls'] > 0 and proof['checks'] > 0
            and Path(proof['checkpoint']).resolve(strict=True) == checkpoint.resolve(strict=True)
            and proof['checkpoint_sha256'] == pin(checkpoint)['sha256'],
            'Saved proof must bind exactly the scored checkpoint bytes')
    witness = proof_path.parent / (('initial0' if updates == 0 else 'final2048') + '-causality.json')
    fields(read(witness), {
        'status': 'passed', 'all_cutoffs': 128, 'suffix_cases': 2350,
        'internal_temporal_cases': 30, 'state_unchanged': True,
    }, 'Complete saved cutoff/suffix/internal witness required')

fields(fresh, {
    'kind': 'cpb_v4_fresh_initialization_v1', 'status': 'passed',
    'initialization_kind': 'fresh-seeded-constructor-no-import-v1',
    'initialization_seed': 943001, 'pretrained_ancestor': 'none',
    'inherited_learned_parameters': 0, 'fresh_learned_parameter_values': 293672,
    'analytic_fixed_prior_values': 3888, 'source_updates': 0, 'causal_updates': 0,
    'old_model_artifacts_opened': 0, 'old_normalization_artifacts_opened': 0,
    'all_initial_parameters_equal_direct_seeded_constructor': True,
    'parameter_tensors_checked': 76, 'parameter_values_checked': 297560,
}, 'Full direct seeded-constructor fresh ancestry required')
require(fresh['initial_state_identity'] == fresh['direct_constructor_state_identity']
        and verified_artifact(fresh['initial_checkpoint'], args.training)
        == initial_path.resolve(strict=True), 'Actual initialized checkpoint must bind fresh constructor')
calibration = read(args.training / 'calibration/summary.json')
fields(calibration, {
    'kind': 'cpb_v4_fresh_TRAIN_calibration_v1', 'status': 'passed',
    'TRAIN_anchors': 4096, 'history': 128, 'input_scaler_fits': 1,
    'bank_normalizer_fits': 1, 'validation_rows_used': 0, 'test_rows_used': 0,
    'encoder_calls': 0, 'old_normalization_artifacts_opened': 0,
}, 'Fresh permitted TRAIN-only market statistics required')
for artifact in calibration['artifacts']:
    verified_artifact({'path': artifact['file'], 'bytes': artifact['bytes'],
                       'sha256': artifact['sha256']}, args.training / 'calibration')
require(calibration['input_scaler_identity'] == fresh['fresh_input_scaler_identity']
        and calibration['bank_identity'] == fresh['fresh_bank_identity'],
        'Actual fresh calibration must belong to initialized model')

fields(training, {
    'kind': 'causal_fresh_v4_training_summary_v1', 'status': 'passed',
    'model': 'CPB-v4.market-01', 'source_updates': 0, 'new_causal_updates': 2048,
    'gradient_policy': 'variance_input_gradient_stop-v1',
    'encoder_gradient_kind': 'partial-NLL-gradient;live-mean-residual;variance-input-stopped',
    'conditional_variance_head_trainable': True, 'batch_size': 8,
    'TRAIN_windows': 4096, 'device': 'cuda:0', 'typed_resume_512_exact': True,
    'fixed_prior_scaler_normalizer_unchanged': True,
    'new_scaler_fits': 1, 'new_normalizer_fits': 1, 'saved_quality_features_reused': False,
}, 'Complete actual fresh partial-gradient trajectory required')
require(all(training[key] == 0 for key in (
    'old_model_artifacts_opened', 'old_normalization_artifacts_opened',
    'calibration_encoder_calls', 'calibration_validation_rows', 'CPU_encoder_calls',
    'old_encoder_calls', 'head_fits', 'training_replays'))
    and training['parameter_groups']
    and all(group['nonzero_gradient_tensors'] > 0 and group['moved_tensors'] > 0
            for group in training['parameter_groups']),
    'No replay/old-state reuse and actual CUDA movement of every learned group')
fields(read(args.training / 'typed-resume-512.json'), {
    'status': 'passed', 'completed_updates': 512,
    'same_parameters_buffers_moments_steps_order': True,
    'same_attempt_counter_and_three_seeds': True,
    'continued_actual_trajectory_from_saved_state': True, 'repeated_training_updates': 0,
}, 'Actual trajectory must resume exactly from typed update512')

args.output.mkdir(parents=False, exist_ok=False)
stages = {}


def run(name, command):
    started = time.monotonic()
    with (args.output / (name + '.stdout')).open('x') as stdout, (
            args.output / (name + '.stderr')).open('x') as stderr:
        process = subprocess.run(list(map(str, command)), cwd=args.source,
                                 stdout=stdout, stderr=stderr)
    stages[name] = time.monotonic() - started
    with (args.output / (name + '-stage.json')).open('x') as stream:
        json.dump({'stage': name, 'argv': list(map(str, command)),
                   'exit_code': process.returncode,
                   'elapsed_seconds': stages[name]}, stream, indent=2)
        stream.write('\n')
    require(process.returncode == 0,
            f'{name} failed; preserve partial output and do not repeat fits')


settings = args.source / 'code/encoders/causal_patch_bottleneck_mae/versions/v4/config/v4.conf'
card = args.source / 'doc/CAUSAL_FRESH_V4_CARD.json'
manifest = args.source / 'doc/causal_fresh_v4/input-manifest.json'
require(pin(card)['sha256'] == admission['card_sha256'], 'Frozen card changed')
require(pin(manifest)['sha256'] == read(card)['input_manifest']['sha256']
        == 'a6bff90ffc9dd917c6a9c446659c52e0c76b5fb34a9f67201788eb3168948eb9',
        'Frozen input manifest changed')
features = args.output / 'features'
run('extraction', [args.build / 'causal_fresh_v4_extract',
    '--initial-checkpoint', initial_path, '--final-checkpoint', final_path,
    '--settings', settings, '--output', features, '--data-root', args.data_root])
shared = args.data_root / 'output/runs/market-capacity-study/shared-MKT-HIST-2-v1'
for stage in ('initial', 'final'):
    (args.output / stage).mkdir(exist_ok=False)
    run(stage + '_market_heads', [args.build / 'causal_fresh_v4_market_read',
        '--train-targets', shared / 'features/train-targets.pt',
        '--validation-targets', shared / 'features/validation-targets.pt',
        '--method', 'cpb_v4', features / stage / 'market/cpb_v4-train.pt',
        features / stage / 'market/cpb_v4-validation.pt',
        '--output-directory', args.output / stage / 'market-readouts',
        '--card-sha256', admission['card_sha256'],
        '--input-manifest-sha256', pin(manifest)['sha256'],
        '--dataset-id', 'MKT-HIST-2'])
    run(stage + '_synthetic_heads', [args.build / 'causal_fresh_v4_synthetic_read',
        '--features', features / stage / 'synthetic',
        '--output', args.output / stage / 'synthetic-readouts', '--stage', stage])
receipt = {
    'kind': 'causal_fresh_v4_quality_execution_v1', 'status': 'passed',
    'protocol': 'causal-fresh-v4', 'model': 'CPB-v4.market-01',
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'card_sha256': admission['card_sha256'],
    'initial_checkpoint': pin(initial_path), 'final_checkpoint': pin(final_path),
    'initial_saved_proof': pin(args.initial_proof),
    'post_training_proof': pin(args.final_proof),
    'fresh_initialization': pin(fresh_path), 'build_admission': pin(args.admission),
    'engineering_proof': pin(args.engineering_proof),
    'variance_gradient_proof': pin(args.gradient_proof),
    'gradient_policy': 'variance_input_gradient_stop-v1',
    'stage_seconds': stages, 'old_encoder_calls': 0,
    'initial_quality_reused': False, 'initial_and_final_heads_new': True,
    'checkpoint_selection': 'fresh initial0 and final2048 only',
}
receipt_path = args.output / 'quality-receipt.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
run('report', [sys.executable,
    args.source / 'code/research/causal_fresh_v4/evaluation/report.py',
    '--run', args.output, '--training', args.training, '--card', card,
    '--input-manifest', manifest,
    '--raw-controls', args.source / 'doc/causal_fresh_v4/raw-controls.json',
    '--quality-receipt', receipt_path, '--output', args.output / 'report'])
run('comparison', [sys.executable,
    args.source / 'code/research/causal_fresh_v4/evaluation/compare_baseline.py',
    '--card', card, '--candidate-table', args.output / 'report/table_rows.json',
    '--source', args.source, '--output', args.output / 'comparison'])
print(json.dumps({'status': 'passed', 'quality_root': str(args.output),
                  'stage_seconds': stages}))

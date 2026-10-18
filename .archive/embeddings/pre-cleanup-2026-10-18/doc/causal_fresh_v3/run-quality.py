# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
"""One initial/final fixed-head evaluation after fresh-provenance admission."""
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
             'admission', 'data-root', 'output'):
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
require(admission['status'] == 'passed', 'Build admission required')
for binary in admission['binaries']:
    require(pin(Path(binary['path'])) == binary, 'Admitted binary changed')
engineering = read(args.engineering_proof)
require(engineering['status'] == 'passed' and engineering['source_updates'] == 0
        and engineering['old_checkpoints_opened'] == 0
        and engineering['all_initial_tensors_equal_seeded_constructor']
        and engineering['first_typed_artifact_exact']
        and engineering['typed_resume_next_loss_model_moments_bitwise_exact']
        and engineering['typed_negative_cases'] >= 30,
        'Fresh CUDA engineering, causal matrices and typed refusal proof required')
training = read(args.training / 'training-summary.json')
require(training['status'] == 'passed' and training['new_causal_updates'] == 2048
        and training['source_updates'] == 0,
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
            and proof['causal_updates'] == updates
            and proof['new_training_updates'] == 0
            and proof['future_values_and_masks_invariant']
            and proof['checkpoint_sha256'] == pin(path)['sha256'],
            'Saved initial/final checkpoint causal proof must bind scored bytes')

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


settings = args.source / 'code/encoders/causal_patch_bottleneck_mae/versions/v3/config/v3.conf'
card = args.source / 'doc/CAUSAL_FRESH_V3_CARD.json'
manifest = args.source / 'doc/causal_fresh_v3/input-manifest.json'
require(pin(card)['sha256'] == admission['card_sha256'], 'Frozen card changed')
require(pin(manifest)['sha256'] == read(card)['input_manifest']['sha256'],
        'Frozen input manifest changed')
features = args.output / 'features'
run('extraction', [args.build / 'causal_fresh_v3_extract',
    '--initial-checkpoint', initial_path, '--final-checkpoint', final_path,
    '--settings', settings, '--output', features, '--data-root', args.data_root])
shared = args.data_root / 'output/runs/market-capacity-study/shared-MKT-HIST-2-v1'
for stage in ('initial', 'final'):
    (args.output / stage).mkdir(exist_ok=False)
    run(stage + '_market_heads', [args.build / 'causal_fresh_v3_market_read',
        '--train-targets', shared / 'features/train-targets.pt',
        '--validation-targets', shared / 'features/validation-targets.pt',
        '--method', 'cpb_v3', features / stage / 'market/cpb_v3-train.pt',
        features / stage / 'market/cpb_v3-validation.pt',
        '--output-directory', args.output / stage / 'market-readouts',
        '--card-sha256', admission['card_sha256'],
        '--input-manifest-sha256', pin(manifest)['sha256'],
        '--dataset-id', 'MKT-HIST-2'])
    run(stage + '_synthetic_heads', [args.build / 'causal_fresh_v3_synthetic_read',
        '--features', features / stage / 'synthetic',
        '--output', args.output / stage / 'synthetic-readouts', '--stage', stage])
receipt = {
    'kind': 'causal_fresh_v3_quality_execution_v1', 'status': 'passed',
    'protocol': 'causal-fresh-v3', 'model': 'CPB-v3.market-01',
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'card_sha256': admission['card_sha256'],
    'initial_checkpoint': pin(initial_path), 'final_checkpoint': pin(final_path),
    'initial_saved_proof': pin(args.initial_proof),
    'post_training_proof': pin(args.final_proof),
    'fresh_initialization': pin(fresh_path), 'build_admission': pin(args.admission),
    'engineering_proof': pin(args.engineering_proof),
    'stage_seconds': stages, 'old_encoder_calls': 0,
    'initial_quality_reused': False, 'initial_and_final_heads_new': True,
    'checkpoint_selection': 'fresh initial0 and final2048 only',
}
receipt_path = args.output / 'quality-receipt.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
run('report', [sys.executable,
    args.source / 'code/research/causal_fresh_v3/evaluation/report.py',
    '--run', args.output, '--training', args.training, '--card', card,
    '--input-manifest', manifest,
    '--raw-controls', args.source / 'doc/causal_fresh_v3/raw-controls.json',
    '--quality-receipt', receipt_path, '--output', args.output / 'report'])
print(json.dumps({'status': 'passed', 'quality_root': str(args.output),
                  'stage_seconds': stages}))

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
"""Run the single final extraction/readout pass after all engineering gates."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
for name in ('build', 'source', 'training', 'proof', 'admission', 'data-root', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def pin(path):
    content = path.read_bytes()
    return {'path': str(path), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


admission = read(args.admission)
require(admission['status'] == 'passed', 'Build admission required')
for binary in admission['binaries']:
    require(pin(Path(binary['path'])) == binary, 'Admitted binary changed')
training = read(args.training / 'training-summary.json')
require(training['status'] == 'passed' and training['new_causal_updates'] == 2048,
        'One completed 2048-update causal trajectory required')
require(read(args.training / 'typed-resume-512.json')['status'] == 'passed',
        'Actual trajectory typed resume required')
parity_path = args.training / 'initial-parity.json'
parity = read(parity_path)
require(parity['status'] == 'passed' and parity['rows_checked'] == 144 and all(parity[key] for key in (
    'all_initial_weights_exact', 'all_initial_scaler_bank_exact',
    'initial_feature_witness_exact', 'initial_saved_features_reused')),
    'Exact baseline initialization and all saved market feature parity required')
final_path = args.training / 'checkpoints/inference-2048.pt'
proof = read(args.proof)
require(proof['status'] == 'passed', 'Final saved-checkpoint causal proof required')
proof_sha = proof.get('final_checkpoint_sha256', proof.get('checkpoint_sha256'))
require(proof_sha == pin(final_path)['sha256'], 'Final proof must bind the scored checkpoint')
require(proof.get('causal_updates') == 2048 and proof.get('new_training_updates') == 0,
        'Final artifact proof must be frozen inference at2048 updates')
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
                   'exit_code': process.returncode, 'elapsed_seconds': stages[name]}, stream, indent=2)
        stream.write('\n')
    require(process.returncode == 0, f'{name} failed; preserve partial outputs and do not repeat fits')


settings = args.source / 'code/encoders/causal_patch_bottleneck_mae/versions/v2/config/v2.conf'
card = args.source / 'doc/CAUSAL_FINETUNE_V2_CARD.json'
features = args.output / 'features'
run('extraction', [args.build / 'causal_finetune_v2_extract', '--checkpoint', final_path,
    '--settings', settings, '--output', features, '--data-root', args.data_root])
shared = args.data_root / 'output/runs/market-capacity-study/shared-MKT-HIST-2-v1'
run('market_heads', [args.build / 'causal_finetune_v2_market_read',
    '--train-targets', shared / 'features/train-targets.pt',
    '--validation-targets', shared / 'features/validation-targets.pt',
    '--method', 'cpb_v2', features / 'market/cpb_v2-train.pt', features / 'market/cpb_v2-validation.pt',
    '--output-directory', args.output / 'market-readouts', '--card-sha256', admission['card_sha256'],
    '--input-manifest-sha256', '828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8',
    '--dataset-id', 'MKT-HIST-2'])
run('synthetic_heads', [args.build / 'causal_finetune_v2_synthetic_read',
    '--features', features / 'synthetic', '--output', args.output / 'synthetic-readouts'])
receipt = {
    'kind': 'causal_finetune_v2_quality_execution_v1', 'status': 'passed',
    'protocol': 'causal-finetune-v2', 'model': 'CPB-v2.market-01',
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'final_checkpoint': pin(final_path), 'post_training_proof': pin(args.proof),
    'initial_parity': pin(parity_path), 'build_admission': pin(args.admission),
    'stage_seconds': stages, 'old_encoder_calls': 0, 'initial_head_refits': 0,
    'checkpoint_selection': 'final2048 only',
}
receipt_path = args.output / 'quality-receipt.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
run('report', [sys.executable, args.source / 'code/research/causal_finetune_v2/evaluation/report.py',
    '--run', args.output, '--training', args.training, '--data-root', args.data_root,
    '--card', card, '--initial-parity', parity_path, '--quality-receipt', receipt_path,
    '--output', args.output / 'report'])
print(json.dumps({'status': 'passed', 'quality_root': str(args.output), 'stage_seconds': stages}))

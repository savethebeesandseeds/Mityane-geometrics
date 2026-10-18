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
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
if not Path('/.dockerenv').is_file():
    raise RuntimeError('Existing project container required')
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command or not a.name.replace('-', '').replace('_', '').isalnum():
    raise RuntimeError('Explicit command and simple stage name required')
log = a.root / (a.name + '.log')
receipt = a.root / (a.name + '-stage.json')
if log.exists() or receipt.exists():
    raise RuntimeError('Preserve the existing stage; refuse replay')


def pin(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


start = time.monotonic()
with log.open('x') as out:
    process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
result = {
    'kind': 'causal_fresh_v4_exclusive_stage_execution_v1',
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
    if (Path(command[0]).name != 'test_cpb_v4_variance_gradient'
            or 'PASS CPB-v4 variance_input_gradient_stop-v1:' not in observed
            or 'manual full-NLL change4 gradient parity' not in observed
            or 'synthetic optimizer steps=1' not in observed):
        raise RuntimeError('Actual dedicated CUDA test PASS output required')
    proof = {
        'status': 'passed', 'model': 'CPB-v4',
        'gradient_policy': 'variance_input_gradient_stop-v1', 'actual_CUDA': True,
        'native_variance_gradient_blocked': True, 'mean_path_live': True,
        'variance_head_learning': True, 'forward_variance_input_dependence': True,
        'manual_partial_gradient_parity': True, 'nonvacuous_nonzero_fixture': True,
        'binary': result['binary'], 'log': result['log'],
        'execution': pin(receipt),
    }
    with (a.root / 'variance-gradient-proof.json').open('x') as out:
        json.dump(proof, out, indent=2)
        out.write('\n')

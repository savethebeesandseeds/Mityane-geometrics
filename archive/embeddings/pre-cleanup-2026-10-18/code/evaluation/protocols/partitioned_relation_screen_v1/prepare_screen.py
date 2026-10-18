#!/usr/bin/env python3
"""Freeze a small architecture screen without executing any model or head."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
MASTERS = {75272, 76373, 77474, 78575, 79676}


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-list', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--masters', required=True)
    parser.add_argument('--sdk-proof', default='/opt/cuwacunu_embedding/setup/sdk-20261008T234205Z-6a74defe83e3/proof.json', type=Path)
    args = parser.parse_args()
    root = Path('/embedding')
    masters = [int(x) for x in args.masters.split(',')]
    if not masters or len(set(masters)) != len(masters) or not set(masters) <= MASTERS:
        raise ValueError('declared retained masters only')
    paths = sorted(set(args.source_list.read_text().splitlines()))
    if not paths or any(not x or Path(x).is_absolute() or '..' in Path(x).parts for x in paths):
        raise ValueError('closed relative source list required')
    sources = ''.join(f'{sha(root / p)}  {p}\n' for p in paths)
    inputs = ''.join(f'{sha(args.input_root / p)}  {p}\n' for m in masters
                     for view in ('training', 'validation', 'validation-deleted')
                     for p in [f'results/seed-{m}-lag_sign/controlled-{view}.pt'])
    card = root / 'code/evaluation/cards/partitioned_relation_screen_v1.md'
    sdk_sha = sha(args.sdk_proof)
    if sdk_sha != 'e370b91a50e264739c284c0fc5b461436b60a50735b36a2777750a6603b3a0bb':
        raise ValueError('existing verified SDK identity changed')
    args.output.mkdir(parents=False, exist_ok=False)
    (args.output / 'sources.sha256').write_text(sources)
    (args.output / 'inputs.sha256').write_text(inputs)
    shutil.copyfile(card, args.output / 'card.md')
    shutil.copyfile(args.sdk_proof, args.output / 'sdk-proof.json')
    for p in paths:
        target = args.output / 'SOURCE' / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / p, target)
    freeze = {'protocol': 'partitioned-relation-screen-v1', 'masters': masters,
              'input_root': str(args.input_root), 'source_files': len(paths),
              'source_fingerprint': hashlib.sha256(sources.encode()).hexdigest(),
              'inputs_sha256': sha(args.output / 'inputs.sha256'),
              'card_sha256': sha(card), 'sdk_proof_sha256': sdk_sha,
              'model_execution': False, 'encoder_training': False,
              'head_fitting': False, 'old_artifacts_replaced': False}
    (args.output / 'freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    print(json.dumps(freeze, sort_keys=True))


if __name__ == '__main__':
    main()

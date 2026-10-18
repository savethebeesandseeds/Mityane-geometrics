#!/usr/bin/env python3
"""Freeze sources and optionally fresh input files; no model/head execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
MASTERS = [80787, 81888, 82989, 84090, 85191]
PROTOCOL = 'fixed-prior-fresh-confirmation-v1'
SDK_SHA = 'e370b91a50e264739c284c0fc5b461436b60a50735b36a2777750a6603b3a0bb'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-list', required=True, type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sdk-proof', type=Path,
                        default='/opt/cuwacunu_embedding/setup/sdk-20261008T234205Z-6a74defe83e3/proof.json')
    args = parser.parse_args()
    root = Path('/embedding')
    paths = sorted(set(args.source_list.read_text().splitlines()))
    if not paths or any(not p or Path(p).is_absolute() or '..' in Path(p).parts for p in paths):
        raise ValueError('closed relative source list required')
    sources = ''.join(f'{sha(root / p)}  {p}\n' for p in paths)
    card = root / 'code/evaluation/cards/fixed_prior_fresh_confirmation_v1.md'
    if sha(args.sdk_proof) != SDK_SHA:
        raise ValueError('existing verified SDK identity changed')
    inputs = None
    if args.input_root is not None:
        data_report = json.loads((args.input_root / 'data-report.json').read_text())
        if data_report['protocol'] != PROTOCOL or data_report['masters'] != MASTERS:
            raise ValueError('complete prospectively declared fresh cohorts required')
        if (data_report['source_fingerprint'] != hashlib.sha256(sources.encode()).hexdigest()
                or data_report['card_sha256'] != sha(card)):
            raise ValueError('fresh writer must match the exact compiled sources and prospective card')
        inputs = ''.join(f'{sha(args.input_root / p)}  {p}\n' for m in MASTERS
                         for view in ('training', 'validation', 'validation-deleted')
                         for p in [f'results/seed-{m}-lag_sign/controlled-{view}.pt'])
        declared_files = {r['path']: r['sha256'] for r in data_report['files']}
        if len(data_report['files']) != 15 or len(declared_files) != 15:
            raise ValueError('exact complete fresh file matrix required')
        for line in inputs.splitlines():
            digest, name = line.split('  ', 1)
            if declared_files.get(name) != digest:
                raise ValueError('fresh archive differs from the writer report')
    args.output.mkdir(parents=False, exist_ok=False)
    (args.output / 'sources.sha256').write_text(sources)
    shutil.copyfile(card, args.output / 'card.md')
    shutil.copyfile(args.sdk_proof, args.output / 'sdk-proof.json')
    for p in paths:
        target = args.output / 'SOURCE' / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / p, target)
    freeze = {'protocol': PROTOCOL, 'instance_bundle': 'RPB-v17.alt-01', 'masters': MASTERS,
              'input_root': str(args.input_root) if args.input_root else None,
              'source_files': len(paths),
              'source_fingerprint': hashlib.sha256(sources.encode()).hexdigest(),
              'card_sha256': sha(card), 'sdk_proof_sha256': SDK_SHA,
              'model_execution': False, 'encoder_training': False, 'head_fitting': False,
              'old_artifacts_replaced': False}
    if inputs is not None:
        (args.output / 'inputs.sha256').write_text(inputs)
        shutil.copyfile(args.input_root / 'data-report.json', args.output / 'data-report.json')
        freeze['inputs_sha256'] = sha(args.output / 'inputs.sha256')
        freeze['data_report_sha256'] = sha(args.output / 'data-report.json')
    (args.output / 'freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    print(json.dumps(freeze, sort_keys=True))


if __name__ == '__main__':
    main()

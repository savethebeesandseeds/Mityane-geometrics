#!/usr/bin/env python3
"""Freeze the new data-only experiment before engineering cohorts are generated."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/embedding')
SDK = Path('/opt/cuwacunu_embedding/setup/sdk-20261008T234205Z-6a74defe83e3/proof.json')
SDK_SHA = 'e370b91a50e264739c284c0fc5b461436b60a50735b36a2777750a6603b3a0bb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-list', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(set(args.source_list.read_text().splitlines()))
    if not paths or any(not p or Path(p).is_absolute() or '..' in Path(p).parts for p in paths):
        raise ValueError('closed relative source list required')
    if sha(SDK) != SDK_SHA:
        raise ValueError('existing verified SDK proof changed')
    card = ROOT / 'code/evaluation/cards/two_component_information_v1.md'
    manifest = ''.join(f'{sha(ROOT / p)}  {p}\n' for p in paths)
    args.output.mkdir(exist_ok=False)
    (args.output / 'sources.sha256').write_text(manifest)
    for p in paths:
        target = args.output / 'SOURCE' / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, target)
    shutil.copyfile(card, args.output / 'card.md')
    shutil.copyfile(SDK, args.output / 'sdk-proof.json')
    record = {
        'protocol': 'two-component-information-v1',
        'source_files': len(paths),
        'source_fingerprint': hashlib.sha256(manifest.encode()).hexdigest(),
        'card_sha256': sha(card), 'sdk_proof_sha256': SDK_SHA,
        'engineering_masters': [910901, 910902],
        'datasets': ['TEMPO-4', 'AMP-2'],
        'designed_complexity_level_each': 5, 'complexity_scale_max': 5,
        'model_execution': False, 'head_fitting': False,
        'quality_generation': False, 'old_artifacts_replaced': False}
    (args.output / 'freeze.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Hash-only before/after preservation receipt; never loads or executes a model."""
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
import argparse
import hashlib
import json
import pathlib
import time

ROOT = pathlib.Path('/cuwacunu')
EXPECTED = {
    'output/runs/market-capacity-study/run-base-gryiAV/checkpoints/inference-2048.pt':
        'e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c',
    'output/runs/market-consecutive-changes/run-hAxXvh/checkpoints/inference-2048.pt':
        '50ade502e1da9b11b71fad60afbf7daee7ecd714a4e719cc5b397191207ba004',
    'output/runs/market-separate-changes/run-Lr58TR/checkpoints/inference-2048.pt':
        'f2e0cb87de670b36fcabc5add7595ca1e4cfb79f9dafbdc5c9193622ba8263c6',
    'output/runs/market-separate-changes/run-Lr58TR/checkpoints/training-2048.pt':
        'd2dee0a4d05c7131fcf5ec478471ff2c8ebc753f472f5bc7a18d8ff108e9f328',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def historical(path):
    name = path.relative_to(ROOT).as_posix()
    return ('native36' not in name and path.is_file()
            and not path.is_symlink())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['record', 'verify'])
    parser.add_argument('manifest', type=pathlib.Path)
    parser.add_argument('--receipt', type=pathlib.Path)
    args = parser.parse_args()
    assert pathlib.Path('/.dockerenv').is_file(), 'Use managed container'
    assert ROOT.resolve() == pathlib.Path.cwd().resolve(), 'Use integration cwd'
    start = time.monotonic()
    for name, expected in EXPECTED.items():
        assert sha(ROOT / name) == expected, 'Handoff checkpoint mismatch: ' + name
    committed_artifacts = {}
    for receipt, key in [
        ('doc/results/market_training_study_v1/capacity-completion.json', 'original_artifacts'),
        ('doc/results/consecutive_change_v1/completion.json', 'source_assets'),
        ('doc/results/frozen_change_v1/completion.json', 'source_assets'),
        ('doc/results/separate_change_v1/completion.json', 'source_assets'),
    ]:
        for item in json.loads((ROOT / receipt).read_text(encoding='utf-8'))[key]:
            name = item['source']
            assert name not in committed_artifacts or committed_artifacts[name] == item['sha256'], name
            committed_artifacts[name] = item['sha256']
    for name, expected in committed_artifacts.items():
        assert sha(ROOT / name) == expected, 'Committed historical artifact mismatch: ' + name
    if args.mode == 'record':
        # All old ignored run artifacts, durable result trees and source/config
        # files are protected; mutable live status documents are outside scope.
        paths = set()
        for directory in ['output', 'doc/results', 'code', 'config']:
            paths.update(p for p in (ROOT / directory).rglob('*') if historical(p))
        paths.update(ROOT / name for name in [
            'doc/MARKET_SEPARATE_CHANGE_CARD.md',
            'doc/MARKET_SEPARATE_CHANGE_RESULTS.md',
            'doc/MARKET_CONSECUTIVE_CHANGE_CARD.md',
            'doc/MARKET_CONSECUTIVE_CHANGE_RESULTS.md',
            'doc/MARKET_FROZEN_CHANGE_CARD.md',
            'doc/MARKET_FROZEN_CHANGE_RESULTS.md',
            'doc/HANDOFF_2026-10-14.md',
        ])
        members = {p.relative_to(ROOT).as_posix(): {
            'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(paths)}
        payload = {'kind': 'native36_historical_preservation_v1',
                   'handoff_checkpoints': EXPECTED, 'files': members}
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
    else:
        payload = json.loads(args.manifest.read_text(encoding='utf-8'))
        members = payload['files']
        for name, info in members.items():
            p = ROOT / name
            assert p.is_file() and p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], name
    receipt = {'kind': 'native36_preservation_check_v1', 'mode': args.mode,
               'status': 'passed', 'files_checked': len(members),
               'output_files_checked': sum(p.startswith('output/') for p in members),
               'historical_checkpoint_files_checked': sum(p.endswith('.pt') and '/checkpoints/' in p for p in members),
               'handoff_checkpoints_checked': len(EXPECTED),
               'committed_historical_artifacts_checked': len(committed_artifacts),
               'manifest_sha256': sha(args.manifest),
               'seconds': time.monotonic() - start, 'model_execution_calls': 0}
    if args.receipt:
        with args.receipt.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

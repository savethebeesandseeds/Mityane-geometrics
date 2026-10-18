#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Claim the single CPB-v1 quality pass before delegating to the frozen runner."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/cuwacunu')
PARENT = ROOT / 'output/runs/causal-patch-bottleneck-mae-v1'
CLAIM = PARENT / 'quality-card-claim.json'
RESULT = PARENT / 'quality-guard-result.json'
RUNNER = ROOT / 'code/scripts/run-causal-patch-bottleneck-mae-v1.py'
CARD = ROOT / 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_CARD.md'
CARD_SHA = '3b059b84a60a7d193ffcb034480f54fa5aba532ed6cb6b541fa620c4ab584d5f'
ADDENDUM = ROOT / 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_LAYOUT_ADDENDUM.md'
ADDENDUM_SHA = 'ae0a8adae317a7b2daee3f58a645c9000faab130cde17ba6662a66fae06047dd'


def require(value, why):
    if not value:
        raise RuntimeError(why)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def resolve(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def pin(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify(items):
    for item in items:
        path = resolve(item['path'])
        require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Admitted source/artifact changed: ' + str(path))


def exclusive(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engineering', type=Path, required=True)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Existing cuwacunu container only')
    leaf = args.engineering.resolve(strict=True)
    require(leaf.is_relative_to(ROOT / 'output/engineering/causal-patch-bottleneck-mae-v1'), 'Declared engineering leaf')
    require(sha(CARD) == CARD_SHA and sha(ADDENDUM) == ADDENDUM_SHA, 'Frozen prospective card/addendum')
    require(not CLAIM.exists() and not RESULT.exists() and not list(PARENT.glob('run-*')) and
            not (leaf / 'quality-claimed.json').exists(), 'Preserve any earlier quality intent/attempt; no silent retry')
    tools_path = leaf / 'quality-tools-admission.json'
    tools = load(tools_path)
    require(tools['status'] == 'passed' and tools['sources'], 'Prospective quality-tools admission required')
    verify(tools['sources'])
    require(any(resolve(item['path']).resolve() == Path(__file__).resolve() for item in tools['sources']),
            'This global quality guard must be source-admitted')
    admission_path = leaf / 'admission.json'
    admission = load(admission_path)
    verify(admission['source'])
    require(any(resolve(item['path']).resolve() == RUNNER for item in admission['source']), 'Exact frozen runner is admitted')
    require(load(leaf / 'source-review.json')['status'] == load(leaf / 'independent-audit.json')['status'] == 'passed',
            'Independent source and causal proof audit required')
    stage = load(leaf / 'proof-stage.json')
    require(stage['status'] == 'passed', 'Successful CUDA causality proof required')
    verify([stage['proof'], stage['causal_checkpoint']])
    require(load(resolve(stage['proof']['path']))['status'] == 'passed', 'Actual saved proof status')
    PARENT.mkdir(parents=True, exist_ok=True)
    # Recheck immediately before the exclusive global claim. A competing launcher
    # cannot pass open('x'); a failed/abandoned claim is deliberately never removed.
    require(not list(PARENT.glob('run-*')) and not (leaf / 'quality-claimed.json').exists(),
            'An earlier quality attempt appeared; preserve it')
    exclusive(CLAIM, {'kind': 'causal_patch_bottleneck_mae_v1_global_quality_claim_v1',
                     'status': 'claimed_before_quality', 'claimed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'protocol': 'causal-patch-bottleneck-mae-v1', 'model': 'CPB-v1.market-01',
                     'engineering': str(leaf), 'card': pin(CARD), 'layout_addendum': pin(ADDENDUM),
                     'quality_tools_admission': pin(tools_path), 'admitted_tool_sources': tools['sources'],
                     'build_admission': pin(admission_path), 'proof': stage['proof'],
                     'causal_checkpoint': stage['causal_checkpoint'], 'frozen_runner': pin(RUNNER),
                     'guard': pin(Path(__file__).resolve()), 'retry_policy': 'No second global claim or run under this card, including failed/empty attempts'})
    print('CPB_GLOBAL_QUALITY_CLAIM=' + str(CLAIM), flush=True)
    result = subprocess.run([sys.executable, str(RUNNER), 'quality', '--engineering', str(leaf)], cwd=ROOT)
    exclusive(RESULT, {'kind': 'causal_patch_bottleneck_mae_v1_global_quality_guard_result_v1',
                      'status': 'passed' if result.returncode == 0 else 'failed_preserved_no_retry',
                      'exit_code': result.returncode, 'claim': pin(CLAIM),
                      'run_roots': [str(path) for path in sorted(PARENT.glob('run-*'))],
                      'guard': pin(Path(__file__).resolve()), 'claim_removed': False})
    require(result.returncode == 0, 'Quality failed; claim, run and result preserved; no silent retry')


if __name__ == '__main__':
    main()

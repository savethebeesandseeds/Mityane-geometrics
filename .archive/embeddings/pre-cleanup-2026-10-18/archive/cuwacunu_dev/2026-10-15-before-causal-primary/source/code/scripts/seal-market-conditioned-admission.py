#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Bind the passed artificial fixture and final producer before quality starts."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/cuwacunu')
OUT = ROOT / 'doc/results/market_conditioned_changes_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def require(value, why):
    if not value:
        raise RuntimeError(why)


def main():
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Use existing container')
    require(not (OUT / 'admission-review.json').exists(), 'Preserve sealed admission')
    require(not list((ROOT / 'output/runs/market-conditioned-changes').glob('run-*')), 'Quality must not have started')
    fixture = json.loads((OUT / 'engineering-admission.json').read_text())
    require(fixture['status'] == 'passed' and fixture['gpu_fixture_invocations'] == 1 and
            fixture['failed_gpu_fixture_invocations'] == 0, 'First artificial CUDA admission required')
    for key in ('card', 'binary', 'pass_log'):
        path = Path(fixture[key]['path'])
        if not path.is_absolute():
            path = ROOT / path
        require(sha(path) == fixture[key]['sha256'], 'Fixture binding: ' + key)
    require(sha(ROOT / 'doc/MARKET_CONDITIONED_CHANGE_CARD.md') ==
            '66de5fbe159230c82c04bd45a0d48cdf0694e101a082319da2f8bf2bd8b10e84', 'Sealed card')
    pairs = json.loads((OUT / 'portable-sync.json').read_text())
    require(pairs['status'] == 'passed' and pairs['total_pairs'] == 64, '64 portable pairs required')
    for pair in pairs['pairs']:
        path = ROOT / pair['integration_path']
        require(sha(path) == pair['sha256'] and path.stat().st_size == pair['bytes'], 'Portable source changed')
    sources = [path for path in (ROOT / 'code/encoders/rpb_conditioned').rglob('*') if path.is_file()]
    sources += [path for path in (ROOT / 'code/encoders/raw_patch_bottleneck_mae/versions/v26').rglob('*') if path.is_file()]
    sources += [ROOT / name for name in (
        'code/training/src/market_conditioned_changes_main.cpp', 'code/training/market_conditioned_changes.mk',
        'code/scripts/run-market-conditioned-changes.sh', 'code/scripts/execute-market-conditioned-changes.py',
        'code/scripts/audit-market-conditioned-changes.py', 'code/scripts/report-market-conditioned-changes.py',
        'code/scripts/update-encoder-search-ledger.py', 'code/scripts/preflight-market-conditioned-changes.ps1',
        'code/scripts/sync-receipt-market-conditioned-changes.ps1',
        'config/market_studies/encoder_conditioned_changes.conf', 'config/market_studies/conditioned_changes.conf',
        'doc/MARKET_CONDITIONED_CHANGE_CARD.md', 'Makefile', 'code/scripts/task.sh')]
    sources += [Path(__file__).resolve(), Path('/opt/cuwacunu/build/market-conditioned/market_conditioned_changes'),
                Path('/opt/cuwacunu/build/conditioned-v26-engineering/test_conditioned_changes'),
                OUT / 'build-03.log', OUT / 'engineering-admission.json', OUT / 'portable-sync.json']
    require((OUT / 'build-03.log').is_file(), 'Final producer build log required')
    require(sha(Path('/opt/cuwacunu/build/market-conditioned/market_conditioned_changes')) ==
            '90de190073639aa58fa49f32b152d10ec92cc84cc515c46edf89ccc4df19418d',
            'Final reviewed producer binary required')
    assets = [ROOT / pin['path'] for pin in fixture['artifacts']]
    for path, pin in zip(assets, fixture['artifacts']):
        require(sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'], 'Fixture artifact changed')
    assets += [OUT / 'engineering-01.log', OUT / 'engineering-launch-01.log']
    result = dict(kind='market_conditioned_changes_admission_review_v1', status='passed',
                  protocol='market-conditioned-changes-v1', instance='RPB-v26.market-01', quality_started=False,
                  artificial_gpu_fixture_invocations=1, existing_container='cuwacunu',
                  primary='RPB-v22', balanced_reference='RPB-v24.market-01.alt-01',
                  reviews='Root mathematical/masking review and two independent read-only protocol/typed-state reviews passed',
                  protected_sources=[record(path) for path in sorted(set(sources))],
                  fixture_assets=[record(path) for path in assets], portable_pairs=64,
                  preflight_sha256=sha(OUT / 'preflight.json'))
    with (OUT / 'admission-review.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('CONDITIONED_ADMISSION_SEALED=' + sha(OUT / 'admission-review.json'))


if __name__ == '__main__':
    main()

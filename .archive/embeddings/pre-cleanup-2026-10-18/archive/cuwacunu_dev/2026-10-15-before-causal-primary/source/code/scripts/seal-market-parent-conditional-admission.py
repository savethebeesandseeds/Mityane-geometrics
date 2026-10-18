#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Bind the passed artificial fixture and final producer before quality starts."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/cuwacunu')
OUT = ROOT / 'doc/results/market_parent_conditional_changes_v1'


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
    require(not list((ROOT / 'output/runs/market-parent-conditional-changes').glob('run-*')), 'Quality must not have started')
    fixture = json.loads((OUT / 'engineering-admission.json').read_text())
    require(fixture['status'] == 'passed' and fixture['gpu_fixture_invocations'] >= 1 and
            0 <= fixture['failed_gpu_fixture_invocations'] < fixture['gpu_fixture_invocations'],
            'Passed artificial CUDA admission with all attempts preserved required')
    for key in ('card', 'passed_binary', 'pass_log'):
        path = Path(fixture[key]['path'])
        if not path.is_absolute():
            path = ROOT / path
        require(sha(path) == fixture[key]['sha256'], 'Fixture binding: ' + key)
    require(sha(ROOT / 'doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md') ==
            '6e24e48bfdd05148f99446cf1421a713a7736d522f6c00159ba156e43b39c29b', 'Sealed card')
    pairs = json.loads((OUT / 'portable-sync.json').read_text())
    require(pairs['status'] == 'passed' and pairs['total_pairs'] == 73, '73 portable pairs required')
    for pair in pairs['pairs']:
        path = ROOT / pair['integration_path']
        require(sha(path) == pair['sha256'] and path.stat().st_size == pair['bytes'], 'Portable source changed')
    sources = [path for path in (ROOT / 'code/encoders/rpb_conditional').rglob('*') if path.is_file()]
    sources += [path for path in (ROOT / 'code/encoders/raw_patch_bottleneck_mae/versions/v27').rglob('*') if path.is_file()]
    sources += [ROOT / name for name in (
        'code/training/src/market_parent_conditional_changes_main.cpp', 'code/training/market_parent_conditional_changes.mk',
        'code/scripts/run-market-parent-conditional-changes.sh', 'code/scripts/execute-market-parent-conditional-changes.py',
        'code/scripts/audit-market-parent-conditional-changes.py', 'code/scripts/report-market-parent-conditional-changes.py',
        'code/scripts/append-encoder-search-ledger.py', 'code/scripts/preflight-market-parent-conditional-changes.ps1',
        'code/scripts/sync-receipt-market-parent-conditional-changes.ps1',
        'config/market_studies/encoder_parent_conditional_changes.conf', 'config/market_studies/parent_conditional_changes.conf',
        'doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md', 'Makefile', 'code/scripts/task.sh')]
    sources += [Path(__file__).resolve(), Path('/opt/cuwacunu/build/market-parent-conditional/market_parent_conditional_changes'),
                Path('/opt/cuwacunu/build/parent-conditional-v27-engineering/test_parent_conditional_changes'),
                OUT / 'build-01.log', OUT / 'engineering-admission.json', OUT / 'portable-sync.json']
    require((OUT / 'build-01.log').is_file(), 'Final producer build log required')
    require(sha(Path('/opt/cuwacunu/build/market-parent-conditional/market_parent_conditional_changes')) ==
            '8d0e7c83e0e3a4866d04fcd04d81f4ebee297d6eb961847eb9e3e4cb1e070cde',
            'Final reviewed producer binary required')
    assets = [ROOT / pin['path'] for pin in fixture['artifacts']]
    for path, pin in zip(assets, fixture['artifacts']):
        require(sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'], 'Fixture artifact changed')
    assets += sorted(OUT.glob('engineering-*.log'))
    result = dict(kind='market_parent_conditional_changes_admission_review_v1', status='passed',
                  protocol='market-parent-conditional-changes-v1', instance='RPB-v27.market-01', quality_started=False,
                  artificial_gpu_fixture_invocations=fixture['gpu_fixture_invocations'],
                  failed_artificial_gpu_fixture_invocations=fixture['failed_gpu_fixture_invocations'],
                  existing_container='cuwacunu',
                  primary='RPB-v22', balanced_reference='RPB-v24.market-01.alt-01',
                  reviews='Root mathematical/masking review and independent read-only protocol and source audits passed',
                  protected_sources=[record(path) for path in sorted(set(sources))],
                  fixture_assets=[record(path) for path in assets], portable_pairs=73,
                  preflight_sha256=sha(OUT / 'preflight.json'))
    with (OUT / 'admission-review.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('PARENT_CONDITIONAL_ADMISSION_SEALED=' + sha(OUT / 'admission-review.json'))


if __name__ == '__main__':
    main()

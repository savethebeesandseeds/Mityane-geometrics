#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Bind the passed artificial fixture and final producer before quality starts."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/cuwacunu')
OUT = ROOT / 'doc/results/market_relative_changes_v1'


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
    require(not list((ROOT / 'output/runs/market-relative-changes').glob('run-*')), 'Quality must not have started')
    fixture = json.loads((OUT / 'engineering-admission.json').read_text())
    require(fixture['status'] == 'passed' and fixture['gpu_fixture_invocations'] >= 1 and
            0 <= fixture['failed_gpu_fixture_invocations'] < fixture['gpu_fixture_invocations'],
            'Passed artificial CUDA admission with all attempts preserved required')
    for key in ('card', 'passed_binary', 'pass_log'):
        path = Path(fixture[key]['path'])
        if not path.is_absolute():
            path = ROOT / path
        require(sha(path) == fixture[key]['sha256'], 'Fixture binding: ' + key)
    source_capture = fixture['source_capture']
    manifest_path = ROOT / source_capture['path']
    require(sha(manifest_path) == source_capture['sha256'], 'Pre-invocation source capsule binding')
    captured = json.loads(manifest_path.read_text())
    require(captured['file_count'] == len(captured['files']) == 49 and
            captured['captured_before_compilation'] and captured['captured_before_gpu_invocation'],
            'Complete source capsule precedes actual GPU admission')
    capsule_root = ROOT / source_capture['source_root']
    for pin in captured['files']:
        for path in (ROOT / pin['path'], capsule_root / pin['path']):
            require(sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'],
                    'Exact artificial source/capsule unchanged: ' + pin['path'])
    require(sha(ROOT / 'doc/MARKET_RELATIVE_CHANGE_CARD.md') ==
            'f4378d4a76185fc17b6c0adbadab5af27accd772ed66e5a8bc628ab7848f94a2', 'Sealed card')
    pairs = json.loads((OUT / 'portable-sync.json').read_text())
    require(pairs['status'] == 'passed' and pairs['total_pairs'] == 83, '83 portable pairs required')
    for pair in pairs['pairs']:
        path = ROOT / pair['integration_path']
        require(sha(path) == pair['sha256'] and path.stat().st_size == pair['bytes'], 'Portable source changed')
    sources = [path for path in (ROOT / 'code/encoders/rpb_relative').rglob('*') if path.is_file()]
    sources += [path for path in (ROOT / 'code/encoders/raw_patch_bottleneck_mae/versions/v28').rglob('*') if path.is_file()]
    sources += [ROOT / name for name in (
        'code/training/src/market_relative_changes_main.cpp', 'code/training/market_relative_changes.mk',
        'code/training/include/cuwacunu/training/market_raw_f64_data.h',
        'code/scripts/run-market-relative-changes.sh', 'code/scripts/execute-market-relative-changes.py',
        'code/scripts/audit-market-relative-changes.py', 'code/scripts/report-market-relative-changes.py',
        'code/scripts/append-encoder-search-ledger.py', 'code/scripts/preflight-market-relative-changes.ps1',
        'code/scripts/sync-receipt-market-relative-changes.ps1',
        'config/market_studies/encoder_relative_changes.conf', 'config/market_studies/relative_changes.conf',
        'doc/MARKET_RELATIVE_CHANGE_CARD.md', 'Makefile', 'code/scripts/task.sh')]
    sources += [Path(__file__).resolve(), Path('/opt/cuwacunu/build/market-relative/market_relative_changes'),
                Path('/opt/cuwacunu/build/relative-v28-engineering/test_relative_changes'),
                OUT / 'build-02.log', OUT / 'engineering-admission.json', OUT / 'portable-sync.json']
    sources += sorted(OUT.glob('build-*.log'))
    sources += sorted(path for path in OUT.rglob('*failed*') if path.is_file())
    sources += sorted(path for path in (OUT / 'build-01-source').rglob('*') if path.is_file())
    sources += [manifest_path] + sorted(path for path in capsule_root.rglob('*') if path.is_file())
    require((OUT / 'build-02.log').is_file(), 'Final producer build log required')
    require(sha(Path('/opt/cuwacunu/build/market-relative/market_relative_changes')) ==
            'd0f602dfb43264cce3c8962eac60d1ed00fec7c6adf8af7bd917e7c193d755f6',
            'Final reviewed producer binary required')
    assets = [ROOT / pin['path'] for pin in fixture['artifacts']]
    for path, pin in zip(assets, fixture['artifacts']):
        require(sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'], 'Fixture artifact changed')
    assets += sorted(OUT.glob('engineering-*.log'))
    result = dict(kind='market_relative_changes_admission_review_v1', status='passed',
                  protocol='market-relative-changes-v1', instance='RPB-v28.market-01', quality_started=False,
                  artificial_gpu_fixture_invocations=fixture['gpu_fixture_invocations'],
                  failed_artificial_gpu_fixture_invocations=fixture['failed_gpu_fixture_invocations'],
                  existing_container='cuwacunu',
                  primary='RPB-v22', balanced_reference='RPB-v24.market-01.alt-01',
                  reviews='Root mathematical/masking review and independent read-only protocol and source audits passed',
                  protected_sources=[record(path) for path in sorted(set(sources))],
                  fixture_assets=[record(path) for path in assets], portable_pairs=83,
                  preflight_sha256=sha(OUT / 'preflight.json'))
    with (OUT / 'admission-review.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('RELATIVE_ADMISSION_SEALED=' + sha(OUT / 'admission-review.json'))


if __name__ == '__main__':
    main()

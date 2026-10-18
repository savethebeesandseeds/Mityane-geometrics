#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Bind reviewed artificial engineering and compiled sources before quality."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/cuwacunu')
OUT = ROOT / 'doc/results/market_location_scale_decoder_v1'
CARD = 'f620298e9ecf9f73a32ae61737b67295f8dcde709416aa3d6ed16feee613e45c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def require(value, why):
    if not value:
        raise RuntimeError(why)


def check_pin(path, pin):
    require(sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'],
            'Reviewed source or artifact changed: ' + str(path))


def main():
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Use existing container')
    require(not (OUT / 'admission-review.json').exists(), 'Preserve sealed admission')
    require(not list((ROOT / 'output/runs/market-location-scale-decoder').glob('run-*')),
            'Quality must not have started')
    require(sha(ROOT / 'doc/MARKET_LOCATION_SCALE_DECODER_CARD.md') == CARD, 'Sealed prospective card')
    fixture = json.loads((OUT / 'engineering-admission.json').read_text())
    require(fixture['status'] == 'passed' and fixture['gpu_fixture_invocations'] >= 1 and
            0 <= fixture['failed_gpu_fixture_invocations'] < fixture['gpu_fixture_invocations'],
            'Passed artificial CUDA admission with all attempts preserved required')
    for key in ('card', 'passed_binary', 'pass_log'):
        path = Path(fixture[key]['path'])
        path = path if path.is_absolute() else ROOT / path
        check_pin(path, fixture[key])
    capture = fixture['source_capture']
    manifest_path = ROOT / capture['path']
    require(sha(manifest_path) == capture['sha256'], 'Fixture source manifest changed')
    captured = json.loads(manifest_path.read_text())
    require(captured['file_count'] == len(captured['files']) >= 50 and
            captured['captured_before_compilation'] and captured['captured_before_gpu_invocation'],
            'Full fixture source capsule must precede compilation/invocation')
    capsule_root = ROOT / capture['source_root']
    for pin in captured['files']:
        check_pin(ROOT / pin['path'], pin)
        check_pin(capsule_root / pin['path'], pin)
    pairs = json.loads((OUT / 'portable-sync.json').read_text())
    require(pairs['status'] == 'passed' and pairs['total_pairs'] == 99, '99 portable pairs required')
    for pin in pairs['pairs']:
        check_pin(ROOT / pin['integration_path'], pin)
    review_path = OUT / 'root-source-review.json'
    review = json.loads(review_path.read_text())
    require(review['status'] == 'passed' and review['quality_started'] is False and
            review['card_sha256'] == CARD, 'Root and independent source review required')
    for pin in review['reviewed_sources']:
        check_pin(ROOT / pin['path'], pin)
    build_path = OUT / 'producer-build-admission.json'
    build = json.loads(build_path.read_text())
    require(build['status'] == 'passed' and build['exit_code'] == build['warnings'] == 0,
            'Reviewed warning-free producer compilation required')
    producer_binary = Path('/opt/cuwacunu/build/market-location-scale/market_location_scale_decoder')
    require(sha(producer_binary) == build['binary_sha256'] == review['producer_binary_sha256'],
            'Exact reviewed producer executable required')
    require(sha(Path(fixture['passed_binary']['path'])) == review['fixture_binary_sha256'],
            'Exact reviewed fixture executable required')
    dependency_file = ROOT / build['compiled_dependency_file']
    require(sha(dependency_file) == build['compiled_dependency_sha256'], 'Compiler dependency capture changed')
    for pin in build['compiled_repository_dependencies']:
        check_pin(ROOT / pin['path'], pin)
    producer_manifest = ROOT / build['source_capsule_manifest']
    require(sha(producer_manifest) == build['source_capsule_manifest_sha256'], 'Producer source manifest changed')
    producer_capsule = producer_manifest.parent
    for pin in json.loads(producer_manifest.read_text()):
        check_pin(ROOT / pin['path'], pin)
        check_pin(producer_capsule / pin['path'], pin)
    sources = [path for path in (ROOT / 'code/encoders/rpb_location_scale').rglob('*') if path.is_file()]
    sources += [path for path in (ROOT / 'code/encoders/raw_patch_bottleneck_mae/versions/v30').rglob('*') if path.is_file()]
    sources += [ROOT / name for name in (
        'code/training/src/market_location_scale_decoder_main.cpp', 'code/training/market_location_scale_decoder.mk',
        'code/training/include/cuwacunu/training/market_raw_f64_data.h',
        'code/scripts/run-market-location-scale-decoder.sh', 'code/scripts/execute-market-location-scale-decoder.py',
        'code/scripts/audit-market-location-scale-decoder.py', 'code/scripts/report-market-location-scale-decoder.py',
        'code/scripts/append-encoder-search-ledger-v3.py', 'code/scripts/declare-market-location-scale-decoder.py',
        'code/scripts/preflight-market-location-scale-decoder.ps1', 'code/scripts/sync-receipt-market-location-scale-decoder.ps1',
        'config/market_studies/encoder_location_scale_decoder.conf', 'config/market_studies/location_scale_decoder.conf',
        'doc/MARKET_LOCATION_SCALE_DECODER_CARD.md', 'Makefile', 'code/scripts/task.sh')]
    sources += [Path(__file__).resolve(), producer_binary, Path(fixture['passed_binary']['path']),
                OUT / 'engineering-admission.json', OUT / 'portable-sync.json', review_path,
                build_path, dependency_file, manifest_path]
    sources += [path for path in producer_capsule.rglob('*') if path.is_file()]
    sources += [path for path in capsule_root.rglob('*') if path.is_file()]
    sources += sorted(OUT.glob('build-*.log')) + sorted(OUT.glob('engineering-*.log'))
    sources += sorted(path for path in OUT.rglob('*failed*') if path.is_file())
    assets = [ROOT / pin['path'] for pin in fixture['artifacts']]
    for path, pin in zip(assets, fixture['artifacts']):
        check_pin(path, pin)
    result = dict(kind='market_location_scale_decoder_admission_review_v1', status='passed',
                  protocol='market-location-scale-decoder-v1', instance='RPB-v30.market-01', quality_started=False,
                  artificial_gpu_fixture_invocations=fixture['gpu_fixture_invocations'],
                  failed_artificial_gpu_fixture_invocations=fixture['failed_gpu_fixture_invocations'],
                  existing_container='cuwacunu', primary='RPB-v22', balanced_reference='RPB-v24.market-01.alt-01',
                  reviews='Root mathematical/masking review plus independent protocol and source review',
                  protected_sources=[record(path) for path in sorted(set(sources))],
                  fixture_assets=[record(path) for path in assets], portable_pairs=99,
                  preflight_sha256=sha(OUT / 'preflight.json'))
    with (OUT / 'admission-review.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('LOCATION_SCALE_ADMISSION_SEALED=' + sha(OUT / 'admission-review.json'))


if __name__ == '__main__':
    main()

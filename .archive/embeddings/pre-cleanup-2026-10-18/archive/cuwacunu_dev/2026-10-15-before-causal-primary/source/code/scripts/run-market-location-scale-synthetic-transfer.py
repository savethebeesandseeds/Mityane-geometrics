#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Build, seal, then execute the one declared frozen V30 transfer in cuwacunu."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path('/cuwacunu')
EVIDENCE = ROOT / 'doc/results/market_location_scale_synthetic_transfer_v1'
INPUTS = ROOT / 'output/inputs/market-synthetic-transfer-v1'
PARENT = ROOT / 'output/runs/market-location-scale-synthetic-transfer'
BUILD = Path('/opt/cuwacunu/build/market-location-scale-transfer')
ENGINEERING = ROOT / 'output/engineering/location-scale-transfer-build'
CARD = ROOT / 'doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md'
SETTINGS = ROOT / 'config/market_studies/encoder_location_scale_decoder.conf'
CHECKPOINT = ROOT / 'output/runs/market-location-scale-decoder/run-MAmvTJ/checkpoints/inference-2048.pt'
AUDITOR = ROOT / 'code/scripts/audit-market-location-scale-synthetic-transfer.py'
NAMES = ('market_location_scale_synthetic_transfer_extract', 'market_location_scale_synthetic_transfer_read')
EXPECTED = {
    CARD: 'c3cfe0a04b174734599b2b01363c90c3f9b81fd5964fb30ffd4ce8a21976260e',
    SETTINGS: '43320396447ca80be5fde4324d17c3a9ab129e116bf184455fcb6261aeb6923c',
    CHECKPOINT: 'c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb',
    INPUTS / 'manifest.json': 'fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb',
    ROOT / 'doc/encoder_search_ledger.json': 'b8d3ab33dae4937a0202ac64302f58a5b6c803c024b7277604f26855c76a4012',
    EVIDENCE / 'preflight.json': 'b0d18e954de48684f73f08fe1a45e0d8d837baab028ef2ed34151f546bf07704',
}


def require(value, why):
    if not value:
        raise RuntimeError(why)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def record(path):
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pins(paths):
    return {str(path): record(path) for path in sorted(set(paths))}


def verify(records):
    for name, expected in records.items():
        require(record(Path(name)) == expected, 'Protected file changed: ' + name)


def sources():
    paths = [ROOT / 'Makefile', CARD, EVIDENCE / 'preflight.json',
             ROOT / 'doc/encoder_search_ledger.json', ROOT / 'doc/ENCODER_SEARCH_LEDGER.md']
    for folder in ('code', 'config'):
        paths.extend(path for path in (ROOT / folder).rglob('*') if path.is_file())
    return sorted(set(paths))


def inputs():
    for path, digest in EXPECTED.items():
        require(sha(path) == digest, 'Sealed prospective pin changed: ' + str(path))
    preflight = load(EVIDENCE / 'preflight.json')
    require(preflight['status'] == 'passed' and preflight['checkpoints_verified'] == 55 and
            preflight['portable_pairs_verified'] == 99 and preflight['all_portable_pairs_match'] and
            preflight['all_checkpoints_available_hash_matched_and_ignored'], 'Host preflight incomplete')
    paths = list(EXPECTED)
    for pin in preflight['checkpoints'] + preflight['frozen_records']:
        path = ROOT / pin['path']
        require(record(path) == {key: pin[key] for key in ('sha256', 'bytes')}, str(path))
        paths.append(path)
    for pair in preflight['portable_pairs']:
        path = ROOT / pair['integration_path']
        require(record(path) == {key: pair[key] for key in ('sha256', 'bytes')}, str(path))
        paths.append(path)
    manifest = load(INPUTS / 'manifest.json')
    require(len(manifest['members']) == 18, 'All original staged inputs required')
    for member in manifest['members']:
        path = INPUTS / member['path']
        require(path.resolve().is_relative_to(INPUTS) and record(path) ==
                {key: member[key] for key in ('sha256', 'bytes')}, str(path))
        paths.append(path)
    fixture = load(ROOT / 'doc/results/market_location_scale_decoder_v1/engineering-admission.json')
    require(fixture['status'] == 'passed', 'Existing V30 artificial CUDA admission required')
    for key in ('card', 'passed_binary', 'pass_log'):
        pin = fixture[key]
        path = Path(pin['path'])
        path = path if path.is_absolute() else ROOT / path
        require(record(path) == {key: pin[key] for key in ('sha256', 'bytes')}, str(path))
        paths.append(path)
    return sorted(set(paths))


def command(args, log):
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run([str(arg) for arg in args], stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def build():
    inputs()
    ENGINEERING.mkdir(parents=True, exist_ok=True)
    attempt = ENGINEERING / 'attempt-01'
    attempt.mkdir()  # An existing attempt requires explicit additive repair, never overwrite.
    before = pins(sources())
    capsule = attempt / 'sources'
    for name in before:
        path = Path(name)
        target = capsule / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    dump(attempt / 'source-capsule.json', before)
    toolchain = subprocess.check_output(['bash', '-lc',
        'source /etc/profile.d/embedding.sh; g++ --version; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'], text=True)
    with (attempt / 'toolchain.txt').open('x') as stream:
        stream.write(toolchain)
    started = time.monotonic()
    code = command(['bash', ROOT / 'code/scripts/task.sh', 'market-location-scale-transfer', '-j2',
                    'build-market-location-scale-synthetic-transfer-extract',
                    'build-market-location-scale-synthetic-transfer-read'], attempt / 'build.log')
    seconds = time.monotonic() - started
    verify(before)
    result = {'kind': 'market_location_scale_transfer_build_v1', 'exit_code': code,
              'status': 'passed' if code == 0 else 'failed', 'seconds': seconds,
              'quality_started': False, 'source_capsule': str(attempt / 'source-capsule.json'),
              'source_capsule_sha256': sha(attempt / 'source-capsule.json'),
              'toolchain_sha256': sha(attempt / 'toolchain.txt'),
              'build_log': record(attempt / 'build.log')}
    if code == 0:
        dependencies = set()
        for dep in sorted(BUILD.glob('*.d')):
            shutil.copyfile(dep, attempt / dep.name)
            # Make-generated repository paths contain no spaces in this workspace.
            body = dep.read_text().replace('\\\n', ' ')
            for token in body.split(':', 1)[1].split():
                candidate = Path(token)
                if token.endswith(':'):
                    continue
                if not candidate.is_absolute():
                    candidate = ROOT / candidate
                if candidate.is_file() and candidate.resolve().is_relative_to(ROOT):
                    dependencies.add(candidate)
        require(dependencies, 'Compiler must expose repository dependencies')
        for path in dependencies:
            require(str(path) in before, 'Uncaptured compiler dependency: ' + str(path))
        result['compiled_repository_dependencies'] = pins(dependencies)
        result['binaries'] = pins(BUILD / name for name in NAMES)
        for name in NAMES:
            shutil.copyfile(BUILD / name, attempt / name)
        result['warnings'] = (attempt / 'build.log').read_text().count('warning:')
    dump(attempt / 'build-receipt.json', result)
    require(code == 0 and result['warnings'] == 0, 'Preserved build failed or warned; inspect before any additive repair')
    print('TRANSFER_BUILD_PASSED=' + str(attempt), flush=True)


def seal():
    input_paths = inputs()
    require(not list(PARENT.glob('run-*')), 'No transfer quality before admission')
    attempt = ENGINEERING / 'attempt-01'
    build_record = load(attempt / 'build-receipt.json')
    require(build_record['status'] == 'passed' and build_record['warnings'] == 0,
            'Warning-free captured build required')
    captured = load(attempt / 'source-capsule.json')
    verify(captured)
    verify(build_record['compiled_repository_dependencies'])
    verify(build_record['binaries'])
    for name, pin in captured.items():
        require(record(attempt / 'sources' / Path(name).relative_to(ROOT)) == pin, 'Captured source changed')
    review = load(EVIDENCE / 'root-source-review.json')
    require(review['status'] == 'passed' and review['quality_started'] is False and
            review['card_sha256'] == EXPECTED[CARD], 'Independent static review and root approval required')
    verify(review['reviewed_sources'])
    require(AUDITOR.is_file(), 'Independent saved-arithmetic auditor required')
    protected = sources() + input_paths + [AUDITOR, EVIDENCE / 'root-source-review.json']
    protected += [path for path in attempt.rglob('*') if path.is_file()]
    protected += [BUILD / name for name in NAMES]
    dump(EVIDENCE / 'admission-review.json', {
        'kind': 'market_location_scale_synthetic_transfer_admission_v1', 'status': 'passed',
        'quality_started': False, 'card_sha256': EXPECTED[CARD],
        'protected': pins(protected), 'encoder_updates': 0, 'quality_encoder_calls': 0,
        'core_cuda_fixture_reused': True, 'artificial_encoder_calls_new': 0,
        'contract': '768fullCUDAencodes+6same-checkpointparent-onlycorrectnesscalls;0decoder/optimizer/calibrationcalls'})
    print('TRANSFER_ADMISSION_SEALED', flush=True)


def execute():
    input_paths = inputs()
    admission = load(EVIDENCE / 'admission-review.json')
    require(admission['status'] == 'passed' and admission['quality_started'] is False,
            'Frozen admission required')
    verify(admission['protected'])
    PARENT.mkdir(parents=True, exist_ok=True)
    require(not list(PARENT.glob('run-*')), 'Preserve existing extraction; no quality repetition')
    historical = pins(path for folder in ('output', 'doc/results')
                      for path in (ROOT / folder).rglob('*') if path.is_file())
    protected = pins(sources() + input_paths + [EVIDENCE / 'admission-review.json'])
    run = Path(tempfile.mkdtemp(prefix='run-', dir=PARENT))
    dump(run / 'historical-before.json', historical)
    dump(run / 'sources-before.json', protected)
    print('LOCATION_SCALE_TRANSFER_RUN=' + str(run), flush=True)
    started = time.monotonic()
    code = command([BUILD / NAMES[0], '--inputs', INPUTS, '--output', run / 'features',
                    '--card', CARD, '--checkpoint', CHECKPOINT, '--settings', SETTINGS], run / 'extract.log')
    extraction_seconds = time.monotonic() - started
    verify(protected)
    verify(historical)
    require(code == 0, 'Extraction failed; preserve attempt, no quality retry')
    features = pins(path for path in (run / 'features').rglob('*') if path.is_file())
    dump(run / 'features-before-heads.json', features)
    print('Frozen extraction passed; fixed saved-feature classifiers next.', flush=True)
    started = time.monotonic()
    code = command([BUILD / NAMES[1], '--features', run / 'features', '--output', run / 'readouts'], run / 'readout.log')
    head_seconds = time.monotonic() - started
    verify(features)
    verify(protected)
    verify(historical)
    require(code == 0, 'Fixed readout failed; preserve all attempts and fitted files')
    started = time.monotonic()
    code = command(['python3', AUDITOR, '--run', run, '--output', run / 'independent-audit.json'], run / 'audit.log')
    audit_seconds = time.monotonic() - started
    verify(features)
    verify(protected)
    verify(historical)
    verify(admission['protected'])
    require(code == 0 and load(run / 'independent-audit.json')['status'] == 'passed', 'Independent arithmetic audit failed')
    dump(run / 'verification.json', {
        'kind': 'market_location_scale_synthetic_transfer_execution_v1', 'status': 'passed',
        'checkpoint_sha256': EXPECTED[CHECKPOINT], 'card_sha256': EXPECTED[CARD],
        'historical_files_unchanged': len(historical), 'source_files_unchanged': len(protected),
        'feature_files_unchanged': len(features), 'admission_sha256': sha(EVIDENCE / 'admission-review.json'),
        'extraction_wall_seconds_including_provenance_correctness_and_io': extraction_seconds,
        'fixed_head_fitting_scoring_and_io_wall_seconds': head_seconds,
        'saved_arithmetic_audit_wall_seconds': audit_seconds,
        'encoder_updates': 0, 'old_encoder_calls': 0, 'old_head_fits': 0,
        'new_full_cuda_encode_calls': 768, 'new_same_checkpoint_parent_correctness_calls': 6,
        'bank_calibration_fits': 0, 'decoder_calls': 0,
        'broad_preservation_scan_pending_goal_closure': True})
    print('LOCATION_SCALE_FROZEN_TRANSFER_COMPLETE=' + str(run), flush=True)


def main():
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Use existing cuwacunu container')
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('build', 'seal', 'run'))
    args = parser.parse_args()
    {'build': build, 'seal': seal, 'run': execute}[args.mode]()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Run one declared frozen transfer extraction and saved-feature evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path('/cuwacunu')
BUILD = Path('/opt/cuwacunu/build/market-transfer')
INPUTS = ROOT / 'output/inputs/market-synthetic-transfer-v1'
CHECKPOINT = ROOT / 'output/runs/market-separate-changes-native36/run-4SauBf/checkpoints/inference-2048.pt'
SETTINGS = ROOT / 'config/market_studies/encoder_separate_native36.conf'
CARD = ROOT / 'doc/MARKET_SYNTHETIC_TRANSFER_CARD.md'
EXPECTED = {
    CHECKPOINT: 'a3939b13df8b1704fc3621876cf3adc6efaddd98951147ff43ec6408978fa98c',
    SETTINGS: '52827353ff7dd327e61671d6d2ae1becfdc140671a5d84f11827bc0173fddc1d',
    CARD: 'b0552d60cd50582a3bff213a2778e867ceabe1e20e52cd48e02aec4c34bea82b',
    INPUTS / 'manifest.json': 'fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def pins(paths):
    return {str(path): {'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in sorted(set(paths))}


def verify(records):
    for name, record in records.items():
        path = Path(name)
        assert path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name


def execute(args, log):
    with log.open('x', encoding='utf-8') as stream:
        subprocess.run([str(arg) for arg in args], stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    assert Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, str(path)
    manifest = json.loads((INPUTS / 'manifest.json').read_text())
    assert len(manifest['members']) == 18
    for member in manifest['members']:
        path = INPUTS / member['path']
        assert path.resolve().is_relative_to(INPUTS) and path.stat().st_size == member['bytes'] and sha(path) == member['sha256'], str(path)
    files = list(EXPECTED) + [ROOT / 'Makefile', ROOT / 'code/scripts/task.sh', Path(__file__).resolve(),
                             BUILD / 'market_synthetic_transfer_extract', BUILD / 'market_synthetic_transfer_read']
    for directory in ['code/evaluation/market_synthetic_transfer_v1', 'code/config/include',
                      'code/encoders/rpb_v19/include', 'code/encoders/rpb_spectral/include',
                      'code/encoders/rpb_content/include', 'code/encoders/rpb_changes/include',
                      'code/encoders/rpb_separate/include',
                      'code/encoders/raw_patch_bottleneck_mae/shared/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v13/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v14/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v16/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v17/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v18/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v19/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v20/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v22/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v23/include',
                      'code/encoders/raw_patch_bottleneck_mae/versions/v24/include']:
        files.extend(path for path in (ROOT / directory).rglob('*') if path.is_file())
    files.extend(INPUTS / member['path'] for member in manifest['members'])
    before = pins(files)
    parent = ROOT / 'output/runs/market-synthetic-transfer'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    write(run / 'provenance-before.json', before)
    print('TRANSFER_RUN_DIRECTORY=' + str(run), flush=True)
    started = time.monotonic()
    execute([BUILD / 'market_synthetic_transfer_extract', '--inputs', INPUTS, '--output', run / 'features',
             '--card', CARD, '--checkpoint', CHECKPOINT, '--settings', SETTINGS], run / 'extract.log')
    extraction_seconds = time.monotonic() - started
    verify(before)
    feature_pins = pins(path for path in (run / 'features').rglob('*') if path.is_file())
    write(run / 'features-before-readout.json', feature_pins)
    print('Frozen GPU extraction passed; fitting fixed saved-feature heads.', flush=True)
    started = time.monotonic()
    execute([BUILD / 'market_synthetic_transfer_read', '--features', run / 'features', '--output', run / 'readouts'], run / 'readout.log')
    readout_seconds = time.monotonic() - started
    verify(before)
    verify(feature_pins)
    write(run / 'verification.json', {
        'kind': 'market_synthetic_transfer_execution_v1', 'status': 'passed',
        'files_unchanged': len(before), 'saved_feature_files_unchanged': len(feature_pins),
        'encoder_updates': 0, 'old_encoder_calls': 0, 'old_head_fits': 0,
        'extraction_wall_seconds_including_provenance_and_io': extraction_seconds,
        'saved_feature_readout_wall_seconds_including_scoring_and_io': readout_seconds,
        'checkpoint_sha256': EXPECTED[CHECKPOINT], 'card_sha256': EXPECTED[CARD],
        'staged_manifest_sha256': EXPECTED[INPUTS / 'manifest.json'],
        'combined_provenance_sha256': sha(run / 'provenance-before.json'),
    })
    print('FROZEN_TRANSFER_COMPLETE=' + str(run), flush=True)


if __name__ == '__main__':
    main()

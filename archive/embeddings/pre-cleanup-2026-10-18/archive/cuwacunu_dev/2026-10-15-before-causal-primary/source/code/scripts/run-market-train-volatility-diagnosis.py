#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Run saved TRAIN arithmetic once; no encoder, solver, fit or held-out arrays."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path('/cuwacunu')
CARD = ROOT / 'doc/MARKET_TRAIN_VOLATILITY_DIAGNOSIS_CARD.md'
MANIFEST = ROOT / 'doc/results/market_train_volatility_diagnosis_v1/input-manifest.json'
BINARY = Path('/opt/cuwacunu/build/market-train-diagnosis/market_train_volatility_diagnosis')
EXPECTED_CARD = '89a2005b87cf714a53da7aa4ed65a15da477712aca49a36ec159c75fd9de5a87'
EXPECTED_MANIFEST = '4630c3f16a3ce6201105983733b31faa446bf22e03a123a79a8c38cbb664e8df'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def dump(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(records):
    for name, expected in records.items():
        require(record(Path(name)) == expected, 'Protected file changed: ' + name)


def main():
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT,
            'Run in the existing managed cuwacunu container')
    require(sha(CARD) == EXPECTED_CARD and sha(MANIFEST) == EXPECTED_MANIFEST,
            'Prospective card/input manifest differs')
    manifest = json.loads(MANIFEST.read_text())
    require(manifest['card_sha256'] == EXPECTED_CARD and len(manifest['members']) == 22,
            'Input declaration differs')
    for member in manifest['members']:
        path = ROOT / member['path']
        require(path.resolve().is_relative_to(ROOT / 'output/runs'), 'Unsafe input path')
        require('validation' not in member['path'].lower() and 'test' not in member['path'].lower(),
                'Non-TRAIN diagnostic input')
        require(record(path) == {key: member[key] for key in ['sha256', 'bytes']},
                'Declared input differs: ' + member['path'])
    historical = {
        str(path): record(path)
        for directory in ['output', 'doc/results']
        for path in sorted((ROOT / directory).rglob('*')) if path.is_file()
    }
    handoff = json.loads((ROOT / 'doc/results/separate_change_native36_v1/preservation-before.json').read_text())
    for name, expected in handoff['handoff_checkpoints'].items():
        require(sha(ROOT / name) == expected, 'Original handoff checkpoint changed: ' + name)
    completion = json.loads((ROOT / 'doc/results/separate_change_native36_v1/completion.json').read_text())
    checkpoints = [entry for entry in completion['source_assets'] if '/checkpoints/' in entry['source']]
    require(len(checkpoints) == 5, 'Native36 checkpoint inventory differs')
    for entry in checkpoints:
        require(record(ROOT / entry['source']) == {'sha256': entry['sha256'], 'bytes': entry['bytes']},
                'Native36 checkpoint changed: ' + entry['source'])
    sources = [CARD, MANIFEST, Path(__file__).resolve(), BINARY, ROOT / 'Makefile',
               ROOT / 'code/scripts/task.sh', ROOT / 'code/evaluation/src/fixed_regression.cpp',
               ROOT / 'code/scripts/audit-market-train-volatility.py']
    sources.extend(path for path in (ROOT / 'code/evaluation/market_train_volatility_diagnosis_v1').rglob('*')
                   if path.is_file())
    protected = {str(path): record(path) for path in sources}
    parent = ROOT / 'output/runs/market-train-volatility-diagnosis'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    dump(run / 'historical-preservation-before.json', historical)
    dump(run / 'provenance-before.json', protected)
    print('TRAIN_DIAGNOSIS_RUN_DIRECTORY=' + str(run), flush=True)
    started = time.monotonic()
    with (run / 'reader.log').open('x') as log:
        subprocess.run([str(BINARY), str(run / 'reader')], check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    wall_seconds = time.monotonic() - started
    verify(protected)
    verify(historical)
    dump(run / 'historical-preservation-after.json', {
        'status': 'passed', 'files_unchanged': len(historical),
        'original_handoff_checkpoints_unchanged': 4, 'native36_checkpoints_unchanged': 5,
        'baseline_sha256': sha(run / 'historical-preservation-before.json'),
    })
    dump(run / 'verification.json', {
        'protocol': manifest['protocol'], 'status': 'reader_completed_preservation_passed_audit_pending',
        'diagnostic_input_archives': 22, 'encoder_calls': 0, 'encoder_updates': 0,
        'head_fits': 0, 'scaler_fits': 0, 'held_out_tensor_archives': 0,
        'card_sha256': EXPECTED_CARD, 'input_manifest_sha256': EXPECTED_MANIFEST,
        'reader_wall_seconds_including_io': wall_seconds,
        'protected_source_binary_files': len(protected), 'historical_files_unchanged': len(historical),
        'binary_sha256': sha(BINARY), 'provenance_sha256': sha(run / 'provenance-before.json'),
    })
    print('TRAIN_DIAGNOSIS_READER_COMPLETE=' + str(run), flush=True)


if __name__ == '__main__':
    main()

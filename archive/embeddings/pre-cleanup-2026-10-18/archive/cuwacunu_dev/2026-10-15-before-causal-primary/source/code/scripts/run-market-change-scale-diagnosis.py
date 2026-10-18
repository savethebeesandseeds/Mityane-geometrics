#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Execute one declared saved-TRAIN diagnosis; no encoder or fitting calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path('/cuwacunu')
EVIDENCE = ROOT / 'doc/results/market_change_scale_diagnosis_v1'
PINS = {
    ROOT / 'doc/MARKET_CHANGE_SCALE_DIAGNOSIS_CARD.md':
        '848c5dbf3b6dbce0450c4287c3bd30d0d0b7978ea5571a925d86f492e11d0260',
    ROOT / 'doc/MARKET_CHANGE_SCALE_DIAGNOSIS_ADDENDUM.md':
        '19a3dc4455a79d40ebd7e6086be1ad44d2d077a26763d2f3826557c3b789fa8c',
    EVIDENCE / 'input-manifest.json':
        '241625e6815fa32175f082e4ffc73d460eec9b050de3d2f750649186b10488e3',
    EVIDENCE / 'preflight.json':
        '178ecb1e7a34eba32ec7db59e214779ecb939c30b9af2d20e9fd41397da94873',
}
BINARY = Path('/opt/cuwacunu/build/market-scale-diagnosis/market_change_scale_diagnosis')
AUDITOR = ROOT / 'code/scripts/audit-market-change-scale.py'


def require(value, why):
    if not value:
        raise RuntimeError(why)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return dict(sha256=sha(path), bytes=path.stat().st_size)


def dump(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(records):
    for name, expected in records.items():
        require(record(Path(name)) == expected, 'Protected file changed: ' + name)


def main():
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT,
            'Use the existing managed cuwacunu container')
    for path, expected in PINS.items():
        require(sha(path) == expected, 'Prospective declaration changed: ' + str(path))
    manifest = json.loads((EVIDENCE / 'input-manifest.json').read_text())
    preflight = json.loads((EVIDENCE / 'preflight.json').read_text())
    require(len(manifest['members']) == 12 and manifest['training_rows'] == 112,
            'Exact TRAIN allowlist required')
    require(preflight['all_checkpoints_available_hash_matched_and_ignored'] and
            preflight['all_pairs_match_prior54_expected_hashes_and_sizes'], 'Preflight incomplete')
    inputs = {}
    for member in manifest['members']:
        path = ROOT / member['path']
        require(path.resolve().is_relative_to(ROOT / 'output/runs') and
                not any(word in member['path'].lower() for word in ('validation', 'test')),
                'Only declared TRAIN inputs permitted')
        expected = {key: member[key] for key in ('sha256', 'bytes')}
        require(record(path) == expected, 'Declared input changed: ' + str(path))
        require(str(path) not in inputs, 'Duplicate diagnostic input')
        inputs[str(path)] = expected
    sources = list(PINS) + [Path(__file__).resolve(), BINARY, AUDITOR,
                           ROOT / 'Makefile', ROOT / 'code/scripts/task.sh']
    sources += [p for p in (ROOT / 'code/evaluation/market_change_scale_diagnosis_v1').rglob('*')
                if p.is_file()]
    protected = {str(path): record(path) for path in sources}
    # Recheck existing integration evidence byte for byte. The previous complete
    # two-repository 77,831-file scan remains bound in preflight; no embedding
    # process is started for this read-only calculation.
    historical = {str(path): record(path)
                  for folder in ('output', 'doc/results')
                  for path in sorted((ROOT / folder).rglob('*')) if path.is_file()}
    parent = ROOT / 'output/runs/market-change-scale-diagnosis'
    parent.mkdir(parents=True, exist_ok=True)
    for prior in parent.glob('run-*'):
        require(not (prior / 'verification.json').exists(),
                'Use the already completed saved diagnosis; do not repeat it')
    run = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    dump(run / 'historical-preservation-before.json', historical)
    dump(run / 'provenance-before.json', protected)
    dump(run / 'input-manifest.json', manifest)
    print('SCALE_DIAGNOSIS_RUN_DIRECTORY=' + str(run), flush=True)
    started = time.monotonic()
    with (run / 'reader.log').open('x') as stream:
        subprocess.run([str(BINARY), str(run / 'reader')], check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
    reader_seconds = time.monotonic() - started
    verify(inputs)
    verify(protected)
    with (run / 'audit.log').open('x') as stream:
        subprocess.run(['python3', str(AUDITOR), str(run)], check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
    audit = json.loads((run / 'independent-audit.json').read_text())
    require(audit['status'] == 'passed', 'Independent arithmetic audit incomplete')
    verify(inputs)
    verify(protected)
    verify(historical)
    dump(run / 'historical-preservation-after.json', dict(
        status='passed', integration_files_unchanged=len(historical),
        baseline_sha256=sha(run / 'historical-preservation-before.json'),
        preceding_two_repository_complete_scan_bound_by_preflight=True))
    dump(run / 'verification.json', dict(
        kind='market_change_scale_diagnosis_verification_v1', status='passed',
        protocol=manifest['protocol'], dataset_id='MKT-HIST-2', training_rows=112,
        diagnostic_input_archives=12, held_out_tensor_inputs=0,
        encoder_calls=0, encoder_updates=0, head_fits=0, scaler_fits=0,
        reader_seconds_including_io=reader_seconds,
        reader_and_audit_seconds_including_io=time.monotonic() - started,
        card_sha256=PINS[ROOT / 'doc/MARKET_CHANGE_SCALE_DIAGNOSIS_CARD.md'],
        addendum_sha256=PINS[ROOT / 'doc/MARKET_CHANGE_SCALE_DIAGNOSIS_ADDENDUM.md'],
        input_manifest_sha256=PINS[EVIDENCE / 'input-manifest.json'],
        binary_sha256=sha(BINARY), auditor_sha256=sha(AUDITOR),
        provenance_sha256=sha(run / 'provenance-before.json'),
        independent_audit_sha256=sha(run / 'independent-audit.json'),
        independent_arithmetic_checks=audit['checks'],
        historical_integration_files_unchanged=len(historical),
        checkpoint_preflight_count=14, portable_pairs=54))
    print('SCALE_DIAGNOSIS_COMPLETE=' + str(run), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""One admitted v28 trajectory, with immutable historical/source checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/cuwacunu')
EVIDENCE = ROOT / 'doc/results/market_relative_changes_v1'
PINS = {
    ROOT / 'doc/MARKET_RELATIVE_CHANGE_CARD.md':
        'f4378d4a76185fc17b6c0adbadab5af27accd772ed66e5a8bc628ab7848f94a2',
    EVIDENCE / 'preflight.json':
        '554b471b99410cdf617a20851898de1cc164e34ad4fde258da48113db17c1219',
}
FIXED = Path('/opt/cuwacunu/build/market-study/market_fixed_regression')
AUDITOR = ROOT / 'code/scripts/audit-market-relative-changes.py'


def require(value, message):
    if not value:
        raise RuntimeError(message)


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
    require(sha(FIXED) == '5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14',
            'Unchanged fixed evaluator required')
    preflight = json.loads((EVIDENCE / 'preflight.json').read_text())
    require(preflight['status'] == 'passed' and
            preflight['all_checkpoints_available_hash_matched_and_ignored'] and
            preflight['all_prior_pairs_unchanged'], 'Historical preflight incomplete')
    for pin in preflight['checkpoints'] + preflight['frozen_records']:
        require(record(ROOT / pin['path']) == {k: pin[k] for k in ('sha256', 'bytes')},
                'Frozen availability/hash mismatch: ' + pin['path'])
    admission_path = EVIDENCE / 'admission-review.json'
    admission = json.loads(admission_path.read_text())
    require(admission['status'] == 'passed' and admission['protocol'] ==
            'market-relative-changes-v1' and admission['quality_started'] is False,
            'Prospective artificial CUDA admission required')
    for pin in admission['protected_sources'] + admission['fixture_assets']:
        require(record(Path(pin['path'])) == {k: pin[k] for k in ('sha256', 'bytes')},
                'Admitted source/fixture changed: ' + pin['path'])
    parent = ROOT / 'output/runs/market-relative-changes'
    parent.mkdir(parents=True, exist_ok=True)
    require(not list(parent.glob('run-*')), 'Preserve existing trajectory; do not repeat')
    # Hash preexisting evidence before creating any new run/wrapper files.
    historical = {str(path): record(path)
                  for folder in ('output', 'doc/results')
                  for path in sorted((ROOT / folder).rglob('*')) if path.is_file()}
    sources = list(PINS) + [Path(__file__).resolve(), AUDITOR, FIXED, admission_path]
    sources += [path for folder in ('code', 'config')
                for path in sorted((ROOT / folder).rglob('*')) if path.is_file()]
    protected = {str(path): record(path) for path in sources}
    wrapper = parent / 'execution-preservation'
    wrapper.mkdir()
    dump(wrapper / 'historical-before.json', historical)
    dump(wrapper / 'sources-before.json', protected)
    print('RELATIVE_EXECUTION_DIRECTORY=' + str(wrapper), flush=True)
    started = time.monotonic()
    with (wrapper / 'execution.log').open('x') as stream:
        result = subprocess.run(['bash', str(ROOT / 'code/scripts/run-market-relative-changes.sh')],
                                stdout=stream, stderr=subprocess.STDOUT)
    # Preserve source/history checks even when an engineering/quality process fails.
    verify(protected)
    verify(historical)
    dump(wrapper / 'historical-after.json', dict(status='passed',
        integration_files_unchanged=len(historical),
        prior_full_two_repository_scan_bound_not_repeated=True,
        baseline_sha256=sha(wrapper / 'historical-before.json')))
    require(result.returncode == 0, 'Candidate execution failed; preserve every attempt/log')
    runs = list(parent.glob('run-*'))
    require(len(runs) == 1 and (runs[0] / 'summary.json').is_file() and
            (runs[0] / 'readouts/report.json').is_file(), 'One completed candidate required')
    run = runs[0]
    with (run / 'audit.log').open('x') as stream:
        subprocess.run(['python3', str(AUDITOR), str(run)], check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
    audit = json.loads((run / 'independent-audit.json').read_text())
    require(audit['status'] == 'passed', 'Independent arithmetic verification required')
    verify(protected)
    verify(historical)
    dump(run / 'verification.json', dict(
        kind='market_relative_changes_verification_v1', status='passed',
        protocol='market-relative-changes-v1', dataset_id='MKT-HIST-2',
        instance='RPB-v28.market-01', card_sha256=PINS[ROOT / 'doc/MARKET_RELATIVE_CHANGE_CARD.md'],
        preflight_sha256=PINS[EVIDENCE / 'preflight.json'],
        admission_sha256=sha(admission_path), independent_audit_sha256=sha(run / 'independent-audit.json'),
        independent_arithmetic_checks=audit['checks'],
        historical_integration_files_unchanged=len(historical),
        sources_before_sha256=sha(wrapper / 'sources-before.json'),
        historical_before_sha256=sha(wrapper / 'historical-before.json'),
        historical_after_sha256=sha(wrapper / 'historical-after.json'),
        existing_checkpoint_pins_verified=preflight['checkpoints_verified'],
        preceding_complete_two_repository_scan_files=77831,
        broad_scan_repeated=False, one_new_training_trajectory=True,
        total_wrapper_seconds_including_build_io_and_head_fitting=time.monotonic() - started))
    print('RELATIVE_COMPLETE=' + str(run), flush=True)


if __name__ == '__main__':
    main()

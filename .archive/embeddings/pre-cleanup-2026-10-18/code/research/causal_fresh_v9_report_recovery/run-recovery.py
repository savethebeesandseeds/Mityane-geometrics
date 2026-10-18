#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Once-only saved-report aggregation recovery; never launches scientific stages."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/opt/cuwacunu/research/causal-fresh-v9-report-recovery/attempt-001')
MAIN = Path('/opt/cuwacunu/research/causal-fresh-v9/attempt-001')
SCIENCE = ('extraction', 'initial_market_heads', 'initial_synthetic_heads',
           'final_market_heads', 'final_synthetic_heads')
IDENTITY = dict(card_sha256='94488bf2d42751b5a38c054eb5b6bbd7674e45d6e0bd444818b7f6fbcb34346d',
    manifest_sha256='43a44dbba0f7b885258766ab5741c9e806cc37ff0817237954b31d09de940750',
    archive_sha256='b54ffb00a1489dc4973efe7d93819196b4e5bcd445140851d977ba14732aa867',
    source_files=118, binary_count=7, translation_units=11)


def pin(path):
    path = Path(path)
    assert not path.is_symlink() and path.is_file()
    data = path.read_bytes()
    return dict(path=str(path.resolve(strict=True)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def bound(value):
    actual = pin(value['path'])
    assert all(actual[key] == value[key] for key in ('bytes', 'sha256'))
    return Path(actual['path'])


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=Path, required=True)
    parser.add_argument('--card-sha256', required=True)
    args = parser.parse_args()
    assert Path('/.dockerenv').is_file() and ROOT.is_dir() and not ROOT.is_symlink()
    assert pin(args.card)['sha256'] == args.card_sha256
    card = load(args.card)
    assert card['kind'] == 'cpb_v9_report_only_recovery_card_v1' and card['root_reviewed'] is True
    assert card['main_source_identity'] == IDENTITY
    assert bound(card['runner']) == Path(__file__).resolve(strict=True)
    assert pin(bound(card['main_card']))['sha256'] == IDENTITY['card_sha256']
    assert pin(bound(card['source_manifest']))['sha256'] == IDENTITY['manifest_sha256']
    assert pin(bound(card['source_archive']))['sha256'] == IDENTITY['archive_sha256']
    failure = load(bound(card['failure_mirror']))
    assert card['failure_mirror']['sha256'] == 'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532'
    assert failure['status'] == 'passed' and failure['failure_preserved'] and failure['source_files_verified'] == 118
    manifest = load(bound(card['source_manifest']))
    assert len(manifest['files']) == 118
    for item in manifest['files']:
        bound(dict(item, path=str(MAIN/'source'/item['path'])))
    failed_quality = load(bound(card['failed_quality_stage']))
    failed_report = load(bound(card['failed_report_stage']))
    assert failed_quality['status'] == 'failed' and failed_quality['exit_code'] == 1
    assert failed_report['stage'] == 'report' and failed_report['exit_code'] == 1
    assert set(card['scientific_stages']) == set(SCIENCE)
    for name in SCIENCE:
        value = load(bound(card['scientific_stages'][name]))
        assert value['stage'] == name and value['exit_code'] == 0
    quality = MAIN/'quality-001'
    assert card['outputs'] == dict(report=str(quality/'report'), comparison=str(quality/'comparison'))
    assert not (quality/'report').exists() and not (quality/'comparison').exists()
    assert len(card['scientific_inputs']) == len(failure['files'])
    assert {tuple(item[key] for key in ('path','bytes','sha256')) for item in card['scientific_inputs']} == {
        (item['container_path'],item['bytes'],item['sha256']) for item in failure['files']}
    before = [pin(bound(value)) for value in card['scientific_inputs']]
    report_source = bound(card['reporter'])
    comparator = bound(card['comparator'])
    assert comparator == MAIN/'source/code/research/causal_fresh_v9/evaluation/compare_baseline.py'
    bound(card['original_report'])
    output = ROOT/'run-001'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    def run(name, command):
        assert name in ('report', 'comparison')
        start = time.monotonic()
        with (output/(name+'.stdout')).open('xb') as stdout, (output/(name+'.stderr')).open('xb') as stderr:
            done = subprocess.run(command, stdout=stdout, stderr=stderr,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), check=False)
        stage = dict(kind='cpb_v9_report_only_recovery_stage_v1', stage=name,
            status='passed' if done.returncode == 0 else 'failed', exit_code=done.returncode,
            argv=command, elapsed_seconds=time.monotonic()-start,
            stdout=pin(output/(name+'.stdout')), stderr=pin(output/(name+'.stderr')))
        write(output/(name+'-stage.json'), stage)
        assert done.returncode == 0, 'Preserve failed reporting recovery; no implicit retry'
        return pin(output/(name+'-stage.json'))
    with Path('/opt/cuwacunu/research/cpb-v9-report-only-recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        report_stage = run('report', [sys.executable, str(report_source), '--run', str(quality),
            '--training', str(MAIN/'run-001'), '--card', str(bound(card['main_card'])),
            '--input-manifest', str(bound(card['input_manifest'])), '--raw-controls', str(bound(card['raw_controls'])),
            '--quality-receipt', str(bound(card['quality_receipt'])), '--output', str(quality/'report'),
            '--recovery-card', str(args.card), '--recovery-card-sha256', args.card_sha256])
        comparison_stage = run('comparison', [sys.executable, str(comparator), '--card', str(bound(card['main_card'])),
            '--candidate-table', str(quality/'report/table_rows.json'), '--source', str(MAIN/'source'),
            '--output', str(quality/'comparison')])
    after = [pin(bound(value)) for value in card['scientific_inputs']]
    assert before == after
    for item in manifest['files']:
        bound(dict(item, path=str(MAIN/'source'/item['path'])))
    result = dict(kind='cpb_v9_report_only_recovery_receipt_v1', status='passed',
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        card=pin(args.card), main_source_identity=IDENTITY, failure_mirror=card['failure_mirror'],
        failed_quality_stage=card['failed_quality_stage'], failed_report_stage=card['failed_report_stage'],
        original_quality_stage_status='failed', scientific_stages=card['scientific_stages'],
        completed_scientific_stages=5, scientific_inputs_before=before, scientific_inputs_after=after,
        scientific_inputs_before_after_exact=True, frozen_scientific_source_unchanged=True,
        report=dict(stage=report_stage, output_pins=[pin(quality/'report'/name) for name in
            ('completion.json','table_rows.json','asset_hashes.json','RESULTS.md')]),
        comparison=dict(stage=comparison_stage, output_pins=[pin(quality/'comparison'/name) for name in
            ('comparison.json','report.md')]),
        reporter=pin(report_source), comparator=pin(comparator), elapsed_seconds=time.monotonic()-started,
        encoder_calls=0, head_fits=0, training_updates=0, science_stages_replayed=0)
    write(output/'recovery.json', result)
    print(json.dumps(dict(status='passed', receipt=pin(output/'recovery.json'), elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()

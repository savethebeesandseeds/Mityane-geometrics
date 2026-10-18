#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Owned V8-derived once-only saved training audit recorder; never imports models."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def pin(path):
    path = Path(path)
    assert not path.is_symlink() and path.is_file()
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def same(path, expected):
    actual = pin(path)
    assert actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256']
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--kind', choices=('training',), required=True)
    parser.add_argument('--execute-once', action='store_true', required=True)
    for name in ('audit-card-sha256', 'stage-receipt-sha256', 'manifest-sha256', 'zip-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    assert args.execute_once and Path('/.dockerenv').is_file()
    assert all(re.fullmatch('[0-9a-f]{64}', value) for value in (
        args.audit_card_sha256, args.stage_receipt_sha256, args.manifest_sha256, args.zip_sha256))
    root = args.root.resolve(strict=True)
    assert str(root) == '/opt/cuwacunu/research/causal-fresh-v9-training-audit/attempt-001'
    source = root / 'source'
    # Every externally supplied transport binding precedes JSON parsing or body import.
    for path, digest in ((root/'source-stage.json', args.stage_receipt_sha256),
                         (source/'audit-card.json', args.audit_card_sha256),
                         (source/'SOURCE_MANIFEST.json', args.manifest_sha256),
                         (root/'source.zip', args.zip_sha256)):
        assert pin(path)['sha256'] == digest, 'External transport hash before JSON parse: ' + str(path)
    stage = read(root/'source-stage.json')
    assert stage['kind'] == 'cpb_v9_training_audit_source_stage_v1' and stage['status'] == 'passed'
    assert stage['numeric_archive_reads'] == 0 and stage['source_root'] == str(source)
    assert stage['source_manifest_sha256'] == args.manifest_sha256
    assert stage['source_zip_sha256'] == args.zip_sha256 and stage['audit_card_sha256'] == args.audit_card_sha256
    assert stage['files'] == 6 and stage['python_AST_files'] == 2
    manifest = read(source/'SOURCE_MANIFEST.json')
    assert manifest['container_root'] == str(root) and manifest['audit_card_sha256'] == args.audit_card_sha256
    entries = manifest['files']
    assert len(entries) == 6 and {entry['file'] for entry in entries} == {
        'audit_training.py', 'README.md', 'provenance.json', 'audit-card.json', 'LICENSE', 'execute-audit.py'}
    assert all(Path(entry['file']).name == entry['file'] for entry in entries)
    before = [same(source/entry['file'], entry) for entry in entries]
    card = read(source/'audit-card.json')
    assert card['kind'] == 'cpb_v9_saved_training_independent_audit_card_v1'
    assert card['stage'] == 'declared_before_independent_saved_data_execution' and card['execution_pending']
    assert card['model'] == 'CPB-v9.market-01' and card['expected_audit_root'] == str(root)
    same(__file__, card['execution_wrapper'])
    main_paths = card['expected_main_paths']
    attempt = Path(main_paths['attempt'])
    assert str(attempt) == '/opt/cuwacunu/research/causal-fresh-v9/attempt-001'
    main_source, training = attempt/'source', attempt/'run-001'
    assert main_paths['source'] == str(main_source) and main_paths['run'] == str(training)
    assert pin(main_source/'CAPSULE_MANIFEST.json')['sha256'] == card['main_source_manifest_sha256']
    assert pin(attempt/'source.zip')['sha256'] == card['main_source_zip_sha256']
    assert pin(main_source/'doc/CAUSAL_FRESH_V9_CARD.json')['sha256'] == card['frozen_training_card_sha256']
    same(attempt/'training-stage.json', card['completed_training_stage'])
    same(training/'training-summary.json', card['completed_training_summary'])
    completed = read(attempt/'training-stage.json')
    summary = read(training/'training-summary.json')
    assert completed['status'] == 'passed' and completed['exit_code'] == 0
    assert summary['status'] == 'passed' and summary['model'] == 'CPB-v9.market-01'
    assert summary['new_causal_updates'] == 2048 and 2048 <= summary['attempted_batches'] <= 4096
    output = root/'run-001'
    assert not output.exists(), 'Preserve any prior audit invocation, including failures'
    output.mkdir()
    command = [sys.executable, str(source/'audit_training.py'), '--run', str(training),
               '--attempt', str(attempt), '--source', str(main_source), '--data-root', '/cuwacunu',
               '--audit-card', str(source/'audit-card.json'),
               '--engineering-proof', str(attempt/'engineering-proof/proof.json'),
               '--gradient-proof', str(attempt/'variance-gradient-proof.json'),
               '--temporal-proof', str(attempt/'temporal-pooling-proof.json'),
               '--output', str(output/'raw-result.json')]
    transport = dict(audit_card=pin(source/'audit-card.json'), source_stage=pin(root/'source-stage.json'),
                     source_manifest=pin(source/'SOURCE_MANIFEST.json'), source_zip=pin(root/'source.zip'))
    write(output/'invocation.json', dict(kind='cpb_v9_independent_training_audit_invocation_v1', argv=command,
          main_stage=pin(attempt/'training-stage.json'), wrapper=pin(__file__), source_before=before,
          external_transport_sha256=vars(args) | {'root':str(root)}, model_calls=0, head_fits=0,
          training_updates=0, **transport))
    lock_path = Path('/opt/cuwacunu/research/cpb-v9-independent-training-audit.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with (output/'stdout.log').open('wb') as stdout, (output/'stderr.log').open('wb') as stderr:
            finished = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, check=False)
        elapsed = time.monotonic() - started
    after = [pin(source/entry['file']) for entry in entries]
    result = read(output/'raw-result.json') if (output/'raw-result.json').is_file() else None
    transport_unchanged = (pin(root/'source-stage.json') == transport['source_stage']
                           and pin(source/'SOURCE_MANIFEST.json') == transport['source_manifest']
                           and pin(root/'source.zip') == transport['source_zip'])
    status = 'passed' if (finished.returncode == 0 and result and result['status'] == 'passed'
                          and before == after and transport_unchanged) else 'failed'
    execution = dict(kind='causal_fresh_v9_training_audit_execution_v1', status=status,
                     exit_code=finished.returncode, argv=command, started_utc=start_utc,
                     elapsed_seconds=elapsed, source_before=before, source_after=after,
                     stdout=pin(output/'stdout.log'), stderr=pin(output/'stderr.log'),
                     wrapper=pin(__file__), transport_unchanged=transport_unchanged,
                     model_calls=0, head_fits=0, training_updates=0, **transport)
    if result is not None:
        execution['raw_result'] = pin(output/'raw-result.json')
    write(output/'execution.json', execution)
    inventory = []
    # Inventory only this audit leaf, never concurrently running main proof/quality directories.
    for file in sorted(root.rglob('*')):
        assert not file.is_symlink()
        if file.is_file():
            inventory.append(dict(relative=file.relative_to(root).as_posix(), **pin(file)))
    write(root/'mirror-inventory.json', dict(kind='cpb_v9_training_audit_complete_file_inventory_v1',
                                           status=status, files=inventory))
    names = ('numeric_statistics', 'saved_tensor_state', 'manual_saved_Gaussian',
             'actual_pre_step_composite_objective', 'temporal_pooling_saved_learning')
    print(json.dumps(dict(status=status, kind='training', elapsed_seconds=elapsed, output=str(output),
                          core_checks=result.get('checks') if result else None,
                          module_checks={name:result[name]['checks'] for name in names} if result else None)))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

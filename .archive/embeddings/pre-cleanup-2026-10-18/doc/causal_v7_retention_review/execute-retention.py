#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Once-only saved-aggregate derivation/review recorder; no producer imports."""
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

ROOT = Path('/opt/cuwacunu/research/causal-v7-retention-review/attempt-001')


def pin(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError('Regular nonlinked metadata file required')
    value = path.read_bytes()
    return dict(path=str(path), bytes=len(value), sha256=hashlib.sha256(value).hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--kind', choices=('derivation', 'review'), required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    assert sys.platform.startswith('linux') and Path('/.dockerenv').is_file() and root == ROOT
    source = root/'source'
    stage, capsule = read(root/'source-stage.json'), read(source/'CAPSULE_MANIFEST.json')
    assert stage['status'] == 'passed_AST_only' and stage['module_executions'] == 0
    assert stage['manifest_sha256'] == pin(source/'CAPSULE_MANIFEST.json')['sha256']
    assert stage['source_zip_sha256'] == pin(root/'source.zip')['sha256']
    assert len(capsule['files']) == 17
    before = []
    for item in capsule['files']:
        relative = Path(item['file'])
        assert not relative.is_absolute() and '..' not in relative.parts
        actual = pin(source/relative)
        assert all(actual[key] == item[key] for key in ('bytes', 'sha256'))
        before.append(actual)
    manifest = read(source/'admitted-manifest.json')
    assert manifest['kind'] == 'clean_causal_v7_retention_derivation_manifest_v1'
    assert manifest['authorized_for_derivation'] is True
    assert manifest['execution_wrapper']['sha256'] == pin(__file__)['sha256']
    assert manifest['root_review']['parent_manifest_sha256'] == pin(source/'manifest.json')['sha256']
    record = root/(args.kind+'-run-001')
    assert not record.exists()
    if args.kind == 'derivation':
        assert not (root/'output-001').exists()
        command = [sys.executable, str(source/'derive.py'), '--manifest', str(source/'admitted-manifest.json'),
                   '--output', str(root/'output-001')]
        result_path = root/'output-001'/'derivation.json'
        expected_status = 'passed_proposal_not_applied'
    else:
        prior = read(root/'derivation-run-001'/'execution.json')
        assert prior['status'] == 'passed' and prior['exit_code'] == 0
        assert prior['source_after'] == before
        command = [sys.executable, str(source/'independent_review.py'), '--manifest', str(source/'admitted-manifest.json'),
                   '--protocol', str(root/'output-001'/'clean_causal_encoder_search_goal_v7_retention.json'),
                   '--output', str(record/'independent-review.json')]
        result_path, expected_status = record/'independent-review.json', 'passed'
    record.mkdir()
    write(record/'invocation.json', dict(argv=command, source_before=before,
          admitted_manifest=pin(source/'admitted-manifest.json'), wrapper=pin(__file__),
          model_calls=0, head_fits=0, training_updates=0))
    with Path('/opt/cuwacunu/research/clean-causal-v7-retention-metadata.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, utc = time.monotonic(), datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (record/'stdout.log').open('wb') as stdout, (record/'stderr.log').open('wb') as stderr:
            finished = subprocess.run(command, stdout=stdout, stderr=stderr,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), check=False)
        elapsed = time.monotonic()-started
    after = [pin(source/item['file']) for item in capsule['files']]
    result = read(result_path) if result_path.is_file() else None
    status = 'passed' if finished.returncode == 0 and result and result['status'] == expected_status and before == after else 'failed'
    execution = dict(kind='clean_causal_v7_retention_'+args.kind+'_execution_v1', status=status,
        exit_code=finished.returncode, argv=command, started_utc=utc, elapsed_seconds=elapsed,
        source_before=before, source_after=after, wrapper=pin(__file__),
        admitted_manifest=pin(source/'admitted-manifest.json'), stdout=pin(record/'stdout.log'),
        stderr=pin(record/'stderr.log'), model_calls=0, head_fits=0, training_updates=0,
        new_quality_scores=0, live_protocol_files_written=0)
    if result is not None:
        execution['result'] = pin(result_path)
    write(record/'execution.json', execution)
    inventory = [dict(relative=p.relative_to(root).as_posix(), **pin(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    write(root/('mirror-inventory-'+args.kind+'.json'),
          dict(kind='clean_causal_v7_retention_inventory_v1', phase=args.kind, status=status, files=inventory))
    print(json.dumps(dict(status=status, elapsed_seconds=elapsed, result=execution.get('result'))))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

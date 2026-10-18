#!/usr/bin/env python3
"""Immutable stdlib-only receipts for the saved-coordinate diagnostic."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

ORIGINAL_MANIFEST = '62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83'
ORIGINAL_SOURCE = '8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1'
PROTOCOL = 'configurable-branch-v1'
BINARY = 'embedding_configurable_branch_readout'


def check(ok, why):
    if not ok:
        raise ValueError(why)


def direct(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
          'direct absolute path required: ' + str(path))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def document(path):
    direct(path)
    check(path.is_file() and path.stat().st_nlink == 1, 'single-link regular receipt required')
    return json.loads(path.read_text())


def freeze(root, original, build):
    for path in (root, original, build):
        direct(path)
    check(sha(original / 'artifacts.sha256') == ORIGINAL_MANIFEST, 'original manifest differs')
    done = document(original / 'complete.json')
    check(done['protocol'] == 'configurable-capacity-v1' and done['status'] == 'complete'
          and done['artifacts_manifest_sha256'] == ORIGINAL_MANIFEST, 'original completion differs')
    original_receipt = document(original / 'freeze/receipt.json')
    check(original_receipt['source_id'] == ORIGINAL_SOURCE, 'original source receipt differs')
    out = root / 'freeze'
    out.mkdir()
    source_list = subprocess.check_output(
        ['make', '-s', 'print-configurable-branch-sources', f'BUILD_DIR={build}'], text=True)
    paths = source_list.splitlines()
    check(paths and paths == sorted(set(paths)), 'closed sorted source list')
    expected = {}
    for name in paths:
        path = Path(name)
        check(not path.is_absolute() and '..' not in path.parts and path.is_file()
              and path.resolve(strict=True) == Path.cwd() / path and not path.is_symlink(), 'direct source file')
        expected[name] = sha(path)
    manifest = ''.join(f'{expected[name]}  {name}\n' for name in paths)
    new(out / 'source-list.txt', source_list)
    new(out / 'sources.sha256', manifest)
    source_id = sha(out / 'sources.sha256')
    compiled = subprocess.check_output([str(build / BINARY), '--source-id'], text=True).strip()
    check(compiled == source_id, 'compiled reader differs from captured source')
    for name in paths:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
        check(sha(destination) == expected[name] == sha(Path(name)), 'source changed during capture')
    shutil.copyfile('code/evaluation/cards/configurable_branch_v1.md', out / 'card.md')
    shutil.copyfile(original / 'artifacts.sha256', out / 'original-artifacts.sha256')
    new(out / 'binaries.sha256', f'{sha(build / BINARY)}  {build / BINARY}\n')
    binding = dict(protocol=PROTOCOL, original_capsule=str(original),
                   original_source_fingerprint=ORIGINAL_SOURCE,
                   original_artifacts_manifest_sha256=ORIGINAL_MANIFEST,
                   original_card_sha256=original_receipt['card_sha256'],
                   original_config_sha256=original_receipt['config_sha256'])
    new(out / 'original-evidence.json', json.dumps(binding, indent=2) + '\n')
    receipt = dict(binding, source_fingerprint=source_id, card_sha256=sha(out / 'card.md'),
                   source_files=len(paths), build_directory=str(build),
                   git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    new(out / 'receipt.json', json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


def finish(root):
    direct(root)
    check(not (root / 'complete.json').exists() and not (root / 'artifacts.sha256').exists(),
          'completed diagnostic cannot be finalized again')
    receipt = document(root / 'freeze/receipt.json')
    reader = document(root / 'readouts/complete.json')
    report = document(root / 'readouts/report.json')
    summary = document(root / 'published/summary.json')
    check(reader['protocol'] == PROTOCOL and reader['status'] == 'complete'
          and reader['encoder_calls'] == 0 and reader['head_fitted_pairs'] == 27
          and reader['saved_arithmetic_fits_replayed'] == 27
          and reader['report_sha256'] == sha(root / 'readouts/report.json')
          and reader['arithmetic_audit_sha256'] == sha(root / 'readouts/arithmetic-audit.json'),
          'reader completion or artifact binding differs')
    # The independent publisher carries the final audited fit/scope counters.
    check(summary['protocol'] == PROTOCOL and summary['encoder_calls'] == 0
          and summary['encoder_updates'] == 0 and summary['new_head_fit_pairs'] == 27
          and summary['new_classifiers'] == 54,
          'audited diagnostic scope differs')
    check(summary['source_fingerprint'] == receipt['source_fingerprint']
          and summary['card_sha256'] == receipt['card_sha256']
          and report['source_fingerprint'] == receipt['source_fingerprint']
          and report['card_sha'] == receipt['card_sha256']
          and summary['original_source_fingerprint'] == ORIGINAL_SOURCE
          and summary['original_artifact_manifest_sha'] == ORIGINAL_MANIFEST
          and summary['branch_report_sha'] == sha(root / 'readouts/report.json'),
          'published original/source/card/report binding differs')
    check(sha(root / 'freeze/sources.sha256') == receipt['source_fingerprint']
          and sha(root / 'freeze/card.md') == receipt['card_sha256']
          and sha(root / 'freeze/original-artifacts.sha256') == ORIGINAL_MANIFEST,
          'captured evidence differs from freeze')
    for name, count in (('original-before-diagnostic.log', 815),
                        ('original-after-diagnostic.log', 815),
                        ('source-after-diagnostic.log', receipt['source_files']),
                        ('binary-after-diagnostic.log', 1)):
        lines = (root / name).read_text().splitlines()
        check(len(lines) == count and all(line.endswith(': OK') for line in lines),
              'incomplete checksum verification: ' + name)
    engineering = document(root / 'engineering/complete.json')
    check(engineering['status'] == 'passed' and engineering['fitted_pairs'] == 9
          and engineering['encoder_calls'] == 0, 'artificial engineering scope differs')
    times = {}
    for stage in ('audit-test', 'audit-test-optimized', 'engineering', 'readout', 'arithmetic-audit', 'publish'):
        seconds = float((root / (stage + '.seconds')).read_text())
        check(math.isfinite(seconds) and seconds >= 0, 'finite stage wall time')
        times[stage] = seconds
    new(root / 'stage-timings.json', json.dumps(dict(unit='wall seconds', stages=times,
        quality_head_fits=27, artificial_engineering_head_fits=9,
        encoder_calls=0, encoder_updates=0), indent=2) + '\n')
    paths = sorted(path for path in root.rglob('*') if path.is_file())
    for path in paths:
        direct(path)
        check(path.stat().st_nlink == 1, 'single-link retained diagnostic artifact')
    new(root / 'artifacts.sha256', ''.join(f'{sha(path)}  {path.relative_to(root)}\n' for path in paths))
    new(root / 'complete.json', json.dumps(dict(protocol=PROTOCOL, status='complete',
        new_head_fit_pairs=27, new_classifiers=54, encoder_calls=0, encoder_updates=0,
        artificial_engineering_head_fit_pairs=9, artificial_engineering_classifiers=18,
        original_artifacts_manifest_sha256=ORIGINAL_MANIFEST,
        artifacts_manifest_sha256=sha(root / 'artifacts.sha256')), indent=2) + '\n')
    print('BRANCH_DIAGNOSTIC_COMPLETE', root)


def main():
    check(Path('/.dockerenv').exists(), 'use managed development container')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'finish'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--build', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        check(args.original is not None and args.build is not None, 'freeze requires original/build')
        freeze(args.root, args.original, args.build)
    else:
        finish(args.root)


if __name__ == '__main__':
    main()

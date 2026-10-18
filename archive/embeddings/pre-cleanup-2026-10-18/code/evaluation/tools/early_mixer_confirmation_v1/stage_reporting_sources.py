#!/usr/bin/env python3
"""Materialize exact blocked reporting SOURCE definitions; never run them."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile

sys.dont_write_bytecode = True
SOURCE = Path('/embedding/code/evaluation/tools/early_mixer_confirmation_v1')
DESTINATION = Path('/embedding/output/runs/rpb-early-mixer-confirmation/report-tools')
STAGER_SHA = '92b58a7de0a4a8c0f31443b99f78dfa7d1f515f7827b2685b3b3bf946eeb6a73'
MANIFEST_SHA = '04df04d87373aa375e9c3200d33213451c641ca88c3198060502edf041abbacd'
NAMES = ('write_early_mixer_confirmation_summary_v1.py',
         'prepare_confirmation_renderer_release_v1.py',
         'save_and_qa_confirmation_report_v1.py',
         'prepare_confirmation_saver_release_v1.py')


def check(ok, why):
    if not ok:
        raise AssertionError(why)


def admit(paths):
    for path in paths:
        check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
              'canonical explicit SOURCE path')
        info = path.stat()
        check(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'unaliased regular SOURCE')
    check(len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) == len(paths), 'distinct SOURCE inode matrix')


def helpers():
    path = SOURCE / 'stage_source_tools.py'
    admit([path])
    check(hashlib.sha256(path.read_bytes()).hexdigest() == STAGER_SHA, 'reviewed stager SOURCE pin')
    spec = importlib.util.spec_from_file_location('confirmation_source_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Definitions only; no reader or reporting imports.
    return module


def self_test():
    module = helpers()
    negatives = 0
    with tempfile.TemporaryDirectory(prefix='confirmation-report-source-fixture-') as temporary:
        root = Path(temporary).resolve()
        absent, conflict = root / 'absent', root / 'conflict'
        conflict.write_bytes(b'existing artificial SOURCE')
        try:
            module.materialize_sources([(absent, b'new SOURCE'), (conflict, b'wrong SOURCE')])
        except AssertionError:
            negatives += 1
        else:
            raise AssertionError('late conflicting SOURCE accepted')
        check(not absent.exists(), 'complete target admission precedes all copies')
        alias = root / 'alias'; alias.symlink_to(conflict)
        try:
            admit([conflict, alias])
        except AssertionError:
            negatives += 1
        else:
            raise AssertionError('redirected SOURCE accepted')
        module.materialize_sources([(absent, b'new SOURCE'), (conflict, b'existing artificial SOURCE')])
        check(absent.read_bytes() == b'new SOURCE', 'absent SOURCE and identical reuse')
    return {'status': 'passed', 'artificial_only': True, 'negative_cases': negatives,
            'quality_or_tensor_reads': 0, 'report_or_release_execution': False,
            'actual_reporting_source_stage': False}


def stage():
    manifest, helper = SOURCE / 'report-source-manifest.json', SOURCE / 'stage_source_tools.py'
    paths = [manifest, helper, *[SOURCE / name for name in NAMES]]
    admit(paths)  # Admit the entire fixed SOURCE matrix before reading any body.
    contents = {p.name: p.read_bytes() for p in paths}
    check(hashlib.sha256(contents[manifest.name]).hexdigest() == MANIFEST_SHA and
          hashlib.sha256(contents[helper.name]).hexdigest() == STAGER_SHA, 'exact reviewed SOURCE declarations')
    module = helpers(); value = module.parse(contents[manifest.name])
    check(set(value) == {'protocol', 'files'} and value['protocol'] == 'early-mixer-confirmation-v1', 'SOURCE manifest scope')
    rows = value['files']
    check(len(rows) == 4 and [row['name'] for row in rows] == list(NAMES), 'exact four reporting definitions')
    for row in rows:
        raw = contents[row['name']]
        check(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'byte-exact reviewed reporting SOURCE')
    check(contents[NAMES[0]].count(b'AUTHORIZE_COMPLETED_METADATA = False\n') == 1 and
          contents[NAMES[2]].count(b'AUTHORIZED_COMPLETED_METADATA = False\n') == 1, 'both reporting entrypoints remain blocked')
    module.materialize_sources([(DESTINATION / name, contents[name]) for name in NAMES])
    check(all(p.read_bytes() == contents[p.name] for p in paths), 'tracked SOURCE preserved')
    return {'status': 'staged-blocked-source', 'definitions': list(NAMES),
            'destination': str(DESTINATION), 'manifest_sha256': MANIFEST_SHA,
            'quality_or_tensor_reads': 0, 'report_or_release_execution': False}


def main():
    check(Path('/.dockerenv').is_file(), 'existing managed container only')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else stage(), sort_keys=True))


if __name__ == '__main__':
    main()

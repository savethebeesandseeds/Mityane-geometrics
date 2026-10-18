#!/usr/bin/env python3
"""Prepare a one-flag pooled renderer release from an explicit authorized record.

SOURCE-only --self-test uses artificial bytes and opens no quality files.
The release branch must be invoked later with the root-approved record and its
exact SHA. It binds named completed JSON/source files, never a tensor archive.
It neither emits reports nor executes the reader, renderer, model or heads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/embedding/output/runs/rpb-pooled-context/report-tools')
PROTOCOL = 'pooled-context-v1'
BLOCKED = ROOT / 'write_pooled_context_summary_v1.py'
BLOCKED_SHA = 'e51cc6b5ebe1d18e81cb9fc820052e1a242febaf50d10bfa95b4e4cb94124f86'
CARD = 'code/evaluation/cards/pooled_context_v1.md'
CARD_SHA = 'b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7'
METADATA = ['results/report.json', 'results/complete.json', 'artifact-integrity.json',
            'launch-plan.json', 'recipe-plan.json', 'admission/passed.json',
            'admission/build-and-tests.log', 'source/' + CARD]
FALSE = b'AUTHORIZE_COMPLETED_METADATA = False\n'
TRUE = b'AUTHORIZE_COMPLETED_METADATA = True\n'


def check(ok, why):
    if not ok:
        raise AssertionError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def regular(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and
          not path.is_symlink() and path.stat().st_nlink == 1, 'regular unaliased explicit metadata/source file')
    check(all(not p.is_symlink() for p in path.parents), 'metadata ancestor redirect')


def parse(data):
    def unique(items):
        result = {}
        for key, value in items:
            check(key not in result, 'unique authorization/metadata JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise AssertionError('nonfinite metadata literal ' + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def released_bytes(old):
    check(old.count(FALSE) == 1 and TRUE not in old, 'exact reviewed blocked authorization declaration')
    new = old.replace(FALSE, TRUE)
    check(new.count(TRUE) == 1 and new.replace(TRUE, FALSE) == old, 'one-flag reverse-byte proof')
    return new


def authorization(value):
    keys = {'protocol', 'capsule', 'audit_validation', 'audit_reader', 'source_fingerprint',
            'expected_sha256', 'audit_checks', 'audit_archive_decodes', 'root_approved_completed_metadata'}
    check(set(value) == keys and value['protocol'] == PROTOCOL and
          value['root_approved_completed_metadata'] is True, 'explicit closed root-authorized release record')
    capsule = Path(value['capsule'])
    check(capsule.is_absolute() and capsule.parent == ROOT.parent and
          capsule.name.startswith('pooled-context-'), 'only the new curve capsule')
    audit = Path(value['audit_validation']); reader = Path(value['audit_reader'])
    check(audit.is_absolute() and reader.is_absolute() and
          audit.is_relative_to(ROOT.parent / 'audit-tools') and reader.is_relative_to(ROOT.parent / 'audit-tools') and
          audit.name == 'validation.json' and reader.name == 'validate_pooled_context.py',
          'named new audit JSON and reader SOURCE only')
    paths = [capsule / name for name in METADATA] + [audit, reader]
    check(set(value['expected_sha256']) == set(str(p) for p in paths) and
          all(re.fullmatch('[a-f0-9]{64}', h) for h in value['expected_sha256'].values()) and
          re.fullmatch('[a-f0-9]{64}', value['source_fingerprint']) is not None,
          'closed root-approved whole metadata/source hash matrix')
    check(all(type(value[key]) is int and value[key]>0 for key in ('audit_checks','audit_archive_decodes')),
          'explicit successful independent audit counts')
    return capsule, audit, reader, paths


def write_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def self_test():
    old = b'prefix\n' + FALSE + b'suffix\n'
    new = released_bytes(old)
    check(new == b'prefix\n' + TRUE + b'suffix\n', 'single artificial flag replacement')
    record = {'protocol': PROTOCOL, 'capsule': str(ROOT.parent / 'pooled-context-artificial'),
              'audit_validation': str(ROOT.parent / 'audit-tools/run-artificial/validation.json'),
              'audit_reader': str(ROOT.parent / 'audit-tools/reader-artificial/validate_pooled_context.py'),
              'source_fingerprint': 'a' * 64, 'expected_sha256': {},
              'audit_checks':1, 'audit_archive_decodes':1,
              'root_approved_completed_metadata': True}
    paths = ([Path(record['capsule']) / name for name in METADATA] +
             [Path(record['audit_validation']), Path(record['audit_reader'])])
    record['expected_sha256'] = {str(p): 'b' * 64 for p in paths}
    check(authorization(record)[3] == paths, 'artificial closed binding declaration')
    negative = 0
    def reject(call):
        nonlocal negative
        try:
            call()
        except AssertionError:
            negative += 1
        else:
            raise AssertionError('invalid artificial release record admitted')
    reject(lambda: released_bytes(old + FALSE))
    reject(lambda: released_bytes(new))
    reject(lambda: authorization({**record, 'root_approved_completed_metadata': False}))
    reject(lambda: authorization({**record, 'protocol': 'historical-protocol'}))
    reject(lambda: authorization({**record, 'capsule': '/embedding/output/runs/old/pooled-context-artificial'}))
    reject(lambda: authorization({**record, 'expected_sha256': {}}))
    reject(lambda: authorization({**record, 'extra': 'unbound'}))
    reject(lambda: parse('{"a":1,"a":2}'))
    return {'status': 'passed', 'artificial_only': True, 'negative_cases': negative,
            'quality_metadata_reads': 0, 'tensor_reads': 0, 'source_authorization_flag_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--fixture-output')
    parser.add_argument('--authorization')
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--release-directory')
    args = parser.parse_args()
    check(Path('/.dockerenv').is_file(), 'managed container only')
    if args.self_test:
        check(not any((args.authorization, args.authorization_sha256, args.release_directory)), 'artificial mode cannot open a release record')
        result=self_test()
        if args.fixture_output:
            target=Path(args.fixture_output)
            check(target.is_absolute() and target.parent==ROOT and ROOT.resolve(strict=True)==ROOT and not target.exists(),
                  'exclusive source-only fixture output')
            write_new(target,(json.dumps(result,indent=2)+'\n').encode())
        print(json.dumps(result, sort_keys=True)); return
    check(args.authorization and args.authorization_sha256 and args.release_directory,
          'explicit later root authorization record, SHA and exclusive release path required')
    authorization_path = Path(args.authorization); release = Path(args.release_directory)
    check(authorization_path.suffix == '.json' and authorization_path.parent == ROOT,
          'single new root-approved record under this protocol report-tools')
    check(re.fullmatch('[a-f0-9]{64}', args.authorization_sha256) is not None and
          release.is_absolute() and release.parent == ROOT and release.name.startswith('released-renderer-') and
          ROOT.resolve(strict=True)==ROOT and not release.exists() and not release.is_symlink(), 'explicit unique release directory and authorization hash')
    regular(authorization_path)
    authorization_bytes = authorization_path.read_bytes()
    check(sha(authorization_bytes) == args.authorization_sha256, 'exact root-approved authorization record bytes')
    approved = parse(authorization_bytes)
    capsule, audit_path, reader, measured = authorization(approved)
    for path in measured + [BLOCKED]:
        regular(path)
    # Entire closed matrix was admitted before any quality/source bytes below.
    contents = {path: path.read_bytes() for path in measured + [BLOCKED]}
    for path in measured:
        check(sha(contents[path]) == approved['expected_sha256'][str(path)], 'root-approved completed metadata/source byte hash')
    check(sha(contents[BLOCKED]) == BLOCKED_SHA, 'exact peer-reviewed blocked renderer')
    audit = parse(contents[audit_path]); complete = parse(contents[capsule / 'results/complete.json'])
    check(audit['status'] == 'passed' and audit['protocol'] == complete['protocol'] == PROTOCOL and
          complete['status'] == 'complete' and not complete['testing_accessed'] and not complete['stress_accessed'] and
          not complete['promotion'] and not complete['selection'], 'actual successful audit/complete new quality without selection')
    for key, name in (('checks','audit_checks'),('archive_decodes','audit_archive_decodes')):
        check(audit[key] == approved[name], 'root-authorized actual independent audit count')
    source = audit['source']
    check(source['source_fingerprint'] == approved['source_fingerprint'] and
          source['reader_sha256'] == approved['expected_sha256'][str(reader)] and
          source['inventory_sha256'] == approved['expected_sha256'][str(capsule / 'artifact-integrity.json')] and
          source['human_card_sha256'] == CARD_SHA == sha(contents[capsule / ('source/' + CARD)]),
          'completed source/reader/inventory/frozen card binding')
    old = contents[BLOCKED]; new = released_bytes(old)
    binding = dict(approved)
    proof = {'protocol': PROTOCOL, 'status': 'passed', 'authorization_path': str(authorization_path),
             'authorization_sha256': args.authorization_sha256, 'blocked_source_sha256': sha(old),
             'released_source_sha256': sha(new), 'authorization_flag_only_reverse_byte_proof': True,
             'quality_metadata_paths': [str(path) for path in measured], 'tensor_reads': 0,
             'model_head_PCA_or_audit_execution': False}
    release.mkdir(exist_ok=False)
    write_new(release / 'reviewed-blocked-source.py', old)
    write_new(release / 'write_pooled_context_summary_v1.py', new)
    for name, value in [('completed-binding.json', binding), ('renderer-release-proof.json', proof)]:
        write_new(release / name, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({'status': 'prepared', 'release': str(release), 'released_source_sha256': sha(new),
                      'binding_sha256': sha((release / 'completed-binding.json').read_bytes()),
                      'tensor_reads': 0, 'report_emissions': 0}, sort_keys=True))


if __name__ == '__main__':
    main()

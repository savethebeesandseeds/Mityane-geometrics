#!/usr/bin/env python3
"""Prepare a one-flag durable-saver release from a later explicit root binding.

Artificial --self-test reads only the pinned blocked saver SOURCE. The release
branch requires one root-approved binding and its exact hash, admits its whole
closed metadata matrix before reads, and creates an exclusive new source copy.
It does not run the saver, renderer, audit, model, heads or a tensor codec.
"""
import argparse
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
TOOLS = Path('/embedding/output/runs/rpb-early-mixer-confirmation/report-tools')
BLOCKED = TOOLS / 'save_and_qa_confirmation_report_v1.py'
BLOCKED_SHA = '0cb0b4abd5e6f093033159c34ebb0dbe3f2028bba921148e3b144b7b598e0108'
PROTOCOL = 'early-mixer-confirmation-v1'
CARD_SHA = '98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea'
FALSE = b'AUTHORIZED_COMPLETED_METADATA = False\n'
TRUE = b'AUTHORIZED_COMPLETED_METADATA = True\n'


def check(ok, why):
    if not ok:
        raise AssertionError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and
          path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
          'explicit canonical regular metadata or source file')
    check(all(not ancestor.is_symlink() for ancestor in path.parents),
          'no redirected metadata ancestor')


def parse(raw):
    def unique(items):
        result = {}
        for key, value in items:
            check(key not in result, 'duplicate JSON binding key')
            result[key] = value
        return result
    def invalid(value):
        raise AssertionError('nonfinite JSON binding literal ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def blocked_source():
    regular(BLOCKED)
    old = BLOCKED.read_bytes()
    check(sha(old) == BLOCKED_SHA, 'exact reviewed blocked saver SOURCE')
    released_bytes(old)
    spec = importlib.util.spec_from_file_location('blocked_confirmation_saver_source', BLOCKED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check(module.AUTHORIZED_COMPLETED_METADATA is False and module.PROTOCOL == PROTOCOL,
          'imported definitions remain blocked')
    return module, old


def released_bytes(old):
    check(old.count(FALSE) == 1 and TRUE not in old, 'one disabled saver authorization line')
    new = old.replace(FALSE, TRUE)
    check(new.count(TRUE) == 1 and new.replace(TRUE, FALSE) == old,
          'exact one-flag reverse-byte proof')
    return new


def self_test():
    module, source = blocked_source()
    check(source.count(FALSE) == 1 and module.AUTHORIZED_COMPLETED_METADATA is False,
          'self-test leaves original source blocked')
    old = b'prefix\n' + FALSE + b'suffix\n'
    check(released_bytes(old) == b'prefix\n' + TRUE + b'suffix\n', 'artificial one-line replacement')
    capsule = TOOLS.parent / 'early-mixer-confirmation-artificial'
    emitted = TOOLS / 'emitted-artificial'
    audit = TOOLS.parent / 'audit-tools/run-artificial/validation.json'
    renderer = TOOLS / 'released-renderer-artificial/write_early_mixer_confirmation_summary_v1.py'
    proof = renderer.parent / 'renderer-release-proof.json'
    paths = [emitted/module.SUMMARY_NAME, emitted/module.REPORT_NAME, audit,
             capsule/'results/report.json', capsule/'results/complete.json',
             capsule/'artifact-integrity.json', renderer, proof]
    binding = {'protocol': PROTOCOL, 'root_authorized_completed_metadata': True,
               'capsule': str(capsule), 'emitted_directory': str(emitted),
               'audit_validation': str(audit), 'released_renderer': str(renderer),
               'renderer_release_proof': str(proof),
               'expected_sha256': {str(path): 'a' * 64 for path in paths}}
    check(module.binding_paths(binding)[-1] == paths, 'actual closed saver binding branch')
    negatives = 0
    def reject(call):
        nonlocal negatives
        try:
            call()
        except AssertionError:
            negatives += 1
        else:
            raise AssertionError('invalid artificial saver release accepted')
    reject(lambda: released_bytes(old + FALSE))
    reject(lambda: released_bytes(released_bytes(old)))
    reject(lambda: module.binding_paths({**binding, 'root_authorized_completed_metadata': False}))
    reject(lambda: module.binding_paths({**binding, 'protocol': 'historical-protocol'}))
    reject(lambda: module.binding_paths({**binding, 'expected_sha256': {}}))
    reject(lambda: module.binding_paths({**binding, 'extra': 'unbound'}))
    reject(lambda: parse('{"a":1,"a":2}'))
    reject(lambda: parse('{"a":NaN}'))
    admitted = []
    def late_invalid(path):
        admitted.append(path)
        check(path != 'last-invalid', 'late artificial path')
    reject(lambda: module.admit_metadata(['first', 'last-invalid'], late_invalid))
    check(admitted == ['first', 'last-invalid'], 'all paths precede any content phase')
    scope = module.unresolved_globals(Path(__file__).read_text(), set(globals()) | set(dir(builtins)))
    check(not scope['unresolved_globals'], 'all conditional release globals resolve')
    return {'protocol': PROTOCOL, 'status': 'passed', 'artificial_only': True,
            'negative_cases': negatives, 'completed_metadata_reads': 0, 'tensor_reads': 0,
            'release_writes': 0, 'source_authorization_flag_unchanged': True,
            'blocked_saver_sha256': BLOCKED_SHA, 'global_resolution': scope}


def write_new(path, raw):
    with path.open('xb') as out:
        out.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--binding'); parser.add_argument('--binding-sha256')
    parser.add_argument('--release-directory')
    args = parser.parse_args()
    check(Path('/.dockerenv').is_file(), 'existing managed container only')
    if args.self_test:
        check(not any((args.binding, args.binding_sha256, args.release_directory)),
              'artificial mode cannot open a completed record')
        print(json.dumps(self_test(), sort_keys=True)); return
    check(args.binding and args.binding_sha256 and args.release_directory,
          'later exact root-approved binding and exclusive release directory required')
    approved_path = Path(args.binding); release = Path(args.release_directory)
    check(approved_path.is_absolute() and approved_path.parent == TOOLS and
          approved_path.suffix == '.json' and re.fullmatch('[a-f0-9]{64}', args.binding_sha256),
          'single explicit new root approval path and hash')
    check(release.is_absolute() and release.parent == TOOLS and
          release.name.startswith('released-saver-') and not release.exists() and
          not release.is_symlink() and TOOLS.resolve(strict=True) == TOOLS,
          'exclusive canonical new saver release leaf')
    regular(approved_path); binding_raw = approved_path.read_bytes()
    check(sha(binding_raw) == args.binding_sha256, 'root-approved exact binding bytes')
    binding = parse(binding_raw)
    module, old = blocked_source()
    capsule, emitted, audit_path, renderer, proof_path, paths = module.binding_paths(binding)
    module.admit_metadata(paths)
    contents = {path: path.read_bytes() for path in paths}
    for path in paths:
        check(sha(contents[path]) == binding['expected_sha256'][str(path)],
              'whole explicit root-approved completed metadata/source bytes')
    audit = parse(contents[audit_path]); complete = parse(contents[capsule/'results/complete.json'])
    summary = parse(contents[emitted/module.SUMMARY_NAME]); proof = parse(contents[proof_path])
    check(audit['protocol'] == complete['protocol'] == summary['protocol'] == PROTOCOL and
          audit['status'] == 'passed' and complete['status'] == 'complete' and
          type(audit['checks']) is int and audit['checks'] > 0,
          'actual passed same-run audit and completed emission')
    check(not complete['testing_accessed'] and not complete['stress_accessed'] and
          not complete['selection'] and not complete['promotion'], 'completed declared scope')
    check(summary['bindings']['capsule'] == str(capsule) and
          summary['bindings']['audit_sha256'] == sha(contents[audit_path]) and
          summary['bindings']['inventory_sha256'] == sha(contents[capsule/'artifact-integrity.json']) and
          summary['bindings']['source_fingerprint'] == audit['source']['source_fingerprint'] and
          summary['bindings']['card_sha256'] == audit['source']['human_card_sha256'] == CARD_SHA,
          'same-run source/card/audit/inventory emission binding')
    check(proof['protocol'] == PROTOCOL and proof['status'] == 'passed' and
          proof['blocked_source_sha256'] == module.RENDERER_SHA and
          proof['released_source_sha256'] == sha(contents[renderer]) and
          proof['authorization_flag_only_reverse_byte_proof'] is True,
          'actual reviewed renderer release identity')
    check(contents[renderer].count(b'AUTHORIZE_COMPLETED_METADATA = True\n') == 1 and
          sha(contents[renderer].replace(b'AUTHORIZE_COMPLETED_METADATA = True\n',
                                        b'AUTHORIZE_COMPLETED_METADATA = False\n')) == module.RENDERER_SHA,
          'exact one-flag renderer bytes, no arithmetic or schema change')
    new = released_bytes(old)
    release.mkdir(exist_ok=False)
    write_new(release/'reviewed-blocked-source.py', old)
    write_new(release/BLOCKED.name, new)
    result = {'protocol': PROTOCOL, 'status': 'prepared', 'binding_path': str(approved_path),
              'binding_sha256': args.binding_sha256, 'blocked_source_sha256': sha(old),
              'released_source_sha256': sha(new), 'authorization_flag_only_reverse_byte_proof': True,
              'metadata_paths': [str(path) for path in paths], 'tensor_reads': 0,
              'saver_renderer_audit_model_head_execution': False, 'durable_report_writes': 0}
    write_new(release/'saver-release-proof.json',
              (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    regular(approved_path); regular(BLOCKED)
    check(approved_path.read_bytes() == binding_raw and BLOCKED.read_bytes() == old,
          'approved binding and blocked source preserved')
    for path in paths:
        regular(path)
        check(path.read_bytes() == contents[path], 'all admitted completed bytes preserved')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

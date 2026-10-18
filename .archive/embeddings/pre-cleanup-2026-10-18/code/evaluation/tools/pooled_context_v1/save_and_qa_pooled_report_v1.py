#!/usr/bin/env python3
"""Blocked durable-copy and metadata QA for the new pooled temporal context comparison.

The source-only fixture mode imports only the pinned, blocked renderer source
and uses its artificial dictionaries.  Completed metadata access requires a
separate root-authorized released copy and exact closed input hashes.  This tool
does not edit the registry or a prior report, decode archives, execute a reader,
or run an encoder, optimizer, classifier, PCA, or bootstrap.
"""
import argparse
import builtins
import copy
import hashlib
import importlib.util
import json
import re
import sys
import symtable
from pathlib import Path

sys.dont_write_bytecode = True
AUTHORIZED_COMPLETED_METADATA = False
PROTOCOL = 'pooled-context-v1'
REPO = Path('/embedding')
ROOT = REPO / 'output/runs/rpb-pooled-context'
TOOLS = ROOT / 'report-tools'
RENDERER = TOOLS / 'write_pooled_context_summary_v1.py'
RENDERER_SHA = 'e51cc6b5ebe1d18e81cb9fc820052e1a242febaf50d10bfa95b4e4cb94124f86'
CARD_SHA = 'b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7'
REPORT_REL = 'code/encoders/raw_patch_bottleneck_mae/POOLED_CONTEXT_DIAGNOSTIC.md'
SUMMARY_REL = 'doc/results/pooled_context_v1.json'
REPORT_NAME = 'POOLED_CONTEXT_DIAGNOSTIC.md'
SUMMARY_NAME = 'pooled_context_v1.json'
REGISTRY = REPO / 'doc/embedding_versions.json'


def check(ok, why):
    if not ok:
        raise AssertionError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and
          path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
          'explicit canonical regular metadata/source file')
    check(all(not p.is_symlink() for p in path.parents), 'no redirected metadata ancestor')
    return path


def admit_metadata(paths, admit=regular):
    check(len(paths) == len(set(paths)), 'unique complete metadata path matrix')
    for path in paths:
        admit(path)
    # No content hash or read occurs until this entire pass returns.
    return paths


def parse(raw):
    def unique(items):
        result = {}
        for key, value in items:
            check(key not in result, 'duplicate JSON metadata key')
            result[key] = value
        return result
    def invalid(value):
        raise AssertionError('nonfinite JSON literal: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def unresolved_globals(source, known):
    scopes = []; references = 0; missing = set()
    def visit(table):
        nonlocal references
        scopes.append(table)
        for symbol in table.get_symbols():
            if symbol.is_referenced() and symbol.is_global():
                references += 1
                if symbol.get_name() not in known:
                    missing.add(symbol.get_name())
        for child in table.get_children():
            visit(child)
    visit(symtable.symtable(source, '<saved-metadata-source>', 'exec'))
    return {'scopes': len(scopes), 'global_references': references,
            'unresolved_globals': sorted(missing)}


def renderer_source():
    regular(RENDERER)
    raw = RENDERER.read_bytes()
    check(sha(raw) == RENDERER_SHA and
          raw.count(b'AUTHORIZE_COMPLETED_METADATA = False\n') == 1,
          'exact reviewed blocked renderer SOURCE')
    spec = importlib.util.spec_from_file_location('blocked_pooled_renderer_source', RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check(module.AUTHORIZE_COMPLETED_METADATA is False, 'imported source remains blocked')
    return module


def metadata_qa(module, report, audit, complete, summary, markdown_bytes):
    module.validate_pair(report, audit, complete)
    expected = module.summarize(report, audit, complete, summary['bindings'])
    check(summary == expected, 'every emitted field matches saved producer and passed audit metadata')
    check(markdown_bytes == module.markdown(summary).encode('utf-8'),
          'exact reviewed renderer Markdown; no changed scores or intervals')
    check(len(summary['quality']) == 4 and len(summary['training']) == 2 and
          len(summary['cohorts']) == 5 and
          all(len(p['methods']) == 7 and len(p['paired_effects']) == 5
              for p in summary['quality']), 'all task/view/budget/cohort panels retained')
    check(summary['completion']['encoder_updates_each'] == 512 and
          summary['promotion'] is summary['completion']['selection'] is False,
          'ordinary joint reconstruction only; no promotion or selected budget')
    check(b'| Encoder | Updates | Train error \xe2\x86\x93 | Validation error \xe2\x86\x93 | GPU training seconds |'
          in markdown_bytes, 'standard five training headers and order')
    return {'protocol': PROTOCOL, 'status': 'passed', 'panels': 4,
            'cohorts': 5, 'training_rows': 2,
            'all_summary_fields_reproduced_from_metadata': True,
            'exact_renderer_markdown': True, 'tensor_decodes': 0,
            'model_optimizer_head_PCA_or_audit_execution': False}


def registry_qa(registry, summary, report_sha, summary_sha):
    latest = registry['latest_completed_diagnostic']
    check(latest['protocol'] == PROTOCOL and latest['instance_bundles'] ==
          ['RPB-v10.alt-03', 'RPB-v12'], 'same new registry experiment and group labels')
    check(latest['measured_record'] == REPORT_REL and latest['durable_summary'] == SUMMARY_REL and
          latest['measured_record_sha256'] == report_sha and
          latest['durable_summary_sha256'] == summary_sha, 'exact durable registry file bindings')
    bindings = summary['bindings']
    for field, key in [('source_fingerprint', 'source_fingerprint'),
                       ('artifact_inventory_sha256', 'inventory_sha256'),
                       ('audit_validation_sha256', 'audit_sha256'),
                       ('audit_reader_sha256', 'reader_sha256')]:
        check(latest[field] == bindings[key], 'registry measured scope: ' + field)
    check(latest['audit_checks'] == summary['independent_audit']['checks'] and
          latest['timing_master_seeds'] == [53151, 54252, 55353, 56454, 57555] and
          latest['amplitude_data_master_seeds'] == [58656, 59757, 60858, 61959, 63060] and
          not latest['testing_accessed'] and not latest['stress_accessed'] and
          not latest['promotion'] and latest['existing_groups_replaced'] is False,
          'registry preserves cohorts, evidence scope and historical groups')
    return {'status': 'passed', 'registry_mutated': False,
            'durable_file_scope_bindings_checked': True}


def binding_paths(binding):
    keys = {'protocol', 'root_authorized_completed_metadata', 'capsule',
            'emitted_directory', 'audit_validation', 'released_renderer',
            'renderer_release_proof', 'expected_sha256'}
    check(set(binding) == keys and binding['protocol'] == PROTOCOL and
          binding['root_authorized_completed_metadata'] is True, 'explicit closed root approval')
    capsule = Path(binding['capsule']); emitted = Path(binding['emitted_directory'])
    check(capsule.is_absolute() and capsule.parent == ROOT and
          capsule.name.startswith('pooled-context-'), 'only new completed capsule metadata')
    check(emitted.is_absolute() and emitted.parent == TOOLS and
          emitted.name.startswith('emitted-'), 'only new explicit renderer emission')
    audit = Path(binding['audit_validation']); released = Path(binding['released_renderer'])
    proof = Path(binding['renderer_release_proof'])
    check(audit.is_absolute() and audit.is_relative_to(ROOT/'audit-tools') and
          audit.name == 'validation.json', 'new passed audit JSON only')
    check(released.is_absolute() and released.parent.parent == TOOLS and
          released.parent.name.startswith('released-renderer-') and
          released.name == RENDERER.name and proof == released.parent/'renderer-release-proof.json',
          'explicit released renderer and its proof')
    paths = [emitted/SUMMARY_NAME, emitted/REPORT_NAME, audit,
             capsule/'results/report.json', capsule/'results/complete.json',
             capsule/'artifact-integrity.json', released, proof]
    check(set(binding['expected_sha256']) == set(str(p) for p in paths) and
          all(re.fullmatch('[a-f0-9]{64}', h) for h in binding['expected_sha256'].values()),
          'closed exact metadata/source SHA matrix')
    return capsule, emitted, audit, released, proof, paths


def self_test():
    module = renderer_source()
    report, audit, complete = module.artificial_metadata()
    summary = module.summarize(report, audit, complete, {})
    body = module.markdown(summary).encode()
    metadata_qa(module, report, audit, complete, summary, body)
    negatives = 0
    def reject(call):
        nonlocal negatives
        try:
            call()
        except AssertionError:
            negatives += 1
        else:
            raise AssertionError('bad artificial metadata admitted')
    changed = copy.deepcopy(summary); changed['quality'][0]['methods'][0]['linear']['mean']['decimal'] += .001
    reject(lambda: metadata_qa(module, report, audit, complete, changed, body))
    changed = copy.deepcopy(summary); progress = next(row['encoder_progress'] for row in changed['cohorts'][0]['encoders'] if row.get('encoder_progress') and row['encoder_progress']['losses'])
    progress['losses'][0][3] += .1
    reject(lambda: metadata_qa(module, report, audit, complete, changed, body))
    reject(lambda: metadata_qa(module, report, audit, complete, summary, body + b'changed'))
    reject(lambda: completed(None))  # Actual guard rejects before touching any path.
    reject(lambda: parse('{"a":1,"a":2}'))
    reject(lambda: parse('{"a":NaN}'))
    seen = []
    def late_invalid(path):
        seen.append(path)
        check(path != 'last-invalid', 'late invalid artificial metadata path')
    reject(lambda: admit_metadata(['first', 'last-invalid'], late_invalid))
    check(seen == ['first', 'last-invalid'], 'whole admission reaches late invalid before content phase')
    bindings = {'source_fingerprint': 'a'*64, 'inventory_sha256': 'b'*64,
                'audit_sha256': 'c'*64, 'reader_sha256': 'd'*64}
    summary['bindings'] = bindings
    latest = {'protocol': PROTOCOL, 'instance_bundles': ['RPB-v10.alt-03', 'RPB-v12'],
              'measured_record': REPORT_REL, 'durable_summary': SUMMARY_REL,
              'measured_record_sha256': 'e'*64, 'durable_summary_sha256': 'f'*64,
              'source_fingerprint': 'a'*64, 'artifact_inventory_sha256': 'b'*64,
              'audit_validation_sha256': 'c'*64, 'audit_reader_sha256': 'd'*64,
              'audit_checks': summary['independent_audit']['checks'],
              'timing_master_seeds': [53151,54252,55353,56454,57555],
              'amplitude_data_master_seeds': [58656,59757,60858,61959,63060],
              'testing_accessed': False, 'stress_accessed': False, 'promotion': False,
              'existing_groups_replaced': False}
    registry_qa({'latest_completed_diagnostic': latest}, summary, 'e'*64, 'f'*64)
    bad = copy.deepcopy(latest); bad['audit_validation_sha256'] = 'e'*64
    reject(lambda: registry_qa({'latest_completed_diagnostic': bad}, summary, 'e'*64, 'f'*64))
    scope = unresolved_globals(Path(__file__).read_text(), set(globals()) | set(dir(builtins)))
    check(not scope['unresolved_globals'], 'all conditional completed branches resolve globals')
    check(unresolved_globals('def bad():\n    return omitted_global\n', set(dir(builtins)))['unresolved_globals']
          == ['omitted_global'], 'actual scanner detects a missing measured-branch name')
    return {'protocol': PROTOCOL, 'status': 'passed', 'artificial_only': True,
            'negative_cases': negatives, 'completed_metadata_reads': 0,
            'tensor_decodes': 0, 'durable_writes': 0, 'registry_mutations': 0,
            'metadata_authorized': AUTHORIZED_COMPLETED_METADATA,
            'global_resolution': scope,
            'pinned_blocked_renderer_sha256': RENDERER_SHA}


def completed(args):
    check(AUTHORIZED_COMPLETED_METADATA, 'blocked before completed metadata path admission or reads')
    approval = Path(args.binding)
    check(approval.is_absolute() and approval.parent == TOOLS and approval.suffix == '.json' and
          re.fullmatch('[a-f0-9]{64}', args.binding_sha256) is not None,
          'explicit later root-approved metadata binding and SHA')
    regular(approval); approval_raw = approval.read_bytes()
    check(sha(approval_raw) == args.binding_sha256, 'exact root-approved binding bytes')
    binding = parse(approval_raw)
    capsule, emitted, audit_path, released, proof_path, paths = binding_paths(binding)
    admit_metadata(paths)
    contents = {p: p.read_bytes() for p in paths}
    for path in paths:
        check(sha(contents[path]) == binding['expected_sha256'][str(path)], 'authorized exact metadata bytes')
    summary = parse(contents[emitted/SUMMARY_NAME]); report = parse(contents[capsule/'results/report.json'])
    audit = parse(contents[audit_path]); complete = parse(contents[capsule/'results/complete.json'])
    inventory = parse(contents[capsule/'artifact-integrity.json']); proof = parse(contents[proof_path])
    check(proof['protocol'] == PROTOCOL and proof['status'] == 'passed' and
          proof['blocked_source_sha256'] == RENDERER_SHA and
          proof['released_source_sha256'] == sha(contents[released]) and
          proof['authorization_flag_only_reverse_byte_proof'] is True and proof['tensor_reads'] == 0,
          'actual reviewed one-flag renderer release')
    old_source = contents[released].replace(b'AUTHORIZE_COMPLETED_METADATA = True\n',
                                           b'AUTHORIZE_COMPLETED_METADATA = False\n')
    check(contents[released].count(b'AUTHORIZE_COMPLETED_METADATA = True\n') == 1 and
          sha(old_source) == RENDERER_SHA, 'released renderer exact one-flag source')
    bindings = summary['bindings']; source = audit['source']
    check(bindings['capsule'] == str(capsule) and bindings['audit_validation'] == str(audit_path) and
          bindings['audit_sha256'] == sha(contents[audit_path]) and
          bindings['inventory_sha256'] == sha(contents[capsule/'artifact-integrity.json']) and
          bindings['renderer_sha256'] == sha(contents[released]) and
          bindings['source_fingerprint'] == source['source_fingerprint'] == report['source_fingerprint'] and
          bindings['card_sha256'] == source['human_card_sha256'] == CARD_SHA,
          'same capsule/audit/source/inventory/card/emission binding')
    index = {row['path']: row for row in inventory['files']}
    check(len(index) == len(inventory['files']), 'unique completed inventory metadata')
    for name in ('results/report.json', 'results/complete.json'):
        check(index[name]['sha256'] == sha(contents[capsule/name]), 'inventory-bound report and completion')
    result = metadata_qa(renderer_source(), report, audit, complete, summary, contents[emitted/REPORT_NAME])
    report_sha = sha(contents[emitted/REPORT_NAME]); summary_sha = sha(contents[emitted/SUMMARY_NAME])
    if args.registry:
        check(Path(args.registry) == REGISTRY and args.registry_sha256 and
              re.fullmatch('[a-f0-9]{64}', args.registry_sha256), 'explicit root registry metadata only')
        regular(REGISTRY); raw = REGISTRY.read_bytes()
        check(sha(raw) == args.registry_sha256, 'exact approved registry bytes')
        result['registry_qa'] = registry_qa(parse(raw), summary, report_sha, summary_sha)
    output = Path(args.proof_directory)
    check(output.is_absolute() and output.parent == TOOLS and
          output.parent.resolve(strict=True) == TOOLS and output.name.startswith('durable-qa-') and
          not output.exists() and not output.is_symlink(), 'exclusive new metadata QA evidence directory')
    if args.save_durable:
        targets = [REPO/REPORT_REL, REPO/SUMMARY_REL]
        check(all(p.parent.resolve(strict=True) == p.parent and not p.exists() and
                  not p.is_symlink() for p in targets), 'new canonical durable destinations; never overwrite')
        with targets[0].open('xb') as out:
            out.write(contents[emitted/REPORT_NAME])
        with targets[1].open('xb') as out:
            out.write(contents[emitted/SUMMARY_NAME])
        check(sha(targets[0].read_bytes()) == report_sha and sha(targets[1].read_bytes()) == summary_sha,
              'durable copy exactly preserves reviewed emission')
    # Verify no admitted source/metadata input changed during QA or copying.
    for path in paths:
        regular(path)
        check(path.read_bytes() == contents[path], 'all admitted metadata/source bytes preserved after QA')
    result.update(report=str(REPO/REPORT_REL), summary=str(REPO/SUMMARY_REL),
                  report_sha256=report_sha, summary_sha256=summary_sha,
                  audit_sha256=bindings['audit_sha256'], source_fingerprint=bindings['source_fingerprint'],
                  input_binding_sha256=args.binding_sha256, tool_sha256=sha(Path(__file__).read_bytes()),
                  durable_copy_performed=bool(args.save_durable), input_bytes_unchanged=True,
                  registry_mutations=0)
    output.mkdir()
    with (output/'qa.json').open('x', encoding='utf-8', newline='\n') as out:
        json.dump(result, out, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--binding'); parser.add_argument('--binding-sha256')
    parser.add_argument('--proof-directory'); parser.add_argument('--save-durable', action='store_true')
    parser.add_argument('--registry'); parser.add_argument('--registry-sha256')
    args = parser.parse_args()
    check(Path('/.dockerenv').is_file(), 'existing managed container only')
    if args.self_test:
        check(not any((args.binding,args.binding_sha256,args.proof_directory,args.save_durable,
                       args.registry,args.registry_sha256)), 'artificial mode has no completed paths')
        print(json.dumps(self_test(), sort_keys=True)); return
    check(AUTHORIZED_COMPLETED_METADATA, 'unreleased saver refuses before completed metadata access')
    check(args.binding and args.binding_sha256 and args.proof_directory,
          'explicit later root approval SHA and exclusive QA output required')
    check(bool(args.registry) == bool(args.registry_sha256), 'registry path and identity supplied together')
    completed(args)


if __name__ == '__main__':
    main()

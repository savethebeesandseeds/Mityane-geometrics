#!/usr/bin/env python3
"""SOURCE-only evidence for three inherited cached objects; no archive/data access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path('/embedding')
RUN = ROOT / 'output/runs/rpb-early-mixer-confirmation'
CAPSULE = RUN / 'early-mixer-confirmation-uJ1Ack'
PRIOR = RUN / 'admission/admission-FXJe90'
FIXTURE = RUN / 'admission/source-scope-fixtures/test-api-repair-v2/canonical-scope.json'
FIXTURE_SHA = '5f210bcaa6cfc5f944688c5eb02ad28bc18f132a1dd9fba371199205cf9682a0'
CARD_SHA = '98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea'
CURRENT_SOURCE = 'fa1cbd0e67b4ce6226d6d3a0c5c5c39a71298764e64158dfdbc50decad7e1f12'
PRIOR_SOURCE = 'efa44759e3488147e1c13b2fe16a4c63644321b7e9978f5c4bdd49f9c4af2f1d'
CURRENT_LOG_SHA = '55bf81f6099165fccefc21f9f6ddbe28b16f9497f2c55c7c238835f864225efe'
INVENTORY_SHA = 'd7f6db6dccad050eec984196d28e99b2cc1091c537edf40049ce2be117454846'
BASE_READER = RUN / 'audit-tools/independent-early-mixer-confirmation-released-uJ1Ack-v3/validate_early_mixer_confirmation.py'
BASE_READER_SHA = '6ba1100b5e983a76ddf83369a893e07241a0193f009c1b6c00b340ab270ce0a0'
TEST = 'code/encoders/raw_patch_bottleneck_mae/tests/early_mixer_confirmation_adapter_test.cpp'
OLD_TEST_SHA = 'b38d05143c86a2030e99cf015a7a3183d710a4fce9c3b088f1ba09455add4fcb'
NEW_TEST_SHA = 'f213e17178142ec57f13b7914e31a43006ff90a703e8c572164e989cea0c4f2a'
OBJECT_ROOT = '/opt/cuwacunu_embedding/build/rpb-paired-pooling/code/encoders/raw_patch_bottleneck_mae/'
SCOPES = (
    ('core_writer', 'RPB_PROVENANCE_INPUTS', 'workflow.cpp', 'workflow.o', 'RPB_SOURCE_ID',
     '6a8a526087a7cf14fdd9f9f4d0e35218349f766bc24c7a7c58f4f388681bb7b1'),
    ('curve_training', 'EVALUATION_PROVENANCE_INPUTS', 'learning_curve_adapter.cpp', 'learning_curve_adapter.o', 'EVALUATION_SOURCE_ID',
     '63eae8c8b907885f14a9848d5f203ad6aef09c281489c34bc0b07cf3c7cfd1de'),
    ('early_adapter', 'EARLY_MIXER_RELIABILITY_INPUTS', 'early_mixer_adapter.cpp', 'early_mixer_adapter.o', 'EARLY_MIXER_ADAPTER_SOURCE_ID',
     '4620215f84e2c96c712ec1a1397696503a16a66367232b38cd39da3691f09c23'),
)


def check(ok, why):
    if not ok:
        raise AssertionError(why)


def admit(paths):
    """Complete direct regular/inode matrix before any content hash/read."""
    seen = set()
    for path in paths:
        check(path.is_absolute() and path.is_relative_to(ROOT), 'bounded absolute SOURCE path')
        cursor = path
        while cursor != ROOT.parent:
            check(not cursor.is_symlink(), 'no SOURCE path component aliases')
            cursor = cursor.parent
        check(path.resolve() == path, 'canonical SOURCE path')
        item = path.stat()
        check(stat.S_ISREG(item.st_mode) and item.st_nlink == 1, 'regular single-link SOURCE')
        identity = (item.st_dev, item.st_ino)
        check(identity not in seen, 'distinct admitted SOURCE inodes')
        seen.add(identity)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def canonical(records):
    return ''.join(row['sha256'] + '  ' + row['path'] + '\n' for row in records).encode()


def source_names(rows):
    names = [row['path'] for row in rows]
    check(names == sorted(set(names)) and len(names) == 182, 'complete canonical182 SOURCE matrix')
    for name in names:
        check(name == Path(name).as_posix() and not Path(name).is_absolute() and '..' not in Path(name).parts,
              'canonical relative SOURCE name')
        check(name in ('Makefile', 'setup.sh', 'dependencies.lock') or name.startswith('code/'), 'closed SOURCE namespace')
    return names


def command(log, filename, object_name, macro):
    source = 'code/encoders/raw_patch_bottleneck_mae/src/' + filename
    object_path = OBJECT_ROOT + object_name
    rows = [line for line in log.splitlines()
            if re.search(r'(?:^|\s)-c\s+' + re.escape(source) + r'(?:\s|$)', line)
            and re.search(r'(?:^|\s)-o\s+' + re.escape(object_path) + r'(?:\s|$)', line)]
    check(len(rows) == 1, 'one exact inherited compile target: ' + object_name)
    values = re.findall(r'-D' + re.escape(macro) + r'=\\?"([0-9a-f]{64})\\?"', rows[0])
    check(len(values) == 1, 'one exact macro on inherited compile')
    return rows[0], values[0]


def absent_current(log, filename):
    source = 'code/encoders/raw_patch_bottleneck_mae/src/' + filename
    check(not any(re.search(r'(?:^|\s)-c\s+' + re.escape(source) + r'(?:\s|$)', line)
                  for line in log.splitlines()), 'current inherited source compile genuinely absent')


def build(output):
    check(Path('/.dockerenv').is_file() and ROOT.is_dir(), 'managed Linux container only')
    expected = RUN / 'audit-tools/cached-source-evidence-uJ1Ack-v2'
    check(output == expected and output.is_absolute() and not output.exists(), 'exact absent future evidence leaf')
    cursor = output.parent
    while cursor != ROOT.parent:
        check(cursor.is_dir() and not cursor.is_symlink(), 'canonical existing output ancestors')
        cursor = cursor.parent
    check(output.parent.resolve() == output.parent, 'canonical fixed output parent before reads')
    paths = {
        'current_source_manifest': CAPSULE / 'source-manifest.json',
        'current_source_inputs': CAPSULE / 'source-inputs.sha256',
        'current_admission_manifest': CAPSULE / 'admission/source-manifest.json',
        'current_admission_source_inputs': CAPSULE / 'admission/source-inputs.sha256',
        'current_admission_log': CAPSULE / 'admission/build-and-tests.log',
        'current_admission_passed': CAPSULE / 'admission/passed.json',
        'current_launch_plan': CAPSULE / 'launch-plan.json',
        'current_inventory': CAPSULE / 'artifact-integrity.json',
        'prior_source_manifest': PRIOR / 'source-manifest.json',
        'prior_source_inputs': PRIOR / 'source-inputs.sha256',
        'prior_admission_log': PRIOR / 'build-and-tests.log',
        'prior_admission_plan': PRIOR / 'admission-plan.json',
        'canonical_scope_fixture': FIXTURE,
        'unchanged_released_base_reader': BASE_READER,
    }
    admit(list(paths.values()))
    before = {role: record(path) for role, path in paths.items()}
    check(before['current_admission_log']['sha256'] == CURRENT_LOG_SHA and
          before['current_inventory']['sha256'] == INVENTORY_SHA and
          before['canonical_scope_fixture']['sha256'] == FIXTURE_SHA and
          before['unchanged_released_base_reader']['sha256'] == BASE_READER_SHA, 'exact allowed provenance input pins')
    current = json.loads(paths['current_source_manifest'].read_text())
    prior = json.loads(paths['prior_source_manifest'].read_text())
    prior_plan = json.loads(paths['prior_admission_plan'].read_text())
    passed = json.loads(paths['current_admission_passed'].read_text())
    launch = json.loads(paths['current_launch_plan'].read_text())
    inventory = json.loads(paths['current_inventory'].read_text())
    fixture = json.loads(FIXTURE.read_text())
    check(current['source_fingerprint'] == CURRENT_SOURCE and prior['source_fingerprint'] == PRIOR_SOURCE,
          'exact current and prequality prior SOURCE identities')
    check(prior_plan['quality_generated'] is False and prior_plan['historical_payload_roles'] == 0 and
          prior_plan['source_fingerprint'] == PRIOR_SOURCE and prior_plan['human_card_sha256'] == CARD_SHA,
          'failed attempt was prequality engineering only')
    check(passed['status'] == 'passed' and passed['source_fingerprint'] == CURRENT_SOURCE and
          passed['quality_generated'] is False and passed['source_preserved'] is True and
          passed['human_card_sha256'] == CARD_SHA and passed['log_sha256'] == CURRENT_LOG_SHA and
          launch['admission_log_sha256'] == CURRENT_LOG_SHA and
          launch['admission_passed_sha256'] == before['current_admission_passed']['sha256'],
          'actual passed current admission/launch binding')
    current_rows, prior_rows = current['sources'], prior['sources']
    names = source_names(current_rows)
    check(source_names(prior_rows) == names, 'same prior/current closed SOURCE names')
    for manifest, rows, expected in ((current, current_rows, CURRENT_SOURCE), (prior, prior_rows, PRIOR_SOURCE)):
        check(manifest['protocol'] == 'early-mixer-confirmation-v1' and
              manifest['human_card'] == 'code/evaluation/cards/early_mixer_confirmation_v1.md' and
              hashlib.sha256(canonical(rows)).hexdigest() == expected, 'canonical complete SOURCE manifest identity')
    check(paths['current_source_inputs'].read_bytes() == canonical(current_rows) and
          paths['current_admission_source_inputs'].read_bytes() == canonical(current_rows) and
          json.loads(paths['current_admission_manifest'].read_text())['sources'] == current_rows and
          paths['prior_source_inputs'].read_bytes() == canonical(prior_rows), 'exact manifest and SOURCE-input byte binding')
    current_index, prior_index = ({row['path']: row for row in rows} for rows in (current_rows, prior_rows))
    changed = [name for name in names if current_index[name] != prior_index[name]]
    check(changed == [TEST] and prior_index[TEST]['sha256'] == OLD_TEST_SHA and
          current_index[TEST]['sha256'] == NEW_TEST_SHA, 'only announced test API repair differs')
    source_paths = [CAPSULE / 'source' / name for name in names] + [PRIOR / 'source' / name for name in names]
    admit(source_paths)
    source_before = {str(path): sha(path) for path in source_paths}
    for base, rows in ((CAPSULE, current_rows), (PRIOR, prior_rows)):
        for row in rows:
            path = base / 'source' / row['path']
            check(path.stat().st_size == row['bytes'] and source_before[str(path)] == row['sha256'],
                  'all captured SOURCE actual bytes match manifests')
    old_test = (PRIOR / 'source' / TEST).read_bytes()
    new_test = (CAPSULE / 'source' / TEST).read_bytes()
    check(old_test.count(b'rpb::load_checkpoint(left,oa,') == 1 and
          old_test.count(b'rpb::load_checkpoint(right,ob,') == 1 and
          old_test.replace(b'rpb::load_checkpoint(left,oa,', b'rpb::load_optimizer(left,oa,')
                  .replace(b'rpb::load_checkpoint(right,ob,', b'rpb::load_optimizer(right,ob,') == new_test,
          'exact two compile-API token substitutions only')
    check(fixture['status'] == 'passed' and fixture['make_invocations'] == 1 and
          fixture['builds'] == 0 and fixture['quality_reads'] == 0, 'prequality one-Make SOURCE scope fixture')
    current_log = paths['current_admission_log'].read_text()
    prior_log = paths['prior_admission_log'].read_text()
    binary = passed['compiled_binary']['path']
    link_rows = [line for line in current_log.splitlines() if ' -c ' not in line and
                 re.search(r'(?:^|\s)-o\s+' + re.escape(binary) + r'(?:\s|$)', line)]
    check(len(link_rows) == 1, 'one exact successful current binary link')
    link = link_rows[0]
    markers = ('Early mixer model CUDA admission passed', 'Early mixer CUDA adapter admission passed',
               'Early mixer confirmation CUDA admission passed', 'Fixed feature readout tests passed',
               'Frozen role guard checks passed', 'Container SDK proof:', CURRENT_SOURCE,
               'EARLY_MIXER_ADAPTER_SOURCE_ID=4620215f84e2c96c712ec1a1397696503a16a66367232b38cd39da3691f09c23')
    check(all(marker in current_log for marker in markers) and
          '/embedding/.external/libtorch' not in current_log and 'not found' not in current_log,
          'current actual engineering markers/internal runtime retained')
    scope_records = {}
    for role, scope_name, filename, object_name, macro, expected in SCOPES:
        scope = fixture['scopes'][scope_name]
        members = scope['paths']
        check(members == sorted(set(members)) and len(members) == scope['files'] and
              set(members).issubset(names) and TEST not in members, 'closed unchanged inherited SOURCE subset')
        records = [current_index[name] for name in members]
        check(all(current_index[name] == prior_index[name] for name in members) and
              hashlib.sha256(canonical(records)).hexdigest() == scope['canonical_sha256'] == expected,
              'same exact inherited scope dependency bytes and canonical macro')
        absent_current(current_log, filename)
        compile_line, actual = command(prior_log, filename, object_name, macro)
        check(actual == expected and re.search(r'(?:^|\s)' + re.escape(OBJECT_ROOT + object_name) + r'(?:\s|$)', link),
              'prior exact compiled object/macro reused in successful current link')
        scope_records[role] = {'role': role, 'scope_name': scope_name, 'macro': macro,
            'source_path': 'code/encoders/raw_patch_bottleneck_mae/src/' + filename,
            'object_path': OBJECT_ROOT + object_name, 'canonical_sha256': expected, 'expected_source_id': expected,
            'source_records': records, 'files': len(records), 'prior_compile_command': compile_line,
            'current_compile_absent': True, 'current_link_contains_exact_object': True,
            'prior_current_scope_source_bytes_exact': True}
    mandatory = {}
    for name, filename, object_path, macro in (
        ('main', 'code/evaluation/src/early_mixer_confirmation_main.cpp',
         '/opt/cuwacunu_embedding/build/rpb-paired-pooling/code/evaluation/early_mixer_confirmation_main.o', 'EVALUATION_SOURCE_ID'),
        ('confirmation_adapter', 'code/encoders/raw_patch_bottleneck_mae/src/early_mixer_confirmation_adapter.cpp',
         OBJECT_ROOT + 'early_mixer_confirmation_adapter.o', 'EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID')):
        rows = [line for line in current_log.splitlines()
                if re.search(r'(?:^|\s)-c\s+' + re.escape(filename) + r'(?:\s|$)', line)
                and re.search(r'(?:^|\s)-o\s+' + re.escape(object_path) + r'(?:\s|$)', line)]
        check(len(rows) == 1, 'one mandatory current compile: ' + name)
        values = re.findall(r'-D' + re.escape(macro) + r'=\\?"([0-9a-f]{64})\\?"', rows[0])
        check(values == [CURRENT_SOURCE], 'mandatory current compile macro matches source')
        mandatory[name] = {'source_path': filename, 'object_path': object_path, 'macro': macro,
                           'expected_source_id': CURRENT_SOURCE, 'current_compile_command': rows[0]}
    # Inventory association is SOURCE-only: no quality body read or hash.
    inventory_records = inventory.get('files', inventory.get('artifacts', []))
    check(isinstance(inventory_records, list), 'inventory SOURCE index rows')
    inventory_index = {row['path']: row for row in inventory_records}
    for row in current_rows:
        entry = inventory_index['source/' + row['path']]
        check(entry['sha256'] == row['sha256'] and entry['bytes'] == row['bytes'], 'current captured SOURCE inventory association')
    admit(list(paths.values()))
    admit(source_paths)
    check({role: record(path) for role, path in paths.items()} == before and
          all(sha(path) == source_before[str(path)] for path in source_paths), 'all admitted SOURCE/metadata inputs unchanged')
    evidence = {'schema': 'early-mixer-confirmation-cached-source-evidence-v1', 'status': 'passed',
        'protocol': 'early-mixer-confirmation-cached-source-evidence-v1', 'quality_protocol': 'early-mixer-confirmation-v1',
        'capsule': str(CAPSULE), 'inventory_sha256': INVENTORY_SHA,
        'source_fingerprint': CURRENT_SOURCE, 'human_card_sha256': CARD_SHA,
        'base_released_reader_sha256': BASE_READER_SHA,
        'failed_base_audit': {'checks': 18502, 'archive_decodes': 0,
             'validation_sha256': 'd474f53a0e25622db16e7cda165bf74970cc874bcc8e3f37628edc260a1ac9f3'},
        'inputs': before, 'prior_source_fingerprint': PRIOR_SOURCE,
        'source_files_per_admission': len(names), 'actual_source_files_checked': len(source_paths),
        'source_roles': {
            'current': [{'path': str(CAPSULE / 'source' / row['path']), 'bytes': row['bytes'], 'sha256': row['sha256']} for row in current_rows],
            'prior': [{'path': str(PRIOR / 'source' / row['path']), 'bytes': row['bytes'], 'sha256': row['sha256']} for row in prior_rows]},
        'source_differences': [{'path': TEST, 'prior_sha256': OLD_TEST_SHA, 'current_sha256': NEW_TEST_SHA,
             'exact_two_API_token_substitutions': True, 'outside_all_inherited_scopes': True}],
        'scopes': scope_records, 'current_successful_binary_link': {
            'command': link, 'output': binary, 'inherited_objects': [OBJECT_ROOT + item[3] for item in SCOPES]},
        'current_mandatory_compile_records': mandatory,
        'current_required_runtime_markers': list(markers),
        'prior_admission_status': 'preserved_compile_failed_new_test_only_no_quality',
        'copied_commands_status': 'strict_separately_bound_prior_compile_evidence_not_current_compilation',
        'required_downstream_checks': 'All unchanged base reader typed CP companion/training/snapshot/confirmation scope and numerical checks remain mandatory.',
        'limits': {'object_file_bytes_not_rehashed': True, 'no_claim_of_fresh_current_inherited_compilation': True,
             'compile_link_source_and_CUDA_admission_bound_only': True},
        'quality_score_metadata_reads': 0, 'archive_decodes': 0, 'model_execution': False, 'head_refits': 0,
        'quality_or_audit_reruns': 0, 'input_bytes_preserved': True}
    check(not output.exists() and output.is_absolute() and output.is_relative_to(RUN / 'audit-tools'), 'exclusive bounded evidence leaf')
    output.mkdir()
    destination = output / 'cached-source-evidence.json'
    with destination.open('x', newline='\n') as stream:
        json.dump(evidence, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'path': str(destination), 'sha256': sha(destination),
          'source_files_checked': len(source_paths), 'inherited_scope_files': [x['files'] for x in scope_records.values()],
          'archive_decodes': 0, 'quality_reads': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)

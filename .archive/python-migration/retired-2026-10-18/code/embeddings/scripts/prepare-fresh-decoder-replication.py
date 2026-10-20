#!/usr/bin/env python3
"""Freeze source/card/admission metadata for the fresh decoder replication.

This helper does not generate quality data, read historical quality payloads,
construct a model, fit a head or execute an audit. Source lists are explicit and
fully admitted before hashing; only the newly completed capsule is inventoried.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

PROTOCOL = 'fresh-decoder-replication-v1'
CARD = 'code/embeddings/evaluation/cards/fresh_decoder_replication_v1.md'
ROOT = 'output/runs/rpb-fresh-decoder-replication'
MARKER = 'Fresh decoder replication CUDA admission passed'
OLD_MARKER = 'V7 decoder calibration CUDA admission passed'
FIXED_MARKER = 'Fixed feature readout tests passed'
MASTERS = [9109, 10210, 11311, 12412, 13513]
REQUIRED_SOURCES = {CARD, 'Makefile', 'dependencies.lock',
    'code/embeddings/evaluation/src/fresh_decoder_replication_main.cpp',
    'code/embeddings/scripts/check-fresh-decoder-replication.sh',
    'code/embeddings/scripts/evaluate-fresh-decoder-replication.sh',
    'code/embeddings/scripts/prepare-fresh-decoder-replication.py',
    'code/shared/representation/include/embedding/shared/fixed_feature_readouts.h',
    'code/shared/representation/src/fixed_feature_readouts.cpp', 'code/shared/representation/tests/fixed_feature_readouts_test.cpp',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/include/embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/src/frozen_decoder_calibration.cpp',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/tests/frozen_decoder_calibration_test.cpp'}
SOURCE_SUFFIXES = {'.h', '.cpp', '.py', '.sh', '.md', '.conf', '.lock'}

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(content):
    return hashlib.sha256(content).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def write_new(path, content):
    mode = 'xb' if isinstance(content, bytes) else 'x'
    kwargs = {} if isinstance(content, bytes) else {'encoding': 'utf-8', 'newline': '\n'}
    with path.open(mode, **kwargs) as out:
        out.write(content)
        out.flush()
        os.fsync(out.fileno())

def write_json(path, value):
    write_new(path, json.dumps(value, indent=2, allow_nan=False)+'\n')

def read_json(path):
    require(path.suffix == '.json' and path.is_file() and not path.is_symlink(), 'regular JSON metadata required')
    return json.loads(path.read_text(encoding='utf-8'))

def relative_name(name):
    require(isinstance(name, str) and bool(name) and '\\' not in name and
            not any(c.isspace() for c in name), 'safe relative source name')
    pure = PurePosixPath(name)
    require(not pure.is_absolute() and '..' not in pure.parts and '.' not in pure.parts and
            pure.as_posix() == name, 'normalized relative source name')
    require(name == 'Makefile' or pure.suffix in SOURCE_SUFFIXES, 'source-only suffix required')
    return pure

def admit_paths(repo, names):
    """Entire matrix/path/inode pass; deliberately performs no payload hashes."""
    require(names and names == sorted(names) and len(names) == len(set(names)), 'unique sorted explicit Make source list')
    require(REQUIRED_SOURCES.issubset(names), 'card/main/launchers/helper/build metadata must be source-bound')
    admitted, inodes = [], set()
    for name in names:
        relative_name(name)
        lexical = repo/name
        resolved = lexical.resolve(strict=True)
        require(resolved == lexical and resolved.is_relative_to(repo) and resolved.is_file(), 'source path redirects/bounds')
        require(not any((repo/Path(*Path(name).parts[:i])).is_symlink() for i in range(1, len(Path(name).parts)+1)), 'source symlink')
        stat = resolved.stat()
        inode = (stat.st_dev, stat.st_ino)
        require(inode not in inodes, 'source aliases are forbidden')
        inodes.add(inode)
        admitted.append((name, resolved, stat.st_size))
    return admitted

def source_records(repo, names, card_sha):
    admitted = admit_paths(repo, names)
    records = [{'path': name, 'bytes': size, 'sha256': file_sha(path)} for name, path, size in admitted]
    require(next(x['sha256'] for x in records if x['path'] == CARD) == card_sha, 'frozen human card differs')
    return records

def manifest_bytes(records):
    return ''.join(x['sha256']+'  '+x['path']+'\n' for x in records).encode('utf-8')

def bounded_directory(repo, path, kind):
    resolved = path.resolve(strict=True)
    require(resolved == path and not path.is_symlink() and path.is_dir(), 'regular exclusive target directory')
    parent = repo/ROOT/('admission' if kind == 'admission' else '')
    prefix = 'admission-' if kind == 'admission' else 'fresh-decoder-replication-'
    require(resolved.parent == parent and resolved.name.startswith(prefix), 'target protocol directory bounds')
    return resolved

def copy_sources(repo, target, records):
    source = target/'source'
    source.mkdir()
    for row in records:
        dest = source/row['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_new(dest, (repo/row['path']).read_bytes())
        require(dest.stat().st_size == row['bytes'] and file_sha(dest) == row['sha256'], 'source copy mutation')
    manifest = manifest_bytes(records)
    source_id = digest(manifest)
    write_new(target/'source-inputs.sha256', manifest)
    write_new(target/'source-fingerprint.txt', source_id+'\n')
    write_json(target/'source-manifest.json', {'protocol': PROTOCOL, 'algorithm': 'sha256-source-manifest-v1',
        'human_card': CARD, 'source_fingerprint': source_id, 'sources': records})
    return source_id

def source_state(repo, target, card_sha):
    metadata = read_json(target/'source-manifest.json')
    require(metadata['protocol'] == PROTOCOL and metadata['human_card'] == CARD, 'source metadata protocol')
    records = metadata['sources']
    names = [x['path'] for x in records]
    require(source_records(repo, names, card_sha) == records, 'current sources changed')
    source = target/'source'
    require({p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() or p.is_symlink()} == set(names), 'captured source matrix has extra/missing files')
    # Admit the complete captured list before reading any copied source byte.
    captured = admit_paths(source, names)
    for row, (_, path, size) in zip(records, captured):
        require(row['bytes'] == size and row['sha256'] == file_sha(path), 'captured source changed')
    manifest = manifest_bytes(records)
    require((target/'source-inputs.sha256').read_bytes() == manifest, 'canonical source manifest changed')
    source_id = digest(manifest)
    require(metadata['source_fingerprint'] == source_id and
            (target/'source-fingerprint.txt').read_text().strip() == source_id, 'source aggregate identity')
    return metadata, source_id

def binary_metadata(binary):
    path = Path(binary)
    resolved = path.resolve(strict=True)
    require(path.is_absolute() and resolved == path and not path.is_symlink() and path.is_file() and
            path.is_relative_to(Path('/opt/mityane-geometrics/embeddings/build')) and
            path.name == 'embedding_fresh_decoder_replication', 'named container binary path')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path)}

def admission_paths(admission, sealed):
    metadata = read_json(admission/'source-manifest.json')
    expected = {'source/'+x['path'] for x in metadata['sources']}
    expected |= {'source-inputs.sha256', 'source-fingerprint.txt', 'source-manifest.json',
                 'admission-plan.json', 'build-and-tests.log'}
    if sealed:
        expected |= {'source-preserved-after.json', 'passed.json'}
    paths = sorted(x for x in admission.rglob('*') if x.is_file() or x.is_symlink())
    require({x.relative_to(admission).as_posix() for x in paths} == expected, 'closed admission source/metadata/log matrix')
    for path in paths:
        require(not path.is_symlink() and path.resolve(strict=True) == path and path.is_relative_to(admission), 'admission source redirect')
        require(path.suffix not in ('.pt', '.bin', '.npz'), 'no model/tensor admission copies')
    return paths

def freeze_admission(repo, target, names, card_sha):
    require(not any(target.iterdir()), 'new admission must be empty')
    records = source_records(repo, names, card_sha)
    source_id = copy_sources(repo, target, records)
    write_json(target/'admission-plan.json', {'protocol': PROTOCOL, 'created_utc': utc(),
        'human_card': CARD, 'human_card_sha256': card_sha, 'source_fingerprint': source_id,
        'quality_generated': False, 'historical_quality_inputs': [], 'historical_payload_roles': 0,
        'scope': 'engineering admission only; no declared fresh quality generation'})
    source_state(repo, target, card_sha)
    return source_id

def admission_record(repo, admission, card_sha, compiled_id, binary, sealed=False):
    admission_paths(admission, sealed)
    metadata, source_id = source_state(repo, admission, card_sha)
    require(compiled_id == source_id, 'compiled binary/source fingerprint differs')
    plan = read_json(admission/'admission-plan.json')
    require(plan['source_fingerprint'] == source_id and plan['human_card_sha256'] == card_sha and
            plan['quality_generated'] is False and plan['historical_quality_inputs'] == [], 'admission plan binding')
    log = admission/'build-and-tests.log'
    require(log.is_file() and not log.is_symlink(), 'regular actual CUDA admission log')
    text = log.read_text(encoding='utf-8', errors='strict')
    require(MARKER in text and OLD_MARKER in text and FIXED_MARKER in text and source_id in text,
            'source-bound new/legacy actual CUDA and fixed-readout admission markers')
    return {'protocol': PROTOCOL, 'status': 'passed', 'created_utc': utc(), 'source_fingerprint': source_id,
        'human_card_sha256': card_sha, 'source_preserved': True, 'log_sha256': file_sha(log),
        'compiled_binary': binary_metadata(binary),
        'quality_generated': False, 'historical_payload_roles': 0, 'head_fits': 0,
        'gates': ['actual CUDA paired encoder/frozen-decoder lifecycle contract on separate engineering fixtures',
            'declared quality recipe is encoder512 plus fresh decoder128; engineering admission does not train those quality cohorts',
            'frozen encoder/scaler/native outputs; original query and direct/split fresh decoder AdamW parity',
            'legacy v7 typed decoder adapter preserved', 'fresh initialization/data/streams, immutable snapshots and bounded counters']}

def admit(repo, target, card_sha, compiled_id, binary):
    record = admission_record(repo, target, card_sha, compiled_id, binary)
    write_json(target/'source-preserved-after.json', {'protocol': PROTOCOL, 'source_preserved': True,
        'source_fingerprint': compiled_id, 'created_utc': utc()})
    write_json(target/'passed.json', record)
    return record

def check_plan(plan):
    require(plan['protocol'] == PROTOCOL and plan['master_seeds'] == MASTERS, 'fixed fresh protocol/masters')
    require(plan['tags'] == ['RPB-v4', 'RPB-v7'] and plan['encoder_updates'] == 512 and
            plan['decoder_updates'] == 128, 'fixed model and encoder/decoder budgets')
    require(plan['train_pairs'] == 128 and plan['validation_pairs'] == 64 and plan['test_pairs'] == 0,
            'fixed fresh development-only cohort sizes')
    require(plan['prior_quality_input_roles'] == [] and plan['quality_generated'] is False and
            plan['testing_accessed'] is False and plan['stress_accessed'] is False, 'no historical/test/stress/generation plan')

def copy_admission(repo, admission, target, card_sha, compiled_id, binary):
    current = admission_record(repo, admission, card_sha, compiled_id, binary, sealed=True)
    passed = read_json(admission/'passed.json')
    for key in ('protocol', 'status', 'source_fingerprint', 'human_card_sha256', 'source_preserved',
                'log_sha256', 'compiled_binary', 'quality_generated', 'historical_payload_roles', 'head_fits', 'gates'):
        require(passed[key] == current[key], 'passed admission evidence mismatch: '+key)
    # Admission contains only source/metadata/logs; never copy model/quality assets.
    paths = admission_paths(admission, sealed=True)
    copied = target/'admission'
    copied.mkdir()
    for path in paths:
        dest = copied/path.relative_to(admission)
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        write_new(dest, content)
        require(file_sha(dest) == digest(content), 'admission byte copy differs')
    require(file_sha(copied/'passed.json') == file_sha(admission/'passed.json'), 'passed record copy differs')
    return passed

def freeze_run(repo, target, names, card_sha, admission, compiled_id, binary):
    require(set(x.name for x in target.iterdir()) == {'recipe-plan.json'}, 'new run only contains source-only plan')
    plan = read_json(target/'recipe-plan.json')
    check_plan(plan)
    records = source_records(repo, names, card_sha)
    source_id = digest(manifest_bytes(records))
    require(source_id == compiled_id, 'run binary/source/card identity mismatch')
    passed = copy_admission(repo, admission, target, card_sha, compiled_id, binary)
    require(copy_sources(repo, target, records) == compiled_id, 'run source capture differs')
    write_json(target/'input-role-plan.json', {'protocol': PROTOCOL, 'historical_payload_roles': 0,
        'prior_quality_input_roles': [], 'fresh_quality_inputs': 'generated only after launch plan inside measured main'})
    write_json(target/'launch-plan.json', {'protocol': PROTOCOL, 'created_utc': utc(), 'status': 'frozen_before_quality_generation',
        'human_card': CARD, 'human_card_sha256': card_sha, 'source_fingerprint': source_id,
        'recipe_plan_sha256': file_sha(target/'recipe-plan.json'), 'recipe': plan,
        'admission_passed_sha256': file_sha(target/'admission/passed.json'),
        'admission_log_sha256': passed['log_sha256'], 'compiled_binary': passed['compiled_binary'], 'historical_payload_roles': 0,
        'argv': [binary, '--output', str(target/'results'), '--admission-log', str(target/'admission/build-and-tests.log'),
                 '--admission-sha256', passed['log_sha256']],
        'testing_accessed': False, 'stress_accessed': False, 'quality_generated': False})
    source_state(repo, target, card_sha)
    return source_id

def finalize(repo, target, card_sha, compiled_id):
    metadata, source_id = source_state(repo, target, card_sha)
    require(source_id == compiled_id, 'compiled source differs after quality run')
    launch = read_json(target/'launch-plan.json')
    require(launch['source_fingerprint'] == source_id and launch['human_card_sha256'] == card_sha and
            launch['historical_payload_roles'] == 0, 'launch source/card/input boundary')
    check_plan(read_json(target/'recipe-plan.json'))
    require(file_sha(target/'recipe-plan.json') == launch['recipe_plan_sha256'], 'recipe changed after generation')
    passed = read_json(target/'admission/passed.json')
    require(file_sha(target/'admission/passed.json') == launch['admission_passed_sha256'] and
            file_sha(target/'admission/build-and-tests.log') == launch['admission_log_sha256'] and
            passed['source_fingerprint'] == source_id and passed['status'] == 'passed', 'copied admission changed')
    admission_paths(target/'admission', sealed=True)
    require(binary_metadata(launch['argv'][0]) == launch['compiled_binary'] == passed['compiled_binary'], 'tested binary bytes changed during measurement')
    complete = read_json(target/'results/complete.json')
    report = read_json(target/'results/report.json')
    require(complete['protocol'] == report['protocol'] == PROTOCOL and complete['status'] == 'complete', 'new measurement not complete')
    require(complete['testing_accessed'] is False and complete['stress_accessed'] is False, 'fresh development scope violation')
    write_json(target/'source-preserved-after.json', {'protocol': PROTOCOL, 'source_preserved': True,
        'source_fingerprint': source_id, 'historical_payload_roles': 0, 'created_utc': utc(),
        'copied_admission_preserved': True, 'recipe_preserved': True})
    # Inventory only this NEW capsule. Admit all regular paths first, then hash.
    paths = sorted(x for x in target.rglob('*') if x.is_file() or x.is_symlink())
    for path in paths:
        require(path.resolve(strict=True) == path and not path.is_symlink() and path.is_relative_to(target), 'capsule redirect')
    files = [{'path': path.relative_to(target).as_posix(), 'bytes': path.stat().st_size, 'sha256': file_sha(path)} for path in paths]
    record = {'schema_version': 1, 'protocol': PROTOCOL, 'created_utc': utc(),
        'inventory_excludes_itself': True, 'checksum_algorithm': 'sha256-file-bytes',
        'file_count': len(files), 'total_bytes': sum(x['bytes'] for x in files), 'files': files}
    write_json(target/'artifact-integrity.json', record)
    return {'protocol': PROTOCOL, 'status': 'complete', 'source_fingerprint': source_id,
        'inventory_sha256': file_sha(target/'artifact-integrity.json'), 'file_count': len(files), 'total_bytes': record['total_bytes']}

def self_test():
    # Metadata-only actual plan/parser fixtures; no source/quality payload access.
    plan = {'protocol': PROTOCOL, 'master_seeds': MASTERS, 'tags': ['RPB-v4', 'RPB-v7'],
            'encoder_updates': 512, 'decoder_updates': 128, 'train_pairs': 128, 'validation_pairs': 64,
            'test_pairs': 0, 'prior_quality_input_roles': [], 'quality_generated': False,
            'testing_accessed': False, 'stress_accessed': False}
    check_plan(plan)
    negatives = 0
    for key, value in [('protocol', 'old'), ('master_seeds', MASTERS[:-1]), ('encoder_updates', 128),
                       ('decoder_updates', 512), ('test_pairs', 64), ('prior_quality_input_roles', ['old/model.pt']),
                       ('quality_generated', True), ('testing_accessed', True), ('stress_accessed', True)]:
        wrong = dict(plan);wrong[key] = value
        try:
            check_plan(wrong)
        except AssertionError:
            negatives += 1
        else:
            raise AssertionError('unsafe actual recipe accepted: '+key)
    require(manifest_bytes([{'path': 'code/source.cpp', 'sha256': 'a'*64}]) == ('a'*64+'  code/source.cpp\n').encode(), 'GNU SHA manifest law')
    for name in ('../model.pt', '/model.pt', 'code/model.pt', 'code\\source.cpp', 'code/a b.cpp'):
        try:
            relative_name(name)
        except AssertionError:
            negatives += 1
        else:
            raise AssertionError('unsafe actual source name accepted')
    hashes = []
    original_hash = globals()['file_sha']
    def count_hash(path):
        hashes.append(str(path))
        return original_hash(path)
    with tempfile.TemporaryDirectory(prefix='fresh-decoder-source-only-') as temporary:
        fixture = Path(temporary).resolve(strict=True)
        names = sorted(REQUIRED_SOURCES)
        for name in names:
            path = fixture/name;path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('source-only fixture\n', encoding='utf-8')
        globals()['file_sha'] = count_hash
        try:
            try:
                source_records(fixture, sorted(names+['zz_missing.cpp']), digest(b'source-only fixture\n'))
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('late missing source accepted')
            require(hashes == [], 'a source hash occurred before the entire matrix was admitted')
            records = source_records(fixture, names, digest(b'source-only fixture\n'))
            require(len(records) == len(names) and len(hashes) == len(names), 'actual valid source matrix/hash fixture')
            hashes.clear()
            os.link(fixture/'Makefile', fixture/'zz_alias.cpp')
            try:
                source_records(fixture, sorted(names+['zz_alias.cpp']), digest(b'source-only fixture\n'))
            except AssertionError:
                pass
            else:
                raise AssertionError('late hardlink alias accepted')
            require(hashes == [], 'an alias matrix hashed a source')
        finally:
            globals()['file_sha'] = original_hash
    return {'protocol': PROTOCOL, 'source_only_self_test': 'passed', 'actual_plan_positive': 1,
            'negative_fixtures': negatives, 'whole_matrix_before_hash_fixtures': 3,
            'quality_payload_reads': 0, 'quality_generation': 0}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['role-plan', 'self-test', 'freeze-admission', 'admit', 'freeze-run', 'finalize'])
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--target')
    parser.add_argument('--card-sha256')
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--compiled-source-id')
    parser.add_argument('--admission')
    parser.add_argument('--binary')
    args = parser.parse_args()
    require(Path('/.dockerenv').exists(), 'existing managed Linux container required')
    if args.mode == 'role-plan':
        print(json.dumps({'protocol': PROTOCOL, 'master_seeds': MASTERS, 'historical_payload_roles': 0,
                          'prior_quality_input_roles': [], 'quality_generated': False, 'testing_accessed': False, 'stress_accessed': False}, indent=2))
        return
    if args.mode == 'self-test':
        print(json.dumps(self_test(), indent=2));return
    require(args.target and args.card_sha256 and re.fullmatch('[0-9a-f]{64}', args.card_sha256), 'explicit target/frozen card SHA required')
    repo = Path(args.repo_root).resolve(strict=True)
    require(repo.is_dir(), 'repository root')
    target = bounded_directory(repo, Path(args.target), 'admission' if args.mode in ('freeze-admission', 'admit') else 'run')
    if args.mode in ('admit', 'freeze-run', 'finalize'):
        require(args.compiled_source_id and re.fullmatch('[0-9a-f]{64}', args.compiled_source_id), 'compiled source SHA required')
    if args.mode == 'freeze-admission':
        result = {'source_fingerprint': freeze_admission(repo, target, args.sources, args.card_sha256)}
    elif args.mode == 'admit':
        require(args.binary, 'explicit tested container binary required')
        result = admit(repo, target, args.card_sha256, args.compiled_source_id, args.binary)
    elif args.mode == 'freeze-run':
        require(args.admission and args.binary, 'explicit matching admission/binary required')
        admission = bounded_directory(repo, Path(args.admission), 'admission')
        result = {'source_fingerprint': freeze_run(repo, target, args.sources, args.card_sha256, admission, args.compiled_source_id, args.binary)}
    else:
        result = finalize(repo, target, args.card_sha256, args.compiled_source_id)
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()

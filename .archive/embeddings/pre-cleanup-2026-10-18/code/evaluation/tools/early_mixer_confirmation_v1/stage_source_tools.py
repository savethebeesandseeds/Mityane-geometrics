#!/usr/bin/env python3
"""Stage reviewed confirmation SOURCE and a fresh blocked fixture/seal triple.

Run only in the existing managed container. This creates new SOURCE/fixture
evidence; it does not build, generate quality, release a measured reader, decode
archives, emit reports or run models/heads. --self-test is artificial only.
"""
import argparse
import ast
import builtins
import datetime
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import symtable
import tempfile

sys.dont_write_bytecode = True
REPO = Path('/embedding')
ROOT = REPO / 'output/runs/rpb-early-mixer-confirmation'
SOURCE = REPO / 'code/evaluation/tools/early_mixer_confirmation_v1'
PROTOCOL = 'early-mixer-confirmation-v1'
CARD_SHA = '98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea'
CODEC = REPO / 'output/runs/archive-controls/input-audit-20261006T191016Z-d7b99529/validate_phase1.py'
CODEC_SHA = '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d'
NAMES = {'validate_early_mixer_confirmation.py', 'validate_phase1.py',
         'release_confirmation_reader_v3.py'}


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    require(path.is_absolute() and path.resolve(strict=True) == path and
            not path.is_symlink() and stat.S_ISREG(path.stat().st_mode) and
            path.stat().st_nlink == 1, 'canonical regular SOURCE with no aliases')
    require(all(not p.is_symlink() for p in path.parents), 'no redirected ancestor')


def admit(paths, check=regular):
    require(len(paths) == len(set(paths)), 'unique SOURCE matrix')
    for path in paths:
        check(path)


def parse(raw):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(AssertionError('nonfinite JSON ' + x)))


def blocked(raw):
    for name in (b'REVIEWED_SCHEMA', b'MEASURED_IMPLEMENTATION'):
        require(len(re.findall(rb'(?m)^' + name + rb' = False$', raw)) == 1 and
                not re.findall(rb'(?m)^' + name + rb' = True$', raw),
                'exact FALSE reader flags')


def derive_release(raw, relative, pins):
    require(relative.startswith('audit-tools/independent-early-mixer-confirmation-staged-') and
            '..' not in relative and '\\' not in relative, 'closed fresh reader leaf')
    require(set(pins) == {'validate_early_mixer_confirmation.py', 'source-fixtures.json', 'reader-seal.json'} and
            all(re.fullmatch('[a-f0-9]{64}', value) for value in pins.values()), 'three exact fresh pins')
    text = raw.decode('utf-8')
    replacements = [
        (r"(?m)^ORIGINAL = ROOT / '[^']+'$", "ORIGINAL = ROOT / " + repr(relative)),
        (r'(?ms)^PINS = \{.*?^\}', 'PINS = ' + repr(pins)),
    ]
    for pattern, replacement in replacements:
        text, count = re.subn(pattern, lambda _: replacement, text)
        require(count == 1, 'one top-level release declaration')
    new = text.encode('utf-8')
    def functions(source):
        tree = ast.parse(source)
        return [ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    require(functions(new) == functions(raw), 'all reviewed release function ASTs unchanged')
    before, after = ast.parse(raw).body, ast.parse(new).body
    require(len(before) == len(after), 'same release module structure')
    differences = []
    for left, right in zip(before, after):
        if ast.dump(left, include_attributes=False) != ast.dump(right, include_attributes=False):
            require(isinstance(left, ast.Assign) and isinstance(right, ast.Assign) and
                    len(left.targets) == len(right.targets) == 1 and
                    isinstance(left.targets[0], ast.Name) and isinstance(right.targets[0], ast.Name),
                    'only named global declarations can differ')
            differences.append(left.targets[0].id)
            require(left.targets[0].id == right.targets[0].id, 'same changed declaration')
    require(differences == ['ORIGINAL', 'PINS'], 'only path and three pins differ')
    return new


def write_new(path, raw):
    with path.open('xb') as out:
        out.write(raw)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def materialize_sources(pairs):
    require(len(pairs) == len({path for path, _ in pairs}), 'unique fixed SOURCE destinations')
    for path, _ in pairs:
        require(path.is_absolute() and all(not p.is_symlink() for p in [path, *path.parents]),
                'no redirected fixed SOURCE destinations')
        if path.exists():
            regular(path)
    # Validate every existing source before copying any absent source. A
    # conflicting/partial existing SOURCE is never replaced or repaired here.
    for path, raw in pairs:
        if path.exists():
            require(path.read_bytes() == raw, 'existing fixed SOURCE bytes exact; conflicts preserved')
    for path, raw in pairs:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            require(path.parent.resolve(strict=True) == path.parent, 'canonical fixed SOURCE parent')
            write_new(path, raw)
        regular(path); require(path.read_bytes() == raw, 'staged fixed SOURCE bytes exact')


def self_test():
    negative = 0
    def reject(call):
        nonlocal negative
        try:
            call()
        except (AssertionError, FileNotFoundError):
            negative += 1
        else:
            raise AssertionError('invalid artificial SOURCE accepted')
    raw = b'REVIEWED_SCHEMA = False\nMEASURED_IMPLEMENTATION = False\n'
    blocked(raw)
    reject(lambda: blocked(raw.replace(b'False', b'True', 1)))
    reject(lambda: blocked(raw + b'REVIEWED_SCHEMA = False\n'))
    reject(lambda: parse('{"a":1,"a":2}'))
    reject(lambda: parse('{"a":NaN}'))
    template = b"from pathlib import Path\nROOT = Path('/x')\nORIGINAL = ROOT / 'old'\nPINS = {\n    'old': 'a',\n}\ndef unchanged():\n    return ORIGINAL, PINS\n"
    pins = {name: 'a' * 64 for name in ('validate_early_mixer_confirmation.py', 'source-fixtures.json', 'reader-seal.json')}
    derived = derive_release(template, 'audit-tools/independent-early-mixer-confirmation-staged-fixture', pins)
    require(b'def unchanged():' in derived, 'actual declaration-only derivation branch')
    reject(lambda: derive_release(template, '../escape', pins))
    reject(lambda: derive_release(template, 'audit-tools/independent-early-mixer-confirmation-staged-fixture', {}))
    seen = []
    def late(path):
        seen.append(path)
        require(path != 'late-invalid', 'late artificial invalid path')
    reject(lambda: admit(['first', 'late-invalid'], late))
    require(seen == ['first', 'late-invalid'], 'whole admission precedes any hash phase')
    with tempfile.TemporaryDirectory(prefix='confirmation-source-stage-fixture-') as directory:
        base = Path(directory).resolve()
        first = base / 'first'; first.write_bytes(b'artificial')
        alias = base / 'alias'; alias.symlink_to(first)
        reject(lambda: regular(alias))
        absent = base / 'not-created'; bad = base / 'conflict'; bad.write_bytes(b'conflict')
        reject(lambda: materialize_sources([(absent, b'expected'), (bad, b'expected')]))
        require(not absent.exists() and bad.read_bytes() == b'conflict', 'late conflict causes zero copies')
        materialize_sources([(absent, b'expected'), (first, b'artificial')])
        require(absent.read_bytes() == b'expected', 'absent SOURCE creation and identical reuse')
    tree = symtable.symtable(Path(__file__).read_text(), str(Path(__file__)), 'exec')
    known = set(tree.get_identifiers()) | set(dir(builtins)) | {'__file__', '__name__'}
    stack = [tree]; scopes = references = 0; unresolved = set()
    while stack:
        node = stack.pop(); scopes += 1; stack.extend(node.get_children())
        for symbol in node.get_symbols():
            if symbol.is_referenced() and symbol.is_global():
                references += 1
                if symbol.get_name() not in known:
                    unresolved.add(symbol.get_name())
    require(not unresolved, 'all conditional stager globals resolve')
    return {'protocol': PROTOCOL, 'status': 'passed', 'artificial_only': True,
            'negative_cases': negative, 'quality_or_tensor_reads': 0,
            'reader_imports_or_fixture_reruns': 0, 'real_staging_or_releases': 0,
            'scopes': scopes, 'global_references': references, 'unresolved_globals': sorted(unresolved)}


def stage(target):
    require(SOURCE.resolve(strict=True) == SOURCE, 'tracked SOURCE directory')
    manifest = SOURCE / 'source-manifest.json'; regular(manifest)
    manifest_raw = manifest.read_bytes(); value = parse(manifest_raw)
    require(set(value) == {'protocol', 'files'} and value['protocol'] == PROTOCOL,
            'closed tracked SOURCE manifest')
    rows = value['files']
    require(len(rows) == len(NAMES) and {row['name'] for row in rows} == NAMES,
            'exact three reviewed SOURCE files')
    require(all(set(row) == {'name', 'bytes', 'sha256', 'original_source_role'} and
                type(row['bytes']) is int and row['bytes'] > 0 and
                re.fullmatch('[a-f0-9]{64}', row['sha256']) for row in rows), 'SOURCE manifest fields')
    paths = [SOURCE / row['name'] for row in rows]
    admit(paths)
    contents = {row['name']: path.read_bytes() for row, path in zip(rows, paths)}
    for row in rows:
        require(len(contents[row['name']]) == row['bytes'] and
                digest(contents[row['name']]) == row['sha256'], 'exact reviewed tracked SOURCE bytes')
    blocked(contents['validate_early_mixer_confirmation.py'])
    require(digest(contents['validate_phase1.py']) == CODEC_SHA, 'pinned codec SOURCE')
    require(target.is_absolute() and target.parent == ROOT / 'audit-tools' and
            target.parent.resolve(strict=True) == target.parent and
            target.name.startswith('independent-early-mixer-confirmation-staged-') and
            not target.exists() and not target.is_symlink(), 'exclusive fresh SOURCE stage')
    # Preserved sources use these fixed SOURCE-only paths. Stage definitions,
    # never execute their historical main or release/report branches.
    fixed = [(CODEC, contents['validate_phase1.py'])]
    materialize_sources(fixed)
    target.mkdir(exist_ok=False)
    reader = target / 'validate_early_mixer_confirmation.py'; write_new(reader, contents[reader.name])
    fixture = target / 'source-fixtures.json'
    subprocess.run([sys.executable, '-B', str(reader), '--self-test', '--fixture-output', str(fixture)],
                   cwd=REPO, check=True)
    regular(fixture); fixture_raw = fixture.read_bytes(); tested = parse(fixture_raw)
    require(tested['status'] == 'passed' and not tested['measured_execution_enabled'] and
            tested['archive_payload_reads'] == tested['quality_payload_hashes'] == 0 and
            tested['source_sha256'] == digest(contents[reader.name]), 'fresh source-only fixture boundary')
    seal = {'protocol': PROTOCOL, 'status': 'passed',
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_sha256': digest(contents[reader.name]), 'fixtures_sha256': digest(fixture_raw),
            'human_card_sha256': CARD_SHA, 'reviewed_source_manifest_sha256': digest(manifest_raw),
            'reviewed_schema_flag': False, 'measured_implementation_flag': False,
            'measured_execution_enabled': False, 'archive_payload_reads': 0,
            'quality_payload_hashes': 0, 'fixture_checks': tested['checks'],
            'fixture_negative_cases': tested['negative_cases'],
            'review_scope': 'byte-exact reviewed SOURCE; fresh artificial fixtures, no historical seal reuse'}
    seal_path = target / 'reader-seal.json'; seal_raw = json_bytes(seal); write_new(seal_path, seal_raw)
    pins = {reader.name: digest(contents[reader.name]), fixture.name: digest(fixture_raw),
            seal_path.name: digest(seal_raw)}
    relative = target.relative_to(ROOT).as_posix()
    release = derive_release(contents['release_confirmation_reader_v3.py'], relative, pins)
    release_path = target / 'release_staged_reader.py'; write_new(release_path, release)
    paths_file = target / 'reader.paths'; write_new(paths_file, ('\n'.join(map(str, [reader, fixture, seal_path])) + '\n').encode())
    write_new(target / 'stage-proof.json', json_bytes({
        'protocol': PROTOCOL, 'status': 'passed', 'source_manifest_sha256': digest(manifest_raw),
        'fresh_reader_triple': pins, 'release_template_sha256': digest(contents['release_confirmation_reader_v3.py']),
        'derived_release_sha256': digest(release), 'release_changes_only_ORIGINAL_and_PINS': True,
        'all_release_function_ASTs_unchanged': True, 'codec_sha256': CODEC_SHA,
        'quality_or_tensor_reads': 0, 'model_or_head_execution': False,
        'reader_execution_enabled': False, 'historical_fixture_or_seal_reused': False}))
    require(manifest.read_bytes() == manifest_raw, 'tracked manifest unchanged')
    for path in paths:
        regular(path); require(path.read_bytes() == contents[path.name], 'tracked SOURCE bytes unchanged')
    print(json.dumps({'status': 'staged-blocked', 'reader_paths': str(paths_file),
                      'release_tool': str(release_path), 'triple': pins, 'quality_generated': False}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--stage-directory')
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'existing managed container only')
    if args.self_test:
        require(not args.stage_directory, 'artificial mode cannot stage SOURCE')
        print(json.dumps(self_test(), sort_keys=True)); return
    require(args.stage_directory, 'explicit exclusive SOURCE stage directory')
    stage(Path(args.stage_directory))


if __name__ == '__main__':
    main()

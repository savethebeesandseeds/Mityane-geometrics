#!/usr/bin/env python3
"""Explicit post-completion release; source-only fixtures never resolve a capsule.

Do not invoke the release branch before the coordinator authorizes the completed
inventory. This tool reads only the supplied inventory, completion metadata and
captured prequality reader triple. It never imports a reader or archive codec.
"""
import argparse
import datetime
import hashlib
import json
import re
import stat
import sys
import tempfile
from pathlib import Path
sys.dont_write_bytecode = True

PROTOCOL = 'early-mixer-confirmation-v1'
ROOT = Path('/embedding/output/runs/rpb-early-mixer-confirmation')
ORIGINAL = ROOT / 'audit-tools/independent-early-mixer-confirmation-20261009-v3'
SOURCE_NAME = 'validate_early_mixer_confirmation.py'
PINS = {
    SOURCE_NAME: 'b114d0e1db46fd7dd9bba16594eca220b0703269e82f1bd2dea0c399088b4516',
    'source-fixtures.json': '35065efa3c165c57581d797848f93365f649f6fee1414a76bf1b70f26f3cc60c',
    'reader-seal.json': '56cada04dead0073ec4005fdbfc2548316ab4f5c851ea4f100a9fb9b47188745',
}


def require(ok, why):
    if not ok:
        raise AssertionError(why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
            'explicit absolute canonical path, no aliases')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'regular unaliased file')
    return path


def flags_only(raw):
    released = raw
    for name in (b'REVIEWED_SCHEMA', b'MEASURED_IMPLEMENTATION'):
        false = rb'(?m)^' + name + rb' = False$'
        true = rb'(?m)^' + name + rb' = True$'
        require(len(re.findall(false, released)) == 1 and not re.findall(true, released),
                'one original disabled assignment for ' + name.decode())
        released = re.sub(false, name + b' = True', released)
    restored = released
    for name in (b'REVIEWED_SCHEMA', b'MEASURED_IMPLEMENTATION'):
        true = rb'(?m)^' + name + rb' = True$'
        require(len(re.findall(true, restored)) == 1, 'one released assignment')
        restored = re.sub(true, name + b' = False', restored)
    require(restored == raw, 'exact two-line reverse-byte proof')
    return released


def self_test():
    raw = b'REVIEWED_SCHEMA = False\nMEASURED_IMPLEMENTATION = False\n# False stays untouched\nx = "True"\n'
    expected = b'REVIEWED_SCHEMA = True\nMEASURED_IMPLEMENTATION = True\n# False stays untouched\nx = "True"\n'
    require(flags_only(raw) == expected, 'actual anchored flag-only branch')
    negative = 0
    for bad in (raw + b'REVIEWED_SCHEMA = False\n', raw.replace(b'REVIEWED_SCHEMA = False', b'REVIEWED_SCHEMA = True'),
                raw.replace(b'MEASURED_IMPLEMENTATION = False\n', b''), raw.replace(b'\n', b'\r\n')):
        try:
            flags_only(bad)
        except AssertionError:
            negative += 1
        else:
            raise AssertionError('invalid release source accepted')
    return {'status': 'passed', 'negative_cases': negative, 'quality_metadata_reads': 0,
            'archive_payload_reads': 0, 'model_head_optimizer_or_SVD_execution': False}


def release(capsule, output, inventory_sha):
    require(Path('/.dockerenv').is_file(), 'use the existing managed container')
    capsule = Path(capsule)
    require(capsule.is_absolute() and capsule.parent == ROOT and capsule.resolve(strict=True) == capsule and
            capsule.name.startswith('early-mixer-confirmation-') and not capsule.is_symlink(),
            'one explicit completed protocol capsule')
    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / 'audit-tools' and
            output.parent.resolve(strict=True) == output.parent and not output.exists() and
            output.name.startswith('independent-early-mixer-confirmation-') and 'released' in output.name,
            'new exclusive released leaf outside the measured capsule')
    require(re.fullmatch('[0-9a-f]{64}', inventory_sha) is not None, 'explicit inventory identity')
    # Admit every metadata source before reading any bytes; no tensor record is read.
    paths = [capsule / 'artifact-integrity.json', capsule / 'results/complete.json']
    names = (SOURCE_NAME, 'source-fixtures.json', 'reader-seal.json')
    captured_names = ('source.py', 'source-fixtures.json', 'reader-seal.json')
    paths += [ORIGINAL / name for name in names]
    paths += [capsule / 'independent-reader' / name for name in captured_names]
    admitted = [regular(path) for path in paths]
    require(len({(p.stat().st_dev, p.stat().st_ino) for p in admitted}) == len(admitted), 'whole metadata inode matrix')
    raw_inventory = paths[0].read_bytes()
    require(digest(raw_inventory) == inventory_sha, 'authorized exact completed inventory')
    inventory = json.loads(raw_inventory)
    require(inventory['protocol'] == PROTOCOL and inventory['inventory_excludes_itself'] is True,
            'completed protocol inventory')
    index = {row['path']: row for row in inventory['files']}
    require(len(index) == len(inventory['files']), 'unique inventory records')
    complete_raw = paths[1].read_bytes()
    require(digest(complete_raw) == index['results/complete.json']['sha256'], 'inventory-bound completion')
    complete = json.loads(complete_raw)
    require(complete['protocol'] == PROTOCOL and complete['status'] == 'complete' and
            complete['cohorts'] == 5 and complete['tasks_each'] == 2 and complete['retained_points'] == 20 and
            complete['encoder_trajectories'] == 10 and complete['encoder_updates_each'] == 512 and
            complete['skipped_attempts'] == 0 and complete['full_native_export_calls'] == 120 and
            complete['initial_shared_state_exact_before_training_and_heads'] is True and complete['separate_initial_controls'] == 2,
            'all unselected trajectories completed; no skip or quality replay')
    originals = {}
    for name, captured_name in zip(names, captured_names):
        raw = (ORIGINAL / name).read_bytes()
        saved = (capsule / 'independent-reader' / captured_name).read_bytes()
        require(digest(raw) == PINS[name] and saved == raw and
                index['independent-reader/' + captured_name]['sha256'] == PINS[name],
                'immutable original/captured source-only triple ' + name)
        originals[name] = raw
    fixture = json.loads(originals['source-fixtures.json']); seal = json.loads(originals['reader-seal.json'])
    require(fixture['status'] == seal['status'] == 'passed' and not fixture['measured_execution_enabled'] and
            fixture['archive_payload_reads'] == fixture['quality_payload_hashes'] == 0 and
            seal['source_sha256'] == PINS[SOURCE_NAME] and seal['fixtures_sha256'] == PINS['source-fixtures.json'],
            'captured approved source-only fixtures and seal')
    released = flags_only(originals[SOURCE_NAME])
    output.mkdir()
    with (output / SOURCE_NAME).open('xb') as out:
        out.write(released)
    proof = {'protocol': PROTOCOL, 'status': 'passed',
             'released_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'authorization': 'root explicitly authorized flags-only release after this completed inventory',
             'capsule': str(capsule), 'inventory_sha256': inventory_sha,
             'prequality_source': str(ORIGINAL / SOURCE_NAME), 'prequality_source_sha256': PINS[SOURCE_NAME],
             'released_source': str(output / SOURCE_NAME), 'released_source_sha256': digest(released),
             'source_fixtures_sha256': PINS['source-fixtures.json'], 'prequality_seal_sha256': PINS['reader-seal.json'],
             'two_flag_reverse_byte_proof': True, 'original_prequality_triple_unchanged': True,
             'release_utility_sha256': digest(Path(__file__).read_bytes()),
             'archive_payload_reads': 0, 'score_metadata_reads': 0,
             'model_head_optimizer_or_SVD_execution': False}
    with (output / 'release-proof.json').open('x', encoding='utf-8', newline='\n') as out:
        json.dump(proof, out, indent=2, allow_nan=False); out.write('\n')
    for name in names:
        require(digest((ORIGINAL / name).read_bytes()) == PINS[name], 'original triple still immutable')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--authorized-completed-capsule'); parser.add_argument('--inventory-sha256')
    parser.add_argument('--released-directory')
    args = parser.parse_args()
    if args.self_test:
        require(not any((args.authorized_completed_capsule, args.inventory_sha256, args.released_directory)),
                'source fixtures cannot also release')
        print(json.dumps(self_test())); return
    require(all((args.authorized_completed_capsule, args.inventory_sha256, args.released_directory)),
            'all explicit root-authorized completed release bindings required')
    print(json.dumps(release(args.authorized_completed_capsule, args.released_directory, args.inventory_sha256)))


if __name__ == '__main__':
    main()

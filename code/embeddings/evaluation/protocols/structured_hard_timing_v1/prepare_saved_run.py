#!/usr/bin/env python3
"""Closed source capture and saved-run admission. No tensors or models are loaded."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True
PROTOCOL = 'structured-hard-timing-comparison-v2'
RUN_ROOT = '.temp/embeddings/runs/rpb-structured-hard-timing'
CARD = 'code/embeddings/evaluation/cards/structured_hard_timing_comparison_v2.md'
CARD_SHA = '35f7aa9987f293bafbe735506cc86cd2af02f2ab39bec42dd1dc2d2dedfa758c'
DATA_ROOT = 'code/embeddings/evaluation/benchmarks/structured_hard_timing'
DATA_BINARY = Path('/opt/mityane-geometrics/embeddings/build/structured-hard-timing-data/code/benchmarks/structured_hard_timing/cross_feature_test')
DATA_MARKER = 'Cross-feature timing data-only fixtures passed: 54 affine noiseless cases; no encoder/head fitting or quality evaluation'
RULE = 'observed-cross-feature-determinant-v2'
PRODUCER_ROOT = 'code/embeddings/evaluation/protocols/structured_hard_timing_v1'
QUALITY_BINARY = Path('/opt/mityane-geometrics/embeddings/build/rpb-structured-hard-timing/embedding_structured_hard_timing')
QUALITY_MARKERS = ['Early mixer model CUDA admission passed', 'Early mixer CUDA adapter admission passed',
                   'Structured hard timing CUDA admission passed', 'Frozen role guard checks passed',
                   'Fixed feature readout tests passed']


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(path, directory=False):
    p = Path(path)
    require(p.is_absolute() and p.resolve(strict=True) == p, 'absolute canonical path required: ' + str(p))
    for node in (p, *p.parents):
        require(not node.is_symlink(), 'symlink ancestor forbidden: ' + str(node))
    s = p.stat()
    require(stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode), 'wrong path type: ' + str(p))
    if not directory:
        require(s.st_nlink == 1, 'hardlinked file forbidden: ' + str(p))
    return p


def admit_paths(paths):
    result = [canonical(p) for p in paths]
    require(len(set(result)) == len(result), 'duplicate paths')
    inodes = [(p.stat().st_dev, p.stat().st_ino) for p in result]
    require(len(set(inodes)) == len(inodes), 'file aliases forbidden')
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(paths):
    admitted = admit_paths(paths)  # Complete path/inode matrix before first hash.
    return [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in admitted]


def read_json(path):
    return json.loads(canonical(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def safe_name(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and str(p) == name and not p.is_absolute() and
            all(x not in ('', '.', '..') for x in p.parts) and '\\' not in name,
            'normalized relative source name required')
    require(name in ('Makefile', 'dependencies.lock', 'setup.sh', '.archive/embeddings/pre-cleanup-2026-10-18/doc/dataset_registry.json') or name.startswith('code/'),
            'closed source namespace')
    return name


def empty_target(repo, target, family):
    repo = canonical(repo, True)
    p = Path(target)
    require(p.is_absolute() and p.parent == repo / RUN_ROOT / family, 'exclusive target family')
    canonical(p.parent, True)
    if p.exists():
        canonical(p, True)
        require(not any(p.iterdir()), 'target must be empty; preserve prior attempts')
    else:
        p.mkdir()
    return repo, p


def information_sources(repo):
    names = ['Makefile', 'dependencies.lock', 'setup.sh', 'code/scripts/verify-libtorch.mjs',
             'code/scripts/node.sh', 'code/scripts/node-runtime.sh', CARD,
             DATA_ROOT + '/Makefile', DATA_ROOT + '/structured_hard_timing.h',
             DATA_ROOT + '/structured_hard_timing.cpp', DATA_ROOT + '/cross_feature_test.cpp',
             DATA_ROOT + '/cross_feature_solvability.h', DATA_ROOT + '/cross_feature_solvability.cpp',
             'code/shared/representation/src/feature_harness.cpp']
    names += [str(p.relative_to(repo)).replace(os.sep, '/') for p in (repo / 'code/shared/representation/include').rglob('*.h')]
    return sorted(set(names))


def capture(repo, target, names, card_sha):
    names = sorted(names)
    require(names and len(names) == len(set(names)), 'unique nonempty source closure')
    for name in names:
        safe_name(name)
    require(CARD in names and {'setup.sh', 'code/scripts/verify-libtorch.mjs', 'code/scripts/node.sh',
                              'code/scripts/node-runtime.sh', 'dependencies.lock', 'Makefile'} <= set(names),
            'card and internal SDK provenance required')
    source_records = records([repo / name for name in names])
    relative = [dict(item, path=name) for name, item in zip(names, source_records)]
    card = next(x for x in relative if x['path'] == CARD)
    require(card_sha == CARD_SHA == card['sha256'], 'frozen prospective card hash')
    manifest = ''.join(x['sha256'] + '  ' + x['path'] + '\n' for x in relative)
    fingerprint = hashlib.sha256(manifest.encode()).hexdigest()
    for name in names:
        destination = target / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write((repo / name).read_bytes())
    copied = records([target / 'source' / name for name in names])
    require(all(a['bytes'] == b['bytes'] and a['sha256'] == b['sha256'] for a, b in zip(source_records, copied)), 'captured source bytes')
    (target / 'source-inputs.sha256').write_text(manifest, encoding='utf-8', newline='\n')
    (target / 'source-fingerprint.txt').write_text(fingerprint + '\n', encoding='utf-8', newline='\n')
    write_json(target / 'source-manifest.json', {'protocol': PROTOCOL, 'source_sha256': fingerprint, 'files': relative})
    return fingerprint


def verify_sources(repo, target):
    manifest_path = canonical(target / 'source-manifest.json')
    value = read_json(manifest_path)
    files = value['files']
    names = [safe_name(x['path']) for x in files]
    require(names == sorted(set(names)), 'complete sorted source-name matrix')
    combined = records([repo / n for n in names] + [target / 'source' / n for n in names])
    current, copied = combined[:len(names)], combined[len(names):]
    for expected, a, b in zip(files, current, copied):
        require(a['bytes'] == b['bytes'] == expected['bytes'] and a['sha256'] == b['sha256'] == expected['sha256'], 'source preserved: ' + expected['path'])
    body = ''.join(x['sha256'] + '  ' + x['path'] + '\n' for x in files)
    require((target / 'source-inputs.sha256').read_text() == body and
            (target / 'source-fingerprint.txt').read_text() == value['source_sha256'] + '\n' and
            hashlib.sha256(body.encode()).hexdigest() == value['source_sha256'], 'canonical source identity')
    return value


def validate_information(value):
    require(set(value) == {'protocol', 'dataset_id', 'rule_id', 'complexity', 'complexity_maximum', 'seeds', 'passed', 'encoder_or_head_fits', 'TEST_generated'}, 'closed raw information schema')
    require(value['protocol'] == 'structured-hard-timing-information-v2' and value['rule_id'] == RULE and value['dataset_id'] == 'TEMPO-3' and
            value['complexity'] == 4 and value['complexity_maximum'] == 5 and value['encoder_or_head_fits'] == 0 and
            value['TEST_generated'] is False, 'information recipe and zero fit scope')
    require(len(value['seeds']) == 2, 'both fixed engineering seeds')
    passed = True
    for expected, row in zip((82585, 83686), value['seeds']):
        require(set(row) == {'seed', 'training_source_pairs', 'validation_source_pairs', 'intact', 'deleted'} and
                row['seed'] == expected and row['training_source_pairs'] == row['validation_source_pairs'] == 512, 'engineering source populations')
        for view, acc_min, cov_min in [('intact', .99, .99), ('deleted', .98, .95)]:
            score = row[view]
            require(set(score) == {'total', 'valid', 'correct', 'accuracy', 'coverage', 'full_population_correctness', 'passed'}, 'closed information score schema')
            total, valid, correct = (score[k] for k in ('total', 'valid', 'correct'))
            require(all(type(x) is int for x in (total, valid, correct)) and total == 1024 and 0 <= correct <= valid <= total, 'information integer bounds')
            for name, expected_number in [('accuracy', correct / valid if valid else 0), ('coverage', valid / total), ('full_population_correctness', correct / total)]:
                actual = score[name]
                require(type(actual) in (float, int) and math.isfinite(actual) and abs(actual - expected_number) <= 2e-15, 'exact rational information score: ' + name)
            ok = score['accuracy'] >= acc_min and score['coverage'] >= cov_min
            require(score['passed'] is ok, 'fixed information gate')
            passed = passed and ok
    require(value['passed'] is passed, 'whole information gate')
    return passed


def compile_commands(log, required, binary):
    require('/embedding/.external/libtorch' not in log and 'not found' not in log, 'internal SDK runtime')
    commands = {}
    lines = log.splitlines()
    outputs = []
    for source, object_path in required:
        candidates = []
        for line in lines:
            if ' -c ' not in line:
                continue
            try:
                args = shlex.split(line)
            except ValueError:
                continue
            if '-c' in args and '-o' in args and args[args.index('-c') + 1] == source:
                out = args[args.index('-o') + 1]
                if out == str(object_path):
                    candidates.append(line)
        require(len(candidates) == 1, 'one current exact compile record: ' + source + ' -> ' + str(object_path))
        commands[source] = candidates[0]
        outputs.append(str(object_path))
    links = []
    for line in lines:
        try:
            args = shlex.split(line)
        except ValueError:
            continue
        if '-c' not in args and '-o' in args and args[args.index('-o') + 1] == str(binary):
            links.append(line)
    require(len(links) == 1, 'one current exact binary link')
    link_args = shlex.split(links[0])
    require(all(link_args.count(out) == 1 for out in outputs), 'every current compiled object appears once in exact binary link')
    return {'commands': commands, 'link': links[0]}


def freeze_information(args):
    repo, target = empty_target(args.repo_root, args.target, 'information-admission')
    names = args.sources or information_sources(repo)
    require(set(information_sources(repo)) <= set(names), 'all data compilation dependencies required')
    fingerprint = capture(repo, target, names, args.card_sha256)
    write_json(target / 'freeze.json', {'protocol': 'structured-hard-timing-information-v2', 'rule_id': RULE, 'source_sha256': fingerprint, 'card_sha256': CARD_SHA, 'utc': utc(), 'quality_generated': False})
    print(json.dumps({'status': 'frozen', 'target': str(target), 'source_sha256': fingerprint, 'sources': len(names)}))


def admit_information(args):
    repo, target = canonical(args.repo_root, True), canonical(args.target, True)
    require(target.parent == repo / RUN_ROOT / 'information-admission', 'fixed information family')
    binary = canonical(args.binary)
    require(binary == DATA_BINARY, 'fixed data-only binary')
    paths = [target / 'information.json', target / 'build-and-tests.log', target / 'source-manifest.json', target / 'source-inputs.sha256', target / 'source-fingerprint.txt', target / 'freeze.json', binary]
    before = records(paths)
    manifest = verify_sources(repo, target)
    log = (target / 'build-and-tests.log').read_text()
    require(DATA_MARKER in log, 'actual 54-case data fixtures passed marker')
    compiled = compile_commands(log, [(DATA_ROOT + '/structured_hard_timing.cpp', binary.parent / 'structured_hard_timing.o'),
                (DATA_ROOT + '/cross_feature_test.cpp', binary.parent / 'cross_feature_test.o'),
                (DATA_ROOT + '/cross_feature_solvability.cpp', binary.parent / 'cross_feature_solvability.o')], binary)
    value = read_json(target / 'information.json')
    passed = validate_information(value)
    verify_sources(repo, target)
    require(records(paths) == before, 'information admission inputs preserved after validation')
    record = {'protocol': 'structured-hard-timing-information-binding-v2', 'rule_id': RULE, 'status': 'passed' if passed else 'failed', 'passed': passed,
              'utc': utc(), 'card_sha256': CARD_SHA, 'source_sha256': manifest['source_sha256'], 'inputs': before,
              'compiler_evidence': compiled, 'information': value, 'source_preserved_after': True,
              'encoder_or_head_fits': 0, 'archive_reads': 0, 'quality_generated': False}
    write_json(target / 'information-record.json', record)
    print(json.dumps({'status': record['status'], 'path': str(target / 'information-record.json'), 'sha256': sha(target / 'information-record.json')}))
    require(passed, 'information admission failed; preserve recipe/result; do not train encoders')


def verify_information_record(repo, record_path):
    record_path = canonical(record_path)
    require(record_path.name == 'information-record.json' and record_path.parent.parent == repo / RUN_ROOT / 'information-admission', 'closed information binding path')
    record = read_json(record_path)
    require(record['protocol'] == 'structured-hard-timing-information-binding-v2' and record['rule_id'] == RULE and record['passed'] is True and record['status'] == 'passed' and
            record['card_sha256'] == CARD_SHA and record['source_preserved_after'] is True and record['archive_reads'] == 0 and record['encoder_or_head_fits'] == 0,
            'passed source-bound information record')
    require(validate_information(record['information']), 'fixed information gates')
    inputs = record['inputs']
    actual = records([x['path'] for x in inputs])
    require(actual == inputs, 'information bytes/binary/log preserved')
    require(read_json(record_path.parent / 'information.json') == record['information'], 'raw information association')
    require(verify_sources(repo, record_path.parent)['source_sha256'] == record['source_sha256'], 'information source closure')
    return record


def freeze_admission(args):
    repo, target = empty_target(args.repo_root, args.target, 'admission')
    fingerprint = capture(repo, target, args.sources, args.card_sha256)
    expanded = subprocess.run(['make', '-s', '-f', PRODUCER_ROOT + '/Makefile', 'print-structured-hard-timing-scope-inputs'],
                              cwd=repo, capture_output=True, text=True, check=True).stdout
    scopes = {}
    for line in expanded.splitlines():
        name, path = line.split('|', 1)
        require(name in ('core_writer', 'curve_training', 'early_adapter', 'structured_adapter'), 'closed compiler scope name')
        safe_name(path)
        scopes.setdefault(name, []).append(path)
    require(set(scopes) == {'core_writer', 'curve_training', 'early_adapter', 'structured_adapter'}, 'all actual compiled scopes')
    manifest = read_json(target / 'source-manifest.json')
    index = {x['path']: x for x in manifest['files']}
    scope_records = {}
    for name, paths in scopes.items():
        require(paths == sorted(set(paths)) and set(paths) <= set(index), 'full canonical scope subset: ' + name)
        rows = [index[p] for p in paths]
        identity = hashlib.sha256(''.join(x['sha256'] + '  ' + x['path'] + '\n' for x in rows).encode()).hexdigest()
        scope_records[name] = {'source_sha256': identity, 'files': rows}
    require(scope_records['structured_adapter']['source_sha256'] == fingerprint, 'enclosing scope canonical equality')
    verify_sources(repo, target)
    write_json(target / 'source-scopes.json', scope_records)
    write_json(target / 'freeze.json', {'protocol': PROTOCOL, 'source_sha256': fingerprint, 'card_sha256': CARD_SHA, 'utc': utc(), 'quality_generated': False})
    print(json.dumps({'status': 'frozen', 'target': str(target), 'source_sha256': fingerprint}))


def metadata_command(binary, option):
    result = subprocess.run([str(binary), option], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def validate_plan(plan):
    expected = {'protocol': PROTOCOL, 'timing_master_seeds': [75272,76373,77474,78575,79676],
                'dataset_codename':'TEMPO-3','dataset_generator_protocol':'structured-hard-timing-v1','dataset_id':'TEMPO-3',
                'recipe_id':'structured-hard-timing-v1','complexity_is_designed_not_accuracy_derived':True,
                'designed_complexity_level':4,'complexity_scale_max':5,'task':'lag_sign','tags':['RPB-v7.alt-05','RPB-v10.alt-05'],
                'encoder_trajectories':10,'encoder_updates_each':512,'attempt_limit':1024,'log_every':1,
                'skipped_attempts_permitted':False,'retained_points':20,'decoder_updates':0,'extra_decoder_calibration_updates':0,
                'decoder_update_scope':'joint reconstruction during encoder updates; no extra decoder-only calibration',
                'batch_size':8,'sampled_rows':40960,'train_pairs':128,'validation_pairs':64,'test_pairs':0,'shape':[3,32,3],
                'native_width':32,'parameter_count_each':225805,'channel_mixer_placements':[0,1],'global_pool_input_sources':[0,0],
                'checkpoint_roles_per_point':5,'external_fit_protocol':PROTOCOL+'/lag_sign',
                'implementation_fit_protocol':'early-mixer-reliability-v1/lag_sign','point_binding_suffix':'.structured-timing.pt',
                'head_repetitions':[2701,2802,2903],'planned_pipelines':105,'planned_heads':210,'deletion_rate':.30,
                'methods_each_task':7,'separate_initial_controls':2,'planned_unique_quality_native_exports':60,
                'planned_initial_counterpart_exports':0,'planned_native_export_calls':60,'planned_query_writer_calls':20,
                'planned_query_forwards':80,'driver_raw_outer_fits':5,'helper_outer_train_fits':25,'raw_outer_fit_shared_with_PCA':True,
                'quality_gain':False,'analytic_solver_fits':0,'analytic_individual_heads':0,'analytic_surface_evaluations':15,
                'information_rule_id':RULE,'encoder_device':'CUDA','prior_quality_input_roles':[],'quality_generated':False,
                'testing_accessed':False,'stress_accessed':False,'selection':False,'promotion':False,'human_card_sha256':CARD_SHA}
    require(set(plan) == set(expected) | {'source_fingerprint'}, 'closed actual producer plan schema')
    for key, value in expected.items():
        require(plan[key] == value, 'fixed prospective plan: ' + key)
    require(isinstance(plan['source_fingerprint'], str) and len(plan['source_fingerprint']) == 64 and
            all(c in '0123456789abcdef' for c in plan['source_fingerprint']), 'actual compiled source fingerprint')
    return expected


def admit_quality(args):
    repo, target = canonical(args.repo_root, True), canonical(args.target, True)
    require(target.parent == repo / RUN_ROOT / 'admission', 'admission family')
    binary = canonical(args.binary)
    require(binary == QUALITY_BINARY, 'named quality binary')
    information = verify_information_record(repo, args.information)
    manifest = verify_sources(repo, target)
    require(args.compiled_source_id == manifest['source_sha256'] == metadata_command(binary, '--source-id'), 'actual enclosing source identity')
    log_path = canonical(target / 'build-and-tests.log')
    log = log_path.read_text()
    require(all(marker in log for marker in QUALITY_MARKERS), 'all actual CUDA/readout/role gates')
    old_root = 'code/embeddings/encoders/raw_patch_bottleneck_mae/shared/src/'
    old_objects = binary.parent / 'code/embeddings/encoders/raw_patch_bottleneck_mae'
    new_objects = binary.parent / 'code/protocols/structured_hard_timing_v1'
    required = [(old_root + 'workflow.cpp', old_objects / 'workflow.o'), (old_root + 'learning_curve_adapter.cpp', old_objects / 'learning_curve_adapter.o'),
                (old_root + 'early_mixer_adapter.cpp', old_objects / 'early_mixer_adapter.o'),
                (PRODUCER_ROOT + '/structured_hard_timing_adapter.cpp', new_objects / 'structured_hard_timing_adapter.o'),
                (PRODUCER_ROOT + '/paired_timing_run.cpp', new_objects / 'paired_timing_run.o'),
                (PRODUCER_ROOT + '/structured_hard_timing_main.cpp', new_objects / 'structured_hard_timing_main.o'),
                (DATA_ROOT + '/structured_hard_timing.cpp', new_objects / 'structured_hard_timing_generator.o'),
                (DATA_ROOT + '/cross_feature_solvability.cpp', new_objects / 'cross_feature_solvability.o')]
    compiled = compile_commands(log, required, binary)
    scopes = read_json(target / 'source-scopes.json')
    macro_bindings = [(old_root + 'workflow.cpp','RPB_SOURCE_ID','core_writer'),
                      (old_root + 'learning_curve_adapter.cpp','EVALUATION_SOURCE_ID','curve_training'),
                      (old_root + 'early_mixer_adapter.cpp','EARLY_MIXER_ADAPTER_SOURCE_ID','early_adapter'),
                      (PRODUCER_ROOT + '/structured_hard_timing_adapter.cpp','STRUCTURED_HARD_TIMING_ADAPTER_SOURCE_ID','structured_adapter'),
                      (PRODUCER_ROOT + '/paired_timing_run.cpp','EVALUATION_SOURCE_ID','structured_adapter'),
                      (PRODUCER_ROOT + '/structured_hard_timing_main.cpp','EVALUATION_SOURCE_ID','structured_adapter'),
                      (DATA_ROOT + '/cross_feature_solvability.cpp','EVALUATION_SOURCE_ID','structured_adapter'),
                      (DATA_ROOT + '/structured_hard_timing.cpp','EVALUATION_SOURCE_ID','structured_adapter')]
    for path, macro, scope in macro_bindings:
        args_compiled = shlex.split(compiled['commands'][path])
        values = [a.split('=',1)[1].strip('"') for a in args_compiled if a.startswith('-D' + macro + '=')]
        require(values == [scopes[scope]['source_sha256']], 'exact current compiled scope macro: ' + path)
    require('Container SDK proof:' in log, 'installed internal SDK proof and ldd recorded')
    for source in [PRODUCER_ROOT + '/structured_hard_timing_adapter.cpp', PRODUCER_ROOT + '/paired_timing_run.cpp', PRODUCER_ROOT + '/structured_hard_timing_main.cpp']:
        require(manifest['source_sha256'] in compiled['commands'][source], 'new producer compiled scope')
    plan = json.loads(metadata_command(binary, '--plan'))
    validate_plan(plan)
    info_dir = target / 'information'; info_dir.mkdir()
    for name in ['information-record.json', 'information.json', 'build-and-tests.log', 'source-manifest.json', 'source-inputs.sha256', 'source-fingerprint.txt', 'freeze.json']:
        with (info_dir / name).open('xb') as out:
            out.write((Path(args.information).parent / name).read_bytes())
    with log_path.open('a', encoding='utf-8', newline='\n') as out:
        out.write('TEMPO-3 observed-only information admission passed: ' + sha(args.information) + '\n')
    verify_sources(repo, target)
    verify_information_record(repo, args.information)
    write_json(target / 'recipe-plan.json', plan)
    write_json(target / 'passed.json', {'protocol': PROTOCOL, 'status': 'passed', 'source_sha256': manifest['source_sha256'], 'card_sha256': CARD_SHA,
              'binary': records([binary])[0], 'log': records([log_path])[0], 'compiler_evidence': compiled,
              'information_record': records([args.information])[0], 'information_source_sha256': information['source_sha256'],
              'quality_generated': False, 'utc': utc()})
    print(json.dumps({'status': 'passed', 'admission': str(target), 'source_sha256': manifest['source_sha256']}))


def reader_bundle(directory):
    directory = canonical(directory, True)
    seal = read_json(directory / 'reader-seal.json')
    require(seal['status'] == 'passed' and seal.get('measured_execution_enabled') is False, 'reviewed blocked reader bundle')
    names = [x['path'] for x in seal['files']]
    require(names == sorted(set(names)) and all('/' not in n and '\\' not in n and n not in ('.', '..') and not n.endswith('.pt') for n in names), 'closed reader source file names')
    require({'validate_structured_hard_timing.py', 'source-fixtures.json', 'saved_cpu_math.py', 'archive_codec.py'} <= set(names), 'complete independent reader source bundle')
    actual = records([directory / n for n in names])
    for expected, row in zip(seal['files'], actual):
        require(expected['bytes'] == row['bytes'] and expected['sha256'] == row['sha256'], 'reader file bytes')
    source = (directory / 'validate_structured_hard_timing.py').read_text()
    require(source.count('\nREVIEWED_SCHEMA = False\n') == 1 and source.count('\nMEASURED_IMPLEMENTATION = False\n') == 1, 'prequality false flags')
    fixtures = read_json(directory / 'source-fixtures.json')
    require(fixtures['status'] == 'passed' and fixtures['measured_execution_enabled'] is False and fixtures['archive_reads'] == 0, 'artificial-only reader fixtures')
    return seal, names + ['reader-seal.json']


def freeze_run(args):
    repo = canonical(args.repo_root, True)
    admission = canonical(args.admission, True)
    require(admission.parent == repo / RUN_ROOT / 'admission', 'approved admission family')
    passed = read_json(admission / 'passed.json')
    require(passed['status'] == 'passed' and passed['protocol'] == PROTOCOL and passed['card_sha256'] == CARD_SHA, 'actual quality admission passed')
    verify_sources(repo, admission)
    require(records([passed['binary']['path']])[0] == passed['binary'] and records([passed['log']['path']])[0] == passed['log'], 'admitted binary/log preserved')
    verify_information_record(repo, passed['information_record']['path'])
    require(records([passed['information_record']['path']])[0] == passed['information_record'], 'admitted information binding preserved')
    seal, names = reader_bundle(args.reader_dir)
    target = Path(args.target)
    require(target.is_absolute() and target.parent == repo / RUN_ROOT and target.name.startswith('structured-hard-timing-'), 'new quality capsule family')
    canonical(target, True); require(not any(target.iterdir()), 'exclusive empty quality capsule')
    source_manifest = read_json(admission / 'source-manifest.json')
    capture(repo, target, [x['path'] for x in source_manifest['files']], CARD_SHA)
    admission_paths = sorted(p for p in admission.rglob('*') if p.is_file())
    records(admission_paths)
    for p in admission_paths:
        dest = target / 'admission' / p.relative_to(admission)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as out:
            out.write(p.read_bytes())
    reader_paths = [Path(args.reader_dir) / n for n in names]
    before_reader = records(reader_paths)
    (target / 'independent-reader').mkdir()
    for p in reader_paths:
        with (target / 'independent-reader' / p.name).open('xb') as out:
            out.write(p.read_bytes())
    require(records(reader_paths) == before_reader, 'reader bundle preserved')
    verify_sources(repo, admission); verify_sources(repo, target)
    write_json(target / 'launch-plan.json', {'protocol': PROTOCOL, 'source_sha256': passed['source_sha256'], 'card_sha256': CARD_SHA,
              'binary': passed['binary'], 'admission': str(admission), 'admission_sha256': sha(admission / 'passed.json'),
              'information_record': passed['information_record'], 'reader_original': str(args.reader_dir), 'reader_files': before_reader,
              'historical_payload_roles': [], 'quality_generated_before_freeze': False, 'utc': utc()})
    print(json.dumps({'status': 'frozen', 'capsule': str(target), 'source_sha256': passed['source_sha256']}))


def finalize(args):
    repo, target = canonical(args.repo_root, True), canonical(args.target, True)
    require(target.parent == repo / RUN_ROOT and target.name.startswith('structured-hard-timing-'), 'completed capsule family')
    complete = read_json(target / 'results/complete.json')
    require(complete['protocol'] == PROTOCOL and complete['status'] == 'complete' and complete['cohorts'] == 5 and complete['encoder_trajectories'] == 10 and
            complete['encoder_updates_each'] == 512 and complete['sampled_rows'] == 40960, 'full fixed measurement complete')
    launch = read_json(target / 'launch-plan.json')
    verify_sources(repo, target)
    require(records([launch['binary']['path']])[0] == launch['binary'], 'quality binary preserved')
    verify_information_record(repo, launch['information_record']['path'])
    require(records([launch['information_record']['path']])[0] == launch['information_record'], 'information binding preserved')
    require(records([x['path'] for x in launch['reader_files']]) == launch['reader_files'], 'original blocked reader preserved')
    seal, names = reader_bundle(target / 'independent-reader')
    write_json(target / 'source-preserved-after.json', {'status': 'passed', 'source_sha256': launch['source_sha256'], 'binary_preserved': True, 'reader_preserved': True, 'information_preserved': True})
    paths = sorted(p for p in target.rglob('*') if p.is_file())
    all_records = records(paths)
    relative = [dict(row, path=str(p.relative_to(target))) for p, row in zip(paths, all_records)]
    write_json(target / 'artifact-integrity.json', {'protocol': PROTOCOL, 'status': 'complete', 'files': relative, 'files_count': len(relative), 'total_bytes': sum(x['bytes'] for x in relative), 'excludes': ['artifact-integrity.json']})
    print(json.dumps({'status': 'complete', 'inventory_sha256': sha(target / 'artifact-integrity.json'), 'files': len(relative)}))


def self_test(args):
    score = {'total': 1024, 'valid': 1024, 'correct': 1024, 'accuracy': 1., 'coverage': 1., 'full_population_correctness': 1., 'passed': True}
    data = {'protocol': 'structured-hard-timing-information-v2', 'dataset_id': 'TEMPO-3', 'rule_id': RULE, 'complexity': 4, 'complexity_maximum': 5,
            'seeds': [{'seed': s, 'training_source_pairs': 512, 'validation_source_pairs': 512, 'intact': dict(score), 'deleted': dict(score)} for s in (82585, 83686)],
            'passed': True, 'encoder_or_head_fits': 0, 'TEST_generated': False}
    require(validate_information(data), 'positive information fixture')
    negatives = 0
    for mutation in [('seed', 12), ('accuracy', .5), ('valid', 1025), ('passed', False)]:
        changed = json.loads(json.dumps(data))
        if mutation[0] == 'seed':
            changed['seeds'][1]['seed'] = mutation[1]
        else:
            changed['seeds'][1]['deleted'][mutation[0]] = mutation[1]
        try:
            validate_information(changed)
        except ValueError:
            negatives += 1
        else:
            raise ValueError('negative information fixture escaped')
    for name in ('../escape', '/absolute', 'code/../escape', 'setup.sh.bak', 'code\\escape'):
        try:
            safe_name(name)
        except ValueError:
            negatives += 1
        else:
            raise ValueError('negative source-name fixture escaped')
    with tempfile.TemporaryDirectory() as temporary:
        a = Path(temporary) / 'a'; a.write_text('source')
        b = Path(temporary) / 'b'; os.link(a, b)
        try:
            admit_paths([a, b])
        except ValueError:
            negatives += 1
        else:
            raise ValueError('hardlink fixture escaped')
    source = canonical(Path('/mityane') / PRODUCER_ROOT / 'paired_timing_run.cpp').read_text()
    body = source.split('std::string plan() {', 1)[1].split('\n}', 1)[0]
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|quote\(protocol\)|quote\(card_sha\)|quote\(EVALUATION_SOURCE_ID\)', body)
    values = {'quote(protocol)': json.dumps(PROTOCOL), 'quote(card_sha)': json.dumps(CARD_SHA),
              'quote(EVALUATION_SOURCE_ID)': json.dumps('0' * 64)}
    actual_plan = json.loads(''.join(values[t] if t in values else json.loads(t) for t in tokens))
    validate_plan(actual_plan)
    source_files = [('code/a.cpp', Path('/opt/build/a.o')), ('code/b.cpp', Path('/opt/build/b.o'))]
    fixture_binary = Path('/opt/build/probe')
    compile_log = 'g++ -c code/a.cpp -o /opt/build/a.o\ng++ -c code/b.cpp -o /opt/build/b.o\ng++ /opt/build/a.o /opt/build/b.o -o /opt/build/probe\n'
    compile_commands(compile_log, source_files, fixture_binary)
    for bad in [compile_log.replace('/opt/build/a.o', '/other/a.o'),
                compile_log.replace('g++ /opt/build/a.o /opt/build/b.o', 'g++ /opt/build/b.o'),
                compile_log + 'g++ -c code/a.cpp -o /opt/build/a.o\n']:
        try:
            compile_commands(bad, source_files, fixture_binary)
        except ValueError:
            negatives += 1
        else:
            raise ValueError('compile/output/link fixture escaped')
    print(json.dumps({'status': 'passed', 'negative_fixtures': negatives, 'archive_reads': 0, 'encoder_or_head_fits': 0, 'real_SDK_reads': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('self-test').set_defaults(call=self_test)
    freeze = sub.add_parser('freeze-information')
    freeze.add_argument('--repo-root', required=True); freeze.add_argument('--target', required=True)
    freeze.add_argument('--card-sha256', default=CARD_SHA); freeze.add_argument('--sources', nargs='+')
    freeze.set_defaults(call=freeze_information)
    admit = sub.add_parser('admit-information')
    admit.add_argument('--repo-root', required=True); admit.add_argument('--target', required=True)
    admit.add_argument('--binary', default=str(DATA_BINARY)); admit.set_defaults(call=admit_information)
    freeze = sub.add_parser('freeze-admission')
    freeze.add_argument('--repo-root', required=True); freeze.add_argument('--target', required=True)
    freeze.add_argument('--card-sha256', required=True); freeze.add_argument('--sources', nargs='+', required=True)
    freeze.set_defaults(call=freeze_admission)
    admit = sub.add_parser('admit')
    for name in ['repo-root', 'target', 'binary', 'compiled-source-id', 'information']:
        admit.add_argument('--' + name, required=True)
    admit.set_defaults(call=admit_quality)
    run = sub.add_parser('freeze-run')
    for name in ['repo-root', 'target', 'admission', 'reader-dir']:
        run.add_argument('--' + name, required=True)
    run.set_defaults(call=freeze_run)
    finish = sub.add_parser('finalize')
    finish.add_argument('--repo-root', required=True); finish.add_argument('--target', required=True)
    finish.set_defaults(call=finalize)
    args = parser.parse_args()
    require(Path('/.dockerenv').exists(), 'run in the managed container')
    args.call(args)


if __name__ == '__main__':
    raise SystemExit('Metadata CLI migrated to prepare_saved_run.mjs; this Python module remains only as the independent captured scientific contract API.')

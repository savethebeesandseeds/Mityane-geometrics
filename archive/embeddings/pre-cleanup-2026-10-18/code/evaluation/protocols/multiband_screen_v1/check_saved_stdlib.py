#!/usr/bin/env python3
"""Replay saved native32 heads and original queries; never execute a model.

The two tasks retain separate source populations. This small check verifies saved
scores, TRAIN scalers, all 512 reported updates and fixed relation coordinates.
It does not replay bootstrap intervals or decode checkpoints/optimizer bodies.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import struct
import sys

sys.dont_write_bytecode = True
ROOT = Path('/embedding')
MODULE_ROOT = ROOT / 'code/evaluation/protocols/visible_difference_v1/reader-v2'
RUN_ROOT = ROOT / 'output/runs/rpb-multiband-screen'
MASTERS = [920903, 920904]
TASKS = ['slow_lag_sign', 'component_balance']
DATASETS = dict(zip(TASKS, ('TEMPO-4', 'AMP-2')))
DECODES = 0
READ_SHA = {}
PINNED_MODULES = {
    'archive_codec.py': '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d',
    'saved_cpu_math.py': '1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9',
    'legacy_implementation.py': 'f0c64b9fe67a90a9d95196433e05dec527515778d1635bdc909d1703fa278b68'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct(path, regular=False):
    if not path.is_absolute() or path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError('canonical direct path required: ' + str(path))
    info = path.stat()
    if regular and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise ValueError('distinct direct regular file required: ' + str(path))
    return info.st_dev, info.st_ino


def admit(paths):
    identities = [direct(path, True) for path in dict.fromkeys(paths)]
    if len(set(identities)) != len(identities):
        raise ValueError('aliased saved role files')


def remember(path):
    digest = sha(path)
    if path in READ_SHA and READ_SHA[path] != digest:
        raise ValueError('input changed while checking: ' + str(path))
    READ_SHA[path] = digest
    return digest


def read_json(path):
    remember(path)
    return json.loads(path.read_text())


def module(name, filename):
    path = MODULE_ROOT / filename
    direct(path, True)
    if remember(path) != PINNED_MODULES[filename]:
        raise ValueError('pinned saved-arithmetic module changed: ' + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


R = module('multiband_archive_codec', 'archive_codec.py')
M = module('saved_cpu_math', 'saved_cpu_math.py')
M.R = R
L = module('multiband_saved_contracts', 'legacy_implementation.py')
L.R = R


def load(path):
    global DECODES
    direct(path, True)
    remember(path)
    DECODES += 1
    return R.load(path)


def manifest(path):
    remember(path)
    records = []
    for line in path.read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = Path(name)
        M.check(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
                and bool(name) and not relative.is_absolute() and '..' not in relative.parts
                and str(relative) == name, 'canonical relative SHA manifest row')
        records.append((digest, name))
    M.check(records and len({name for _, name in records}) == len(records),
            'nonempty distinct closed SHA rows')
    return records


def controlled(path, rows, master, task):
    a = load(path)
    M.check(set(a) == {'observations', 'feature_mask', 'source_ids', 'labels_scoring_only'},
            'exact four-key legal controlled archive')
    data = M.tensor(a['observations'], 'DoubleStorage', [rows, 3, 32, 3])
    mask = M.tensor(a['feature_mask'], 'BoolStorage', [rows, 3, 32, 3])
    numeric = M.tensor(a['source_ids'], 'LongStorage', [rows])
    labels = M.tensor(a['labels_scoring_only'], 'LongStorage', [rows])
    M.check(rows in (256, 128) and master in MASTERS and task in TASKS,
            'closed task/master/population')
    M.check(all(math.isfinite(x) and (seen or x == 0) for x, seen in zip(data, mask)),
            'finite legal observations with hidden zeros')
    ids = []
    previous = -1
    for row in range(0, rows, 2):
        packed = numeric[row]
        M.check(packed >= 0 and numeric[row + 1] == packed and packed > previous
                and packed >> 31 == master and packed & 1 == TASKS.index(task)
                and (packed & 0x7fffffff) >> 1 < 192, 'ascending packed master/task/source pairs')
        previous = packed
        M.check({labels[row], labels[row + 1]} == {0, 1}, 'opposite randomized pair labels')
        first, second = row * 288, (row + 1) * 288
        M.check(mask[first:first + 288] == mask[second:second + 288], 'pair-shared legal mask')
        M.check(data[first + 192:first + 288] == data[second + 192:second + 288],
                'label-independent channel2 paired values')
        if task == 'slow_lag_sign':
            M.check(data[first:first + 96] == data[second:second + 96], 'timing channel0 pair equality')
        identifier = f'two-component-v1/{task}/source-{packed}'
        ids.extend((identifier, identifier))
    return data, mask, ids, labels


def check_splits(splits, master, task):
    train, intact, deleted = splits
    ti, vi = set(train[2]), set(intact[2])
    expected = {f'two-component-v1/{task}/source-{(master << 31) | (s << 1) | TASKS.index(task)}'
                for s in range(192)}
    M.check(len(ti) == 128 and len(vi) == 64 and not ti & vi and ti | vi == expected,
            'pair-disjoint TRAIN/VAL cover exact source universe')
    M.check(intact[2] == deleted[2] and list(intact[3]) == list(deleted[3]),
            'deleted source/label association unchanged')
    M.check(all(not m or (parent and x == original) for x, m, original, parent in
                zip(deleted[0], deleted[1], intact[0], intact[1])), 'exact deleted legal value subset')
    for row in range(0, 128, 2):
        packed = int(intact[2][row].rsplit('-', 1)[1])
        rng = M.MT19937_64(M.stream_seed(M.stream_seed(master, 0x74632d64656c7631), packed))
        for coordinate in range(288):
            erase = (rng.draw() >> 11) / 9007199254740992.0 < .30
            for paired in (row, row + 1):
                index = paired * 288 + coordinate
                M.check(bool(deleted[1][index]) == (bool(intact[1][index]) and not erase),
                        'exact source-keyed .30 MT19937_64 deletion')
    return ti | vi


def check_train_scaler(scaler, training):
    mean, scale, identity = M.scaler(scaler)
    count = M.tensor(scaler['count'], 'LongStorage', [3, 3])
    floors = M.tensor(scaler['floor_applied'], 'BoolStorage', [3, 3])
    data, mask, _, _ = training
    for c in range(3):
        for f in range(3):
            values = [data[b * 288 + c * 96 + h * 3 + f] for b in range(256) for h in range(32)
                      if mask[b * 288 + c * 96 + h * 3 + f]]
            i = c * 3 + f
            M.check(len(values) == count[i] and bool(values), 'legal TRAIN scaler count')
            wanted_mean = math.fsum(values) / len(values)
            sd = math.sqrt(math.fsum((x - wanted_mean) ** 2 for x in values) / len(values))
            M.close(mean[i], wanted_mean, 'legal TRAIN scaler mean')
            M.close(scale[i], max(1e-6, sd), 'legal TRAIN population scale')
            M.check(bool(floors[i]) == (sd < 1e-6), 'legal TRAIN scaler floor condition')
    return identity


def check_progress(progress, identity):
    expected = dict(attempted=512, completed=512, sampled_rows=4096, parameter_count=226877,
                    cuda_parameter_count=226877, trainable_parameter_count=226445,
                    frozen_parameter_count=432, training_device='cuda:0',
                    preprocessing_id=identity, weights_changed=True)
    for key, value in expected.items():
        M.check(progress[key] == value, 'full fixed-budget CUDA progress ' + key)
    M.check(math.isfinite(progress['training_seconds']) and progress['training_seconds'] >= 0,
            'finite mixed training loop timer')
    columns = ['attempted', 'completed', 'target_cells', 'total_loss', 'waveform_loss',
               'dynamics_loss', 'gradient_norm', 'dynamics_valid_dimensions', 'requested', 'actual', 'restored']
    M.check(progress['trace_columns'] == columns and len(progress['losses']) == 512,
            'all 512 update rows with exact declared columns')
    sums = [0, 0, 0]
    for i, row in enumerate(progress['losses']):
        M.check(len(row) == 11 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                     and math.isfinite(x) for x in row), 'finite complete update trace')
        M.check(row[:2] == [i + 1, i + 1] and 0 < row[2] <= 8 * 288 and int(row[2]) == row[2],
                'contiguous successful update/target population')
        M.check(row[3] == row[4] and row[3] >= 0 and row[5] == 0 and row[6] > 0 and row[7] == 0,
                'sole waveform loss, finite positive gradient, no dynamics auxiliary')
        M.check(all(int(x) == x and 0 <= x <= 8 * 288 for x in row[8:])
                and row[8] == row[9] + row[10], 'requested/actual/restored update counts')
        for j in range(3):
            sums[j] += row[8 + j]
    for key, value in zip(('context_requested_deleted_coordinates', 'context_actual_deleted_coordinates',
                           'context_restored_coordinates'), sums):
        M.check(progress[key] == value, 'full-trace cumulative context count ' + key)


def check_heads(path, metadata, splits, candidate, master):
    M.check(read_json(path / 'report.json') == metadata, 'embedded/readout metadata association')
    M.check(metadata['protocol'] == 'fixed-feature-readouts-v1' and metadata['master_seed'] == str(master),
            'fixed-readout master association')
    M.check(metadata['recipe'] == {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100,
            'tiny_learning_rate': .01, 'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods',
            'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'unchanged fixed head recipe')
    M.check(len(metadata['methods']) == 1 and metadata['methods'][0]['method'] == 'native_' + candidate
            and metadata['pairs'] == [], 'one candidate native method without baseline refits')
    item = metadata['methods'][0]
    M.check(item['size'] == 32 and item['inputs_train_prepared'] is False
            and item['status'] in ('measured', 'unsupported_fit'), 'native32 ordinary TRAIN map')
    M.check(len(item['repetitions']) == 3, 'complete three fixed head repetitions')
    root = path / item['method']
    surfaces = []
    for view, split, population in zip(L.VIEWS, splits, ('training_population', 'validation_intact_population',
                                                       'validation_deleted_population')):
        surface = M.feature_archive(load(root / (view + '-features.pt')), len(split[2]), 32,
                                    split[2], split[3], 'DoubleStorage')
        features, valid = surface
        M.check(all(not ok or any(split[1][b * 288:(b + 1) * 288]) for b, ok in enumerate(valid)),
                'native support belongs to legal observations')
        M.check(item[population] == M.population(valid, split[3], split[2]), 'saved view population')
        surfaces.append(surface)
    valid = surfaces[0][1]
    supported = sum(valid) >= 2 and {splits[0][3][i] for i, keep in enumerate(valid) if keep} == {0, 1}
    M.check(item['status'] == ('measured' if supported else 'unsupported_fit')
            and bool(item['reason']) == (not supported), 'ordinary TRAIN support and unsupported retention')
    count = dict(outer_train_normalizer_fits=int(supported), ridge_fits=3 if supported else 0,
                 tiny_fits=3 if supported else 0, validation_fits=0)
    M.check(read_json(root / 'fit-counts.json') == count
            and item['outer_train_normalizer_fits'] == count['outer_train_normalizer_fits'],
            'one TRAIN outer map and three paired heads only when supported')
    M.check(metadata['fit_counts'] == dict(count, encoder_calls=0, pca_fits=0), 'exact supported fit counts')
    for rep, record in zip((2701, 2802, 2903), item['repetitions']):
        repid = f'rep-{rep}'
        M.check(record['id'] == repid and record['status'] == item['status']
                and record['actual_probe_seed_decimal'] == str(M.stream_seed(rep, 32))
                and record['ridge_parameters'] == 66 and record['neural_parameters'] == 562,
                'fixed width-paired repetitions and head sizes')
        directory = root / repid
        if not supported:
            M.check(not directory.exists() and not any(k in record for k in
                    ('fit_artifact', 'training', 'validation_intact', 'validation_deleted')),
                    'unsupported TRAIN fit invents no head/prediction')
            continue
        M.check(record['fit_artifact'] == f'{item["method"]}/{repid}/fit.pt', 'fixed TRAIN fit artifact')
        fit = load(directory / 'fit.pt')
        fitted = M.fit_schema(fit, 32, splits[0][2], rep, False, *surfaces[0])
        for view, split, (features, keep) in zip(L.VIEWS, splits, surfaces):
            prediction = load(directory / (view + '-predictions.pt'))
            ridge, tiny, _ = M.infer_saved_fit(fit, features, keep, split[3], split[2], prediction, False, 32, fitted)
            declared = record[view.replace('-', '_')]
            for head, predictions in (('ridge', ridge), ('tiny_secondary', tiny)):
                M.check_saved_score(M.score(predictions, split[3], keep), declared[head])
    return surfaces, count


def saved_role_paths(run, report):
    paths = []
    for cohort in report['cohorts']:
        base = run / f'seed-{cohort["master"]}-{cohort["task"]}'
        paths.extend((base / 'scaler.pt', base / 'progress.json'))
        for point in cohort['points']:
            root = base / f'point-{point["updates"]}'
            paths.extend((root / 'checkpoint.pt', root / 'readouts/report.json'))
            M.check(point['checkpoint'] == str(root / 'checkpoint.pt'), 'canonical checkpoint role association')
            method = point['readouts']['methods'][0]
            method_root = root / 'readouts' / ('native_' + report['candidate'])
            M.check(method['method'] == 'native_' + report['candidate'], 'closed native artifact directory')
            paths.append(method_root / 'fit-counts.json')
            paths.extend(method_root / (view + '-features.pt') for view in L.VIEWS)
            M.check(len(method['repetitions']) == 3, 'all declared repetition roles before decoding')
            for rep, record in zip((2701, 2802, 2903), method['repetitions']):
                M.check(record['id'] == f'rep-{rep}', 'closed repetition role name')
                if method['status'] == 'measured':
                    directory = method_root / record['id']
                    paths.append(directory / 'fit.pt')
                    paths.extend(directory / (view + '-predictions.pt') for view in L.VIEWS)
            if point['updates'] == 512:
                paths.extend(root / (name + '-reconstruction.pt') for name in ('training', 'validation'))
    M.check(set(run.rglob('*.pt')) == {p for p in paths if p.suffix == '.pt'},
            'exact saved archive role matrix; checkpoints retained but not decoded')
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    for directory in (args.run, args.input_root, args.freeze, args.output.parent):
        direct(directory)
        M.check(directory.is_relative_to(RUN_ROOT), 'bounded multiband protocol directory')
    M.check(args.output.is_absolute() and not args.output.exists() and not args.output.is_symlink(),
            'exclusive new saved-check proof path')
    metadata = [args.run / name for name in ('report.json', 'complete.json', 'input-manifest.json')]
    metadata += [args.freeze / name for name in ('freeze.json', 'inputs.sha256', 'sources.sha256', 'card.md', 'sdk-proof.json')]
    admit(metadata)
    report = read_json(args.run / 'report.json')
    complete = read_json(args.run / 'complete.json')
    frozen = read_json(args.freeze / 'freeze.json')
    candidate = report['candidate']
    M.check(candidate in ('v18', 'v19') and frozen['candidates'] == ['v18', 'v19'], 'closed candidate choices')
    M.check(report['protocol'] == frozen['protocol'] == 'multiband-screen-v1', 'same declared protocol')
    M.check(report['masters'] == frozen['masters'] == MASTERS and report['tasks'] == frozen['tasks'] == TASKS,
            'all four predeclared task/cohort trajectories')
    M.check([(c['master'], c['task']) for c in report['cohorts']] == [(m, t) for m in MASTERS for t in TASKS],
            'complete ordered four-cohort records')
    M.check(complete == dict(status='complete', protocol='multiband-screen-v1', candidate=candidate,
            masters=MASTERS, tasks=TASKS, updates_each=512, source_fingerprint=report['source_fingerprint'],
            report_sha256=remember(args.run / 'report.json')), 'exact completed report association')
    for report_key, freeze_key in (('source_fingerprint', 'source_fingerprint'), ('human_card_sha256', 'card_sha256'),
                                   ('sdk_proof_sha256', 'sdk_proof_sha256'), ('inputs_sha256', 'inputs_sha256')):
        M.check(report[report_key] == frozen[freeze_key], 'frozen source/card/SDK/input association')
    M.check(remember(args.freeze / 'card.md') == frozen['card_sha256']
            and remember(args.freeze / 'sdk-proof.json') == frozen['sdk_proof_sha256']
            and remember(args.freeze / 'sources.sha256') == frozen['source_fingerprint']
            and remember(args.freeze / 'inputs.sha256') == frozen['inputs_sha256']
            and frozen['input_root'] == str(args.input_root) and frozen['input_files'] == 12,
            'captured metadata byte and input-root binding')
    M.check(report['registered_parameter_count'] == 226877 and report['trainable_parameter_count'] == 226445
            and report['frozen_parameter_count'] == 432 and report['frozen_parameter_name'] == 'grouped_odd_relation_projection.weight',
            'registered/trainable/frozen capacities')
    M.check(report['data_recipe'] == 'two-component-v1' and report['datasets'] == DATASETS
            and report['designed_complexity_level_each'] == 5 and report['complexity_scale_max'] == 5,
            'separate dataset identity and designed complexity')
    M.check(all(report[key] is False for key in ('tasks_averaged', 'testing_accessed', 'stress_accessed', 'promotion')),
            'no task pooling, TEST/stress access or promotion')
    for cohort in report['cohorts']:
        M.check([p['updates'] for p in cohort['points']] == [0, 512], 'all initial/fixed-budget points')
    input_rows = manifest(args.freeze / 'inputs.sha256')
    expected_names = [f'results/seed-{m}-{t}/controlled-{v}.pt' for m in MASTERS for t in TASKS
                      for v in ('training', 'validation', 'validation-deleted')]
    M.check([name for _, name in input_rows] == expected_names, 'exact ordered twelve input roles')
    source_rows = manifest(args.freeze / 'sources.sha256')
    M.check(len(source_rows) == frozen['source_files'] and [name for _, name in source_rows] == sorted(name for _, name in source_rows),
            'canonical frozen source scope')
    sources = [p for _, name in source_rows for p in (ROOT / name, args.freeze / 'SOURCE' / name)]
    input_paths = [args.input_root / name for _, name in input_rows]
    saved_paths = saved_role_paths(args.run, report)
    admit(sources + input_paths + saved_paths)
    for digest, name in source_rows:
        M.check(remember(ROOT / name) == digest and remember(args.freeze / 'SOURCE' / name) == digest,
                'current and captured SOURCE bytes unchanged')
    for digest, name in input_rows:
        M.check(remember(args.input_root / name) == digest, 'once-generated quality input unchanged')
    declared_inputs = report['input_manifest']
    M.check(read_json(args.run / 'input-manifest.json') == declared_inputs
            and declared_inputs['checksum_algorithm'] == 'sha256-file-bytes' and declared_inputs['closed_roles'] is True,
            'producer closed input manifest association')
    M.check(declared_inputs['files'] == [dict(manifest_path=name, path=str(args.input_root / name), sha256=digest,
            bytes=(args.input_root / name).stat().st_size) for digest, name in sorted(input_rows, key=lambda x: x[1])],
            'every producer input role/hash/size association')
    splits_by_cohort = {}
    universe = set()
    for master in MASTERS:
        for task in TASKS:
            base = args.input_root / f'results/seed-{master}-{task}'
            splits = [controlled(base / ('controlled-' + view + '.pt'), rows, master, task)
                      for view, rows in (('training', 256), ('validation', 128), ('validation-deleted', 128))]
            sources_for_cohort = check_splits(splits, master, task)
            M.check(not universe & sources_for_cohort, 'task/master/split source namespaces independent')
            universe.update(sources_for_cohort)
            splits_by_cohort[master, task] = splits
    M.check(len(universe) == 768, 'all twelve legal inputs checked before saved head replay')
    findings = []
    relation_values_checked = pipelines = outer = 0
    for cohort in report['cohorts']:
        master, task = cohort['master'], cohort['task']
        base = args.run / f'seed-{master}-{task}'
        M.check(cohort['model_tag'] == ('RPB-v18.alt-01' if candidate == 'v18' else 'RPB-v19')
                and cohort['dataset_id'] == DATASETS[task] and cohort['designed_complexity_level'] == 5
                and cohort['complexity_scale_max'] == 5, 'candidate instance/dataset association')
        splits = splits_by_cohort[master, task]
        scaler = load(base / 'scaler.pt')
        identity = check_train_scaler(scaler, splits[0])
        M.check(read_json(base / 'progress.json') == cohort['progress'], 'full embedded/saved progress association')
        check_progress(cohort['progress'], identity)
        M.check(set(cohort['costs']) == {'binding_checkpoint_snapshot_seconds', 'CUDA_feature_transfer_seconds',
                'CPU_heads_and_intervals_IO_seconds', 'CUDA_query_transfer_IO_seconds'}
                and all(math.isfinite(x) and x >= 0 for x in cohort['costs'].values()), 'finite declared cost scopes')
        initial_relations = None
        for point in cohort['points']:
            root = base / f'point-{point["updates"]}'
            surfaces, count = check_heads(root / 'readouts', point['readouts'], splits, candidate, master)
            pipelines += count['ridge_fits']
            outer += count['outer_train_normalizer_fits']
            relations = [([row[20:32] for row in features], keep) for features, keep in surfaces]
            if point['updates'] == 0:
                initial_relations = relations
            else:
                for (before, support_before), (after, support_after) in zip(initial_relations, relations):
                    M.check(support_before == support_after and len(before) == len(after), 'same relation support at0/512')
                    M.check(all(struct.pack('<d', a) == struct.pack('<d', b)
                                for lhs, rhs in zip(before, after) for a, b in zip(lhs, rhs)),
                            'exact fixed relation12 coordinates at0/512; spectral coordinates may be even or odd')
                    relation_values_checked += len(after) * 12
                for split, name in zip(splits[:2], ('training', 'validation')):
                    summary = cohort[name + '_reconstruction']
                    artifact = name + '-reconstruction.pt'
                    M.check(summary['artifact'] == artifact, 'fixed query basename role')
                    asset = load(root / artifact)
                    replay = L.query_archive(asset, split, scaler)
                    L.query_summary(summary, replay, asset, artifact)
        findings.append(dict(master=master, task=task, dataset_id=DATASETS[task], status='passed'))
    M.check(report['counts'] == dict(encoder_trajectories=4, encoder_updates_each=512, sampled_rows=16384,
            retained_points=8, head_pipelines=pipelines, individual_heads=2 * pipelines, helper_outer_train_fits=outer,
            native_exports=24, query_writer_calls=8, necessary_query_forwards=32, generator_calls=0,
            information_fit_calls=0, PCA_fits=0, old_encoder_or_head_refits=0, CPU_encoder_calls=0, skipped_attempts=0),
            'actual supported heads and declared execution totals')
    M.check(relation_values_checked == 24576, 'all four cohorts/all three views fixed relation witness count')
    for path, digest in READ_SHA.items():
        M.check(sha(path) == digest, 'all read input/SOURCE bytes preserved')
    proof = dict(protocol='multiband-screen-saved-check-v1', status='passed', candidate=candidate,
                 masters=MASTERS, tasks=TASKS, source_fingerprint=report['source_fingerprint'],
                 human_card_sha256=report['human_card_sha256'], inputs_sha256=report['inputs_sha256'],
                 report_sha256=sha(args.run / 'report.json'), complete_sha256=sha(args.run / 'complete.json'),
                 reader_sha256=sha(Path(__file__)), modules_sha256=PINNED_MODULES,
                 checks=M.CHECKS + R.CHECKS, CPU_archive_decodes=DECODES, cohorts=findings,
                 fixed_relation_coordinate_values_checked=relation_values_checked,
                 encoder_forwards=0, encoder_updates=0, head_fits=0, PCA_fits=0,
                 information_fits=0, bootstrap_replays=0, checkpoint_bodies_decoded=0,
                 limits='CUDA execution and checkpoints remain producer/admission claims; saved logits, scores, TRAIN scalers, '
                        'original query reductions and reported trace/counters were checked without model execution.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, sort_keys=True))


if __name__ == '__main__':
    main()

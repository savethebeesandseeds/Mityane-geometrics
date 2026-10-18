#!/usr/bin/env python3
"""Saved-only check using the existing managed container's standard library.

Reuses the pinned, passed CPU archive codec/arithmetic, without its old run driver.
No encoder/optimizer load, model execution, head fitting, PCA or bootstrap.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/embedding')
MODULE_ROOT = ROOT / 'code/evaluation/protocols/visible_difference_v1/reader-v2'
DECODES = 0
MASTERS = [80787, 81888, 82989, 84090, 85191]
PINNED_MODULES = {
    'archive_codec.py': '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d',
    'saved_cpu_math.py': '1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9',
    'legacy_implementation.py': 'f0c64b9fe67a90a9d95196433e05dec527515778d1635bdc909d1703fa278b68'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, filename):
    if sha(MODULE_ROOT / filename) != PINNED_MODULES[filename]:
        raise ValueError('pinned saved-arithmetic module changed: ' + filename)
    spec = importlib.util.spec_from_file_location(name, MODULE_ROOT / filename)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


R = module('architecture_archive_codec', 'archive_codec.py')
M = module('saved_cpu_math', 'saved_cpu_math.py')
M.R = R
L = module('architecture_saved_contracts', 'legacy_implementation.py')
L.R = R


def load(path):
    global DECODES
    DECODES += 1
    return R.load(path)


def controlled(path, master, deleted=False):
    a = load(path)
    expected = {'observations', 'feature_mask', 'source_ids_json', 'labels_scoring_only'}
    if deleted:
        expected.add('requested_erasure')
    M.check(set(a) == expected, 'exact fresh controlled role schema')
    rows = a['observations']['shape'][0]
    data = M.tensor(a['observations'], 'DoubleStorage', [rows, 3, 32, 3])
    mask = M.tensor(a['feature_mask'], 'BoolStorage', [rows, 3, 32, 3])
    ids = json.loads(R.text(a['source_ids_json']))
    labels = M.tensor(a['labels_scoring_only'], 'LongStorage', [rows])
    M.check(rows in (256, 128) and len(ids) == rows, 'fresh fixed row populations')
    M.check(all(math.isfinite(x) and (seen or x == 0) for x, seen in zip(data, mask)),
            'finite legal observations with hidden zeros')
    prefix = f'fixed-prior-fresh-confirmation-v1/lag_sign/structured-hard-timing-v1/seed-{master}/lag_sign/source-'
    previous = -1
    for i in range(0, rows, 2):
        M.check(ids[i].startswith(prefix) and ids[i + 1] == ids[i], 'fresh paired source namespace')
        suffix = ids[i][len(prefix):]
        M.check(suffix.isdigit() and str(int(suffix)) == suffix and previous < int(suffix) < 192,
                'ascending original source universe')
        previous = int(suffix)
        M.check({labels[i], labels[i + 1]} == {0, 1}, 'opposite pair labels')
        M.check(mask[i*288:(i+1)*288] == mask[(i+1)*288:(i+2)*288], 'pair-shared observation mask')
    if deleted:
        erasure = M.tensor(a['requested_erasure'], 'BoolStorage', [rows, 3, 32, 3])
        M.check(all(not (e and m) for e, m in zip(erasure, mask)), 'requested erasures hidden')
    return data, mask, ids, labels


def check_splits(splits):
    train, intact, deleted = splits
    ti, vi = set(train[2]), set(intact[2])
    M.check(len(ti) == 128 and len(vi) == 64 and len(ti | vi) == 192,
            'fresh source-disjoint split covers universe')
    M.check(intact[2] == deleted[2] and list(intact[3]) == list(deleted[3]),
            'deleted source/label association unchanged')
    M.check(all(not m or (parent and x == original) for x, m, original, parent in
                zip(deleted[0], deleted[1], intact[0], intact[1])), 'exact deletion subset values')


def check_train_scaler(scaler, training):
    mean, scale, identity = M.scaler(scaler)
    count = M.tensor(scaler['count'], 'LongStorage', [3, 3])
    floors = M.tensor(scaler['floor_applied'], 'BoolStorage', [3, 3])
    data, mask, _, _ = training
    for c in range(3):
        for f in range(3):
            values = [data[b*288+c*96+h*3+f] for b in range(256) for h in range(32)
                      if mask[b*288+c*96+h*3+f]]
            i = c*3+f
            M.check(len(values) == count[i] and bool(values), 'legal TRAIN scaler count')
            wanted_mean = math.fsum(values) / len(values)
            sd = math.sqrt(math.fsum((x-wanted_mean)**2 for x in values) / len(values))
            M.close(mean[i], wanted_mean, 'legal TRAIN scaler mean')
            M.close(scale[i], max(1e-6, sd), 'legal TRAIN population scale')
            M.check(bool(floors[i]) == (sd < 1e-6), 'legal TRAIN scaler floor condition')
    return identity


def check_heads(path, metadata, splits):
    M.check(json.loads((path / 'report.json').read_text()) == metadata, 'embedded/readout metadata association')
    M.check(metadata['recipe'] == {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100,
            'tiny_learning_rate': .01, 'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods',
            'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'fixed head recipe')
    M.check(len(metadata['methods']) == 1 and metadata['methods'][0]['method'] == 'native_v18', 'fixed-prior native method')
    for item in metadata['methods']:
        M.check(item['size'] == 32 and item['status'] in ('measured', 'unsupported_fit'), 'native32 status')
        M.check(len(item['repetitions']) == 3, 'complete three fixed head repetitions')
        root = path / item['method']
        surfaces = [M.feature_archive(load(root / (view + '-features.pt')), len(split[2]), 32,
                    split[2], split[3], 'DoubleStorage') for view, split in zip(L.VIEWS, splits)]
        for rep, record in zip((2701, 2802, 2903), item['repetitions']):
            M.check(record['id'] == f'rep-{rep}' and record['actual_probe_seed_decimal'] == str(M.stream_seed(rep, 32)), 'fixed repetitions/paired seeds')
            if item['status'] == 'unsupported_fit':
                M.check(record['status'] == 'unsupported_fit' and bool(item['reason']), 'unsupported fit retained')
                continue
            directory = root / record['id']
            fit = load(directory / 'fit.pt')
            fitted = M.fit_schema(fit, 32, splits[0][2], rep, False, *surfaces[0])
            for view, split, (features, valid) in zip(L.VIEWS, splits, surfaces):
                prediction = load(directory / (view + '-predictions.pt'))
                ridge, tiny, _ = M.infer_saved_fit(fit, features, valid, split[3], split[2], prediction, False, 32, fitted)
                scores = {'ridge': M.score(ridge, split[3], valid), 'tiny_secondary': M.score(tiny, split[3], valid)}
                declared = record[view.replace('-', '_')]
                for head in ('ridge', 'tiny_secondary'):
                    M.check_saved_score(scores[head], declared[head])
    return surfaces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text())
    frozen = json.loads((args.freeze / 'freeze.json').read_text())
    M.check(report['protocol'] == frozen['protocol'] == 'unit-relation-screen-v1', 'fresh protocol')
    selected = frozen['masters']
    M.check(report['masters'] == selected and set(selected) <= set(MASTERS), 'all declared cohorts retained')
    M.check([c['master'] for c in report['cohorts']] == selected, 'complete ordered cohort records')
    for report_key, freeze_key in (('source_fingerprint', 'source_fingerprint'),
                                   ('human_card_sha256', 'card_sha256'),
                                   ('sdk_proof_sha256', 'sdk_proof_sha256'),
                                   ('inputs_sha256', 'inputs_sha256')):
        M.check(report[report_key] == frozen[freeze_key], 'exact frozen source/card/SDK/input association')
    for line in (args.freeze / 'inputs.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        M.check(sha(args.input_root / name) == digest, 'fresh input bytes unchanged')
    findings = []
    odd_values_checked = 0
    for cohort in report['cohorts']:
        master = cohort['master']
        base = args.input_root / f'results/seed-{master}-lag_sign'
        M.check(cohort['model_tag'] == 'RPB-v18', 'instance bundle preserved')
        M.check([p['updates'] for p in cohort['points']] == [0, 512], 'complete initial/fixed-budget points')
        splits = [controlled(base / ('controlled-' + view + '.pt'), master, view == 'validation-deleted')
                  for view in ('training', 'validation', 'validation-deleted')]
        check_splits(splits)
        scaler = load(args.run / f'seed-{master}/scaler.pt')
        identity = check_train_scaler(scaler, splits[0])
        M.check(cohort['progress']['preprocessing_id'] == identity, 'TRAIN scaler/progress association')
        initial_odd = None
        for point in cohort['points']:
            root = args.run / f'seed-{master}/point-{point["updates"]}'
            surfaces = check_heads(root / 'readouts', point['readouts'], splits)
            odd = [([row[20:32] for row in features], valid) for features, valid in surfaces]
            if point['updates'] == 0:
                initial_odd = odd
            if point['updates'] == 512:
                M.check(initial_odd is not None, 'initial fixed-prior witness retained')
                for (before, valid_before), (after, valid_after) in zip(initial_odd, odd):
                    M.check(valid_before == valid_after, 'same support at0/512')
                    M.check(len(before) == len(after) and all(
                        struct.pack('<d', a) == struct.pack('<d', b)
                        for lhs, rhs in zip(before, after) for a, b in zip(lhs, rhs)),
                        'exact fixed odd12 coordinates at0/512')
                    odd_values_checked += len(after) * 12
                for split, name in zip(splits[:2], ('training', 'validation')):
                    summary = cohort[name + '_reconstruction']
                    asset = load(root / summary['artifact'])
                    replay = L.query_archive(asset, split, scaler)
                    L.query_summary(summary, replay, asset, summary['artifact'])
        findings.append({'master': master, 'status': 'passed'})
    proof = {'protocol': 'unit-relation-screen-saved-check-v1', 'status': 'passed',
             'report_sha256': sha(args.run / 'report.json'), 'reader_sha256': sha(Path(__file__)),
             'modules_sha256': {name: sha(MODULE_ROOT / name) for name in
                 ('archive_codec.py', 'saved_cpu_math.py', 'legacy_implementation.py')},
             'checks': M.CHECKS + R.CHECKS, 'CPU_archive_decodes': DECODES, 'cohorts': findings,
             'fixed_odd_coordinate_values_checked': odd_values_checked,
             'encoder_forwards': 0, 'encoder_updates': 0, 'head_fits': 0, 'PCA_fits': 0}
    with args.output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, sort_keys=True))


if __name__ == '__main__':
    main()

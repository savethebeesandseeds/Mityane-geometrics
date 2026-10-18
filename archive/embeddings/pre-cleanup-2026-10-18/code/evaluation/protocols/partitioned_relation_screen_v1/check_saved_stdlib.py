#!/usr/bin/env python3
"""Saved-only check using the existing managed container's standard library.

Reuses the pinned, passed CPU archive codec/arithmetic, without its old run driver.
No encoder/optimizer load, model execution, head fitting, PCA or bootstrap.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/embedding')
MODULE_ROOT = ROOT / 'code/evaluation/protocols/visible_difference_v1/reader-v2'
DECODES = 0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, filename):
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


def controlled(path):
    a = load(path)
    rows = a['observations']['shape'][0]
    data = M.tensor(a['observations'], 'DoubleStorage', [rows, 3, 32, 3])
    mask = M.tensor(a['feature_mask'], 'BoolStorage', [rows, 3, 32, 3])
    ids = json.loads(R.text(a['source_ids_json']))
    labels = M.tensor(a['labels_scoring_only'], 'LongStorage', [rows])
    return data, mask, ids, labels


def check_heads(path, metadata, splits):
    M.check(json.loads((path / 'report.json').read_text()) == metadata, 'embedded/readout metadata association')
    M.check(metadata['recipe'] == {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100,
            'tiny_learning_rate': .01, 'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods',
            'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'fixed head recipe')
    for item in metadata['methods']:
        M.check(item['size'] == 32 and item['status'] in ('measured', 'unsupported_fit'), 'native32 status')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text())
    M.check(report['protocol'] == 'partitioned-relation-screen-v1', 'screen protocol')
    findings = []
    for cohort in report['cohorts']:
        master = cohort['master']
        base = args.input_root / f'results/seed-{master}-lag_sign'
        splits = [controlled(base / ('controlled-' + view + '.pt')) for view in ('training', 'validation', 'validation-deleted')]
        scaler = load(args.run / f'seed-{master}/scaler.pt')
        M.scaler(scaler)
        for point in cohort['points']:
            root = args.run / f'seed-{master}/point-{point["updates"]}'
            check_heads(root / 'readouts', point['readouts'], splits)
            if point['updates'] == 512:
                for split, name in zip(splits[:2], ('training', 'validation')):
                    summary = cohort[name + '_reconstruction']
                    asset = load(root / summary['artifact'])
                    replay = L.query_archive(asset, split, scaler)
                    L.query_summary(summary, replay, asset, summary['artifact'])
        findings.append({'master': master, 'status': 'passed'})
    proof = {'protocol': 'partitioned-relation-screen-saved-check-v1', 'status': 'passed',
             'report_sha256': sha(args.run / 'report.json'), 'reader_sha256': sha(Path(__file__)),
             'modules_sha256': {name: sha(MODULE_ROOT / name) for name in
                 ('archive_codec.py', 'saved_cpu_math.py', 'legacy_implementation.py')},
             'checks': M.CHECKS + R.CHECKS, 'CPU_archive_decodes': DECODES, 'cohorts': findings,
             'encoder_forwards': 0, 'encoder_updates': 0, 'head_fits': 0, 'PCA_fits': 0}
    with args.output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, sort_keys=True))


if __name__ == '__main__':
    main()

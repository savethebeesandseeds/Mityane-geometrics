#!/usr/bin/env python3
"""Check saved CPU features, fitted-head arithmetic and fixed-query reductions.

Never load model/optimizer checkpoints or execute an encoder, fit, PCA or bootstrap.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

sys.dont_write_bytecode = True
CHECKS = 0
DECODES = 0


def require(value, why):
    global CHECKS
    CHECKS += 1
    if not bool(value):
        raise ValueError(why)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    global DECODES
    DECODES += 1
    # Only explicitly named CPU feature/fit/prediction/query/scaler/data roles.
    module = torch.jit.load(str(path), map_location='cpu')
    tensors = dict(module.state_dict())
    require(bool(tensors) and all(x.device.type == 'cpu' for x in tensors.values()), 'saved CPU roles')
    return tensors


def text(tensor):
    require(tensor.dtype == torch.uint8 and tensor.ndim == 1, 'UTF8 metadata tensor')
    return bytes(tensor.tolist()).decode()


def near(left, right, why):
    require(math.isfinite(left) and math.isfinite(right)
            and abs(left - right) <= 2e-9 + 2e-9 * abs(right), why)


def same(left, right, why):
    require(left.shape == right.shape and torch.equal(left, right), why)


def close(left, right, why):
    require(left.shape == right.shape and torch.isfinite(left).all()
            and torch.isfinite(right).all() and torch.allclose(left, right, atol=2e-9, rtol=2e-8), why)


def score(classes, valid, labels, summary):
    total = valid.numel()
    kept = int(valid.sum())
    correct = int(((classes == labels) & valid).sum())
    require((summary['total'], summary['valid'], summary['correct']) == (total, kept, correct), 'exact score counts')
    near(summary['coverage'], kept / total, 'coverage')
    if kept:
        near(summary['accuracy'], correct / kept, 'conditional accuracy')
    else:
        require(summary['accuracy'] is None, 'unsupported score remains undefined')


def check_heads(directory, metadata, controlled):
    require(json.loads((directory / 'report.json').read_text()) == metadata, 'embedded/readout report association')
    require(metadata['recipe'] == {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100,
            'tiny_learning_rate': .01, 'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods',
            'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'fixed head recipe')
    for method in metadata['methods']:
        require(method['status'] in ('measured', 'unsupported_fit') and method['size'] == 32, 'declared native32 status')
        root = directory / method['method']
        surfaces = {view: load(root / f'{view}-features.pt') for view in controlled}
        for view, surface in surfaces.items():
            source = controlled[view]
            same(surface['labels_scoring_only'], source['labels_scoring_only'], 'feature labels retain original rows')
            same(surface['source_ids_json'], source['source_ids_json'], 'feature source IDs retain original rows')
            require(surface['features'].shape == (surface['valid'].numel(), 32)
                    and surface['valid'].dtype == torch.bool and torch.isfinite(surface['features']).all(), 'finite native features')
        require([x['id'] for x in method['repetitions']] == ['rep-2701', 'rep-2802', 'rep-2903'], 'all fixed repetitions')
        if method['status'] == 'unsupported_fit':
            require(bool(method['reason']) and all(r['status'] == 'unsupported_fit' for r in method['repetitions']),
                    'unsupported fits retain their reason and all repetitions')
            continue
        for rep in method['repetitions']:
            repdir = root / rep['id']
            fit = load(repdir / 'fit.pt')
            require(fit['ridge_penalty'].item() == 1 and fit['tiny_hidden'].item() == 16
                    and fit['tiny_steps'].item() == 100 and fit['tiny_learning_rate'].item() == .01, 'saved head budgets')
            require(text(fit['actual_probe_seed_decimal']) == rep['actual_probe_seed_decimal'], 'paired head seed')
            same(fit['training_source_ids_json'], controlled['training']['source_ids_json'], 'fit TRAIN lineage')
            for view, surface in surfaces.items():
                p = load(repdir / f'{view}-predictions.pt')
                for key in ('labels_scoring_only', 'source_ids_json', 'valid'):
                    same(p[key], surface[key], 'prediction/feature association')
                x = surface['features'].double()
                if fit['outer_normalizer_applied'].item():
                    x = (x - fit['feature_mean']) / fit['feature_scale']
                close(p['probe_input_features'], x, 'saved TRAIN outer normalization')
                ridge_logits = ((x - fit['ridge_mean']) / fit['ridge_scale']) @ fit['ridge_weights'] + fit['ridge_intercept']
                hidden = ((x - fit['tiny_mean']) / fit['tiny_scale']) @ fit['tiny_w1'] + fit['tiny_b1']
                logits = hidden.tanh() @ fit['tiny_w2'] + fit['tiny_b2']
                close(ridge_logits, p['ridge_logits'], 'frozen ridge arithmetic')
                close(hidden, p['tiny_hidden_preactivation'], 'frozen neural hidden arithmetic')
                close(logits, p['tiny_logits'], 'frozen neural logits')
                same(p['ridge'], p['ridge_logits'].argmax(1), 'saved ridge decision')
                same(p['tiny_secondary'], p['tiny_logits'].argmax(1), 'saved neural decision')
                row = rep[view.replace('-', '_')]
                score(p['ridge'], p['valid'], p['labels_scoring_only'], row['ridge'])
                score(p['tiny_secondary'], p['valid'], p['labels_scoring_only'], row['tiny_secondary'])


def check_query(path, summary, controlled, scaler):
    q = load(path)
    same(q['source_ids_json'], controlled['source_ids_json'], 'query original source order')
    prediction, target, mask = q['standardized_prediction'], q['standardized_target'], q['target_mask']
    require(prediction.shape == target.shape == mask.shape and mask.dtype == torch.bool, 'query dimensions')
    observed = controlled['feature_mask'].unsqueeze(0).expand_as(mask)
    require(mask.shape == (4, controlled['feature_mask'].shape[0], 3, 32, 3), 'original four patch queries')
    requested, visible, eligible, effective = [], [], [], []
    for trial in range(4):
        artificial = torch.zeros_like(controlled['feature_mask'])
        artificial[:, :, trial*8:(trial+1)*8, :] = True
        target = controlled['feature_mask'] & artificial
        context = controlled['feature_mask'] & ~artificial
        legal_channel = (context.reshape(context.shape[0], 3, 4, 24).any(-1).sum(-1) >= 2) & target.flatten(2).any(2)
        actual_artificial = artificial & legal_channel[:, :, None, None]
        requested.append(target)
        eligible.append(legal_channel)
        visible.append(controlled['feature_mask'] & ~actual_artificial)
        effective.append(controlled['feature_mask'] & actual_artificial)
    same(q['requested_observed_target_mask'], torch.stack(requested), 'exact original requested query law')
    same(q['trial_channel_eligible'], torch.stack(eligible), 'exact original query channel eligibility')
    same(q['visible_mask'], torch.stack(visible), 'exact original query visible context')
    same(mask, torch.stack(effective), 'exact original effective query population')
    require(not (mask & ~observed).any(), 'query targets never naturally hidden')
    require(not (mask & q['visible_mask']).any(), 'query target/context separation')
    require(torch.isfinite(prediction).all() and torch.isfinite(target).all(), 'finite saved query')
    raw = controlled['observations'].double()
    legal = controlled['feature_mask']
    scaled = torch.where(legal, (torch.where(legal, raw, 0) - scaler['mean'][None, :, None, :])
                         / scaler['scale'][None, :, None, :], 0).float().double()
    expected_target = torch.where(mask, scaled.unsqueeze(0).expand_as(mask), 0)
    same(target, expected_target, 'query target equals original legal data in frozen TRAIN units')
    error = (prediction - target).abs()
    counts = mask.sum((0, 3, 4))
    valid = counts > 0
    mae = (error * mask).sum((0, 3, 4)) / counts.clamp_min(1)
    huber = torch.where(error <= 1, .5 * error.square(), error - .5)
    channel_huber = (huber * mask).sum((0, 3, 4)) / counts.clamp_min(1)
    example_valid = valid.any(1)
    example_mae = mae.sum(1) / valid.sum(1).clamp_min(1)
    example_huber = channel_huber.sum(1) / valid.sum(1).clamp_min(1)
    same(counts, q['channel_target_counts'], 'query cell counts')
    same(valid, q['channel_valid'], 'query channel support')
    same(example_valid, q['example_valid'], 'query example support')
    close(mae, q['channel_standardized_mae'], 'query channel MAE')
    close(channel_huber, q['channel_standardized_huber'], 'query channel Huber')
    close(example_mae, q['example_standardized_mae'], 'query hierarchical MAE')
    if example_valid.any():
        near(float(example_mae[example_valid].mean()), summary['standardized_mae'], 'fixed-query reported MAE')
        near(float(example_huber[example_valid].mean()), summary['standardized_huber'], 'fixed-query reported Huber')
    else:
        require(summary['standardized_mae'] is None and summary['standardized_huber'] is None,
                'unsupported query error stays undefined')
    require(int(counts.sum()) == summary['valid_target_cells'] and int(example_valid.sum()) == summary['valid_examples'], 'reported query support')
    require(summary['total_examples'] == controlled['feature_mask'].shape[0]
            and summary['valid_channels'] == int(valid.sum())
            and summary['requested_observed_target_cells'] == int(torch.stack(requested).sum())
            and summary['status'] == ('measured' if example_valid.any() else 'unsupported_zero_support'), 'query summary support/status')
    near(float(example_valid.double().mean()), summary['example_coverage'], 'query example coverage')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = json.loads((args.run / 'report.json').read_text())
    require(report['protocol'] == 'architecture-screen-v1', 'screen protocol')
    findings = []
    for cohort in report['cohorts']:
        master = cohort['master']
        base = args.input_root / f'results/seed-{master}-lag_sign'
        controlled = {view: load(base / f'controlled-{suffix}.pt') for view, suffix in
                      [('training', 'training'), ('validation-intact', 'validation'), ('validation-deleted', 'validation-deleted')]}
        scaler_path = args.run / f'seed-{master}/scaler.pt'
        scaler = load(scaler_path)
        raw, legal = controlled['training']['observations'], controlled['training']['feature_mask']
        counts = legal.sum((0, 2))
        mean = torch.where(legal, raw, 0).sum((0, 2)) / counts
        centered = torch.where(legal, raw - mean[None, :, None, :], 0)
        scale = (centered.square().sum((0, 2)) / counts).sqrt().clamp_min(1e-6)
        same(scaler['count'], counts, 'scaler legal TRAIN counts')
        close(scaler['mean'], mean, 'scaler TRAIN-only means')
        close(scaler['scale'], scale, 'scaler TRAIN-only scales')
        roots = []
        for point in cohort['points']:
            matches = [p for p in args.run.rglob('readouts/report.json')
                       if f'seed-{master}' in str(p) and f'point-{point["updates"]}' in p.parts]
            require(len(matches) == 1, 'one explicit readout role per cohort/point')
            check_heads(matches[0].parent, point['readouts'], controlled)
            if point['updates'] == 512:
                roots.append(matches[0].parent.parent)
        require(len(roots) == 1, 'one retained trained point')
        for split in ('training', 'validation'):
            summary = cohort[f'{split}_reconstruction']
            path = roots[0] / summary['artifact']
            check_query(path, summary, controlled['training' if split == 'training' else 'validation-intact'], scaler)
        findings.append({'master': master, 'status': 'passed'})
    proof = {'protocol': 'architecture-screen-saved-check-v1', 'status': 'passed',
             'report_sha256': sha(args.run / 'report.json'), 'reader_sha256': sha(Path(__file__)),
             'checks': CHECKS, 'CPU_archive_decodes': DECODES, 'cohorts': findings,
             'encoder_forwards': 0, 'encoder_updates': 0, 'head_fits': 0, 'PCA_fits': 0}
    with args.output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, sort_keys=True))


if __name__ == '__main__':
    main()

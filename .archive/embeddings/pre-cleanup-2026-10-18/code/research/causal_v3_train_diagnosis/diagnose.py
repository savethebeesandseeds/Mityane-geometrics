#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Card-bound scalar diagnosis of saved CPB-v3 TRAIN data; no model execution.

The sibling archive_reader supplies inert C++ OutputArchive descriptors only.
This program imports no Torch/NumPy and performs no fitting or neural forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from archive_reader import Archive, Record, Tensor


TRAIN_BEGIN = 1704067259999
TRAIN_END = 1705881599999
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
REPS = (2701, 2802, 2903)
BLOCKS = (('learned20', 0, 20), ('fixed_spectral12', 20, 32), ('change4', 32, 36))
ALLOWED_IDS = {'initial_features', 'final_features', 'train_targets',
               'initial_query', 'final_query', 'initial_query_summary',
               'final_query_summary', 'initial_checkpoint', 'final_checkpoint'} | {
    f'{stage}_{kind}_{rep}' for stage in ('initial', 'final')
    for kind in ('head', 'fit') for rep in REPS}
QUANTILES = (0., .01, .05, .25, .5, .75, .95, .99, 1.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_pin(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def mean(values):
    return math.fsum(values) / len(values) if values else None


def quantiles(values):
    if not values:
        return None
    require(all(math.isfinite(v) for v in values), 'Nonfinite descriptive value')
    ordered = sorted(values)
    result = {}
    for q in QUANTILES:
        position = q * (len(ordered) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        result[str(q)] = ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])
    return result


def variance(values):
    center = mean(values)
    return mean([(v - center) ** 2 for v in values]) if values else None


def pearson(x, y):
    require(len(x) == len(y), 'Correlation support differs')
    if len(x) < 2:
        return None
    xm, ym = mean(x), mean(y)
    xx = math.fsum((v - xm) ** 2 for v in x)
    yy = math.fsum((v - ym) ** 2 for v in y)
    if xx == 0. or yy == 0.:
        return None
    return max(-1., min(1., math.fsum((a-xm)*(b-ym) for a, b in zip(x, y)) / math.sqrt(xx*yy)))


def score(prediction, target, valid):
    require(len(prediction) == len(target) == len(valid), 'Score lengths differ')
    rows = [(p, t) for p, t, ok in zip(prediction, target, valid) if ok]
    if not rows:
        return {'total': len(valid), 'valid': 0, 'coverage': 0., 'r2': None,
                'mae': None, 'rmse': None}
    require(all(math.isfinite(p) and math.isfinite(t) for p, t in rows), 'Nonfinite supported prediction')
    errors = [p-t for p, t in rows]
    target_mean = mean([t for _, t in rows])
    denominator = math.fsum((t-target_mean)**2 for _, t in rows)
    squared = math.fsum(e*e for e in errors)
    return {'total': len(valid), 'valid': len(rows), 'coverage': len(rows)/len(valid),
            'r2': 1.-squared/denominator if denominator > 0. else None,
            'mae': mean([abs(e) for e in errors]), 'rmse': math.sqrt(squared/len(rows))}


def near(a, b, label, absolute=1e-10, relative=1e-9):
    require(math.isfinite(a) and math.isfinite(b) and
            abs(a-b) <= absolute + relative*abs(b), f'{label}: {a!r} != {b!r}')


class Inputs:
    def __init__(self, card):
        require(card.get('baseline') == 'CPB-v3.market-01' and
                card.get('train_rows') == 112 and card.get('query_rows') == 8 and
                card.get('train_end_ms') == TRAIN_END, 'Card TRAIN scope differs')
        require([(b['begin'], b['end']) for b in card['native_blocks']] ==
                [(0,20), (20,32), (32,36)], 'Card native blocks differ')
        pins = card['allowed_inputs']
        require(isinstance(pins, list) and len(pins) == len(ALLOWED_IDS), 'Exactly21 declared input IDs required')
        self.pins = {}
        self.opened = {}
        self.archives = {}
        self.numeric_fields = []
        for pin in pins:
            identity = pin['id']
            require(identity in ALLOWED_IDS and identity not in self.pins, f'Unknown/duplicate input ID: {identity}')
            path = Path(pin['container_path'])
            require(path.is_absolute() and 'validation' not in path.name.lower(),
                    f'Validation numeric archive or relative input refused: {identity}')
            self.pins[identity] = dict(pin)
        require(set(self.pins) == ALLOWED_IDS, 'Missing card input IDs')
        self.verified = []
        for pin in pins:
            self.verified.append(self.verify(pin))
        self.preservation = [self.verify(pin) for pin in card.get('preservation_pins', [])]

    @staticmethod
    def verify(pin):
        path = Path(pin.get('container_path', pin.get('path', '')))
        require(path.is_absolute() and path.is_file(), f'Pinned file unavailable: {path}')
        actual = file_pin(path)
        require(actual['bytes'] == pin['bytes'] and actual['sha256'] == pin['sha256'],
                f'Pinned file differs: {path}')
        if 'id' in pin:
            actual['id'] = pin['id']
        return actual

    def archive(self, identity):
        require(identity in self.pins and not identity.endswith('_summary'), 'Archive ID outside card')
        if identity not in self.archives:
            path = self.pins[identity]['container_path']
            self.archives[identity] = Archive(path)
            self.opened[identity] = 'inert_archive_metadata'
        return self.archives[identity]

    def tensor(self, identity, *fields, shape=None):
        require(all('validation' not in name.lower() for name in fields), 'Validation numeric field refused')
        value = self.archive(identity).field(*fields)
        require(isinstance(value, Tensor), f'Expected inert tensor: {identity}:{fields}')
        if shape is not None:
            require(tuple(value.shape) == tuple(shape), f'Tensor shape differs: {identity}:{fields}: {value.shape}')
        values = value.values()
        require(len(values) == math.prod(value.shape), 'Tensor value count differs')
        self.numeric_fields.append({'id': identity, 'field': '.'.join(fields),
                                    'shape': list(value.shape), 'values': len(values)})
        return values

    def summary(self, identity):
        require(identity in ('initial_query_summary', 'final_query_summary'), 'Summary ID outside card')
        with Path(self.pins[identity]['container_path']).open(encoding='utf-8-sig') as source:
            container = json.load(source)
        require(isinstance(container, dict) and isinstance(container.get('train'), dict), 'Expected aggregate query JSON with TRAIN key')
        # The full file is pinned/decoded, but no validation numeric field is accessed.
        self.opened[identity] = 'aggregate_JSON_train_key_only'
        self.numeric_fields.append({'id': identity, 'field': 'train_only'})
        return container['train']

    def close(self):
        for archive in self.archives.values():
            archive.close()


def train_ids(values, expected=None):
    require(all(isinstance(v, int) and not isinstance(v, bool) and TRAIN_BEGIN <= v <= TRAIN_END for v in values),
            'Row ID outside declared TRAIN cutoff')
    require(len(set(values)) == len(values), 'Duplicate TRAIN row IDs')
    if expected is not None:
        require(values == expected, 'TRAIN row association differs')
    return values


def boolean(values, label):
    require(all(v in (False, True) for v in values), f'Nonbinary support: {label}')
    return [bool(v) for v in values]


def features(inputs, stage, rows, targets, target_valid):
    identity = stage + '_features'
    train_ids(inputs.tensor(identity, 'row_ids', shape=(112,)), rows)
    values = inputs.tensor(identity, 'values', shape=(112, 36))
    valid = boolean(inputs.tensor(identity, 'valid', shape=(112,)), identity)
    require(all(math.isfinite(v) for v in values), 'Nonfinite native feature')
    columns = [[values[r*36+d] for r in range(112)] for d in range(36)]
    coordinate_rows = []
    for d, column in enumerate(columns):
        x = [v for v, ok in zip(column, valid) if ok]
        correlations = {}
        for t, task in enumerate(TASKS):
            indices = [r for r in range(112) if valid[r] and target_valid[r*4+t]]
            correlations[task] = pearson([column[r] for r in indices], [targets[r*4+t] for r in indices])
        coordinate_rows.append({'coordinate': d, 'block': next(name for name, lo, hi in BLOCKS if lo <= d < hi),
                                'mean': mean(x), 'population_std': math.sqrt(variance(x)) if x else None,
                                'minimum': min(x) if x else None, 'maximum': max(x) if x else None,
                                'pearson_to_TRAIN_targets': correlations})
    blocks = {}
    for name, lo, hi in BLOCKS + (('complete36', 0, 36),):
        chosen = [[column[r] for r in range(112) if valid[r]] for column in columns[lo:hi]]
        centers = [mean(column) for column in chosen]
        n = sum(valid)
        trace, squared_trace = 0., 0.
        if n:
            for i, x in enumerate(chosen):
                for j, y in enumerate(chosen):
                    covariance = math.fsum((a-centers[i])*(b-centers[j]) for a, b in zip(x, y))/n
                    if i == j:
                        trace += covariance
                    squared_trace += covariance*covariance
        blocks[name] = {'width': hi-lo, 'valid_rows': n, 'covariance_trace': trace,
                        'covariance_frobenius_squared': squared_trace,
                        'participation_ratio': trace*trace/squared_trace if squared_trace > 0 else None,
                        'definition': 'trace(population_covariance)^2 / sum_ij(covariance_ij^2); no eigensolver or fit'}
    return {'rows': 112, 'valid_rows': sum(valid), 'coordinates': coordinate_rows, 'blocks': blocks}, values, valid


def head_scores(inputs, stage, rows, groups, targets, target_valid, native, feature_valid):
    reports = []
    for rep in REPS:
        identity = f'{stage}_head_{rep}'
        train_ids(inputs.tensor(identity, 'row_ids', shape=(112,)), rows)
        require(inputs.tensor(identity, 'group_ids', shape=(112,)) == groups, 'Head TRAIN groups differ')
        require(inputs.tensor(identity, 'targets', shape=(112, 4)) == targets, 'Saved head targets differ')
        require(boolean(inputs.tensor(identity, 'target_valid', shape=(112, 4)), identity) == target_valid,
                'Saved target support differs')
        valid = boolean(inputs.tensor(identity, 'valid', shape=(112, 4)), identity)
        require(all(not ok or (feature_valid[r] and target_valid[r*4+t])
                    for r in range(112) for t in range(4) for ok in [valid[r*4+t]]), 'Head support exceeds TRAIN support')
        ridge = inputs.tensor(identity, 'ridge_predictions', shape=(112, 4))
        neural = inputs.tensor(identity, 'neural_predictions', shape=(112, 4))
        task_scores = {}
        for t, task in enumerate(TASKS):
            task_scores[task] = {head: score(pred[t::4], targets[t::4], valid[t::4])
                                 for head, pred in (('linear', ridge), ('neural', neural))}
        fit_id = f'{stage}_fit_{rep}'
        mu = inputs.tensor(fit_id, 'feature_mean', shape=(36,))
        scale = inputs.tensor(fit_id, 'feature_scale', shape=(36,))
        weights = inputs.tensor(fit_id, 'ridge_weights', shape=(36, 4))
        target_scale = inputs.tensor(fit_id, 'target_scale', shape=(4,))
        require(all(math.isfinite(v) and v > 0 for v in scale+target_scale), 'Invalid saved readout scale')
        require(all(math.isfinite(v) for v in mu+weights), 'Invalid saved readout coefficients')
        seed = inputs.tensor(fit_id, 'actual_seed')
        require(len(seed) == 1, 'Saved head seed must be scalar')
        contributions = {}
        for task_index, task in enumerate(TASKS):
            per_block = {}
            for name, lo, hi in BLOCKS:
                # Arithmetic decomposition of an ALREADY FITTED linear readout;
                # no predictor object, optimization, new fit or neural forward.
                component = [target_scale[task_index] * math.fsum(
                    (native[r*36+d]-mu[d])/scale[d]*weights[d*4+task_index]
                    for d in range(lo, hi)) for r in range(112)]
                selected = [r for r in range(112) if valid[r*4+task_index]]
                vals = [component[r] for r in selected]
                per_block[name] = {'saved_standardized_weight_l2': math.sqrt(math.fsum(weights[d*4+task_index]**2 for d in range(lo, hi))),
                                   'TRAIN_contribution_population_std_in_target_units': math.sqrt(variance(vals)) if vals else None,
                                   'TRAIN_contribution_pearson_to_target': pearson(vals, [targets[r*4+task_index] for r in selected])}
            contributions[task] = per_block
        reports.append({'repetition': rep, 'saved_seed_signed_int64': seed[0], 'tasks': task_scores,
                        'saved_linear_readout_block_decomposition': contributions})
    return reports


def query(inputs, stage, expected_rows):
    identity = stage + '_query'
    rows = train_ids(inputs.tensor(identity, 'row_ids', shape=(8,)))
    require(rows == expected_rows[:8], 'Detailed query witness must be first8 TRAIN probes')
    shape = (8, 3, 128, 9)
    timestamps = inputs.tensor(identity, 'timestamps', shape=(8, 128))
    for b, endpoint in enumerate(rows):
        require(timestamps[b*128:(b+1)*128] == [endpoint+(h-127)*60000 for h in range(128)],
                'Query timestamps exceed or differ from exact TRAIN window')
    support = boolean(inputs.tensor(identity, 'target_mask', shape=shape), identity)
    visible = boolean(inputs.tensor(identity, 'visible', shape=shape), identity)
    observed = boolean(inputs.tensor(identity, 'observed', shape=shape), identity)
    require(all(not s or (o and not v) for s, o, v in zip(support, observed, visible)), 'Query target/visible/original support differs')
    target = inputs.tensor(identity, 'targets', shape=shape)
    predicted = inputs.tensor(identity, 'mean', shape=shape)
    var = inputs.tensor(identity, 'variance', shape=shape)
    native = inputs.tensor(identity, 'native', shape=(8, 36))
    require(all(math.isfinite(v) for v in target+predicted+var+native), 'Nonfinite TRAIN query values')
    require(all(v > 0 for v, s in zip(var, support) if s), 'Nonpositive supported query variance')
    index = lambda b, c, h, f: ((b*3+c)*128+h)*9+f
    cell_stats, row_mae, row_nll, row_counts = [], [], [], []
    for c in range(3):
        for f in range(9):
            chosen = [index(b,c,h,f) for b in range(8) for h in range(128) if support[index(b,c,h,f)]]
            error = [predicted[i]-target[i] for i in chosen]
            variances = [var[i] for i in chosen]
            squared_standardized = [e*e/v for e, v in zip(error, variances)]
            log_term = [.5*math.log(v) for v in variances]
            residual_term = [.5*v for v in squared_standardized]
            standardized = [e/math.sqrt(v) for e, v in zip(error, variances)]
            cell_stats.append({'channel_id': c, 'feature': f, 'target_cells': len(chosen),
                               'mean_absolute_error': mean([abs(e) for e in error]),
                               'mean_squared_error': mean([e*e for e in error]), 'mean_variance': mean(variances),
                               'mean_half_squared_error_over_variance': mean(residual_term),
                               'mean_half_log_variance': mean(log_term),
                               'mean_Gaussian_NLL': mean([a+b for a,b in zip(residual_term,log_term)]),
                               'variance_quantiles': quantiles(variances),
                               'standardized_residual_mean': mean(standardized),
                               'standardized_residual_quantiles': quantiles(standardized),
                               'squared_standardized_residual_quantiles': quantiles(squared_standardized)})
    for b in range(8):
        channel_mae, channel_nll = [], []
        total = 0
        for c in range(3):
            chosen = [index(b,c,h,f) for h in range(128) for f in range(9) if support[index(b,c,h,f)]]
            total += len(chosen)
            if chosen:
                channel_mae.append(mean([abs(predicted[i]-target[i]) for i in chosen]))
                channel_nll.append(mean([.5*((predicted[i]-target[i])**2/var[i]+math.log(var[i])) for i in chosen]))
        require(channel_mae, 'Detailed query example lacks support')
        row_mae.append(mean(channel_mae)); row_nll.append(mean(channel_nll)); row_counts.append(total)
    saved_mae = inputs.tensor(identity, 'row_mae', shape=(8,))
    saved_nll = inputs.tensor(identity, 'row_nll', shape=(8,))
    for b in range(8):
        near(row_mae[b], saved_mae[b], 'Saved B8 row MAE')
        near(row_nll[b], saved_nll[b], 'Saved B8 row NLL')
    summary = inputs.summary(stage + '_query_summary')
    require(summary['valid_rows'] == 112 and len(summary['row_mae']) == len(summary['row_nll']) == 112,
            'Aggregate TRAIN query population must be112')
    require(len(summary['row_target_cells']) == 112, 'Aggregate TRAIN support rows differ')
    for b in range(8):
        near(row_mae[b], summary['row_mae'][b], 'Aggregate TRAIN B8 MAE')
        near(row_nll[b], summary['row_nll'][b], 'Aggregate TRAIN B8 NLL')
        require(row_counts[b] == summary['row_target_cells'][b], 'Aggregate TRAIN B8 target count')
    near(mean(summary['row_mae']), summary['standardized_fixed_query_MAE'], 'All112 TRAIN mean MAE')
    near(mean(summary['row_nll']), summary['Gaussian_NLL_F64'], 'All112 TRAIN mean NLL')
    require(sum(summary['row_target_cells']) == summary['target_cells'], 'All112 TRAIN target count')
    report = {'detailed_population': 8, 'detailed_row_ids': rows, 'full_TRAIN_probe_population': 112,
              'scope_limit': 'Cellwise statistics describe only first8 saved TRAIN queries, not112 or VALIDATION',
              'per_channel_feature': cell_stats,
              'B8_hierarchical_mean_MAE': mean(row_mae), 'B8_hierarchical_mean_NLL': mean(row_nll),
              'B8_target_cells': sum(row_counts),
              'all112_TRAIN': {'standardized_fixed_query_MAE': summary['standardized_fixed_query_MAE'],
                               'Gaussian_NLL_F64': summary['Gaussian_NLL_F64'], 'target_cells': summary['target_cells'],
                               'row_mae_quantiles': quantiles(summary['row_mae']),
                               'row_nll_quantiles': quantiles(summary['row_nll']),
                               'row_target_cells': summary['row_target_cells']}}
    return report, {'row_ids': rows, 'native': native, 'support': support, 'mean': predicted,
                    'target': target, 'variance': var}


def checkpoint_metadata(inputs, stage):
    identity = stage + '_checkpoint'
    archive = inputs.archive(identity)
    text = lambda field: bytes(inputs.tensor(identity, field)).decode('utf-8')
    require(text('artifact_kind') == 'cpb_v3_inference_checkpoint_v1' and text('model_tag') == 'CPB-v3', 'Only typed CPB-v3 checkpoints admitted')
    require(text('initialization_kind') == 'fresh-seeded-constructor-no-import-v1' and text('pretrained_ancestor') == 'none', 'Fresh checkpoint provenance differs')
    source = inputs.tensor(identity, 'source_updates')
    updates = inputs.tensor(identity, 'causal_updates')
    seed = inputs.tensor(identity, 'initialization_seed')
    require(source == [0] and updates == [0 if stage == 'initial' else 2048] and seed == [943001], 'Checkpoint counters/seed differ')
    require(inputs.tensor(identity, 'channel_ids', shape=(3,)) == [0,1,2], 'Canonical channel order required')
    return archive, {'source_updates': source[0], 'causal_updates': updates[0], 'initialization_seed': seed[0],
                     'pretrained_ancestor': 'none'}


def analytic_change_gradients(inputs, witness):
    """Double-precision chain-rule arithmetic on saved tensors, not autograd.

    Values approximate the actual F32 intermediate gradient; this is not a
    replay, newly evaluated model or bit-parity claim.
    """
    identity = 'final_checkpoint'
    decoder = inputs.tensor(identity, 'payload', 'model', 'encoder', 'change_decoder', shape=(3456, 4))
    projection = inputs.tensor(identity, 'payload', 'model', 'variance_projection', shape=(27, 4))
    bias = inputs.tensor(identity, 'payload', 'model', 'variance_bias', shape=(27,))
    constants = inputs.tensor(identity, 'payload', 'link_constants_f32', shape=(3,))
    eps, shift, coefficient = constants
    require(all(math.isfinite(v) for v in decoder+projection+bias+constants), 'Nonfinite saved decoder tensor')
    require(0 < eps < 1 and coefficient > 0, 'Saved link constants invalid')
    softplus = lambda x: x if x > 20 else math.log1p(math.exp(x))
    sigmoid = lambda x: 1. if x > 20 else (1./(1.+math.exp(-x)) if x >= 0 else math.exp(x)/(1.+math.exp(x)))
    u0 = softplus(shift)
    gm, gv, rows = [], [], []
    maximum_variance_error = 0.
    support = witness['support']
    at = lambda b,c,h,f: ((b*3+c)*128+h)*9+f
    for b in range(8):
        counts = [sum(support[at(b,c,h,f)] for h in range(128) for f in range(9)) for c in range(3)]
        eligible_channels = sum(n > 0 for n in counts)
        require(eligible_channels > 0, 'Gradient query row lacks eligible channels')
        z = witness['native'][b*36+32:b*36+36]
        mean_gradient, variance_gradient = [0.]*4, [0.]*4
        for c in range(3):
            if not counts[c]:
                continue
            factor = 1./(8*eligible_channels*counts[c])
            for f in range(9):
                cf = c*9+f
                linear = math.fsum(z[d]*projection[cf*4+d] for d in range(4))+bias[cf]
                rebuilt_variance = eps+coefficient*softplus(linear+shift)/u0
                link_derivative = coefficient*sigmoid(linear+shift)/u0
                variance_derivative = 0.
                for h in range(128):
                    i = at(b,c,h,f)
                    stored_variance = witness['variance'][i]
                    maximum_variance_error = max(maximum_variance_error, abs(rebuilt_variance-stored_variance))
                    near(rebuilt_variance, stored_variance, 'Saved variance/link/canonical mapping', absolute=5e-6, relative=5e-5)
                    if not support[i]:
                        continue
                    error = witness['mean'][i]-witness['target'][i]
                    dmean = factor*error/stored_variance
                    variance_derivative += factor*.5*(1./stored_variance-error*error/(stored_variance*stored_variance))
                    for d in range(4):
                        mean_gradient[d] += dmean*decoder[((c*128+h)*9+f)*4+d]
                for d in range(4):
                    variance_gradient[d] += variance_derivative*link_derivative*projection[cf*4+d]
        gm.extend(mean_gradient); gv.extend(variance_gradient)
        norm_m, norm_v = math.sqrt(math.fsum(v*v for v in mean_gradient)), math.sqrt(math.fsum(v*v for v in variance_gradient))
        dot = math.fsum(a*b for a,b in zip(mean_gradient, variance_gradient))
        rows.append({'row_id': witness['row_ids'][b], 'mean_path_gradient': mean_gradient,
                     'variance_path_gradient': variance_gradient, 'mean_path_l2': norm_m,
                     'variance_path_l2': norm_v, 'variance_to_mean_norm_ratio': norm_v/norm_m if norm_m else None,
                     'dot_product': dot, 'cosine': dot/(norm_m*norm_v) if norm_m and norm_v else None})
    nm, nv = math.sqrt(math.fsum(v*v for v in gm)), math.sqrt(math.fsum(v*v for v in gv))
    dot = math.fsum(a*b for a,b in zip(gm,gv))
    return {'status': 'computed_from_saved_tensors', 'population': 8,
            'scope': 'Gradient with respect to each B8 query change4 native; no parameter gradient, model execution or autograd',
            'precision': 'Analytic F64 arithmetic approximates F32 intermediate chain rule; not gradient bit parity',
            'normalization': 'same eligible-example/channel/target-cell hierarchy as saved Gaussian NLL',
            'canonical_order': [0,1,2], 'stored_variance_consistency_max_abs_error': maximum_variance_error,
            'rows': rows, 'flattened_mean_path_l2': nm, 'flattened_variance_path_l2': nv,
            'flattened_variance_to_mean_norm_ratio': nv/nm if nm else None,
            'flattened_dot_product': dot, 'flattened_cosine': dot/(nm*nv) if nm and nv else None,
            'negative_dot_product_rows': sum(r['dot_product'] < 0 for r in rows)}


def diagnose(card, inputs):
    rows = train_ids(inputs.tensor('train_targets', 'row_ids', shape=(112,)))
    groups = inputs.tensor('train_targets', 'group_ids', shape=(112,))
    targets = inputs.tensor('train_targets', 'targets', shape=(112,4))
    target_valid = boolean(inputs.tensor('train_targets', 'valid', shape=(112,4)), 'TRAIN targets')
    require(all(math.isfinite(v) for v in targets), 'Nonfinite TRAIN target')
    stages, query_witnesses = {}, {}
    for stage in ('initial', 'final'):
        feature_report, native, valid = features(inputs, stage, rows, targets, target_valid)
        query_report, witness = query(inputs, stage, rows)
        _, metadata = checkpoint_metadata(inputs, stage)
        heads = head_scores(inputs, stage, rows, groups, targets, target_valid, native, valid)
        means = {task: {head: {metric: mean([rep['tasks'][task][head][metric] for rep in heads
                                           if rep['tasks'][task][head][metric] is not None])
                              for metric in ('r2', 'mae', 'rmse', 'coverage')}
                       for head in ('linear', 'neural')} for task in TASKS}
        stages[stage] = {'checkpoint_provenance': metadata, 'native_features': feature_report,
                         'saved_TRAIN_head_scores': heads,
                         'per_task_mean_across_saved_repetitions': means,
                         'fixed_query_TRAIN_diagnosis': query_report}
        query_witnesses[stage] = witness
    require(query_witnesses['initial']['support'] == query_witnesses['final']['support'] and
            query_witnesses['initial']['target'] == query_witnesses['final']['target'], 'Initial/final fixed TRAIN queries differ')
    try:
        gradients = analytic_change_gradients(inputs, query_witnesses['final'])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        gradients = {'status': 'not_computed_schema_or_consistency_refusal', 'reason': str(error),
                     'fallback_model_calls': 0,
                     'limitation': 'No gradient-direction conclusion is permitted without complete safe schema and variance consistency'}
    return {'kind': 'cpb_v3_saved_TRAIN_relation_variance_diagnosis_v1', 'status': 'passed',
            'instance': 'CPB-v3.market-01', 'scope': 'Saved TRAIN diagnostic arithmetic; not new quality evidence',
            'TRAIN_cutoff_ms': TRAIN_END, 'TRAIN_rows': rows, 'TRAIN_probe_population': 112,
            'tasks': list(TASKS), 'stages': stages, 'analytic_change4_gradient_diagnosis': gradients,
            'counters': {'model_calls': 0, 'encoder_calls': 0, 'CPU_encoder_calls': 0, 'old_encoder_calls': 0,
                         'decoder_calls': 0, 'GPU_calls': 0, 'model_instantiations': 0,
                         'training_updates': 0, 'new_head_fits': 0, 'new_scaler_fits': 0,
                         'head_fits': 0, 'validation_numeric_reads': 0, 'raw_cache_numeric_reads': 0,
                         'neural_head_forward_calls': 0, 'validation_numeric_fields_accessed': 0,
                         'test_numeric_fields_accessed': 0, 'new_quality_scores': 0},
            'interpretation_limits': ['Known TRAIN diagnostics do not establish validation improvement or causation',
                                      'Detailed query tensors cover8 TRAIN examples;112-row summary statistics do not expand cellwise coverage',
                                      'Saved linear readout block arithmetic is descriptive, without fitting or selecting new heads',
                                      'No candidate, hyperparameter, gate, checkpoint promotion or oracle retention floor is chosen by this script']}


def write_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n'); output.flush(); os.fsync(output.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(sys.platform.startswith('linux') and Path('/.dockerenv').is_file(), 'Run only inside the existing managed Linux container')
    output = args.output.resolve()
    require(args.output.is_absolute() and output != Path('/'), 'Absolute exclusive output leaf required')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Never overwrite or reuse an earlier diagnosis.
    started = time.monotonic()
    inputs = None
    try:
        card_pin = file_pin(args.card)
        with args.card.open(encoding='utf-8-sig') as source:
            card = json.load(source)
        inputs = Inputs(card)  # ALL allowed/preservation pins verified before numeric access.
        report = diagnose(card, inputs)
        # Recheck all source data/metadata bytes after analysis, without reading
        # numeric fields from any preservation-only file.
        for pin in card['allowed_inputs'] + card.get('preservation_pins', []):
            inputs.verify(pin)
        require(file_pin(args.card) == card_pin, 'Card changed during diagnosis')
        report.update({'card': card_pin, 'diagnostic_source': file_pin(__file__),
                       'inert_archive_reader_source': file_pin(Path(__file__).with_name('archive_reader.py')),
                       'verified_allowed_inputs': inputs.verified, 'verified_preservation_pins': inputs.preservation,
                       'opened_allowed_ids': inputs.opened, 'numeric_fields_accessed': inputs.numeric_fields,
                       'elapsed_seconds': time.monotonic()-started})
        write_exclusive(output/'diagnosis.json', report)
        print(json.dumps({'status': report['status'], 'output': str(output/'diagnosis.json'),
                          'sha256': sha256(output/'diagnosis.json'),
                          'analytic_gradient_status': report['analytic_change4_gradient_diagnosis']['status']}))
    except Exception as error:
        write_exclusive(output/'failure.json', {'kind': 'cpb_v3_TRAIN_diagnosis_refusal_v1',
                                               'status': 'failed', 'reason': str(error),
                                               'encoder_calls': 0, 'head_fits': 0})
        raise
    finally:
        if inputs is not None:
            inputs.close()


if __name__ == '__main__':
    main()

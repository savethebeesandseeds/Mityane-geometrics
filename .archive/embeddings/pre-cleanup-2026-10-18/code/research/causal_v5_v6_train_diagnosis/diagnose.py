#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Card-bound scalar diagnosis of saved CPB-v5/CPB-v6 TRAIN data; no model execution.

The sibling archive_reader supplies inert C++ OutputArchive descriptors only.
This program imports no Torch/NumPy and performs no fitting or neural forward.
"""
from __future__ import annotations

import argparse
import csv
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
COMMON_BLOCKS = (('learned20', 0, 20), ('fixed_spectral12', 20, 32), ('mean4', 32, 36), ('uncertainty4', 36, 40))
WIDTHS = {'v5': 40, 'v6': 40}
BLOCKS = {'v5': COMMON_BLOCKS, 'v6': COMMON_BLOCKS}
POLICIES = {'v5': 'separate-mean-and-uncertainty-bank-projections-live-gaussian-nll-v1',
            'v6': 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'}
OBJECTIVE_POLICY = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
DIRECT_ROUTES = 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
HEAD_SEEDS_U64 = {'v5': (6049515622977958931, 8262941649267994858, 2095371932330426401),
                  'v6': (6049515622977958931, 8262941649267994858, 2095371932330426401)}
V6_LAYOUT = ('parent32[0:32];mean4[32:36];uncertainty4[36:40];'
             'independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;'
             'variance-reads-live-uncertainty4-only')
BASE_ALLOWED_IDS = {'initial_features', 'final_features', 'train_targets',
               'initial_query', 'final_query', 'initial_query_summary',
               'final_query_summary', 'initial_checkpoint', 'final_checkpoint'} | {
    f'{stage}_{kind}_{rep}' for stage in ('initial', 'final')
    for kind in ('head', 'fit') for rep in REPS}
VERSIONS = ('v5', 'v6')
COMMON_IDS = {'train_targets'} | {version + '_' + identity for version in VERSIONS for identity in BASE_ALLOWED_IDS - {'train_targets'}}
EXTRA_IDS = {'v6_actual_update_1', 'v6_actual_update_2048', 'v6_actual_bindings', 'v6_training_csv'}
ALLOWED_IDS = COMMON_IDS | EXTRA_IDS
QUANTILES = (0., .01, .05, .25, .5, .75, .95, .99, 1.)
FIELD_GROUPS = {
    'targets': {'row_ids','group_ids','targets','valid'},
    'features': {'row_ids','values','valid'},
    'query': {'row_ids','timestamps','target_mask','visible','observed','hidden','targets',
              'mean','variance','native','row_mae','row_nll'},
    'head': {'row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions'},
    'fit': {'feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed'},
    'checkpoint': {'artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates',
                   'causal_updates','initialization_seed','channel_ids','payload.model.encoder.change_decoder',
                   'payload.model.variance_projection','payload.model.variance_bias',
                   'payload.link_constants_f32','payload.softplus_parameters_f64'},
    'actual': {'update','attempt','row_ids','timestamps','observed','hidden','context_deleted','visible',
               'target_mask','eligible_channels','targets','mean','variance','native','row_mean_huber',
               'row_variance_nll','row_total_objective','mean_Huber_F64','variance_NLL_F64',
               'total_objective_F64','Gaussian_NLL_F64','pre_clip_global_norm','clip_factor'},
}
COMPONENT_FIELDS = {'row_mean_huber','row_variance_nll','row_total_objective'}
OBJECTIVE_FIELDS = {'objective_policy','mean_huber_delta_f64','loss_coefficients_f64',
                    'variance_error_detached','direct_gradient_routes','reporting_nll_backpropagated'}
CSV_FIELDS = ('update','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64',
              'pre_clip_global_norm','clip_factor')


def expected_fields(identity):
    if identity in {'v6_actual_update_1','v6_actual_update_2048'}:
        return FIELD_GROUPS['actual']
    if identity == 'train_targets':
        return FIELD_GROUPS['targets']
    if identity.endswith('_query_summary') or identity in {'v6_actual_bindings','v6_training_csv'}:
        return set()
    for kind in ('features','query','checkpoint','head','fit'):
        if identity.endswith('_'+kind) or ('_'+kind+'_') in identity:
            fields = set(FIELD_GROUPS[kind])
            if kind == 'checkpoint':
                fields.update({'gradient_policy', 'payload.gradient_policy'})
                fields.update({'native_layout', 'mean_width', 'uncertainty_width', 'native_width',
                               'payload.native_layout', 'payload.mean_width',
                               'payload.uncertainty_width', 'payload.native_width'})
                if identity.startswith('v6_'):
                    fields.update(OBJECTIVE_FIELDS | {'payload.'+field for field in OBJECTIVE_FIELDS})
            if kind == 'query' and identity.startswith('v6_'):
                fields.update(COMPONENT_FIELDS)
            return fields
    raise ValueError('Unknown declared input kind')


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
        require(card.get('kind') == 'cpb_v5_v6_saved_train_diagnosis_card_v1' and
                card.get('stage') == 'prospective_before_diagnostic_arithmetic' and
                card.get('baseline') == 'CPB-v3.market-01' and
                card.get('comparison_references') == ['CPB-v5.market-01', 'CPB-v6.market-01'] and
                card.get('versions') == list(VERSIONS) and
                card.get('train_rows') == 112 and card.get('query_rows') == 8 and
                card.get('train_end_ms') == TRAIN_END, 'Card TRAIN scope differs')
        require(card.get('gradient_objectives') == ['Gaussian_NLL','MAE','Huber_delta1']
                and card.get('Huber_delta') == 1., 'Prospective local derivative objectives differ')
        require(card.get('native_widths') == WIDTHS, 'Card native widths differ')
        require({version: [(b['name'], b['begin'], b['end']) for b in card['native_blocks'][version]]
                 for version in VERSIONS} ==
                {version: list(BLOCKS[version]) for version in VERSIONS}, 'Card native blocks differ')
        require(card.get('cross_basis_cosine_policy') == 'forbidden-between-independent-mean4-and-uncertainty4',
                'Independent latent bases must not be compared with a cosine')
        pins = card['allowed_inputs']
        require(isinstance(pins, list) and len(pins) == len(ALLOWED_IDS), 'Exactly45 declared input IDs required')
        self.pins = {}
        self.opened = {}
        self.archives = {}
        self.numeric_fields = []
        for pin in pins:
            identity = pin['id']
            require(identity in ALLOWED_IDS and identity not in self.pins, f'Unknown/duplicate input ID: {identity}')
            path = Path(pin['container_path'])
            require(path.is_absolute() and 'validation' not in str(path).lower(),
                    f'Validation numeric archive or relative input refused: {identity}')
            role = ('aggregate_JSON_TRAIN_key_only' if identity.endswith('_query_summary') else
                    'actual_batch_binding_metadata' if identity == 'v6_actual_bindings' else
                    'TRAIN_CSV_2048_scalar_rows_only' if identity == 'v6_training_csv' else 'TRAIN_archive')
            require(pin.get('role') == role,
                    f'Undeclared numeric input role: {identity}')
            if pin['role'] == 'TRAIN_archive':
                require(isinstance(pin.get('allowed_tensor_fields'), list)
                        and set(pin['allowed_tensor_fields']) == expected_fields(identity),
                        'Exact independent prospective tensor-field allowlist required')
            if identity=='v6_training_csv':
                require(pin.get('numeric_update_scope')=='all2048successfulupdates' and
                        set(pin.get('allowed_CSV_fields',[]))==set(CSV_FIELDS) and
                        pin.get('fixed_update_blocks')==[[1,512],[513,1024],[1025,1536],[1537,2048]],
                        'CSV numeric row/field scope differs')
            self.pins[identity] = dict(pin)
        require(set(self.pins) == ALLOWED_IDS, 'Missing card input IDs')
        self.verified = []
        for pin in pins:
            self.verified.append(self.verify(pin))
        require(len(card.get('preservation_pins', [])) == 31 and
                len({pin['id'] for pin in card['preservation_pins']}) == 31,
                'Exactly31 distinct declared preservation-only pins required')
        require(card.get('actual_batches') == [1,2048] and
                card.get('actual_batch_decoder_adjoints') == 'forbidden-no-matched-prestep-decoder-weights' and
                card.get('v5_full112_component_rows') == 'absent-never-reconstructed' and
                card.get('v5_actual_batch_tensors') == 'absent-never-replayed', 'Evidence asymmetry declaration differs')
        self.preservation = [self.verify(pin) for pin in card['preservation_pins']]
        self.protocol_preservation = self.verify(card['pre_execution_preservation'])

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
        require(identity in self.pins and self.pins[identity]['role'] == 'TRAIN_archive', 'Archive ID outside card')
        if identity not in self.archives:
            path = self.pins[identity]['container_path']
            self.archives[identity] = Archive(path)
            self.opened[identity] = 'inert_archive_metadata'
        return self.archives[identity]

    def tensor(self, identity, *fields, shape=None):
        require(all('validation' not in name.lower() for name in fields), 'Validation numeric field refused')
        require('.'.join(fields) in self.pins[identity]['allowed_tensor_fields'],
                f'Tensor field outside prospective allowlist: {identity}:{fields}')
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
        require(identity in {version + '_' + stage + '_query_summary' for version in VERSIONS
                             for stage in ('initial', 'final')}, 'Summary ID outside card')
        require(self.pins[identity]['role'] == 'aggregate_JSON_TRAIN_key_only', 'TRAIN-only JSON role required')
        require(self.pins[identity]['numeric_json_selector'] == 'train', 'Only TRAIN JSON selector admitted')
        with Path(self.pins[identity]['container_path']).open(encoding='utf-8-sig') as source:
            container = json.load(source)
        require(isinstance(container, dict) and isinstance(container.get('train'), dict), 'Expected aggregate query JSON with TRAIN key')
        # The full file is pinned/decoded, but no validation numeric field is accessed.
        self.opened[identity] = 'aggregate_JSON_train_key_only'
        self.numeric_fields.append({'id': identity, 'field': 'train_only'})
        return container['train']

    def actual_bindings(self):
        identity = 'v6_actual_bindings'
        with Path(self.pins[identity]['container_path']).open(encoding='utf-8-sig') as source:
            bindings = json.load(source)
        require(isinstance(bindings,list) and len(bindings)==2, 'Exactly two actual-batch bindings required')
        for binding,update in zip(bindings,(1,2048)):
            pin=self.pins[f'v6_actual_update_{update}']
            require(binding.get('update')==update and binding.get('pre_step_forward') is True and
                    binding.get('extra_encoder_calls')==0 and binding.get('path')==pin['container_path'] and
                    binding.get('bytes')==pin['bytes'] and binding.get('sha256')==pin['sha256'],
                    'Actual pre-step binding differs')
        self.opened[identity]='exact-two-prestep-bindings-metadata'
        return bindings

    def actual_csv_rows(self):
        identity='v6_training_csv'
        rows=[]
        with Path(self.pins[identity]['container_path']).open(encoding='utf-8-sig',newline='') as source:
            reader=csv.DictReader(source)
            require(set(CSV_FIELDS).issubset(reader.fieldnames or []), 'CSV component columns unavailable')
            for index,row in enumerate(reader):
                require(index<2048,'CSV has more than2048 successful rows')
                # No attempt/anchor/seed/count/gradient-tensor column is read.
                item={key:float(row[key]) for key in CSV_FIELDS if key!='update'}
                item['update']=int(row['update'])
                require(item['update']==index+1 and all(math.isfinite(value) for value in item.values()),
                        'CSV successful update sequence or scalar finiteness differs')
                require(item['mean_Huber_F64']>=0 and item['pre_clip_global_norm']>=0 and
                        0<item['clip_factor']<=1,'CSV Huber/norm/clip range differs')
                near(item['mean_Huber_F64']+item['variance_NLL_F64'],item['total_objective_F64'],
                     'CSV unit component sum')
                near(item['variance_NLL_F64'],item['Gaussian_NLL_F64'],'CSV calibration/reporting NLL numeric identity')
                near(min(1.,1./(item['pre_clip_global_norm']+1e-6)),item['clip_factor'],
                     'CSV observed clip-factor arithmetic')
                rows.append(item)
            require(len(rows)==2048,'Actual trajectory must contain2048 CSV rows')
        groups={}
        for label,lo,hi in (('all2048',0,2048),('updates1_512',0,512),('updates513_1024',512,1024),
                            ('updates1025_1536',1024,1536),('updates1537_2048',1536,2048)):
            chosen=rows[lo:hi]
            clipped=sum(item['clip_factor']<1. for item in chosen)
            groups[label]={'first_update':lo+1,'last_update':hi,'updates':hi-lo,
                           'recorded_clip_factor_below1_updates':clipped,
                           'observed_clipping_frequency':clipped/(hi-lo),'scalars':{}}
            for field in CSV_FIELDS[1:]:
                values=[item[field] for item in chosen]
                ordered=sorted(values);n=len(ordered)
                median=(ordered[(n-1)//2]+ordered[n//2])/2.
                groups[label]['scalars'][field]={'minimum':min(values),'mean':mean(values),
                                                'median':median,'maximum':max(values)}
        self.opened[identity]='CSV-only-seven-declared-numeric-columns-all2048-successful-updates'
        self.numeric_fields.append({'id':identity,'field':','.join(CSV_FIELDS),'successful_updates':2048})
        return rows,{'status':'computed_from_saved_TRAIN_CSV_scalars','numeric_columns':list(CSV_FIELDS),
                     'successful_updates':2048,'fixed_overall_and512_blocks':groups,
                     'all_recorded_factor_formulas_checked':True,
                     'limits':'Observed norm/factor and loss scalars only; no full gradient-norm reconstruction, objective-specific gradient decomposition, loss-ratio-to-gradient-dominance inference or causal explanation. Shared clip1/AdamW coupling remains.'}

    def close(self):
        for archive in self.archives.values():
            archive.close()


class VersionInputs:
    """Bind copied arithmetic helpers to one version's explicit allowed IDs."""
    def __init__(self, inputs, version):
        require(version in VERSIONS, 'Unknown version view')
        self.inputs, self.version = inputs, version

    def identity(self, identity):
        return identity if identity == 'train_targets' else self.version + '_' + identity

    def tensor(self, identity, *fields, shape=None):
        return self.inputs.tensor(self.identity(identity), *fields, shape=shape)

    def archive(self, identity):
        return self.inputs.archive(self.identity(identity))

    def summary(self, identity):
        return self.inputs.summary(self.identity(identity))


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
    width, block_specs = WIDTHS[inputs.version], BLOCKS[inputs.version]
    train_ids(inputs.tensor(identity, 'row_ids', shape=(112,)), rows)
    values = inputs.tensor(identity, 'values', shape=(112, width))
    valid = boolean(inputs.tensor(identity, 'valid', shape=(112,)), identity)
    require(all(valid), 'Complete112 TRAIN feature coverage required')
    require(all(math.isfinite(v) for v in values), 'Nonfinite native feature')
    columns = [[values[r*width+d] for r in range(112)] for d in range(width)]
    coordinate_rows = []
    for d, column in enumerate(columns):
        x = [v for v, ok in zip(column, valid) if ok]
        correlations = {}
        for t, task in enumerate(TASKS):
            indices = [r for r in range(112) if valid[r] and target_valid[r*4+t]]
            correlations[task] = pearson([column[r] for r in indices], [targets[r*4+t] for r in indices])
        coordinate_rows.append({'coordinate': d, 'block': next(name for name, lo, hi in block_specs if lo <= d < hi),
                                'mean': mean(x), 'population_std': math.sqrt(variance(x)) if x else None,
                                'minimum': min(x) if x else None, 'maximum': max(x) if x else None,
                                'pearson_to_TRAIN_targets': correlations})
    blocks = {}
    for name, lo, hi in block_specs + ((f'complete{width}', 0, width),):
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
    return {'rows': 112, 'native_width': width, 'valid_rows': sum(valid),
            'coordinates': coordinate_rows, 'blocks': blocks}, values, valid


def head_scores(inputs, stage, rows, groups, targets, target_valid, native, feature_valid):
    width, block_specs = WIDTHS[inputs.version], BLOCKS[inputs.version]
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
        mu = inputs.tensor(fit_id, 'feature_mean', shape=(width,))
        scale = inputs.tensor(fit_id, 'feature_scale', shape=(width,))
        weights = inputs.tensor(fit_id, 'ridge_weights', shape=(width, 4))
        target_scale = inputs.tensor(fit_id, 'target_scale', shape=(4,))
        target_mean = inputs.tensor(fit_id, 'target_mean', shape=(4,))
        intercept = inputs.tensor(fit_id, 'ridge_intercept', shape=(4,))
        require(all(math.isfinite(v) and v > 0 for v in scale+target_scale), 'Invalid saved readout scale')
        require(all(math.isfinite(v) for v in mu+weights), 'Invalid saved readout coefficients')
        seed = inputs.tensor(fit_id, 'actual_seed')
        require(len(seed) == 1 and isinstance(seed[0], int) and
                (seed[0] & ((1 << 64)-1)) == HEAD_SEEDS_U64[inputs.version][REPS.index(rep)],
                'Saved head seed differs from the original frozen dimension-dependent recipe')
        reconstructed = [target_mean[t] + target_scale[t] * (intercept[t] + math.fsum(
            (native[r*width+d]-mu[d])/scale[d]*weights[d*4+t] for d in range(width)))
            for r in range(112) for t in range(4)]
        for i, ok in enumerate(valid):
            if ok:
                near(reconstructed[i], ridge[i], 'Saved ridge coefficient/prediction reconstruction')
        chronological = {}
        for label, lo, hi in (('early56', 0, 56), ('late56', 56, 112)):
            chronological[label] = {task: {head: score(pred[lo*4+t:hi*4:4], targets[lo*4+t:hi*4:4],
                                                       valid[lo*4+t:hi*4:4])
                                              for head, pred in (('linear', ridge), ('neural', neural))}
                                    for t, task in enumerate(TASKS)}
        contributions = {}
        for task_index, task in enumerate(TASKS):
            per_block = {}
            components = {}
            selected = [r for r in range(112) if valid[r*4+task_index]]
            residuals = [ridge[r*4+task_index]-targets[r*4+task_index] for r in selected]
            for name, lo, hi in block_specs:
                # Arithmetic decomposition of an ALREADY FITTED linear readout;
                # no predictor object, optimization, new fit or neural forward.
                component = [target_scale[task_index] * math.fsum(
                    (native[r*width+d]-mu[d])/scale[d]*weights[d*4+task_index]
                    for d in range(lo, hi)) for r in range(112)]
                vals = [component[r] for r in selected]
                components[name] = component
                per_block[name] = {'saved_standardized_weight_l2': math.sqrt(math.fsum(weights[d*4+task_index]**2 for d in range(lo, hi))),
                                   'TRAIN_contribution_population_std_in_target_units': math.sqrt(variance(vals)) if vals else None,
                                   'TRAIN_contribution_pearson_to_target': pearson(vals, [targets[r*4+task_index] for r in selected]),
                                   'TRAIN_contribution_pearson_to_saved_residual': pearson(vals, residuals),
                                   'row_contributions_in_target_units': component}
            covariance_pairs = []
            names = [b[0] for b in block_specs]
            for a, name in enumerate(names):
                for other in names[a+1:]:
                    x = [components[name][r] for r in selected]
                    y = [components[other][r] for r in selected]
                    covariance_pairs.append({'blocks': [name, other], 'population_covariance':
                        mean([(v-mean(x))*(w-mean(y)) for v, w in zip(x, y)]), 'pearson': pearson(x, y)})
            total = [math.fsum(components[name][r] for name in names) for r in selected]
            marginal = math.fsum(variance([components[name][r] for r in selected]) for name in names)
            contributions[task] = {'blocks': per_block, 'pairwise_component_covariance': covariance_pairs,
                'sum_contribution_population_variance': variance(total),
                'sum_of_block_population_variances': marginal,
                'sum_to_marginal_variance_ratio': variance(total)/marginal if marginal else None,
                'constant_target_units': target_mean[task_index]+target_scale[task_index]*intercept[task_index],
                'chronological_TRAIN_halves': {
                    label: {name: {
                        'valid_rows': len(chosen),
                        'contribution_population_std_in_target_units': math.sqrt(variance([components[name][r] for r in chosen])) if chosen else None,
                        'contribution_pearson_to_target': pearson([components[name][r] for r in chosen], [targets[r*4+task_index] for r in chosen]),
                        'contribution_pearson_to_saved_residual': pearson([components[name][r] for r in chosen], [ridge[r*4+task_index]-targets[r*4+task_index] for r in chosen])}
                            for name in names}
                    for label, lo, hi in (('early56',0,56),('late56',56,112))
                    for chosen in [[r for r in range(lo,hi) if valid[r*4+task_index]]]},
                'interpretation': 'Exact decomposition of saved fitted ridge only; no block ablation predictor or refit; component correlations are scalar readout contributions, not latent-gradient conflict'}
        reports.append({'repetition': rep, 'saved_seed_signed_int64': seed[0], 'tasks': task_scores,
                        'saved_seed_unsigned_decimal': str(seed[0] & ((1 << 64)-1)),
                        'chronological_TRAIN_halves': chronological,
                        'ridge_prediction_reconstruction_checked': True,
                        'saved_linear_readout_block_decomposition': contributions})
    return reports


def reduced_components(witness):
    """Declared F64 scalar hierarchy from saved supported distributions only."""
    support=witness['support']
    at=lambda b,c,h,f:((b*3+c)*128+h)*9+f
    result={key:[] for key in ('row_mae','row_nll','row_mean_huber','row_variance_nll','row_total_objective','row_target_cells')}
    for b in range(8):
        channels={key:[] for key in ('mae','nll','huber')}
        cells=0
        for c in range(3):
            chosen=[at(b,c,h,f) for h in range(128) for f in range(9) if support[at(b,c,h,f)]]
            cells+=len(chosen)
            if not chosen:
                continue
            errors=[float(witness['mean'][i])-float(witness['target'][i]) for i in chosen]
            variances=[float(witness['variance'][i]) for i in chosen]
            channels['mae'].append(mean([abs(e) for e in errors]))
            channels['nll'].append(mean([.5*(e*e/v+math.log(v)) for e,v in zip(errors,variances)]))
            channels['huber'].append(mean([.5*e*e if abs(e)<1. else abs(e)-.5 for e in errors]))
        require(channels['mae'], 'Saved distribution row lacks eligible support')
        mae,nll,huber=(mean(channels[key]) for key in ('mae','nll','huber'))
        for key,value in (('row_mae',mae),('row_nll',nll),('row_mean_huber',huber),
                          ('row_variance_nll',nll),('row_total_objective',huber+nll),('row_target_cells',cells)):
            result[key].append(value)
    return result


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
    hidden = boolean(inputs.tensor(identity, 'hidden', shape=shape), identity)
    require(all(s == (o and h) and v == (o and not h) for s,o,v,h in zip(support,observed,visible,hidden)),
            'Exact query target/visible/original support differs')
    target = inputs.tensor(identity, 'targets', shape=shape)
    predicted = inputs.tensor(identity, 'mean', shape=shape)
    var = inputs.tensor(identity, 'variance', shape=shape)
    native = inputs.tensor(identity, 'native', shape=(8, WIDTHS[inputs.version]))
    require(all(math.isfinite(v) for v in target+predicted+var+native), 'Nonfinite TRAIN query values')
    require(all(v > 0 for v, s in zip(var, support) if s), 'Nonpositive supported query variance')
    index = lambda b, c, h, f: ((b*3+c)*128+h)*9+f
    cell_stats, row_mae, row_nll, row_counts = [], [], [], []
    for c in range(3):
        for f in range(9):
            chosen = [index(b,c,h,f) for b in range(8) for h in range(128) if support[index(b,c,h,f)]]
            error = [predicted[i]-target[i] for i in chosen]
            variances = [var[i] for i in chosen]
            inverse_variance = [1./v for v in variances]
            squared_standardized = [e*e/v for e, v in zip(error, variances)]
            log_term = [.5*math.log(v) for v in variances]
            residual_term = [.5*v for v in squared_standardized]
            standardized = [e/math.sqrt(v) for e, v in zip(error, variances)]
            cell_stats.append({'channel_id': c, 'feature': f, 'target_cells': len(chosen),
                               'mean_absolute_error': mean([abs(e) for e in error]),
                               'mean_squared_error': mean([e*e for e in error]), 'mean_variance': mean(variances),
                               'mean_inverse_variance': mean(inverse_variance),
                               'inverse_variance_quantiles': quantiles(inverse_variance),
                               'abs_error_inverse_variance_pearson': pearson([abs(e) for e in error], inverse_variance),
                               'mean_abs_NLL_mean_derivative': mean([abs(e)/v for e,v in zip(error, variances)]),
                               'mean_abs_MAE_mean_derivative': mean([float(e != 0.) for e in error]),
                               'mean_abs_Huber_delta1_mean_derivative': mean([min(abs(e),1.) for e in error]),
                               'fraction_abs_error_at_least1': mean([float(abs(e) >= 1.) for e in error]),
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
    require(all(math.isfinite(x) and x >= 0 for x in summary['row_mae']) and
            all(math.isfinite(x) for x in summary['row_nll']) and
            all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                for x in summary['row_target_cells']), 'Invalid all112 TRAIN summary values')
    report = {'detailed_population': 8, 'detailed_row_ids': rows, 'full_TRAIN_probe_population': 112,
              'scope_limit': 'Cellwise statistics describe only first8 saved TRAIN queries, not112 or VALIDATION',
              'per_channel_feature': cell_stats,
              'B8_hierarchical_mean_MAE': mean(row_mae), 'B8_hierarchical_mean_NLL': mean(row_nll),
              'B8_target_cells': sum(row_counts),
              'all112_TRAIN': {'standardized_fixed_query_MAE': summary['standardized_fixed_query_MAE'],
                               'Gaussian_NLL_F64': summary['Gaussian_NLL_F64'], 'target_cells': summary['target_cells'],
                               'row_mae_quantiles': quantiles(summary['row_mae']),
                               'row_nll_quantiles': quantiles(summary['row_nll']),
                               'row_target_cells': summary['row_target_cells'],
                               'row_mae': summary['row_mae'], 'row_nll': summary['row_nll'],
                               'chronological_TRAIN_halves': {
                                   label: {'rows': hi-lo,
                                           'standardized_fixed_query_MAE': mean(summary['row_mae'][lo:hi]),
                                           'Gaussian_NLL_F64': mean(summary['row_nll'][lo:hi]),
                                           'target_cells': sum(summary['row_target_cells'][lo:hi])}
                                   for label,lo,hi in (('early56',0,56),('late56',56,112))},
                               'row_association': 'Ordered saved TRAIN query plan; first8 row IDs/masks checked directly, complete plan/trajectory association already independently audited'}}
    witness={'row_ids': rows, 'native': native, 'support': support, 'mean': predicted,
             'target': target, 'variance': var}
    components=reduced_components(witness)
    report['B8_fixed_Huber_delta1']={'mean':mean(components['row_mean_huber']),
        'rows':components['row_mean_huber'],
        'status':'applied-mean-objective' if inputs.version=='v6' else 'hypothetical-unapplied-diagnostic'}
    if inputs.version=='v6':
        for field,scalar in (('row_mean_huber','mean_Huber_F64'),('row_variance_nll','variance_NLL_F64'),
                             ('row_total_objective','total_objective_F64')):
            saved=inputs.tensor(identity,field,shape=(8,))
            require(len(summary[field])==112, 'V6 full112 component rows unavailable')
            require(all(math.isfinite(value) for value in summary[field]), 'Nonfinite V6 component summary')
            for b in range(8):
                near(components[field][b],saved[b],'Saved first8 '+field)
                near(components[field][b],summary[field][b],'Aggregate first8 '+field)
            near(mean(summary[field]),summary[scalar],'Full112 '+scalar)
            report['all112_TRAIN'][scalar]=summary[scalar]
            report['all112_TRAIN'][field]=summary[field]
            for label,lo,hi in (('early56',0,56),('late56',56,112)):
                report['all112_TRAIN']['chronological_TRAIN_halves'][label][scalar]=mean(summary[field][lo:hi])
        for i in range(112):
            near(summary['row_variance_nll'][i],summary['row_nll'][i],'V6 detached variance/reporting NLL numeric identity')
            near(summary['row_mean_huber'][i]+summary['row_variance_nll'][i],summary['row_total_objective'][i],
                 'V6 unit component sum')
        report['all112_component_evidence']='saved-V6-only;no-symmetric-V5-component-comparison'
    else:
        require(not any(field in summary for field in COMPONENT_FIELDS), 'V5 summary unexpectedly supplies component rows')
        report['all112_component_evidence']='absent-V5;never-reconstructed-from-MAE/NLL'
    return report,witness


def checkpoint_metadata(inputs, stage):
    identity = stage + '_checkpoint'
    archive = inputs.archive(identity)
    text = lambda field: bytes(inputs.tensor(identity, field)).decode('utf-8')
    version = inputs.version
    require(text('artifact_kind') == f'cpb_{version}_inference_checkpoint_v1'
            and text('model_tag') == 'CPB-' + version, 'Only declared typed fresh checkpoints admitted')
    require(text('initialization_kind') == 'fresh-seeded-constructor-no-import-v1' and text('pretrained_ancestor') == 'none', 'Fresh checkpoint provenance differs')
    source = inputs.tensor(identity, 'source_updates')
    updates = inputs.tensor(identity, 'causal_updates')
    seed = inputs.tensor(identity, 'initialization_seed')
    require(source == [0] and updates == [0 if stage == 'initial' else 2048] and seed == [943001], 'Checkpoint counters/seed differ')
    require(inputs.tensor(identity, 'channel_ids', shape=(3,)) == [0,1,2], 'Canonical channel order required')
    require(text('gradient_policy') == POLICIES[version], 'Typed native gradient policy differs')
    require(bytes(inputs.tensor(identity,'payload','gradient_policy')).decode('utf-8') == POLICIES[version],
            'Nested decoder gradient policy differs')
    require(text('native_layout') == V6_LAYOUT and
                inputs.tensor(identity, 'mean_width') == [4] and
                inputs.tensor(identity, 'uncertainty_width') == [4] and
                inputs.tensor(identity, 'native_width') == [40], 'Independent native40 layout differs')
    require(bytes(inputs.tensor(identity,'payload','native_layout')).decode('utf-8') == V6_LAYOUT and
                inputs.tensor(identity,'payload','mean_width') == [4] and
                inputs.tensor(identity,'payload','uncertainty_width') == [4] and
                inputs.tensor(identity,'payload','native_width') == [40], 'Nested independent native40 layout differs')
    if version=='v6':
        for prefix in ((),('payload',)):
            require(bytes(inputs.tensor(identity,*prefix,'objective_policy')).decode('utf-8')==OBJECTIVE_POLICY and
                    inputs.tensor(identity,*prefix,'mean_huber_delta_f64')==[1.] and
                    inputs.tensor(identity,*prefix,'loss_coefficients_f64',shape=(2,))==[1.,1.] and
                    inputs.tensor(identity,*prefix,'variance_error_detached')==[1] and
                    inputs.tensor(identity,*prefix,'reporting_nll_backpropagated')==[0] and
                    bytes(inputs.tensor(identity,*prefix,'direct_gradient_routes')).decode('utf-8')==DIRECT_ROUTES,
                    'V6 typed composite objective/detach/reporting contract differs')
    return archive, {'source_updates': source[0], 'causal_updates': updates[0], 'initialization_seed': seed[0],
                     'pretrained_ancestor': 'none', 'gradient_policy': POLICIES[version],
                     'native_width': WIDTHS[version],
                     'objective':'joint-Gaussian-NLL' if version=='v5' else OBJECTIVE_POLICY,
                     'variance_reads': 'live-independent-uncertainty4',
                     'applied_variance_gradient_into_mean4': 'exactly-zero'}


def vector_comparison(a, b):
    require(len(a) == len(b), 'Same-basis vector dimensions differ')
    na = math.sqrt(math.fsum(x*x for x in a))
    nb = math.sqrt(math.fsum(x*x for x in b))
    dot = math.fsum(x*y for x,y in zip(a,b))
    return {'left_l2': na, 'right_l2': nb, 'dot_product': dot,
            'cosine': dot/(na*nb) if na and nb else None}


def analytic_change_gradients(inputs, witness, stage):
    """Local scalar adjoints of saved outputs/coefficients, never a model replay.

    NLL, MAE and Huber1 share the same mean4 basis, support and reductions.
    Both variance routes act in independent uncertainty4 bases. V5 applies the
    mean-NLL adjoint; V6 applies Huber1 to mean, never reporting-NLL backward.
    No dot/cosine between
    those bases, or claim of conflict from their norms, is mathematically valid.
    Parent32 and upstream parameter Jacobians are deliberately unavailable.
    """
    identity = stage + '_checkpoint'
    decoder = inputs.tensor(identity, 'payload', 'model', 'encoder', 'change_decoder', shape=(3456, 4))
    projection = inputs.tensor(identity, 'payload', 'model', 'variance_projection', shape=(27, 4))
    bias = inputs.tensor(identity, 'payload', 'model', 'variance_bias', shape=(27,))
    constants = inputs.tensor(identity, 'payload', 'link_constants_f32', shape=(3,))
    require(inputs.tensor(identity, 'payload', 'softplus_parameters_f64', shape=(2,)) == [1.,20.],
            'Saved softplus beta/threshold differ from admitted scalar derivative')
    eps, shift, coefficient = constants
    require(all(math.isfinite(v) for v in decoder+projection+bias+constants), 'Nonfinite saved decoder tensor')
    require(0 < eps < 1 and coefficient > 0, 'Saved link constants invalid')
    softplus = lambda x: x if x > 20 else math.log1p(math.exp(x))
    sigmoid = lambda x: 1. if x > 20 else (1./(1.+math.exp(-x)) if x >= 0 else math.exp(x)/(1.+math.exp(x)))
    u0 = softplus(shift)
    names = ('mean_NLL', 'mean_MAE', 'mean_Huber_delta1')
    flattened = {name: [] for name in names}
    variance_flattened = []
    rows, maximum_variance_error = [], 0.
    support = witness['support']
    at = lambda b,c,h,f: ((b*3+c)*128+h)*9+f
    by_cf = [{'channel_id': c, 'feature': f, 'target_cells': 0, 'inverse_variance': [],
              'absolute_error': [], 'sum_abs_NLL_mean_derivative_hierarchical': 0.,
              'sum_abs_MAE_mean_derivative_hierarchical': 0.,
              'sum_abs_Huber_delta1_mean_derivative_hierarchical': 0.,
              'mean4_gradient_vectors': {name: [] for name in names},
              'variance_route_gradient_vectors': []}
             for c in range(3) for f in range(9)]
    projection_gradient, bias_gradient = [0.]*108, [0.]*27
    for b in range(8):
        counts = [sum(support[at(b,c,h,f)] for h in range(128) for f in range(9)) for c in range(3)]
        eligible_channels = sum(n > 0 for n in counts)
        require(eligible_channels > 0, 'Gradient query row lacks eligible channels')
        width = WIDTHS[inputs.version]
        variance_offset = 36
        variance_native = witness['native'][b*width+variance_offset:b*width+variance_offset+4]
        gradients = {name: [0.]*4 for name in names}
        variance_gradient = [0.]*4
        for c in range(3):
            if not counts[c]:
                continue
            factor = 1./(8*eligible_channels*counts[c])
            for f in range(9):
                cf = c*9+f
                local = {name: [0.]*4 for name in names}
                linear = math.fsum(variance_native[d]*projection[cf*4+d] for d in range(4))+bias[cf]
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
                    # Tensor.values promotes each stored F32 operand to a Python
                    # F64 scalar before this subtraction, as the declared NLL
                    # does. Unsupported cells never enter scalar loss algebra.
                    promoted_mean, promoted_target = float(witness['mean'][i]), float(witness['target'][i])
                    error = promoted_mean-promoted_target
                    derivatives = {'mean_NLL': factor*error/stored_variance,
                                   'mean_MAE': factor*((error > 0.)-(error < 0.)),
                                   'mean_Huber_delta1': factor*max(-1., min(1., error))}
                    variance_derivative += factor*.5*(1./stored_variance-error*error/(stored_variance*stored_variance))
                    for name, derivative in derivatives.items():
                        for d in range(4):
                            local[name][d] += derivative*decoder[((c*128+h)*9+f)*4+d]
                    cell = by_cf[cf]
                    cell['target_cells'] += 1
                    cell['inverse_variance'].append(1./stored_variance)
                    cell['absolute_error'].append(abs(error))
                    cell['sum_abs_NLL_mean_derivative_hierarchical'] += abs(derivatives['mean_NLL'])
                    cell['sum_abs_MAE_mean_derivative_hierarchical'] += abs(derivatives['mean_MAE'])
                    cell['sum_abs_Huber_delta1_mean_derivative_hierarchical'] += abs(derivatives['mean_Huber_delta1'])
                raw_variance = variance_derivative*link_derivative
                bias_gradient[cf] += raw_variance
                local_variance = [raw_variance*projection[cf*4+d] for d in range(4)]
                for d in range(4):
                    variance_gradient[d] += local_variance[d]
                    projection_gradient[cf*4+d] += raw_variance*variance_native[d]
                by_cf[cf]['variance_route_gradient_vectors'].extend(local_variance)
                for name in names:
                    by_cf[cf]['mean4_gradient_vectors'][name].extend(local[name])
                    for d in range(4):
                        gradients[name][d] += local[name][d]
        for name in names:
            flattened[name].extend(gradients[name])
        variance_flattened.extend(variance_gradient)
        applied_uncertainty = variance_gradient
        applied_mean=gradients['mean_NLL' if inputs.version=='v5' else 'mean_Huber_delta1']
        comparisons = {left + '_vs_' + right: vector_comparison(gradients[left], gradients[right])
                       for left,right in (('mean_NLL','mean_MAE'), ('mean_NLL','mean_Huber_delta1'),
                                          ('mean_MAE','mean_Huber_delta1'))}
        rows.append({'row_id': witness['row_ids'][b], 'mean4_native_gradients': gradients,
                     'applied_variance_into_mean4_gradient': [0.]*4,
                     'applied_training_mean4_gradient': applied_mean,
                     'applied_training_mean_objective':'Gaussian_NLL' if inputs.version=='v5' else 'Huber_delta1',
                     'applied_uncertainty4_NLL_gradient': applied_uncertainty,
                     'mean4_same_basis_comparisons': comparisons})
    for cell in by_cf:
        cell['mean_inverse_variance'] = mean(cell.pop('inverse_variance'))
        cell['mean_absolute_error'] = mean(cell.pop('absolute_error'))
        cell['mean4_flattened_gradient_l2'] = {name: math.sqrt(math.fsum(x*x for x in values))
                                               for name,values in cell.pop('mean4_gradient_vectors').items()}
        cell['variance_route_local_sensitivity_l2'] = math.sqrt(math.fsum(x*x for x in cell.pop('variance_route_gradient_vectors')))
    pairs = (('mean_NLL','mean_MAE'),
             ('mean_NLL','mean_Huber_delta1'), ('mean_MAE','mean_Huber_delta1'))
    return {'status': 'computed_from_saved_tensors', 'population': 8, 'stage': stage,
            'scope': 'Local mean4 and separately identified variance-route adjoints from saved coefficients; no parent32/upstream parameter gradient replay',
            'precision': 'Analytic F64 approximates F32 intermediate chain rule; not gradient bit parity',
            'normalization': 'Same eligible-example/channel/target-cell hierarchy as saved Gaussian NLL',
            'MAE_zero_subgradient': 0., 'Huber_delta_standardized_target_units': 1.,
            'canonical_order': [0,1,2], 'stored_variance_consistency_max_abs_error': maximum_variance_error,
            'rows': rows, 'mean4_same_basis_objective_comparisons': {left+'_vs_'+right: vector_comparison(flattened[left],flattened[right])
                                                  for left,right in pairs},
            'mean4_flattened_gradient_l2': {name: math.sqrt(math.fsum(x*x for x in values))
                                            for name,values in flattened.items()},
            'applied_variance_into_mean4_l2': 0.,
            'applied_mean_into_uncertainty4_l2': 0.,
            'applied_mean_objective':'Gaussian_NLL' if inputs.version=='v5' else 'Huber_delta1',
            'V6_reporting_NLL_mean_derivative_applied':False if inputs.version=='v6' else None,
            'applied_mean4_gradient_l2':math.sqrt(math.fsum(value*value for value in
                flattened['mean_NLL' if inputs.version=='v5' else 'mean_Huber_delta1'])),
            'variance_route': {
                'coordinate_basis': 'independent-uncertainty4',
                'applied_native_gradient': True,
                'applied_uncertainty4_l2': math.sqrt(math.fsum(x*x for x in variance_flattened)),
                'scale_warning': 'Latent adjoint norms depend on coordinate scale and decoder parameterization; no cross-basis magnitude dominance, cosine, orthogonality or conflict inference'},
            'variance_head_local_gradient_l2': {
                'projection': math.sqrt(math.fsum(x*x for x in projection_gradient)),
                'bias': math.sqrt(math.fsum(x*x for x in bias_gradient))},
            'by_channel_feature': by_cf,
            'limits': 'V5 Huber1 is hypothetical/unapplied; V6 mean-NLL/MAE are unapplied diagnostic adjoints. Both uncertainty4 branches are independent; no cross-basis cosine. Local adjoints omit upstream Jacobians and optimizer/clip coupling. No next encoder or coefficients selected.'}

def residual_allocation(inputs,witness,stage):
    decoder=inputs.tensor(stage+'_checkpoint','payload','model','encoder','change_decoder',shape=(3456,4))
    residual=[math.fsum(witness['native'][b*40+32+d]*decoder[i*4+d] for d in range(4))
              for b in range(8) for i in range(3456)]
    parent=[total-part for total,part in zip(witness['mean'],residual)]
    rows=[]
    for c in range(3):
        for f in range(9):
            indices=[((b*3+c)*128+h)*9+f for b in range(8) for h in range(128)
                     if witness['support'][((b*3+c)*128+h)*9+f]]
            r=[residual[i] for i in indices];p=[parent[i] for i in indices]
            rows.append({'channel_id':c,'feature':f,'target_cells':len(indices),
                         'residual_quantiles':quantiles(r),'parent_remainder_quantiles':quantiles(p),
                         'parent_residual_pearson':pearson(p,r)})
    return {'population':8,'per_channel_feature':rows,
            'interpretation':'Saved linear mean4 residual only; parent=stored total-minus-F64 reconstructed residual approximates F32 addition. No parent nonlinear decoder execution, content20/spectral12 decoder split or counterfactual predictor.'}


def actual_batches(inputs):
    bindings=inputs.actual_bindings()
    csv_rows,trajectory=inputs.actual_csv_rows()
    reports=[]
    shape=(8,3,128,9)
    for binding,csv_row,update in zip(bindings,(csv_rows[0],csv_rows[-1]),(1,2048)):
        identity=f'v6_actual_update_{update}'
        tensor=lambda *field,shape=None:inputs.tensor(identity,*field,shape=shape)
        require(tensor('update')==[update] and tensor('attempt')==[update-1], 'Actual batch counters differ')
        rows=tensor('row_ids',shape=(8,))
        require(all(isinstance(row,int) and not isinstance(row,bool) and TRAIN_BEGIN<=row<=TRAIN_END for row in rows),
                'Actual-batch row outside TRAIN')
        require(csv_row['update']==update,'CSV/witness successful update association differs')
        timestamps=tensor('timestamps',shape=(8,128))
        for b,endpoint in enumerate(rows):
            require(timestamps[b*128:(b+1)*128]==[endpoint+(h-127)*60000 for h in range(128)],
                    'Actual batch timestamps exceed TRAIN window')
        masks={name:boolean(tensor(name,shape=shape),name)
               for name in ('observed','hidden','context_deleted','visible','target_mask')}
        require(all(t==(o and h) and v==(o and not h and not d)
                    for o,h,d,v,t in zip(masks['observed'],masks['hidden'],masks['context_deleted'],masks['visible'],masks['target_mask'])),
                'Actual target/context/visible support differs')
        eligible=boolean(tensor('eligible_channels',shape=(8,3)),'actual eligible channels')
        at=lambda b,c,h,f:((b*3+c)*128+h)*9+f
        require(eligible==[any(masks['target_mask'][at(b,c,h,f)] for h in range(128) for f in range(9))
                           for b in range(8) for c in range(3)], 'Actual channel eligibility differs')
        witness={'row_ids':rows,'support':masks['target_mask'],'target':tensor('targets',shape=shape),
                 'mean':tensor('mean',shape=shape),'variance':tensor('variance',shape=shape),
                 'native':tensor('native',shape=(8,40))}
        require(all(math.isfinite(value) for key in ('target','mean','variance','native') for value in witness[key]),
                'Nonfinite actual saved distribution')
        require(all(v>0 for v,s in zip(witness['variance'],witness['support']) if s), 'Actual supported variance not positive')
        components=reduced_components(witness)
        scalars={}
        for row_field,scalar in (('row_mean_huber','mean_Huber_F64'),('row_variance_nll','variance_NLL_F64'),
                                 ('row_total_objective','total_objective_F64')):
            saved=tensor(row_field,shape=(8,))
            for a,b in zip(components[row_field],saved):near(a,b,'Actual '+row_field)
            scalar_values=tensor(scalar)
            require(len(scalar_values)==1,'Actual scalar shape differs')
            near(mean(saved),scalar_values[0],'Actual '+scalar)
            near(scalar_values[0],csv_row[scalar],'Actual CSV '+scalar)
            scalars[scalar]=scalar_values[0]
        reporting=tensor('Gaussian_NLL_F64')
        require(len(reporting)==1,'Actual reporting scalar shape differs')
        near(mean(components['row_nll']),reporting[0],'Actual reporting Gaussian NLL')
        near(reporting[0],csv_row['Gaussian_NLL_F64'],'Actual CSV reporting NLL')
        scalars['Gaussian_NLL_F64']=reporting[0]
        norm,factor=tensor('pre_clip_global_norm'),tensor('clip_factor')
        require(len(norm)==len(factor)==1 and math.isfinite(norm[0]) and norm[0]>=0,'Observed norm invalid')
        near(min(1.,1./(norm[0]+1e-6)),factor[0],'Actual observed clip-factor arithmetic')
        near(norm[0],csv_row['pre_clip_global_norm'],'Actual CSV observed norm')
        near(factor[0],csv_row['clip_factor'],'Actual CSV clip factor')
        reports.append({'update':update,'pre_step_state_updates':update-1,'row_ids':rows,'components':scalars,
                        'recomputed_rows':components,'observed_pre_clip_global_norm':norm[0],
                        'clip_factor':factor[0],'binding':binding,
                        'scope':'Actual V6-only pre-step distribution; no matched decoder coefficients or local adjoints; norm observed, not reconstructed; two differing batches, not a training curve'})
    return {'status':'computed_from_saved_tensors','batches':reports,'V5_actual_batch_evidence':'absent-never-replayed',
            'full_parameter_gradient_reconstruction':False,'decoder_adjoint_reconstruction':False,
            'TRAIN_trajectory_scalar_diagnosis':trajectory,
            'numeric_CSV_successful_updates_accessed':2048,
            'actual_tensor_updates_accessed':[1,2048]}


def diagnose_version(inputs, rows, groups, targets, target_valid):
    stages, query_witnesses, native_by_stage = {}, {}, {}
    for stage in ('initial', 'final'):
        feature_report, native, valid = features(inputs, stage, rows, targets, target_valid)
        query_report, witness = query(inputs, stage, rows)
        _, metadata = checkpoint_metadata(inputs, stage)
        heads = head_scores(inputs, stage, rows, groups, targets, target_valid, native, valid)
        means = {task: {head: {metric: mean([rep['tasks'][task][head][metric] for rep in heads
                                           if rep['tasks'][task][head][metric] is not None])
                              for metric in ('r2', 'mae', 'rmse', 'coverage')}
                       for head in ('linear', 'neural')} for task in TASKS}
        gradients = analytic_change_gradients(inputs, witness, stage)
        require(gradients['status'] == 'computed_from_saved_tensors', 'Complete local derivative diagnosis required')
        stages[stage] = {'checkpoint_provenance': metadata, 'native_features': feature_report,
                         'saved_TRAIN_head_scores': heads,
                         'per_task_mean_across_saved_repetitions': means,
                         'fixed_query_TRAIN_diagnosis': query_report,
                         'analytic_local_native_gradient_diagnosis': gradients,
                         'saved_mean_residual_allocation':residual_allocation(inputs,witness,stage)}
        query_witnesses[stage] = witness
        native_by_stage[stage] = (native, valid)
    require(query_witnesses['initial']['support'] == query_witnesses['final']['support'] and
            query_witnesses['initial']['target'] == query_witnesses['final']['target'],
            'Initial/final fixed TRAIN queries differ')
    return {'instance': 'CPB-' + inputs.version + '.market-01', 'stages': stages}, native_by_stage, query_witnesses


def diagnose(card, inputs):
    rows = train_ids(inputs.tensor('train_targets', 'row_ids', shape=(112,)))
    require(rows == sorted(rows), 'Prespecified early/late halves require chronological rows')
    groups = inputs.tensor('train_targets', 'group_ids', shape=(112,))
    targets = inputs.tensor('train_targets', 'targets', shape=(112,4))
    target_valid = boolean(inputs.tensor('train_targets', 'valid', shape=(112,4)), 'TRAIN targets')
    require(all(math.isfinite(v) for v in targets), 'Nonfinite TRAIN target')
    models, natives, witnesses = {}, {}, {}
    for version in VERSIONS:
        models[version], natives[version], witnesses[version] = diagnose_version(
            VersionInputs(inputs, version), rows, groups, targets, target_valid)
    x, xv = natives['v5']['initial']
    y, yv = natives['v6']['initial']
    require(xv == yv and all(x[r*40+d] == y[r*40+d] for r in range(112) for d in range(40)),
            'Paired initial whole40 full-context TRAIN features/support differ')
    for stage in ('initial', 'final'):
        require(witnesses['v5'][stage]['row_ids'] == witnesses['v6'][stage]['row_ids']
                and witnesses['v5'][stage]['support'] == witnesses['v6'][stage]['support']
                and witnesses['v5'][stage]['target'] == witnesses['v6'][stage]['target'],
                'Cross-version TRAIN query row/target support differs')
    require(all(witnesses['v5']['initial'][key] == witnesses['v6']['initial'][key]
                for key in ('mean', 'variance')), 'Paired initial masked TRAIN distributions differ')
    a, b = witnesses['v5']['initial']['native'], witnesses['v6']['initial']['native']
    require(all(a[r*40+d] == b[r*40+d] for r in range(8) for d in range(40)),
            'Paired initial masked TRAIN whole40 differs')
    extra_initial = [y[r*40+d] for r in range(112) for d in range(36,40)]
    require(all(math.isfinite(value) for value in extra_initial) and any(value != 0 for value in extra_initial),
            'V6 fresh independent uncertainty4 is nonfinite or vacuously all zero')
    block_comparison, head_comparison, query_comparison, gradient_comparison = [], [], [], []
    for stage in ('initial', 'final'):
        left, right = models['v5']['stages'][stage], models['v6']['stages'][stage]
        for name, lo, hi in COMMON_BLOCKS:
            a, b = left['native_features']['blocks'][name], right['native_features']['blocks'][name]
            pair = {'stage': stage, 'block': name, 'v5': a, 'v6': b}
            for metric in ('covariance_trace','covariance_frobenius_squared','participation_ratio'):
                pair[metric + '_v6_minus_v5'] = b[metric]-a[metric] if a[metric] is not None and b[metric] is not None else None
            x, xv = natives['v5'][stage]
            y, yv = natives['v6'][stage]
            require(xv == yv, 'Paired native feature eligibility differs')
            pair['exact_same_block_values'] = all(x[r*40+d] == y[r*40+d]
                                                   for r in range(112) for d in range(lo,hi))
            pair['mean_absolute_coordinate_difference'] = mean([abs(y[r*40+d]-x[r*40+d])
                                                                  for r in range(112) if xv[r] for d in range(lo,hi)])
            block_comparison.append(pair)
        for task in TASKS:
            for head in ('linear','neural'):
                a = left['per_task_mean_across_saved_repetitions'][task][head]
                b = right['per_task_mean_across_saved_repetitions'][task][head]
                head_comparison.append({'stage': stage, 'task': task, 'head': head, 'v5': a, 'v6': b,
                                        'v6_minus_v5': {metric: b[metric]-a[metric] if a[metric] is not None and b[metric] is not None else None
                                                        for metric in ('r2','mae','rmse','coverage')}})
        qa, qb = left['fixed_query_TRAIN_diagnosis'], right['fixed_query_TRAIN_diagnosis']
        query_comparison.append({'stage': stage, 'all112_TRAIN_v5': qa['all112_TRAIN'],
                                 'all112_TRAIN_v6': qb['all112_TRAIN'],
                                 'all112_MAE_v6_minus_v5': qb['all112_TRAIN']['standardized_fixed_query_MAE']-qa['all112_TRAIN']['standardized_fixed_query_MAE'],
                                 'all112_NLL_v6_minus_v5': qb['all112_TRAIN']['Gaussian_NLL_F64']-qa['all112_TRAIN']['Gaussian_NLL_F64'],
                                 'B8_channel_features': [
                                     {'channel_id': a['channel_id'], 'feature': a['feature'],
                                      'v6_minus_v5': {key: b[key]-a[key] if b[key] is not None and a[key] is not None else None for key in
                                          ('mean_absolute_error','mean_squared_error','mean_variance','mean_inverse_variance',
                                           'mean_half_squared_error_over_variance','mean_half_log_variance','mean_Gaussian_NLL')}}
                                     for a,b in zip(qa['per_channel_feature'],qb['per_channel_feature'])],
                                 'limit': 'Cellwise/gradient evidence first8 TRAIN only;112 summaries do not expand detailed coverage'})
        ga, gb = left['analytic_local_native_gradient_diagnosis'], right['analytic_local_native_gradient_diagnosis']
        gradient_comparison.append({'stage': stage,
                                    'v5_mean4_norms': ga['mean4_flattened_gradient_l2'],
                                    'v6_mean4_norms': gb['mean4_flattened_gradient_l2'],
                                    'v5_variance_route': ga['variance_route'],
                                    'v6_variance_route': gb['variance_route'],
                                    'v5_mean4_same_basis_objective_comparisons': ga['mean4_same_basis_objective_comparisons'],
                                    'v6_mean4_same_basis_objective_comparisons': gb['mean4_same_basis_objective_comparisons'],
                                    'variance_into_mean4_applied_l2_both': [0.,0.],
                                    'cross_basis_comparison': 'Not computed: independent mean4/uncertainty4 coordinate bases; norms are parameterization dependent'})
    return {'kind': 'cpb_v5_v6_saved_TRAIN_diagnosis_v1', 'status': 'passed',
            'baseline': 'CPB-v3.market-01', 'comparison_references': ['CPB-v5.market-01','CPB-v6.market-01'],
            'candidate_declared': False, 'scope': 'Saved TRAIN diagnostic arithmetic; not new quality evidence',
            'TRAIN_cutoff_ms': TRAIN_END, 'TRAIN_rows': rows, 'TRAIN_probe_population': 112,
            'tasks': list(TASKS), 'models': models,
            'paired_initial_whole40_TRAIN_values_exact': True,
            'V6_initial_uncertainty4_finite_nonzero': True,
            'actual_V6_pre_step_batch_diagnosis':actual_batches(inputs),
            'comparison': {'native_blocks': block_comparison, 'saved_TRAIN_head_scores': head_comparison,
                           'masked_reconstruction': query_comparison, 'local_native_adjoints': gradient_comparison},
            'counters': {'model_calls': 0, 'encoder_calls': 0, 'CPU_encoder_calls': 0, 'old_encoder_calls': 0,
                         'decoder_calls': 0, 'GPU_calls': 0, 'model_instantiations': 0,
                         'training_updates': 0, 'new_head_fits': 0, 'new_scaler_fits': 0,
                         'head_fits': 0, 'validation_numeric_reads': 0, 'raw_cache_numeric_reads': 0,
                         'neural_head_forward_calls': 0, 'validation_numeric_fields_accessed': 0,
                         'test_numeric_fields_accessed': 0, 'new_quality_scores': 0},
            'mixed_summary_JSON_policy': {'whole_files_parsed': 4, 'selected_numeric_key': 'train',
                                          'VALIDATION_numeric_fields_used': 0,
                                          'files_contain_both_splits': True},
            'interpretation_limits': ['Known TRAIN diagnostics do not establish held-out improvement or causation',
                                      'Detailed query tensors and local derivatives cover first8 TRAIN examples only',
                                      'Aggregate query JSON files contain both splits and are parsed, but only TRAIN key values are accessed',
                                      'Head-fit archive metadata may describe neural or other fields; only prospective ridge/normalizer tensor selectors are read',
                                      'Ridge reconstruction/components are descriptive arithmetic of already fitted coefficients, without new head or block ablation selection',
                                      'Local derivatives omit parent32/encoder parameter Jacobians and optimizer trajectory; no claim of full training gradient replay',
                                      'V6 mean4/uncertainty4 are independent coordinate bases: cross-basis cosine, orthogonality and norm-dominance claims are invalid',
                                      'V5 applies joint Gaussian mean/uncertainty derivatives; V6 applies Huber mean plus detached-error variance. Reporting Gaussian NLL is never a V6 backward objective',
                                      'V5/V6 native40 capacity and fixed head settings match; TRAIN differences still do not isolate a causal explanation or seed robustness',
                                      'V5 Huber/MAE and V6 mean-NLL/MAE adjoints are hypothetical diagnostic alternatives; no next objective is chosen',
                                      'V6 actual two-batch tensors admit component/support/CSV/observedclip arithmetic only; no matching pre-step2047 decoder exists, no actual-batch adjoints or V5 replay',
                                      'No candidate, hyperparameter, gate, checkpoint promotion or oracle floor is chosen by this script']}

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
        source_root = Path(__file__).resolve().parent
        source_before = {name: file_pin(source_root/name) for name in
                         ('diagnose.py','archive_reader.py','schema.py','README.md','provenance.json')}
        with args.card.open(encoding='utf-8-sig') as source:
            card = json.load(source)
        inputs = Inputs(card)  # ALL allowed/preservation pins verified before numeric access.
        report = diagnose(card, inputs)
        # Recheck all source data/metadata bytes after analysis, without reading
        # numeric fields from any preservation-only file.
        for pin in card['allowed_inputs'] + card.get('preservation_pins', []) + [card['pre_execution_preservation']]:
            inputs.verify(pin)
        require(file_pin(args.card) == card_pin, 'Card changed during diagnosis')
        require(all(file_pin(source_root/name) == binding for name,binding in source_before.items()),
                'Owned diagnostic source changed during diagnosis')
        report.update({'card': card_pin, 'diagnostic_source': file_pin(__file__),
                       'inert_archive_reader_source': file_pin(Path(__file__).with_name('archive_reader.py')),
                       'all_owned_source_files_unchanged': True, 'owned_source_pins': source_before,
                       'verified_allowed_inputs': inputs.verified, 'verified_preservation_pins': inputs.preservation,
                       'verified_pre_execution_preservation': inputs.protocol_preservation,
                       'opened_allowed_ids': inputs.opened, 'numeric_fields_accessed': inputs.numeric_fields,
                       'elapsed_seconds': time.monotonic()-started})
        write_exclusive(output/'diagnosis.json', report)
        print(json.dumps({'status': report['status'], 'output': str(output/'diagnosis.json'),
                          'sha256': sha256(output/'diagnosis.json'),
                          'local_gradient_stages': 4}))
    except Exception as error:
        write_exclusive(output/'failure.json', {'kind': 'cpb_v5_v6_TRAIN_diagnosis_refusal_v1',
                                               'status': 'failed', 'reason': str(error),
                                               'encoder_calls': 0, 'head_fits': 0})
        raise
    finally:
        if inputs is not None:
            inputs.close()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Owned V12-only saved TRAIN information diagnosis with closed audited V11 cache.

SOURCE PREPARATION ONLY. No actual card or numerical invocation is admitted by
these source files. All affine contractions below are disclosed scalar algebra,
never a model/head module, refit, ablation predictor, or prior V11 body replay.
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
from archive_reader import Archive, Tensor

TRAIN_BEGIN, TRAIN_END = 1704067259999, 1705881599999
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
REPS = (2701, 2802, 2903)
STAGES = ('initial', 'final')
COMMON_BLOCKS = (('learned20',0,20),('fixed_spectral12',20,32),('mean4',32,36),('uncertainty4',36,40))
WIDTHS, BLOCKS = {'v12':40}, {'v12':COMMON_BLOCKS}
HEAD_SEEDS_U64 = {'v12':(6049515622977958931,8262941649267994858,2095371932330426401)}
OBJECTIVE_POLICY = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
GRADIENT_POLICY = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
DIRECT_ROUTES = 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
NATIVE_LAYOUT = ('parent32[0:32];mean4[32:36];uncertainty4[36:40];'
 'independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;variance-reads-live-uncertainty4-only')
POOL_POLICY = 'original-position-four-bin-attention-mean4-uniform-centered-variance4-v1'
MOMENT_POLICY = 'componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1'
EXPORT_LAYOUT = 'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'
EMPTY_POLICY = 'zero-whole-export-slice-after-bias-v1'
CACHE_ID = 'cached_v11_prior_TRAIN_diagnosis'
CACHE_SELECTORS = ('models.v11.stages.initial','models.v11.stages.final',
                   'temporal_moment_observations.csv','temporal_moment_observations.actual_witnesses')
BASE_IDS = {'train_targets',CACHE_ID,'v12_temporal_moment_csv','v12_actual_update_1','v12_actual_update_2048'}
ALLOWED_IDS = BASE_IDS | {'v12_'+stage+'_'+kind for stage in STAGES
 for kind in ('features','query','query_summary','checkpoint')} | {
 'v12_'+stage+'_'+kind+'_'+str(rep) for stage in STAGES for kind in ('head','fit') for rep in REPS}
V11_GUARD_STEMS = ('initial_features','initial_query','initial_query_summary','initial_checkpoint',
 'final_features','final_query','final_query_summary','final_checkpoint','actual_update_1','actual_update_2048','temporal_moment_csv',
 'actual_bindings','training_csv','temporal_learning_receipt','temporal_learning_csv','temporal_gauge_csv','temporal_between_csv')
V11_GUARD_IDS = {'v11_'+name for name in V11_GUARD_STEMS} | {
 'v11_'+stage+'_'+kind+'_'+str(rep) for stage in STAGES for kind in ('head','fit') for rep in REPS}
V12_GUARD_IDS = {'v12_'+name for name in ('actual_bindings','training_csv','temporal_learning_receipt',
 'temporal_learning_csv','temporal_gauge_csv','temporal_between_csv')}
HASH_GUARD_IDS = V11_GUARD_IDS | V12_GUARD_IDS
CONTEXT_IDS = {'old_card','old_audit','old_main_mirror','old_audit_mirror','v12_training_audit',
 'v12_final_establishment','v12_final_preservation','current_goal_protocol','retention_independent_review',
 'retention_application_admission','current_ledger','current_catalog','old_completion','v12_quality_audit'}
CONTEXT_ASSERTIONS = (
 {'id':'old_completion_passed','input':'old_completion','path':['status'],'equals':'passed'},
 {'id':'old_independent_V11_mismatches_zero','input':'old_completion','path':['independent_mismatches'],'equals':0},
 {'id':'v12_preservation_passed','input':'v12_final_preservation','path':['status'],'equals':'passed'},
 {'id':'v12_critical114','input':'v12_final_preservation','path':['critical_checkpoint_total'],'equals':114},
 {'id':'v12_TRAIN_passed','input':'v12_training_audit','path':['status'],'equals':'passed'},
 {'id':'v12_quality_passed','input':'v12_quality_audit','path':['status'],'equals':'passed'},
 {'id':'v12_fulltrust','input':'v12_final_establishment','path':['trust_admission_passed'],'equals':True},
 {'id':'retention_review_passed','input':'retention_independent_review','path':['status'],'equals':'passed'},
 {'id':'retention_applied','input':'retention_application_admission','path':['applied'],'equals':True},
 {'id':'retention_eligible','input':'retention_application_admission','path':['prospective_future_use_eligible'],'equals':True},
 {'id':'V3_ledger','input':'current_ledger','path':['baseline'],'equals':'CPB-v3.market-01'},
 {'id':'V3_catalog','input':'current_catalog','path':['active_application'],'equals':'CPB-v3.market-01'})
QUERY_FIELDS = {'row_ids','target_mask','targets','mean','variance','native','row_mae','row_nll',
                'row_mean_huber','row_variance_nll','row_total_objective'}
OBJECTIVE_FIELDS = {'objective_policy','mean_huber_delta_f64','loss_coefficients_f64',
 'variance_error_detached','direct_gradient_routes','reporting_nll_backpropagated','gradient_policy'}
LAYOUT_FIELDS = {'native_layout','mean_width','uncertainty_width','native_width'}
TEMPORAL_FIELDS = {'temporal_pooling_policy','temporal_pooling_partition_v1','temporal_pooling_mix_f64',
 'empty_local_bin_policy','temporal_moment_policy','temporal_export_layout','temporal_moment_partition_v1'}
CHECKPOINT_FIELDS = {'artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates',
 'causal_updates','initialization_seed','channel_ids','payload.model.encoder.change_decoder'} | {
 prefix+field for prefix in ('','payload.') for field in OBJECTIVE_FIELDS|LAYOUT_FIELDS|TEMPORAL_FIELDS}
ACTUAL_FIELDS = {'update','attempt','clip_factor','temporal_export_weight_before','temporal_export_bias_before',
 'temporal_patch_positions','temporal_patch_valid','temporal_patch_rows'} | {
 prefix+field for prefix in ('temporal_','temporal_contextual_') for field in (
 'pool_mixture_weights','pool_bin_valid','pool_bin_mean','pool_bin_variance','pool_dispersion_weights','pool_dispersion_mean',
 'pool_projected_upstream_defined','pool_projected_upstream_pre_clip')}
MOMENT_CSV_FIELDS = ('update','bin','moment','export_row_begin','export_row_end','weight_gradient_l2',
 'bias_gradient_l2','weight_movement_l2','bias_movement_l2','nonzero_export_gradient_values','moved_export_values')
UPDATE_BLOCKS = ((1,512),(513,1024),(1025,1536),(1537,2048))
QUANTILES = (0.,.01,.05,.25,.5,.75,.95,.99,1.)
COUNTERS = {'manual_saved_affine_contractions':{}, 'scalar_summary_function_calls':{},
            'captured_center_gap_scalar_subtractions':0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def file_pin(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'Physical input file required')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path)}


def same_pin(left,right):
    return left['bytes']==right['bytes'] and left['sha256']==right['sha256']


def count_scalar(name):
    counts=COUNTERS['scalar_summary_function_calls']
    counts[name]=counts.get(name,0)+1


def record_affine(kind, outputs, terms):
    counter=COUNTERS['manual_saved_affine_contractions'].setdefault(kind,{'dot_outputs':0,'multiply_add_terms':0})
    counter['dot_outputs']+=outputs
    counter['multiply_add_terms']+=outputs*terms


def expected_fields(identity):
    require(identity in ALLOWED_IDS,'Unknown numeric input identity')
    if identity=='train_targets': return {'row_ids','group_ids','targets','valid'}
    if identity in ('v12_actual_update_1','v12_actual_update_2048'): return ACTUAL_FIELDS
    if identity.endswith('_features'): return {'row_ids','values','valid'}
    if identity.endswith('_query'): return QUERY_FIELDS
    if identity.endswith('_checkpoint'): return CHECKPOINT_FIELDS
    if '_head_' in identity:
        return {'row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions'}
    if '_fit_' in identity:
        return {'feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed'}
    return set()


def role_for(identity):
    if identity=='train_targets':return 'shared_TRAIN_only_targets'
    if identity==CACHE_ID:return 'audited_prior_TRAIN_output_exact_four_V11_subtrees_only'
    if identity=='v12_temporal_moment_csv':return 'V12_TRAIN_moment_subgroup_16384_rows'
    if identity.endswith('_query_summary'):return 'aggregate_JSON_TRAIN_key_only'
    return 'TRAIN_archive'


def at(record, selector):
    for key in selector.split('.'):
        record=record[key]
    return record


class Inputs:
    """Hash every allowed/guard/context binding before the first storage value.

    Archive metadata may contain unselected fields, but only tensor() can read
    values. Mixed summaries return only train; cached JSON returns only4V11 paths.
    """
    def __init__(self,card):
        require(card.get('kind')=='cpb_v11_v12_saved_train_diagnosis_card_v1' and
          card.get('stage')=='prospective_before_diagnostic_arithmetic' and card.get('execution_admitted') is True,
          'Separately frozen and admitted diagnosis card required')
        require(card.get('baseline')=='CPB-v3.market-01' and card.get('computed_versions')==['v12'] and
          card.get('cached_reference_versions')==['v11'] and card.get('train_rows')==112 and
          card.get('query_rows')==8 and card.get('train_begin_ms')==TRAIN_BEGIN and card.get('train_end_ms')==TRAIN_END,
          'Closed TRAIN population/version declaration')
        require(card.get('fixed_update_blocks')==[list(x) for x in UPDATE_BLOCKS] and
          card.get('cached_v11_selectors')==list(CACHE_SELECTORS),'Fixed blocks/closed cache selectors required')
        require(card.get('manual_saved_affine_contractions_declared') is True and
          card.get('module_forwards_allowed') is False and card.get('new_quality_or_selection_allowed') is False,
          'Manual saved arithmetic and no model/selection boundary')
        self.card,self.pins,self.archives,self.opened,self.numeric_fields=card,{},{},{},[]
        self.verified=[]
        rows=card.get('allowed_inputs',[])
        require(len(rows)==25 and {x['id'] for x in rows}==ALLOWED_IDS,'Exactly25 closed numeric roles')
        for row in rows:
            identity=row['id'];require(identity not in self.pins,'Duplicate numeric identity')
            require(row.get('role')==role_for(identity),'Exact source-declared role')
            require('validation' not in row['container_path'].lower(),'Held-out input refused')
            if row['role'] in ('TRAIN_archive','shared_TRAIN_only_targets'):
                fields=row.get('allowed_tensor_fields',[])
                require(len(fields)==len(set(fields)) and set(fields)==expected_fields(identity),
                        'Exact unique tensor allowlist, no literal payload. entry')
            if identity.endswith('_query_summary'):
                require(row.get('numeric_json_selectors')==['train'],'Only query TRAIN selector')
            if identity==CACHE_ID:
                require(row.get('numeric_json_selectors')==list(CACHE_SELECTORS),'Only4cached V11 subtrees')
            if identity=='v12_temporal_moment_csv':
                require(row.get('allowed_CSV_columns')==list(MOMENT_CSV_FIELDS),'Exact11 moment CSV columns')
            self.pins[identity]=row
            self.verified.append(self.verify(row))
        guards=card.get('hash_only_guards',[])
        require(len(guards)==35 and {x['id'] for x in guards}==HASH_GUARD_IDS,'29V11+6V12 hash-only guards')
        self.hash_guards=[self.verify(row) for row in guards]
        contexts=card.get('context_inputs',[])
        require(len(contexts)==14 and {x['id'] for x in contexts}==CONTEXT_IDS,'14 actual final-context bindings')
        self.context_bindings={x['id']:x for x in contexts}
        self.context={}
        for row in contexts:
            self.verify(row)
            with Path(row['container_path']).open(encoding='utf-8-sig') as stream:
                self.context[row['id']]=json.load(stream)
        self.preservation=[self.verify(row) for row in card.get('preservation_pins',[])]
        self.pre_execution_preservation=self.verify(card['pre_execution_preservation'])
        # Literal ARRAY objects are the reviewed12 context assertions; typed equals
        # prevents bool/int substitution. No trust algorithm or old body is replayed.
        assertions=card.get('context_assertions',[])
        require(isinstance(assertions,list) and len(assertions)==12,'Explicit12 context assertion array')
        for row,expected in zip(assertions,CONTEXT_ASSERTIONS):
            require(row==expected and type(row['equals']) is type(expected['equals']),
                    'Exact typed final-context assertion object')
            actual=self.context[row['input']]
            for key in row['path']:actual=actual[key]
            require(actual==row['equals'] and type(actual) is type(row['equals']),
                    'Actual final-context assertion: '+row['id'])
        old=self.context['old_audit']
        require(old['status']=='passed' and not old['mismatches'] and old['checks']==68264 and
          same_pin(old['card'],self.context_bindings['old_card']) and same_pin(old['diagnosis'],self.pins[CACHE_ID]),
          'Preserved independent old audit binds cached V11 source card/result')
        main_mirror,audit_mirror=self.context['old_main_mirror'],self.context['old_audit_mirror']
        require(main_mirror['status']=='passed' and main_mirror['container_before_after_exact'] is True and
          main_mirror['all_sizes_sha256_exact'] is True and
          any(same_pin(x,self.pins[CACHE_ID]) for x in main_mirror['files']) and
          audit_mirror['status']=='passed' and audit_mirror['source_before_after_exact'] is True and
          same_pin(audit_mirror['independent_audit'],self.context_bindings['old_audit']),
          'Closed prior source/mirror/audit provenance, no V11 replay')
        old_card=self.context['old_card']
        completion=self.context['old_completion']
        require(completion['independent_checks']==68264 and
          all(same_pin(completion[key],binding) for key,binding in (
            ('main_card',self.context_bindings['old_card']),('main_diagnosis',self.pins[CACHE_ID]),
            ('independent_audit',self.context_bindings['old_audit']),
            ('main_mirror',self.context_bindings['old_main_mirror']),
            ('independent_audit_mirror',self.context_bindings['old_audit_mirror']))) and
          same_pin(main_mirror['canonical_diagnosis'],self.pins[CACHE_ID]),
          'Completed V11 metadata binds the five prior card/result/audit/mirror files')
        prior_guards=[x for x in old_card['allowed_inputs']+old_card['hash_only_guards']
                      if x['id'].startswith('v11_')]
        require(len(prior_guards)==29 and {x['id'] for x in prior_guards}==V11_GUARD_IDS,
                'Old V11 guard union is exactly23 prior numeric plus6 prior hash-only files')
        for row in guards:
            if row['id'].startswith('v11_'):
                require(any(x['id']==row['id'] and same_pin(x,row) for x in prior_guards),
                  'Every V11 raw guard remains bound to its original prospective card')
        for row in old_card['owned_source_files']:
            require(any(same_pin(item,row) for item in main_mirror['files']),
                    'Old source copies remain exact in completed prior mirror')
        require(self.context['v12_training_audit']['status']=='passed' and
          self.context['v12_final_establishment']['measured_candidate_established'] is True and
          self.context['v12_final_establishment']['trust_admission_passed'] is True,
          'Use completed V12 trust metadata without repeating trust arithmetic')

    @staticmethod
    def verify(row):
        path=Path(row.get('container_path',row.get('path','')))
        require(path.is_absolute(),'Explicit absolute pinned container path')
        actual=file_pin(path)
        require(same_pin(actual,row),'Pinned bytes changed: '+str(path))
        if 'id' in row:actual['id']=row['id']
        return actual

    def archive(self,identity):
        require(identity in self.pins and (self.pins[identity]['role']=='TRAIN_archive' or
          (identity=='train_targets' and self.pins[identity]['role']=='shared_TRAIN_only_targets')),
          'Closed archive identity')
        if identity not in self.archives:
            self.archives[identity]=Archive(self.pins[identity]['container_path'])
            self.opened[identity]='inert_archive_metadata'
        return self.archives[identity]

    def tensor(self,identity,*fields,shape=None,dtype=None):
        key='.'.join(fields)
        require(identity in self.pins and key in expected_fields(identity),'Unselected tensor refused: '+identity+':'+key)
        value=self.archive(identity).field(*fields)
        require(isinstance(value,Tensor),'Inert tensor required')
        require(shape is None or value.shape==tuple(shape),'Declared tensor shape: '+identity+':'+key)
        require(dtype is None or value.storage.dtype==dtype,'Declared tensor dtype: '+identity+':'+key)
        values=value.values()
        require(len(values)==math.prod(value.shape),'Tensor logical size')
        self.numeric_fields.append({'id':identity,'field':key,'shape':list(value.shape),'values':len(values)})
        return values

    def summary(self,identity):
        require(identity in ('v12_initial_query_summary','v12_final_query_summary'),'Closed summary identity')
        with Path(self.pins[identity]['container_path']).open(encoding='utf-8-sig') as stream:
            container=json.load(stream)
        self.opened[identity]='JSON_decode_whole_select_train_only'
        self.numeric_fields.append({'id':identity,'field':'train_only'})
        return container['train']

    def cached(self):
        with Path(self.pins[CACHE_ID]['container_path']).open(encoding='utf-8-sig') as stream:
            container=json.load(stream)
        selected=[at(container,path) for path in CACHE_SELECTORS]
        self.opened[CACHE_ID]='JSON_decode_whole_select_exact4V11subtrees'
        self.numeric_fields.extend({'id':CACHE_ID,'field':path} for path in CACHE_SELECTORS)
        return {'stages':{'initial':selected[0],'final':selected[1]},
          'temporal_moment_observations':{'csv':selected[2],'actual_witnesses':selected[3]}}

    def close(self):
        for archive in self.archives.values():archive.close()


class VersionInputs:
    def __init__(self,inputs):self.inputs,self.version=inputs,'v12'
    def identity(self,identity):return identity if identity=='train_targets' else 'v12_'+identity
    def tensor(self,identity,*fields,**kwargs):return self.inputs.tensor(self.identity(identity),*fields,**kwargs)
    def archive(self,identity):return self.inputs.archive(self.identity(identity))
    def summary(self,identity):return self.inputs.summary(self.identity(identity))

def mean(values):
    count_scalar('mean')
    return math.fsum(values) / len(values) if values else None


def quantiles(values):
    count_scalar('quantiles')
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
    count_scalar('variance')
    center = mean(values)
    return mean([(v - center) ** 2 for v in values]) if values else None


def pearson(x, y):
    count_scalar('pearson')
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
    count_scalar('score')
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
        record_affine('ridge_total',112*4,width)
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
                record_affine('ridge_block',112,hi-lo)
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
                        'manual_saved_affine_contractions': True,
                        'saved_TRAIN_predictions': {'linear':ridge,'neural':neural,'support':valid},
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
    support = boolean(inputs.tensor(identity, 'target_mask', shape=shape), identity)
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
        'status':'applied-mean-objective'}
    if inputs.version == 'v12':
        for field,scalar in (('row_mean_huber','mean_Huber_F64'),('row_variance_nll','variance_NLL_F64'),
                             ('row_total_objective','total_objective_F64')):
            saved=inputs.tensor(identity,field,shape=(8,))
            require(len(summary[field])==112, 'Full112 component rows unavailable')
            require(all(math.isfinite(value) for value in summary[field]), 'Nonfinite component summary')
            for b in range(8):
                near(components[field][b],saved[b],'Saved first8 '+field)
                near(components[field][b],summary[field][b],'Aggregate first8 '+field)
            near(mean(summary[field]),summary[scalar],'Full112 '+scalar)
            report['all112_TRAIN'][scalar]=summary[scalar]
            report['all112_TRAIN'][field]=summary[field]
            for label,lo,hi in (('early56',0,56),('late56',56,112)):
                report['all112_TRAIN']['chronological_TRAIN_halves'][label][scalar]=mean(summary[field][lo:hi])
        for i in range(112):
            near(summary['row_variance_nll'][i],summary['row_nll'][i],'Detached variance/reporting NLL numeric identity')
            near(summary['row_mean_huber'][i]+summary['row_variance_nll'][i],summary['row_total_objective'][i],
                 'Unit component sum')
        report['all112_component_evidence']='saved-V12-no-reconstruction-from-MAE/NLL'

    return report,witness



def residual_allocation(inputs,witness,stage):
    decoder=inputs.tensor(stage+'_checkpoint','payload','model','encoder','change_decoder',shape=(3456,4))
    residual=[math.fsum(witness['native'][b*40+32+d]*decoder[i*4+d] for d in range(4))
              for b in range(8) for i in range(3456)]
    record_affine('mean4_decoder',8*3456,4)
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




def checkpoint_metadata(inputs,stage):
    identity=stage+'_checkpoint'
    def text(*fields):
        return bytes(inputs.tensor(identity,*fields,dtype='ByteStorage')).decode('utf-8')
    def scalar(field):return inputs.tensor(identity,field,dtype='LongStorage')
    require(text('artifact_kind')=='cpb_v12_inference_checkpoint_v1' and text('model_tag')=='CPB-v12',
            'Exact V12 inference type/tag')
    require(text('initialization_kind')=='fresh-seeded-constructor-no-import-v1' and
            text('pretrained_ancestor')=='none' and scalar('source_updates')==[0] and
            scalar('causal_updates')==[0 if stage=='initial' else 2048] and
            scalar('initialization_seed')==[943001], 'Exact fresh same-stage checkpoint lineage')
    require(inputs.tensor(identity,'channel_ids',shape=(3,),dtype='LongStorage')==[0,1,2], 'Canonical channels')
    for prefix in ((),('payload',)):
        for field,expected in (('objective_policy',OBJECTIVE_POLICY),('gradient_policy',GRADIENT_POLICY),
          ('direct_gradient_routes',DIRECT_ROUTES),('native_layout',NATIVE_LAYOUT),
          ('temporal_pooling_policy',POOL_POLICY),('empty_local_bin_policy',EMPTY_POLICY),
          ('temporal_moment_policy',MOMENT_POLICY),('temporal_export_layout',EXPORT_LAYOUT)):
            require(text(*prefix,field)==expected,'Typed declaration: '+'.'.join(prefix+(field,)))
        for field,expected in (('mean_width',4),('uncertainty_width',4),('native_width',40),
          ('variance_error_detached',1),('reporting_nll_backpropagated',0)):
            require(inputs.tensor(identity,*prefix,field,dtype='LongStorage')==[expected],'Typed numeric layout')
        require(inputs.tensor(identity,*prefix,'mean_huber_delta_f64',dtype='DoubleStorage')==[1.]
          and inputs.tensor(identity,*prefix,'loss_coefficients_f64',shape=(2,),dtype='DoubleStorage')==[1.,1.]
          and inputs.tensor(identity,*prefix,'temporal_pooling_mix_f64',shape=(2,),dtype='DoubleStorage')==[.5,.5]
          and inputs.tensor(identity,*prefix,'temporal_pooling_partition_v1',shape=(5,),dtype='LongStorage')==[4,16,32,4,8]
          and inputs.tensor(identity,*prefix,'temporal_moment_partition_v1',shape=(2,),dtype='LongStorage')==[4,4],
          'Exact typed objective/pooling/moment partitions')
    return {'source_updates':0,'causal_updates':0 if stage=='initial' else 2048,
      'initialization_seed':943001,'pretrained_ancestor':'none','gradient_policy':GRADIENT_POLICY,
      'native_width':40,'objective':OBJECTIVE_POLICY,'direct_gradient_routes':DIRECT_ROUTES,
      'clipping':'single-global-norm2-clip1','temporal_pooling':POOL_POLICY,
      'temporal_moment_policy':MOMENT_POLICY,'temporal_export_layout':EXPORT_LAYOUT,
      'temporal_moment_partition':[4,4], 'variance_reads':'live-independent-uncertainty4',
      'typed_metadata_scope':'outer/payload declarations and same-stage change_decoder only; no other model parameters'}


def augment_head_halves(reports,targets):
    for report in reports:
        support=report['saved_TRAIN_predictions']['support']
        for task_index,task in enumerate(TASKS):
            decomposition=report['saved_linear_readout_block_decomposition'][task]
            components={name:decomposition['blocks'][name]['row_contributions_in_target_units']
                        for name,_,_ in COMMON_BLOCKS}
            for label,lo,hi in (('early56',0,56),('late56',56,112)):
                selected=[r for r in range(lo,hi) if support[r*4+task_index]]
                pairs=[]
                for index,name in enumerate(components):
                    for other in list(components)[index+1:]:
                        x=[components[name][r] for r in selected];y=[components[other][r] for r in selected]
                        mx,my=mean(x),mean(y)
                        pairs.append({'blocks':[name,other], 'population_covariance':
                          mean([(a-mx)*(b-my) for a,b in zip(x,y)]), 'pearson':pearson(x,y)})
                decomposition['chronological_TRAIN_halves'][label]['pairwise_component_covariance']=pairs


def reconstruction_associations(heads,query_report,targets):
    full=query_report['all112_TRAIN']
    component_names=('row_mae','row_nll','row_mean_huber','row_variance_nll','row_total_objective')
    records=[]
    for report in heads:
        predictions=report['saved_TRAIN_predictions']
        for task_index,task in enumerate(TASKS):
            for head in ('linear','neural'):
                for label,lo,hi in (('all112',0,112),('early56',0,56),('late56',56,112)):
                    selected=[r for r in range(lo,hi) if predictions['support'][r*4+task_index]
                              and full['row_target_cells'][r]>0]
                    residual=[predictions[head][r*4+task_index]-targets[r*4+task_index] for r in selected]
                    records.append({'repetition':report['repetition'],'task':task,'head':head,'scope':label,
                      'eligible_rows':len(selected),'readout_residual_units':'physical task target units',
                      'reconstruction_units':'standardized query units', 'components':{name:{
                        'pearson_to_signed_readout_residual':pearson([full[name][r] for r in selected],residual),
                        'pearson_to_absolute_readout_residual':pearson([full[name][r] for r in selected],[abs(x) for x in residual])}
                        for name in component_names}})
    return records


def moment_csv(inputs):
    identity='v12_temporal_moment_csv';pin=inputs.pins[identity]
    records=[]
    with Path(pin['container_path']).open(encoding='utf-8-sig',newline='') as source:
        reader=csv.DictReader(source)
        require(tuple(reader.fieldnames or ())==MOMENT_CSV_FIELDS,'Exact11 existing moment columns')
        for row in reader:
            require(len(records)<16384,'Bounded declared moment rows')
            item={key:int(row[key]) for key in MOMENT_CSV_FIELDS[:2]+MOMENT_CSV_FIELDS[3:5]}
            item['moment']=row['moment']
            require(1<=item['update']<=2048 and 0<=item['bin']<4 and
              item['moment'] in ('mean','centered_variance'),'Fixed moment row identity')
            for key in MOMENT_CSV_FIELDS[5:]:
                item[key]=float(row[key])
                require(math.isfinite(item[key]),'Finite recorded moment descriptor')
            records.append(item)
    require(len(records)==16384,'Exact captured 16384-row population')
    blocks=[]
    for name,lo,hi in (('overall',1,2048),('updates1_512',1,512),('updates513_1024',513,1024),
                       ('updates1025_1536',1025,1536),('updates1537_2048',1537,2048)):
        groups=[]
        for bin_index in range(4):
            for half,moment in enumerate(('mean','centered_variance')):
                selected=[row for row in records if lo<=row['update']<=hi and
                          row['bin']==bin_index and row['moment']==moment]
                require(len(selected)==hi-lo+1,'Fixed group population, no selected update subset')
                groups.append({'bin':bin_index,'moment':moment,'export_row_begin':bin_index*8+half*4,
                  'export_row_end':bin_index*8+(half+1)*4,'records':len(selected),
                  'scalars':{key:{'mean':mean([x[key] for x in selected]),
                                    'quantiles':quantiles([x[key] for x in selected])}
                             for key in MOMENT_CSV_FIELDS[5:]}})
        blocks.append({'name':name,'first_update':lo,'last_update':hi,'updates':hi-lo+1,'subgroups':groups})
    inputs.opened[identity]='all16384rows_exact11CSVfields_fixedbins_moments_and512blocks'
    inputs.numeric_fields.append({'id':identity,'fields':list(MOMENT_CSV_FIELDS),'rows':len(records)})
    return {'overall_and_fixed512':blocks,'CSV_columns':list(MOMENT_CSV_FIELDS),'recorded_rows':len(records),
      'interpretation':'Recorded postclip gradient/movement magnitudes only; movement includes AdamW decay. No event/partition/learning trust check is rerun; coordinate units prevent branch importance or dominance inference.'}


def actual_moment_descriptors(inputs,update):
    identity='v12_actual_update_'+str(update)
    read=lambda name,shape=None,dtype=None:inputs.tensor(identity,name,shape=shape,dtype=dtype)
    require(read('update',dtype='LongStorage')==[update],'Exact captured successful update')
    attempt=read('attempt',dtype='LongStorage');factor=read('clip_factor',dtype='DoubleStorage')
    positions_tensor=inputs.archive(identity).field('temporal_patch_positions')
    require(isinstance(positions_tensor,Tensor) and len(positions_tensor.shape)==2,'Captured packed positions rank')
    R,T=positions_tensor.shape
    require(0<=R<=24 and 0<=T<=16,'Captured bounded packed geometry')
    positions=read('temporal_patch_positions',(R,T),'LongStorage')
    valid=boolean(read('temporal_patch_valid',(R,T),'BoolStorage'),'packed')
    packed_rows=read('temporal_patch_rows',(R,),'LongStorage')
    weight=read('temporal_export_weight_before',(32,64),'FloatStorage')
    bias=read('temporal_export_bias_before',(32,),'FloatStorage')
    routes=[]
    for route,prefix in (('ordinary','temporal_'),('contextual','temporal_contextual_')):
        bin_valid=boolean(read(prefix+'pool_bin_valid',(R,4),'BoolStorage'),'bin')
        means=read(prefix+'pool_bin_mean',(R,4,64),'FloatStorage')
        variances=read(prefix+'pool_bin_variance',(R,4,64),'FloatStorage')
        mixture=read(prefix+'pool_mixture_weights',(R,4,T),'FloatStorage')
        uniform=read(prefix+'pool_dispersion_weights',(R,4,T),'FloatStorage')
        centers=read(prefix+'pool_dispersion_mean',(R,4,64),'FloatStorage')
        defined=read(prefix+'pool_projected_upstream_defined',(), 'LongStorage')
        upstream=read(prefix+'pool_projected_upstream_pre_clip',(R,32),'FloatStorage')
        require(defined in ([0],[1]),'Explicit saved used/unused route flag')
        require(all(math.isfinite(x) for x in means+variances+mixture+uniform+centers+upstream+weight+bias),
                'Finite selected saved moment/export values')
        bins=[]
        for j in range(4):
            present=[r for r in range(R) if bin_valid[r*4+j]]
            local_counts=[sum(valid[r*T+t] and 4*j<=positions[r*T+t]<4*(j+1) for t in range(T))
                          for r in range(R)]
            sum_squares=[];effective=[];maximum=[]
            uniform_squares=[];uniform_effective=[];uniform_maximum=[]
            for r in present:
                weights=[mixture[(r*4+j)*T+t] for t in range(T) if valid[r*T+t]]
                squared=math.fsum(x*x for x in weights)
                sum_squares.append(squared)
                require(squared>0,'Present captured mixture must have positive concentration')
                effective.append(1./squared)
                maximum.append(max(weights) if weights else 0.)
                uniform_weights=[uniform[(r*4+j)*T+t] for t in range(T) if valid[r*T+t]]
                uniform_squared=math.fsum(x*x for x in uniform_weights)
                require(uniform_squared>0,'Present captured uniform weights must have positive concentration')
                uniform_squares.append(uniform_squared)
                uniform_effective.append(1./uniform_squared)
                uniform_maximum.append(max(uniform_weights) if uniform_weights else 0.)
            center_gap=[means[(r*4+j)*64+i]-centers[(r*4+j)*64+i]
                        for r in present for i in range(64)]
            COUNTERS['captured_center_gap_scalar_subtractions']+=len(center_gap)
            exports=[]
            for half,moment in enumerate((means,variances)):
                values=[]
                for r in range(R):
                    for d in range(4):
                        export_row=j*8+half*4+d
                        value=bias[export_row]+math.fsum(moment[(r*4+j)*64+i]*weight[export_row*64+i]
                                                         for i in range(64))
                        values.append(value if bin_valid[r*4+j] else 0.)
                record_affine('temporal_mean4_export' if half==0 else 'temporal_variance4_export',R*4,64)
                exports.append(values)
            bins.append({'bin':j,'present_local_rows':len(present),'local_member_counts':local_counts,
              'mean64_quantiles':quantiles([means[(r*4+j)*64+i] for r in present for i in range(64)]),
              'centered_variance64_quantiles':quantiles([variances[(r*4+j)*64+i] for r in present for i in range(64)]),
              'mixture_sum_squares_quantiles':quantiles(sum_squares),
              'mixture_effective_weight_count_quantiles':quantiles(effective),
              'mixture_max_weight_quantiles':quantiles(maximum),
              'uniform_sum_squares_quantiles':quantiles(uniform_squares),
              'uniform_effective_weight_count_quantiles':quantiles(uniform_effective),
              'uniform_max_weight_quantiles':quantiles(uniform_maximum),
              'uniform_own_center64_quantiles':quantiles([centers[(r*4+j)*64+i] for r in present for i in range(64)]),
              'captured_attention_mean_minus_uniform_center64_quantiles':quantiles(center_gap),
              'concentration_semantics':'mixture_* uses captured attention-mean weights; uniform_* uses captured mask-only dispersion weights; valid-token support, present local-bin rows; no renormalization',
              'manual_export_mean4_values':exports[0],'manual_export_variance4_values':exports[1],
              'manual_export_mean4_quantiles':quantiles([exports[0][r*4+i] for r in present for i in range(4)]),
              'manual_export_variance4_quantiles':quantiles([exports[1][r*4+i] for r in present for i in range(4)]),
              'PRECLIP_upstream_mean4_quantiles':quantiles([upstream[r*32+j*8+i] for r in present for i in range(4)]),
              'PRECLIP_upstream_variance4_quantiles':quantiles([upstream[r*32+j*8+4+i] for r in present for i in range(4)])})
        routes.append({'route':route,'upstream_defined':bool(defined[0]),'bins':bins})
    return {'update':update,'attempt':attempt[0],'pre_step_completed_updates':update-1,'clip_factor':factor[0],
      'packed_rows':packed_rows,'packed_positions':positions,'packed_valid':valid,'geometry':{'R':R,'T':T},
      'routes':routes,'interpretation':'Captured same-forward moment/export allocation. Attention mean and uniform OWN dispersion center remain distinct. Center gap subtracts captured arrays only; no hidden, center, variance, score or adjoint reconstruction, no renormalization. All routes retained, present-bin descriptors only; allR affine rows then post-bias empty-bin zeros. PRECLIP upstream magnitudes/defined flag descriptive, no global-norm/optimizer reconstruction. Update2048 is pre-step state2047, never paired to final2048 decoder.'}


def scalar_reference_comparison(stages,cached):
    readouts=[];blocks=[];reconstruction=[]
    for stage in STAGES:
        old,new=cached['stages'][stage],stages[stage]
        for task in TASKS:
            for head in ('linear','neural'):
                for metric in ('r2','mae','rmse','coverage'):
                    a=old['per_task_mean_across_saved_repetitions'][task][head][metric]
                    b=new['per_task_mean_across_saved_repetitions'][task][head][metric]
                    readouts.append({'stage':stage,'task':task,'head':head,'metric':metric,
                      'cached_V11':a,'new_V12':b,'V12_minus_cached_V11':b-a if a is not None and b is not None else None})
        for name,_,_ in COMMON_BLOCKS+(('complete40',0,40),):
            for metric in ('covariance_trace','covariance_frobenius_squared','participation_ratio'):
                a=old['native_features']['blocks'][name][metric];b=new['native_features']['blocks'][name][metric]
                blocks.append({'stage':stage,'block':name,'metric':metric,'cached_V11':a,'new_V12':b,
                  'V12_minus_cached_V11':b-a if a is not None and b is not None else None})
        for metric in ('standardized_fixed_query_MAE','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64'):
            a=old['fixed_query_TRAIN_diagnosis']['all112_TRAIN'][metric]
            b=new['fixed_query_TRAIN_diagnosis']['all112_TRAIN'][metric]
            reconstruction.append({'stage':stage,'metric':metric,'cached_V11':a,'new_V12':b,'V12_minus_cached_V11':b-a})
    return {'saved_TRAIN_readout_scalar_differences':readouts,'native_block_descriptor_differences':blocks,
      'full112_reconstruction_scalar_differences':reconstruction,
      'limits':'Subtraction of already cached V11 scalars only; no old V11 reductions/refits or latent-basis alignment. V11 moment CSV/witness objects are cached opaque. No V11 uniform-center/gap capture exists. Missing packed global anchor IDs prevent paired-input counterfactual claims; current moment descriptions provide no validation cause.'}


def diagnose(card,inputs):
    rows=train_ids(inputs.tensor('train_targets','row_ids',shape=(112,),dtype='LongStorage'))
    groups=inputs.tensor('train_targets','group_ids',shape=(112,),dtype='LongStorage')
    targets=inputs.tensor('train_targets','targets',shape=(112,4))
    valid=boolean(inputs.tensor('train_targets','valid',shape=(112,4)),'task targets')
    cached=inputs.cached()
    vi=VersionInputs(inputs);stages={};witnesses={}
    for stage in STAGES:
        native_report,native,feature_valid=features(vi,stage,rows,targets,valid)
        query_report,witness=query(vi,stage,rows)
        metadata=checkpoint_metadata(vi,stage)
        heads=head_scores(vi,stage,rows,groups,targets,valid,native,feature_valid)
        augment_head_halves(heads,targets)
        means={task:{head:{metric:mean([rep['tasks'][task][head][metric] for rep in heads
          if rep['tasks'][task][head][metric] is not None]) for metric in ('r2','mae','rmse','coverage')}
          for head in ('linear','neural')} for task in TASKS}
        stages[stage]={'checkpoint_provenance':metadata,'native_features':native_report,
          'saved_TRAIN_head_scores':heads,'per_task_mean_across_saved_repetitions':means,
          'fixed_query_TRAIN_diagnosis':query_report,
          'saved_mean_residual_allocation':residual_allocation(vi,witness,stage),
          'reconstruction_readout_associations':reconstruction_associations(heads,query_report,targets)}
        witnesses[stage]=witness
    require(witnesses['initial']['row_ids']==witnesses['final']['row_ids'] and
      witnesses['initial']['support']==witnesses['final']['support'] and
      witnesses['initial']['target']==witnesses['final']['target'],'Same fixed first8 TRAIN query capture plan')
    moments={'csv':moment_csv(inputs),
      'actual_witnesses':[actual_moment_descriptors(inputs,update) for update in (1,2048)],
      'limits':'V11 moment objects are reused only as opaque cache; no old re-quantiling or moment arithmetic. Missing global anchor IDs prevent matched-batch claims. No V11 own-uniform-center capture or fabricated comparison. Gradient/movement descriptors are coordinate dependent, not importance, branch dominance, alternate clipping or a task-level causal explanation.'}
    comparison=scalar_reference_comparison(stages,cached)
    return {'kind':'cpb_v11_v12_TRAIN_saved_information_diagnosis_v1','status':'passed_readonly_saved_TRAIN_diagnosis',
      'baseline':'CPB-v3.market-01','new_encoder_experiment':False,'candidate_declared':False,
      'dataset_id':'MKT-HIST-2','designed_complexity_level':4,'complexity_scale_max':5,
      'dataset_legend':'Joint BTC/ETH/SOL minute histories C3/H128/F9;112 chronological TRAIN windows, past-window information retention; complexity4/5 ordinal. One historical trajectory perversion; initial0/final2048. No VALIDATION/TEST/stress/TEMPO numeric input.',
      'models':{'v12':{'stages':stages}},'cached_v11_TRAIN':cached,
      'cached_V11_numeric_selectors':list(CACHE_SELECTORS),'V11_new_numerical_reductions':0,
      'cached_V11_independent_audit_reused':inputs.context_bindings['old_audit'],
      'temporal_moment_observations':moments,'comparison':comparison,
      'manual_arithmetic_counters':COUNTERS,
      'manual_arithmetic_counter_units':{'dot_outputs':'One scalar dot-product output from saved arrays, including allR masked export rows computed then zeroed.',
        'multiply_add_terms':'Scalar pair products entering disclosed saved dot contractions; bias additions excluded.',
        'scalar_summary_function_calls':'Instrumented mean/quantiles/variance/pearson/score function invocations; implementation-dependent, not primitive FLOPs.',
        'captured_center_gap_scalar_subtractions':'One captured attention-mean minus captured own-uniform-center scalar coordinate for each present route/bin row; not a reconstruction or module call.'},
      'model_calls':0,'encoder_calls':0,'decoder_calls':0,'head_calls':0,'module_forwards':0,
      'neural_forwards':0,'fits':0,'head_fits':0,'training_updates':0,'held_out_tensor_reads':0,
      'old_V11_raw_numeric_reads':0,'old_V9_numeric_reads':0,'old_V8_numeric_reads':0,'local_adjoint_rechecks':0,
      'learning_gauge_Hadamard_trust_rechecks':0,'new_quality_scores':False,
      'diagnostic_gate_or_next_encoder_selected':False,'application_changed':False,
      'limits':['TRAIN predictions are in-sample saved readouts, not generalization or3independent encoder runs.',
        'No PCA/eigensolver, fitted transform, new representation, refit, ablation predictor or coefficient/budget rescue.',
        'All112 rows have no saved intermediate moments;2actual batches cannot trace per-task moment contribution.',
        'Captured moments/affine descriptors and TRAIN residual associations do not explain validation gains causally.',
        'Existing V12 causal/learning/typedresume/local-adjoint proofs are reused unchanged, never replayed.']}


def write_exclusive(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card',required=True,type=Path)
    parser.add_argument('--card-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    require(sys.platform.startswith('linux') and Path('/.dockerenv').is_file(),'Existing managed Linux container only')
    root=Path('/opt/cuwacunu/research/causal-v11-v12-train-diagnosis')
    require(args.output.is_absolute() and args.output.resolve().is_relative_to(root) and
      args.output.resolve()!=root and not args.output.exists(),'Exclusive bounded new diagnosis output')
    card_pin=file_pin(args.card)
    require(card_pin['sha256']==args.card_sha256,'Externally admitted exact actual diagnosiscard hash')
    with args.card.open(encoding='utf-8-sig') as stream:card=json.load(stream)
    own=Path(__file__).resolve().parent
    source_before={name:file_pin(own/name) for name in ('diagnose.py','archive_reader.py','schema.py','README.md','provenance.json')}
    source_rows=card.get('owned_source_files',[])
    require(len(source_rows)==5 and all(any(same_pin(row,pin) for row in source_rows)
      for pin in source_before.values()),'Exact actual frozen5owned source bytes before numericalbody')
    args.output.mkdir(parents=True)
    started=time.monotonic();inputs=None
    try:
        inputs=Inputs(card)
        result=diagnose(card,inputs)
        for row in card['allowed_inputs']+card['hash_only_guards']+card['context_inputs']+card.get('preservation_pins',[])+[card['pre_execution_preservation']]:
            inputs.verify(row)
        require(file_pin(args.card)==card_pin and all(file_pin(own/name)==pin for name,pin in source_before.items()),
          'Actual card/source unchanged throughout saved arithmetic')
        result.update({'card':card_pin,'owned_source_pins':source_before,'all_owned_source_files_unchanged':True,
          'verified_allowed_inputs':inputs.verified,'verified_hash_only_guards':inputs.hash_guards,
          'verified_context_inputs':[file_pin(row['container_path']) for row in card['context_inputs']],
          'verified_pre_execution_preservation':inputs.pre_execution_preservation,
          'opened_allowed_ids':inputs.opened,'numeric_fields_accessed':inputs.numeric_fields,
          'elapsed_seconds':time.monotonic()-started})
        write_exclusive(args.output/'diagnosis.json',result)
        print(json.dumps({'status':result['status'],'output':str(args.output/'diagnosis.json'),
          'sha256':sha256(args.output/'diagnosis.json'),'new_V12_stages':2,'cached_V11_selectors':4,
          'old_encoder_or_head_calls':0,'manual_saved_affine_contractions':COUNTERS['manual_saved_affine_contractions']}))
    except Exception as error:
        write_exclusive(args.output/'failure.json',{'kind':'cpb_v11_v12_TRAIN_diagnosis_refusal_v1',
          'status':'failed','reason':str(error),'model_calls':0,'head_fits':0,'training_updates':0})
        raise
    finally:
        if inputs is not None:inputs.close()


if __name__=='__main__':main()

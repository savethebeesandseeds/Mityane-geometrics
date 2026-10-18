# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Owned inert metadata-schema reader; no tensor storage values or main import.

SOURCE PREPARATION ONLY. Later metadata invocation needs separate actual-card
review and managed-container admission. Exact field schemas are physical copies,
not an import of the diagnosis algorithm or an older implementation.
"""
import hashlib
import json
from pathlib import Path
import sys
from archive_reader import Archive, Tensor

def require(condition,message):
    if not condition:raise ValueError(message)
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




if not Path('/.dockerenv').is_file():
    raise RuntimeError('Existing managed container only')
card_path=Path(sys.argv[1]);external=sys.argv[2]
require(hashlib.sha256(card_path.read_bytes()).hexdigest()==external,'Externally reviewed actual card hash')
card=json.loads(card_path.read_text(encoding='utf-8-sig'))
require(card.get('kind')=='cpb_v11_v12_saved_train_diagnosis_card_v1' and
 card.get('stage')=='prospective_before_diagnostic_arithmetic' and
 card.get('computed_versions')==['v12'] and card.get('cached_reference_versions')==['v11'] and
 card.get('execution_admitted') is True, 'Exact admitted diagnosis scope')
require(card.get('fixed_update_blocks')==[list(x) for x in UPDATE_BLOCKS] and
 card.get('cached_v11_selectors')==list(CACHE_SELECTORS),'Fixed blocks/closed cache selectors')
require(card.get('baseline')=='CPB-v3.market-01' and card.get('train_rows')==112 and
 card.get('query_rows')==8 and card.get('train_begin_ms')==TRAIN_BEGIN and card.get('train_end_ms')==TRAIN_END,
 'Exact TRAIN population and baseline')
require(card.get('manual_saved_affine_contractions_declared') is True and
 card.get('module_forwards_allowed') is False and card.get('new_quality_or_selection_allowed') is False,
 'Declared saved arithmetic only')
rows=card.get('allowed_inputs',[])
require(len(rows)==25 and {r['id'] for r in rows}==ALLOWED_IDS,'Exactly25 selected roles')
guards=card.get('hash_only_guards',[])
require(len(guards)==35 and {r['id'] for r in guards}==HASH_GUARD_IDS,'Exactly35 hash-only guards')
contexts=card.get('context_inputs',[])
require(len(contexts)==14 and {r['id'] for r in contexts}==CONTEXT_IDS,'14 actual final-context inputs')
# ALL data/context bytes verified before any archive metadata is decoded.
for row in rows+guards+contexts+card.get('preservation_pins',[])+[card['pre_execution_preservation']]:
    path=Path(row.get('container_path',row.get('path','')))
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),'Physical explicit input')
    require(path.stat().st_size==row['bytes'],'Input length')
    with path.open('rb') as stream:
        require(hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256'],'Input SHA256')
context={row['id']:json.loads(Path(row['container_path']).read_text(encoding='utf-8-sig')) for row in contexts}
assertions=card.get('context_assertions',[])
require(isinstance(assertions,list) and len(assertions)==12,'Explicit12 context assertion array')
for row,expected in zip(assertions,CONTEXT_ASSERTIONS):
    require(row==expected and type(row['equals']) is type(expected['equals']),'Exact typed context assertion')
    actual=context[row['input']]
    for key in row['path']:actual=actual[key]
    require(actual==row['equals'] and type(actual) is type(row['equals']),'Actual metadata context assertion')
out={}
for row in rows:
    identity=row['id']
    require(row['role']==role_for(identity),'Exact role')
    if identity.endswith('_query_summary'):
        require(row.get('numeric_json_selectors')==['train'],'Only TRAIN JSON selector')
    if identity==CACHE_ID:
        require(row.get('numeric_json_selectors')==list(CACHE_SELECTORS),'Exact four cached V11 subtrees')
    if identity=='v12_temporal_moment_csv':
        require(row.get('allowed_CSV_columns')==list(MOMENT_CSV_FIELDS),'Exact11 CSV columns')
    if row['role'] not in ('TRAIN_archive','shared_TRAIN_only_targets'):continue
    selected=row.get('allowed_tensor_fields',[])
    require(len(selected)==len(set(selected)) and set(selected)==expected_fields(identity), 'Exact unique field allowlist')
    archive=Archive(row['container_path'])
    try:
        fields={}
        for name in selected:
            tensor=archive.field(*name.split('.'))
            require(isinstance(tensor,Tensor),'Inert tensor metadata')
            fields[name]={'shape':list(tensor.shape),'dtype':tensor.storage.dtype}
        out[identity]=fields
    finally:archive.close()
print(json.dumps({'kind':'cpb_v11_v12_TRAIN_metadata_schema_v1','status':'passed_metadata_only',
 'numeric_roles':25,'hash_only_guards':35,'context_inputs':14,'context_assertions':12,'numeric_storage_reads':0,
 'model_calls':0,'head_fits':0,'training_updates':0,'fields':out},indent=2))

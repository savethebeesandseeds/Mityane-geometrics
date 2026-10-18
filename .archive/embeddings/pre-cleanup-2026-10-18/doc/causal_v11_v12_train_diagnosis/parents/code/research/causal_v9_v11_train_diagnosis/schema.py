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
WIDTHS, BLOCKS = {'v11':40}, {'v11':COMMON_BLOCKS}
HEAD_SEEDS_U64 = {'v11':(6049515622977958931,8262941649267994858,2095371932330426401)}
OBJECTIVE_POLICY = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
GRADIENT_POLICY = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
DIRECT_ROUTES = 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
NATIVE_LAYOUT = ('parent32[0:32];mean4[32:36];uncertainty4[36:40];'
 'independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;variance-reads-live-uncertainty4-only')
POOL_POLICY = 'original-position-four-bin-half-mixture-mean4-centered-variance4-v1'
MOMENT_POLICY = 'componentwise-centered-variance-two-sum-mean-live-attention-v1'
EXPORT_LAYOUT = 'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'
EMPTY_POLICY = 'zero-whole-export-slice-after-bias-v1'
CACHE_ID = 'cached_v9_prior_TRAIN_diagnosis'
CACHE_SELECTORS = ('models.v9.stages.initial','models.v9.stages.final',
                   'actual_pre_step_batch_diagnosis.v9','temporal_learning_observations.v9')
BASE_IDS = {'train_targets',CACHE_ID,'v11_temporal_moment_csv','v11_actual_update_1','v11_actual_update_2048'}
ALLOWED_IDS = BASE_IDS | {'v11_'+stage+'_'+kind for stage in STAGES
 for kind in ('features','query','query_summary','checkpoint')} | {
 'v11_'+stage+'_'+kind+'_'+str(rep) for stage in STAGES for kind in ('head','fit') for rep in REPS}
V9_GUARD_STEMS = ('initial_features','initial_query','initial_query_summary','initial_checkpoint',
 'final_features','final_query','final_query_summary','final_checkpoint','actual_update_1','actual_update_2048',
 'actual_bindings','training_csv','temporal_learning_receipt','temporal_learning_csv','temporal_gauge_csv','temporal_between_csv')
V9_GUARD_IDS = {'v9_'+name for name in V9_GUARD_STEMS} | {
 'v9_'+stage+'_'+kind+'_'+str(rep) for stage in STAGES for kind in ('head','fit') for rep in REPS}
V11_GUARD_IDS = {'v11_'+name for name in ('actual_bindings','training_csv','temporal_learning_receipt',
 'temporal_learning_csv','temporal_gauge_csv','temporal_between_csv')}
HASH_GUARD_IDS = V9_GUARD_IDS | V11_GUARD_IDS
CONTEXT_IDS = {'old_card','old_audit','old_main_mirror','old_audit_mirror','v11_training_audit',
 'v11_final_establishment','v11_final_preservation','current_goal_protocol','retention_independent_review',
 'retention_application_admission','current_ledger','current_catalog'}
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
 'pool_mixture_weights','pool_bin_valid','pool_bin_mean','pool_bin_variance',
 'pool_projected_upstream_defined','pool_projected_upstream_pre_clip')}
MOMENT_CSV_FIELDS = ('update','bin','moment','export_row_begin','export_row_end','weight_gradient_l2',
 'bias_gradient_l2','weight_movement_l2','bias_movement_l2','nonzero_export_gradient_values','moved_export_values')
UPDATE_BLOCKS = ((1,512),(513,1024),(1025,1536),(1537,2048))
QUANTILES = (0.,.01,.05,.25,.5,.75,.95,.99,1.)

def expected_fields(identity):
    require(identity in ALLOWED_IDS,'Unknown numeric input identity')
    if identity=='train_targets': return {'row_ids','group_ids','targets','valid'}
    if identity in ('v11_actual_update_1','v11_actual_update_2048'): return ACTUAL_FIELDS
    if identity.endswith('_features'): return {'row_ids','values','valid'}
    if identity.endswith('_query'): return QUERY_FIELDS
    if identity.endswith('_checkpoint'): return CHECKPOINT_FIELDS
    if '_head_' in identity:
        return {'row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions'}
    if '_fit_' in identity:
        return {'feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed'}
    return set()


def role_for(identity):
    if identity==CACHE_ID:return 'audited_prior_TRAIN_output_selected_V9_subtrees_only'
    if identity=='v11_temporal_moment_csv':return 'V11_TRAIN_moment_subgroup_16384_rows'
    if identity.endswith('_query_summary'):return 'aggregate_JSON_TRAIN_key_only'
    return 'TRAIN_archive'




if not Path('/.dockerenv').is_file():
    raise RuntimeError('Existing managed container only')
card_path=Path(sys.argv[1]);external=sys.argv[2]
require(hashlib.sha256(card_path.read_bytes()).hexdigest()==external,'Externally reviewed actual card hash')
card=json.loads(card_path.read_text(encoding='utf-8-sig'))
require(card.get('kind')=='cpb_v9_v11_saved_train_diagnosis_card_v1' and
 card.get('stage')=='prospective_before_diagnostic_arithmetic' and
 card.get('computed_versions')==['v11'] and card.get('cached_reference_versions')==['v9'], 'Exact diagnosis scope')
rows=card.get('allowed_inputs',[])
require(len(rows)==25 and {r['id'] for r in rows}==ALLOWED_IDS,'Exactly25 selected roles')
guards=card.get('hash_only_guards',[])
require(len(guards)==34 and {r['id'] for r in guards}==HASH_GUARD_IDS,'Exactly34 hash-only guards')
contexts=card.get('context_inputs',[])
require(len(contexts)==12 and {r['id'] for r in contexts}==CONTEXT_IDS,'12 actual final-context inputs')
# ALL data/context bytes verified before any archive metadata is decoded.
for row in rows+guards+contexts+card.get('preservation_pins',[])+[card['pre_execution_preservation']]:
    path=Path(row.get('container_path',row.get('path','')))
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),'Physical explicit input')
    require(path.stat().st_size==row['bytes'],'Input length')
    with path.open('rb') as stream:
        require(hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256'],'Input SHA256')
out={}
for row in rows:
    identity=row['id']
    require(row['role']==role_for(identity),'Exact role')
    if row['role']!='TRAIN_archive':continue
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
print(json.dumps({'kind':'cpb_v9_v11_TRAIN_metadata_schema_v1','status':'passed_metadata_only',
 'numeric_roles':25,'hash_only_guards':34,'context_inputs':12,'numeric_storage_reads':0,
 'model_calls':0,'head_fits':0,'training_updates':0,'fields':out},indent=2))
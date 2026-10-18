#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent V11 saved-TRAIN descriptors and exact cached V9 carry-forward.

Owned stdlib scalar implementation. Never imports main diagnosis or old project
algorithms and never executes a model/head, neural weights, fitting or updates.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from archive_reader import Archive, Tensor
from component_math import reduce_components, describe_moment_csv, describe_moment_witness

MAIN_CARD_KIND = 'cpb_v9_v11_saved_train_diagnosis_card_v1'
TASKS = ('past_volatility','past_trend','past_activity','past_relation')
REPS = (2701,2802,2903)
BLOCKS = (('learned20',0,20),('fixed_spectral12',20,32),('mean4',32,36),('uncertainty4',36,40))
QUANTILES = (0.,.01,.05,.25,.5,.75,.95,.99,1.)
POLICY = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
OBJECTIVE = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
DIRECT_ROUTES = 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
NATIVE_LAYOUT = ('parent32[0:32];mean4[32:36];uncertainty4[36:40];'
                'independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;'
                'variance-reads-live-uncertainty4-only')
TEMPORAL_POLICY = 'original-position-four-bin-half-mixture-mean4-centered-variance4-v1'
MOMENT_POLICY = 'componentwise-centered-variance-two-sum-mean-live-attention-v1'
EXPORT_LAYOUT = 'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'
MOMENT_COLUMNS = ('update','bin','moment','export_row_begin','export_row_end',
                 'weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2',
                 'nonzero_export_gradient_values','moved_export_values')
CACHED_SELECTORS = ('models.v9.stages.initial','models.v9.stages.final',
                    'actual_pre_step_batch_diagnosis.v9','temporal_learning_observations.v9')
ALLOWED_IDS = {'train_targets','cached_v9_prior_TRAIN_diagnosis','v11_actual_update_1',
               'v11_actual_update_2048','v11_temporal_moment_csv'} | {
                  'v11_'+stage+'_'+kind for stage in ('initial','final')
                  for kind in ('features','query','query_summary','checkpoint')} | {
                  'v11_'+stage+'_'+kind+'_'+str(rep) for stage in ('initial','final')
                  for kind in ('head','fit') for rep in REPS}
CONTEXT_IDS = {'old_card','old_audit','old_main_mirror','old_audit_mirror',
               'v11_training_audit','v11_final_establishment','v11_final_preservation',
               'current_goal_protocol','retention_independent_review',
               'retention_application_admission','current_ledger','current_catalog'}
HASH_ONLY_IDS = {'v9_'+stage+'_'+kind for stage in ('initial','final')
                 for kind in ('features','query','query_summary','checkpoint')} | {
                 'v9_'+stage+'_'+kind+'_'+str(rep) for stage in ('initial','final')
                 for kind in ('head','fit') for rep in REPS} | {
                 'v9_'+kind for kind in ('actual_update_1','actual_update_2048','actual_bindings','training_csv',
                 'temporal_learning_receipt','temporal_learning_csv','temporal_gauge_csv','temporal_between_csv')} | {
                 'v11_'+kind for kind in ('actual_bindings','training_csv','temporal_learning_receipt',
                 'temporal_learning_csv','temporal_gauge_csv','temporal_between_csv')}
OLD_DIAGNOSIS_SHA = '805d80db8f260e59a86b1b0184fe54fb3e28659e642f09f6e4e8ab99aa2a36cf'
OLD_AUDIT_SHA = 'c0c8d5c93e95d24c8a79f3e1d6a94b597a4ac853407c38aaeea27235715d8da9'
OLD_CARD_SHA = 'ceb4c2648a2e4b3d594b2028ae91eab6a9a626f1846a1d11b2b9b0bd1f4cd950'

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def pin(path):
    p=Path(path)
    if not p.is_absolute() or not p.is_file() or p.is_symlink():raise ValueError('Explicit absolute physical file pin required')
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}

def average(x):return math.fsum(x)/len(x) if x else None
def var(x):
    m=average(x);return average([(a-m)**2 for a in x]) if x else None
def norm(x):return math.sqrt(math.fsum(a*a for a in x))
def pearson(x,y):
    if len(x)<2:return None
    xm,ym=average(x),average(y)
    a=[v-xm for v in x];b=[v-ym for v in y]
    denominator=norm(a)*norm(b)
    return max(-1.,min(1.,math.fsum(p*q for p,q in zip(a,b))/denominator)) if denominator else None
def quantile(x):
    if not x:return None
    x=sorted(x);out={}
    for q in QUANTILES:
        i=q*(len(x)-1);lo=int(math.floor(i));hi=int(math.ceil(i))
        out[str(q)]=x[lo]+(i-lo)*(x[hi]-x[lo])
    return out
def scores(p,y,ok):
    pairs=[(x,t) for x,t,v in zip(p,y,ok) if v]
    if not pairs:return {'total':len(ok),'valid':0,'coverage':0.,'r2':None,'mae':None,'rmse':None}
    center=average([t for _,t in pairs]);errors=[x-t for x,t in pairs]
    sse=math.fsum(e*e for e in errors);sst=math.fsum((t-center)**2 for _,t in pairs)
    return {'total':len(ok),'valid':len(pairs),'coverage':len(pairs)/len(ok),
            'r2':1-sse/sst if sst else None,'mae':average([abs(e) for e in errors]),'rmse':math.sqrt(sse/len(pairs))}
def dimension_seed(rep,width):
    mask=(1<<64)-1;x=(rep+0x9e3779b97f4a7c15*width)&mask
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask;x=((x^(x>>27))*0x94d049bb133111eb)&mask
    return x^(x>>31)

def independent_allowed_fields(identity):
    if identity in ('v11_actual_update_1','v11_actual_update_2048'):
        return {'update','attempt','clip_factor','temporal_export_weight_before',
                'temporal_export_bias_before','temporal_patch_positions',
                'temporal_patch_valid','temporal_patch_rows'} | {
                    prefix+'_'+field for prefix in ('temporal_pool','temporal_contextual_pool')
                    for field in ('mixture_weights','bin_valid','bin_mean','bin_variance',
                                  'projected_upstream_defined','projected_upstream_pre_clip')}
    if identity=='train_targets':return {'row_ids','group_ids','targets','valid'}
    if identity.endswith('_features'):return {'row_ids','values','valid'}
    if identity.endswith('_query'):
        return {'row_ids','target_mask','targets','mean','variance','native','row_mae','row_nll',
                'row_mean_huber','row_variance_nll','row_total_objective'}
    if '_head_' in identity:
        return {'row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions'}
    if '_fit_' in identity:
        return {'feature_mean','feature_scale','ridge_weights','target_scale','target_mean',
                'ridge_intercept','actual_seed'}
    if identity.endswith('_checkpoint'):
        fields={'artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates',
                'causal_updates','initialization_seed','channel_ids','payload.model.encoder.change_decoder'}
        double={'gradient_policy','native_layout','mean_width','uncertainty_width','native_width',
                'objective_policy','mean_huber_delta_f64','loss_coefficients_f64',
                'variance_error_detached','direct_gradient_routes','reporting_nll_backpropagated',
                'temporal_pooling_policy','temporal_pooling_partition_v1','temporal_pooling_mix_f64',
                'empty_local_bin_policy','temporal_moment_policy','temporal_export_layout',
                'temporal_moment_partition_v1'}
        return fields | double | {'payload.'+field for field in double}
    if identity.endswith('_query_summary') or identity in ('cached_v9_prior_TRAIN_diagnosis','v11_temporal_moment_csv'):
        return set()
    raise ValueError('Unknown independently closed numeric role '+identity)

def feature_audit(a,identity,report,ids,target,target_valid):
    width=40
    block_specs=BLOCKS
    x=a.array(identity,'values',(112,width));valid=a.array(identity,'valid',(112,))
    a.ck(a.array(identity,'row_ids',(112,))==ids,'Feature exact TRAIN association')
    a.ck(all(valid) and all(math.isfinite(v) for v in x),'112 finite TRAIN native rows')
    columns=[[x[r*width+d] for r in range(112)] for d in range(width)]
    for d,col in enumerate(columns):
        expected={'coordinate':d,'block':next(n for n,l,h in block_specs if l<=d<h),
            'mean':average(col),'population_std':math.sqrt(var(col)),'minimum':min(col),'maximum':max(col),
            'pearson_to_TRAIN_targets':{task:pearson([col[r] for r in range(112) if target_valid[r*4+t]],
                [target[r*4+t] for r in range(112) if target_valid[r*4+t]]) for t,task in enumerate(TASKS)}}
        a.tree(report['coordinates'][d],expected,'Coordinate '+identity+str(d))
    for name,lo,hi in block_specs+((f'complete{width}',0,width),):
        selected=columns[lo:hi];center=[average(c) for c in selected]
        covariance=[[math.fsum((u-center[i])*(v-center[j]) for u,v in zip(c,d))/112
                     for j,d in enumerate(selected)] for i,c in enumerate(selected)]
        trace=math.fsum(covariance[d][d] for d in range(hi-lo))
        squared=math.fsum(v*v for row in covariance for v in row)
        a.tree(report['blocks'][name],{'width':hi-lo,'valid_rows':112,'covariance_trace':trace,
            'covariance_frobenius_squared':squared,'participation_ratio':trace*trace/squared if squared else None},'Block '+identity+name)
    return x,valid,columns

def head_audit(a,prefix,report,x,feature_valid,ids,groups,target,target_valid):
    width=40
    block_specs=BLOCKS
    expected_reps=[]
    for index,rep in enumerate(REPS):
        hid=prefix+'_head_'+str(rep);fid=prefix+'_fit_'+str(rep)
        p={k:a.array(hid,k,(112,4)) for k in ('targets','target_valid','valid','ridge_predictions','neural_predictions')}
        a.ck(a.array(hid,'row_ids',(112,))==ids and a.array(hid,'group_ids',(112,))==groups,'Head row/group association')
        a.ck(p['targets']==target and p['target_valid']==target_valid,'Exact original TRAIN targets')
        a.ck(p['valid']==[bool(feature_valid[r] and target_valid[r*4+t]) for r in range(112) for t in range(4)],'Exact head eligibility')
        f={k:a.array(fid,k,shape) for k,shape in
           (('feature_mean',(width,)),('feature_scale',(width,)),('ridge_weights',(width,4)),('target_scale',(4,)),
            ('target_mean',(4,)),('ridge_intercept',(4,)),('actual_seed',()))}
        a.ck(f['actual_seed'][0]%(1<<64)==dimension_seed(rep,width),'Exact fixed dimension-dependent seed')
        for d in range(width):
            column=[x[r*width+d] for r in range(112) if feature_valid[r]]
            a.close(f['feature_mean'][d],average(column),'Saved TRAIN feature mean')
            a.close(f['feature_scale'][d],max(1e-8,math.sqrt(var(column))),'Saved TRAIN feature scale')
        for t in range(4):
            col=[target[r*4+t] for r in range(112) if target_valid[r*4+t]]
            a.close(f['target_mean'][t],average(col),'Saved TRAIN target mean')
            a.close(f['target_scale'][t],max(1e-8,math.sqrt(var(col))),'Saved TRAIN target scale')
        reconstructed=[]
        for r in range(112):
            for t in range(4):
                raw=math.fsum((x[r*width+d]-f['feature_mean'][d])/f['feature_scale'][d]*f['ridge_weights'][d*4+t] for d in range(width))
                reconstructed.append(f['target_mean'][t]+f['target_scale'][t]*(raw+f['ridge_intercept'][t]))
        a.affine('ridge_total',112*4,112*4*40)
        a.affine('ridge_block',112*4*len(BLOCKS),112*4*40)
        for got,want,ok in zip(p['ridge_predictions'],reconstructed,p['valid']):
            if ok:a.close(got,want,'Saved ridge reconstruction')
        task_values={task:{head:scores(p[field][t::4],target[t::4],p['valid'][t::4])
                     for head,field in (('linear','ridge_predictions'),('neural','neural_predictions'))} for t,task in enumerate(TASKS)}
        halves={label:{task:{head:scores(p[field][lo*4+t:hi*4:4],target[lo*4+t:hi*4:4],p['valid'][lo*4+t:hi*4:4])
                    for head,field in (('linear','ridge_predictions'),('neural','neural_predictions'))} for t,task in enumerate(TASKS)}
                    for label,lo,hi in (('early56',0,56),('late56',56,112))}
        saved=report['saved_TRAIN_head_scores'][index]
        a.tree(saved,{'repetition':rep,'saved_seed_signed_int64':f['actual_seed'][0],'tasks':task_values,'chronological_TRAIN_halves':halves},'Saved head '+prefix+str(rep))
        a.tree(saved['saved_TRAIN_predictions'],{'linear':p['ridge_predictions'],'neural':p['neural_predictions'],'support':p['valid']},'Saved predictions '+prefix+str(rep))
        a.ck(saved['manual_saved_affine_contractions'] is True,'Declared existing ridge affine algebra')
        for t,task in enumerate(TASKS):
            component={}
            indices=[r for r in range(112) if p['valid'][r*4+t]]
            y=[target[r*4+t] for r in indices];residual=[p['ridge_predictions'][r*4+t]-target[r*4+t] for r in indices]
            expected_blocks={}
            for name,lo,hi in block_specs:
                z=[f['target_scale'][t]*math.fsum((x[r*width+d]-f['feature_mean'][d])/f['feature_scale'][d]*f['ridge_weights'][d*4+t] for d in range(lo,hi)) for r in range(112)]
                component[name]=z;chosen=[z[r] for r in indices]
                expected_blocks[name]={'saved_standardized_weight_l2':norm([f['ridge_weights'][d*4+t] for d in range(lo,hi)]),
                    'TRAIN_contribution_population_std_in_target_units':math.sqrt(var(chosen)),
                    'TRAIN_contribution_pearson_to_target':pearson(chosen,y),'TRAIN_contribution_pearson_to_saved_residual':pearson(chosen,residual),
                    'row_contributions_in_target_units':z}
            pairs=[]
            for i,(name,_,_) in enumerate(block_specs):
                for other,_,_ in block_specs[i+1:]:
                    u=[component[name][r] for r in indices];v=[component[other][r] for r in indices]
                    pairs.append({'blocks':[name,other],'population_covariance':average([(p-average(u))*(q-average(v)) for p,q in zip(u,v)]),'pearson':pearson(u,v)})
            total=[math.fsum(component[n][r] for n,_,_ in block_specs) for r in indices]
            marginal=math.fsum(var([component[n][r] for r in indices]) for n,_,_ in block_specs)
            a.tree(saved['saved_linear_readout_block_decomposition'][task],{'blocks':expected_blocks,'pairwise_component_covariance':pairs,
                'sum_contribution_population_variance':var(total),'sum_of_block_population_variances':marginal,
                'sum_to_marginal_variance_ratio':var(total)/marginal if marginal else None,
                'constant_target_units':f['target_mean'][t]+f['target_scale'][t]*f['ridge_intercept'][t]},'Ridge components '+prefix+task)
            expected_halves={}
            for label,lo,hi in (('early56',0,56),('late56',56,112)):
                chosen=[r for r in range(lo,hi) if p['valid'][r*4+t]]
                expected_halves[label]={}
                for name,_,_ in block_specs:
                    values=[component[name][r] for r in chosen]
                    expected_halves[label][name]={'valid_rows':len(chosen),
                        'contribution_population_std_in_target_units':math.sqrt(var(values)) if chosen else None,
                        'contribution_pearson_to_target':pearson(values,[target[r*4+t] for r in chosen]),
                        'contribution_pearson_to_saved_residual':pearson(values,[p['ridge_predictions'][r*4+t]-target[r*4+t] for r in chosen])}
            for label,lo,hi in (('early56',0,56),('late56',56,112)):
                chosen=[r for r in range(lo,hi) if p['valid'][r*4+t]]
                pairs=[]
                for index,(name,_,_) in enumerate(block_specs):
                    for other,_,_ in block_specs[index+1:]:
                        left=[component[name][r] for r in chosen];right=[component[other][r] for r in chosen]
                        pairs.append(dict(blocks=[name,other],
                            population_covariance=average([(x-average(left))*(y-average(right)) for x,y in zip(left,right)]),
                            pearson=pearson(left,right)))
                expected_halves[label]['pairwise_component_covariance']=pairs
            a.tree(saved['saved_linear_readout_block_decomposition'][task]['chronological_TRAIN_halves'],expected_halves,'Ridge block chronological halves '+prefix+task)
        expected_reps.append(task_values)
    means={task:{head:{metric:average([r[task][head][metric] for r in expected_reps]) for metric in ('r2','mae','rmse','coverage')} for head in ('linear','neural')} for task in TASKS}
    a.tree(report['per_task_mean_across_saved_repetitions'],means,'Three saved head repetition means '+prefix)
    return means

class Audit:
    def __init__(self,card):
        self.card=card;self.checks=0;self.errors=[];self.numeric=[];self.archives={}
        self.allowed={row['id']:row for row in card['allowed_inputs']}
        self.context={row['id']:row for row in card['context_inputs']}
        self.verified=[];self.manual={name:{'dot_outputs':0,'multiply_add_terms':0} for name in
            ('ridge_total','ridge_block','mean4_decoder','temporal_mean4_export','temporal_variance4_export')}
        self.ck(card['kind']==MAIN_CARD_KIND and card['stage']=='prospective_before_diagnostic_arithmetic'
                and card['computed_versions']==['v11'] and card['cached_reference_versions']==['v9']
                and card['baseline']=='CPB-v3.market-01' and card['train_rows']==112
                and card['query_rows']==8 and card.get('execution_admitted') is True
                and card['train_begin_ms']==1704067259999 and card['train_end_ms']==1705881599999
                and card['fixed_update_blocks']==[[1,512],[513,1024],[1025,1536],[1537,2048]]
                and card['cached_v9_selectors']==list(CACHED_SELECTORS)
                and card['manual_saved_affine_contractions_declared'] is True
                and card['module_forwards_allowed'] is False and card['new_quality_or_selection_allowed'] is False,'Prospective V11 computation and cached V9 scope')
        self.ck(len(self.allowed)==len(card['allowed_inputs'])==25 and set(self.allowed)==ALLOWED_IDS,
                'Exactly twenty-five independent closed numeric roles')
        self.ck(set(self.context)==CONTEXT_IDS and len(card['context_inputs'])==len(CONTEXT_IDS),
                'Exactly frozen context roles; no context is an arithmetic archive')
        guards=card['hash_only_guards']
        self.ck(len(guards)==34 and {row['id'] for row in guards}==HASH_ONLY_IDS,
                '28 old V9 plus six V11 hash-only guards')
        self.ck(not(set(self.allowed)&{row['id'] for row in guards}),'Numeric and hash-only roles disjoint')
        for row in card['allowed_inputs']+card['context_inputs']+guards+card.get('preservation_pins',[]):
            actual=pin(row['container_path'])
            self.ck((actual['bytes'],actual['sha256'])==(row['bytes'],row['sha256']),
                    'Frozen input bytes '+row['id'])
            self.verified.append(actual)
        for identity,row in self.allowed.items():
            if identity=='cached_v9_prior_TRAIN_diagnosis':
                self.ck(row['role']=='audited_prior_TRAIN_output_selected_V9_subtrees_only'
                        and row['numeric_json_selectors']==list(CACHED_SELECTORS)
                        and row['sha256']==OLD_DIAGNOSIS_SHA,'Only four cached audited V9 numeric selectors')
            elif identity.endswith('_query_summary'):
                self.ck(row['role']=='aggregate_JSON_TRAIN_key_only'
                        and row['numeric_json_selector']=='train','TRAIN-only mixed query JSON selector')
            elif identity=='v11_temporal_moment_csv':
                self.ck(row['role']=='V11_TRAIN_moment_subgroup_16384_rows'
                        and row['allowed_CSV_fields']==list(MOMENT_COLUMNS),
                        'Eleven observed CSV columns and fixed update blocks')
            else:
                fields=row['allowed_tensor_fields']
                self.ck(row['role']=='TRAIN_archive' and len(fields)==len(set(fields))
                        and set(fields)==independent_allowed_fields(identity),
                        'Independently enumerated typed/tensor selectors '+identity)
            self.ck('validation' not in row['container_path'].lower()
                    and 'test' not in Path(row['container_path']).name.lower(),
                    'No held-out numerical archive path')
        pre=card['pre_execution_preservation'];actual=pin(pre.get('container_path') or pre.get('path'))
        self.ck((actual['bytes'],actual['sha256'])==(pre['bytes'],pre['sha256']),'Frozen preexecution preservation')
        self.verified.append(actual)
        self.metadata_context()
        if self.errors:raise ValueError('Independent scope/hash admission failed before tensor values: '+repr(self.errors))
    def ck(self,ok,label):
        self.checks+=1
        if not ok and len(self.errors)<200:self.errors.append(label)
    def close(self,actual,expected,label,absolute=1e-10,relative=1e-9):
        if actual is None or expected is None:self.ck(actual is None and expected is None,label);return
        self.ck(math.isfinite(actual) and math.isfinite(expected)
                and abs(actual-expected)<=absolute+relative*abs(expected),label)
    def tree(self,actual,expected,label,absolute=1e-10,relative=1e-9):
        if isinstance(expected,dict):
            self.ck(isinstance(actual,dict),label+' object')
            if not isinstance(actual,dict):return
            for key,value in expected.items():
                self.ck(key in actual,label+' missing '+str(key))
                if key in actual:self.tree(actual[key],value,label+'/'+str(key),absolute,relative)
        elif isinstance(expected,list):
            self.ck(isinstance(actual,list) and len(actual)==len(expected),label+' length')
            if isinstance(actual,list):
                for index,(left,right) in enumerate(zip(actual,expected)):
                    self.tree(left,right,label+'/'+str(index),absolute,relative)
        elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
            self.close(actual,expected,label,absolute,relative)
        else:self.ck(actual==expected,label)
    def affine(self,name,outputs,terms):
        self.manual[name]['dot_outputs']+=outputs
        self.manual[name]['multiply_add_terms']+=terms
    def tensor(self,identity,field):
        row=self.allowed[identity]
        if row['role']!='TRAIN_archive' or field not in independent_allowed_fields(identity):
            raise ValueError('Unallowed numeric tensor selector')
        if identity not in self.archives:self.archives[identity]=Archive(row['container_path'])
        value=self.archives[identity].field(*field.split('.'))
        if not isinstance(value,Tensor):raise ValueError('Inert typed tensor required')
        return value
    def array(self,identity,field,shape=None):
        value=self.tensor(identity,field)
        if shape is not None:self.ck(tuple(value.shape)==tuple(shape),'Shape '+identity+':'+field)
        values=value.values();self.numeric.append(dict(id=identity,field=field,values=len(values)))
        return values
    def summary(self,identity):
        row=self.allowed[identity]
        if row['numeric_json_selector']!='train':raise ValueError('Only saved train subtree may be selected')
        self.numeric.append(dict(id=identity,field='train_only'))
        return load(row['container_path'])['train']
    def close_archives(self):
        for value in self.archives.values():value.close()
    def metadata_context(self):
        # Context is saved hash/receipt metadata. Never execute old sources or
        # select old TRAIN descriptor values here.
        contexts={name:load(row['container_path']) for name,row in self.context.items()}
        old=contexts['old_audit'];old_card=self.context['old_card']
        self.ck(self.context['old_audit']['sha256']==OLD_AUDIT_SHA
                and old_card['sha256']==OLD_CARD_SHA and old['status']=='passed'
                and old['checks']==687825 and not old['mismatches'],
                'Exact preserved prior 687825-check audit')
        same=lambda left,right:all(left[key]==right[key] for key in ('bytes','sha256'))
        self.ck(same(old['card'],old_card)
                and same(old['diagnosis'],self.allowed['cached_v9_prior_TRAIN_diagnosis']),
                'Prior audit binds old card and cached diagnosis exactly')
        main_mirror=contexts['old_main_mirror'];audit_mirror=contexts['old_audit_mirror']
        self.ck(main_mirror['status']=='passed' and main_mirror['all_sizes_sha256_exact'] is True
                and main_mirror['container_before_after_exact'] is True
                and any(same(row,self.allowed['cached_v9_prior_TRAIN_diagnosis']) for row in main_mirror['files'])
                and any(same(row,old_card) for row in main_mirror['files']),
                'Old main complete mirror preserves exact card/diagnosis/source context')
        self.ck(audit_mirror['status']=='passed' and audit_mirror['all_hashes_exact'] is True
                and audit_mirror['source_before_after_exact'] is True
                and any(same(row['host'],self.context['old_audit']) for row in audit_mirror['files'])
                and any(same(row['host'],old['source']) for row in audit_mirror['files']),
                'Old independent mirror preserves exact audit/source context')
        trust=contexts['v11_final_establishment'];saved=contexts['v11_final_preservation']
        self.ck(trust['status']=='passed' and trust['candidate']=='CPB-v11.market-01'
                and trust['trust_admission_passed'] is True
                and trust['measured_candidate_established'] is True
                and trust['initial_partial_native20_40_pairings_passed'] is True,
                'Completed V11 trust is reused, not recomputed')
        self.ck(saved['status']=='passed' and saved['candidate']=='CPB-v11.market-01'
                and saved['critical_checkpoint_total']==len(saved['critical_checkpoint_pins'])==109
                and saved['baseline']=='CPB-v3.market-01' and saved['application_changed'] is False
                and same(saved['final_establishment'],self.context['v11_final_establishment'])
                and same(saved['current_ledger'],self.context['current_ledger'])
                and same(saved['current_catalog'],self.context['current_catalog']),
                'Actual completed109 preservation binds V3/current metadata')
        training=contexts['v11_training_audit']
        self.ck(training['status']=='passed' and training['checks']==trust['training_audit_core_checks']
                and all(training[name]['status']=='passed' and training[name]['checks']==count
                        for name,count in trust['training_audit_five_module_checks'].items()),
                'Existing V11 core/five-module learning/local-adjoint audit counts are metadata only')
        goal=contexts['current_goal_protocol'];retention=contexts['retention_independent_review']
        self.ck(goal['criterion_count']==len(goal['criteria'])==22
                and goal['baseline']==goal['application']=='CPB-v3.market-01'
                and goal['additional_train_NLL_gate'] is False
                and retention['status']=='passed'
                and same(retention['protocol'],self.context['current_goal_protocol']),
                'Reviewed stronger future floor protocol is an external prerequisite')
        expected={'v11_preservation_status':('v11_final_preservation','passed'),
          'v11_preservation_checkpoint_count':('v11_final_preservation',109),
          'retention_review_status':('retention_independent_review','passed'),
          'retention_applied':('retention_application_admission',True),
          'ledger_baseline':('current_ledger','CPB-v3.market-01'),
          'catalog_baseline':('current_catalog','CPB-v3.market-01')}
        self.ck(set(self.card['context_assertions'])==set(expected),'Exactly six source-declared final-context assertions')
        for name,(identity,want) in expected.items():
            assertion=self.card['context_assertions'][name]
            self.ck(assertion['input']==identity and isinstance(assertion['path'],list)
                    and bool(assertion['path']) and assertion['equals']==want
                    and type(assertion['equals']) is type(want),'Exact context assertion '+name)
            observed=contexts[identity]
            for part in assertion['path']:observed=observed[part]
            self.ck(observed==want and type(observed) is type(want),'Actual context assertion '+name)
        for guard in self.card['hash_only_guards']:
            if guard['id'].startswith('v9_'):
                self.ck(any(row['id']==guard['id'] and same(row,guard)
                            for row in contexts['old_card']['allowed_inputs']),
                        'Old raw guard remains attached to its original frozen card')
        for row in contexts['old_card']['owned_source_files']:
            self.ck(any(same(member,row) for member in main_mirror['files']),
                    'Old full source copies remain pinned in prior complete mirror')
        self.context_data=contexts


def query_audit(a,prefix,report,ids):
    width=40
    qid=prefix+'_query';shape=(8,3,128,9)
    q={k:a.array(qid,k,shape) for k in ('target_mask','targets','mean','variance')}
    q['native']=a.array(qid,'native',(8,width));q['row_ids']=a.array(qid,'row_ids',(8,))
    a.ck(q['row_ids']==ids[:8],'First8 TRAIN query rows only')
    a.ck(all(type(value) is bool for value in q['target_mask']),'Declared boolean query target support')
    a.ck(all(math.isfinite(value) for field in ('native','mean','variance','targets') for value in q[field]),
         'Finite selected saved query values')
    a.ck(all(value>0 for value,ok in zip(q['variance'],q['target_mask']) if ok),'Positive supported saved variance')
    index=lambda b,c,h,f:((b*3+c)*128+h)*9+f
    row_mae=[];row_nll=[];counts=[]
    for b in range(8):
        maes=[];nlls=[];cells=0
        for c in range(3):
            chosen=[index(b,c,h,f) for h in range(128) for f in range(9) if q['target_mask'][index(b,c,h,f)]]
            cells+=len(chosen)
            if chosen:
                error=[q['mean'][i]-q['targets'][i] for i in chosen]
                maes.append(average([abs(e) for e in error]))
                nlls.append(average([.5*(e*e/q['variance'][i]+math.log(q['variance'][i])) for e,i in zip(error,chosen)]))
        row_mae.append(average(maes));row_nll.append(average(nlls));counts.append(cells)
    a.tree(a.array(qid,'row_mae',(8,)),row_mae,'Saved query MAE');a.tree(a.array(qid,'row_nll',(8,)),row_nll,'Saved query NLL')
    summary=a.summary(prefix+'_query_summary')
    for b in range(8):
        a.close(summary['row_mae'][b],row_mae[b],'B8 summary MAE');a.close(summary['row_nll'][b],row_nll[b],'B8 summary NLL')
        a.ck(summary['row_target_cells'][b]==counts[b],'B8 support count')
    a.close(average(summary['row_mae']),summary['standardized_fixed_query_MAE'],'112 TRAIN query MAE')
    a.close(average(summary['row_nll']),summary['Gaussian_NLL_F64'],'112 TRAIN query NLL')
    a.ck(summary['valid_rows']==112 and sum(summary['row_target_cells'])==summary['target_cells'],'112 TRAIN query support')
    cf_values=[]
    for c in range(3):
        for f in range(9):
            chosen=[index(b,c,h,f) for b in range(8) for h in range(128) if q['target_mask'][index(b,c,h,f)]]
            e=[q['mean'][i]-q['targets'][i] for i in chosen];v=[q['variance'][i] for i in chosen]
            inv=[1/x for x in v];standard=[p/math.sqrt(x) for p,x in zip(e,v)];squared=[p*p/x for p,x in zip(e,v)]
            expected={'channel_id':c,'feature':f,'target_cells':len(chosen),'mean_absolute_error':average([abs(x) for x in e]),
                'mean_squared_error':average([x*x for x in e]),'mean_variance':average(v),'mean_inverse_variance':average(inv),
                'inverse_variance_quantiles':quantile(inv),'abs_error_inverse_variance_pearson':pearson([abs(x) for x in e],inv),
                'fraction_abs_error_at_least1':average([float(abs(x)>=1.) for x in e]),
                'mean_half_squared_error_over_variance':average([.5*x for x in squared]),'mean_half_log_variance':average([.5*math.log(x) for x in v]),
                'mean_Gaussian_NLL':average([.5*(x+math.log(y)) for x,y in zip(squared,v)]),
                'variance_quantiles':quantile(v),'standardized_residual_mean':average(standard),
                'standardized_residual_quantiles':quantile(standard),'squared_standardized_residual_quantiles':quantile(squared)}
            cf_values.append(expected);a.tree(report['per_channel_feature'][c*9+f],expected,'Query CF '+prefix+str(c*9+f))
    a.tree(report,{'detailed_population':8,'detailed_row_ids':ids[:8],'full_TRAIN_probe_population':112,
        'B8_hierarchical_mean_MAE':average(row_mae),'B8_hierarchical_mean_NLL':average(row_nll),'B8_target_cells':sum(counts),
        'all112_TRAIN':{'standardized_fixed_query_MAE':summary['standardized_fixed_query_MAE'],'Gaussian_NLL_F64':summary['Gaussian_NLL_F64'],
            'target_cells':summary['target_cells'],'row_mae_quantiles':quantile(summary['row_mae']),'row_nll_quantiles':quantile(summary['row_nll']),
            'row_target_cells':summary['row_target_cells'],'row_mae':summary['row_mae'],'row_nll':summary['row_nll'],
            'chronological_TRAIN_halves':{label:{'rows':hi-lo,
                'standardized_fixed_query_MAE':average(summary['row_mae'][lo:hi]),
                'Gaussian_NLL_F64':average(summary['row_nll'][lo:hi]),
                'target_cells':sum(summary['row_target_cells'][lo:hi])}
                for label,lo,hi in (('early56',0,56),('late56',56,112))}}},'Query population '+prefix)
    components=reduce_components(q['targets'],q['mean'],q['variance'],q['target_mask'])
    huber=[row['means'][2] for row in components['rows']]
    a.tree(report['B8_fixed_Huber_delta1'],
           {'mean':average(huber),'rows':huber,
            'status':'applied-mean-objective'},
           'First8 supported Huber evidence '+prefix)
    for field,scalar,column in (('row_mean_huber','mean_Huber_F64',2),
                                ('row_variance_nll','variance_NLL_F64',1),
                                ('row_total_objective','total_objective_F64',3)):
        a.ck(len(summary[field])==112 and all(math.isfinite(x) for x in summary[field]),
             'Both versions have complete112 saved component rows')
        values=[row['means'][column] for row in components['rows']]
        a.tree(a.array(qid,field,(8,)),values,'Saved first8 component '+prefix)
        a.tree(summary[field][:8],values,'Full112 first8 component association')
        a.close(average(summary[field]),summary[scalar],'Full112 component reduction')
        a.tree(report['all112_TRAIN'][field],summary[field],'Reported full112 component rows')
        a.close(report['all112_TRAIN'][scalar],summary[scalar],'Reported full112 component scalar')
        for label,lo,hi in (('early56',0,56),('late56',56,112)):
            a.close(report['all112_TRAIN']['chronological_TRAIN_halves'][label][scalar],average(summary[field][lo:hi]),
                    'Component fixed TRAIN half')
    for i in range(112):
        a.close(summary['row_variance_nll'][i],summary['row_nll'][i],'Numeric variance/reporting identity')
        a.close(summary['row_total_objective'][i],summary['row_mean_huber'][i]+summary['row_variance_nll'][i],
                'Unit component sum')
    a.ck(report['all112_component_evidence']=='saved-V11-no-reconstruction-from-MAE/NLL',
         'Saved V11 full112 component evidence')
    return q,cf_values,summary

def residual_audit(a,prefix,saved,q):
    decoder=a.array(prefix+'_checkpoint','payload.model.encoder.change_decoder',(3456,4))
    residual=[math.fsum(q['native'][b*40+32+d]*decoder[i*4+d] for d in range(4))
              for b in range(8) for i in range(3456)]
    a.affine('mean4_decoder',8*3456,8*3456*4)
    parent=[float(total)-part for total,part in zip(q['mean'],residual)]
    a.ck(saved['population']==8,'Residual allocation limited to fixed-query first8')
    at=lambda b,c,h,f:((b*3+c)*128+h)*9+f
    for c in range(3):
        for f in range(9):
            selected=[at(b,c,h,f) for b in range(8) for h in range(128) if q['target_mask'][at(b,c,h,f)]]
            r=[residual[i] for i in selected];p=[parent[i] for i in selected]
            a.tree(saved['per_channel_feature'][c*9+f],
                   {'channel_id':c,'feature':f,'target_cells':len(selected),
                    'residual_quantiles':quantile(r),'parent_remainder_quantiles':quantile(p),
                    'parent_residual_pearson':pearson(p,r)},'Approximate saved linear mean4 residual allocation')

def checkpoint_audit(a,stage,report):
    identity='v11_'+stage+'_checkpoint'
    def read(field,dtype,shape=None):
        a.ck(a.tensor(identity,field).storage.dtype==dtype,'Typed checkpoint field '+field)
        return a.array(identity,field,shape)
    def text(field):return bytes(read(field,'ByteStorage')).decode('utf-8')
    a.ck(text('artifact_kind')=='cpb_v11_inference_checkpoint_v1' and text('model_tag')=='CPB-v11',
         'Restricted same-stage V11 inference type')
    a.ck(text('initialization_kind')=='fresh-seeded-constructor-no-import-v1'
         and text('pretrained_ancestor')=='none' and read('source_updates','LongStorage')==[0]
         and read('causal_updates','LongStorage')==[0 if stage=='initial' else 2048]
         and read('initialization_seed','LongStorage')==[943001]
         and read('channel_ids','LongStorage',(3,))==[0,1,2],'Fresh typed selected checkpoint lineage')
    for prefix in ('','payload.'):
        for field,want in (('gradient_policy',POLICY),('objective_policy',OBJECTIVE),
                ('direct_gradient_routes',DIRECT_ROUTES),('native_layout',NATIVE_LAYOUT),
                ('temporal_pooling_policy',TEMPORAL_POLICY),('temporal_moment_policy',MOMENT_POLICY),
                ('temporal_export_layout',EXPORT_LAYOUT),('empty_local_bin_policy','zero-whole-export-slice-after-bias-v1')):
            a.ck(text(prefix+field)==want,'Restricted typed text '+prefix+field)
        for field,want in (('mean_width',4),('uncertainty_width',4),('native_width',40),
                           ('variance_error_detached',1),('reporting_nll_backpropagated',0)):
            a.ck(read(prefix+field,'LongStorage')==[want],'Typed scalar '+prefix+field)
        for field,want,shape,dtype in (
                ('mean_huber_delta_f64',[1.],None,'DoubleStorage'),
                ('loss_coefficients_f64',[1.,1.],(2,),'DoubleStorage'),
                ('temporal_pooling_mix_f64',[.5,.5],(2,),'DoubleStorage'),
                ('temporal_pooling_partition_v1',[4,16,32,4,8],(5,),'LongStorage'),
                ('temporal_moment_partition_v1',[4,4],(2,),'LongStorage')):
            a.ck(read(prefix+field,dtype,shape)==want,'Typed geometry/objective '+prefix+field)
    a.tree(report,dict(source_updates=0,causal_updates=0 if stage=='initial' else 2048,
        initialization_seed=943001,pretrained_ancestor='none',gradient_policy=POLICY,native_width=40,
        objective=OBJECTIVE,direct_gradient_routes=DIRECT_ROUTES,clipping='single-global-norm2-clip1',
        temporal_pooling=TEMPORAL_POLICY,temporal_moment_policy=MOMENT_POLICY,
        temporal_export_layout=EXPORT_LAYOUT,temporal_moment_partition=[4,4],
        variance_reads='live-independent-uncertainty4'),'Checkpoint report '+stage)


def association_audit(a,stage,saved,summary,targets):
    expected=[]
    full=summary
    for report in saved['saved_TRAIN_head_scores']:
        values=report['saved_TRAIN_predictions']
        for task_index,task in enumerate(TASKS):
            for head in ('linear','neural'):
                for label,start,end in (('all112',0,112),('early56',0,56),('late56',56,112)):
                    indices=[row for row in range(start,end) if values['support'][row*4+task_index]
                             and full['row_target_cells'][row]>0]
                    residual=[values[head][row*4+task_index]-targets[row*4+task_index] for row in indices]
                    component={}
                    for name in ('row_mae','row_nll','row_mean_huber','row_variance_nll','row_total_objective'):
                        observed=[full[name][row] for row in indices]
                        component[name]=dict(pearson_to_signed_readout_residual=pearson(observed,residual),
                            pearson_to_absolute_readout_residual=pearson(observed,[abs(value) for value in residual]))
                    expected.append(dict(repetition=report['repetition'],task=task,head=head,scope=label,
                        eligible_rows=len(indices),readout_residual_units='physical task target units',
                        reconstruction_units='standardized query units',components=component))
    a.tree(saved['reconstruction_readout_associations'],expected,'Fixed all-task/readout/reconstruction associations '+stage)


def cached_audit(a,report):
    identity='cached_v9_prior_TRAIN_diagnosis'
    root=load(a.allowed[identity]['container_path'])
    values=[]
    for path in CACHED_SELECTORS:
        selected=root
        for part in path.split('.'):selected=selected[part]
        values.append(selected)
        a.numeric.append(dict(id=identity,field=path))
    expected={'stages':{'initial':values[0],'final':values[1]},
              'actual_pre_step_batch_diagnosis':values[2], 'temporal_learning_observations':values[3]}
    # Whole-object equality carries already audited values forward. No V9
    # arrays, score arithmetic or old diagnostic reduction is repeated.
    a.ck(report['cached_v9_TRAIN']==expected,'Exact four-subtree cached V9 carry-forward')
    a.ck(set(report['cached_v9_TRAIN'])==set(expected),'No additional V9/V8 cached result tree')
    a.ck(report['cached_V9_numeric_selectors']==list(CACHED_SELECTORS)
         and report['V9_new_numerical_reductions']==0 and report['old_V9_raw_numeric_reads']==0
         and report['old_V8_numeric_reads']==0,'Cached historical scope with no raw re-reduction')
    same=lambda left,right:all(left[key]==right[key] for key in ('bytes','sha256'))
    a.ck(same(report['cached_V9_independent_audit_reused'],a.context['old_audit']),
         'Cached result explicitly relies on original independent audit')
    return expected


def moment_audit(a,report):
    identity='v11_temporal_moment_csv';rows=[]
    with Path(a.allowed[identity]['container_path']).open(encoding='utf-8-sig',newline='') as stream:
        source=csv.DictReader(stream)
        a.ck(tuple(source.fieldnames or ())==MOMENT_COLUMNS,'Exact source-declared11 CSV selectors')
        for line in source:
            if len(rows)>=16384:raise ValueError('Bounded complete moment observation population')
            row={key:int(line[key]) for key in MOMENT_COLUMNS[:2]+MOMENT_COLUMNS[3:5]}
            row['moment']=line['moment']
            if not(1<=row['update']<=2048 and row['bin'] in range(4)
                   and row['moment'] in ('mean','centered_variance')):raise ValueError('Declared moment identity')
            for name in MOMENT_COLUMNS[5:]:
                row[name]=float(line[name])
                if not math.isfinite(row[name]):raise ValueError('Finite captured scalar descriptor')
            rows.append(row)
    a.numeric.append(dict(id=identity,fields=list(MOMENT_COLUMNS),rows=len(rows)))
    a.tree(report['csv'],describe_moment_csv(rows),'Independent all8 moment groups/fixed512 summaries')
    for index,update in enumerate((1,2048)):
        identity='v11_actual_update_'+str(update)
        def read(field,shape=None,dtype=None):
            if dtype:a.ck(a.tensor(identity,field).storage.dtype==dtype,'Actual typed descriptor '+field)
            return a.array(identity,field,shape)
        a.ck(read('update',dtype='LongStorage')==[update],'Declared saved TRAIN update')
        attempt=read('attempt',dtype='LongStorage');clip=read('clip_factor',dtype='DoubleStorage')
        positions_tensor=a.tensor(identity,'temporal_patch_positions')
        a.ck(len(positions_tensor.shape)==2,'Packed descriptor rank')
        R,T=positions_tensor.shape
        a.ck(0<=R<=24 and 0<=T<=16,'Bounded saved packed geometry')
        positions=read('temporal_patch_positions',(R,T),'LongStorage')
        valid=read('temporal_patch_valid',(R,T),'BoolStorage')
        packed_rows=read('temporal_patch_rows',(R,),'LongStorage')
        weight=read('temporal_export_weight_before',(32,64),'FloatStorage')
        bias=read('temporal_export_bias_before',(32,),'FloatStorage')
        a.ck(all(math.isfinite(value) for value in weight+bias),'Finite before-step saved affine coefficients')
        routes=[]
        for name,prefix in (('ordinary','temporal_pool'),('contextual','temporal_contextual_pool')):
            route={}
            for field,shape,dtype in (('bin_valid',(R,4),'BoolStorage'),('bin_mean',(R,4,64),'FloatStorage'),
                    ('bin_variance',(R,4,64),'FloatStorage'),('mixture_weights',(R,4,T),'FloatStorage'),
                    ('projected_upstream_defined',(),'LongStorage'),('projected_upstream_pre_clip',(R,32),'FloatStorage')):
                value=read(prefix+'_'+field,shape,dtype)
                if field=='projected_upstream_defined':
                    a.ck(value in ([0],[1]),'Explicit used/unused route flag')
                    route['upstream_defined']=bool(value[0])
                elif field=='projected_upstream_pre_clip':route['upstream_pre_clip']=value
                else:route[field]=value
            a.ck(all(math.isfinite(value) for field in ('bin_mean','bin_variance','mixture_weights','upstream_pre_clip')
                     for value in route[field]),'Finite selected same-forward descriptors')
            routes.append((name,route))
        expected=describe_moment_witness(update,attempt[0],clip[0],positions,valid,packed_rows,
                                         weight,bias,routes,packed_shape=(R,T))
        a.tree(report['actual_witnesses'][index],expected,'Independent captured moment/export allocation '+str(update))
        for family in ('temporal_mean4_export','temporal_variance4_export'):
            a.affine(family,2*4*R*4,2*4*R*4*64)
    a.ck(len(report['actual_witnesses'])==2,'Both fixed observed actual TRAIN minibatches only')


def comparison_audit(a,comparison,stages,cached):
    readout=[];native=[];queries=[]
    for stage in ('initial','final'):
        old=cached['stages'][stage];new=stages[stage]
        for task in TASKS:
            for head in ('linear','neural'):
                for metric in ('r2','mae','rmse','coverage'):
                    left=old['per_task_mean_across_saved_repetitions'][task][head][metric]
                    right=new['per_task_mean_across_saved_repetitions'][task][head][metric]
                    readout.append(dict(stage=stage,task=task,head=head,metric=metric,cached_V9=left,new_V11=right,
                        V11_minus_cached_V9=right-left if left is not None and right is not None else None))
        for block,_,_ in BLOCKS+(('complete40',0,40),):
            for metric in ('covariance_trace','covariance_frobenius_squared','participation_ratio'):
                left=old['native_features']['blocks'][block][metric];right=new['native_features']['blocks'][block][metric]
                native.append(dict(stage=stage,block=block,metric=metric,cached_V9=left,new_V11=right,
                    V11_minus_cached_V9=right-left if left is not None and right is not None else None))
        for metric in ('standardized_fixed_query_MAE','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64'):
            left=old['fixed_query_TRAIN_diagnosis']['all112_TRAIN'][metric]
            right=new['fixed_query_TRAIN_diagnosis']['all112_TRAIN'][metric]
            queries.append(dict(stage=stage,metric=metric,cached_V9=left,new_V11=right,V11_minus_cached_V9=right-left))
    a.tree(comparison,dict(saved_TRAIN_readout_scalar_differences=readout,
        native_block_descriptor_differences=native,full112_reconstruction_scalar_differences=queries),
        'Differences of independently checked V11 and cached V9 scalar descriptors only')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('card','diagnosis','audit-card','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--audit-card-sha256',required=True)
    args=parser.parse_args()
    root=Path('/opt/cuwacunu/research/causal-v9-v11-train-diagnosis-audit')
    if not(sys.platform.startswith('linux') and Path('/.dockerenv').is_file()
           and args.output.is_absolute() and args.output.resolve().is_relative_to(root)
           and args.output.resolve()!=root and not args.output.exists()):
        raise ValueError('Existing managed container and exclusive new independent audit output')
    if digest(args.audit_card)!=args.audit_card_sha256:raise ValueError('External exact audit-card hash before parse')
    declaration=load(args.audit_card)
    if declaration['kind']!='cpb_v9_v11_TRAIN_diagnosis_independent_audit_card_v1' or declaration['execution_admitted'] is not True:
        raise ValueError('Separate frozen independent audit execution admission required')
    same=lambda left,right:all(left[key]==right[key] for key in ('bytes','sha256'))
    if not same(pin(args.card),declaration['main_card']) or not same(pin(args.diagnosis),declaration['main_diagnosis']):
        raise ValueError('Exact frozen main card and saved new diagnosis required')
    source_names={'audit.py','archive_reader.py','component_math.py','README.md','provenance.json'}
    if len(declaration['auditor_sources'])!=5 or {row['file'] for row in declaration['auditor_sources']}!=source_names:
        raise ValueError('Five complete owned source files independently frozen')
    own=Path(__file__).resolve().parent
    for row in declaration['auditor_sources']:
        if not same(pin(own/row['file']),row):raise ValueError('Changed frozen independent auditor source')
    frozen=[args.card,args.diagnosis,args.audit_card]+[own/name for name in sorted(source_names)]
    before=[pin(path) for path in frozen]
    started=time.monotonic();card=load(args.card);report=load(args.diagnosis);a=Audit(card)
    try:
        a.ck(report['kind']=='cpb_v9_v11_TRAIN_saved_information_diagnosis_v1'
             and report['status']=='passed_readonly_saved_TRAIN_diagnosis'
             and report['baseline']=='CPB-v3.market-01' and set(report['models'])=={'v11'}
             and set(report['models']['v11']['stages'])=={'initial','final'},'New V11 descriptor identity/population')
        a.ck(same(report['card'],declaration['main_card']),'Exact diagnosis-to-main-card binding')
        zeros=('model_calls','encoder_calls','decoder_calls','head_calls','module_forwards','neural_forwards',
               'fits','head_fits','training_updates','held_out_tensor_reads','old_V9_raw_numeric_reads',
               'old_V8_numeric_reads','local_adjoint_rechecks','learning_gauge_Hadamard_trust_rechecks')
        a.ck(all(type(report[name]) is int and report[name]==0 for name in zeros),'Zero prohibited execution/refit/replay counters')
        a.ck(report['new_encoder_experiment'] is False and report['candidate_declared'] is False
             and report['new_quality_scores'] is False and report['diagnostic_gate_or_next_encoder_selected'] is False
             and report['application_changed'] is False,'No new experiment, quality gate or promotion')
        target=a.array('train_targets','targets',(112,4));support=a.array('train_targets','valid',(112,4))
        ids=a.array('train_targets','row_ids',(112,));groups=a.array('train_targets','group_ids',(112,))
        a.ck(len(set(ids))==112 and ids==sorted(ids)
             and all(1704067259999<=value<=1705881599999 for value in ids),'Only complete chronological TRAIN row IDs')
        a.ck(all(type(value) is bool for value in support),'Boolean saved task support')
        cached=cached_audit(a,report);stages=report['models']['v11']['stages'];results={}
        for stage in ('initial','final'):
            prefix='v11_'+stage;saved=stages[stage]
            x,valid,columns=feature_audit(a,prefix+'_features',saved['native_features'],ids,target,support)
            head_audit(a,prefix,saved,x,valid,ids,groups,target,support)
            q,_,summary=query_audit(a,prefix,saved['fixed_query_TRAIN_diagnosis'],ids)
            checkpoint_audit(a,stage,saved['checkpoint_provenance'])
            residual_audit(a,prefix,saved['saved_mean_residual_allocation'],q)
            association_audit(a,stage,saved,summary,target)
            results[stage]=q
            a.ck(all(not any('derivative' in key for key in cell) for cell in saved['fixed_query_TRAIN_diagnosis']['per_channel_feature']),
                 'No local derivative descriptors in new V11 query panel')
            forbidden=('analytic_local_native_gradient_diagnosis','fixed_four32_temporal_bins')
            a.ck(not any(key in saved for key in forbidden)
                 and 'fixed_four32_temporal_bins' not in saved['fixed_query_TRAIN_diagnosis'],
                 'No old local-adjoint/raw-age diagnostic module output')
        for field in ('row_ids','targets','target_mask'):
            a.ck(results['initial'][field]==results['final'][field],'Same captured fixed TRAIN query plan '+field)
        moment_audit(a,report['temporal_moment_observations'])
        comparison_audit(a,report['comparison'],stages,cached)
        a.ck(set(report['manual_arithmetic_counters'])=={'manual_saved_affine_contractions','scalar_summary_function_calls'},
             'Separate disclosed manual scalar arithmetic counters')
        actual_affine=report['manual_arithmetic_counters']['manual_saved_affine_contractions']
        a.ck(actual_affine==a.manual,'Exact independently counted affine output/product quantities')
        for row in actual_affine.values():
            a.ck(all(type(value) is int and value>=0 for value in row.values()),'Explicit nonnegative manual contraction counts')
        scalar_calls=report['manual_arithmetic_counters']['scalar_summary_function_calls']
        a.ck(set(scalar_calls)=={'mean','quantiles','variance','pearson','score'}
             and all(type(value) is int and value>0 for value in scalar_calls.values()),
             'Positive disclosed implementation-dependent scalar summary calls, not primitive FLOPs')
        for row in report['numeric_fields_accessed']:
            identity=row['id'];a.ck(identity in a.allowed,'Producer closed numeric role '+identity)
            if identity not in a.allowed:continue
            if identity=='cached_v9_prior_TRAIN_diagnosis':
                a.ck(row['field'] in CACHED_SELECTORS,'Producer only four cached V9 selectors')
            elif identity.endswith('_query_summary'):
                a.ck(row['field']=='train_only','Producer mixed JSON selects train only')
            elif identity=='v11_temporal_moment_csv':
                a.ck(row['fields']==list(MOMENT_COLUMNS) and row['rows']==16384,'Producer entire fixed moment CSV scope')
            else:a.ck(row['field'] in independent_allowed_fields(identity),'Producer exact typed/tensor selector')
        a.ck(set(report['opened_allowed_ids']).issubset(ALLOWED_IDS),'No numeric opened hash guard/context role')
        for row in card['allowed_inputs']+card['hash_only_guards']+card['context_inputs']+card.get('preservation_pins',[])+[card['pre_execution_preservation']]:
            a.ck(same(pin(row.get('container_path') or row.get('path')),row),'Every original input unchanged after independent arithmetic')
        a.ck([pin(path) for path in frozen]==before,'All main/audit cards, diagnosis and owned source bytes unchanged')
        result=dict(kind='cpb_v9_v11_TRAIN_diagnosis_independent_audit_v1',
            status='passed' if not a.errors else 'failed',checks=a.checks,mismatches=a.errors,
            source=pin(__file__),reader=pin(own/'archive_reader.py'),card=pin(args.card),
            diagnosis=pin(args.diagnosis),audit_card=pin(args.audit_card),verified_inputs=a.verified,
            numeric_fields_accessed=a.numeric,computed_versions=['v11'],cached_reference_versions=['v9'],
            newly_independently_recomputed=['two V11 native/readout/query/mean4 stages',
                'all112 per-task/readout/reconstruction associations','16384 observed subgroup rows/fixed blocks',
                'two actual ordinary/contextual same-forward moment/export descriptors'],
            cached_independence_scope='Only four exact cached V9 subtrees; original687825-check audit reused, no second V9 reduction',
            cached_V9_selectors=list(CACHED_SELECTORS),cached_V9_original_audit=a.context['old_audit'],
            manual_saved_affine_contractions=a.manual,
            scalar_summary_calls_policy='Auditor performs independent scalar reductions; main instrumented counts are implementation-dependent and checked positive, not forced equal',
            model_calls=0,encoder_calls=0,decoder_calls=0,head_calls=0,module_forwards=0,neural_forwards=0,
            fits=0,head_fits=0,training_updates=0,held_out_tensor_reads=0,old_V9_raw_numeric_reads=0,
            old_V8_numeric_reads=0,local_adjoint_rechecks=0,learning_gauge_Hadamard_trust_rechecks=0,
            new_quality_scores=0,next_encoder_selected=False,application_changed=False,
            limits=['No saved moment captures for V9 or all112 query rows are manufactured.',
                'Actual2048 is pre-step2047 and never receives final2048 decoder coefficients.',
                'Gradient/movement magnitudes depend on coordinate units; no importance/dominance/counterfactual claim.',
                'TRAIN readout residual associations and cached scalar differences do not identify validation causes.',
                'No existing V11 trust proof, local adjoint, gauge/Hadamard audit or old source body is rerun.'],
            elapsed_seconds=time.monotonic()-started)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps(dict(status=result['status'],checks=a.checks,mismatches=a.errors,output=str(args.output))))
        return 0 if not a.errors else 1
    finally:a.close_archives()


if __name__=='__main__':raise SystemExit(main())

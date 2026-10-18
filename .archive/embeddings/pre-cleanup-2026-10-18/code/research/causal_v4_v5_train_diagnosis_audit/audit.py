#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent stdlib saved-TRAIN arithmetic; no diagnosis-module/model import."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from archive_reader import Archive, Tensor

MAIN_CARD_SHA = 'fcafd7c6dff4a0cbab6e103573e129532a8d9a733374971289c0da601f2d0aaa'
TASKS = ('past_volatility','past_trend','past_activity','past_relation')
REPS = (2701,2802,2903)
BLOCKS = (('learned20',0,20),('fixed_spectral12',20,32),('mean4',32,36))
QUANTILES = (0.,.01,.05,.25,.5,.75,.95,.99,1.)

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def pin(path):
    p=Path(path);return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}

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
def vector(x,y):
    nx,ny=norm(x),norm(y);dot=math.fsum(a*b for a,b in zip(x,y))
    return {'left_l2':nx,'right_l2':ny,'dot_product':dot,'cosine':dot/(nx*ny) if nx and ny else None}
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
    groups={
        'targets':{'row_ids','group_ids','targets','valid'},
        'features':{'row_ids','values','valid'},
        'query':{'row_ids','timestamps','target_mask','visible','observed','hidden','targets','mean','variance','native','row_mae','row_nll'},
        'head':{'row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions'},
        'fit':{'feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed'},
        'checkpoint':{'artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates','causal_updates','initialization_seed',
                      'channel_ids','payload.model.encoder.change_decoder','payload.model.variance_projection','payload.model.variance_bias',
                      'payload.link_constants_f32','payload.softplus_parameters_f64','gradient_policy','payload.gradient_policy'}}
    if identity=='train_targets':return groups['targets']
    if identity.endswith('_query_summary'):return set()
    for kind in ('features','query','checkpoint','head','fit'):
        if identity.endswith('_'+kind) or '_'+kind+'_' in identity:
            fields=set(groups[kind])
            if kind=='checkpoint' and identity.startswith('v5_'):
                fields.update({'native_layout','mean_width','uncertainty_width','native_width',
                               'payload.native_layout','payload.mean_width','payload.uncertainty_width','payload.native_width'})
            return fields
    raise ValueError('Unregistered input identity '+identity)

class Audit:
    def __init__(self,card):
        self.card=card;self.checks=0;self.errors=[];self.numeric=[];self.archives={}
        self.allowed={p['id']:p for p in card['allowed_inputs']};self.verified=[]
        self.finite_checks=0;self.maximum_finite_error=0.
        self.ck(len(self.allowed)==41,'41 unique allowed inputs')
        expected={'train_targets'}
        for version in ('v4','v5'):
            for stage in ('initial','final'):
                expected.update(version+'_'+stage+'_'+kind for kind in ('features','query','query_summary','checkpoint'))
                expected.update(version+'_'+stage+'_'+kind+'_'+str(rep) for kind in ('head','fit') for rep in REPS)
        self.ck(set(self.allowed)==expected,'Independent exact41 IDs')
        self.ck(len(card['preservation_pins'])==23,'23 preservation-only pins')
        for p in card['allowed_inputs']+card['preservation_pins']+[card['pre_execution_preservation']]:
            actual=pin(p['container_path'])
            self.ck(actual['bytes']==p['bytes'] and actual['sha256']==p['sha256'],'Input hash '+p.get('id',''))
            self.verified.append(actual)
        for p in card['allowed_inputs']:
            self.ck('validation' not in p['container_path'].lower(),'No held-out tensor path')
            if p['id'].endswith('_query_summary'):
                self.ck(p['role']=='aggregate_JSON_TRAIN_key_only' and p['numeric_json_selector']=='train','Strict TRAIN summary selector')
            else:
                self.ck(p['role']=='TRAIN_archive' and set(p['allowed_tensor_fields'])==independent_allowed_fields(p['id']),
                        'Independent exact tensor-field allowlist')
        if self.errors:raise ValueError('Input preservation/scope admission failed before numeric values')
    def ck(self,ok,label):
        self.checks+=1
        if not ok and len(self.errors)<200:self.errors.append(label)
    def close(self,actual,expected,label,absolute=1e-10,relative=1e-9):
        if actual is None or expected is None:self.ck(actual is None and expected is None,label);return
        self.ck(math.isfinite(actual) and math.isfinite(expected) and
                abs(actual-expected)<=absolute+relative*abs(expected),label)
    def tree(self,actual,expected,label):
        if isinstance(expected,dict):
            for k,v in expected.items():
                self.ck(k in actual,label+' missing '+str(k))
                if k in actual:self.tree(actual[k],v,label+'/'+str(k))
        elif isinstance(expected,list):
            self.ck(len(actual)==len(expected),label+' length')
            for i,(x,y) in enumerate(zip(actual,expected)):self.tree(x,y,label+'/'+str(i))
        elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
            self.close(actual,expected,label)
        else:self.ck(actual==expected,label)
    def array(self,identity,field,shape=None):
        p=self.allowed[identity]
        self.ck(p['role']=='TRAIN_archive' and field in p['allowed_tensor_fields'],'Field allowed '+identity+field)
        if field not in p['allowed_tensor_fields']:raise ValueError('Unallowed field')
        if identity not in self.archives:self.archives[identity]=Archive(p['container_path'])
        value=self.archives[identity].field(*field.split('.'))
        if not isinstance(value,Tensor):raise ValueError('Tensor required')
        if shape is not None:self.ck(tuple(value.shape)==tuple(shape),'Shape '+identity+field)
        data=value.values();self.numeric.append({'id':identity,'field':field,'values':len(data)})
        return data
    def summary(self,identity):
        p=self.allowed[identity];self.ck(p['role']=='aggregate_JSON_TRAIN_key_only','TRAIN JSON role')
        self.ck(p['numeric_json_selector']=='train','Only admitted TRAIN numeric JSON key')
        value=load(p['container_path'])['train']
        self.numeric.append({'id':identity,'field':'train_only'})
        return value
    def finite(self,actual,expected,label):
        self.finite_checks+=1
        self.maximum_finite_error=max(self.maximum_finite_error,abs(actual-expected))
        self.close(actual,expected,label,absolute=1e-6,relative=2e-4)
    def close_archives(self):
        for archive in self.archives.values():archive.close()

def feature_audit(a,identity,report,ids,target,target_valid):
    width=40 if identity.startswith('v5_') else 36
    block_specs=BLOCKS+((('uncertainty4',36,40),) if width==40 else ())
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
    width=40 if prefix.startswith('v5_') else 36
    block_specs=BLOCKS+((('uncertainty4',36,40),) if width==40 else ())
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
        for got,want,ok in zip(p['ridge_predictions'],reconstructed,p['valid']):
            if ok:a.close(got,want,'Saved ridge reconstruction')
        task_values={task:{head:scores(p[field][t::4],target[t::4],p['valid'][t::4])
                     for head,field in (('linear','ridge_predictions'),('neural','neural_predictions'))} for t,task in enumerate(TASKS)}
        halves={label:{task:{head:scores(p[field][lo*4+t:hi*4:4],target[lo*4+t:hi*4:4],p['valid'][lo*4+t:hi*4:4])
                    for head,field in (('linear','ridge_predictions'),('neural','neural_predictions'))} for t,task in enumerate(TASKS)}
                    for label,lo,hi in (('early56',0,56),('late56',56,112))}
        saved=report['saved_TRAIN_head_scores'][index]
        a.tree(saved,{'repetition':rep,'saved_seed_signed_int64':f['actual_seed'][0],'tasks':task_values,'chronological_TRAIN_halves':halves},'Saved head '+prefix+str(rep))
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
            a.tree(saved['saved_linear_readout_block_decomposition'][task]['chronological_TRAIN_halves'],expected_halves,'Ridge block chronological halves '+prefix+task)
        expected_reps.append(task_values)
    means={task:{head:{metric:average([r[task][head][metric] for r in expected_reps]) for metric in ('r2','mae','rmse','coverage')} for head in ('linear','neural')} for task in TASKS}
    a.tree(report['per_task_mean_across_saved_repetitions'],means,'Three saved head repetition means '+prefix)
    return means

def query_audit(a,prefix,report,ids):
    width=40 if prefix.startswith('v5_') else 36
    qid=prefix+'_query';shape=(8,3,128,9)
    q={k:a.array(qid,k,shape) for k in ('target_mask','visible','observed','hidden','targets','mean','variance')}
    q['native']=a.array(qid,'native',(8,width));q['row_ids']=a.array(qid,'row_ids',(8,))
    timestamps=a.array(qid,'timestamps',(8,128))
    a.ck(q['row_ids']==ids[:8],'First8 TRAIN query rows only')
    for b,row in enumerate(q['row_ids']):a.ck(timestamps[b*128:(b+1)*128]==[row+(h-127)*60000 for h in range(128)],'Exact TRAIN query time grid')
    for s,v,o,h in zip(q['target_mask'],q['visible'],q['observed'],q['hidden']):a.ck(s==(o and h) and v==(o and not h),'Exact masked support')
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
                'mean_abs_NLL_mean_derivative':average([abs(x)/y for x,y in zip(e,v)]),
                'mean_abs_MAE_mean_derivative':average([float(x!=0) for x in e]),'mean_abs_Huber_delta1_mean_derivative':average([min(abs(x),1.) for x in e]),
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
    return q,cf_values,summary

def gradient_audit(a,prefix,saved,q):
    """Independent F64 local chain rules; no model or cross-basis comparison."""
    cid=prefix+'_checkpoint';version=prefix[:2];width=40 if version=='v5' else 36
    decoder=a.array(cid,'payload.model.encoder.change_decoder',(3456,4))
    projection=a.array(cid,'payload.model.variance_projection',(27,4));bias=a.array(cid,'payload.model.variance_bias',(27,))
    eps,shift,coefficient=a.array(cid,'payload.link_constants_f32',(3,))
    a.ck(a.array(cid,'payload.softplus_parameters_f64',(2,))==[1.,20.],'Scalar softplus contract')
    a.ck(a.array(cid,'channel_ids',(3,))==[0,1,2],'Canonical coefficient channel order')
    a.ck(a.array(cid,'source_updates')==[0] and a.array(cid,'causal_updates')==[0 if prefix.endswith('initial') else 2048]
         and a.array(cid,'initialization_seed')==[943001],'Saved typed provenance')
    text=lambda k:bytes(a.array(cid,k)).decode('utf-8')
    a.ck(text('pretrained_ancestor')=='none' and text('initialization_kind')=='fresh-seeded-constructor-no-import-v1','Fresh typed ancestry')
    a.ck(text('artifact_kind')=='cpb_'+version+'_inference_checkpoint_v1' and text('model_tag')=='CPB-'+version,'Exact typed family')
    policy='variance_input_gradient_stop-v1' if version=='v4' else 'separate-mean-and-uncertainty-bank-projections-live-gaussian-nll-v1'
    a.ck(text('gradient_policy')==text('payload.gradient_policy')==policy,'Outer/nested disjoint or detached variance policy')
    if version=='v5':
        layout='parent32[0:32];mean4[32:36];uncertainty4[36:40];independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;variance-reads-live-uncertainty4-only'
        for first in ('','payload.'):
            a.ck(text(first+'native_layout')==layout,'Exact V5 disjoint native layout')
            for name,want in (('mean_width',4),('uncertainty_width',4),('native_width',40)):
                a.ck(a.array(cid,first+name)==[want],'Exact V5 typed native width')
    softplus=lambda x:x if x>20 else math.log1p(math.exp(x))
    sigmoid=lambda x:1. if x>20 else (1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x)))
    base=softplus(shift);at=lambda b,c,h,f:((b*3+c)*128+h)*9+f
    mean_names=('mean_NLL','mean_MAE','mean_Huber_delta1');all_names=mean_names+('variance_route',)
    flat={n:[] for n in all_names};local_by_cf=[{n:[] for n in all_names} for _ in range(27)]
    cf_abs=[{n:[] for n in mean_names} for _ in range(27)];cf_e=[[] for _ in range(27)];cf_inv=[[] for _ in range(27)]
    projection_grad=[[] for _ in range(108)];bias_grad=[[] for _ in range(27)]
    max_variance_error=0.;selected_cells=[]
    pairs=(('mean_NLL','mean_MAE'),('mean_NLL','mean_Huber_delta1'),('mean_MAE','mean_Huber_delta1'))
    for b in range(8):
        counts=[sum(q['target_mask'][at(b,c,h,f)] for h in range(128) for f in range(9)) for c in range(3)]
        nchannel=sum(x>0 for x in counts);a.ck(nchannel>0,'Every detailed query example eligible')
        offset=36 if version=='v5' else 32;z=q['native'][b*width+offset:b*width+offset+4]
        row_grad={n:[[] for _ in range(4)] for n in all_names}
        for c in range(3):
            if not counts[c]:continue
            factor=1/(8*nchannel*counts[c])
            for f in range(9):
                cf=c*9+f;raw=math.fsum(z[d]*projection[cf*4+d] for d in range(4))+bias[cf]
                rebuilt=eps+coefficient*softplus(raw+shift)/base;link=coefficient*sigmoid(raw+shift)/base
                mean_local={n:[[] for _ in range(4)] for n in mean_names};var_derivatives=[]
                for h in range(128):
                    i=at(b,c,h,f);v=float(q['variance'][i])
                    max_variance_error=max(max_variance_error,abs(rebuilt-v))
                    a.close(rebuilt,v,'Saved variance/F64 reconstructed link',absolute=5e-6,relative=5e-5)
                    if not q['target_mask'][i]:continue
                    e=float(q['mean'][i])-float(q['targets'][i])
                    derivatives={'mean_NLL':factor*e/v,'mean_MAE':factor*((e>0)-(e<0)),
                                 'mean_Huber_delta1':factor*max(-1.,min(1.,e))}
                    for name,g in derivatives.items():
                        cf_abs[cf][name].append(abs(g))
                        for d in range(4):mean_local[name][d].append(g*decoder[((c*128+h)*9+f)*4+d])
                    var_derivatives.append(factor*.5*(1/v-e*e/(v*v)))
                    cf_e[cf].append(abs(e));cf_inv[cf].append(1/v)
                    if len(selected_cells)<16:selected_cells.append((e,v))
                raw_g=math.fsum(var_derivatives)*link;bias_grad[cf].append(raw_g)
                for d in range(4):projection_grad[cf*4+d].append(raw_g*z[d])
                for name in all_names:
                    values=[math.fsum(mean_local[name][d]) for d in range(4)] if name in mean_local else [raw_g*projection[cf*4+d] for d in range(4)]
                    local_by_cf[cf][name].extend(values)
                    for d in range(4):row_grad[name][d].append(values[d])
        values={name:[math.fsum(v) for v in row_grad[name]] for name in all_names}
        for name in all_names:flat[name].extend(values[name])
        expected={'row_id':q['row_ids'][b],'mean4_native_gradients':{n:values[n] for n in mean_names},
                  'applied_variance_into_mean4_gradient':[0.]*4,
                  'applied_total_NLL_mean4_gradient':values['mean_NLL'],
                  'applied_uncertainty4_NLL_gradient':values['variance_route'] if version=='v5' else None,
                  'hypothetical_detached_mean4_variance_sensitivity':values['variance_route'] if version=='v4' else None,
                  'mean4_same_basis_comparisons':{left+'_vs_'+right:vector(values[left],values[right]) for left,right in pairs}}
        a.tree(saved['rows'][b],expected,'Independent local disjoint adjoints '+prefix+str(b))
    route={'coordinate_basis':'independent-uncertainty4' if version=='v5' else 'detached-mean4',
           'applied_native_gradient':version=='v5',
           'applied_uncertainty4_l2':norm(flat['variance_route']) if version=='v5' else None,
           'hypothetical_detached_mean4_sensitivity_l2':norm(flat['variance_route']) if version=='v4' else None}
    expected={'population':8,'stage':'initial' if prefix.endswith('initial') else 'final',
              'MAE_zero_subgradient':0.,'Huber_delta_standardized_target_units':1.,'canonical_order':[0,1,2],
              'mean4_flattened_gradient_l2':{n:norm(flat[n]) for n in mean_names},
              'mean4_same_basis_objective_comparisons':{left+'_vs_'+right:vector(flat[left],flat[right]) for left,right in pairs},
              'applied_variance_into_mean4_l2':0.,'applied_mean_into_uncertainty4_l2':0. if version=='v5' else None,
              'stored_variance_consistency_max_abs_error':max_variance_error,'variance_route':route,
              'variance_head_local_gradient_l2':{'projection':norm([math.fsum(x) for x in projection_grad]),'bias':norm([math.fsum(x) for x in bias_grad])}}
    a.tree(saved,expected,'Independent flattened local native adjoints '+prefix)
    for cf in range(27):
        expected_cf={'channel_id':cf//9,'feature':cf%9,'target_cells':len(cf_e[cf]),
                     'mean_inverse_variance':average(cf_inv[cf]),'mean_absolute_error':average(cf_e[cf]),
                     'sum_abs_NLL_mean_derivative_hierarchical':math.fsum(cf_abs[cf]['mean_NLL']),
                     'sum_abs_MAE_mean_derivative_hierarchical':math.fsum(cf_abs[cf]['mean_MAE']),
                     'sum_abs_Huber_delta1_mean_derivative_hierarchical':math.fsum(cf_abs[cf]['mean_Huber_delta1']),
                     'mean4_flattened_gradient_l2':{n:norm(local_by_cf[cf][n]) for n in mean_names},
                     'variance_route_local_sensitivity_l2':norm(local_by_cf[cf]['variance_route'])}
        a.tree(saved['by_channel_feature'][cf],expected_cf,'Independent CF adjoints '+prefix+str(cf))
    for index,(e,v) in enumerate(selected_cells):
        h=min(1e-6*max(1.,abs(e)),.25*abs(e)) if e else 1e-6
        likelihood=lambda error,variance:.5*(error*error/variance+math.log(variance))
        huber=lambda error:.5*error*error if abs(error)<=1 else abs(error)-.5
        a.finite((likelihood(e+h,v)-likelihood(e-h,v))/(2*h),e/v,'NLL mean finite difference '+prefix+str(index))
        a.finite((abs(e+h)-abs(e-h))/(2*h),float((e>0)-(e<0)),'MAE finite difference '+prefix+str(index))
        a.finite((huber(e+h)-huber(e-h))/(2*h),max(-1.,min(1.,e)),'Huber1 finite difference '+prefix+str(index))
        hv=1e-5*v
        a.finite((likelihood(e,v+hv)-likelihood(e,v-hv))/(2*hv),.5*(1/v-e*e/(v*v)),'NLL variance finite difference '+prefix+str(index))
    return {'mean_norms':{n:norm(flat[n]) for n in mean_names},'variance_route':route,
            'same_basis_comparisons':{left+'_vs_'+right:vector(flat[left],flat[right]) for left,right in pairs}}

def main():
    p=argparse.ArgumentParser()
    for n in ('card','diagnosis','audit-card','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    if not Path('/.dockerenv').is_file():raise RuntimeError('Existing managed container required')
    if args.output.exists():raise RuntimeError('Exclusive audit output required')
    started=time.monotonic()
    card=load(args.card);declaration=load(args.audit_card)
    if declaration['kind']!='cpb_v4_v5_TRAIN_diagnosis_independent_audit_card_v1' or declaration['main_card']['sha256']!=MAIN_CARD_SHA:
        raise ValueError('Frozen independent diagnosis-audit declaration')
    if digest(args.card)!=MAIN_CARD_SHA or digest(args.diagnosis)!=declaration['main_diagnosis']['sha256']:
        raise ValueError('Frozen main inputs changed')
    if args.diagnosis.stat().st_size!=declaration['main_diagnosis']['bytes']:
        raise ValueError('Frozen diagnosis size changed')
    for entry in declaration['auditor_sources']:
        actual=pin(Path(__file__).parent/entry['file'])
        if (actual['bytes'],actual['sha256'])!=(entry['bytes'],entry['sha256']):raise ValueError('Frozen independent source changed')
    frozen_files=[args.card,args.diagnosis,args.audit_card,Path(__file__),Path(__file__).with_name('archive_reader.py')]
    frozen_before=[pin(path) for path in frozen_files]
    report=load(args.diagnosis);a=Audit(card)
    try:
        a.ck(report['status']=='passed' and report['baseline']=='CPB-v3.market-01' and report['comparison_references']==['CPB-v4.market-01','CPB-v5.market-01'],'Diagnosis identity')
        a.ck(all(value==0 for value in report['counters'].values()),'No forbidden execution or held-out analysis')
        a.tree(report['mixed_summary_JSON_policy'],{'whole_files_parsed':4,'selected_numeric_key':'train','VALIDATION_numeric_fields_used':0,'files_contain_both_splits':True},'Mixed JSON scope')
        target=a.array('train_targets','targets',(112,4));valid=a.array('train_targets','valid',(112,4))
        ids=a.array('train_targets','row_ids',(112,));groups=a.array('train_targets','group_ids',(112,))
        a.ck(len(set(ids))==112 and ids==sorted(ids) and all(1704067259999<=x<=1705881599999 for x in ids),'Chronological TRAIN ids')
        results={}
        for version in ('v4','v5'):
            for stage in ('initial','final'):
                prefix=version+'_'+stage;saved=report['models'][version]['stages'][stage]
                x,fv,columns=feature_audit(a,prefix+'_features',saved['native_features'],ids,target,valid)
                means=head_audit(a,prefix,saved,x,fv,ids,groups,target,valid)
                q,cf,summary=query_audit(a,prefix,saved['fixed_query_TRAIN_diagnosis'],ids)
                g=gradient_audit(a,prefix,saved['analytic_local_native_gradient_diagnosis'],q)
                results[prefix]={'x':x,'valid':fv,'means':means,'q':q,'cf':cf,'summary':summary,'g':g}
        a.ck(all(results['v4_initial']['x'][r*36+d]==results['v5_initial']['x'][r*40+d]
                 for r in range(112) for d in range(36)),'Exact initial TRAIN common36 pairing')
        extra=[results['v5_initial']['x'][r*40+d] for r in range(112) for d in range(36,40)]
        a.ck(all(math.isfinite(x) for x in extra) and any(x!=0 for x in extra),'New uncertainty4 initial values finite/nonzero')
        for key in ('mean','variance'):a.ck(results['v4_initial']['q'][key]==results['v5_initial']['q'][key],'Exact initial masked pairing '+key)
        a.ck(all(results['v4_initial']['q']['native'][r*36+d]==results['v5_initial']['q']['native'][r*40+d]
                 for r in range(8) for d in range(36)),'Exact masked initial common36')
        for version in ('v4','v5'):
            for key in ('target_mask','targets'):a.ck(results[version+'_initial']['q'][key]==results[version+'_final']['q'][key],'Fixed stage query support/targets')
        for stage in ('initial','final'):
            left,right=results['v4_'+stage],results['v5_'+stage]
            for key in ('row_ids','target_mask','targets'):a.ck(left['q'][key]==right['q'][key],'Cross-model query exact '+stage+key)
            block_rows=[r for r in report['comparison']['native_blocks'] if r['stage']==stage]
            for name,lo,hi in BLOCKS:
                saved=next(r for r in block_rows if r['block']==name)
                a.ck(saved['exact_same_block_values']==all(left['x'][r*36+d]==right['x'][r*40+d] for r in range(112) for d in range(lo,hi)),'Cross-model same native block')
                a.close(saved['mean_absolute_coordinate_difference'],average([abs(left['x'][r*36+d]-right['x'][r*40+d]) for r in range(112) for d in range(lo,hi)]),'Cross-model native mean difference')
                for metric in ('covariance_trace','covariance_frobenius_squared','participation_ratio'):
                    l=report['models']['v4']['stages'][stage]['native_features']['blocks'][name][metric]
                    r=report['models']['v5']['stages'][stage]['native_features']['blocks'][name][metric]
                    a.close(saved[metric+'_v5_minus_v4'],r-l if l is not None and r is not None else None,'Cross-model covariance difference')
            for row in (r for r in report['comparison']['saved_TRAIN_head_scores'] if r['stage']==stage):
                l=left['means'][row['task']][row['head']];r=right['means'][row['task']][row['head']]
                a.tree(row['v5_minus_v4'],{k:r[k]-l[k] if l[k] is not None and r[k] is not None else None for k in ('r2','mae','rmse','coverage')},'Cross-model TRAIN score difference')
            row=next(r for r in report['comparison']['masked_reconstruction'] if r['stage']==stage)
            a.close(row['all112_MAE_v5_minus_v4'],right['summary']['standardized_fixed_query_MAE']-left['summary']['standardized_fixed_query_MAE'],'Cross-model TRAIN MAE difference')
            a.close(row['all112_NLL_v5_minus_v4'],right['summary']['Gaussian_NLL_F64']-left['summary']['Gaussian_NLL_F64'],'Cross-model TRAIN NLL difference')
            for actual,l,r in zip(row['B8_channel_features'],left['cf'],right['cf']):
                a.tree(actual['v5_minus_v4'],{k:r[k]-l[k] for k in ('mean_absolute_error','mean_squared_error','mean_variance','mean_inverse_variance','mean_half_squared_error_over_variance','mean_half_log_variance','mean_Gaussian_NLL')},'Cross-model B8 CF difference')
            local=next(x for x in report['comparison']['local_native_adjoints'] if x['stage']==stage)
            a.tree(local,{'v4_mean4_norms':left['g']['mean_norms'],'v5_mean4_norms':right['g']['mean_norms'],
                          'v4_variance_route':left['g']['variance_route'],'v5_variance_route':right['g']['variance_route'],
                          'v4_mean4_same_basis_objective_comparisons':left['g']['same_basis_comparisons'],
                          'v5_mean4_same_basis_objective_comparisons':right['g']['same_basis_comparisons'],
                          'variance_into_mean4_applied_l2_both':[0.,0.]},'Reported local adjoint comparison')
            a.ck(local['cross_basis_comparison'].startswith('Not computed: independent mean4/uncertainty4'),
                 'Independent-basis cosine/orthogonality/conflict is not computed')
        for accessed in report['numeric_fields_accessed']:
            a.ck(accessed['id'] in a.allowed,'Producer only allowed input id')
            fields=a.allowed[accessed['id']].get('allowed_tensor_fields',[])
            a.ck(accessed['field']=='train_only' or accessed['field'] in fields,'Producer only allowed tensor field')
        for binding in card['allowed_inputs']+card['preservation_pins']+[card['pre_execution_preservation']]:
            a.ck(pin(binding['container_path'])['sha256']==binding['sha256'],'All inputs remain unchanged')
        a.ck([pin(path) for path in frozen_files]==frozen_before,'Card/report/auditor bytes remain exact after arithmetic')
        a.ck(a.finite_checks==256,'Complete prespecified scalar finite-difference coverage')
        result={'kind':'cpb_v4_v5_TRAIN_diagnosis_independent_audit_v1','status':'passed' if not a.errors else 'failed',
            'checks':a.checks,'mismatches':a.errors,'source':pin(__file__),'reader':pin(Path(__file__).with_name('archive_reader.py')),
            'card':pin(args.card),'diagnosis':pin(args.diagnosis),'audit_card':pin(args.audit_card),
            'finite_difference_checks':a.finite_checks,'finite_difference_max_absolute_error':a.maximum_finite_error,
            'finite_difference_policy':'First16 supported cells in canonical b,c,f,h traversal per stage; NLL mean/variance, MAE,Huber1 scalar contracts; fixed positive step rules',
            'verified_inputs':a.verified,'numeric_fields_accessed':a.numeric,
            'native_dimensions':[36,40],'all4_task_ridge_blocks_checked':True,'independent_basis_cosines_computed':0,
            'four_model_stages':4,'TRAIN_rows_each':112,'detailed_query_rows_each':8,
            'model_calls':0,'neural_forwards':0,'fits':0,'training_updates':0,'held_out_tensor_reads':0,
            'scope':'Independent source implementation, saved data and F64 algebra only; never imports diagnosis code, Torch or numerical frameworks.',
            'limits':['Scalar finite differences verify local likelihood kernels; full encoder/optimizer Jacobians are not replayed.',
                      'Detailed query/gradient evidence remains first8 TRAIN rows per stage.',
                      'Four JSON summary files are parsed whole but only train values accessed.',
                      'Saved coefficient reconstruction and population moments are descriptive checks, not fitting.',
                      'No candidate/objective/coefficient or validation gate is selected.'],
            'elapsed_seconds':time.monotonic()-started}
        with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({'status':result['status'],'checks':a.checks,'mismatches':a.errors,'finite_difference_checks':a.finite_checks,'output':str(args.output)}))
        return 0 if not a.errors else 1
    finally:a.close_archives()

if __name__=='__main__':raise SystemExit(main())


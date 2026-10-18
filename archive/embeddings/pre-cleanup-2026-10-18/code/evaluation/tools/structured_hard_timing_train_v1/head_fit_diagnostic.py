#!/usr/bin/env python3
"""Replay saved TRAIN heads and describe margins; no fitting or held-out arrays."""
import argparse
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys
import tempfile
import os

sys.dont_write_bytecode=True
ROOT=Path('/embedding')
TOOLS=ROOT/'code/evaluation/tools/structured_hard_timing_train_v1'
CAPSULE=ROOT/'output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS'
RUN_ROOT=ROOT/'output/runs/rpb-structured-hard-timing-train'
CARD=ROOT/'code/evaluation/cards/structured_hard_timing_train_diagnostic_v1.md'
PROTOCOL='structured-hard-timing-train-diagnostic-v1'
MASTERS=[75272,76373,77474,78575,79676]
METHODS=['raw','native_late','native_early']
WIDTHS={'raw':576,'native_late':32,'native_early':32}
REPS=[2701,2802,2903]
QUANTILES=[0,.05,.25,.5,.75,.95,1]
INVENTORY_SHA='9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa'
SUMMARY_SHA='1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4'
ROLES_SHA='96521655c9b2f8266d94db79672791f0ef12af2d7fb08215bf62a7fe3635a849'
MODULE_PINS={'archive_codec.py':'4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d',
             'saved_cpu_math.py':'1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9'}
CHECKS=ARCHIVES=0

def require(ok,why):
    global CHECKS
    CHECKS+=1
    if not ok:raise ValueError(why)

def sha(body):return hashlib.sha256(body).hexdigest()

def admit(paths):
    seen=set()
    for p in paths:
        require(p.is_absolute() and p.resolve(strict=True)==p and all(not x.is_symlink() for x in (p,*p.parents)), 'direct canonical role')
        s=p.stat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and (s.st_dev,s.st_ino) not in seen,'distinct regular one-link role')
        seen.add((s.st_dev,s.st_ino))

def expected_roles():
    roles=[]
    for master in MASTERS:
        base=f'results/seed-{master}-lag_sign'
        roles.append((base+'/controlled-training.pt','controlled_training',master,None,None))
        for method in METHODS:
            path=base+'/lag_sign/readouts/'+method
            roles.append((path+'/training-features.pt','features',master,method,None))
            for rep in REPS:
                for role,name in [('fit','fit.pt'),('predictions','training-predictions.pt')]:
                    roles.append((path+f'/rep-{rep}/'+name,role,master,method,rep))
    return sorted(roles)

def module(name):
    p=TOOLS/name;require(sha(p.read_bytes())==MODULE_PINS[name],'unchanged saved arithmetic module')
    spec=importlib.util.spec_from_file_location('_head_'+p.stem,p);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def distribution(values):
    require(all(math.isfinite(x) for x in values),'finite diagnostic gaps')
    if not values:return {'count':0,'mean':None,'population_SD':None,'min':None,'max':None,'quantiles':{str(p):None for p in QUANTILES}}
    ordered=sorted(values);q={}
    for p in QUANTILES:
        x=(len(ordered)-1)*p;i=math.floor(x);j=math.ceil(x)
        q[str(p)]=ordered[i]+(ordered[j]-ordered[i])*(x-i)
    average=math.fsum(values)/len(values)
    sd=math.sqrt(math.fsum((x-average)**2 for x in values)/len(values))
    return {'count':len(values),'mean':average,'population_SD':sd,'min':ordered[0],'max':ordered[-1],'quantiles':q}

def describe(logits,predictions,valid,labels,ids):
    n=len(labels)
    require(len(logits)==len(predictions)==len(valid)==len(ids)==n and n>0,'row population')
    require(all(len(x)==2 and all(math.isfinite(v) for v in x) for x in logits),'finite binary saved logits')
    require(all(predictions[i]==int(logits[i][1]>logits[i][0]) for i in range(n)),'classes from own saved logits, first-class ties')
    groups={}
    for i,s in enumerate(ids):
        require(isinstance(s,str) and bool(s) and labels[i] in (0,1),'legal TRAIN labels/sources')
        groups.setdefault(s,[]).append(i)
    gaps=[x[1]-x[0] for x in logits]
    signed=[(2*y-1)*g for y,g in zip(labels,gaps)]
    selected=[i for i,v in enumerate(valid) if v]
    pairs=[];contrasts=[]
    for source,indices in sorted(groups.items()):
        require(len(indices)==2 and {labels[i] for i in indices}=={0,1},'two opposite-label variants per source')
        i0=next(i for i in indices if labels[i]==0);i1=next(i for i in indices if labels[i]==1)
        supported=bool(valid[i0] and valid[i1]);contrast=gaps[i1]-gaps[i0]
        pair={'source':source,'label0_row':i0,'label1_row':i1,'supported':supported,
              'gap_label0':gaps[i0] if supported else None,'gap_label1':gaps[i1] if supported else None,
              'label1_minus_label0_gap':contrast if supported else None,
              'strict_correct_order':contrast>0 if supported else None,
              'strict_opposite_signs':((gaps[i0]<0<gaps[i1]) or (gaps[i1]<0<gaps[i0])) if supported else None,
              'both_correct':predictions[i0]==0 and predictions[i1]==1 if supported else None,
              'same_predicted_class':predictions[i0]==predictions[i1] if supported else None}
        pairs.append(pair)
        if supported:contrasts.append(contrast)
    supported_pairs=[x for x in pairs if x['supported']]
    count=len(selected);pair_count=len(supported_pairs)
    metrics={}
    for key in ('strict_correct_order','strict_opposite_signs','both_correct','same_predicted_class'):
        total=sum(bool(x[key]) for x in supported_pairs)
        metrics[key]={'count':total,'supported_pair_fraction':total/pair_count if pair_count else None,
                      'full_source_pair_fraction':total/len(pairs)}
    return {'total_rows':n,'valid_rows':count,'correct_rows':sum(predictions[i]==labels[i] for i in selected),
            'accuracy':sum(predictions[i]==labels[i] for i in selected)/count if count else None,'coverage':count/n,
            'full_population_correctness':sum(predictions[i]==labels[i] for i in selected)/n,
            'class_valid_rows':[sum(labels[i]==c for i in selected) for c in (0,1)],
            'true_class_margin':distribution([signed[i] for i in selected]),
            'true_class_margin_sign_counts':{'positive':sum(signed[i]>0 for i in selected),
                                            'zero':sum(signed[i]==0 for i in selected),'negative':sum(signed[i]<0 for i in selected)},
            'source_pairs':len(pairs),'supported_source_pairs':pair_count,'source_pair_coverage':pair_count/len(pairs),
            'source_pair_contrast':distribution(contrasts),'source_pair_metrics':metrics,
            'rows':{'sources':ids,'labels':labels,'valid':valid,'classes':predictions,'saved_logit_gap':gaps,
                    'true_class_margin':signed},'pairs':pairs}

def frozen_score_matches(diagnostic,saved):
    for key,actual in [('total',diagnostic['total_rows']),('valid',diagnostic['valid_rows']),('correct',diagnostic['correct_rows'])]:
        require(saved[key]==actual,'exact existing TRAIN score '+key)
    require(saved['accuracy']==diagnostic['accuracy'] and saved['coverage']==diagnostic['coverage'],'exact existing TRAIN conditional score')

def mean(values):
    return None if any(v is None for v in values) else float(sum(Fraction(v) for v in values)/len(values))

def aggregate(cohorts):
    result=[]
    for method in METHODS:
        for head in ('ridge','tiny_secondary'):
            per=[]
            for cohort in cohorts:
                reps=next(x for x in cohort['methods'] if x['method']==method)['repetitions']
                per.append({'master':cohort['master'],
                            **{view:mean([x['heads'][head][view] for x in reps]) for view in ('TRAIN_accuracy','validation_intact_accuracy','validation_deleted_accuracy')},
                            'TRAIN_coverage':mean([x['heads'][head]['TRAIN']['coverage'] for x in reps]),
                            'validation_intact_coverage':mean([x['heads'][head]['validation_intact_metadata']['coverage'] for x in reps]),
                            'validation_deleted_coverage':mean([x['heads'][head]['validation_deleted_metadata']['coverage'] for x in reps]),
                            'TRAIN_source_order_fraction':mean([x['heads'][head]['TRAIN']['source_pair_metrics']['strict_correct_order']['supported_pair_fraction'] for x in reps]),
                            'TRAIN_both_correct_pair_fraction':mean([x['heads'][head]['TRAIN']['source_pair_metrics']['both_correct']['supported_pair_fraction'] for x in reps])})
            item={'method':method,'head':head,'per_master':per}
            for field in ('TRAIN_accuracy','validation_intact_accuracy','validation_deleted_accuracy','TRAIN_coverage','validation_intact_coverage','validation_deleted_coverage','TRAIN_source_order_fraction','TRAIN_both_correct_pair_fraction'):
                item[field+'_mean']=mean([x[field] for x in per])
            item['TRAIN_minus_intact_accuracy_mean']=None if any(item[k] is None for k in ('TRAIN_accuracy_mean','validation_intact_accuracy_mean')) else item['TRAIN_accuracy_mean']-item['validation_intact_accuracy_mean']
            result.append(item)
    return result

def report(value):
    out=['# TEMPO-3 saved-TRAIN fixed-head diagnosis','',
         'Retained TRAIN heads are replayed without fitting. VALIDATION values are copied from the previously audited metadata; no held-out tensor is read.','',
         'Dataset: **TEMPO-3** · timing · **designed complexity4/5** · variable delay, positive gains, offsets and3-tick channel gaps · TRAIN · retained fixed heads, equal means over all five masters and all three repetitions.','',
         '| Method | Size | Linear head % | Neural head % | Coverage % |',
         '| --- | --- | --- | --- | --- |']
    def pct(v):return 'undefined' if v is None else f'{100*v:.2f}'
    labels={'raw':'Raw data — no encoder','native_late':'RPB-v7.alt-05','native_early':'RPB-v10.alt-05'}
    for name in METHODS:
        heads=[next(x for x in value['summary'] if x['method']==name and x['head']==h) for h in ('ridge','tiny_secondary')]
        out.append('| '+' | '.join([labels[name],str(WIDTHS[name]),pct(heads[0]['TRAIN_accuracy_mean']),pct(heads[1]['TRAIN_accuracy_mean']),pct(heads[0]['TRAIN_coverage_mean'])])+' |')
    out+=['','These are conditional accuracies of the original fitted heads. Coverage, all per-master scores, original saved-logit margins and source-pair denominators remain in JSON. No bootstrap or significance threshold is added.','']
    for view,label in [('validation_intact','intact VALIDATION'),('validation_deleted','extra30% deletion VALIDATION')]:
        out+=['Dataset: **TEMPO-3** · timing · **designed complexity4/5** · same recipe · '+label+' · copied from existing audited metadata.','',
              '| Method | Size | Linear head % | Neural head % | Coverage % |',
              '| --- | --- | --- | --- | --- |']
        for name in METHODS:
            heads=[next(x for x in value['summary'] if x['method']==name and x['head']==h) for h in ('ridge','tiny_secondary')]
            out.append('| '+' | '.join([labels[name],str(WIDTHS[name]),pct(heads[0][view+'_accuracy_mean']),pct(heads[1][view+'_accuracy_mean']),pct(heads[0][view+'_coverage_mean'])])+' |')
        out.append('')
    for head in ('ridge','tiny_secondary'):
        out+=['Dataset: **TEMPO-3** · timing · **designed complexity4/5** · same recipe · all paired TRAIN/VALIDATION cohorts · '+head+'.','',
              '| Master | Raw TRAIN/intact/deleted % | RPB-v7.alt-05 TRAIN/intact/deleted % | RPB-v10.alt-05 TRAIN/intact/deleted % |',
              '| --- | --- | --- | --- |']
        for master in MASTERS:
            cells=[]
            for name in METHODS:
                item=next(x for x in value['summary'] if x['method']==name and x['head']==head)
                row=next(x for x in item['per_master'] if x['master']==master)
                cells.append('/'.join(pct(row[v]) for v in ('TRAIN_accuracy','validation_intact_accuracy','validation_deleted_accuracy')))
            out.append('| '+' | '.join([str(master),*cells])+' |')
        out.append('')
    out+=['True-class margin is (2×label−1)×(saved logit1−saved logit0). Source contrast is gap(label1)−gap(label0); strict order, opposite signs and exact both-correct classes are distinct statistics. Ties retain class0. Unsupported rows and pairs stay in the total denominators; zero-support statistics remain undefined.','',
          'A large TRAIN/VALIDATION gap describes limited generalization by that fixed fitted head; weak TRAIN separation describes limited fitting or head-accessible information. Neither outcome alone proves encoder information loss or identifies an internal architectural cause. Raw fixed-head results are an essential comparator. Saved query-bank timing is a separate diagnostic.','',
          f"CPU archives decoded: {value['archive_decodes']}; no encoder/head/PCA fits, model forward, GPU execution, TEST/stress or historical payload reads.",'']
    return '\n'.join(out)

def run(args):
    global ARCHIVES
    require(Path('/.dockerenv').is_file(),'existing managed container only')
    require(args.roles_sha256==ROLES_SHA,'frozen TRAIN role manifest pin')
    metadata=[TOOLS/'head_roles.json',CAPSULE/'artifact-integrity.json',ROOT/'doc/results/structured_hard_timing_comparison_v2.json',CARD,Path(__file__).resolve(),*[TOOLS/name for name in MODULE_PINS]]
    admit(metadata)
    before_meta=[p.read_bytes() for p in metadata]
    require([sha(b) for b in before_meta[:3]]==[ROLES_SHA,INVENTORY_SHA,SUMMARY_SHA],'sealed input metadata bindings')
    require(sha(before_meta[3])==args.card_sha256 and sha(before_meta[4])==args.source_sha256,'reviewed card/source')
    roles,inventory,prior=map(json.loads,before_meta[:3])
    require(roles['protocol']==PROTOCOL and roles['capsule']==str(CAPSULE),'exact saved TRAIN namespace')
    records=roles['role_records'];wanted=expected_roles()
    require(len(records)==110 and [(x['path'],x['role'],x['master'],x.get('method'),x.get('repetition')) for x in records]==wanted,'closed TRAIN-only matrix')
    entries={x['path']:x for x in inventory['files']}
    require(inventory['status']=='complete' and prior['audit']['status']=='passed' and prior['audit']['source']['capsule']==str(CAPSULE),'completed parent audit association')
    require(prior['audit']['source']['inventory_sha256']==INVENTORY_SHA and prior['audit']['checks']==52934423,'original passed audit kept unchanged')
    paths=[CAPSULE/x['path'] for x in records]
    admit(paths)  # Admit every role before hashing or decoding any payload.
    for record,p in zip(records,paths):
        require(record['sha256']==entries[record['path']]['sha256'] and record['bytes']==entries[record['path']]['bytes']==p.stat().st_size,'exact inventory role association')
    hashes=[sha(p.read_bytes()) for p in paths]
    require(hashes==[x['sha256'] for x in records],'all saved TRAIN bytes sealed before arithmetic')
    output=Path(args.output)
    require(output.parent==RUN_ROOT and output.name.startswith('head-fit-') and not output.exists() and not output.is_symlink(),'exclusive diagnostic leaf')
    ancestor=RUN_ROOT.parent;require(ancestor.resolve(strict=True)==ancestor and all(not p.is_symlink() for p in ancestor.parents),'canonical output ancestors')
    require(not RUN_ROOT.exists() or (RUN_ROOT.is_dir() and RUN_ROOT.resolve(strict=True)==RUN_ROOT and not RUN_ROOT.is_symlink()),'canonical diagnostic root')
    m=module('saved_cpu_math.py');codec=module('archive_codec.py');m.R=codec
    def load(relative):
        global ARCHIVES
        require(relative in entries and relative in {x['path'] for x in records},'only admitted TRAIN payload')
        ARCHIVES+=1;return codec.load(CAPSULE/relative)
    cohorts=[]
    for master in MASTERS:
        base=f'results/seed-{master}-lag_sign'
        controlled=load(base+'/controlled-training.pt')
        require(set(controlled)=={'observations','feature_mask','labels_scoring_only','source_ids_json'},'literal controlled TRAIN writer schema')
        labels=list(m.tensor(controlled['labels_scoring_only'],'LongStorage',[256]));ids=json.loads(m.text(controlled,'source_ids_json'))
        masks=list(m.tensor(controlled['feature_mask'],'BoolStorage',[256,3,32,3]));m.tensor(controlled['observations'],'DoubleStorage',[256,3,32,3])
        groups=m.validate_pairs(labels,ids,[masks[i*288:(i+1)*288] for i in range(256)]);require(len(groups)==128,'all TRAIN source pairs')
        audit=next(x for x in prior['audit']['per_master'] if x['timing_master']==master)
        task=next(x for x in audit['tasks'] if x['task']=='lag_sign')
        cohort={'master':master,'methods':[]}
        for name in METHODS:
            path=base+'/lag_sign/readouts/'+name;width=WIDTHS[name];prepared=name=='raw'
            rows,valid=m.feature_archive(load(path+'/training-features.pt'),256,width,ids,labels,'DoubleStorage' if prepared else 'FloatStorage')
            saved_method=next(x for x in task['methods'] if x['method']==name)
            require(saved_method['status']=='measured','exact previously fitted method')
            method={'method':name,'width':width,'prepared_outer_identity':prepared,'repetitions':[]}
            for rep in REPS:
                repid=f'rep-{rep}';fit=load(path+'/'+repid+'/fit.pt');pred=load(path+'/'+repid+'/training-predictions.pt')
                fitted=m.fit_schema(fit,width,ids,rep,prepared,rows,valid)
                ridge,tiny,_=m.infer_saved_fit(fit,rows,valid,labels,ids,pred,prepared,width,fitted)
                previous=next(x for x in saved_method['repetitions'] if x['repetition']==repid)
                result={'repetition':rep,'actual_probe_seed_decimal':m.text(fit,'actual_probe_seed_decimal'),'heads':{}}
                for head,classes,key in [('ridge',ridge,'ridge_logits'),('tiny_secondary',tiny,'tiny_logits')]:
                    values=list(m.tensor(pred[key],'DoubleStorage',[256,2]));logits=[values[i*2:(i+1)*2] for i in range(256)]
                    diagnostic=describe(logits,classes,valid,labels,ids);frozen_score_matches(diagnostic,previous['training'][head])
                    result['heads'][head]={'TRAIN':diagnostic,'TRAIN_accuracy':diagnostic['accuracy'],
                        'validation_intact_accuracy':previous['validation_intact'][head]['accuracy'],
                        'validation_deleted_accuracy':previous['validation_deleted'][head]['accuracy'],
                        'validation_intact_metadata':copy.deepcopy(previous['validation_intact'][head]),
                        'validation_deleted_metadata':copy.deepcopy(previous['validation_deleted'][head])}
                method['repetitions'].append(result)
            cohort['methods'].append(method)
        cohorts.append(cohort)
    require(ARCHIVES==110,'exact closed archive count, no duplicate views')
    require([sha(p.read_bytes()) for p in paths]==hashes and [p.read_bytes() for p in metadata]==before_meta,'all saved inputs/source preserved')
    value={'status':'passed','protocol':PROTOCOL,'dataset':'TEMPO-3','designed_complexity_level':4,'complexity_scale_max':5,
           'source_sha256':args.source_sha256,'card_sha256':args.card_sha256,'roles_sha256':ROLES_SHA,
           'capsule':str(CAPSULE),'inventory_sha256':INVENTORY_SHA,'validation_metadata_sha256':SUMMARY_SHA,
           'arithmetic_modules':MODULE_PINS,'tolerance':{'absolute':2e-9,'relative':2e-9},'quantiles':QUANTILES,
           'margin_source':'original saved logits; exact own argmax with first-class ties','pair_order':'lexical source ID',
           'cohorts':cohorts,'summary':aggregate(cohorts),'checks':CHECKS+m.CHECKS+codec.CHECKS,'archive_decodes':ARCHIVES,
           'limits':{'encoder_updates':0,'head_refits':0,'PCA_fits':0,'bootstrap_replays':0,'model_forward':False,'GPU_execution':False,
                     'heldout_tensor_reads':0,'TEST_or_stress_reads':0,'historical_payload_reads':0,'all_inputs_preserved':True}}
    RUN_ROOT.mkdir(exist_ok=True);output.mkdir()
    with (output/'head-fit-diagnostic.json').open('x',encoding='utf-8',newline='\n') as out:out.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    with (output/'HEAD_FIT_DIAGNOSTIC.md').open('x',encoding='utf-8',newline='\n') as out:out.write(report(value))
    print(json.dumps({'status':'passed','checks':value['checks'],'archive_decodes':ARCHIVES,'output':str(output)}))

def self_test():
    m=module('saved_cpu_math.py');codec=module('archive_codec.py');m.R=codec
    require(len(expected_roles())==110 and all('validation' not in x[0] for x in expected_roles()),'closed TRAIN fixture roles')
    ids=['b','b','a','a'];labels=[0,1,0,1];valid=[True]*4
    known=[[-1,1],[1,-1],[-1,-1],[1,1]]
    # Literal saved-fit writer keys; two independent, standardized coordinates.
    fit={};width=2
    for key in ('feature_mean','ridge_mean','tiny_mean'):fit[key]=m.fake('DoubleStorage',[width],[0,0])
    for key in ('feature_scale','ridge_scale','tiny_scale'):fit[key]=m.fake('DoubleStorage',[width],[1,1])
    fit.update({'outer_normalizer_applied':m.fake('BoolStorage',[],[False]),'outer_fitted_rows':m.fake('LongStorage',[],[0]),
                'fitted_rows':m.fake('LongStorage',[],[4]),'ridge_weights':m.fake('DoubleStorage',[2,2],[-1,1,0,0]),
                'ridge_intercept':m.fake('DoubleStorage',[2],[0,0]),'tiny_w1':m.fake('DoubleStorage',[2,16],[1]+[0]*31),
                'tiny_b1':m.fake('DoubleStorage',[16],[0]*16),'tiny_w2':m.fake('DoubleStorage',[16,2],[-1,1]+[0]*30),
                'tiny_b2':m.fake('DoubleStorage',[2],[0,0]),'actual_probe_seed_decimal':m.fake_text(str(m.stream_seed(2701,2))),
                'training_source_ids_json':m.fake_text(json.dumps(ids)),'ridge_penalty':m.fake('DoubleStorage',[],[1]),
                'tiny_hidden':m.fake('LongStorage',[],[16]),'tiny_steps':m.fake('LongStorage',[],[100]),
                'tiny_learning_rate':m.fake('DoubleStorage',[],[.01])})
    ridge_logits=[[-x[0],x[0]] for x in known];tiny_logits=[[-math.tanh(x[0]),math.tanh(x[0])] for x in known]
    pred={'ridge':m.fake('LongStorage',[4],labels),'tiny_secondary':m.fake('LongStorage',[4],labels),
          'valid':m.fake('BoolStorage',[4],valid),'probe_input_features':m.fake('DoubleStorage',[4,2],[v for x in known for v in x]),
          'ridge_logits':m.fake('DoubleStorage',[4,2],[v for x in ridge_logits for v in x]),
          'tiny_hidden_preactivation':m.fake('DoubleStorage',[4,16],[v for x in known for v in ([x[0]]+[0]*15)]),
          'tiny_logits':m.fake('DoubleStorage',[4,2],[v for x in tiny_logits for v in x]),
          'labels_scoring_only':m.fake('LongStorage',[4],labels),'source_ids_json':m.fake_text(json.dumps(ids))}
    require(m.fit_schema(fit,2,ids,2701,True,known,valid)==4,'retained synthetic fit schema')
    classes=m.infer_saved_fit(fit,known,valid,labels,ids,pred,True,2,4)
    for logits,predictions in [(ridge_logits,classes[0]),(tiny_logits,classes[1])]:
        d=describe(logits,predictions,valid,labels,ids)
        require(d['accuracy']==1 and d['source_pair_metrics']['both_correct']['count']==2 and d['pairs'][0]['source']=='a','known fit exact/source order')
    require(distribution([0,10])['quantiles']['0.25']==2.5,'linear quantile law')
    require(distribution([0,10])['population_SD']==5 and distribution([0,10])['min']==0 and distribution([0,10])['max']==10,'population dispersion and bounds')
    zero=describe([[0,0]]*4,[0]*4,[False]*4,labels,ids)
    require(zero['accuracy'] is None and zero['source_pair_metrics']['both_correct']['supported_pair_fraction'] is None and zero['source_pairs']==2,'no support stays undefined/denominators retained')
    tie=describe([[0,0],[0,1],[0,-1],[0,1]],[0,1,0,1],valid,labels,ids)
    require(tie['accuracy']==1 and tie['true_class_margin_sign_counts']['zero']==1 and tie['source_pair_metrics']['strict_opposite_signs']['count']==1,'zero margin distinct from exact class/tie')
    tiny=describe([[1e-200,0],[0,1e-200]]*2,labels,valid,labels,ids)
    require(tiny['source_pair_metrics']['strict_opposite_signs']['count']==2,'tiny opposite signs do not rely on underflowing products')
    negative=0
    def rejects(fn):
        nonlocal negative
        try:fn()
        except (ValueError,AssertionError,KeyError,OSError):negative+=1;return
        raise AssertionError('negative fixture did not reject')
    for index in range(4):
        bad=copy.deepcopy(pred);bad['ridge']['values'][index]=1-bad['ridge']['values'][index]
        rejects(lambda bad=bad:m.infer_saved_fit(fit,known,valid,labels,ids,bad,True,2,4))
    bad=copy.deepcopy(pred);bad['ridge_logits']['values'][0]+=.01
    rejects(lambda:m.infer_saved_fit(fit,known,valid,labels,ids,bad,True,2,4))
    badfit=copy.deepcopy(fit);badfit['ridge_scale']['values'][0]=0
    rejects(lambda:m.infer_saved_fit(badfit,known,valid,labels,ids,pred,True,2,4))
    rejects(lambda:describe([[0,float('nan')]]*4,[0]*4,valid,labels,ids))
    rejects(lambda:describe(ridge_logits,labels,valid,[0,0,0,1],ids))
    rejects(lambda:frozen_score_matches(tie,{'total':4,'valid':3,'correct':4,'accuracy':1,'coverage':1}))
    partial=describe(ridge_logits,labels,[False,True,True,True],labels,ids)
    require(partial['valid_rows']==3 and partial['source_pairs']==2 and partial['supported_source_pairs']==1,'partial pair stays in full denominator')
    require(partial['source_pair_metrics']['both_correct']['full_source_pair_fraction']==.5,'pair full-population correctness')
    fake_cohorts=[]
    for master in MASTERS:
        methods=[]
        for name in METHODS:
            repetitions=[]
            for rep in REPS:
                head={'TRAIN':tie,'TRAIN_accuracy':tie['accuracy'],'validation_intact_accuracy':.5,'validation_deleted_accuracy':.5,
                      'validation_intact_metadata':{'coverage':1},'validation_deleted_metadata':{'coverage':1}}
                repetitions.append({'repetition':rep,'heads':{'ridge':head,'tiny_secondary':head}})
            methods.append({'method':name,'repetitions':repetitions})
        fake_cohorts.append({'master':master,'methods':methods})
    fake_summary=aggregate(fake_cohorts);rendered=report({'summary':fake_summary,'archive_decodes':0})
    require(rendered.count('| Method | Size | Linear head % | Neural head % | Coverage % |')==3,'three standard split panels')
    require(all(x['TRAIN_accuracy_mean']==1 and x['validation_intact_accuracy_mean']==.5 for x in fake_summary),'equal-master/equal-repetition means')
    with tempfile.TemporaryDirectory(prefix='head-source-fixture-') as temp:
        base=Path(temp);one=base/'one';two=base/'two';one.write_bytes(b'one');two.write_bytes(b'two')
        admit([one,two])
        rejects(lambda:admit([one,one]))
        os.link(one,base/'hardlink');rejects(lambda:admit([one,two]))
        (base/'hardlink').unlink()
        (base/'symlink').symlink_to(two);rejects(lambda:admit([one,base/'symlink']))
        rejects(lambda:admit([one,base/'absent']))
    require(ARCHIVES==0,'synthetic archive opens remain zero')
    return {'status':'passed','checks':CHECKS+m.CHECKS+codec.CHECKS,'negative_fixtures':negative,
            'archive_decodes':0,'model_or_head_fits':0,'source_only':True,'module_pins':MODULE_PINS}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true')
    for name in ('card-sha256','roles-sha256','source-sha256','output'):p.add_argument('--'+name)
    a=p.parse_args()
    require(Path('/.dockerenv').is_file(),'managed container only')
    if a.self_test:print(json.dumps(self_test()));return
    require(all(getattr(a,k) for k in ('card_sha256','roles_sha256','source_sha256','output')),'explicit frozen source/card/role pins')
    run(a)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Prospective TRAIN-only saved-query arithmetic; explicit pinned --run only."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
import time

sys.dont_write_bytecode=True
PROTOCOL='structured-hard-timing-train-diagnostic-v1'
REPO=Path('/embedding')
PARENT=REPO/'output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS'
INVENTORY_SHA='9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa'
AUDIT=REPO/'output/runs/rpb-structured-hard-timing/audit-tools/run-xrZMAS-v3-ordering/validation.json'
AUDIT_SHA='cf79abdca511f22778622d2b08e62b74b592c10618d1090ea2bd53eefd84539c'
MASTERS=[75272,76373,77474,78575,79676]
PINS={'archive_codec.py':'4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d',
      'saved_cpu_math.py':'1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9'}
CHECKS=0
R=M=None

def check(ok,reason):
    global CHECKS
    CHECKS+=1
    if not ok:raise AssertionError(reason)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def admit(paths):
    identities=set()
    for path in paths:
        check(path.is_absolute() and path.resolve(strict=True)==path and all(not p.is_symlink() for p in (path,*path.parents)),'canonical direct role')
        info=path.stat();check(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'regular nonhardlinked role')
        check((info.st_dev,info.st_ino) not in identities,'complete role inode uniqueness');identities.add((info.st_dev,info.st_ino))

def modules():
    global R,M
    paths=[Path(__file__).parent/name for name in PINS];admit(paths)
    for name,path in zip(PINS,paths):check(sha(path)==PINS[name],'unchanged pinned CPU SOURCE')
    loaded=[]
    for name,path in zip(PINS,paths):
        spec=importlib.util.spec_from_file_location(name[:-3],path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);loaded.append(mod)
    R,M=loaded;M.R=R

def role_names():
    result=[]
    for master in MASTERS:
        root=f'results/seed-{master}-lag_sign/'
        result.extend(root+n for n in ('controlled-training.pt','late/point-512/training-reconstruction.pt',
                      'early/point-512/training-reconstruction.pt','late/point-512/checkpoint.pt.scaler.pt'))
    return sorted(result,key=PurePosixPath)

def role_records(manifest,parent):
    check(manifest['protocol']==PROTOCOL and manifest['component']=='same_bank_query_train' and
          manifest['parent_inventory_sha256']==INVENTORY_SHA and manifest['capsule']==str(PARENT),'frozen query role manifest')
    records=manifest['files']
    check([r['path'] for r in records]==role_names() and len(records)==20,'exact twenty TRAIN roles')
    check(all(set(r)=={'path','bytes','sha256'} and type(r['bytes']) is int and r['bytes']>0 and
              len(r['sha256'])==64 and all(c in '0123456789abcdef' for c in r['sha256']) for r in records),'closed selected role records')
    check(all(r==parent[r['path']] for r in records),'selected roles exactly immutable inventory')
    return records

def timing(values,support,rows):
    check(len(values)==len(support)==4*rows*288 and rows>0,'complete four-bank query geometry')
    def ix(bank,row,channel,h,f):return (((bank*rows+row)*3+channel)*32+h)*3+f
    for bank in range(4):
        for row in range(rows):
            for channel in range(3):
                for h in range(32):
                    for f in range(3):
                        i=ix(bank,row,channel,h,f)
                        check(not support[i] or (h//8==bank and math.isfinite(values[i])),'legal finite same-bank cells')
    output=[]
    for row in range(rows):
        centres=[];bank_counts=[]
        for bank in range(4):
            count=0
            for centre in range(8*bank+1,8*bank+7):
                spacings=[]
                for k in range(1,5):
                    if centre-k<8*bank or centre+k>=8*bank+8:continue
                    diffs=[]
                    for channel in (0,1):
                        features=[f for f in range(3) if all(support[ix(bank,row,channel,h,f)] for h in (centre-k,centre,centre+k))]
                        if not features:break
                        left=math.fsum(values[ix(bank,row,channel,centre,f)]-values[ix(bank,row,channel,centre-k,f)] for f in features)/len(features)
                        right=math.fsum(values[ix(bank,row,channel,centre+k,f)]-values[ix(bank,row,channel,centre,f)] for f in features)/len(features)
                        diffs.append((left,right))
                    if len(diffs)==2:
                        determinant=diffs[1][0]*diffs[0][1]-diffs[0][0]*diffs[1][1]
                        check(math.isfinite(determinant),'finite legal determinant');spacings.append(determinant)
                if spacings:centres.append(math.fsum(spacings)/len(spacings));count+=1
            bank_counts.append(count)
        margin=math.fsum(centres)/len(centres) if centres else None
        valid=len(centres)>=4 and margin is not None and math.isfinite(margin) and margin!=0
        output.append({'margin':margin,'supported_centres':len(centres),'bank_centres':bank_counts,
                       'valid':valid,'prediction':int(margin>0) if valid else 0})
    return output

def distribution(values):
    check(all(math.isfinite(x) for x in values),'finite descriptive distribution')
    if not values:return {'count':0,'mean':None,'population_sd':None,'minimum':None,'maximum':None,'quantiles':None}
    ordered=sorted(values);mean=math.fsum(values)/len(values)
    def quantile(p):
        at=(len(values)-1)*p;left=math.floor(at);right=math.ceil(at)
        return ordered[left]+(ordered[right]-ordered[left])*(at-left)
    return {'count':len(values),'mean':mean,'population_sd':math.sqrt(math.fsum((x-mean)**2 for x in values)/len(values)),
            'minimum':ordered[0],'maximum':ordered[-1],'quantiles':[quantile(p) for p in (0,.05,.25,.5,.75,.95,1)]}

def summarize(rows,labels,ids):
    check(len(rows)==len(labels)==len(ids)>0,'complete diagnostic row geometry')
    groups={}
    for i,source in enumerate(ids):groups.setdefault(source,[]).append(i)
    pairs=[]
    for source,indices in sorted(groups.items()):
        check(len(indices)==2 and {labels[i] for i in indices}=={0,1},'complete label-independent source pair')
        a=next(i for i in indices if labels[i]==0);b=next(i for i in indices if labels[i]==1)
        supported=rows[a]['valid'] and rows[b]['valid']
        contrast=rows[b]['margin']-rows[a]['margin'] if supported else None
        pairs.append({'source_id':source,'row_label0':a,'row_label1':b,'supported':supported,'margin_contrast':contrast,
                      'positive_order':contrast>0 if supported else None,
                      'opposite_signs':((rows[a]['margin']<0<rows[b]['margin']) or (rows[b]['margin']<0<rows[a]['margin'])) if supported else None,
                      'both_correct':rows[a]['prediction']==0 and rows[b]['prediction']==1 if supported else None,
                      'same_prediction':rows[a]['prediction']==rows[b]['prediction'] if supported else None})
    score=M.score([r['prediction'] for r in rows],labels,[r['valid'] for r in rows]);score['full_population_correctness']=score['correct']/len(rows)
    supported_pairs=[p for p in pairs if p['supported']]
    stats={'total_pairs':len(pairs),'supported_pairs':len(supported_pairs),'pair_coverage':len(supported_pairs)/len(pairs)}
    for key in ('positive_order','opposite_signs','both_correct','same_prediction'):
        stats[key+'_count']=sum(p[key] for p in supported_pairs)
        stats[key+'_supported_fraction']=stats[key+'_count']/len(supported_pairs) if supported_pairs else None
        stats[key+'_full_population_fraction']=stats[key+'_count']/len(pairs)
    return {'score':score,'pair_summary':stats,'margin':distribution([r['margin'] for r in rows if r['valid']]),
            'true_class_margin':distribution([(2*y-1)*r['margin'] for r,y in zip(rows,labels) if r['valid']]),
            'pair_contrast':distribution([p['margin_contrast'] for p in supported_pairs]),'rows':rows,'pairs':pairs}

def common_effect(reference,candidate,labels,ids):
    check(len(reference)==len(candidate)==len(labels)==len(ids)>0,'common-support row geometry')
    common=[i for i in range(len(ids)) if reference[i]['valid'] and candidate[i]['valid']]
    rc=sum(reference[i]['prediction']==labels[i] for i in common)
    cc=sum(candidate[i]['prediction']==labels[i] for i in common)
    grouped={}
    for i,source in enumerate(ids):grouped.setdefault(source,[]).append(i)
    complete=sum(all(i in common for i in indices) for indices in grouped.values())
    return {'total_rows':len(ids),'common_supported_rows':len(common),'common_coverage':len(common)/len(ids),
            'reference_correct_on_common':rc,'candidate_correct_on_common':cc,
            'candidate_minus_reference_accuracy':(cc-rc)/len(common) if common else None,
            'candidate_minus_reference_full_population_correctness':(cc-rc)/len(ids),
            'total_source_pairs':len(grouped),'complete_common_source_pairs':complete}

def aggregate(masters):
    result={}
    for method in ('saved_target','late_prediction','early_prediction'):
        result[method]={}
        for field in ('accuracy','coverage','full_population_correctness'):
            values=[r['methods'][method]['score'][field] for r in masters]
            result[method][field]={'per_master':values,'mean':math.fsum(values)/len(values) if all(v is not None for v in values) else None,
                                   'minimum':min(values) if all(v is not None for v in values) else None,
                                   'maximum':max(values) if all(v is not None for v in values) else None}
    return result

def controlled(asset,master):
    check(set(asset)=={'observations','feature_mask','labels_scoring_only','source_ids_json'},'controlled TRAIN keys')
    values=M.tensor(asset['observations'],'DoubleStorage',[256,3,32,3]);mask=list(M.tensor(asset['feature_mask'],'BoolStorage',[256,3,32,3]))
    labels=list(M.tensor(asset['labels_scoring_only'],'LongStorage',[256]));M.tensor(asset['source_ids_json'],'ByteStorage',[len(asset['source_ids_json']['values'])]);ids=json.loads(R.text(asset['source_ids_json']))
    check(len(ids)==256 and all(type(s) is str and s.startswith(f'structured-hard-timing-comparison-v2/lag_sign/structured-hard-timing-v1/seed-{master}/lag_sign/source-') for s in ids),'exact TRAIN source scope')
    M.finite(values,'finite controlled legal storage');check(all(ok or x==0 for x,ok in zip(values,mask)),'hidden TRAIN zeros')
    check(len(M.validate_pairs(labels,ids,[mask[i*288:(i+1)*288] for i in range(256)]))==128,'fixed TRAIN pair population')
    return values,mask,labels,ids

def data_identifiers(data):
    # Literal workflow.cpp schema_identity/dataset_identity byte order; no raw
    # companion or checkpoint body is opened to establish the association.
    def text(value):return (value+'\0').encode('utf-8')
    raw=text('rpb_raw_uniform_history_v1')+text('unitless,unitless,unitless')
    raw+=b''.join(text(str(v)) for v in (3,32,3,7))+struct.pack('<d',1.)
    raw+=b''.join(text(str(v)) for v in (0,1,2))
    schema='rpb-schema-fnv1a-v1-'+M.fnv(raw)
    raw=text(schema)
    for enum,dtype,shape,values in ((7,'DoubleStorage',[256,3,32,3],data[0]),(11,'BoolStorage',[256,3,32,3],data[1]),
                                   (4,'LongStorage',[3],[0,1,2]),(7,'DoubleStorage',[256],[31.]*256)):
        raw+=b''.join(text(str(v)) for v in (enum,len(shape),*shape))+M.bytes_of(M.fake(dtype,shape,values))
    return schema,'rpb-dataset-fnv1a-v1-'+M.fnv(raw)

def scaler_companion(asset,data):
    check(set(asset)=={'encoder_id','format_version','artifact_kind','schema_id','preprocessing_id','fit_dataset_id','scaler'},'exact nested scaler companion schema')
    check(M.text(asset,'encoder_id')=='raw_patch_bottleneck_mae_v1' and M.scalar(asset,'format_version')==1 and M.text(asset,'artifact_kind')=='rpb_frozen_training_scaler_v1','typed original scaler envelope')
    frozen=M.group(asset['scaler']);mean,scale,identity=M.scaler(frozen)
    schema,dataset=data_identifiers(data)
    check(M.text(asset,'preprocessing_id')==identity and M.text(asset,'schema_id')==schema and M.text(asset,'fit_dataset_id')==dataset,'exact controlled TRAIN/scaler schema and data association')
    for c in range(3):
        for f in range(3):
            selected=[data[0][(row*3+c)*96+h*3+f] for row in range(256) for h in range(32) if data[1][(row*3+c)*96+h*3+f]]
            d=3*c+f;check(bool(selected),'observed TRAIN scaler coordinate support')
            average=math.fsum(selected)/len(selected);sd=math.sqrt(math.fsum((x-average)**2 for x in selected)/len(selected))
            check(frozen['count']['values'][d]==len(selected) and bool(frozen['floor_applied']['values'][d])==(sd<1e-6),'original frozen TRAIN count/floor witness')
            M.close(mean[d],average,'original TRAIN scaler mean');M.close(scale[d],max(1e-6,sd),'original TRAIN scaler population SD')
    return mean,scale,identity

def query(asset,data,mean,scale):
    values,mask,labels,ids=data
    expected={'standardized_prediction','standardized_target','target_mask','requested_observed_target_mask','visible_mask','trial_channel_eligible',
              'channel_target_counts','channel_valid','channel_standardized_mae','channel_standardized_huber','example_valid','example_standardized_mae','example_standardized_huber','source_ids_json'}
    check(set(asset)==expected,'original query writer closed schema')
    pred=list(M.tensor(asset['standardized_prediction'],'DoubleStorage',[4,256,3,32,3]));target=list(M.tensor(asset['standardized_target'],'DoubleStorage',[4,256,3,32,3]))
    q=list(M.tensor(asset['target_mask'],'BoolStorage',[4,256,3,32,3]));requested=list(M.tensor(asset['requested_observed_target_mask'],'BoolStorage',[4,256,3,32,3]))
    visible=list(M.tensor(asset['visible_mask'],'BoolStorage',[4,256,3,32,3]));eligible=list(M.tensor(asset['trial_channel_eligible'],'BoolStorage',[4,256,3]))
    check((requested,visible,q,eligible)==M.query_masks(mask,256),'original observed/Q eligibility and bank masks')
    for key,dtype,shape in (('channel_target_counts','LongStorage',[256,3]),('channel_valid','BoolStorage',[256,3]),
                ('channel_standardized_mae','DoubleStorage',[256,3]),('channel_standardized_huber','DoubleStorage',[256,3]),
                ('example_valid','BoolStorage',[256]),('example_standardized_mae','DoubleStorage',[256]),('example_standardized_huber','DoubleStorage',[256])):
        saved=M.tensor(asset[key],dtype,shape)
        if dtype=='DoubleStorage':M.finite(saved,'finite original query reduction witness')
    M.tensor(asset['source_ids_json'],'ByteStorage',[len(asset['source_ids_json']['values'])])
    check(json.loads(R.text(asset['source_ids_json']))==ids,'query source/row identity')
    raw_pred=[0.]*len(pred);raw_target=[0.]*len(target)
    for i,ok in enumerate(q):
        check(math.isfinite(pred[i]) and math.isfinite(target[i]) and (ok or pred[i]==target[i]==0),'saved finite zero-hidden query storage')
        if ok:
            original=i%(256*288);channel=(original//96)%3;feature=original%3;d=3*channel+feature
            want=M.f32((values[original]-mean[d])/scale[d])
            check(abs(target[i]-want)<=2e-6+2e-5*abs(want),'original TRAIN scaler/F32 target arithmetic')
            raw_pred[i]=pred[i]*scale[d]+mean[d];raw_target[i]=target[i]*scale[d]+mean[d]
    return raw_pred,raw_target,q

def run(args):
    started=time.monotonic();source=Path(__file__);card=Path(args.card);roles=Path(args.roles);inventory=PARENT/'artifact-integrity.json'
    metadata=[source,card,roles,inventory,AUDIT,*[source.parent/name for name in PINS]]
    paths=[PARENT/name for name in role_names()];admit(metadata+paths)
    before={str(p):sha(p) for p in metadata}
    check(before[str(source)]==args.source_sha256 and before[str(card)]==args.card_sha256 and before[str(roles)]==args.roles_sha256,'explicit root SOURCE/card/role pins')
    check(sha(inventory)==INVENTORY_SHA and sha(AUDIT)==AUDIT_SHA,'immutable parent inventory/prior audit provenance; audit not decoded')
    manifest=json.loads(roles.read_text())
    parent={r['path']:r for r in json.loads(inventory.read_text())['files']}
    records=role_records(manifest,parent)
    check(all(p.stat().st_size==r['bytes'] for p,r in zip(paths,records)),'whole selected sizes before hashes')
    for path,row in zip(paths,records):check(sha(path)==row['sha256'],'selected TRAIN bytes')
    modules();index=dict(zip((r['path'] for r in records),paths));results=[]
    for master in MASTERS:
        base=f'results/seed-{master}-lag_sign/';data=controlled(R.load(index[base+'controlled-training.pt']),master)
        mean,scale,identity=scaler_companion(R.load(index[base+'late/point-512/checkpoint.pt.scaler.pt']),data)
        arrays=[query(R.load(index[base+role+'/point-512/training-reconstruction.pt']),data,mean,scale) for role in ('late','early')]
        check(arrays[0][1:]==arrays[1][1:],'paired saved target/support exact; one matched baseline')
        labels,ids=data[2:];methods={'saved_target':summarize(timing(arrays[0][1],arrays[0][2],256),labels,ids),
                                   'late_prediction':summarize(timing(arrays[0][0],arrays[0][2],256),labels,ids),
                                   'early_prediction':summarize(timing(arrays[1][0],arrays[1][2],256),labels,ids)}
        effects={candidate+'_minus_'+reference:common_effect(methods[reference]['rows'],methods[candidate]['rows'],labels,ids)
                 for reference,candidate in (('saved_target','late_prediction'),('saved_target','early_prediction'),('late_prediction','early_prediction'))}
        results.append({'master':master,'scaler_id':identity,'methods':methods,'common_support_effects':effects,'labels_scoring_only':labels,'source_ids':ids})
    for path,row in zip(paths,records):check(sha(path)==row['sha256'],'selected TRAIN bytes preserved after arithmetic')
    check(all(sha(p)==before[str(p)] for p in metadata),'SOURCE/card/role/module/parent metadata preserved after arithmetic')
    return {'status':'passed','protocol':PROTOCOL,'source_sha256':sha(source),'card_sha256':sha(card),'roles_sha256':sha(roles),
            'parent_inventory_sha256':INVENTORY_SHA,'parent_audit_sha256':AUDIT_SHA,'per_master':results,'aggregates':aggregate(results),'checks':CHECKS+M.CHECKS,
            'archive_decodes':20,'elapsed_seconds':time.monotonic()-started,'limits':{'TRAIN_only':True,'same_bank_patch_only':True,
            'head_PCA_or_encoder_fits':0,'model_forward':False,'VAL_payload_reads':0,'bootstrap':False,'not_single_full_context_latent':True,
            'prior_MAE_reductions_not_reaudited':True,'all_selected_inputs_and_SOURCE_preserved':True}}

def self_test():
    modules();negatives=0
    def rejected(call):
        nonlocal negatives
        try:call()
        except (AssertionError,KeyError):negatives+=1
        else:raise AssertionError('corrupted SOURCE fixture accepted')
    for period in (10.,16.,24.):
        for lag in (.25,.5,1.):
            for sign in (-1,1):
                values=[0.]*1152;mask=[False]*1152
                for bank in range(4):
                    for c,feature in ((0,0),(1,2)):
                        for h in range(8*bank,8*bank+8):
                            i=((bank*3+c)*32+h)*3+feature;values[i]=(1+.2*c)*math.sin(2*math.pi*(h+sign*lag*c)/period+.4)+.7*c;mask[i]=True
                result=timing(values,mask,1)[0];check(result['valid'] and (result['margin']>0)==(sign>0),'affine cross-feature noiseless sign')
                normalized=[(x-1.3)/.8 if ok else 0 for x,ok in zip(values,mask)]
                inverse=[x*.8+1.3 if ok else 0 for x,ok in zip(normalized,mask)]
                check(abs(timing(inverse,mask,1)[0]['margin']-result['margin'])<=2e-12,'legal affine inversion')
    for length in (5,6):
        values=[0.]*1152;mask=[False]*1152
        for c in (0,1):
            for h in range(length):
                i=(c*32+h)*3;values[i]=math.sin(2*math.pi*(h+.5*c)/16);mask[i]=True
        row=timing(values,mask,1)[0];check(row['supported_centres']==length-2 and row['valid']==(length==6),'three/four centre boundary')
    values=[0.]*1152;mask=[False]*1152
    for bank,positions in ((0,(6,7)),(1,(8,9))):
        for c in (0,1):
            for h in positions:mask[((bank*3+c)*32+h)*3]=True
    check(timing(values,mask,1)[0]['supported_centres']==0,'never stitch bank seam')
    values=[0.]*1152;mask=[False]*1152
    for c in (0,1):
        for h in range(8):mask[(c*32+h)*3]=True
    check(not timing(values,mask,1)[0]['valid'],'zero margin abstention')
    for bad in ('shape','legal_nan','outside_patch'):
        v=list(values);q=list(mask)
        if bad=='shape':v.pop()
        elif bad=='legal_nan':v[0]=float('nan')
        else:q[8*3]=True
        try:timing(v,q,1)
        except AssertionError:negatives+=1
        else:raise AssertionError('invalid bank geometry/finite support accepted')
    check(len(role_names())==len(set(role_names()))==20 and all('validation' not in n for n in role_names()),'closed TRAIN roles')

    records=[{'path':name,'bytes':1,'sha256':'0'*64} for name in role_names()]
    manifest={'protocol':PROTOCOL,'component':'same_bank_query_train','parent_inventory_sha256':INVENTORY_SHA,'capsule':str(PARENT),'files':records}
    parent={r['path']:r for r in records};check(role_records(manifest,parent)==records,'actual selected-role binding positive')
    for bad in ('missing','duplicate','VAL','wrong_sha','extra_key'):
        items=[dict(r) for r in records]
        if bad=='missing':items.pop()
        elif bad=='duplicate':items[-1]=dict(items[0])
        elif bad=='VAL':items[0]['path']=items[0]['path'].replace('training','validation')
        elif bad=='wrong_sha':items[0]['sha256']='1'*64
        else:items[0]['extra']=True
        rejected(lambda:role_records(dict(manifest,files=items),parent))

    ids=[f'structured-hard-timing-comparison-v2/lag_sign/structured-hard-timing-v1/seed-75272/lag_sign/source-{i//2}' for i in range(256)]
    labels=[i%2 for i in range(256)]
    original=[math.sin(2*math.pi*(h+(.5 if labels[row] else -.5)*int(c==1))/16)+.4*c+.1*f
              for row in range(256) for c in range(3) for h in range(32) for f in range(3)]
    observed=[True]*len(original)
    data_asset={'observations':M.fake('DoubleStorage',[256,3,32,3],original),'feature_mask':M.fake('BoolStorage',[256,3,32,3],observed),
                'labels_scoring_only':M.fake('LongStorage',[256],labels),'source_ids_json':M.fake_text(json.dumps(ids))}
    data=controlled(data_asset,75272)
    rejected(lambda:controlled(dict(data_asset,labels_scoring_only=M.fake('LongStorage',[256],[0]*256)),75272))
    bad_mask=list(observed);bad_mask[0]=False
    rejected(lambda:controlled(dict(data_asset,feature_mask=M.fake('BoolStorage',[256,3,32,3],bad_mask)),75272))
    rejected(lambda:controlled(data_asset,76373))
    means=[];scales=[];counts=[];floors=[]
    for c in range(3):
        for f in range(3):
            selected=[original[(row*3+c)*96+h*3+f] for row in range(256) for h in range(32)]
            m=math.fsum(selected)/len(selected);s=math.sqrt(math.fsum((x-m)**2 for x in selected)/len(selected))
            means.append(m);scales.append(max(1e-6,s));counts.append(len(selected));floors.append(s<1e-6)
    frozen={'mean':M.fake('DoubleStorage',[3,3],means),'scale':M.fake('DoubleStorage',[3,3],scales),
            'count':M.fake('LongStorage',[3,3],counts),'floor_applied':M.fake('BoolStorage',[3,3],floors),
            'channel_ids':M.fake('LongStorage',[3],[0,1,2]),'scale_floor':M.fake('DoubleStorage',[],[1e-6])}
    schema,dataset=data_identifiers(data);identity=M.scaler(frozen)[2]
    companion={'encoder_id':M.fake_text('raw_patch_bottleneck_mae_v1'),'format_version':M.fake('LongStorage',[],[1]),'artifact_kind':M.fake_text('rpb_frozen_training_scaler_v1'),
               'schema_id':M.fake_text(schema),'fit_dataset_id':M.fake_text(dataset),'preprocessing_id':M.fake_text(identity),'scaler':frozen}
    check(scaler_companion(companion,data)==(frozen['mean']['values'],frozen['scale']['values'],identity),'actual nested frozen scaler companion positive')
    rejected(lambda:scaler_companion(frozen,data))
    for field in ('encoder_id','artifact_kind','schema_id','fit_dataset_id','preprocessing_id'):
        rejected(lambda:scaler_companion(dict(companion,**{field:M.fake_text('wrong')}),data))
    requested,visible,q,eligible=M.query_masks(observed,256)
    mean=[.31+.03*d for d in range(9)];scale=[.7+.02*d for d in range(9)]
    target=[M.f32((original[i%(256*288)]-mean[((i%(256*288))//96)%3*3+i%3])/scale[((i%(256*288))//96)%3*3+i%3]) if ok else 0 for i,ok in enumerate(q)]
    asset={'standardized_prediction':M.fake('DoubleStorage',[4,256,3,32,3],target),'standardized_target':M.fake('DoubleStorage',[4,256,3,32,3],target),
           'target_mask':M.fake('BoolStorage',[4,256,3,32,3],q),'requested_observed_target_mask':M.fake('BoolStorage',[4,256,3,32,3],requested),
           'visible_mask':M.fake('BoolStorage',[4,256,3,32,3],visible),'trial_channel_eligible':M.fake('BoolStorage',[4,256,3],eligible),
           'channel_target_counts':M.fake('LongStorage',[256,3],[96]*768),'channel_valid':M.fake('BoolStorage',[256,3],[True]*768),
           'channel_standardized_mae':M.fake('DoubleStorage',[256,3],[0]*768),'channel_standardized_huber':M.fake('DoubleStorage',[256,3],[0]*768),
           'example_valid':M.fake('BoolStorage',[256],[True]*256),'example_standardized_mae':M.fake('DoubleStorage',[256],[0]*256),
           'example_standardized_huber':M.fake('DoubleStorage',[256],[0]*256),'source_ids_json':M.fake_text(json.dumps(ids))}
    raw_pred,raw_target,support=query(asset,data,mean,scale)
    check(raw_pred==raw_target and support==q,'actual typed saved-query positive affine replay')
    rows=timing(raw_target,q,256);scores=summarize(rows,labels,ids)
    check(scores['score']['accuracy']==1 and scores['pair_summary']['both_correct_count']==128,'typed target/noiseless source pairs')
    bad=list(target);bad[next(i for i,ok in enumerate(q) if not ok)]=1
    rejected(lambda:query(dict(asset,standardized_prediction=M.fake('DoubleStorage',[4,256,3,32,3],bad)),data,mean,scale))
    bad=list(target);bad[0]+=.01
    rejected(lambda:query(dict(asset,standardized_target=M.fake('DoubleStorage',[4,256,3,32,3],bad)),data,mean,scale))
    rejected(lambda:query(dict(asset,trial_channel_eligible=M.fake('BoolStorage',[4,256,3],[False]*3072)),data,mean,scale))
    rejected(lambda:query(dict(asset,unexpected=M.fake('LongStorage',[],[0])),data,mean,scale))
    effect=common_effect(rows,rows,labels,ids);check(effect['candidate_minus_reference_accuracy']==0 and effect['complete_common_source_pairs']==128,'common-support same row arrays')
    tiny=[{'margin':-1e-300,'valid':True,'prediction':0},{'margin':1e-300,'valid':True,'prediction':1}]
    check(summarize(tiny,[0,1],['same','same'])['pair_summary']['opposite_signs_count']==1,'strict signs without underflowing product')
    unsupported=[dict(r,valid=False,prediction=0) for r in rows]
    empty=summarize(unsupported,labels,ids);check(empty['score']['accuracy'] is None and empty['pair_summary']['supported_pairs']==0,'unsupported rows remain original denominator')
    check(common_effect(rows,unsupported,labels,ids)['candidate_minus_reference_accuracy'] is None,'undefined common-support effect remains null')
    rejected(lambda:summarize(rows,labels[:-1],ids))
    rejected(lambda:summarize(rows,[0]*256,ids))
    masters=[{'methods':{method:empty for method in ('saved_target','late_prediction','early_prediction')}} for _ in MASTERS]
    check(aggregate(masters)['saved_target']['accuracy']['mean'] is None and aggregate(masters)['saved_target']['coverage']['mean']==0,'all-five null support summary')
    return {'status':'passed','source_sha256':sha(Path(__file__)),'module_sha256':PINS,'checks':CHECKS+M.CHECKS,'negative_fixtures':negatives,'archive_reads':0,'TRAIN_or_VAL_payload_reads':0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--fixture-output');p.add_argument('--run',action='store_true')
    for name in ('source-sha256','card','card-sha256','roles','roles-sha256','output'):p.add_argument('--'+name)
    args=p.parse_args();check(Path('/.dockerenv').exists(),'managed container only')
    if args.self_test:
        result=self_test()
        if args.fixture_output:
            with Path(args.fixture_output).open('x',encoding='utf-8') as out:json.dump(result,out,indent=2);out.write('\n')
        print(json.dumps(result));return
    check(args.run and all(getattr(args,k) for k in ('source_sha256','card','card_sha256','roles','roles_sha256','output')),'explicit root pinned TRAIN execution only')
    output=Path(args.output);check(output.is_absolute() and output.parent.resolve(strict=True)==output.parent and not output.exists(),'exclusive new output')
    output.mkdir()
    try:
        result=run(args)
        with (output/'query-diagnostic.json').open('x',encoding='utf-8') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
        print(json.dumps({'status':'passed','checks':result['checks'],'archive_decodes':20,'elapsed_seconds':result['elapsed_seconds'],'result_sha256':sha(output/'query-diagnostic.json')}))
    except Exception as error:
        with (output/'failure.json').open('x',encoding='utf-8') as out:json.dump({'status':'failed','error':str(error),'checks':CHECKS,'source_sha256':sha(Path(__file__))},out,indent=2);out.write('\n')
        raise

if __name__=='__main__':main()

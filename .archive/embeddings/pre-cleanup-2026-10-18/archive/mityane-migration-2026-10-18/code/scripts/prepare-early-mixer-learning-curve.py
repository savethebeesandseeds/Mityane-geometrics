#!/usr/bin/env python3
"""Bind the prospective early-mixer source, actual GPU admission and reader.

Uses unchanged source-capture utilities only; no historical quality entrypoint
or payload is accessed. Project execution is restricted to the managed container.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('source_capture', Path(__file__).with_name('prepare-fresh-decoder-replication.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
PROTOCOL = 'early-mixer-learning-curve-v1'
CARD = 'code/evaluation/cards/early_mixer_learning_curve_v1.md'
CARD_SHA = '869d1a113ed8b6493295632555d69700a279f34e85b01754459761abcc7c4a98'
ROOT = 'output/runs/rpb-early-mixer-learning-curve'
TIMING = [30130,31231,32332,33433,34534]
AMPLITUDE = [35635,36736,37837,38938,40039]
base.PROTOCOL,base.CARD,base.ROOT = PROTOCOL,CARD,ROOT
base.REQUIRED_SOURCES = {CARD,'Makefile','dependencies.lock','setup.sh',
    'code/scripts/install-libtorch.py',
    'code/evaluation/src/early_mixer_learning_curve_main.cpp',
    'code/evaluation/include/frozen_role_guard.h',
    'code/shared/include/embedding/shared/fixed_feature_readouts.h',
    'code/shared/src/fixed_feature_readouts.cpp','code/shared/tests/fixed_feature_readouts_test.cpp',
    'code/encoders/raw_patch_bottleneck_mae/shared/include/embedding/encoders/raw_patch_bottleneck_mae/config.h',
    'code/encoders/raw_patch_bottleneck_mae/shared/include/embedding/encoders/raw_patch_bottleneck_mae/model.h',
    'code/encoders/raw_patch_bottleneck_mae/shared/include/embedding/encoders/raw_patch_bottleneck_mae/early_mixer_adapter.h',
    'code/encoders/raw_patch_bottleneck_mae/shared/src/early_mixer_adapter.cpp',
    'code/encoders/raw_patch_bottleneck_mae/shared/src/learning_curve_adapter.cpp',
    'code/encoders/raw_patch_bottleneck_mae/shared/src/workflow.cpp',
    'code/encoders/raw_patch_bottleneck_mae/versions/v10/tests/early_mixer_model_test.cpp',
    'code/encoders/raw_patch_bottleneck_mae/shared/tests/early_mixer_adapter_test.cpp',
    'code/encoders/raw_patch_bottleneck_mae/shared/tests/early_mixer_curve_adapter_test.cpp',
    'code/scripts/prepare-fresh-decoder-replication.py','code/scripts/prepare-early-mixer-learning-curve.py',
    'code/scripts/check-early-mixer-learning-curve.sh','code/scripts/evaluate-early-mixer-learning-curve.sh'}
require,file_sha,write_new,write_json = base.require,base.file_sha,base.write_new,base.write_json
original_source_admission = base.admit_paths

def source_admission(repo,names):
    admitted = original_source_admission(repo,names)
    for _,path,_ in admitted:
        require(path.stat().st_nlink==1,'source hardlink outside the captured matrix')
    return admitted

base.admit_paths = source_admission

def directory(repo,path,admission=False):
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_dir() and not path.is_symlink(),'regular exclusive directory')
    require(path.parent==repo/ROOT/('admission' if admission else '') and
            path.name.startswith('admission-' if admission else 'early-mixer-learning-curve-'),'protocol directory bounds')
    return path

def admit_matrix(paths,root):
    require(paths==sorted(set(paths)),'unique sorted whole matrix')
    inodes,result=set(),[]
    for path in paths:
        require(path.is_absolute() and path.is_relative_to(root) and path.resolve(strict=True)==path
                and path.is_file() and not path.is_symlink(),'regular closed role')
        for ancestor in [path,*path.parents]:
            if ancestor==root.parent: break
            require(not ancestor.is_symlink(),'ancestor redirect')
        stat=path.stat(); inode=(stat.st_dev,stat.st_ino)
        require(stat.st_nlink==1 and inode not in inodes,'hardlink/alias')
        inodes.add(inode); result.append((path,stat.st_size))
    return result

def read_json(path):
    require(path.suffix=='.json','JSON metadata role only')
    admit_matrix([path],path.parent)
    return json.loads(path.read_text(encoding='utf-8'))

base.read_json = read_json
original_source_state = base.source_state

def source_state(repo,target,card_sha):
    metadata=read_json(target/'source-manifest.json')
    names=[row['path'] for row in metadata['sources']]
    source_admission(repo,names)
    source=target/'source'
    require({p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() or p.is_symlink()}==set(names),
            'closed captured source matrix before any source hash')
    source_admission(source,names)
    return original_source_state(repo,target,card_sha)

base.source_state = source_state

def binary_record(binary):
    path=Path(binary)
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink()
            and path.is_relative_to(Path('/opt/cuwacunu_embedding/build')) and
            path.name=='embedding_early_mixer_learning_curve','named container binary')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path)}

def admission_record(repo,target,card_sha,source_id,binary,sealed=False):
    admit_matrix(base.admission_paths(target,sealed),target)
    require(base.source_state(repo,target,card_sha)[1]==source_id,'actual compiled/source binding')
    plan=read_json(target/'admission-plan.json')
    require(plan['source_fingerprint']==source_id and plan['human_card_sha256']==card_sha
            and not plan['quality_generated'] and plan['historical_quality_inputs']==[],'source-only engineering admission')
    log=target/'build-and-tests.log'; text=log.read_text(encoding='utf-8')
    for marker in ['Early mixer model CUDA admission passed','Early mixer CUDA adapter admission passed',
                   'Early mixer curve CUDA admission passed',
                   'Fixed feature readout tests passed','Frozen role guard checks passed',source_id]:
        require(marker in text,'actual source-bound CUDA/fixed-readout admission: '+marker)
    return {'protocol':PROTOCOL,'status':'passed','source_fingerprint':source_id,
        'human_card_sha256':card_sha,'source_preserved':True,'log_sha256':file_sha(log),
        'compiled_binary':binary_record(binary),'quality_generated':False,'historical_payload_roles':0,
        'gates':['default/early CUDA model contracts and typed architecture archives',
                 'unchanged continuous coordinate15 CUDA updates and immutable snapshots',
                 'curve horizon/direct/segmented/interleaved CUDA AdamW and live CPU-state witnesses',
                 'fixed generic paired readouts and whole-role guard fixtures']}

def check_plan(plan):
    expected={'protocol':PROTOCOL,'timing_master_seeds':TIMING,'amplitude_data_master_seeds':AMPLITUDE,
        'tags':['RPB-v7.alt-03','RPB-v10.alt-01'],'encoder_trajectories':10,'encoder_updates_each':2048,
        'milestones':[0,512,1024,2048],'attempt_limit':4096,'skipped_attempts_permitted':False,
        'decoder_updates':0,'batch_size':8,'sampled_rows':163840,'train_pairs':128,'validation_pairs':64,
        'test_pairs':0,'shape':[3,32,3],'native_width':32,'parameter_count_each':225805,
        'head_repetitions':[2701,2802,2903],'planned_pipelines':270,'planned_heads':540,'deletion_rate':.30,
        'timing_methods':11,'amplitude_methods':7,'planned_native_export_calls':180,'planned_query_writer_calls':60,
        'planned_query_forwards':240,'quality_exports_repeated':False,'raw_outer_fit_shared_with_PCA':True,
        'encoder_device':'CUDA','prior_quality_input_roles':[],'quality_generated':False,
        'testing_accessed':False,'stress_accessed':False,'selection':False,'promotion':False}
    for key,value in expected.items(): require(plan[key]==value,'fixed early-mixer recipe: '+key)

def bind_reader(repo,target,args):
    originals=[Path(args.reader_source),Path(args.reader_fixtures),Path(args.reader_seal)]
    admit_matrix(sorted(originals),repo/ROOT/'audit-tools')
    seal=read_json(originals[2])
    require(seal['status']=='passed' and seal['source_sha256']==file_sha(originals[0])
            and seal['fixtures_sha256']==file_sha(originals[1]),'presealed independent source-only reader')
    reader_text=originals[0].read_text(encoding='utf-8')
    require(re.search(r'^REVIEWED_SCHEMA = False$',reader_text,re.M) and
            re.search(r'^MEASURED_IMPLEMENTATION = False$',reader_text,re.M),
            'reviewed reader captured with both execution flags FALSE before generation')
    fixtures=read_json(originals[1])
    require(fixtures['status']=='passed' and not fixtures['measured_execution_enabled'],
            'passed source-only fixtures remain prospectively blocked')
    copied=target/'independent-reader'; copied.mkdir();records=[]
    for path,name in zip(originals,['source.py','source-fixtures.json','reader-seal.json']):
        dest=copied/name;content=path.read_bytes();write_new(dest,content)
        records.append({'path':str(path),'captured':str(dest),'bytes':len(content),'sha256':base.digest(content)})
    return records

def freeze_run(args,repo,target):
    require({p.name for p in target.iterdir()}=={'recipe-plan.json'},'exclusive source-only plan capsule')
    check_plan(read_json(target/'recipe-plan.json'))
    records=base.source_records(repo,args.sources,args.card_sha256)
    require(base.digest(base.manifest_bytes(records))==args.compiled_source_id,'compiled closure binding')
    admission=directory(repo,Path(args.admission),True)
    current=admission_record(repo,admission,args.card_sha256,args.compiled_source_id,args.binary,True)
    require(read_json(admission/'passed.json')==current,'unchanged admission')
    copied=target/'admission';copied.mkdir()
    for path in base.admission_paths(admission,True):
        dest=copied/path.relative_to(admission);dest.parent.mkdir(parents=True,exist_ok=True);write_new(dest,path.read_bytes())
    require(base.copy_sources(repo,target,records)==args.compiled_source_id,'source capture')
    reader=bind_reader(repo,target,args)
    write_json(target/'input-role-plan.json',{'protocol':PROTOCOL,'historical_payload_roles':0,
        'prior_quality_input_roles':[],'fresh_quality_inputs':'generated only after frozen source/admission/reader launch'})
    argv=[args.binary,'--output',str(target/'results'),'--admission-log',str(copied/'build-and-tests.log'),
          '--admission-sha256',current['log_sha256'],'--card',str(target/'source'/CARD),'--card-sha256',args.card_sha256]
    write_json(target/'launch-plan.json',{'protocol':PROTOCOL,'status':'frozen_before_quality_generation',
        'created_utc':base.utc(),'source_fingerprint':args.compiled_source_id,'human_card_sha256':args.card_sha256,
        'compiled_binary':current['compiled_binary'],'admission_passed_sha256':file_sha(copied/'passed.json'),
        'admission_log_sha256':current['log_sha256'],'reader':reader,'historical_payload_roles':0,'argv':argv,
        'recipe_sha256':file_sha(target/'recipe-plan.json'),'encoder_trajectories':10,'new_heads_planned':540,
        'milestones':[0,512,1024,2048],'sampled_rows':163840,'historical_quality_inputs':[],
        'reader_execution_enabled_before_generation':False})
    base.source_state(repo,target,args.card_sha256)
    return {'protocol':PROTOCOL,'source_fingerprint':args.compiled_source_id,'historical_payload_roles':0}

def finalize(args,repo,target):
    require(base.source_state(repo,target,args.card_sha256)[1]==args.compiled_source_id,'live/captured source preserved')
    launch=read_json(target/'launch-plan.json');require(binary_record(launch['argv'][0])==launch['compiled_binary'],'tested binary preserved')
    require(file_sha(target/'recipe-plan.json')==launch['recipe_sha256'],'recipe preserved');check_plan(read_json(target/'recipe-plan.json'))
    require(file_sha(target/'admission/passed.json')==launch['admission_passed_sha256'] and
            file_sha(target/'admission/build-and-tests.log')==launch['admission_log_sha256'],'admission preserved')
    for row in launch['reader']:
        require(file_sha(Path(row['path']))==file_sha(Path(row['captured']))==row['sha256'],'reader bytes preserved')
    complete=read_json(target/'results/complete.json')
    require(complete['protocol']==PROTOCOL and complete['status']=='complete' and complete['encoder_trajectories']==10 and
            complete['encoder_updates_each']==2048 and complete['attempt_limit']==4096 and complete['sampled_rows']==163840 and
            complete['milestones']==[0,512,1024,2048] and complete['retained_points']==40 and complete['skipped_attempts']==0 and
            complete['head_pipelines']<=270 and complete['individual_heads']==2*complete['head_pipelines'] and
            complete['planned_pipelines']==270 and complete['planned_heads']==540 and
            complete['full_native_export_calls']==180 and complete['query_evaluation_calls']==60 and
            complete['necessary_query_forwards']==240 and complete['quality_export_replays']==0 and
            complete['driver_raw_outer_fits']==10 and complete['helper_outer_train_fits']<=70 and
            complete['earlier_parent_bytes_and_prefixes_preserved'] and not complete['selection'] and not complete['promotion'] and
            not complete['CPU_encoder_training'] and not complete['CPU_encoder_forward'] and
            not complete['testing_accessed'] and not complete['stress_accessed'],'fixed completed measurement')
    write_json(target/'source-preserved-after.json',{'protocol':PROTOCOL,'source_fingerprint':args.compiled_source_id,
        'source_preserved':True,'historical_payload_roles':0,'reader_preserved':True,'created_utc':base.utc()})
    files=[{'path':p.relative_to(target).as_posix(),'bytes':size,'sha256':file_sha(p)}
           for p,size in admit_matrix(sorted(p for p in target.rglob('*') if p.is_file() or p.is_symlink()),target)]
    write_json(target/'artifact-integrity.json',{'schema_version':1,'protocol':PROTOCOL,'created_utc':base.utc(),
        'inventory_excludes_itself':True,'checksum_algorithm':'sha256-file-bytes','file_count':len(files),
        'total_bytes':sum(x['bytes'] for x in files),'files':files})
    return {'protocol':PROTOCOL,'status':'complete','source_fingerprint':args.compiled_source_id,
        'inventory_sha256':file_sha(target/'artifact-integrity.json'),'file_count':len(files)}

def self_test():
    fixture={'protocol':PROTOCOL,'timing_master_seeds':TIMING,'amplitude_data_master_seeds':AMPLITUDE,
        'tags':['RPB-v7.alt-03','RPB-v10.alt-01'],'encoder_trajectories':10,'encoder_updates_each':2048,'decoder_updates':0,
        'milestones':[0,512,1024,2048],'attempt_limit':4096,'skipped_attempts_permitted':False,
        'batch_size':8,'sampled_rows':163840,'train_pairs':128,'validation_pairs':64,'test_pairs':0,'shape':[3,32,3],
        'native_width':32,'parameter_count_each':225805,'head_repetitions':[2701,2802,2903],
        'planned_pipelines':270,'planned_heads':540,'deletion_rate':.30,'encoder_device':'CUDA',
        'timing_methods':11,'amplitude_methods':7,'planned_native_export_calls':180,'planned_query_writer_calls':60,
        'planned_query_forwards':240,'quality_exports_repeated':False,'raw_outer_fit_shared_with_PCA':True,
        'prior_quality_input_roles':[],'quality_generated':False,'testing_accessed':False,'stress_accessed':False,'selection':False,'promotion':False}
    check_plan(fixture);negatives=0
    for key,value in [('timing_master_seeds',TIMING[:-1]),('tags',['RPB-v7','RPB-v10']),('encoder_updates_each',512),
        ('milestones',[0,512,2048]),('attempt_limit',1000),('skipped_attempts_permitted',True),
        ('decoder_updates',128),('planned_heads',420),('test_pairs',64),('prior_quality_input_roles',['old.pt']),
        ('quality_generated',True),('encoder_device','CPU'),('quality_exports_repeated',True),('raw_outer_fit_shared_with_PCA',False)]:
        wrong=dict(fixture);wrong[key]=value
        try:check_plan(wrong)
        except AssertionError:negatives+=1
        else:raise AssertionError('unsafe plan admitted: '+key)
    with tempfile.TemporaryDirectory(prefix='early-mixer-source-fixture-') as temporary:
        root=Path(temporary).resolve(strict=True);first,last=root/'a',root/'z';first.write_text('artificial fixture');os.link(first,last)
        try:admit_matrix([first,last],root)
        except AssertionError:negatives+=1
        else:raise AssertionError('alias admitted')
        last.unlink();last.symlink_to(first)
        try:admit_matrix([first,last],root)
        except AssertionError:negatives+=1
        else:raise AssertionError('redirect admitted')
    return {'status':'passed','protocol':PROTOCOL,'negative_cases':negatives,'quality_payload_reads':0,'quality_generated':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['self-test','freeze-admission','admit','freeze-run','finalize'])
    for name in ['repo-root','target','card-sha256','compiled-source-id','admission','binary','reader-source','reader-fixtures','reader-seal']:
        parser.add_argument('--'+name,default='.' if name=='repo-root' else None)
    parser.add_argument('--sources',nargs='+');args=parser.parse_args();require(Path('/.dockerenv').exists(),'managed container required')
    if args.mode=='self-test':result=self_test()
    else:
        require(args.target and args.card_sha256==CARD_SHA,'explicit target/frozen card SHA')
        repo=Path(args.repo_root).resolve(strict=True);target=directory(repo,Path(args.target),args.mode in ['freeze-admission','admit'])
        if args.mode=='freeze-admission':result={'source_fingerprint':base.freeze_admission(repo,target,args.sources,args.card_sha256)}
        elif args.mode=='admit':
            result=admission_record(repo,target,args.card_sha256,args.compiled_source_id,args.binary)
            write_json(target/'source-preserved-after.json',{'protocol':PROTOCOL,'source_fingerprint':args.compiled_source_id,'source_preserved':True});write_json(target/'passed.json',result)
        elif args.mode=='freeze-run':result=freeze_run(args,repo,target)
        else:result=finalize(args,repo,target)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()

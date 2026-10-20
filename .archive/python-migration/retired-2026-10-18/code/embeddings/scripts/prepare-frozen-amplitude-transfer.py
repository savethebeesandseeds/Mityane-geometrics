#!/usr/bin/env python3
"""Freeze a source-bound, closed-role frozen-encoder transfer run.

Reuse unchanged source-capture utilities, never the prior quality entrypoint.
All project execution is restricted to the managed container.
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
PROTOCOL = 'frozen-amplitude-transfer-v1'
CARD = 'code/embeddings/evaluation/cards/frozen_amplitude_transfer_v1.md'
ROOT = 'output/runs/rpb-frozen-amplitude-transfer'
PARENT = 'output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b'
PARENT_SHA = 'a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90'
OLD = [9109,10210,11311,12412,13513]
NEW = [14614,15715,16816,17917,19018]
HEADER = ['id','display_tag','old_master','new_master','method','policy','updates',
          'controlled_training','checkpoint','audit','scaler','training_raw','native_training']
base.PROTOCOL,base.CARD,base.ROOT = PROTOCOL,CARD,ROOT
base.REQUIRED_SOURCES = {CARD,'Makefile','dependencies.lock',
    'code/embeddings/evaluation/src/frozen_amplitude_transfer_main.cpp',
    'code/embeddings/evaluation/include/frozen_role_guard.h',
    'code/shared/representation/src/fixed_feature_readouts.cpp',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/include/embedding/encoders/raw_patch_bottleneck_mae/frozen_native_feature_adapter.h',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/src/frozen_decoder_calibration.cpp',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/tests/frozen_native_feature_adapter_test.cpp',
    'code/embeddings/scripts/prepare-fresh-decoder-replication.py',
    'code/embeddings/scripts/prepare-frozen-amplitude-transfer.py',
    'code/embeddings/scripts/check-frozen-amplitude-transfer.sh',
    'code/embeddings/scripts/evaluate-frozen-amplitude-transfer.sh'}
require,file_sha,write_new,write_json,read_json = base.require,base.file_sha,base.write_new,base.write_json,base.read_json

def directory(repo,path,admission=False):
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_dir() and not path.is_symlink(),'regular new directory')
    require(path.parent==repo/ROOT/('admission' if admission else '') and
            path.name.startswith('admission-' if admission else 'frozen-amplitude-transfer-'),'protocol bounds')
    return path

def admit_matrix(paths,root):
    require(paths==sorted(set(paths)),'unique sorted complete role list')
    inodes,result=set(),[]
    for path in paths:
        require(path.is_absolute() and path.is_relative_to(root) and path.resolve(strict=True)==path
                and path.is_file() and not path.is_symlink(),'closed regular role')
        for ancestor in [path,*path.parents]:
            if ancestor==root.parent: break
            require(not ancestor.is_symlink(),'role ancestor redirect')
        stat=path.stat(); inode=(stat.st_dev,stat.st_ino)
        require(stat.st_nlink==1 and inode not in inodes,'role hardlink/alias')
        inodes.add(inode); result.append((path,stat.st_size))
    return result

def binary_record(binary):
    path=Path(binary)
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink()
            and path.is_relative_to(Path('/opt/mityane-geometrics/embeddings/build'))
            and path.name=='embedding_frozen_amplitude_transfer','named container binary')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path)}

def admission_record(repo,target,card_sha,source_id,binary,sealed=False):
    base.admission_paths(target,sealed)
    require(base.source_state(repo,target,card_sha)[1]==source_id,'compiled/source binding')
    plan=read_json(target/'admission-plan.json')
    require(plan['source_fingerprint']==source_id and plan['human_card_sha256']==card_sha
            and not plan['quality_generated'] and plan['historical_quality_inputs']==[],'source-only admission')
    log=target/'build-and-tests.log'; text=log.read_text(encoding='utf-8')
    for marker in ['Frozen native feature CUDA admission passed','Fresh decoder replication CUDA admission passed',
                   'Fixed feature readout tests passed','Frozen role guard checks passed',source_id]:
        require(marker in text,'actual source-bound CUDA/fixed readout admission: '+marker)
    return {'protocol':PROTOCOL,'status':'passed','source_fingerprint':source_id,
        'human_card_sha256':card_sha,'source_preserved':True,'log_sha256':file_sha(log),
        'compiled_binary':binary_record(binary),'quality_generated':False,'historical_payload_roles':0,
        'gates':['new feature-only strict parent CUDA serving fixtures','legacy frozen decoder contracts',
                 'fixed feature readout and whole-role guard fixtures']}

def check_plan(plan):
    expected={'protocol':PROTOCOL,'old_master_seeds':OLD,'new_data_master_seeds':NEW,
        'display_tags':['RPB-v4.alt-01','RPB-v7.alt-01'],'retained_snapshots':15,'unique_parent_roles':80,
        'encoder_updates':0,'decoder_updates':0,'planned_pipelines':90,'planned_heads':180,
        'train_pairs':128,'validation_pairs':64,'test_pairs':0,'testing_accessed':False,
        'stress_accessed':False,'quality_generated':False}
    for key,value in expected.items(): require(plan[key]==value,'fixed transfer plan: '+key)

def declared_rows(repo):
    rows=[]
    for old,new in zip(OLD,NEW):
        root=repo/PARENT/'results'/f'seed-{old}-lag_sign'
        for method,tag,version,updates,policy in [
            ('untrained_native','Untrained encoder','v4',0,'ordinary_v4'),
            ('native_v4','RPB-v4.alt-01','v4',512,'ordinary_v4'),
            ('native_v7','RPB-v7.alt-01','v7',512,'coordinate15_v7')]:
            checkpoint=root/version/f'point-{updates}'/'checkpoint.pt'
            paths=[root/'controlled-training.pt',checkpoint,Path(str(checkpoint)+'.audit.pt'),
                   Path(str(checkpoint)+'.scaler.pt'),Path(str(checkpoint)+'.training-raw.pt'),
                   root/'readouts'/method/'training-features.pt']
            rows.append([f'{method}-{old}',tag,str(old),str(new),method,policy,str(updates),*[str(p) for p in paths]])
    return rows

def input_records(repo,rows):
    paths=sorted({Path(p) for row in rows for p in row[7:]})
    require(len(paths)==80,'exact80 unique parent roles')
    inventory=repo/PARENT/'artifact-integrity.json'
    admitted=admit_matrix(sorted([inventory,*paths]),repo/PARENT)
    require(file_sha(inventory)==PARENT_SHA,'pinned parent inventory')
    meta=read_json(inventory)
    require(meta['protocol']=='fresh-decoder-replication-v1' and meta['file_count']==948
            and len(meta['files'])==948,'parent inventory shape')
    index={x['path']:x for x in meta['files']}; require(len(index)==948,'inventory duplicates')
    records=[]
    for path,size in admitted:
        if path==inventory: continue
        name=path.relative_to(repo/PARENT).as_posix()
        require(name in index and index[name]['bytes']==size and
                re.fullmatch('[0-9a-f]{64}',index[name]['sha256']),'whole pinned role/size/hash association')
        records.append({'path':str(path),'bytes':size,'sha256':index[name]['sha256']})
    for row in records: require(file_sha(Path(row['path']))==row['sha256'],'pinned parent bytes')
    return records,inventory

def bind_reader(repo,target,args):
    originals=[Path(args.reader_source),Path(args.reader_fixtures),Path(args.reader_seal)]
    admit_matrix(sorted(originals),repo/ROOT/'audit-tools')
    seal=read_json(originals[2])
    require(seal['status']=='passed' and seal['source_sha256']==file_sha(originals[0])
            and seal['fixtures_sha256']==file_sha(originals[1]),'presealed source-only independent reader')
    copied=target/'independent-reader'; copied.mkdir(); records=[]
    for path,name in zip(originals,['source.py','source-fixtures.json','reader-seal.json']):
        dest=copied/name; content=path.read_bytes(); write_new(dest,content)
        records.append({'path':str(path),'captured':str(dest),'bytes':len(content),'sha256':base.digest(content)})
    return records

def freeze_run(args,repo,target):
    require({p.name for p in target.iterdir()}=={'recipe-plan.json'},'exclusive plan-only capsule')
    check_plan(read_json(target/'recipe-plan.json'))
    records=base.source_records(repo,args.sources,args.card_sha256)
    require(base.digest(base.manifest_bytes(records))==args.compiled_source_id,'compiled closure binding')
    admission=directory(repo,Path(args.admission),True)
    current=admission_record(repo,admission,args.card_sha256,args.compiled_source_id,args.binary,True)
    require(read_json(admission/'passed.json')==current,'unchanged admission')
    copied=target/'admission'; copied.mkdir()
    for path in base.admission_paths(admission,True):
        dest=copied/path.relative_to(admission); dest.parent.mkdir(parents=True,exist_ok=True); write_new(dest,path.read_bytes())
    require(base.copy_sources(repo,target,records)==args.compiled_source_id,'source capture')
    reader=bind_reader(repo,target,args)
    rows=declared_rows(repo); roles,inventory=input_records(repo,rows)
    write_new(target/'parent-inventory.json',inventory.read_bytes())
    write_new(target/'instances.tsv','\t'.join(HEADER)+'\n'+''.join('\t'.join(row)+'\n' for row in rows))
    write_new(target/'input-checksums.sha256',''.join(x['sha256']+'  '+x['path']+'\n' for x in roles))
    write_json(target/'input-role-plan.json',{'protocol':PROTOCOL,'roles':roles,'unique_parent_roles':80,
        'parent_inventory_sha256':PARENT_SHA,'old_heldout_roles':0,'prior_mixed_audit_payload_read':False})
    argv=[args.binary,'--instances',str(target/'instances.tsv'),'--checksums',str(target/'input-checksums.sha256'),
          '--input-root',str(repo),'--output',str(target/'results'),'--admission-log',str(copied/'build-and-tests.log'),
          '--admission-sha256',current['log_sha256'],'--card',str(target/'source'/CARD),'--card-sha256',args.card_sha256]
    write_json(target/'launch-plan.json',{'protocol':PROTOCOL,'status':'frozen_before_quality_generation',
        'created_utc':base.utc(),'source_fingerprint':args.compiled_source_id,'human_card_sha256':args.card_sha256,
        'compiled_binary':current['compiled_binary'],'admission_passed_sha256':file_sha(copied/'passed.json'),
        'admission_log_sha256':current['log_sha256'],'reader':reader,'roles':roles,'argv':argv,
        'instances_sha256':file_sha(target/'instances.tsv'),'checksums_sha256':file_sha(target/'input-checksums.sha256'),
        'recipe_sha256':file_sha(target/'recipe-plan.json'),'encoder_updates':0,'decoder_updates':0,'new_heads_planned':180})
    base.source_state(repo,target,args.card_sha256)
    return {'protocol':PROTOCOL,'source_fingerprint':args.compiled_source_id,'unique_parent_roles':80}

def finalize(args,repo,target):
    require(base.source_state(repo,target,args.card_sha256)[1]==args.compiled_source_id,'unchanged live/captured source')
    launch=read_json(target/'launch-plan.json')
    require(binary_record(launch['argv'][0])==launch['compiled_binary'],'tested binary preserved')
    for name,key in [('instances.tsv','instances_sha256'),('input-checksums.sha256','checksums_sha256'),('recipe-plan.json','recipe_sha256')]:
        require(file_sha(target/name)==launch[key],'run metadata preserved')
    check_plan(read_json(target/'recipe-plan.json'))
    require(file_sha(target/'admission/passed.json')==launch['admission_passed_sha256'] and
            file_sha(target/'admission/build-and-tests.log')==launch['admission_log_sha256'],'admission preserved')
    admit_matrix(sorted(Path(x['path']) for x in launch['roles']),repo/PARENT)
    for row in launch['roles']:
        path=Path(row['path']); require(path.stat().st_size==row['bytes'] and file_sha(path)==row['sha256'],'input preserved')
    for row in launch['reader']:
        require(file_sha(Path(row['path']))==file_sha(Path(row['captured']))==row['sha256'],'reader preserved')
    complete=read_json(target/'results/complete.json')
    require(complete['protocol']==PROTOCOL and complete['status']=='complete','measurement complete')
    write_json(target/'source-preserved-after.json',{'protocol':PROTOCOL,'source_fingerprint':args.compiled_source_id,
        'source_preserved':True,'inputs_preserved':True,'reader_preserved':True,'created_utc':base.utc()})
    files=[{'path':p.relative_to(target).as_posix(),'bytes':size,'sha256':file_sha(p)}
           for p,size in admit_matrix(sorted(p for p in target.rglob('*') if p.is_file() or p.is_symlink()),target)]
    write_json(target/'artifact-integrity.json',{'schema_version':1,'protocol':PROTOCOL,'created_utc':base.utc(),
        'inventory_excludes_itself':True,'checksum_algorithm':'sha256-file-bytes','file_count':len(files),
        'total_bytes':sum(x['bytes'] for x in files),'files':files})
    return {'protocol':PROTOCOL,'status':'complete','source_fingerprint':args.compiled_source_id,
            'inventory_sha256':file_sha(target/'artifact-integrity.json'),'file_count':len(files)}

def self_test():
    rows=declared_rows(Path('/embedding'))
    require(len(rows)==15 and all(len(r)==13 for r in rows) and len({p for r in rows for p in r[7:]})==80,'closed source-only enumeration')
    with tempfile.TemporaryDirectory(prefix='frozen-amplitude-source-fixture-') as temp:
        root=Path(temp).resolve(strict=True); first,last=root/'a',root/'z'
        first.write_text('artificial fixture'); os.link(first,last)
        try: admit_matrix([first,last],root)
        except AssertionError: pass
        else: raise AssertionError('alias accepted')
        last.unlink(); last.symlink_to(first)
        try: admit_matrix([first,last],root)
        except AssertionError: pass
        else: raise AssertionError('redirect accepted')
    return {'status':'passed','protocol':PROTOCOL,'parent_roles':80,'snapshots':15,'quality_payload_reads':0,'negative_cases':2}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['self-test','freeze-admission','admit','freeze-run','finalize'])
    for name in ['repo-root','target','card-sha256','compiled-source-id','admission','binary','reader-source','reader-fixtures','reader-seal']:
        parser.add_argument('--'+name,default='.' if name=='repo-root' else None)
    parser.add_argument('--sources',nargs='+'); args=parser.parse_args()
    require(Path('/.dockerenv').exists(),'managed container required')
    if args.mode=='self-test': result=self_test()
    else:
        require(args.target and args.card_sha256 and re.fullmatch('[0-9a-f]{64}',args.card_sha256),'target/card binding')
        repo=Path(args.repo_root).resolve(strict=True)
        target=directory(repo,Path(args.target),args.mode in ['freeze-admission','admit'])
        if args.mode=='freeze-admission': result={'source_fingerprint':base.freeze_admission(repo,target,args.sources,args.card_sha256)}
        elif args.mode=='admit':
            result=admission_record(repo,target,args.card_sha256,args.compiled_source_id,args.binary)
            write_json(target/'source-preserved-after.json',{'protocol':PROTOCOL,'source_fingerprint':args.compiled_source_id,'source_preserved':True})
            write_json(target/'passed.json',result)
        elif args.mode=='freeze-run': result=freeze_run(args,repo,target)
        else: result=finalize(args,repo,target)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()

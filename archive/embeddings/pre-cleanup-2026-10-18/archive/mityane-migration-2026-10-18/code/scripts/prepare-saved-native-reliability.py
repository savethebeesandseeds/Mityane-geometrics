#!/usr/bin/env python3
"""Bind a closed saved-TRAIN diagnosis without model execution or head fitting.

Reuse the unchanged source-capture utilities from the prior protocol. Only those
source/metadata functions are called; its old main and quality code never run.
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
_spec = importlib.util.spec_from_file_location('source_capture', Path(__file__).with_name('prepare-fresh-decoder-replication.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
PROTOCOL = 'saved-native-reliability-v1'
CARD = 'code/evaluation/cards/saved_native_reliability_v1.md'
ROOT = 'output/runs/rpb-saved-native-reliability'
MARKER = 'Saved native reliability CPU fixtures passed'
MASTERS = [9109, 10210, 11311, 12412, 13513]
PARENT = 'output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b'
PARENT_SHA = 'a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90'
HEADER = ['id','display_tag','master_seed','controlled_training','native_training','encoder_progress',
          'fit_2701','pred_2701','fit_2802','pred_2802','fit_2903','pred_2903']
base.PROTOCOL, base.CARD, base.ROOT = PROTOCOL, CARD, ROOT
base.REQUIRED_SOURCES = {CARD, 'Makefile', 'dependencies.lock',
    'code/evaluation/src/saved_feature_reliability_main.cpp',
    'code/evaluation/include/frozen_role_guard.h',
    'code/shared/include/embedding/shared/saved_feature_reliability.h',
    'code/shared/src/saved_feature_reliability.cpp',
    'code/shared/tests/saved_feature_reliability_test.cpp',
    'code/scripts/prepare-fresh-decoder-replication.py',
    'code/scripts/prepare-saved-native-reliability.py',
    'code/scripts/check-saved-native-reliability.sh',
    'code/scripts/evaluate-saved-native-reliability.sh'}
base.REQUIRED_SOURCES |= {'code/evaluation/src/fresh_decoder_replication_main.cpp',
                         'code/shared/src/fixed_feature_readouts.cpp'}
require, file_sha, write_new, write_json, read_json = base.require, base.file_sha, base.write_new, base.write_json, base.read_json

def directory(repo, path, admission=False):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir() and not path.is_symlink(), 'regular new directory')
    parent = repo/ROOT/('admission' if admission else '')
    prefix = 'admission-' if admission else 'saved-native-reliability-'
    require(path.parent == parent and path.name.startswith(prefix), 'protocol directory bounds')
    return path

def binary_record(binary):
    path = Path(binary)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
            and path.is_relative_to(Path('/opt/cuwacunu_embedding/build'))
            and path.name == 'embedding_saved_feature_reliability', 'named container binary')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path)}

def admission_record(repo, target, card_sha, compiled_id, binary, sealed=False):
    base.admission_paths(target, sealed)
    _, source_id = base.source_state(repo, target, card_sha)
    require(source_id == compiled_id, 'compiled/source identity')
    plan = read_json(target/'admission-plan.json')
    require(plan['source_fingerprint'] == source_id and plan['human_card_sha256'] == card_sha
            and plan['quality_generated'] is False and plan['historical_quality_inputs'] == [], 'source-only admission plan')
    log = target/'build-and-tests.log'
    text = log.read_text(encoding='utf-8')
    require(MARKER in text and source_id in text and 'Frozen role guard checks passed' in text, 'actual source-bound CPU fixtures')
    return {'protocol': PROTOCOL, 'status': 'passed', 'source_fingerprint': source_id,
        'human_card_sha256': card_sha, 'source_preserved': True, 'log_sha256': file_sha(log),
        'compiled_binary': binary_record(binary), 'quality_generated': False,
        'historical_payload_roles': 0, 'head_fits': 0,
        'gates': ['synthetic CPU saved-map/logit/class/margin/geometry/trace fixtures',
                  'closed role/path/alias rejection before payload hashing; no encoder or fitted-head code linked']}

def check_plan(plan):
    require(plan['protocol'] == PROTOCOL and plan['master_seeds'] == MASTERS
            and plan['display_tags'] == ['RPB-v4.alt-01','RPB-v7.alt-01'], 'fixed instances and labels')
    for key, value in [('instances',10),('unique_train_roles',85),('encoder_updates',0),
                       ('decoder_updates',0),('head_refits',0),('heldout_roles',0),
                       ('testing_accessed',False),('stress_accessed',False)]:
        require(plan[key] == value, 'saved TRAIN plan: '+key)

def declared_rows(repo):
    rows = []
    for master in MASTERS:
        root = repo/PARENT/'results'/f'seed-{master}-lag_sign'
        for method in ('v4','v7'):
            paths = [root/'controlled-training.pt', root/'readouts'/f'native_{method}'/'training-features.pt',
                     root/method/'encoder-progress.json']
            for repetition in (2701,2802,2903):
                rep = root/'readouts'/f'native_{method}'/f'rep-{repetition}'
                paths += [rep/'fit.pt', rep/'training-predictions.pt']
            rows.append([f'{method}-{master}',f'RPB-{method}.alt-01',str(master),*[str(p) for p in paths]])
    return rows

def admit_matrix(paths, root):
    """All paths and aliases checked before any supplied file is hashed."""
    require(paths == sorted(set(paths)), 'unique sorted complete role list')
    inodes, result = set(), []
    for path in paths:
        require(path.is_absolute() and path.is_relative_to(root) and path.resolve(strict=True) == path
                and path.is_file() and not path.is_symlink(), 'closed regular role path')
        for ancestor in [path, *path.parents]:
            if ancestor == root.parent: break
            require(not ancestor.is_symlink(), 'redirected role ancestor')
        stat = path.stat(); inode = (stat.st_dev,stat.st_ino)
        require(stat.st_nlink == 1, 'external role hardlink')
        require(inode not in inodes, 'unrelated role inode alias'); inodes.add(inode)
        result.append((path,stat.st_size))
    return result

def input_records(repo, rows):
    paths = sorted({Path(p) for row in rows for p in row[3:]})
    require(len(paths) == 85, 'exact85 unique TRAIN roles')
    inventory = repo/PARENT/'artifact-integrity.json'
    # Admit metadata and the complete role matrix before its hashes/payload reads.
    admitted = admit_matrix(sorted([inventory,*paths]),repo/PARENT)
    require(file_sha(inventory) == PARENT_SHA, 'pinned parent inventory')
    metadata = read_json(inventory)
    require(metadata['protocol'] == 'fresh-decoder-replication-v1' and metadata['file_count'] == 948
            and len(metadata['files']) == 948, 'parent metadata shape')
    indexed = {x['path']:x for x in metadata['files']}
    require(len(indexed) == 948, 'duplicate inventory role')
    records = []
    for path,size in admitted:
        if path == inventory: continue
        name = path.relative_to(repo/PARENT).as_posix()
        require(name in indexed and size == indexed[name]['bytes'], 'pinned TRAIN role/size')
        expected = indexed[name]['sha256']
        require(re.fullmatch('[0-9a-f]{64}',expected), 'pinned TRAIN role digest format')
        records.append({'path':str(path),'bytes':size,'sha256':expected})
    # The entire inventory role/size/digest association is now admitted. Only
    # then may any of the 85 payload bytes be read for hashing.
    for record in records:
        require(file_sha(Path(record['path'])) == record['sha256'], 'pinned TRAIN role hash')
    return records, inventory

def bind_reader(repo, target, source, fixtures, seal):
    paths = sorted([Path(source),Path(fixtures),Path(seal)])
    require(len(set(paths)) == 3, 'distinct reader evidence')
    admit_matrix(paths,repo/ROOT/'audit-tools')
    metadata = read_json(Path(seal))
    require(metadata['status'] == 'passed' and metadata['source_sha256'] == file_sha(Path(source))
            and metadata['fixtures_sha256'] == file_sha(Path(fixtures)), 'presealed independent reader/fixtures')
    copied = target/'independent-reader'; copied.mkdir()
    names = {source:'source.py',fixtures:'source-fixtures.json',seal:'reader-seal.json'}
    records = []
    for original,name in names.items():
        path = Path(original); dest = copied/name; content = path.read_bytes(); write_new(dest,content)
        records.append({'path':str(path),'captured':str(dest),'bytes':len(content),'sha256':base.digest(content)})
    return records

def source_schema_check(repo):
    names = ['code/evaluation/src/fresh_decoder_replication_main.cpp',
             'code/shared/src/fixed_feature_readouts.cpp','code/shared/src/saved_feature_reliability.cpp']
    paths = sorted(repo/name for name in names); admit_matrix(paths,repo)
    fresh,readout,reader = [(repo/name).read_text(encoding='utf-8') for name in names]
    def body(text, signature):
        start = text.index(signature); return text[start:text.index('\n}',start)+2]
    def writes(text,variable): return set(re.findall(r'\b'+variable+r'\.write\("([a-z0-9_]+)"',text))
    def gets(text,variable): return set(re.findall(r'get\('+variable+r', "([a-z0-9_]+)"\)',text))
    actual = {
        'controlled':writes(body(fresh,'void save_observations('),'a')-{'requested_erasure'},
        'native':writes(body(readout,'void save_surface('),'out'),
        'fit':writes(body(readout,'void save_fit('),'out'),
        'prediction':writes(body(readout,'void save_prediction('),'out')}
    needed = {'controlled':gets(reader,'observations'),'native':gets(reader,'features'),
        'fit':gets(body(reader,'SavedFeatureFit load_fit('),'in') | {'outer_fitted_rows','fitted_rows','tiny_hidden','tiny_steps'},
        'prediction':gets(body(reader,'SavedFeaturePrediction load_prediction('),'in')}
    require(actual == needed,'reader/synthetic assumptions differ from actual producer archive keys')
    # The exact defect from the first blocked attempt must fail this source-only fixture.
    bad = {key:set(value) for key,value in needed.items()}
    bad['controlled'].remove('observations'); bad['controlled'].add('observed')
    require(actual != bad,'wrong observation key source fixture')
    return {key:sorted(value) for key,value in actual.items()}

def freeze_run(args, repo, target):
    require({p.name for p in target.iterdir()} == {'recipe-plan.json'}, 'exclusive new plan-only capsule')
    plan = read_json(target/'recipe-plan.json'); check_plan(plan)
    records = base.source_records(repo,args.sources,args.card_sha256)
    require(base.digest(base.manifest_bytes(records)) == args.compiled_source_id, 'run compiled/source identity')
    admission = directory(repo,Path(args.admission),True)
    current = admission_record(repo,admission,args.card_sha256,args.compiled_source_id,args.binary,True)
    require(read_json(admission/'passed.json') == current, 'unchanged passed admission')
    copied = target/'admission'; copied.mkdir()
    for path in base.admission_paths(admission,True):
        dest = copied/path.relative_to(admission); dest.parent.mkdir(parents=True,exist_ok=True); write_new(dest,path.read_bytes())
    require(base.copy_sources(repo,target,records) == args.compiled_source_id, 'source capture')
    reader = bind_reader(repo,target,args.reader_source,args.reader_fixtures,args.reader_seal)
    rows = declared_rows(repo)
    roles,inventory = input_records(repo,rows)
    write_new(target/'parent-inventory.json',inventory.read_bytes())
    write_new(target/'instances.tsv','\t'.join(HEADER)+'\n'+''.join('\t'.join(row)+'\n' for row in rows))
    write_new(target/'input-checksums.sha256',''.join(x['sha256']+'  '+x['path']+'\n' for x in roles))
    write_json(target/'input-role-plan.json',{'protocol':PROTOCOL,'roles':roles,'unique_train_roles':85,
        'parent_inventory_sha256':PARENT_SHA,'heldout_roles':0,'prior_mixed_audit_payload_read':False})
    argv = [args.binary,'--instances',str(target/'instances.tsv'),'--checksums',str(target/'input-checksums.sha256'),
        '--input-root',str(repo),'--output',str(target/'results'),'--admission-log',str(copied/'build-and-tests.log'),
        '--admission-sha256',current['log_sha256'],'--card',str(target/'source'/CARD),'--card-sha256',args.card_sha256]
    write_json(target/'launch-plan.json',{'protocol':PROTOCOL,'created_utc':base.utc(),'status':'frozen_before_TRAIN_analysis',
        'source_fingerprint':args.compiled_source_id,'human_card_sha256':args.card_sha256,
        'compiled_binary':current['compiled_binary'],'admission_passed_sha256':file_sha(copied/'passed.json'),
        'admission_log_sha256':current['log_sha256'],'reader':reader,'roles':roles,'argv':argv,
        'instances_sha256':file_sha(target/'instances.tsv'),'checksums_sha256':file_sha(target/'input-checksums.sha256'),
        'recipe_sha256':file_sha(target/'recipe-plan.json'),'encoder_updates':0,'decoder_updates':0,'head_refits':0,'heldout_roles':0})
    base.source_state(repo,target,args.card_sha256)
    return {'protocol':PROTOCOL,'source_fingerprint':args.compiled_source_id,'unique_train_roles':85}

def finalize(args, repo, target):
    _,source_id = base.source_state(repo,target,args.card_sha256)
    require(source_id == args.compiled_source_id, 'unchanged live/captured sources')
    launch = read_json(target/'launch-plan.json')
    require(binary_record(launch['argv'][0]) == launch['compiled_binary'], 'tested binary preserved')
    for name,key in [('instances.tsv','instances_sha256'),('input-checksums.sha256','checksums_sha256'),('recipe-plan.json','recipe_sha256')]:
        require(file_sha(target/name) == launch[key], 'run metadata preserved')
    check_plan(read_json(target/'recipe-plan.json'))
    require(file_sha(target/'admission/passed.json') == launch['admission_passed_sha256']
            and file_sha(target/'admission/build-and-tests.log') == launch['admission_log_sha256'], 'admission preserved')
    paths = sorted(Path(x['path']) for x in launch['roles']); admit_matrix(paths,repo/PARENT)
    for record in launch['roles']:
        path = Path(record['path']); require(path.stat().st_size == record['bytes'] and file_sha(path) == record['sha256'], 'input preserved')
    for record in launch['reader']:
        require(file_sha(Path(record['path'])) == file_sha(Path(record['captured'])) == record['sha256'], 'reader preserved')
    complete = read_json(target/'results/complete.json')
    require(complete['protocol'] == PROTOCOL and complete['status'] == 'complete', 'measurement complete')
    write_json(target/'source-preserved-after.json',{'protocol':PROTOCOL,'source_fingerprint':source_id,
        'source_preserved':True,'inputs_preserved':True,'reader_preserved':True,'created_utc':base.utc()})
    paths = sorted(p for p in target.rglob('*') if p.is_file() or p.is_symlink()); admitted = admit_matrix(paths,target)
    files = [{'path':p.relative_to(target).as_posix(),'bytes':size,'sha256':file_sha(p)} for p,size in admitted]
    write_json(target/'artifact-integrity.json',{'schema_version':1,'protocol':PROTOCOL,'created_utc':base.utc(),
        'inventory_excludes_itself':True,'checksum_algorithm':'sha256-file-bytes','file_count':len(files),
        'total_bytes':sum(x['bytes'] for x in files),'files':files})
    return {'protocol':PROTOCOL,'status':'complete','source_fingerprint':source_id,
        'inventory_sha256':file_sha(target/'artifact-integrity.json'),'file_count':len(files)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['self-test','freeze-admission','admit','freeze-run','finalize'])
    for name in ['repo-root','target','card-sha256','compiled-source-id','admission','binary','reader-source','reader-fixtures','reader-seal']:
        parser.add_argument('--'+name,default='.' if name == 'repo-root' else None)
    parser.add_argument('--sources',nargs='+'); args = parser.parse_args()
    require(Path('/.dockerenv').exists(),'managed container required')
    if args.mode == 'self-test':
        # Synthetic source and metadata fixtures only; no measured files opened.
        plan = {'protocol':PROTOCOL,'master_seeds':MASTERS,'display_tags':['RPB-v4.alt-01','RPB-v7.alt-01'],
            'instances':10,'unique_train_roles':85,'encoder_updates':0,'decoder_updates':0,'head_refits':0,
            'heldout_roles':0,'testing_accessed':False,'stress_accessed':False}
        check_plan(plan); negatives = 0
        for key,value in [('instances',9),('unique_train_roles',84),('encoder_updates',1),('head_refits',1),('heldout_roles',1),('testing_accessed',True)]:
            bad = dict(plan); bad[key] = value
            try: check_plan(bad)
            except AssertionError: negatives += 1
            else: raise AssertionError('unsafe plan accepted')
        rows = declared_rows(Path('/embedding'))
        require(len(rows) == 10 and all(len(row) == 12 for row in rows)
                and len({p for row in rows for p in row[3:]}) == 85, 'closed source-only role enumeration')
        hashes = []; original_hash = base.file_sha
        def counted_hash(path):
            hashes.append(str(path)); return original_hash(path)
        with tempfile.TemporaryDirectory(prefix='saved-native-source-only-') as temporary:
            fixture = Path(temporary).resolve(strict=True); names = sorted(base.REQUIRED_SOURCES)
            for name in names:
                path = fixture/name; path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('source-only fixture\n',encoding='utf-8')
            base.file_sha = counted_hash
            try:
                for failure in ('missing','alias'):
                    extra = 'zz_missing.cpp' if failure == 'missing' else 'zz_alias.cpp'
                    if failure == 'alias': os.link(fixture/'Makefile',fixture/extra)
                    try: base.source_records(fixture,sorted(names+[extra]),base.digest(b'source-only fixture\n'))
                    except (AssertionError,FileNotFoundError): pass
                    else: raise AssertionError('unsafe entire source matrix accepted')
                    require(not hashes,'source hash occurred before complete matrix admission')
                records = base.source_records(fixture,names,base.digest(b'source-only fixture\n'))
                require(len(records) == len(hashes) == len(names),'valid source matrix hashes')
            finally: base.file_sha = original_hash
            a,b = fixture/'role-a.json',fixture/'role-b.json'
            a.write_text('{}\n',encoding='utf-8'); os.link(a,b)
            try: admit_matrix(sorted([a,b]),fixture)
            except AssertionError: pass
            else: raise AssertionError('unsafe metadata inode alias accepted')
        print(json.dumps({'protocol':PROTOCOL,'status':'passed','negative_plan_fixtures':negatives,
            'whole_matrix_source_fixtures':3,'metadata_alias_fixture':1,
            'producer_source_schemas':source_schema_check(Path(args.repo_root).resolve(strict=True)),
            'instances':10,'unique_train_roles':85,'quality_payload_reads':0})); return
    require(args.target and args.card_sha256 and re.fullmatch('[0-9a-f]{64}',args.card_sha256),'target/card binding')
    repo = Path(args.repo_root).resolve(strict=True)
    target = directory(repo,Path(args.target),args.mode in ('freeze-admission','admit'))
    if args.mode == 'freeze-admission':
        result = {'source_fingerprint':base.freeze_admission(repo,target,args.sources,args.card_sha256)}
    elif args.mode == 'admit':
        result = admission_record(repo,target,args.card_sha256,args.compiled_source_id,args.binary)
        write_json(target/'source-preserved-after.json',{'protocol':PROTOCOL,'source_fingerprint':args.compiled_source_id,'source_preserved':True})
        write_json(target/'passed.json',result)
    elif args.mode == 'freeze-run': result = freeze_run(args,repo,target)
    else: result = finalize(args,repo,target)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Independent saved CPU arithmetic for TEMPO-3. Seal SOURCE before generation;
authorize only a new two-flag copy after the completed capsule inventory.
No model, optimizer, head, PCA/SVD or dataset generator is executed here.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tempfile
import time

sys.dont_write_bytecode = True
REVIEWED_SCHEMA = False
MEASURED_IMPLEMENTATION = False
PROTOCOL = 'structured-hard-timing-comparison-v2'
FIT = 'early-mixer-reliability-v1/lag_sign'
EXTERNAL = PROTOCOL + '/lag_sign'
RULE = 'observed-cross-feature-determinant-v2'
CARD = 'code/evaluation/cards/structured_hard_timing_comparison_v2.md'
CARD_SHA = '35f7aa9987f293bafbe735506cc86cd2af02f2ab39bec42dd1dc2d2dedfa758c'
ROOT = Path('/embedding/output/runs/rpb-structured-hard-timing')
MASTERS = (75272,76373,77474,78575,79676)
TAGS = ('RPB-v7.alt-05','RPB-v10.alt-05')
MODULE_PINS = {'saved_cpu_math.py':'1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9',
               'legacy_implementation.py':'f0c64b9fe67a90a9d95196433e05dec527515778d1635bdc909d1703fa278b68',
               'archive_codec.py':'4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d'}
M = L = R = None
CHECKS = ARCHIVES = 0
ADMITTED = {}
CACHE = {}


def check(ok, message):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(path, directory=False):
    path = Path(path)
    check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(), 'absolute canonical direct path')
    check(all(not p.is_symlink() for p in path.parents), 'no aliased path ancestors')
    check(path.is_dir() if directory else path.is_file() and path.stat().st_nlink == 1, 'regular non-hardlinked file/directory')
    return path


def whole_paths(paths):
    paths = [canonical(p) for p in paths]
    check(len(paths) == len(set(paths)), 'unique whole path matrix')
    inodes = [(p.stat().st_dev,p.stat().st_ino) for p in paths]
    check(len(inodes) == len(set(inodes)), 'no cross-file inode aliases')
    return paths


def metadata(path):
    path = canonical(path)
    def pairs(items):
        result = {}
        for key,value in items:
            check(key not in result,'unique metadata keys')
            result[key] = value
        return result
    def invalid(value):
        raise AssertionError('nonfinite JSON '+value)
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=invalid)


def import_source(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def initialize_modules():
    global M,L,R
    directory = canonical(Path(__file__).parent,True)
    paths = whole_paths([directory/name for name in MODULE_PINS])
    for path in paths:
        check(sha(path) == MODULE_PINS[path.name], 'exact standalone SOURCE module '+path.name)
    M = import_source(directory/'saved_cpu_math.py','saved_cpu_math')
    R = import_source(directory/'archive_codec.py','structured_timing_archive_codec')
    M.R = R
    L = import_source(directory/'legacy_implementation.py','structured_timing_legacy_contracts')
    L.load,L.read_json,L.sha = load,metadata,sha


def load(path):
    global ARCHIVES
    path = Path(path)
    check(path.name != 'checkpoint.pt','ordinary CUDA checkpoint bodies are hash/FNV bound and never decoded')
    check(path in ADMITTED and path.suffix == '.pt','only the closed new saved-archive matrix may be decoded')
    if path not in CACHE:
        ARCHIVES += 1
        CACHE[path] = R.load(path)
    return CACHE[path]


def relative_name(name):
    p = PurePosixPath(name)
    check(isinstance(name,str) and str(p)==name and not p.is_absolute() and '\\' not in name and
          all(x not in ('','.','..') for x in p.parts),'normalized relative inventory name')
    return name


def admit_capsule(capsule):
    capsule = canonical(capsule,True)
    check(capsule.parent==ROOT and capsule.name.startswith('structured-hard-timing-'),'sole new protocol capsule')
    inventory_path = canonical(capsule/'artifact-integrity.json')
    inventory = metadata(inventory_path)
    check(inventory['protocol']==PROTOCOL and inventory['status']=='complete','completed immutable inventory')
    rows = inventory['files']; names = [relative_name(x['path']) for x in rows]
    check(names==sorted(set(names)) and 'artifact-integrity.json' not in names,'whole sorted inventory matrix')
    actual_names = sorted(p.relative_to(capsule).as_posix() for p in capsule.rglob('*') if p.is_file() and p!=inventory_path)
    check(actual_names==names,'no omitted or added capsule files')
    paths = whole_paths([capsule/name for name in names])
    check(all(p.stat().st_size==row['bytes'] for p,row in zip(paths,rows)),'whole sizes admitted before hashes')
    index = {}
    for path,row in zip(paths,rows):
        check(re.fullmatch('[0-9a-f]{64}',row['sha256']) and sha(path)==row['sha256'],'saved immutable capsule bytes')
        index[row['path']] = row
    check(inventory['files_count']==len(rows) and inventory['total_bytes']==sum(x['bytes'] for x in rows),'inventory population and total bytes')
    return index


def release_binding(capsule):
    before = (capsule/'independent-reader/validate_structured_hard_timing.py').read_bytes()
    runtime = Path(__file__).read_bytes(); normalized = runtime
    for name in (b'REVIEWED_SCHEMA',b'MEASURED_IMPLEMENTATION'):
        true = rb'(?m)^'+name+rb' = True$'; false = rb'(?m)^'+name+rb' = False$'
        check(len(re.findall(true,normalized))==1 and not re.findall(false,normalized),'one enabled release line')
        check(len(re.findall(false,before))==1 and not re.findall(true,before),'one prequality disabled line')
        normalized = re.sub(true,name+b' = False',normalized)
    check(normalized==before,'full two-line reverse byte proof, no numeric change')
    seal = metadata(capsule/'independent-reader/reader-seal.json')
    fixtures = metadata(capsule/'independent-reader/source-fixtures.json')
    sealed_names=[relative_name(row['path']) for row in seal['files']]
    check(sealed_names==sorted(set(MODULE_PINS)|{'validate_structured_hard_timing.py','source-fixtures.json'}),'complete exact prequality bundle file matrix')
    sealed_paths=whole_paths([capsule/'independent-reader'/name for name in sealed_names])
    for row,path in zip(seal['files'],sealed_paths):
        check(row['bytes']==path.stat().st_size and row['sha256']==sha(path),'every prequality SOURCE/seal role bytes')
    check(seal['status']=='passed' and seal['measured_execution_enabled'] is False and
          seal['source_sha256']==hashlib.sha256(before).hexdigest() and
          seal['fixtures_sha256']==sha(capsule/'independent-reader/source-fixtures.json') and
          fixtures['status']=='passed' and fixtures['archive_reads']==0 and fixtures['measured_execution_enabled'] is False,
          'prospective sealed SOURCE and artificial-only fixtures')
    for name,pin in MODULE_PINS.items():
        check(sha(capsule/'independent-reader'/name)==pin,'captured reusable SOURCE module')
    return {'reader_sha256':hashlib.sha256(runtime).hexdigest(),'prequality_reader_sha256':hashlib.sha256(before).hexdigest(),
            'fixtures_sha256':seal['fixtures_sha256'],'two_flag_reverse_byte_proof':True,'modules':MODULE_PINS}


def source_binding(capsule,index):
    manifest = metadata(capsule/'source-manifest.json')
    rows = manifest['files']; names = [relative_name(x['path']) for x in rows]
    check(names==sorted(set(names)) and all(n.startswith('code/') or n in ('Makefile','setup.sh','dependencies.lock','doc/dataset_registry.json') for n in names),
          'whole closed compiled SOURCE name matrix')
    check({CARD,'setup.sh','code/scripts/install-libtorch.py'}<=set(names),'card and internal SDK SOURCE')
    for row in rows:
        for prefix in ('source/','admission/source/'):
            check(prefix+row['path'] in index and all(index[prefix+row['path']][k]==row[k] for k in ('bytes','sha256')),'whole measured/admission SOURCE equality')
    body = ''.join(x['sha256']+'  '+x['path']+'\n' for x in rows)
    identity = hashlib.sha256(body.encode()).hexdigest()
    check(identity==manifest['source_sha256'] and manifest['protocol']==PROTOCOL and
          (capsule/'source-inputs.sha256').read_text()==body and (capsule/'source-fingerprint.txt').read_text()==identity+'\n' and
          metadata(capsule/'admission/source-manifest.json')==manifest,'canonical enclosing SOURCE ID')
    check(index['source/'+CARD]['sha256']==CARD_SHA,'fixed card literal')
    helper_path = capsule/'source/code/evaluation/protocols/structured_hard_timing_v1/prepare_saved_run.py'
    helper = import_source(helper_path,'captured_structured_timing_metadata_contract')
    # Import only this captured, inventory-bound stdlib SOURCE. Never call its build/run/generator functions.
    plan = metadata(capsule/'results/recipe-plan.json'); helper.validate_plan(plan)
    check(plan['source_fingerprint']==identity,'actual compiled plan SOURCE ID')
    scopes = metadata(capsule/'admission/source-scopes.json')
    check(set(scopes)=={'core_writer','curve_training','early_adapter','structured_adapter'},'four separate compiled scopes')
    for name,scope in scopes.items():
        paths = [relative_name(x['path']) for x in scope['files']]
        check(paths==sorted(set(paths)) and set(paths)<=set(names),'canonical SOURCE scope subset')
        check(scope['files']==[next(x for x in rows if x['path']==n) for n in paths],'scope byte records exactly captured')
        canonical_scope = ''.join(x['sha256']+'  '+x['path']+'\n' for x in scope['files'])
        check(hashlib.sha256(canonical_scope.encode()).hexdigest()==scope['source_sha256'],'derived compiler SOURCE scope identity')
    check(scopes['structured_adapter']['source_sha256']==identity,'new loader enclosing scope')
    launch = metadata(capsule/'launch-plan.json'); passed = metadata(capsule/'admission/passed.json')
    check(launch['source_sha256']==passed['source_sha256']==identity and launch['card_sha256']==passed['card_sha256']==CARD_SHA and
          passed['status']=='passed' and passed['quality_generated'] is False and launch['historical_payload_roles']==[] and
          launch['quality_generated_before_freeze'] is False,'actual admission and frozen launch')
    check(launch['admission_sha256']==sha(capsule/'admission/passed.json') and passed['log']['sha256']==sha(capsule/'admission/build-and-tests.log') and
          launch['binary']==passed['binary'],'binary/log/admission association')
    log = (capsule/'admission/build-and-tests.log').read_text()
    check(all(x in log for x in helper.QUALITY_MARKERS+['Container SDK proof:','TEMPO-3 observed-only information admission passed:']), 'actual CUDA/head/role/SDK/information gates')
    binary = Path(passed['binary']['path']); old = binary.parent/'code/encoders/raw_patch_bottleneck_mae'; new=binary.parent/'code/protocols/structured_hard_timing_v1'
    old_source='code/encoders/raw_patch_bottleneck_mae/src/'; new_source='code/evaluation/protocols/structured_hard_timing_v1/'; data_source='code/evaluation/benchmarks/structured_hard_timing/'
    contracts=[(old_source+'workflow.cpp',old/'workflow.o','RPB_SOURCE_ID','core_writer'),
               (old_source+'learning_curve_adapter.cpp',old/'learning_curve_adapter.o','EVALUATION_SOURCE_ID','curve_training'),
               (old_source+'early_mixer_adapter.cpp',old/'early_mixer_adapter.o','EARLY_MIXER_ADAPTER_SOURCE_ID','early_adapter'),
               (new_source+'structured_hard_timing_adapter.cpp',new/'structured_hard_timing_adapter.o','STRUCTURED_HARD_TIMING_ADAPTER_SOURCE_ID','structured_adapter'),
               (new_source+'paired_timing_run.cpp',new/'paired_timing_run.o','EVALUATION_SOURCE_ID','structured_adapter'),
               (new_source+'structured_hard_timing_main.cpp',new/'structured_hard_timing_main.o','EVALUATION_SOURCE_ID','structured_adapter'),
               (data_source+'structured_hard_timing.cpp',new/'structured_hard_timing_generator.o','EVALUATION_SOURCE_ID','structured_adapter'),
               (data_source+'cross_feature_solvability.cpp',new/'cross_feature_solvability.o','EVALUATION_SOURCE_ID','structured_adapter')]
    commands = helper.compile_commands(log,[(s,o) for s,o,_,_ in contracts],binary)
    for path,_,macro,scope in contracts:
        args = shlex.split(commands['commands'][path]); found=[a.split('=',1)[1].strip('"') for a in args if a.startswith('-D'+macro+'=')]
        check(found==[scopes[scope]['source_sha256']],'one current exact object/macro record, no cached fallback')
    info = metadata(capsule/'admission/information/information-record.json')
    check(info['status']=='passed' and info['passed'] is True and info['protocol']=='structured-hard-timing-information-binding-v2' and
          info['rule_id']==RULE and info['card_sha256']==CARD_SHA and info['archive_reads']==info['encoder_or_head_fits']==0 and
          info['source_preserved_after'] is True,'separate fixed information PASS admission')
    helper.validate_information(info['information'])
    check(metadata(capsule/'admission/information/information.json')==info['information'],'raw admitted information metadata')
    check(passed['information_record']['sha256']==sha(capsule/'admission/information/information-record.json'),'exact admitted information binding bytes')
    information_manifest=metadata(capsule/'admission/information/source-manifest.json')
    information_names=[relative_name(x['path']) for x in information_manifest['files']]
    check(information_names==sorted(set(information_names)) and set(information_names)<=set(names),'complete information SOURCE subset before record binding')
    for row in information_manifest['files']:
        check(all(index['source/'+row['path']][k]==row[k] for k in ('bytes','sha256')),'same exact generator/rule/card dependencies as information admission')
    information_body=''.join(x['sha256']+'  '+x['path']+'\n' for x in information_manifest['files'])
    check(hashlib.sha256(information_body.encode()).hexdigest()==info['source_sha256']==information_manifest['source_sha256'] and
          (capsule/'admission/information/source-inputs.sha256').read_text()==information_body and
          (capsule/'admission/information/source-fingerprint.txt').read_text()==info['source_sha256']+'\n','canonical separately captured information SOURCE identity')
    allowed_info={'information.json','build-and-tests.log','source-manifest.json','source-inputs.sha256','source-fingerprint.txt','freeze.json'}
    input_names=[Path(x['path']).name for x in info['inputs']]
    check(set(input_names)==allowed_info|{'cross_feature_test'} and len(input_names)==7,'closed information metadata/binary evidence roles')
    for row in info['inputs']:
        basename=Path(row['path']).name
        if basename=='cross_feature_test':check(Path(row['path'])==helper.DATA_BINARY,'actual fixed data-only binary declaration')
        else:check(all(index['admission/information/'+basename][k]==row[k] for k in ('bytes','sha256')),'captured information evidence bytes')
    info_log=(capsule/'admission/information/build-and-tests.log').read_text()
    check(helper.DATA_MARKER in info_log,'actual analytic engineering fixture marker')
    helper.compile_commands(info_log,[('code/evaluation/benchmarks/structured_hard_timing/structured_hard_timing.cpp',helper.DATA_BINARY.parent/'structured_hard_timing.o'),
                           ('code/evaluation/benchmarks/structured_hard_timing/cross_feature_solvability.cpp',helper.DATA_BINARY.parent/'cross_feature_solvability.o'),
                           ('code/evaluation/benchmarks/structured_hard_timing/cross_feature_test.cpp',helper.DATA_BINARY.parent/'cross_feature_test.o')],helper.DATA_BINARY)
    check(metadata(capsule/'results/card.json')=={'protocol':PROTOCOL,'human_card_sha256':CARD_SHA,'source_fingerprint':identity,
          'historical_payload_roles':0,'testing_accessed':False,'stress_accessed':False},'exact instantiated prospective card')
    check(metadata(capsule/'admission/recipe-plan.json')==plan,'same admission and measured recipe')
    preserved=metadata(capsule/'source-preserved-after.json')
    check(preserved=={'status':'passed','source_sha256':identity,'binary_preserved':True,'reader_preserved':True,'information_preserved':True},'capture inputs preserved after quality')
    for fixture_row in metadata(capsule/'independent-reader/source-fixtures.json')['writer_source_schema']:
        check(index['source/'+fixture_row['path']]['sha256']==fixture_row['sha256'],'SOURCE fixture bound to actual captured writer')
    return {'source_fingerprint':identity,'human_card_sha256':CARD_SHA,'actual_compile_scopes':{k:v['source_sha256'] for k,v in scopes.items()},
            'admission_sha256':sha(capsule/'admission/passed.json'),'admission_log_sha256':sha(capsule/'admission/build-and-tests.log'),
            'information_sha256':sha(capsule/'admission/information/information-record.json'),'inventory_sha256':sha(capsule/'artifact-integrity.json'),
            'captured_source_files':len(rows),'capsule':str(capsule),**release_binding(capsule)}


def observations(value,rows,master,deleted=False):
    keys={'observations','feature_mask','labels_scoring_only','source_ids_json'}|({'requested_erasure'} if deleted else set())
    check(set(value)==keys,'literal controlled archive keys')
    data=M.tensor(value['observations'],'DoubleStorage',[rows,3,32,3]); mask=M.tensor(value['feature_mask'],'BoolStorage',[rows,3,32,3])
    labels=list(M.tensor(value['labels_scoring_only'],'LongStorage',[rows])); ids=json.loads(M.text(value,'source_ids_json'))
    pattern=re.escape(EXTERNAL+'/structured-hard-timing-v1/seed-'+str(master)+'/lag_sign/source-')+'[0-9]+'
    check(len(ids)==rows and len(set(ids))==rows//2 and all(isinstance(s,str) and re.fullmatch(pattern,s) for s in ids),'new source namespace and pair populations')
    M.finite(data,'finite legal controlled observations')
    check(all(keep or x==0 for x,keep in zip(data,mask)),'zero hidden storage')
    M.validate_pairs(labels,ids,[list(mask[b*288:(b+1)*288]) for b in range(rows)])
    return data,mask,ids,labels


def deleted_view(base,deleted,asset,master):
    data,mask,ids,labels=base; dv,dm,di,dl=deleted
    check(di==ids and dl==labels,'deletion preserves row/label/source order')
    namespace=EXTERNAL+'/validation-coordinate-dropout'; seed=M.stream_seed(master,0x746d70332d64656c)
    expected=[]
    for source in ids:
        key=namespace+'/lag_sign/coordinate/'+str(len(source.encode()))+':'+source
        engine=M.MT19937_64(M.stream_seed(seed,M.fnv64(key)))
        expected.extend((engine.draw()>>11)*(2.**-53)<.30 for _ in range(288))
    requested=list(M.tensor(asset['requested_erasure'],'BoolStorage',[128,3,32,3]))
    check(requested==expected,'fixed source-pair shared MT64 coordinate deletion')
    wanted,support=M.deletion_view(data,mask,requested)
    check(list(dv)==wanted and list(dm)==support,'deletion only; no invented observations or mask repair')
    return {'rate':.30,'namespace':namespace,'actual_seed_decimal':str(seed),'requested_coordinates':sum(requested),
            'observed_coordinates_before':sum(mask),'observed_coordinates_after':sum(dm)}


def analytic_values(data,mask,rows):
    check(len(data)==len(mask)==rows*288 and rows>0,'closed analytic C3/H32/F3 geometry')
    margins=[]; counts=[]
    for row in range(rows):
        centres=[]
        for centre in range(1,31):
            spacings=[]
            for k in range(1,5):
                if centre-k<0 or centre+k>=32: continue
                channels=[]
                for c in range(2):
                    left=[]; right=[]
                    for f in range(3):
                        indices=[row*288+c*96+h*3+f for h in (centre-k,centre,centre+k)]
                        if not all(mask[i] for i in indices): continue
                        left.append(data[indices[1]]-data[indices[0]]); right.append(data[indices[2]]-data[indices[1]])
                    channels.append((sum(left)/len(left),sum(right)/len(right)) if left else None)
                if None not in channels:
                    a,b=channels; spacings.append(b[0]*a[1]-a[0]*b[1])
            if spacings: centres.append(sum(spacings)/len(spacings))
        counts.append(len(centres)); margins.append(sum(centres)/len(centres) if centres else 0.)
    M.finite(margins,'finite observed-only analytic margins')
    return margins,counts


def analytic_archive(asset,split,declared):
    data,mask,ids,labels=split; rows=len(ids)
    check(set(asset)=={'predictions','valid','margins','supported_time_positions','labels_scoring_only','source_ids_json','observed_only','fitted_heads','information_rule_id'},'closed observed-only analytic archive')
    predictions=list(M.tensor(asset['predictions'],'LongStorage',[rows]));valid=list(M.tensor(asset['valid'],'BoolStorage',[rows]))
    saved=list(M.tensor(asset['margins'],'DoubleStorage',[rows])); counts=list(M.tensor(asset['supported_time_positions'],'LongStorage',[rows]))
    check(M.scalar(asset,'observed_only','BoolStorage')==1 and M.scalar(asset,'fitted_heads')==0 and M.text(asset,'information_rule_id')==RULE and
          list(M.tensor(asset['labels_scoring_only'],'LongStorage',[rows]))==labels and json.loads(M.text(asset,'source_ids_json'))==ids,'analytic scoring-only labels and no fitted inputs')
    expected_margins,expected_counts=analytic_values(data,mask,rows)
    check(counts==expected_counts,'exact distinct supported centre counts')
    for actual,wanted in zip(saved,expected_margins):M.close(actual,wanted,'independent cross-feature determinant arithmetic')
    check(valid==[count>=4 and margin!=0 for count,margin in zip(counts,saved)] and
          predictions==[int(margin>0) if keep else 0 for margin,keep in zip(saved,valid)],'saved own-margin decisions and exact invalid-zero law')
    supported=sum(valid); correct=sum(int(p==y) for p,y,keep in zip(predictions,labels,valid) if keep)
    wanted={'status':'measured' if supported else 'unsupported_view','method':'analytic_observed_only','information_rule_id':RULE,
            'fitted_heads':0,'total_examples':rows,'valid_examples':supported,'correct_supported':correct,'coverage':supported/rows,
            'full_population_correctness':correct/rows,'accuracy':correct/supported if supported else None,
            'minimum_supported_time_positions':4,'nonzero_margin_required':True,'observed_only':True}
    check(set(declared)==set(wanted),'closed analytic reported schema')
    for key,value in wanted.items():
        if type(value) is float:M.close(declared[key],value,'analytic reported rational score')
        else:check(declared[key]==value,'analytic reported support/null/recipe '+key)
    return wanted


def external_binding(asset,audit,info,cp,placement,point,ids,scopes,snapshot=False):
    fields={'artifact_kind':'rpb_structured_hard_timing_cuda_snapshot_v2' if snapshot else 'rpb_structured_hard_timing_binding_v2',
            'structured_timing_protocol_id':PROTOCOL,'information_rule_id':RULE,'dataset_codename':'TEMPO-3',
            'dataset_generator_protocol_id':'structured-hard-timing-v1','designed_complexity_level':'4','designed_complexity_scale_max':'5',
            'external_fit_protocol_id':EXTERNAL,'training_implementation_fit_protocol_id':FIT,'structured_timing_scope':'quality',
            'structured_timing_role':'early' if placement else 'late','structured_timing_instance_group':TAGS[placement],
            'structured_timing_adapter_source_fingerprint':scopes['structured_adapter'],'parent_core_writer_source_fingerprint':scopes['core_writer'],
            'parent_training_producer_source_fingerprint':scopes['curve_training'],'parent_snapshot_adapter_source_fingerprint':scopes['early_adapter'],
            'training_dataset_id':info['dataset_id'],'training_schema_id':info['schema_id'],'preprocessing_id':info['scaler_id'],
            'original_training_source_manifest':M.source_manifest(ids),'original_training_source_manifest_id':M.text(audit,'fit_source_manifest_id'),
            'initialization_seed':M.text(audit,'initialization_seed'),'actual_training_seed':M.text(audit,'actual_training_seed'),
            'training_policy_id':L.POLICY,'architecture_id':L.ARCHITECTURES[placement],'feature_units':'unitless,unitless,unitless',
            'snapshot_policy':'new-structured-timing-binding;unchanged-reliability-CUDA-serving;no-old-quality-input'}
    for key,suffix in zip(('content_checkpoint','content_audit','content_scaler','content_training_raw'),('','.audit.pt','.scaler.pt','.training-raw.pt')):
        fields[key]='fnv1a64-runtime-content-v1-'+M.fnv(Path(str(cp)+suffix).read_bytes())
    if snapshot:fields['content_structured_timing']='fnv1a64-runtime-content-v1-'+M.fnv(Path(str(cp)+'.structured-timing.pt').read_bytes())
    values={'channel_mixer_placement_value':placement,'attempted_steps':point,'completed_steps':point,'sampled_rows':point*8,'parameter_count_value':225805}
    values.update(dict(zip(('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'),info['counts'])))
    check(set(asset)==set(fields)|set(values),'closed fifth companion/supplemental snapshot schema')
    for key,value in fields.items():check(M.text(asset,key)==value,'external versus truthful implementation/source identity '+key)
    for key,value in values.items():check(M.scalar(asset,key)==value,'typed external point/counter '+key)
    check(M.text(audit,'protocol_id')==FIT and FIT!=EXTERNAL,'old implementation protocol not retagged')
    return fields


def archive_matrix(capsule,report,index):
    wanted=set(); results=capsule/'results'
    for cohort,master in zip(report['cohorts'],MASTERS):
        base=results/f'seed-{master}-lag_sign'
        for name in ('training','validation','validation-deleted'):wanted.add(base/('controlled-'+name+'.pt'))
        wanted.add(base/'initial-pair.pt')
        for role in ('late','early'):
            for point in (0,512):
                cp=base/role/f'point-{point}/checkpoint.pt'
                wanted.update(Path(str(cp)+suffix) for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.structured-timing.pt'))
                wanted.update(cp.parent/'snapshot-assets'/name for name in ('early-mixer-snapshot-audit.pt','structured-timing-snapshot-audit.pt'))
                if point:wanted.update(cp.parent/(name+'-reconstruction.pt') for name in ('training','validation'))
        task=base/'lag_sign';wanted.update(task/name for name in ('raw-scaler.pt','raw-outer-normalizer.pt','analytic-training.pt','analytic-validation-intact.pt','analytic-validation-deleted.pt'))
        pca=metadata(task/'pca-status.json')
        if pca['status']=='measured':wanted.add(task/'pca-preprocessing.pt')
        methods=cohort['tasks'][0]['readouts']['methods']
        check([x['method'] for x in methods]==list(L.METHODS),'all seven fixed methods before archive access')
        for method in methods:
            path=task/'readouts'/method['method']
            wanted.update(path/(view+'-features.pt') for view in L.VIEWS)
            for repetition in method['repetitions']:
                if method['status']=='measured':
                    rep=path/repetition['id'];wanted.add(rep/'fit.pt');wanted.update(rep/(view+'-predictions.pt') for view in L.VIEWS)
    actual={capsule/name for name in index if name.endswith('.pt')}
    check(actual==wanted,'complete exact new archive roles; no unknown/old/TEST/continuation inputs')
    for path in wanted:ADMITTED[path]=index[path.relative_to(capsule).as_posix()]
    return len(wanted)


def run(capsule):
    started=time.monotonic();index=admit_capsule(capsule)
    initialize_modules();source=source_binding(capsule,index); scopes=source['actual_compile_scopes']
    results=capsule/'results';report=metadata(results/'report.json');complete=metadata(results/'complete.json')
    check(report['protocol']==PROTOCOL and report['source_fingerprint']==source['source_fingerprint'] and report['information_rule_id']==RULE and
          len(report['cohorts'])==5 and all(report[k] is False for k in ('testing_accessed','stress_accessed','selection','promotion')),'complete five-cohort fixed report')
    roles=archive_matrix(capsule,report,index);audited=[];all_sources=set();pipelines=helper_outer=0
    for cohort,master in zip(report['cohorts'],MASTERS):
        check(cohort['timing_master']==master and len(cohort['encoder_points'])==2 and len(cohort['tasks'])==1,'fixed single-task paired cohort matrix')
        base=results/f'seed-{master}-lag_sign';files=[load(base/('controlled-'+n+'.pt')) for n in ('training','validation','validation-deleted')]
        data=[observations(a,n,master,i==2) for i,(a,n) in enumerate(zip(files,(256,128,128)))];view=deleted_view(data[1],data[2],files[2],master)
        check(set(data[0][2]).isdisjoint(data[1][2]),'TRAIN/VAL source split')
        for split in data[:2]:
            ids=set(split[2]);check(all_sources.isdisjoint(ids),'all five source populations independent');all_sources.update(ids)
        initial=L.initial_witness(load(base/'initial-pair.pt'))
        init=metadata(base/'initialization-audit.json')
        expected_init={'common_parameters_exact':'true','common_buffers_exact':'true','scaler_exact':'true','training_dataset_exact':'true','counter_streams_exact':'true',
                       'parameter_count':'225805','late_architecture_id':L.ARCHITECTURES[0],'early_architecture_id':L.ARCHITECTURES[1],
                       'initialization_seed':str(M.mixed(master^0x7270622d696e6974)),'initial_features_equality_required':'false',
                       'snapshot_loader_source_fingerprint':scopes['early_adapter'],'structured_timing_protocol_id':PROTOCOL,
                       'external_fit_protocol_id':EXTERNAL,'training_implementation_fit_protocol_id':FIT,
                       'structured_timing_adapter_source_fingerprint':scopes['structured_adapter'],'historical_quality_input_roles':'0'}
        check(init==expected_init,'complete paired initializer/source/dual namespace witness')
        check(metadata(base/'initial-pair.json')=={'full_named_parameters_exact':True,'buffers_exact':True,'scaler_exact':True,'initial_features_assumed_equal':False,
              'initial_control_sets':2,'per_step_trace_counts_exact':True,'initial_shared_state_exact_before_training_and_heads':True},'paired initialization before all training/head fits')
        providers={};infos={};traces={};queries={};encoders=[]
        for placement,role in enumerate(('late','early')):
            for point in (0,512):
                cp=base/role/f'point-{point}/checkpoint.pt';audit=load(Path(str(cp)+'.audit.pt'))
                info=L.replay_parent(audit,load(Path(str(cp)+'.training-raw.pt')),load(Path(str(cp)+'.scaler.pt')),data[0],master,placement,point,scopes)
                check(info['scaler_id']==initial[3],'four points same initial original TRAIN scaler')
                fields=external_binding(load(Path(str(cp)+'.structured-timing.pt')),audit,info,cp,placement,point,data[0][2],scopes)
                external_binding(load(cp.parent/'snapshot-assets/structured-timing-snapshot-audit.pt'),audit,info,cp,placement,point,data[0][2],scopes,True)
                provider=L.snapshot_binding(load(cp.parent/'snapshot-assets/early-mixer-snapshot-audit.pt'),audit,info,cp,placement,point,scopes,initial)
                base_provenance=('RPB-v10' if placement else 'RPB-v7')+'; immutable timing-protocol checkpoint; exact contextual-global32 CUDA serving; ordinary original-Q decoder'
                check(provider['extractor_provenance']==base_provenance+'; '+base_provenance,'literal old provider provenance preserved')
                provider['extractor_provenance']=base_provenance+'; '+base_provenance+';external-structured-hard-timing-cohort='+EXTERNAL+';training-implementation='+FIT
                providers[('native_' if point else 'untrained_')+role]=provider;infos[role,point]=info
                if point:
                    actual=metadata(base/role/'encoder-progress.json');traces[role]=L.progress(actual,info);declared=cohort['encoder_points'][placement]
                    check(declared['model_tag']==TAGS[placement] and declared['placement']==placement and declared['encoder_progress']==actual,'all512 actual encoder trace entries and declared tags')
                    for key,value in metadata(base/role/'trainer-audit.json').items():
                        if key=='snapshot_adapter_source_fingerprint':check(value==scopes['early_adapter'],'literal old snapshot adapter source')
                        elif key=='snapshot_policy':check(value=='new_protocol_bound_CUDA_only;historical_CPU_snapshot_not_called','no CPU model snapshot')
                        elif key in fields:check(value==fields[key],'new static external-binding field')
                        else:check(value==M.text(audit,key),'static delegated trainer association')
                    fixed={}
                    for name,split in [('training',data[0]),('validation',data[1])]:
                        asset=load(cp.parent/(name+'-reconstruction.pt'));replay=L.query_archive(asset,split,info['frozen_scaler']);queries[role,name]=asset
                        L.query_summary(declared[name+'_reconstruction'],replay,asset,name+'-reconstruction.pt')
                        fixed[name]={'standardized_mae':replay['mae'],'standardized_huber':replay['huber'],'valid_examples':replay['valid_examples'],
                                     'total_examples':replay['total_examples'],'valid_target_cells':replay['valid_target_cells'],'requested_observed_target_cells':replay['requested_observed_target_cells']}
                    encoders.append({'tag':TAGS[placement],'role':role,'budget':512,'placement':placement,'architecture_id':L.ARCHITECTURES[placement],
                                     'training_seconds':info['training_seconds'],'attempted':512,'completed':512,'sampled_rows':4096,'context_counts':info['counts'],'fixed_query':fixed})
        check(infos['late',512]['common_settings']==infos['early',512]['common_settings'] and infos['late',512]['counts']==infos['early',512]['counts'],'same settings/context stream, sole mixer placement')
        for left,right in zip(traces['late'],traces['early']):check(left[:3]==right[:3],'paired exact absolute trace target/count prefix')
        for name in ('training','validation'):
            for key in ('standardized_target','target_mask','requested_observed_target_mask','visible_mask','trial_channel_eligible','channel_target_counts','channel_valid','example_valid','source_ids_json'):
                M.exact(queries['late',name][key],queries['early',name][key],'paired original-Q witness '+key)
        task=base/'lag_sign';declared=cohort['tasks'][0]
        check(declared['task']=='lag_sign' and declared['data_master']==master and declared['dataset_codename']=='TEMPO-3' and
              declared['designed_complexity_level']==4 and declared['complexity_scale_max']==5,'single fixed dataset/task metadata')
        analytic={}
        for filename,key,split in zip(('training','validation-intact','validation-deleted'),('training','validation_intact','validation_deleted'),data):
            analytic[key]=analytic_archive(load(task/('analytic-'+filename+'.pt')),split,declared['analytic_solvability'][key])
        controls,status=L.control_rows(task,*data);summary,original=L.audit_readouts(task,data,controls,status,providers,master,'lag_sign')
        check(original==declared['readouts'],'full embedded readout report/status/intervals exactly retained')
        check(metadata(task/'preprocessing-fit-counts.json')=={'ObservationScaler_fits':1,'driver_raw_outer_fits':1,'driver_raw_outer_shared_with_PCA':True,
              'helper_outer_train_fits':original['fit_counts']['outer_train_normalizer_fits'],'planned_helper_outer_train_fits':5,'validation_fits':0,'native_PCA_fits':0},'one raw/PCA map; supported native/mask maps only')
        pipelines+=original['fit_counts']['ridge_fits'];helper_outer+=original['fit_counts']['outer_train_normalizer_fits']
        summary.update({'task':'lag_sign','validation_deletion':view,'PCA_status':status,'analytic_solvability':analytic})
        costs=cohort['costs'];check(set(costs)=={'generation_and_observation_io_seconds','binding_and_checkpoint_io_seconds','CUDA_query_transfer_verification_io_seconds',
              'CUDA_native_transfer_verification_seconds','CPU_baseline_preparation_io_seconds','CPU_head_bootstrap_io_seconds','CPU_analytic_observed_only_and_asset_IO_seconds'},'separate declared mixed wall scopes')
        for key,value in costs.items():M.finite_timer(value,key)
        audited.append({'timing_master':master,'encoders':encoders,'tasks':[summary],'costs':costs})
        print('Audited TEMPO-3 paired master '+str(master),flush=True)
    expected={'protocol':PROTOCOL,'dataset_codename':'TEMPO-3','designed_complexity_level':4,'complexity_scale_max':5,'status':'complete','cohorts':5,'tasks_each':1,
              'encoder_trajectories':10,'encoder_updates_each':512,'attempt_limit':1024,'retained_points':20,'skipped_attempts':0,'sampled_rows':40960,
              'head_pipelines':pipelines,'individual_heads':2*pipelines,'planned_pipelines':105,'planned_heads':210,'unique_quality_native_exports':60,'initial_counterpart_exports':0,
              'full_native_export_calls':60,'query_evaluation_calls':20,'necessary_query_forwards':80,'driver_raw_outer_fits':5,'helper_outer_train_fits':helper_outer,
              'initial_shared_state_exact_before_training_and_heads':True,'separate_initial_controls':2,'analytic_solver_fits':0,'analytic_individual_heads':0,
              'analytic_surface_evaluations':15,'information_rule_id':RULE,'extra_decoder_calibration_updates':0,
              'decoder_update_scope':'joint reconstruction during encoder updates; no extra decoder-only calibration','selection':False,'CPU_encoder_training':False,
              'CPU_encoder_forward':False,'testing_accessed':False,'stress_accessed':False,'promotion':False}
    check(complete==expected and len(all_sources)==960,'complete fixed counts with supported/null head fits retained')
    for name,row in index.items():check(sha(capsule/name)==row['sha256'],'all immutable capture bytes preserved after arithmetic')
    return {'status':'passed','protocol':PROTOCOL,'dataset':{'codename':'TEMPO-3','recipe':'structured-hard-timing-v1','designed_complexity_level':4,'maximum':5,'rule_id':RULE},
            'source':source,'validated_metadata':{'report_sha256':sha(results/'report.json'),'complete_sha256':sha(results/'complete.json')},
            'checks':CHECKS+M.CHECKS,'archive_decodes':ARCHIVES,'elapsed_seconds':time.monotonic()-started,'per_master':audited,
            'counts':{'encoder_runs':10,'retained_points':20,'sampled_rows':40960,'source_groups':960,'head_pipelines':pipelines,'individual_heads':2*pipelines,
                      'planned_pipelines':105,'planned_heads':210,'native_full_export_calls':60,'query_writers':20,'query_forwards':80,'analytic_surfaces':15,
                      'driver_raw_outer_fits':5,'helper_outer_fits':helper_outer,'new_archive_roles':roles},
            'limits':{'saved_arithmetic_only':True,'encoder_or_head_fits':0,'PCA_or_SVD_fits':0,'model_forward':False,'optimizer_execution':False,
                      'ordinary_CUDA_checkpoint_body_decodes':0,'historical_payload_reads':0,'TEST_or_stress_reads':0,'across_encoder_interval':False,
                      'checkpoint_feature_association':'captured source/current CUDA admission/typed snapshot bound; no neural reenactment',
                      'optimizer_and_training':'source and actual CUDA admission plus exact saved trace/target/context counters; not independently replayed',
                      'costs':'synchronized training loop includes CPU trace; other named mixed transfer/verification/I/O walls are not pure GPU timings',
                      'interpretation':'five retained fresh pairs, separate analytic information rule; no causal internal loss claim, selection or promotion'}}


def self_test():
    initialize_modules()
    negatives=0
    check(M.own_argmax([[1.,1.],[1.,2.]])==[0,1],'saved-logit own ties')
    try:load(Path('/checkpoint.pt'))
    except AssertionError:negatives+=1
    else:raise AssertionError('CUDA checkpoint decode rejection fixture')
    data=[0.]*288;mask=[False]*288
    for c,f in ((0,0),(1,1)):
        for h in range(32):
            i=c*96+h*3+f;data[i]=(.7 if c==0 else 1.3)*math.sin(2*math.pi*(h+(.6 if c else 0))/16)+(.9 if c else -.4);mask[i]=True
    margins,counts=analytic_values(data,mask,1)
    check(counts==[30] and margins[0]>0,'distinct legal features retain positive physical sign')
    for h in range(6,32):
        for c in range(2):
            for f in range(3):mask[c*96+h*3+f]=False
    check(analytic_values(data,mask,1)[1]==[4],'four distinct legal centres')
    for c in range(2):
        for f in range(3):mask[c*96+5*3+f]=False
    check(analytic_values(data,mask,1)[1]==[3],'three centres remain unsupported')
    hidden=list(data)
    for i,keep in enumerate(mask):
        if not keep:hidden[i]=float('nan')
    check(analytic_values(data,mask,1)==analytic_values(hidden,mask,1),'hidden values never enter analytic information')
    check(analytic_values([.2]*288,[True]*288,1)==([0.],[30]),'exact zero analytic margin')
    def reject(call):
        nonlocal negatives
        try:call()
        except (AssertionError,ValueError,KeyError):negatives+=1
        else:raise AssertionError('negative artificial branch escaped')
    M.close(1.+3.5e-9,1.,'both fixed absolute/relative tolerance terms')
    reject(lambda:M.close(1.+5e-9,1.,'outside fixed F64 bound'))
    reject(lambda:M.own_argmax([[math.nan,0.]]))
    reject(lambda:M.deletion_view([1.,3.],[True,False],[False,False]))
    reject(lambda:M.validate_pairs([0,0],['pair','pair'],[[True],[True]]))
    reject(lambda:M.validate_pairs([0,1],['pair','pair'],[[True],[False]]))
    check(M.normalize([[5.,7.],[10.,11.]],[True,False],[1.,3.],[2.,2.])==[[2.,2.],[0.,0.]],'saved affine maps and unsupported zeros')
    reject(lambda:M.normalize([[1.]],[True],[0.],[0.]))
    check(M.score([0,1],[0,1],[False,False])['accuracy'] is None,'undefined view score retained')
    reject(lambda:M.exact(M.fake('FloatStorage',[1],[0.]),M.fake('FloatStorage',[1],[-0.]),'signed-zero exact byte witness'))
    q=M.query_masks([True]*576,2)[2];pred=[0.]*len(q);target=[1. if keep else 0. for keep in q]
    reduced=M.reductions(pred,target,q,2)
    check(reduced['mae']==1. and reduced['huber']==.5 and all(reduced['example_valid']),'query cell/channel/example reduction hierarchy')
    corrupt=target.copy();corrupt[next(i for i,keep in enumerate(q) if not keep)]=1.
    reject(lambda:M.reductions(pred,corrupt,q,2))
    check(M.reductions(pred,pred,[False]*len(q),2)['mae'] is None,'unsupported query has no invented mean')
    check(M.bootstrap_groups([[1,2],[0,2]],123,1000)['estimate']==.25,'whole source bootstrap arithmetic')
    reject(lambda:M.bootstrap_groups([[1,2]],123,999))
    # Actual analytic archive branch, including positive margins with invalid support.
    margins,centres=analytic_values(data,mask,1);split=(data,mask,['artificial-source'],[1])
    analytic={'predictions':M.fake('LongStorage',[1],[0]),'valid':M.fake('BoolStorage',[1],[False]),
              'margins':M.fake('DoubleStorage',[1],margins),'supported_time_positions':M.fake('LongStorage',[1],centres),
              'labels_scoring_only':M.fake('LongStorage',[1],[1]),'source_ids_json':M.fake_text(json.dumps(split[2])),
              'observed_only':M.fake('BoolStorage',[],[True]),'fitted_heads':M.fake('LongStorage',[],[0]),'information_rule_id':M.fake_text(RULE)}
    declared={'status':'unsupported_view','method':'analytic_observed_only','information_rule_id':RULE,'fitted_heads':0,'total_examples':1,
              'valid_examples':0,'correct_supported':0,'coverage':0.,'full_population_correctness':0.,'accuracy':None,
              'minimum_supported_time_positions':4,'nonzero_margin_required':True,'observed_only':True}
    analytic_archive(analytic,split,declared)
    bad=dict(analytic,predictions=M.fake('LongStorage',[1],[1]));reject(lambda:analytic_archive(bad,split,declared))
    bad=dict(analytic,margins=M.fake('FloatStorage',[1],margins));reject(lambda:analytic_archive(bad,split,declared))
    bad=dict(analytic,unknown=M.fake('LongStorage',[],[0]));reject(lambda:analytic_archive(bad,split,declared))
    bad=dict(declared,accuracy=1.);reject(lambda:analytic_archive(analytic,split,bad))
    # The source-only fifth-companion schema is exercised without loading an archive.
    with tempfile.TemporaryDirectory() as temp:
        cp=Path(temp)/'checkpoint.pt'
        for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.structured-timing.pt'):
            Path(str(cp)+suffix).write_bytes(('SOURCE fixture '+suffix).encode())
        audit={k:M.fake_text(v) for k,v in {'fit_source_manifest_id':'manifest','initialization_seed':'123','actual_training_seed':'456','protocol_id':FIT}.items()}
        info={'dataset_id':'dataset','schema_id':'schema','scaler_id':'scaler','counts':[0,0,0]};scopes={k:'0'*64 for k in ('core_writer','curve_training','early_adapter','structured_adapter')}
        fields={'artifact_kind':'rpb_structured_hard_timing_binding_v2','structured_timing_protocol_id':PROTOCOL,'information_rule_id':RULE,'dataset_codename':'TEMPO-3',
                'dataset_generator_protocol_id':'structured-hard-timing-v1','designed_complexity_level':'4','designed_complexity_scale_max':'5',
                'external_fit_protocol_id':EXTERNAL,'training_implementation_fit_protocol_id':FIT,'structured_timing_scope':'quality','structured_timing_role':'late',
                'structured_timing_instance_group':TAGS[0],'structured_timing_adapter_source_fingerprint':'0'*64,'parent_core_writer_source_fingerprint':'0'*64,
                'parent_training_producer_source_fingerprint':'0'*64,'parent_snapshot_adapter_source_fingerprint':'0'*64,'training_dataset_id':'dataset',
                'training_schema_id':'schema','preprocessing_id':'scaler','original_training_source_manifest':M.source_manifest(['id']),
                'original_training_source_manifest_id':'manifest','initialization_seed':'123','actual_training_seed':'456','training_policy_id':L.POLICY,
                'architecture_id':L.ARCHITECTURES[0],'feature_units':'unitless,unitless,unitless','snapshot_policy':'new-structured-timing-binding;unchanged-reliability-CUDA-serving;no-old-quality-input'}
        for key,suffix in zip(('content_checkpoint','content_audit','content_scaler','content_training_raw'),('','.audit.pt','.scaler.pt','.training-raw.pt')):
            fields[key]='fnv1a64-runtime-content-v1-'+M.fnv(Path(str(cp)+suffix).read_bytes())
        asset={k:M.fake_text(v) for k,v in fields.items()}
        asset.update({k:M.fake('LongStorage',[],[v]) for k,v in {'channel_mixer_placement_value':0,'attempted_steps':0,'completed_steps':0,'sampled_rows':0,
                    'parameter_count_value':225805,'context_requested_deleted_coordinates':0,'context_actual_deleted_coordinates':0,'context_restored_coordinates':0}.items()})
        external_binding(asset,audit,info,cp,0,0,['id'],scopes)
        for placement,point in ((0,512),(1,0),(1,512)):
            changed=dict(asset)
            for key,val in {'structured_timing_role':'early' if placement else 'late','structured_timing_instance_group':TAGS[placement],
                            'architecture_id':L.ARCHITECTURES[placement]}.items():changed[key]=M.fake_text(val)
            for key,val in {'channel_mixer_placement_value':placement,'attempted_steps':point,'completed_steps':point,'sampled_rows':point*8}.items():
                changed[key]=M.fake('LongStorage',[],[val])
            external_binding(changed,audit,info,cp,placement,point,['id'],scopes)
            snapshot=dict(changed)
            snapshot['artifact_kind']=M.fake_text('rpb_structured_hard_timing_cuda_snapshot_v2')
            snapshot['content_structured_timing']=M.fake_text('fnv1a64-runtime-content-v1-'+M.fnv(Path(str(cp)+'.structured-timing.pt').read_bytes()))
            external_binding(snapshot,audit,info,cp,placement,point,['id'],scopes,True)
            bad=dict(snapshot);bad['content_structured_timing']=M.fake_text('wrong')
            reject(lambda bad=bad,placement=placement,point=point:external_binding(bad,audit,info,cp,placement,point,['id'],scopes,True))
        for key,value in [('external_fit_protocol_id',FIT),('content_checkpoint','wrong'),('information_rule_id','old-rule')]:
            corrupt=dict(asset);corrupt[key]=M.fake_text(value)
            reject(lambda corrupt=corrupt:external_binding(corrupt,audit,info,cp,0,0,['id'],scopes))
        corrupt=dict(asset);corrupt['completed_steps']=M.fake('DoubleStorage',[],[0.])
        reject(lambda:external_binding(corrupt,audit,info,cp,0,0,['id'],scopes))
    for dtype,shape,values in [('FloatStorage',[1],[1.]),('LongStorage',[2],[1,2])]:
        good=M.fake(dtype,shape,values);M.tensor(good,dtype,shape)
        try:M.tensor(good,dtype,[3])
        except AssertionError:negatives+=1
        else:raise AssertionError('wrong tensor geometry accepted')
    for name in ('../escape','/absolute','results/../escape','results\\escape'):
        try:relative_name(name)
        except AssertionError:negatives+=1
        else:raise AssertionError('bad relative role accepted')
    writers=['code/evaluation/protocols/structured_hard_timing_v1/paired_timing_run.cpp',
             'code/evaluation/protocols/structured_hard_timing_v1/structured_hard_timing_adapter.h',
             'code/evaluation/protocols/structured_hard_timing_v1/structured_hard_timing_adapter.cpp',
             'code/evaluation/benchmarks/structured_hard_timing/structured_hard_timing.cpp',
             'code/evaluation/benchmarks/structured_hard_timing/cross_feature_solvability.cpp',
             'code/shared/src/fixed_feature_readouts.cpp','code/shared/src/paired_pooling.cpp',
             'code/encoders/raw_patch_bottleneck_mae/src/early_mixer_adapter.cpp',
             'code/encoders/raw_patch_bottleneck_mae/src/learning_curve_adapter.cpp']
    paths=whole_paths([Path('/embedding')/n for n in writers]);bound=[{'path':n,'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in zip(writers,paths)]
    source=(Path('/embedding')/writers[0]).read_text();adapter=(Path('/embedding')/writers[2]).read_text()
    check('observations' in source and 'analytic-validation' in source and 'information_rule_id' in source and 'content_structured_timing' in adapter,'literal writer schema SOURCE fixtures')
    check(ARCHIVES==0,'no SOURCE fixture archive reads')
    return {'status':'passed','checks':CHECKS+M.CHECKS,'negative_fixtures':negatives,'archive_reads':0,'quality_reads':0,
            'measured_execution_enabled':bool(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION),'writer_source_schema':bound,
            'source_sha256':sha(Path(__file__)),'module_sha256':MODULE_PINS,'scope':'artificial saved arithmetic and frozen writer SOURCE, no models/fits'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--fixture-output');parser.add_argument('--capsule');parser.add_argument('--output');args=parser.parse_args()
    check(Path('/.dockerenv').exists(),'managed container only')
    if args.self_test:
        result=self_test()
        if args.fixture_output:
            with Path(args.fixture_output).open('x',encoding='utf-8') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
        print(json.dumps(result,allow_nan=False));return
    check(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION,'measured execution blocked before quality path/import/hash')
    check(args.capsule and args.output,'explicit completed capsule and new audit output')
    output=Path(args.output);check(output.is_absolute() and output.parent==ROOT/'audit-tools' and output.name.startswith('run-'),'exclusive protocol audit output')
    canonical(output.parent,True);output.mkdir()
    started=time.monotonic()
    try:
        result=run(Path(args.capsule))
        with (output/'validation.json').open('x',encoding='utf-8') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
        print(json.dumps({'status':'passed','checks':result['checks'],'archive_decodes':ARCHIVES,'elapsed_seconds':result['elapsed_seconds'],
                          'validation_sha256':sha(output/'validation.json'),'path':str(output/'validation.json')}))
    except Exception as error:
        with (output/'failure.json').open('x',encoding='utf-8') as out:json.dump({'status':'failed','error':str(error),'checks':CHECKS+(M.CHECKS if M else 0),
            'archive_decodes':ARCHIVES,'elapsed_seconds':time.monotonic()-started,'reader_sha256':sha(Path(__file__))},out,indent=2);out.write('\n')
        raise


if __name__=='__main__':main()

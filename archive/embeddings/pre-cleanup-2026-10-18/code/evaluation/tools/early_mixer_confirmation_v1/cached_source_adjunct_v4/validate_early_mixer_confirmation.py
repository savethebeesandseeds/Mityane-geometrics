#!/usr/bin/env python3
"""Blocked, additive compiler-evidence adjunct for one completed confirmation.

The exact released v3 arithmetic reader is imported without editing its bytes.
Only its compile-scope resolver can use three explicitly bound, absent inherited
compile records from the preserved prequality FXJe90 attempt. All numerical,
archive, source, typed provenance and admission predicates remain unchanged.
An independently reviewed flags-only copy requires separate ROOT authorization.
"""
import argparse
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import stat
import symtable
import time

AUTHORIZE_SAVED_ARITHMETIC = False
ROOT = Path('/embedding')
RUN = ROOT / 'output/runs/rpb-early-mixer-confirmation'
CAPSULE = RUN / 'early-mixer-confirmation-uJ1Ack'
PRIOR = RUN / 'admission/admission-FXJe90'
BASE = RUN / 'audit-tools/independent-early-mixer-confirmation-released-uJ1Ack-v3/validate_early_mixer_confirmation.py'
BLOCKED_ADJUNCT = RUN / 'audit-tools/independent-confirmation-cached-source-adjunct-v4/validate_early_mixer_confirmation.py'
BASE_SHA = '6ba1100b5e983a76ddf83369a893e07241a0193f009c1b6c00b340ab270ce0a0'
EVIDENCE = RUN / 'audit-tools/cached-source-evidence-uJ1Ack-v1/cached-source-evidence.json'
EVIDENCE_SHA = '2084b926ca034e18f8d0ffa09d436d304be4c7735ab00a1558a1cca3d0bee1e8'
INVENTORY_SHA = 'd7f6db6dccad050eec984196d28e99b2cc1091c537edf40049ce2be117454846'
SOURCE = 'fa1cbd0e67b4ce6226d6d3a0c5c5c39a71298764e64158dfdbc50decad7e1f12'
PRIOR_SOURCE = 'efa44759e3488147e1c13b2fe16a4c63644321b7e9978f5c4bdd49f9c4af2f1d'
CARD_SHA = '98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea'
CURRENT_LOG_SHA = '55bf81f6099165fccefc21f9f6ddbe28b16f9497f2c55c7c238835f864225efe'
PRIOR_LOG_SHA = 'e562e198f110a3195fd6d7b0d341855308b4235ebfc46131d6773b94c24364ef'
FAILED_AUDIT_SHA = 'd474f53a0e25622db16e7cda165bf74970cc874bcc8e3f37628edc260a1ac9f3'
TEST = 'code/encoders/raw_patch_bottleneck_mae/tests/early_mixer_confirmation_adapter_test.cpp'
OLD_TEST_SHA = 'b38d05143c86a2030e99cf015a7a3183d710a4fce9c3b088f1ba09455add4fcb'
NEW_TEST_SHA = 'f213e17178142ec57f13b7914e31a43006ff90a703e8c572164e989cea0c4f2a'
SOURCE_PREFIX = 'code/encoders/raw_patch_bottleneck_mae/src/'
OBJECT_ROOT = '/opt/cuwacunu_embedding/build/rpb-paired-pooling/'
ENCODER_OBJECT = OBJECT_ROOT + 'code/encoders/raw_patch_bottleneck_mae/'
BINARY = OBJECT_ROOT + 'embedding_early_mixer_confirmation'
SCOPES = {
    'core_writer': ('RPB_PROVENANCE_INPUTS', SOURCE_PREFIX+'workflow.cpp', ENCODER_OBJECT+'workflow.o', 'RPB_SOURCE_ID',
        '6a8a526087a7cf14fdd9f9f4d0e35218349f766bc24c7a7c58f4f388681bb7b1', 36),
    'curve_training': ('EVALUATION_PROVENANCE_INPUTS', SOURCE_PREFIX+'learning_curve_adapter.cpp', ENCODER_OBJECT+'learning_curve_adapter.o', 'EVALUATION_SOURCE_ID',
        '63eae8c8b907885f14a9848d5f203ad6aef09c281489c34bc0b07cf3c7cfd1de', 115),
    'early_adapter': ('EARLY_MIXER_RELIABILITY_INPUTS', SOURCE_PREFIX+'early_mixer_adapter.cpp', ENCODER_OBJECT+'early_mixer_adapter.o', 'EARLY_MIXER_ADAPTER_SOURCE_ID',
        '4620215f84e2c96c712ec1a1397696503a16a66367232b38cd39da3691f09c23', 160),
}
NEW_COMPILES = {
    'main': ('code/evaluation/src/early_mixer_confirmation_main.cpp', OBJECT_ROOT+'code/evaluation/early_mixer_confirmation_main.o', 'EVALUATION_SOURCE_ID'),
    'confirmation_adapter': (SOURCE_PREFIX+'early_mixer_confirmation_adapter.cpp', ENCODER_OBJECT+'early_mixer_confirmation_adapter.o', 'EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID'),
}
MARKERS = ('Early mixer model CUDA admission passed', 'Early mixer CUDA adapter admission passed',
    'Early mixer confirmation CUDA admission passed', 'Fixed feature readout tests passed',
    'Frozen role guard checks passed', 'Container SDK proof:', SOURCE,
    'EARLY_MIXER_ADAPTER_SOURCE_ID='+SCOPES['early_adapter'][4])
CHECKS = 0
HASHES = 0
BASE_MODULE = None


def check(ok, why):
    global CHECKS
    CHECKS += 1
    if not ok: raise AssertionError(why)


def admit(paths, root=ROOT):
    """Whole explicit metadata/SOURCE matrix, with no content access."""
    seen = set()
    for path in paths:
        path = Path(path)
        check(path.is_absolute() and path.is_relative_to(root), 'bounded absolute SOURCE/metadata path')
        cursor = path
        while cursor != root.parent:
            check(not cursor.is_symlink(), 'no path component symlink')
            cursor = cursor.parent
        check(path.resolve(strict=True) == path, 'canonical direct path')
        item = path.stat()
        check(stat.S_ISREG(item.st_mode) and item.st_nlink == 1, 'regular single-link SOURCE/metadata')
        identity = (item.st_dev, item.st_ino)
        check(identity not in seen, 'unique whole-matrix inode')
        seen.add(identity)


def sha(path):
    global HASHES
    HASHES += 1
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, 'JSON duplicate key rejection')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON '+value)))


def canonical(rows):
    return ''.join(row['sha256']+'  '+row['path']+'\n' for row in rows).encode()


def absolute_source_records(base,rows):
    return [{'path':str(base/'source'/row['path']),'bytes':row['bytes'],'sha256':row['sha256']} for row in rows]


def repair_binding(old,new):
    before=(b'rpb::load_checkpoint(left,oa,',b'rpb::load_checkpoint(right,ob,')
    after=(b'rpb::load_optimizer(left,oa,',b'rpb::load_optimizer(right,ob,')
    check(all(old.count(token)==1 for token in before) and old.replace(before[0],after[0]).replace(before[1],after[1])==new,
        'exact compile-only API substitutions, no arithmetic changes')


def source_names(rows, expected_count=182):
    names = []
    for row in rows:
        check(set(row) == {'path','bytes','sha256'}, 'closed SOURCE record keys')
        name = row['path']; p = PurePosixPath(name)
        check(isinstance(name,str) and p.as_posix()==name and not p.is_absolute() and '..' not in p.parts and
            '\\' not in name and (name in ('Makefile','setup.sh','dependencies.lock') or name.startswith('code/')),
            'closed canonical relative SOURCE name')
        check(type(row['bytes']) is int and row['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',row['sha256']), 'typed SOURCE size/hash')
        names.append(name)
    check(names == sorted(set(names)) and len(names)==expected_count, 'complete sorted SOURCE matrix')
    return names


def compile_rows(log, filename):
    return [line for line in log.splitlines()
        if re.search(r'(?:^|\s)-c\s+'+re.escape(filename)+r'(?:\s|$)',line)]


def exact_compile(log, filename, object_path, macro, expected):
    rows = compile_rows(log,filename)
    check(len(rows)==1, 'one exact compile record for source '+filename)
    line = rows[0]
    outputs = re.findall(r'(?:^|\s)-o\s+(\S+)',line)
    inputs = re.findall(r'(?:^|\s)-c\s+(\S+)',line)
    values = re.findall(r'-D'+re.escape(macro)+r'=\\?"([0-9a-f]{64})\\?"',line)
    check(inputs==[filename] and outputs==[object_path] and values==[expected], 'exact source/object/macro compile binding')
    return line


def link_binding(log, expected_line=None):
    rows = [line for line in log.splitlines() if ' -c ' not in line and
        re.search(r'(?:^|\s)-o\s+'+re.escape(BINARY)+r'(?:\s|$)',line)]
    check(len(rows)==1 and (expected_line is None or rows[0]==expected_line), 'one current successful exact binary link')
    tokens = rows[0].split()
    for _,_,object_path,_,_,_ in SCOPES.values(): check(tokens.count(object_path)==1, 'linked exact inherited object once')
    for _,object_path,_ in NEW_COMPILES.values(): check(tokens.count(object_path)==1, 'linked mandatory new object once')
    check(all(marker in log for marker in MARKERS) and '/embedding/.external/libtorch' not in log and 'not found' not in log,
        'actual current CUDA admission/runtime/internal SDK markers')
    for filename,object_path,macro in NEW_COMPILES.values(): exact_compile(log,filename,object_path,macro,SOURCE)
    return rows[0]


def current_link_evidence(log,record):
    check(set(record)=={'command','output','inherited_objects'} and record['output']==BINARY and
        record['inherited_objects']==[v[2] for v in SCOPES.values()], 'exact current link evidence roles/output')
    return link_binding(log,record['command'])


def subset_binding(role,record,scope,ci,pi):
    scope_name,filename,object_path,macro,expected,count=SCOPES[role]
    members=scope['paths']; source_names([ci[n] for n in members],count)
    check(scope['files']==count and scope['canonical_sha256']==expected and TEST not in members and
        filename in members and all(ci[n]==pi[n] for n in members) and
        hashlib.sha256(canonical([ci[n] for n in members])).hexdigest()==expected,
        'unchanged exact inherited dependency subset')
    check(record['scope_name']==scope_name and record['source_path']==filename and record['object_path']==object_path and
        record['macro']==macro and record['canonical_sha256']==expected and record['files']==count and
        record['source_records']==[ci[n] for n in members], 'evidence role maps actual source subset')


def allowed_inputs():
    return {
        'current_source_manifest':CAPSULE/'source-manifest.json', 'current_source_inputs':CAPSULE/'source-inputs.sha256',
        'current_admission_manifest':CAPSULE/'admission/source-manifest.json', 'current_admission_source_inputs':CAPSULE/'admission/source-inputs.sha256',
        'current_admission_log':CAPSULE/'admission/build-and-tests.log', 'current_admission_passed':CAPSULE/'admission/passed.json',
        'current_launch_plan':CAPSULE/'launch-plan.json', 'current_inventory':CAPSULE/'artifact-integrity.json',
        'prior_source_manifest':PRIOR/'source-manifest.json', 'prior_source_inputs':PRIOR/'source-inputs.sha256',
        'prior_admission_log':PRIOR/'build-and-tests.log', 'prior_admission_plan':PRIOR/'admission-plan.json',
        'canonical_scope_fixture':RUN/'admission/source-scope-fixtures/test-api-repair-v2/canonical-scope.json',
        'unchanged_released_base_reader':BASE,
    }


def verify_record(path,row):
    check(set(row)=={'path','bytes','sha256'} and row['path']==str(path) and type(row['bytes']) is int and
        path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'exact admitted record bytes '+str(path))


def evidence_binding():
    paths = allowed_inputs()
    admit([EVIDENCE,*paths.values()])
    check(sha(EVIDENCE)==EVIDENCE_SHA, 'reviewed exact adjunct source evidence')
    evidence = read_json(EVIDENCE)
    check(evidence['schema']=='early-mixer-confirmation-cached-source-evidence-v1' and evidence['status']=='passed' and
        evidence['protocol']=='early-mixer-confirmation-cached-source-evidence-v1' and
        evidence['quality_protocol']=='early-mixer-confirmation-v1' and evidence['capsule']==str(CAPSULE) and
        evidence['inventory_sha256']==INVENTORY_SHA and evidence['source_fingerprint']==SOURCE and
        evidence['human_card_sha256']==CARD_SHA and evidence['base_released_reader_sha256']==BASE_SHA and
        evidence['prior_source_fingerprint']==PRIOR_SOURCE,
        'closed evidence/capsule/card/base scope')
    check(set(evidence['inputs'])==set(paths), 'exact provenance metadata roles')
    for role,path in paths.items(): verify_record(path,evidence['inputs'][role])
    check(evidence['inputs']['current_inventory']['sha256']==INVENTORY_SHA and
        evidence['inputs']['unchanged_released_base_reader']['sha256']==BASE_SHA and
        evidence['inputs']['current_admission_log']['sha256']==CURRENT_LOG_SHA and
        evidence['inputs']['prior_admission_log']['sha256']==PRIOR_LOG_SHA, 'fixed compiler/inventory/base pins')
    current = read_json(paths['current_source_manifest']); prior = read_json(paths['prior_source_manifest'])
    names = source_names(current['sources']); check(source_names(prior['sources'])==names, 'same closed182 source names')
    source_paths = [base/'source'/name for base in (CAPSULE,PRIOR) for name in names]
    admit(source_paths)
    for base,manifest,expected,role in ((CAPSULE,current,SOURCE,'current'),(PRIOR,prior,PRIOR_SOURCE,'prior')):
        rows=manifest['sources']; raw=canonical(rows)
        check(manifest['protocol']=='early-mixer-confirmation-v1' and manifest['human_card']=='code/evaluation/cards/early_mixer_confirmation_v1.md' and
            manifest['source_fingerprint']==expected==hashlib.sha256(raw).hexdigest(), 'canonical whole admission SOURCE identity')
        check(paths[role+'_source_inputs'].read_bytes()==raw, 'canonical sha manifest actual bytes')
        check(evidence['source_roles'][role]==absolute_source_records(base,rows), 'evidence complete absolute captured SOURCE rows')
        for row in rows:
            path=base/'source'/row['path']; check(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'actual captured SOURCE bytes')
    check(read_json(paths['current_admission_manifest'])['sources']==current['sources'] and
        paths['current_admission_source_inputs'].read_bytes()==canonical(current['sources']), 'successful source/admission equality')
    ci={r['path']:r for r in current['sources']}; pi={r['path']:r for r in prior['sources']}
    check([n for n in names if ci[n]!=pi[n]]==[TEST] and pi[TEST]['sha256']==OLD_TEST_SHA and ci[TEST]['sha256']==NEW_TEST_SHA,
        'only preserved two-token test repair differs')
    old=(PRIOR/'source'/TEST).read_bytes(); new=(CAPSULE/'source'/TEST).read_bytes()
    repair_binding(old,new)
    inventory=read_json(paths['current_inventory']); inventory_rows=inventory.get('files',inventory.get('artifacts',[]))
    ii={r['path']:r for r in inventory_rows}
    for row in current['sources']:
        check(ii['source/'+row['path']]['sha256']==row['sha256'] and ii['source/'+row['path']]['bytes']==row['bytes'], 'inventory-bound current SOURCE')
    passed=read_json(paths['current_admission_passed']); launch=read_json(paths['current_launch_plan']); plan=read_json(paths['prior_admission_plan'])
    check(passed['status']=='passed' and passed['source_fingerprint']==SOURCE and passed['human_card_sha256']==CARD_SHA and
        passed['source_preserved'] is True and passed['quality_generated'] is False and passed['log_sha256']==CURRENT_LOG_SHA and
        passed['compiled_binary']['path']==BINARY and launch['admission_log_sha256']==CURRENT_LOG_SHA and
        launch['admission_passed_sha256']==evidence['inputs']['current_admission_passed']['sha256'], 'successful current source/admission/binary binding')
    check(plan['quality_generated'] is False and plan['historical_payload_roles']==0 and plan['source_fingerprint']==PRIOR_SOURCE and
        plan['human_card_sha256']==CARD_SHA, 'prior compile evidence is prequality')
    fixture=read_json(paths['canonical_scope_fixture'])
    check(fixture['status']=='passed' and fixture['make_invocations']==1 and fixture['builds']==0 and fixture['quality_reads']==0,
        'SOURCE-only one-Make subset evidence')
    current_log=paths['current_admission_log'].read_text(); prior_log=paths['prior_admission_log'].read_text()
    current_link_evidence(current_log,evidence['current_successful_binary_link'])
    check(evidence['current_required_runtime_markers']==list(MARKERS) and
        set(evidence['current_mandatory_compile_records'])==set(NEW_COMPILES), 'exact mandatory current compile/runtime roles')
    for role,(filename,object_path,macro) in NEW_COMPILES.items():
        record=evidence['current_mandatory_compile_records'][role]
        check(record=={'source_path':filename,'object_path':object_path,'macro':macro,'expected_source_id':SOURCE,
            'current_compile_command':exact_compile(current_log,filename,object_path,macro,SOURCE)}, 'current new compile record equality')
    check(evidence['failed_base_audit']=={'checks':18502,'archive_decodes':0,'validation_sha256':FAILED_AUDIT_SHA} and
        evidence['quality_score_metadata_reads']==0 and evidence['archive_decodes']==0 and evidence['model_execution'] is False and
        evidence['head_refits']==0 and evidence['quality_or_audit_reruns']==0 and evidence['input_bytes_preserved'] is True,
        'truthful preserved failure and SOURCE-only evidence limits')
    check(set(evidence['scopes'])==set(SCOPES), 'only exact three inherited fallback roles')
    for role,(scope_name,filename,object_path,macro,expected,count) in SCOPES.items():
        scope=fixture['scopes'][scope_name]
        record=evidence['scopes'][role]
        subset_binding(role,record,scope,ci,pi)
        check(exact_compile(prior_log,filename,object_path,macro,expected)==record['prior_compile_command'], 'prior exact target command')
        check(not compile_rows(current_log,filename), 'fallback limited to absent current inherited compile')
    return evidence,current_log


def resolver(original,current_log,evidence,used):
    def resolve(log,filename,macro,object_name=None):
        candidates=[(role,values) for role,values in SCOPES.items()
            if values[1]==filename and values[3]==macro and object_name in (None,Path(values[2]).name)]
        if not candidates: return original(log,filename,macro,object_name)
        check(log==current_log and len(candidates)==1, 'fallback only exact current log and one declared role')
        role,values=candidates[0]; _,source,object_path,macro,expected,_=values
        if compile_rows(log,source):
            exact_compile(log,source,object_path,macro,expected)
            value=original(log,filename,macro,object_name); check(value==expected, 'present compile remains strict original predicate')
            return value
        check(role not in used and evidence['scopes'][role]['canonical_sha256']==expected and
            evidence['scopes'][role]['object_path']==object_path, 'one verified absent inherited compile per role')
        used.add(role)
        return expected
    return resolve


def import_base():
    admit([BASE]); check(sha(BASE)==BASE_SHA, 'unchanged arithmetic base SOURCE')
    spec=importlib.util.spec_from_file_location('unchanged_confirmation_arithmetic_v3',BASE)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    check(module.ARCHIVES==0 and module.REVIEWED_SCHEMA and module.MEASURED_IMPLEMENTATION, 'definition-only base import with no archives')
    return module


def external_source_preservation(evidence):
    paths=allowed_inputs()
    external={role:path for role,path in paths.items() if not path.is_relative_to(CAPSULE)}
    prior=[Path(row['path']) for row in evidence['source_roles']['prior']]
    admit([EVIDENCE,*external.values(),*prior])
    check(sha(EVIDENCE)==EVIDENCE_SHA,'compiler evidence unchanged after saved arithmetic')
    for role,path in external.items():verify_record(path,evidence['inputs'][role])
    for path,row in zip(prior,evidence['source_roles']['prior']):verify_record(path,row)
    return {'passed':True,'external_metadata_files':len(external)+1,'prior_source_files':len(prior),
        'all_admitted_bytes_exact':True,'archive_or_numerical_rerun':False}


def authorization_proof():
    paths=[Path(__file__),BLOCKED_ADJUNCT];admit(paths)
    released=paths[0].read_bytes();blocked=paths[1].read_bytes()
    true_line=rb'^AUTHORIZE_SAVED_ARITHMETIC = True$'
    false_line=b'AUTHORIZE_SAVED_ARITHMETIC = False'
    check(len(re.findall(true_line,released,re.M))==1 and
        re.sub(true_line,false_line,released,count=1,flags=re.M)==blocked,
        'post-completion adjunct release changes only one anchored authorization flag')
    return {'released_sha256':hashlib.sha256(released).hexdigest(),'blocked_sha256':hashlib.sha256(blocked).hexdigest(),
        'changed_assignment':'AUTHORIZE_SAVED_ARITHMETIC = False -> True','exact_reverse_byte_proof':True,
        'created_after_zero_archive_base_failure':True,'original_prequality_base_triple_unchanged':True}


def run(output):
    global BASE_MODULE
    started=time.monotonic();release=authorization_proof();evidence,current_log=evidence_binding(); module=import_base(); used=set()
    BASE_MODULE = module
    module.compile_scope=resolver(module.compile_scope,current_log,evidence,used)
    result=module.run(CAPSULE,output)
    check(used==set(SCOPES) and result['source']['reader_sha256']==BASE_SHA and
        result['source']['actual_compile_scopes']=={**{k:v[4] for k,v in SCOPES.items()},'confirmation_adapter':SOURCE},
        'all exact three fallbacks and unchanged downstream source checks completed')
    preservation=external_source_preservation(evidence)
    check(sha(Path(__file__))==release['released_sha256'] and sha(BLOCKED_ADJUNCT)==release['blocked_sha256'],
        'released and original blocked adjunct SOURCE unchanged through audit')
    # Add provenance only; numerical arrays and decision predicates are untouched.
    result['source']['base_arithmetic_reader_sha256']=BASE_SHA
    result['source']['reader_sha256']=sha(Path(__file__))
    result['source']['composite_reader']={'adjunct_sha256':result['source']['reader_sha256'],'base_sha256':BASE_SHA,
        'evidence_sha256':EVIDENCE_SHA,'fallback_roles':sorted(used),'compiler_evidence':evidence,
        'only_runtime_override':'compile_scope; missing-only exact inherited targets',
        'preserved_failed_base_audit_sha256':FAILED_AUDIT_SHA,'external_source_preservation':preservation,
        'adjunct_authorization_release':release}
    result['source']['base_checks']=result['checks']; result['adjunct_source_checks']=CHECKS
    result['checks']+=CHECKS; result['elapsed_seconds']=time.monotonic()-started
    result['limits']['cached_inherited_objects']='exact source subsets plus preserved FXJe90 compile/current link/CUDA admission; object-file bytes not independently rehashed'
    result['limits']['compiler_evidence_adjunct']={'adjunct_sha256':result['source']['reader_sha256'],'base_sha256':BASE_SHA,
        'evidence_sha256':EVIDENCE_SHA,'prior_compile_log_sha256':PRIOR_LOG_SHA,'current_compile_log_sha256':CURRENT_LOG_SHA,
        'missing_only_roles':sorted(used),'no_claim_of_fresh_current_inherited_compilation':True,
        'numerical_reader_functions_and_tolerances_unchanged':True,'external_source_preservation':preservation,
        'adjunct_authorization_release':release}
    return result


def self_test():
    global CHECKS,HASHES
    check(Path('/.dockerenv').is_file(), 'managed SOURCE fixtures only')
    module=import_base(); negative=0
    def reject(call):
        nonlocal negative
        try: call()
        except (AssertionError,ValueError,KeyError,FileNotFoundError): negative+=1
        else: raise AssertionError('negative SOURCE fixture accepted')
    evidence={'scopes':{r:{'canonical_sha256':v[4],'object_path':v[2]} for r,v in SCOPES.items()}}
    current='\n'.join('g++ -D'+macro+'=\\"'+SOURCE+'\\" -c '+source+' -o '+obj for source,obj,macro in NEW_COMPILES.values())
    current+='\ng++ '+' '.join(v[2] for v in SCOPES.values())+' '+' '.join(v[1] for v in NEW_COMPILES.values())+' -o '+BINARY
    current+='\n'+'\n'.join(MARKERS)
    link_binding(current); used=set(); resolve=resolver(module.compile_scope,current,evidence,used)
    for role,v in SCOPES.items(): check(resolve(current,v[1],v[3],Path(v[2]).name if role=='early_adapter' else None)==v[4], 'actual absent resolver positive')
    check(used==set(SCOPES), 'all three exact artificial fallbacks')
    v=SCOPES['core_writer']; reject(lambda:resolve(current,v[1],v[3]))
    reject(lambda:resolver(module.compile_scope,current,evidence,set())(current+'changed',v[1],v[3]))
    reject(lambda:resolve(current,'code/unknown.cpp','RPB_SOURCE_ID'))
    for role,v in SCOPES.items():
        line='g++ -D'+v[3]+'=\\"'+v[4]+'\\" -c '+v[1]+' -o '+v[2]
        present=current+'\n'+line; used=set(); call=resolver(module.compile_scope,present,evidence,used)
        check(call(present,v[1],v[3],Path(v[2]).name if role=='early_adapter' else None)==v[4] and not used, 'present original resolver positive')
        for bad in (line.replace(v[4],'0'*64),line.replace(v[2],v[2]+'.other'),line+'\n'+line,
                    line.replace('-D'+v[3],'-DWRONG_MACRO')):
            wrong=current+'\n'+bad; reject(lambda:resolver(module.compile_scope,wrong,evidence,set())(wrong,v[1],v[3]))
        reject(lambda:exact_compile('',v[1],v[2],v[3],v[4]))
    reject(lambda:link_binding(current.replace(SCOPES['core_writer'][2],'')))
    reject(lambda:link_binding(current.replace(BINARY,BINARY+'.other')))
    reject(lambda:link_binding(current.replace(MARKERS[-1],MARKERS[-1]+'wrong').replace(SCOPES['early_adapter'][4], '0'*64)))
    reject(lambda:link_binding(current+'\n'+current.splitlines()[2]))
    for name in ('../escape','/absolute','code/../escape','unexpected-root.sh','code\\alias.cpp'):
        reject(lambda:source_names([{'path':name,'bytes':1,'sha256':'0'*64}],1))
    rows=[{'path':'code/a.cpp','bytes':1,'sha256':'1'*64}]
    source_names(rows,1); check(hashlib.sha256(canonical(rows)).hexdigest()!=hashlib.sha256(canonical([dict(rows[0],sha256='2'*64)])).hexdigest(), 'subset corruption changes canonical identity')
    check(absolute_source_records(CAPSULE,rows)==[{'path':str(CAPSULE/'source/code/a.cpp'),'bytes':1,'sha256':'1'*64}], 'actual absolute source evidence schema')
    old=b'rpb::load_checkpoint(left,oa,x);rpb::load_checkpoint(right,ob,y);';new=old.replace(b'load_checkpoint',b'load_optimizer')
    repair_binding(old,new);reject(lambda:repair_binding(old,new+b'changed'));reject(lambda:repair_binding(old,old))
    link_record={'command':current.splitlines()[2],'output':BINARY,'inherited_objects':[v[2] for v in SCOPES.values()]}
    current_link_evidence(current,link_record);reject(lambda:current_link_evidence(current,dict(link_record,output=BINARY+'.bad')))
    # Exercise actual subset gate with synthetic closed SOURCE, including a valid canonical identity.
    role='core_writer'; saved_scope=SCOPES[role]
    synthetic=sorted([{'path':saved_scope[1],'bytes':1,'sha256':'1'*64}]+[
        {'path':f'code/fixture-{n:03}.cpp','bytes':1,'sha256':'2'*64} for n in range(saved_scope[5]-1)],key=lambda row:row['path'])
    synthetic_id=hashlib.sha256(canonical(synthetic)).hexdigest()
    SCOPES[role]=(*saved_scope[:4],synthetic_id,saved_scope[5])
    try:
        ci={row['path']:row for row in synthetic};pi=dict(ci)
        scope={'paths':list(ci),'files':len(ci),'canonical_sha256':synthetic_id}
        record={'scope_name':saved_scope[0],'source_path':saved_scope[1],'object_path':saved_scope[2],'macro':saved_scope[3],
            'canonical_sha256':synthetic_id,'files':len(ci),'source_records':synthetic}
        subset_binding(role,record,scope,ci,pi)
        bad=dict(pi);bad[saved_scope[1]]=dict(pi[saved_scope[1]],sha256='0'*64)
        reject(lambda:subset_binding(role,record,scope,ci,bad))
        reject(lambda:subset_binding(role,record,dict(scope,paths=list(reversed(scope['paths']))),ci,pi))
        reject(lambda:subset_binding(role,record,dict(scope,canonical_sha256='0'*64),ci,pi))
        reject(lambda:subset_binding(role,dict(record,object_path=saved_scope[2]+'.other'),scope,ci,pi))
    finally:SCOPES[role]=saved_scope
    # Late invalid/alias path proves the real whole admission pass hashes nothing.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='confirmation-source-fixture-') as temp:
        root=Path(temp); one=root/'one';two=root/'two';one.write_text('x');two.write_text('y');start=HASHES
        reject(lambda:admit([one,root/'missing'],root));check(HASHES==start,'late invalid admission zero hashes')
        import os
        alias=root/'alias';os.link(one,alias)
        reject(lambda:admit([two,alias],root));check(HASHES==start,'late hardlink admission zero hashes')
    check(module.ARCHIVES==0, 'SOURCE fixtures zero archive decodes')
    tree=symtable.symtable(Path(__file__).read_text(),str(__file__),'exec');unresolved=[];scopes=0;refs=0
    def walk(scope):
        nonlocal scopes,refs
        scopes+=1
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global():
                refs+=1
                if symbol.get_name() not in globals() and symbol.get_name() not in vars(builtins): unresolved.append(symbol.get_name())
        for child in scope.get_children():walk(child)
    walk(tree);check(not unresolved, 'all adjunct globals resolved')
    return {'status':'passed','source_sha256':sha(Path(__file__)),'base_source_sha256':BASE_SHA,'evidence_sha256':EVIDENCE_SHA,
        'checks':CHECKS,'negative_cases':negative,'global_scopes':scopes,'global_references':refs,'unresolved_globals':unresolved,
        'archive_payload_reads':0,'quality_score_metadata_reads':0,'model_or_head_execution':False,
        'measured_execution_enabled':AUTHORIZE_SAVED_ARITHMETIC}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--source-evidence-check',action='store_true')
    parser.add_argument('--fixture-output');parser.add_argument('--capsule');parser.add_argument('--output');args=parser.parse_args()
    if args.self_test:
        result=self_test()
        if args.fixture_output:
            path=Path(args.fixture_output);check(not path.exists(),'exclusive SOURCE fixture record')
            path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True);return
    if args.source_evidence_check:
        check(Path('/.dockerenv').is_file(),'managed SOURCE-only evidence preflight')
        evidence,_=evidence_binding()
        preservation=external_source_preservation(evidence)
        result={'status':'passed','source_sha256':sha(Path(__file__)),'evidence_sha256':EVIDENCE_SHA,
            'checks':CHECKS,'source_files_checked':364,'archive_payload_reads':0,'quality_score_metadata_reads':0,
            'measured_execution_enabled':AUTHORIZE_SAVED_ARITHMETIC,'scopes':{k:v[4] for k,v in SCOPES.items()},
            'external_source_preservation':preservation}
        if args.fixture_output:
            path=Path(args.fixture_output);check(not path.exists(),'exclusive SOURCE evidence preflight record')
            path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True);return
    check(AUTHORIZE_SAVED_ARITHMETIC, 'blocked adjunct refuses before measured path/source/archive access')
    check(Path('/.dockerenv').is_file() and args.capsule==str(CAPSULE) and args.output,'managed exact completed capsule required')
    output=Path(args.output);check(output.is_absolute() and output.is_relative_to(RUN/'audit-tools') and
        output.resolve(strict=False)==output and not output.exists() and not output.is_symlink() and
        not output.is_relative_to(CAPSULE) and not output.is_relative_to(PRIOR),'exclusive output outside captured evidence')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:result=run(output)
    except Exception as error:
        result={'protocol':'early-mixer-confirmation-v1','status':'failed','checks':CHECKS,'archive_decodes':0,
            'elapsed_seconds':time.monotonic()-started,'reader_sha256':sha(Path(__file__)),
            'base_reader_sha256':BASE_SHA,'evidence_sha256':EVIDENCE_SHA,'error':str(error),'preserved_measured_capsule':str(CAPSULE)}
        # If the unchanged base reached arithmetic, preserve its actual counters.
        if BASE_MODULE is not None:
            result['base_checks']=BASE_MODULE.CHECKS;result['checks']+=BASE_MODULE.CHECKS
            result['archive_decodes']=BASE_MODULE.ARCHIVES
        (output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True);raise
    (output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'checks':result['checks'],'archive_decodes':result['archive_decodes'],
        'elapsed_seconds':result['elapsed_seconds'],'validation_sha256':sha(output/'validation.json')}),flush=True)


if __name__=='__main__':main()

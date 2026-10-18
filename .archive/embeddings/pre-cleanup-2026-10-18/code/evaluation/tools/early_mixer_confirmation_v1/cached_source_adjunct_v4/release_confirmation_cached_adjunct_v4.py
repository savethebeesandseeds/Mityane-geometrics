#!/usr/bin/env python3
"""SOURCE-reviewed, metadata-only one-flag release utility; invoke only when ROOT authorizes.

No reader import, archive decode, score read, model execution or audit is performed.
The immutable post-failure blocked adjunct is copied to one exclusive new leaf.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat

ROOT=Path('/embedding')
RUN=ROOT/'output/runs/rpb-early-mixer-confirmation'
CAPSULE=RUN/'early-mixer-confirmation-uJ1Ack'
ORIGINAL=RUN/'audit-tools/independent-confirmation-cached-source-adjunct-v4'
SOURCE_SHA='72e41a922a63459e0fa57ae56c6ec1bc955a75611709898f6e1582f948b8b588'
FIXTURES_SHA='9c5295e1f0154670de16f346ee9c5827bf1f16c47de6f8d47c144459ac11567c'
SEAL_SHA='1b03cc9f3af884de851e23909a3fb4e07ebb426c698a8cc172b76db60a3f761d'
PREFLIGHT_SHA='25ed014551b01d80ab848003cf7bc467058eb61d0118f83462ce1a16fdf02d52'
EVIDENCE_SHA='2084b926ca034e18f8d0ffa09d436d304be4c7735ab00a1558a1cca3d0bee1e8'
BASE_SHA='6ba1100b5e983a76ddf83369a893e07241a0193f009c1b6c00b340ab270ce0a0'
CAPTURED_FALSE_SHA='b114d0e1db46fd7dd9bba16594eca220b0703269e82f1bd2dea0c399088b4516'
INVENTORY_SHA='d7f6db6dccad050eec984196d28e99b2cc1091c537edf40049ce2be117454846'
SOURCE_ID='fa1cbd0e67b4ce6226d6d3a0c5c5c39a71298764e64158dfdbc50decad7e1f12'


def check(ok,why):
    if not ok:raise AssertionError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit(paths,root=ROOT):
    seen=set()
    for path in paths:
        check(path.is_absolute() and path.is_relative_to(root),'bounded explicit metadata/SOURCE')
        cursor=path
        while cursor!=root.parent:
            check(not cursor.is_symlink(),'no path component symlink');cursor=cursor.parent
        check(path.resolve(strict=True)==path,'direct canonical metadata/SOURCE')
        item=path.stat();check(stat.S_ISREG(item.st_mode) and item.st_nlink==1,'regular single-link metadata/SOURCE')
        identity=(item.st_dev,item.st_ino);check(identity not in seen,'whole explicit inode matrix');seen.add(identity)


def release_bytes(blocked):
    false=rb'^AUTHORIZE_SAVED_ARITHMETIC = False$';true=b'AUTHORIZE_SAVED_ARITHMETIC = True'
    check(len(re.findall(false,blocked,re.M))==1 and not re.findall(rb'^AUTHORIZE_SAVED_ARITHMETIC = True$',blocked,re.M),
        'one exact blocked authorization assignment')
    released=re.sub(false,true,blocked,count=1,flags=re.M)
    check(re.sub(rb'^AUTHORIZE_SAVED_ARITHMETIC = True$',b'AUTHORIZE_SAVED_ARITHMETIC = False',released,count=1,flags=re.M)==blocked,
        'only one anchored assignment and complete reverse-byte proof')
    return released


def completion_binding(inventory,actual):
    rows=inventory.get('files',inventory.get('artifacts',[]))
    matches=[row for row in rows if row['path']=='results/complete.json']
    check(len(matches)==1 and matches[0]['bytes']==actual['bytes'] and matches[0]['sha256']==actual['sha256'],
        'completion metadata exact bytes/SHA associated with pinned completed inventory')


def prepare(output):
    check(Path('/.dockerenv').is_file(),'existing managed container only')
    paths={
        'blocked_adjunct':ORIGINAL/'validate_early_mixer_confirmation.py',
        'artificial_fixtures':ORIGINAL/'source-fixtures-final.json',
        'reader_seal':ORIGINAL/'reader-seal.json',
        'source_evidence_preflight':ORIGINAL/'source-evidence-final.json',
        'compiler_source_evidence':RUN/'audit-tools/cached-source-evidence-uJ1Ack-v1/cached-source-evidence.json',
        'unchanged_released_base':RUN/'audit-tools/independent-early-mixer-confirmation-released-uJ1Ack-v3/validate_early_mixer_confirmation.py',
        'original_captured_false_base':CAPSULE/'independent-reader/source.py',
        'completed_inventory':CAPSULE/'artifact-integrity.json',
        'completion_metadata':CAPSULE/'results/complete.json',
    }
    admit(list(paths.values()))
    before={role:{'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)} for role,path in paths.items()}
    pins={'blocked_adjunct':SOURCE_SHA,'artificial_fixtures':FIXTURES_SHA,'reader_seal':SEAL_SHA,
        'source_evidence_preflight':PREFLIGHT_SHA,'compiler_source_evidence':EVIDENCE_SHA,'unchanged_released_base':BASE_SHA,
        'original_captured_false_base':CAPTURED_FALSE_SHA,'completed_inventory':INVENTORY_SHA}
    check(all(before[role]['sha256']==value for role,value in pins.items()),'all explicit reviewed source/fixture/evidence/inventory pins')
    load=lambda role:json.loads(paths[role].read_text())
    completion_binding(load('completed_inventory'),before['completion_metadata'])
    seal=load('reader_seal');fixtures=load('artificial_fixtures');preflight=load('source_evidence_preflight');complete=load('completion_metadata')
    check(seal['status']=='passed' and seal['source_sha256']==SOURCE_SHA and seal['fixtures_sha256']==FIXTURES_SHA and
        seal['source_evidence_preflight_sha256']==PREFLIGHT_SHA and seal['compiler_evidence_sha256']==EVIDENCE_SHA and
        seal['measured_execution_enabled'] is False,'reviewed blocked adjunct seal')
    check(fixtures['status']=='passed' and fixtures['source_sha256']==SOURCE_SHA and fixtures['archive_payload_reads']==0 and
        fixtures['quality_score_metadata_reads']==0 and fixtures['measured_execution_enabled'] is False and
        preflight['status']=='passed' and preflight['source_sha256']==SOURCE_SHA and preflight['evidence_sha256']==EVIDENCE_SHA and
        preflight['archive_payload_reads']==0 and preflight['quality_score_metadata_reads']==0 and
        preflight['measured_execution_enabled'] is False and preflight['external_source_preservation']['passed'],
        'artificial and actual SOURCE-only preflight passed, still blocked')
    check(complete['protocol']=='early-mixer-confirmation-v1' and complete['status']=='complete' and
        complete['encoder_trajectories']==10 and complete['encoder_updates_each']==512 and complete['retained_points']==20 and
        complete['unique_quality_native_exports']==120 and complete['separate_initial_controls']==2 and
        complete['query_evaluation_calls']==20 and complete['necessary_query_forwards']==80 and
        complete['testing_accessed'] is False and complete['stress_accessed'] is False and complete['promotion'] is False,
        'exact completed immutable confirmation scope')
    oldbase=paths['unchanged_released_base'].read_bytes()
    for flag in (b'REVIEWED_SCHEMA',b'MEASURED_IMPLEMENTATION'):
        check(len(re.findall(rb'^'+flag+rb' = True$',oldbase,re.M))==1,'exact released base flag')
        oldbase=re.sub(rb'^'+flag+rb' = True$',flag+b' = False',oldbase,count=1,flags=re.M)
    check(oldbase==paths['original_captured_false_base'].read_bytes(),'original two-flag base reverse proof preserved')
    released=release_bytes(paths['blocked_adjunct'].read_bytes())
    check(output==RUN/'audit-tools/independent-confirmation-cached-source-adjunct-released-uJ1Ack-v4' and
        output.is_absolute() and output.resolve(strict=False)==output and not output.exists() and not output.is_symlink(),
        'one exact exclusive new release leaf')
    cursor=output.parent
    while cursor!=ROOT.parent:
        check(not cursor.is_symlink(),'release ancestor aliases forbidden');cursor=cursor.parent
    output.mkdir(exist_ok=False)
    destination=output/'validate_early_mixer_confirmation.py'
    with destination.open('xb') as stream:stream.write(released)
    proof={'protocol':'early-mixer-confirmation-cached-source-adjunct-v4','status':'passed',
        'root_authorized_completed_capsule':str(CAPSULE),'completed_inventory_sha256':INVENTORY_SHA,'source_fingerprint':SOURCE_ID,
        'original_blocked_sha256':SOURCE_SHA,'released_sha256':sha(destination),'changed_assignment':'AUTHORIZE_SAVED_ARITHMETIC = False -> True',
        'exact_reverse_byte_proof':True,'unchanged_released_base_sha256':BASE_SHA,'compiler_evidence_sha256':EVIDENCE_SHA,
        'inputs':before,'reader_imports':0,'archive_decodes':0,'score_metadata_reads':0,'model_or_head_or_audit_execution':False}
    admit(list(paths.values()))
    check(all(sha(path)==before[role]['sha256'] for role,path in paths.items()),'all original inputs unchanged after new release')
    with (output/'release-proof.json').open('x',newline='\n') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'passed','reader_path':str(destination),'reader_sha256':proof['released_sha256'],
        'proof_sha256':sha(output/'release-proof.json'),'archives':0,'audit_execution':False}),flush=True)


def self_test():
    negative=0
    def reject(call):
        nonlocal negative
        try:call()
        except (AssertionError,FileNotFoundError):negative+=1
        else:raise AssertionError('negative fixture accepted')
    blocked=b'prefix\nAUTHORIZE_SAVED_ARITHMETIC = False\nsuffix\n'
    check(release_bytes(blocked)==b'prefix\nAUTHORIZE_SAVED_ARITHMETIC = True\nsuffix\n','actual one-flag positive')
    for wrong in (blocked+blocked,blocked.replace(b'False',b'True'),blocked.replace(b' = ',b'='),blocked.replace(b'False',b'False # comment')):
        reject(lambda:release_bytes(wrong))
    actual={'bytes':1,'sha256':'a'*64};inventory={'files':[{'path':'results/complete.json',**actual}]}
    completion_binding(inventory,actual)
    reject(lambda:completion_binding(inventory,dict(actual,sha256='b'*64)))
    reject(lambda:completion_binding(inventory,dict(actual,bytes=2)))
    reject(lambda:completion_binding({'files':[]},actual))
    import tempfile,os
    with tempfile.TemporaryDirectory(prefix='confirmation-release-source-') as temp:
        root=Path(temp);one=root/'one';two=root/'two';one.write_text('x');two.write_text('y')
        admit([one,two],root);reject(lambda:admit([one,root/'missing'],root))
        os.link(one,root/'alias');reject(lambda:admit([two,root/'alias'],root))
    return {'status':'passed','negative_cases':negative,'reader_imports':0,'archive_decodes':0,'quality_score_reads':0,
        'releases_or_audits':0,'source_sha256':sha(Path(__file__))}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()),flush=True)
    else:check(args.output is not None,'ROOT-authorized explicit exclusive release path required');prepare(args.output)

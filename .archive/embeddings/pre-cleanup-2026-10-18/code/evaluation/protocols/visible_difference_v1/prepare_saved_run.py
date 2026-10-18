#!/usr/bin/env python3
"""Closed SOURCE/admission/candidate capture. Loads metadata only, never tensors."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
PROTOCOL = 'visible-difference-v1'
ROOT = 'output/runs/rpb-visible-difference'
CODE = 'code/evaluation/protocols/visible_difference_v1'
CARD = 'code/evaluation/cards/visible_difference_v1.md'
CARD_SHA = '75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc'
MASTERS = [75272,76373,77474,78575,79676]
BINARY = Path('/opt/cuwacunu_embedding/build/rpb-visible-difference/embedding_visible_difference')
PARENT = Path('/embedding/output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS')
PARENT_META = {
    'artifact-integrity.json': (PARENT/'artifact-integrity.json', '9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa'),
    'parent-validation.json': (Path('/embedding/output/runs/rpb-structured-hard-timing/audit-tools/run-xrZMAS-v3-ordering/validation.json'), 'cf79abdca511f22778622d2b08e62b74b592c10618d1090ea2bd53eefd84539c'),
    'parent-summary.json': (Path('/embedding/doc/results/structured_hard_timing_comparison_v2.json'), '1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4'),
}

def require(ok, why):
    if not ok:
        raise ValueError(why)

def canonical(path, directory=False):
    p = Path(path)
    require(p.is_absolute() and p.resolve(strict=True) == p, 'absolute canonical path: '+str(p))
    require(all(not n.is_symlink() for n in (p,*p.parents)), 'no symlink ancestor')
    s = p.stat()
    require(stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'regular nonhardlinked role')
    return p

def admit(paths):
    paths = [canonical(p) for p in paths]
    require(len(set(paths)) == len(paths), 'unique path matrix')
    nodes = [(p.stat().st_dev,p.stat().st_ino) for p in paths]
    require(len(set(nodes)) == len(nodes), 'unique inode matrix before bytes')
    return paths

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda:src.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def records(paths):
    return [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in admit(paths)]

def read_json(path):
    return json.loads(canonical(path).read_text())

def write_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as out:
        json.dump(value,out,indent=2,sort_keys=True,allow_nan=False)
        out.write('\n')

def copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as out,src.open('rb') as inp:
        for block in iter(lambda:inp.read(1048576),b''):
            out.write(block)

def relative(name):
    p = PurePosixPath(name)
    require(isinstance(name,str) and str(p)==name and not p.is_absolute() and '\\' not in name and
            all(v not in ('','.','..') for v in p.parts), 'normalized relative name')
    return name

def source_name(name):
    relative(name)
    require(name in ('Makefile','dependencies.lock','setup.sh') or name.startswith('code/'), 'closed SOURCE namespace')
    return name

def parent_roles():
    out=[]
    for m in MASTERS:
        b=f'results/seed-{m}-lag_sign/'
        out += [b+n for n in ['controlled-training.pt','controlled-validation.pt','controlled-validation-deleted.pt','initial-pair.pt']]
        out += [b+'early/point-0/checkpoint.pt'+s for s in ['', '.audit.pt','.scaler.pt','.training-raw.pt','.structured-timing.pt']]
        out += [b+'lag_sign/readouts/untrained_early/'+v+'-features.pt' for v in ['training','validation-intact','validation-deleted']]
        out += [b+f'lag_sign/readouts/{c}/rep-{r}/{v}-predictions.pt' for c in ['native_late','native_early'] for r in [2701,2802,2903] for v in ['validation-intact','validation-deleted']]
        out += [b+'early/point-512/'+v+'-reconstruction.pt' for v in ['training','validation']]
    require(len(out)==130 and len(set(out))==130,'closed130 retained roles')
    return sorted(out)

def parent_metadata():
    before=records([v[0] for v in PARENT_META.values()])
    for row,(_,pin) in zip(before,PARENT_META.values()):
        require(row['sha256']==pin,'immutable parent metadata')
    values={n:read_json(p) for n,(p,_) in PARENT_META.items()}
    require(values['parent-validation.json']['status']=='passed','passed independent parent audit')
    require(records([v[0] for v in PARENT_META.values()])==before,'parent metadata preserved')
    return values,before

def selected_roles(inventory):
    files=inventory['files'];names=[relative(x['path']) for x in files]
    require(len(names)==len(set(names)) and inventory['status']=='complete','unique complete parent inventory')
    index={x['path']:x for x in files}
    roles=parent_roles()
    require(set(roles)<=set(index),'all130 present in pinned parent inventory')
    rows=[index[n] for n in roles]
    for row in rows:
        require(set(row)=={'path','bytes','sha256'} and type(row['bytes']) is int and row['bytes']>0 and
                len(row['sha256'])==64 and all(c in '0123456789abcdef' for c in row['sha256']),'typed selected parent identity')
    return rows

def progress_text(audit,summary):
    require(len(audit['per_master'])==len(summary['cohorts'])==5,'all five parent metadata records')
    amap={x['timing_master']:x for x in audit['per_master']}
    smap={x['timing_master']:x for x in summary['cohorts']}
    require(set(amap)==set(smap)==set(MASTERS),'fixed masters in parent metadata')
    lines=['master\tattempted\tcompleted\tsampled_rows\trequested\tactual\trestored\ttarget_cells']
    for m in MASTERS:
        es=[x for x in amap[m]['encoders'] if x['role']=='early']
        ps=[x for x in smap[m]['encoder_points'] if x.get('placement')==1]
        require(len(es)==len(ps)==1,'one matched early parent encoder')
        e,p=es[0],ps[0]['encoder_progress']
        require([e['attempted'],e['completed'],e['sampled_rows']]==[p['attempted'],p['completed'],p['sampled_rows']]==[512,512,4096], 'exact parent budget')
        cc=e['context_counts']
        require(len(cc)==3 and all(type(v) is int and v>=0 for v in cc),'parent cumulative erasure triple')
        losses=p['losses']
        require(len(losses)==512 and all(len(l)==5 and l[:2]==[i+1,i+1] and type(l[2]) is int and l[2]>0 for i,l in enumerate(losses)),'full parent target-count trace')
        lines.append('\t'.join(map(str,[m,512,512,4096,*cc]))+'\t'+','.join(str(l[2]) for l in losses))
    return '\n'.join(lines)+'\n'

def capture(repo,target,names):
    require(names==sorted(set(names)) and names,'sorted nonempty SOURCE closure')
    for n in names:source_name(n)
    require({CARD,'Makefile','setup.sh','dependencies.lock','code/scripts/install-libtorch.py',CODE+'/retained-roles.json'}<=set(names),'card/SDK/roles SOURCE closure')
    actual=records([repo/n for n in names])
    rows=[dict(r,path=n) for n,r in zip(names,actual)]
    require(next(r['sha256'] for r in rows if r['path']==CARD)==CARD_SHA,'fixed card')
    body=''.join(r['sha256']+'  '+r['path']+'\n' for r in rows)
    fp=hashlib.sha256(body.encode()).hexdigest()
    for n in names:copy(repo/n,target/'source'/n)
    write_json(target/'source-manifest.json',dict(protocol=PROTOCOL,source_sha256=fp,files=rows))
    (target/'source-inputs.sha256').write_text(body)
    (target/'source-fingerprint.txt').write_text(fp+'\n')
    require(records([repo/n for n in names])==actual,'SOURCE preserved during capture')
    return fp

def verify_sources(repo,target):
    value=read_json(target/'source-manifest.json');rows=value['files'];names=[source_name(r['path']) for r in rows]
    require(names==sorted(set(names)),'unique canonical SOURCE names')
    actual=records([repo/n for n in names]+[target/'source'/n for n in names])
    for i,row in enumerate(rows):
        require(all(x['bytes']==row['bytes'] and x['sha256']==row['sha256'] for x in (actual[i],actual[i+len(names)])),'SOURCE remains exact: '+row['path'])
    body=''.join(r['sha256']+'  '+r['path']+'\n' for r in rows)
    require(hashlib.sha256(body.encode()).hexdigest()==value['source_sha256'] and (target/'source-inputs.sha256').read_text()==body and
            (target/'source-fingerprint.txt').read_text()==value['source_sha256']+'\n','canonical SOURCE identity')
    return value

def target_dir(repo,path,family,prefix=''):
    p=Path(path);require(p.is_absolute() and p.parent==repo/ROOT/family and p.name.startswith(prefix),'exclusive named target')
    canonical(p.parent,True)
    if p.exists():
        canonical(p,True);require(not any(p.iterdir()),'preserve existing outputs')
    else:p.mkdir()
    return p

def freeze(args):
    repo=canonical(args.repo_root,True);target=target_dir(repo,args.target,'admission','admission-')
    require(args.card_sha256==CARD_SHA,'frozen card CLI pin')
    fp=capture(repo,target,sorted(args.sources))
    expanded=subprocess.run(['make','-s','-f',CODE+'/Makefile','print-visible-difference-scope-inputs'],cwd=repo,text=True,capture_output=True,check=True).stdout
    scopes={}
    for line in expanded.splitlines():
        name,path=line.split('|',1);require(name in ['core_writer','curve_training','early_adapter','visible_adapter'],'closed scope');source_name(path)
        scopes.setdefault(name,[]).append(path)
    require(set(scopes)=={'core_writer','curve_training','early_adapter','visible_adapter'},'all compiled scopes')
    index={r['path']:r for r in read_json(target/'source-manifest.json')['files']};out={}
    for name,paths in scopes.items():
        require(paths==sorted(set(paths)) and set(paths)<=set(index),'full canonical source subset')
        rows=[index[n] for n in paths];identity=hashlib.sha256(''.join(r['sha256']+'  '+r['path']+'\n' for r in rows).encode()).hexdigest()
        out[name]=dict(source_sha256=identity,files=rows)
    require(out['visible_adapter']['source_sha256']==fp,'enclosing scope equals wrapper scope')
    write_json(target/'source-scopes.json',out);verify_sources(repo,target)
    print(json.dumps(dict(status='frozen',target=str(target),source_sha256=fp)))

def compile_records(log,binary,scopes):
    require('/embedding/.external/libtorch' not in log and 'not found' not in log,'internal LibTorch runtime')
    old='code/encoders/raw_patch_bottleneck_mae/shared/src/';od=binary.parent/'code/encoders/raw_patch_bottleneck_mae';nd=binary.parent/'code/protocols/visible_difference_v1'
    requirements=[(old+'workflow.cpp',od/'workflow.o','RPB_SOURCE_ID','core_writer'),
                  (old+'learning_curve_adapter.cpp',od/'learning_curve_adapter.o','EVALUATION_SOURCE_ID','curve_training'),
                  (old+'early_mixer_adapter.cpp',od/'early_mixer_adapter.o','EARLY_MIXER_ADAPTER_SOURCE_ID','early_adapter'),
                  ('code/encoders/raw_patch_bottleneck_mae/versions/v13/src/visible_difference_adapter.cpp',nd/'visible_difference_adapter.o','VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID','visible_adapter'),
                  (CODE+'/visible_difference_main.cpp',nd/'visible_difference_main.o','VISIBLE_DIFFERENCE_SOURCE_ID','visible_adapter')]
    lines=[]
    for line in log.splitlines():
        try:lines.append((line,shlex.split(line)))
        except ValueError:pass
    commands={};outputs=[]
    for src,obj,macro,scope in requirements:
        found=[(l,a) for l,a in lines if '-c' in a and '-o' in a and a[a.index('-c')+1]==src and a[a.index('-o')+1]==str(obj)]
        require(len(found)==1,'one current exact object compile: '+src)
        line,argv=found[0];values=[v.split('=',1)[1].strip('"') for v in argv if v.startswith('-D'+macro+'=')]
        require(values==[scopes[scope]['source_sha256']],'exact current object scope macro')
        commands[src]=line;outputs.append(str(obj))
    links=[(l,a) for l,a in lines if '-c' not in a and '-o' in a and a[a.index('-o')+1]==str(binary)]
    require(len(links)==1 and all(links[0][1].count(o)==1 for o in outputs),'all exact compiled objects linked once')
    return dict(commands=commands,link=links[0][0])

def expected_plan():
    return dict(protocol=PROTOCOL,dataset_id='TEMPO-3',recipe_id='structured-hard-timing-v1',task='lag_sign',timing_master_seeds=MASTERS,
      designed_complexity_level=4,complexity_scale_max=5,candidate_tag='RPB-v13',matched_control_tag='RPB-v10.alt-05',other_control_tag='RPB-v7.alt-05',
      parent_inventory_sha256=PARENT_META['artifact-integrity.json'][1],reused_payload_roles=130,encoder_trajectories=5,encoder_updates_each=512,batch_size=8,
      sampled_rows=20480,attempt_limit=1024,retained_points=10,checkpoint_roles_per_point=6,candidate_parameter_count=228877,common_parameter_count=225805,
      difference_parameter_count=3072,train_pairs=128,validation_pairs=64,shape=[3,32,3],native_width=32,temporal_difference_input=1,head_repetitions=[2701,2802,2903],
      planned_pipelines=15,planned_heads=30,helper_outer_train_fits=5,initial_parity_exports=15,unique_quality_native_exports=15,full_native_export_calls=30,
      query_writer_calls=10,necessary_query_forwards=40,baseline_or_control_refits=0,parent_model_forwards=0,generator_calls=0,PCA_fits=0,
      extra_decoder_calibration_updates=0,selection=False,promotion=False,testing_accessed=False,stress_accessed=False,human_card_sha256=CARD_SHA)

def admit_quality(args):
    repo=canonical(args.repo_root,True);target=canonical(args.target,True);require(target.parent==repo/ROOT/'admission','admission family')
    binary=canonical(args.binary);require(binary==BINARY,'named current binary')
    source=verify_sources(repo,target);log_path=canonical(target/'build-and-tests.log');log=log_path.read_text()
    require(all(m in log for m in ['Visible difference retained input loader fixtures passed','Visible difference model CUDA admission passed','Visible difference adapter CUDA admission passed','Fixed feature readout tests passed','Frozen role guard checks passed','Container SDK proof:']),'actual serialized input/CUDA/shared/internal SDK gates')
    scopes=read_json(target/'source-scopes.json');compiled=compile_records(log,binary,scopes)
    sourceid=subprocess.run([str(binary),'--source-id'],text=True,capture_output=True,check=True).stdout.strip()
    require(sourceid==args.compiled_source_id==source['source_sha256'],'actual binary/source identity')
    plan=json.loads(subprocess.run([str(binary),'--plan'],text=True,capture_output=True,check=True).stdout)
    require(set(plan)==set(expected_plan())|{'source_fingerprint'} and all(plan[k]==v for k,v in expected_plan().items()) and plan['source_fingerprint']==sourceid,'exact prospective producer plan')
    verify_sources(repo,target);write_json(target/'recipe-plan.json',plan)
    write_json(target/'passed.json',dict(protocol=PROTOCOL,status='passed',source_sha256=sourceid,card_sha256=CARD_SHA,binary=records([binary])[0],log=records([log_path])[0],compiler_evidence=compiled,quality_generated=False))
    print(json.dumps(dict(status='passed',admission=str(target),source_sha256=sourceid)))

def reader_bundle(directory):
    directory=canonical(directory,True);seal=read_json(directory/'reader-seal.json')
    require(seal['status']=='passed','prospectively reviewed reader SOURCE fixture seal')
    names=[relative(r['path']) for r in seal['files']]
    require(names==sorted(set(names)) and set(names)=={'validate_visible_difference.py','source-fixtures.json','saved_cpu_math.py','archive_codec.py','legacy_implementation.py'},'complete exact independent SOURCE modules')
    actual=records([directory/n for n in names])
    require(all(a['bytes']==e['bytes'] and a['sha256']==e['sha256'] for a,e in zip(actual,seal['files'])),'sealed reader bytes')
    fixtures=read_json(directory/'source-fixtures.json');require(fixtures['status']=='passed' and fixtures.get('archive_reads',0)==0,'artificial reader fixture PASS before quality')
    return names+['reader-seal.json']

def verify_retained(target,manifest):
    names=[r['path'] for r in manifest['files']];require(names==parent_roles(),'exact retained130 role names')
    paths=[PARENT/n for n in names]+[target/'retained-inputs'/n for n in names]
    actual=records(paths)
    for i,row in enumerate(manifest['files']):
        require(all(a['bytes']==row['bytes'] and a['sha256']==row['sha256'] for a in (actual[i],actual[i+len(names)])),'original/captured parent bytes preserved')
    return actual

def freeze_run(args):
    repo=canonical(args.repo_root,True);admission=canonical(args.admission,True);require(admission.parent==repo/ROOT/'admission','approved admission family')
    passed=read_json(admission/'passed.json');require(passed['status']=='passed' and passed['protocol']==PROTOCOL and passed['card_sha256']==CARD_SHA,'current actual CUDA admission')
    source=verify_sources(repo,admission)
    require(records([passed['binary']['path'],passed['log']['path']])==[passed['binary'],passed['log']],'binary/admission preserved')
    reader_names=reader_bundle(args.reader_dir);reader_paths=[Path(args.reader_dir)/n for n in reader_names]
    reader_before=records(reader_paths)
    metadata,metadata_before=parent_metadata();rows=selected_roles(metadata['artifact-integrity.json'])
    require(read_json(repo/CODE/'retained-roles.json')==dict(protocol=PROTOCOL,parent_inventory_sha256=PARENT_META['artifact-integrity.json'][1],files=rows),'frozen reused-role manifest exact inventory association')
    # All 130 original payload paths/sizes/inodes precede the first payload hash.
    paths=admit([PARENT/r['path'] for r in rows])
    require(all(p.stat().st_size==r['bytes'] for p,r in zip(paths,rows)),'all original sizes admitted before payload hashes')
    before=records(paths);require(all(a['sha256']==r['sha256'] for a,r in zip(before,rows)),'all original selected SHA pins')
    target=target_dir(repo,args.target,'','visible-difference-')
    require(capture(repo,target,[r['path'] for r in source['files']])==passed['source_sha256'],'current copied SOURCE')
    admission_paths=sorted((p for p in admission.rglob('*') if p.is_file()),key=lambda p:p.relative_to(admission).as_posix())
    admission_before=records(admission_paths)
    for p in admission_paths:copy(p,target/'admission'/p.relative_to(admission))
    for p in reader_paths:copy(p,target/'independent-reader'/p.name)
    for (name,(p,_)) in PARENT_META.items():copy(p,target/'parent-metadata'/name)
    for p,row in zip(paths,rows):copy(p,target/'retained-inputs'/row['path'])
    manifest=dict(protocol=PROTOCOL,parent_inventory_sha256=PARENT_META['artifact-integrity.json'][1],files=rows)
    write_json(target/'retained-inputs-manifest.json',manifest)
    (target/'retained-inputs.sha256').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in rows))
    (target/'parent-progress.tsv').write_text(progress_text(metadata['parent-validation.json'],metadata['parent-summary.json']))
    verify_retained(target,manifest);verify_sources(repo,target)
    require(records(reader_paths)==reader_before and records(admission_paths)==admission_before and records([v[0] for v in PARENT_META.values()])==metadata_before,'all reader/admission/metadata originals preserved')
    write_json(target/'launch-plan.json',dict(protocol=PROTOCOL,source_sha256=passed['source_sha256'],card_sha256=CARD_SHA,binary=passed['binary'],log=passed['log'],admission=str(admission),admission_sha256=sha(admission/'passed.json'),reader_files=reader_before,parent_metadata=metadata_before,retained_payload_roles=130,parent_progress_sha256=sha(target/'parent-progress.tsv'),quality_generated_before_freeze=False))
    print(json.dumps(dict(status='frozen',capsule=str(target),source_sha256=passed['source_sha256'])))

def finalize(args):
    repo=canonical(args.repo_root,True);target=canonical(args.target,True);require(target.parent==repo/ROOT and target.name.startswith('visible-difference-'),'capsule family')
    complete=read_json(target/'results/complete.json');require(complete['protocol']==PROTOCOL and complete['status']=='complete','completed candidate run')
    fixed=dict(cohorts=5,encoder_trajectories=5,encoder_updates_each=512,skipped_attempts=0,sampled_rows=20480,retained_points=10,checkpoint_roles_per_point=6,retained_payload_roles=130,planned_pipelines=15,planned_heads=30,quality_native_exports=15,initial_parity_exports=15,full_native_export_calls=30,query_writer_calls=10,necessary_query_forwards=40,baseline_or_control_refits=0,parent_model_forwards=0,generator_calls=0,PCA_fits=0,extra_decoder_calibration_updates=0,CPU_encoder_forward=False,CPU_encoder_training=False,selection=False,promotion=False,testing_accessed=False,stress_accessed=False)
    require(all(complete[k]==v for k,v in fixed.items()) and complete['head_pipelines']<=15 and complete['individual_heads']==2*complete['head_pipelines'] and complete['helper_outer_train_fits']<=5,'full candidate budget and only declared fits')
    launch=read_json(target/'launch-plan.json');verify_sources(repo,target)
    require(records([launch['binary']['path'],launch['log']['path']])==[launch['binary'],launch['log']] and records([r['path'] for r in launch['reader_files']])==launch['reader_files'],'original binary/log/reader preserved')
    reader_bundle(target/'independent-reader')
    require(records([v[0] for v in PARENT_META.values()])==launch['parent_metadata'],'parent metadata originals preserved')
    verify_retained(target,read_json(target/'retained-inputs-manifest.json'))
    for name,(_,pin) in PARENT_META.items():require(sha(target/'parent-metadata'/name)==pin,'captured parent metadata exact')
    require(sha(target/'parent-progress.tsv')==launch['parent_progress_sha256'],'parent progress preserved')
    write_json(target/'source-preserved-after.json',dict(status='passed',source_sha256=launch['source_sha256'],parent_inputs_preserved=True,reader_preserved=True,binary_preserved=True))
    # Normalize first, then string-sort; producer and independent reader agree.
    names=sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file())
    actual=records([target/n for n in names]);rows=[dict(r,path=n) for n,r in zip(names,actual)]
    write_json(target/'artifact-integrity.json',dict(protocol=PROTOCOL,status='complete',files=rows,files_count=len(rows),total_bytes=sum(r['bytes'] for r in rows),excludes=['artifact-integrity.json']))
    print(json.dumps(dict(status='complete',inventory_sha256=sha(target/'artifact-integrity.json'),files=len(rows))))

def manifest(args):
    values,_=parent_metadata();rows=selected_roles(values['artifact-integrity.json'])
    write_json(args.output,dict(protocol=PROTOCOL,parent_inventory_sha256=PARENT_META['artifact-integrity.json'][1],files=rows))
    print(json.dumps(dict(status='passed',payload_reads=0,roles=len(rows),manifest_sha256=sha(args.output))))

def self_test(args):
    checks=0;negatives=0
    def ok(v,why):
        nonlocal checks;checks+=1;require(v,why)
    def reject(fn):
        nonlocal negatives
        try:fn()
        except (ValueError,KeyError):negatives+=1;return
        raise AssertionError('expected artificial rejection')
    rows=[dict(path=n,bytes=7,sha256='a'*64) for n in parent_roles()]
    inv=dict(status='complete',files=rows)
    ok(selected_roles(inv)==rows,'selected closed130')
    reject(lambda:selected_roles(dict(status='complete',files=rows[:-1])))
    reject(lambda:selected_roles(dict(status='complete',files=rows+[rows[0]])))
    for n in ['../code/a','/code/a','code//a','code/a/../b','code\\a','other.txt']:reject(lambda n=n:source_name(n))
    a=[];s=[]
    for m in MASTERS:
        a.append(dict(timing_master=m,encoders=[dict(role='early',attempted=512,completed=512,sampled_rows=4096,context_counts=[23,21,2])]))
        s.append(dict(timing_master=m,encoder_points=[dict(placement=1,encoder_progress=dict(attempted=512,completed=512,sampled_rows=4096,losses=[[i+1,i+1,4,.1,.2] for i in range(512)]))]))
    text=progress_text(dict(per_master=a),dict(cohorts=s));ok(len(text.splitlines())==6 and text.splitlines()[1].split('\t')[7].count(',')==511,'full metadata-derived target trace')
    s[0]['encoder_points'][0]['encoder_progress']['losses'].pop();reject(lambda:progress_text(dict(per_master=a),dict(cohorts=s)))
    scopes={n:dict(source_sha256='a'*64) for n in ['core_writer','curve_training','early_adapter','visible_adapter']}
    old='code/encoders/raw_patch_bottleneck_mae/shared/src/';od=BINARY.parent/'code/encoders/raw_patch_bottleneck_mae';nd=BINARY.parent/'code/protocols/visible_difference_v1'
    specs=[(old+'workflow.cpp',od/'workflow.o','RPB_SOURCE_ID'),(old+'learning_curve_adapter.cpp',od/'learning_curve_adapter.o','EVALUATION_SOURCE_ID'),(old+'early_mixer_adapter.cpp',od/'early_mixer_adapter.o','EARLY_MIXER_ADAPTER_SOURCE_ID'),('code/encoders/raw_patch_bottleneck_mae/versions/v13/src/visible_difference_adapter.cpp',nd/'visible_difference_adapter.o','VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID'),(CODE+'/visible_difference_main.cpp',nd/'visible_difference_main.o','VISIBLE_DIFFERENCE_SOURCE_ID')]
    lines=['g++ -D'+macro+'='+('a'*64)+' -c '+src+' -o '+str(obj) for src,obj,macro in specs]
    link='g++ '+' '.join(str(o) for _,o,_ in specs)+' -o '+str(BINARY)
    log='\n'.join([*lines,link]);ok(len(compile_records(log,BINARY,scopes)['commands'])==5,'exact object/macro/link positives')
    reject(lambda:compile_records('\n'.join(lines+[link.replace(str(specs[0][1]),'other.o')]),BINARY,scopes))
    reject(lambda:compile_records(log+'\n'+lines[0],BINARY,scopes))
    reject(lambda:compile_records(log.replace('-DRPB_SOURCE_ID='+'a'*64,'-DRPB_SOURCE_ID='+'b'*64),BINARY,scopes))
    with tempfile.TemporaryDirectory(prefix='visible-difference-helper-source-') as d:
        root=Path(d);p=root/'one';p.write_text('x');ok(len(records([p]))==1,'regular source synthetic path')
        reject(lambda:records([p,p]));q=root/'redirect';q.symlink_to(p);reject(lambda:records([q]))
        names=sorted(['source-fingerprint.txt','source/code/a','a-z.txt','a/z.txt']);ok(names[0]=='a-z.txt' and names[-1]=='source/code/a','relative string inventory order')
    value=dict(status='passed',checks=checks,negative_cases=negatives,archive_reads=0,models_or_fits=0,quality_generated=False,source_sha256=sha(Path(__file__).resolve()))
    if args.output:write_json(args.output,value)
    print(json.dumps(value))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    m=sub.add_parser('manifest');m.add_argument('--output',required=True);m.set_defaults(fn=manifest)
    f=sub.add_parser('freeze-admission');f.add_argument('--repo-root',required=True);f.add_argument('--target',required=True);f.add_argument('--sources',nargs='+',required=True);f.add_argument('--card-sha256',required=True);f.set_defaults(fn=freeze)
    a=sub.add_parser('admit');a.add_argument('--repo-root',required=True);a.add_argument('--target',required=True);a.add_argument('--binary',required=True);a.add_argument('--compiled-source-id',required=True);a.set_defaults(fn=admit_quality)
    f=sub.add_parser('freeze-run');f.add_argument('--repo-root',required=True);f.add_argument('--target',required=True);f.add_argument('--admission',required=True);f.add_argument('--reader-dir',required=True);f.set_defaults(fn=freeze_run)
    f=sub.add_parser('finalize');f.add_argument('--repo-root',required=True);f.add_argument('--target',required=True);f.set_defaults(fn=finalize)
    t=sub.add_parser('self-test');t.add_argument('--output');t.set_defaults(fn=self_test)
    args=p.parse_args();args.fn(args)

if __name__=='__main__':main()

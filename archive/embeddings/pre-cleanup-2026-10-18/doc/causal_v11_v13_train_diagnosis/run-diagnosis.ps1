# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Complete owned physical phased helper, adapted for V13 arithmetic and cached V11 TRAIN context.
# No phase creates a container; only the existing managed cuwacunu is admitted.
param(
 [Parameter(Mandatory=$true)][ValidateSet('Stage','Schema','Run','Mirror')][string]$Phase,
 [ValidateRange(1,1)][int]$Attempt=1,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PreflightSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$CardSha256,
 [string]$StageReceiptSha256='',
 [string]$SchemaReceiptSha256='',
 [switch]$RootAdmittedStage,
 [switch]$RootAdmittedSchema,
 [switch]$RootAdmittedRun
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$label='attempt-{0:d3}'-f$Attempt
$containerRoot='/opt/cuwacunu/research/causal-v11-v13-train-diagnosis'
$stageHostRoot=Join-Path $researchRoot ('output/engineering/causal-v11-v13-train-diagnosis/'+$label)
$cardRelative='doc/CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'
$preflightRelative='doc/causal_v11_v13_train_diagnosis/preflight-review.json'
$family='code/research/causal_v11_v13_train_diagnosis/'
$names=@('archive_reader.py','diagnose.py','schema.py','README.md','provenance.json')
function Require([bool]$ok,[string]$why){if(-not$ok){throw $why}}
function Host-Path([string]$relative,[string]$repo='cuwacunu_embedding'){
 Require ($repo-in@('cuwacunu_embedding','cuwacunu_dev')-and-not[IO.Path]::IsPathRooted($relative)-and$relative-notmatch':'-and@($relative.Replace([char]92,[char]47).Split('/')|Where-Object {$_-in@('','..','.')} ).Count-eq0) 'Bounded existing-repository path required'
 $base=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repo));$path=[IO.Path]::GetFullPath((Join-Path $base $relative))
 Require ($path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Host path escaped repository'
 $walk=$path
 while($walk-ne$base){if(Test-Path -LiteralPath $walk){Require (((Get-Item -LiteralPath $walk).Attributes-band[IO.FileAttributes]::ReparsePoint)-eq0) 'No linked host metadata ancestor'};$walk=[IO.Path]::GetDirectoryName($walk)}
 return $path
}
function Pin([string]$path,[string]$name){$file=Get-Item -LiteralPath $path;Require (-not$file.PSIsContainer-and-not($file.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'Unlinked regular file required';[ordered]@{path=$name;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Verify([string]$path,$pin){$actual=Pin $path $path;Require ($actual.bytes-eq$pin.bytes-and$actual.sha256-eq$pin.sha256) ('Pinned bytes changed: '+$path)}
function Write-NewJSON([string]$path,$value){$stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write);try{$data=[Text.UTF8Encoding]::new($false).GetBytes(($value|ConvertTo-Json -Depth 30).Replace([string][char]13,'')+"`n");$stream.Write($data,0,$data.Length)}finally{$stream.Dispose()}}
function Existing-Container{
 $raw=@(& docker.exe inspect cuwacunu);Require ($LASTEXITCODE-eq0) 'Existing managed container inspection'
 $observed=@(($raw-join[char]10)|ConvertFrom-Json);Require ($observed.Count-eq1) 'One named managed container'
 $c=$observed[0]
 Require ($c.Id-eq'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-and$c.Image-eq'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-and$c.State.Running) 'Exact immutable existing running container/image'
 Require ($c.Mounts.Count-eq1-and$c.Mounts[0].Type-eq'bind'-and$c.Mounts[0].Source-eq(Join-Path $workspaceRoot 'cuwacunu_dev')-and$c.Mounts[0].Destination-eq'/cuwacunu'-and$c.Mounts[0].RW) 'Sole documented DEV bind; no research-host mount added'
 $g=@($c.HostConfig.DeviceRequests);Require ($g.Count-eq1-and$g[0].Count-eq-1-and@($g[0].Capabilities|ForEach-Object{$_}|ForEach-Object{$_}|Where-Object{$_-eq'gpu'}).Count-gt0) 'Existing GPU-all access'
 return [ordered]@{id=$c.Id;image=$c.Image;running=[bool]$c.State.Running;mounts=$c.Mounts;devices=$g}
}
function Mirror-Snapshot{
 $inventoryCode=@'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); assert r.is_dir(), 'Existing evidence root required'
out=[]
for p in sorted(r.rglob('*')):
    assert not p.is_symlink(), 'No linked evidence: '+str(p)
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        out.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(out))
'@
 $before=@(& docker.exe exec cuwacunu python3 -B -c $inventoryCode $containerRoot);Require ($LASTEXITCODE-eq0) 'Read-only container evidence inventory'
 $inventory=@(($before-join"`n")|ConvertFrom-Json)
 $index=1;while(Test-Path -LiteralPath (Join-Path $stageHostRoot ('mirror-{0:d3}'-f$index))){$index++;Require ($index-le999) 'Preserve every previous mirror'}
 $mirror=Join-Path $stageHostRoot ('mirror-{0:d3}'-f$index)
 & docker.exe cp ('cuwacunu:'+$containerRoot) $mirror;Require ($LASTEXITCODE-eq0) 'Preserve partial mirror after copy failure'
 foreach($pin in $inventory){
  Require (-not[IO.Path]::IsPathRooted($pin.path)-and$pin.path-notmatch'[\\:]'-and@($pin.path.Split('/')|Where-Object {$_-in@('','..','.')} ).Count-eq0) 'Bounded mirrored evidence path'
  $path=[IO.Path]::GetFullPath((Join-Path $mirror $pin.path));Require ($path.StartsWith($mirror+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Mirror path escaped';Verify $path $pin
 }
 Require (@(Get-ChildItem -LiteralPath $mirror -Recurse -Force -File).Count-eq$inventory.Count) 'No unrecorded or missing mirrored files'
 $hostManifestPath=Join-Path $stageHostRoot 'manifest.json'
 Verify (Join-Path $mirror 'manifest.json') (Pin $hostManifestPath 'manifest.json')
 $transport=Get-Content -LiteralPath $hostManifestPath -Raw|ConvertFrom-Json
 Verify (Join-Path $mirror 'source.zip') $transport.archive
 Verify (Join-Path $mirror 'phase-wrapper.py') $transport.wrapper
 Verify (Join-Path $mirror 'source/CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json') $transport.card
 Verify (Join-Path $mirror 'source/preflight-review.json') $transport.preflight
 Require ($transport.card.sha256-eq$CardSha256-and$transport.preflight.sha256-eq$PreflightSha256) 'Mirror binds externally reviewed card/preflight'
 Verify $PSCommandPath $transport.helper
 if($Phase-ne'Stage'){Require ((Pin (Join-Path $mirror 'source-stage.json') 'source-stage.json').sha256-eq$StageReceiptSha256) 'Mirror binds externally reviewed source-stage receipt'}
 $mirroredStage=Get-Content -LiteralPath (Join-Path $mirror 'source-stage.json') -Raw|ConvertFrom-Json
 if($mirroredStage.status-eq'passed'){Verify $hostManifestPath $mirroredStage.manifest;Require ($mirroredStage.card.sha256-eq$CardSha256-and$mirroredStage.source_capsule_files-eq$transport.source_capsule_files-and$mirroredStage.input_pins_verified-eq$transport.input_pins) 'Passed stage binds actual dynamic manifest/card counts'}
 $after=@(& docker.exe exec cuwacunu python3 -B -c $inventoryCode $containerRoot);Require ($LASTEXITCODE-eq0-and($before-join"`n")-ceq($after-join"`n")) 'Container bytes changed during mirror; preserve both copies'
 $receipt=[ordered]@{kind='cpb_v11_v13_TRAIN_diagnosis_exact_mirror_v1';status='passed';phase=$Phase;verified_utc=[DateTime]::UtcNow.ToString('o');container_root=$containerRoot;host_root=$mirror;files=$inventory;all_sizes_sha256_exact=$true;container_before_after_exact=$true;source_capsule_files=$transport.source_capsule_files;input_pin_occurrences=$transport.input_pins;model_calls=0;head_fits=0;training_updates=0;deletions=0;helper=(Pin $PSCommandPath 'doc/causal_v11_v13_train_diagnosis/run-diagnosis.ps1')}
 $receiptPath=Join-Path $stageHostRoot ('mirror-{0:d3}.json'-f$index);Write-NewJSON $receiptPath $receipt
 $executionPath=Join-Path $mirror 'execution-stage.json';$diagnosisPath=Join-Path $mirror 'run-001/diagnosis.json'
 if((Test-Path -LiteralPath $executionPath)-and(Test-Path -LiteralPath $diagnosisPath)){
  $execution=Get-Content -LiteralPath $executionPath -Raw|ConvertFrom-Json
  if($execution.status-eq'passed'){
   $report=Get-Content -LiteralPath $diagnosisPath -Raw|ConvertFrom-Json
   Require ($report.kind-eq'cpb_v11_v13_TRAIN_saved_information_diagnosis_v1'-and$report.status-eq'passed_readonly_saved_TRAIN_diagnosis'-and$report.card.sha256-eq$CardSha256) 'Canonical result is exact completed V13-only diagnosis'
   $canonical=Join-Path $researchRoot 'doc/causal_v11_v13_train_diagnosis/diagnosis.json'
   if(Test-Path -LiteralPath $canonical){Verify $canonical (Pin $diagnosisPath 'diagnosis.json')}else{[IO.File]::Copy($diagnosisPath,$canonical,$false)}
   $executionCanonical=Join-Path $researchRoot 'doc/causal_v11_v13_train_diagnosis/execution.json'
   if(Test-Path -LiteralPath $executionCanonical){Verify $executionCanonical (Pin $executionPath 'execution.json')}else{[IO.File]::Copy($executionPath,$executionCanonical,$false)}
   $canonicalMirror=Join-Path $researchRoot 'doc/causal_v11_v13_train_diagnosis/mirror.json'
   if(-not(Test-Path -LiteralPath $canonicalMirror)){
    $receipt.canonical_diagnosis=Pin $canonical 'doc/causal_v11_v13_train_diagnosis/diagnosis.json'
    $receipt.canonical_execution=Pin $executionCanonical 'doc/causal_v11_v13_train_diagnosis/execution.json'
    $receipt.raw_mirror_receipt=Pin $receiptPath $receiptPath
    Write-NewJSON $canonicalMirror $receipt
   }
  }
 }
 return [ordered]@{path=$mirror;receipt=$receiptPath;files=$inventory.Count}
}

$identity=Existing-Container
if($Phase-eq'Mirror'){
 Require (Test-Path -LiteralPath $stageHostRoot -PathType Container) 'Existing host attempt required for additive mirror'
 Require ($StageReceiptSha256-match'^[0-9a-f]{64}$') 'Manual mirror requires reviewed source-stage SHA'
 Mirror-Snapshot|ConvertTo-Json -Depth 5;return
}
if($Phase-eq'Stage'){Require ([bool]$RootAdmittedStage) 'Separate AST-only Stage admission after actual card/preflight review required'}
if($Phase-eq'Schema'){Require ($RootAdmittedSchema-and$StageReceiptSha256-match'^[0-9a-f]{64}$') 'Explicit metadata-only schema admission and reviewed stage hash required'}
if($Phase-eq'Run'){Require ($RootAdmittedRun-and$StageReceiptSha256-match'^[0-9a-f]{64}$'-and$SchemaReceiptSha256-match'^[0-9a-f]{64}$') 'Explicit numeric-run admission and reviewed Stage/Schema hashes required'}
$cardPath=Host-Path $cardRelative;$preflightPath=Host-Path $preflightRelative
Require ((Pin $cardPath $cardRelative).sha256-eq$CardSha256) 'Externally reviewed ADMITTED V11/V13 card hash'
Require ((Pin $preflightPath $preflightRelative).sha256-eq$PreflightSha256) 'Explicit independently reviewed preflight hash'
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json;$preflight=Get-Content -LiteralPath $preflightPath -Raw|ConvertFrom-Json
Require ($card.kind-eq'cpb_v11_v13_saved_train_diagnosis_card_v1'-and$card.stage-eq'prospective_before_diagnostic_arithmetic'-and$card.execution_admitted-is[bool]-and$card.execution_admitted-and$card.allowed_inputs.Count-eq25-and$card.hash_only_guards.Count-eq35-and$card.context_inputs.Count-eq14) 'Exact separately ADMITTED25/35/14 card; phase admission remains separate'
Require ($preflight.status-eq'passed'-and$preflight.card_sha256-eq$CardSha256) 'Independent static preflight of this exact card'
Require ($card.metadata_copies_required_before_execution.Count-eq($card.context_inputs.Count+2)-and$card.pre_execution_preservation.container_path-eq($containerRoot+'/source/preservation/pre-execution-preservation.json')) 'Explicit cache1/context14/fresh preservation1 copies'
$sourceNames=@(Get-ChildItem -LiteralPath (Host-Path $family.TrimEnd('/')) -File|ForEach-Object Name|Sort-Object)
Require (($sourceNames-join'|')-ceq(($names|Sort-Object)-join'|')) 'Exactly five physical owned source files'
foreach($relative in (@($names|ForEach-Object {$family+$_})+@($cardRelative))){
 $bound=@($preflight.files|Where-Object {$_.path -eq $relative});Require ($bound.Count-eq1) ('Preflight exact source/card pin required: '+$relative);Verify (Host-Path $relative) $bound[0]
}
foreach($pin in $preflight.files){Verify (Host-Path $pin.path) $pin}
$inputPins=@($card.allowed_inputs)+@($card.hash_only_guards)+@($card.context_inputs)+@($card.preservation_pins)+@($card.pre_execution_preservation)
Require ($inputPins.Count-eq$card.logical_pin_occurrences) 'Dynamic logical role occurrences; exact physical aliases allowed'
foreach($pin in $inputPins){Verify (Host-Path $pin.host_path $pin.host_repository) $pin}

# This owned stdlib recorder is physically staged and hashed separately from the dynamic capsule.
# Stage hashes every role and AST-parses three sources, without owned project imports or values.
$wrapper=@'
import argparse,ast,datetime,hashlib,json,os,pathlib,subprocess,sys,time,traceback,zipfile
def pin(p):
    p=pathlib.Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def verify(p,x):
    p=pathlib.Path(p)
    assert p.is_absolute() and p.is_file() and not p.is_symlink() and p.resolve()==p,str(p)
    a=pin(p);assert a['bytes']==x['bytes'] and a['sha256']==x['sha256'],str(p)
def write(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
p=argparse.ArgumentParser();p.add_argument('phase',choices=('Stage','Schema','Run'));p.add_argument('root',type=pathlib.Path);p.add_argument('--stage-sha',default='');p.add_argument('--schema-sha',default='')
for name in ('manifest-sha','card-sha','preflight-sha','zip-sha','wrapper-sha'):p.add_argument('--'+name,required=True)
a=p.parse_args()
r=a.root;s=r/'source';phase=a.phase.lower();stage_name={'Stage':'source-stage.json','Schema':'schema-stage.json','Run':'execution-stage.json'}[a.phase]
assert pathlib.Path('/.dockerenv').is_file() and r.is_absolute()
assert r==pathlib.Path('/opt/cuwacunu/research/causal-v11-v13-train-diagnosis')
assert not (r/stage_name).exists(),'Preserve prior phase receipt; no replay'
started=time.monotonic();receipt={'kind':'cpb_v11_v13_TRAIN_diagnosis_'+phase+'_v1','phase':a.phase,'status':'failed','model_calls':0,'head_fits':0,'training_updates':0}
exit_code=1
try:
    for path,expected in ((r/'manifest.json',a.manifest_sha),(s/'CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json',a.card_sha),
                          (s/'preflight-review.json',a.preflight_sha),(r/'source.zip',a.zip_sha),(pathlib.Path(__file__),a.wrapper_sha)):
        assert path.is_file() and not path.is_symlink() and pin(path)['sha256']==expected,'External frozen transport hash: '+str(path)
    m=json.loads((r/'manifest.json').read_text());c=json.loads((s/'CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json').read_text())
    assert m['kind']=='cpb_v11_v13_TRAIN_diagnosis_execution_capsule_v1'
    assert m['container_root']==str(r) and len(m['files'])==m['source_capsule_files']
    verify(pathlib.Path(__file__),m['wrapper']);verify(r/'source.zip',m['archive'])
    assert m['card']['sha256']==a.card_sha and m['preflight']['sha256']==a.preflight_sha
    assert m['wrapper']['sha256']==a.wrapper_sha and m['archive']['sha256']==a.zip_sha
    def verify_all():
        actual=[]
        for q in s.rglob('*'):
            assert not q.is_symlink(),'No linked source capsule member'
            if q.is_file():actual.append(q.relative_to(s).as_posix())
        assert len(actual)==len(m['files'])==m['source_capsule_files'] and set(actual)=={x['path'] for x in m['files']},'Exact source membership'
        for x in m['files']:
            q=pathlib.PurePosixPath(x['path']);assert not q.is_absolute() and '..' not in q.parts and chr(92) not in x['path']
            verify(s/x['path'],x)
        pins=c['allowed_inputs']+c['hash_only_guards']+c['context_inputs']+c.get('preservation_pins',[])+[c['pre_execution_preservation']]
        assert c['execution_admitted'] is True and len(c['allowed_inputs'])==25 and len(c['hash_only_guards'])==35 and len(c['context_inputs'])==14
        assert len(pins)==c['logical_pin_occurrences']==m['input_pins']
        seen={}
        for x in pins:
            p=pathlib.Path(x['container_path']);key=str(p.resolve())
            if key in seen:assert (seen[key]['bytes'],seen[key]['sha256'])==(x['bytes'],x['sha256']),'Conflicting runtime alias'
            else:verify(p,x);seen[key]=x
        assert len(seen)==c['unique_container_files']
        return pins
    pins=verify_all()
    verify(s/'CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json',m['card']);verify(s/'preflight-review.json',m['preflight'])
    pre=json.loads((s/'preflight-review.json').read_text());assert pre['status']=='passed' and pre['card_sha256']==m['card']['sha256']
    with zipfile.ZipFile(r/'source.zip') as z:
        expected={x['path']:x for x in m['files']};assert len(expected)==m['source_capsule_files'] and set(z.namelist())==set(expected) and len(z.namelist())==len(set(z.namelist()))==len(expected)
        for name,x in expected.items():
            b=z.read(name);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
    if a.phase=='Stage':
        for name in ('archive_reader.py','diagnose.py','schema.py'):ast.parse((s/name).read_text(encoding='utf-8-sig'),filename=name)
        receipt.update(ast_sources=3,numeric_storage_reads=0,project_module_imports=0,input_pins_verified=len(pins),source_capsule_files=len(m['files']),metadata_copies=len(c['metadata_copies_required_before_execution']),unique_container_paths=c['unique_container_files'])
        exit_code=0
    else:
        assert pin(r/'source-stage.json')['sha256']==a.stage_sha,'Reviewed exact stage before JSON parse'
        stage=json.loads((r/'source-stage.json').read_text());assert stage['status']=='passed'
        assert stage['input_pins_verified']==len(pins) and stage['source_capsule_files']==len(m['files']) and stage['ast_sources']==3
        verify(r/'manifest.json',stage['manifest'])
        schema=r/'schema-stage.json'
        if a.phase=='Run':
            assert schema.is_file() and not schema.is_symlink(),'Mandatory Schema phase before numeric Run'
            assert a.schema_sha and pin(schema)['sha256']==a.schema_sha,'Explicit schema SHA before parse'
            observed=json.loads(schema.read_text())
            assert observed['kind']=='cpb_v11_v13_TRAIN_diagnosis_schema_v1' and observed['status']=='passed'
            assert observed['reviewed_stage_sha256']==a.stage_sha,'Schema binds the same externally reviewed Stage'
            verify(r/'manifest.json',observed['manifest']);verify(s/'CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json',observed['card'])
            verify(pathlib.Path(__file__),observed['wrapper'])
            assert observed['external_transport_sha256']=={'manifest':a.manifest_sha,'card':a.card_sha,'preflight':a.preflight_sha,'ZIP':a.zip_sha,'wrapper':a.wrapper_sha},'Schema preserves exact source/card/transport closure'
        start='schema-start.json' if a.phase=='Schema' else 'execution-start.json'
        admission={'kind':'cpb_v11_v13_'+phase+'_root_admission_v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewed_stage_sha256':a.stage_sha,'reviewed_schema_sha256':a.schema_sha or None,'once_only':True}
        write(r/start,admission)
        if a.phase=='Schema':argv=['/usr/bin/flock','-n',str(r/'execution.lock'),'/usr/bin/python3','-B',str(s/'schema.py'),str(s/'CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'),a.card_sha]
        else:argv=['/usr/bin/flock','-n',str(r/'execution.lock'),'/usr/bin/python3','-B',str(s/'diagnose.py'),'--card',str(s/'CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'),'--card-sha256',a.card_sha,'--output',str(r/'run-001')]
        with (r/(phase+'-stdout.log')).open('xb') as out,(r/(phase+'-stderr.log')).open('xb') as err:
            child=subprocess.run(argv,stdout=out,stderr=err,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        exit_code=child.returncode;receipt.update(argv=argv,exit_code=exit_code,once_only=True,admission=pin(r/start),stdout=pin(r/(phase+'-stdout.log')),stderr=pin(r/(phase+'-stderr.log')))
        if a.phase=='Schema' and exit_code==0:
            result=json.loads((r/'schema-stdout.log').read_text());assert result['status']=='passed_metadata_only' and result['numeric_storage_reads']==0 and result['model_calls']==result['head_fits']==result['training_updates']==0
        report=r/'run-001'/('diagnosis.json' if exit_code==0 else 'failure.json')
        if a.phase=='Run' and report.is_file():receipt['report']=pin(report)
    verify_all()
    receipt.update(status='passed' if exit_code==0 else 'failed',input_pins_verified=len(pins),source_capsule_files=len(m['files']),unique_container_paths=c['unique_container_files'],manifest=pin(r/'manifest.json'),card=pin(s/'CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'),wrapper=pin(pathlib.Path(__file__)),reviewed_stage_sha256=a.stage_sha or None,reviewed_schema_sha256=a.schema_sha or None,external_transport_sha256={'manifest':a.manifest_sha,'card':a.card_sha,'preflight':a.preflight_sha,'ZIP':a.zip_sha,'wrapper':a.wrapper_sha})
except Exception as e:
    receipt.update(error=str(e),traceback=traceback.format_exc());exit_code=1
receipt['elapsed_seconds']=time.monotonic()-started
write(r/stage_name,receipt)
print(json.dumps({'status':receipt['status'],'phase':a.phase,'receipt':pin(r/stage_name)}))
sys.exit(exit_code)
'@

if($Phase-eq'Stage'){
 if(Test-Path -LiteralPath $stageHostRoot){Require ((Get-Item -LiteralPath $stageHostRoot).PSIsContainer-and@(Get-ChildItem -LiteralPath $stageHostRoot -Force).Count-eq0) 'Preserve every nonempty host attempt; no historical exception'}
 $null=[IO.Directory]::CreateDirectory($stageHostRoot);$source=Join-Path $stageHostRoot 'source';$null=[IO.Directory]::CreateDirectory($source)
 $capsule=@();foreach($name in $names){$capsule+=[ordered]@{host_repository='cuwacunu_embedding';host_path=($family+$name);name=$name}}
 $capsule+=@([ordered]@{host_repository='cuwacunu_embedding';host_path=$cardRelative;name='CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'},[ordered]@{host_repository='cuwacunu_embedding';host_path=$preflightRelative;name='preflight-review.json'})
 foreach($item in $card.metadata_copies_required_before_execution){
  Require ($item.container_path.StartsWith($containerRoot+'/source/',[StringComparison]::Ordinal)) 'Context copy escapes exact source leaf'
  $name=$item.container_path.Substring(($containerRoot+'/source/').Length)
  Require ($name.StartsWith('preservation/',[StringComparison]::Ordinal)-and$name-notmatch'[\\:]'-and@($name.Split('/')|Where-Object {$_-in@('','..','.')} ).Count-eq0-and$name-ne'final-preservation.json') 'Bounded distinct context metadata copy'
  $declared=@((@($card.allowed_inputs|Where-Object id -eq cached_v11_prior_TRAIN_diagnosis)+@($card.context_inputs)+@($card.pre_execution_preservation))|Where-Object {$_.id-eq$item.id-and$_.container_path-eq$item.container_path-and$_.host_repository-eq$item.host_repository-and$_.host_path-eq$item.host_path-and$_.bytes-eq$item.bytes-and$_.sha256-eq$item.sha256})
  Require ($declared.Count-eq1) 'Every context copy exactly aliases one metadata-only role'
  $capsule+=[ordered]@{host_repository=$item.host_repository;host_path=$item.host_path;name=$name}
 }
 Require ($capsule.Count-eq($names.Count+2+$card.metadata_copies_required_before_execution.Count)-and@($capsule.name|Sort-Object -Unique).Count-eq$capsule.Count) 'Actual unique capsule membership; cache, contexts and dedicated fresh receipt each once'
 $pins=@();foreach($item in $capsule){$target=Join-Path $source $item.name;$null=[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target));[IO.File]::WriteAllBytes($target,[IO.File]::ReadAllBytes((Host-Path $item.host_path $item.host_repository)));$pins+=Pin $target $item.name}
 $wrapperPath=Join-Path $stageHostRoot 'phase-wrapper.py';[IO.File]::WriteAllText($wrapperPath,$wrapper.Replace([string][char]13,'')+"`n",[Text.UTF8Encoding]::new($false))
 Add-Type -AssemblyName System.IO.Compression.FileSystem
 $zipPath=Join-Path $stageHostRoot 'source.zip';$zip=[IO.Compression.ZipFile]::Open($zipPath,[IO.Compression.ZipArchiveMode]::Create)
 try{foreach($pin in $pins){$null=[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,(Join-Path $source $pin.path),$pin.path,[IO.Compression.CompressionLevel]::Optimal)}}finally{$zip.Dispose()}
 $manifest=[ordered]@{kind='cpb_v11_v13_TRAIN_diagnosis_execution_capsule_v1';files=$pins;archive=(Pin $zipPath 'source.zip');wrapper=(Pin $wrapperPath 'phase-wrapper.py');card=(Pin $cardPath $cardRelative);preflight=(Pin $preflightPath $preflightRelative);helper=(Pin $PSCommandPath 'doc/causal_v11_v13_train_diagnosis/run-diagnosis.ps1');parent_preservation=$card.parent_preservation;container_identity=$identity;container_root=$containerRoot;source_capsule_files=$pins.Count;input_pins=$inputPins.Count;metadata_copies=$card.metadata_copies_required_before_execution.Count;model_calls=0;head_fits=0;training_updates=0}
 $manifestPath=Join-Path $stageHostRoot 'manifest.json';Write-NewJSON $manifestPath $manifest
 & docker.exe exec cuwacunu python3 -B -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert not p.exists() or (p.is_dir() and not any(p.iterdir())), "Preserve nonempty container attempt"; p.mkdir(parents=True,exist_ok=True)' $containerRoot
 Require ($LASTEXITCODE-eq0) 'Exclusive existing-container namespace; no nonempty reuse'
 & docker.exe cp $source ('cuwacunu:'+$containerRoot+'/source');Require ($LASTEXITCODE-eq0) 'Dynamic source capsule copy; preserve partial state on failure'
 foreach($path in @($manifestPath,$zipPath,$wrapperPath)){& docker.exe cp $path ('cuwacunu:'+$containerRoot+'/'+[IO.Path]::GetFileName($path));Require ($LASTEXITCODE-eq0) 'Capsule orchestration copy'}
}else{
 Require (Test-Path -LiteralPath $stageHostRoot -PathType Container) 'Previously staged exclusive attempt required'
 $manifest=Get-Content -LiteralPath (Join-Path $stageHostRoot 'manifest.json') -Raw|ConvertFrom-Json
 Require ($manifest.container_root-eq$containerRoot-and$manifest.card.sha256-eq$CardSha256-and$manifest.preflight.sha256-eq$PreflightSha256) 'Exact previously staged declaration'
 Verify $PSCommandPath $manifest.helper
}
$stdout=Join-Path $stageHostRoot ($Phase.ToLowerInvariant()+'-orchestration-stdout.log');$stderr=Join-Path $stageHostRoot ($Phase.ToLowerInvariant()+'-orchestration-stderr.log')
Require (-not(Test-Path -LiteralPath $stdout)-and-not(Test-Path -LiteralPath $stderr)) 'Preserve every earlier phase invocation; no replay'
$arguments=@('exec','cuwacunu','python3','-B',($containerRoot+'/phase-wrapper.py'),$Phase,$containerRoot)
$arguments+=@('--manifest-sha',(Pin (Join-Path $stageHostRoot 'manifest.json') 'manifest.json').sha256,'--card-sha',$CardSha256,'--preflight-sha',$PreflightSha256,'--zip-sha',$manifest.archive.sha256,'--wrapper-sha',$manifest.wrapper.sha256)
if($Phase-ne'Stage'){$arguments+=@('--stage-sha',$StageReceiptSha256);if($SchemaReceiptSha256){Require ($SchemaReceiptSha256-match'^[0-9a-f]{64}$') 'Reviewed schema hash format';$arguments+=@('--schema-sha',$SchemaReceiptSha256)}}
Write-NewJSON (Join-Path $stageHostRoot ($Phase.ToLowerInvariant()+'-invocation.json')) ([ordered]@{phase=$Phase;argv=(@('docker.exe')+$arguments);card_sha256=$CardSha256;preflight_sha256=$PreflightSha256;root_admitted_schema=[bool]$RootAdmittedSchema;root_admitted_run=[bool]$RootAdmittedRun;no_model_calls=$true})
& docker.exe @arguments 1> $stdout 2> $stderr
$phaseExit=$LASTEXITCODE
$mirrored=Mirror-Snapshot
Require ($phaseExit-eq0) ('Phase failed; all source/log/stage/report evidence preserved in '+$mirrored.path)
[ordered]@{status='passed';phase=$Phase;container_root=$containerRoot;mirror=$mirrored;numeric_diagnosis_executed=($Phase-eq'Run');metadata_only=($Phase-in@('Stage','Schema'))}|ConvertTo-Json -Depth 6


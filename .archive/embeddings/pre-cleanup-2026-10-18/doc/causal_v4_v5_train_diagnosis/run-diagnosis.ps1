# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host orchestration only. Saved-data scalar arithmetic runs once in cuwacunu.
param([ValidateRange(1,999)][int]$Attempt=1)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$label='attempt-{0:d3}' -f $Attempt
$root='/opt/cuwacunu/research/causal-v4-v5-train-diagnosis/'+$label
$hostRoot=Join-Path $repo ('output/engineering/causal-v4-v5-train-diagnosis/'+$label)
$cardRelative='doc/CAUSAL_V4_V5_TRAIN_DIAGNOSIS_CARD.json'
$preflightRelative='doc/causal_v4_v5_train_diagnosis/preflight-review.json'
$family='code/research/causal_v4_v5_train_diagnosis/'
$names=@('archive_reader.py','diagnose.py','schema.py','README.md','provenance.json')
function Get-Pin([string]$Path,[string]$Label){
    $f=Get-Item -LiteralPath $Path
    if($f.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Linked file refused'}
    [ordered]@{path=$Label;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Test-Pin([string]$Path,$Pin){
    $actual=Get-Pin $Path $Pin.path
    if($actual.bytes -ne $Pin.bytes -or $actual.sha256 -ne $Pin.sha256){throw "Pinned bytes differ: $Path"}
}
$cardPath=Join-Path $repo $cardRelative
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
if($card.stage -ne 'prospective_before_diagnostic_arithmetic'){throw 'Frozen prospective diagnosis card required'}
if($card.allowed_inputs.Count -ne 41 -or $card.preservation_pins.Count -ne 23){throw 'Exact41/23 input cardinality required'}
if($card.pre_execution_preservation.container_path -ne ($root+'/source/final-preservation.json')){throw 'Card must bind this exact exclusive attempt; never rewrite it implicitly'}
$preflightPath=Join-Path $repo $preflightRelative
$preflight=Get-Content -LiteralPath $preflightPath -Raw|ConvertFrom-Json
if($preflight.status -ne 'passed'){throw 'Independent static preflight must pass'}
$sourceFiles=@(Get-ChildItem -LiteralPath (Join-Path $repo $family) -File|ForEach-Object Name|Sort-Object)
if(($sourceFiles -join '|') -ne (($names|Sort-Object) -join '|')){throw 'Exactly five owned source files required'}
foreach($relative in (@($names|ForEach-Object {$family+$_})+@($cardRelative))){
    $pin=@($preflight.files|Where-Object {$_.path -eq $relative})
    if($pin.Count -ne 1){throw "Preflight must bind exact source/card: $relative"}
    Test-Pin (Join-Path $repo $relative) $pin[0]
}
$inputPins=@($card.allowed_inputs)+@($card.preservation_pins)+@($card.pre_execution_preservation)
foreach($pin in $inputPins){
    if($pin.host_repository -notin @('cuwacunu_embedding','cuwacunu_dev')){throw 'Unknown host input repository'}
    $checkout=[IO.Path]::GetFullPath((Join-Path $repo ('../'+$pin.host_repository)))
    $path=[IO.Path]::GetFullPath((Join-Path $checkout $pin.host_path))
    if(-not $path.StartsWith(($checkout+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)){throw 'Input escaped checkout'}
    Test-Pin $path $pin
}
$identity=docker inspect cuwacunu --format '{{.Id}} {{.State.Status}} {{.Image}}'
$expected='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e running sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'
if($LASTEXITCODE -ne 0 -or $identity -ne $expected){throw 'Existing managed container/image/running guard failed'}
if((Test-Path -LiteralPath $hostRoot) -and @(Get-ChildItem -LiteralPath $hostRoot -Force).Count){throw 'Preserve nonempty host attempt'}
docker exec cuwacunu python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert not p.exists() or not any(p.iterdir()), "Preserve nonempty container attempt"' $root
if($LASTEXITCODE -ne 0){throw 'Exclusive container target guard'}
$source=Join-Path $hostRoot 'source'
[IO.Directory]::CreateDirectory($source)|Out-Null
$capsule=@()
foreach($name in $names){$capsule+=@{original=($family+$name);name=$name}}
$capsule+=@{original=$cardRelative;name='CAUSAL_V4_V5_TRAIN_DIAGNOSIS_CARD.json'}
$capsule+=@{original=$preflightRelative;name='preflight-review.json'}
$capsule+=@{original=$card.pre_execution_preservation.host_path;name='final-preservation.json'}
$pins=@()
foreach($item in $capsule){
    $original=Join-Path $repo $item.original
    $target=Join-Path $source $item.name
    [IO.File]::WriteAllBytes($target,[IO.File]::ReadAllBytes($original))
    $pins+=Get-Pin $target $item.name
}
if($pins.Count -ne 8){throw 'Exact eight capsule inputs required'}
$zip=Join-Path $hostRoot 'source.zip'
Compress-Archive -LiteralPath @($pins|ForEach-Object {Join-Path $source $_.path}) -DestinationPath $zip
$manifest=[ordered]@{kind='cpb_v4_v5_TRAIN_diagnosis_execution_capsule_v1';files=$pins;archive=(Get-Pin $zip 'source.zip');helper=(Get-Pin $PSCommandPath 'doc/causal_v4_v5_train_diagnosis/run-diagnosis.ps1');container_identity=$identity;container_root=$root;model_calls=0;head_fits=0}
$manifestPath=Join-Path $hostRoot 'manifest.json'
[IO.File]::WriteAllText($manifestPath,(($manifest|ConvertTo-Json -Depth 12).Replace([string][char]13,'')+[string][char]10),[Text.UTF8Encoding]::new($false))
docker exec cuwacunu python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert not p.exists() or not any(p.iterdir()); p.mkdir(parents=True,exist_ok=True)' $root
if($LASTEXITCODE -ne 0){throw 'Exclusive container leaf creation'}
docker cp $source ('cuwacunu:'+$root+'/source')
if($LASTEXITCODE -ne 0){throw 'Source copy failed'}
foreach($path in @($manifestPath,$zip)){docker cp $path ('cuwacunu:'+$root+'/'+[IO.Path]::GetFileName($path));if($LASTEXITCODE -ne 0){throw 'Capsule receipt/archive copy failed'}}
$runCode=@'
import ast,datetime,hashlib,json,os,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]); s=r/'source'; m=json.loads((r/'manifest.json').read_text())
def pin(p):
    d=p.read_bytes(); return {'path':str(p),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
def verify(p,x):
    assert p.is_file() and not p.is_symlink(), str(p)
    a=pin(p); assert a['bytes']==x['bytes'] and a['sha256']==x['sha256'], str(p)
assert len(m['files'])==8
for x in m['files']: verify(s/x['path'],x)
verify(r/'source.zip',m['archive'])
c=json.loads((s/'CAUSAL_V4_V5_TRAIN_DIAGNOSIS_CARD.json').read_text())
inputs=c['allowed_inputs']+c['preservation_pins']+[c['pre_execution_preservation']]
assert len(inputs)==65
for x in inputs: verify(pathlib.Path(x['container_path']),x)
for name in ('archive_reader.py','diagnose.py','schema.py'): ast.parse((s/name).read_text())
with (r/'execution-start.json').open('x') as out: json.dump({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'once_only':True},out)
argv=['/usr/bin/flock','-n',str(r/'execution.lock'),'/usr/bin/python3',str(s/'diagnose.py'),'--card',str(s/'CAUSAL_V4_V5_TRAIN_DIAGNOSIS_CARD.json'),'--output',str(r/'run-001')]
t=time.monotonic()
with (r/'stdout.log').open('xb') as stdout,(r/'stderr.log').open('xb') as stderr:
    result=subprocess.run(argv,stdout=stdout,stderr=stderr,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
for x in m['files']: verify(s/x['path'],x)
for x in inputs: verify(pathlib.Path(x['container_path']),x)
receipt={'kind':'cpb_v4_v5_saved_TRAIN_diagnosis_execution_v1','status':'passed' if result.returncode==0 else 'failed','exit_code':result.returncode,'argv':argv,'elapsed_seconds':time.monotonic()-t,'source_files':8,'input_pins_verified':65,'stdout':pin(r/'stdout.log'),'stderr':pin(r/'stderr.log'),'manifest':pin(r/'manifest.json'),'model_calls':0,'head_fits':0,'new_training_updates':0}
p=r/'run-001'/('diagnosis.json' if result.returncode==0 else 'failure.json')
if p.is_file(): receipt['report']=pin(p)
with (r/'execution-stage.json').open('x') as out: json.dump(receipt,out,indent=2); out.write('\n')
print(json.dumps({'status':receipt['status'],'exit_code':result.returncode,'root':str(r)}))
'@
docker exec cuwacunu python3 -c $runCode $root
if($LASTEXITCODE -ne 0){throw 'Execution wrapper failed; preserve container and host state'}
$mirror=Join-Path $hostRoot 'container-mirror'
if(Test-Path -LiteralPath $mirror){throw 'Preserve existing mirror'}
docker cp ('cuwacunu:'+$root) $mirror
if($LASTEXITCODE -ne 0){throw 'Mirror failed; preserve partial state'}
$inventoryCode='import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); print(json.dumps([{ "path":p.relative_to(r).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(r.rglob("*")) if p.is_file()]))'
$inventory=docker exec cuwacunu python3 -c $inventoryCode $root
if($LASTEXITCODE -ne 0){throw 'Mirror hash inventory failed'}
$mirrored=@($inventory|ConvertFrom-Json)
foreach($pin in $mirrored){Test-Pin (Join-Path $mirror $pin.path) $pin}
$stage=Get-Content -LiteralPath (Join-Path $mirror 'execution-stage.json') -Raw|ConvertFrom-Json
$receipt=[ordered]@{kind='cpb_v4_v5_TRAIN_diagnosis_exact_mirror_v1';status=$stage.status;container_root=$root;host_root=$mirror;files=$mirrored;all_sizes_sha256_exact=$true;source_capsule_files=8;manifest=(Get-Pin $manifestPath 'manifest.json');helper=$manifest.helper;model_calls=0;head_fits=0;training_updates=0;failed_attempts_deleted=0}
$receiptPath=Join-Path $hostRoot 'mirror-receipt.json'
[IO.File]::WriteAllText($receiptPath,(($receipt|ConvertTo-Json -Depth 12).Replace([string][char]13,'')+[string][char]10),[Text.UTF8Encoding]::new($false))
if($stage.status -ne 'passed'){throw 'Saved-data diagnosis failed; report/stdout/stderr/source/stage mirrors preserved'}
[ordered]@{status='passed';report=(Join-Path $mirror 'run-001/diagnosis.json');receipt=$receiptPath}|ConvertTo-Json

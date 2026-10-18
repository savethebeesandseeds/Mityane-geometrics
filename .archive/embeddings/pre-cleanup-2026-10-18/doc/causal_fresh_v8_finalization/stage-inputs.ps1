# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
<#
Host file/hash/ZIP/Docker orchestration only. Invoke only after root approves
the actual completed-evidence plan, with -Plan FILE -PlanSha256 APPROVED_SHA.
Plan kind: cpb_v8_metadata_finalization_input_plan_v1; JSON approval flag
approved_for_metadata_card_generation must already be true.
Each of37 unique inputs has id,path,bytes,sha256,format,source_path.
path is its durable research-relative forward-slash host label. source_path
MUST ALREADY be /opt/cuwacunu/research/causal-fresh-v8-finalization/attempt-001/source/inputs/ID.json;
only results/review/detail use .txt and format=text. No plan bytes are rewritten.
The helper copies five frozen tools+37 inputs+plan, verifies bounded ZIP/hash
paths in the existing container and ast.parse on three Python files. It NEVER
imports/executes those modules, prepare_card.py, finalize.py, models or heads.
Existing prior-live and all other siblings remain untouched. No lifecycle calls.
#>
param(
 [Parameter(Mandatory=$true)][string]$Plan,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$stageRoot=Join-Path $researchRoot 'output/engineering/causal-fresh-v8-finalization/attempt-001/input-stage-001'
$sourceRelative='code/research/causal_fresh_v8_finalization'
$container='cuwacunu'
$containerId='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$containerImage='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'
$containerAttempt='/opt/cuwacunu/research/causal-fresh-v8-finalization/attempt-001'
$containerSource=$containerAttempt+'/source'
$containerArchive=$containerAttempt+'/input-stage-001.zip'
$containerReceipt=$containerAttempt+'/input-stage-001-receipt.json'
$roles=@('ledger','catalog','prior_snapshot','descriptor','experiment_card',
 'source_manifest','build','training','resume','initialization','calibration',
 'engineering','gradient','initial_proof','final_proof','training_audit',
 'quality_audit','pairings','establishment','mirror','quality_receipt',
 'comparison','v3_table','v4_table','v5_table','v6_table','v7_table',
 'v6_establishment','v8_table','v7_establishment','temporal_proof','temporal_learning','host_recovery_card','host_recovery_review','results','review','detail')
$tools=@(
 [ordered]@{name='finalize.py';bytes=44631;sha256='e40969573ba56f8522a0fb167ad5b6bcf86b6703f38eb4b91ea4637374890bf3'},
 [ordered]@{name='prepare_card.py';bytes=10116;sha256='71c8edb24c5474c7acb4c3286cefc19b88d9d9138c38bce2a68608362ef95e9f'},
 [ordered]@{name='review_proposals.py';bytes=26441;sha256='4777f528dd410ffd12c2ca7e0f9f95e25aea8c35d181fde5be1f3fb9d9a6445f'},
 [ordered]@{name='README.md';bytes=11707;sha256='f54fe495c78821b48b12119086e68a35b66a452742878f635c5d3d0e9845e1d5'},
 [ordered]@{name='provenance.json';bytes=5933;sha256='1b458d4fec688127327482630dbe51c87d996e8f11b82d3e96ea818cd65d3b9d'}
)
function Require([bool]$ok,[string]$message){if(-not $ok){throw $message}}
function Research-File([string]$label){
 Require (-not [string]::IsNullOrWhiteSpace($label)) 'Nonempty research-relative label required'
 Require ($label -eq $label.Replace([char]92,[char]47)) 'Plan labels must already use forward slashes'
 Require (-not [IO.Path]::IsPathRooted($label) -and $label -notmatch ':' -and -not $label.StartsWith('/')) 'Absolute/drive label forbidden'
 Require (@($label.Split('/')|Where-Object{$_ -in @('','..','.')}).Count -eq 0) 'Empty/dot/traversal segment forbidden'
 $full=[IO.Path]::GetFullPath((Join-Path $researchRoot $label))
 $prefix=$researchRoot.TrimEnd([char]92,[char]47)+[IO.Path]::DirectorySeparatorChar
 Require ($full.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) 'File escaped research workspace'
 $item=Get-Item -LiteralPath $full
 Require (-not $item.PSIsContainer) ('Expected file: '+$label)
 $walk=$item
 while($walk.FullName -ne $researchRoot){
  Require (($walk.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) ('Linked path forbidden: '+$walk.FullName)
  $walk=Get-Item -LiteralPath ([IO.Path]::GetDirectoryName($walk.FullName))
 }
 return $full
}
function File-Pin([string]$path){
 return [ordered]@{bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Require-Pin([string]$path,$expected){
 $actual=File-Pin $path
 Require ($actual.bytes -eq $expected.bytes -and $actual.sha256 -eq $expected.sha256) ('Changed approved file: '+$path)
}
function Write-Json([string]$path,$value){
 Require (-not (Test-Path -LiteralPath $path)) ('Preserve existing file: '+$path)
 [IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 30)+([string][char]10),[Text.UTF8Encoding]::new($false))
}
function Docker-Exit([string]$label){if($LASTEXITCODE -ne 0){throw ('Docker failure; preserve partial stage: '+$label)}}

# Inspect approved bytes before creating only the new input-stage child.
Require (-not (Test-Path -LiteralPath $stageRoot)) 'Preserve existing input-stage-001 leaf'
$absolute=if([IO.Path]::IsPathRooted($Plan)){[IO.Path]::GetFullPath($Plan)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $Plan))}
$planRelative=[IO.Path]::GetRelativePath($researchRoot,$absolute).Replace([char]92,[char]47)
$planPath=Research-File $planRelative
$planPin=File-Pin $planPath
Require ($planPin.sha256 -eq $PlanSha256) 'Plan differs from explicit approved SHA256'
$document=Get-Content -LiteralPath $planPath -Raw|ConvertFrom-Json
Require ($document.kind -eq 'cpb_v8_metadata_finalization_input_plan_v1') 'Own reviewed plan required'
Require ($document.approved_for_metadata_card_generation -is [bool] -and $document.approved_for_metadata_card_generation -eq $true) 'Plan must explicitly contain approval=true'
Require ($document.container_attempt -eq '/opt/cuwacunu/research/causal-fresh-v8/attempt-001') 'Only completed V8 attempt admitted'
Require ($document.inputs.Count -eq 37) 'Exactly37 approved metadata inputs'
$unique=@($document.inputs.id|Sort-Object -Unique)
Require ($unique.Count -eq 37 -and -not (Compare-Object $roles $unique)) 'Exact37 unique prospective roles'
$files=@()
foreach($tool in $tools){
 $hostPath=Research-File ($sourceRelative+'/'+$tool.name)
 Require-Pin $hostPath $tool
 $files += [ordered]@{path=$tool.name;source_path=($sourceRelative+'/'+$tool.name);bytes=$tool.bytes;sha256=$tool.sha256;role='owned_source'}
}
Require ($document.generator.bytes -eq $tools[0].bytes -and $document.generator.sha256 -eq $tools[0].sha256) 'Plan binds actual frozen finalizer'
foreach($row in $document.inputs){
 Require ($row.id -in $roles -and $row.sha256 -cmatch '^[0-9a-f]{64}$' -and $row.bytes -gt 0) 'Exact role/content pin required'
 $format=if($row.id -in @('results','review','detail')){'text'}else{'json'}
 $extension=if($format -eq 'text'){'.txt'}else{'.json'}
 $destination='inputs/'+$row.id+$extension
 Require ($row.format -eq $format -and $row.source_path -eq ($containerSource+'/'+$destination)) 'Preapproved source_path must exactly name staged file; no rewrite'
 $hostPath=Research-File $row.path
 Require-Pin $hostPath $row
 $files += [ordered]@{path=$destination;source_path=$row.path;bytes=$row.bytes;sha256=$row.sha256;role=$row.id}
}
$files += [ordered]@{path='input-plan.json';source_path=$planRelative;bytes=$planPin.bytes;sha256=$planPin.sha256;role='approved_input_plan'}
Require ($files.Count -eq 43 -and @($files.path|Sort-Object -Unique).Count -eq 43) 'Five sources+37 inputs+unchanged plan'
$inspection=@(& docker.exe inspect --format '{{.Id}}|{{.Image}}|{{.State.Running}}' $container)
Docker-Exit 'inspect existing managed container'
Require ($inspection.Count -eq 1 -and $inspection[0] -eq ($containerId+'|'+$containerImage+'|true')) 'Expected immutable existing running container only'
$null=New-Item -ItemType Directory -Path $stageRoot -ErrorAction Stop
$hostSource=Join-Path $stageRoot 'source'
$null=New-Item -ItemType Directory -Path $hostSource -ErrorAction Stop
try{
 foreach($row in $files){
  $from=Research-File $row.source_path
  Require-Pin $from $row
  $to=Join-Path $hostSource $row.path
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($to))|Out-Null
  Require (-not (Test-Path -LiteralPath $to)) 'Exclusive staged copy required'
  Copy-Item -LiteralPath $from -Destination $to
  Require-Pin $to $row
 }
 $manifest=[ordered]@{kind='cpb_v8_finalization_input_capsule_v1';status='prepared_not_executed';created_utc=[DateTime]::UtcNow.ToString('o');
  container=$container;container_id=$containerId;container_image=$containerImage;container_source=$containerSource;
  approved_plan=[ordered]@{path=$planRelative;bytes=$planPin.bytes;sha256=$planPin.sha256;approval_flag=$true};
  stager=[ordered]@{path='doc/causal_fresh_v8_finalization/stage-inputs.ps1';bytes=(File-Pin $PSCommandPath).bytes;sha256=(File-Pin $PSCommandPath).sha256};
  files=$files;source_files=5;metadata_inputs=37;plan_files=1;module_executions=0;model_calls=0;head_fits=0;training_updates=0;live_files_written=0}
 $manifestPath=Join-Path $stageRoot 'source-manifest.json'
 Write-Json $manifestPath $manifest
 $manifestPin=File-Pin $manifestPath
 Add-Type -AssemblyName System.IO.Compression
 Add-Type -AssemblyName System.IO.Compression.ZipFile
 $zipPath=Join-Path $stageRoot 'source.zip'
 $stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
 $zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
 try{
  foreach($row in $files){
   $path=Join-Path $hostSource $row.path
   Require-Pin $path $row
   [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$row.path,[IO.Compression.CompressionLevel]::Fastest)|Out-Null
  }
  [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$manifestPath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null
 }finally{$zip.Dispose();$stream.Dispose()}
 $zipPin=File-Pin $zipPath
 $guard=@'
import pathlib,sys
source,archive,receipt=map(pathlib.Path,sys.argv[1:])
root=pathlib.Path('/opt/cuwacunu/research/causal-fresh-v8-finalization/attempt-001')
assert source==root/'source' and archive==root/'input-stage-001.zip' and receipt==root/'input-stage-001-receipt.json'
assert not source.exists() and not archive.exists() and not receipt.exists()
for path in (root,*root.parents):
    if path.exists(): assert path.is_dir() and not path.is_symlink()
root.mkdir(parents=True,exist_ok=True)
source.mkdir()
print('exclusive existing-container metadata source leaf created')
'@
 [IO.File]::WriteAllText((Join-Path $stageRoot 'container-guard.py'),$guard+([string][char]10),[Text.UTF8Encoding]::new($false))
 $guardOutput=@($guard|& docker.exe exec -i $container python3 - $containerSource $containerArchive $containerReceipt 2>&1)
 $guardOutput|Set-Content -LiteralPath (Join-Path $stageRoot 'guard.log') -Encoding utf8NoBOM
 Docker-Exit 'exclusive container source guard'
 $copyOutput=@(& docker.exe cp $zipPath ($container+':'+$containerArchive) 2>&1)
 $copyOutput|Set-Content -LiteralPath (Join-Path $stageRoot 'copy.log') -Encoding utf8NoBOM
 Docker-Exit 'copy frozen metadata archive'
 $verify=@'
import ast,hashlib,json,pathlib,stat,sys,zipfile
archive,source,receipt=map(pathlib.Path,sys.argv[1:4])
zip_sha,manifest_sha,plan_sha=sys.argv[4:]
root=pathlib.Path('/opt/cuwacunu/research/causal-fresh-v8-finalization/attempt-001')
assert archive==root/'input-stage-001.zip' and source==root/'source' and receipt==root/'input-stage-001-receipt.json'
assert source.is_dir() and not source.is_symlink() and not list(source.iterdir()) and not receipt.exists()
assert archive.is_file() and not archive.is_symlink()
digest=lambda b:hashlib.sha256(b).hexdigest()
assert digest(archive.read_bytes())==zip_sha
with zipfile.ZipFile(archive) as package:
    names=package.namelist()
    assert len(names)==len(set(names))==44
    manifest_bytes=package.read('CAPSULE_MANIFEST.json')
    assert digest(manifest_bytes)==manifest_sha
    manifest=json.loads(manifest_bytes)
    assert manifest['kind']=='cpb_v8_finalization_input_capsule_v1' and manifest['container_source']==str(source)
    files=manifest['files']
    assert len(files)==43 and set(names)=={r['path'] for r in files}|{'CAPSULE_MANIFEST.json'}
    assert manifest['approved_plan']['approval_flag'] is True and manifest['approved_plan']['sha256']==plan_sha
    for info in package.infolist():
        relative=pathlib.PurePosixPath(info.filename)
        assert not relative.is_absolute() and '..' not in relative.parts and chr(92) not in info.filename
        assert not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16)
    for row in files:
        data=package.read(row['path'])
        assert len(data)==row['bytes'] and digest(data)==row['sha256']
        target=source/pathlib.PurePosixPath(row['path'])
        assert target.resolve().is_relative_to(source.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream: stream.write(data)
    with (source/'CAPSULE_MANIFEST.json').open('xb') as stream: stream.write(manifest_bytes)
plan=json.loads((source/'input-plan.json').read_text(encoding='utf-8-sig'))
assert digest((source/'input-plan.json').read_bytes())==plan_sha
assert plan['kind']=='cpb_v8_metadata_finalization_input_plan_v1' and plan['approved_for_metadata_card_generation'] is True
assert plan['container_attempt']=='/opt/cuwacunu/research/causal-fresh-v8/attempt-001'
roles={'ledger','catalog','prior_snapshot','descriptor','experiment_card','source_manifest','build','training','resume',
 'initialization','calibration','engineering','gradient','initial_proof','final_proof','training_audit','quality_audit',
 'pairings','establishment','mirror','quality_receipt','comparison','v3_table','v4_table','v5_table','v6_table','v7_table',
 'v6_establishment','v8_table','v7_establishment','temporal_proof','temporal_learning','host_recovery_card','host_recovery_review','results','review','detail'}
assert len(plan['inputs'])==37 and {r['id'] for r in plan['inputs']}==roles
for row in plan['inputs']:
    fmt='text' if row['id'] in {'results','review','detail'} else 'json'
    extension='.txt' if fmt=='text' else '.json'
    path=source/'inputs'/(row['id']+extension)
    assert row['format']==fmt and row['source_path']==str(path)
    data=path.read_bytes()
    assert len(data)==row['bytes'] and digest(data)==row['sha256']
    text=data.decode('utf-8-sig')
    if fmt=='json': json.loads(text)
parsed=[]
for name in ('finalize.py','prepare_card.py','review_proposals.py'):
    ast.parse((source/name).read_text(encoding='utf-8-sig'),filename=str(source/name))
    parsed.append(name)
actual=[p for p in source.rglob('*') if p.is_file()]
assert len(actual)==44 and not any(p.is_symlink() for p in source.rglob('*'))
result=dict(kind='cpb_v8_finalization_input_stage_v1',status='passed_staged_AST_only',
 source=str(source),archive=str(archive),archive_sha256=zip_sha,manifest_sha256=manifest_sha,plan_sha256=plan_sha,
 approved_for_metadata_card_generation=True,staged_files=44,owned_code_files=5,metadata_inputs=37,AST_only_files=parsed,
 files=files,module_executions=0,card_generations=0,metadata_proposals=0,model_calls=0,head_fits=0,training_updates=0,live_files_written=0)
with receipt.open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(status=result['status'],receipt=str(receipt),staged_files=44,AST_files=3)))
'@
 [IO.File]::WriteAllText((Join-Path $stageRoot 'verify-unpack-ast.py'),$verify+([string][char]10),[Text.UTF8Encoding]::new($false))
 $argv=@('exec','-i',$container,'python3','-',$containerArchive,$containerSource,$containerReceipt,$zipPin.sha256,$manifestPin.sha256,$PlanSha256)
 Write-Json (Join-Path $stageRoot 'execution.json') ([ordered]@{kind='cpb_v8_finalization_staging_execution_v1';argv=$argv;stdin=File-Pin (Join-Path $stageRoot 'verify-unpack-ast.py');approved_plan=$manifest.approved_plan;numeric_generator_invoked=$false})
 $verifyOutput=@($verify|& docker.exe @argv 2>&1)
 $verifyOutput|Set-Content -LiteralPath (Join-Path $stageRoot 'verify.log') -Encoding utf8NoBOM
 Docker-Exit 'container content/path/AST verification'
 $receiptPath=Join-Path $stageRoot 'container-receipt.json'
 $mirrorOutput=@(& docker.exe cp ($container+':'+$containerReceipt) $receiptPath 2>&1)
 $mirrorOutput|Set-Content -LiteralPath (Join-Path $stageRoot 'receipt-copy.log') -Encoding utf8NoBOM
 Docker-Exit 'mirror exact container receipt'
 $receiptDocument=Get-Content -LiteralPath $receiptPath -Raw|ConvertFrom-Json
 Require ($receiptDocument.status -eq 'passed_staged_AST_only' -and $receiptDocument.plan_sha256 -eq $PlanSha256 -and $receiptDocument.archive_sha256 -eq $zipPin.sha256 -and $receiptDocument.manifest_sha256 -eq $manifestPin.sha256) 'Exact successful staging receipt required'
 foreach($row in $files){Require-Pin (Research-File $row.source_path) $row;Require-Pin (Join-Path $hostSource $row.path) $row}
 Require-Pin $planPath $planPin
 Write-Json (Join-Path $stageRoot 'stage.json') ([ordered]@{kind='cpb_v8_finalization_host_input_stage_v1';status='passed_staged_AST_only';container=$container;immutable_id=$containerId;container_source=$containerSource;plan=$manifest.approved_plan;zip=$zipPin;manifest=$manifestPin;container_receipt=File-Pin $receiptPath;owned_sources_verified=5;metadata_inputs_verified=37;plan_bytes_unchanged=$true;source_before_after_exact=$true;numeric_card_generation_performed=$false;finalizer_invoked=$false;live_files_written=0})
 Get-Content -LiteralPath (Join-Path $stageRoot 'stage.json') -Raw
}catch{
 $failurePath=Join-Path $stageRoot 'failure.json'
 if(-not (Test-Path -LiteralPath $failurePath)){
  Write-Json $failurePath ([ordered]@{kind='cpb_v8_finalization_input_stage_failure_v1';status='failed_preserved';error=$_.Exception.Message;cleanup_performed=$false;numeric_generators_invoked=$false;live_files_written=0})
 }
 throw
}

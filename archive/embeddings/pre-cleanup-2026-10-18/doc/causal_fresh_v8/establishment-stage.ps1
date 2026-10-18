# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Full physical V7 counterpart copy; stage14/15JSON roles and3owned join files.
# AST and JSON syntax only. No join, auditor, model or scientific arithmetic execution.
param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ManifestSha256,[string]$Manifest='doc/causal_fresh_v8/establishment-input-manifest.json')
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerName='cuwacunu'
$containerRoot='/opt/cuwacunu/research/causal-fresh-v8-establishment/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-fresh-v8-establishment/attempt-001'
function File-Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No linked source'};[ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Bounded([string]$relative){if([IO.Path]::IsPathRooted($relative)-or$relative-match'(^|/)\.\.(/|$)'-or$relative.Contains(':')-or$relative.Contains([char]92)){throw 'Bounded forward-slash repository path required'};$p=[IO.Path]::GetFullPath((Join-Path $researchRoot $relative));if(-not$p.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Path escaped workspace'};$p}
function Require-Pin([string]$path,$expected){$p=File-Pin $path;if($p.bytes-ne$expected.bytes-or$p.sha256-ne$expected.sha256){throw "Frozen join input changed: $path"}}
$manifestPath=Bounded $Manifest
if((File-Pin $manifestPath).sha256-ne$ManifestSha256){throw 'Explicit reviewed join-card hash required'}
$declaration=Get-Content -Raw -LiteralPath $manifestPath|ConvertFrom-Json
if($declaration.kind-ne'causal_fresh_v8_establishment_join_card_v1'-or$declaration.stage-ne'declared_after_completed_audits_before_metadata_join'-or$declaration.root_reviewed-ne$true){throw 'Actual completed-audit reviewed join card required'}
$roles=@('comparison','table_rows','quality_audit','training_audit','complete_mirror','card','quality_receipt','pairings','source_manifest','build_admission','temporal_pooling_proof','temporal_learning','host_recovery_card','host_recovery_review')
$ids=@($declaration.inputs.id)
if(@($ids|Sort-Object -Unique).Count-ne$ids.Count-or$ids.Count-lt14-or$ids.Count-gt15-or@($roles|Where-Object {$_-notin$ids}).Count-or@($ids|Where-Object {$_-notin($roles+@('declaration_review'))}).Count){throw 'Exact14 roles plus only optional declaration_review required'}
Require-Pin $PSCommandPath $declaration.stage_helper
$sourceSupport=@($declaration.source_support)
if($sourceSupport.Count-ne2-or(Compare-Object @($sourceSupport|ForEach-Object {[IO.Path]::GetFileName($_.path)}|Sort-Object) @(@('README.md','provenance.json')|Sort-Object))){throw 'Exactly README/provenance owned support sources required'}
$entries=@()
$programs=@([ordered]@{file='establish.py';binding=$declaration.source},[ordered]@{file='execute.py';binding=$declaration.execution_wrapper})+@($sourceSupport|ForEach-Object {[ordered]@{file=[IO.Path]::GetFileName($_.path);binding=$_}})
foreach($item in $programs){
 $path=Bounded $item.binding.path;Require-Pin $path $item.binding;$entries +=[ordered]@{file=$item.file;repository_path=$item.binding.path;bytes=$item.binding.bytes;sha256=$item.binding.sha256}
}
if($declaration.source.path-ne'code/research/causal_fresh_v8_establishment/establish.py'-or$declaration.source.sha256-ne'2bb8ac51761f6f38d86fd491bc16073ad9f40e64eb30474ac7aa7b375e818f47'-or$declaration.execution_wrapper.path-ne'doc/causal_fresh_v8/establishment-execute.py'){throw 'Independently reviewed owned join/executor source required'}
foreach($support in $sourceSupport){if($support.path-notin@('code/research/causal_fresh_v8_establishment/README.md','code/research/causal_fresh_v8_establishment/provenance.json')){throw 'Exact owned support paths required'}}
foreach($item in $declaration.inputs){if($item.staged_path-ne("inputs/$($item.id).json")){throw 'Exact role path required'};$path=Bounded $item.path;Require-Pin $path $item;$entries +=[ordered]@{file=$item.staged_path;repository_path=$item.path;bytes=$item.bytes;sha256=$item.sha256}}
foreach($item in @([ordered]@{file='input-manifest.json';path=$Manifest},[ordered]@{file='LICENSE';path='LICENSE'})){$p=File-Pin (Bounded $item.path);$entries +=[ordered]@{file=$item.file;repository_path=$item.path;bytes=$p.bytes;sha256=$p.sha256}}
$expectedCount=$ids.Count+6
if($entries.Count-ne$expectedCount-or@($entries.file|Sort-Object -Unique).Count-ne$expectedCount){throw 'Exact20/21 capsule files required'}
if(Test-Path -LiteralPath $hostRoot){throw 'Preserve existing establishment attempt'}
$observed=docker.exe inspect $containerName|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$observed.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$observed.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$observed.State.Running){throw 'Exact existing running container required'}
[IO.Directory]::CreateDirectory($hostRoot)|Out-Null
$capsule=[ordered]@{kind='cpb_v8_metadata_establishment_source_capsule_v1';scope='Metadata/AST only until separate root execution admission';container_root=$containerRoot;join_card_sha256=$ManifestSha256;metadata_roles=$ids.Count;owned_join_sources=3;files=$entries;model_calls=0;head_fits=0;training_updates=0}
$capsulePath=Join-Path $hostRoot 'source-manifest.json'
[IO.File]::WriteAllText($capsulePath,($capsule|ConvertTo-Json -Depth 10)+"`n",[Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $hostRoot 'source.zip';$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($item in $entries){$path=Bounded $item.repository_path;Require-Pin $path $item;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$item.file,[IO.Compression.CompressionLevel]::Fastest)|Out-Null};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$capsulePath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$zip.Dispose();$stream.Dispose()}
$archivePin=File-Pin $zipPath
$capsulePin=File-Pin $capsulePath
$setup=@'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);assert str(p)=='/opt/cuwacunu/research/causal-fresh-v8-establishment/attempt-001' and not p.exists()
p.mkdir(parents=True)
'@
docker.exe exec $containerName python3 -c $setup $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'setup.log')
if($LASTEXITCODE-ne0){throw 'Exclusive container root refused; preserve host capsule'}
docker.exe cp $zipPath "${containerName}:${containerRoot}/source.zip"
if($LASTEXITCODE-ne0){throw 'Capsule copy failed; preserve state'}
$stage=@'
import ast,hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]);expected_card,expected_archive,expected_capsule=sys.argv[2:5]
assert str(root)=='/opt/cuwacunu/research/causal-fresh-v8-establishment/attempt-001'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(root/'source.zip')==expected_archive
source=root/'source';source.mkdir()
with zipfile.ZipFile(root/'source.zip') as z:
 capsule_bytes=z.read('CAPSULE_MANIFEST.json')
 assert hashlib.sha256(capsule_bytes).hexdigest()==expected_capsule
 m=json.loads(capsule_bytes);assert m['container_root']==str(root) and m['join_card_sha256']==expected_card
 assert m['metadata_roles'] in (14,15) and m['owned_join_sources']==3
 assert len(m['files'])==m['metadata_roles']+6 and len({e['file'] for e in m['files']})==len(m['files'])
 assert len(z.namelist())==len(m['files'])+1 and set(z.namelist())=={e['file'] for e in m['files']}|{'CAPSULE_MANIFEST.json'}
 for e in m['files']:
  relative=pathlib.PurePosixPath(e['file']);assert not relative.is_absolute() and '..' not in relative.parts and chr(92) not in str(relative)
  data=z.read(e['file']);assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
  path=source/relative;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
 (source/'CAPSULE_MANIFEST.json').write_bytes(capsule_bytes)
 assert digest(source/'input-manifest.json')==m['join_card_sha256']==expected_card
 for name in ('establish.py','execute.py'):ast.parse((source/name).read_text(),filename=name)
 for path in source.rglob('*.json'):json.loads(path.read_text(encoding='utf-8-sig'))
 receipt=dict(kind='cpb_v8_metadata_establishment_source_stage_v1',status='passed_AST_only',files=len(m['files']),metadata_roles=m['metadata_roles'],owned_join_sources=3,AST_files=2,module_executions=0,model_calls=0,head_fits=0,training_updates=0,source_root=str(source),join_card_sha256=m['join_card_sha256'],source_zip_sha256=digest(root/'source.zip'),manifest_sha256=digest(source/'CAPSULE_MANIFEST.json'))
 (root/'source-stage.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
'@
docker.exe exec $containerName python3 -c $stage $containerRoot $ManifestSha256 $archivePin.sha256 $capsulePin.sha256 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'source-stage.log')
if($LASTEXITCODE-ne0){throw 'Source/AST staging failed; preserve every file'}
docker.exe cp "${containerName}:${containerRoot}/source-stage.json" (Join-Path $hostRoot 'source-stage.json')
if($LASTEXITCODE-ne0){throw 'Source-stage receipt copy failed'}
Write-Output 'Metadata capsule staged; execution requires separate root admission.'

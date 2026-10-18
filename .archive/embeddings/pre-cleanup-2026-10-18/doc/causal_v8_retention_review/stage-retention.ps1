# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Full owned V7 retention stager copy, adapted for eleven V8 metadata roles. AST only.
param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AdmittedManifestSha256)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$containerRoot='/opt/cuwacunu/research/causal-v8-retention-review/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-v8-retention-review/attempt-001'
function Bounded([string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains('\')-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded repository path required'}
 $path=[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))
 if(-not$path.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Escaped repository path'}
 $current=$path
 while($current-ne$researchRoot){$item=Get-Item -LiteralPath $current;if($item.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No linked capsule input'};$current=[IO.Path]::GetDirectoryName($current)}
 $path
}
function Pin([string]$path){$item=Get-Item -LiteralPath $path;if($item.PSIsContainer){throw 'File required'};[ordered]@{bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$expected){$actual=Pin $path;if($actual.bytes-ne$expected.bytes-or$actual.sha256-ne$expected.sha256){throw "Frozen metadata changed: $path"}}
$manifestRelative='doc/causal_v8_retention_review/admitted-manifest.json'
$manifestPath=Bounded $manifestRelative
if((Pin $manifestPath).sha256-ne$AdmittedManifestSha256){throw 'Exact root-admitted manifest hash required'}
$declaration=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json
if($declaration.kind-ne'clean_causal_v8_retention_derivation_manifest_v1'-or$declaration.authorized_for_derivation-isnot[bool]-or-not$declaration.authorized_for_derivation){throw 'Explicit admitted V8 manifest required'}
if($declaration.container_id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$declaration.container_scope-ne'/opt/cuwacunu/research/causal-v8-retention-review'){throw 'Exact existing-container declaration required'}
$prospectiveRelative='doc/causal_v8_retention_review/derivation-manifest.json'
$prospectivePath=Bounded $prospectiveRelative
if($declaration.root_review.parent_manifest_sha256-ne(Pin $prospectivePath).sha256){throw 'Preserve exact pre-review manifest'}
$roles=@('parent','v3','v4','v5','v6','v7','v8','ledger','establishment','complete_mirror','preservation')
if($declaration.inputs.Count-ne11-or(Compare-Object ($roles|Sort-Object) ($declaration.inputs.id|Sort-Object))){throw 'Exactly eleven metadata roles required'}
$sourceNames=@('derive.py','independent_review.py','README.md','provenance.json')
if($declaration.owned_sources.Count-ne4-or(Compare-Object ($sourceNames|Sort-Object) (@($declaration.owned_sources|ForEach-Object{[IO.Path]::GetFileName($_.path)})|Sort-Object))){throw 'Exactly four owned sources required'}
$entries=@()
foreach($row in $declaration.owned_sources){$name=[IO.Path]::GetFileName($row.path);if($row.path-ne('code/research/causal_v8_retention_review/'+$name)){throw 'Owned source location differs'};$path=Bounded $row.path;Require-Pin $path $row;$entries +=[ordered]@{file=$name;repository_path=$row.path;bytes=$row.bytes;sha256=$row.sha256}}
if($declaration.source.sha256-ne'e65061d256ada129e2dbbbdcbc699c0c7629d92fa9299bae19754ae9966373c0'-or$declaration.independent_reviewer.sha256-ne'63e0cd4bffc842e2d6b882f4690a8895e22bcf283506014724b7042666a9c9f3'){throw 'Root-reviewed derive/review bytes required'}
foreach($row in $declaration.orchestration_helpers){Require-Pin (Bounded $row.path) $row}
if($declaration.orchestration_helpers.Count-ne4-or(Compare-Object @('prepare-manifest.ps1','stage-retention.ps1','execute-retention.py','mirror-retention.ps1') @($declaration.orchestration_helpers|ForEach-Object{[IO.Path]::GetFileName($_.path)}))){throw 'Exactly four owned orchestration helpers'}
foreach($row in @($declaration.manifest_schema,$declaration.orchestration_provenance)){Require-Pin (Bounded $row.path) $row}
if($declaration.execution_wrapper.path-ne'doc/causal_v8_retention_review/execute-retention.py'){throw 'Owned execution recorder required'}
$wrapper=$declaration.execution_wrapper;Require-Pin (Bounded $wrapper.path) $wrapper
$entries +=[ordered]@{file='execute.py';repository_path=$wrapper.path;bytes=$wrapper.bytes;sha256=$wrapper.sha256}
foreach($row in $declaration.inputs){if($row.staged_path-ne("inputs/$($row.id).json")){throw 'Exact role staging path required'};Require-Pin (Bounded $row.path) $row;$entries +=[ordered]@{file=$row.staged_path;repository_path=$row.path;bytes=$row.bytes;sha256=$row.sha256}}
foreach($row in @([ordered]@{file='manifest.json';path=$prospectiveRelative},[ordered]@{file='admitted-manifest.json';path=$manifestRelative},[ordered]@{file='LICENSE';path='LICENSE'})){$p=Pin (Bounded $row.path);$entries +=[ordered]@{file=$row.file;repository_path=$row.path;bytes=$p.bytes;sha256=$p.sha256}}
if($entries.Count-ne19-or@($entries.file|Sort-Object -Unique).Count-ne19){throw 'Exactly19 unique capsule files required'}
if(Test-Path -LiteralPath $hostRoot){throw 'Preserve prior retention attempt'}
$inspection=docker.exe inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$inspection.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$inspection.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$inspection.State.Running){throw 'Exact existing running managed container required'}
[IO.Directory]::CreateDirectory($hostRoot)|Out-Null
$capsule=[ordered]@{kind='clean_causal_v8_retention_source_capsule_v1';scope='Metadata/AST only until separate root execution admission';container_root=$containerRoot;admitted_manifest_sha256=$AdmittedManifestSha256;files=$entries;model_calls=0;head_fits=0;training_updates=0}
$capsulePath=Join-Path $hostRoot 'source-manifest.json'
[IO.File]::WriteAllText($capsulePath,(($capsule|ConvertTo-Json -Depth 12).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $hostRoot 'source.zip';$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($row in $entries){$path=Bounded $row.repository_path;Require-Pin $path $row;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$row.file,[IO.Compression.CompressionLevel]::Fastest)|Out-Null};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$capsulePath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$zip.Dispose();$stream.Dispose()}
$zipPin=Pin $zipPath
$capsulePin=Pin $capsulePath
$setup=@'
from pathlib import Path
import sys
p=Path(sys.argv[1]);assert str(p)=='/opt/cuwacunu/research/causal-v8-retention-review/attempt-001' and not p.exists()
p.mkdir(parents=True)
'@
docker.exe exec cuwacunu python3 -c $setup $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'setup.log')
if($LASTEXITCODE-ne0){throw 'Exclusive container root refused; preserve capsule'}
docker.exe cp $zipPath "cuwacunu:${containerRoot}/source.zip"
if($LASTEXITCODE-ne0){throw 'Copy failed; preserve all partial state'}
$stage=@'
import ast,hashlib,json,pathlib,stat,sys,zipfile
root=pathlib.Path(sys.argv[1]);assert str(root)=='/opt/cuwacunu/research/causal-v8-retention-review/attempt-001'
expected_admitted,expected_zip,expected_capsule=sys.argv[2:5]
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(root/'source.zip')==expected_zip
source=root/'source';source.mkdir()
with zipfile.ZipFile(root/'source.zip') as archive:
 capsule_bytes=archive.read('CAPSULE_MANIFEST.json')
 assert hashlib.sha256(capsule_bytes).hexdigest()==expected_capsule
 capsule=json.loads(capsule_bytes);assert capsule['container_root']==str(root)
 assert capsule['admitted_manifest_sha256']==expected_admitted
 assert len(capsule['files'])==19 and len(archive.namelist())==len(set(archive.namelist()))==20
 assert set(archive.namelist())=={row['file'] for row in capsule['files']}|{'CAPSULE_MANIFEST.json'}
 for info in archive.infolist():assert not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16)
 for row in capsule['files']:
  relative=pathlib.PurePosixPath(row['file']);assert not relative.is_absolute() and '..' not in relative.parts and chr(92) not in str(relative)
  data=archive.read(row['file']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  path=source/relative;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
 (source/'CAPSULE_MANIFEST.json').write_bytes(capsule_bytes)
assert digest(source/'admitted-manifest.json')==capsule['admitted_manifest_sha256']==expected_admitted
for name in ('derive.py','independent_review.py','execute.py'):ast.parse((source/name).read_text(encoding='utf-8-sig'),filename=name)
for path in source.rglob('*.json'):json.loads(path.read_text(encoding='utf-8-sig'))
receipt=dict(kind='clean_causal_v8_retention_source_stage_v1',status='passed_AST_only',files=19,metadata_roles=11,AST_files=3,module_executions=0,model_calls=0,head_fits=0,training_updates=0,source_root=str(source),admitted_manifest_sha256=expected_admitted,source_zip_sha256=digest(root/'source.zip'),manifest_sha256=digest(source/'CAPSULE_MANIFEST.json'))
with (root/'source-stage.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt))
'@
docker.exe exec cuwacunu python3 -c $stage $containerRoot $AdmittedManifestSha256 $zipPin.sha256 $capsulePin.sha256 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'source-stage.log')
if($LASTEXITCODE-ne0){throw 'Source/AST staging failed; preserve every file'}
docker.exe cp "cuwacunu:${containerRoot}/source-stage.json" (Join-Path $hostRoot 'source-stage.json')
if($LASTEXITCODE-ne0){throw 'Stage receipt copy failed'}
Write-Output 'Retention capsule staged; no aggregate arithmetic executed.'

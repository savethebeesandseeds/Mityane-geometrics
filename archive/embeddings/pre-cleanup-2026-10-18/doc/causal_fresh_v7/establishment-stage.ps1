# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Stage eight metadata inputs and two stdlib programs; AST only, no module execution.
param()
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerName='cuwacunu'
$containerRoot='/opt/cuwacunu/research/causal-fresh-v7-establishment/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-fresh-v7-establishment/attempt-001'
$manifestPath=Join-Path $researchRoot 'doc/causal_fresh_v7/establishment-input-manifest.json'
$declaration=Get-Content -Raw -LiteralPath $manifestPath|ConvertFrom-Json
if($declaration.kind-ne'cpb_v7_postmeasurement_evidence_join_manifest_v1'-or$declaration.inputs.Count-ne8){throw 'Exact eight-role declaration required'}
$roles=@('comparison','table_rows','quality_audit','training_audit','complete_mirror','card','quality_receipt','pairings')
if(Compare-Object ($roles|Sort-Object) ($declaration.inputs.id|Sort-Object)){throw 'Exact evidence roles required'}
function File-Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No linked source'};[ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Bounded([string]$relative){if([IO.Path]::IsPathRooted($relative)-or$relative-match'(^|/)\.\.(/|$)'-or$relative.Contains(':')){throw 'Bounded repository path required'};$p=[IO.Path]::GetFullPath((Join-Path $researchRoot $relative));if(-not$p.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Path escaped workspace'};$p}
function Require-Pin([string]$path,$expected){$p=File-Pin $path;if($p.bytes-ne$expected.bytes-or$p.sha256-ne$expected.sha256){throw "Frozen join input changed: $path"}}
$entries=@()
foreach($item in @([ordered]@{file='establish.py';binding=$declaration.source},[ordered]@{file='execute.py';binding=$declaration.execution_wrapper})){
 $path=Bounded $item.binding.path;Require-Pin $path $item.binding;$entries +=[ordered]@{file=$item.file;repository_path=$item.binding.path;bytes=$item.binding.bytes;sha256=$item.binding.sha256}
}
if($declaration.source.sha256-ne'c2932f7bd86a5f9a9af33cf8cffb4eaaa3d253539b4f7ddc8b22b9ed9bdb69ab'){throw 'Independently reviewed join source required'}
foreach($item in $declaration.inputs){if($item.staged_path-ne("inputs/$($item.id).json")){throw 'Exact role path required'};$path=Bounded $item.path;Require-Pin $path $item;$entries +=[ordered]@{file=$item.staged_path;repository_path=$item.path;bytes=$item.bytes;sha256=$item.sha256}}
foreach($item in @([ordered]@{file='input-manifest.json';path='doc/causal_fresh_v7/establishment-input-manifest.json'},[ordered]@{file='LICENSE';path='LICENSE'})){$p=File-Pin (Bounded $item.path);$entries +=[ordered]@{file=$item.file;repository_path=$item.path;bytes=$p.bytes;sha256=$p.sha256}}
if($entries.Count-ne12-or@($entries.file|Sort-Object -Unique).Count-ne12){throw 'Exact12 capsule files'}
if(Test-Path -LiteralPath $hostRoot){throw 'Preserve existing establishment attempt'}
$observed=docker inspect $containerName|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$observed.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$observed.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$observed.State.Running){throw 'Exact existing running container required'}
[IO.Directory]::CreateDirectory($hostRoot)|Out-Null
$capsule=[ordered]@{kind='cpb_v7_metadata_establishment_source_capsule_v1';scope='Metadata/AST only until separate root execution admission';container_root=$containerRoot;files=$entries;model_calls=0;head_fits=0;training_updates=0}
$capsulePath=Join-Path $hostRoot 'source-manifest.json'
[IO.File]::WriteAllText($capsulePath,($capsule|ConvertTo-Json -Depth 10)+"`n",[Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $hostRoot 'source.zip';$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($item in $entries){$path=Bounded $item.repository_path;Require-Pin $path $item;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$item.file,[IO.Compression.CompressionLevel]::Fastest)|Out-Null};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$capsulePath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$zip.Dispose();$stream.Dispose()}
$setup=@'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);assert str(p)=='/opt/cuwacunu/research/causal-fresh-v7-establishment/attempt-001' and not p.exists()
p.mkdir(parents=True)
'@
docker exec $containerName python3 -c $setup $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'setup.log')
if($LASTEXITCODE-ne0){throw 'Exclusive container root refused; preserve host capsule'}
docker cp $zipPath "${containerName}:${containerRoot}/source.zip"
if($LASTEXITCODE-ne0){throw 'Capsule copy failed; preserve state'}
$stage=@'
import ast,hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]);assert str(root)=='/opt/cuwacunu/research/causal-fresh-v7-establishment/attempt-001'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=root/'source';source.mkdir()
with zipfile.ZipFile(root/'source.zip') as z:
 m=json.loads(z.read('CAPSULE_MANIFEST.json'));assert m['container_root']==str(root)
 assert len(m['files'])==12 and len(z.namelist())==13 and set(z.namelist())=={e['file'] for e in m['files']}|{'CAPSULE_MANIFEST.json'}
 for e in m['files']:
  relative=pathlib.PurePosixPath(e['file']);assert not relative.is_absolute() and '..' not in relative.parts and chr(92) not in str(relative)
  data=z.read(e['file']);assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
  path=source/relative;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
 (source/'CAPSULE_MANIFEST.json').write_bytes(z.read('CAPSULE_MANIFEST.json'))
 for name in ('establish.py','execute.py'):ast.parse((source/name).read_text(),filename=name)
 for path in source.rglob('*.json'):json.loads(path.read_text(encoding='utf-8-sig'))
 receipt=dict(kind='cpb_v7_metadata_establishment_source_stage_v1',status='passed_AST_only',files=12,metadata_roles=8,AST_files=2,module_executions=0,model_calls=0,head_fits=0,training_updates=0,source_root=str(source),source_zip_sha256=digest(root/'source.zip'),manifest_sha256=digest(source/'CAPSULE_MANIFEST.json'))
 (root/'source-stage.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
'@
docker exec $containerName python3 -c $stage $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'source-stage.log')
if($LASTEXITCODE-ne0){throw 'Source/AST staging failed; preserve every file'}
docker cp "${containerName}:${containerRoot}/source-stage.json" (Join-Path $hostRoot 'source-stage.json')
if($LASTEXITCODE-ne0){throw 'Source-stage receipt copy failed'}
Write-Output 'Metadata capsule staged; execution requires separate root admission.'

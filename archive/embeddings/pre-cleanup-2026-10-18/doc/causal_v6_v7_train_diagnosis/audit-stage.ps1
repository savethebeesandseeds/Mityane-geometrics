# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Owned neutral-audit capsule workflow; hashes and AST only, never audit body.
param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSha256)
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerRoot='/opt/cuwacunu/research/causal-v6-v7-train-diagnosis-audit/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-v6-v7-train-diagnosis-audit/attempt-001'
function Bounded([string]$root,[string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains('\')-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded repository-relative path required'}
 $path=[IO.Path]::GetFullPath((Join-Path $root $relative))
 if(-not$path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Escaped repository path'}
 $current=$path
 while($current-ne$root){$item=Get-Item -LiteralPath $current;if($item.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No linked input'};$current=[IO.Path]::GetDirectoryName($current)}
 $path
}
function Pin([string]$path){$item=Get-Item -LiteralPath $path;if($item.PSIsContainer){throw 'File required'};[ordered]@{bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$expected){$actual=Pin $path;if($actual.bytes-ne$expected.bytes-or$actual.sha256-ne$expected.sha256){throw "Frozen bytes changed: $path"}}
$cardRelative='doc/causal_v6_v7_train_diagnosis/independent-audit-card.json'
$cardPath=Bounded $researchRoot $cardRelative
if((Pin $cardPath).sha256-ne$AuditCardSha256){throw 'Exact root-reviewed independent card hash required'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
if($card.kind-ne'cpb_v6_v7_TRAIN_diagnosis_independent_audit_card_v1'-or$card.auditor_sources.Count-ne5-or$card.container.output_root-ne$containerRoot){throw 'Own independent five-source audit declaration required'}
if($card.main_card.sha256-ne'6067c3f5128991619e72197dbc5b8ea3f44f6e7e4186a56f4962f70c43b03304'){throw 'Exact frozen main card required'}
$expected=@('audit.py','archive_reader.py','component_math.py','README.md','provenance.json')
if(Compare-Object ($expected|Sort-Object) ($card.auditor_sources.file|Sort-Object)){throw 'Exact owned source names required'}
$entries=@()
foreach($row in $card.auditor_sources){if($row.path-ne('code/research/causal_v6_v7_train_diagnosis_audit/'+$row.file)){throw 'Owned source path differs'};Require-Pin (Bounded $researchRoot $row.path) $row;$entries +=[ordered]@{file=$row.file;repository_path=$row.path;bytes=$row.bytes;sha256=$row.sha256}}
if(($card.auditor_sources|Where-Object file -eq'audit.py').sha256-ne'ba69ecbc748a19e98eafdef60d2642a70e5e142b9bd65fe5e888c0621d655cba'-or($card.auditor_sources|Where-Object file -eq'component_math.py').sha256-ne'7a42b5524a41a987f3df1fc9c582b144dd59ccbfec810be29adf88ccb3f7f68f'){throw 'Root-admitted independent scalar source required'}
foreach($row in $card.orchestration_helpers){Require-Pin (Bounded $researchRoot $row.path) $row}
if($card.execution_wrapper.path-ne'doc/causal_v6_v7_train_diagnosis/audit-execute.py'){throw 'Owned recorder required'}
$wrapper=$card.execution_wrapper;Require-Pin (Bounded $researchRoot $wrapper.path) $wrapper
$entries +=[ordered]@{file='execute.py';repository_path=$wrapper.path;bytes=$wrapper.bytes;sha256=$wrapper.sha256}
foreach($row in @([ordered]@{file='MAIN_CARD.json';binding=$card.main_card},[ordered]@{file='diagnosis.json';binding=$card.main_diagnosis})){
 Require-Pin (Bounded $researchRoot $row.binding.path) $row.binding
 $entries +=[ordered]@{file=$row.file;repository_path=$row.binding.path;bytes=$row.binding.bytes;sha256=$row.binding.sha256}
}
foreach($row in @([ordered]@{file='audit-card.json';path=$cardRelative},[ordered]@{file='LICENSE';path='LICENSE'})){$p=Pin (Bounded $researchRoot $row.path);$entries +=[ordered]@{file=$row.file;repository_path=$row.path;bytes=$p.bytes;sha256=$p.sha256}}
if($entries.Count-ne10-or@($entries.file|Sort-Object -Unique).Count-ne10){throw 'Exactly10 unique capsule files required'}
$main=Get-Content -LiteralPath (Bounded $researchRoot $card.main_card.path) -Raw|ConvertFrom-Json
if($main.allowed_inputs.Count-ne49-or$main.preservation_pins.Count-ne$main.preservation_pin_count){throw 'Exact49 main data roles and declared preservation count required'}
$pins=@($main.allowed_inputs)+@($main.preservation_pins)+@($main.pre_execution_preservation)
foreach($row in $pins){
 $root=switch($row.host_repository){'cuwacunu_embedding'{$researchRoot};'cuwacunu_dev'{'C:\Work\cuwacunu\cuwacunu_dev'};default{throw 'Unknown pinned repository'}}
 Require-Pin (Bounded $root $row.host_path) $row
}
if(Test-Path -LiteralPath $hostRoot){throw 'Preserve earlier independent audit attempt'}
$inspection=docker inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$inspection.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$inspection.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$inspection.State.Running){throw 'Exact existing running managed container required'}
[IO.Directory]::CreateDirectory($hostRoot)|Out-Null
$capsule=[ordered]@{kind='cpb_v6_v7_TRAIN_independent_audit_source_capsule_v1';container_root=$containerRoot;audit_card_sha256=$AuditCardSha256;files=$entries;scope='Hash/AST admission only; audit arithmetic requires explicit root signal';model_calls=0;head_fits=0;training_updates=0}
$capsulePath=Join-Path $hostRoot 'source-manifest.json'
[IO.File]::WriteAllText($capsulePath,(($capsule|ConvertTo-Json -Depth 12).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $hostRoot 'source.zip';$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($row in $entries){$path=Bounded $researchRoot $row.repository_path;Require-Pin $path $row;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$row.file,[IO.Compression.CompressionLevel]::Fastest)|Out-Null};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$capsulePath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$zip.Dispose();$stream.Dispose()}
$setup=@'
from pathlib import Path
import sys
p=Path(sys.argv[1]);assert str(p)=='/opt/cuwacunu/research/causal-v6-v7-train-diagnosis-audit/attempt-001' and not p.exists()
p.mkdir(parents=True)
'@
docker exec cuwacunu python3 -c $setup $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'setup.log')
if($LASTEXITCODE-ne0){throw 'Exclusive audit root refused; preserve host capsule'}
docker cp $zipPath "cuwacunu:${containerRoot}/source.zip"
if($LASTEXITCODE-ne0){throw 'Capsule copy failed; preserve partial state'}
$stage=@'
import ast,hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]);assert str(root)=='/opt/cuwacunu/research/causal-v6-v7-train-diagnosis-audit/attempt-001'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=root/'source';source.mkdir()
with zipfile.ZipFile(root/'source.zip') as archive:
 capsule=json.loads(archive.read('CAPSULE_MANIFEST.json'));assert capsule['container_root']==str(root)
 assert len(capsule['files'])==10 and len(archive.namelist())==11
 assert set(archive.namelist())=={row['file'] for row in capsule['files']}|{'CAPSULE_MANIFEST.json'}
 for row in capsule['files']:
  relative=pathlib.PurePosixPath(row['file']);assert not relative.is_absolute() and '..' not in relative.parts and chr(92) not in str(relative)
  data=archive.read(row['file']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  path=source/relative;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
 (source/'CAPSULE_MANIFEST.json').write_bytes(archive.read('CAPSULE_MANIFEST.json'))
for name in ('audit.py','archive_reader.py','component_math.py','execute.py'):ast.parse((source/name).read_text(encoding='utf-8-sig'),filename=name)
for path in source.glob('*.json'):json.loads(path.read_text(encoding='utf-8-sig'))
assert digest(source/'audit-card.json')==capsule['audit_card_sha256']
main=json.loads((source/'MAIN_CARD.json').read_text(encoding='utf-8-sig'))
pins=main['allowed_inputs']+main['preservation_pins']+[main['pre_execution_preservation']]
assert len(main['allowed_inputs'])==49 and len(main['preservation_pins'])==main['preservation_pin_count']
for row in pins:
 path=pathlib.Path(row['container_path']);assert path.is_absolute() and path.is_file() and not path.is_symlink()
 assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
receipt=dict(kind='cpb_v6_v7_TRAIN_independent_audit_source_stage_v1',status='passed_AST_only',files=10,AST_files=4,data_and_preservation_pins=len(pins),module_executions=0,numeric_storage_reads=0,model_calls=0,head_fits=0,training_updates=0,source_root=str(source),source_zip_sha256=digest(root/'source.zip'),manifest_sha256=digest(source/'CAPSULE_MANIFEST.json'))
with (root/'source-stage.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt))
'@
docker exec cuwacunu python3 -c $stage $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'source-stage.log')
if($LASTEXITCODE-ne0){throw 'Hash/AST staging failed; preserve all state'}
docker cp "cuwacunu:${containerRoot}/source-stage.json" (Join-Path $hostRoot 'source-stage.json')
if($LASTEXITCODE-ne0){throw 'Stage receipt copy failed'}
Write-Output 'Independent audit capsule staged AST-only; body awaits explicit root signal.'

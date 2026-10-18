# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Owned neutral-audit capsule workflow; hashes and AST only, never audit body.
param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSha256)
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$containerRoot='/opt/cuwacunu/research/causal-v9-v11-train-diagnosis-audit/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-v9-v11-train-diagnosis-audit/attempt-001'
function Bounded([string]$root,[string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains([char]92)-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded repository-relative path required'}
 $path=[IO.Path]::GetFullPath((Join-Path $root $relative))
 if(-not$path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Escaped repository path'}
 $current=$path
 while($current-ne$root){$item=Get-Item -LiteralPath $current;if($item.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No linked input'};$current=[IO.Path]::GetDirectoryName($current)}
 $path
}
function Pin([string]$path){$item=Get-Item -LiteralPath $path;if($item.PSIsContainer){throw 'File required'};[ordered]@{bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$expected){$actual=Pin $path;if($actual.bytes-ne$expected.bytes-or$actual.sha256-ne$expected.sha256){throw "Frozen bytes changed: $path"}}
$cardRelative='doc/causal_v9_v11_train_diagnosis/independent-audit-card.ADMITTED.json'
$cardPath=Bounded $researchRoot $cardRelative
if((Pin $cardPath).sha256-ne$AuditCardSha256){throw 'Exact root-reviewed independent card hash required'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
if($card.kind-ne'cpb_v9_v11_TRAIN_diagnosis_independent_audit_card_v1'-or$card.execution_admitted-isnot[bool]-or$card.execution_admitted-ne$true-or$card.auditor_sources.Count-ne5-or$card.container.output_root-ne$containerRoot){throw 'Own independent five-source audit declaration required'}
if($card.main_card.sha256-ne$MainCardSha256-or$card.root_supplied_main_card_sha256-ne$MainCardSha256){throw 'Exact external actual main card required'}
$expected=@('audit.py','archive_reader.py','component_math.py','README.md','provenance.json')
if(Compare-Object ($expected|Sort-Object) ($card.auditor_sources.file|Sort-Object)){throw 'Exact owned source names required'}
$entries=@()
foreach($row in $card.auditor_sources){if($row.path-ne('code/research/causal_v9_v11_train_diagnosis_audit/'+$row.file)){throw 'Owned source path differs'};Require-Pin (Bounded $researchRoot $row.path) $row;$entries +=[ordered]@{file=$row.file;repository_path=$row.path;bytes=$row.bytes;sha256=$row.sha256}}
if($card.scientific_source_ready.sha256-ne'fd681ba57560663e62e84f159a66fcdfb54093a4e761d685d562cdd17e96c917'){throw 'Root-reviewed fixed scientific source authority required'}
Require-Pin (Bounded $researchRoot $card.scientific_source_ready.path) $card.scientific_source_ready
$science=Get-Content -Raw -LiteralPath (Bounded $researchRoot $card.scientific_source_ready.path)|ConvertFrom-Json
foreach($row in $card.auditor_sources){$bound=@($science.independent_auditor_owned_sources|Where-Object{$_.path-eq$row.path});if($bound.Count-ne1-or$bound[0].bytes-ne$row.bytes-or$bound[0].sha256-ne$row.sha256){throw 'Actual auditor source differs from reviewed fixed10 science'}}
foreach($row in $card.orchestration_helpers){Require-Pin (Bounded $researchRoot $row.path) $row}
if($card.execution_wrapper.path-ne'doc/causal_v9_v11_train_diagnosis/audit-execute.py'){throw 'Owned recorder required'}
$wrapper=$card.execution_wrapper;Require-Pin (Bounded $researchRoot $wrapper.path) $wrapper
$entries +=[ordered]@{file='execute.py';repository_path=$wrapper.path;bytes=$wrapper.bytes;sha256=$wrapper.sha256}
foreach($row in @([ordered]@{file='MAIN_CARD.json';binding=$card.main_card},[ordered]@{file='diagnosis.json';binding=$card.main_diagnosis})){
 Require-Pin (Bounded $researchRoot $row.binding.path) $row.binding
 $entries +=[ordered]@{file=$row.file;repository_path=$row.binding.path;bytes=$row.binding.bytes;sha256=$row.binding.sha256}
}
foreach($row in @([ordered]@{file='audit-card.json';path=$cardRelative},[ordered]@{file='LICENSE';path='LICENSE'})){$p=Pin (Bounded $researchRoot $row.path);$entries +=[ordered]@{file=$row.file;repository_path=$row.path;bytes=$p.bytes;sha256=$p.sha256}}
$expectedPayload=$card.auditor_sources.Count+5
if($entries.Count-ne$expectedPayload-or@($entries.file|Sort-Object -Unique).Count-ne$expectedPayload){throw 'Exact derived audit capsule membership required'}
$main=Get-Content -LiteralPath (Bounded $researchRoot $card.main_card.path) -Raw|ConvertFrom-Json
if($main.kind-ne'cpb_v9_v11_saved_train_diagnosis_card_v1'-or$main.execution_admitted-isnot[bool]-or$main.execution_admitted-ne$true-or$main.allowed_inputs.Count-ne25-or$main.hash_only_guards.Count-ne34-or$main.context_inputs.Count-ne12){throw 'Actual admitted25/34/12 main card required'}
$preservation=@();if($null-ne$main.preservation_pins){$preservation=@($main.preservation_pins)}
$pins=(@($main.allowed_inputs)+@($main.hash_only_guards)+@($main.context_inputs)+$preservation+@($main.pre_execution_preservation))
$hostFiles=@{};$containerFiles=@{}
foreach($row in $pins){
 $repository=switch($row.host_repository){'cuwacunu_embedding'{$researchRoot};'cuwacunu_dev'{[IO.Path]::GetFullPath((Join-Path $researchRoot '../cuwacunu_dev'))};default{throw 'Unknown pinned repository'}}
 $path=Bounded $repository $row.host_path;Require-Pin $path $row
 $identity=([string]$row.bytes)+':'+$row.sha256;$hostKey=$path.ToLowerInvariant();$containerKey=[string]$row.container_path
 if($hostFiles.ContainsKey($hostKey)-and$hostFiles[$hostKey]-ne$identity){throw 'Host aliases differ'};$hostFiles[$hostKey]=$identity
 if($containerFiles.ContainsKey($containerKey)-and$containerFiles[$containerKey]-ne$identity){throw 'Container aliases differ'};$containerFiles[$containerKey]=$identity
}
$counts=$card.main_input_binding_counts
if($counts.numeric_roles-ne25-or$counts.hash_only_guards-ne34-or$counts.context_roles-ne12-or$counts.pre_execution_roles-ne1-or$counts.preservation_roles-ne$preservation.Count-or$counts.occurrences-ne$pins.Count-or$counts.unique_host_files-ne$hostFiles.Count-or$counts.unique_container_paths-ne$containerFiles.Count){throw 'Actual alias occurrences and distinct physical-file counts changed'}
if(Test-Path -LiteralPath $hostRoot){throw 'Preserve earlier independent audit attempt'}
$inspection=docker.exe inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$inspection.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$inspection.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$inspection.State.Running){throw 'Exact existing running managed container required'}
[IO.Directory]::CreateDirectory($hostRoot)|Out-Null
$capsule=[ordered]@{kind='cpb_v9_v11_TRAIN_independent_audit_source_capsule_v1';container_root=$containerRoot;audit_card_sha256=$AuditCardSha256;files=$entries;scope='Hash/AST admission only; audit arithmetic requires explicit root signal';model_calls=0;head_fits=0;training_updates=0}
$capsulePath=Join-Path $hostRoot 'source-manifest.json'
[IO.File]::WriteAllText($capsulePath,(($capsule|ConvertTo-Json -Depth 12).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $hostRoot 'source.zip';$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($row in $entries){$path=Bounded $researchRoot $row.repository_path;Require-Pin $path $row;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$row.file,[IO.Compression.CompressionLevel]::Fastest)|Out-Null};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$capsulePath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$zip.Dispose();$stream.Dispose()}
$zipPin=Pin $zipPath
$capsulePin=Pin $capsulePath
$setup=@'
from pathlib import Path
import sys
p=Path(sys.argv[1]);assert str(p)=='/opt/cuwacunu/research/causal-v9-v11-train-diagnosis-audit/attempt-001' and not p.exists()
p.mkdir(parents=True)
'@
docker.exe exec cuwacunu python3 -c $setup $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'setup.log')
if($LASTEXITCODE-ne0){throw 'Exclusive audit root refused; preserve host capsule'}
docker.exe cp $zipPath "cuwacunu:${containerRoot}/source.zip"
if($LASTEXITCODE-ne0){throw 'Capsule copy failed; preserve partial state'}
$stage=@'
import ast,hashlib,json,pathlib,stat,sys,zipfile
root=pathlib.Path(sys.argv[1]);assert str(root)=='/opt/cuwacunu/research/causal-v9-v11-train-diagnosis-audit/attempt-001'
expected_card,expected_zip,expected_capsule,expected_main=sys.argv[2:6]
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(root/'source.zip')==expected_zip,'Externally approved source archive bytes before parse'
source=root/'source';source.mkdir()
with zipfile.ZipFile(root/'source.zip') as archive:
 raw=archive.read('CAPSULE_MANIFEST.json');assert hashlib.sha256(raw).hexdigest()==expected_capsule
 capsule=json.loads(raw);assert capsule['container_root']==str(root) and capsule['audit_card_sha256']==expected_card
 assert capsule['kind']=='cpb_v9_v11_TRAIN_independent_audit_source_capsule_v1'
 payload_count=len(capsule['files']);assert payload_count>=5 and len(archive.namelist())==len(set(archive.namelist()))==payload_count+1
 assert set(archive.namelist())=={row['file'] for row in capsule['files']}|{'CAPSULE_MANIFEST.json'}
 for info in archive.infolist():assert not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16)
 for row in capsule['files']:
  relative=pathlib.PurePosixPath(row['file']);assert not relative.is_absolute() and '..' not in relative.parts and chr(92) not in str(relative)
  data=archive.read(row['file']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  path=source/relative;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
 (source/'CAPSULE_MANIFEST.json').write_bytes(raw)
assert digest(source/'audit-card.json')==expected_card
assert digest(source/'MAIN_CARD.json')==expected_main
for name in ('audit.py','archive_reader.py','component_math.py','execute.py'):ast.parse((source/name).read_text(encoding='utf-8-sig'),filename=name)
for path in source.glob('*.json'):json.loads(path.read_text(encoding='utf-8-sig'))
assert digest(source/'audit-card.json')==capsule['audit_card_sha256']
declaration=json.loads((source/'audit-card.json').read_text(encoding='utf-8-sig'))
assert declaration['main_card']['sha256']==expected_main and declaration['root_supplied_main_card_sha256']==expected_main
assert declaration['execution_wrapper']['sha256']==digest(source/'execute.py')
assert declaration['main_diagnosis']['sha256']==digest(source/'diagnosis.json')
main=json.loads((source/'MAIN_CARD.json').read_text(encoding='utf-8-sig'))
assert declaration['kind']=='cpb_v9_v11_TRAIN_diagnosis_independent_audit_card_v1' and declaration['execution_admitted'] is True
assert main['kind']=='cpb_v9_v11_saved_train_diagnosis_card_v1' and main['stage']=='prospective_before_diagnostic_arithmetic' and main['execution_admitted'] is True
assert main['computed_versions']==['v11'] and main['cached_reference_versions']==['v9']
pins=main['allowed_inputs']+main['hash_only_guards']+main['context_inputs']+main.get('preservation_pins',[])+[main['pre_execution_preservation']]
assert len(main['allowed_inputs'])==25 and len(main['hash_only_guards'])==34 and len(main['context_inputs'])==12
unique={}
for row in pins:
 path=pathlib.Path(row.get('container_path') or row.get('path'));assert path.is_absolute() and path.is_file() and not path.is_symlink()
 assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
 key=str(path.resolve());identity=(row['bytes'],row['sha256'])
 assert key not in unique or unique[key]==identity
 unique[key]=identity
counts=declaration['main_input_binding_counts']
assert counts['numeric_roles']==25 and counts['hash_only_guards']==34 and counts['context_roles']==12
assert counts['occurrences']==len(pins) and counts['preservation_roles']==len(main.get('preservation_pins',[])) and counts['pre_execution_roles']==1
assert counts['unique_container_paths']==len(unique)
assert payload_count==len(declaration['auditor_sources'])+5 and len(declaration['auditor_sources'])==5
receipt=dict(kind='cpb_v9_v11_TRAIN_independent_audit_source_stage_v1',status='passed_AST_only',files=payload_count,AST_files=4,data_guard_occurrences=len(pins),unique_container_physical_files=len(unique),main_input_binding_counts=counts,module_executions=0,numeric_storage_reads=0,model_calls=0,head_fits=0,training_updates=0,source_root=str(source),source_zip_sha256=digest(root/'source.zip'),manifest_sha256=digest(source/'CAPSULE_MANIFEST.json'),audit_card_sha256=expected_card,main_card_sha256=expected_main,main_diagnosis_sha256=digest(source/'diagnosis.json'),execution_wrapper_sha256=digest(source/'execute.py'))
with (root/'source-stage.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt))
'@
docker.exe exec cuwacunu python3 -c $stage $containerRoot $AuditCardSha256 $zipPin.sha256 $capsulePin.sha256 $MainCardSha256 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'source-stage.log')
if($LASTEXITCODE-ne0){throw 'Hash/AST staging failed; preserve all state'}
docker.exe cp "cuwacunu:${containerRoot}/source-stage.json" (Join-Path $hostRoot 'source-stage.json')
if($LASTEXITCODE-ne0){throw 'Stage receipt copy failed'}
Write-Output 'Independent audit capsule staged AST-only; body awaits explicit root signal.'

# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
param([ValidateSet('training')][string]$Kind='training',[ValidateSet('001')][string]$Attempt='001',
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSHA256,[Parameter(Mandatory=$true)][switch]$StageAdmitted)
$ErrorActionPreference='Stop'
if(-not $StageAdmitted){throw 'AST-only Stage requires separate explicit admission'}
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerName='cuwacunu'
$containerId='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$containerImage='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'
function File-Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No source links'};[ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$expected){$actual=File-Pin $path;if($actual.bytes-ne$expected.bytes-or$actual.sha256-ne$expected.sha256){throw "Frozen source changed: $path"}}
$sourceRel="code/research/causal_fresh_v11_${Kind}_audit"
$cardRel='doc/causal_fresh_v11_training_audit/audit-card.json'
$cardPath=Join-Path $researchRoot $cardRel
if((File-Pin $cardPath).sha256-ne$AuditCardSHA256){throw 'Externally reviewed audit card required before parse'}
$card=Get-Content -Raw -LiteralPath $cardPath|ConvertFrom-Json
if($card.stage-ne'declared_before_independent_saved_data_execution'-or-not$card.execution_pending-or$card.model-ne'CPB-v11.market-01'){throw 'Independent static review/frozen declaration required'}
Require-Pin $PSCommandPath $card.stage_helper
$sourceFiles=@('audit_training.py','README.md','provenance.json')
$declared=@($card.auditor_source)+@($card.auditor_support)
$declaredNames=@($declared|ForEach-Object{[IO.Path]::GetFileName($_.path)})
if($declared.Count-ne3-or@($declaredNames|Select-Object -Unique).Count-ne3-or(Compare-Object $sourceFiles $declaredNames)){throw 'Exactly three prospective training auditor files required'}
foreach($entry in $declared){$name=if($entry.path){[IO.Path]::GetFileName($entry.path)}else{$entry.file};Require-Pin (Join-Path $researchRoot "$sourceRel/$name") $entry}
$wrapperPath=Join-Path $researchRoot 'doc/causal_fresh_v11_training_audit/execute.py'
Require-Pin $wrapperPath $card.execution_wrapper
$hostRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v11-$Kind-audit/attempt-$Attempt"
if(Test-Path -LiteralPath $hostRoot){throw 'Preserve existing audit attempt'}
$containerRoot="/opt/cuwacunu/research/causal-fresh-v11-$Kind-audit/attempt-$Attempt"
if($containerRoot-ne$card.expected_audit_root){throw 'Prospectively declared audit leaf required'}
$observed=docker.exe inspect $containerName|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$observed.Id-ne$containerId-or$observed.Image-ne$containerImage-or-not$observed.State.Running){throw 'Exact existing running managed container required'}
[IO.Directory]::CreateDirectory($hostRoot)|Out-Null
$entries=@()
foreach($name in $sourceFiles){$path=Join-Path $researchRoot "$sourceRel/$name";$p=File-Pin $path;$entries +=[ordered]@{file=$name;repository_path="$sourceRel/$name";bytes=$p.bytes;sha256=$p.sha256}}
foreach($item in @([ordered]@{file='audit-card.json';path=$cardPath},[ordered]@{file='LICENSE';path=(Join-Path $researchRoot 'LICENSE')})){$p=File-Pin $item.path;$entries +=[ordered]@{file=$item.file;repository_path=$item.path;bytes=$p.bytes;sha256=$p.sha256}}
$wrapperPin=File-Pin $wrapperPath;$entries +=[ordered]@{file='execute-audit.py';repository_path=$wrapperPath;bytes=$wrapperPin.bytes;sha256=$wrapperPin.sha256}
$manifest=[ordered]@{kind="cpb_v11_${Kind}_independent_audit_source_manifest_v1";stage='AST_only_no_numeric';container_id=$containerId;container_image=$containerImage;container_root=$containerRoot;audit_card_sha256=$AuditCardSHA256;files=$entries}
$manifestPath=Join-Path $hostRoot 'source-manifest.json'
[IO.File]::WriteAllText($manifestPath,($manifest|ConvertTo-Json -Depth 10)+"`n",[Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $hostRoot 'source.zip'
$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($entry in $entries){$path=if($entry.file-eq'audit-card.json'){$cardPath}elseif($entry.file-eq'LICENSE'){Join-Path $researchRoot 'LICENSE'}elseif($entry.file-eq'execute-audit.py'){$wrapperPath}else{Join-Path $researchRoot "$sourceRel/$($entry.file)"};Require-Pin $path $entry;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$path,$entry.file,[IO.Compression.CompressionLevel]::Fastest)|Out-Null};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$manifestPath,'SOURCE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$archive.Dispose();$stream.Dispose()}
$manifestPin=File-Pin $manifestPath;$zipPin=File-Pin $zipPath
$setup=@'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);assert str(p).startswith('/opt/cuwacunu/research/causal-fresh-v11-') and not p.exists()
p.mkdir(parents=True)
'@
docker.exe exec $containerName python3 -c $setup $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'setup.log')
if($LASTEXITCODE-ne0){throw 'Exclusive container audit leaf refused; preserve host capsule'}
docker.exe cp $zipPath "${containerName}:${containerRoot}/source.zip"
if($LASTEXITCODE-ne0){throw 'Audit capsule copy failed; preserve attempt'}
$stage=@'
import ast,datetime,hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]);kind,expected_zip,expected_manifest,expected_card=sys.argv[2:]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(root/'source.zip')==expected_zip,'External ZIP hash before ZIP open'
with zipfile.ZipFile(root/'source.zip') as z:
 raw_manifest=z.read('SOURCE_MANIFEST.json')
 assert hashlib.sha256(raw_manifest).hexdigest()==expected_manifest,'External raw manifest hash before JSON parse'
 m=json.loads(raw_manifest);names=[e['file'] for e in m['files']]
 assert len(names)==len(set(names))==6 and len(z.namelist())==7
 assert set(names)=={'audit_training.py','README.md','provenance.json','audit-card.json','LICENSE','execute-audit.py'}
 assert set(z.namelist())==set(names+['SOURCE_MANIFEST.json'])
 assert m['container_root']==str(root) and m['stage']=='AST_only_no_numeric' and m['audit_card_sha256']==expected_card
 assert hashlib.sha256(z.read('audit-card.json')).hexdigest()==expected_card,'External card before extraction/AST'
 source=root/'source';source.mkdir()
 for e in m['files']:
  assert pathlib.PurePosixPath(e['file']).name==e['file']
  data=z.read(e['file']);assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
  (source/e['file']).write_bytes(data)
 (source/'SOURCE_MANIFEST.json').write_bytes(raw_manifest)
 python_files=sorted(source.glob('*.py'))
 assert [p.name for p in python_files]==['audit_training.py','execute-audit.py']
 for p in python_files:
  tree=ast.parse(p.read_text(),filename=str(p))
  forbidden={'torch','numpy','subprocess','ctypes','importlib'}
  for node in ast.walk(tree):
   if isinstance(node,(ast.Import,ast.ImportFrom)):
    modules=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [(node.module or '').split('.')[0]]
    assert not(set(modules)&(forbidden-({'subprocess'} if p.name=='execute-audit.py' else set())))
 receipt={'kind':'cpb_v11_'+kind+'_audit_source_stage_v1','status':'passed','stage':'AST_only_no_numeric','files':len(names),'python_AST_files':len(python_files),'source_root':str(source),'source_manifest_sha256':digest(source/'SOURCE_MANIFEST.json'),'source_zip_sha256':digest(root/'source.zip'),'audit_card_sha256':expected_card,'numeric_archive_reads':0,'model_calls':0,'head_fits':0,'training_updates':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 with (root/'source-stage.json').open('x') as output:output.write(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
'@
docker.exe exec $containerName python3 -c $stage $containerRoot $Kind $zipPin.sha256 $manifestPin.sha256 $AuditCardSHA256 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'source-stage.log')
if($LASTEXITCODE-ne0){throw 'Source/AST admission failed; preserve every attempt'}
docker.exe cp "${containerName}:${containerRoot}/source-stage.json" (Join-Path $hostRoot 'source-stage.json')
if($LASTEXITCODE-ne0){throw 'Source-stage receipt mirror failed'}
Write-Output "Staged AST-only $Kind audit at $containerRoot; numerical execution remains separately gated."

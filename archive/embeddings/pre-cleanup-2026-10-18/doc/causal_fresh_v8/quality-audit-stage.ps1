# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
param([ValidateSet('quality')][string]$Kind='quality',[ValidatePattern('^[0-9]{3}$')][string]$Attempt='001')
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerName='cuwacunu'
$containerId='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$containerImage='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'
function File-Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No source links'};[ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$expected){$actual=File-Pin $path;if($actual.bytes-ne$expected.bytes-or$actual.sha256-ne$expected.sha256){throw "Frozen source changed: $path"}}
$sourceRel="code/research/causal_fresh_v8_${Kind}_audit"
$cardRel="doc/causal_fresh_v8/$Kind-audit-card.json"
$cardPath=Join-Path $researchRoot $cardRel
$card=Get-Content -Raw -LiteralPath $cardPath|ConvertFrom-Json
if($card.stage-ne'declared_after_static_review_before_any_saved_numeric_audit'-or-not$card.execution_pending-or$card.model-ne'CPB-v8.market-01'){throw 'Independent static review/frozen declaration required'}
Require-Pin $PSCommandPath $card.stage_helper
$sourceFiles=if($Kind-eq'training'){@('audit_training.py','README.md','provenance.json')}else{@('audit.py','archive_reader.py','README.md','provenance.json')}
$declared=if($Kind-eq'training'){@($card.auditor_source)+@($card.auditor_support)}else{@($card.auditor_sources)}
foreach($entry in $declared){$name=if($entry.path){[IO.Path]::GetFileName($entry.path)}else{$entry.file};Require-Pin (Join-Path $researchRoot "$sourceRel/$name") $entry}
$wrapperPath=Join-Path $researchRoot "doc/causal_fresh_v8/$Kind-audit-execute.py"
Require-Pin $wrapperPath $card.execution_wrapper
$hostRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v8-$Kind-audit/attempt-$Attempt"
if(Test-Path -LiteralPath $hostRoot){throw 'Preserve existing audit attempt'}
$containerRoot="/opt/cuwacunu/research/causal-fresh-v8-$Kind-audit/attempt-$Attempt"
$observed=docker.exe inspect $containerName|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$observed.Id-ne$containerId-or$observed.Image-ne$containerImage-or-not$observed.State.Running){throw 'Exact existing running managed container required'}
[IO.Directory]::CreateDirectory($hostRoot)|Out-Null
$entries=@()
foreach($name in $sourceFiles){$path=Join-Path $researchRoot "$sourceRel/$name";$p=File-Pin $path;$entries +=[ordered]@{file=$name;repository_path="$sourceRel/$name";bytes=$p.bytes;sha256=$p.sha256}}
foreach($item in @([ordered]@{file='audit-card.json';path=$cardPath},[ordered]@{file='LICENSE';path=(Join-Path $researchRoot 'LICENSE')})){$p=File-Pin $item.path;$entries +=[ordered]@{file=$item.file;repository_path=$item.path;bytes=$p.bytes;sha256=$p.sha256}}
$wrapperPin=File-Pin $wrapperPath;$entries +=[ordered]@{file='execute-audit.py';repository_path=$wrapperPath;bytes=$wrapperPin.bytes;sha256=$wrapperPin.sha256}
$manifest=[ordered]@{kind="cpb_v8_${Kind}_independent_audit_source_manifest_v1";stage='AST_only_no_numeric';container_id=$containerId;container_image=$containerImage;container_root=$containerRoot;files=$entries}
$manifestPath=Join-Path $hostRoot 'source-manifest.json'
[IO.File]::WriteAllText($manifestPath,($manifest|ConvertTo-Json -Depth 10)+"`n",[Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $hostRoot 'source.zip'
$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($entry in $entries){$path=if($entry.file-eq'audit-card.json'){$cardPath}elseif($entry.file-eq'LICENSE'){Join-Path $researchRoot 'LICENSE'}elseif($entry.file-eq'execute-audit.py'){$wrapperPath}else{Join-Path $researchRoot "$sourceRel/$($entry.file)"};Require-Pin $path $entry;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$path,$entry.file,[IO.Compression.CompressionLevel]::Fastest)|Out-Null};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$manifestPath,'SOURCE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$archive.Dispose();$stream.Dispose()}
$setup=@'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);assert str(p).startswith('/opt/cuwacunu/research/causal-fresh-v8-') and not p.exists()
p.mkdir(parents=True)
'@
docker.exe exec $containerName python3 -c $setup $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'setup.log')
if($LASTEXITCODE-ne0){throw 'Exclusive container audit leaf refused; preserve host capsule'}
docker.exe cp $zipPath "${containerName}:${containerRoot}/source.zip"
if($LASTEXITCODE-ne0){throw 'Audit capsule copy failed; preserve attempt'}
$stage=@'
import ast,datetime,hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]);kind=sys.argv[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(root/'source.zip') as z:
 m=json.loads(z.read('SOURCE_MANIFEST.json'));names=[e['file'] for e in m['files']]
 assert len(names)==len(set(names)) and set(z.namelist())==set(names+['SOURCE_MANIFEST.json'])
 assert m['container_root']==str(root) and m['stage']=='AST_only_no_numeric'
 source=root/'source';source.mkdir()
 for e in m['files']:
  assert pathlib.PurePosixPath(e['file']).name==e['file']
  data=z.read(e['file']);assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
  (source/e['file']).write_bytes(data)
 (source/'SOURCE_MANIFEST.json').write_bytes(z.read('SOURCE_MANIFEST.json'))
 python_files=sorted(source.glob('*.py'))
 for p in python_files:
  tree=ast.parse(p.read_text(),filename=str(p))
  forbidden={'torch','numpy','subprocess','ctypes','importlib'}
  for node in ast.walk(tree):
   if isinstance(node,(ast.Import,ast.ImportFrom)):
    modules=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [(node.module or '').split('.')[0]]
    assert not(set(modules)&(forbidden-({'subprocess'} if p.name=='execute-audit.py' else set())))
 receipt={'kind':'cpb_v8_'+kind+'_audit_source_stage_v1','status':'passed','stage':'AST_only_no_numeric','files':len(names),'python_AST_files':len(python_files),'source_root':str(source),'source_manifest_sha256':digest(source/'SOURCE_MANIFEST.json'),'source_zip_sha256':digest(root/'source.zip'),'numeric_archive_reads':0,'model_calls':0,'head_fits':0,'training_updates':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (root/'source-stage.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
'@
docker.exe exec $containerName python3 -c $stage $containerRoot $Kind 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'source-stage.log')
if($LASTEXITCODE-ne0){throw 'Source/AST admission failed; preserve every attempt'}
docker.exe cp "${containerName}:${containerRoot}/source-stage.json" (Join-Path $hostRoot 'source-stage.json')
if($LASTEXITCODE-ne0){throw 'Source-stage receipt mirror failed'}
Write-Output "Staged AST-only $Kind audit at $containerRoot; numerical execution remains separately gated."

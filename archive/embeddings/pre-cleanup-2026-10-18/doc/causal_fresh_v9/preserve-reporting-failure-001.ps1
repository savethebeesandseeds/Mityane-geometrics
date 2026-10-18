# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Byte preservation only; no model, export, fit, report or numerical audit.
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$receiptPath=Join-Path $PSScriptRoot 'failure-mirror-before-report-recovery-001.json'
if(Test-Path -LiteralPath $receiptPath){throw 'Preserve the existing failure mirror'}
$state=docker.exe inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$state.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$state.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$state.State.Running){throw 'Exact existing container required'}
$inventoryCode=@'
import hashlib,json
from pathlib import Path
root=Path('/opt/cuwacunu/research/causal-fresh-v9/attempt-001')
source=root/'source'
raw=(source/'CAPSULE_MANIFEST.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='43a44dbba0f7b885258766ab5741c9e806cc37ff0817237954b31d09de940750'
manifest=json.loads(raw.decode('utf-8-sig'))
assert manifest['card_sha256']=='94488bf2d42751b5a38c054eb5b6bbd7674e45d6e0bd444818b7f6fbcb34346d'
assert len(manifest['files'])==118
for entry in manifest['files']:
 p=source/entry['path'];data=p.read_bytes()
 assert not p.is_symlink() and len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
def read(p):return json.loads(p.read_text())
quality=root/'quality-001'
failed=read(root/'quality-stage.json')
assert failed['status']=='failed' and failed['exit_code']==1
assert read(quality/'report-stage.json')['exit_code']==1
assert not (quality/'comparison-stage.json').exists()
stages={name:read(quality/(name+'-stage.json')) for name in
 ('extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads')}
assert all(item['exit_code']==0 for item in stages.values())
assert read(quality/'quality-receipt.json')['status']=='passed'
files=[]
prefix='output/engineering/causal-fresh-v9/attempt-001'
for p in sorted(root.rglob('*')):
 if not p.is_file() or 'source'==p.relative_to(root).parts[0]:continue
 assert not p.is_symlink()
 rel=p.relative_to(root);parts=rel.parts
 if parts[0]=='run-001':dest='output/runs/causal-fresh-v9/run-001/'+Path(*parts[1:]).as_posix()
 elif parts[0]=='quality-001':dest='output/runs/causal-fresh-v9/run-001/quality-001/'+Path(*parts[1:]).as_posix()
 else:dest=prefix+'/'+rel.as_posix()
 data=p.read_bytes()
 files.append({'container_path':str(p),'path':dest,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'kind':'cpb_v9_failed_reporting_exact_inventory_v1','source_files_verified':118,
 'failed_quality_stage':failed,'completed_scientific_stages':stages,'files':files,
 'model_calls':0,'head_fits':0,'training_updates':0}))
'@
$inventoryText=docker.exe exec cuwacunu python3 -c $inventoryCode
if($LASTEXITCODE-ne0){throw 'Failed-workflow byte inventory refused'}
$inventory=$inventoryText|ConvertFrom-Json
# These three immutable completed subtrees have never been mirrored locally.
# Bulk copy only when their destination is wholly absent; otherwise verify below.
foreach($pair in @(
 @{source='initial-proof';destination='output/engineering/causal-fresh-v9/attempt-001/initial-proof'},
 @{source='final-proof';destination='output/engineering/causal-fresh-v9/attempt-001/final-proof'},
 @{source='quality-001';destination='output/runs/causal-fresh-v9/run-001/quality-001'})){
 $destination=Join-Path $repo $pair.destination
 if(-not(Test-Path -LiteralPath $destination)){
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))|Out-Null
  docker.exe cp ('cuwacunu:/opt/cuwacunu/research/causal-fresh-v9/attempt-001/'+$pair.source) $destination
  if($LASTEXITCODE-ne0){throw 'Preserve partial immutable subtree copy'}
 }
}
$total=[int64]0
foreach($pin in $inventory.files){
 $destination=[IO.Path]::GetFullPath((Join-Path $repo $pin.path))
 $allowed=[IO.Path]::GetFullPath((Join-Path $repo 'output'))+[IO.Path]::DirectorySeparatorChar
 if(-not$destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Bounded output path required'}
 if(-not(Test-Path -LiteralPath $destination)){
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))|Out-Null
  docker.exe cp ('cuwacunu:'+$pin.container_path) $destination
  if($LASTEXITCODE-ne0){throw 'Preserve partial failure mirror'}
 }
 $file=Get-Item -LiteralPath $destination
 if(($file.Attributes-band[IO.FileAttributes]::ReparsePoint)-or$file.Length-ne$pin.bytes-or(Get-FileHash -LiteralPath $destination).Hash.ToLowerInvariant()-ne$pin.sha256){throw ('Refuse mismatched prior bytes: '+$pin.path)}
 $total+=$file.Length
}
$pt=@($inventory.files|Where-Object {$_.path.EndsWith('.pt')}|ForEach-Object {$_.path})
$ignored=@()
for($start=0;$start-lt$pt.Count;$start+=128){
 $last=[Math]::Min($start+127,$pt.Count-1)
 $ignored+=@(& git -c ('safe.directory='+$repo.Replace([char]92,[char]47)) -C $repo check-ignore -- @($pt[$start..$last]))
 if($LASTEXITCODE-ne0){throw 'Ignored archive inspection failed'}
}
if($ignored.Count-ne$pt.Count-or(Compare-Object @($ignored|Sort-Object -Unique) @($pt|Sort-Object -Unique))){throw 'Every preserved PT must remain ignored'}
$receipt=[ordered]@{kind='cpb_v9_failure_mirror_before_report_recovery_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');files=$inventory.files;file_count=$inventory.files.Count;bytes=$total;source_files_verified=118;ignored_pt_count=$pt.Count;all_mirrored_pt_ignored=$true;failed_quality_stage=$inventory.failed_quality_stage;completed_scientific_stages=$inventory.completed_scientific_stages;failure_preserved=$true;container_files_preserved=$true;reporting_recovery_not_executed=$true;model_calls=0;head_fits=0;training_updates=0;helper=[ordered]@{path='doc/causal_fresh_v9/preserve-reporting-failure-001.ps1';bytes=(Get-Item -LiteralPath $PSCommandPath).Length;sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()}}
[IO.File]::WriteAllText($receiptPath,($receipt|ConvertTo-Json -Depth 20)+[char]10,[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';files=$receipt.file_count;bytes=$total;ignored_pt=$pt.Count;receipt_bytes=(Get-Item -LiteralPath $receiptPath).Length;receipt_sha256=(Get-FileHash -LiteralPath $receiptPath).Hash.ToLowerInvariant()}|ConvertTo-Json

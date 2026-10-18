# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Exact source/card/result/log copies only; no audit or model execution.
param()
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerRoot='/opt/cuwacunu/research/causal-v6-v7-train-diagnosis-audit/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-v6-v7-train-diagnosis-audit/attempt-001'
$mirror=Join-Path $hostRoot 'container-mirror'
if(-not(Test-Path -LiteralPath $hostRoot)-or(Test-Path -LiteralPath $mirror)){throw 'Existing stage and exclusive full mirror required'}
$inspection=docker inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$inspection.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$inspection.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'){throw 'Exact existing managed container required'}
[IO.Directory]::CreateDirectory($mirror)|Out-Null
docker cp "cuwacunu:${containerRoot}/." $mirror
if($LASTEXITCODE-ne0){throw 'Preserve partial full mirror; copy failed'}
function Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.PSIsContainer-or($f.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Regular nonlinked file required'};[ordered]@{path=$f.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$binding){$actual=Pin $path;if($actual.bytes-ne$binding.bytes-or$actual.sha256-ne$binding.sha256){throw "Exact mirror mismatch: $path"};$actual}
$inventoryPath=Join-Path $mirror 'mirror-inventory.json'
$inventory=Get-Content -LiteralPath $inventoryPath -Raw|ConvertFrom-Json
if($inventory.kind-ne'cpb_v6_v7_TRAIN_independent_audit_inventory_v1'){throw 'Exact independent audit inventory required'}
$files=@();$names=@();$prefix=[IO.Path]::GetFullPath($mirror).TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
foreach($row in $inventory.files){
 $path=[IO.Path]::GetFullPath((Join-Path $mirror $row.relative))
 if([IO.Path]::IsPathRooted($row.relative)-or$row.relative-match'(^|/)\.\.(/|$)'-or-not$path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Unbounded mirrored path'}
 $p=Require-Pin $path $row;$names+=$row.relative;$files +=[ordered]@{container=$row.path;relative=$row.relative;host=$p}
}
$names+='mirror-inventory.json'
$actual=@(Get-ChildItem -LiteralPath $mirror -Recurse -File|ForEach-Object{$_.FullName.Substring($mirror.Length+1).Replace([char]92,[char]47)})
if($actual.Count-ne$names.Count-or(Compare-Object ($actual|Sort-Object) ($names|Sort-Object))){throw 'Missing or unrecorded audit mirror file'}
foreach($row in @([ordered]@{from='source.zip';path=(Join-Path $hostRoot 'source.zip')},[ordered]@{from='source-stage.json';path=(Join-Path $hostRoot 'source-stage.json')},[ordered]@{from='source/audit-card.json';path=(Join-Path $researchRoot 'doc/causal_v6_v7_train_diagnosis/independent-audit-card.json')},[ordered]@{from='source/MAIN_CARD.json';path=(Join-Path $researchRoot 'doc/CAUSAL_V6_V7_TRAIN_DIAGNOSIS_CARD.json')},[ordered]@{from='source/diagnosis.json';path=(Join-Path $researchRoot 'doc/causal_v6_v7_train_diagnosis/diagnosis.json')})){$null=Require-Pin (Join-Path $mirror $row.from) (Pin $row.path)}
$executionPath=Join-Path $mirror 'run-001/execution.json'
$execution=Get-Content -LiteralPath $executionPath -Raw|ConvertFrom-Json
if($execution.status-ne$inventory.status){throw 'Execution/inventory disposition mismatch'}
$receipt=[ordered]@{kind='cpb_v6_v7_TRAIN_independent_audit_mirror_v1';status=$execution.status;container_root=$containerRoot;host_root=$mirror;file_count=$actual.Count;files=$files;inventory=(Pin $inventoryPath);execution=(Pin $executionPath);all_hashes_exact=$true;source_before_after_exact=$true;helper=(Pin $PSCommandPath);model_calls=0;head_fits=0;training_updates=0;deletions=0}
$hostReceipt=Join-Path $hostRoot 'mirror-receipt.json'
if(Test-Path -LiteralPath $hostReceipt){throw 'Preserve earlier audit mirror receipt'}
[IO.File]::WriteAllText($hostReceipt,(($receipt|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
if($execution.status-ne'passed'){throw 'Failed audit remains fully mirrored; do not publish a passed result'}
$raw=Join-Path $mirror 'run-001/audit.json';$null=Require-Pin $raw $execution.result
$durable=Join-Path $researchRoot 'doc/causal_v6_v7_train_diagnosis/independent-audit.json'
$executionCopy=Join-Path $researchRoot 'doc/causal_v6_v7_train_diagnosis/independent-audit-execution.json'
$durableReceipt=Join-Path $researchRoot 'doc/causal_v6_v7_train_diagnosis/independent-audit-mirror.json'
foreach($path in @($durable,$executionCopy,$durableReceipt)){if(Test-Path -LiteralPath $path){throw 'Preserve all earlier durable audit evidence'}}
[IO.File]::Copy($raw,$durable,$false);$receipt['independent_audit']=Require-Pin $durable (Pin $raw)
[IO.File]::Copy($executionPath,$executionCopy,$false);$receipt['durable_execution']=Require-Pin $executionCopy (Pin $executionPath)
[IO.File]::WriteAllText($durableReceipt,(($receipt|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';file_count=$actual.Count;independent_audit=$receipt.independent_audit;execution=$receipt.durable_execution;mirror_receipt=(Pin $durableReceipt)}|ConvertTo-Json -Depth 5

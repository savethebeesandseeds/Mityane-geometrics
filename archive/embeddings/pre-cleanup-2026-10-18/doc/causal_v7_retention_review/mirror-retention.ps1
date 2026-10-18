# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Exact byte mirror only. Future protocol application is a separate root action.
param([ValidateSet('derivation','review')][string]$Phase='review')
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerRoot='/opt/cuwacunu/research/causal-v7-retention-review/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-v7-retention-review/attempt-001'
$mirror=Join-Path $hostRoot ('container-mirror-'+$Phase)
if(-not(Test-Path -LiteralPath $hostRoot)-or(Test-Path -LiteralPath $mirror)){throw 'Existing stage and exclusive phase mirror required'}
$inspection=docker inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$inspection.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$inspection.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'){throw 'Exact existing container required'}
[IO.Directory]::CreateDirectory($mirror)|Out-Null
docker cp "cuwacunu:${containerRoot}/." $mirror
if($LASTEXITCODE-ne0){throw 'Preserve partial phase mirror; copy failed'}
function Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.PSIsContainer-or($f.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Regular nonlinked metadata file required'};[ordered]@{path=$f.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$binding){$actual=Pin $path;if($actual.bytes-ne$binding.bytes-or$actual.sha256-ne$binding.sha256){throw "Byte mismatch: $path"};$actual}
$inventoryName='mirror-inventory-'+$Phase+'.json'
$inventoryPath=Join-Path $mirror $inventoryName
$inventory=Get-Content -LiteralPath $inventoryPath -Raw|ConvertFrom-Json
if($inventory.kind-ne'clean_causal_v7_retention_inventory_v1'-or$inventory.phase-ne$Phase){throw 'Exact phase inventory required'}
$files=@();$names=@();$prefix=[IO.Path]::GetFullPath($mirror).TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
foreach($entry in $inventory.files){
 $path=[IO.Path]::GetFullPath((Join-Path $mirror $entry.relative))
 if([IO.Path]::IsPathRooted($entry.relative)-or$entry.relative-match'(^|/)\.\.(/|$)'-or-not$path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Unbounded mirrored path'}
 $p=Require-Pin $path $entry;$names+=$entry.relative;$files +=[ordered]@{container=$entry.path;relative=$entry.relative;host=$p}
}
$names+=$inventoryName
$actual=@(Get-ChildItem -LiteralPath $mirror -Recurse -File|ForEach-Object{$_.FullName.Substring($mirror.Length+1).Replace([char]92,[char]47)})
if($actual.Count-ne$names.Count-or(Compare-Object ($actual|Sort-Object) ($names|Sort-Object))){throw 'Missing or unrecorded phase mirror file'}
$null=Require-Pin (Join-Path $mirror 'source.zip') (Pin (Join-Path $hostRoot 'source.zip'))
$null=Require-Pin (Join-Path $mirror 'source-stage.json') (Pin (Join-Path $hostRoot 'source-stage.json'))
$null=Require-Pin (Join-Path $mirror 'source/manifest.json') (Pin (Join-Path $researchRoot 'doc/causal_v7_retention_review/derivation-manifest.json'))
$null=Require-Pin (Join-Path $mirror 'source/admitted-manifest.json') (Pin (Join-Path $researchRoot 'doc/causal_v7_retention_review/admitted-manifest.json'))
$executionPath=Join-Path $mirror ($Phase+'-run-001/execution.json')
$execution=Get-Content -LiteralPath $executionPath -Raw|ConvertFrom-Json
if($execution.status-ne$inventory.status){throw 'Execution/inventory disposition mismatch'}
$receipt=[ordered]@{kind='clean_causal_v7_retention_complete_mirror_v1';status=$inventory.status;phase=$Phase;container_root=$containerRoot;host_root=$mirror;file_count=$actual.Count;files=$files;inventory=(Pin $inventoryPath);execution=(Pin $executionPath);all_hashes_exact=$true;helper=(Pin $PSCommandPath);model_calls=0;head_fits=0;training_updates=0;live_protocol_files_written=0;deletions=0}
$hostReceipt=Join-Path $hostRoot ('mirror-'+$Phase+'-receipt.json')
if(Test-Path -LiteralPath $hostReceipt){throw 'Preserve earlier mirror receipt'}
[IO.File]::WriteAllText($hostReceipt,(($receipt|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
if($execution.status-ne'passed'){throw 'Failed arithmetic remains mirrored; do not publish passed receipts'}
if($Phase-eq'review'){
 $copies=@([ordered]@{from='output-001/derivation.json';to='doc/causal_v7_retention_review/derivation.json'},[ordered]@{from='review-run-001/independent-review.json';to='doc/causal_v7_retention_review/independent-review.json'},[ordered]@{from='derivation-run-001/execution.json';to='doc/causal_v7_retention_review/derivation-execution.json'},[ordered]@{from='review-run-001/execution.json';to='doc/causal_v7_retention_review/review-execution.json'})
 $durableReceipt=Join-Path $researchRoot 'doc/causal_v7_retention_review/mirror-receipt.json'
 foreach($row in $copies){if(Test-Path -LiteralPath (Join-Path $researchRoot $row.to)){throw 'Preserve earlier durable evidence'}}
 if(Test-Path -LiteralPath $durableReceipt){throw 'Preserve earlier durable mirror receipt'}
 $durable=@()
 foreach($row in $copies){$source=Join-Path $mirror $row.from;$dest=Join-Path $researchRoot $row.to;[IO.File]::Copy($source,$dest,$false);$durable+=Require-Pin $dest (Pin $source)}
 $receipt.durable_evidence=$durable
 [IO.File]::WriteAllText($durableReceipt,(($receipt|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
}
[ordered]@{status=$execution.status;phase=$Phase;file_count=$actual.Count;mirror_receipt=(Pin $hostReceipt);future_protocol_applied=$false}|ConvertTo-Json -Depth 5

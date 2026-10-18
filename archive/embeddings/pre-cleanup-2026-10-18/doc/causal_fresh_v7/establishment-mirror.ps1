# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Exact metadata byte copies and hashes only; no join/model/audit execution.
param()
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerRoot='/opt/cuwacunu/research/causal-fresh-v7-establishment/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-fresh-v7-establishment/attempt-001'
$mirror=Join-Path $hostRoot 'container-mirror'
if(-not(Test-Path -LiteralPath $hostRoot)-or(Test-Path -LiteralPath $mirror)){throw 'Existing stage and exclusive mirror required'}
$inspection=docker inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$inspection.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$inspection.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'){throw 'Exact existing container required'}
[IO.Directory]::CreateDirectory($mirror)|Out-Null
docker cp "cuwacunu:${containerRoot}/." $mirror
if($LASTEXITCODE-ne0){throw 'Preserve partial mirror; copy failed'}
function Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No links'};[ordered]@{path=$f.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$binding){$p=Pin $path;if($p.bytes-ne$binding.bytes-or$p.sha256-ne$binding.sha256){throw "Bytes changed: $path"};$p}
$inventory=Get-Content -Raw -LiteralPath (Join-Path $mirror 'mirror-inventory.json')|ConvertFrom-Json
if($inventory.kind-ne'cpb_v7_metadata_establishment_inventory_v1'){throw 'Exact inventory required'}
$files=@();$names=@();$prefix=[IO.Path]::GetFullPath($mirror).TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
foreach($entry in $inventory.files){$path=[IO.Path]::GetFullPath((Join-Path $mirror $entry.relative));if([IO.Path]::IsPathRooted($entry.relative)-or-not$path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Unbounded mirror path'};$p=Require-Pin $path $entry;$names+=$entry.relative;$files +=[ordered]@{container=$entry.path;relative=$entry.relative;host=$p}}
$names+='mirror-inventory.json';$actual=@(Get-ChildItem -LiteralPath $mirror -Recurse -File|ForEach-Object{$_.FullName.Substring($mirror.Length+1).Replace([char]92,[char]47)})
if($actual.Count-ne$names.Count-or(Compare-Object ($actual|Sort-Object) ($names|Sort-Object))){throw 'Missing or unrecorded mirror file'}
$null=Require-Pin (Join-Path $mirror 'source.zip') (Pin (Join-Path $hostRoot 'source.zip'))
$null=Require-Pin (Join-Path $mirror 'source-stage.json') (Pin (Join-Path $hostRoot 'source-stage.json'))
$null=Require-Pin (Join-Path $mirror 'source/input-manifest.json') (Pin (Join-Path $researchRoot 'doc/causal_fresh_v7/establishment-input-manifest.json'))
$executionPath=Join-Path $mirror 'run-001/execution.json';$execution=Get-Content -Raw -LiteralPath $executionPath|ConvertFrom-Json
if($execution.status-ne$inventory.status){throw 'Disposition mismatch'}
$raw=Join-Path $mirror 'run-001/final-establishment.json'
$null=Require-Pin $raw $execution.result
$durable=Join-Path $researchRoot 'doc/causal_fresh_v7/final-establishment.json'
$executionCopy=Join-Path $researchRoot 'doc/causal_fresh_v7/establishment-execution.json'
$receiptPath=Join-Path $researchRoot 'doc/causal_fresh_v7/establishment-mirror.json'
foreach($path in @($durable,$executionCopy,$receiptPath)){if(Test-Path -LiteralPath $path){throw 'Preserve prior durable receipt'}}
if($execution.status-ne'passed'){throw 'Failed execution remains mirrored; do not publish passed establishment'}
[IO.File]::Copy($raw,$durable,$false);$durablePin=Require-Pin $durable (Pin $raw)
[IO.File]::Copy($executionPath,$executionCopy,$false);$executionPin=Require-Pin $executionCopy (Pin $executionPath)
$receipt=[ordered]@{kind='cpb_v7_metadata_establishment_mirror_v1';status='passed';join_status=$execution.status;files=$actual.Count;container_root=$containerRoot;host_root=$mirror;mapped_files=$files;inventory=(Pin (Join-Path $mirror 'mirror-inventory.json'));final_establishment=$durablePin;execution=$executionPin;helper=(Pin $PSCommandPath);source_before_after_exact=$true;model_calls=0;head_fits=0;training_updates=0;deletions=0}
[IO.File]::WriteAllText($receiptPath,($receipt|ConvertTo-Json -Depth 12)+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';files=$actual.Count;final_establishment=$durablePin;execution=$executionPin;mirror=(Pin $receiptPath)}|ConvertTo-Json -Depth 5

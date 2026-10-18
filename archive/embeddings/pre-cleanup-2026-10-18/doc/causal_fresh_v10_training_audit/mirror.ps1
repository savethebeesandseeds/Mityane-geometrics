# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Exact file copying/hash inspection only; never executes an auditor or model.
param([ValidateSet('training')][string]$Kind='training',[ValidateSet('001')][string]$Attempt='001',
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$StageReceiptSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ZipSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExecutionSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InventorySHA256)
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerName='cuwacunu'
$containerRoot="/opt/cuwacunu/research/causal-fresh-v10-$Kind-audit/attempt-$Attempt"
$hostRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v10-$Kind-audit/attempt-$Attempt"
$hostMirror=Join-Path $hostRoot 'container-mirror'
function File-Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Reparse point forbidden'};[ordered]@{path=$f.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$entry){$p=File-Pin $path;if($p.bytes-ne$entry.bytes-or$p.sha256-ne$entry.sha256){throw "Mirror bytes differ: $path"};$p}
$cardPath=Join-Path $researchRoot 'doc/causal_fresh_v10_training_audit/audit-card.json'
if((File-Pin $cardPath).sha256-ne$AuditCardSHA256){throw 'External audit card hash before parse required'}
$card=Get-Content -Raw -LiteralPath $cardPath|ConvertFrom-Json
$null=Require-Pin $PSCommandPath $card.mirror_helper
$hostStagePath=Join-Path $hostRoot 'source-stage.json'
if((File-Pin $hostStagePath).sha256-ne$StageReceiptSha256){throw 'External host Stage hash before parse required'}
$stage=Get-Content -Raw -LiteralPath $hostStagePath|ConvertFrom-Json
if($stage.kind-ne'cpb_v10_training_audit_source_stage_v1'-or$stage.status-ne'passed'-or$stage.audit_card_sha256-ne$AuditCardSHA256-or$stage.source_manifest_sha256-ne$ManifestSHA256-or$stage.source_zip_sha256-ne$ZipSHA256){throw 'Exact passed source Stage and external transport required'}
if(-not(Test-Path -LiteralPath $hostRoot)-or(Test-Path -LiteralPath $hostMirror)){throw 'Existing staging leaf and new mirror required; preserve every attempt'}
$observed=docker.exe inspect $containerName|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$observed.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$observed.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'){throw 'Exact existing managed container required'}
[IO.Directory]::CreateDirectory($hostMirror)|Out-Null
docker.exe cp "${containerName}:${containerRoot}/." $hostMirror
if($LASTEXITCODE-ne0){throw 'Audit mirror copy failed; retain partial copy'}
$inventoryPath=Join-Path $hostMirror 'mirror-inventory.json'
if((File-Pin $inventoryPath).sha256-ne$InventorySHA256){throw 'External inventory hash before parse required'}
$inventory=Get-Content -Raw -LiteralPath $inventoryPath|ConvertFrom-Json
if($inventory.kind-ne'cpb_v10_training_audit_complete_file_inventory_v1'){throw 'Exact completed wrapper inventory required'}
$expected=@();$mapped=@();$mirrorPrefix=[IO.Path]::GetFullPath($hostMirror).TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
foreach($entry in $inventory.files){
 if($entry.relative-match'(^|/)\.\.(/|$)'-or[IO.Path]::IsPathRooted($entry.relative)){throw 'Unbounded inventory path'}
 $path=[IO.Path]::GetFullPath((Join-Path $hostMirror $entry.relative))
 if(-not$path.StartsWith($mirrorPrefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Inventory path outside new mirror'}
 $actual=Require-Pin $path $entry
 $expected+=$entry.relative;$mapped +=[ordered]@{container=$entry.path;relative=$entry.relative;host=$actual}
}
$expected+='mirror-inventory.json'
$actualFiles=@(Get-ChildItem -LiteralPath $hostMirror -Recurse -File|ForEach-Object{$_.FullName.Substring($hostMirror.Length+1).Replace([char]92,[char]47)})
if($expected.Count-ne$actualFiles.Count-or(Compare-Object ($expected|Sort-Object) ($actualFiles|Sort-Object))){throw 'Uninventoried or missing mirrored audit file'}
$null=Require-Pin (Join-Path $hostMirror 'source.zip') (File-Pin (Join-Path $hostRoot 'source.zip'))
$null=Require-Pin (Join-Path $hostMirror 'source-stage.json') (File-Pin (Join-Path $hostRoot 'source-stage.json'))
$null=Require-Pin (Join-Path $hostMirror 'source/audit-card.json') (File-Pin (Join-Path $researchRoot 'doc/causal_fresh_v10_training_audit/audit-card.json'))
$manifestPath=Join-Path $hostMirror 'source/SOURCE_MANIFEST.json'
if((File-Pin $manifestPath).sha256-ne$stage.source_manifest_sha256-or(File-Pin (Join-Path $hostRoot 'source-manifest.json')).sha256-ne$stage.source_manifest_sha256-or(File-Pin (Join-Path $hostMirror 'source.zip')).sha256-ne$stage.source_zip_sha256){throw 'Raw source manifest/ZIP must bind passed Stage'}
$manifest=Get-Content -Raw -LiteralPath $manifestPath|ConvertFrom-Json
if($manifest.audit_card_sha256-ne$AuditCardSHA256-or$manifest.container_root-ne$containerRoot){throw 'Exact card/source manifest context required'}
$executionPath=Join-Path $hostMirror 'run-001/execution.json'
if((File-Pin $executionPath).sha256-ne$ExecutionSHA256){throw 'External execution hash before parse required'}
$execution=Get-Content -Raw -LiteralPath $executionPath|ConvertFrom-Json
if($execution.kind-ne'causal_fresh_v10_training_audit_execution_v1'-or$execution.status-ne$inventory.status-or$execution.audit_card.sha256-ne$AuditCardSHA256-or$execution.source_stage.sha256-ne$StageReceiptSha256-or$execution.source_manifest.sha256-ne$stage.source_manifest_sha256){throw 'Execution/inventory/card/Stage disposition mismatch'}
$sourceEquality=$true
if($execution.source_before.Count-ne$manifest.files.Count-or$execution.source_after.Count-ne$manifest.files.Count){$sourceEquality=$false}
foreach($entry in $manifest.files){
 $before=@($execution.source_before|Where-Object { $_.path -eq ($containerRoot+'/source/'+$entry.file) })
 $after=@($execution.source_after|Where-Object { $_.path -eq ($containerRoot+'/source/'+$entry.file) })
 $current=File-Pin (Join-Path $hostMirror ('source/'+$entry.file))
 if($before.Count-ne1-or$after.Count-ne1-or$before[0].bytes-ne$entry.bytes-or$after[0].bytes-ne$entry.bytes-or$before[0].sha256-ne$entry.sha256-or$after[0].sha256-ne$entry.sha256-or$current.bytes-ne$entry.bytes-or$current.sha256-ne$entry.sha256){$sourceEquality=$false}
}
if($execution.status-eq'passed'-and-not$sourceEquality){throw 'Passed durable audit requires actual source-before/after/manifest equality'}
$rawPath=Join-Path $hostMirror 'run-001/raw-result.json'
if(Test-Path -LiteralPath $rawPath){$null=Require-Pin $rawPath $execution.raw_result}
$docRoot=Join-Path $researchRoot 'doc/causal_fresh_v10'
$canonical=if($Kind-eq'training'){'independent-training-audit.json'}else{'quality-independent-audit.json'}
if($execution.status-ne'passed'){$canonical=$canonical.Replace('.json',".failed-attempt$Attempt.json")}
$durable=$null
if(Test-Path -LiteralPath $rawPath){$durablePath=Join-Path $docRoot $canonical;if(Test-Path -LiteralPath $durablePath){throw 'Preserve prior durable audit'};[IO.File]::Copy($rawPath,$durablePath,$false);$durable=Require-Pin $durablePath (File-Pin $rawPath)}
$executionCopy=Join-Path $docRoot "$Kind-audit-execution-attempt$Attempt.json"
if(Test-Path -LiteralPath $executionCopy){throw 'Preserve prior execution receipt'}
[IO.File]::Copy($executionPath,$executionCopy,$false)
$executionPin=Require-Pin $executionCopy (File-Pin $executionPath)
$receipt=[ordered]@{kind="cpb_v10_${Kind}_independent_audit_mirror_v1";status='passed';audit_status=$execution.status;container_root=$containerRoot;host_root=$hostMirror;files=$actualFiles.Count;exact_file_inventory=$true;mapped_files=$mapped;container_inventory=(File-Pin $inventoryPath);execution=$executionPin;durable_raw_audit=$durable;mirror_helper=(File-Pin $PSCommandPath);source_card=(File-Pin $cardPath);source_stage=(File-Pin $hostStagePath);source_before_after_manifest_equal=$sourceEquality;model_calls=0;head_fits=0;training_updates=0;interpretation='Pure byte copy/hash mapping; raw audit and every source/log/stage remain exact; no arithmetic replay'}
$receiptPath=Join-Path $docRoot "$Kind-audit-mirror-attempt$Attempt.json"
if(Test-Path -LiteralPath $receiptPath){throw 'Preserve prior audit mirror receipt'}
[IO.File]::WriteAllText($receiptPath,($receipt|ConvertTo-Json -Depth 12)+"`n",[Text.UTF8Encoding]::new($false))
$summary=[ordered]@{mirror='passed';audit=$execution.status;files=$actualFiles.Count;durable=$durable;execution=$executionPin;receipt=(File-Pin $receiptPath)}
$summary|ConvertTo-Json -Depth 5

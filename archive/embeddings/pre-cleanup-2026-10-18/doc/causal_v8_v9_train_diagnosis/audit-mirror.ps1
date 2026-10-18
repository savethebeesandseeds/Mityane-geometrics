# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Owned complete physical V6/V8 mirror copy, adapted to strict external V8/V9 admission.
# Exact failed/passed source/card/result/log preservation; no audit or model execution.
param(
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$StageReceiptSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExecutionSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InventorySha256
)
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$containerRoot='/opt/cuwacunu/research/causal-v8-v9-train-diagnosis-audit/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-v8-v9-train-diagnosis-audit/attempt-001'
$mirror=Join-Path $hostRoot 'container-mirror'
if(-not(Test-Path -LiteralPath $hostRoot)-or(Test-Path -LiteralPath $mirror)){throw 'Existing stage and exclusive full mirror required'}
$inspection=docker.exe inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$inspection.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$inspection.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$inspection.State.Running){throw 'Exact existing running managed container required'}
[IO.Directory]::CreateDirectory($mirror)|Out-Null
docker.exe cp "cuwacunu:${containerRoot}/." $mirror
if($LASTEXITCODE-ne0){throw 'Preserve partial full mirror; copy failed'}
function Pin([string]$path){
 $f=Get-Item -LiteralPath $path
 if($f.PSIsContainer-or($f.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Regular nonlinked file required'}
 [ordered]@{path=$f.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Require-Pin([string]$path,$binding){$actual=Pin $path;if($actual.bytes-ne$binding.bytes-or$actual.sha256-ne$binding.sha256){throw "Exact mirror mismatch: $path"};$actual}
$stagePath=Join-Path $mirror 'source-stage.json'
$cardPath=Join-Path $mirror 'source/audit-card.json'
$executionPath=Join-Path $mirror 'run-001/execution.json'
$inventoryPath=Join-Path $mirror 'mirror-inventory.json'
if((Pin $stagePath).sha256-ne$StageReceiptSha256-or(Pin $cardPath).sha256-ne$AuditCardSha256-or(Pin $executionPath).sha256-ne$ExecutionSha256-or(Pin $inventoryPath).sha256-ne$InventorySha256){throw 'External exact Stage/card/Execution/Inventory hashes required before any JSON parse'}
$stage=Get-Content -LiteralPath $stagePath -Raw|ConvertFrom-Json
if($stage.kind-ne'cpb_v8_v9_TRAIN_independent_audit_source_stage_v1'-or$stage.status-ne'passed_AST_only'-or$stage.files-ne10-or$stage.AST_files-ne4-or$stage.data_and_preservation_pins-ne93-or$stage.audit_card_sha256-ne$AuditCardSha256){throw 'Exact10-file/AST4/93-pin source admission required'}
$capsulePath=Join-Path $mirror 'source/CAPSULE_MANIFEST.json'
if((Pin $capsulePath).sha256-ne$stage.manifest_sha256-or(Pin (Join-Path $hostRoot 'source-manifest.json')).sha256-ne$stage.manifest_sha256-or(Pin (Join-Path $mirror 'source.zip')).sha256-ne$stage.source_zip_sha256){throw 'Externally bound Stage must pin raw capsule and ZIP before parse'}
$capsule=Get-Content -LiteralPath $capsulePath -Raw|ConvertFrom-Json
if($capsule.kind-ne'cpb_v8_v9_TRAIN_independent_audit_source_capsule_v1'-or$capsule.container_root-ne$containerRoot-or$capsule.audit_card_sha256-ne$AuditCardSha256-or$capsule.files.Count-ne10-or@($capsule.files.file|Sort-Object -Unique).Count-ne10){throw 'Exact independent audit capsule required'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
$execution=Get-Content -LiteralPath $executionPath -Raw|ConvertFrom-Json
$inventory=Get-Content -LiteralPath $inventoryPath -Raw|ConvertFrom-Json
if($card.kind-ne'cpb_v8_v9_TRAIN_diagnosis_independent_audit_card_v1'-or$inventory.kind-ne'cpb_v8_v9_TRAIN_independent_audit_inventory_v1'-or$execution.kind-ne'cpb_v8_v9_TRAIN_independent_audit_execution_v1'-or$execution.status-ne$inventory.status-or$execution.status-notin@('passed','failed')){throw 'Own exact audit disposition required'}
foreach($metadata in @($inventory,$execution)){
 if($metadata.externally_reviewed_audit_card_sha256-ne$AuditCardSha256-or$metadata.externally_reviewed_stage_sha256-ne$StageReceiptSha256){throw 'External execution/inventory admission mismatch'}
}
if($card.main_card.sha256-ne$stage.main_card_sha256-or$card.root_supplied_main_card_sha256-ne$stage.main_card_sha256-or$card.main_diagnosis.sha256-ne$stage.main_diagnosis_sha256-or$card.execution_wrapper.sha256-ne$stage.execution_wrapper_sha256){throw 'Stage must bind actual main card, completed diagnosis and wrapper'}
$files=@();$names=@();$prefix=[IO.Path]::GetFullPath($mirror).TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
foreach($row in $inventory.files){
 $path=[IO.Path]::GetFullPath((Join-Path $mirror $row.relative))
 if([IO.Path]::IsPathRooted($row.relative)-or$row.relative.Contains([char]92)-or$row.relative-match'(^|/)\.\.(/|$)'-or-not$path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)-or$row.path-ne($containerRoot+'/'+$row.relative)){throw 'Unbounded mirrored path'}
 $p=Require-Pin $path $row;$names+=$row.relative;$files +=[ordered]@{container=$row.path;relative=$row.relative;host=$p}
}
$names+='mirror-inventory.json'
$actual=@(Get-ChildItem -LiteralPath $mirror -Recurse -File|ForEach-Object{$_.FullName.Substring($mirror.Length+1).Replace([char]92,[char]47)})
if($actual.Count-ne$names.Count-or@($names|Sort-Object -Unique).Count-ne$names.Count-or(Compare-Object ($actual|Sort-Object) ($names|Sort-Object))){throw 'Missing or unrecorded audit mirror file'}
foreach($row in @([ordered]@{from='source.zip';path=(Join-Path $hostRoot 'source.zip')},[ordered]@{from='source-stage.json';path=(Join-Path $hostRoot 'source-stage.json')},[ordered]@{from='source/audit-card.json';path=(Join-Path $researchRoot 'doc/causal_v8_v9_train_diagnosis/independent-audit-card.json')},[ordered]@{from='source/MAIN_CARD.json';path=(Join-Path $researchRoot 'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json')},[ordered]@{from='source/diagnosis.json';path=(Join-Path $researchRoot 'doc/causal_v8_v9_train_diagnosis/diagnosis.json')})){$null=Require-Pin (Join-Path $mirror $row.from) (Pin $row.path)}
$sourceExact=@($execution.source_before).Count-eq10-and@($execution.source_after).Count-eq10-and(($execution.source_before|ConvertTo-Json -Depth 15 -Compress)-ceq($execution.source_after|ConvertTo-Json -Depth 15 -Compress))
foreach($row in $capsule.files){
 if([IO.Path]::IsPathRooted($row.file)-or$row.file.Contains(':')-or$row.file.Contains([char]92)-or$row.file-match'(^|/)\.\.(/|$)'){throw 'Bounded capsule file required'}
 $null=Require-Pin (Join-Path $mirror ('source/'+$row.file)) $row
 $expected=$containerRoot+'/source/'+$row.file
 foreach($array in @(@{items=@($execution.source_before)},@{items=@($execution.source_after)})){
  $matching=@($array.items|Where-Object {$_.path-eq$expected})
  if($matching.Count-ne1-or$matching[0].bytes-ne$row.bytes-or$matching[0].sha256-ne$row.sha256){$sourceExact=$false}
 }
}
if($execution.wrapper.sha256-ne$card.execution_wrapper.sha256-or$execution.wrapper.sha256-ne$stage.execution_wrapper_sha256-or$execution.audit_card.sha256-ne$AuditCardSha256-or$execution.model_calls-ne0-or$execution.head_fits-ne0-or$execution.training_updates-ne0){throw 'Exact admitted recorder and zero forbidden execution required'}
$mirrorStatus=if($execution.status-eq'passed'-and-not$sourceExact){'failed'}else{$execution.status}
if($mirrorStatus-eq'passed'){
 $raw=Join-Path $mirror 'run-001/audit.json';$null=Require-Pin $raw $execution.result
 $result=Get-Content -LiteralPath $raw -Raw|ConvertFrom-Json
 if($result.kind-ne'cpb_v8_v9_TRAIN_diagnosis_independent_audit_v1'-or$result.status-ne'passed'-or@($result.mismatches).Count-ne0-or$result.card.sha256-ne$stage.main_card_sha256-or$result.diagnosis.sha256-ne$stage.main_diagnosis_sha256-or$result.audit_card.sha256-ne$AuditCardSha256){$mirrorStatus='failed'}
}
$receipt=[ordered]@{kind='cpb_v8_v9_TRAIN_independent_audit_mirror_v1';status=$mirrorStatus;container_root=$containerRoot;host_root=$mirror;file_count=$actual.Count;files=$files;inventory=(Pin $inventoryPath);execution=(Pin $executionPath);all_hashes_exact=$true;source_before_after_exact=$sourceExact;externally_reviewed_audit_card_sha256=$AuditCardSha256;externally_reviewed_stage_sha256=$StageReceiptSha256;externally_reviewed_execution_sha256=$ExecutionSha256;externally_reviewed_inventory_sha256=$InventorySha256;helper=(Pin $PSCommandPath);model_calls=0;head_fits=0;training_updates=0;deletions=0}
$hostReceipt=Join-Path $hostRoot 'mirror-receipt.json'
if(Test-Path -LiteralPath $hostReceipt){throw 'Preserve earlier audit mirror receipt'}
[IO.File]::WriteAllText($hostReceipt,(($receipt|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
if($mirrorStatus-ne'passed'){throw 'Failed audit or changed source remains fully mirrored; no canonical passed copies'}
$durable=Join-Path $researchRoot 'doc/causal_v8_v9_train_diagnosis/independent-audit.json'
$executionCopy=Join-Path $researchRoot 'doc/causal_v8_v9_train_diagnosis/independent-audit-execution.json'
$durableReceipt=Join-Path $researchRoot 'doc/causal_v8_v9_train_diagnosis/independent-audit-mirror.json'
foreach($path in @($durable,$executionCopy,$durableReceipt)){if(Test-Path -LiteralPath $path){throw 'Preserve all earlier durable audit evidence'}}
[IO.File]::Copy($raw,$durable,$false);$receipt['independent_audit']=Require-Pin $durable (Pin $raw)
[IO.File]::Copy($executionPath,$executionCopy,$false);$receipt['durable_execution']=Require-Pin $executionCopy (Pin $executionPath)
[IO.File]::WriteAllText($durableReceipt,(($receipt|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';file_count=$actual.Count;independent_audit=$receipt.independent_audit;execution=$receipt.durable_execution;mirror_receipt=(Pin $durableReceipt)}|ConvertTo-Json -Depth 5

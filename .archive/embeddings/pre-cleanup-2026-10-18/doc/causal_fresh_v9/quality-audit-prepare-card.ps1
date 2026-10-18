# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# File/JSON metadata only. Never executes an auditor or numerical project code.
param([Parameter(Mandatory=$true)][switch]$SourcesReviewed,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QualityReceiptSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$CompleteMirrorSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ReportingRecoveryCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ReportingRecoveryReceiptSHA256)
$ErrorActionPreference='Stop'
if(-not $SourcesReviewed){throw 'Independent source review must precede audit-card freeze'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$output=Join-Path $PSScriptRoot 'quality-audit-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve the prior prospective audit card'}
function Pin([string]$relative){
 $f=Get-Item -LiteralPath (Join-Path $researchRoot $relative)
 if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Physical source/evidence required'}
 [ordered]@{path=$relative;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$mainCardRelative='doc/CAUSAL_FRESH_V9_CARD.json'
$mainPin=Pin $mainCardRelative
if($mainPin.sha256-ne$MainCardSHA256){throw 'Externally reviewed main card required before parse'}
$main=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $mainCardRelative)|ConvertFrom-Json
$inputPin=Pin 'doc/causal_fresh_v9/input-manifest.json'
if($main.model-ne'CPB-v9'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Frozen actual V9 prospective card required'}
if($inputPin.bytes-ne$main.input_manifest.bytes-or$inputPin.sha256-ne$main.input_manifest.sha256){throw 'Exact main-card-bound input manifest required'}
$packageRoot='output/engineering/causal-fresh-v9/attempt-001'
$manifestPin=Pin "$packageRoot/source-manifest.json"
if($manifestPin.sha256-ne$MainManifestSHA256){throw 'Externally reviewed main manifest required before parse'}
$manifest=Get-Content -Raw -LiteralPath (Join-Path $researchRoot "$packageRoot/source-manifest.json")|ConvertFrom-Json
$package=Get-Content -Raw -LiteralPath (Join-Path $researchRoot "$packageRoot/package.json")|ConvertFrom-Json
$zipPin=Pin "$packageRoot/source.zip"
if($manifest.card_sha256-ne$mainPin.sha256-or$manifest.files.Count-le0-or$package.source_files-ne$manifest.files.Count-or$package.manifest_sha256-ne$manifestPin.sha256-or$package.zip_sha256-ne$zipPin.sha256){throw 'Exact immutable main package required'}
$qualityPin=Pin 'output/runs/causal-fresh-v9/run-001/quality-001/quality-receipt.json'
$completePin=Pin 'doc/causal_fresh_v9/complete-mirror-attempt-001.json'
if($qualityPin.sha256-ne$QualityReceiptSHA256-or$completePin.sha256-ne$CompleteMirrorSHA256){throw 'Actual externally reviewed quality and Complete mirror required before audit card'}
$quality=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $qualityPin.path)|ConvertFrom-Json
$complete=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $completePin.path)|ConvertFrom-Json
if($quality.status-ne'passed'-or$quality.model-ne'CPB-v9.market-01'-or$quality.card_sha256-ne$MainCardSHA256-or$complete.status-ne'passed'-or$complete.scope-ne'Complete'-or$complete.attempt-ne'attempt-001'-or$complete.card_sha256-ne$MainCardSHA256-or$complete.source_manifest_sha256-ne$MainManifestSHA256-or-not$complete.all_mirrored_pt_ignored){throw 'Complete measured V9 evidence before audit declaration required'}
$recoveryCardPin=Pin 'doc/causal_fresh_v9/reporting-recovery-card-001.json'
$recoveryReceiptPin=Pin 'doc/causal_fresh_v9/reporting-recovery-receipt-001.json'
if($recoveryCardPin.sha256-ne$ReportingRecoveryCardSHA256-or$recoveryReceiptPin.sha256-ne$ReportingRecoveryReceiptSHA256){throw 'Explicit actual report-only recovery hashes before parse'}
$recoveryCard=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $recoveryCardPin.path)|ConvertFrom-Json
$recoveryReceipt=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $recoveryReceiptPin.path)|ConvertFrom-Json
if($recoveryCard.kind-ne'cpb_v9_report_only_recovery_card_v1'-or$recoveryCard.root_reviewed-ne$true-or$recoveryReceipt.kind-ne'cpb_v9_report_only_recovery_receipt_v1'-or$recoveryReceipt.status-ne'passed'-or$recoveryReceipt.card.bytes-ne$recoveryCardPin.bytes-or$recoveryReceipt.card.sha256-ne$ReportingRecoveryCardSHA256-or$recoveryCard.main_source_identity.card_sha256-ne$MainCardSHA256-or$recoveryCard.main_source_identity.manifest_sha256-ne$MainManifestSHA256){throw 'Exact separately reviewed completed report-only recovery required'}
if($complete.report_only_recovery.status-ne'passed'-or$complete.report_only_recovery.card_sha256-ne$ReportingRecoveryCardSHA256-or$complete.report_only_recovery.receipt_sha256-ne$ReportingRecoveryReceiptSHA256-or$complete.report_only_recovery.original_quality_stage_status-ne'failed'-or$complete.report_only_recovery.completed_scientific_stages-ne5-or$complete.report_only_recovery.science_stages_replayed-ne0-or-not$complete.report_only_recovery.scientific_inputs_before_after_exact){throw 'Complete mirror must retain failed wrapper and separately admit reporting recovery'}
$null=Pin 'output/runs/causal-fresh-v9/run-001/quality-001/report/table_rows.json'
$null=Pin 'output/runs/causal-fresh-v9/run-001/quality-001/comparison/comparison.json'
if(($main.initial_pairing.native_coordinate_slice -join ',')-ne'20,40'-or$main.initial_pairing.content20_equality_required-or$main.initial_pairing.references.Count-ne13){throw 'Declared partial native20:40 pairing only'}
$sources=@(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/research/causal_fresh_v9_quality_audit') -File|Sort-Object Name|ForEach-Object{
 $relative=$_.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47)
 $p=Pin $relative
 [ordered]@{file=$_.Name;bytes=$p.bytes;sha256=$p.sha256}
})
if(($sources.file -join ',')-ne'archive_reader.py,audit.py,provenance.json,README.md'){throw 'Exactly four reviewed owned auditor files required'}
$card=[ordered]@{
 kind='causal_fresh_v9_saved_quality_audit_card_v1'
 copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 declared_utc=[DateTime]::UtcNow.ToString('o')
 stage='declared_after_static_review_before_any_saved_numeric_audit'
 execution_pending=$true
 model='CPB-v9.market-01'
 model_calls=0;head_fits=0;training_updates=0;old_model_numeric_reads=0
 main_card=$mainPin
 input_manifest=$inputPin
 main_source_manifest_sha256=$manifestPin.sha256
 main_source_zip_sha256=$zipPin.sha256
 main_source_files=$manifest.files.Count
 main_attempt='attempt-001'
 outside_frozen_main_source=$true
 auditor_sources=$sources
 expected_main_paths=[ordered]@{training='/opt/cuwacunu/research/causal-fresh-v9/attempt-001/run-001';quality='/opt/cuwacunu/research/causal-fresh-v9/attempt-001/quality-001';card='/opt/cuwacunu/research/causal-fresh-v9/attempt-001/source/doc/CAUSAL_FRESH_V9_CARD.json';reporting_recovery_card='/opt/cuwacunu/research/causal-fresh-v9-report-recovery/attempt-001/source/recovery-card.json';reporting_recovery_receipt='/opt/cuwacunu/research/causal-fresh-v9-report-recovery/attempt-001/run-001/recovery.json'}
 paired_initial_reference_inputs=$main.initial_pairing.references
 native_coordinate_slice=@(20,40)
 whole_native40_equality_claimed=$false
 goal_protocol=(Pin 'doc/causal_fresh_v9/goal-protocol.json')
 source_parent=(Pin 'code/research/causal_fresh_v8_quality_audit/audit.py')
 completed_quality_receipt=$qualityPin
 completed_main_mirror=$completePin
 reporting_recovery_card=$recoveryCardPin
 reporting_recovery_receipt=$recoveryReceiptPin
 report_only_recovery_required=$true
 scope=@('Saved market predictions/physical errors/ridge reconstruction,112/32 row association,all repetitions and fixed D40 recipes','TEMPO5 saved count/group/interval/mean/coverage arithmetic;1572 recorded initial/final exports;no forwards or fits','Exactly13 V8 references: eight partial20:40 DATA banks with whole remaining declared fields,five whole TRAIN-statistic archives;new own initial quality','Six clean controls,V6/V7/V8 separately established trust and unchanged pending raw tables,e79222 max/min/positivity rules,chronological ties,no TRAIN NLL gate','Seven binaries/eleven translation units,separate objective and temporal CUDA proof,policy/partition/half-mix/empty-local fields,75 identifiable tensors with294751 numel capacity;not every-scalar gradient proof','Saved temporal8192/gauge2048/between2048 row counts/receipt pins and declared Hadamard learning predicates;independent training audit owns detailed observed gradient-array/group arithmetic')
 fresh_schema_contract='Only MAIN fresh constructor JSON omits gradient_policy; require absence there and strict policy in all other actual objective-bearing trust records.'
 execution_wrapper=(Pin 'doc/causal_fresh_v9/quality-audit-execute.py')
 stage_helper=(Pin 'doc/causal_fresh_v9/quality-audit-stage.ps1')
 mirror_helper=(Pin 'doc/causal_fresh_v9/quality-audit-mirror.ps1')
 card_generator=(Pin 'doc/causal_fresh_v9/quality-audit-prepare-card.ps1')
 forbidden=@('Torch/NumPy or producer/comparator imports','Encoder/decoder/head forward or fitting','Old model tensor reads','Whole40 initial pairing claim','Post-encoder PCA/tuning/selection','Mutation of actual main/card/source/results or previous evidence')
}
[IO.File]::WriteAllText($output,($card|ConvertTo-Json -Depth 30)+"`n",[Text.UTF8Encoding]::new($false))
Pin 'doc/causal_fresh_v9/quality-audit-card.json'|ConvertTo-Json

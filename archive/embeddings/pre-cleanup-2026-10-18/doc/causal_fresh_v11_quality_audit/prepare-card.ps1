# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Complete physical FINAL V10 normal helper copy, adapted to NORMAL V11 quality.
# Host file/hash/JSON metadata only. No Docker, AST, model or auditor execution.
param(
 [Parameter(Mandatory=$true)][switch]$SourcesAndCompletedEvidenceReviewed,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainZipSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QualityReceiptSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QualityStageSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$CompleteMirrorSHA256,
 [Parameter(Mandatory=$true)][string]$SourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SourceReviewSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$HelpersProvenanceSHA256,
 [ValidateRange(1,999)][int]$MainAttempt=1
)
$ErrorActionPreference='Stop'
if(-not$SourcesAndCompletedEvidenceReviewed){throw 'Independent source and actual completed-quality review must precede audit-card freeze'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$helperRelative='doc/causal_fresh_v11_quality_audit'
$output=Join-Path $PSScriptRoot 'audit-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve any existing prospective audit card'}
function Rooted([string]$relative){
 $full=if([IO.Path]::IsPathRooted($relative)){[IO.Path]::GetFullPath($relative)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))}
 if(-not$full.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Explicit repository-bounded file required'}
 for($cursor=$full;$cursor.Length-ge$researchRoot.Length;$cursor=[IO.Path]::GetDirectoryName($cursor)){
  if((Get-Item -LiteralPath $cursor).Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked path refused'}
  if($cursor-eq$researchRoot){break}
 }
 $full
}
function Pin([string]$relative){
 $path=Rooted $relative;$f=Get-Item -LiteralPath $path
 [ordered]@{path=$path.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Require-Pin($pin){$actual=Pin $pin.path;if($actual.bytes-ne$pin.bytes-or$actual.sha256-ne$pin.sha256){throw ('Exact file required: '+$pin.path)}}
$workflowPin=Pin ($helperRelative+'/provenance.json')
if($workflowPin.sha256-ne$HelpersProvenanceSHA256){throw 'External reviewed helper provenance SHA required before parse'}
$workflow=Get-Content -Raw -LiteralPath (Rooted $workflowPin.path)|ConvertFrom-Json
if($workflow.kind-ne'cpb_v11_owned_quality_audit_orchestration_source_v1'-or$workflow.current_sources.Count-ne4){throw 'Exact reviewed four-helper closure required'}
foreach($entry in $workflow.current_sources){Require-Pin $entry}
$self=@($workflow.current_sources|Where-Object{$_.path-eq($helperRelative+'/prepare-card.ps1')})
if($self.Count-ne1-or$self[0].sha256-ne(Pin $PSCommandPath).sha256){throw 'Generator self pin required'}
$reviewPin=Pin $SourceReviewPath
if($reviewPin.sha256-ne$SourceReviewSHA256){throw 'External auditor source-review SHA required before parse'}
$review=Get-Content -Raw -LiteralPath (Rooted $reviewPin.path)|ConvertFrom-Json
if($review.kind-ne'cpb_v11_quality_auditor_independent_static_source_review_v1'-or$review.status-ne'passed'-or$review.sources.Count-ne4){throw 'Passed actual four-source independent review required'}
$sourceRelative='code/research/causal_fresh_v11_quality_audit'
$sources=@(Get-ChildItem -LiteralPath (Rooted $sourceRelative) -File|Sort-Object Name|ForEach-Object{
 $relative=$_.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);$pin=Pin $relative
 $binding=@($review.sources|Where-Object{$_.path-eq$relative})
 if($binding.Count-ne1-or$binding[0].bytes-ne$pin.bytes-or$binding[0].sha256-ne$pin.sha256){throw 'Current auditor must match independent review exactly'}
 [ordered]@{file=$_.Name;bytes=$pin.bytes;sha256=$pin.sha256}
})
if(($sources.file-join',')-cne'archive_reader.py,audit.py,provenance.json,README.md'){throw 'Exactly four owned auditor sources required'}
$mainPin=Pin 'doc/CAUSAL_FRESH_V11_CARD.json'
if($mainPin.sha256-ne$MainCardSHA256){throw 'External actual main card SHA required before parse'}
$main=Get-Content -Raw -LiteralPath (Rooted $mainPin.path)|ConvertFrom-Json
$inputPin=Pin 'doc/causal_fresh_v11/input-manifest.json'
if($main.model-ne'CPB-v11'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Actual prospectively frozen V11 scientific card required'}
if($inputPin.bytes-ne$main.input_manifest.bytes-or$inputPin.sha256-ne$main.input_manifest.sha256){throw 'Exact main-card data-only input identity required'}
if(($main.initial_pairing.native_coordinate_slice-join',')-ne'20,40'-or$main.initial_pairing.content20_equality_required-ne$false-or$main.initial_pairing.native_coordinates_compared-ne20-or$main.initial_pairing.references.Count-ne13){throw 'Exact partial20:40 eight-DATA/five-whole-statistic pairing declaration required'}
$label='attempt-{0:D3}' -f $MainAttempt
$packageRoot='output/engineering/causal-fresh-v11/'+$label
$runtimeMain='/opt/cuwacunu/research/causal-fresh-v11/'+$label
$manifestPin=Pin ($packageRoot+'/source-manifest.json')
$zipPin=Pin ($packageRoot+'/source.zip')
if($manifestPin.sha256-ne$MainManifestSHA256-or$zipPin.sha256-ne$MainZipSHA256){throw 'External actual main manifest/ZIP SHA guards before parse'}
$manifest=Get-Content -Raw -LiteralPath (Rooted $manifestPin.path)|ConvertFrom-Json
$package=Get-Content -Raw -LiteralPath (Rooted ($packageRoot+'/package.json'))|ConvertFrom-Json
if($manifest.card_sha256-ne$MainCardSHA256-or$manifest.files.Count-le0-or$package.source_files-ne$manifest.files.Count-or$package.manifest_sha256-ne$MainManifestSHA256-or$package.zip_sha256-ne$MainZipSHA256){throw 'Immutable actual main package identity required'}
$qualityPin=Pin 'output/runs/causal-fresh-v11/run-001/quality-001/quality-receipt.json'
$qualityStageHost=Pin ($packageRoot+'/quality-stage.json')
$completePin=Pin ('doc/causal_fresh_v11/complete-mirror-'+$label+'.json')
if($qualityPin.sha256-ne$QualityReceiptSHA256-or$qualityStageHost.sha256-ne$QualityStageSHA256-or$completePin.sha256-ne$CompleteMirrorSHA256){throw 'External passed quality receipt/stage/Complete hashes required before parse'}
$quality=Get-Content -Raw -LiteralPath (Rooted $qualityPin.path)|ConvertFrom-Json
$qualityStage=Get-Content -Raw -LiteralPath (Rooted $qualityStageHost.path)|ConvertFrom-Json
$complete=Get-Content -Raw -LiteralPath (Rooted $completePin.path)|ConvertFrom-Json
if($quality.kind-ne'causal_fresh_v11_quality_execution_v1'-or$quality.status-ne'passed'-or$quality.model-ne'CPB-v11.market-01'-or$quality.card_sha256-ne$MainCardSHA256-or$quality.gradient_policy-ne'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'){throw 'Actual passed normal V11 quality receipt required'}
if($qualityStage.kind-ne'causal_fresh_v11_exclusive_stage_execution_v1'-or$qualityStage.status-ne'passed'-or$qualityStage.exit_code-ne0){throw 'Actual completed outer quality wrapper required,not a reporting-recovery assumption'}
if($complete.status-ne'passed'-or$complete.scope-ne'Complete'-or$complete.attempt-ne$label-or$complete.card_sha256-ne$MainCardSHA256-or$complete.source_manifest_sha256-ne$MainManifestSHA256-or-not$complete.all_mirrored_pt_ignored){throw 'Passed exact completed main mirror required'}
function Require-CompletedFile([string]$relative,[string]$runtime){
 $matches=@($complete.files|Where-Object{$_.path-eq$relative-and$_.container_path-eq$runtime})
 if($matches.Count-ne1){throw ('One exact completed mirror member required: '+$relative)}
 Require-Pin $matches[0]
 $matches[0]
}
$null=Require-CompletedFile $qualityPin.path ($runtimeMain+'/quality-001/quality-receipt.json')
$null=Require-CompletedFile $qualityStageHost.path ($runtimeMain+'/quality-stage.json')
$science=@('extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads')
if((@($quality.stage_seconds.PSObject.Properties.Name|Sort-Object)-join',')-cne(@($science|Sort-Object)-join',')){throw 'The receipt contains five scientific durations,before report/comparison'}
foreach($name in @($science)+@('report','comparison')){
 $relative='output/runs/causal-fresh-v11/run-001/quality-001/'+$name+'-stage.json'
 $null=Require-CompletedFile $relative ($runtimeMain+'/quality-001/'+$name+'-stage.json')
 $stage=Get-Content -Raw -LiteralPath (Rooted $relative)|ConvertFrom-Json
 if($stage.stage-ne$name-or$stage.exit_code-ne0){throw 'All seven normal local stages must have passed'}
}
foreach($relative in @('report/table_rows.json','report/completion.json','comparison/comparison.json')){
 $null=Require-CompletedFile ('output/runs/causal-fresh-v11/run-001/quality-001/'+$relative) ($runtimeMain+'/quality-001/'+$relative)
}
$qualityRuntimePin=[ordered]@{path=($runtimeMain+'/quality-stage.json');bytes=$qualityStageHost.bytes;sha256=$qualityStageHost.sha256}
$card=[ordered]@{
 kind='causal_fresh_v11_saved_quality_audit_card_v1';copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 declared_utc=[DateTime]::UtcNow.ToString('o');stage='declared_after_static_review_before_any_saved_numeric_audit'
 execution_pending=$true;model='CPB-v11.market-01';model_calls=0;head_fits=0;training_updates=0;old_model_numeric_reads=0
 main_card=$mainPin;input_manifest=$inputPin;main_source_manifest_sha256=$manifestPin.sha256;main_source_zip_sha256=$zipPin.sha256;main_source_files=$manifest.files.Count
 main_attempt=$label;audit_attempt='attempt-001';outside_frozen_main_source=$true;auditor_sources=$sources
 expected_main_paths=[ordered]@{training=($runtimeMain+'/run-001');quality=($runtimeMain+'/quality-001');card=($runtimeMain+'/source/doc/CAUSAL_FRESH_V11_CARD.json')}
 paired_initial_reference_inputs=$main.initial_pairing.references;native_coordinate_slice=@(20,40);native_coordinates_compared=20;content20_equality_required=$false;whole_native40_equality_claimed=$false
 goal_protocol=(Pin 'doc/causal_fresh_v11/goal-protocol.json')
 source_parent=(Pin 'code/research/causal_fresh_v10_quality_audit/audit.py')
 independent_source_review=$reviewPin;helpers_provenance=$workflowPin
 completed_quality_receipt=$qualityPin;completed_quality_stage=$qualityRuntimePin;completed_main_mirror=$completePin
 report_only_recovery_required=$false
 scope=@('All112TRAIN/32knownVALIDATION saved market predictions,physical errors,ridge reconstruction,allrepetitions,fixedD40recipes;no fitting','TEMPO5 saved count/group/interval/mean/coverage arithmetic;1572 recorded initial/final exports;no model or head forward','13savedV9 references:eight initialDATA native20:40 tails and five whole fresh-statistic archives;content0:20 may differ;no imported model/statistics or old score reuse','Eight clean V3–V10 controls,five separate V6–V10 trust joins;historicalV8/V9PARTIAL andV10FULL40 semantics and opaqueV9report-only recovery remain unchanged','221f22exactmax/min/positivity operators,chronological ties,noTRAINNLLgate,no automatic winner or appselection','Sevenbinary11TU actualsource closure;restoredV9 detached-variance objective proof and new full-attention mean4/centeredvariance4 temporal proof;75mandatorytensors plusregisteredgauge capacity','Actual passed normal outerquality wrapper/five scientific receipt durations/seven local exit0 stage files and report/comparison outputs;no current recovery','Saved8192group/2048gauge/2048between/16384moment-subgroup evidence/receipt bindings;independent training audit owns detailed saved-array arithmetic')
 fresh_schema_contract='Only main fresh-constructor JSON omits gradient_policy; requireabsence there and strict policies elsewhere.'
 execution_wrapper=(Pin ($helperRelative+'/execute.py'));stage_helper=(Pin ($helperRelative+'/stage.ps1'));mirror_helper=(Pin ($helperRelative+'/mirror.ps1'));card_generator=(Pin ($helperRelative+'/prepare-card.ps1'))
 forbidden=@('Torch/NumPy or producer imports','Encoder/decoder/head forward or fitting','Old trained checkpoint numeric reads','Whole40 current initial equality claim','Current reporting recovery arguments or prior-failure assumptions','PCA/tuning/selection','Mutation of source/cards/results/old evidence')
}
[IO.File]::WriteAllText($output,($card|ConvertTo-Json -Depth 30)+"`n",[Text.UTF8Encoding]::new($false))
Pin ($helperRelative+'/audit-card.json')|ConvertTo-Json -Depth 5

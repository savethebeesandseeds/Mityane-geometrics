# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
# New owned metadata generator derived from the exact copied V8 card schema.
# Source preparation is not admission. Invoke only after actual final artifacts exist.
param([Parameter(Mandatory=$true)][string]$InputPlan,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InputPlanSHA256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedActualInputs,
 [Parameter(Mandatory=$true)][switch]$FinalMetadataApplied)
$ErrorActionPreference='Stop'
if(-not$RootReviewedActualInputs-or-not$FinalMetadataApplied){throw 'Actual root-reviewed artifacts and final metadata application required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$output=Join-Path $researchRoot 'doc/causal_fresh_v9/preservation-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve any existing preservation card'}
function Resolve-Research([string]$relative){
 $path=if([IO.Path]::IsPathRooted($relative)){[IO.Path]::GetFullPath($relative)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))}
 if(-not$path.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Bounded research path required'}
 $path
}
function Pin([string]$relative){
 $file=Get-Item -LiteralPath (Resolve-Research $relative)
 if($file.PSIsContainer-or($file.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Physical file required'}
 [ordered]@{path=[IO.Path]::GetRelativePath($researchRoot,$file.FullName).Replace([char]92,[char]47);bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Require-Pin($binding){$actual=Pin $binding.path;if($actual.bytes-ne$binding.bytes-or$actual.sha256-ne$binding.sha256){throw ('Actual reviewed input changed: '+$binding.path)};$actual}
$planPin=Pin $InputPlan
if($planPin.sha256-ne$InputPlanSHA256){throw 'External input-plan hash required before JSON parse'}
$plan=Get-Content -Raw -LiteralPath (Resolve-Research $InputPlan)|ConvertFrom-Json
if($plan.kind-ne'causal_fresh_v9_final_preservation_input_plan_v1'-or$plan.status-ne'root_reviewed_actual_inputs'-or-not$plan.authorized){throw 'Explicit actual-input plan required'}
foreach($field in @('compiled_binaries','concurrent_DEV_exclusions','protected_inventories','protected_pins','additional_protected_roots')){
 if($field-notin@($plan.PSObject.Properties.Name)-or$null-eq$plan.$field){throw ('Explicit array, even if empty, required: '+$field)}
}
$template=Pin 'code/research/causal_fresh_v9_preservation/v8-preservation-card.template.json'
if($template.bytes-ne134629-or$template.sha256-ne'04058b51806411d8bcf0193144e2423654d4e211a02eedd273b236d592c9f569'){throw 'Exact physical V8 card template required'}
$card=Get-Content -Raw -LiteralPath (Resolve-Research $template.path)|ConvertFrom-Json -AsHashtable
$required=@('prior_live_snapshot','complete_mirror','source_manifest','source_zip','main_card','metadata_application','proposal_review','final_establishment','compiled_binary_mirror','reporting_recovery_card','reporting_recovery_receipt','failure_mirror')
foreach($role in $required){if($null-eq$plan.$role){throw ('Missing actual role '+$role)};$card[$role]=Require-Pin $plan.$role}
$prior=Pin 'doc/causal_fresh_v8/final-preservation.json'
if($prior.bytes-ne140977-or$prior.sha256-ne'919b4f11610e3ab61b63afbc3612145506e4c47a3be3c777f119d47c49e7126a'){throw 'Exact94-checkpoint parent receipt required'}
$parent=Get-Content -Raw -LiteralPath (Resolve-Research $prior.path)|ConvertFrom-Json
if($parent.status-ne'passed'-or$parent.critical_checkpoint_pins.Count-ne94-or$parent.old_pins-ne1030){throw 'Parent must preserve94 actual critical checkpoints and1030oldpins'}
$card.Remove('prior_V7_final')|Out-Null
$card['prior_V8_final']=$prior
$manifest=Get-Content -Raw -LiteralPath (Resolve-Research $card.source_manifest.path)|ConvertFrom-Json
$mirror=Get-Content -Raw -LiteralPath (Resolve-Research $card.complete_mirror.path)|ConvertFrom-Json
$main=Get-Content -Raw -LiteralPath (Resolve-Research $card.main_card.path)|ConvertFrom-Json
$established=Get-Content -Raw -LiteralPath (Resolve-Research $card.final_establishment.path)|ConvertFrom-Json
$applied=Get-Content -Raw -LiteralPath (Resolve-Research $card.metadata_application.path)|ConvertFrom-Json
$review=Get-Content -Raw -LiteralPath (Resolve-Research $card.proposal_review.path)|ConvertFrom-Json
if($main.model-ne'CPB-v9'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Actual frozen V9 scientific declaration required'}
if($manifest.kind-ne'causal_fresh_v9_source_capsule_v1'-or$manifest.files.Count-le0-or$manifest.card_sha256-ne$card.main_card.sha256){throw 'Actual admitted source count/card binding required'}
if($mirror.kind-ne'causal_fresh_v9_container_to_local_evidence_mirror_v1'-or$mirror.status-ne'passed'-or$mirror.scope-ne'Complete'-or$mirror.attempt-ne'attempt-001'-or$mirror.source_files_verified-ne$manifest.files.Count-or$mirror.card_sha256-ne$card.main_card.sha256-or$mirror.source_manifest_sha256-ne$card.source_manifest.sha256-or-not$mirror.all_mirrored_pt_ignored){throw 'Exact completed mirror required'}
if($card.reporting_recovery_card.sha256-ne'9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'-or$card.reporting_recovery_receipt.sha256-ne'5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72'-or$card.failure_mirror.sha256-ne'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532'){throw 'Exact actual reporting-only recovery and pre-recovery failure receipt required'}
$recovery=Get-Content -Raw -LiteralPath (Resolve-Research $card.reporting_recovery_receipt.path)|ConvertFrom-Json
$recoveryCard=Get-Content -Raw -LiteralPath (Resolve-Research $card.reporting_recovery_card.path)|ConvertFrom-Json
if($recovery.kind-ne'cpb_v9_report_only_recovery_receipt_v1'-or$recovery.status-ne'passed'-or$recoveryCard.kind-ne'cpb_v9_report_only_recovery_card_v1'-or-not$recoveryCard.root_reviewed-or$recovery.card.sha256-ne$card.reporting_recovery_card.sha256-or$recovery.original_quality_stage_status-ne'failed'-or$recovery.completed_scientific_stages-ne5-or$recovery.science_stages_replayed-ne0-or$recovery.encoder_calls-ne0-or$recovery.head_fits-ne0-or$recovery.training_updates-ne0-or-not$recovery.frozen_scientific_source_unchanged-or-not$recovery.scientific_inputs_before_after_exact-or($recovery.scientific_inputs_before|ConvertTo-Json -Depth 12 -Compress)-cne($recovery.scientific_inputs_after|ConvertTo-Json -Depth 12 -Compress)){throw 'Strict reporting-only recovery preserves failed stages and science/source with zero replay'}
if($recovery.main_source_identity.source_files-ne118-or$recovery.main_source_identity.card_sha256-ne$card.main_card.sha256-or$recovery.main_source_identity.manifest_sha256-ne$card.source_manifest.sha256-or$recovery.main_source_identity.archive_sha256-ne$card.source_zip.sha256-or$mirror.report_only_recovery.status-ne'passed'-or$mirror.report_only_recovery.card_sha256-ne$card.reporting_recovery_card.sha256-or$mirror.report_only_recovery.receipt_sha256-ne$card.reporting_recovery_receipt.sha256){throw 'Complete mirror must bind exact separate recovery and immutable118 source identity'}
$new5=@($mirror.files|Where-Object {$_.path-like'output/runs/causal-fresh-v9/run-001/checkpoints/*.pt'})
$expected=@('inference-0.pt','inference-2048.pt','training-512.pt','training-1024.pt','training-2048.pt')
if($new5.Count-ne5-or(Compare-Object @($new5|ForEach-Object{Split-Path -Leaf $_.path}|Sort-Object) @($expected|Sort-Object))){throw 'Do not create a99-checkpoint declaration before the exact five actual V9 checkpoints exist'}
foreach($pin in $new5){$null=Require-Pin $pin}
if($established.status-ne'passed'-or$established.candidate-ne'CPB-v9.market-01'-or-not$established.measured_candidate_established-or-not$established.trust_admission_passed-or-not$established.full_coverage-or$established.initial_pairing_count-ne13-or$established.numeric_total-ne22-or$established.overall_winner-or$established.winner_selection_made){throw 'Actual separate trusted measured candidate required; no winner selection'}
if($applied.status-ne'passed'-or$applied.baseline_changed-or$applied.application_changed-or$review.status-ne'passed'){throw 'Root application and independent proposal review required'}
$buildEntry=@($mirror.files|Where-Object {$_.path.EndsWith('/build-admission.json')})
if($buildEntry.Count-ne1){throw 'One build-admission pin required'}
$null=Require-Pin $buildEntry[0]
$build=Get-Content -Raw -LiteralPath (Resolve-Research $buildEntry[0].path)|ConvertFrom-Json
$binaryMirror=Get-Content -Raw -LiteralPath (Resolve-Research $card.compiled_binary_mirror.path)|ConvertFrom-Json
if($binaryMirror.kind-ne'cpb_v9_seven_build_binaries_mirror_v1'-or$binaryMirror.status-ne'passed'-or$binaryMirror.file_count-ne7-or$binaryMirror.files.Count-ne7-or$binaryMirror.included_in_main_Complete_files-ne$false-or-not$binaryMirror.container_files_preserved){throw 'Exact separately preserved seven-binary mirror receipt required'}
if($binaryMirror.build_admission.path-ne$buildEntry[0].path-or$binaryMirror.build_admission.bytes-ne$buildEntry[0].bytes-or$binaryMirror.build_admission.sha256-ne$buildEntry[0].sha256){throw 'Separate binary mirror must bind the same actual build admission'}
$binaries=@($plan.compiled_binaries)
if($binaries.Count-ne7-or$build.binaries.Count-ne7){throw 'Seven separately mirrored actual compiled binaries required'}
$expectedBinaryNames=@('test_causal_patch_bottleneck_mae_v9','test_cpb_v9_variance_gradient','test_cpb_v9_temporal_pooling','causal_fresh_v9_train','causal_fresh_v9_extract','causal_fresh_v9_market_read','causal_fresh_v9_synthetic_read')
foreach($set in @(@{items=$binaries;label='host'},@{items=@($build.binaries);label='build'},@{items=@($binaryMirror.files);label='receipt'})){
 $names=@($set.items|ForEach-Object {Split-Path -Leaf $_.path})
 if(@($set.items.path|Sort-Object -Unique).Count-ne7-or@($names|Sort-Object -Unique).Count-ne7-or(Compare-Object @($names|Sort-Object) @($expectedBinaryNames|Sort-Object))){throw ('Exact unique admitted seven-binary set required: '+$set.label)}
}
foreach($pin in $binaries){
 $null=Require-Pin $pin
 $match=@($build.binaries|Where-Object {(Split-Path -Leaf $_.path)-eq(Split-Path -Leaf $pin.path)})
 $mirrored=@($binaryMirror.files|Where-Object {$_.path-eq$pin.path})
 if($match.Count-ne1-or$match[0].bytes-ne$pin.bytes-or$match[0].sha256-ne$pin.sha256-or$mirrored.Count-ne1-or$mirrored[0].bytes-ne$pin.bytes-or$mirrored[0].sha256-ne$pin.sha256-or$mirrored[0].container_path-ne$match[0].path){throw 'Host binary must match both separate mirror receipt and actual container build pin'}
}
$card['compiled_binaries']=$binaries
$card['main_source_files']=$manifest.files.Count
$card['concurrent_DEV_exclusions']=@($plan.concurrent_DEV_exclusions)
$card['protected_inventories']=@($card.protected_inventories)+@($plan.protected_inventories)
$unique=[ordered]@{}
foreach($pin in @($card.protected_pins)+@($plan.protected_pins)){$repo=if($pin.repository){$pin.repository}else{'cuwacunu_embedding'};$unique[$repo+'|'+$pin.path+'|'+$pin.bytes+'|'+$pin.sha256]=$pin}
# The existing V8 receipt predates future-retention/diagnosis completion and V9
# pre-admission snapshots. Add those immutable contexts as explicit file pins.
$requiredRoots=@('code/research/causal_v8_retention_review','doc/causal_v8_retention_review',
 'code/research/causal_v6_v8_train_diagnosis','doc/causal_v6_v8_train_diagnosis',
 'code/research/causal_fresh_v9_preservation','code/research/causal_fresh_v9_report_recovery','doc/causal_fresh_v9',
 'output/engineering/causal-fresh-v9-report-recovery/attempt-001',
 'output/engineering/causal-fresh-v9-report-recovery-preparation',
 'output/engineering/causal-fresh-v9-preflight-repairs',
 'output/engineering/causal-fresh-v9-quality-audit-preflight-repairs',
 'output/engineering/causal-fresh-v9-training-audit-preflight-repairs',
 'output/engineering/causal-fresh-v9-protocol-preparation',
 'output/engineering/causal-fresh-v9-preservation-preflight-repairs')
foreach($relative in @($requiredRoots)+@($plan.additional_protected_roots)){
 $directory=Get-Item -LiteralPath (Resolve-Research $relative)
 if(-not$directory.PSIsContainer-or($directory.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Explicit physical context directory required'}
 foreach($file in Get-ChildItem -LiteralPath $directory.FullName -Recurse -File){$pin=Pin $file.FullName;$unique['cuwacunu_embedding|'+$pin.path+'|'+$pin.bytes+'|'+$pin.sha256]=$pin}
}
foreach($path in @('doc/clean_causal_encoder_search_goal_v8_retention.json','doc/clean_causal_encoder_search_goal_v8_retention.md','doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.json','doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.md','doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS.md')){$pin=Pin $path;$unique['cuwacunu_embedding|'+$pin.path+'|'+$pin.bytes+'|'+$pin.sha256]=$pin}
$card['protected_pins']=@($unique.Values)
$card['kind']='causal_fresh_v9_final_preservation_card_v1'
$card['status']='frozen_admitted_after_actual_five_checkpoints_complete_trust_and_metadata_application'
$card['authorized']=$true
$card['created_utc']=[DateTime]::UtcNow.ToString('o')
$card['historical_critical_checkpoints']=94
$card['new_actual_checkpoints']=5
$card['expected_final_critical_checkpoints']=99
$card['verifier']=Pin 'code/research/causal_fresh_v9_preservation/verify.ps1'
$card['generator']=Pin 'code/research/causal_fresh_v9_preservation/prepare-card.ps1'
$card['input_plan']=$planPin
$card['parent_template']=$template
[IO.File]::WriteAllText($output,(($card|ConvertTo-Json -Depth 90).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Pin 'doc/causal_fresh_v9/preservation-card.json'|ConvertTo-Json

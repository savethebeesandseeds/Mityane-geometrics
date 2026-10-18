# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
# Complete physical FINAL V11 generator copy, adapted to normal V12 preservation.
# Metadata only; no verification, Docker, AST or scientific program is invoked here.
param([Parameter(Mandatory=$true)][string]$InputPlan,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InputPlanSHA256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedActualInputs,
 [Parameter(Mandatory=$true)][switch]$FinalMetadataApplied)
$ErrorActionPreference='Stop'
if(-not$RootReviewedActualInputs-or-not$FinalMetadataApplied){throw 'Actual independently/root-reviewed inputs and final metadata application required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$output=Join-Path $researchRoot 'doc/causal_fresh_v12/preservation-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve any existing preservation card'}
function Resolve-Research([string]$relative){
 $path=if([IO.Path]::IsPathRooted($relative)){[IO.Path]::GetFullPath($relative)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))}
 if(-not$path.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Bounded research path required'}
 for($cursor=$path;$cursor.Length-ge$researchRoot.Length;$cursor=[IO.Path]::GetDirectoryName($cursor)){
  if((Get-Item -LiteralPath $cursor).Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked ancestor refused'}
  if($cursor-eq$researchRoot){break}
 }
 $path
}
function Pin([string]$relative){
 $file=Get-Item -LiteralPath (Resolve-Research $relative)
 if($file.PSIsContainer){throw 'Physical file required'}
 [ordered]@{path=[IO.Path]::GetRelativePath($researchRoot,$file.FullName).Replace([char]92,[char]47);bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Require-Pin($binding){$actual=Pin $binding.path;if($actual.bytes-ne$binding.bytes-or$actual.sha256-ne$binding.sha256){throw ('Actual reviewed input changed: '+$binding.path)};$actual}
$planPin=Pin $InputPlan
if($planPin.sha256-ne$InputPlanSHA256){throw 'External input-plan hash before JSON parse required'}
$plan=Get-Content -Raw -LiteralPath (Resolve-Research $InputPlan)|ConvertFrom-Json
if($plan.kind-ne'causal_fresh_v12_final_preservation_input_plan_v1'-or$plan.status-ne'root_reviewed_actual_inputs'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true){throw 'Explicit actual-input Boolean admission required'}
foreach($field in @('compiled_binaries','new_checkpoint_pins','concurrent_DEV_exclusions','protected_inventories','protected_pins','additional_protected_roots','completed_evidence_roots','source_workflow')){
 if($field-notin@($plan.PSObject.Properties.Name)-or$null-eq$plan.$field){throw ('Explicit field required, including empty arrays where declared: '+$field)}
}
# Require this owned workflow and all four exact historical DATA templates before copying metadata.
foreach($binding in $plan.source_workflow){$null=Require-Pin $binding}
$workflowExpected=@('verify.ps1','prepare-card.ps1','README.md','provenance.json','v8-preservation-card.template.json','v9-preservation-card.template.json','v10-preservation-card.template.json','v11-preservation-card.template.json')
if($plan.source_workflow.Count-ne8-or(Compare-Object @($plan.source_workflow.path|Sort-Object) @($workflowExpected|ForEach-Object{'code/research/causal_fresh_v12_preservation/'+$_}|Sort-Object))){throw 'Eight exact owned workflow assets required'}
$self=@($plan.source_workflow|Where-Object{$_.path-eq'code/research/causal_fresh_v12_preservation/prepare-card.ps1'})
if($self.Count-ne1-or$self[0].sha256-ne(Pin $PSCommandPath).sha256){throw 'Reviewed generator self binding required'}
$template=Pin 'code/research/causal_fresh_v12_preservation/v11-preservation-card.template.json'
if($template.bytes-ne1035137-or$template.sha256-ne'516690f776d8cdb0306288135ba894c7611586044e4db2a9794e1aa828bc9325'){throw 'Exact final V11 card DATA template required;no historical execution'}
$priorTemplate=Get-Content -Raw -LiteralPath (Resolve-Research $template.path)|ConvertFrom-Json
$prior=Pin 'doc/causal_fresh_v11/final-preservation.json'
if($prior.bytes-ne1066891-or$prior.sha256-ne'805aa2183f14dbfd3a7bb521a67f5d01e33b22008b5244da45745baa8c51bab0'){throw 'Exact final V11 receipt required'}
$parent=Get-Content -Raw -LiteralPath (Resolve-Research $prior.path)|ConvertFrom-Json
if($parent.status-ne'passed'-or$parent.critical_checkpoint_pins.Count-ne109-or$parent.critical_checkpoint_total-ne109-or$parent.old_pins-ne1030-or$parent.declaration.sha256-ne$template.sha256){throw 'Actual109 inherited critical pins and1030 old pins required'}
$card=[ordered]@{prior_V11_final=$prior;protected_inventories=@($priorTemplate.protected_inventories);protected_pins=@($priorTemplate.protected_pins)}
$required=@('prior_live_snapshot','complete_mirror','source_manifest','source_zip','main_card','metadata_application','proposal_review','final_establishment','compiled_binary_mirror')
foreach($role in $required){if($null-eq$plan.$role){throw ('Missing actual role '+$role)};$card[$role]=Require-Pin $plan.$role}
if(@($plan.PSObject.Properties.Name|Where-Object{$_-in@('reporting_recovery_card','reporting_recovery_receipt','failure_mirror')}).Count){throw 'No invented current V12 recovery roles; V9 historical recovery remains inherited'}
$manifest=Get-Content -Raw -LiteralPath (Resolve-Research $card.source_manifest.path)|ConvertFrom-Json
$mirror=Get-Content -Raw -LiteralPath (Resolve-Research $card.complete_mirror.path)|ConvertFrom-Json
$main=Get-Content -Raw -LiteralPath (Resolve-Research $card.main_card.path)|ConvertFrom-Json
$established=Get-Content -Raw -LiteralPath (Resolve-Research $card.final_establishment.path)|ConvertFrom-Json
$applied=Get-Content -Raw -LiteralPath (Resolve-Research $card.metadata_application.path)|ConvertFrom-Json
$review=Get-Content -Raw -LiteralPath (Resolve-Research $card.proposal_review.path)|ConvertFrom-Json
if($main.model-ne'CPB-v12'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Frozen scientific V12 declaration required'}
if($manifest.kind-ne'causal_fresh_v12_source_capsule_v1'-or$manifest.files.Count-ne137-or$manifest.card_sha256-ne$card.main_card.sha256-or$manifest.attempt-notmatch'^attempt-[0-9]{3}$'){throw 'Actual admitted137-source/card/attempt identity required'}
if($mirror.kind-ne'causal_fresh_v12_container_to_local_evidence_mirror_v1'-or$mirror.status-ne'passed'-or$mirror.scope-ne'Complete'-or$mirror.attempt-ne$manifest.attempt-or$mirror.source_files_verified-ne$manifest.files.Count-or$mirror.card_sha256-ne$card.main_card.sha256-or$mirror.source_manifest_sha256-ne$card.source_manifest.sha256-or-not$mirror.all_mirrored_pt_ignored){throw 'Exact passed normal Complete mirror required'}
$expected=@('inference-0.pt','inference-2048.pt','training-512.pt','training-1024.pt','training-2048.pt')
$new5=@($mirror.files|Where-Object{$_.path-like'output/runs/causal-fresh-v12/run-001/checkpoints/*.pt'})
if($new5.Count-ne5-or(Compare-Object @($new5|ForEach-Object{Split-Path -Leaf $_.path}|Sort-Object) @($expected|Sort-Object))){throw 'Exactly five actual canonical V12 checkpoint files required before declaration'}
if($plan.new_checkpoint_pins.Count-ne5){throw 'Five independently reviewed actual checkpoint pins required'}
foreach($binding in $new5){
 $null=Require-Pin $binding;$matches=@($plan.new_checkpoint_pins|Where-Object{$_.path-eq$binding.path-and$_.bytes-eq$binding.bytes-and$_.sha256-eq$binding.sha256-and$_.container_path-eq$binding.container_path})
 if($matches.Count-ne1){throw 'Every actual checkpoint must match the reviewed plan and Complete exactly'}
}
if($established.status-ne'passed'-or$established.candidate-ne'CPB-v12.market-01'-or-not$established.measured_candidate_established-or-not$established.trust_admission_passed-or-not$established.full_coverage-or-not$established.initial_partial_native20_40_pairings_passed-or($established.initial_pairing_native_coordinate_slice-join',')-ne'20,40'-or$established.whole_native40_initial_equality_claimed-ne$false-or$established.initial_pairing_count-ne13-or$established.numeric_total-ne22-or$established.overall_winner-or$established.winner_selection_made){throw 'Full trust/partial native20:40 measured candidate required; no winner selection'}
if($established.actual_component_archive_fields-ne60-or$established.same_forward_temporal_fields-ne25-or$established.new_uniform_dispersion_trace_fields-ne4-or$established.uniform_dispersion_direct_score_adjoint-ne0-or$established.local_hidden_adjoint_array_available-ne$false-or$established.score_to_hidden_route_reconstructed-ne$false-or$established.hidden_reference_is_observed_gradient_parity-ne$false-or$established.all_eight_temporal_moment_subgroups_verified-ne$true-or$established.whole8_two4_squared_norms_and_element_counts_verified-ne$true-or$established.actual_first_last_local_moment_adjoints_verified-ne$true-or$established.full_encoder_gradient_or_global_norm_reconstructed-ne$false-or$established.common_score_bias_gauge_tight_parity_required-ne$false){throw 'Actual60/25 dual-support/three-local-adjoint scope;fixed-score hidden reference descriptive only'}
$dualSupportFlags=@('attention_mean_and_uniform_dispersion_full_expression_adjoints','uniform_dispersion_own_center','uniform_dispersion_score_independent','uniform_dispersion_hidden_gradient_live','attention_mean_score_gradient_live','dual_support_actual_trace')
foreach($flag in $dualSupportFlags){if($established.$flag-ne$true){throw ('Passed exact dual-support fixture flag required: '+$flag)}}
if($established.temporal_pooling_direct_kernel_calls-ne47-or$established.temporal_pooling_F64_finite_difference_coordinates-ne18-or$established.temporal_pooling_F32_finite_difference_coordinates-ne18-or$main.advancement.goal_protocol.sha256-ne'866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'-or$main.scientific_recipe_parent-ne'CPB-v11 source only;fresh weights/optimizer/statistics,not trained ancestry'){throw 'Frozen47/18/18,strongest866 protocol and V11 source-only lineage required'}
if($established.normal_prospective_source_admission.normal_quality_all_stages_passed-ne$true-or$established.normal_prospective_source_admission.scientific_model_head_or_training_replay-ne$false){throw 'Normal completed current quality without scientific replay required'}
if($applied.status-ne'passed'-or$applied.baseline_changed-or$applied.application_changed-or$review.status-ne'passed'){throw 'Actual reviewed and root-applied metadata required'}
$buildEntry=@($mirror.files|Where-Object{$_.path.EndsWith('/build-admission.json')})
if($buildEntry.Count-ne1){throw 'One actual admitted build required'}
$null=Require-Pin $buildEntry[0]
$build=Get-Content -Raw -LiteralPath (Resolve-Research $buildEntry[0].path)|ConvertFrom-Json
$binaryMirror=Get-Content -Raw -LiteralPath (Resolve-Research $card.compiled_binary_mirror.path)|ConvertFrom-Json
if($binaryMirror.kind-ne'cpb_v12_actual_build_binary_mirror_v1'-or$binaryMirror.status-ne'passed'-or$binaryMirror.source_files-ne$manifest.files.Count-or$binaryMirror.binaries-ne7-or$binaryMirror.translation_units-ne11-or$binaryMirror.files.Count-ne18){throw 'Actual mirror must bind manifest-derived sources and seven binaries plus eleven dependency files'}
if($binaryMirror.build_admission.path-ne$buildEntry[0].path-or$binaryMirror.build_admission.bytes-ne$buildEntry[0].bytes-or$binaryMirror.build_admission.sha256-ne$buildEntry[0].sha256){throw 'Separate binary mirror must bind exact host build-admission pin'}
$mirroredBinaries=@($binaryMirror.files|Where-Object{-not$_.path.EndsWith('.d')})
$dependencies=@($binaryMirror.files|Where-Object{$_.path.EndsWith('.d')})
$dependencyPins=@($build.dependency_closures.PSObject.Properties|ForEach-Object{$_.Value.file})
if($mirroredBinaries.Count-ne7-or$dependencies.Count-ne11-or$dependencyPins.Count-ne11-or@($binaryMirror.files.path|Sort-Object -Unique).Count-ne18-or@($binaryMirror.files.container_path|Sort-Object -Unique).Count-ne18){throw 'Exactly eighteen unique build evidence members required'}
foreach($binding in $binaryMirror.files){$null=Require-Pin $binding}
foreach($binding in $dependencies){$match=@($dependencyPins|Where-Object{$_.path-eq$binding.container_path});if($match.Count-ne1-or$match[0].bytes-ne$binding.bytes-or$match[0].sha256-ne$binding.sha256){throw 'Every dependency-file mirror must match the actual admitted translation-unit closure'}}
$binaries=@($plan.compiled_binaries)
if($binaries.Count-ne7-or$build.binaries.Count-ne7){throw 'Seven separately mirrored actual compiled binaries required'}
$expectedBinaryNames=@('test_causal_patch_bottleneck_mae_v12','test_cpb_v12_variance_gradient','test_cpb_v12_temporal_pooling','causal_fresh_v12_train','causal_fresh_v12_extract','causal_fresh_v12_market_read','causal_fresh_v12_synthetic_read')
foreach($set in @(@{items=$binaries;label='host'},@{items=@($build.binaries);label='build'},@{items=$mirroredBinaries;label='receipt'})){
 $names=@($set.items|ForEach-Object{Split-Path -Leaf $_.path})
 if(@($set.items.path|Sort-Object -Unique).Count-ne7-or@($names|Sort-Object -Unique).Count-ne7-or(Compare-Object @($names|Sort-Object) @($expectedBinaryNames|Sort-Object))){throw ('Exact admitted unique seven-binary set required: '+$set.label)}
}
foreach($binding in $binaries){
 $null=Require-Pin $binding;$match=@($build.binaries|Where-Object{(Split-Path -Leaf $_.path)-eq(Split-Path -Leaf $binding.path)});$mirrored=@($binaryMirror.files|Where-Object{$_.path-eq$binding.path})
 if($match.Count-ne1-or$match[0].bytes-ne$binding.bytes-or$match[0].sha256-ne$binding.sha256-or$mirrored.Count-ne1-or$mirrored[0].bytes-ne$binding.bytes-or$mirrored[0].sha256-ne$binding.sha256-or$mirrored[0].container_path-ne$match[0].path){throw 'Each host binary must match the separate receipt and actual build pin'}
}
# Current audit/join/finalization evidence is outside main Complete and must be explicit.
foreach($role in @('training_audit','quality_audit','establishment','finalization')){
 $items=@($plan.protected_inventories|Where-Object{$_.role-eq$role})
 if($items.Count-ne1){throw ('One completed current protected inventory required: '+$role)}
}
if(Compare-Object @($plan.completed_evidence_roots.PSObject.Properties.Name|Sort-Object) @('training_audit','quality_audit','establishment','finalization'|Sort-Object)){throw 'Four explicit completed host evidence roots required; no future path guesses'}
$completedRoots=[ordered]@{}
foreach($entry in $plan.completed_evidence_roots.PSObject.Properties){$directory=Resolve-Research $entry.Value;if(-not(Get-Item -LiteralPath $directory).PSIsContainer){throw 'Completed evidence root must be an actual directory'};$completedRoots[$entry.Name]=[IO.Path]::GetRelativePath($researchRoot,$directory).Replace([char]92,[char]47)}
foreach($inventory in $plan.protected_inventories){$null=Require-Pin $inventory.receipt;if(-not$inventory.inventory_field){throw 'Exact inventory field required'}}
$card['compiled_binaries']=$binaries;$card['compiled_dependency_files']=$dependencies;$card['new_checkpoint_pins']=$new5;$card['main_source_files']=$manifest.files.Count;$card['main_attempt']=$manifest.attempt
$card['concurrent_DEV_exclusions']=@($plan.concurrent_DEV_exclusions)
$card['completed_evidence_roots']=$completedRoots
$card['protected_inventories']=@($card.protected_inventories)+@($plan.protected_inventories)
$card['current_protected_inventories']=@($plan.protected_inventories|Where-Object{$_.role-in@('training_audit','quality_audit','establishment','finalization')})
$unique=[ordered]@{}
foreach($binding in @($card.protected_pins)+@($plan.protected_pins)){$repo=if($binding.repository){$binding.repository}else{'cuwacunu_embedding'};$unique[$repo+'|'+$binding.path+'|'+$binding.bytes+'|'+$binding.sha256]=$binding}
# Retain every inherited root plus the whole completed current preparation/evidence context.
$requiredRoots=@('code/research/causal_v9_retention_review','doc/causal_v9_retention_review',
 'code/research/causal_v8_v9_train_diagnosis','code/research/causal_v8_v9_train_diagnosis_audit','doc/causal_v8_v9_train_diagnosis',
 'output/engineering/causal-v9-retention-review','output/engineering/causal-v9-retention-source-preparation',
 'output/engineering/causal-v8-v9-train-diagnosis','output/engineering/causal-v8-v9-train-diagnosis-audit',
 'output/engineering/causal-v8-v9-train-diagnosis-audit-orchestration-preflight-repairs','output/engineering/causal-v8-v9-train-diagnosis-audit-preflight-repairs',
 'output/engineering/causal-v8-v9-train-diagnosis-card-preparation','output/engineering/causal-v8-v9-train-diagnosis-human-report',
 'output/engineering/causal-v8-v9-train-diagnosis-orchestration-preparation','output/engineering/causal-v8-v9-train-diagnosis-preparation',
 'code/research/causal_fresh_v12_preservation','code/research/causal_fresh_v12_training_audit','code/research/causal_fresh_v12_quality_audit',
 'code/research/causal_fresh_v12_establishment','code/research/causal_fresh_v12_finalization',
 'doc/causal_fresh_v12','doc/causal_fresh_v12_quality_audit','doc/causal_fresh_v12_finalization','doc/results/causal_fresh_v12',
 'output/engineering/causal-fresh-v12','output/runs/causal-fresh-v12','output/engineering/causal-fresh-v12-source-preparation',
 'output/engineering/causal-fresh-v12-protocol-preparation',
 'output/engineering/causal-fresh-v12-postmeasurement-source-preparation',
 'doc/causal_fresh_v12_training_audit','output/engineering/causal-fresh-v12-training-audit-workflow-source-preparation',
 'output/engineering/causal-fresh-v12-quality-audit-workflow-source-preparation')
foreach($relative in @($requiredRoots)+@($plan.additional_protected_roots)+@($completedRoots.Values)){
 $directory=Get-Item -LiteralPath (Resolve-Research $relative)
 if(-not$directory.PSIsContainer){throw 'Explicit completed physical directory required'}
 foreach($file in Get-ChildItem -LiteralPath $directory.FullName -Recurse -File){$binding=Pin $file.FullName;$unique['cuwacunu_embedding|'+$binding.path+'|'+$binding.bytes+'|'+$binding.sha256]=$binding}
}
foreach($relative in @('doc/clean_causal_encoder_search_goal_v9_retention.json','doc/clean_causal_encoder_search_goal_v9_retention.md','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.md','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md','doc/CAUSAL_FRESH_V12_RESULTS.md','doc/CAUSAL_FRESH_V12_DETAIL.md','doc/CAUSAL_FRESH_V12_REVIEW.md','doc/CAUSAL_FRESH_V12_CARD.json','doc/CAUSAL_FRESH_V12_CARD.md','code/encoders/causal_patch_bottleneck_mae/versions/CPB-v12.json','code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v12.market-01.json','doc/causal_fresh_v11/draft-metadata-amendment-001.json')){$binding=Pin $relative;$unique['cuwacunu_embedding|'+$binding.path+'|'+$binding.bytes+'|'+$binding.sha256]=$binding}
# This is explicitly inherited V11 documentary history, never a current V12 amendment.
$amendment=Get-Content -Raw -LiteralPath (Resolve-Research 'doc/causal_fresh_v11/draft-metadata-amendment-001.json')|ConvertFrom-Json
$null=Require-Pin $amendment.preserved_before
$card['preexecution_draft_amendment']=Pin 'doc/causal_fresh_v11/draft-metadata-amendment-001.json'
$card['preexecution_draft_amendment_scope']='inherited_V11_documentary_history_only_not_current_V12'
$card['preexecution_draft_amendment_before_snapshot']=Require-Pin $amendment.preserved_before
$card['protected_pins']=@($unique.Values)
$card['kind']='causal_fresh_v12_final_preservation_card_v1';$card['status']='frozen_admitted_after_actual_five_checkpoints_complete_trust_and_metadata_application';$card['authorized']=$true
$card['created_utc']=[DateTime]::UtcNow.ToString('o');$card['historical_critical_checkpoints']=109;$card['new_actual_checkpoints']=5;$card['expected_final_critical_checkpoints']=114
$card['current_reporting_recovery_required']=$false
$card['verifier']=Pin 'code/research/causal_fresh_v12_preservation/verify.ps1';$card['generator']=Pin 'code/research/causal_fresh_v12_preservation/prepare-card.ps1'
$card['input_plan']=$planPin;$card['parent_template']=$template;$card['source_workflow']=@($plan.source_workflow)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes((($card|ConvertTo-Json -Depth 90).Replace("`r`n","`n")+"`n"))
$stream=[IO.File]::Open($output,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Pin 'doc/causal_fresh_v12/preservation-card.json'|ConvertTo-Json

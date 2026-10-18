# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
# Complete physical FINAL V10 generator copy, adapted to normal V11 preservation.
# Metadata only; no verification, Docker, AST or scientific program is invoked here.
param([Parameter(Mandatory=$true)][string]$InputPlan,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InputPlanSHA256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedActualInputs,
 [Parameter(Mandatory=$true)][switch]$FinalMetadataApplied)
$ErrorActionPreference='Stop'
if(-not$RootReviewedActualInputs-or-not$FinalMetadataApplied){throw 'Actual independently/root-reviewed inputs and final metadata application required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$output=Join-Path $researchRoot 'doc/causal_fresh_v11/preservation-card.json'
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
if($plan.kind-ne'causal_fresh_v11_final_preservation_input_plan_v1'-or$plan.status-ne'root_reviewed_actual_inputs'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true){throw 'Explicit actual-input Boolean admission required'}
foreach($field in @('compiled_binaries','new_checkpoint_pins','concurrent_DEV_exclusions','protected_inventories','protected_pins','additional_protected_roots','completed_evidence_roots','source_workflow')){
 if($field-notin@($plan.PSObject.Properties.Name)-or$null-eq$plan.$field){throw ('Explicit field required, including empty arrays where declared: '+$field)}
}
# Require this owned workflow and all three exact data templates before copying metadata.
foreach($binding in $plan.source_workflow){$null=Require-Pin $binding}
$workflowExpected=@('verify.ps1','prepare-card.ps1','README.md','provenance.json','v8-preservation-card.template.json','v9-preservation-card.template.json','v10-preservation-card.template.json')
if($plan.source_workflow.Count-ne7-or(Compare-Object @($plan.source_workflow.path|Sort-Object) @($workflowExpected|ForEach-Object{'code/research/causal_fresh_v11_preservation/'+$_}|Sort-Object))){throw 'Seven exact owned workflow assets required'}
$self=@($plan.source_workflow|Where-Object{$_.path-eq'code/research/causal_fresh_v11_preservation/prepare-card.ps1'})
if($self.Count-ne1-or$self[0].sha256-ne(Pin $PSCommandPath).sha256){throw 'Reviewed generator self binding required'}
$template=Pin 'code/research/causal_fresh_v11_preservation/v10-preservation-card.template.json'
if($template.bytes-ne673975-or$template.sha256-ne'4a381d78ade3261cfcbd5ed0214a4c921ff239abb9235d3e4af37796deda7f7d'){throw 'Exact final V10 card data template required'}
$priorTemplate=Get-Content -Raw -LiteralPath (Resolve-Research $template.path)|ConvertFrom-Json
$prior=Pin 'doc/causal_fresh_v10/final-preservation.json'
if($prior.bytes-ne709732-or$prior.sha256-ne'fa6eb72bd45b9cb43d8dc7ea523ed70908eee072cc1c2e046e45995cd3a4c24d'){throw 'Exact final V10 receipt required'}
$parent=Get-Content -Raw -LiteralPath (Resolve-Research $prior.path)|ConvertFrom-Json
if($parent.status-ne'passed'-or$parent.critical_checkpoint_pins.Count-ne104-or$parent.critical_checkpoint_total-ne104-or$parent.old_pins-ne1030-or$parent.declaration.sha256-ne$template.sha256){throw 'Actual104 inherited critical pins and1030 old pins required'}
$card=[ordered]@{prior_V10_final=$prior;protected_inventories=@($priorTemplate.protected_inventories);protected_pins=@($priorTemplate.protected_pins)}
$required=@('prior_live_snapshot','complete_mirror','source_manifest','source_zip','main_card','metadata_application','proposal_review','final_establishment','compiled_binary_mirror')
foreach($role in $required){if($null-eq$plan.$role){throw ('Missing actual role '+$role)};$card[$role]=Require-Pin $plan.$role}
if(@($plan.PSObject.Properties.Name|Where-Object{$_-in@('reporting_recovery_card','reporting_recovery_receipt','failure_mirror')}).Count){throw 'No invented current V11 recovery roles; V9 historical recovery remains inherited'}
$manifest=Get-Content -Raw -LiteralPath (Resolve-Research $card.source_manifest.path)|ConvertFrom-Json
$mirror=Get-Content -Raw -LiteralPath (Resolve-Research $card.complete_mirror.path)|ConvertFrom-Json
$main=Get-Content -Raw -LiteralPath (Resolve-Research $card.main_card.path)|ConvertFrom-Json
$established=Get-Content -Raw -LiteralPath (Resolve-Research $card.final_establishment.path)|ConvertFrom-Json
$applied=Get-Content -Raw -LiteralPath (Resolve-Research $card.metadata_application.path)|ConvertFrom-Json
$review=Get-Content -Raw -LiteralPath (Resolve-Research $card.proposal_review.path)|ConvertFrom-Json
if($main.model-ne'CPB-v11'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Frozen scientific V11 declaration required'}
if($manifest.kind-ne'causal_fresh_v11_source_capsule_v1'-or$manifest.files.Count-le0-or$manifest.card_sha256-ne$card.main_card.sha256-or$manifest.attempt-notmatch'^attempt-[0-9]{3}$'){throw 'Actual admitted source/card/attempt identity required'}
if($mirror.kind-ne'causal_fresh_v11_container_to_local_evidence_mirror_v1'-or$mirror.status-ne'passed'-or$mirror.scope-ne'Complete'-or$mirror.attempt-ne$manifest.attempt-or$mirror.source_files_verified-ne$manifest.files.Count-or$mirror.card_sha256-ne$card.main_card.sha256-or$mirror.source_manifest_sha256-ne$card.source_manifest.sha256-or-not$mirror.all_mirrored_pt_ignored){throw 'Exact passed normal Complete mirror required'}
$expected=@('inference-0.pt','inference-2048.pt','training-512.pt','training-1024.pt','training-2048.pt')
$new5=@($mirror.files|Where-Object{$_.path-like'output/runs/causal-fresh-v11/run-001/checkpoints/*.pt'})
if($new5.Count-ne5-or(Compare-Object @($new5|ForEach-Object{Split-Path -Leaf $_.path}|Sort-Object) @($expected|Sort-Object))){throw 'Exactly five actual canonical V11 checkpoint files required before declaration'}
if($plan.new_checkpoint_pins.Count-ne5){throw 'Five independently reviewed actual checkpoint pins required'}
foreach($binding in $new5){
 $null=Require-Pin $binding;$matches=@($plan.new_checkpoint_pins|Where-Object{$_.path-eq$binding.path-and$_.bytes-eq$binding.bytes-and$_.sha256-eq$binding.sha256-and$_.container_path-eq$binding.container_path})
 if($matches.Count-ne1){throw 'Every actual checkpoint must match the reviewed plan and Complete exactly'}
}
if($established.status-ne'passed'-or$established.candidate-ne'CPB-v11.market-01'-or-not$established.measured_candidate_established-or-not$established.trust_admission_passed-or-not$established.full_coverage-or-not$established.initial_partial_native20_40_pairings_passed-or($established.initial_pairing_native_coordinate_slice-join',')-ne'20,40'-or$established.whole_native40_initial_equality_claimed-ne$false-or$established.initial_pairing_count-ne13-or$established.numeric_total-ne22-or$established.overall_winner-or$established.winner_selection_made){throw 'Full trust/partial native20:40 measured candidate required; no winner selection'}
if($established.actual_component_archive_fields-ne56-or$established.same_forward_temporal_fields-ne21-or$established.all_eight_temporal_moment_subgroups_verified-ne$true-or$established.whole8_two4_squared_norms_and_element_counts_verified-ne$true-or$established.actual_first_last_local_moment_adjoints_verified-ne$true-or$established.full_encoder_gradient_or_global_norm_reconstructed-ne$false-or$established.common_score_bias_gauge_tight_parity_required-ne$false){throw 'Actual saved moment/subgroup/three-local-adjoint scope required'}
if($established.normal_prospective_source_admission.normal_quality_all_stages_passed-ne$true-or$established.normal_prospective_source_admission.scientific_model_head_or_training_replay-ne$false){throw 'Normal completed current quality without scientific replay required'}
if($applied.status-ne'passed'-or$applied.baseline_changed-or$applied.application_changed-or$review.status-ne'passed'){throw 'Actual reviewed and root-applied metadata required'}
$buildEntry=@($mirror.files|Where-Object{$_.path.EndsWith('/build-admission.json')})
if($buildEntry.Count-ne1){throw 'One actual admitted build required'}
$null=Require-Pin $buildEntry[0]
$build=Get-Content -Raw -LiteralPath (Resolve-Research $buildEntry[0].path)|ConvertFrom-Json
$binaryMirror=Get-Content -Raw -LiteralPath (Resolve-Research $card.compiled_binary_mirror.path)|ConvertFrom-Json
if($binaryMirror.kind-ne'cpb_v11_actual_build_binary_mirror_v1'-or$binaryMirror.status-ne'passed'-or$binaryMirror.source_files-ne$manifest.files.Count-or$binaryMirror.binaries-ne7-or$binaryMirror.translation_units-ne11-or$binaryMirror.files.Count-ne18){throw 'Actual mirror must bind manifest-derived sources and seven binaries plus eleven dependency files'}
if($binaryMirror.build_admission.path-ne$buildEntry[0].path-or$binaryMirror.build_admission.bytes-ne$buildEntry[0].bytes-or$binaryMirror.build_admission.sha256-ne$buildEntry[0].sha256){throw 'Separate binary mirror must bind exact host build-admission pin'}
$mirroredBinaries=@($binaryMirror.files|Where-Object{-not$_.path.EndsWith('.d')})
$dependencies=@($binaryMirror.files|Where-Object{$_.path.EndsWith('.d')})
$dependencyPins=@($build.dependency_closures.PSObject.Properties|ForEach-Object{$_.Value.file})
if($mirroredBinaries.Count-ne7-or$dependencies.Count-ne11-or$dependencyPins.Count-ne11-or@($binaryMirror.files.path|Sort-Object -Unique).Count-ne18-or@($binaryMirror.files.container_path|Sort-Object -Unique).Count-ne18){throw 'Exactly eighteen unique build evidence members required'}
foreach($binding in $binaryMirror.files){$null=Require-Pin $binding}
foreach($binding in $dependencies){$match=@($dependencyPins|Where-Object{$_.path-eq$binding.container_path});if($match.Count-ne1-or$match[0].bytes-ne$binding.bytes-or$match[0].sha256-ne$binding.sha256){throw 'Every dependency-file mirror must match the actual admitted translation-unit closure'}}
$binaries=@($plan.compiled_binaries)
if($binaries.Count-ne7-or$build.binaries.Count-ne7){throw 'Seven separately mirrored actual compiled binaries required'}
$expectedBinaryNames=@('test_causal_patch_bottleneck_mae_v11','test_cpb_v11_variance_gradient','test_cpb_v11_temporal_pooling','causal_fresh_v11_train','causal_fresh_v11_extract','causal_fresh_v11_market_read','causal_fresh_v11_synthetic_read')
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
 'code/research/causal_fresh_v11_preservation','code/research/causal_fresh_v11_training_audit','code/research/causal_fresh_v11_quality_audit',
 'code/research/causal_fresh_v11_establishment','code/research/causal_fresh_v11_finalization',
 'doc/causal_fresh_v11','doc/causal_fresh_v11_quality_audit','doc/causal_fresh_v11_finalization','doc/results/causal_fresh_v11',
 'output/engineering/causal-fresh-v11','output/engineering/causal-fresh-v11-family-preparation',
 'output/engineering/causal-fresh-v11-protocol-preparation',
 'output/engineering/causal-fresh-v11-establishment-source-preparation',
 'doc/causal_fresh_v11_training_audit','output/engineering/causal-fresh-v11-preservation-source-preparation')
foreach($relative in @($requiredRoots)+@($plan.additional_protected_roots)+@($completedRoots.Values)){
 $directory=Get-Item -LiteralPath (Resolve-Research $relative)
 if(-not$directory.PSIsContainer){throw 'Explicit completed physical directory required'}
 foreach($file in Get-ChildItem -LiteralPath $directory.FullName -Recurse -File){$binding=Pin $file.FullName;$unique['cuwacunu_embedding|'+$binding.path+'|'+$binding.bytes+'|'+$binding.sha256]=$binding}
}
foreach($relative in @('doc/clean_causal_encoder_search_goal_v9_retention.json','doc/clean_causal_encoder_search_goal_v9_retention.md','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.md','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md','doc/CAUSAL_FRESH_V11_RESULTS.md','doc/CAUSAL_FRESH_V11_DETAIL.md','doc/CAUSAL_FRESH_V11_REVIEW.md','doc/causal_fresh_v11/draft-metadata-amendment-001.json')){$binding=Pin $relative;$unique['cuwacunu_embedding|'+$binding.path+'|'+$binding.bytes+'|'+$binding.sha256]=$binding}
$amendment=Get-Content -Raw -LiteralPath (Resolve-Research 'doc/causal_fresh_v11/draft-metadata-amendment-001.json')|ConvertFrom-Json
$null=Require-Pin $amendment.preserved_before
$card['preexecution_draft_amendment']=Pin 'doc/causal_fresh_v11/draft-metadata-amendment-001.json'
$card['preexecution_draft_amendment_before_snapshot']=Require-Pin $amendment.preserved_before
$card['protected_pins']=@($unique.Values)
$card['kind']='causal_fresh_v11_final_preservation_card_v1';$card['status']='frozen_admitted_after_actual_five_checkpoints_complete_trust_and_metadata_application';$card['authorized']=$true
$card['created_utc']=[DateTime]::UtcNow.ToString('o');$card['historical_critical_checkpoints']=104;$card['new_actual_checkpoints']=5;$card['expected_final_critical_checkpoints']=109
$card['current_reporting_recovery_required']=$false
$card['verifier']=Pin 'code/research/causal_fresh_v11_preservation/verify.ps1';$card['generator']=Pin 'code/research/causal_fresh_v11_preservation/prepare-card.ps1'
$card['input_plan']=$planPin;$card['parent_template']=$template;$card['source_workflow']=@($plan.source_workflow)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes((($card|ConvertTo-Json -Depth 90).Replace("`r`n","`n")+"`n"))
$stream=[IO.File]::Open($output,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Pin 'doc/causal_fresh_v11/preservation-card.json'|ConvertTo-Json

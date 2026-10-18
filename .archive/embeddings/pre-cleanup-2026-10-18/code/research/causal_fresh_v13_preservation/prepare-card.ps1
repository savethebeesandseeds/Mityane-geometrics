# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
# Complete physical FINAL V12 generator copy, adapted to normal V13 preservation.
# Metadata only; no verification, Docker, AST or scientific program is invoked here.
param([Parameter(Mandatory=$true)][string]$InputPlan,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InputPlanSHA256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedActualInputs,
 [Parameter(Mandatory=$true)][switch]$FinalMetadataApplied)
$ErrorActionPreference='Stop'
if(-not$RootReviewedActualInputs-or-not$FinalMetadataApplied){throw 'Actual independently/root-reviewed inputs and final metadata application required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$output=Join-Path $researchRoot 'doc/causal_fresh_v13/preservation-card.json'
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
if($plan.kind-ne'causal_fresh_v13_final_preservation_input_plan_v1'-or$plan.status-ne'root_reviewed_actual_inputs'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true){throw 'Explicit actual-input Boolean admission required'}
$proposalPin=Require-Pin $plan.proposed_input_plan
$proposal=Get-Content -Raw -LiteralPath (Resolve-Research $proposalPin.path)|ConvertFrom-Json
$peerPin=Require-Pin $plan.independent_input_plan_review
$peer=Get-Content -Raw -LiteralPath (Resolve-Research $peerPin.path)|ConvertFrom-Json
if($proposal.kind-cne$plan.kind-or$proposal.authorized-isnot[bool]-or$proposal.authorized-ne$false-or$proposal.status-cne'PROPOSED_pending_independent_and_root_review'){throw 'Immutable unauthorized PROPOSED actual-input plan required'}
if($peer.kind-cne'cpb_v13_final_preservation_actual_input_plan_independent_review_v1'-or$peer.status-cne'passed'-or$peer.metadata_only-isnot[bool]-or-not$peer.metadata_only-or$peer.independent_of_plan_author-isnot[bool]-or-not$peer.independent_of_plan_author-or$peer.actual_plan.path-cne$proposalPin.path-or$peer.actual_plan.bytes-ne$proposalPin.bytes-or$peer.actual_plan.sha256-cne$proposalPin.sha256){throw 'Nonself actual-plan review must bind false PROPOSED bytes,never futureADMITTED bytes'}
if($plan.separate_root_admission.proposed_input_plan_sha256-cne$proposalPin.sha256-or$plan.separate_root_admission.independent_review_sha256-cne$peerPin.sha256-or-not$plan.separate_root_admission.admitted_utc){throw 'Separate root admission binds exact proposal/peer pins and timestamp'}
$admissionFields=@('proposed_input_plan','independent_input_plan_review','separate_root_admission','admitted_utc')
$proposalKeys=@($proposal.PSObject.Properties.Name)
foreach($key in @($plan.PSObject.Properties.Name)){if($key-notin($proposalKeys+$admissionFields)){throw 'Only admission metadata may be added to proposal'}}
foreach($key in $proposalKeys){if($key-in@('authorized','status')){continue};if($key-notin@($plan.PSObject.Properties.Name)-or(($plan.$key|ConvertTo-Json -Depth 90 -Compress)-cne($proposal.$key|ConvertTo-Json -Depth 90 -Compress))){throw ('Frozen proposal/source/input invariant changed: '+$key)}}
foreach($field in @('compiled_binaries','new_checkpoint_pins','concurrent_DEV_exclusions','protected_inventories','protected_pins','additional_protected_roots','completed_evidence_roots','source_workflow')){
 if($field-notin@($plan.PSObject.Properties.Name)-or$null-eq$plan.$field){throw ('Explicit field required, including empty arrays where declared: '+$field)}
}
# Require this owned workflow and all five exact historical DATA templates before copying metadata.
foreach($binding in $plan.source_workflow){$null=Require-Pin $binding}
$workflowExpected=@('verify.ps1','prepare-card.ps1','README.md','provenance.json','v8-preservation-card.template.json','v9-preservation-card.template.json','v10-preservation-card.template.json','v11-preservation-card.template.json','v12-preservation-card.template.json')
if($plan.source_workflow.Count-ne9-or(Compare-Object @($plan.source_workflow.path|Sort-Object) @($workflowExpected|ForEach-Object{'code/research/causal_fresh_v13_preservation/'+$_}|Sort-Object))){throw 'Nine exact owned workflow assets required'}
$self=@($plan.source_workflow|Where-Object{$_.path-eq'code/research/causal_fresh_v13_preservation/prepare-card.ps1'})
if($self.Count-ne1-or$self[0].sha256-ne(Pin $PSCommandPath).sha256){throw 'Reviewed generator self binding required'}
$template=Pin 'code/research/causal_fresh_v13_preservation/v12-preservation-card.template.json'
if($template.bytes-ne1518070-or$template.sha256-ne'a92bffc4970dd086a8af00060805e8b8b01fcf1d902ff12ba0f4cd05b5ab95e9'){throw 'Exact final V12 card DATA template required;no historical execution'}
$priorTemplate=Get-Content -Raw -LiteralPath (Resolve-Research $template.path)|ConvertFrom-Json
$prior=Pin 'doc/causal_fresh_v12/final-preservation.json'
if($prior.bytes-ne1551705-or$prior.sha256-ne'db4e5f31737688e041784bd20d53747f10fc58b1e189041ca45bc2c2846fded7'){throw 'Exact final V12 receipt required'}
$parent=Get-Content -Raw -LiteralPath (Resolve-Research $prior.path)|ConvertFrom-Json
if($parent.status-ne'passed'-or$parent.critical_checkpoint_pins.Count-ne114-or$parent.critical_checkpoint_total-ne114-or$parent.old_pins-ne1030-or$parent.declaration.sha256-ne$template.sha256){throw 'Actual114 inherited critical pins and1030 old pins required'}
$card=[ordered]@{prior_V12_final=$prior;protected_inventories=@($priorTemplate.protected_inventories);protected_pins=@($priorTemplate.protected_pins)}
$required=@('prior_live_snapshot','complete_mirror','source_manifest','source_zip','main_card','metadata_application','proposal_review','final_establishment','compiled_binary_mirror')
foreach($role in $required){if($null-eq$plan.$role){throw ('Missing actual role '+$role)};$card[$role]=Require-Pin $plan.$role}
if(@($plan.PSObject.Properties.Name|Where-Object{$_-in@('reporting_recovery_card','reporting_recovery_receipt','failure_mirror')}).Count){throw 'No invented current V13 recovery roles; V9 historical recovery remains inherited'}
$manifest=Get-Content -Raw -LiteralPath (Resolve-Research $card.source_manifest.path)|ConvertFrom-Json
$mirror=Get-Content -Raw -LiteralPath (Resolve-Research $card.complete_mirror.path)|ConvertFrom-Json
$main=Get-Content -Raw -LiteralPath (Resolve-Research $card.main_card.path)|ConvertFrom-Json
$established=Get-Content -Raw -LiteralPath (Resolve-Research $card.final_establishment.path)|ConvertFrom-Json
$applied=Get-Content -Raw -LiteralPath (Resolve-Research $card.metadata_application.path)|ConvertFrom-Json
$review=Get-Content -Raw -LiteralPath (Resolve-Research $card.proposal_review.path)|ConvertFrom-Json
if($main.model-ne'CPB-v13'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Frozen scientific V13 declaration required'}
if($manifest.kind-ne'causal_fresh_v13_source_capsule_v1'-or$manifest.files.Count-ne143-or$manifest.card_sha256-ne$card.main_card.sha256-or$manifest.attempt-notmatch'^attempt-[0-9]{3}$'){throw 'Actual admitted143-source/card/attempt identity required'}
if($mirror.kind-ne'causal_fresh_v13_container_to_local_evidence_mirror_v1'-or$mirror.status-ne'passed'-or$mirror.scope-ne'Complete'-or$mirror.attempt-ne$manifest.attempt-or$mirror.source_files_verified-ne$manifest.files.Count-or$mirror.card_sha256-ne$card.main_card.sha256-or$mirror.source_manifest_sha256-ne$card.source_manifest.sha256-or-not$mirror.all_mirrored_pt_ignored){throw 'Exact passed normal Complete mirror required'}
$expected=@('inference-0.pt','inference-2048.pt','training-512.pt','training-1024.pt','training-2048.pt')
$new5=@($mirror.files|Where-Object{$_.path-like'output/runs/causal-fresh-v13/run-001/checkpoints/*.pt'})
if($new5.Count-ne5-or(Compare-Object @($new5|ForEach-Object{Split-Path -Leaf $_.path}|Sort-Object) @($expected|Sort-Object))){throw 'Exactly five actual canonical V13 checkpoint files required before declaration'}
if($plan.new_checkpoint_pins.Count-ne5){throw 'Five independently reviewed actual checkpoint pins required'}
foreach($binding in $new5){
 $null=Require-Pin $binding;$matches=@($plan.new_checkpoint_pins|Where-Object{$_.path-eq$binding.path-and$_.bytes-eq$binding.bytes-and$_.sha256-eq$binding.sha256-and$_.container_path-eq$binding.container_path})
 if($matches.Count-ne1){throw 'Every actual checkpoint must match the reviewed plan and Complete exactly'}
}
if($established.status-ne'passed'-or$established.candidate-ne'CPB-v13.market-01'-or-not$established.measured_candidate_established-or-not$established.trust_admission_passed-or-not$established.full_coverage-or-not$established.initial_partial_native20_40_pairings_passed-or($established.initial_pairing_native_coordinate_slice-join',')-ne'20,40'-or$established.whole_native40_initial_equality_claimed-ne$false-or$established.initial_pairing_count-ne13-or$established.numeric_total-ne22-or$established.overall_winner-or$established.winner_selection_made){throw 'Full trust/partial native20:40 measured candidate required; no winner selection'}
if($established.actual_component_archive_fields-ne60-or$established.same_forward_temporal_fields-ne25-or$established.new_regularized_dispersion_trace_fields-ne4-or$established.regularized_sqrt_uses_actual_saved_root-ne$true-or$established.attention_dispersion_score_and_position_routes_live-ne$true-or$established.raw_root_absent_offset_exception-ne$true-or$established.local_hidden_adjoint_array_available-ne$false-or$established.score_to_hidden_route_reconstructed-ne$false-or$established.hidden_reference_is_observed_gradient_parity-ne$false-or$established.all_eight_temporal_moment_subgroups_verified-ne$true-or$established.whole8_two4_squared_norms_and_element_counts_verified-ne$true-or$established.actual_first_last_local_moment_adjoints_verified-ne$true-or$established.full_encoder_gradient_or_global_norm_reconstructed-ne$false-or$established.common_score_bias_gauge_tight_parity_required-ne$false){throw 'Actual60/25 regularized-root/three-local-adjoint scope;nohidden/globalnorm/unmatcheddecoder reconstruction'}
$proofFlags=@('mean4_regularized_stddev4_export_layout','full_attention_dependent_regularized_moment_adjoints','live_dispersion_score_gradient','raw_variance_root_dispersion_trace','zero_tiny_large_analytic_dispersion_adjoints','supported_nonfinite_and_variance_overflow_refused','absent_local_huge_singleton_sanitized','captured_F32_residual_correction','raw_root_absent_offset_exception')
foreach($flag in $proofFlags){if($established.$flag-isnot[bool]-or-not$established.$flag){throw ('Passed exact regularized-stddev fixture flag required: '+$flag)}}
if($established.temporal_pooling_direct_kernel_calls-ne51-or$established.temporal_pooling_F64_finite_difference_coordinates-ne18-or$established.temporal_pooling_F32_finite_difference_coordinates-ne18-or$established.temporal_pooling_isolated_autograd_grad_calls-ne4-or$established.engineering_typed_negative_cases-ne636-or$main.advancement.goal_protocol.sha256-ne'866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'-or$main.scientific_recipe_parent-ne'CPB-v11 source only;fresh weights/optimizer/statistics,not trained ancestry'){throw 'Frozen51/18/18/4/636,strongest866 protocol and V11 SOURCE-only lineage required'}
foreach($record in @($main.mechanism,$main.training,$established)){if($record.temporal_pooling_policy-cne'original-position-four-bin-half-mixture-mean4-regularized-stddev4-v1'-or$record.temporal_moment_policy-cne'componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1'-or$record.temporal_dispersion_transform_policy-cne'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'-or$record.temporal_dispersion_regularizer-ne0.0009765625){throw 'Frozen actual V13 regularized-stddev/tau fields must agree'}}
if($established.normal_prospective_source_admission.normal_quality_all_stages_passed-ne$true-or$established.normal_prospective_source_admission.scientific_model_head_or_training_replay-ne$false){throw 'Normal completed current quality without scientific replay required'}
if($applied.status-ne'passed'-or$applied.baseline_changed-or$applied.application_changed-or$review.status-ne'passed'){throw 'Actual reviewed and root-applied metadata required'}
$buildEntry=@($mirror.files|Where-Object{$_.path.EndsWith('/build-admission.json')})
if($buildEntry.Count-ne1){throw 'One actual admitted build required'}
$null=Require-Pin $buildEntry[0]
$build=Get-Content -Raw -LiteralPath (Resolve-Research $buildEntry[0].path)|ConvertFrom-Json
$binaryMirror=Get-Content -Raw -LiteralPath (Resolve-Research $card.compiled_binary_mirror.path)|ConvertFrom-Json
if($binaryMirror.kind-ne'cpb_v13_actual_build_binary_mirror_v1'-or$binaryMirror.status-ne'passed'-or$binaryMirror.source_files-ne$manifest.files.Count-or$binaryMirror.binaries-ne7-or$binaryMirror.translation_units-ne11-or$binaryMirror.files.Count-ne18){throw 'Actual mirror must bind manifest-derived sources and seven binaries plus eleven dependency files'}
if($binaryMirror.build_admission.path-ne$buildEntry[0].path-or$binaryMirror.build_admission.bytes-ne$buildEntry[0].bytes-or$binaryMirror.build_admission.sha256-ne$buildEntry[0].sha256){throw 'Separate binary mirror must bind exact host build-admission pin'}
$mirroredBinaries=@($binaryMirror.files|Where-Object{-not$_.path.EndsWith('.d')})
$dependencies=@($binaryMirror.files|Where-Object{$_.path.EndsWith('.d')})
$dependencyPins=@($build.dependency_closures.PSObject.Properties|ForEach-Object{$_.Value.file})
if($mirroredBinaries.Count-ne7-or$dependencies.Count-ne11-or$dependencyPins.Count-ne11-or@($binaryMirror.files.path|Sort-Object -Unique).Count-ne18-or@($binaryMirror.files.container_path|Sort-Object -Unique).Count-ne18){throw 'Exactly eighteen unique build evidence members required'}
foreach($binding in $binaryMirror.files){$null=Require-Pin $binding}
foreach($binding in $dependencies){$match=@($dependencyPins|Where-Object{$_.path-eq$binding.container_path});if($match.Count-ne1-or$match[0].bytes-ne$binding.bytes-or$match[0].sha256-ne$binding.sha256){throw 'Every dependency-file mirror must match the actual admitted translation-unit closure'}}
$binaries=@($plan.compiled_binaries)
if($binaries.Count-ne7-or$build.binaries.Count-ne7){throw 'Seven separately mirrored actual compiled binaries required'}
$expectedBinaryNames=@('test_causal_patch_bottleneck_mae_v13','test_cpb_v13_variance_gradient','test_cpb_v13_temporal_pooling','causal_fresh_v13_train','causal_fresh_v13_extract','causal_fresh_v13_market_read','causal_fresh_v13_synthetic_read')
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
 'code/research/causal_fresh_v13_preservation','code/research/causal_fresh_v13_training_audit','code/research/causal_fresh_v13_quality_audit',
 'code/research/causal_fresh_v13_establishment','code/research/causal_fresh_v13_finalization',
 'doc/causal_fresh_v13','doc/causal_fresh_v13_quality_audit','doc/causal_fresh_v13_finalization','doc/results/causal_fresh_v13',
 'output/engineering/causal-fresh-v13','output/runs/causal-fresh-v13','output/engineering/causal-fresh-v13-source-preparation',
 'output/engineering/causal-fresh-v13-protocol-preparation',
 'output/engineering/causal-fresh-v13-postmeasurement-source-preparation',
 'doc/causal_fresh_v13_training_audit','output/engineering/causal-fresh-v13-training-audit-workflow-source-preparation',
 'output/engineering/causal-fresh-v13-quality-audit-workflow-source-preparation',
 'output/engineering/causal-fresh-v13-preservation-source-preparation',
 'output/engineering/causal-fresh-v13-finalization-source-preparation',
 'code/research/causal_v11_v12_train_diagnosis','code/research/causal_v11_v12_train_diagnosis_audit',
 'code/research/causal_v11_v12_train_diagnosis_audit_repair_001','doc/causal_v11_v12_train_diagnosis',
 'output/engineering/causal-v11-v12-train-diagnosis','output/engineering/causal-v11-v12-train-diagnosis-audit')
foreach($relative in @($requiredRoots)+@($plan.additional_protected_roots)+@($completedRoots.Values)){
 $directory=Get-Item -LiteralPath (Resolve-Research $relative)
 if(-not$directory.PSIsContainer){throw 'Explicit completed physical directory required'}
 foreach($file in Get-ChildItem -LiteralPath $directory.FullName -Recurse -File){$binding=Pin $file.FullName;$unique['cuwacunu_embedding|'+$binding.path+'|'+$binding.bytes+'|'+$binding.sha256]=$binding}
}
foreach($relative in @('doc/clean_causal_encoder_search_goal_v9_retention.json','doc/clean_causal_encoder_search_goal_v9_retention.md','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.md','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md','doc/CAUSAL_FRESH_V13_RESULTS.md','doc/CAUSAL_FRESH_V13_DETAIL.md','doc/CAUSAL_FRESH_V13_REVIEW.md','doc/CAUSAL_FRESH_V13_CARD.json','doc/CAUSAL_FRESH_V13_CARD.md','code/encoders/causal_patch_bottleneck_mae/versions/CPB-v13.json','code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v13.market-01.json','doc/causal_fresh_v11/draft-metadata-amendment-001.json')){$binding=Pin $relative;$unique['cuwacunu_embedding|'+$binding.path+'|'+$binding.bytes+'|'+$binding.sha256]=$binding}
# This is explicitly inherited V11 documentary history, never a current V13 amendment.
$amendment=Get-Content -Raw -LiteralPath (Resolve-Research 'doc/causal_fresh_v11/draft-metadata-amendment-001.json')|ConvertFrom-Json
$null=Require-Pin $amendment.preserved_before
$card['preexecution_draft_amendment']=Pin 'doc/causal_fresh_v11/draft-metadata-amendment-001.json'
$card['preexecution_draft_amendment_scope']='inherited_V11_documentary_history_only_not_current_V13'
$card['preexecution_draft_amendment_before_snapshot']=Require-Pin $amendment.preserved_before
$card['protected_pins']=@($unique.Values)
$card['kind']='causal_fresh_v13_final_preservation_card_v1';$card['status']='frozen_admitted_after_actual_five_checkpoints_complete_trust_and_metadata_application';$card['authorized']=$true
$card['created_utc']=[DateTime]::UtcNow.ToString('o');$card['historical_critical_checkpoints']=114;$card['new_actual_checkpoints']=5;$card['expected_final_critical_checkpoints']=119
$card['current_reporting_recovery_required']=$false
$card['verifier']=Pin 'code/research/causal_fresh_v13_preservation/verify.ps1';$card['generator']=Pin 'code/research/causal_fresh_v13_preservation/prepare-card.ps1'
$card['input_plan']=$planPin;$card['proposed_input_plan']=$proposalPin;$card['independent_input_plan_review']=$peerPin;$card['parent_template']=$template;$card['source_workflow']=@($plan.source_workflow)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes((($card|ConvertTo-Json -Depth 90).Replace("`r`n","`n")+"`n"))
$stream=[IO.File]::Open($output,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Pin 'doc/causal_fresh_v13/preservation-card.json'|ConvertTo-Json

# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Owned complete prior workflow copy, simplified after exact whole-parent preservation.
# Host metadata/hash only; no Docker, Python, archive values or diagnosis invocation.
param(
 [Parameter(Mandatory=$true)][ValidateSet('Propose','Admit')][string]$Phase,
 [string]$ContextPlan='', [string]$ContextPlanSha256='',
 [string]$SourceReview='', [string]$SourceReviewSha256='',
 [string]$ProposedCardSha256='',
 [Parameter(Mandatory=$true)][switch]$RootReviewedCompletedContext
)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$doc='doc/causal_v11_v12_train_diagnosis/'
$sourceRoot='/opt/cuwacunu/research/causal-v11-v12-train-diagnosis/source'
$proposed='doc/CAUSAL_V11_V12_TRAIN_DIAGNOSIS_CARD.PROPOSED.json'
$admitted='doc/CAUSAL_V11_V12_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'
function Require([bool]$ok,[string]$why){if(-not$ok){throw $why}}
function Host-File([string]$relative,[string]$repository='cuwacunu_embedding'){
 Require ($repository-in@('cuwacunu_embedding','cuwacunu_dev')-and-not[IO.Path]::IsPathRooted($relative)-and$relative-notmatch'[\\:]'-and@($relative.Split('/')|Where-Object{$_-in@('','..','.')} ).Count-eq0) 'Explicit bounded repository-relative path'
 $base=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repository));$path=[IO.Path]::GetFullPath((Join-Path $base $relative))
 Require ($path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'No repository escape'
 $walk=$path;while($walk-ne$base){$item=Get-Item -LiteralPath $walk;Require (-not($item.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'No linked file/ancestor';$walk=[IO.Path]::GetDirectoryName($walk)}
 Require (-not(Get-Item -LiteralPath $path).PSIsContainer) 'Physical regular file';return $path
}
function Pin([string]$relative,[string]$repository='cuwacunu_embedding'){$p=Host-File $relative $repository;[ordered]@{path=$relative;bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Same($a,$b){$a.bytes-eq$b.bytes-and$a.sha256-ceq$b.sha256}
function Equal($a,$b){($a|ConvertTo-Json -Depth 80 -Compress)-ceq($b|ConvertTo-Json -Depth 80 -Compress)}
function Exact($a,$b,[string]$why){Require (@($a).Count-eq@($b).Count-and@($a|Sort-Object -Unique).Count-eq@($b).Count-and-not(Compare-Object @($a|Sort-Object) @($b|Sort-Object))) $why}
function Verify($row){Require ($row.bytes-gt0-and$row.sha256-cmatch'^[0-9a-f]{64}$') 'Positive typed binding';Require (Same (Pin $row.host_path $row.host_repository) $row) ('Changed binding '+$row.id)}
function Read-Binding($row){Verify $row;Get-Content -LiteralPath (Host-File $row.host_path $row.host_repository) -Raw|ConvertFrom-Json}
function Write-New([string]$relative,[string]$text){$stream=[IO.File]::Open((Join-Path $researchRoot $relative),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write);try{$b=[Text.UTF8Encoding]::new($false).GetBytes($text.Replace([string][char]13,'')+[char]10);$stream.Write($b,0,$b.Length)}finally{$stream.Dispose()}}
function Check-Review($pin){
 Require (Same (Pin $pin.path) $pin) 'Exact externally reviewed source review'
 $review=Get-Content -LiteralPath (Host-File $pin.path) -Raw|ConvertFrom-Json
 Require ($review.kind-eq'causal_v11_v12_diagnosis_source_root_review_v1'-and$review.status-eq'passed_source_review'-and$review.execution_admitted-is[bool]-and-not$review.execution_admitted) 'Combined source-only root review'
 $main=@('archive_reader.py','diagnose.py','schema.py','README.md','provenance.json')|ForEach-Object{'code/research/causal_v11_v12_train_diagnosis/'+$_}
 $audit=@('archive_reader.py','audit.py','component_math.py','README.md','provenance.json')|ForEach-Object{'code/research/causal_v11_v12_train_diagnosis_audit/'+$_}
 $workflow=@('prepare-card.ps1','prepare-preflight.ps1','run-diagnosis.ps1','verify-completion.ps1','orchestration-provenance.json','audit-card-generator.ps1','audit-stage.ps1','audit-execute.py','audit-mirror.ps1','audit-orchestration-provenance.json')|ForEach-Object{$doc+$_}
 Exact @($review.scientific_files.path) @($main+$audit) 'Exactly ten reviewed scientific files'
 Exact @($review.workflow_files.path) @($workflow) 'Exactly ten reviewed workflow files'
 foreach($row in @($review.scientific_files)+@($review.workflow_files)+@($review.design)+@($review.parent_preservation)){Require (Same (Pin $row.path) $row) 'Current reviewed source/design/parent bytes'}
 Require ($review.design.path-eq($doc+'design-inventory-001.PROPOSED.json')-and$review.parent_preservation.path-eq($doc+'parents-preservation.json')) 'Current design and whole-parent authority paths'
 return $review
}
Require ([bool]$RootReviewedCompletedContext) 'Root review required'
if($Phase-eq'Admit'){
 Require ($ProposedCardSha256-cmatch'^[0-9a-f]{64}$') 'External exact proposed SHA'
 $prior=Pin $proposed;Require ($prior.sha256-ceq$ProposedCardSha256) 'External proposed SHA before parse'
 $card=Get-Content -LiteralPath (Host-File $proposed) -Raw|ConvertFrom-Json
 Require ($card.kind-eq'cpb_v11_v12_saved_train_diagnosis_card_v1'-and$card.stage-eq'prospective_before_diagnostic_arithmetic'-and$card.execution_admitted-is[bool]-and-not$card.execution_admitted) 'Preserved false proposed card'
 $null=Check-Review $card.scientific_source_review
 Require (Same (Pin $card.completed_context_plan.path) $card.completed_context_plan) 'Completed metadata plan remains exact before admission'
 foreach($row in @($card.allowed_inputs)+@($card.hash_only_guards)+@($card.context_inputs)+@($card.preservation_pins)+@($card.pre_execution_preservation)){Verify $row}
 $card.execution_admitted=$true;$card|Add-Member -NotePropertyName separate_root_admission -NotePropertyValue ([ordered]@{prospective_card=$prior;source_review=$card.scientific_source_review;admitted_utc=[DateTime]::UtcNow.ToString('o');separate_Stage_Schema_Run_admissions_required=$true})
 Write-New $admitted ($card|ConvertTo-Json -Depth 80)
 Write-New ($admitted.Replace('.json','.md')) ('# Saved TRAIN V11/V12 diagnosis admission'+[char]10+[char]10+'Separate admitted copy of proposed '+$prior.sha256+'. V12-only saved arithmetic; four exact audited V11 subtrees. V3 remains selected. Actual admitted card review and separate Stage/Schema/Run admissions remain required.')
 Pin $admitted|ConvertTo-Json;return
}
Require ($ContextPlan-and$ContextPlanSha256-cmatch'^[0-9a-f]{64}$'-and$SourceReview-and$SourceReviewSha256-cmatch'^[0-9a-f]{64}$') 'External completed-plan and combined source-review paths/SHA'
$planPin=Pin $ContextPlan;$reviewPin=Pin $SourceReview
Require ($planPin.sha256-ceq$ContextPlanSha256-and$reviewPin.sha256-ceq$SourceReviewSha256) 'External metadata hashes before parse'
$review=Check-Review $reviewPin;$inventoryPin=$review.design
$inventory=Get-Content -LiteralPath (Host-File $inventoryPin.path) -Raw|ConvertFrom-Json
$plan=Get-Content -LiteralPath (Host-File $ContextPlan) -Raw|ConvertFrom-Json
Require ($plan.kind-eq'cpb_v11_v12_diagnosis_completed_context_plan_v1'-and$plan.status-eq'root_reviewed_completed_inputs'-and$plan.authorized_execution-is[bool]-and-not$plan.authorized_execution) 'Metadata-only completed context plan'
$cache=@('models.v11.stages.initial','models.v11.stages.final','temporal_moment_observations.csv','temporal_moment_observations.actual_witnesses')
Require (Equal $inventory.cached_V11_selectors $cache) 'Four actual V11 cache paths'
$query=@('row_ids','target_mask','targets','mean','variance','native','row_mae','row_nll','row_mean_huber','row_variance_nll','row_total_objective')
$actual=@('update','attempt','clip_factor','temporal_export_weight_before','temporal_export_bias_before','temporal_patch_positions','temporal_patch_valid','temporal_patch_rows')
foreach($prefix in @('temporal_','temporal_contextual_')){foreach($f in @('pool_mixture_weights','pool_bin_valid','pool_bin_mean','pool_bin_variance','pool_projected_upstream_defined','pool_projected_upstream_pre_clip','pool_dispersion_weights','pool_dispersion_mean')){$actual+=($prefix+$f)}}
$checkpoint=@('artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates','causal_updates','initialization_seed','channel_ids','payload.model.encoder.change_decoder')
foreach($f in @('gradient_policy','native_layout','mean_width','uncertainty_width','native_width','objective_policy','mean_huber_delta_f64','loss_coefficients_f64','variance_error_detached','direct_gradient_routes','reporting_nll_backpropagated','temporal_pooling_policy','temporal_pooling_partition_v1','temporal_pooling_mix_f64','empty_local_bin_policy','temporal_moment_policy','temporal_export_layout','temporal_moment_partition_v1')){$checkpoint+=@($f,('payload.'+$f))}
$columns=@('update','bin','moment','export_row_begin','export_row_end','weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2','nonzero_export_gradient_values','moved_export_values')
function Numeric-Fields([string]$id){if($id-eq'train_targets'){return @('row_ids','group_ids','targets','valid')};if($id-match'^v12_actual_update_(1|2048)$'){return $actual};if($id.EndsWith('_features')){return @('row_ids','values','valid')};if($id.EndsWith('_query')){return $query};if($id.EndsWith('_checkpoint')){return $checkpoint};if($id-match'_head_'){return @('row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions')};if($id-match'_fit_'){return @('feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed')};throw 'Unselected tensor role'}
function Design-Binding($row,[string]$role){$b=[ordered]@{id=$row.id;role=$role;host_repository=$row.host_repository;host_path=$row.host_path;path=$row.host_path;container_path=$row.container_path;bytes=$row.bytes;sha256=$row.sha256};Require ($b.container_path.StartsWith('/')-and$b.container_path-notmatch'(^|/)\.\.(/|$)|\\') 'Absolute normal container path';Verify $b;return $b}
$allowed=@();foreach($row in $inventory.prospective_numeric_roles){$b=Design-Binding $row $row.role;Require ($b.container_path-notmatch'(?i)validation|/test[^/]*/') 'No held-out numerical path';if($b.id-eq'cached_v11_prior_TRAIN_diagnosis'){$b.numeric_json_selectors=$cache;Require (Equal $row.numeric_json_selectors $cache) 'Exact cached selectors'}elseif($b.id-eq'v12_temporal_moment_csv'){$b.allowed_CSV_columns=$columns;Require (Equal $row.allowed_CSV_columns $columns) 'Exact CSV columns'}elseif($b.id.EndsWith('_query_summary')){$b.numeric_json_selectors=@('train');Require (Equal $row.numeric_json_selectors @('train')) 'Exact TRAIN selector'}else{$b.allowed_tensor_fields=@(Numeric-Fields $b.id);Exact @($b.allowed_tensor_fields) @($row.allowed_tensor_fields) 'Exact tensor selector set'};$allowed+=$b}
Require ($allowed.Count-eq25-and@($allowed.id|Sort-Object -Unique).Count-eq25-and$actual.Count-eq24-and$checkpoint.Count-eq45-and$query.Count-eq11) 'Exact25 numeric /24 witness /45 checkpoint /11 query selectors'
$guards=@();foreach($row in @($inventory.historical_V11_hash_only_guard_roles)+@($inventory.additional_V12_hash_only_guard_roles)){$guards+=Design-Binding $row 'hash_only_no_numeric_access'}
Require ($guards.Count-eq35-and@($guards.id|Sort-Object -Unique).Count-eq35) 'Exactly35 disjoint hash guards'
Exact @($plan.context_inputs.id) @($inventory.context_inputs.id) 'Fourteen exact context IDs'
Require ($plan.context_inputs.Count-eq14-and$plan.context_assertions-is[array]-and$plan.context_assertions.Count-eq12-and(Equal $plan.context_assertions $inventory.prospective_context_assertions)) 'Exactly twelve context assertions as ARRAY'
$contexts=@();$metadata=@{};$map=@{}
foreach($row in $plan.context_inputs){$binding=@($inventory.context_inputs|Where-Object id -eq $row.id);Require ($binding.Count-eq1-and$row.host_repository-eq$binding[0].host_repository-and$row.host_path-eq$binding[0].host_path-and(Same $row $binding[0])-and$row.container_path-eq($sourceRoot+'/preservation/'+$row.id+'.json')) 'Exact completed context binding/path';Verify $row;$contexts+=$row;$map[$row.id]=$row;$metadata[$row.id]=Read-Binding $row}
foreach($assertion in $plan.context_assertions){$value=$metadata[$assertion.input];Require ($assertion.path-is[array]-and$assertion.path.Count-gt0) 'Literal assertion path';foreach($part in $assertion.path){Require ($part-is[string]-and$part) 'Literal selector';$value=$value.$part};Require (Equal $value $assertion.equals) ('Completed typed assertion '+$assertion.id)}
$old=$metadata.old_audit;$completion=$metadata.old_completion;$cacheRow=@($allowed|Where-Object id -eq 'cached_v11_prior_TRAIN_diagnosis')[0]
Require ($old.status-eq'passed'-and$old.checks-eq68264-and@($old.mismatches).Count-eq0-and(Same $old.card $map.old_card)-and(Same $old.diagnosis $cacheRow)-and(Same $completion.main_card $map.old_card)-and(Same $completion.main_diagnosis $cacheRow)-and(Same $completion.independent_audit $map.old_audit)-and(Same $completion.main_mirror $map.old_main_mirror)-and(Same $completion.independent_audit_mirror $map.old_audit_mirror)) 'Opaque completed68264 prior V11 chain; no cached numeric decoding'
Require ($metadata.old_main_mirror.status-eq'passed'-and$metadata.old_main_mirror.container_before_after_exact-and$metadata.old_main_mirror.all_sizes_sha256_exact-and(Same $metadata.old_main_mirror.canonical_diagnosis $cacheRow)-and$metadata.old_audit_mirror.status-eq'passed'-and$metadata.old_audit_mirror.source_before_after_exact) 'Opaque full prior mirrors'
$final=$metadata.v12_final_preservation;$trust=$metadata.v12_final_establishment
Require ($final.status-eq'passed'-and$final.critical_checkpoint_total-eq114-and$final.critical_checkpoint_pins.Count-eq114-and$final.V12_source_files-eq137-and-not$final.current_reporting_recovery_required-and-not$final.application_changed-and-not$final.baseline_changed-and(Same $final.current_ledger $map.current_ledger)-and(Same $final.current_catalog $map.current_catalog)-and(Same $final.final_establishment $map.v12_final_establishment)) 'Actual normal V12 114/137 preservation and current metadata'
Require ($trust.measured_candidate_established-and$trust.trust_admission_passed-and$metadata.v12_training_audit.status-eq'passed'-and$metadata.v12_quality_audit.status-eq'passed') 'Completed V12 trust reused without replay'
$goal=$metadata.current_goal_protocol;$retention=$metadata.retention_independent_review;$application=$metadata.retention_application_admission
Require ($goal.criterion_count-eq22-and$goal.criteria.Count-eq22-and-not$goal.additional_train_NLL_gate-and$application.status-eq'passed'-and$application.applied-and$application.prospective_future_use_eligible-and-not$application.application_changed-and-not$application.historical_results_or_counts_recomputed-and-not$application.additional_TRAIN_NLL_gate-and(Same $retention.protocol $map.current_goal_protocol)-and(Same $application.independent_review $map.retention_independent_review)) 'Unchanged strongest22 future protocol and applied review; historical V11 ancestry retained'
Require (@($application.files|Where-Object{$_.path-eq$map.current_goal_protocol.host_path-and(Same $_ $map.current_goal_protocol)}).Count-eq1) 'Applied protocol bytes exact; historical application ledger/catalog pins are not current'
$oldRoles=@($metadata.old_card.allowed_inputs)+@($metadata.old_card.hash_only_guards)
foreach($guard in @($guards|Where-Object{$_.id.StartsWith('v11_')})){Require (@($oldRoles|Where-Object{$_.id-eq$guard.id-and(Same $_ $guard)}).Count-eq1) 'Historical29 guards bind union of old numeric+hash roles'}
Require (Same $goal.v11_final_preservation $metadata.old_card.pre_execution_preservation) 'Strongest protocol retains historical V11 ancestor, distinct from current V12 preservation'
$pre=$plan.pre_execution_preservation;Verify $pre;Require ($pre.container_path-eq($sourceRoot+'/preservation/pre-execution-preservation.json')) 'Separate dedicated fresh preexecution copy'
$preMetadata=Read-Binding $pre;Require ($preMetadata.kind-eq'causal_v11_v12_TRAIN_diagnosis_pre_execution_preservation_v1'-and$preMetadata.status-eq'passed'-and$preMetadata.critical_checkpoint_count-eq114-and$preMetadata.critical_checkpoint_pins.Count-eq114-and$preMetadata.all_availability_bytes_sha256_exact-and(Same $preMetadata.parent_final_preservation $map.v12_final_preservation)-and(Same $preMetadata.parent_preservation $review.parent_preservation)-and$preMetadata.science_stages_replayed-eq0-and-not$preMetadata.DEV_changed) 'Actual fresh114 receipt and completed V12 binding, no broad re-verification here'
$extras=@($plan.preservation_pins);foreach($row in $extras){Verify $row;Require (-not$row.container_path.StartsWith($sourceRoot+'/')) 'Extra guards cannot add undeclared source copies'}
$all=@($allowed)+@($guards)+@($contexts)+@($extras)+@($pre);$hostFiles=@{};$runtime=@{}
foreach($row in $all){$taskHostKey=$row.host_repository+'/'+$row.host_path;if($hostFiles.ContainsKey($taskHostKey)){Require (Same $hostFiles[$taskHostKey] $row) 'Conflicting host alias'}else{$hostFiles[$taskHostKey]=$row};if($runtime.ContainsKey($row.container_path)){Require (Same $runtime[$row.container_path] $row) 'Conflicting runtime alias'}else{$runtime[$row.container_path]=$row}}
$card=[ordered]@{kind='cpb_v11_v12_saved_train_diagnosis_card_v1';stage='prospective_before_diagnostic_arithmetic';execution_admitted=$false;candidate_declared=$false;baseline='CPB-v3.market-01';baseline_or_application_change=$false;computed_versions=@('v12');cached_reference_versions=@('v11');train_rows=112;query_rows=8;train_begin_ms=1704067259999;train_end_ms=1705881599999;fixed_update_blocks=@(@(1,512),@(513,1024),@(1025,1536),@(1537,2048));cached_v11_selectors=$cache;manual_saved_affine_contractions_declared=$true;module_forwards_allowed=$false;new_quality_or_selection_allowed=$false;allowed_inputs=$allowed;hash_only_guards=$guards;context_inputs=$contexts;context_assertions=$plan.context_assertions;preservation_pins=$extras;pre_execution_preservation=$pre;metadata_copies_required_before_execution=@($contexts)+@($pre);owned_source_files=@($review.scientific_files|Where-Object{$_.path.StartsWith('code/research/causal_v11_v12_train_diagnosis/')} );scientific_source_files=$review.scientific_files;orchestration_helpers=$review.workflow_files;scientific_source_review=$reviewPin;closed_design_inventory=$inventoryPin;parent_preservation=$review.parent_preservation;completed_context_plan=$planPin;logical_pin_occurrences=$all.Count;unique_host_files=$hostFiles.Count;unique_container_files=$runtime.Count;container=[ordered]@{name='cuwacunu';id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';image='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7';root=$sourceRoot.Substring(0,$sourceRoot.Length-7)};scope='V12-only saved TRAIN arithmetic and exactly four cached audited V11 subtrees; no old raw reductions, trust replay, model/head/fit, heldout diagnosis, quality gate or candidate selection';checkpoint_schema_correction='45 total selectors: one numeric decoder parameter plus44 typed metadata selectors; immutable proposed design001 wording is preserved';missing_captures=$inventory.missing_captures_and_limits}
Write-New $proposed ($card|ConvertTo-Json -Depth 80)
Write-New ($proposed.Replace('.json','.md')) ('# Prospective saved TRAIN V11/V12 diagnosis'+[char]10+[char]10+'Execution is false. V12 initial/final saved TRAIN112, first8 query rows, fixed512 blocks and actual1/2048; exactly four audited V11 cached subtrees. Numeric25/hash35/context14; twelve assertions; witness24of60/checkpoint45. Five manual affine families are separate from zero model/head/module calls. V3 remains selected. Separate source/card/Stage/Schema/Run/audit admissions remain required.')
Pin $proposed|ConvertTo-Json

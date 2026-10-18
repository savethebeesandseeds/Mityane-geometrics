# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Owned physical V8/V9 generator copy, adapted to closed V11-only arithmetic.
# Host hashes/JSON metadata only. Never invokes a project helper, Docker or Python.
param(
 [Parameter(Mandatory=$true)][ValidateSet('Propose','Admit')][string]$Phase,
 [string]$ContextPlan='',
 [string]$ContextPlanSha256='',
 [string]$ProposedCardSha256='',
 [Parameter(Mandatory=$true)][switch]$RootReviewedCompletedContext
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootReviewedCompletedContext){throw 'Root review of exact source and completed context/prospective card required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$sourceRoot='/opt/cuwacunu/research/causal-v9-v11-train-diagnosis/attempt-001/source'
$documentRoot='doc/causal_v9_v11_train_diagnosis/'
$proposed='doc/CAUSAL_V9_V11_TRAIN_DIAGNOSIS_CARD.PROPOSED.json'
$admitted='doc/CAUSAL_V9_V11_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'
function Require([bool]$ok,[string]$why){if(-not$ok){throw $why}}
function Host-File([string]$relative,[string]$repository='cuwacunu_embedding'){
 Require ($repository-in@('cuwacunu_embedding','cuwacunu_dev')-and-not[IO.Path]::IsPathRooted($relative)-and$relative-notmatch'[\\:]'-and@($relative.Split('/')|Where-Object{$_-in@('','..','.')} ).Count-eq0) 'Bounded explicit repository-relative path'
 $base=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repository));$path=[IO.Path]::GetFullPath((Join-Path $base $relative))
 Require ($path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'No repository escape'
 $walk=$path
 while($walk-ne$base){$item=Get-Item -LiteralPath $walk;Require (-not($item.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'No linked input ancestor';$walk=[IO.Path]::GetDirectoryName($walk)}
 Require (-not(Get-Item -LiteralPath $path).PSIsContainer) 'Physical regular input file'
 return $path
}
function Pin([string]$relative,[string]$repository='cuwacunu_embedding'){
 $p=Host-File $relative $repository
 [ordered]@{path=$relative;bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Same($a,$b){$a.bytes-eq$b.bytes-and$a.sha256-ceq$b.sha256}
function Verify($row){
 Require ($row.host_repository-in@('cuwacunu_embedding','cuwacunu_dev')-and$row.bytes-is[ValueType]-and$row.bytes-gt0-and$row.sha256-cmatch'^[0-9a-f]{64}$') 'Typed host binding required'
 $p=Pin $row.host_path $row.host_repository;Require (Same $p $row) ('Changed host binding: '+$row.host_path)
 return $p
}
function Read-Context($row){$null=Verify $row;Get-Content -LiteralPath (Host-File $row.host_path $row.host_repository) -Raw|ConvertFrom-Json}
function Write-New([string]$relative,[string]$text){
 $p=Join-Path $researchRoot $relative;$stream=[IO.File]::Open($p,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
 try{$b=[Text.UTF8Encoding]::new($false).GetBytes($text.Replace([string][char]13,'')+[char]10);$stream.Write($b,0,$b.Length)}finally{$stream.Dispose()}
}
function Exact($observed,$expected,[string]$reason){
 Require (@($observed).Count-eq@($expected).Count-and@($observed|Sort-Object -Unique).Count-eq@($expected).Count-and-not(Compare-Object @($observed|Sort-Object) @($expected|Sort-Object))) $reason
}
$ready=Pin 'output/engineering/causal-v9-v11-train-diagnosis-source-preparation/attempt-001/source-ready-002.json'
Require ($ready.bytes-eq23879-and$ready.sha256-ceq'fd681ba57560663e62e84f159a66fcdfb54093a4e761d685d562cdd17e96c917') 'Exact reviewed scientific source readiness002'
$sourceReview=Pin ($documentRoot+'source-root-review-002.json')
Require ($sourceReview.bytes-eq4309-and$sourceReview.sha256-ceq'bcc963d6b9d2528a3cd0d1ecd0e80d7ec86dcadaf4f6efc3c776ec1b489ef6e6') 'Exact root ten-source admission'
$review=Get-Content -LiteralPath (Host-File $sourceReview.path) -Raw|ConvertFrom-Json
Require ($review.status-eq'passed_static_review_workflow_source_preparation_only'-and$review.source_files.Count-eq10) 'Source-only authority, not execution admission'
foreach($row in $review.source_files){Require (Same (Pin $row.path) $row) 'Every reviewed main/auditor source remains exact'}
$helperNames=@('prepare-card.ps1','prepare-preflight.ps1','run-diagnosis.ps1','verify-completion.ps1','orchestration-provenance.json','audit-card-generator.ps1','audit-stage.ps1','audit-execute.py','audit-mirror.ps1','audit-orchestration-provenance.json')
$helpers=@($helperNames|ForEach-Object{Pin ($documentRoot+$_)})
$sources=@($review.source_files|Where-Object{$_.path.StartsWith('code/research/causal_v9_v11_train_diagnosis/',[StringComparison]::Ordinal)})
Require ($sources.Count-eq5) 'Five main owned physical source pins'

if($Phase-eq'Admit'){
 Require ($ProposedCardSha256-cmatch'^[0-9a-f]{64}$') 'Externally reviewed proposed-card SHA required'
 $prior=Pin $proposed;Require ($prior.sha256-ceq$ProposedCardSha256) 'Exact prospective false bytes before parse'
 $card=Get-Content -LiteralPath (Host-File $proposed) -Raw|ConvertFrom-Json
 Require ($card.kind-eq'cpb_v9_v11_saved_train_diagnosis_card_v1'-and$card.stage-eq'prospective_before_diagnostic_arithmetic'-and$card.execution_admitted-is[bool]-and-not$card.execution_admitted) 'Preserved prospective false card required'
 foreach($row in @($card.owned_source_files)+@($card.orchestration_helpers)) {Require (Same (Pin $row.path) $row) 'Frozen source/helper unchanged before separate admission'}
 foreach($row in @($card.allowed_inputs)+@($card.hash_only_guards)+@($card.context_inputs)+@($card.preservation_pins)+@($card.pre_execution_preservation)){$null=Verify $row}
 $card.execution_admitted=$true
 $card|Add-Member -NotePropertyName separate_root_admission -NotePropertyValue ([ordered]@{prospective_card=$prior;source_review=$sourceReview;admitted_utc=[DateTime]::UtcNow.ToString('o');requires_actual_admitted_card_review_before_Stage=$true})
 Write-New $admitted ($card|ConvertTo-Json -Depth 80)
 $md=$admitted.Replace('.json','.md')
 Write-New $md ('# Saved TRAIN V9/V11 diagnosis admission'+[char]10+[char]10+'Separately admitted copy of preserved prospective card '+$prior.sha256+'. Exact admitted JSON must be independently reviewed before Stage. This metadata operation runs no diagnosis, schema, model, head or training.'+[char]10+'V11-only new saved arithmetic; four audited V9 cached subtrees; V3 remains selected.')
 Pin $admitted|ConvertTo-Json;return
}

Require ($ContextPlanSha256-cmatch'^[0-9a-f]{64}$'-and$ContextPlan) 'Externally reviewed completed-context plan SHA and bounded path required'
$planPin=Pin $ContextPlan;Require ($planPin.sha256-ceq$ContextPlanSha256) 'External context-plan hash before parse'
$plan=Get-Content -LiteralPath (Host-File $ContextPlan) -Raw|ConvertFrom-Json
Require ($plan.kind-eq'cpb_v9_v11_diagnosis_completed_context_plan_v1'-and$plan.status-eq'root_reviewed_completed_inputs'-and$plan.authorized_execution-is[bool]-and-not$plan.authorized_execution) 'Metadata-only root-reviewed completed context plan'
$inventoryPin=Pin ($documentRoot+'design-inventory-002.PROPOSED.json')
Require ($inventoryPin.bytes-eq44835-and$inventoryPin.sha256-ceq'de0aed954d263e4e94e17638e30cba9a0a79108ddd270c07b2e7814e25029941') 'Closed design002 immutable binding'
$inventory=Get-Content -LiteralPath (Host-File $inventoryPin.path) -Raw|ConvertFrom-Json
$cacheSelectors=@('models.v9.stages.initial','models.v9.stages.final','actual_pre_step_batch_diagnosis.v9','temporal_learning_observations.v9')
$queryFields=@('row_ids','target_mask','targets','mean','variance','native','row_mae','row_nll','row_mean_huber','row_variance_nll','row_total_objective')
$actualFields=@('update','attempt','clip_factor','temporal_export_weight_before','temporal_export_bias_before','temporal_patch_positions','temporal_patch_valid','temporal_patch_rows')
foreach($prefix in @('temporal_','temporal_contextual_')){foreach($f in @('pool_mixture_weights','pool_bin_valid','pool_bin_mean','pool_bin_variance','pool_projected_upstream_defined','pool_projected_upstream_pre_clip')){$actualFields+=($prefix+$f)}}
$checkpointFields=@('artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates','causal_updates','initialization_seed','channel_ids','payload.model.encoder.change_decoder')
foreach($f in @('gradient_policy','native_layout','mean_width','uncertainty_width','native_width','objective_policy','mean_huber_delta_f64','loss_coefficients_f64','variance_error_detached','direct_gradient_routes','reporting_nll_backpropagated','temporal_pooling_policy','temporal_pooling_partition_v1','temporal_pooling_mix_f64','empty_local_bin_policy','temporal_moment_policy','temporal_export_layout','temporal_moment_partition_v1')){$checkpointFields+=@($f,('payload.'+$f))}
$csvFields=@('update','bin','moment','export_row_begin','export_row_end','weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2','nonzero_export_gradient_values','moved_export_values')
function Numeric-Fields([string]$id){
 if($id-eq'train_targets'){return @('row_ids','group_ids','targets','valid')}
 if($id-match'^v11_actual_update_(1|2048)$'){return $actualFields}
 if($id.EndsWith('_features')){return @('row_ids','values','valid')}
 if($id.EndsWith('_query')){return $queryFields}
 if($id.EndsWith('_checkpoint')){return $checkpointFields}
 if($id-match'_head_'){return @('row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions')}
 if($id-match'_fit_'){return @('feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed')}
 throw 'Unselected numeric tensor identity'
}
function Design-Binding($row,[string]$role){
 $repo='cuwacunu_embedding';$path=[string]$row.path
 if($path.StartsWith('../cuwacunu_dev/',[StringComparison]::Ordinal)){$repo='cuwacunu_dev';$path=$path.Substring(16)}
 $out=[ordered]@{id=$row.id;role=$role;host_repository=$repo;host_path=$path;path=$path;container_path=$row.container_path;bytes=$row.bytes;sha256=$row.sha256}
 Require ($out.container_path.StartsWith('/')-and$out.container_path-notmatch'(^|/)\.\.(/|$)|\\') 'Explicit normal absolute container path'
 $null=Verify $out
 return $out
}
$allowed=@()
foreach($row in $inventory.prospective_numeric_roles){
 $role=[string]$row.proposed_role;$out=Design-Binding $row $role
 Require ($out.container_path-notmatch'(?i)validation|/test[^/]*/') 'No held-out numeric source'
 if($role-eq'TRAIN_archive'){$out.allowed_tensor_fields=@(Numeric-Fields $row.id)}
 elseif($row.id-eq'cached_v9_prior_TRAIN_diagnosis'){$out.numeric_json_selectors=$cacheSelectors}
 elseif($row.id-eq'v11_temporal_moment_csv'){$out.allowed_CSV_fields=$csvFields}
 elseif($row.id-match'^v11_(initial|final)_query_summary$'){$out.numeric_json_selector='train'}
 else{throw 'Unexpected source-declared numeric role'}
 $allowed+=$out
}
Require ($allowed.Count-eq25-and@($allowed.id|Sort-Object -Unique).Count-eq25-and$checkpointFields.Count-eq45-and$actualFields.Count-eq20-and$queryFields.Count-eq11) 'Closed25/45/20/11 source schema'
$guards=@();foreach($row in @($inventory.historical_V9_hash_only_guard_roles)+@($inventory.additional_V11_hash_only_guard_roles)){$guards+=Design-Binding $row 'hash_only_no_numeric_access'}
Require ($guards.Count-eq34-and@($guards.id|Sort-Object -Unique).Count-eq34) 'Closed34 original hash guards'
$contextIds=@('old_card','old_audit','old_main_mirror','old_audit_mirror','v11_training_audit','v11_final_establishment','v11_final_preservation','current_goal_protocol','retention_independent_review','retention_application_admission','current_ledger','current_catalog')
Exact @($plan.context_inputs.id) $contextIds 'Exactly12 actual completed metadata contexts'
$contexts=@();$metadata=@{}
foreach($row in $plan.context_inputs){
 $null=Verify $row
 $copy=[ordered]@{id=$row.id;role='metadata_context_only';host_repository=$row.host_repository;host_path=$row.host_path;path=$row.host_path;container_path=($sourceRoot+'/preservation/'+$row.id+'.json');bytes=$row.bytes;sha256=$row.sha256}
 Require ($copy.host_repository-eq'cuwacunu_embedding') 'Contexts are saved research metadata'
 $contexts+=$copy;$metadata[$copy.id]=Read-Context $copy
}
$expected=[ordered]@{v11_preservation_status='passed';v11_preservation_checkpoint_count=109;retention_review_status='passed';retention_applied=$true;ledger_baseline='CPB-v3.market-01';catalog_baseline='CPB-v3.market-01'}
Exact @($plan.context_assertions.PSObject.Properties.Name) @($expected.Keys) 'Exactly six completed context assertions'
foreach($name in $expected.Keys){
 $row=$plan.context_assertions.$name
 $id=if($name.StartsWith('v11_preservation_')){'v11_final_preservation'}elseif($name-eq'retention_review_status'){'retention_independent_review'}elseif($name-eq'retention_applied'){'retention_application_admission'}elseif($name-eq'ledger_baseline'){'current_ledger'}else{'current_catalog'}
 Require ($row.input-eq$id-and$row.path-is[array]-and$row.path.Count-gt0-and($row.equals|ConvertTo-Json -Compress)-ceq($expected[$name]|ConvertTo-Json -Compress)) 'Exact typed assertion source/value'
 $v=$metadata[$id];foreach($key in $row.path){Require ($key-is[string]-and$key) 'Literal metadata selector';$v=$v.$key}
 Require (($v|ConvertTo-Json -Compress)-ceq($row.equals|ConvertTo-Json -Compress)) ('Actual completed metadata assertion: '+$name)
}
$final=$metadata.v11_final_preservation;$trust=$metadata.v11_final_establishment
Require ($final.status-eq'passed'-and$final.critical_checkpoint_total-eq109-and$final.critical_checkpoint_pins.Count-eq109-and$final.V11_source_files-eq132-and-not$final.current_reporting_recovery_required-and$final.baseline-eq'CPB-v3.market-01'-and-not$final.application_changed-and-not$final.baseline_changed) 'Completed normal V11 109/132 preservation; V3 selected'
Require ($trust.measured_candidate_established-and$trust.trust_admission_passed-and$metadata.v11_training_audit.status-eq'passed') 'Completed V11 trust reused without scientific replay'
$goal=$metadata.current_goal_protocol;$retention=$metadata.retention_independent_review;$applied=$metadata.retention_application_admission
Require ($goal.criterion_count-eq22-and$goal.criteria.Count-eq22-and-not$goal.additional_train_NLL_gate-and$retention.status-eq'passed'-and$applied.status-eq'passed'-and$applied.applied-and$applied.prospective_future_use_eligible-and-not$applied.historical_results_or_counts_recomputed-and-not$applied.application_changed-and-not$applied.additional_TRAIN_NLL_gate) 'Actual stronger prospective22 protocol only; no new gate/old rescoring'
$contextMap=@{};foreach($row in $contexts){$contextMap[$row.id]=$row}
Require (Same $retention.protocol $contextMap.current_goal_protocol) 'Independent review binds exact current protocol'
Require (Same $applied.independent_review $contextMap.retention_independent_review) 'Application binds exact independent review'
Require (@($applied.files|Where-Object{$_.path-eq$contextMap.current_goal_protocol.host_path-and(Same $_ $contextMap.current_goal_protocol)}).Count-eq1) 'Applied bytes are current protocol'
Require (Same $goal.v11_final_preservation $contextMap.v11_final_preservation) 'Stronger protocol binds completed V11 preservation'
Require (Same $final.current_ledger $contextMap.current_ledger) 'Final preservation binds unchanged current ledger'
Require (Same $final.current_catalog $contextMap.current_catalog) 'Final preservation binds unchanged current catalog'
$cache=@($allowed|Where-Object id -eq 'cached_v9_prior_TRAIN_diagnosis')[0];$old=$metadata.old_audit
Require ($old.status-eq'passed'-and$old.checks-eq687825-and@($old.mismatches).Count-eq0-and(Same $old.card $contextMap.old_card)-and(Same $old.diagnosis $cache)) 'Opaque prior687825 audit binds exact cached V9 JSON; cache not decoded'
Require ($metadata.old_main_mirror.status-eq'passed'-and$metadata.old_main_mirror.container_before_after_exact-and$metadata.old_main_mirror.all_sizes_sha256_exact-and$metadata.old_audit_mirror.status-eq'passed'-and$metadata.old_audit_mirror.source_before_after_exact-and(Same $metadata.old_audit_mirror.independent_audit $contextMap.old_audit)) 'Opaque completed prior mirrors'
$extras=@();if($plan.PSObject.Properties.Name-contains'preservation_pins'){
 foreach($row in $plan.preservation_pins){
  $null=Verify $row
  Require ($row.container_path.StartsWith('/')-and$row.container_path-notmatch'(^|/)\.\.(/|$)|\\') 'Absolute extra preservation path'
  $out=[ordered]@{id=$row.id;role='hash_only_extra_preservation';host_repository=$row.host_repository;host_path=$row.host_path;path=$row.host_path;container_path=$row.container_path;bytes=$row.bytes;sha256=$row.sha256}
  if($out.container_path.StartsWith($sourceRoot+'/',[StringComparison]::Ordinal)){Require (@($contexts|Where-Object{$_.container_path-eq$out.container_path-and$_.host_path-eq$out.host_path-and(Same $_ $out)}).Count-eq1) 'Copied extras must alias an existing exact context'}
  $extras+=$out
 }
}
$pre=[ordered]@{};foreach($key in $contextMap.v11_final_preservation.Keys){$pre[$key]=$contextMap.v11_final_preservation[$key]};$pre.id='completed_v11_pre_execution_preservation';$pre.role='hash_only_final_preservation'
$all=@($allowed)+@($guards)+@($contexts)+@($extras)+@($pre)
$hostAliases=@{};$runtimeAliases=@{}
foreach($row in $all){
 $taskHostKey=$row.host_repository+'/'+$row.host_path
 if($hostAliases.ContainsKey($taskHostKey)){Require (Same $hostAliases[$taskHostKey] $row) 'Exact host alias pins'}else{$hostAliases[$taskHostKey]=$row}
 if($runtimeAliases.ContainsKey($row.container_path)){Require (Same $runtimeAliases[$row.container_path] $row) 'Exact container alias pins'}else{$runtimeAliases[$row.container_path]=$row}
}
$card=[ordered]@{
 kind='cpb_v9_v11_saved_train_diagnosis_card_v1';stage='prospective_before_diagnostic_arithmetic';execution_admitted=$false;candidate_declared=$false;baseline='CPB-v3.market-01';baseline_or_application_change=$false;computed_versions=@('v11');cached_reference_versions=@('v9');train_rows=112;query_rows=8;train_begin_ms=1704067259999;train_end_ms=1705881599999
 fixed_update_blocks=@(@(1,512),@(513,1024),@(1025,1536),@(1537,2048));cached_v9_selectors=$cacheSelectors;manual_saved_affine_contractions_declared=$true;module_forwards_allowed=$false;new_quality_or_selection_allowed=$false
 allowed_inputs=$allowed;hash_only_guards=$guards;context_inputs=$contexts;context_assertions=$plan.context_assertions;preservation_pins=$extras;pre_execution_preservation=$pre;metadata_copies_required_before_execution=$contexts
 owned_source_files=$sources;orchestration_helpers=$helpers;scientific_source_ready=$ready;scientific_source_review=$sourceReview;closed_design_inventory=$inventoryPin;completed_context_plan=$planPin
 container=[ordered]@{name='cuwacunu';id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';image='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7';root=$sourceRoot.Substring(0,$sourceRoot.Length-7)}
 logical_pin_occurrences=$all.Count;unique_host_files=$hostAliases.Count;unique_container_files=$runtimeAliases.Count
 scope='New V11 saved TRAIN manual arithmetic only; four exact cached audited V9 subtrees. No old V9 raw reductions, new fit, model/head/module execution, held-out diagnosis or next candidate admission.'
 missing_captures=@('No matched V9 moment panel','No all112 intermediate moments','No pre-step2047 decoder; actual2048 is not final2048','No other2046 raw gradients or optimizer counterfactual')
}
Write-New $proposed ($card|ConvertTo-Json -Depth 80)
Write-New ($proposed.Replace('.json','.md')) ('# Prospective saved TRAIN V9/V11 diagnosis'+[char]10+[char]10+'Execution is false. V11 initial/final saved TRAIN112, first8 query rows, four fixed512 blocks and actual updates1/2048; four prior audited V9 subtrees are cached. Manual affine contractions/scalar reductions are disclosed; model/head/module calls, fitting and training are forbidden. V3 remains selected.'+[char]10+'Exactly25 numeric,34 hash-only and12 metadata context roles; optional preservation aliases and actual capsule counts are dynamic. Separate root admission creates an ADMITTED copy and does not change these prospective bytes.')
Pin $proposed|ConvertTo-Json

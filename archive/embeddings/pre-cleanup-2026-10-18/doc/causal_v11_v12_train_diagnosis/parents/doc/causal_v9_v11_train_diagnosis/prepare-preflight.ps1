# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Independently owned host file/hash/JSON preflight. No Docker, Python, imports,
# archive decoding, saved numeric JSON parsing or scientific arithmetic.
param(
 [Parameter(Mandatory=$true)][switch]$RootReviewedSourceAndCard,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$CardSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$OrchestrationHelperSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootReviewedSourceAndCard){throw 'Independent actual admitted-card/source/helper review required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$doc='doc/causal_v9_v11_train_diagnosis/'
$destination=Join-Path $PSScriptRoot 'preflight-review.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve every prior preflight; no overwrite'}
$script:checks=0
function Check([bool]$ok,[string]$why){$script:checks++;if(-not$ok){throw $why}}
function Host-File([string]$relative,[string]$repository='cuwacunu_embedding'){
 Check ($repository-in@('cuwacunu_embedding','cuwacunu_dev')-and-not[IO.Path]::IsPathRooted($relative)-and$relative-notmatch'[\\:]'-and@($relative.Split('/')|Where-Object{$_-in@('','..','.')} ).Count-eq0) 'Bounded explicit repository label'
 $base=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repository));$p=[IO.Path]::GetFullPath((Join-Path $base $relative))
 Check ($p.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Host file stays within repository'
 $walk=$p
 while($walk-ne$base){$item=Get-Item -LiteralPath $walk;Check (-not($item.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'No linked file/ancestor';$walk=[IO.Path]::GetDirectoryName($walk)}
 Check (-not(Get-Item -LiteralPath $p).PSIsContainer) 'Physical regular file'
 return $p
}
function Pin([string]$relative,[string]$repository='cuwacunu_embedding'){
 $p=Host-File $relative $repository
 [ordered]@{path=$relative;bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Same($a,$b){$a.bytes-eq$b.bytes-and$a.sha256-ceq$b.sha256}
function Exact($actual,$expected,[string]$why){
 Check (@($actual).Count-eq@($expected).Count-and@($actual|Sort-Object -Unique).Count-eq@($expected).Count-and-not(Compare-Object @($actual|Sort-Object) @($expected|Sort-Object))) $why
}
function Equal($a,$b){($a|ConvertTo-Json -Depth 80 -Compress)-ceq($b|ConvertTo-Json -Depth 80 -Compress)}
$files=@()
function Verify-Metadata($binding){
 $actual=Pin $binding.path
 Check (Same $actual $binding) ('Exact metadata/source file: '+$binding.path)
 return $actual
}
$cardRelative='doc/CAUSAL_V9_V11_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'
$cardPin=Pin $cardRelative;Check ($cardPin.sha256-ceq$CardSha256) 'External admitted-card SHA before parse'
$card=Get-Content -LiteralPath (Host-File $cardRelative) -Raw|ConvertFrom-Json
Check ($card.kind-eq'cpb_v9_v11_saved_train_diagnosis_card_v1'-and$card.stage-eq'prospective_before_diagnostic_arithmetic'-and$card.execution_admitted-is[bool]-and$card.execution_admitted-and-not$card.candidate_declared-and-not$card.baseline_or_application_change) 'Separate true card, no new encoder/application admission'
Check ($card.baseline-eq'CPB-v3.market-01'-and(Equal $card.computed_versions @('v11'))-and(Equal $card.cached_reference_versions @('v9'))-and$card.train_rows-eq112-and$card.query_rows-eq8-and$card.train_begin_ms-eq1704067259999-and$card.train_end_ms-eq1705881599999) 'Closed TRAIN versions/population'
Check ((Equal $card.fixed_update_blocks @(@(1,512),@(513,1024),@(1025,1536),@(1537,2048)))-and$card.manual_saved_affine_contractions_declared-and-not$card.module_forwards_allowed-and-not$card.new_quality_or_selection_allowed) 'Fixed blocks and disclosed manual arithmetic only'
$cached=@('models.v9.stages.initial','models.v9.stages.final','actual_pre_step_batch_diagnosis.v9','temporal_learning_observations.v9')
Check (Equal $card.cached_v9_selectors $cached) 'Four cached audited V9 paths only'
$prior=Verify-Metadata $card.separate_root_admission.prospective_card
Check ($prior.path-eq'doc/CAUSAL_V9_V11_TRAIN_DIAGNOSIS_CARD.PROPOSED.json') 'Preserved separate prospective path'
$proposed=Get-Content -LiteralPath (Host-File $prior.path) -Raw|ConvertFrom-Json
Check (-not$proposed.execution_admitted) 'Prospective bytes retain false execution'
foreach($property in $proposed.PSObject.Properties){
 if($property.Name-ne'execution_admitted'){Check (Equal $card.($property.Name) $property.Value) ('Admission changes no prospective field: '+$property.Name)}
}
Exact @($card.PSObject.Properties.Name) @(@($proposed.PSObject.Properties.Name)+@('separate_root_admission')) 'Only separate admission object added'
$sourceReady=Verify-Metadata $card.scientific_source_ready
Check ($sourceReady.bytes-eq23879-and$sourceReady.sha256-ceq'fd681ba57560663e62e84f159a66fcdfb54093a4e761d685d562cdd17e96c917') 'Reviewed scientific source readiness002'
$sourceReview=Verify-Metadata $card.scientific_source_review
Check ($sourceReview.bytes-eq4309-and$sourceReview.sha256-ceq'bcc963d6b9d2528a3cd0d1ecd0e80d7ec86dcadaf4f6efc3c776ec1b489ef6e6') 'Exact ten-source root review002'
$review=Get-Content -LiteralPath (Host-File $sourceReview.path) -Raw|ConvertFrom-Json
Check ($review.source_files.Count-eq10) 'Ten frozen scientific files'
foreach($row in $review.source_files){$files+=Verify-Metadata $row}
$names=@('archive_reader.py','diagnose.py','schema.py','README.md','provenance.json')
Exact @($card.owned_source_files.path) @($names|ForEach-Object{'code/research/causal_v9_v11_train_diagnosis/'+$_}) 'Exact five main source paths'
foreach($row in $card.owned_source_files){Check (@($review.source_files|Where-Object{$_.path-eq$row.path-and(Same $_ $row)}).Count-eq1) 'Main source is reviewed exact file'}
$helperNames=@('prepare-card.ps1','prepare-preflight.ps1','run-diagnosis.ps1','verify-completion.ps1','orchestration-provenance.json','audit-card-generator.ps1','audit-stage.ps1','audit-execute.py','audit-mirror.ps1','audit-orchestration-provenance.json')
Exact @($card.orchestration_helpers.path) @($helperNames|ForEach-Object{$doc+$_}) 'Exactly ten owned workflow files'
foreach($row in $card.orchestration_helpers){$files+=Verify-Metadata $row}
$helper=Pin ($doc+'run-diagnosis.ps1');Check ($helper.sha256-ceq$OrchestrationHelperSha256) 'External complete phased-helper SHA'
$inventoryPin=Verify-Metadata $card.closed_design_inventory
Check ($inventoryPin.bytes-eq44835-and$inventoryPin.sha256-ceq'de0aed954d263e4e94e17638e30cba9a0a79108ddd270c07b2e7814e25029941') 'Exact closed design002'
$inventory=Get-Content -LiteralPath (Host-File $inventoryPin.path) -Raw|ConvertFrom-Json
$files+=@($sourceReady,$sourceReview,$inventoryPin,(Verify-Metadata $card.completed_context_plan),$prior,$cardPin,(Pin $cardRelative.Replace('.json','.md')),(Pin $prior.path.Replace('.json','.md')))
$numericIds=@('train_targets','cached_v9_prior_TRAIN_diagnosis','v11_temporal_moment_csv','v11_actual_update_1','v11_actual_update_2048')
foreach($stage in @('initial','final')){
 foreach($kind in @('features','query','query_summary','checkpoint')){$numericIds+=('v11_'+$stage+'_'+$kind)}
 foreach($rep in @(2701,2802,2903)){foreach($kind in @('head','fit')){$numericIds+=('v11_'+$stage+'_'+$kind+'_'+$rep)}}
}
$guardIds=@();foreach($row in @($inventory.historical_V9_hash_only_guard_roles)+@($inventory.additional_V11_hash_only_guard_roles)){$guardIds+=$row.id}
$contextIds=@('old_card','old_audit','old_main_mirror','old_audit_mirror','v11_training_audit','v11_final_establishment','v11_final_preservation','current_goal_protocol','retention_independent_review','retention_application_admission','current_ledger','current_catalog')
Exact @($card.allowed_inputs.id) $numericIds 'Exactly25 closed numeric identities'
Exact @($card.hash_only_guards.id) $guardIds 'Exactly34 source-pinned hash identities'
Check ($guardIds.Count-eq34) 'No inferred old numeric roles'
Exact @($card.context_inputs.id) $contextIds 'Exactly12 actual metadata contexts'
Exact @($card.metadata_copies_required_before_execution.id) $contextIds 'Exactly12 declared context copies'
$query=@('row_ids','target_mask','targets','mean','variance','native','row_mae','row_nll','row_mean_huber','row_variance_nll','row_total_objective')
$actual=@('update','attempt','clip_factor','temporal_export_weight_before','temporal_export_bias_before','temporal_patch_positions','temporal_patch_valid','temporal_patch_rows')
foreach($prefix in @('temporal_','temporal_contextual_')){foreach($f in @('pool_mixture_weights','pool_bin_valid','pool_bin_mean','pool_bin_variance','pool_projected_upstream_defined','pool_projected_upstream_pre_clip')){$actual+=($prefix+$f)}}
$checkpoint=@('artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates','causal_updates','initialization_seed','channel_ids','payload.model.encoder.change_decoder')
foreach($f in @('gradient_policy','native_layout','mean_width','uncertainty_width','native_width','objective_policy','mean_huber_delta_f64','loss_coefficients_f64','variance_error_detached','direct_gradient_routes','reporting_nll_backpropagated','temporal_pooling_policy','temporal_pooling_partition_v1','temporal_pooling_mix_f64','empty_local_bin_policy','temporal_moment_policy','temporal_export_layout','temporal_moment_partition_v1')){$checkpoint+=@($f,('payload.'+$f))}
$columns=@('update','bin','moment','export_row_begin','export_row_end','weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2','nonzero_export_gradient_values','moved_export_values')
foreach($row in $card.allowed_inputs){
 $id=$row.id;$source=@($inventory.prospective_numeric_roles|Where-Object id -eq $id);Check ($source.Count-eq1) 'One exact approved design role'
 $repo='cuwacunu_embedding';$path=$source[0].path
 if($path.StartsWith('../cuwacunu_dev/')){$repo='cuwacunu_dev';$path=$path.Substring(16)}
 Check ($row.host_repository-eq$repo-and$row.host_path-eq$path-and$row.path-eq$path-and$row.container_path-eq$source[0].container_path-and(Same $row $source[0])) 'Exact numeric host/runtime paths and original byte pins'
 Check ($row.container_path-notmatch'(?i)validation|/test[^/]*/') 'No held-out numeric source'
 $expectedRole=if($id-eq'cached_v9_prior_TRAIN_diagnosis'){'audited_prior_TRAIN_output_selected_V9_subtrees_only'}elseif($id-eq'v11_temporal_moment_csv'){'V11_TRAIN_moment_subgroup_16384_rows'}elseif($id.EndsWith('_query_summary')){'aggregate_JSON_TRAIN_key_only'}else{'TRAIN_archive'}
 Check ($row.role-eq$expectedRole) 'Source-declared exact access role'
 if($id-eq'cached_v9_prior_TRAIN_diagnosis'){Check (Equal $row.numeric_json_selectors $cached) 'Cached4 only';continue}
 if($id-eq'v11_temporal_moment_csv'){Check (Equal $row.allowed_CSV_fields $columns) 'Eleven moment CSV columns';continue}
 if($id.EndsWith('_query_summary')){Check ($row.numeric_json_selector-eq'train') 'Mixed JSON train only';continue}
 $fields=if($id-eq'train_targets'){@('row_ids','group_ids','targets','valid')}elseif($id-match'^v11_actual_update_'){$actual}elseif($id.EndsWith('_features')){@('row_ids','values','valid')}elseif($id.EndsWith('_query')){$query}elseif($id.EndsWith('_checkpoint')){$checkpoint}elseif($id-match'_head_'){@('row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions')}else{@('feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed')}
 Exact @($row.allowed_tensor_fields) @($fields) 'Independent exact unique typed/tensor allowlist'
}
foreach($row in $card.hash_only_guards){
 $source=@(@($inventory.historical_V9_hash_only_guard_roles)+@($inventory.additional_V11_hash_only_guard_roles)|Where-Object id -eq $row.id)
 Check ($source.Count-eq1-and$row.host_repository-eq'cuwacunu_embedding'-and$row.host_path-eq$source[0].path-and$row.container_path-eq$source[0].container_path-and(Same $row $source[0])) 'Hash-only original design guard remains exact'
}
$all=@($card.allowed_inputs)+@($card.hash_only_guards)+@($card.context_inputs)+@($card.preservation_pins)+@($card.pre_execution_preservation)
$byHost=@{};$byRuntime=@{};$verified=@()
foreach($row in $all){
 Check ($row.bytes-is[ValueType]-and$row.bytes-gt0-and$row.sha256-cmatch'^[0-9a-f]{64}$'-and$row.container_path.StartsWith('/')-and$row.container_path-notmatch'(^|/)\.\.(/|$)|\\') 'Positive typed pins and normal absolute runtime path'
 $key=$row.host_repository+'/'+$row.host_path
 if($byHost.ContainsKey($key)){Check (Same $byHost[$key] $row) 'Exact physical host alias'}else{$p=Pin $row.host_path $row.host_repository;Check (Same $p $row) 'Host availability/exact byte hash';$byHost[$key]=$p}
 if($byRuntime.ContainsKey($row.container_path)){Check (Same $byRuntime[$row.container_path] $row) 'Exact runtime alias'}else{$byRuntime[$row.container_path]=$row}
 $verified+=[ordered]@{id=$row.id;host_repository=$row.host_repository;host_path=$row.host_path;container_path=$row.container_path;bytes=$row.bytes;sha256=$row.sha256}
}
Check ($card.logical_pin_occurrences-eq$all.Count-and$card.unique_host_files-eq$byHost.Count-and$card.unique_container_files-eq$byRuntime.Count) 'Dynamic occurrence and unique physical counts'
$contexts=@{}
foreach($row in $card.context_inputs){
 Check ($row.host_repository-eq'cuwacunu_embedding'-and$row.container_path-eq('/opt/cuwacunu/research/causal-v9-v11-train-diagnosis/attempt-001/source/preservation/'+$row.id+'.json')) 'Frozen owned metadata copy path'
 $copy=@($card.metadata_copies_required_before_execution|Where-Object{$_.id-eq$row.id-and(Equal $_ $row)});Check ($copy.Count-eq1) 'Every context exactly copied once'
 $contexts[$row.id]=Get-Content -LiteralPath (Host-File $row.host_path) -Raw|ConvertFrom-Json
}
$expected=[ordered]@{v11_preservation_status='passed';v11_preservation_checkpoint_count=109;retention_review_status='passed';retention_applied=$true;ledger_baseline='CPB-v3.market-01';catalog_baseline='CPB-v3.market-01'}
Exact @($card.context_assertions.PSObject.Properties.Name) @($expected.Keys) 'Exactly6 context assertions'
foreach($name in $expected.Keys){
 $row=$card.context_assertions.$name;$id=if($name.StartsWith('v11_preservation_')){'v11_final_preservation'}elseif($name-eq'retention_review_status'){'retention_independent_review'}elseif($name-eq'retention_applied'){'retention_application_admission'}elseif($name-eq'ledger_baseline'){'current_ledger'}else{'current_catalog'}
 Check ($row.input-eq$id-and$row.path-is[array]-and$row.path.Count-gt0-and(Equal $row.equals $expected[$name])) 'Independent exact assertion source/type/value'
 $v=$contexts[$id];foreach($key in $row.path){Check ($key-is[string]-and$key) 'Literal metadata selector';$v=$v.$key};Check (Equal $v $row.equals) 'Actual completed context assertion'
}
$map=@{};foreach($row in $card.context_inputs){$map[$row.id]=$row}
Check ($card.pre_execution_preservation.host_path-eq$map.v11_final_preservation.host_path-and$card.pre_execution_preservation.container_path-eq$map.v11_final_preservation.container_path-and(Same $card.pre_execution_preservation $map.v11_final_preservation)) 'Pre-execution broad receipt aliases frozen V11 context'
$final=$contexts.v11_final_preservation
Check ($final.status-eq'passed'-and$final.critical_checkpoint_total-eq109-and$final.critical_checkpoint_pins.Count-eq109-and$final.V11_source_files-eq132-and-not$final.current_reporting_recovery_required-and-not$final.application_changed-and-not$final.baseline_changed) 'Reuse actual normal109/132 preservation; no109 checkpoint rehash claim'
Check ($contexts.v11_training_audit.status-eq'passed'-and$contexts.v11_final_establishment.measured_candidate_established-and$contexts.v11_final_establishment.trust_admission_passed) 'Completed learning/quality trust reused without replay'
$goal=$contexts.current_goal_protocol;$review=$contexts.retention_independent_review;$application=$contexts.retention_application_admission
Check ($goal.criterion_count-eq22-and$goal.criteria.Count-eq22-and-not$goal.additional_train_NLL_gate-and$application.applied-and$application.status-eq'passed'-and$application.prospective_future_use_eligible-and-not$application.application_changed-and-not$application.historical_results_or_counts_recomputed-and-not$application.additional_TRAIN_NLL_gate) 'Actual future22 protocol/application; no floor reduction'
Check ((Same $review.protocol $map.current_goal_protocol)-and(Same $application.independent_review $map.retention_independent_review)-and(Same $goal.v11_final_preservation $map.v11_final_preservation)) 'Stronger protocol/review/application/preservation exact chain'
Check (@($application.files|Where-Object{$_.path-eq$map.current_goal_protocol.host_path-and(Same $_ $map.current_goal_protocol)}).Count-eq1) 'Applied current protocol bytes'
Check ((Same $final.current_ledger $map.current_ledger)-and(Same $final.current_catalog $map.current_catalog)) 'Preserved current V3 ledger/catalog exact'
$old=$contexts.old_audit;$cache=@($card.allowed_inputs|Where-Object id -eq 'cached_v9_prior_TRAIN_diagnosis')[0]
Check ($old.status-eq'passed'-and$old.checks-eq687825-and@($old.mismatches).Count-eq0-and(Same $old.card $map.old_card)-and(Same $old.diagnosis $cache)) 'Opaque old687825 audit binds cache; cache JSON not decoded'
Check ($contexts.old_main_mirror.status-eq'passed'-and$contexts.old_main_mirror.container_before_after_exact-and$contexts.old_main_mirror.all_sizes_sha256_exact-and$contexts.old_audit_mirror.status-eq'passed'-and$contexts.old_audit_mirror.source_before_after_exact-and(Same $contexts.old_audit_mirror.independent_audit $map.old_audit)) 'Opaque prior source/mirror chain'
# Bounded current ignore checks cover only declared PT inputs/guards, not the
# historical109 checkpoint inventory or its large transitive preservation tree.
$ignoreEvidence=@()
foreach($repository in @('cuwacunu_embedding','cuwacunu_dev')){
 $paths=@($all|Where-Object{$_.host_repository-eq$repository-and$_.host_path.EndsWith('.pt')}|ForEach-Object host_path|Sort-Object -Unique)
 if($paths.Count-gt0){
  $root=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repository))
  $ignored=@()
  for($begin=0;$begin-lt$paths.Count;$begin+=128){
   $batch=@($paths|Select-Object -Skip $begin -First 128)
   $part=@(& git.exe -c ('safe.directory='+$root.Replace([char]92,[char]47)) -C $root check-ignore --no-index -- @batch)
   Check ($LASTEXITCODE-eq0) 'Every declared opaque PT remains ignored in bounded argv batch'
   $ignored+=@($part)
  }
  Exact $ignored $paths 'Current declared PT ignore set exact'
  $ignoreEvidence+=[ordered]@{repository=$repository;declared_PT=$paths.Count;all_ignored=$true}
 }
}
Check ($final.all_pinned_PT_ignored.Count-eq2-and@($final.all_pinned_PT_ignored|Where-Object{-not$_.all_ignored}).Count-eq0) 'Historical broad ignore evidence retained as metadata'
$files=@($files|Group-Object path|ForEach-Object{$_.Group[0]})
$result=[ordered]@{kind='cpb_v9_v11_saved_TRAIN_diagnosis_independent_static_preflight_v1';status='passed';reviewed_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks;card_sha256=$CardSha256;card=$cardPin;prospective_card=$prior;files=$files;orchestration_helper=$helper;allowed_input_pins=25;hash_only_guards=34;metadata_contexts=12;metadata_copies=12;logical_pin_occurrences=$all.Count;unique_host_files=$byHost.Count;unique_container_files=$byRuntime.Count;verified_file_bindings=$verified;pre_execution_preservation=$card.pre_execution_preservation;reused_completed_checkpoint_count=109;checkpoint_archives_rehashed_by_this_preflight=0;declared_PT_ignore=$ignoreEvidence;root_review_of_exact_source_card_acknowledged=$true;limits=@('Hashes are opaque; no tensor/archive values or mixed/cached numerical JSON are decoded','Existing V11 trust and old687825 audit are reused, not recomputed','Later separate Stage/Schema/Run admissions remain required');actions=[ordered]@{Docker=0;AST=0;project_imports=0;numeric_storage_reads=0;models=0;heads=0;training_updates=0;diagnosis_arithmetic=0;live_metadata_writes=0;DEV_writes=0}}
$stream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try{$b=[Text.UTF8Encoding]::new($false).GetBytes(($result|ConvertTo-Json -Depth 80).Replace([string][char]13,'')+[char]10);$stream.Write($b,0,$b.Length)}finally{$stream.Dispose()}
Pin ($doc+'preflight-review.json')|ConvertTo-Json

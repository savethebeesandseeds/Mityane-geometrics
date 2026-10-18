# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Source only: bounded host file/hash/JSON metadata generator. Never invokes Docker,
# Python, a project module, AST, schema, a model, a head or numerical diagnosis.
param(
 [Parameter(Mandatory=$true)][string]$ContextPlan,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ContextPlanSha256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedCompletedContext
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootReviewedCompletedContext){throw 'Explicit root review of completed context/source plan required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$sourceRoot='/opt/cuwacunu/research/causal-v8-v9-train-diagnosis/attempt-001/source'
$jsonRelative='doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json'
$mdRelative='doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.md'
function Resolve-File([string]$repository,[string]$relative){
 if($repository-notin@('cuwacunu_embedding','cuwacunu_dev')-or[IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains([char]92)-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded explicit repository-relative physical file required'}
 $root=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repository))
 $path=[IO.Path]::GetFullPath((Join-Path $root $relative))
 if(-not$path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Path escaped repository'}
 $cursor=$path
 while($cursor-ne$root){$item=Get-Item -LiteralPath $cursor;if($item.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No linked source/input ancestors'};$cursor=[IO.Path]::GetDirectoryName($cursor)}
 if((Get-Item -LiteralPath $path).PSIsContainer){throw 'Regular file required'}
 $path
}
function Pin([string]$repository,[string]$relative){
 $p=Resolve-File $repository $relative
 [ordered]@{path=$relative;bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Require-Pin($binding){
 $repository=if($binding.Contains('host_repository')){$binding.host_repository}else{'cuwacunu_embedding'}
 $relative=if($binding.Contains('host_path')){$binding.host_path}else{$binding.path}
 $actual=Pin $repository $relative
 if($binding.sha256-cnotmatch'^[0-9a-f]{64}$'-or$actual.bytes-ne$binding.bytes-or$actual.sha256-ne$binding.sha256){throw ('Reviewed input changed: '+$relative)}
 $actual
}
function Remap-Text([string]$text){
 $text.Replace('v8','__OLD_HIGH__').Replace('V8','__OLD_HIGH_UPPER__').Replace('v6','v8').Replace('V6','V8').Replace('__OLD_HIGH__','v9').Replace('__OLD_HIGH_UPPER__','V9')
}
function Remap-Value($value){
 if($value-is[string]){return (Remap-Text $value)}
 if($value-is[Collections.IDictionary]){$copy=[ordered]@{};foreach($key in $value.Keys){$copy[(Remap-Text $key)]=Remap-Value $value[$key]};return $copy}
 if($value-is[Collections.IEnumerable]){return ,@($value|ForEach-Object {Remap-Value $_})}
 $value
}
function Clone-Value($value){$value|ConvertTo-Json -Depth 90|ConvertFrom-Json -AsHashtable}
function Stamp-HostPin($entry){
 $p=Pin $entry.host_repository $entry.host_path
 $entry['bytes']=$p.bytes;$entry['sha256']=$p.sha256
 $entry
}
function Context-PlainPin($entry){[ordered]@{path=$entry.host_path;bytes=$entry.bytes;sha256=$entry.sha256}}
function Exact-Set($actual,$expected,[string]$reason){
 if(@($actual).Count-ne@($expected).Count-or(Compare-Object @($actual|Sort-Object) @($expected|Sort-Object))){throw $reason}
}
$outputJSON=Join-Path $researchRoot $jsonRelative;$outputMD=Join-Path $researchRoot $mdRelative
if((Test-Path -LiteralPath $outputJSON)-or(Test-Path -LiteralPath $outputMD)){throw 'Preserve every existing card/Markdown; use additive attempts'}
$planPin=Pin 'cuwacunu_embedding' $ContextPlan
if($planPin.sha256-ne$ContextPlanSha256){throw 'External context-plan SHA before JSON parse required'}
$plan=Get-Content -Raw -LiteralPath (Resolve-File 'cuwacunu_embedding' $ContextPlan)|ConvertFrom-Json -AsHashtable
if($plan.kind-ne'cpb_v8_v9_train_diagnosis_context_input_plan_v1'-or$plan.status-ne'root_reviewed_completed_V9_context'-or$plan.authorized-isnot[bool]-or-not$plan.authorized){throw 'Separate root-admitted completed-context plan required'}
$selfRelative=[IO.Path]::GetRelativePath($researchRoot,$PSCommandPath).Replace([char]92,[char]47)
$self=Require-Pin $plan.generator
if($self.path-ne$selfRelative){throw 'Root plan must bind this exact generator'}
$null=Require-Pin $plan.generator_provenance
$templatePin=Require-Pin $plan.template
if($templatePin.path-ne'doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.json'-or$templatePin.bytes-ne81588-or$templatePin.sha256-ne'49b22f1360593f2f0dd6d8ebb7fd8b0d4baa4749fe01ad8875fa79e5fe529829'){throw 'Exact immutable final V6/V8 JSON metadata template required'}
$template=Get-Content -Raw -LiteralPath (Resolve-File 'cuwacunu_embedding' $templatePin.path)|ConvertFrom-Json -AsHashtable
if($template.allowed_inputs.Count-ne52-or$template.preservation_pins.Count-ne33-or$template.metadata_copies_required_before_execution.Count-ne13){throw 'Unexpected immutable template schema'}
$requiredSourceNames=@('archive_reader.py','diagnose.py','provenance.json','README.md','schema.py')
Exact-Set @($plan.owned_source_files|ForEach-Object {[IO.Path]::GetFileName($_.path)}) $requiredSourceNames 'Exactly five source files required'
foreach($row in $plan.owned_source_files){if($row.path-ne('code/research/causal_v8_v9_train_diagnosis/'+[IO.Path]::GetFileName($row.path))){throw 'Exact owned main source paths'};$null=Require-Pin $row}
$main=@($plan.owned_source_files|Where-Object {$_.path.EndsWith('/diagnose.py')})
if($main.Count-ne1-or$main[0].sha256-ne'5743c08ceacd127db4ac463a7a4e56e9c3da8d86619839a2f8bf190a4c8fb537'){throw 'Frozen independently reviewed main diagnostic source required'}
$helperNames=@('run-diagnosis.ps1','prepare-preflight.ps1','verify-completion.ps1','orchestration-provenance.json')
Exact-Set @($plan.orchestration_source_files|ForEach-Object {[IO.Path]::GetFileName($_.path)}) $helperNames 'Exactly four reviewed orchestration sources required'
foreach($row in $plan.orchestration_source_files){if($row.path-ne('doc/causal_v8_v9_train_diagnosis/'+[IO.Path]::GetFileName($row.path))){throw 'Exact owned helper path'};$null=Require-Pin $row}
foreach($row in $plan.independent_reviews){$pin=Require-Pin $row;$review=Get-Content -Raw -LiteralPath (Resolve-File 'cuwacunu_embedding' $pin.path)|ConvertFrom-Json -AsHashtable;if($review.status-notin@('passed','passed_static_source_only')){throw 'Actual independent source/context reviews required'}}
if($plan.independent_reviews.Count-lt1){throw 'Independent reviewed source/context bindings required'}
$contextNames=@('final_preservation','goal_protocol','retention_independent_review','retention_application_admission','current_clean_ledger','current_catalog','reporting_recovery_card','reporting_recovery_receipt','before_recovery_failure_mirror')
Exact-Set @($plan.context.Keys) $contextNames 'Exactly nine external context roles required'
$contexts=[ordered]@{}
foreach($key in $contextNames){$null=Require-Pin $plan.context[$key];$contexts[$key]=$plan.context[$key]}
$broad=Get-Content -Raw -LiteralPath (Resolve-File $contexts.final_preservation.host_repository $contexts.final_preservation.host_path)|ConvertFrom-Json -AsHashtable
if($broad.kind-ne'causal_fresh_v9_final_preservation_v1'-or$broad.status-ne'passed'-or$broad.critical_checkpoint_total-ne99-or$broad.selected_V3_source_files_each_repo-ne47){throw 'Completed final99/selectedV3 preservation required before card generation'}
if($broad.current_ledger.sha256-ne$contexts.current_clean_ledger.sha256-or$broad.current_catalog.sha256-ne$contexts.current_catalog.sha256){throw 'Final99 receipt must bind current ledger/catalog'}
$future=Get-Content -Raw -LiteralPath (Resolve-File $contexts.goal_protocol.host_repository $contexts.goal_protocol.host_path)|ConvertFrom-Json -AsHashtable
$futureReview=Get-Content -Raw -LiteralPath (Resolve-File $contexts.retention_independent_review.host_repository $contexts.retention_independent_review.host_path)|ConvertFrom-Json -AsHashtable
$futureApplication=Get-Content -Raw -LiteralPath (Resolve-File $contexts.retention_application_admission.host_repository $contexts.retention_application_admission.host_path)|ConvertFrom-Json -AsHashtable
if($future.kind-ne'clean_causal_encoder_search_goal_v9_retention_protocol_v1'-or$future.criterion_count-ne22-or$futureReview.status-ne'passed'-or$futureApplication.status-ne'passed'-or-not$futureApplication.prospective_future_use_eligible-or$futureApplication.application_changed-or$futureApplication.new_candidate_declared-or$futureApplication.historical_results_or_counts_recomputed){throw 'Completed future22 protocol/review/application context required; no derivation here'}
$futureFile=@($futureApplication.files|Where-Object {$_.path-eq$contexts.goal_protocol.host_path-and$_.bytes-eq$contexts.goal_protocol.bytes-and$_.sha256-eq$contexts.goal_protocol.sha256})
if($futureFile.Count-ne1-or$futureApplication.independent_review.bytes-ne$contexts.retention_independent_review.bytes-or$futureApplication.independent_review.sha256-ne$contexts.retention_independent_review.sha256){throw 'Future application must bind this exact protocol and review'}
if($contexts.reporting_recovery_card.sha256-ne'9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'-or$contexts.reporting_recovery_receipt.sha256-ne'5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72'-or$contexts.before_recovery_failure_mirror.sha256-ne'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532'){throw 'Exact completed report-only recovery context'}
$card=Remap-Value $template
foreach($key in @('frozen_utc','root_and_independent_frozen_card_review','preserved_original_unfrozen_draft','context_changes_after_draft_review','preserved_pre_wording_context')){$card.Remove($key)}
$card.kind='cpb_v8_v9_saved_train_diagnosis_card_v1';$card.declared_utc=[DateTime]::UtcNow.ToString('o')
$card.stage='prospective_before_diagnostic_arithmetic';$card.versions=@('v8','v9')
$card.comparison_references=@('CPB-v8.market-01','CPB-v9.market-01')
$card.candidate_declared=$false;$card.diagnosis_executed=$false;$card.execution_admitted=$false
$card.freeze_can_proceed=$false;$card.pending_before_freeze=@('Independent actual-card/source/transport preflight and separate root phase admissions')
$card.parent_card_pattern=$templatePin;$card.planning_handoff='Root-authorized bounded savedTRAIN V8/V9 diagnosis; source preparation and metadata card generation do not admit AST/body or a new model.'
$card.owned_source_files=$plan.owned_source_files;$card.orchestration_source_files=$plan.orchestration_source_files
$card.card_generator=[ordered]@{source=$self;provenance=$plan.generator_provenance;context_plan=$planPin;project_source_executed=$false;immutable_template=$templatePin}
$versions=@('v8','v9');$temporalStems=@('temporal_export_weight','temporal_export_bias','temporal_pool_positions','temporal_pool_score_bias')
$actualFields=@('update','attempt','row_ids','timestamps','observed','hidden','context_deleted','visible','target_mask','eligible_channels','targets','mean','variance','native','row_mean_huber','row_variance_nll','row_total_objective','mean_Huber_F64','variance_NLL_F64','total_objective_F64','Gaussian_NLL_F64','pre_clip_global_norm','clip_factor')+@($temporalStems|ForEach-Object {$stem=$_;@('before','after','clipped_gradient')|ForEach-Object {$stem+'_'+$_}})
$checkpointFields=@('artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates','causal_updates','initialization_seed','channel_ids','payload.model.encoder.change_decoder','payload.model.variance_projection','payload.model.variance_bias','payload.link_constants_f32','payload.softplus_parameters_f64','gradient_policy','payload.gradient_policy')
$layoutFields=@('native_layout','mean_width','uncertainty_width','native_width','objective_policy','mean_huber_delta_f64','loss_coefficients_f64','variance_error_detached','direct_gradient_routes','reporting_nll_backpropagated','temporal_pooling_policy','temporal_pooling_partition_v1')
foreach($field in $layoutFields){$checkpointFields+=@($field,('payload.'+$field))}
$otherFields=[ordered]@{targets=@('row_ids','group_ids','targets','valid');features=@('row_ids','values','valid');query=@('row_ids','timestamps','target_mask','visible','observed','hidden','raw','targets','mean','variance','native','row_mae','row_nll','row_mean_huber','row_variance_nll','row_total_objective');head=@('row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions');fit=@('feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed')}
$inputs=@()
foreach($row in $template.allowed_inputs){
 if($row.id-like'v8_temporal_*'){continue}
 $entry=Remap-Value $row;$id=$entry.id
 if($entry.role-eq'TRAIN_archive'){
  if($id-match'_actual_update_'){$entry.allowed_tensor_fields=@($actualFields|Sort-Object)}
  elseif($id.EndsWith('_checkpoint')){$fields=@($checkpointFields);if($id.StartsWith('v9_')){$fields+=@('temporal_pooling_mix_f64','payload.temporal_pooling_mix_f64','empty_local_bin_policy','payload.empty_local_bin_policy')};$entry.allowed_tensor_fields=@($fields|Sort-Object)}
  else{$kind=if($id-eq'train_targets'){'targets'}elseif($id-match'_head_'){'head'}elseif($id-match'_fit_'){'fit'}elseif($id.EndsWith('_features')){'features'}else{'query'};$entry.allowed_tensor_fields=@($otherFields[$kind]|Sort-Object)}
 }
 $inputs+=Stamp-HostPin $entry
}
$betweenFields=@('basis','basis_signs','full_position_divisor','position_bin_width','rows','gradient_steps','movement_steps','final_contrast_changes','final_projection_change_l2','gradient_definition','movement_includes_AdamW_weight_decay','weight_decay_counted_as_data_learning','extra_encoder_calls')
foreach($version in $versions){
 foreach($row in $template.allowed_inputs|Where-Object {$_.id-like'v8_temporal_*'}){
  $entry=if($version-eq'v8'){Clone-Value $row}else{Remap-Value $row}
  if($entry.id.EndsWith('_receipt')-and$version-eq'v9'){$entry.allowed_JSON_fields+=@('temporal_pooling_mix','empty_local_bin_policy','between_bin_positions');$entry.allowed_JSON_nested_fields.between_bin_positions=$betweenFields}
  $inputs+=Stamp-HostPin $entry
 }
}
$between=[ordered]@{id='v9_temporal_between_csv';host_repository='cuwacunu_embedding';host_path='output/runs/causal-fresh-v9/run-001/temporal-between-bin-position-learning.csv';container_path='/opt/cuwacunu/research/causal-fresh-v9/attempt-001/run-001/temporal-between-bin-position-learning.csv';role='V9_TRAIN_between_bin_2048_rows';allowed_CSV_fields=@('update','between_bin_gradient_l2','between_bin_movement_l2','contrast_0_gradient','contrast_1_gradient','contrast_2_gradient','contrast_0_movement','contrast_1_movement','contrast_2_movement');numeric_update_scope='all2048successfulupdates'}
$inputs+=Stamp-HostPin $between
if($inputs.Count-ne56-or@($inputs.id|Sort-Object -Unique).Count-ne56){throw 'Exactly56 distinct TRAIN numeric roles required'}
foreach($entry in $inputs){if($entry.container_path-match'validation'-or-not$entry.container_path.StartsWith('/')){throw 'No held-out or relative numeric input'}}
$card.allowed_inputs=$inputs
$guards=@()
foreach($row in $card.preservation_pins){
 if($row.id-eq'preserve-v8-complete-mirror'){$row.host_path='doc/causal_fresh_v8/complete-mirror-attempt-001.json'}
 $guards+=Stamp-HostPin $row
}
$contextIDs=[ordered]@{current_clean_ledger='preserve-current-clean-ledger';current_catalog='preserve-current-catalog';goal_protocol='preserve-future-v9-retention';retention_independent_review='preserve-future-v9-retention-review';retention_application_admission='preserve-future-v9-retention-admission';reporting_recovery_card='preserve-reporting-recovery-card';reporting_recovery_receipt='preserve-reporting-recovery-receipt';before_recovery_failure_mirror='preserve-before-recovery-failure-mirror'}
foreach($key in $contextIDs.Keys){
 $id=$contextIDs[$key];$ctx=$contexts[$key];$matches=@($guards|Where-Object {$_.id-eq$id})
 $entry=[ordered]@{id=$id;host_repository=$ctx.host_repository;host_path=$ctx.host_path;container_path=($sourceRoot+'/preservation/'+$id+'.json');bytes=$ctx.bytes;sha256=$ctx.sha256}
 if($matches.Count-eq1){$guards=@($guards|ForEach-Object {if($_.id-eq$id){$entry}else{$_}})}elseif($matches.Count-eq0){$guards+=$entry}else{throw 'Distinct preservation IDs required'}
 $card[$key]=Context-PlainPin $entry
}
if($guards.Count-ne36-or@($guards.id|Sort-Object -Unique).Count-ne36){throw 'Exactly36 distinct preservation-only roles required'}
$card.preservation_pins=$guards;$card.preservation_pin_count=36
$broadContext=$contexts.final_preservation
$card.pre_execution_preservation=[ordered]@{id='final-v9-preservation';host_repository=$broadContext.host_repository;host_path=$broadContext.host_path;container_path=($sourceRoot+'/final-preservation.json');bytes=$broadContext.bytes;sha256=$broadContext.sha256}
$metadata=@($guards|Where-Object {$_.container_path.StartsWith($sourceRoot+'/preservation/')})
if($metadata.Count-ne16){throw 'Exactly16 preservation metadata source copies required'}
$card.metadata_copies_required_before_execution=$metadata
$card.final_establishment_context=Context-PlainPin (@($guards|Where-Object {$_.id-eq'preserve-v9-establishment'})[0])
$card.temporal_observation_scope='both-saved8192group2048gauge-two12field-witnesses-V9-between2048'
$card.temporal_policy_by_version=[ordered]@{v8='original-position-four-bin-disjoint-export-rows-v1';v9='original-position-four-bin-shared-global-half-mixture-v1'}
$card.V9_temporal_mix=@(.5,.5);$card.V9_empty_local_bin_policy='zero-whole-export-slice-after-bias-v1'
$card.between_bin_basis='hadamard-full16-signs-div4-v1';$card.between_bin_signs=@(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1));$card.individual_final_coefficient_nonzero_gate=$false
$card.actual_witness_attempt_context=[ordered]@{v8=@{attempted_batches=2048;successful_updates=2048;skipped_batches=0};v9=@{attempted_batches=2048;successful_updates=2048;skipped_batches=0};scope='these two pinned historical runs only; not a generic future training assumption'}
$card.numeric_policy='Hash56 TRAIN roles+36preservation-only roles+broad final99 receipt (93guards) before storage;recheck all pins/source/card afterward. Metadata source copies16, future capsule24, no held-out numeric reads.'
$card.view='TRAIN112 saved features/heads/components;first8 stage-matched query raw/support/local adjoints;both actual1/2048;both2048 observed global-clipping traces;BOTH8192temporal-group/2048gauge/two12-array witnesses;V9-only2048between-bin observations'
$card.diagnostics=@(
 'Both2048 globalclip traces:component/norm/factor observations overall+four512blocks;actual1/2048component/support/CSV association without matched pre-step decoder weights.',
 'All112 TRAIN native40/four-block statistics and exact already-fitted ridge contributions;all three saved neural prediction sets,no forward/refit.',
 'First8 initial/final fixed queries:four original32-tick raw/error/support bins,not age controlled;stage-matched mean4 and separate uncertainty4 local adjoints only.',
 'BOTH versions8192group+2048gauge logs and two actual12-array F64-before-subtraction checks;V9 alone adds2048signed Hadamard between-bin rows/three direction events/positive overall final projection. Both expose observed global norms only;no UNSAVED subgroup/gradient counterfactual,full Jacobian or optimizer replay.'
)
$card.forbidden=@(
 'Full encoder/nonlinear-parent-decoder/library-model/neural/head execution;bounded manual saved linear-residual/variance-link/finite-difference scalar kernels are explicitly allowed',
 'New training/head/scaler/normalizer/statistic fits',
 'Either version branch-split inference,UNSAVED subgroup or gradient counterfactual,global-clip or optimizer replay;both actual trajectories expose observed global norms only',
 'Actual-batch decoder pairing or full upstream Jacobian',
 'Held-out numeric fields/raw-cache reads/post-encoder PCA',
 'New encoder/candidate/rescue/quality floor/TRAIN-NLL gate from this diagnosis',
 'Dev promotion/publication/Git mutations'
)
$card.temporal_observation_contract=[ordered]@{versions=@('v8','v9');observed_group_rows_each=8192;observed_gauge_rows_each=2048;actual_updates_each=@(1,2048);actual_array_fields_each=12;V9_observed_between_rows=2048;other2046_raw_gradients_reconstructed=$false;counterfactual_gradient_or_cause_claim=$false;movement_includes_AdamW_decay=$true;decay_is_data_learning_proof=$false}
$card.tolerances.temporal_between_projection_arithmetic='absolute2e-12+relative2e-12;F64 operands before subtraction;exact signed direction/count/row association'
$card.admission='Prospective metadata card only: execution_admitted=false. Root+independent actual card/source/helper preflight, external card/preflight/ZIP/capsule pins, and separate Stage/Schema/Run admission are mandatory. No generator phase admits a project call.'
$card.interpretation='Saved TRAIN evidence only.112rows/fourmodelstages,first8stage-matched local adjoints/four32tickbins,twoactualbatches,observed2048scalars andtemporal records have separate scopes. No V10,cause,gradient-dominance or held-out explanation selected.'
foreach($entry in @($inputs)+@($guards)+@($card.pre_execution_preservation)){$null=Require-Pin $entry}
foreach($entry in @($plan.owned_source_files)+@($plan.orchestration_source_files)){$null=Require-Pin $entry}
if((Pin 'cuwacunu_embedding' $ContextPlan).sha256-ne$ContextPlanSha256){throw 'Context plan changed during metadata generation'}
$jsonBytes=[Text.UTF8Encoding]::new($false).GetBytes((($card|ConvertTo-Json -Depth 95).Replace([string][char]13+[char]10,[string][char]10)+[char]10))
$markdown=@'
<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE. -->
# Prospective saved TRAIN V8/V9 diagnosis

This machine card declares no new candidate and admits no execution. It uses56
TRAIN numeric roles,36preservation-only roles and final99 preservation:93guards;
16metadata source copies belong to the later24file capsule. The source/card and
transport preflight must pass before separate root Stage/Schema/Run admissions.

Both native40 versions use fixed saved readouts and identical objective/global
clipping.112TRAIN rows,first8stage-matched query tensors,two actual batches and
2048 observed trajectory scalars have separate scopes. Both8192 temporal groups,
2048gauge observations and two12-array witnesses are allowed;V9 alone adds2048
signed R16Hadamard records. No unsaved gradient or optimizer replay is permitted.
Each Hadamard direction needs recorded gradient/movement events; only OVERALL
final projection change must be positive. Signed final coefficients are descriptive.

Manual saved linear residual contractions, variance-link reconstruction and scalar
finite differences are bounded arithmetic, not full encoder/nonlinear parent/
library model/head execution. Mixed query JSON exposes only TRAIN numeric keys.
No held-out tensor arithmetic,refit,old model/head replay or new quality follows.
Unequal four32tick-bin support is not age controlled or proof of pooling cause.
No cross-basis/model/stage cosine,scalar-loss gradient dominance or V10 selection.

The companion JSON contains exact source,all file pins,scoped selectors,independently
reviewed future retention/current ledger/catalog/final99 and report-recovery context.
V3 remains sole dev selection; every V3–V9 scoped strength remains protected.
'@
$mdBytes=[Text.UTF8Encoding]::new($false).GetBytes(($markdown.Replace([string][char]13+[char]10,[string][char]10)+[char]10))
foreach($item in @(@{path=$outputJSON;bytes=$jsonBytes},@{path=$outputMD;bytes=$mdBytes})){$stream=[IO.File]::Open($item.path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None);try{$stream.Write($item.bytes,0,$item.bytes.Length)}finally{$stream.Dispose()}}
@((Pin 'cuwacunu_embedding' $jsonRelative),(Pin 'cuwacunu_embedding' $mdRelative))|ConvertTo-Json

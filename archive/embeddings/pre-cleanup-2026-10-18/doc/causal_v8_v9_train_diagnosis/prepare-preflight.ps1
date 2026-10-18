# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host file/hash/JSON binding checks only, based on the preserved V6/V8 preflight contract.
# No Docker, AST, project import, tensor decoding or diagnostic arithmetic.
param(
 [Parameter(Mandatory=$true)][switch]$RootReviewedSourceAndCard,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$CardSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$OrchestrationHelperSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootReviewedSourceAndCard){throw 'Actual final source/card and this helper require prior root review'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$destination=Join-Path $PSScriptRoot 'preflight-review.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve the previous preflight; do not overwrite or replay'}
$script:checks=0
function Check([bool]$ok,[string]$reason){$script:checks++;if(-not$ok){throw $reason}}
function HostFile([string]$relative,[string]$repository='cuwacunu_embedding'){
 Check ($repository-in@('cuwacunu_embedding','cuwacunu_dev')-and-not[IO.Path]::IsPathRooted($relative)-and$relative-notmatch'[\\:]'-and@($relative.Split('/')|Where-Object{$_-in@('','..','.')}).Count-eq0) 'Bounded repository file label'
 $base=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repository));$path=[IO.Path]::GetFullPath((Join-Path $base $relative))
 Check ($path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'File stays in exact repository'
 $walk=$path
 while($walk-ne$base){$item=Get-Item -LiteralPath $walk;Check (($item.Attributes-band[IO.FileAttributes]::ReparsePoint)-eq0) 'No linked file/ancestor';$walk=[IO.Path]::GetDirectoryName($walk)}
 Check (-not(Get-Item -LiteralPath $path).PSIsContainer) 'Regular file required'
 return $path
}
function Pin([string]$relative,[string]$repository='cuwacunu_embedding'){
 $path=HostFile $relative $repository
 [ordered]@{repository=$repository;path=$relative;bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Verify($binding,[string]$relative,[string]$repository='cuwacunu_embedding'){
 Check (($binding.bytes-is[int]-or$binding.bytes-is[long]-or$binding.bytes-is[ulong])-and$binding.bytes-gt0-and$binding.sha256-is[string]-and$binding.sha256-cmatch'^[0-9a-f]{64}$') 'Positive integral byte count and exact SHA256 type'
 $actual=Pin $relative $repository
 Check ($actual.bytes-eq$binding.bytes-and$actual.sha256-ceq$binding.sha256) ('Exact file binding: '+$repository+'/'+$relative)
 return $actual
}
function Equal($left,$right){($left|ConvertTo-Json -Depth 30 -Compress)-ceq($right|ConvertTo-Json -Depth 30 -Compress)}
function ExactStrings($observed,$expected,[string]$reason){Check (@($observed).Count-eq@($expected).Count-and@($observed|Sort-Object -Unique).Count-eq@($expected).Count-and-not(Compare-Object @($observed|Sort-Object) @($expected|Sort-Object))) $reason}
function ReadMetadata($binding){$null=Verify $binding $binding.path;Get-Content -LiteralPath (HostFile $binding.path) -Raw|ConvertFrom-Json}

$cardRelative='doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json';$cardPin=Pin $cardRelative
Check ($cardPin.sha256-ceq$CardSha256) 'Externally reviewed final card bytes before JSON parse'
$card=Get-Content -LiteralPath (HostFile $cardRelative) -Raw|ConvertFrom-Json
Check ($card.kind-eq'cpb_v8_v9_saved_train_diagnosis_card_v1'-and$card.stage-eq'prospective_before_diagnostic_arithmetic'-and$card.execution_admitted-is[bool]-and-not$card.execution_admitted-and$card.candidate_declared-is[bool]-and-not$card.candidate_declared) 'Prospective frozen card; execution/candidate not declared'
Check ($card.baseline-eq'CPB-v3.market-01'-and$card.train_rows-eq112-and$card.query_rows-eq8-and-not$card.baseline_or_application_change) 'V3 baseline and bounded TRAIN support'
Check (Equal $card.versions @('v8','v9')) 'Only V8/V9'
Check (Equal $card.comparison_references @('CPB-v8.market-01','CPB-v9.market-01')) 'Exact clean comparison identities'
Check ($card.native_widths.v8-eq40-and$card.native_widths.v9-eq40-and(Equal $card.initial_pairing_coordinate_slice @(20,40))-and-not$card.whole_native40_equality_required-and-not$card.first_fresh_mean_variance_equality_required) 'Native40 with partial initial20:40 scope'
Check ($card.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and$card.mean_huber_delta-eq1-and(Equal $card.loss_coefficients @(1.0,1.0))-and$card.variance_error_detached-and-not$card.reporting_nll_backpropagated) 'Unchanged composite objective metadata'
Check ($card.V8_split_norm_inference-eq'forbidden'-and$card.V9_split_norm_inference-eq'forbidden'-and$card.actual_batch_decoder_adjoints-eq'forbidden-no-matched-prestep-decoder-weights') 'No branch inference or unmatched actual-batch adjoints'
Check ((Equal $card.temporal_bins @(@(0,32),@(32,64),@(64,96),@(96,128)))-and(Equal $card.actual_batches @(1,2048))) 'Fixed four original-time bins and actual1/2048 witnesses'
Check ($card.temporal_observation_scope-eq'both-saved8192group2048gauge-two12field-witnesses-V9-between2048') 'Observed-only both-version temporal evidence and V9 between-bin observations'
Check ((Equal $card.temporal_policy_by_version ([ordered]@{v8='original-position-four-bin-disjoint-export-rows-v1';v9='original-position-four-bin-shared-global-half-mixture-v1'}))-and(Equal $card.V9_temporal_mix @(.5,.5))-and$card.V9_empty_local_bin_policy-eq'zero-whole-export-slice-after-bias-v1') 'Distinct original V8/V9 pooling metadata, fixed half mix and empty-bin policy'
Check ($card.between_bin_basis-eq'hadamard-full16-signs-div4-v1'-and(Equal $card.between_bin_signs @(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1)))-and-not$card.individual_final_coefficient_nonzero_gate) 'Three fixed R16 Hadamard directions; no individual final nonzero gate'

$sourceRelative='code/research/causal_v8_v9_train_diagnosis/';$names=@('archive_reader.py','diagnose.py','schema.py','README.md','provenance.json')
ExactStrings @(Get-ChildItem -LiteralPath (HostFile ($sourceRelative+'README.md')|Split-Path -Parent) -File|ForEach-Object Name) $names 'Exactly five owned files'
Check ($card.owned_source_files.Count-eq5) 'Five source pins in card'
$sources=@()
foreach($name in $names){$label=$sourceRelative+$name;$bound=@($card.owned_source_files|Where-Object{$_.path-eq$label});Check ($bound.Count-eq1) 'Unique exact owned source location';$sources+=Verify $bound[0] $label}
$helperRelative='doc/causal_v8_v9_train_diagnosis/run-diagnosis.ps1';$helperPin=Pin $helperRelative
Check ($helperPin.sha256-ceq$OrchestrationHelperSha256) 'Externally reviewed complete phased helper'
$companion=Pin 'doc/causal_v8_v9_train_diagnosis/orchestration-provenance.json'

$roles=@('train_targets')
foreach($version in @('v8','v9')){
 foreach($stage in @('initial','final')){
  foreach($kind in @('features','query','query_summary','checkpoint')){$roles+=($version+'_'+$stage+'_'+$kind)}
  foreach($repeat in @(2701,2802,2903)){foreach($kind in @('head','fit')){$roles+=($version+'_'+$stage+'_'+$kind+'_'+$repeat)}}
 }
 foreach($kind in @('actual_update_1','actual_update_2048','actual_bindings','training_csv')){$roles+=($version+'_'+$kind)}
}
foreach($version in @('v8','v9')){foreach($kind in @('temporal_learning_receipt','temporal_learning_csv','temporal_gauge_csv')){$roles+=($version+'_'+$kind)}}
$roles+='v9_temporal_between_csv'
ExactStrings @($card.allowed_inputs.id) $roles 'Exactly56 unique numeric role identities'
Check ($card.preservation_pins.Count-eq36-and$card.preservation_pin_count-eq36-and@($card.preservation_pins.id|Sort-Object -Unique).Count-eq36) 'Exactly36 preservation identities'
$all=@($card.allowed_inputs)+@($card.preservation_pins)+@($card.pre_execution_preservation)
Check ($all.Count-eq93-and@($all.id|Sort-Object -Unique).Count-eq93) 'Exact93 source-bound file identities'
$verified=@()
foreach($binding in $all){
 Check ($binding.host_repository-in@('cuwacunu_embedding','cuwacunu_dev')-and$binding.container_path-is[string]-and$binding.container_path.StartsWith('/')) 'Actual path/repository types'
 $verified+=Verify $binding $binding.host_path $binding.host_repository
}
foreach($binding in $card.allowed_inputs){
 Check ($binding.container_path-notmatch'(?i)validation') 'No held-out numeric file'
 if($binding.role-eq'TRAIN_archive'){
  Check ($binding.allowed_tensor_fields-is[array]-and$binding.allowed_tensor_fields.Count-gt0-and@($binding.allowed_tensor_fields|Where-Object{$_-isnot[string]-or$_-match'(?i)validation'}).Count-eq0-and@($binding.allowed_tensor_fields|Sort-Object -Unique).Count-eq$binding.allowed_tensor_fields.Count) 'Explicit unique TRAIN-only tensor selectors; no archive decoding here'
 }else{Check ($binding.role-in@('aggregate_JSON_TRAIN_key_only','actual_batch_binding_metadata','TRAIN_CSV_2048_scalar_rows_only','V8_TRAIN_temporal_receipt_restricted_JSON','V8_TRAIN_temporal_8192_rows','V8_TRAIN_gauge_2048_rows','V9_TRAIN_temporal_receipt_restricted_JSON','V9_TRAIN_temporal_8192_rows','V9_TRAIN_gauge_2048_rows','V9_TRAIN_between_bin_2048_rows')) 'Exact scalar/metadata role kind'}
}
$copies=@($card.metadata_copies_required_before_execution);Check ($copies.Count-eq16-and@($copies.container_path|Sort-Object -Unique).Count-eq16-and@($copies.id|Sort-Object -Unique).Count-eq16) 'Sixteen distinct context metadata copies'
foreach($copy in $copies){Check ($copy.container_path.StartsWith('/opt/cuwacunu/research/causal-v8-v9-train-diagnosis/attempt-001/source/preservation/')) 'Bounded owned context copy';$same=@($card.preservation_pins|Where-Object{$_.id-eq$copy.id-and$_.host_repository-eq$copy.host_repository-and$_.host_path-eq$copy.host_path-and$_.container_path-eq$copy.container_path-and$_.bytes-eq$copy.bytes-and$_.sha256-ceq$copy.sha256});Check ($same.Count-eq1) 'Context copy exactly aliases one preservation-only role'}
Check ($card.pre_execution_preservation.container_path-eq'/opt/cuwacunu/research/causal-v8-v9-train-diagnosis/attempt-001/source/final-preservation.json'-and$card.pre_execution_preservation.host_path-eq'doc/causal_fresh_v9/final-preservation.json') 'Exact card-pinned broad final99 guard before any future storage access'

$final=Get-Content -LiteralPath (HostFile $card.pre_execution_preservation.host_path) -Raw|ConvertFrom-Json
Check ($final.kind-eq'causal_fresh_v9_final_preservation_v1'-and$final.status-eq'passed'-and$final.critical_checkpoint_total-eq99-and$final.critical_checkpoint_pins.Count-eq99-and$final.old_pins-eq1030-and$final.selected_V3_source_files_each_repo-eq47) 'Actual final99/old1030/selected47 preservation types'
Check ($final.baseline-eq'CPB-v3.market-01'-and-not$final.baseline_changed-and-not$final.application_changed-and$final.prior_whole_metadata_chain_preserved) 'Protected application/history chain'
Check ($final.all_pinned_PT_ignored-is[array]-and$final.all_pinned_PT_ignored.Count-eq2-and@($final.all_pinned_PT_ignored|Where-Object{-not$_.all_ignored}).Count-eq0) 'Ignore evidence is two repository records, not a scalar boolean'
$critical=@();foreach($pin in $final.critical_checkpoint_pins){$critical+=Verify $pin $pin.path $pin.repository}
Check (@($critical|ForEach-Object{$_.repository+'/'+$_.path}|Sort-Object -Unique).Count-eq99) 'Ninety-nine exact unique available checkpoint byte hashes; no deserialization'
foreach($repository in @('cuwacunu_embedding','cuwacunu_dev')){$baseline=Pin 'output/baselines/CPB-v3.market-01/inference.pt' $repository;Check ($baseline.sha256-ceq'50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e') 'Exact selected V3 checkpoint in each repository'}
$goal=ReadMetadata $card.goal_protocol;$review=ReadMetadata $card.retention_independent_review;$application=ReadMetadata $card.retention_application_admission
Check ($card.goal_protocol.path-eq'doc/clean_causal_encoder_search_goal_v9_retention.json'-and$goal.kind-eq'clean_causal_encoder_search_goal_v9_retention_protocol_v1'-and$goal.criterion_count-eq22-and$goal.criteria.Count-eq22-and-not$goal.additional_train_NLL_gate) 'Actual card-pinned applied future22 protocol; no floor recomputation'
Check ($review.status-eq'passed'-and$application.status-eq'passed'-and$application.prospective_future_use_eligible-and-not$application.application_changed-and-not$application.new_candidate_declared-and-not$application.historical_results_or_counts_recomputed) 'Independent future retention review/application metadata; no new candidate or historical rescoring'
Check ($review.protocol.bytes-eq$card.goal_protocol.bytes-and$review.protocol.sha256-ceq$card.goal_protocol.sha256-and$application.independent_review.bytes-eq$card.retention_independent_review.bytes-and$application.independent_review.sha256-ceq$card.retention_independent_review.sha256) 'Review and application bind the exact card-pinned future protocol'
$applied=@($application.files|Where-Object{$_.path-eq$card.goal_protocol.path-and$_.bytes-eq$card.goal_protocol.bytes-and$_.sha256-ceq$card.goal_protocol.sha256})
Check ($applied.Count-eq1-and$application.criteria-eq22-and-not$application.additional_TRAIN_NLL_gate) 'Actual application copied this future22 protocol; no TRAIN NLL gate'
$ledger=ReadMetadata $card.current_clean_ledger;$catalog=ReadMetadata $card.current_catalog
Check ($ledger.baseline-eq'CPB-v3.market-01'-and$catalog.active_application-eq'CPB-v3.market-01'-and$catalog.fresh_research_candidate.instance-eq'CPB-v9.market-01') 'Actual V3 selection and preserved measured V9 context'
Check ($final.current_ledger.sha256-ceq$card.current_clean_ledger.sha256-and$final.current_catalog.sha256-ceq$card.current_catalog.sha256) 'Final preservation binds exact current context'

# These are opaque preservation/context records, not extra scientific numeric inputs.
function ContextAlias($pin,[string]$label){
 Check ($pin.path-eq$label) 'Explicit canonical context metadata path'
 $null=Verify $pin $label
 $guard=@($card.preservation_pins|Where-Object{$_.host_repository-eq'cuwacunu_embedding'-and$_.host_path-eq$label-and$_.bytes-eq$pin.bytes-and$_.sha256-ceq$pin.sha256})
 $copy=@($copies|Where-Object{$_.host_repository-eq'cuwacunu_embedding'-and$_.host_path-eq$label-and$_.bytes-eq$pin.bytes-and$_.sha256-ceq$pin.sha256})
 Check ($guard.Count-eq1-and$copy.Count-eq1) 'Explicit context pin aliases one preservation-only role and one staged context copy'
}
foreach($binding in @($card.goal_protocol,$card.retention_independent_review,$card.retention_application_admission,$card.current_clean_ledger,$card.current_catalog)){ContextAlias $binding $binding.path}
ContextAlias $card.reporting_recovery_card 'doc/causal_fresh_v9/reporting-recovery-card-001.json'
ContextAlias $card.reporting_recovery_receipt 'doc/causal_fresh_v9/reporting-recovery-receipt-001.json'
ContextAlias $card.before_recovery_failure_mirror 'doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json'
$recoveryCard=ReadMetadata $card.reporting_recovery_card;$recovery=ReadMetadata $card.reporting_recovery_receipt;$failure=ReadMetadata $card.before_recovery_failure_mirror
Check ($card.reporting_recovery_card.sha256-ceq'9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'-and$card.reporting_recovery_receipt.sha256-ceq'5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72'-and$card.before_recovery_failure_mirror.sha256-ceq'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532') 'Exact preserved V9 reporting-only recovery and original failure evidence'
Check ($recoveryCard.kind-eq'cpb_v9_report_only_recovery_card_v1'-and$recoveryCard.root_reviewed-and$recovery.kind-eq'cpb_v9_report_only_recovery_receipt_v1'-and$recovery.status-eq'passed'-and$failure.status-eq'passed') 'Actual root-reviewed recovery metadata and passed failure mirror'
Check ($recovery.card.sha256-ceq$card.reporting_recovery_card.sha256-and$recovery.card.bytes-eq$card.reporting_recovery_card.bytes-and$recovery.scientific_inputs_before.Count-eq531-and(Equal $recovery.scientific_inputs_before $recovery.scientific_inputs_after)-and$recovery.encoder_calls-eq0-and$recovery.head_fits-eq0-and$recovery.training_updates-eq0-and$recovery.science_stages_replayed-eq0) 'Recovery binds its card and unchanged531 original inputs with zero scientific replay'
Check ($final.reporting_recovery_card.sha256-ceq$card.reporting_recovery_card.sha256-and$final.reporting_recovery_receipt.sha256-ceq$card.reporting_recovery_receipt.sha256-and$final.before_recovery_failure_mirror.sha256-ceq$card.before_recovery_failure_mirror.sha256-and$final.science_stages_replayed-eq0) 'Actual final99 preservation retains the explicit report-only recovery chain'
foreach($version in @('v8','v9')){
 $observed=$card.actual_witness_attempt_context.$version
 Check ($observed.attempted_batches-eq2048-and$observed.successful_updates-eq2048-and$observed.skipped_batches-eq0) 'Context-specific known2048attempt/zero-skip observations; not future training policy'
}
Check ($card.actual_witness_attempt_context.scope-eq'these two pinned historical runs only; not a generic future training assumption') 'Explicit limit on inherited actual attempt/update association'

$markdown=Pin 'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.md'
$result=[ordered]@{kind='cpb_v8_v9_saved_TRAIN_diagnosis_independent_static_preflight_v1';status='passed';reviewed_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks;card_sha256=$CardSha256;card=$cardPin;markdown=$markdown;sources=$sources;files=(@($sources)+@($cardPin));reviewer=(Pin 'doc/causal_v8_v9_train_diagnosis/prepare-preflight.ps1');orchestration_helper=$helperPin;orchestration_provenance=$companion;allowed_input_pins=56;preservation_pins=36;pre_execution_preservation=$card.pre_execution_preservation;metadata_copies=16;current_goal=$card.goal_protocol;current_goal_admission=$card.retention_application_admission;verified_file_bindings=$verified;critical_checkpoint_pins=$critical;root_review_of_exact_source_card_acknowledged=$true;scope='Host availability/byte/hash/JSON-shape binding checks after root+independent static source/card review; immutable card binds exact selector allowlists; no producer/diagnosis import or code execution.';limits=@('PT/checkpoints are hashed as opaque bytes, never decoded','No claim of runtime dtype/shape or arithmetic verification; later separately admitted container Schema and diagnosis provide those','No numerical floor reduction or metric calculation','Independent root admission for Stage,Schema,Run remains required');actions=[ordered]@{Docker=0;AST=0;project_modules=0;tensor_values=0;models=0;heads=0;training_updates=0;diagnosis_arithmetic=0;live_metadata_writes=0;Git_mutations=0}}
$stream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($result|ConvertTo-Json -Depth 30).Replace("`r`n","`n")+"`n");$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Pin 'doc/causal_v8_v9_train_diagnosis/preflight-review.json'|ConvertTo-Json

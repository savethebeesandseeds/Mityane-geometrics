# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host file/hash/JSON binding checks only, based on the preserved V6/V7 preflight contract.
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

$cardRelative='doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.json';$cardPin=Pin $cardRelative
Check ($cardPin.sha256-ceq$CardSha256) 'Externally reviewed final card bytes before JSON parse'
$card=Get-Content -LiteralPath (HostFile $cardRelative) -Raw|ConvertFrom-Json
Check ($card.kind-eq'cpb_v6_v8_saved_train_diagnosis_card_v1'-and$card.stage-eq'prospective_before_diagnostic_arithmetic'-and$card.execution_admitted-is[bool]-and-not$card.execution_admitted-and$card.candidate_declared-is[bool]-and-not$card.candidate_declared) 'Prospective frozen card; execution/candidate not declared'
Check ($card.baseline-eq'CPB-v3.market-01'-and$card.train_rows-eq112-and$card.query_rows-eq8-and-not$card.baseline_or_application_change) 'V3 baseline and bounded TRAIN support'
Check (Equal $card.versions @('v6','v8')) 'Only V6/V8'
Check (Equal $card.comparison_references @('CPB-v6.market-01','CPB-v8.market-01')) 'Exact clean comparison identities'
Check ($card.native_widths.v6-eq40-and$card.native_widths.v8-eq40-and(Equal $card.initial_pairing_coordinate_slice @(20,40))-and-not$card.whole_native40_equality_required-and-not$card.first_fresh_mean_variance_equality_required) 'Native40 with partial initial20:40 scope'
Check ($card.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and$card.mean_huber_delta-eq1-and(Equal $card.loss_coefficients @(1.0,1.0))-and$card.variance_error_detached-and-not$card.reporting_nll_backpropagated) 'Unchanged composite objective metadata'
Check ($card.V6_split_norm_inference-eq'forbidden'-and$card.V8_split_norm_inference-eq'forbidden'-and$card.actual_batch_decoder_adjoints-eq'forbidden-no-matched-prestep-decoder-weights') 'No branch inference or unmatched actual-batch adjoints'
Check ((Equal $card.temporal_bins @(@(0,32),@(32,64),@(64,96),@(96,128)))-and(Equal $card.actual_batches @(1,2048))) 'Fixed four original-time bins and actual1/2048 witnesses'
Check ($card.temporal_observation_scope-eq'V8-only-saved8192group2048gauge-two12field-witnesses') 'Observed-only V8 temporal evidence'

$sourceRelative='code/research/causal_v6_v8_train_diagnosis/';$names=@('archive_reader.py','diagnose.py','schema.py','README.md','provenance.json')
ExactStrings @(Get-ChildItem -LiteralPath (HostFile ($sourceRelative+'README.md')|Split-Path -Parent) -File|ForEach-Object Name) $names 'Exactly five owned files'
Check ($card.owned_source_files.Count-eq5) 'Five source pins in card'
$sources=@()
foreach($name in $names){$label=$sourceRelative+$name;$bound=@($card.owned_source_files|Where-Object{$_.path-eq$label});Check ($bound.Count-eq1) 'Unique exact owned source location';$sources+=Verify $bound[0] $label}
$helperRelative='doc/causal_v6_v8_train_diagnosis/run-diagnosis.ps1';$helperPin=Pin $helperRelative
Check ($helperPin.sha256-ceq$OrchestrationHelperSha256) 'Externally reviewed complete phased helper'
$companion=Pin 'doc/causal_v6_v8_train_diagnosis/orchestration-provenance.json'

$roles=@('train_targets')
foreach($version in @('v6','v8')){
 foreach($stage in @('initial','final')){
  foreach($kind in @('features','query','query_summary','checkpoint')){$roles+=($version+'_'+$stage+'_'+$kind)}
  foreach($repeat in @(2701,2802,2903)){foreach($kind in @('head','fit')){$roles+=($version+'_'+$stage+'_'+$kind+'_'+$repeat)}}
 }
 foreach($kind in @('actual_update_1','actual_update_2048','actual_bindings','training_csv')){$roles+=($version+'_'+$kind)}
}
$roles+=@('v8_temporal_learning_receipt','v8_temporal_learning_csv','v8_temporal_gauge_csv')
ExactStrings @($card.allowed_inputs.id) $roles 'Exactly52 unique numeric role identities'
Check ($card.preservation_pins.Count-eq33-and$card.preservation_pin_count-eq33-and@($card.preservation_pins.id|Sort-Object -Unique).Count-eq33) 'Exactly33 preservation identities'
$all=@($card.allowed_inputs)+@($card.preservation_pins)+@($card.pre_execution_preservation)
Check ($all.Count-eq86-and@($all.id|Sort-Object -Unique).Count-eq86) 'Exact86 source-bound file identities'
$verified=@()
foreach($binding in $all){
 Check ($binding.host_repository-in@('cuwacunu_embedding','cuwacunu_dev')-and$binding.container_path-is[string]-and$binding.container_path.StartsWith('/')) 'Actual path/repository types'
 $verified+=Verify $binding $binding.host_path $binding.host_repository
}
foreach($binding in $card.allowed_inputs){
 Check ($binding.container_path-notmatch'(?i)validation') 'No held-out numeric file'
 if($binding.role-eq'TRAIN_archive'){
  Check ($binding.allowed_tensor_fields-is[array]-and$binding.allowed_tensor_fields.Count-gt0-and@($binding.allowed_tensor_fields|Where-Object{$_-isnot[string]-or$_-match'(?i)validation'}).Count-eq0-and@($binding.allowed_tensor_fields|Sort-Object -Unique).Count-eq$binding.allowed_tensor_fields.Count) 'Explicit unique TRAIN-only tensor selectors; no archive decoding here'
 }else{Check ($binding.role-in@('aggregate_JSON_TRAIN_key_only','actual_batch_binding_metadata','TRAIN_CSV_2048_scalar_rows_only','V8_TRAIN_temporal_receipt_restricted_JSON','V8_TRAIN_temporal_8192_rows','V8_TRAIN_gauge_2048_rows')) 'Exact scalar/metadata role kind'}
}
$copies=@($card.metadata_copies_required_before_execution);Check ($copies.Count-eq13) 'Thirteen context metadata copies'
foreach($copy in $copies){Check ($copy.container_path.StartsWith('/opt/cuwacunu/research/causal-v6-v8-train-diagnosis/attempt-001/source/preservation/')) 'Bounded owned context copy';$same=@($card.preservation_pins|Where-Object{$_.id-eq$copy.id-and$_.host_repository-eq$copy.host_repository-and$_.host_path-eq$copy.host_path-and$_.container_path-eq$copy.container_path-and$_.bytes-eq$copy.bytes-and$_.sha256-ceq$copy.sha256});Check ($same.Count-eq1) 'Context copy exactly aliases one preservation-only role'}
Check ($card.pre_execution_preservation.container_path-eq'/opt/cuwacunu/research/causal-v6-v8-train-diagnosis/attempt-001/source/final-preservation.json'-and$card.pre_execution_preservation.sha256-ceq'919b4f11610e3ab61b63afbc3612145506e4c47a3be3c777f119d47c49e7126a') 'Exact broad final94 guard before any future storage access'

$final=Get-Content -LiteralPath (HostFile $card.pre_execution_preservation.host_path) -Raw|ConvertFrom-Json
Check ($final.kind-eq'causal_fresh_v8_final_preservation_v1'-and$final.status-eq'passed'-and$final.critical_checkpoint_total-eq94-and$final.critical_checkpoint_pins.Count-eq94-and$final.old_pins-eq1030-and$final.selected_V3_source_files_each_repo-eq47) 'Actual final94/old1030/selected47 preservation types'
Check ($final.baseline-eq'CPB-v3.market-01'-and-not$final.baseline_changed-and-not$final.application_changed-and$final.prior_whole_metadata_chain_preserved) 'Protected application/history chain'
Check ($final.all_pinned_PT_ignored-is[array]-and$final.all_pinned_PT_ignored.Count-eq2-and@($final.all_pinned_PT_ignored|Where-Object{-not$_.all_ignored}).Count-eq0) 'Ignore evidence is two repository records, not a scalar boolean'
$critical=@();foreach($pin in $final.critical_checkpoint_pins){$critical+=Verify $pin $pin.path $pin.repository}
Check (@($critical|ForEach-Object{$_.repository+'/'+$_.path}|Sort-Object -Unique).Count-eq94) 'Ninety-four exact unique available checkpoint byte hashes; no deserialization'
foreach($repository in @('cuwacunu_embedding','cuwacunu_dev')){$baseline=Pin 'output/baselines/CPB-v3.market-01/inference.pt' $repository;Check ($baseline.sha256-ceq'50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e') 'Exact selected V3 checkpoint in each repository'}
$goal=ReadMetadata $card.goal_protocol;$review=ReadMetadata $card.retention_independent_review;$application=ReadMetadata $card.retention_application_admission
Check ($card.goal_protocol.sha256-ceq'e7920442fbc5528cbe7e67f71ada65101974a6a28ad4d39bc57dea461c477699'-and$goal.kind-eq'clean_causal_encoder_search_goal_v8_retention_protocol_v1'-and$goal.criterion_count-eq22-and$goal.criteria.Count-eq22-and-not$goal.additional_train_NLL_gate) 'Applied future22 protocol; no floor recomputation'
Check ($review.status-eq'passed'-and$application.status-eq'passed') 'Independent future retention review/application metadata'
$ledger=ReadMetadata $card.current_clean_ledger;$catalog=ReadMetadata $card.current_catalog
Check ($ledger.baseline-eq'CPB-v3.market-01'-and$catalog.active_application-eq'CPB-v3.market-01'-and$catalog.fresh_research_candidate.instance-eq'CPB-v8.market-01') 'Actual V3 selection and preserved measured V8 context'
Check ($final.current_ledger.sha256-ceq$card.current_clean_ledger.sha256-and$final.current_catalog.sha256-ceq$card.current_catalog.sha256) 'Final preservation binds exact current context'

$markdown=Pin 'doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.md'
$result=[ordered]@{kind='cpb_v6_v8_saved_TRAIN_diagnosis_independent_static_preflight_v1';status='passed';reviewed_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks;card_sha256=$CardSha256;card=$cardPin;markdown=$markdown;sources=$sources;files=(@($sources)+@($cardPin));reviewer=(Pin 'doc/causal_v6_v8_train_diagnosis/prepare-preflight.ps1');orchestration_helper=$helperPin;orchestration_provenance=$companion;allowed_input_pins=52;preservation_pins=33;pre_execution_preservation=$card.pre_execution_preservation;metadata_copies=13;current_goal=$card.goal_protocol;current_goal_admission=$card.retention_application_admission;verified_file_bindings=$verified;critical_checkpoint_pins=$critical;root_review_of_exact_source_card_acknowledged=$true;scope='Host availability/byte/hash/JSON-shape binding checks after root+independent static source/card review; immutable card binds exact selector allowlists; no producer/diagnosis import or code execution.';limits=@('PT/checkpoints are hashed as opaque bytes, never decoded','No claim of runtime dtype/shape or arithmetic verification; later separately admitted container Schema and diagnosis provide those','No numerical floor reduction or metric calculation','Independent root admission for Stage,Schema,Run remains required');actions=[ordered]@{Docker=0;AST=0;project_modules=0;tensor_values=0;models=0;heads=0;training_updates=0;diagnosis_arithmetic=0;live_metadata_writes=0;Git_mutations=0}}
$stream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($result|ConvertTo-Json -Depth 30).Replace("`r`n","`n")+"`n");$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Pin 'doc/causal_v6_v8_train_diagnosis/preflight-review.json'|ConvertTo-Json

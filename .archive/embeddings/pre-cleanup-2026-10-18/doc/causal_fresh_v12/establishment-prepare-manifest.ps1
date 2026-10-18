# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Owned host metadata generator physically copied from the complete final V11 schema.
# File/hash/JSON inspection only. No Docker, AST, project body or tensor access.
param(
 [Parameter(Mandatory=$true)][string]$InputPlan,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InputPlanSHA256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedCompletedInputs
)
$ErrorActionPreference='Stop'
if(-not $RootReviewedCompletedInputs){throw 'Actual completed quality/audit/Complete inputs require root review'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$destination='doc/causal_fresh_v12/establishment-input-manifest.json'
function Bounded([string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains([char]92)-or$relative-match'(^|/)\.\.(/|$)'){throw 'Forward-slash research-relative file path required'}
 $path=[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))
 if(-not $path.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Path escaped research root'}
 $path
}
function File-Pin([string]$relative){
 $path=Bounded $relative;$file=Get-Item -LiteralPath $path
 if($file.PSIsContainer-or($file.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Physical metadata file required'}
 [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Verify-Pin($expected){
 if($null-eq$expected.path-or$null-eq$expected.bytes-or$expected.sha256-notmatch'^[0-9a-f]{64}$'){throw 'Explicit path/bytes/SHA256 binding required'}
 $actual=File-Pin $expected.path
 if($actual.bytes-ne$expected.bytes-or$actual.sha256-ne$expected.sha256){throw ('Reviewed metadata changed: '+$expected.path)}
 $actual
}
function Read-Pinned($binding){$null=Verify-Pin $binding;Get-Content -Raw -LiteralPath (Bounded $binding.path)|ConvertFrom-Json}
if(Test-Path -LiteralPath (Bounded $destination)){throw 'Preserve the existing join card; additive attempts require a separate declaration'}
$planPin=File-Pin $InputPlan
if($planPin.sha256-ne$InputPlanSHA256){throw 'External input-plan hash required before JSON parse'}
$plan=Get-Content -Raw -LiteralPath (Bounded $InputPlan)|ConvertFrom-Json
if($plan.kind-ne'causal_fresh_v12_establishment_input_plan_v1'-or$plan.status-ne'root_reviewed_completed_quality_audits_and_mirror'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true){throw 'Explicit actual completed-input plan admission required'}
$own=File-Pin ([IO.Path]::GetRelativePath($researchRoot,$PSCommandPath).Replace([char]92,[char]47))
if($plan.generator.path-ne$own.path-or$plan.generator.bytes-ne$own.bytes-or$plan.generator.sha256-ne$own.sha256){throw 'Exact root-reviewed generator source required'}
$roles=@('comparison','table_rows','quality_audit','training_audit','complete_mirror','card','quality_receipt','pairings','source_manifest','build_admission','temporal_pooling_proof','temporal_learning','prospective_freeze','protocol_preflight')
$ids=@($plan.inputs.id)
if(@($ids|Sort-Object -Unique).Count-ne$ids.Count-or$ids.Count-lt14-or$ids.Count-gt15-or@($roles|Where-Object {$_-notin$ids}).Count-or@($ids|Where-Object {$_-notin($roles+@('declaration_review'))}).Count){throw 'Exactly14 required JSON roles; only optional declaration_review'}
$requiredSources=@(
 'code/research/causal_fresh_v12_establishment/establish.py',
 'code/research/causal_fresh_v12_establishment/README.md',
 'code/research/causal_fresh_v12_establishment/provenance.json',
 'doc/causal_fresh_v12/establishment-stage.ps1',
 'doc/causal_fresh_v12/establishment-execute.py',
 'doc/causal_fresh_v12/establishment-mirror.ps1')
$reviewedSources=@($plan.reviewed_sources)
if($reviewedSources.Count-ne6-or@($reviewedSources.path|Sort-Object -Unique).Count-ne6-or(Compare-Object @($reviewedSources.path|Sort-Object) @($requiredSources|Sort-Object))){throw 'Exactly six independently reviewed join/support/transport source pins required'}
$sources=@{}
foreach($pin in $reviewedSources){$sources[$pin.path]=Verify-Pin $pin}
$inputs=@();$data=@{};$bindings=@{}
foreach($row in $plan.inputs){
 if(-not $row.path.EndsWith('.json')){throw 'Metadata inputs must be explicit JSON files'}
 $binding=Verify-Pin $row
 $data[$row.id]=Read-Pinned $binding;$bindings[$row.id]=$binding
 $inputs +=[ordered]@{id=$row.id;path=$binding.path;bytes=$binding.bytes;sha256=$binding.sha256;staged_path=('inputs/'+$row.id+'.json')}
}
if($bindings.pairings.path-ne$bindings.quality_audit.path-or$bindings.pairings.bytes-ne$bindings.quality_audit.bytes-or$bindings.pairings.sha256-ne$bindings.quality_audit.sha256){throw 'Pairings must be an exact alias of the complete quality audit receipt'}
$archive=File-Pin 'output/engineering/causal-fresh-v12/attempt-001/source.zip'
$identity=[ordered]@{card_sha256=$bindings.card.sha256;manifest_sha256=$bindings.source_manifest.sha256;archive_sha256=$archive.sha256;source_files=$data.source_manifest.files.Count;binary_count=$data.card.source_counts.binaries;translation_units=$data.card.source_counts.translation_units}
if($data.card.instance-ne'CPB-v12.market-01'-or$data.card.stage.Contains('DRAFT')-or-not$data.card.frozen_utc-or-not$data.card.execution_admitted-or$identity.source_files-ne137-or$data.card.source_counts.capsule_sources-ne$identity.source_files-or$data.source_manifest.card_sha256-ne$identity.card_sha256){throw 'Actual frozen V12 declaration and its137-file source capsule required'}
if($data.card.scientific_recipe_parent-ne'CPB-v11 source only;fresh weights/optimizer/statistics,not trained ancestry'-or$data.card.scoped_reference-ne'CPB-v11.market-01'-or$data.prospective_freeze.critical_checkpoints_verified-ne109){throw 'Fresh V11-source-only recipe and all109 protected historical checkpoints required'}
if($data.quality_audit.kind-ne'causal_fresh_v12_quality_independent_audit_v1'-or$data.quality_audit.status-ne'passed'-or$data.training_audit.kind-ne'causal_fresh_v12_training_independent_audit_v1'-or$data.training_audit.status-ne'passed'){throw 'Both independently completed audits must pass before join-card declaration'}
if($data.complete_mirror.kind-ne'causal_fresh_v12_container_to_local_evidence_mirror_v1'-or$data.complete_mirror.status-ne'passed'-or$data.complete_mirror.scope-ne'Complete'-or$data.complete_mirror.attempt-ne'attempt-001'-or$data.complete_mirror.source_files_verified-ne$identity.source_files-or$data.complete_mirror.file_count-ne$data.complete_mirror.files.Count-or-not$data.complete_mirror.all_mirrored_pt_ignored){throw 'Actual exact normal Complete mirror required; counts derived from the receipt'}
if($data.complete_mirror.source_stage.status-ne'passed'-or$data.complete_mirror.source_stage.card_sha256-ne$identity.card_sha256-or$data.complete_mirror.source_stage.manifest_sha256-ne$identity.manifest_sha256-or$data.complete_mirror.source_stage.archive_sha256-ne$identity.archive_sha256){throw 'Complete mirror must bind the same immutable source transport'}
if($data.quality_receipt.status-ne'passed'-or$data.quality_receipt.card_sha256-ne$identity.card_sha256-or$data.build_admission.status-ne'passed'-or$data.build_admission.source_files-ne$identity.source_files-or$data.build_admission.binaries.Count-ne7-or$identity.binary_count-ne7-or$identity.translation_units-ne11-or@($data.build_admission.dependency_closures.PSObject.Properties).Count-ne11){throw 'Completed normal quality and actual seven-binary/eleven-TU build required'}
if($data.prospective_freeze.kind-ne'causal_fresh_v12_prospective_freeze_v1'-or$data.prospective_freeze.status-ne'passed'-or$data.prospective_freeze.card.sha256-ne$identity.card_sha256-or$data.prospective_freeze.declared_capsule_sources-ne$identity.source_files-or$data.protocol_preflight.status-ne'passed'-or$data.protocol_preflight.card_sha256-ne$identity.card_sha256-or($data.protocol_preflight.sources.Count+1)-ne$identity.source_files){throw 'Actual normal prospective freeze/preflight bindings required'}
$objective='huber1-mean-plus-detached-error-gaussian-variance-v1';$gradient='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
foreach($record in @($data.card.training,$data.table_rows,$data.comparison,$data.training_audit,$data.quality_audit)){if($record.objective_policy-ne$objective-or$record.gradient_policy-ne$gradient-or$record.variance_error_detached-ne$true-or$record.reporting_nll_backpropagated-ne$false){throw 'Actual V9 detached-error objective plus V12 mean4/centered-variance4 contract required'}}
$pool='original-position-four-bin-attention-mean4-uniform-centered-variance4-v1'
$moment='componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1'
$layout='per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'
foreach($record in @($data.card.mechanism,$data.table_rows,$data.comparison,$data.training_audit,$data.quality_audit,$data.temporal_pooling_proof,$data.temporal_learning)){
 if($record.temporal_pooling_policy-ne$pool-or($record.temporal_pooling_partition-join',')-ne'4,16,32,4,8'-or($record.temporal_pooling_mix-join',')-ne'0.5,0.5'-or$record.empty_local_bin_policy-ne'zero-whole-export-slice-after-bias-v1'-or$record.temporal_moment_policy-ne$moment-or$record.temporal_export_layout-ne$layout-or($record.temporal_moment_partition-join',')-ne'4,4'){throw 'Seven exact current pooling/moment fields must agree across actual sources'}
}
$pairs=$data.quality_audit.paired_initial_equality
if($pairs.reference_data_archives-ne13-or$pairs.feature_archives-ne8-or$pairs.statistic_archives-ne5-or($pairs.feature_coordinate_slice-join',')-ne'20,40'-or$pairs.native_coordinates_compared-ne20-or$pairs.content0_20_equality_required-ne$false-or$pairs.whole_native40_equality_claimed-ne$false-or-not$pairs.all_declared_numeric_tensor_values_exact){throw 'Actual partial20:40 plus five whole-statistic pairing must pass;historical V10 whole40 is separate'}
$local=$data.training_audit.temporal_pooling_saved_learning.temporal_moment_saved_learning
if($local.status-ne'passed'-or$local.mismatches.Count-ne0-or$local.moment_subgroup_rows_checked-ne16384-or$local.whole_group_two_moment_partition_rows_checked-ne8192-or-not$local.whole_group_partition_squared_norms_and_counts_verified-or$local.actual_same_forward_moment_witnesses.Count-ne2-or$data.temporal_learning.moment_subgroups.Count-ne8){throw 'Actual independent eight-subgroup/wholepartition/two-witness local audit required'}
foreach($witness in $local.actual_same_forward_moment_witnesses){
 if($witness.total_component_archive_fields-ne60-or$witness.new_temporal_trace_fields-ne25-or$witness.new_uniform_dispersion_trace_fields-ne4-or$witness.parameter_array_fields-ne12-or$witness.uniform_dispersion_direct_score_adjoint-ne0-or$witness.local_hidden_adjoint_array_available-ne$false-or$witness.score_to_hidden_route_reconstructed-ne$false-or$witness.hidden_reference_is_observed_gradient_parity-ne$false-or$witness.full_hidden_encoder_or_global_norm_reconstruction-ne$false-or$witness.matched_pre_step_decoder_adjoint_replay-ne$false){throw 'Actual60/25 dual-support fields and descriptive-only fixed-score hidden reference required'}
}
$dualSupportFlags=@('attention_mean_and_uniform_dispersion_full_expression_adjoints','uniform_dispersion_own_center','uniform_dispersion_score_independent','uniform_dispersion_hidden_gradient_live','attention_mean_score_gradient_live','dual_support_actual_trace')
foreach($flag in $dualSupportFlags){if($data.temporal_pooling_proof.$flag-ne$true){throw ('Actual dual-support fixture predicate required: '+$flag)}}
if($data.temporal_pooling_proof.direct_pool_kernel_calls-ne47-or$data.temporal_pooling_proof.F64_finite_difference_coordinates-ne18-or$data.temporal_pooling_proof.F32_finite_difference_coordinates-ne18-or'full_attention_dependent_moment_adjoints'-in@($data.temporal_pooling_proof.PSObject.Properties.Name)){throw 'Source-bound47 calls/18 F64/18 F32;no stale V11 learned-variance predicate'}
if($data.card.advancement.goal_protocol.sha256-ne'866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'){throw 'Exact current prospective22 protocol required'}
foreach($relative in @('training-stage.json','initial-saved-proof-stage.json','final-saved-proof-stage.json','quality-stage.json','quality-001/extraction-stage.json','quality-001/initial_market_heads-stage.json','quality-001/initial_synthetic_heads-stage.json','quality-001/final_market_heads-stage.json','quality-001/final_synthetic_heads-stage.json','quality-001/report-stage.json','quality-001/comparison-stage.json')){
 $containerPath='/opt/cuwacunu/research/causal-fresh-v12/attempt-001/'+$relative
 $matches=@($data.complete_mirror.files|Where-Object{$_.container_path-eq$containerPath});if($matches.Count-ne1){throw 'Every normal completed stage must belong to Complete mirror'}
 $stage=Read-Pinned $matches[0];if($stage.exit_code-ne0){throw 'Normal V12 quality/proofs must pass without an anticipated recovery'}
}
# These checks admit only a prospective JSON/hash join, never establish candidate
# trust or numerical advancement on the host. The separately reviewed join body
# validates the complete evidence graph inside the existing managed container.
$record=[ordered]@{
 kind='causal_fresh_v12_establishment_join_card_v1';stage='declared_after_completed_audits_before_metadata_join';declared_utc=[DateTime]::UtcNow.ToString('o');root_reviewed=$true;
 independent_actual_input_review_required_before_execution=$true;
 source=$sources['code/research/causal_fresh_v12_establishment/establish.py'];
 source_support=@($sources['code/research/causal_fresh_v12_establishment/README.md'],$sources['code/research/causal_fresh_v12_establishment/provenance.json']);
 execution_wrapper=$sources['doc/causal_fresh_v12/establishment-execute.py'];stage_helper=$sources['doc/causal_fresh_v12/establishment-stage.ps1'];mirror_helper=$sources['doc/causal_fresh_v12/establishment-mirror.ps1'];
 main_source_identity=$identity;inputs=$inputs;metadata_generator=$own;actual_input_plan=$planPin;
 source_static_review='Independently reviewed owned full V11-derived join and transports; actual input-card review and root Stage/body admission remain separate';
 model_calls=0;head_fits=0;training_updates=0;scientific_metric_recomputations=0;Docker_calls=0;AST_calls=0;project_body_calls=0;
 raw_producer_pending_fields_must_remain_exact=$true;baseline='CPB-v3.market-01';application_changed=$false;goal_completion_not_assessed=$true}
foreach($pin in @($reviewedSources)+@($plan.inputs)){$null=Verify-Pin $pin}
if((File-Pin $InputPlan).sha256-ne$InputPlanSHA256){throw 'Input plan changed during metadata generation'}
$outputBytes=[Text.UTF8Encoding]::new($false).GetBytes((($record|ConvertTo-Json -Depth 25).Replace("`r`n","`n")+"`n"))
$stream=[IO.File]::Open((Bounded $destination),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$stream.Write($outputBytes,0,$outputBytes.Length)}finally{$stream.Dispose()}
File-Pin $destination|ConvertTo-Json

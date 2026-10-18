# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Full physical final V13 normal workflow SOURCE copy; owned V14 metadata adapter.
# No Docker, project import, AST, auditor body, model or numerical archive execution.
param([Parameter(Mandatory=$true)][switch]$SourcesReviewed,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainZipSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$TrainingMirrorSHA256,
 [Parameter(Mandatory=$true)][string]$SourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SourceReviewSHA256,
 [Parameter(Mandatory=$true)][string]$WorkflowSourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WorkflowSourceReviewSHA256)
$ErrorActionPreference='Stop'
if(-not$SourcesReviewed){throw 'Independent scientific and workflow source review must precede metadata preparation'}
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$output=Join-Path $PSScriptRoot 'audit-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve existing audit card;no implicit retry'}
function Rooted([string]$relative){
 if([IO.Path]::IsPathRooted($relative)){throw 'Repository-relative metadata paths only'}
 $path=[IO.Path]::GetFullPath((Join-Path $repo $relative))
 if(-not$path.StartsWith($repo.TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Metadata path outside repository'}
 for($walk=$path;$walk-ne$repo;$walk=[IO.Path]::GetDirectoryName($walk)){if(-not$walk-or((Get-Item -LiteralPath $walk).Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Physical metadata ancestors required'}}
 $path
}
function Pin([string]$relative){$f=Get-Item -LiteralPath (Rooted $relative);if($f.PSIsContainer){throw 'Expected file'};[ordered]@{path=$relative.Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Read([string]$relative){Get-Content -LiteralPath (Rooted $relative) -Raw|ConvertFrom-Json -AsHashtable}
function Exact($a,$b){if($a.bytes-ne$b.bytes-or$a.sha256-cne$b.sha256){throw 'Exact reviewed metadata/source bytes and SHA required'}}
function Fields($actual,$expected,[string]$why){
 if($expected-is[Collections.IDictionary]){if($actual-isnot[Collections.IDictionary]){throw $why};foreach($key in $expected.Keys){if(-not$actual.Contains($key)){throw $why};Fields $actual[$key] $expected[$key] ($why+':'+$key)}}
 elseif($expected-is[Array]){if($actual-isnot[Array]-or$actual.Count-ne$expected.Count){throw $why};for($index=0;$index-lt$expected.Count;$index++){Fields $actual[$index] $expected[$index] $why}}
 elseif($expected-is[bool]){if($actual-isnot[bool]-or$actual-ne$expected){throw $why}}
 elseif($expected-is[ValueType]){if($actual-isnot[ValueType]-or$actual-is[bool]-or$actual-ne$expected){throw $why}}
 elseif($actual-cne$expected){throw $why}
}
$reviewPin=Pin $SourceReviewPath
if($reviewPin.sha256-cne$SourceReviewSHA256){throw 'External scientific review hash before parse'}
$review=Read $SourceReviewPath
if($review.kind-ne'cpb_v14_family_and_training_auditor_independent_static_review_v1'-or$review.status-ne'passed'-or-not$review.Contains('blockers')-or$review.blockers-isnot[Array]-or$review.blockers.Count-ne0-or$review.reviewed_owned_files.Count-ne88-or@($review.reviewed_owned_files.path|Sort-Object -Unique).Count-ne88){throw 'Passed scientific88:88 complete reviewed_owned_files and explicit blockers[] required'}
$auditor=Pin 'code/research/causal_fresh_v14_training_audit/audit_training.py'
$readme=Pin 'code/research/causal_fresh_v14_training_audit/README.md'
$provenance=Pin 'code/research/causal_fresh_v14_training_audit/provenance.json'
$auditorPaths=@($auditor.path,$readme.path,$provenance.path)
$reviewedSources=@($review.reviewed_owned_files|Where-Object{$_.path-cin$auditorPaths})
if($reviewedSources.Count-ne3-or@($reviewedSources.path|Sort-Object -Unique).Count-ne3){throw 'Select exact3 actual auditor paths from independently reviewed88'}
foreach($pin in @($auditor,$readme,$provenance)){$matches=@($reviewedSources|Where-Object{$_.path-ceq$pin.path});if($matches.Count-ne1){throw 'Exact independently reviewed auditor path'};Exact $pin $matches[0]}
$workflowReviewPin=Pin $WorkflowSourceReviewPath
if($workflowReviewPin.sha256-cne$WorkflowSourceReviewSHA256){throw 'External workflow review hash before parse'}
$workflowReview=Read $WorkflowSourceReviewPath
if($workflowReview.kind-ne'cpb_v14_training_audit_workflow_independent_static_review_v1'-or$workflowReview.status-ne'passed'-or-not$workflowReview.Contains('blockers')-or$workflowReview.blockers-isnot[Array]-or$workflowReview.blockers.Count-ne0){throw 'Separate passed7-file workflow review with explicit blockers[] required'}
$prefix='doc/causal_fresh_v14_training_audit/'
$workflowNames=@('prepare-card.ps1','stage.ps1','run.ps1','mirror.ps1','execute.py','provenance.json','README.md')
$workflowSources=@($workflowReview.reviewed_owned_files)
if($workflowSources.Count-ne7-or@($workflowSources.path|Sort-Object -Unique).Count-ne7){throw 'Exactly7 complete externally reviewed workflow files'}
foreach($name in $workflowNames){$current=Pin ($prefix+$name);$matches=@($workflowSources|Where-Object{$_.path-ceq$current.path});if($matches.Count-ne1){throw 'Exact seven workflow path set'};Exact $current $matches[0]}
$workflowProvenance=Read ($prefix+'provenance.json')
if($reviewPin.path-cne$workflowProvenance.independent_scientific_source_review.path){throw 'Actual scientific002 authority required'}
Exact $reviewPin $workflowProvenance.independent_scientific_source_review
$mainPin=Pin 'doc/CAUSAL_FRESH_V14_CARD.json'
if($mainPin.sha256-cne$MainCardSHA256){throw 'External main card hash before parse'}
if($mainPin.path-cne$workflowProvenance.expected_main_paths.card){throw 'Exact declared future main-card path'}
$main=Read $mainPin.path
$md=Pin 'doc/CAUSAL_FRESH_V14_CARD.md'
$input=Pin 'doc/causal_fresh_v14/input-manifest.json'
if($md.path-cne$workflowProvenance.expected_main_paths.markdown){throw 'Exact declared future Markdown path'}
Exact $input $main.input_manifest
if($main.model-ne'CPB-v14'-or$main.instance-ne'CPB-v14.market-01'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or$main.execution_admitted-ne$true-or$main.geometry.native_width-ne40){throw 'Actual frozen admitted native40 V14 main declaration'}
$manifestPin=Pin 'output/engineering/causal-fresh-v14/attempt-001/source-manifest.json'
if($manifestPin.sha256-cne$MainManifestSHA256){throw 'External main manifest hash before parse'}
if($manifestPin.path-cne$workflowProvenance.expected_main_paths.source_manifest){throw 'Exact declared future manifest path'}
$manifest=Read $manifestPin.path
$zip=Pin 'output/engineering/causal-fresh-v14/attempt-001/source.zip'
if($zip.sha256-cne$MainZipSHA256){throw 'External actual whole main ZIP hash before package parse'}
$package=Read 'output/engineering/causal-fresh-v14/attempt-001/package.json'
if($manifest.card_sha256-cne$MainCardSHA256-or$manifest.files.Count-ne$main.source_counts.capsule_sources-or$manifest.files.Count-le0-or$package.source_files-ne$manifest.files.Count-or$package.manifest_sha256-cne$MainManifestSHA256-or$package.zip_sha256-cne$zip.sha256){throw 'Actual current main source count and immutable whole package'}
$preflightPin=Pin 'doc/causal_fresh_v14/protocol-preflight.json'
foreach($entry in @($mainPin,$md,$input,$preflightPin)){$matches=@($manifest.files|Where-Object{$_.path-ceq$entry.path});if($matches.Count-ne1){throw 'Exact main declaration/input/preflight role in capsule'};Exact $entry $matches[0]}
$preflight=Read $preflightPin.path
if($preflight.status-ne'passed'-or$preflight.card_sha256-cne$MainCardSHA256-or($preflight.sources.Count+1)-ne$manifest.files.Count){throw 'Passed actual main preflight/capsule closure'}
$goalPin=Pin $main.advancement.goal_protocol.path
Exact $goalPin $main.advancement.goal_protocol
$goalEntry=@($manifest.files|Where-Object{$_.path-ceq$goalPin.path})
if($goalEntry.Count-ne1){throw 'Strongest protocol uniquely in main capsule'}
Exact $goalPin $goalEntry[0]
$goal=Read $goalPin.path
Exact $goalPin $workflowProvenance.canonical_retention_protocol
if($goal.kind-ne'clean_causal_encoder_search_goal_v13_retention_protocol_v1'-or$main.advancement.criterion_count-ne22-or$goal.criteria.Count-ne22-or$goal.additional_train_NLL_gate-ne$false){throw 'Actual strongest V3-V13/22,noTRAIN-NLLgate'}
Fields $goal.criteria $main.advancement.criteria 'Full ordered prospective criterion objects'
$retentionReviewPin=Pin $main.retention_independent_review.path;Exact $retentionReviewPin $main.retention_independent_review
$retentionApplicationPin=Pin $main.retention_application_admission.path;Exact $retentionApplicationPin $main.retention_application_admission
$retentionReview=Read $retentionReviewPin.path;$retentionApplication=Read $retentionApplicationPin.path
if($retentionReview.kind-ne'clean_causal_encoder_search_goal_v13_retention_independent_review_v1'-or$retentionReview.status-ne'passed'-or$retentionReview.checks-le0-or$retentionApplication.kind-ne'clean_causal_encoder_search_goal_v13_retention_future_application_v1'-or$retentionApplication.status-ne'passed'-or$retentionApplication.applied-ne$true-or$retentionApplication.prospective_future_use_eligible-ne$true-or$retentionApplication.application_changed-ne$false-or$retentionApplication.historical_results_or_counts_recomputed-ne$false-or$retentionApplication.additional_TRAIN_NLL_gate-ne$false){throw 'Actual independently reviewed/applied future protocol, no invented receipts or historical rescoring'}
Exact $retentionReview.protocol $goalPin;Exact $retentionApplication.protocol $goalPin;Exact $retentionApplication.independent_review $retentionReviewPin
$predecessorPin=Pin $main.baseline_preservation.path
Exact $predecessorPin $main.baseline_preservation
$predecessor=Read $predecessorPin.path
if($predecessorPin.sha256-ne'ae11f853b6bd8462c766a939f69a513922aaa0d791a5bf9b156e485692f7adce'-or$predecessor.status-ne'passed'-or$predecessor.critical_checkpoint_pins.Count-ne119-or$predecessor.V13_source_files-ne143-or$main.latest_frozen_source_preservation.source_files-ne143){throw 'Current completedV13 ancestry119/143 distinct from historical866/V11-109 and current V14 count'}
$mirrorPin=Pin 'doc/causal_fresh_v14/training-mirror-attempt-001.json'
if($mirrorPin.sha256-cne$TrainingMirrorSHA256){throw 'External completed Training mirror hash before parse'}
$mirror=Read $mirrorPin.path
if($mirror.kind-ne'causal_fresh_v14_container_to_local_evidence_mirror_v1'-or$mirror.status-ne'passed'-or$mirror.scope-ne'Training'-or$mirror.attempt-ne'attempt-001'-or$mirror.card_sha256-cne$MainCardSHA256-or$mirror.source_manifest_sha256-cne$MainManifestSHA256-or$mirror.all_mirrored_pt_ignored-ne$true-or$mirror.source_files_verified-ne$manifest.files.Count){throw 'Completed exact current Training mirror;no quality dependency'}
foreach($entry in @($manifestPin,$zip)){$matches=@($mirror.preserved_source_packages|Where-Object{$_.path-ceq$entry.path});if($matches.Count-ne1){throw 'Manifest/wholeZIP unique exact preserved_source_packages,not containerpayload files'};Exact $entry $matches[0]}
function Mirrored([string]$relative){$pin=Pin $relative;$matches=@($mirror.files|Where-Object{$_.path-ceq$relative});if($matches.Count-ne1){throw 'Exact completed mirrored containerpayload metadata role'};Exact $pin $matches[0];$pin}
$stagePin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/training-stage.json'
$summaryPin=Mirrored 'output/runs/causal-fresh-v14/run-001/training-summary.json'
$stage=Read $stagePin.path
$summary=Read $summaryPin.path
if($stage.status-ne'passed'-or$stage.exit_code-ne0-or$summary.status-ne'passed'-or$summary.model-ne'CPB-v14.market-01'-or$summary.new_causal_updates-ne2048-or$summary.attempted_batches-lt2048-or$summary.attempted_batches-gt4096-or($summary.attempted_batches-isnot[int]-and$summary.attempted_batches-isnot[long])-or$summary.batch_size-ne8){throw 'Exactly2048 successfulB8 within4096 actual attempts'}
if($summary.calibration_TRAIN_loader_calls-ne512-or$summary.raw_load_calls-ne(512+$summary.attempted_batches+36)-or$summary.source_updates-ne0-or$summary.old_model_artifacts_opened-ne0-or$summary.old_normalization_artifacts_opened-ne0){throw 'Fresh training and actual calibration/attempt/query counters'}
$objective='huber1-mean-plus-detached-error-gaussian-variance-v1'
$gradient='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
$routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
$policy='original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1'
$moment='componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1'
$layout='per-trailing-window-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1'
$transform='sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'
$objectiveFields=[ordered]@{objective_policy=$objective;gradient_policy=$gradient;direct_gradient_routes=$routes;variance_error_detached=$true;reporting_nll_backpropagated=$false;loss_coefficients=@(1,1)}
$temporalFields=[ordered]@{temporal_pooling_policy=$policy;temporal_pooling_geometry=@(4,16,32,8);temporal_pooling_trailing_lengths=@(4,8,12,16);temporal_pooling_mix=@(0.5,0.5);empty_local_bin_policy='zero-whole-export-slice-after-bias-v1';temporal_moment_policy=$moment;temporal_export_layout=$layout;temporal_moment_partition=@(4,4);temporal_dispersion_transform_policy=$transform;temporal_dispersion_regularizer=0.0009765625}
foreach($record in @($main.mechanism,$main.training,$summary)){Fields $record $objectiveFields 'Exactunchanged objective';Fields $record $temporalFields 'Exact attention regularized-root contract'}
$transformFields=@('temporal_pool_bin_regularized_root','temporal_pool_bin_dispersion','temporal_contextual_pool_bin_regularized_root','temporal_contextual_pool_bin_dispersion')
if($summary.all_eight_temporal_moment_subgroups_learned-ne$true-or$main.actual_step_temporal_witness_contract.total_fields-ne60-or$main.actual_step_temporal_witness_contract.base_fields-ne23-or$main.actual_step_temporal_witness_contract.parameter_arrays-ne12-or$main.actual_step_temporal_witness_contract.temporal_trace_fields-ne25-or$main.temporal_learning_receipt_contract.moment_subgroup_rows-ne16384-or($main.temporal_learning_receipt_contract.moment_subgroups.moments-join',')-ne'mean,regularized_stddev'){throw 'Actual60=23+12+25 and mean/regularized_stddev evidence'}
if(Compare-Object ($transformFields|Sort-Object) @($main.actual_step_temporal_witness_contract.dispersion_transform_trace_fields|Sort-Object)){throw 'Exact four same-forward rawroot/dispersion arrays'}
if(($main.actual_step_temporal_witness_contract.per_window_local_weight_fields-join',')-cne'temporal_pool_local_weights,temporal_contextual_pool_local_weights'-or($main.actual_step_temporal_witness_contract.per_window_local_weight_shape-join',')-cne'R,4,T'){throw 'Exact two same-forward local fields[R,4,T] before saved audit'}
if($main.actual_step_temporal_witness_contract.exact_allowlist.Count-ne60-or@($main.actual_step_temporal_witness_contract.exact_allowlist|Sort-Object -Unique).Count-ne60){throw 'Exact sixty unique actual writer field names'}
$flags=@('mean4_regularized_stddev4_export_layout','full_attention_dependent_regularized_moment_adjoints','live_dispersion_score_gradient','raw_variance_root_dispersion_trace','zero_tiny_large_analytic_dispersion_adjoints','supported_nonfinite_and_variance_overflow_refused','absent_local_huge_singleton_sanitized','captured_F32_residual_correction','raw_root_absent_offset_exception','nested_trailing_supports_fixed_edge16','per_window_local_weights_captured','longest_local_equals_global_two_sum','packed_padding_membership_invariance')
$temporalProofPin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/temporal-pooling-proof.json'
$temporalStagePin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/temporal-pooling-stage.json'
$temporalLogPin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/temporal-pooling.log'
$temporalProof=Read $temporalProofPin.path
$temporalStage=Read $temporalStagePin.path
$temporalLog=Get-Content -LiteralPath (Rooted $temporalLogPin.path) -Raw
Fields $temporalProof $main.temporal_pooling_proof_contract.required_proof_fields 'Actual temporal proof equals prospectively frozen fields'
foreach($flag in $flags){if($temporalProof[$flag]-isnot[bool]-or-not$temporalProof[$flag]-or-not$temporalLog.Contains($flag+'=true')){throw 'Actual literal13 temporal predicates'}}
if($temporalProof.direct_pool_kernel_calls-ne53-or$temporalProof.F64_finite_difference_coordinates-ne18-or$temporalProof.F32_finite_difference_coordinates-ne18-or$temporalProof.isolated_autograd_grad_calls-ne4-or$temporalProof.transform_derivative_upper_bound-ne16-or$temporalProof.kind-ne'cpb_v14_temporal_pooling_mechanism_proof_v1'-or$temporalProof.actual_CUDA-ne$true){throw 'Actual reviewed53/18/18/4,bound16 CUDAfixture'}
Exact $temporalProof.frozen_card $mainPin
if($temporalProof.frozen_card.path-cne'/opt/cuwacunu/research/causal-fresh-v14/attempt-001/source/doc/CAUSAL_FRESH_V14_CARD.json'){throw 'Exact current frozen-card runtime role'}
Fields $temporalProof.fixture_source $main.temporal_pooling_proof_contract.fixture_source 'Actual compiled fixture source'
Fields $temporalProof.independent_scientific_source_review $reviewPin 'Exact scientific002 proof authority'
Exact $temporalProof.execution $temporalStagePin
Exact $temporalProof.log $temporalLogPin
Exact $temporalStage.log $temporalLogPin
if($temporalStage.status-ne'passed'-or$temporalStage.exit_code-ne0-or-not$temporalLog.Contains($main.temporal_pooling_proof_contract.PASS_prefix)-or-not$temporalLog.Contains('direct_pool_kernel_calls=53')){throw 'Actual temporal successful execution/stdout'}
$buildPin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/build-admission.json'
$build=Read $buildPin.path
if($build.status-ne'passed'-or$build.card_sha256-cne$MainCardSHA256-or$build.source_files-ne$manifest.files.Count-or$build.binaries.Count-ne7-or$build.dependency_closures.Count-ne11){throw 'Actual main7binary/11TU/source closure'}
$binary=@($build.binaries|Where-Object{[IO.Path]::GetFileName($_.path)-ceq'test_cpb_v14_temporal_pooling'})
if($binary.Count-ne1){throw 'One actual admitted temporal binary'}
Fields $temporalProof.binary $binary[0] 'Temporal proof admitted binary'
$engineeringPin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/engineering-proof/proof.json'
$engineering=Read $engineeringPin.path
if($engineering.status-ne'passed'-or$engineering.source_updates-ne0-or$engineering.old_checkpoints_opened-ne0-or$engineering.typed_negative_cases-ne821-or$main.engineering_typed_negative_cases-ne821-or$engineering.typed_resume_next_loss_model_moments_bitwise_exact-ne$true){throw 'Source0/821typed/refusal and actual fixture continuation engineering'}
Fields $engineering ([ordered]@{typed_negative_full_archive_files=817;typed_negative_direct_loader_requests=4;typed_negative_compact_archive_files=0;typed_negative_restored_target_controls=0;typed_negative_legacy_conflict_files=5}) 'Exact full archive/direct-request/legacy counts'
$gradientPin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/variance-gradient-proof.json'
$gradientStagePin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/variance-gradient-stage.json'
$gradientLogPin=Mirrored 'output/engineering/causal-fresh-v14/attempt-001/variance-gradient.log'
$gradientProof=Read $gradientPin.path
foreach($key in $main.mechanism_proof_contract.Keys|Where-Object{$_-notin@('manual_parity_scope','PASS_prefix','log_tokens')}){Fields $gradientProof[$key] $main.mechanism_proof_contract[$key] 'Unchanged actual objective fixture'}
Exact $gradientProof.execution $gradientStagePin
Exact $gradientProof.log $gradientLogPin
if($gradientProof.status-ne'passed'-or$gradientProof.actual_CUDA-ne$true-or$gradientProof.model-ne'CPB-v14'){throw 'Actual passed current CUDA objective fixture'}
$parents=@($workflowProvenance.preserved_parent_auditor_sources)
if($parents.Count-ne3){throw 'Exactly3 direct V13 auditor SOURCE-only guards'}
foreach($name in @('audit_training.py','README.md','provenance.json')){$pin=Pin ('code/research/causal_fresh_v13_training_audit/'+$name);$matches=@($parents|Where-Object{$_.path-ceq$pin.path});if($matches.Count-ne1-or$matches[0].container_path-cne('/opt/cuwacunu/research/causal-fresh-v13-training-audit/attempt-001/source/'+$name)){throw 'Preserved V13 auditor SOURCE guard only'};Exact $pin $matches[0]}
$card=[ordered]@{
 kind='cpb_v14_saved_training_independent_audit_card_v1';copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 stage='declared_before_independent_saved_data_execution';declared_utc=[DateTime]::UtcNow.ToString('o');frozen_utc=[DateTime]::UtcNow.ToString('o');execution_pending=$true
 model='CPB-v14.market-01';baseline='CPB-v3.market-01';outside_frozen_main_source=$true
 objective_policy=$objective;gradient_policy=$gradient;direct_gradient_routes=$routes;mean_huber_delta=1.0;loss_coefficients=@(1.0,1.0);variance_error_detached=$true;reporting_nll_backpropagated=$false
 temporal_pooling_policy=$policy;temporal_pooling_geometry=@(4,16,32,8);temporal_pooling_trailing_lengths=@(4,8,12,16);temporal_pooling_mix=@(0.5,0.5);empty_local_bin_policy='zero-whole-export-slice-after-bias-v1'
 temporal_moment_policy=$moment;temporal_export_layout=$layout;temporal_moment_partition=@(4,4);temporal_dispersion_transform_policy=$transform;temporal_dispersion_regularizer=0.0009765625;raw_root_absent_offset_exception=$true
 frozen_training_card_sha256=$mainPin.sha256;frozen_training_markdown_sha256=$md.sha256;frozen_input_manifest_sha256=$input.sha256
 main_source_manifest_sha256=$manifestPin.sha256;main_source_zip_sha256=$zip.sha256;main_source_files=$manifest.files.Count
 main_protocol_preflight=$preflightPin;main_input_bindings=@($mainPin,$md,$input,$manifestPin,$zip)
 auditor_source=$auditor;auditor_support=@($readme,$provenance);preserved_parent_sources=$parents;independent_source_review=$reviewPin
 independent_workflow_source_review=$workflowReviewPin;reviewed_workflow_sources=$workflowSources;strongest_prospective_protocol=$goalPin
 completed_training_mirror=$mirrorPin;completed_training_stage=$stagePin;completed_training_summary=$summaryPin
 actual_main_proof_bindings=@($buildPin,$engineeringPin,$gradientPin,$gradientStagePin,$gradientLogPin,$temporalProofPin,$temporalStagePin,$temporalLogPin)
 actual_component_witness_allowlist=$main.actual_step_temporal_witness_contract.exact_allowlist
 completed_predecessor=[ordered]@{preservation=$predecessorPin;critical_checkpoint_files=119;source_files=143;scope='Completed V13 ancestry119/143;866 historical V11/109 is separate from new protocol and current V14 capsule'}
 expected_main_paths=[ordered]@{attempt='/opt/cuwacunu/research/causal-fresh-v14/attempt-001';run='/opt/cuwacunu/research/causal-fresh-v14/attempt-001/run-001';source='/opt/cuwacunu/research/causal-fresh-v14/attempt-001/source';build='/opt/cuwacunu/build/causal-fresh-v14-attempt001'}
 expected_audit_root='/opt/cuwacunu/research/causal-fresh-v14-training-audit/attempt-001'
 execution=[ordered]@{container='cuwacunu';immutable_id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';image='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7';pure_stdlib_saved_data_only=$true;host_numeric_execution=$false;new_containers=0;separate_card_review_Stage_AST_Run_required=$true}
 scope=@('Saved TRAIN arithmetic only:2048successfulB8,actual2048..4096attempts,typed512/1024/2048 and no repeated training.','FreshTRAIN stats,76learned294752/3888fixed,75identifiable capacity/gauge;no every-scalar gradient claim.','UnitHuber1+detached-error Gaussian calibration,sharedclip1;112/32 saved scalar query arithmetic is no quality gate.','8192whole-bin/2048gauge/2048Hadamard/16384mean-regularized_stddev rows;actual1/2048 same-forward60fields.','Raw variance/root/dispersion distinct. beta=c/(2actualroot),full F32two-sum residual/attention score and positions live;sum both PRECLIP routes and apply clip factor ONCE.','No full hidden/encoder/globalnorm or2046uncaptured gradients reconstructed;gauge descriptive,decay not data learning.','Actual source-derived821typed/53kernels/18+18FD/4isolated/thirteen predicates,7binary11TU;metadata proofs are not replayed.')
 expected=[ordered]@{updates=2048;attempted_batch_range=@(2048,4096);batch_size=8;native_width=40;parameter_tensors=77;learned_tensors=76;learned_values=294752;mandatory_nonzero_data_learning_tensors=75;mandatory_identifiable_tensor_value_capacity=294751;fixed_values=3888;global_clip_max_norm=1;binary_count=7;translation_units=11;typed_negative_cases=821;typed_negative_full_archive_files=817;typed_negative_direct_loader_requests=4;typed_negative_compact_archive_files=0;typed_negative_restored_target_controls=0;typed_negative_legacy_conflict_files=5;temporal_group_rows=8192;gauge_rows=2048;between_bin_rows=2048;moment_subgroup_rows=16384;moment_subgroups=8;actual_temporal_raw_array_witness_updates=@(1,2048);parameter_array_fields_per_witness=12;temporal_trace_fields_per_witness=25;base_component_fields_per_witness=23;component_archive_fields_per_witness=60;new_regularized_dispersion_trace_fields_per_witness=4;temporal_fixture_direct_kernel_calls=53;temporal_fixture_F64_coordinates=18;temporal_fixture_F32_coordinates=18;temporal_fixture_isolated_grad_calls=4;transform_derivative_upper_bound=16;temporal_fixture_required_flags=$flags}
 temporal_numeric_contract=[ordered]@{movement='Promote operands before subtraction/centering/contraction';between_basis='hadamard-full16-signs-div4-v1';between_signs=@(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1));full_position_divisor=4;norm_absolute_tolerance=2e-12;norm_relative_tolerance=2e-12;Parseval_squared_absolute=1e-12;Parseval_squared_relative=1e-10;final_gate='Per-direction EVENTS and positiveOVERALL final projection,not each-final-nonzero';unsaved2046_raw_gradients_reconstructed=$false}
 fresh_schema_contract='Only fresh constructor JSON lacks gradient_policy;other typed objective/temporal selectors strict.'
 typed_temporal_nodes=@('inference root','inference payload','TRAIN root','TRAIN encoder','TRAIN encoder.payload')
 typed_dispersion_contract='UTF8 transform String and strict Float64 vector[1]tau2^-10 at all5nodes;reject0-D/F32/otherlength/value/missing'
 actual_moment_adjoint_contract=[ordered]@{same_forward_only=$true;local_weight_fields=@('temporal_pool_local_weights','temporal_contextual_pool_local_weights');local_weight_shape=@('R',4,'T');local_weights_are_exact_per_window=$true;dispersion_transform_trace_fields=$transformFields;root_dispersion_dtype_shape='Float32[R,4,64]';raw_variance_trace_unchanged=$true;raw_root_unmasked_at_zero=0.03125;dispersion_at_zero=0;actual2048_coefficients='captured BEFORE update2048,pre-step2047';regularized_sqrt_uses_actual_saved_root=$true;attention_dispersion_score_and_position_routes_live=$true;derivative='beta=c/(2saved_actual_F32root),center a-2beta*sum(p*(h-mu));full learned-attention score/position paths';projection_upstreams='ordinary/contextual retained PRECLIP outputs;explicit Longdefined flag/zero fallback';clip_order='sum both routes then recordedfactor ONCE';parity_parameters=@('export weight','export bias','pool_positions');absolute_tolerance=1e-9;relative_tolerance=3e-4;nonvacuity='At both actual witnesses expectedPOSTCLIP exportW OR bias max>1e-6';common_pool_score_bias='Finite descriptive actual/reference/error;no tight parity/nonzero gate';whole8_vs_two4_identity_updates=2048;whole8_vs_two4_identity_rows=8192;full_hidden_encoder_or_global_norm_reconstruction=$false;matched_pre_step_decoder_adjoint_replay=$false}
 module_counting='CORE plus five numeric_statistics,saved_tensor_state,manual_saved_Gaussian,actual_pre_step_composite_objective,temporal_pooling_saved_learning. Nested moment included in last;never double-count. Totals deriveactualresult.'
 stage_helper=(Pin ($prefix+'stage.ps1'));run_helper=(Pin ($prefix+'run.ps1'));execution_wrapper=(Pin ($prefix+'execute.py'));mirror_helper=(Pin ($prefix+'mirror.ps1'));card_generator=(Pin ($prefix+'prepare-card.ps1'));workflow_provenance=(Pin ($prefix+'provenance.json'))
 model_calls=0;head_fits=0;training_updates=0;AST_calls=0
 forbidden=@('Torch/NumPy/producer/model imports or executable module deserialization','Encoder/decoder/head/optimizer replay/refit','Old trained model/statistic numeric reads','Fullhidden/globalnorm/2046missing-gradient claims','Quality scores/heads/selection or frozen main/DEV mutation')
}
[IO.File]::WriteAllText($output,($card|ConvertTo-Json -Depth 40)+[char]10,[Text.UTF8Encoding]::new($false))
Pin ($prefix+'audit-card.json')|ConvertTo-Json


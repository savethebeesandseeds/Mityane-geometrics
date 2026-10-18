# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Owned full physical final V11 training-workflow copy; metadata preparation only.
# No Docker, project imports, AST, auditor body, model or numerical data algorithm.
param([Parameter(Mandatory=$true)][switch]$SourcesReviewed,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$TrainingMirrorSHA256,
 [Parameter(Mandatory=$true)][string]$SourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SourceReviewSHA256,
 [Parameter(Mandatory=$true)][string]$WorkflowSourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WorkflowSourceReviewSHA256)
$ErrorActionPreference='Stop'
if(-not $SourcesReviewed){throw 'Independent static review must precede audit-card generation'}
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$output=Join-Path $PSScriptRoot 'audit-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve the existing audit declaration; use an admitted additive attempt'}
function Rooted([string]$relative){
 if([IO.Path]::IsPathRooted($relative)){throw 'Repository-relative metadata paths only'}
 $path=[IO.Path]::GetFullPath((Join-Path $repo $relative));$prefix=$repo.TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
 if(-not $path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Metadata path outside repository'}
 for($walk=$path;$walk-ne$repo;$walk=[IO.Path]::GetDirectoryName($walk)){$item=Get-Item -LiteralPath $walk;if(($item.Attributes-band[IO.FileAttributes]::ReparsePoint)-or-not$walk){throw 'Physical metadata ancestors required'}}
 $path
}
function Pin([string]$relative){$f=Get-Item -LiteralPath (Rooted $relative);if($f.PSIsContainer){throw 'Expected file'};[ordered]@{path=$relative.Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Exact($a,$b){if($a.bytes-ne$b.bytes-or$a.sha256-ne$b.sha256){throw 'Reviewed metadata/source hash mismatch'}}
$reviewPin=Pin $SourceReviewPath
if($reviewPin.sha256-ne$SourceReviewSHA256){throw 'External source-review hash required before parse'}
$review=Get-Content -Raw -LiteralPath (Rooted $SourceReviewPath)|ConvertFrom-Json
if($review.kind-ne'cpb_v12_family_and_training_auditor_independent_static_review_v1'-or$review.status-ne'passed'-or$review.remaining_findings.Count-ne0){throw 'Externally bound independently passed V12 training-auditor source review required'}
$auditor=Pin 'code/research/causal_fresh_v12_training_audit/audit_training.py'
$readme=Pin 'code/research/causal_fresh_v12_training_audit/README.md'
$provenance=Pin 'code/research/causal_fresh_v12_training_audit/provenance.json'
$reviewedSources=@($review.exact_independently_reviewed_training_auditor_files)
if($reviewedSources.Count-ne3-or@($reviewedSources.path|Sort-Object -Unique).Count-ne3){throw 'Exactly three independent auditor source bindings required'}
foreach($pin in @($auditor,$readme,$provenance)){$match=@($reviewedSources|Where-Object{$_.path-eq$pin.path});if($match.Count-ne1){throw 'Exact auditor source path missing from independent review'};Exact $pin $match[0]}
$workflowReviewPin=Pin $WorkflowSourceReviewPath
if($workflowReviewPin.sha256-ne$WorkflowSourceReviewSHA256){throw 'External workflow-review hash required before parse'}
$workflowReview=Get-Content -Raw -LiteralPath (Rooted $WorkflowSourceReviewPath)|ConvertFrom-Json
if($workflowReview.kind-ne'cpb_v12_training_audit_workflow_independent_static_review_v1'-or$workflowReview.status-ne'passed'){throw 'Separately passed current V12 audit workflow review required'}
$workflowNames=@('prepare-card.ps1','stage.ps1','run.ps1','mirror.ps1','execute.py','provenance.json','README.md')
$workflowSources=@($workflowReview.reviewed_owned_files)
if($workflowSources.Count-ne7-or@($workflowSources.path|Sort-Object -Unique).Count-ne7){throw 'Exactly seven externally reviewed complete workflow files required'}
foreach($name in $workflowNames){$current=Pin ('doc/causal_fresh_v12_training_audit/'+$name);$match=@($workflowSources|Where-Object{$_.path-ceq$current.path});if($match.Count-ne1){throw 'Reviewed workflow path set must equal the seven owned files'};Exact $current $match[0]}
$workflowProvenance=Get-Content -Raw -LiteralPath (Rooted 'doc/causal_fresh_v12_training_audit/provenance.json')|ConvertFrom-Json
$mainPin=Pin 'doc/CAUSAL_FRESH_V12_CARD.json'
if($mainPin.sha256-ne$MainCardSHA256){throw 'External main-card hash required before parse'}
$main=Get-Content -Raw -LiteralPath (Rooted $mainPin.path)|ConvertFrom-Json
$md=Pin 'doc/CAUSAL_FRESH_V12_CARD.md';$input=Pin 'doc/causal_fresh_v12/input-manifest.json'
if($main.model-ne'CPB-v12'-or$main.instance-ne'CPB-v12.market-01'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Actually frozen and admitted main card required'}
Exact $input $main.input_manifest
$manifestPin=Pin 'output/engineering/causal-fresh-v12/attempt-001/source-manifest.json'
if($manifestPin.sha256-ne$MainManifestSHA256){throw 'External main-manifest hash required before parse'}
$manifest=Get-Content -Raw -LiteralPath (Rooted $manifestPin.path)|ConvertFrom-Json
$zip=Pin 'output/engineering/causal-fresh-v12/attempt-001/source.zip'
$package=Get-Content -Raw -LiteralPath (Rooted 'output/engineering/causal-fresh-v12/attempt-001/package.json')|ConvertFrom-Json
if($manifest.card_sha256-ne$MainCardSHA256-or$manifest.files.Count-le0-or$package.source_files-ne$manifest.files.Count-or$package.manifest_sha256-ne$MainManifestSHA256-or$package.zip_sha256-ne$zip.sha256){throw 'Actual immutable main source package required'}
$preflightPin=Pin 'doc/causal_fresh_v12/protocol-preflight.json'
$preflightBinding=@($manifest.files|Where-Object{$_.path-eq$preflightPin.path})
if($preflightBinding.Count-ne1){throw 'Main preflight must be externally manifest-bound before parse'}
Exact $preflightPin $preflightBinding[0]
$preflight=Get-Content -Raw -LiteralPath (Rooted $preflightPin.path)|ConvertFrom-Json
if($preflight.status-ne'passed'-or$preflight.card_sha256-ne$MainCardSHA256-or($preflight.sources.Count+1)-ne$manifest.files.Count){throw 'Actual normal main preflight/closure required'}
foreach($entry in @($mainPin,$md,$input,$preflightPin)){$matching=@($manifest.files|Where-Object{$_.path-eq$entry.path});if($matching.Count-ne1){throw 'Main declaration/input/preflight role missing from immutable capsule'};Exact $entry $matching[0]}
$goalPin=Pin $main.advancement.goal_protocol.path
Exact $goalPin $main.advancement.goal_protocol
$goalBinding=@($manifest.files|Where-Object{$_.path-eq$goalPin.path})
if($goalBinding.Count-ne1){throw 'Current strongest prospective protocol must be in the frozen main capsule'}
Exact $goalPin $goalBinding[0]
if($goalPin.sha256-ne'866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'-or$main.advancement.criterion_count-ne22){throw 'Exact strongest-V3-V11 prospective22 protocol required'}
$goal=Get-Content -Raw -LiteralPath (Rooted $goalPin.path)|ConvertFrom-Json
if($goal.criteria.Count-ne22-or$goal.additional_train_NLL_gate-ne$false){throw 'No TRAIN-NLL advancement gate may be introduced'}
$mirrorPin=Pin 'doc/causal_fresh_v12/training-mirror-attempt-001.json'
if($mirrorPin.sha256-ne$TrainingMirrorSHA256){throw 'External completed Training mirror hash required before parse'}
$mirror=Get-Content -Raw -LiteralPath (Rooted $mirrorPin.path)|ConvertFrom-Json
if($mirror.status-ne'passed'-or$mirror.scope-ne'Training'-or$mirror.attempt-ne'attempt-001'-or$mirror.card_sha256-ne$MainCardSHA256-or$mirror.source_manifest_sha256-ne$MainManifestSHA256-or-not$mirror.all_mirrored_pt_ignored){throw 'Completed separately mirrored main training evidence required; no quality dependency'}
if($mirror.kind-ne'causal_fresh_v12_container_to_local_evidence_mirror_v1'-or$mirror.source_files_verified-ne$manifest.files.Count){throw 'Normal V12 Training receipt and actual source count required'}
foreach($entry in @($manifestPin,$zip)){$matches=@($mirror.preserved_source_packages|Where-Object{$_.path-ceq$entry.path});if($matches.Count-ne1){throw 'External completed Training mirror preserved_source_packages must uniquely bind exact source manifest and whole ZIP'};Exact $entry $matches[0]}
foreach($relative in @('output/engineering/causal-fresh-v12/attempt-001/training-stage.json','output/runs/causal-fresh-v12/run-001/training-summary.json')){
 $p=Pin $relative;$matches=@($mirror.files|Where-Object {$_.path-eq$relative});if($matches.Count-ne1){throw 'Exact completed training metadata role missing'};Exact $p $matches[0]
}
$stagePin=Pin 'output/engineering/causal-fresh-v12/attempt-001/training-stage.json'
$stage=Get-Content -Raw -LiteralPath (Rooted $stagePin.path)|ConvertFrom-Json
$summaryPin=Pin 'output/runs/causal-fresh-v12/run-001/training-summary.json'
$summary=Get-Content -Raw -LiteralPath (Rooted $summaryPin.path)|ConvertFrom-Json
if($stage.status-ne'passed'-or$stage.exit_code-ne0-or$summary.status-ne'passed'-or$summary.model-ne'CPB-v12.market-01'-or$summary.new_causal_updates-ne2048-or$summary.attempted_batches-lt2048-or$summary.attempted_batches-gt4096){throw 'Exactly2048 successful main updates within4096 attempts required'}
if(($summary.attempted_batches-isnot[int]-and$summary.attempted_batches-isnot[long])-or$summary.calibration_TRAIN_loader_calls-ne512-or$summary.raw_load_calls-ne(512+$summary.attempted_batches+36)){throw 'Actual calibration+attempt+fixed-query load counters required;no zero-skip assumption'}
$objective='huber1-mean-plus-detached-error-gaussian-variance-v1'
$gradient='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
$routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
foreach($record in @($main.mechanism,$main.training,$summary)){if($record.objective_policy-ne$objective-or$record.gradient_policy-ne$gradient-or$record.direct_gradient_routes-ne$routes-or$record.variance_error_detached-ne$true-or$record.reporting_nll_backpropagated-ne$false-or($record.loss_coefficients-join',')-ne'1,1'){throw 'Unchanged V11 detached-error variance objective contract required'}}
$policy='original-position-four-bin-attention-mean4-uniform-centered-variance4-v1'
foreach($record in @($main.mechanism,$main.training,$summary)){
 if($record.temporal_pooling_policy-ne$policy-or($record.temporal_pooling_partition -join ',')-ne'4,16,32,4,8'-or($record.temporal_pooling_mix -join ',')-ne'0.5,0.5'-or$record.empty_local_bin_policy-ne'zero-whole-export-slice-after-bias-v1'-or$record.temporal_moment_policy-ne'componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1'-or$record.temporal_export_layout-ne'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'-or($record.temporal_moment_partition-join',')-ne'4,4'){throw 'Exact attention-mean/uniform-own-center moment/layout contract required'}
}
if($summary.all_eight_temporal_moment_subgroups_learned-ne$true-or$main.actual_step_temporal_witness_contract.total_fields-ne60-or$main.actual_step_temporal_witness_contract.base_fields-ne23-or$main.actual_step_temporal_witness_contract.parameter_arrays-ne12-or$main.actual_step_temporal_witness_contract.temporal_trace_fields-ne25-or$main.temporal_learning_receipt_contract.moment_subgroup_rows-ne16384){throw 'Actual eight moment groups and exact60=23base+12parameter+25trace same-forward fields required'}
$uniformFields=@('temporal_pool_dispersion_weights','temporal_pool_dispersion_mean','temporal_contextual_pool_dispersion_weights','temporal_contextual_pool_dispersion_mean')
if(Compare-Object ($uniformFields|Sort-Object) @($main.actual_step_temporal_witness_contract.uniform_trace_fields|Sort-Object)){throw 'Both existing routes require explicit uniform weights and OWN centers'}
$temporalProof=$main.temporal_pooling_proof_contract.required_proof_fields
foreach($field in @('attention_mean_and_uniform_dispersion_full_expression_adjoints','uniform_dispersion_own_center','uniform_dispersion_score_independent','uniform_dispersion_hidden_gradient_live','attention_mean_score_gradient_live','dual_support_actual_trace')){if($temporalProof.$field-isnot[bool]-or$temporalProof.$field-ne$true){throw ('Prospective uniform-dispersion proof field required: '+$field)}}
if($temporalProof.direct_pool_kernel_calls-ne47-or$temporalProof.F64_finite_difference_coordinates-ne18-or$temporalProof.F32_finite_difference_coordinates-ne18-or$temporalProof.PSObject.Properties.Name-contains'full_attention_dependent_moment_adjoints'){throw 'Exact reviewed47/18/18 fixture contract without old attention-dependent dispersion claim required'}
$parents=@($workflowProvenance.preserved_parent_auditor_sources)
if($parents.Count-ne3){throw 'Exactly three provenance-bound V11 parent SOURCE guards required'}
foreach($name in @('audit_training.py','README.md','provenance.json')){$p=Pin ('code/research/causal_fresh_v11_training_audit/'+$name);$match=@($parents|Where-Object{$_.path-ceq$p.path});if($match.Count-ne1-or$match[0].container_path-cne('/opt/cuwacunu/research/causal-fresh-v11-training-audit/attempt-001/source/'+$name)){throw 'Only exact preserved V11 auditor SOURCE paths allowed'};Exact $p $match[0]}
$prefix='doc/causal_fresh_v12_training_audit/'
$card=[ordered]@{
 kind='cpb_v12_saved_training_independent_audit_card_v1';copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 stage='declared_before_independent_saved_data_execution';declared_utc=[DateTime]::UtcNow.ToString('o');frozen_utc=[DateTime]::UtcNow.ToString('o');execution_pending=$true
 model='CPB-v12.market-01';baseline='CPB-v3.market-01';outside_frozen_main_source=$true
 objective_policy=$objective;gradient_policy=$gradient;direct_gradient_routes=$routes;mean_huber_delta=1.0;loss_coefficients=@(1.0,1.0);variance_error_detached=$true;reporting_nll_backpropagated=$false
 temporal_pooling_policy=$policy;temporal_pooling_partition=@(4,16,32,4,8);temporal_pooling_mix=@(0.5,0.5);empty_local_bin_policy='zero-whole-export-slice-after-bias-v1'
 temporal_moment_policy='componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1';temporal_export_layout='per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1';temporal_moment_partition=@(4,4)
 frozen_training_card_sha256=$mainPin.sha256;frozen_training_markdown_sha256=$md.sha256;frozen_input_manifest_sha256=$input.sha256
 main_source_manifest_sha256=$manifestPin.sha256;main_source_zip_sha256=$zip.sha256;main_source_files=$manifest.files.Count
 main_protocol_preflight=$preflightPin;main_input_bindings=@($mainPin,$md,$input,$manifestPin,$zip)
 auditor_source=$auditor;auditor_support=@($readme,$provenance);preserved_parent_sources=$parents;independent_source_review=$reviewPin
 independent_workflow_source_review=$workflowReviewPin;reviewed_workflow_sources=$workflowSources;strongest_prospective_protocol=$goalPin
 completed_training_mirror=$mirrorPin;completed_training_stage=$stagePin;completed_training_summary=$summaryPin
 expected_main_paths=[ordered]@{attempt='/opt/cuwacunu/research/causal-fresh-v12/attempt-001';run='/opt/cuwacunu/research/causal-fresh-v12/attempt-001/run-001';source='/opt/cuwacunu/research/causal-fresh-v12/attempt-001/source';build='/opt/cuwacunu/build/causal-fresh-v12-attempt001'}
 expected_audit_root='/opt/cuwacunu/research/causal-fresh-v12-training-audit/attempt-001'
 execution=[ordered]@{container='cuwacunu';immutable_id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';image='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7';pure_stdlib_saved_data_only=$true;host_numeric_execution=$false;new_containers=0;stage_and_run_separately_admitted=$true}
 scope=@('Exactly2048 successful B8 updates with2048..4096 attempted batches; every CSV attempt/skip row,actual seed streams,typed512/1024/2048 progress and actual1/2048 associations;raw loads512+actual attempts+36','FreshTRAIN4096 scaler/bank arithmetic;77 parameter tensors,76 learned294752 capacity,3888 fixed prior;actual512 exact continuation without replay','Unchanged V11 unitF64Huber1 plus unit detached-error variance NLL and detached reporting copy;partial stop-gradient objective proof,shared clip1;known112/32 scalar query arithmetic adds no quality gate','All8192whole-group/2048gauge/2048Hadamard/16384moment subgroup rows;12parameter arrays plus25same-forward trace fields within60component fields at actual1/2048;F64 operands before subtraction','Actual attention two-sum mean and uniform OWN-center dispersion;export-weight/bias and attention-mean-only pool-position adjoints;common-bias finite descriptive values only from retained PRECLIP route upstreams;summed over routes then recorded clipfactor ONCE;no fullencoder/wholehidden/norm reconstruction','UnitR16 Hadamard Parseval/events and positive OVERALL final projection;75 mandatory tensorcapacity/gauge;no per-scalar/every-update claim or decay-as-data-learning','Actual manifest count/admitted7binaries11TUs,new47-call uniform-dispersion fixture with18F64/18F32 coordinates and unchanged V11 objective proof;strict seven typed temporal fields at all5nodes,no model/head/optimizer replay')
 expected=[ordered]@{updates=2048;attempted_batch_range=@(2048,4096);batch_size=8;native_width=40;parameter_tensors=77;learned_tensors=76;learned_values=294752;mandatory_nonzero_data_learning_tensors=75;mandatory_identifiable_tensor_value_capacity=294751;fixed_values=3888;global_clip_max_norm=1;binary_count=7;translation_units=11;temporal_group_rows=8192;gauge_rows=2048;between_bin_rows=2048;moment_subgroup_rows=16384;moment_subgroups=8;actual_temporal_raw_array_witness_updates=@(1,2048);parameter_array_fields_per_witness=12;temporal_trace_fields_per_witness=25;base_component_fields_per_witness=23;component_archive_fields_per_witness=60;new_uniform_trace_fields_per_witness=4;temporal_fixture_direct_kernel_calls=47;temporal_fixture_F64_coordinates=18;temporal_fixture_F32_coordinates=18}
 temporal_numeric_contract=[ordered]@{movement='Promote EACH F32 operand to F64 BEFORE subtraction/centering/contraction';between_basis='hadamard-full16-signs-div4-v1';between_signs=@(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1));full_position_divisor=4;norm_absolute_tolerance=2e-12;norm_relative_tolerance=2e-12;Parseval_squared_absolute=1e-12;Parseval_squared_relative=1e-10;final_gate='Per-direction data-gradient/movement EVENTS and positive OVERALL final projection; no individual final-nonzero gate';unsaved2046_raw_gradients_reconstructed=$false}
 fresh_schema_contract='Only fresh constructor JSON lacks gradient_policy; require absence there and strict typed/other selectors.'
 typed_temporal_nodes=@('inference root','inference payload','TRAIN root','TRAIN encoder','TRAIN encoder.payload')
 actual_moment_adjoint_contract=[ordered]@{same_forward_only=$true;uniform_trace_fields=$uniformFields;uniform_weights_shape='[R,4,T]';uniform_center_shape='[R,4,64]';actual2048_coefficients='captured BEFORE update2048,pre-step2047';dispersion_direct_score_derivative=0;fixed_score_hidden_reference='Descriptive only;separate captured rounded u_actual and F32 reciprocal two-sum center Jacobian;not observed full-hidden parity';local_hidden_adjoint_array_available=$false;score_to_hidden_route_reconstructed=$false;hidden_reference_is_observed_gradient_parity=$false;projection_upstreams='temporal_pool_projected_upstream_pre_clip and contextual counterpart,with explicit Long defined flags';unused_route='flag0 plus zeroarray;no route-activity gate';clip_order='sum both PRECLIP route adjoints,then apply recorded globalclipfactor once';parity_parameters=@('export weight','export bias','pool_positions');absolute_tolerance=1e-9;relative_tolerance=3e-4;nonvacuity='At BOTH actual updates1/2048,max reconstructed POSTCLIP export-weight OR export-bias magnitude >1e-6';common_pool_score_bias='Actual/reference/error finite descriptive values only;no nearzero parity gate;raw scalar gauge CSV checks remain exact';whole8_vs_two4_identity_updates=2048;whole8_vs_two4_identity_rows=8192;whole8_vs_two4_identities='Squared gradient/movement norms and exact nonzero/moved counts;F64 operands before operations';full_hidden_encoder_or_global_norm_reconstruction=$false;matched_pre_step_decoder_adjoint_replay=$false}
 module_counting='CORE plus five distinct modules: numeric_statistics,saved_tensor_state,manual_saved_Gaussian,actual_pre_step_composite_objective,temporal_pooling_saved_learning. The last INCLUDES temporal_moment_saved_learning;do not count it twice. Any total is their actual sum.'
 stage_helper=(Pin ($prefix+'stage.ps1'));run_helper=(Pin ($prefix+'run.ps1'));execution_wrapper=(Pin ($prefix+'execute.py'));mirror_helper=(Pin ($prefix+'mirror.ps1'));card_generator=(Pin ($prefix+'prepare-card.ps1'));workflow_provenance=(Pin ($prefix+'provenance.json'))
 model_calls=0;head_fits=0;training_updates=0;AST_calls=0
 forbidden=@('Torch/NumPy/producer imports or executable tensor module deserialization','Encoder/decoder/head/optimizer replay or refits','Old model/statistic numeric reads','Claiming every scalar data-learns or reconstructing unsaved2046 gradients','Quality scores/heads/selection or mutation of frozen main/old evidence/dev')
}
[IO.File]::WriteAllText($output,($card|ConvertTo-Json -Depth 35)+[char]10,[Text.UTF8Encoding]::new($false))
Pin ($prefix+'audit-card.json')|ConvertTo-Json

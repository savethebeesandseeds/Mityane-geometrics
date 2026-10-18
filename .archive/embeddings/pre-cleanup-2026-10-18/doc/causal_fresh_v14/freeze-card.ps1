# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Prospective host byte/JSON declaration only. No AST/build/Docker/project body.
param(
 [Parameter(Mandatory=$true)][string]$DraftCardSHA256,
 [Parameter(Mandatory=$true)][string]$DraftMarkdownSHA256,
 [Parameter(Mandatory=$true)][string]$OrchestrationProvenanceSHA256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedProspectiveFreeze
)
$ErrorActionPreference='Stop'
if(-not$RootReviewedProspectiveFreeze){throw 'Root independent source/card/auditor/helper review is required'}
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspace=[IO.Path]::GetFullPath((Join-Path $root '..'))
$snapshotRelative='output/engineering/causal-fresh-v14-protocol-preparation/attempt-001/before-prospective-freeze'
$snapshot=Join-Path $root $snapshotRelative
$canonicalRelative='code/encoders/causal_patch_bottleneck_mae/versions/CPB-v14.json'
$canonical=Join-Path $root $canonicalRelative
if((Test-Path -LiteralPath $snapshot)-or(Test-Path -LiteralPath $canonical)){throw 'Preserve any existing freeze or canonical descriptor'}
$script:checks=0
function Check([bool]$value,[string]$why){$script:checks++;if(-not$value){throw $why}}
function Get-Pin([string]$relative){
 $file=Get-Item -LiteralPath (Join-Path $root $relative)
 Check (-not($file.Attributes-band[IO.FileAttributes]::ReparsePoint)) ('Physical declaration input: '+$relative)
 [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Verify($pin,[string]$repository='cuwacunu_embedding'){
 Check ($repository-in@('cuwacunu_embedding','cuwacunu_dev')) 'Known repository required'
 $base=[IO.Path]::GetFullPath((Join-Path $workspace $repository))
 $path=if([IO.Path]::IsPathRooted($pin.path)){[IO.Path]::GetFullPath($pin.path)}else{[IO.Path]::GetFullPath((Join-Path $base $pin.path))}
 Check ($path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin stays in its explicit repository'
 $file=Get-Item -LiteralPath $path
 Check (-not($file.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'Physical protected file'
 Check ($file.Length-eq$pin.bytes-and(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()-eq$pin.sha256) ('Exact protected bytes: '+$pin.path)
}
function Read([string]$relative){Get-Content -LiteralPath (Join-Path $root $relative) -Raw|ConvertFrom-Json -AsHashtable}
function Write-NewJson([string]$relative,$value){
 $path=Join-Path $root $relative
 if(Test-Path -LiteralPath $path){throw ('Preserve existing declaration output: '+$relative)}
 [IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 60)+"`n",[Text.UTF8Encoding]::new($false))
}
$cardRelative='doc/CAUSAL_FRESH_V14_CARD.json';$markdownRelative='doc/CAUSAL_FRESH_V14_CARD.md'
$cardPin=Get-Pin $cardRelative;$markdownPin=Get-Pin $markdownRelative
Check ($cardPin.sha256-eq$DraftCardSHA256-and$markdownPin.sha256-eq$DraftMarkdownSHA256) 'Exact root-reviewed draft bytes'
$orchestrationPin=Get-Pin 'doc/causal_fresh_v14/orchestration-provenance.json'
Check ($orchestrationPin.sha256-eq$OrchestrationProvenanceSHA256) 'Exact reviewed ten-helper derivation'
$card=Read $cardRelative;$orchestration=Read $orchestrationPin.path
Check ($card.stage-eq'DRAFT_NOT_FROZEN_NOT_EXECUTION_ADMITTED'-and-not$card.execution_admitted-and$null-eq$card.frozen_utc) 'Only the unmeasured prospective draft may freeze'
Check ($card.instance-eq'CPB-v14.market-01'-and$card.model-eq'CPB-v14'-and$card.geometry.native_width-eq40-and$card.baseline-eq'CPB-v3.market-01'-and$card.baseline_remains_selected) 'Separately named native40, selected baseline unchanged'
$policy='original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1';$empty='zero-whole-export-slice-after-bias-v1'
foreach($record in @($card.geometry,$card.mechanism,$card.training,$card.trust_receipt_objective_fields,$card.temporal_pooling_proof_contract,$card.temporal_learning_receipt_contract)){
 Check ($record.temporal_pooling_policy-eq$policy-and($record.temporal_pooling_geometry-join',')-eq'4,16,32,8'-and($record.temporal_pooling_trailing_lengths-join',')-eq'4,8,12,16'-and($record.temporal_pooling_mix-join',')-eq'0.5,0.5'-and$record.empty_local_bin_policy-eq$empty) 'Every current temporal declaration agrees'
 Check (-not$record.Contains('temporal_pooling_partition')-and($record.temporal_pooling_geometry-join',')-eq'4,16,32,8'-and($record.temporal_pooling_trailing_lengths-join',')-eq'4,8,12,16') 'New Long4 geometry and trailing lengths at every current record;legacy partition absent'
 Check ($record.temporal_moment_policy-eq'componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1'-and$record.temporal_export_layout-eq'per-trailing-window-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1'-and($record.temporal_moment_partition-join',')-eq'4,4') 'Every current moment/layout declaration agrees'
 Check ($record.temporal_dispersion_transform_policy-eq'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'-and$record.temporal_dispersion_regularizer-eq0.0009765625) 'All current transform/tau declarations agree'
}
Check ($card.mechanism.source_updates-eq0-and$card.mechanism.all_learned_values-eq294752-and$card.mechanism.all_learned_tensors-eq76-and$card.mechanism.mandatory_identifiable_learned_tensors-eq75-and-not$card.mechanism.checkpoint_import_for_initialization-and$card.mechanism.fresh_optimizer) 'Fresh learned/optimizer ancestry and sole scalar gauge exemption'
Check (-not$card.mechanism.Contains('mandatory_identifiable_learned_values')-and$card.mechanism.mandatory_identifiable_tensor_value_capacity-eq294751-and$card.mechanism.mandatory_identifiable_count_semantics-eq'tensor numel capacity, not every-scalar gradient/movement proof'-and$card.mechanism.per_scalar_nonzero_gradient_proof_claimed-eq$false) 'Tensor numel capacity is not every-scalar learning proof'
Check ($card.training.updates-eq2048-and$card.training.batch_size-eq8-and$card.training.global_gradient_clip-eq1-and$card.training.learning_rate-eq0.001-and$card.training.weight_decay-eq0.0001-and$card.training.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and$card.training.gradient_policy-eq'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'-and($card.training.loss_coefficients-join',')-eq'1,1') 'Unchanged objective/shared globalclip/single trajectory budget'
Check ($card.training.initialization_seed-eq943001-and$card.training.row_seed-eq943002-and$card.training.mask_seed-eq943003-and$card.training.context_seed-eq943004) 'Unchanged four seeds'
$routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
Check ($card.mechanism.kind-eq'original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1'-and$card.training.encoder_gradient_kind-eq'Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1') 'Distinct pooling identity and unchanged V11 gradient-kind label'
foreach($record in @($card.mechanism,$card.training,$card.trust_receipt_objective_fields)){
 Check ($record.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and$record.gradient_policy-eq'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'-and$record.direct_gradient_routes-eq$routes-and($record.loss_coefficients-join',')-eq'1,1'-and$record.variance_error_detached-eq$true-and$record.reporting_nll_backpropagated-eq$false) 'All objective/direct-route/detach/reporting declarations agree'
}
$establishment='Full trust including exact initial native20:40 DATA and whole-statistic pairings against saved V13, and independent training/quality audits, establishes a measured candidate. Advancement additionally requires full coverage and all22 declared numeric conditions. Winner and application selection remain separate and false.'
$report=Get-Content -LiteralPath (Join-Path $root 'code/research/causal_fresh_v14/evaluation/report.py') -Raw
$literal=[regex]::Match($report,'(?s)ESTABLISHMENT_CONTRACT = \((?<body>.*?)\)\r?\n')
$reportLiteral=(@([regex]::Matches($literal.Groups['body'].Value,"'([^']*)'")|ForEach-Object{$_.Groups[1].Value})-join'')
Check ($literal.Success-and$reportLiteral-ceq$establishment-and$card.criteria.establishment-ceq$establishment-and$report.Contains('validate_card_contract(card)')) 'Literal reporter/card contract checked before any expensive phase'
Check ($card.mechanism_proof_contract.direct_NLL_mean_gradient_zero-eq$true) 'Unchanged V11 detached-error variance direct mean-gradient zero'
foreach($field in @('direct_joint_NLL_mean_gradient_live','direct_joint_NLL_uncertainty_gradient_live','joint_Gaussian_variance_counted_once','total_equals_Huber_plus_one_joint_Gaussian','independent_composite_F64_finite_differences')){Check (-not$card.mechanism_proof_contract.Contains($field)) ('No V10 joint objective proof inherited: '+$field)}
Check ($card.mechanism_proof_contract.log_tokens-contains'detached-error NLL mean gradient zero') 'Unchanged objective partial-gradient fixture token'
Check ($card.temporal_learning_receipt_contract.moment_subgroup_rows-eq16384-and$card.temporal_learning_receipt_contract.moment_subgroups['count']-eq8-and$card.actual_step_temporal_witness_contract.total_fields-eq60) 'Declared eight moment subgroups and expanded same-forward actual witness'
$engineeringArchives=$card.engineering_archive_contract
Check ($engineeringArchives.format-eq'full_refusal_archives'-and$engineeringArchives.typed_negative_full_archive_files-eq817-and$engineeringArchives.typed_negative_direct_loader_requests-eq4-and$engineeringArchives.typed_negative_compact_archive_files-eq0-and$engineeringArchives.typed_negative_restored_target_controls-eq0-and$engineeringArchives.typed_negative_legacy_conflict_files-eq5-and$engineeringArchives.real_main_checkpoints-eq5-and$engineeringArchives.all_full_negative_files_preserved-is[bool]-and$engineeringArchives.all_full_negative_files_preserved) 'Compiled full-refusal fallback817files+4requests;zero compact/control;all real5 checkpoints retained'
$actualWitness=$card.actual_step_temporal_witness_contract
Check ($actualWitness.total_fields-eq60-and$actualWitness.base_fields-eq23-and$actualWitness.parameter_arrays-eq12-and$actualWitness.temporal_trace_fields-eq25-and$actualWitness.extra_encoder_calls-eq0-and($actualWitness.per_window_local_weight_fields-join',')-ceq'temporal_pool_local_weights,temporal_contextual_pool_local_weights'-and($actualWitness.per_window_local_weight_shape-join',')-ceq'R,4,T') 'Two exact same-forward local softmax captures change toR4T within unchanged60fields;no inverse mixture or extra forward'
Check (($card.engineering_typed_negative_cases-is[int]-or$card.engineering_typed_negative_cases-is[long])-and$card.engineering_typed_negative_cases-eq821) 'NEW compiled research primary821 refusal cases;family676 not executed here'
Verify $card.declaration_input_plan
$fixtureInput=Read $card.declaration_input_plan.path
$fixtureContract=$card.temporal_pooling_proof_contract
$fixturePlan=$fixtureInput.temporal_pooling_proof_contract
foreach($field in @('required_proof_fields','log_tokens','fixture_source','source_review')){Check (($fixtureContract[$field]|ConvertTo-Json -Depth 30 -Compress)-ceq($fixturePlan[$field]|ConvertTo-Json -Depth 30 -Compress)) ('Source-reviewed fixture binding: '+$field)}
Verify $fixtureContract.fixture_source;Verify $fixtureContract.source_review
Check ($fixtureContract.fixture_source.path-eq'code/research/causal_fresh_v14/tests/v14_temporal_pooling_test.cpp'-and$fixtureContract.source_review.sha256-eq$card.independent_training_auditor_source_review.sha256) 'Actual complete fixture plus independent scientific review'
foreach($key in $fixtureContract.required_proof_fields.Keys){Check ($key-notmatch'uniform|dual_support') 'No V12-specific fixture claim inherited'}
Check ($fixtureContract.required_proof_fields.temporal_dispersion_transform_policy-eq'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'-and$fixtureContract.required_proof_fields.temporal_dispersion_regularizer-eq0.0009765625) 'Actual V14 source contract transform/tau'
foreach($field in @('mean4_regularized_stddev4_export_layout','full_attention_dependent_regularized_moment_adjoints','live_dispersion_score_gradient','raw_variance_root_dispersion_trace','zero_tiny_large_analytic_dispersion_adjoints','supported_nonfinite_and_variance_overflow_refused','absent_local_huge_singleton_sanitized','captured_F32_residual_correction','raw_root_absent_offset_exception','nested_trailing_supports_fixed_edge16','per_window_local_weights_captured','longest_local_equals_global_two_sum','packed_padding_membership_invariance')){Check ($fixtureContract.required_proof_fields[$field]-is[bool]-and$fixtureContract.required_proof_fields[$field]-and$fixtureContract.log_tokens-contains($field+'=true')) ('Actual V14 fixture literal flag/token: '+$field)}
$fixtureNumeric=[ordered]@{backbone_encode_calls=1;public_encoder_calls=0;head_fits=0;synthetic_optimizer_steps=1;manual_F32_absolute_tolerance=3e-5;manual_F32_relative_tolerance=3e-5;F64_finite_difference_step=1e-5;F64_finite_difference_absolute_tolerance=2e-8;F64_finite_difference_relative_tolerance=2e-8;F32_finite_difference_step=2e-3;F32_finite_difference_absolute_tolerance=2e-3;F32_finite_difference_relative_tolerance=2e-3;softmax_gauge_tolerance=2e-6}
foreach($key in $fixtureNumeric.Keys){Check ($fixtureContract.required_proof_fields[$key]-eq$fixtureNumeric[$key]) ('Exact source-derived fixture tolerance/count: '+$key)}
Check ($fixtureContract.required_proof_fields.direct_pool_kernel_calls-eq53-and$fixtureContract.required_proof_fields.F64_finite_difference_coordinates-eq18-and$fixtureContract.required_proof_fields.F32_finite_difference_coordinates-eq18-and$fixtureContract.required_proof_fields.isolated_autograd_grad_calls-eq4) 'NEW compiled temporal source53/18/18/4 exactly'
foreach($key in @('direct_pool_kernel_calls','F64_finite_difference_coordinates','F32_finite_difference_coordinates','isolated_autograd_grad_calls')){Check (($fixtureContract.required_proof_fields[$key]-is[int]-or$fixtureContract.required_proof_fields[$key]-is[long])-and$fixtureContract.required_proof_fields[$key]-gt0) ('Actual source-derived NEW fixture count: '+$key)};Check ($fixtureContract.required_proof_fields.transform_derivative_upper_bound-eq16) 'Retained analytic transform bound16'
Check ($card.initial_pairing.references.Count-eq13-and$card.initial_pairing.reference-eq'CPB-v13.market-01 initial0 DATA/statistic artifacts only'-and($card.initial_pairing.native_coordinate_slice-join',')-eq'20,40'-and-not$card.initial_pairing.content20_equality_required-and-not$card.initial_pairing.whole_native40_equality_required-and-not$card.initial_pairing.initial_scores_reused) 'Partial native20:40 eight DATA archives plus whole five statistics; new own initial heads'
foreach($reference in $card.initial_pairing.references){Check ($reference.reference_data_archive.path.StartsWith('/opt/cuwacunu/research/causal-fresh-v13/attempt-001/')-and$reference.candidate_relative_to_attempt-notmatch'cpb_v13-') 'Initial pairing references saved V13 DATA/statistics only';if($reference.Contains('feature_value_coordinate_slice')){Check (($reference.feature_value_coordinate_slice-join',')-eq'20,40') 'Every feature matrix compares20:40 only'}}
$goal=Read $card.advancement.goal_protocol.path
Verify $card.advancement.goal_protocol
foreach($binding in @($card.retention_source_bindings.protocol,$card.retention_source_bindings.independent_review,$card.retention_source_bindings.application_admission)){Verify $binding}
$newProtocol=Read $card.advancement.goal_protocol.path
$declaredDesign=Read 'doc/causal_fresh_v14/design-source-plan.ADMITTED-001.json'
Check ($newProtocol.kind-eq'clean_causal_encoder_search_goal_v13_retention_protocol_v1'-and$newProtocol.criterion_count-eq22-and$newProtocol.criteria.Count-eq22-and$newProtocol.additional_train_NLL_gate-is[bool]-and-not$newProtocol.additional_train_NLL_gate-and$newProtocol.baseline-eq'CPB-v3.market-01') 'Actual newV3–V13 protocol/noTRAINNLL;old866 ancestry only'
Check (($newProtocol.criteria|ConvertTo-Json -Depth 30 -Compress)-ceq($declaredDesign.prospective_criteria_proposal.criteria|ConvertTo-Json -Depth 30 -Compress)) 'Exactly five V13 floors plus17unchanged criterion objects from reviewed design'
Check ($card.advancement.goal_protocol.bytes-eq$card.retention_source_bindings.protocol.bytes-and$card.advancement.goal_protocol.sha256-eq$card.retention_source_bindings.protocol.sha256-and$card.retention_source_bindings.protocol.path-eq'doc/clean_causal_encoder_search_goal_v13_retention.json') 'V14 protocol alias equals actual canonical newprotocol'
$retentionReview=Read $card.retention_independent_review.path
Check ($retentionReview.kind-eq'clean_causal_encoder_search_goal_v13_retention_independent_review_v1'-and$retentionReview.status-eq'passed'-and$retentionReview.checks-gt0-and$retentionReview.protocol.sha256-eq$card.retention_source_bindings.protocol.sha256-and$retentionReview.protocol.bytes-eq$card.retention_source_bindings.protocol.bytes) 'Actual independent newprotocol review,not inherited723'
Check ($card.retention_independent_review.bytes-eq$card.retention_source_bindings.independent_review.bytes-and$card.retention_independent_review.sha256-eq$card.retention_source_bindings.independent_review.sha256-and$card.retention_application_admission.bytes-eq$card.retention_source_bindings.application_admission.bytes-and$card.retention_application_admission.sha256-eq$card.retention_source_bindings.application_admission.sha256) 'Both V14 review/admission aliases equal actual externalsourcebindings'
Check (($card.advancement.criteria|ConvertTo-Json -Depth 25 -Compress)-ceq($goal.criteria|ConvertTo-Json -Depth 25 -Compress)) 'All22 criterion objects copied literally without reductions'
Verify $card.retention_independent_review;Verify $card.retention_application_admission
$appliedRetention=Read $card.retention_application_admission.path
foreach($flag in @('applied','prospective_future_use_eligible','application_changed','historical_results_or_counts_recomputed','additional_TRAIN_NLL_gate')){Check ($appliedRetention[$flag]-is[bool]) ('Typed actual future application flag: '+$flag)}
Check ($appliedRetention.protocol.path-eq$card.retention_source_bindings.protocol.path-and$appliedRetention.protocol.bytes-eq$card.retention_source_bindings.protocol.bytes-and$appliedRetention.protocol.sha256-eq$card.retention_source_bindings.protocol.sha256-and$appliedRetention.independent_review.path-eq$card.retention_source_bindings.independent_review.path-and$appliedRetention.independent_review.bytes-eq$card.retention_source_bindings.independent_review.bytes-and$appliedRetention.independent_review.sha256-eq$card.retention_source_bindings.independent_review.sha256) 'Actual future application binds exact canonical protocol and independent review'
Check ($appliedRetention.files.Count-eq2-and-not(Compare-Object @($appliedRetention.files.path|Sort-Object) @('doc/clean_causal_encoder_search_goal_v13_retention.json','doc/clean_causal_encoder_search_goal_v13_retention.md'))) 'Only canonical new protocol JSON/MD were applied'
foreach($pin in $appliedRetention.files){Verify $pin}
Check ((Read $card.retention_application_admission.path).kind-eq'clean_causal_encoder_search_goal_v13_retention_future_application_v1'-and(Read $card.retention_application_admission.path).status-eq'passed'-and(Read $card.retention_application_admission.path).applied-and(Read $card.retention_application_admission.path).prospective_future_use_eligible-and-not(Read $card.retention_application_admission.path).application_changed-and-not(Read $card.retention_application_admission.path).historical_results_or_counts_recomputed-and-not(Read $card.retention_application_admission.path).additional_TRAIN_NLL_gate) 'Independent newprotocol and separately eligible application,no historicalrescore'
foreach($key in @('v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v6_establishment','v7_establishment','v8_establishment','v9_establishment','v10_establishment','v11_establishment','v12_establishment','v13_establishment')){Verify $card.comparison_control[$key]}
Check (($card.scoped_references-join',')-eq'CPB-v4.market-01,CPB-v5.market-01,CPB-v6.market-01,CPB-v7.market-01,CPB-v8.market-01,CPB-v9.market-01,CPB-v10.market-01,CPB-v11.market-01,CPB-v12.market-01,CPB-v13.market-01'-and$card.scoped_reference-eq'CPB-v13.market-01') 'Eleven historical controls;scientific source/pairing and completed predecessor V13'
Verify $card.input_manifest
foreach($pin in $card.input_pins){Verify $pin $pin.repository}
foreach($key in @('card','result','independent_audit','human_report','human_report_root_review','completion')){Verify $card.saved_TRAIN_diagnosis[$key]}
Check ($card.saved_TRAIN_diagnosis.result.sha256-eq'716959d4428b3b65d71d4e0fa84fe5d96a84bdd9ab40266785b837d84f54ab27'-and$card.saved_TRAIN_diagnosis.independent_audit.sha256-eq'0d47cc5306b6fbfae2a886c0cf1d1a90dc90a9b18c767cffef8357fdb20d3d11'-and$card.saved_TRAIN_diagnosis.human_report.sha256-eq'da2fefe463a789fba490d82ac408639ae979fe2c2a014c1e5113da9bfc2630e2'-and$card.saved_TRAIN_diagnosis.completion.sha256-eq'5b98ba9adeec76b5bb42aed6b398f1c82fd8d014aa903746c09401e620b4ba0f') 'Completed audited V11/V13 diagnosis73778+2035;prior68264 cached scope historical only'
foreach($key in @('card','receipt','failure_mirror')){Verify $card.historical_V9_reporting_recovery[$key]}
Verify $card.declaration_input_plan
$inputPlan=Read $card.declaration_input_plan.path
Check ($card.reviewed_orchestration_provenance.path-eq$orchestrationPin.path-and$card.reviewed_orchestration_provenance.bytes-eq$orchestrationPin.bytes-and$card.reviewed_orchestration_provenance.sha256-eq$orchestrationPin.sha256) 'Frozen declaration retains the exact externally reviewed orchestration provenance'
Verify $card.reviewed_orchestration_provenance
Check ($card.owned_descriptor_draft.path-eq'doc/causal_fresh_v14/CPB-v14.descriptor-draft.json'-and($card.owned_descriptor_draft|ConvertTo-Json -Depth 8 -Compress)-ceq($inputPlan.descriptor_source|ConvertTo-Json -Depth 8 -Compress)) 'Complete owned descriptor draft remains bound to the exact reviewed input plan'
Verify $card.owned_descriptor_draft
Verify $orchestration.card_markdown_template
Check ($card.historical_V9_reporting_recovery.card.sha256-eq'9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'-and$card.historical_V9_reporting_recovery.receipt.sha256-eq'5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72') 'Historical V9 recovery has no new scientific meaning'
Verify $card.baseline_preservation;Verify $card.pre_execution_preservation
$preservation=Read $card.baseline_preservation.path
Check ($preservation.status-eq'passed'-and$preservation.critical_checkpoint_pins.Count-eq119-and$preservation.selected_V3_checkpoint_sha256-eq'50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e') '119 existing critical checkpoints and selected V3 baseline'
$latestSource=$card.latest_frozen_source_preservation
$latestRoles=[ordered]@{card='doc/CAUSAL_FRESH_V13_CARD.json';manifest='output/engineering/causal-fresh-v13/attempt-001/source-manifest.json';archive='output/engineering/causal-fresh-v13/attempt-001/source.zip';protocol_preflight='doc/causal_fresh_v13/protocol-preflight.json'}
foreach($role in $latestRoles.Keys){$binding=$latestSource.$role;Check ($null-ne$binding-and$binding.path-ceq$latestRoles[$role]) ('Exact latest V13 source role: '+$role);Verify $binding}
$latestCard=Read $latestSource.card.path
$latestManifest=Read $latestSource.manifest.path
$latestPreflight=Read $latestSource.protocol_preflight.path
$latestCount=$latestCard.source_counts.capsule_sources
Check (($latestCount-is[int]-or$latestCount-is[long])-and$latestCount-gt0-and$latestCount-eq$preservation.V13_source_files-and$latestManifest.files.Count-eq$latestCount-and@($latestManifest.files.path|Sort-Object -Unique).Count-eq$latestCount-and($latestPreflight.sources.Count+1)-eq$latestCount-and$latestSource.source_files-eq$latestCount) 'Latest V13 source count derives from frozen card/final preservation/preflight/manifest'
Check ($latestCard.model-eq'CPB-v13'-and-not$latestCard.stage.Contains('DRAFT')-and$latestCard.frozen_utc-and$latestManifest.card_sha256-eq$latestSource.card.sha256-and$latestPreflight.status-eq'passed'-and$latestPreflight.card_sha256-eq$latestSource.card.sha256) 'Latest frozen V13 declaration and source identities'
$latestArchives=@($preservation.source_ZIPs|Where-Object{$_.manifest.path-eq$latestSource.manifest.path})
Check ($latestArchives.Count-eq1-and$latestArchives[0].source_files-eq$latestCount-and$latestArchives[0].manifest.sha256-eq$latestSource.manifest.sha256-and$latestArchives[0].archive.sha256-eq$latestSource.archive.sha256) 'Latest V13 whole ZIP and manifest equal independently preserved bytes'
$latestPathsExpected=@($latestPreflight.sources.path)+@($latestSource.protocol_preflight.path)
Check (-not(Compare-Object (@($latestManifest.files.path)|Sort-Object) ($latestPathsExpected|Sort-Object))) 'Exact latest V13 preflight-plus-receipt path set'
foreach($binding in $latestManifest.files){Verify $binding}
foreach($pin in $preservation.critical_checkpoint_pins){$repository=$pin.repository;if(-not$repository){Check ($pin.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')) 'Only archived five V5 pins may omit repository';$repository='cuwacunu_embedding'};Verify $pin $repository}
Check ($card.baseline_frozen_files.Count-eq47) 'Selected V3 source inventory'
foreach($pin in $card.baseline_frozen_files){Verify $pin 'cuwacunu_embedding';Verify $pin 'cuwacunu_dev'}
$family=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v14') -Recurse -File)
$research=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v14') -Recurse -File)
Check ($family.Count-eq$card.source_counts.family_inside-and$research.Count-eq$card.source_counts.research-and$card.source_counts.family_external_header-eq1) 'Actual final owner family/research counts agree with bound complete path set'
Check ($card.source_pins_after_owner_freeze.Count-eq($family.Count+$research.Count+1)) 'All actual owner source pins'
foreach($pin in $card.source_pins_after_owner_freeze){Verify $pin}
$observed=@(@($family)+@($research)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v14.h'))|ForEach-Object{$_.FullName.Substring($root.Length+1).Replace([char]92,[char]47)}|Sort-Object)
Check (-not(Compare-Object $observed @($card.source_pins_after_owner_freeze.path|Sort-Object))) 'Exact actual owner path set, not merely count'
foreach($binding in $orchestration.files){Verify $binding.parent;Verify $binding.owned}
Check ($orchestration.files.Count-eq10) 'Ten complete owned orchestration helpers'
$checkpoint=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v14/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v14/location_scale_decoder_checkpoint.h') -Raw
foreach($field in @('temporal_pooling_policy','temporal_pooling_geometry_v1','temporal_pooling_trailing_lengths_v1','temporal_pooling_mix_f64','empty_local_bin_policy','temporal_moment_policy','temporal_export_layout','temporal_moment_partition_v1','temporal_dispersion_transform_policy','temporal_dispersion_regularizer')){Check ($checkpoint.Contains($field)) ('Typed write/read field: '+$field)}
Check ($checkpoint.Contains('torch::kFloat64,2')-and$checkpoint.Contains('torch::kInt64,4')-and$checkpoint.Contains('torch::kFloat64,1')) 'Typed dimensions/dtypes remain explicit'
$public=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v14.h') -Raw
Check (([regex]::Matches($public,'write_temporal_pooling').Count)-ge2-and([regex]::Matches($public,'read_temporal_pooling').Count)-ge3) 'Static typed outer/TRAIN gateways call the owned validator'
Verify $card.independent_training_auditor_source_review
$trainingReview=Read $card.independent_training_auditor_source_review.path
Check ($trainingReview.status-eq'passed'-and$trainingReview.kind-eq'cpb_v14_family_and_training_auditor_independent_static_review_v1') 'Actual existing independent training-auditor source review complete'
# Identical path declarations to preflight; generated descriptor/preflight do not exist yet.
$docs=@('LICENSE','doc/CAUSAL_FRESH_V14_CARD.json','doc/CAUSAL_FRESH_V14_CARD.md','code/encoders/causal_patch_bottleneck_mae/versions/CPB-v14.json','doc/CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json','doc/CAUSAL_V11_V13_TRAIN_DIAGNOSIS.md','doc/causal_v11_v13_train_diagnosis/diagnosis.json','doc/causal_v11_v13_train_diagnosis/independent-audit.json','doc/causal_v11_v13_train_diagnosis/completion.json','doc/causal_v11_v13_train_diagnosis/human-report-root-finalization-001.json','doc/causal_v11_v13_train_diagnosis/human-report-independent-review-001.json','doc/causal_fresh_v13/final-preservation.json','doc/causal_fresh_v11/final-preservation.json','doc/CAUSAL_FRESH_V13_CARD.json','doc/causal_fresh_v13/protocol-preflight.json','doc/causal_fresh_v13/complete-mirror-attempt-001.json','doc/CAUSAL_FRESH_V11_CARD.json','doc/causal_fresh_v11/protocol-preflight.json','doc/causal_fresh_v11/complete-mirror-attempt-001.json','doc/causal_fresh_v14/design-source-plan.ADMITTED-001.json')
$controls=@('preflight.ps1','stage-source.ps1','unpack-source.py','audit-build.py','execute-stage.py','run-quality.py','verify-mirror.ps1','run-container.ps1','prepare-draft-card.ps1','freeze-card.ps1','input-manifest.json','raw-controls.json','draft-metadata-copy-receipt.json','preservation-before.json','orchestration-provenance.json','family-source-provenance.json','evaluation-source-provenance.json','card-markdown-template.md','baseline-control-v3.json','baseline-control-v4.json','baseline-control-v5.json','baseline-control-v6.json','baseline-control-v7.json','baseline-control-v8.json','baseline-control-v9.json','baseline-control-v10.json','baseline-control-v11.json','baseline-control-v12.json','baseline-control-v13.json','baseline-v6-establishment.json','baseline-v7-establishment.json','baseline-v8-establishment.json','baseline-v9-establishment.json','baseline-v10-establishment.json','baseline-v11-establishment.json','baseline-v12-establishment.json','baseline-v13-establishment.json','goal-protocol.json','goal-protocol-review.json','goal-protocol-admission.json')
$docs+=$card.declaration_input_plan.path
$docs+=$card.independent_training_auditor_source_review.path
$paths=@($observed)+$docs+@($controls|ForEach-Object{'doc/causal_fresh_v14/'+$_})
Check (@($paths|Sort-Object -Unique).Count-eq$paths.Count) 'No duplicate prospective capsule paths'
$card.source_counts.capsule_sources=$paths.Count+1
$card.source_counts.status='Prospectively_frozen_actual_path_set_before_execution;actual_build_closure_audited_later'
Check ($inputPlan.source_derived_counts_status-eq'source_owner_derived_and_independently_reviewed'-and$card.source_counts.translation_units-eq$inputPlan.static_build.translation_units-and$card.source_counts.binaries-eq$inputPlan.static_build.binaries-and$card.engineering_typed_negative_cases-eq$inputPlan.engineering_typed_negative_cases) 'Actual NEW independently source-derived counts equal frozen input plan'
$prospectiveMarkdown=Get-Content -LiteralPath (Join-Path $root $markdownRelative) -Raw
Check (([regex]::Matches($prospectiveMarkdown,'(?m)^# CPB-v14.*?(?=CPB-v3\.market-01 remains)',[Text.RegularExpressions.RegexOptions]::Singleline)).Count-eq1) 'Exactly one prospective introduction is replaceable while preserving the copyright notice'
$now=[DateTime]::UtcNow.ToString('o')
# Snapshot complete current bytes before the first in-place declaration change.
[IO.Directory]::CreateDirectory($snapshot)|Out-Null
$snapshotPins=@()
foreach($relative in @($cardRelative,$markdownRelative,'doc/causal_fresh_v14/CPB-v14.descriptor-draft.json')){
 $pin=Get-Pin $relative;$copy=$snapshotRelative+'/'+[IO.Path]::GetFileName($relative)
 [IO.File]::Copy((Join-Path $root $relative),(Join-Path $root $copy),$false)
 $actual=Get-Pin $copy;Check ($actual.sha256-eq$pin.sha256-and$actual.bytes-eq$pin.bytes) 'Full pre-freeze bytes preserved'
 $snapshotPins+=[ordered]@{original=$pin;snapshot=$actual}
}
Write-NewJson ($snapshotRelative+'/manifest.json') ([ordered]@{kind='causal_fresh_v14_before_prospective_freeze_v1';created_utc=$now;files=$snapshotPins;model_calls=0;head_fits=0;old_files_modified=$false})
$card.declared_utc=$now;$card.frozen_utc=$now;$card.stage='prospectively_frozen_unmeasured_before_build_training_quality';$card.execution_admitted=$true
$card.source_contract_status='Prospectively frozen after owner and independent source/card/helper/auditor reviews;no AST/build/model/head execution before freeze'
$card.temporal_pooling_proof_contract.status='declared_before_any_execution_fixture_source_contract_agreed'
$card.prospective_freeze=[ordered]@{source=(Get-Pin 'doc/causal_fresh_v14/freeze-card.ps1');prior_draft=$cardPin;prior_markdown=$markdownPin;snapshot=($snapshotRelative+'/manifest.json');root_reviewed=$true;scientific_calls=0;execution_admission='Only the declared exclusive workflow;each phase remains explicitly invoked;no automatic body execution'}
[IO.File]::WriteAllText((Join-Path $root $cardRelative),($card|ConvertTo-Json -Depth 60)+"`n",[Text.UTF8Encoding]::new($false))
$markdown=Get-Content -LiteralPath (Join-Path $root $markdownRelative) -Raw
$introduction='# CPB-v14.market-01 — prospective declaration'+[char]10+[char]10+'Root reviewed V13-SOURCE nested trailing4/8/12/16 local supports at '+$now+' after completed V11/V13 TRAIN diagnosis73778/2035. Retain globalhalf,two-sum/live centeredvariance and actual F32 sqrt(v+2^-10)-1/32. All learned values/optimizer/TRAIN statistics fresh. Actual newV3–V13 retention22 is separate from old866 historicalancestry. Current completed V13/119 remains protected. Card: [CAUSAL_FRESH_V14_CARD.json](CAUSAL_FRESH_V14_CARD.json); source closure '+$card.source_counts.capsule_sources+' files. No AST/build/model/head body has run;every phase requires separate hash-bound admission;whole prior draft preserved.'+[char]10+[char]10
$markdown=[regex]::Replace($markdown,'(?m)^# CPB-v14.*?(?=CPB-v3\.market-01 remains)',$introduction,[Text.RegularExpressions.RegexOptions]::Singleline)
$markdown=$markdown.Replace('Actual build/source/binary/TU counts must be derived and','Actual build/source/binary/TU counts have been derived and').Replace('reviewed from the final owners before admission; this draft does not assume','reviewed from the final owners before admission; the later actual dependency audit must confirm').Replace('with no future artifact count is prefilled','with no future artifact count prefilled')
[IO.File]::WriteAllText((Join-Path $root $markdownRelative),$markdown,[Text.UTF8Encoding]::new($false))
$finalCard=Get-Pin $cardRelative
$descriptor=Read 'doc/causal_fresh_v14/CPB-v14.descriptor-draft.json'
$descriptor.kind='causal_encoder_prospective_version_descriptor_v1';$descriptor.family='causal_patch_bottleneck_mae'
$descriptor.stage='prospectively_frozen_unmeasured_before_build_training_quality';$descriptor.immutable_prospective_semantics=$true
$descriptor.created_utc=$now;$descriptor.created_before_build_training_or_quality=$true;$descriptor.created_after_measurement=$false
$descriptor.card=$finalCard;$descriptor.canonical_descriptor_pending_card_freeze=$false
$descriptor.declaration_provenance='Prospective immutable descriptor pins the frozen card;card names descriptor path without its hash,avoiding a cycle. No measured results.'
Write-NewJson $canonicalRelative $descriptor
$receipt=[ordered]@{kind='causal_fresh_v14_prospective_freeze_v1';status='passed';frozen_utc=$now;checks=$script:checks;card=$finalCard;markdown=(Get-Pin $markdownRelative);descriptor=(Get-Pin $canonicalRelative);snapshot=(Get-Pin ($snapshotRelative+'/manifest.json'));orchestration_provenance=$orchestrationPin;owner_source_files=$observed.Count;reviewed_source_paths_before_preflight=$paths.Count;declared_capsule_sources=$card.source_counts.capsule_sources;critical_checkpoints_verified=119;latest_v13_sources_verified=$latestManifest.files.Count;latest_v13_source_manifest=$latestSource.manifest;latest_v13_source_zip=$latestSource.archive;V3_sources_verified_each_repo=47;criteria_objects_copied_exact=22;AST_calls=0;Docker_calls=0;project_body_calls=0;model_calls=0;head_fits=0;training_updates=0;baseline_changed=$false;application_changed=$false}
Write-NewJson 'doc/causal_fresh_v14/prospective-freeze.json' $receipt
$receipt|ConvertTo-Json -Depth 8

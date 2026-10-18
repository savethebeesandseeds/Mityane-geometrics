# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Complete physical FINAL V12 normal helper copy, adapted to NORMAL V14 quality.
# Host file/hash/JSON metadata only. No Docker, AST, model or auditor execution.
param(
 [Parameter(Mandatory=$true)][switch]$SourcesAndCompletedEvidenceReviewed,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainZipSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QualityReceiptSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QualityStageSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$CompleteMirrorSHA256,
 [Parameter(Mandatory=$true)][string]$SourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SourceReviewSHA256,
 [Parameter(Mandatory=$true)][string]$QualitySourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QualitySourceReviewSHA256,
 [Parameter(Mandatory=$true)][string]$EvaluationSourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$EvaluationSourceReviewSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$HelpersProvenanceSHA256,
 [Parameter(Mandatory=$true)][string]$WorkflowSourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WorkflowSourceReviewSHA256,
 [ValidateRange(1,999)][int]$MainAttempt=1
)
$ErrorActionPreference='Stop'
if(-not$SourcesAndCompletedEvidenceReviewed){throw 'Independent source and actual completed-quality review must precede audit-card freeze'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$helperRelative='doc/causal_fresh_v14_quality_audit'
$output=Join-Path $PSScriptRoot 'audit-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve any existing prospective audit card'}
function Rooted([string]$relative){
 $full=if([IO.Path]::IsPathRooted($relative)){[IO.Path]::GetFullPath($relative)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))}
 if(-not$full.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Explicit repository-bounded file required'}
 for($cursor=$full;$cursor.Length-ge$researchRoot.Length;$cursor=[IO.Path]::GetDirectoryName($cursor)){
  if((Get-Item -LiteralPath $cursor).Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked path refused'}
  if($cursor-eq$researchRoot){break}
 }
 $full
}
function Pin([string]$relative){
 $path=Rooted $relative;$f=Get-Item -LiteralPath $path
 [ordered]@{path=$path.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Require-Pin($pin){$actual=Pin $pin.path;if($actual.bytes-ne$pin.bytes-or$actual.sha256-ne$pin.sha256){throw ('Exact file required: '+$pin.path)}}
function Workflow-RepositoryPath([string]$value){
 $full=if([IO.Path]::IsPathRooted($value)){[IO.Path]::GetFullPath($value)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $value))}
 if(-not$full.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Repository-bounded workflow review required'}
 for($cursor=$full;$cursor.Length-ge$researchRoot.Length;$cursor=[IO.Path]::GetDirectoryName($cursor)){if((Get-Item -LiteralPath $cursor).Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked workflow/review path refused'};if($cursor-eq$researchRoot){break}}
 $full
}
function Workflow-Pin([string]$value){
 $full=Workflow-RepositoryPath $value;$file=Get-Item -LiteralPath $full
 [ordered]@{path=$full.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Read-IndependentWorkflowReview{
 $reviewPin=Workflow-Pin $WorkflowSourceReviewPath
 if($reviewPin.sha256-cne$WorkflowSourceReviewSHA256){throw 'External independent workflow-review SHA required before parse'}
 $review=Get-Content -LiteralPath (Workflow-RepositoryPath $reviewPin.path) -Raw|ConvertFrom-Json
 if($review.kind-cne'cpb_v14_quality_audit_workflow_independent_static_review_v1'-or$review.status-cne'passed'-or$review.source_only-isnot[bool]-or-not$review.source_only-or$review.independent_of_author-isnot[bool]-or-not$review.independent_of_author-or$review.scientific_execution_admitted-isnot[bool]-or$review.scientific_execution_admitted-ne$false){throw 'Independent SOURCE-ONLY workflow review with exact kind/status is required'}
 $prefix='doc/causal_fresh_v14_quality_audit/'
 $names=@('prepare-card.ps1','stage.ps1','execute.py','mirror.ps1','provenance.json')
 $paths=@($names|ForEach-Object{$prefix+$_})
 if($review.reviewed_sources.Count-ne$paths.Count-or(@($review.reviewed_sources.path|Sort-Object -Unique).Count)-ne$paths.Count-or(Compare-Object ($paths|Sort-Object) @($review.reviewed_sources.path|Sort-Object))){throw 'Review must bind exactly five owned workflow files including provenance'}
 foreach($entry in $review.reviewed_sources){$actual=Workflow-Pin $entry.path;if($actual.bytes-ne$entry.bytes-or$actual.sha256-cne$entry.sha256){throw ('Reviewed workflow source changed: '+$entry.path)}}
 $provenancePin=Workflow-Pin ($prefix+'provenance.json')
 $provenance=Get-Content -LiteralPath (Workflow-RepositoryPath $provenancePin.path) -Raw|ConvertFrom-Json
 $executablePaths=@($paths|Where-Object{$_-ne($prefix+'provenance.json')})
 if($provenance.kind-cne'cpb_v14_owned_quality_audit_orchestration_source_v1'-or$provenance.current_sources.Count-ne$executablePaths.Count-or(Compare-Object ($executablePaths|Sort-Object) @($provenance.current_sources.path|Sort-Object))){throw 'Exact complete four-helper provenance required'}
 foreach($entry in $provenance.current_sources){$matched=@($review.reviewed_sources|Where-Object{$_.path-ceq$entry.path});if($matched.Count-ne1-or$matched[0].bytes-ne$entry.bytes-or$matched[0].sha256-cne$entry.sha256){throw 'Independent review and current helper provenance disagree'}}
 [ordered]@{review_pin=$reviewPin;review=$review;provenance_pin=$provenancePin;provenance=$provenance}
}
$workflowGuard=Read-IndependentWorkflowReview
$workflowPin=$workflowGuard.provenance_pin
if($workflowPin.sha256-ne$HelpersProvenanceSHA256){throw 'External reviewed helper provenance SHA required before parse'}
$workflow=$workflowGuard.provenance
if($workflow.kind-ne'cpb_v14_owned_quality_audit_orchestration_source_v1'-or$workflow.current_sources.Count-ne4){throw 'Exact reviewed four-helper closure required'}
foreach($entry in $workflow.current_sources){Require-Pin $entry}
$self=@($workflow.current_sources|Where-Object{$_.path-eq($helperRelative+'/prepare-card.ps1')})
if($self.Count-ne1-or$self[0].sha256-ne(Pin $PSCommandPath).sha256){throw 'Generator self pin required'}
$reviewPin=Pin $SourceReviewPath
if($reviewPin.path-cne'doc/causal_fresh_v14/source-reviews/root-complete-source-closure-002.json'-or$reviewPin.sha256-cne$SourceReviewSHA256){throw 'External actual revised root104 source-review SHA required before parse'}
if($reviewPin.bytes-ne$workflow.scientific_source_authorities.combined_root104.bytes-or$reviewPin.sha256-cne$workflow.scientific_source_authorities.combined_root104.sha256){throw 'Exact separately reviewed root104closure002 provenance binding required'}
$qualityReviewPin=Pin $QualitySourceReviewPath
$evaluationReviewPin=Pin $EvaluationSourceReviewPath
if($qualityReviewPin.path-cne'doc/causal_fresh_v14/source-reviews/independent-quality-auditor-source-review-001.json'-or$qualityReviewPin.sha256-cne$QualitySourceReviewSHA256-or$evaluationReviewPin.path-cne'doc/causal_fresh_v14/source-reviews/independent-training-evaluation-auditor-source-review-001.json'-or$evaluationReviewPin.sha256-cne$EvaluationSourceReviewSHA256){throw 'External actual independent quality and evaluation review SHAs required before either parse'}
$review=Get-Content -Raw -LiteralPath (Rooted $reviewPin.path)|ConvertFrom-Json
$qualityReview=Get-Content -Raw -LiteralPath (Rooted $qualityReviewPin.path)|ConvertFrom-Json
$evaluationReview=Get-Content -Raw -LiteralPath (Rooted $evaluationReviewPin.path)|ConvertFrom-Json
if($review.kind-cne'cpb_v14_complete_104_owned_source_root_closure_v1'-or$review.status-cne'passed'-or$review.reviewed_owned_files.Count-ne104){throw 'Actual104-source root closure required;root review is not the independent quality/evaluation review'}
if($qualityReview.kind-cne'cpb_v14_independent_quality_auditor_source_review_v1'-or$qualityReview.status-cne'passed'-or$qualityReview.independent_static_source_review-isnot[bool]-or-not$qualityReview.independent_static_source_review-or$qualityReview.reviewed_owned_files.Count-ne4){throw 'Actual independent quality4 source authority required'}
if($evaluationReview.kind-cne'cpb_v14_training_evaluation_and_training_auditor_independent_static_review_v1'-or$evaluationReview.status-cne'passed'-or$evaluationReview.independent_of_author-isnot[bool]-or-not$evaluationReview.independent_of_author-or$evaluationReview.reviewed_owned_files.Count-ne32){throw 'Actual independent training/evaluation/auditor32 authority required'}
foreach($authority in @($qualityReviewPin,$evaluationReviewPin)){$match=@($review.independent_reviews.PSObject.Properties.Value|Where-Object{$_.path-ceq$authority.path});if($match.Count-ne1-or$match[0].bytes-ne$authority.bytes-or$match[0].sha256-cne$authority.sha256){throw 'Root closure must name each actual independent review exactly once'}}
foreach($entry in $review.reviewed_owned_files){Require-Pin $entry}
$evaluationSources=@($evaluationReview.reviewed_owned_files|Where-Object{$_.path.StartsWith('code/research/causal_fresh_v14/evaluation/')})
if($evaluationSources.Count-ne20-or@($evaluationSources.path|Sort-Object -Unique).Count-ne20){throw 'Exactly20 independently reviewed evaluator assets required'}
$evaluationMetadata=@($evaluationReview.reviewed_owned_files|Where-Object{$_.path-ceq'doc/causal_fresh_v14/evaluation-source-provenance.json'})
if($evaluationMetadata.Count-ne1){throw 'Actual32-source nonself authority must include exactly one evaluation metadata source'}
foreach($entry in @($qualityReview.reviewed_owned_files)+@($evaluationReview.reviewed_owned_files)){Require-Pin $entry;$binding=@($review.reviewed_owned_files|Where-Object{$_.path-ceq$entry.path});if($binding.Count-ne1-or$binding[0].bytes-ne$entry.bytes-or$binding[0].sha256-cne$entry.sha256){throw 'Independent source pin must join current combined root closure'}}
$sourceRelative='code/research/causal_fresh_v14_quality_audit'
$sources=@(Get-ChildItem -LiteralPath (Rooted $sourceRelative) -File|Sort-Object Name|ForEach-Object{
 $relative=$_.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);$pin=Pin $relative
 $binding=@($qualityReview.reviewed_owned_files|Where-Object{$_.path-ceq$relative})
 if($binding.Count-ne1-or$binding[0].bytes-ne$pin.bytes-or$binding[0].sha256-ne$pin.sha256){throw 'Current auditor must match independent review exactly'}
 [ordered]@{file=$_.Name;bytes=$pin.bytes;sha256=$pin.sha256}
})
if(($sources.file-join',')-cne'archive_reader.py,audit.py,provenance.json,README.md'){throw 'Exactly four owned auditor sources required'}
$mainPin=Pin 'doc/CAUSAL_FRESH_V14_CARD.json'
if($mainPin.sha256-ne$MainCardSHA256){throw 'External actual main card SHA required before parse'}
$main=Get-Content -Raw -LiteralPath (Rooted $mainPin.path)|ConvertFrom-Json
$inputPin=Pin 'doc/causal_fresh_v14/input-manifest.json'
if($main.model-ne'CPB-v14'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or$main.execution_admitted-isnot[bool]-or-not$main.execution_admitted){throw 'Actual prospectively frozen V14 scientific card required'}
if($inputPin.bytes-ne$main.input_manifest.bytes-or$inputPin.sha256-ne$main.input_manifest.sha256){throw 'Exact main-card data-only input identity required'}
if(($main.initial_pairing.native_coordinate_slice-join',')-ne'20,40'-or$main.initial_pairing.content20_equality_required-ne$false-or$main.initial_pairing.native_coordinates_compared-ne20-or$main.initial_pairing.references.Count-ne13){throw 'Exact partial20:40 eight-DATA/five-whole-statistic pairing declaration required'}
if($main.initial_pairing.whole_native40_equality_required-ne$false-or$main.initial_pairing.initial_scores_reused-ne$false){throw 'Only saved V13 DATA tails/whole-statistic references;new own initial heads'}
foreach($reference in $main.initial_pairing.references){if(-not$reference.reference_data_archive.path.StartsWith('/opt/cuwacunu/research/causal-fresh-v13/attempt-001/')-or$reference.candidate_relative_to_attempt-match'cpb_v13-'){throw 'V14 references saved V13 DATA only'};if($reference.PSObject.Properties.Name-contains'feature_value_coordinate_slice'){if(($reference.feature_value_coordinate_slice-join',')-cne'20,40'){throw 'Only feature value coordinates20:40 may be paired'}}}
$policy='original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1'
$momentPolicy='componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1'
$transformPolicy='sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'
$tau=0.0009765625
foreach($record in @($main.mechanism,$main.training)){if($record.temporal_pooling_policy-cne$policy-or($record.PSObject.Properties.Name-contains'temporal_pooling_partition')-or($record.temporal_pooling_geometry-join',')-cne'4,16,32,8'-or($record.temporal_pooling_trailing_lengths-join',')-cne'4,8,12,16'-or$record.temporal_moment_policy-cne$momentPolicy-or($record.temporal_moment_partition-join',')-cne'4,4'-or($record.temporal_pooling_mix-join',')-cne'0.5,0.5'-or$record.temporal_dispersion_transform_policy-cne$transformPolicy-or$record.temporal_dispersion_regularizer-ne$tau){throw 'Exact frozen V14 attention-dependent raw variance and regularized-stddev transform required'}}
if(($main.actual_step_temporal_witness_contract.per_window_local_weight_fields-join',')-cne'temporal_pool_local_weights,temporal_contextual_pool_local_weights'-or($main.actual_step_temporal_witness_contract.per_window_local_weight_shape-join',')-cne'R,4,T'){throw 'Exact two same-forward per-window local captures required'}
$proofFlags=@('mean4_regularized_stddev4_export_layout','full_attention_dependent_regularized_moment_adjoints','live_dispersion_score_gradient','raw_variance_root_dispersion_trace','zero_tiny_large_analytic_dispersion_adjoints','supported_nonfinite_and_variance_overflow_refused','absent_local_huge_singleton_sanitized','captured_F32_residual_correction','raw_root_absent_offset_exception','nested_trailing_supports_fixed_edge16','per_window_local_weights_captured','longest_local_equals_global_two_sum','packed_padding_membership_invariance')
foreach($field in $proofFlags){if($main.temporal_pooling_proof_contract.required_proof_fields.$field-isnot[bool]-or-not$main.temporal_pooling_proof_contract.required_proof_fields.$field){throw 'All thirteen V14 proof predicates must have been prospectively frozen'}}
$proof=$main.temporal_pooling_proof_contract.required_proof_fields
if($proof.direct_pool_kernel_calls-ne53-or$proof.F64_finite_difference_coordinates-ne18-or$proof.F32_finite_difference_coordinates-ne18-or$proof.isolated_autograd_grad_calls-ne4-or$proof.transform_derivative_upper_bound-ne16-or$main.engineering_typed_negative_cases-ne821-or$main.actual_step_temporal_witness_contract.total_fields-ne60-or$main.actual_step_temporal_witness_contract.temporal_trace_fields-ne25){throw 'Source-derived53/18/18/4/821 and60-field same-forward archive required'}
$goalPin=Pin $main.advancement.goal_protocol.path;Require-Pin $main.advancement.goal_protocol
if($goalPin.bytes-ne$workflow.canonical_retention_protocol.bytes-or$goalPin.sha256-cne$workflow.canonical_retention_protocol.sha256-or$main.advancement.criterion_count-ne22-or($main.scoped_references-join',')-cne'CPB-v4.market-01,CPB-v5.market-01,CPB-v6.market-01,CPB-v7.market-01,CPB-v8.market-01,CPB-v9.market-01,CPB-v10.market-01,CPB-v11.market-01,CPB-v12.market-01,CPB-v13.market-01'){throw 'Eleven saved controls and actual strongest V3-V13/22 protocol required'}
$goal=Get-Content -LiteralPath (Rooted $goalPin.path) -Raw|ConvertFrom-Json
if($goal.kind-cne'clean_causal_encoder_search_goal_v13_retention_protocol_v1'-or$goal.criterion_count-ne22-or$goal.additional_train_NLL_gate-ne$false-or($goal.criteria|ConvertTo-Json -Depth 20 -Compress)-cne($main.advancement.criteria|ConvertTo-Json -Depth 20 -Compress)){throw 'Actual full ordered22 criterion objects/noTRAIN-NLL gate'}
foreach($pin in @($main.retention_independent_review,$main.retention_application_admission)){Require-Pin $pin}
$retentionReview=Get-Content -LiteralPath (Rooted $main.retention_independent_review.path) -Raw|ConvertFrom-Json
$retentionApplication=Get-Content -LiteralPath (Rooted $main.retention_application_admission.path) -Raw|ConvertFrom-Json
if($retentionReview.kind-cne'clean_causal_encoder_search_goal_v13_retention_independent_review_v1'-or$retentionReview.status-cne'passed'-or$retentionReview.checks-le0-or$retentionApplication.kind-cne'clean_causal_encoder_search_goal_v13_retention_future_application_v1'-or$retentionApplication.status-cne'passed'-or$retentionApplication.applied-ne$true-or$retentionApplication.prospective_future_use_eligible-ne$true-or$retentionApplication.application_changed-ne$false-or$retentionApplication.historical_results_or_counts_recomputed-ne$false-or$retentionApplication.additional_TRAIN_NLL_gate-ne$false){throw 'Actual nonself reviewed/applied protocol required, never future receipt assumptions'}
foreach($p in @($retentionReview.protocol,$retentionApplication.protocol)){if($p.path-cne$workflow.canonical_retention_protocol.path-or$p.bytes-ne$goalPin.bytes-or$p.sha256-cne$goalPin.sha256){throw 'Exact canonical protocol identities behind main alias'}}
if($retentionApplication.independent_review.path-cne$main.retention_source_bindings.independent_review.path-or$retentionApplication.independent_review.bytes-ne$main.retention_independent_review.bytes-or$retentionApplication.independent_review.sha256-cne$main.retention_independent_review.sha256){throw 'Exact canonical independent protocol review path and main-alias bytes/SHA binding'}
$label='attempt-{0:D3}' -f $MainAttempt
$packageRoot='output/engineering/causal-fresh-v14/'+$label
$runtimeMain='/opt/cuwacunu/research/causal-fresh-v14/'+$label
$manifestPin=Pin ($packageRoot+'/source-manifest.json')
$zipPin=Pin ($packageRoot+'/source.zip')
if($manifestPin.sha256-ne$MainManifestSHA256-or$zipPin.sha256-ne$MainZipSHA256){throw 'External actual main manifest/ZIP SHA guards before parse'}
$manifest=Get-Content -Raw -LiteralPath (Rooted $manifestPin.path)|ConvertFrom-Json
$packagePin=Pin ($packageRoot+'/package.json')
$package=Get-Content -Raw -LiteralPath (Rooted $packagePin.path)|ConvertFrom-Json
if($manifest.card_sha256-ne$MainCardSHA256-or$manifest.files.Count-le0-or$package.source_files-ne$manifest.files.Count-or$package.manifest_sha256-ne$MainManifestSHA256-or$package.zip_sha256-ne$MainZipSHA256){throw 'Immutable actual main package identity required'}
if($manifest.files.Count-le0-or$main.source_counts.binaries-ne7-or$main.source_counts.translation_units-ne11-or$main.source_counts.capsule_sources-ne$manifest.files.Count){throw 'Actual dynamically frozen source seven-binary eleven-TU closure required'}
foreach($entry in $evaluationSources){$binding=@($manifest.files|Where-Object{$_.path-ceq$entry.path});if($binding.Count-ne1-or$binding[0].bytes-ne$entry.bytes-or$binding[0].sha256-cne$entry.sha256){throw 'Current independently reviewed evaluator20 must be actual frozen main package members;external quality4 remains bound by root104/nonself quality review/own audit capsule'}}
$qualityPin=Pin 'output/runs/causal-fresh-v14/run-001/quality-001/quality-receipt.json'
$qualityStageHost=Pin ($packageRoot+'/quality-stage.json')
$completePin=Pin ('doc/causal_fresh_v14/complete-mirror-'+$label+'.json')
if($qualityPin.sha256-ne$QualityReceiptSHA256-or$qualityStageHost.sha256-ne$QualityStageSHA256-or$completePin.sha256-ne$CompleteMirrorSHA256){throw 'External passed quality receipt/stage/Complete hashes required before parse'}
$quality=Get-Content -Raw -LiteralPath (Rooted $qualityPin.path)|ConvertFrom-Json
$qualityStage=Get-Content -Raw -LiteralPath (Rooted $qualityStageHost.path)|ConvertFrom-Json
$complete=Get-Content -Raw -LiteralPath (Rooted $completePin.path)|ConvertFrom-Json
if($quality.kind-ne'causal_fresh_v14_quality_execution_v1'-or$quality.status-ne'passed'-or$quality.model-ne'CPB-v14.market-01'-or$quality.card_sha256-ne$MainCardSHA256-or$quality.gradient_policy-ne'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'){throw 'Actual passed normal V14 quality receipt required'}
if($qualityStage.kind-ne'causal_fresh_v14_exclusive_stage_execution_v1'-or$qualityStage.status-ne'passed'-or$qualityStage.exit_code-ne0){throw 'Actual completed outer quality wrapper required,not a reporting-recovery assumption'}
if($complete.status-ne'passed'-or$complete.scope-ne'Complete'-or$complete.attempt-ne$label-or$complete.card_sha256-ne$MainCardSHA256-or$complete.source_manifest_sha256-ne$MainManifestSHA256-or-not$complete.all_mirrored_pt_ignored){throw 'Passed exact completed main mirror required'}
$sourcePackages=@($manifestPin,$zipPin,$packagePin)
foreach($entry in $sourcePackages){$preserved=@($complete.preserved_source_packages|Where-Object{$_.path-ceq$entry.path});if($preserved.Count-ne1-or$preserved[0].bytes-ne$entry.bytes-or$preserved[0].sha256-cne$entry.sha256){throw 'One exact preserved_source_packages host manifest/wholeZIP/package role required;these are not container payload files'};Require-Pin $preserved[0]}
function Require-CompletedFile([string]$relative,[string]$runtime){
 $matches=@($complete.files|Where-Object{$_.path-eq$relative-and$_.container_path-eq$runtime})
 if($matches.Count-ne1){throw ('One exact completed mirror member required: '+$relative)}
 Require-Pin $matches[0]
 $matches[0]
}
$null=Require-CompletedFile $qualityPin.path ($runtimeMain+'/quality-001/quality-receipt.json')
$null=Require-CompletedFile $qualityStageHost.path ($runtimeMain+'/quality-stage.json')
$temporalMatches=@($complete.files|Where-Object{$_.container_path-ceq$quality.temporal_pooling_proof.path})
if($temporalMatches.Count-ne1-or$temporalMatches[0].bytes-ne$quality.temporal_pooling_proof.bytes-or$temporalMatches[0].sha256-cne$quality.temporal_pooling_proof.sha256){throw 'One exact Complete-mirrored current temporal proof required'}
Require-Pin $temporalMatches[0]
$temporal=Get-Content -LiteralPath (Rooted $temporalMatches[0].path) -Raw|ConvertFrom-Json
if($temporal.kind-cne'cpb_v14_temporal_pooling_mechanism_proof_v1'-or$temporal.status-cne'passed'-or$temporal.model-cne'CPB-v14'-or$temporal.temporal_pooling_policy-cne$policy-or$temporal.temporal_moment_policy-cne$momentPolicy-or$temporal.temporal_dispersion_transform_policy-cne$transformPolicy-or$temporal.temporal_dispersion_regularizer-ne$tau-or$temporal.direct_pool_kernel_calls-ne53-or$temporal.F64_finite_difference_coordinates-ne18-or$temporal.F32_finite_difference_coordinates-ne18-or$temporal.isolated_autograd_grad_calls-ne4){throw 'Actual saved53-call V14 regularized-stddev proof must match the prospective recipe'}
foreach($field in $proofFlags){if($temporal.$field-isnot[bool]-or-not$temporal.$field){throw ('Actual V14 temporal proof failed: '+$field)}}
$science=@('extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads')
if((@($quality.stage_seconds.PSObject.Properties.Name|Sort-Object)-join',')-cne(@($science|Sort-Object)-join',')){throw 'The receipt contains five scientific durations,before report/comparison'}
foreach($name in @($science)+@('report','comparison')){
 $relative='output/runs/causal-fresh-v14/run-001/quality-001/'+$name+'-stage.json'
 $null=Require-CompletedFile $relative ($runtimeMain+'/quality-001/'+$name+'-stage.json')
 $stage=Get-Content -Raw -LiteralPath (Rooted $relative)|ConvertFrom-Json
 if($stage.stage-ne$name-or$stage.exit_code-ne0){throw 'All seven normal local stages must have passed'}
}
foreach($relative in @('report/table_rows.json','report/completion.json','comparison/comparison.json')){
 $null=Require-CompletedFile ('output/runs/causal-fresh-v14/run-001/quality-001/'+$relative) ($runtimeMain+'/quality-001/'+$relative)
}
$qualityRuntimePin=[ordered]@{path=($runtimeMain+'/quality-stage.json');bytes=$qualityStageHost.bytes;sha256=$qualityStageHost.sha256}
$payloadFiles=@($sources.file)+@('audit-card.json','LICENSE','execute-audit.py','workflow-review.json','workflow-provenance.json','scientific-source-review.json','quality-source-review.json','evaluation-source-review.json')
if(@($payloadFiles|Sort-Object -Unique).Count-ne$payloadFiles.Count){throw 'Unique flat audit capsule role names required'}
$pythonFiles=@($payloadFiles|Where-Object{$_.EndsWith('.py')}|Sort-Object)
$card=[ordered]@{
 kind='causal_fresh_v14_saved_quality_audit_card_v1';copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 declared_utc=[DateTime]::UtcNow.ToString('o');stage='declared_after_static_review_before_any_saved_numeric_audit'
 execution_pending=$true;model='CPB-v14.market-01';model_calls=0;head_fits=0;training_updates=0;old_model_numeric_reads=0
 main_card=$mainPin;input_manifest=$inputPin;main_source_manifest_sha256=$manifestPin.sha256;main_source_zip_sha256=$zipPin.sha256;main_source_files=$manifest.files.Count
 main_attempt=$label;audit_attempt='attempt-001';outside_frozen_main_source=$true;auditor_sources=$sources
 expected_main_paths=[ordered]@{training=($runtimeMain+'/run-001');quality=($runtimeMain+'/quality-001');card=($runtimeMain+'/source/doc/CAUSAL_FRESH_V14_CARD.json')}
 paired_initial_reference_inputs=$main.initial_pairing.references;native_coordinate_slice=@(20,40);native_coordinates_compared=20;content20_equality_required=$false;whole_native40_equality_claimed=$false
 goal_protocol=(Pin 'doc/causal_fresh_v14/goal-protocol.json')
 source_parent=(Pin 'code/research/causal_fresh_v13_quality_audit/audit.py')
 independent_source_review=$reviewPin;helpers_provenance=$workflowPin;independent_workflow_source_review=$workflowGuard.review_pin
 independent_quality_source_review=$qualityReviewPin;independent_evaluation_source_review=$evaluationReviewPin
 independent_source_review_scope='Combined root104 closure links actual separate nonself independent quality4 and training/evaluation/auditor32 authority selecting evaluator20+documentary1;root review is not relabeled independent'
 staging_transport=[ordered]@{payload_files=$payloadFiles;payload_count=$payloadFiles.Count;ZIP_member_count=($payloadFiles.Count+1);python_AST_files=$pythonFiles;python_AST_count=$pythonFiles.Count;workflow_review_file='workflow-review.json';workflow_provenance_file='workflow-provenance.json';scientific_source_review_file='scientific-source-review.json';quality_source_review_file='quality-source-review.json';evaluation_source_review_file='evaluation-source-review.json';external_workflow_review_SHA_required_at_every_phase=$true}
 temporal_pooling_policy=$policy;temporal_pooling_geometry=@(4,16,32,8);temporal_pooling_trailing_lengths=@(4,8,12,16);temporal_moment_policy=$momentPolicy;temporal_dispersion_transform_policy=$transformPolicy;temporal_dispersion_regularizer=$tau;required_new_temporal_flags=$proofFlags;temporal_fixture_kernel_calls=53;F64_finite_difference_coordinates=18;F32_finite_difference_coordinates=18;isolated_autograd_grad_calls=4;engineering_typed_negative_cases=821;engineering_full_invalid_archive_files=817;engineering_direct_loader_requests=4;engineering_compact_archive_files=0;engineering_restored_target_controls=0;engineering_legacy_extra_partition_files=5;actual_step_local_weight_fields=@('temporal_pool_local_weights','temporal_contextual_pool_local_weights');actual_step_local_weight_shape=@('R',4,'T');actual_step_archive_fields=60;actual_step_temporal_trace_fields=25
 typed_transform_contract='UTF8 String transform-policy plus Float64 vector[1] exact tau2^-10 at all5nodes;reject0-D/F32/wronglength/value;raw root unmasked1/32 at zero/absent,dispersion0,served absent whole8 zero afterbias'
 completed_quality_receipt=$qualityPin;completed_quality_stage=$qualityRuntimePin;completed_main_mirror=$completePin
 completed_main_source_packages=$sourcePackages
 report_only_recovery_required=$false
 scope=@('All112TRAIN/32knownVALIDATION saved market predictions,physical errors,ridge reconstruction,all3repetitions,fixedD40recipes;no fitting','TEMPO5 saved count/group/interval/mean/coverage arithmetic;1572 recorded initial/final exports;no model or head implementation forward','13savedV13 DATA references:eight initial native20:40 tails with whole IDs/support/labels plus five whole fresh-statistic archives;content0:20 unpaired;no imported weights/statistics or old score reuse','Eleven saved clean V3-V13 controls,eight separate V6-V13 trust joins;historical FULL40/PARTIAL semantics plus opaqueV9 recovery unchanged;13methods52marketrows','Actual separately reviewed/applied V3-V13 strongest22 exact ordered max/min/positivity conditions,earlier chronological ties,noTRAINNLLgate;no automatic winner/application selection','Actual declared source count/sevenbinary11TU closure;unchanged V13 detached-error objective;53-call regularized-stddev proof with13flags,18F64/18F32/4isolated,821typedrefusals;75mandatory tensors plus scalar gauge capacity','Actual passed normal outerquality wrapper/five scientific receipt durations/seven local exit0 stage files and report/comparison outputs;no current recovery','Four temporal CSV artifacts8192/2048/2048/16384 and60-field same-existing-forward actual1/2048 witnesses;quality binds producer temporal-learning/proof, separate final join binds independentTRAINaudit;no every-scalar/full-hidden proof claim')
 fresh_schema_contract='Only main fresh-constructor JSON omits gradient_policy; requireabsence there and strict policies elsewhere.'
 execution_wrapper=(Pin ($helperRelative+'/execute.py'));stage_helper=(Pin ($helperRelative+'/stage.ps1'));mirror_helper=(Pin ($helperRelative+'/mirror.ps1'));card_generator=(Pin ($helperRelative+'/prepare-card.ps1'))
 forbidden=@('Torch/NumPy or producer imports','Encoder/decoder/head forward or fitting','Old trained checkpoint numeric reads','Whole40 current initial equality claim','Current reporting recovery arguments or prior-failure assumptions','PCA/tuning/selection','Mutation of source/cards/results/old evidence')
}
[IO.File]::WriteAllText($output,($card|ConvertTo-Json -Depth 30)+"`n",[Text.UTF8Encoding]::new($false))
Pin ($helperRelative+'/audit-card.json')|ConvertTo-Json -Depth 5

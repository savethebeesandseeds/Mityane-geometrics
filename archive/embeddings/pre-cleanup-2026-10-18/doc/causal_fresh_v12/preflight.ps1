# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# File/hash and source inspection only. Project execution stays in cuwacunu.
param([Parameter(Mandatory=$true)][string]$CardSHA256,[Parameter(Mandatory=$true)][switch]$RootReviewedSourceAndCard)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspace=[IO.Path]::GetFullPath((Join-Path $root '..'))
$destination=Join-Path $PSScriptRoot 'protocol-preflight.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve existing preflight receipt'}
$script:checks=0
if(-not$RootReviewedSourceAndCard){throw 'Explicit independent source/card review required'}
function Check([bool]$ok,[string]$why){$script:checks++;if(-not $ok){throw $why}}
function Resolve-Pin($p,[string]$repo='cuwacunu_embedding'){
 if($repo -notin @('cuwacunu_embedding','cuwacunu_dev')){throw 'Explicit known repository required'}
 $base=[IO.Path]::GetFullPath((Join-Path $workspace $repo))
 $path=if([IO.Path]::IsPathRooted($p.path)){[IO.Path]::GetFullPath($p.path)}else{[IO.Path]::GetFullPath((Join-Path $base $p.path))}
 Check ($path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin stays within exact repository'
 $path
}
function Verify($p,[string]$repo='cuwacunu_embedding'){
 $f=Get-Item -LiteralPath (Resolve-Pin $p $repo)
 Check (-not($f.Attributes-band[IO.FileAttributes]::ReparsePoint)) ('Physical input: '+$p.path)
 Check ($f.Length-eq$p.bytes-and(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()-eq$p.sha256) ('Exact input: '+$p.path)
}
function Read([string]$p){Get-Content -LiteralPath (Join-Path $root $p) -Raw|ConvertFrom-Json}
function Pin([string]$p){$f=Get-Item -LiteralPath (Join-Path $root $p);Check (-not($f.Attributes-band[IO.FileAttributes]::ReparsePoint)) ('Physical source: '+$p);[ordered]@{path=$p;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}
Check ((Pin 'doc/CAUSAL_FRESH_V12_CARD.json').sha256-eq$CardSHA256) 'Exact prospective card'
$card=Read 'doc/CAUSAL_FRESH_V12_CARD.json'
Check (-not$card.stage.Contains('DRAFT')-and$null-ne$card.frozen_utc-and$card.execution_admitted) 'Frozen and explicitly admitted before any build/test/learning/quality'
Check ($card.model-eq'CPB-v12'-and$card.instance-eq'CPB-v12.market-01'-and$card.geometry.native_width-eq40) 'Named native40 candidate'
Check ($card.mechanism.kind-eq'original-position-four-bin-attention-mean4-uniform-centered-variance4-v1'-and$card.mechanism.temporal_pooling_policy-eq'original-position-four-bin-attention-mean4-uniform-centered-variance4-v1'-and($card.mechanism.temporal_pooling_partition -join ',')-eq'4,16,32,4,8'-and($card.mechanism.temporal_pooling_mix -join ',')-eq'0.5,0.5'-and$card.mechanism.empty_local_bin_policy-eq'zero-whole-export-slice-after-bias-v1') 'Attention mean4/uniform dispersion4 with unchanged V11 objective'
Check ($card.geometry.temporal_pooling_policy-eq$card.mechanism.temporal_pooling_policy-and($card.geometry.temporal_pooling_partition -join ',')-eq'4,16,32,4,8'-and($card.geometry.temporal_pooling_mix -join ',')-eq'0.5,0.5'-and$card.geometry.empty_local_bin_policy-eq$card.mechanism.empty_local_bin_policy) 'Geometry and mechanism temporal declarations exactly agree'
Check ($card.mechanism.gradient_policy-eq'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'-and$card.training.global_gradient_clip-eq1) 'V11 detached-error objective with unchanged globalclip1'
Check ($card.training.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and($card.training.loss_coefficients -join ',')-eq'1,1'-and$card.training.variance_error_detached-and-not$card.training.reporting_nll_backpropagated) 'V11 Huber1 mean plus detached-error variance objective;detached reporting copy'
$expectedRoutes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
foreach($record in @($card.geometry,$card.mechanism,$card.training,$card.trust_receipt_objective_fields,$card.temporal_pooling_proof_contract,$card.temporal_learning_receipt_contract)){
 Check ($record.temporal_moment_policy-eq'componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1'-and$record.temporal_export_layout-eq'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'-and($record.temporal_moment_partition-join',')-eq'4,4') 'Exact pooling moment policy/layout/partition'
}
foreach($record in @($card.mechanism,$card.training,$card.trust_receipt_objective_fields)){Check ($record.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and$record.gradient_policy-eq'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'-and$record.direct_gradient_routes-eq$expectedRoutes-and$record.variance_error_detached-eq$true-and$record.reporting_nll_backpropagated-eq$false) 'Every objective contract matches current typed producer'}
$expectedEstablishment='Full trust including exact initial native20:40 DATA and whole-statistic pairings against saved V11, and independent training/quality audits, establishes a measured candidate. Advancement additionally requires full coverage and all22 declared numeric conditions. Winner and application selection remain separate and false.'
$report=Get-Content -LiteralPath (Join-Path $root 'code/research/causal_fresh_v12/evaluation/report.py') -Raw
$literal=[regex]::Match($report,'(?s)ESTABLISHMENT_CONTRACT = \((?<body>.*?)\)\r?\n')
$reportEstablishment=(@([regex]::Matches($literal.Groups['body'].Value,"'([^']*)'")|ForEach-Object{$_.Groups[1].Value})-join'')
Check ($literal.Success-and$reportEstablishment-ceq$expectedEstablishment-and$card.criteria.establishment-ceq$expectedEstablishment-and$report.Contains('validate_card_contract(card)')) 'Exact reporter/card literal contract before expensive scientific execution'
Check ($card.initial_pairing.references.Count-eq13-and$card.initial_pairing.reference-eq'CPB-v11.market-01 initial0 DATA/statistic artifacts only'-and($card.initial_pairing.native_coordinate_slice-join',')-eq'20,40'-and-not$card.initial_pairing.content20_equality_required-and-not$card.initial_pairing.whole_native40_equality_required-and-not$card.initial_pairing.initial_scores_reused) 'Partial native20:40 eight saved DATA archives and five fresh-stat archives; new initial heads'
Check ($card.training.updates-eq2048-and$card.training.batch_size-eq8-and$card.training.learning_rate-eq0.001-and$card.training.weight_decay-eq0.0001-and$card.training.initialization_seed-eq943001-and$card.training.row_seed-eq943002-and$card.training.mask_seed-eq943003-and$card.training.context_seed-eq943004) 'One unchanged fresh trajectory'
Check ($card.mechanism.all_learned_values-eq294752-and$card.mechanism.all_learned_tensors-eq76-and$card.mechanism.mandatory_identifiable_learned_tensors-eq75-and$card.mechanism.data_gradient_gauge_exemption-eq'core.encoder.level.base.backbone.pool_score.bias'-and-not$card.mechanism.checkpoint_import_for_initialization) 'Same fresh capacity, explicit mathematical gauge'
Check ($card.mechanism.mandatory_identifiable_tensor_value_capacity-eq294751-and$card.mechanism.mandatory_identifiable_count_semantics-eq'tensor numel capacity, not every-scalar gradient/movement proof'-and$card.mechanism.per_scalar_nonzero_gradient_proof_claimed-eq$false) 'No every-scalar capacity overclaim'
foreach($p in @($card.input_pins)+@($card.historical_preservation_pins)){Verify $p $p.repository}
Verify $card.input_manifest
foreach($p in @($card.saved_TRAIN_diagnosis.card,$card.saved_TRAIN_diagnosis.result,$card.saved_TRAIN_diagnosis.independent_audit,$card.saved_TRAIN_diagnosis.human_report,$card.saved_TRAIN_diagnosis.human_report_root_review,$card.saved_TRAIN_diagnosis.completion,$card.pre_execution_preservation,$card.baseline_preservation,$card.comparison_control.v3,$card.comparison_control.v4,$card.comparison_control.v5,$card.comparison_control.v6,$card.comparison_control.v7,$card.comparison_control.v8,$card.comparison_control.v9,$card.comparison_control.v10,$card.comparison_control.v11,$card.comparison_control.v6_establishment,$card.comparison_control.v7_establishment,$card.comparison_control.v8_establishment,$card.comparison_control.v9_establishment,$card.comparison_control.v10_establishment,$card.comparison_control.v11_establishment,$card.advancement.goal_protocol,$card.retention_independent_review)){Verify $p}
foreach($p in @($card.historical_V9_reporting_recovery.card,$card.historical_V9_reporting_recovery.receipt,$card.historical_V9_reporting_recovery.failure_mirror)){Verify $p}
Verify $card.declaration_input_plan
$inputPlan=Read $card.declaration_input_plan.path
Check ($card.reviewed_orchestration_provenance.path-eq'doc/causal_fresh_v12/orchestration-provenance.json') 'Exact reviewed orchestration provenance role'
Verify $card.reviewed_orchestration_provenance
$orchestration=Read $card.reviewed_orchestration_provenance.path
Check ($orchestration.kind-eq'causal_fresh_v12_complete_owned_orchestration_provenance_v1'-and$orchestration.files.Count-eq10) 'Reviewed complete ten-helper provenance'
foreach($binding in $orchestration.files){Verify $binding.parent;Verify $binding.owned}
Verify $orchestration.card_markdown_template
Check ($card.owned_descriptor_draft.path-eq'doc/causal_fresh_v12/CPB-v12.descriptor-draft.json'-and($card.owned_descriptor_draft|ConvertTo-Json -Depth 8 -Compress)-ceq($inputPlan.descriptor_source|ConvertTo-Json -Depth 8 -Compress)) 'Unchanged complete owned draft remains pinned after canonical descriptor creation'
Verify $card.owned_descriptor_draft
$draftDescriptor=Read $card.owned_descriptor_draft.path
$canonicalDescriptor=Read 'code/encoders/causal_patch_bottleneck_mae/versions/CPB-v12.json'
$derivedFields=@('kind','family','stage','immutable_prospective_semantics','created_utc','created_before_build_training_or_quality','created_after_measurement','card','canonical_descriptor_pending_card_freeze','declaration_provenance')
$expectedKeys=@(@($draftDescriptor.PSObject.Properties.Name)+$derivedFields|Sort-Object -Unique)
Check (-not(Compare-Object $expectedKeys @($canonicalDescriptor.PSObject.Properties.Name|Sort-Object))) 'Canonical descriptor contains exactly the owned draft plus declared freeze fields'
foreach($name in $draftDescriptor.PSObject.Properties.Name){if($name-notin$derivedFields){Check (($canonicalDescriptor.$name|ConvertTo-Json -Depth 30 -Compress)-ceq($draftDescriptor.$name|ConvertTo-Json -Depth 30 -Compress)) ('Canonical descriptor retains reviewed owned field: '+$name)}}
$currentCardPin=Pin 'doc/CAUSAL_FRESH_V12_CARD.json'
Check ($canonicalDescriptor.kind-eq'causal_encoder_prospective_version_descriptor_v1'-and$canonicalDescriptor.family-eq'causal_patch_bottleneck_mae'-and$canonicalDescriptor.stage-eq$card.stage-and$canonicalDescriptor.immutable_prospective_semantics-eq$true-and$canonicalDescriptor.created_utc-eq$card.frozen_utc-and$canonicalDescriptor.created_before_build_training_or_quality-eq$true-and$canonicalDescriptor.created_after_measurement-eq$false-and$canonicalDescriptor.canonical_descriptor_pending_card_freeze-eq$false) 'Canonical descriptor has exact prospective freeze semantics'
Check ($canonicalDescriptor.card.path-eq$currentCardPin.path-and$canonicalDescriptor.card.bytes-eq$currentCardPin.bytes-and$canonicalDescriptor.card.sha256-eq$currentCardPin.sha256-and$canonicalDescriptor.declaration_provenance-eq'Prospective immutable descriptor pins the frozen card;card names descriptor path without its hash,avoiding a cycle. No measured results.') 'Canonical descriptor pins this exact frozen card without a hash cycle'
Verify $card.independent_training_auditor_source_review
$trainingReview=Read $card.independent_training_auditor_source_review.path
Check ($trainingReview.status-eq'passed'-and$trainingReview.kind-eq'cpb_v12_family_and_training_auditor_independent_static_review_v1') 'Existing independent source-review authority retained'
foreach($p in $card.source_pins_after_owner_freeze){Verify $p}
Check ($card.historical_V9_reporting_recovery.card.sha256-eq'9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'-and$card.historical_V9_reporting_recovery.receipt.sha256-eq'5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72') 'Historical V9 report-only recovery retains its exact meaning'
$audit=Read $card.saved_TRAIN_diagnosis.independent_audit.path
Check ($audit.kind-eq'cpb_v9_v11_TRAIN_diagnosis_independent_audit_v1'-and$audit.status-eq'passed'-and$audit.checks-eq68264-and$audit.mismatches.Count-eq0-and$audit.old_V9_raw_numeric_reads-eq0-and$audit.local_adjoint_rechecks-eq0-and($audit.computed_versions-join',')-eq'v11'-and($audit.cached_reference_versions-join',')-eq'v9'-and$card.saved_TRAIN_diagnosis.result.sha256-eq'62f80b93ef4ad7b2a2a994832ab3f2f6921e09f37e96aca441ec283064f3b742'-and$card.saved_TRAIN_diagnosis.independent_audit.sha256-eq'70697acc72fd010b48431516cb15dc1537b615cde5835ed90536cd4de625bc8c'-and$card.saved_TRAIN_diagnosis.completion.sha256-eq'62f63261567763a25ee7d55801156e2568b783432039a576e34fd19929e21f9d') 'Exact completed independently audited V11-only TRAIN arithmetic with opaque cached V9;no old finite-difference/trust replay'
$protocol=Read $card.advancement.goal_protocol.path
Check ($card.advancement.goal_protocol.sha256-eq'866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'-and$card.advancement.criterion_count-eq22-and-not$protocol.additional_train_NLL_gate) 'Protect all V3–V11 scoped strengths;no TRAIN-NLL gate'
Check (($card.advancement.criteria|ConvertTo-Json -Depth 30 -Compress)-ceq($protocol.criteria|ConvertTo-Json -Depth 30 -Compress)) 'Exact prospective conditions'
$review=Read $card.retention_independent_review.path
Check ($review.status-eq'passed') 'Independent retention review passed'
Verify $card.retention_application_admission
Check ($card.retention_application_admission.sha256-eq'e59f8eeaa4993ec116eca2751c2cbb9a957c6ae81b95047917b71e01485f3035') 'Exact applied future floor receipt'
$admission=Read 'doc/causal_fresh_v12/goal-protocol-admission.json'
Check ($admission.status-eq'passed'-and$admission.prospective_future_use_eligible-and$admission.criteria-eq22-and-not$admission.historical_results_or_counts_recomputed) 'Applied future floor admission'
Check ($card.baseline_frozen_files.Count-eq47) '47 selected baseline files per repository'
foreach($p in $card.baseline_frozen_files){foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){Verify $p $repo}}
$preservation=Read $card.baseline_preservation.path
Check ($preservation.status-eq'passed'-and$preservation.critical_checkpoint_pins.Count-eq109) '109 protected checkpoint files;no uncreated V12 artifacts counted'
foreach($p in $preservation.critical_checkpoint_pins){
 $repo=$p.repository
 if(-not$repo){Check ($p.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')) 'Only original five V5 pins lack repository';$repo='cuwacunu_embedding'}
 Verify $p $repo
}
foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){
 $paths=@($preservation.critical_checkpoint_pins|Where-Object{($_.repository-eq$repo)-or(-not$_.repository-and$repo-eq'cuwacunu_embedding')}|ForEach-Object{(Resolve-Pin $_ $repo).Substring((Join-Path $workspace $repo).Length+1).Replace([char]92,[char]47)})
 $ignored=@(& git -c ('safe.directory='+($workspace+'/'+$repo).Replace([char]92,[char]47)) -C (Join-Path $workspace $repo) check-ignore -- @paths)
 Check ($LASTEXITCODE-eq0-and$ignored.Count-eq$paths.Count) ('All critical checkpoints Git ignored: '+$repo)
}
$old=Read 'doc/causal_fresh_v9/protocol-preflight.json'
foreach($p in $old.sources){Verify $p}
$latestSource=$card.latest_frozen_source_preservation
$latestRoles=[ordered]@{card='doc/CAUSAL_FRESH_V11_CARD.json';manifest='output/engineering/causal-fresh-v11/attempt-001/source-manifest.json';archive='output/engineering/causal-fresh-v11/attempt-001/source.zip';protocol_preflight='doc/causal_fresh_v11/protocol-preflight.json'}
foreach($role in $latestRoles.Keys){$binding=$latestSource.$role;Check ($null-ne$binding-and$binding.path-ceq$latestRoles[$role]) ('Exact latest V11 source role: '+$role);Verify $binding}
$latestCard=Read $latestSource.card.path
$latestManifest=Read $latestSource.manifest.path
$latestPreflight=Read $latestSource.protocol_preflight.path
$latestCount=$latestCard.source_counts.capsule_sources
Check (($latestCount-is[int]-or$latestCount-is[long])-and$latestCount-gt0-and$latestCount-eq$preservation.V11_source_files-and$latestManifest.files.Count-eq$latestCount-and@($latestManifest.files.path|Sort-Object -Unique).Count-eq$latestCount-and($latestPreflight.sources.Count+1)-eq$latestCount-and$latestSource.source_files-eq$latestCount) 'Latest V11 source count derives from frozen card/final preservation/preflight/manifest'
Check ($latestCard.model-eq'CPB-v11'-and-not$latestCard.stage.Contains('DRAFT')-and$latestCard.frozen_utc-and$latestManifest.card_sha256-eq$latestSource.card.sha256-and$latestPreflight.status-eq'passed'-and$latestPreflight.card_sha256-eq$latestSource.card.sha256) 'Latest frozen V11 declaration and source identities'
$latestArchives=@($preservation.source_ZIPs|Where-Object{$_.manifest.path-eq$latestSource.manifest.path})
Check ($latestArchives.Count-eq1-and$latestArchives[0].source_files-eq$latestCount-and$latestArchives[0].manifest.sha256-eq$latestSource.manifest.sha256-and$latestArchives[0].archive.sha256-eq$latestSource.archive.sha256) 'Latest V11 whole ZIP and manifest equal independently preserved bytes'
$latestPathsExpected=@($latestPreflight.sources.path)+@($latestSource.protocol_preflight.path)
Check (-not(Compare-Object (@($latestManifest.files.path)|Sort-Object) ($latestPathsExpected|Sort-Object))) 'Exact latest V11 preflight-plus-receipt path set'
foreach($binding in $latestManifest.files){Verify $binding}
$family=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v12') -Recurse -File)
$research=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v12') -Recurse -File)
Check ($null-ne$card.source_counts.family_inside-and$null-ne$card.source_counts.research-and$family.Count-eq$card.source_counts.family_inside-and$research.Count-eq$card.source_counts.research) 'Actual final owner family/research counts required'
foreach($f in @($family)+@($research)){
 Check (-not($f.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'Physical source'
 if($f.Extension-in@('.h','.cpp','.mk','.py')-or$f.Name-eq'Makefile'){
  $s=Get-Content -LiteralPath $f.FullName -Raw
  Check ($s-notmatch'(?m)^\s*#\s*include[^\r\n]*(?:versions/v(?:[1-9]|1[01])(?:/|\.h)|detail/v(?:[1-9]|1[01])/|raw_patch_bottleneck_mae)') ('No historical implementation includes: '+$f.Name)
 }
}
$pool=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v12/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v12/temporal_pooling.h') -Raw
Check ($pool.Contains('pool_original_position_bins')-and$pool.Contains('finite_fallback')-and$pool.Contains('projection_weight.narrow')-and$pool.Contains('torch::where(present.unsqueeze(1),projected')-and$pool.Contains('global_weights')-and$pool.Contains('kTemporalPoolingLocalMix=.5')-and$pool.Contains('kTemporalPoolingGlobalMix=.5')) 'Owned masked safe fixed half-mixture pooling moment kernel'
Check ($pool.Contains('centered.square()')-and$pool.Contains('kTemporalMomentPolicy')-and$pool.Contains('kTemporalExportLayout')) 'Actual centered variance and split export policy'
Check ($pool.Contains('local_uniform')-and$pool.Contains('global_uniform')-and$pool.Contains('dispersion_mixture')-and$pool.Contains('safe_hidden-uniform_pooled.unsqueeze(1)')) 'Uniform mask/count support and its own center in actual kernel'
foreach($field in @('attention_mean_and_uniform_dispersion_full_expression_adjoints','uniform_dispersion_own_center','uniform_dispersion_score_independent','uniform_dispersion_hidden_gradient_live','attention_mean_score_gradient_live','dual_support_actual_trace')){Check ($card.temporal_pooling_proof_contract.required_proof_fields.$field-is[bool]-and$card.temporal_pooling_proof_contract.required_proof_fields.$field) ('Required new prospective temporal proof: '+$field)}
Check ($card.temporal_pooling_proof_contract.required_proof_fields.direct_pool_kernel_calls-eq47-and$card.temporal_pooling_proof_contract.required_proof_fields.F64_finite_difference_coordinates-eq18-and$card.temporal_pooling_proof_contract.required_proof_fields.F32_finite_difference_coordinates-eq18) 'Source-derived47 kernel calls and18/18 finite-difference coordinates'
Check ($card.actual_step_temporal_witness_contract.total_fields-eq60-and$card.temporal_learning_receipt_contract.moment_subgroup_rows-eq16384) 'Actual expanded same-forward evidence declaration'
$train=Get-Content -LiteralPath (Join-Path $root 'code/research/causal_fresh_v12/train_main.cpp') -Raw
Check ($train.Contains('clip_grad_norm_(')-and$train.Contains('total_loss.backward()')-and$train.Contains('typed-resume-512.json')-and$train.Contains('pool_score.bias')) 'Actual global clip, typed continuation, explicit gauge'
Check ($train.Contains('temporal-between-bin-position-learning.csv')-and$train.Contains('hadamard-full16-signs-div4-v1')-and$train.Contains('mandatory_identifiable_tensor_value_capacity')-and$train.Contains('per_scalar_nonzero_gradient_proof_claimed')) 'Observed between-bin evidence and honest tensor capacity labels'
Check ($train.Contains('variance_NLL_F64')-and$train.Contains('row_variance_nll')-and$train.Contains('direct_gradient_routes')-and-not$train.Contains('row_joint_gaussian_nll')) 'Unchanged V11 component/direct-route names agree with declaration'
foreach($field in @('temporal_pool_dispersion_weights','temporal_contextual_pool_dispersion_weights','temporal_pool_dispersion_mean','temporal_contextual_pool_dispersion_mean')){Check ($train.Contains($field)) ('Same-forward uniform trace captured: '+$field)}
$docs=@('LICENSE','doc/CAUSAL_FRESH_V12_CARD.json','doc/CAUSAL_FRESH_V12_CARD.md','code/encoders/causal_patch_bottleneck_mae/versions/CPB-v12.json','doc/CAUSAL_V9_V11_TRAIN_DIAGNOSIS_CARD.ADMITTED.json','doc/CAUSAL_V9_V11_TRAIN_DIAGNOSIS.md','doc/causal_v9_v11_train_diagnosis/diagnosis.json','doc/causal_v9_v11_train_diagnosis/independent-audit.json','doc/causal_v9_v11_train_diagnosis/completion.json','doc/causal_v9_v11_train_diagnosis/human-report-root-review-001.json','doc/causal_v9_v11_train_diagnosis/human-report-independent-review-001.json','doc/causal_fresh_v11/final-preservation.json','doc/CAUSAL_FRESH_V11_CARD.json','doc/causal_fresh_v11/protocol-preflight.json','doc/causal_fresh_v11/complete-mirror-attempt-001.json','doc/causal_fresh_v12/design-source-plan.ADMITTED-001.json')
$controls=@('preflight.ps1','stage-source.ps1','unpack-source.py','audit-build.py','execute-stage.py','run-quality.py','verify-mirror.ps1','run-container.ps1','prepare-draft-card.ps1','freeze-card.ps1','input-manifest.json','raw-controls.json','draft-metadata-copy-receipt.json','preservation-before.json','orchestration-provenance.json','family-source-provenance.json','evaluation-source-provenance.json','card-markdown-template.md','baseline-control-v3.json','baseline-control-v4.json','baseline-control-v5.json','baseline-control-v6.json','baseline-control-v7.json','baseline-control-v8.json','baseline-control-v9.json','baseline-control-v10.json','baseline-control-v11.json','baseline-v6-establishment.json','baseline-v7-establishment.json','baseline-v8-establishment.json','baseline-v9-establishment.json','baseline-v10-establishment.json','baseline-v11-establishment.json','goal-protocol.json','goal-protocol-review.json','goal-protocol-admission.json')
$docs+=$card.declaration_input_plan.path
$docs+=$card.independent_training_auditor_source_review.path
$all=@($family)+@($research)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v12.h'))+@($docs|ForEach-Object{Get-Item -LiteralPath (Join-Path $root $_)})+@($controls|ForEach-Object{Get-Item -LiteralPath (Join-Path $PSScriptRoot $_)})
$sources=@($all|Sort-Object FullName -Unique|ForEach-Object{Pin ($_.FullName.Substring($root.Length+1).Replace([char]92,[char]47))})
Check ($sources.Count-eq$all.Count-and($sources.Count+1)-eq$card.source_counts.capsule_sources) 'Unique actual reviewed paths plus generated preflight equal frozen capsule count'
$receipt=[ordered]@{kind='causal_fresh_v12_source_protocol_preflight_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');card_sha256=$CardSHA256;checks=$script:checks;family_files=$family.Count;extra_full_owned_public_header=1;owned_research_files=$research.Count;critical_checkpoints_verified=109;baseline_sources_verified_each_repo=47;old_v9_sources_verified=$old.sources.Count;latest_v11_sources_verified=$latestManifest.files.Count;latest_v11_source_manifest=$latestSource.manifest;latest_v11_source_zip=$latestSource.archive;sources=$sources;scope='Host file/hash/source/protocol inspection only;no build,test,model,numerical execution';goal_active=$true;application_changed=$false}
$receipt|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $destination -Encoding utf8NoBOM
[ordered]@{status='passed';checks=$script:checks;sources=$sources.Count;receipt=(Pin 'doc/causal_fresh_v12/protocol-preflight.json')}|ConvertTo-Json -Depth 4

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
$snapshotRelative='output/engineering/causal-fresh-v11-protocol-preparation/attempt-001/before-prospective-freeze'
$snapshot=Join-Path $root $snapshotRelative
$canonicalRelative='code/encoders/causal_patch_bottleneck_mae/versions/CPB-v11.json'
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
$cardRelative='doc/CAUSAL_FRESH_V11_CARD.json';$markdownRelative='doc/CAUSAL_FRESH_V11_CARD.md'
$cardPin=Get-Pin $cardRelative;$markdownPin=Get-Pin $markdownRelative
Check ($cardPin.sha256-eq$DraftCardSHA256-and$markdownPin.sha256-eq$DraftMarkdownSHA256) 'Exact root-reviewed draft bytes'
$orchestrationPin=Get-Pin 'doc/causal_fresh_v11/orchestration-provenance.json'
Check ($orchestrationPin.sha256-eq$OrchestrationProvenanceSHA256) 'Exact reviewed ten-helper derivation'
$card=Read $cardRelative;$orchestration=Read $orchestrationPin.path
Check ($card.stage-eq'DRAFT_NOT_FROZEN_NOT_EXECUTION_ADMITTED'-and-not$card.execution_admitted-and$null-eq$card.frozen_utc) 'Only the unmeasured prospective draft may freeze'
Check ($card.instance-eq'CPB-v11.market-01'-and$card.model-eq'CPB-v11'-and$card.geometry.native_width-eq40-and$card.baseline-eq'CPB-v3.market-01'-and$card.baseline_remains_selected) 'Separately named native40, selected baseline unchanged'
$policy='original-position-four-bin-half-mixture-mean4-centered-variance4-v1';$empty='zero-whole-export-slice-after-bias-v1'
foreach($record in @($card.geometry,$card.mechanism,$card.training,$card.trust_receipt_objective_fields,$card.temporal_pooling_proof_contract,$card.temporal_learning_receipt_contract)){
 Check ($record.temporal_pooling_policy-eq$policy-and($record.temporal_pooling_partition-join',')-eq'4,16,32,4,8'-and($record.temporal_pooling_mix-join',')-eq'0.5,0.5'-and$record.empty_local_bin_policy-eq$empty) 'Every current temporal declaration agrees'
 Check ($record.temporal_moment_policy-eq'componentwise-centered-variance-two-sum-mean-live-attention-v1'-and$record.temporal_export_layout-eq'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'-and($record.temporal_moment_partition-join',')-eq'4,4') 'Every current moment/layout declaration agrees'
}
Check ($card.mechanism.source_updates-eq0-and$card.mechanism.all_learned_values-eq294752-and$card.mechanism.all_learned_tensors-eq76-and$card.mechanism.mandatory_identifiable_learned_tensors-eq75-and-not$card.mechanism.checkpoint_import_for_initialization-and$card.mechanism.fresh_optimizer) 'Fresh learned/optimizer ancestry and sole scalar gauge exemption'
Check (-not$card.mechanism.Contains('mandatory_identifiable_learned_values')-and$card.mechanism.mandatory_identifiable_tensor_value_capacity-eq294751-and$card.mechanism.mandatory_identifiable_count_semantics-eq'tensor numel capacity, not every-scalar gradient/movement proof'-and$card.mechanism.per_scalar_nonzero_gradient_proof_claimed-eq$false) 'Tensor numel capacity is not every-scalar learning proof'
Check ($card.training.updates-eq2048-and$card.training.batch_size-eq8-and$card.training.global_gradient_clip-eq1-and$card.training.learning_rate-eq0.001-and$card.training.weight_decay-eq0.0001-and$card.training.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and$card.training.gradient_policy-eq'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'-and($card.training.loss_coefficients-join',')-eq'1,1') 'Unchanged objective/shared globalclip/single trajectory budget'
Check ($card.training.initialization_seed-eq943001-and$card.training.row_seed-eq943002-and$card.training.mask_seed-eq943003-and$card.training.context_seed-eq943004) 'Unchanged four seeds'
$routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
Check ($card.mechanism.kind-eq'original-position-four-bin-half-mixture-mean4-centered-variance4-v1'-and$card.training.encoder_gradient_kind-eq'Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1') 'Distinct pooling identity and restored V9 gradient-kind label'
foreach($record in @($card.mechanism,$card.training,$card.trust_receipt_objective_fields)){
 Check ($record.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and$record.gradient_policy-eq'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'-and$record.direct_gradient_routes-eq$routes-and($record.loss_coefficients-join',')-eq'1,1'-and$record.variance_error_detached-eq$true-and$record.reporting_nll_backpropagated-eq$false) 'All objective/direct-route/detach/reporting declarations agree'
}
$establishment='Full trust including exact initial native20:40 DATA and whole-statistic pairings against saved V9, and independent training/quality audits, establishes a measured candidate. Advancement additionally requires full coverage and all22 declared numeric conditions. Winner and application selection remain separate and false.'
$report=Get-Content -LiteralPath (Join-Path $root 'code/research/causal_fresh_v11/evaluation/report.py') -Raw
$literal=[regex]::Match($report,'(?s)ESTABLISHMENT_CONTRACT = \((?<body>.*?)\)\r?\n')
$reportLiteral=(@([regex]::Matches($literal.Groups['body'].Value,"'([^']*)'")|ForEach-Object{$_.Groups[1].Value})-join'')
Check ($literal.Success-and$reportLiteral-ceq$establishment-and$card.criteria.establishment-ceq$establishment-and$report.Contains('validate_card_contract(card)')) 'Literal reporter/card contract checked before any expensive phase'
Check ($card.mechanism_proof_contract.direct_NLL_mean_gradient_zero-eq$true) 'Restored V9 detached-error variance direct mean-gradient zero'
foreach($field in @('direct_joint_NLL_mean_gradient_live','direct_joint_NLL_uncertainty_gradient_live','joint_Gaussian_variance_counted_once','total_equals_Huber_plus_one_joint_Gaussian','independent_composite_F64_finite_differences')){Check (-not$card.mechanism_proof_contract.Contains($field)) ('No V10 joint objective proof inherited: '+$field)}
Check ($card.mechanism_proof_contract.log_tokens-contains'detached-error NLL mean gradient zero') 'Actual V9 partial-gradient fixture token'
Check ($card.temporal_learning_receipt_contract.moment_subgroup_rows-eq16384-and$card.temporal_learning_receipt_contract.moment_subgroups['count']-eq8-and$card.actual_step_temporal_witness_contract.total_fields-eq56) 'Declared eight moment subgroups and expanded same-forward actual witness'
Check ($card.initial_pairing.references.Count-eq13-and$card.initial_pairing.reference-eq'CPB-v9.market-01 initial0 DATA/statistic artifacts only'-and($card.initial_pairing.native_coordinate_slice-join',')-eq'20,40'-and-not$card.initial_pairing.content20_equality_required-and-not$card.initial_pairing.whole_native40_equality_required-and-not$card.initial_pairing.initial_scores_reused) 'Partial native20:40 eight DATA archives plus whole five statistics; new own initial heads'
foreach($reference in $card.initial_pairing.references){Check ($reference.reference_data_archive.path.StartsWith('/opt/cuwacunu/research/causal-fresh-v9/attempt-001/')-and$reference.candidate_relative_to_attempt-notmatch'cpb_v9-') 'Initial pairing references saved V9 DATA/statistics only';if($reference.Contains('feature_value_coordinate_slice')){Check (($reference.feature_value_coordinate_slice-join',')-eq'20,40') 'Every feature matrix compares20:40 only'}}
$goal=Read $card.advancement.goal_protocol.path
Verify $card.advancement.goal_protocol
Check ($card.advancement.goal_protocol.sha256-eq'221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144'-and$card.advancement.criterion_count-eq22-and$goal.additional_train_NLL_gate-eq$false) 'Exact future22 protocol and no TRAIN-NLL gate'
Check (($card.advancement.criteria|ConvertTo-Json -Depth 25 -Compress)-ceq($goal.criteria|ConvertTo-Json -Depth 25 -Compress)) 'All22 criterion objects copied literally without reductions'
Verify $card.retention_independent_review;Verify $card.retention_application_admission
Check ((Read $card.retention_independent_review.path).status-eq'passed'-and(Read $card.retention_application_admission.path).prospective_future_use_eligible) 'Independently reviewed applied future protocol'
foreach($key in @('v3','v4','v5','v6','v7','v8','v9','v10','v6_establishment','v7_establishment','v8_establishment','v9_establishment','v10_establishment')){Verify $card.comparison_control[$key]}
Check (($card.scoped_references-join',')-eq'CPB-v4.market-01,CPB-v5.market-01,CPB-v6.market-01,CPB-v7.market-01,CPB-v8.market-01,CPB-v9.market-01,CPB-v10.market-01'-and$card.scoped_reference-eq'CPB-v10.market-01') 'Eight historical controls;latest measured reference V10,scientific/pairing parent V9'
Verify $card.input_manifest
foreach($pin in $card.input_pins){Verify $pin $pin.repository}
foreach($key in @('card','result','independent_audit','human_report','human_report_root_review','completion')){Verify $card.saved_TRAIN_diagnosis[$key]}
Check ($card.saved_TRAIN_diagnosis.result.sha256-eq'805d80db8f260e59a86b1b0184fe54fb3e28659e642f09f6e4e8ab99aa2a36cf'-and$card.saved_TRAIN_diagnosis.independent_audit.sha256-eq'c0c8d5c93e95d24c8a79f3e1d6a94b597a4ac853407c38aaeea27235715d8da9'-and$card.saved_TRAIN_diagnosis.human_report.sha256-eq'd9c47cc022ee64900ede794a87e363997474b4bfb08ad07a03278f65b6a80b61'-and$card.saved_TRAIN_diagnosis.completion.sha256-eq'6ec7076b32422a0c37ae8d64c8ad2056ede4279639d46f520782a5a8fdfec83d') 'Completed independently audited V8/V9 diagnosis anchors'
foreach($key in @('card','receipt','failure_mirror')){Verify $card.historical_V9_reporting_recovery[$key]}
Verify $card.declaration_input_plan
Check ($card.historical_V9_reporting_recovery.card.sha256-eq'9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'-and$card.historical_V9_reporting_recovery.receipt.sha256-eq'5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72') 'Historical V9 recovery has no new scientific meaning'
Verify $card.baseline_preservation;Verify $card.pre_execution_preservation
$preservation=Read $card.baseline_preservation.path
Check ($preservation.status-eq'passed'-and$preservation.critical_checkpoint_pins.Count-eq104-and$preservation.selected_V3_checkpoint_sha256-eq'50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e') '104 existing critical checkpoints and selected V3 baseline'
$latestSource=$card.latest_frozen_source_preservation
$latestRoles=[ordered]@{card='doc/CAUSAL_FRESH_V10_CARD.json';manifest='output/engineering/causal-fresh-v10/attempt-001/source-manifest.json';archive='output/engineering/causal-fresh-v10/attempt-001/source.zip';protocol_preflight='doc/causal_fresh_v10/protocol-preflight.json'}
foreach($role in $latestRoles.Keys){$binding=$latestSource.$role;Check ($null-ne$binding-and$binding.path-ceq$latestRoles[$role]) ('Exact latest V10 source role: '+$role);Verify $binding}
$latestCard=Read $latestSource.card.path
$latestManifest=Read $latestSource.manifest.path
$latestPreflight=Read $latestSource.protocol_preflight.path
$latestCount=$latestCard.source_counts.capsule_sources
Check (($latestCount-is[int]-or$latestCount-is[long])-and$latestCount-gt0-and$latestCount-eq$preservation.V10_source_files-and$latestManifest.files.Count-eq$latestCount-and@($latestManifest.files.path|Sort-Object -Unique).Count-eq$latestCount-and($latestPreflight.sources.Count+1)-eq$latestCount-and$latestSource.source_files-eq$latestCount) 'Latest V10 source count derives from frozen card/final preservation/preflight/manifest'
Check ($latestCard.model-eq'CPB-v10'-and-not$latestCard.stage.Contains('DRAFT')-and$latestCard.frozen_utc-and$latestManifest.card_sha256-eq$latestSource.card.sha256-and$latestPreflight.status-eq'passed'-and$latestPreflight.card_sha256-eq$latestSource.card.sha256) 'Latest frozen V10 declaration and source identities'
$latestArchives=@($preservation.source_ZIPs|Where-Object{$_.manifest.path-eq$latestSource.manifest.path})
Check ($latestArchives.Count-eq1-and$latestArchives[0].source_files-eq$latestCount-and$latestArchives[0].manifest.sha256-eq$latestSource.manifest.sha256-and$latestArchives[0].archive.sha256-eq$latestSource.archive.sha256) 'Latest V10 whole ZIP and manifest equal independently preserved bytes'
$latestPathsExpected=@($latestPreflight.sources.path)+@($latestSource.protocol_preflight.path)
Check (-not(Compare-Object (@($latestManifest.files.path)|Sort-Object) ($latestPathsExpected|Sort-Object))) 'Exact latest V10 preflight-plus-receipt path set'
foreach($binding in $latestManifest.files){Verify $binding}
foreach($pin in $preservation.critical_checkpoint_pins){$repository=$pin.repository;if(-not$repository){Check ($pin.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')) 'Only archived five V5 pins may omit repository';$repository='cuwacunu_embedding'};Verify $pin $repository}
Check ($card.baseline_frozen_files.Count-eq47) 'Selected V3 source inventory'
foreach($pin in $card.baseline_frozen_files){Verify $pin 'cuwacunu_embedding';Verify $pin 'cuwacunu_dev'}
$family=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v11') -Recurse -File)
$research=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v11') -Recurse -File)
Check ($family.Count-eq$card.source_counts.family_inside-and$research.Count-eq$card.source_counts.research-and$card.source_counts.family_external_header-eq1) 'Actual final owner family/research counts agree with bound complete path set'
Check ($card.source_pins_after_owner_freeze.Count-eq($family.Count+$research.Count+1)) 'All actual owner source pins'
foreach($pin in $card.source_pins_after_owner_freeze){Verify $pin}
$observed=@(@($family)+@($research)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v11.h'))|ForEach-Object{$_.FullName.Substring($root.Length+1).Replace([char]92,[char]47)}|Sort-Object)
Check (-not(Compare-Object $observed @($card.source_pins_after_owner_freeze.path|Sort-Object))) 'Exact actual owner path set, not merely count'
foreach($binding in $orchestration.files){Verify $binding.parent;Verify $binding.owned}
Check ($orchestration.files.Count-eq10) 'Ten complete owned orchestration helpers'
$checkpoint=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v11/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v11/location_scale_decoder_checkpoint.h') -Raw
foreach($field in @('temporal_pooling_policy','temporal_pooling_partition_v1','temporal_pooling_mix_f64','empty_local_bin_policy','temporal_moment_policy','temporal_export_layout','temporal_moment_partition_v1')){Check ($checkpoint.Contains($field)) ('Typed write/read field: '+$field)}
Check ($checkpoint.Contains('torch::kFloat64,2')-and$checkpoint.Contains('torch::kInt64,5')) 'Typed dimensions/dtypes remain explicit'
$public=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v11.h') -Raw
Check (([regex]::Matches($public,'write_temporal_pooling').Count)-ge2-and([regex]::Matches($public,'read_temporal_pooling').Count)-ge3) 'Static typed outer/TRAIN gateways call the owned validator'
Verify $card.independent_training_auditor_source_review
$trainingReview=Read $card.independent_training_auditor_source_review.path
Check ($trainingReview.status-eq'passed'-and$trainingReview.kind-eq'cpb_v11_family_and_training_auditor_independent_static_review_v1') 'Actual existing independent training-auditor source review complete'
# Identical path declarations to preflight; generated descriptor/preflight do not exist yet.
$docs=@('LICENSE',$cardRelative,$markdownRelative,$canonicalRelative,'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md','doc/causal_v8_v9_train_diagnosis/diagnosis.json','doc/causal_v8_v9_train_diagnosis/independent-audit.json','doc/causal_v8_v9_train_diagnosis/completion.json','doc/causal_fresh_v10/final-preservation.json','doc/CAUSAL_FRESH_V10_CARD.json','doc/causal_fresh_v10/protocol-preflight.json','doc/causal_fresh_v10/complete-mirror-attempt-001.json')
$controls=@('preflight.ps1','stage-source.ps1','unpack-source.py','audit-build.py','execute-stage.py','run-quality.py','verify-mirror.ps1','run-container.ps1','prepare-draft-card.ps1','freeze-card.ps1','input-manifest.json','raw-controls.json','draft-metadata-copy-receipt.json','preservation-before.json','orchestration-provenance.json','family-source-provenance.json','evaluation-source-provenance.json','card-markdown-template.md','baseline-control-v3.json','baseline-control-v4.json','baseline-control-v5.json','baseline-control-v6.json','baseline-control-v7.json','baseline-control-v8.json','baseline-control-v9.json','baseline-control-v10.json','baseline-v6-establishment.json','baseline-v7-establishment.json','baseline-v8-establishment.json','baseline-v9-establishment.json','baseline-v10-establishment.json','goal-protocol.json','goal-protocol-review.json','goal-protocol-admission.json')
$docs+=$card.declaration_input_plan.path
$docs+=$card.independent_training_auditor_source_review.path
$paths=@($observed)+$docs+@($controls|ForEach-Object{'doc/causal_fresh_v11/'+$_})
Check (@($paths|Sort-Object -Unique).Count-eq$paths.Count) 'No duplicate prospective capsule paths'
$card.source_counts.capsule_sources=$paths.Count+1
$card.source_counts.status='Prospectively_frozen_actual_path_set_before_execution;actual_build_closure_audited_later'
Check ($card.source_counts.translation_units-eq11-and$card.source_counts.binaries-eq7) 'Final owners static7binaries/11TUs; actual closure remains future build evidence'
$now=[DateTime]::UtcNow.ToString('o')
# Snapshot complete current bytes before the first in-place declaration change.
[IO.Directory]::CreateDirectory($snapshot)|Out-Null
$snapshotPins=@()
foreach($relative in @($cardRelative,$markdownRelative,'doc/causal_fresh_v11/CPB-v11.descriptor-draft.json')){
 $pin=Get-Pin $relative;$copy=$snapshotRelative+'/'+[IO.Path]::GetFileName($relative)
 [IO.File]::Copy((Join-Path $root $relative),(Join-Path $root $copy),$false)
 $actual=Get-Pin $copy;Check ($actual.sha256-eq$pin.sha256-and$actual.bytes-eq$pin.bytes) 'Full pre-freeze bytes preserved'
 $snapshotPins+=[ordered]@{original=$pin;snapshot=$actual}
}
Write-NewJson ($snapshotRelative+'/manifest.json') ([ordered]@{kind='causal_fresh_v11_before_prospective_freeze_v1';created_utc=$now;files=$snapshotPins;model_calls=0;head_fits=0;old_files_modified=$false})
$card.declared_utc=$now;$card.frozen_utc=$now;$card.stage='prospectively_frozen_unmeasured_before_build_training_quality';$card.execution_admitted=$true
$card.source_contract_status='Prospectively frozen after owner and independent source/card/helper/auditor reviews;no AST/build/model/head execution before freeze'
$card.temporal_pooling_proof_contract.status='declared_before_any_execution_fixture_source_contract_agreed'
$card.prospective_freeze=[ordered]@{source=(Get-Pin 'doc/causal_fresh_v11/freeze-card.ps1');prior_draft=$cardPin;prior_markdown=$markdownPin;snapshot=($snapshotRelative+'/manifest.json');root_reviewed=$true;scientific_calls=0;execution_admission='Only the declared exclusive workflow;each phase remains explicitly invoked;no automatic body execution'}
[IO.File]::WriteAllText((Join-Path $root $cardRelative),($card|ConvertTo-Json -Depth 60)+"`n",[Text.UTF8Encoding]::new($false))
$markdown=Get-Content -LiteralPath (Join-Path $root $markdownRelative) -Raw
$introduction="# CPB-v11.market-01 — prospective declaration`n`nRoot reviewed and froze this separately named mean4 plus centered-variance4 pooling hypothesis under the unchanged V9 objective at $now, after the completed saved TRAIN-only V8/V9 diagnosis. The diagnosis selected no model; root selected this separate experiment afterward. V9 half-mixture attention/native40/fresh constructor draws/globalclip1/heads remain fixed; only the intermediate last4 export rows per bin read centered variance instead of mean. Machine card: [CAUSAL_FRESH_V11_CARD.json](CAUSAL_FRESH_V11_CARD.json). Exact source closure: $($card.source_counts.capsule_sources) files including the generated preflight, with $($card.source_counts.family_inside) internal family files plus the full public header and $($card.source_counts.research) research files. The declared $($card.source_counts.binaries) binaries and $($card.source_counts.translation_units) translation units require actual compiler closure admission later. No AST/build/test/model/head body has run. Every future phase requires explicit invocation, hash-bound preflight and the exclusive managed-container workflow. The canonical immutable descriptor pins the frozen card without a cycle. Complete previous draft bytes are preserved in the freeze snapshot.`n`n"
$markdown=[regex]::Replace($markdown,'\A# CPB-v11.*?(?=CPB-v3\.market-01 remains)',$introduction,[Text.RegularExpressions.RegexOptions]::Singleline)
$markdown=$markdown.Replace('Actual build/source/binary/TU counts must be derived and','Actual build/source/binary/TU counts have been derived and').Replace('reviewed from the final owners before admission; this draft does not assume','reviewed from the final owners before admission; the later actual dependency audit must confirm').Replace('with no future artifact count is prefilled','with no future artifact count prefilled')
[IO.File]::WriteAllText((Join-Path $root $markdownRelative),$markdown,[Text.UTF8Encoding]::new($false))
$finalCard=Get-Pin $cardRelative
$descriptor=Read 'doc/causal_fresh_v11/CPB-v11.descriptor-draft.json'
$descriptor.kind='causal_encoder_prospective_version_descriptor_v1';$descriptor.family='causal_patch_bottleneck_mae'
$descriptor.stage='prospectively_frozen_unmeasured_before_build_training_quality';$descriptor.immutable_prospective_semantics=$true
$descriptor.created_utc=$now;$descriptor.created_before_build_training_or_quality=$true;$descriptor.created_after_measurement=$false
$descriptor.card=$finalCard;$descriptor.canonical_descriptor_pending_card_freeze=$false
$descriptor.declaration_provenance='Prospective immutable descriptor pins the frozen card;card names descriptor path without its hash,avoiding a cycle. No measured results.'
Write-NewJson $canonicalRelative $descriptor
$receipt=[ordered]@{kind='causal_fresh_v11_prospective_freeze_v1';status='passed';frozen_utc=$now;checks=$script:checks;card=$finalCard;markdown=(Get-Pin $markdownRelative);descriptor=(Get-Pin $canonicalRelative);snapshot=(Get-Pin ($snapshotRelative+'/manifest.json'));orchestration_provenance=$orchestrationPin;owner_source_files=$observed.Count;reviewed_source_paths_before_preflight=$paths.Count;declared_capsule_sources=$card.source_counts.capsule_sources;critical_checkpoints_verified=104;latest_v10_sources_verified=$latestManifest.files.Count;latest_v10_source_manifest=$latestSource.manifest;latest_v10_source_zip=$latestSource.archive;V3_sources_verified_each_repo=47;criteria_objects_copied_exact=22;AST_calls=0;Docker_calls=0;project_body_calls=0;model_calls=0;head_fits=0;training_updates=0;baseline_changed=$false;application_changed=$false}
Write-NewJson 'doc/causal_fresh_v11/prospective-freeze.json' $receipt
$receipt|ConvertTo-Json -Depth 8

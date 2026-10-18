# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host metadata/byte preparation only. This does not admit or execute a project phase.
param(
 [Parameter(Mandatory=$true)][string]$InputPlan,
 [Parameter(Mandatory=$true)][string]$InputPlanSHA256,
 [Parameter(Mandatory=$true)][string]$OrchestrationProvenanceSHA256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedSourcesAndCompletedContext
)
$ErrorActionPreference='Stop'
if(-not$RootReviewedSourcesAndCompletedContext){throw 'Root source/context review is required before metadata preparation'}
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspace=[IO.Path]::GetFullPath((Join-Path $root '..'))
$draftPath=Join-Path $root 'doc/CAUSAL_FRESH_V14_CARD.json'
if(Test-Path -LiteralPath $draftPath){throw 'Preserve the existing draft card'}
function Physical-Path([string]$path,[string]$base=$root){
 $full=[IO.Path]::GetFullPath($path)
 if(-not$full.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Bounded repository path required'}
 for($cursor=$full;$cursor.Length-ge$base.Length;$cursor=[IO.Path]::GetDirectoryName($cursor)){if((Get-Item -LiteralPath $cursor).Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked path refused'};if($cursor-eq$base){break}}
 $full
}
function Get-FilePin([string]$relative){
 $file=Get-Item -LiteralPath (Physical-Path (Join-Path $root $relative))
 if($file.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Physical metadata input required'}
 [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Verify-Pin($pin,[string]$repository='cuwacunu_embedding'){
 if($repository-notin@('cuwacunu_embedding','cuwacunu_dev')){throw 'Known explicit repository required'}
 $base=Join-Path $workspace $repository
 $file=Get-Item -LiteralPath (Physical-Path (Join-Path $base $pin.path) $base)
 if($file.Length-ne$pin.bytes-or(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()-ne$pin.sha256){throw ('Exact protected file required: '+$pin.path)}
}
function Copy-ExactMetadata([string]$from,[string]$to){
 $destination=Join-Path $root $to
 if(Test-Path -LiteralPath $destination){throw ('Preserve existing metadata: '+$to)}
 [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))|Out-Null
 [IO.File]::Copy((Join-Path $root $from),$destination,$false)
 $a=Get-FilePin $from;$b=Get-FilePin $to
 if($a.sha256-ne$b.sha256-or$a.bytes-ne$b.bytes){throw 'Exact metadata copy failed'}
 $b
}
function Write-JsonNew([string]$relative,$value){
 $destination=Join-Path $root $relative
 if(Test-Path -LiteralPath $destination){throw ('Preserve existing output: '+$relative)}
 [IO.File]::WriteAllText($destination,($value|ConvertTo-Json -Depth 60)+"`n",[Text.UTF8Encoding]::new($false))
}
$planPath=Physical-Path ([IO.Path]::GetFullPath($InputPlan))
if((Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()-ne$InputPlanSHA256){throw 'External input-plan SHA required before JSON parse'}
$plan=Get-Content -LiteralPath $planPath -Raw|ConvertFrom-Json -AsHashtable
if($plan.kind-ne'cpb_v14_prospective_declaration_input_plan_v1'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true-or$plan.status-ne'root_reviewed_source_and_completed_context'){throw 'Only a root-reviewed completed-context plan can prepare a draft'}
foreach($required in @('doc/CAUSAL_FRESH_V11_CARD.json','doc/CAUSAL_FRESH_V11_CARD.md','doc/causal_fresh_v11/final-preservation.json','doc/CAUSAL_FRESH_V13_CARD.json','doc/CAUSAL_FRESH_V13_CARD.md','doc/causal_fresh_v13/protocol-preflight.json','doc/causal_fresh_v13/complete-mirror-attempt-001.json','doc/causal_fresh_v13/final-preservation.json','doc/causal_fresh_v9/protocol-preflight.json','doc/causal_fresh_v9/reporting-recovery-card-001.json','doc/causal_fresh_v9/reporting-recovery-receipt-001.json','doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json','doc/clean_causal_encoder_search_goal_v11_retention.json','doc/causal_v11_retention_review/independent-review.json','doc/causal_v11_retention_review/applied-future-protocol.json','doc/CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json','doc/causal_v11_v13_train_diagnosis/diagnosis.json','doc/causal_v11_v13_train_diagnosis/independent-audit.json','doc/CAUSAL_V11_V13_TRAIN_DIAGNOSIS.md','doc/causal_v11_v13_train_diagnosis/completion.json','doc/causal_v11_v13_train_diagnosis/human-report-root-finalization-001.json','doc/causal_v11_v13_train_diagnosis/human-report-independent-review-001.json','doc/clean_causal_encoder_research_ledger.json','code/encoders/causal_patch_bottleneck_mae/versions/current.json','doc/causal_fresh_v14/family-source-provenance.json','doc/causal_fresh_v14/evaluation-source-provenance.json','code/research/causal_fresh_v14/training-helper-provenance.json','doc/causal_fresh_v14/CPB-v14.descriptor-draft.json','doc/causal_fresh_v14/design-source-plan.ADMITTED-001.json','output/engineering/causal-fresh-v14-source-preparation/attempt-001/parents/manifest.json')){
 if(@($plan.protected_context|Where-Object{$_.path-eq$required}).Count-ne1){throw ('One exact reviewed context pin required: '+$required)}
}
foreach($pin in $plan.protected_context){Verify-Pin $pin}
if($plan.descriptor_source.path-ne'doc/causal_fresh_v14/CPB-v14.descriptor-draft.json'){throw 'Exact complete owned descriptor draft must be plan-bound before parse'}
Verify-Pin $plan.descriptor_source
foreach($version in 3..13){if(@($plan.protected_context|Where-Object{$_.path-eq('doc/results/causal_fresh_v'+$version+'/table_rows.json')}).Count-ne1){throw 'Every saved historical final control must be externally plan-bound'}}
foreach($version in 6..13){if(@($plan.protected_context|Where-Object{$_.path-eq('doc/causal_fresh_v'+$version+'/final-establishment.json')}).Count-ne1){throw 'Every separate historical establishment must be externally plan-bound'}}
Verify-Pin $plan.independent_training_auditor_source_review
$trainingReview=Get-Content -LiteralPath (Join-Path $root $plan.independent_training_auditor_source_review.path) -Raw|ConvertFrom-Json -AsHashtable
if($trainingReview.kind-ne'cpb_v14_family_and_training_auditor_independent_static_review_v1'-or$trainingReview.status-ne'passed'){throw 'Actual independently passed V14 training-auditor source review required'}
if($plan.source_derived_counts_status-ne'source_owner_derived_and_independently_reviewed'-or($plan.static_build.binaries-isnot[int]-and$plan.static_build.binaries-isnot[long])-or$plan.static_build.binaries-le0-or($plan.static_build.translation_units-isnot[int]-and$plan.static_build.translation_units-isnot[long])-or$plan.static_build.translation_units-le0){throw 'Actual independently reviewed NEW owner binary/TU/count declaration required; no inherited V13 count'}
$orchestration=Get-FilePin 'doc/causal_fresh_v14/orchestration-provenance.json'
if($orchestration.sha256-ne$OrchestrationProvenanceSHA256){throw 'Exact independently reviewed orchestration provenance required'}
$workflow=Get-Content -LiteralPath (Join-Path $root $orchestration.path) -Raw|ConvertFrom-Json -AsHashtable
if($workflow.files.Count-ne10){throw 'Exactly ten owned orchestration helpers required'}
foreach($binding in $workflow.files){Verify-Pin $binding.parent;Verify-Pin $binding.owned}
$ownerFiles=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v14') -Recurse -File)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v14.h'))+@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v14') -Recurse -File)
$observedPaths=@($ownerFiles|ForEach-Object{$_.FullName.Substring($root.Length+1).Replace([char]92,[char]47)}|Sort-Object)
if(Compare-Object $observedPaths @($plan.owner_sources.path|Sort-Object)){throw 'Plan must bind the complete actual family/research path set'}
foreach($pin in $plan.owner_sources){Verify-Pin $pin}
foreach($output in @('doc/CAUSAL_FRESH_V14_CARD.md','doc/causal_fresh_v14/draft-metadata-copy-receipt.json','doc/causal_fresh_v14/preservation-before.json')){if(Test-Path -LiteralPath (Join-Path $root $output)){throw ('Preserve existing output: '+$output)}}
Verify-Pin $plan.card_markdown_template
if((Get-FilePin 'doc/causal_fresh_v13/final-preservation.json').sha256-ne'ae11f853b6bd8462c766a939f69a513922aaa0d791a5bf9b156e485692f7adce'){throw 'Exact119-checkpoint V13 final preservation required before parse'}
$preservation=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v13/final-preservation.json') -Raw|ConvertFrom-Json -AsHashtable
if($preservation.status-ne'passed'-or$preservation.critical_checkpoint_pins.Count-ne119){throw 'Actual final V13 preservation with119 existing critical checkpoints required'}
foreach($pin in $preservation.critical_checkpoint_pins){$repo=$pin.repository;if(-not$repo){if(-not$pin.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')){throw 'Only five historical V5 pins may omit repository'};$repo='cuwacunu_embedding'};Verify-Pin $pin $repo}
$oldCard=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V13_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
if($oldCard.baseline_frozen_files.Count-ne47){throw '47 physical V3 source files required per repository'}
foreach($pin in $oldCard.baseline_frozen_files){foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){Verify-Pin $pin $repo}}
$oldSource=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/protocol-preflight.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($pin in $oldSource.sources){Verify-Pin $pin}
$latestPaths=[ordered]@{card='doc/CAUSAL_FRESH_V13_CARD.json';manifest='output/engineering/causal-fresh-v13/attempt-001/source-manifest.json';archive='output/engineering/causal-fresh-v13/attempt-001/source.zip';protocol_preflight='doc/causal_fresh_v13/protocol-preflight.json'}
foreach($role in $latestPaths.Keys){
 $binding=$plan.latest_frozen_source[$role]
 if($null-eq$binding-or$binding.path-cne$latestPaths[$role]){throw ('Exact latest frozen V13 source role required: '+$role)}
 $matches=@($plan.protected_context|Where-Object{$_.path-eq$binding.path})
 if($matches.Count-ne1-or$matches[0].bytes-ne$binding.bytes-or$matches[0].sha256-ne$binding.sha256){throw 'Latest source roles must be externally bound in the actual reviewed plan'}
 Verify-Pin $binding
}
$latestCard=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.card.path) -Raw|ConvertFrom-Json -AsHashtable
$latestManifest=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.manifest.path) -Raw|ConvertFrom-Json -AsHashtable
$latestPreflight=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.protocol_preflight.path) -Raw|ConvertFrom-Json -AsHashtable
$latestCount=$latestCard.source_counts.capsule_sources
if(($latestCount-isnot[int]-and$latestCount-isnot[long])-or$latestCount-le0-or$latestCount-ne$preservation.V13_source_files-or$latestManifest.files.Count-ne$latestCount-or@($latestManifest.files.path|Sort-Object -Unique).Count-ne$latestCount-or($latestPreflight.sources.Count+1)-ne$latestCount){throw 'Latest source count must derive consistently from actual frozen V13 card/final preservation/preflight/manifest'}
if($latestCard.model-ne'CPB-v13'-or$latestCard.stage.Contains('DRAFT')-or-not$latestCard.frozen_utc-or$latestManifest.card_sha256-ne$plan.latest_frozen_source.card.sha256-or$latestPreflight.status-ne'passed'-or$latestPreflight.card_sha256-ne$plan.latest_frozen_source.card.sha256){throw 'Exact frozen V13 source identities required'}
$latestArchives=@($preservation.source_ZIPs|Where-Object{$_.manifest.path-eq$plan.latest_frozen_source.manifest.path})
if($latestArchives.Count-ne1-or$latestArchives[0].source_files-ne$latestCount-or$latestArchives[0].manifest.sha256-ne$plan.latest_frozen_source.manifest.sha256-or$latestArchives[0].archive.sha256-ne$plan.latest_frozen_source.archive.sha256){throw 'Latest manifest/whole ZIP must match independently passed final V13 preservation'}
$latestPathsExpected=@($latestPreflight.sources.path)+@($plan.latest_frozen_source.protocol_preflight.path)
if(Compare-Object (@($latestManifest.files.path)|Sort-Object) ($latestPathsExpected|Sort-Object)){throw 'Latest source manifest path set must equal the passed V13 preflight plus its own receipt'}
foreach($binding in $latestManifest.files){Verify-Pin $binding}
$latestFrozenSource=[ordered]@{card=$plan.latest_frozen_source.card;manifest=$plan.latest_frozen_source.manifest;archive=$plan.latest_frozen_source.archive;protocol_preflight=$plan.latest_frozen_source.protocol_preflight;source_files=$latestCount;source_entries_rehashed=$latestManifest.files.Count;scope='Host current-file hashes plus externally pinned whole ZIP/manifest/preflight;V9 historical-source rehash retained;V13 is scientific source and completed predecessor;old V11/V12 remain historical;no model or archive numeric read'}
Write-JsonNew 'doc/causal_fresh_v14/preservation-before.json' ([ordered]@{kind='cpb_v14_before_execution_file_preservation_v1';status='passed';created_utc=[DateTime]::UtcNow.ToString('o');input_plan=[ordered]@{path=$planPath.Substring($root.Length+1).Replace([char]92,[char]47);bytes=(Get-Item -LiteralPath $planPath).Length;sha256=$InputPlanSHA256};baseline_preservation=(Get-FilePin 'doc/causal_fresh_v13/final-preservation.json');critical_checkpoint_pins=$preservation.critical_checkpoint_pins;critical_checkpoint_files=119;baseline_sources_each_repository=47;baseline_source_pins=$oldCard.baseline_frozen_files;protected_context=$plan.protected_context;old_V9_source_files_verified=$oldSource.sources.Count;new_owner_sources=$plan.owner_sources;file_hash_inspection_only=$true;archive_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
$parentCopies='output/engineering/causal-fresh-v14-protocol-preparation/attempt-001/parent-copies'
$copied=@()
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V13_CARD.json' ($parentCopies+'/CAUSAL_FRESH_V13_CARD.json')
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V13_CARD.md' ($parentCopies+'/CAUSAL_FRESH_V13_CARD.md')
$card=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V13_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($key in @('retention_protocol','retention_independent_review','retention_application_admission')){
 $binding=$plan[$key]
 if($null-eq$binding-or@($plan.protected_context|Where-Object{$_.path-eq$binding.path-and$_.bytes-eq$binding.bytes-and$_.sha256-eq$binding.sha256}).Count-ne1){throw ('Actual separately reviewed/applied retention pin required: '+$key)}
 Verify-Pin $binding
}
if($plan.retention_protocol.path-ne'doc/clean_causal_encoder_search_goal_v13_retention.json'){throw 'Canonical future V3–V13 protocol path required;866 remains historical ancestry only'}
$goal=Get-Content -LiteralPath (Join-Path $root $plan.retention_protocol.path) -Raw|ConvertFrom-Json -AsHashtable
$retentionReview=Get-Content -LiteralPath (Join-Path $root $plan.retention_independent_review.path) -Raw|ConvertFrom-Json -AsHashtable
$retentionApplied=Get-Content -LiteralPath (Join-Path $root $plan.retention_application_admission.path) -Raw|ConvertFrom-Json -AsHashtable
$sourceDesign=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v14/design-source-plan.ADMITTED-001.json') -Raw|ConvertFrom-Json -AsHashtable
if($goal.kind-ne'clean_causal_encoder_search_goal_v13_retention_protocol_v1'-or$goal.criterion_count-ne22-or$goal.criteria.Count-ne22-or$goal.additional_train_NLL_gate-isnot[bool]-or$goal.additional_train_NLL_gate-or$goal.baseline-ne'CPB-v3.market-01'-or($goal.criteria|ConvertTo-Json -Depth 30 -Compress)-cne($sourceDesign.prospective_criteria_proposal.criteria|ConvertTo-Json -Depth 30 -Compress)){throw 'Actual new22 conditions must literally equal the reviewed design;no historical rescoring/TRAINNLL gate'}
if($retentionReview.kind-ne'clean_causal_encoder_search_goal_v13_retention_independent_review_v1'-or$retentionReview.status-ne'passed'-or$retentionReview.checks-le0-or$retentionReview.protocol.path-ne$plan.retention_protocol.path-or$retentionReview.protocol.sha256-ne$plan.retention_protocol.sha256-or$retentionReview.protocol.bytes-ne$plan.retention_protocol.bytes){throw 'Independent actual protocol review must bind canonical newprotocol'}
if($retentionApplied.kind-ne'clean_causal_encoder_search_goal_v13_retention_future_application_v1'-or$retentionApplied.status-ne'passed'-or-not$retentionApplied.applied-or-not$retentionApplied.prospective_future_use_eligible-or$retentionApplied.application_changed-or$retentionApplied.historical_results_or_counts_recomputed-or$retentionApplied.additional_TRAIN_NLL_gate-or$retentionApplied.protocol.sha256-ne$plan.retention_protocol.sha256-or$retentionApplied.protocol.bytes-ne$plan.retention_protocol.bytes-or$retentionApplied.independent_review.sha256-ne$plan.retention_independent_review.sha256-or$retentionApplied.independent_review.bytes-ne$plan.retention_independent_review.bytes){throw 'Actual separate future application required;no ledger/DEV/history mutation'}
foreach($field in @('applied','prospective_future_use_eligible','application_changed','historical_results_or_counts_recomputed','additional_TRAIN_NLL_gate')){if($retentionApplied[$field]-isnot[bool]){throw ('Actual typed future application flag required: '+$field)}}
if($retentionApplied.protocol.path-ne$plan.retention_protocol.path-or$retentionApplied.independent_review.path-ne$plan.retention_independent_review.path){throw 'Exact actual protocol/review application paths required'}
if($retentionApplied.files.Count-ne2-or(Compare-Object @($retentionApplied.files.path|Sort-Object) @('doc/clean_causal_encoder_search_goal_v13_retention.json','doc/clean_causal_encoder_search_goal_v13_retention.md'))){throw 'Only actual canonical newprotocol JSON+MD may be applied'}
foreach($binding in $retentionApplied.files){Verify-Pin $binding;if(@($plan.protected_context|Where-Object{$_.path-eq$binding.path-and$_.bytes-eq$binding.bytes-and$_.sha256-eq$binding.sha256}).Count-ne1){throw 'Canonical applied protocol files must be externally plan-bound'}}
# Strongest historical866 is retained as ancestry;current authority is the separately applied new canonical protocol above.
$controls=[ordered]@{}
foreach($version in 3..13){$controls['v'+$version]=Copy-ExactMetadata ('doc/results/causal_fresh_v'+$version+'/table_rows.json') ('doc/causal_fresh_v14/baseline-control-v'+$version+'.json')}
foreach($version in 6..13){$controls['v'+$version+'_establishment']=Copy-ExactMetadata ('doc/causal_fresh_v'+$version+'/final-establishment.json') ('doc/causal_fresh_v14/baseline-v'+$version+'-establishment.json')}
$v13Established=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v13/final-establishment.json') -Raw|ConvertFrom-Json -AsHashtable
if($v13Established.status-ne'passed'-or$v13Established.candidate-ne'CPB-v13.market-01'-or$v13Established.measured_candidate_established-ne$true-or$v13Established.trust_admission_passed-ne$true){throw 'V13 historical DATA control requires its separately passed establishment;not overall winner'}
$controls.reuse_only='Exact saved V3/V4/V5/V6/V7/V8/V9/V10/V11/V12/V13 final tables and separate V6/V7/V8/V9/V10/V11/V12/V13 full-trust receipts; preserve pending producer bytes; no old encoder/head execution'
$goalPin=Copy-ExactMetadata $plan.retention_protocol.path 'doc/causal_fresh_v14/goal-protocol.json'
$reviewPin=Copy-ExactMetadata $plan.retention_independent_review.path 'doc/causal_fresh_v14/goal-protocol-review.json'
$admissionPin=Copy-ExactMetadata $plan.retention_application_admission.path 'doc/causal_fresh_v14/goal-protocol-admission.json'
$raw=Copy-ExactMetadata 'doc/causal_fresh_v13/raw-controls.json' 'doc/causal_fresh_v14/raw-controls.json'
$input=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v13/input-manifest.json') -Raw|ConvertFrom-Json -AsHashtable
$input.kind='causal_fresh_v14_data_input_manifest_v1';$input.protocol='causal-fresh-v14';$input.model='CPB-v14.market-01'
Write-JsonNew 'doc/causal_fresh_v14/input-manifest.json' $input
$policy='original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1'
$empty='zero-whole-export-slice-after-bias-v1'
$objective='huber1-mean-plus-detached-error-gaussian-variance-v1'
$gradient='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
$encoderGradient='Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1'
$routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
$establishment='Full trust including exact initial native20:40 DATA and whole-statistic pairings against saved V13, and independent training/quality audits, establishes a measured candidate. Advancement additionally requires full coverage and all22 declared numeric conditions. Winner and application selection remain separate and false.'
$card.kind='causal_fresh_v14_prospective_card_v1';$card.protocol='causal-fresh-v14'
$card.declared_utc=$null;$card.frozen_utc=$null;$card.stage='DRAFT_NOT_FROZEN_NOT_EXECUTION_ADMITTED'
$card.model='CPB-v14';$card.instance='CPB-v14.market-01';$card.initial_quality_label='CPB-v14.market-01.init'
$card.execution_admitted=$false;$card.application_selection_changed=$false
$card.mechanism.Remove('mandatory_identifiable_learned_values')
$card.mechanism.mandatory_identifiable_tensor_value_capacity=294751
$card.mechanism.mandatory_identifiable_count_semantics='tensor numel capacity, not every-scalar gradient/movement proof'
$card.mechanism.per_scalar_nonzero_gradient_proof_claimed=$false
$card.independent_training_auditor_source_review=$plan.independent_training_auditor_source_review
foreach($key in @('prospective_freeze','declared_client_date')){$card.Remove($key)}
foreach($record in @($card.mechanism,$card.training,$card.trust_receipt_objective_fields)){
 $record.objective_policy=$objective;$record.gradient_policy=$gradient;$record.direct_gradient_routes=$routes
 $record.mean_huber_delta=1.0;$record.loss_coefficients=@(1.0,1.0);$record.variance_error_detached=$true;$record.reporting_nll_backpropagated=$false
 $record.shared_global_clip=$true;$record.optimizer_independence_claimed=$false
}
$card.training.encoder_gradient_kind=$encoderGradient;$card.trust_receipt_objective_fields.encoder_gradient_kind=$encoderGradient
$card.mechanism.gradient_interpretation='Retain V13 objective,attention-coupled centered variance and actual F32 sqrt(v+2^-10)-1/32. Only valid original-position LOCAL supports become fixed nested trailing4/8/12/16;globalhalf,two-sum mean,full hidden/center/score/position residual derivatives remain. Shared globalclip1/AdamW couples future states;no validation-cause claim.'
$card.training.objective='R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v + log(v))];coefficients[1,1];supported operands sanitized and promoted separately to CUDAF64,unchanged hierarchical reduction. Only total_loss backpropagates;ordinary reporting NLL is a detached copy.'
$card.training.fixed_query_NLL='Ordinary Gaussian NLL from the detached reporting copy;separate Huber mean and detached-error variance objective components retain V9 routes'
$card.training.optimizer_coupling='One global norm2 clip1 and one AdamW;shared scaling,moments and future residuals couple optimization. No optimizer independence or validation-cause claim.'
$card.training.raw_load_counter_contract=[ordered]@{successful_updates=2048;attempts_min=2048;attempts_max=4096;calibration_loader_calls=512;initial_and_final_query_loader_calls=36;raw_load_calls_formula='512 + attempted_batches + 36';skipped_attempts='attempted_batches - new_causal_updates;derived,not a fabricated summary field'}
$card.mechanism.kind=$policy;$card.mechanism.temporal_pooling_policy=$policy
$card.mechanism.temporal_pooling_mix=@(0.5,0.5);$card.mechanism.empty_local_bin_policy=$empty
$card.mechanism.architecture='V13 attention/two-sum mean/live centeredvariance/F32 regularizedstd remain. Only four local supports become fixed original-position trailing4/8/12/16 ordered shortest-to-longest;globalhalf retained and full16 graph not simplified. Rawroot1/32 atzero/absence;dispersion0;whole8 absent served rows zero AFTERbias. Same32x64+32bias/global99->64->32/content20/native40.'
$card.mechanism.controlled_mechanism='One V13 SOURCE temporal-support hypothesis: fixed original endpoint16,lengths4/8/12/16,shortest-to-longest. No packed-rank/end-visible anchoring,transform/coefficient/length sweep,new parameter/buffer/draw,head or objective change.'
$card.mechanism.capacity_confounds='Same V13 shapes/counts/native40/D40heads and fresh draws;overlap/recency/allocation changes representation and gradient routing. No equal-quality or equal-information guarantee;globalhalf means slices are horizon emphases,not isolated horizons.'
$card.mechanism.allowed_typed_resume='Only fresh V14 lineage. Replace disjoint partition with temporal_pooling_geometry_v1 Long[4]=[4,16,32,8] and temporal_pooling_trailing_lengths_v1 Long[4]=[4,8,12,16] at inference root/payload and TRAIN root/encoder/encoder.payload;retain rank1UTF8 transform/tauDouble[1]=2^-10/objective. Reject legacy/missing/wrong dtype/rank/length/order/value before state load;exact own resume512.'
$card.mechanism.fresh_initialization_order='Same V13 registration/shapes/RNG draw order under943001;all294752 learned values,optimizer and TRAIN statistics fresh. One export tensor shared by both routes;source reuse never imports trained weights.'
$card.mechanism.empty_bin_policy='Valid original positions within[16-length[j],16). Emptylocal/nonemptyglobal zeros whole8 AFTER bias;allpadding finite servedzero,rawroot1/32. Invalid tokens cleared before subtract/square.'
$card.mechanism.causal_endpoint_semantics='Fixed original16-edge belongs to lastH128/P8 left-padded history at current cutoff. Never anchor to packed length/validcount/lastvisible/futureend. Both routes retain causal Transformer/backward relation bank;future values AND masks must not change cutoff embedding. Globalhalf/earlier hidden context remains.'
$card.mechanism.same_export_weights_means='One single freshly initialized export tensor shared across both pooling routes;unchanged shape/registration/draw order;never imported trained values'
$card.training.recipe_id='causal-v14-nested-trailing4-8-12-16-attention-mean4-regularized-stddev4-v13-objective-v1'
$card.training.temporal_pooling_policy=$policy;$card.training.temporal_pooling_mix=@(0.5,0.5);$card.training.empty_local_bin_policy=$empty
$card.training.component_evidence='All2048 successful scalar rows,8192 whole-bin rows,16384 mean/regularized_stddev subgroup rows,2048 gauge and Hadamard rows. Actual1/2048 existing-forward witness schema is externally source-reviewed below;raw variance,unmasked actual root,transformed dispersion retained. Actual2048 uses pre-step2047 coefficients. No extra forward or full-encoder/norm reconstruction.'
$card.quality.primary_methods=@('Raw data — no encoder (full256)','Raw data — no encoder (recent128)','CPB-v14.market-01.init','CPB-v14.market-01')
$card.quality.no_historical_encoder_rows_in_primary_tables=$true
$card.quality.primary_clean_control_rows_only=$false
$card.quality.clean_baseline_control=$controls.reuse_only
$card.quality.primary_report='Detailed producer tables contain8 saved raw-control market rows plus8 new own initial/final market rows=16,with four task tables/four TEMPO panels,physical errors,three neural repetitions,query components and cost. Separate comparison contains11 saved clean final encoders plus own initial/final=13 methods/52 market rows. Pending producer trust stays unchanged until independent audits/pairings and separate join. No automatic winner/application selection.'
$card.quality.saved_clean_comparison_methods=@('CPB-v3.market-01','CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01','CPB-v10.market-01','CPB-v11.market-01','CPB-v12.market-01','CPB-v13.market-01','CPB-v14.market-01.init','CPB-v14.market-01')
$card.advancement.policy='All22 exact newprospective strongest V3–V13 conditions plus mandatory fulltrust/coverage. Frozen old866 remains historicalancestry,never newauthority. Producer numeric/coverage/pendingtrust stayseparate;selection requires separateuserdecision.'
$card.advancement.goal_protocol=$goalPin;$card.advancement.criteria=$goal.criteria;$card.advancement.criterion_count=22
$card.criteria.quality_interpretation='All22 literal prospective V3–V13 objects copied unchanged from independently reviewed/applied actual newprotocol;five V13 scoped floors strengthened,other17 rules unchanged. Historical counts remain on oldcards;TRAIN NLL descriptive only. Nested temporal-support hypothesis,not validation-cause or promised improvement.'
$card.criteria.establishment=$establishment
$card.criteria.trust_admission=@(
 'All76 learned tensors/294752 values and optimizer/statistics fresh under identical V13 constructor draws/source0;75 identifiable tensors+sole observed common score-bias gauge;capacity is not every-scalar learning',
 'Four export/position/eight mean-regularized_stddev subgroups and three Hadamard directions need actual CUDA data-gradient/movement events;weight decay is not data learning',
 'All128cutoffs/2350suffix/30internal cases initial/final: future values AND masks cannot alter native40;past-positive interventions and current-endpoint semantics',
 'All temporal fields plus transform String and strict Float64[1]tau at allfive typed nodes,missing/old/dtype/shape/value negatives and exact typed512resume',
 'Exact exclusive source/card/descriptor/compiler/input/binary closure reviewed/frozen before any execution',
 'Source-derived attention-coupled regularized-stddev full-expression F32-aware adjoints/finite-difference fixtures,nonfinite/zero/tiny/large/empty/sparse/constant/equal-mean support cases;no V12 uniform-score-zero expectation',
 'Exactly13 saved V13 DATA references:feature native20:40 only plus five whole fresh statistics;content0:20/whole40 unpaired;new own initial heads,no old model/head execution',
 'Independent TRAIN/quality audits,fullcoverage/finite metrics and separate trust join;all22 actualnew strongestV3–V13 conditions before advancement;selection separatefalse'
)

$card.input_manifest=Get-FilePin 'doc/causal_fresh_v14/input-manifest.json'
$card.baseline_preservation=Get-FilePin 'doc/causal_fresh_v13/final-preservation.json'
$card.latest_frozen_source_preservation=$latestFrozenSource
$card.pre_execution_preservation=Get-FilePin 'doc/causal_fresh_v14/preservation-before.json'
$card.saved_TRAIN_diagnosis=[ordered]@{
 card=(Get-FilePin 'doc/CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json');result=(Get-FilePin 'doc/causal_v11_v13_train_diagnosis/diagnosis.json')
 independent_audit=(Get-FilePin 'doc/causal_v11_v13_train_diagnosis/independent-audit.json');human_report=(Get-FilePin 'doc/CAUSAL_V11_V13_TRAIN_DIAGNOSIS.md')
 completion=(Get-FilePin 'doc/causal_v11_v13_train_diagnosis/completion.json');human_report_root_review=(Get-FilePin 'doc/causal_v11_v13_train_diagnosis/human-report-root-finalization-001.json')
 scope='Completed V11/V13 TRAIN diagnosis:73778 independent V13 arithmetic checks,2035 documentary checks,0mismatches;four cachedV11objects/latest69952/upstream68264 opaque. No encoder/head/refit/heldout/trust replay.'
 interpretation='Body-time no-next-selection remains historicaltruth. Root separately chose V14 DESIGN-only nested trailing support after completed arithmetic/documentation. Current V13/119 closure is distinct from earlier protocol ancestry;no validation-cause or winner claim.'
}
$card.comparison_control=$controls;$card.scoped_reference='CPB-v13.market-01'
$card.scientific_recipe_parent='CPB-v13 source only;fresh weights/optimizer/statistics,not trained ancestry'
$card.pairing_reference='CPB-v13.market-01 saved initial DATA/statistics only'
$card.scoped_references=@('CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01','CPB-v10.market-01','CPB-v11.market-01','CPB-v12.market-01','CPB-v13.market-01')
$card.retention_independent_review=$reviewPin;$card.retention_application_admission=$admissionPin;$card.retention_source_bindings=[ordered]@{protocol=$plan.retention_protocol;independent_review=$plan.retention_independent_review;application_admission=$plan.retention_application_admission}
$card.geometry_override='Only V13 local support becomes nested trailing4/8/12/16;globalhalf/regularizedsqrt/native40/counts/draws/objective/heads fixed. Content0:20 unpaired;20:40DATA+five whole freshstats againstV13.'
$card.prospective_descriptor='code/encoders/causal_patch_bottleneck_mae/versions/CPB-v14.json'
$card.initial_pairing.reference='CPB-v13.market-01 initial0 DATA/statistic artifacts only'
$card.initial_pairing.native_coordinates_compared=20;$card.initial_pairing.native_coordinate_slice=@(20,40)
$card.initial_pairing.unpaired_learned_content_slice=@(0,20);$card.initial_pairing.content20_equality_required=$false
$card.initial_pairing.whole_native40_equality_required=$false;$card.initial_pairing.statistic_numeric_equality_required=$true
$card.initial_pairing.initial_scores_reused=$false;$card.initial_pairing.new_own_initial_fixed_heads=$true
$v13Mirror=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v13/complete-mirror-attempt-001.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($reference in $card.initial_pairing.references){
 $container=$reference.reference_data_archive.path.Replace('causal-fresh-v11/','causal-fresh-v13/').Replace('cpb_v11-','cpb_v13-')
 $matches=@($v13Mirror.files|Where-Object { $_.container_path -eq $container })
 if($matches.Count-ne1){throw ('One exact V13 initial DATA pin required: '+$container)}
 $actual=Get-FilePin $matches[0].path
 if($actual.sha256-ne$matches[0].sha256-or$actual.bytes-ne$matches[0].bytes){throw 'V13 initial DATA reference changed'}
 $reference.reference_data_archive=[ordered]@{path=$container;bytes=$actual.bytes;sha256=$actual.sha256}
 $reference.candidate_relative_to_attempt=$reference.candidate_relative_to_attempt.Replace('cpb_v13-','cpb_v14-')
 $reference.comparison='Feature value matrices compare20:40 only;valid/IDs/labels and five fresh-statistic archives compare whole declared numeric tensors. Content20 intentionally unpaired;new own initial heads,no old model/head execution.'
 if($reference.Contains('feature_value_coordinate_slice')){$reference.feature_value_coordinate_slice=@(20,40)}
}
$card.trust_receipt_objective_fields.temporal_pooling_policy=$policy
$card.trust_receipt_objective_fields.temporal_pooling_mix=@(0.5,0.5)
$card.trust_receipt_objective_fields.empty_local_bin_policy=$empty
$card.trust_receipt_field_applicability=[ordered]@{encoder_gradient_kind='Main training summary only;not an invented engineering/savedproof/fresh selector';gradient_policy='Strict typed records/engineering/training/savedproofs;absent only in constructor fresh-initialization.json';direct_gradient_routes='Mandatory common objective JSON in all declared records including fresh constructor'}
$card.mechanism_proof_contract.PASS_prefix='PASS CPB-v14 huber1-mean-plus-detached-error-gaussian-variance-v1:'
foreach($key in @('direct_joint_NLL_mean_gradient_live','direct_joint_NLL_uncertainty_gradient_live','joint_Gaussian_variance_counted_once','total_equals_Huber_plus_one_joint_Gaussian','independent_composite_F64_finite_differences')){$card.mechanism_proof_contract.Remove($key)}
$card.mechanism_proof_contract.direct_NLL_mean_gradient_zero=$true
$card.mechanism_proof_contract.manual_parity_scope='Unchanged V11 Huber1 mean plus detached-error Gaussian conditional-variance objective;manual native40 partial adjoints,direct cross-path zeros,live head/projection and shared globalclip coupling'
$card.mechanism_proof_contract.log_tokens=@('manual composite-objective native40 gradient parity','detached-error NLL mean gradient zero','synthetic optimizer steps=1')
$temporal=$plan.temporal_pooling_proof_contract
if($temporal.status-ne'DRAFT_source_owner_frozen_independent_review_before_any_execution'-or$temporal.binary-ne'test_cpb_v14_temporal_pooling'-or$temporal.temporal_pooling_policy-ne$policy-or$temporal.PASS_prefix-ne('PASS CPB-v14 '+$policy+':')-or$temporal.required_binary_count-ne$plan.static_build.binaries-or$temporal.required_TU_count-ne$plan.static_build.translation_units){throw 'Exact source-derived prospective temporal proof contract required'}
$proof=$temporal.required_proof_fields
if($null-eq$proof-or$proof.model-ne'CPB-v14'-or$proof.temporal_dispersion_transform_policy-ne'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'-or$proof.temporal_dispersion_regularizer-ne0.0009765625-or$temporal.log_tokens.Count-lt1){throw 'Actual source-derived V14 transform proof fields/tokens required'}
if($temporal.source_review.path-ne$plan.independent_training_auditor_source_review.path-or$temporal.source_review.sha256-ne$plan.independent_training_auditor_source_review.sha256){throw 'Actual independently reviewed family/TRAIN-auditor authority must bind fixture contract'}
Verify-Pin $temporal.source_review;Verify-Pin $temporal.fixture_source
if(@($trainingReview.reviewed_owned_files|Where-Object{$_.path-eq$temporal.fixture_source.path-and$_.bytes-eq$temporal.fixture_source.bytes-and$_.sha256-eq$temporal.fixture_source.sha256}).Count-ne1-or$trainingReview.blockers.Count-ne0){throw 'Nonself scientific review must actually include exact compiled fixture source and no blockers'}
if($proof.direct_pool_kernel_calls-ne53-or$proof.F64_finite_difference_coordinates-ne18-or$proof.F32_finite_difference_coordinates-ne18-or$proof.isolated_autograd_grad_calls-ne4){throw 'NEW source-derived53direct/18F64/18F32/4isolated contract required'}
if($temporal.fixture_source.path-ne'code/research/causal_fresh_v14/tests/v14_temporal_pooling_test.cpp'-or@($plan.owner_sources|Where-Object{$_.path-eq$temporal.fixture_source.path-and$_.bytes-eq$temporal.fixture_source.bytes-and$_.sha256-eq$temporal.fixture_source.sha256}).Count-ne1){throw 'Fixture contract must bind exact complete owned fixture source'}
$fixtureText=Get-Content -LiteralPath (Join-Path $root $temporal.fixture_source.path) -Raw
foreach($token in $temporal.log_tokens){if($token-isnot[string]-or-not$token-or-not$fixtureText.Contains($token)){throw 'Every frozen proof token must occur in independently reviewed actual fixture source'}}
foreach($key in $proof.Keys){if($key-match'uniform|dual_support'){throw 'No inherited V12 uniform proof expectation'}}
foreach($field in @('mean4_regularized_stddev4_export_layout','full_attention_dependent_regularized_moment_adjoints','live_dispersion_score_gradient','raw_variance_root_dispersion_trace','zero_tiny_large_analytic_dispersion_adjoints','supported_nonfinite_and_variance_overflow_refused','absent_local_huge_singleton_sanitized','captured_F32_residual_correction','raw_root_absent_offset_exception','nested_trailing_supports_fixed_edge16','per_window_local_weights_captured','longest_local_equals_global_two_sum','packed_padding_membership_invariance')){if($proof[$field]-isnot[bool]-or-not$proof[$field]-or$temporal.log_tokens-notcontains($field+'=true')){throw ('Actual new fixture flag/token: '+$field)}}
foreach($key in @('direct_pool_kernel_calls','F64_finite_difference_coordinates','F32_finite_difference_coordinates','isolated_autograd_grad_calls')){if(($proof[$key]-isnot[int]-and$proof[$key]-isnot[long])-or$proof[$key]-le0){throw ('Actual source-derived positive fixture count required: '+$key)}};if($proof.transform_derivative_upper_bound-ne16){throw 'Retained analytic regularized-sqrt derivative bound16'}
if(($plan.engineering_typed_negative_cases-isnot[int]-and$plan.engineering_typed_negative_cases-isnot[long])-or$plan.engineering_typed_negative_cases-ne821){throw 'Actual compiled research source-derived821 refusal cases required;family-only676 is a distinct fixture'}
$card.engineering_typed_negative_cases=$plan.engineering_typed_negative_cases;$card.engineering_archive_contract=[ordered]@{format='full_refusal_archives';typed_negative_full_archive_files=817;typed_negative_direct_loader_requests=4;typed_negative_compact_archive_files=0;typed_negative_restored_target_controls=0;typed_negative_legacy_conflict_files=5;real_main_checkpoints=5;all_full_negative_files_preserved=$true;scope='Compiled research primary821=817fullarchivefiles+4directrequests;family-only676 is not this executed fixture;fullfallback,no compact path'}
$fixtureNumeric=[ordered]@{backbone_encode_calls=1;public_encoder_calls=0;head_fits=0;synthetic_optimizer_steps=1;manual_F32_absolute_tolerance=3e-5;manual_F32_relative_tolerance=3e-5;F64_finite_difference_step=1e-5;F64_finite_difference_absolute_tolerance=2e-8;F64_finite_difference_relative_tolerance=2e-8;F32_finite_difference_step=2e-3;F32_finite_difference_absolute_tolerance=2e-3;F32_finite_difference_relative_tolerance=2e-3;softmax_gauge_tolerance=2e-6}
foreach($key in $fixtureNumeric.Keys){if($proof[$key]-ne$fixtureNumeric[$key]){throw ('Exact source-derived fixture tolerance/count: '+$key)}}
$card.temporal_pooling_proof_contract=$temporal
$card.temporal_learning_receipt_contract.kind='cpb_v14_actual_temporal_pooling_learning_v1'
$card.temporal_learning_receipt_contract.temporal_pooling_policy=$policy
$card.temporal_learning_receipt_contract.temporal_pooling_mix=@(0.5,0.5)
$card.temporal_learning_receipt_contract.empty_local_bin_policy=$empty
$card.temporal_learning_receipt_contract.between_bin_rows=2048
$card.temporal_learning_receipt_contract.receipt_artifacts=@('temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv','temporal-between-bin-position-learning.csv','temporal-moment-subgroup-learning.csv')
$card.temporal_learning_receipt_contract.moment_subgroup_rows=16384
$card.temporal_learning_receipt_contract.moment_subgroups=[ordered]@{count=8;moments=@('mean','regularized_stddev');rows_each=4;columns=@('update','bin','moment','export_row_begin','export_row_end','weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2','nonzero_export_gradient_values','moved_export_values');each_weight_and_bias_data_gradient_and_movement_events_positive=$true;each_final_weight_and_bias_changed=$true;every_scalar_or_update_nonzero_required=$false;pooling_intermediate_not_native_uncertainty_branch=$true}
$card.temporal_learning_receipt_contract.summary_positive_flags=@(@($card.temporal_learning_receipt_contract.summary_positive_flags)+@('all_eight_temporal_moment_subgroups_learned')|Select-Object -Unique)
$card.temporal_learning_receipt_contract.between_bin_positions=[ordered]@{
 basis='hadamard-full16-signs-div4-v1';basis_signs=@(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1));full_position_divisor=4;position_bin_width=4;rows=2048
 columns=@('update','between_bin_gradient_l2','between_bin_movement_l2','contrast_0_gradient','contrast_1_gradient','contrast_2_gradient','contrast_0_movement','contrast_1_movement','contrast_2_movement')
 gradient_definition='observed-post-globalclip-data-gradient';F64_before_projection_and_subtraction=$true;parseval_identity_required=$true
 each_direction_positive_data_gradient_and_movement_events=$true;overall_final_projection_change_l2_positive=$true
 each_final_signed_coefficient_nonzero_required=$false;movement_includes_AdamW_weight_decay=$true;weight_decay_counted_as_data_learning=$false;extra_encoder_calls=0
}
$card.fresh_initialization_JSON_exception.required_other_fields='All objective/delta/coefficients/direct_gradient_routes/detachtrue/reportingNLLfalse and temporalpolicy/geometry/trailing_lengths/mix/empty fields plus75/76gauge fields remain mandatory;gradient_policy absent only in this constructor receipt'
$card.source_contract_status='DRAFT_NOT_ADMITTED_actual_owner_counts_and_independent_source_review_pending'
$card.source_admission='No AST/build/test/CUDA/model/head/body execution authorized by this draft. Exact final owner source counts/compiler/TU/binary closure and independent preflight must be frozen prospectively,without a source/card hash cycle.'
$card.source_pins_after_owner_freeze=$plan.owner_sources
foreach($key in @('independent_static_helper_review','independent_static_helper_review_amendment','source_review_scope')){$card.Remove($key)}
$card.source_counts=[ordered]@{family_inside=@($ownerFiles|Where-Object{$_.FullName.StartsWith((Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v14')+[IO.Path]::DirectorySeparatorChar)}).Count;family_external_header=1;research=@($ownerFiles|Where-Object{$_.FullName.StartsWith((Join-Path $root 'code/research/causal_fresh_v14')+[IO.Path]::DirectorySeparatorChar)}).Count;translation_units=$plan.static_build.translation_units;binaries=$plan.static_build.binaries;capsule_sources=$null;status='SOURCE_ONLY_actual_owner_path_set_bound_capsule_count_derived_at_freeze'}
$card.draft_copy_provenance=[ordered]@{
 kind='complete_physical_V13_SOURCE_then_owned_V14_nested_trailing_support_edits'
 card=(Get-FilePin 'doc/CAUSAL_FRESH_V13_CARD.json');markdown=(Get-FilePin 'doc/CAUSAL_FRESH_V13_CARD.md')
 direct_orchestration_parent=(Get-FilePin 'doc/causal_fresh_v13/run-container.ps1')
 repaired_native_Docker_ancestry=(Get-FilePin 'doc/causal_fresh_v8/run-container-recovery-001.ps1')
 parent_copies=$copied;old_files_modified=$false;source_only=$true
}
$card.geometry.temporal_pooling_policy=$policy;$card.geometry.temporal_pooling_geometry=@(4,16,32,8);$card.geometry.temporal_pooling_trailing_lengths=@(4,8,12,16);$card.geometry.temporal_pooling_mix=@(0.5,0.5);$card.geometry.empty_local_bin_policy=$empty
foreach($record in @($card.geometry,$card.mechanism,$card.training,$card.trust_receipt_objective_fields,$card.temporal_pooling_proof_contract,$card.temporal_learning_receipt_contract,$proof)){
 $record.Remove('temporal_pooling_partition');$record.temporal_pooling_geometry=@(4,16,32,8);$record.temporal_pooling_trailing_lengths=@(4,8,12,16)
 $record.temporal_moment_policy='componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1'
 $record.temporal_export_layout='per-trailing-window-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1'
 $record.temporal_moment_partition=@(4,4)
 $record.temporal_dispersion_transform_policy='sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'
 $record.temporal_dispersion_regularizer=0.0009765625
}
$card.actual_step_temporal_witness_contract=$plan.actual_step_temporal_witness_contract
if($card.actual_step_temporal_witness_contract.total_fields-ne60-or$card.actual_step_temporal_witness_contract.base_fields-ne23-or$card.actual_step_temporal_witness_contract.parameter_arrays-ne12-or$card.actual_step_temporal_witness_contract.temporal_trace_fields-ne25-or$card.actual_step_temporal_witness_contract.extra_encoder_calls-ne0){throw 'Actual independently reviewed writer schema23+12+25=60 required'}
$expectedPerWindowLocalFields=@('temporal_pool_local_weights','temporal_contextual_pool_local_weights')
if(Compare-Object @($card.actual_step_temporal_witness_contract.per_window_local_weight_fields|Sort-Object) @($expectedPerWindowLocalFields|Sort-Object)){throw 'Both existing local-weight fields must be exact captured per-window matrices'}
if(($card.actual_step_temporal_witness_contract.per_window_local_weight_shape-join',')-cne'R,4,T'){throw 'V14 local weight shape must be[R,4,T],not historical V13 aggregate[R,T]'}
$card.actual_step_temporal_witness_contract.local_weight_semantics='Exact same-forward per-window local softmax weights[R,4,T] for ordinary/contextual routes;aggregate pool_weights[R,T] remains diagnostic only. No rounded mixture inversion/renormalization or extraencoderforward;oldV13 archives remain unchanged.'
$expectedTransformFields=@('temporal_pool_bin_regularized_root','temporal_pool_bin_dispersion','temporal_contextual_pool_bin_regularized_root','temporal_contextual_pool_bin_dispersion')
if(Compare-Object @($card.actual_step_temporal_witness_contract.dispersion_transform_trace_fields|Sort-Object) @($expectedTransformFields|Sort-Object)){throw 'Exact raw-root/transformed-dispersion same-forward witness names'}
$card.mechanism.raw_root_empty_semantics='Actual raw root is unmasked1/32 at zero/absent/allpadding;transformed dispersion0 at zero;served absent whole8 slice exact0 after bias. Not every trace is zero on padding.'
$card.results=[ordered]@{status='unmeasured';quality=$null;trust=$null;winner_selection_made=$false;application_selection_changed=$false}
$card.parent_goal_preserved=Get-FilePin 'doc/clean_causal_encoder_search_goal_v9_retention.json'
$card.historical_V9_reporting_recovery=[ordered]@{card=(Get-FilePin 'doc/causal_fresh_v9/reporting-recovery-card-001.json');receipt=(Get-FilePin 'doc/causal_fresh_v9/reporting-recovery-receipt-001.json');failure_mirror=(Get-FilePin 'doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json');scope='HistoricalV9 reporting-only/no-science-replay evidence;not a V14 recovery or recipe change'}
$card.declaration_input_plan=[ordered]@{path=$planPath.Substring($root.Length+1).Replace([char]92,[char]47);bytes=(Get-Item -LiteralPath $planPath).Length;sha256=$InputPlanSHA256}
$card.reviewed_orchestration_provenance=$orchestration
$card.owned_descriptor_draft=$plan.descriptor_source
Verify-Pin $plan.descriptor_source
$descriptor=Get-Content -LiteralPath (Join-Path $root $plan.descriptor_source.path) -Raw|ConvertFrom-Json -AsHashtable
if($descriptor.version-ne'CPB-v14'-or$descriptor.instance-ne'CPB-v14.market-01'-or$null-ne$descriptor.card-or$descriptor.architecture_copy_parent-ne'CPB-v13'-or$descriptor.temporal_pooling_policy-ne$policy-or$descriptor.temporal_moment_policy-ne$card.mechanism.temporal_moment_policy-or$descriptor.temporal_dispersion_transform_policy-ne'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'-or$descriptor.temporal_dispersion_regularizer-ne0.0009765625-or$descriptor.objective_policy-ne$objective-or$descriptor.gradient_policy-ne$gradient-or$descriptor.retention_protocol.bytes-ne$goalPin.bytes-or$descriptor.retention_protocol.sha256-ne$goalPin.sha256-or$descriptor.retention_protocol.path-ne$goalPin.path){throw 'Actual complete owned descriptor draft must match the independently reviewed V14 source/criteria declaration;never rewrite it during metadata preparation'}
Write-JsonNew 'doc/CAUSAL_FRESH_V14_CARD.json' $card
[IO.File]::Copy((Join-Path $root $plan.card_markdown_template.path),(Join-Path $root 'doc/CAUSAL_FRESH_V14_CARD.md'),$false)
Write-JsonNew 'doc/causal_fresh_v14/draft-metadata-copy-receipt.json' ([ordered]@{status='draft_prepared_not_execution_admitted';controls=$controls;goal=$goalPin;review=$reviewPin;admission=$admissionPin;raw_controls=$raw;parent_copies=$copied;owned_descriptor_reused_unchanged=$plan.descriptor_source;initial_DATA_references=13;checkpoint_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
Get-FilePin 'doc/CAUSAL_FRESH_V14_CARD.json'|ConvertTo-Json -Depth 4

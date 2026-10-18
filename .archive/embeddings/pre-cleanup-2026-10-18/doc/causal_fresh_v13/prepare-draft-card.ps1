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
$draftPath=Join-Path $root 'doc/CAUSAL_FRESH_V13_CARD.json'
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
if($plan.kind-ne'cpb_v13_prospective_declaration_input_plan_v1'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true-or$plan.status-ne'root_reviewed_source_and_completed_context'){throw 'Only a root-reviewed completed-context plan can prepare a draft'}
foreach($required in @('doc/CAUSAL_FRESH_V11_CARD.json','doc/CAUSAL_FRESH_V11_CARD.md','doc/causal_fresh_v11/final-preservation.json','doc/CAUSAL_FRESH_V12_CARD.json','doc/CAUSAL_FRESH_V12_CARD.md','doc/causal_fresh_v12/protocol-preflight.json','doc/causal_fresh_v12/complete-mirror-attempt-001.json','doc/causal_fresh_v12/final-preservation.json','doc/causal_fresh_v9/protocol-preflight.json','doc/causal_fresh_v9/reporting-recovery-card-001.json','doc/causal_fresh_v9/reporting-recovery-receipt-001.json','doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json','doc/clean_causal_encoder_search_goal_v11_retention.json','doc/causal_v11_retention_review/independent-review.json','doc/causal_v11_retention_review/applied-future-protocol.json','doc/CAUSAL_V11_V12_TRAIN_DIAGNOSIS_CARD.ADMITTED.json','doc/causal_v11_v12_train_diagnosis/diagnosis.json','doc/causal_v11_v12_train_diagnosis/independent-audit.json','doc/CAUSAL_V11_V12_TRAIN_DIAGNOSIS.md','doc/causal_v11_v12_train_diagnosis/completion.json','doc/causal_v11_v12_train_diagnosis/report-root-review-001.json','doc/causal_v11_v12_train_diagnosis/report-independent-review-001.json','doc/clean_causal_encoder_research_ledger.json','code/encoders/causal_patch_bottleneck_mae/versions/current.json','doc/causal_fresh_v13/family-source-provenance.json','doc/causal_fresh_v13/evaluation-source-provenance.json','code/research/causal_fresh_v13/training-helper-provenance.json','doc/causal_fresh_v13/CPB-v13.descriptor-draft.json','doc/causal_fresh_v13/design-source-plan.ADMITTED-002.json','output/engineering/causal-fresh-v13-source-preparation/attempt-001/parents/manifest.json')){
 if(@($plan.protected_context|Where-Object{$_.path-eq$required}).Count-ne1){throw ('One exact reviewed context pin required: '+$required)}
}
foreach($pin in $plan.protected_context){Verify-Pin $pin}
if($plan.descriptor_source.path-ne'doc/causal_fresh_v13/CPB-v13.descriptor-draft.json'){throw 'Exact complete owned descriptor draft must be plan-bound before parse'}
Verify-Pin $plan.descriptor_source
foreach($version in 3..12){if(@($plan.protected_context|Where-Object{$_.path-eq('doc/results/causal_fresh_v'+$version+'/table_rows.json')}).Count-ne1){throw 'Every saved historical final control must be externally plan-bound'}}
foreach($version in 6..12){if(@($plan.protected_context|Where-Object{$_.path-eq('doc/causal_fresh_v'+$version+'/final-establishment.json')}).Count-ne1){throw 'Every separate historical establishment must be externally plan-bound'}}
Verify-Pin $plan.independent_training_auditor_source_review
$trainingReview=Get-Content -LiteralPath (Join-Path $root $plan.independent_training_auditor_source_review.path) -Raw|ConvertFrom-Json -AsHashtable
if($trainingReview.kind-ne'cpb_v13_family_and_training_auditor_independent_static_review_v1'-or$trainingReview.status-ne'passed'){throw 'Actual independently passed V13 training-auditor source review required'}
if($plan.static_build.binaries-ne7-or$plan.static_build.translation_units-ne11){throw 'Final source owners declare seven binaries and eleven TUs; actual build admission remains separate'}
$orchestration=Get-FilePin 'doc/causal_fresh_v13/orchestration-provenance.json'
if($orchestration.sha256-ne$OrchestrationProvenanceSHA256){throw 'Exact independently reviewed orchestration provenance required'}
$workflow=Get-Content -LiteralPath (Join-Path $root $orchestration.path) -Raw|ConvertFrom-Json -AsHashtable
if($workflow.files.Count-ne10){throw 'Exactly ten owned orchestration helpers required'}
foreach($binding in $workflow.files){Verify-Pin $binding.parent;Verify-Pin $binding.owned}
$ownerFiles=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v13') -Recurse -File)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v13.h'))+@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v13') -Recurse -File)
$observedPaths=@($ownerFiles|ForEach-Object{$_.FullName.Substring($root.Length+1).Replace([char]92,[char]47)}|Sort-Object)
if(Compare-Object $observedPaths @($plan.owner_sources.path|Sort-Object)){throw 'Plan must bind the complete actual family/research path set'}
foreach($pin in $plan.owner_sources){Verify-Pin $pin}
foreach($output in @('doc/CAUSAL_FRESH_V13_CARD.md','doc/causal_fresh_v13/draft-metadata-copy-receipt.json','doc/causal_fresh_v13/preservation-before.json')){if(Test-Path -LiteralPath (Join-Path $root $output)){throw ('Preserve existing output: '+$output)}}
Verify-Pin $plan.card_markdown_template
if((Get-FilePin 'doc/causal_fresh_v12/final-preservation.json').sha256-ne'db4e5f31737688e041784bd20d53747f10fc58b1e189041ca45bc2c2846fded7'){throw 'Exact114-checkpoint V12 final preservation required before parse'}
$preservation=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v12/final-preservation.json') -Raw|ConvertFrom-Json -AsHashtable
if($preservation.status-ne'passed'-or$preservation.critical_checkpoint_pins.Count-ne114){throw 'Actual final V12 preservation with114 existing critical checkpoints required'}
foreach($pin in $preservation.critical_checkpoint_pins){$repo=$pin.repository;if(-not$repo){if(-not$pin.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')){throw 'Only five historical V5 pins may omit repository'};$repo='cuwacunu_embedding'};Verify-Pin $pin $repo}
$oldCard=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V11_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
if($oldCard.baseline_frozen_files.Count-ne47){throw '47 physical V3 source files required per repository'}
foreach($pin in $oldCard.baseline_frozen_files){foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){Verify-Pin $pin $repo}}
$oldSource=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/protocol-preflight.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($pin in $oldSource.sources){Verify-Pin $pin}
$latestPaths=[ordered]@{card='doc/CAUSAL_FRESH_V12_CARD.json';manifest='output/engineering/causal-fresh-v12/attempt-001/source-manifest.json';archive='output/engineering/causal-fresh-v12/attempt-001/source.zip';protocol_preflight='doc/causal_fresh_v12/protocol-preflight.json'}
foreach($role in $latestPaths.Keys){
 $binding=$plan.latest_frozen_source[$role]
 if($null-eq$binding-or$binding.path-cne$latestPaths[$role]){throw ('Exact latest frozen V12 source role required: '+$role)}
 $matches=@($plan.protected_context|Where-Object{$_.path-eq$binding.path})
 if($matches.Count-ne1-or$matches[0].bytes-ne$binding.bytes-or$matches[0].sha256-ne$binding.sha256){throw 'Latest source roles must be externally bound in the actual reviewed plan'}
 Verify-Pin $binding
}
$latestCard=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.card.path) -Raw|ConvertFrom-Json -AsHashtable
$latestManifest=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.manifest.path) -Raw|ConvertFrom-Json -AsHashtable
$latestPreflight=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.protocol_preflight.path) -Raw|ConvertFrom-Json -AsHashtable
$latestCount=$latestCard.source_counts.capsule_sources
if(($latestCount-isnot[int]-and$latestCount-isnot[long])-or$latestCount-le0-or$latestCount-ne$preservation.V12_source_files-or$latestManifest.files.Count-ne$latestCount-or@($latestManifest.files.path|Sort-Object -Unique).Count-ne$latestCount-or($latestPreflight.sources.Count+1)-ne$latestCount){throw 'Latest source count must derive consistently from actual frozen V12 card/final preservation/preflight/manifest'}
if($latestCard.model-ne'CPB-v12'-or$latestCard.stage.Contains('DRAFT')-or-not$latestCard.frozen_utc-or$latestManifest.card_sha256-ne$plan.latest_frozen_source.card.sha256-or$latestPreflight.status-ne'passed'-or$latestPreflight.card_sha256-ne$plan.latest_frozen_source.card.sha256){throw 'Exact frozen V12 source identities required'}
$latestArchives=@($preservation.source_ZIPs|Where-Object{$_.manifest.path-eq$plan.latest_frozen_source.manifest.path})
if($latestArchives.Count-ne1-or$latestArchives[0].source_files-ne$latestCount-or$latestArchives[0].manifest.sha256-ne$plan.latest_frozen_source.manifest.sha256-or$latestArchives[0].archive.sha256-ne$plan.latest_frozen_source.archive.sha256){throw 'Latest manifest/whole ZIP must match independently passed final V12 preservation'}
$latestPathsExpected=@($latestPreflight.sources.path)+@($plan.latest_frozen_source.protocol_preflight.path)
if(Compare-Object (@($latestManifest.files.path)|Sort-Object) ($latestPathsExpected|Sort-Object)){throw 'Latest source manifest path set must equal the passed V12 preflight plus its own receipt'}
foreach($binding in $latestManifest.files){Verify-Pin $binding}
$latestFrozenSource=[ordered]@{card=$plan.latest_frozen_source.card;manifest=$plan.latest_frozen_source.manifest;archive=$plan.latest_frozen_source.archive;protocol_preflight=$plan.latest_frozen_source.protocol_preflight;source_files=$latestCount;source_entries_rehashed=$latestManifest.files.Count;scope='Host current-file hashes plus externally pinned whole ZIP/manifest/preflight;V9 historical-source rehash retained;V11 is scientific source;V12 is completed predecessor/infrastructure;no model or archive numeric read'}
Write-JsonNew 'doc/causal_fresh_v13/preservation-before.json' ([ordered]@{kind='cpb_v13_before_execution_file_preservation_v1';status='passed';created_utc=[DateTime]::UtcNow.ToString('o');input_plan=[ordered]@{path=$planPath.Substring($root.Length+1).Replace([char]92,[char]47);bytes=(Get-Item -LiteralPath $planPath).Length;sha256=$InputPlanSHA256};baseline_preservation=(Get-FilePin 'doc/causal_fresh_v12/final-preservation.json');critical_checkpoint_pins=$preservation.critical_checkpoint_pins;critical_checkpoint_files=114;baseline_sources_each_repository=47;baseline_source_pins=$oldCard.baseline_frozen_files;protected_context=$plan.protected_context;old_V9_source_files_verified=$oldSource.sources.Count;new_owner_sources=$plan.owner_sources;file_hash_inspection_only=$true;archive_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
$parentCopies='output/engineering/causal-fresh-v13-protocol-preparation/attempt-001/parent-copies'
$copied=@()
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V11_CARD.json' ($parentCopies+'/CAUSAL_FRESH_V11_CARD.json')
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V11_CARD.md' ($parentCopies+'/CAUSAL_FRESH_V11_CARD.md')
$card=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V11_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
$goal=Get-Content -LiteralPath (Join-Path $root 'doc/clean_causal_encoder_search_goal_v11_retention.json') -Raw|ConvertFrom-Json -AsHashtable
if((Get-FilePin 'doc/clean_causal_encoder_search_goal_v11_retention.json').sha256-ne'866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'-or$goal.criteria.Count-ne22-or$goal.additional_train_NLL_gate-ne$false){throw 'Exact current22-condition/noTRAINNLL protocol required'}
$controls=[ordered]@{}
foreach($version in 3..12){$controls['v'+$version]=Copy-ExactMetadata ('doc/results/causal_fresh_v'+$version+'/table_rows.json') ('doc/causal_fresh_v13/baseline-control-v'+$version+'.json')}
foreach($version in 6..12){$controls['v'+$version+'_establishment']=Copy-ExactMetadata ('doc/causal_fresh_v'+$version+'/final-establishment.json') ('doc/causal_fresh_v13/baseline-v'+$version+'-establishment.json')}
$v12Established=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v12/final-establishment.json') -Raw|ConvertFrom-Json -AsHashtable
if($v12Established.status-ne'passed'-or$v12Established.candidate-ne'CPB-v12.market-01'-or$v12Established.measured_candidate_established-ne$true-or$v12Established.trust_admission_passed-ne$true){throw 'V12 historical DATA control requires its separately passed establishment;not overall winner'}
$controls.reuse_only='Exact saved V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 final tables and separate V6/V7/V8/V9/V10/V11/V12 full-trust receipts; preserve pending producer bytes; no old encoder/head execution'
$goalPin=Copy-ExactMetadata 'doc/clean_causal_encoder_search_goal_v11_retention.json' 'doc/causal_fresh_v13/goal-protocol.json'
$reviewPin=Copy-ExactMetadata 'doc/causal_v11_retention_review/independent-review.json' 'doc/causal_fresh_v13/goal-protocol-review.json'
$admissionPin=Copy-ExactMetadata 'doc/causal_v11_retention_review/applied-future-protocol.json' 'doc/causal_fresh_v13/goal-protocol-admission.json'
$raw=Copy-ExactMetadata 'doc/causal_fresh_v11/raw-controls.json' 'doc/causal_fresh_v13/raw-controls.json'
$input=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v11/input-manifest.json') -Raw|ConvertFrom-Json -AsHashtable
$input.kind='causal_fresh_v13_data_input_manifest_v1';$input.protocol='causal-fresh-v13';$input.model='CPB-v13.market-01'
Write-JsonNew 'doc/causal_fresh_v13/input-manifest.json' $input
$policy='original-position-four-bin-half-mixture-mean4-regularized-stddev4-v1'
$empty='zero-whole-export-slice-after-bias-v1'
$objective='huber1-mean-plus-detached-error-gaussian-variance-v1'
$gradient='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
$encoderGradient='Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1'
$routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
$establishment='Full trust including exact initial native20:40 DATA and whole-statistic pairings against saved V11, and independent training/quality audits, establishes a measured candidate. Advancement additionally requires full coverage and all22 declared numeric conditions. Winner and application selection remain separate and false.'
$card.kind='causal_fresh_v13_prospective_card_v1';$card.protocol='causal-fresh-v13'
$card.declared_utc=$null;$card.frozen_utc=$null;$card.stage='DRAFT_NOT_FROZEN_NOT_EXECUTION_ADMITTED'
$card.model='CPB-v13';$card.instance='CPB-v13.market-01';$card.initial_quality_label='CPB-v13.market-01.init'
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
$card.mechanism.gradient_interpretation='V11 objective and attention-coupled raw centered variance unchanged. Only dispersion projection input becomes sqrt(v+2^-10)-1/32 in actual hidden dtype, add/sqrt/subtract order. Upstream beta=c/(2*actual_root) retains full actual two-sum F32 residual center correction and live hidden,attention-score and position routes. Shared globalclip1/singleAdamW couples future states; no validation-cause claim.'
$card.training.objective='R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v + log(v))];coefficients[1,1];supported operands sanitized and promoted separately to CUDAF64,unchanged hierarchical reduction. Only total_loss backpropagates;ordinary reporting NLL is a detached copy.'
$card.training.fixed_query_NLL='Ordinary Gaussian NLL from the detached reporting copy;separate Huber mean and detached-error variance objective components retain V9 routes'
$card.training.optimizer_coupling='One global norm2 clip1 and one AdamW;shared scaling,moments and future residuals couple optimization. No optimizer independence or validation-cause claim.'
$card.training.raw_load_counter_contract=[ordered]@{successful_updates=2048;attempts_min=2048;attempts_max=4096;calibration_loader_calls=512;initial_and_final_query_loader_calls=36;raw_load_calls_formula='512 + attempted_batches + 36';skipped_attempts='attempted_batches - new_causal_updates;derived,not a fabricated summary field'}
$card.mechanism.kind=$policy;$card.mechanism.temporal_pooling_policy=$policy
$card.mechanism.temporal_pooling_mix=@(0.5,0.5);$card.mechanism.empty_local_bin_policy=$empty
$card.mechanism.architecture='Both actual temporal paths retain V11 half-local/half-global attention and exact two-sum means. Raw centered variance uses that attention mixture/center. Only last4 inputs per bin use zero-based regularized stddev sqrt(rawvariance+2^-10)-1/32 before unchanged four-row projection/bias. Rawroot is unmasked1/32 at zero/absent; transformed dispersion0 there;whole8 served absent slice zero AFTER bias. Same32x64/32bias,registration/draws,global99->64->32/content20;native40 unchanged.'
$card.mechanism.controlled_mechanism='One V11 SOURCE amplitude-transform hypothesis; no uniform support, detached attention, clamp, RMS, Bessel correction, new parameter/buffer/draw or fitted regularizer. tau2^-10 and offset1/32 are fixed binary-exact constants; no sweep or rescue. Actual root/sqrt derivative is live even if tiny transformed F32 output rounds to zero.'
$card.mechanism.capacity_confounds='Same V11 shapes/counts/native40/D40heads and fresh draws. Transform changes feature scale,gradients,extra arithmetic and potentially all shared clipped AdamW trajectories. No equal-compute,information-importance or retention guarantee;only native20:40 DATA+five whole statistics pair. Native36 controls have different capacity/head seeds.'
$card.mechanism.allowed_typed_resume='Only fresh V13 lineage. Existing temporal/objective fields plus UTF8 String temporal_dispersion_transform_policy and Float64 vector temporal_dispersion_regularizer shape[1]=2^-10 at inference root/payload and TRAIN root/encoder/encoder.payload. Reject old/missing/wrong dtype/shape/value,including0-D/F32/otherlength. Preserve exact optimizer/counter/stream continuation at512.'
$card.mechanism.fresh_initialization_order='Same V11 parameter registration/shapes/RNG draw order under943001;all294752 learned values,optimizer and TRAIN statistics fresh. A single fresh export tensor is shared by both pooling routes;existing structure never means imported trained weights.'
$card.mechanism.empty_bin_policy='Original saved position membership. All-padding rows finite zero; all empty anchored bins zero the entire corresponding export slice after its one bias even when global history is nonempty; invalid hidden values cleared before pooling.'
$card.mechanism.causal_endpoint_semantics='Every slice is a coordinate of the embedding at the current requested cutoff t,not an embedding stamped at its bin endpoint. Shared whole-history uses only valid states at or before t. Later-in-window past may enter older anchored slices only within this endpoint embedding. Unchanged causal Transformer/backward relation bank and exact future-value AND future-mask invariance at all cutoffs are mandatory.'
$card.mechanism.same_export_weights_means='One single freshly initialized export tensor shared across both pooling routes;unchanged shape/registration/draw order;never imported trained values'
$card.training.recipe_id='causal-v13-attention-mean4-zero-based-regularized-stddev4-v11-objective-v1'
$card.training.temporal_pooling_policy=$policy;$card.training.temporal_pooling_mix=@(0.5,0.5);$card.training.empty_local_bin_policy=$empty
$card.training.component_evidence='All2048 successful scalar rows,8192 whole-bin rows,16384 mean/regularized_stddev subgroup rows,2048 gauge and Hadamard rows. Actual1/2048 existing-forward witness schema is externally source-reviewed below;raw variance,unmasked actual root,transformed dispersion retained. Actual2048 uses pre-step2047 coefficients. No extra forward or full-encoder/norm reconstruction.'
$card.quality.primary_methods=@('Raw data — no encoder (full256)','Raw data — no encoder (recent128)','CPB-v13.market-01.init','CPB-v13.market-01')
$card.quality.no_historical_encoder_rows_in_primary_tables=$true
$card.quality.primary_clean_control_rows_only=$false
$card.quality.clean_baseline_control=$controls.reuse_only
$card.quality.primary_report='Detailed producer tables contain8 saved raw-control market rows plus8 new own initial/final market rows=16,with four task tables/four TEMPO panels,physical errors,three neural repetitions,query components and cost. Separate comparison contains10 saved clean final encoders plus own initial/final=12 methods/48 market rows. Pending producer trust stays unchanged until independent audits/pairings and separate join. No automatic winner/application selection.'
$card.quality.saved_clean_comparison_methods=@('CPB-v3.market-01','CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01','CPB-v10.market-01','CPB-v11.market-01','CPB-v12.market-01','CPB-v13.market-01.init','CPB-v13.market-01')
$card.advancement.policy='All22 exact reviewed strongest V3–V11 conditions plus mandatory full trust/coverage;raw producer numeric/coverage and pending trust remain separate. No automatic winner or selection.'
$card.advancement.goal_protocol=$goalPin;$card.advancement.criteria=$goal.criteria;$card.advancement.criterion_count=22
$card.criteria.quality_interpretation='All22 literal866dec criterion objects copied unchanged from the passed723-check externally applied strongest-V3–V11 protocol;five V11 floors are protected. Historical counts retain owncards;no TRAIN NLL gate. Regularized-stddev amplitude hypothesis only;no validation-cause or improvement claim.'
$card.criteria.establishment=$establishment
$card.criteria.trust_admission=@(
 'All76 learned tensors/294752 values and optimizer/statistics fresh under identical V11 constructor draws/source0;75 identifiable tensors+sole observed common score-bias gauge;capacity is not every-scalar learning',
 'Four export/position/eight mean-regularized_stddev subgroups and three Hadamard directions need actual CUDA data-gradient/movement events;weight decay is not data learning',
 'All128cutoffs/2350suffix/30internal cases initial/final: future values AND masks cannot alter native40;past-positive interventions and current-endpoint semantics',
 'All temporal fields plus transform String and strict Float64[1]tau at allfive typed nodes,missing/old/dtype/shape/value negatives and exact typed512resume',
 'Exact exclusive source/card/descriptor/compiler/input/binary closure reviewed/frozen before any execution',
 'Source-derived attention-coupled regularized-stddev full-expression F32-aware adjoints/finite-difference fixtures,nonfinite/zero/tiny/large/empty/sparse/constant/equal-mean support cases;no V12 uniform-score-zero expectation',
 'Exactly13 saved V11 DATA references:feature native20:40 only plus five whole fresh statistics;content0:20/whole40 unpaired;new own initial heads,no old model/head execution',
 'Independent TRAIN/quality audits,fullcoverage/finite metrics and separate trust join;all22 strongest866 conditions before advancement;selection separatefalse'
)

$card.input_manifest=Get-FilePin 'doc/causal_fresh_v13/input-manifest.json'
$card.baseline_preservation=Get-FilePin 'doc/causal_fresh_v12/final-preservation.json'
$card.latest_frozen_source_preservation=$latestFrozenSource
$card.pre_execution_preservation=Get-FilePin 'doc/causal_fresh_v13/preservation-before.json'
$card.saved_TRAIN_diagnosis=[ordered]@{
 card=(Get-FilePin 'doc/CAUSAL_V11_V12_TRAIN_DIAGNOSIS_CARD.ADMITTED.json');result=(Get-FilePin 'doc/causal_v11_v12_train_diagnosis/diagnosis.json')
 independent_audit=(Get-FilePin 'doc/causal_v11_v12_train_diagnosis/independent-audit.json');human_report=(Get-FilePin 'doc/CAUSAL_V11_V12_TRAIN_DIAGNOSIS.md')
 completion=(Get-FilePin 'doc/causal_v11_v12_train_diagnosis/completion.json');human_report_root_review=(Get-FilePin 'doc/causal_v11_v12_train_diagnosis/report-root-review-001.json')
 scope='Completed saved V11/V12 TRAIN diagnosis:69952 first numerical V12 audit checks,2202 metadata completion,zero mismatches;four cached V11 objects reuse original68264 only as historical context. Failed001 metadata-only/no-values state and copied one-line repair preserved. No model/head/refit/heldout/trust replay.'
 interpretation='TRAIN diagnosis selected no encoder or validation cause. Root separately proposed V11 SOURCE regularized-stddev amplitude transform after completion. Current V12/114 closure distinct from frozen866 V11/109 ancestry.'
}
$card.comparison_control=$controls;$card.scoped_reference='CPB-v11.market-01'
$card.scientific_recipe_parent='CPB-v11 source only;fresh weights/optimizer/statistics,not trained ancestry'
$card.pairing_reference='CPB-v11.market-01 saved initial DATA/statistics only'
$card.scoped_references=@('CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01','CPB-v10.market-01','CPB-v11.market-01','CPB-v12.market-01')
$card.retention_independent_review=$reviewPin;$card.retention_application_admission=$admissionPin
$card.geometry_override='Native40/D40heads/counts/draws/objective/globalclip unchanged from fresh V11 source. Attention-centered raw variance remains;only dispersion amplitude becomes sqrt(v+2^-10)-1/32. Content20 unpaired;20:40 DATA+five whole stats against V11 only.'
$card.prospective_descriptor='code/encoders/causal_patch_bottleneck_mae/versions/CPB-v13.json'
$card.initial_pairing.reference='CPB-v11.market-01 initial0 DATA/statistic artifacts only'
$card.initial_pairing.native_coordinates_compared=20;$card.initial_pairing.native_coordinate_slice=@(20,40)
$card.initial_pairing.unpaired_learned_content_slice=@(0,20);$card.initial_pairing.content20_equality_required=$false
$card.initial_pairing.whole_native40_equality_required=$false;$card.initial_pairing.statistic_numeric_equality_required=$true
$card.initial_pairing.initial_scores_reused=$false;$card.initial_pairing.new_own_initial_fixed_heads=$true
$v11Mirror=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v11/complete-mirror-attempt-001.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($reference in $card.initial_pairing.references){
 $container=$reference.reference_data_archive.path.Replace('causal-fresh-v9/','causal-fresh-v11/').Replace('cpb_v9-','cpb_v11-')
 $matches=@($v11Mirror.files|Where-Object { $_.container_path -eq $container })
 if($matches.Count-ne1){throw ('One exact V11 initial DATA pin required: '+$container)}
 $actual=Get-FilePin $matches[0].path
 if($actual.sha256-ne$matches[0].sha256-or$actual.bytes-ne$matches[0].bytes){throw 'V11 initial DATA reference changed'}
 $reference.reference_data_archive=[ordered]@{path=$container;bytes=$actual.bytes;sha256=$actual.sha256}
 $reference.candidate_relative_to_attempt=$reference.candidate_relative_to_attempt.Replace('cpb_v11-','cpb_v13-')
 $reference.comparison='Feature value matrices compare20:40 only;valid/IDs/labels and five fresh-statistic archives compare whole declared numeric tensors. Content20 intentionally unpaired;new own initial heads,no old model/head execution.'
 if($reference.Contains('feature_value_coordinate_slice')){$reference.feature_value_coordinate_slice=@(20,40)}
}
$card.trust_receipt_objective_fields.temporal_pooling_policy=$policy
$card.trust_receipt_objective_fields.temporal_pooling_mix=@(0.5,0.5)
$card.trust_receipt_objective_fields.empty_local_bin_policy=$empty
$card.trust_receipt_field_applicability=[ordered]@{encoder_gradient_kind='Main training summary only;not an invented engineering/savedproof/fresh selector';gradient_policy='Strict typed records/engineering/training/savedproofs;absent only in constructor fresh-initialization.json';direct_gradient_routes='Mandatory common objective JSON in all declared records including fresh constructor'}
$card.mechanism_proof_contract.PASS_prefix='PASS CPB-v13 huber1-mean-plus-detached-error-gaussian-variance-v1:'
foreach($key in @('direct_joint_NLL_mean_gradient_live','direct_joint_NLL_uncertainty_gradient_live','joint_Gaussian_variance_counted_once','total_equals_Huber_plus_one_joint_Gaussian','independent_composite_F64_finite_differences')){$card.mechanism_proof_contract.Remove($key)}
$card.mechanism_proof_contract.direct_NLL_mean_gradient_zero=$true
$card.mechanism_proof_contract.manual_parity_scope='Unchanged V11 Huber1 mean plus detached-error Gaussian conditional-variance objective;manual native40 partial adjoints,direct cross-path zeros,live head/projection and shared globalclip coupling'
$card.mechanism_proof_contract.log_tokens=@('manual composite-objective native40 gradient parity','detached-error NLL mean gradient zero','synthetic optimizer steps=1')
$temporal=$plan.temporal_pooling_proof_contract
if($temporal.status-ne'DRAFT_source_owner_frozen_independent_review_before_any_execution'-or$temporal.binary-ne'test_cpb_v13_temporal_pooling'-or$temporal.temporal_pooling_policy-ne$policy-or$temporal.PASS_prefix-ne('PASS CPB-v13 '+$policy+':')-or$temporal.required_binary_count-ne$plan.static_build.binaries-or$temporal.required_TU_count-ne$plan.static_build.translation_units){throw 'Exact source-derived prospective temporal proof contract required'}
$proof=$temporal.required_proof_fields
if($null-eq$proof-or$proof.model-ne'CPB-v13'-or$proof.temporal_dispersion_transform_policy-ne'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'-or$proof.temporal_dispersion_regularizer-ne0.0009765625-or$temporal.log_tokens.Count-lt1){throw 'Actual source-derived V13 transform proof fields/tokens required'}
if($temporal.source_review.path-ne$plan.independent_training_auditor_source_review.path-or$temporal.source_review.sha256-ne$plan.independent_training_auditor_source_review.sha256){throw 'Actual independently reviewed family/TRAIN-auditor authority must bind fixture contract'}
Verify-Pin $temporal.source_review;Verify-Pin $temporal.fixture_source
if($temporal.fixture_source.path-ne'code/research/causal_fresh_v13/tests/v13_temporal_pooling_test.cpp'-or@($plan.owner_sources|Where-Object{$_.path-eq$temporal.fixture_source.path-and$_.bytes-eq$temporal.fixture_source.bytes-and$_.sha256-eq$temporal.fixture_source.sha256}).Count-ne1){throw 'Fixture contract must bind exact complete owned fixture source'}
$fixtureText=Get-Content -LiteralPath (Join-Path $root $temporal.fixture_source.path) -Raw
foreach($token in $temporal.log_tokens){if($token-isnot[string]-or-not$token-or-not$fixtureText.Contains($token)){throw 'Every frozen proof token must occur in independently reviewed actual fixture source'}}
foreach($key in $proof.Keys){if($key-match'uniform|dual_support'){throw 'No inherited V12 uniform proof expectation'}}
foreach($field in @('mean4_regularized_stddev4_export_layout','full_attention_dependent_regularized_moment_adjoints','live_dispersion_score_gradient','raw_variance_root_dispersion_trace','zero_tiny_large_analytic_dispersion_adjoints','supported_nonfinite_and_variance_overflow_refused','absent_local_huge_singleton_sanitized','captured_F32_residual_correction','raw_root_absent_offset_exception')){if($proof[$field]-isnot[bool]-or-not$proof[$field]-or$temporal.log_tokens-notcontains($field+'=true')){throw ('Actual new fixture flag/token: '+$field)}}
if($proof.direct_pool_kernel_calls-ne51-or$proof.F64_finite_difference_coordinates-ne18-or$proof.F32_finite_difference_coordinates-ne18-or$proof.isolated_autograd_grad_calls-ne4-or$proof.transform_derivative_upper_bound-ne16){throw 'Actual source-derived51/18/18/4/bound16 fixture contract required'}
if($plan.engineering_typed_negative_cases-ne636){throw 'Final source-derived exact636 typed negative cases required'}
$card.engineering_typed_negative_cases=636
$fixtureNumeric=[ordered]@{backbone_encode_calls=1;public_encoder_calls=0;head_fits=0;synthetic_optimizer_steps=1;manual_F32_absolute_tolerance=3e-5;manual_F32_relative_tolerance=3e-5;F64_finite_difference_step=1e-5;F64_finite_difference_absolute_tolerance=2e-8;F64_finite_difference_relative_tolerance=2e-8;F32_finite_difference_step=2e-3;F32_finite_difference_absolute_tolerance=2e-3;F32_finite_difference_relative_tolerance=2e-3;softmax_gauge_tolerance=2e-6}
foreach($key in $fixtureNumeric.Keys){if($proof[$key]-ne$fixtureNumeric[$key]){throw ('Exact source-derived fixture tolerance/count: '+$key)}}
$card.temporal_pooling_proof_contract=$temporal
$card.temporal_learning_receipt_contract.kind='cpb_v13_actual_temporal_pooling_learning_v1'
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
$card.fresh_initialization_JSON_exception.required_other_fields='All objective/delta/coefficients/direct_gradient_routes/detachtrue/reportingNLLfalse and temporalpolicy/partition/mix/empty fields plus75/76gauge fields remain mandatory;gradient_policy absent only in this constructor receipt'
$card.source_contract_status='DRAFT_NOT_ADMITTED_actual_owner_counts_and_independent_source_review_pending'
$card.source_admission='No AST/build/test/CUDA/model/head/body execution authorized by this draft. Exact final owner source counts/compiler/TU/binary closure and independent preflight must be frozen prospectively,without a source/card hash cycle.'
$card.source_pins_after_owner_freeze=$plan.owner_sources
foreach($key in @('independent_static_helper_review','independent_static_helper_review_amendment','source_review_scope')){$card.Remove($key)}
$card.source_counts=[ordered]@{family_inside=@($ownerFiles|Where-Object{$_.FullName.StartsWith((Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v13')+[IO.Path]::DirectorySeparatorChar)}).Count;family_external_header=1;research=@($ownerFiles|Where-Object{$_.FullName.StartsWith((Join-Path $root 'code/research/causal_fresh_v13')+[IO.Path]::DirectorySeparatorChar)}).Count;translation_units=$plan.static_build.translation_units;binaries=$plan.static_build.binaries;capsule_sources=$null;status='SOURCE_ONLY_actual_owner_path_set_bound_capsule_count_derived_at_freeze'}
$card.draft_copy_provenance=[ordered]@{
 kind='full_physical_V11_scientific_card_and_V12_infrastructure_SOURCE_then_owned_V13_regularized_stddev_edits'
 card=(Get-FilePin 'doc/CAUSAL_FRESH_V11_CARD.json');markdown=(Get-FilePin 'doc/CAUSAL_FRESH_V11_CARD.md')
 direct_orchestration_parent=(Get-FilePin 'doc/causal_fresh_v12/run-container.ps1')
 repaired_native_Docker_ancestry=(Get-FilePin 'doc/causal_fresh_v8/run-container-recovery-001.ps1')
 parent_copies=$copied;old_files_modified=$false;source_only=$true
}
$card.geometry.temporal_pooling_policy=$policy;$card.geometry.temporal_pooling_partition=@(4,16,32,4,8);$card.geometry.temporal_pooling_mix=@(0.5,0.5);$card.geometry.empty_local_bin_policy=$empty
foreach($record in @($card.geometry,$card.mechanism,$card.training,$card.trust_receipt_objective_fields,$card.temporal_pooling_proof_contract,$card.temporal_learning_receipt_contract,$proof)){
 $record.temporal_moment_policy='componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1'
 $record.temporal_export_layout='per-original-bin-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1'
 $record.temporal_moment_partition=@(4,4)
 $record.temporal_dispersion_transform_policy='sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'
 $record.temporal_dispersion_regularizer=0.0009765625
}
$card.actual_step_temporal_witness_contract=$plan.actual_step_temporal_witness_contract
if($card.actual_step_temporal_witness_contract.total_fields-ne60-or$card.actual_step_temporal_witness_contract.base_fields-ne23-or$card.actual_step_temporal_witness_contract.parameter_arrays-ne12-or$card.actual_step_temporal_witness_contract.temporal_trace_fields-ne25-or$card.actual_step_temporal_witness_contract.extra_encoder_calls-ne0){throw 'Actual independently reviewed writer schema56+4=60 required'}
$expectedTransformFields=@('temporal_pool_bin_regularized_root','temporal_pool_bin_dispersion','temporal_contextual_pool_bin_regularized_root','temporal_contextual_pool_bin_dispersion')
if(Compare-Object @($card.actual_step_temporal_witness_contract.dispersion_transform_trace_fields|Sort-Object) @($expectedTransformFields|Sort-Object)){throw 'Exact raw-root/transformed-dispersion same-forward witness names'}
$card.mechanism.raw_root_empty_semantics='Actual raw root is unmasked1/32 at zero/absent/allpadding;transformed dispersion0 at zero;served absent whole8 slice exact0 after bias. Not every trace is zero on padding.'
$card.results=[ordered]@{status='unmeasured';quality=$null;trust=$null;winner_selection_made=$false;application_selection_changed=$false}
$card.parent_goal_preserved=Get-FilePin 'doc/clean_causal_encoder_search_goal_v9_retention.json'
$card.historical_V9_reporting_recovery=[ordered]@{card=(Get-FilePin 'doc/causal_fresh_v9/reporting-recovery-card-001.json');receipt=(Get-FilePin 'doc/causal_fresh_v9/reporting-recovery-receipt-001.json');failure_mirror=(Get-FilePin 'doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json');scope='HistoricalV9 reporting-only/no-science-replay evidence;not a V13 recovery or recipe change'}
$card.declaration_input_plan=[ordered]@{path=$planPath.Substring($root.Length+1).Replace([char]92,[char]47);bytes=(Get-Item -LiteralPath $planPath).Length;sha256=$InputPlanSHA256}
$card.reviewed_orchestration_provenance=$orchestration
$card.owned_descriptor_draft=$plan.descriptor_source
Verify-Pin $plan.descriptor_source
$descriptor=Get-Content -LiteralPath (Join-Path $root $plan.descriptor_source.path) -Raw|ConvertFrom-Json -AsHashtable
if($descriptor.version-ne'CPB-v13'-or$descriptor.instance-ne'CPB-v13.market-01'-or$null-ne$descriptor.card-or$descriptor.architecture_copy_parent-ne'CPB-v11'-or$descriptor.temporal_pooling_policy-ne$policy-or$descriptor.temporal_moment_policy-ne$card.mechanism.temporal_moment_policy-or$descriptor.temporal_dispersion_transform_policy-ne'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'-or$descriptor.temporal_dispersion_regularizer-ne0.0009765625-or$descriptor.objective_policy-ne$objective-or$descriptor.gradient_policy-ne$gradient-or$descriptor.retention_protocol.bytes-ne$goalPin.bytes-or$descriptor.retention_protocol.sha256-ne$goalPin.sha256-or$descriptor.retention_protocol.path-ne$goalPin.path){throw 'Actual complete owned descriptor draft must match the independently reviewed V13 source/criteria declaration;never rewrite it during metadata preparation'}
Write-JsonNew 'doc/CAUSAL_FRESH_V13_CARD.json' $card
[IO.File]::Copy((Join-Path $root $plan.card_markdown_template.path),(Join-Path $root 'doc/CAUSAL_FRESH_V13_CARD.md'),$false)
Write-JsonNew 'doc/causal_fresh_v13/draft-metadata-copy-receipt.json' ([ordered]@{status='draft_prepared_not_execution_admitted';controls=$controls;goal=$goalPin;review=$reviewPin;admission=$admissionPin;raw_controls=$raw;parent_copies=$copied;owned_descriptor_reused_unchanged=$plan.descriptor_source;initial_DATA_references=13;checkpoint_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
Get-FilePin 'doc/CAUSAL_FRESH_V13_CARD.json'|ConvertTo-Json -Depth 4

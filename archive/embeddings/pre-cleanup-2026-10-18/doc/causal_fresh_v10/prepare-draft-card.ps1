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
$draftPath=Join-Path $root 'doc/CAUSAL_FRESH_V10_CARD.json'
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
if($plan.kind-ne'cpb_v10_prospective_declaration_input_plan_v1'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true-or$plan.status-ne'root_reviewed_source_and_completed_context'){throw 'Only a root-reviewed completed-context plan can prepare a draft'}
foreach($required in @('doc/CAUSAL_FRESH_V9_CARD.json','doc/CAUSAL_FRESH_V9_CARD.md','doc/causal_fresh_v9/final-preservation.json','doc/causal_fresh_v9/protocol-preflight.json','doc/causal_fresh_v9/complete-mirror-attempt-001.json','doc/causal_fresh_v9/reporting-recovery-card-001.json','doc/causal_fresh_v9/reporting-recovery-receipt-001.json','doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json','doc/clean_causal_encoder_search_goal_v9_retention.json','doc/causal_v9_retention_review/independent-review.json','doc/causal_v9_retention_review/applied-future-protocol.json','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json','doc/causal_v8_v9_train_diagnosis/diagnosis.json','doc/causal_v8_v9_train_diagnosis/independent-audit.json','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md','doc/causal_v8_v9_train_diagnosis/completion.json','doc/causal_v8_v9_train_diagnosis/human-report-root-review.json','doc/clean_causal_encoder_research_ledger.json','code/encoders/causal_patch_bottleneck_mae/versions/current.json','doc/causal_fresh_v10/family-source-provenance.json','doc/causal_fresh_v10/evaluation-source-provenance.json','code/research/causal_fresh_v10/training-helper-provenance.json')){
 if(@($plan.protected_context|Where-Object{$_.path-eq$required}).Count-ne1){throw ('One exact reviewed context pin required: '+$required)}
}
foreach($pin in $plan.protected_context){Verify-Pin $pin}
if(@($plan.protected_context|Where-Object{$_.path-eq'doc/causal_fresh_v9/CPB-v9.descriptor-draft.json'}).Count-ne1){throw 'Exact physical V9 descriptor draft template must be externally plan-bound before parse'}
foreach($version in 3..9){if(@($plan.protected_context|Where-Object{$_.path-eq('doc/results/causal_fresh_v'+$version+'/table_rows.json')}).Count-ne1){throw 'Every saved historical final control must be externally plan-bound'}}
foreach($version in 6..9){if(@($plan.protected_context|Where-Object{$_.path-eq('doc/causal_fresh_v'+$version+'/final-establishment.json')}).Count-ne1){throw 'Every separate historical establishment must be externally plan-bound'}}
Verify-Pin $plan.independent_training_auditor_source_review
$trainingReview=Get-Content -LiteralPath (Join-Path $root $plan.independent_training_auditor_source_review.path) -Raw|ConvertFrom-Json -AsHashtable
if($trainingReview.kind-ne'cpb_v10_family_and_training_auditor_independent_static_review_v1'-or$trainingReview.status-ne'passed'-or$plan.independent_training_auditor_source_review.sha256-ne'124ffe4801072e1877f58c40608cbb2f865f9044ec7683bccc8d3b4d7d1c4eee'){throw 'Actual independently passed V10 training-auditor source review required'}
if($plan.static_build.binaries-ne7-or$plan.static_build.translation_units-ne11){throw 'Final source owners declare seven binaries and eleven TUs; actual build admission remains separate'}
$orchestration=Get-FilePin 'doc/causal_fresh_v10/orchestration-provenance.json'
if($orchestration.sha256-ne$OrchestrationProvenanceSHA256){throw 'Exact independently reviewed orchestration provenance required'}
$workflow=Get-Content -LiteralPath (Join-Path $root $orchestration.path) -Raw|ConvertFrom-Json -AsHashtable
if($workflow.files.Count-ne10){throw 'Exactly ten owned orchestration helpers required'}
foreach($binding in $workflow.files){Verify-Pin $binding.parent;Verify-Pin $binding.owned}
$ownerFiles=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v10') -Recurse -File)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v10.h'))+@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v10') -Recurse -File)
$observedPaths=@($ownerFiles|ForEach-Object{$_.FullName.Substring($root.Length+1).Replace([char]92,[char]47)}|Sort-Object)
if(Compare-Object $observedPaths @($plan.owner_sources.path|Sort-Object)){throw 'Plan must bind the complete actual family/research path set'}
foreach($pin in $plan.owner_sources){Verify-Pin $pin}
foreach($output in @('doc/CAUSAL_FRESH_V10_CARD.md','doc/causal_fresh_v10/CPB-v10.descriptor-draft.json','doc/causal_fresh_v10/draft-metadata-copy-receipt.json','doc/causal_fresh_v10/preservation-before.json')){if(Test-Path -LiteralPath (Join-Path $root $output)){throw ('Preserve existing output: '+$output)}}
Verify-Pin $plan.card_markdown_template
$preservation=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/final-preservation.json') -Raw|ConvertFrom-Json -AsHashtable
if($preservation.status-ne'passed'-or$preservation.critical_checkpoint_pins.Count-ne99){throw 'Actual final V9 preservation with99 existing critical checkpoints required'}
foreach($pin in $preservation.critical_checkpoint_pins){$repo=$pin.repository;if(-not$repo){if(-not$pin.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')){throw 'Only five historical V5 pins may omit repository'};$repo='cuwacunu_embedding'};Verify-Pin $pin $repo}
$oldCard=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V9_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
if($oldCard.baseline_frozen_files.Count-ne47){throw '47 physical V3 source files required per repository'}
foreach($pin in $oldCard.baseline_frozen_files){foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){Verify-Pin $pin $repo}}
$oldSource=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/protocol-preflight.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($pin in $oldSource.sources){Verify-Pin $pin}
Write-JsonNew 'doc/causal_fresh_v10/preservation-before.json' ([ordered]@{kind='cpb_v10_before_execution_file_preservation_v1';status='passed';created_utc=[DateTime]::UtcNow.ToString('o');input_plan=[ordered]@{path=$planPath.Substring($root.Length+1).Replace([char]92,[char]47);bytes=(Get-Item -LiteralPath $planPath).Length;sha256=$InputPlanSHA256};baseline_preservation=(Get-FilePin 'doc/causal_fresh_v9/final-preservation.json');critical_checkpoint_pins=$preservation.critical_checkpoint_pins;critical_checkpoint_files=99;baseline_sources_each_repository=47;baseline_source_pins=$oldCard.baseline_frozen_files;protected_context=$plan.protected_context;old_V9_source_files_verified=$oldSource.sources.Count;new_owner_sources=$plan.owner_sources;file_hash_inspection_only=$true;archive_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
$parentCopies='output/engineering/causal-fresh-v10-protocol-preparation/attempt-001/parent-copies'
$copied=@()
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V9_CARD.json' ($parentCopies+'/CAUSAL_FRESH_V9_CARD.json')
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V9_CARD.md' ($parentCopies+'/CAUSAL_FRESH_V9_CARD.md')
$card=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V9_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
$goal=Get-Content -LiteralPath (Join-Path $root 'doc/clean_causal_encoder_search_goal_v9_retention.json') -Raw|ConvertFrom-Json -AsHashtable
if((Get-FilePin 'doc/clean_causal_encoder_search_goal_v9_retention.json').sha256-ne'221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144'-or$goal.criteria.Count-ne22-or$goal.additional_train_NLL_gate-ne$false){throw 'Exact current22-condition/noTRAINNLL protocol required'}
$controls=[ordered]@{}
foreach($version in 3..9){$controls['v'+$version]=Copy-ExactMetadata ('doc/results/causal_fresh_v'+$version+'/table_rows.json') ('doc/causal_fresh_v10/baseline-control-v'+$version+'.json')}
foreach($version in 6..9){$controls['v'+$version+'_establishment']=Copy-ExactMetadata ('doc/causal_fresh_v'+$version+'/final-establishment.json') ('doc/causal_fresh_v10/baseline-v'+$version+'-establishment.json')}
$controls.reuse_only='Exact saved V3/V4/V5/V6/V7/V8/V9 final tables and separate V6/V7/V8/V9 full-trust receipts; preserve pending producer bytes; no old encoder/head execution'
$goalPin=Copy-ExactMetadata 'doc/clean_causal_encoder_search_goal_v9_retention.json' 'doc/causal_fresh_v10/goal-protocol.json'
$reviewPin=Copy-ExactMetadata 'doc/causal_v9_retention_review/independent-review.json' 'doc/causal_fresh_v10/goal-protocol-review.json'
$admissionPin=Copy-ExactMetadata 'doc/causal_v9_retention_review/applied-future-protocol.json' 'doc/causal_fresh_v10/goal-protocol-admission.json'
$raw=Copy-ExactMetadata 'doc/causal_fresh_v9/raw-controls.json' 'doc/causal_fresh_v10/raw-controls.json'
$input=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/input-manifest.json') -Raw|ConvertFrom-Json -AsHashtable
$input.kind='causal_fresh_v10_data_input_manifest_v1';$input.protocol='causal-fresh-v10';$input.model='CPB-v10.market-01'
Write-JsonNew 'doc/causal_fresh_v10/input-manifest.json' $input
$policy='original-position-four-bin-shared-global-half-mixture-v1'
$empty='zero-whole-export-slice-after-bias-v1'
$objective='huber1-mean-plus-joint-gaussian-nll-v1'
$gradient='huber1-mean-live;joint-gaussian-mean-live-and-uncertainty4-live;shared-global-clip1-v1'
$encoderGradient='Huber1-mean-gradient;joint-Gaussian-mean-and-variance;live-uncertainty4;shared-global-clip1'
$routes='Huber->parent32+mean4;Gaussian-mean->parent32+mean4;Gaussian-variance->uncertainty4;mean-reads0:36;variance-reads36:40;shared-global-clip1'
$establishment='Full trust including exact full-native40 initial/statistic pairings and independent training/quality audits establishes a measured candidate. Advancement additionally requires full coverage and all22 declared numeric conditions. Winner and application selection remain separate and false.'
$card.kind='causal_fresh_v10_prospective_card_v1';$card.protocol='causal-fresh-v10'
$card.declared_utc=$null;$card.frozen_utc=$null;$card.stage='DRAFT_NOT_FROZEN_NOT_EXECUTION_ADMITTED'
$card.model='CPB-v10';$card.instance='CPB-v10.market-01';$card.initial_quality_label='CPB-v10.market-01.init'
$card.execution_admitted=$false;$card.application_selection_changed=$false
$card.mechanism.Remove('mandatory_identifiable_learned_values')
$card.mechanism.mandatory_identifiable_tensor_value_capacity=294751
$card.mechanism.mandatory_identifiable_count_semantics='tensor numel capacity, not every-scalar gradient/movement proof'
$card.mechanism.per_scalar_nonzero_gradient_proof_claimed=$false
$card.independent_training_auditor_source_review=$plan.independent_training_auditor_source_review
foreach($key in @('prospective_freeze','declared_client_date')){$card.Remove($key)}
foreach($record in @($card.mechanism,$card.training,$card.trust_receipt_objective_fields)){
 $record.objective_policy=$objective;$record.gradient_policy=$gradient;$record.direct_gradient_routes=$routes
 $record.mean_huber_delta=1.0;$record.loss_coefficients=@(1.0,1.0);$record.variance_error_detached=$false;$record.reporting_nll_backpropagated=$false
 $record.shared_global_clip=$true;$record.optimizer_independence_claimed=$false
}
$card.training.encoder_gradient_kind=$encoderGradient;$card.trust_receipt_objective_fields.encoder_gradient_kind=$encoderGradient
$card.mechanism.gradient_interpretation='Huber1 contributes the clipped residual mean gradient and the ONE joint Gaussian contributes inverse-variance-weighted residual mean gradient plus its variance gradient. Native mean coordinates0:36 and uncertainty36:40 retain separate direct decoder reads. One globalclip1/singleAdamW and future residuals couple optimization;no optimizer independence or VAL-cause claim.'
$card.training.objective='R[Huber_delta1(mu-y)] + R[0.5*((mu-y)^2/v + log(v))]; ONE live joint Gaussian,coefficients[1,1];support sanitized,F32 operands promoted separately to CUDAF64 before subtraction,unchanged equal-example/equal-channel/supported-cell reduction. Only total_loss backpropagates;separate ordinary reporting NLL uses detached mean and variance.'
$card.training.fixed_query_NLL='Unchanged ordinary Gaussian NLL forward value from the detached reporting copy;the separate ONE joint Gaussian objective has live mean and variance routes'
$card.training.optimizer_coupling='One global norm2 clip1 and one AdamW;new inverse-variance mean gradient can alter globalclip scaling,moments and later distributions. Same scalar objective at identical distributions does not mean same optimization.'
$card.training.raw_load_counter_contract=[ordered]@{successful_updates=2048;attempts_min=2048;attempts_max=4096;calibration_loader_calls=512;initial_and_final_query_loader_calls=36;raw_load_calls_formula='512 + attempted_batches + 36';skipped_attempts='attempted_batches - new_causal_updates;derived,not a fabricated summary field'}
$card.mechanism.kind='huber1-mean-plus-joint-gaussian-nll-v1';$card.mechanism.temporal_pooling_policy=$policy
$card.mechanism.temporal_pooling_mix=@(0.5,0.5);$card.mechanism.empty_local_bin_policy=$empty
$card.mechanism.architecture='V9 causal contextual patch states; each nonempty anchored original-position bin forms half its local masked-softmax hidden plus half the shared whole-history masked-softmax hidden, before one application of that bin''s eight rows plus bias from a single freshly initialized32x64 export tensor. Empty local bins zero the whole eight-coordinate slice after bias. Concatenate32, then unchanged global mixing and content20+literal12+mean4+uncertainty4 native40.'
$card.mechanism.controlled_mechanism='Only the Gaussian objective mean route changes versus freshly constructed V9: retain Huber1 and replace detached-mean variance calibration with ONE joint Gaussian NLL live through mean and variance,coefficients[1,1]. Same forward/pooling/native40/parameter registration/RNG streams/globalclip1/singleAdamW/fixed heads; all weights,optimizer and TRAIN stats fresh,never imported.'
$card.mechanism.capacity_confounds='Same native40/counts/constructor order/forward/D40 heads as V9. Identical distribution scalars equal V9,while inverse-variance mean weighting,total gradients,one globalclip and AdamW moments can differ. Whole native40 initial DATA+five freshstats are paired; no causal/VAL/generalization inference. Relative V3/V4native36 readout width/seeds/capacity remain confounds.'
$card.mechanism.allowed_typed_resume='Only fresh V10 lineage; objective/gradient/layout plus temporal_pooling_policy,Long[5] partition,Double[2] mix in anchored_local/whole_history order,and empty_local_bin_policy at inference root/payload and TRAIN root/encoder/encoder.payload. Reject missing,wrong value/dtype/shape/order,old V9/V6 kinds; preserve all optimizer moments/counters/order/data identities.'
$card.mechanism.fresh_initialization_order='Same V9 parameter registration/shapes/RNG draw order under943001; all294752 learned values and optimizer state fresh. A single fresh export tensor is shared by the two pooling routes; existing never means imported trained weights.'
$card.mechanism.empty_bin_policy='Original saved position membership. All-padding rows finite zero; all empty anchored bins zero the entire corresponding export slice after its one bias even when global history is nonempty; invalid hidden values cleared before pooling.'
$card.mechanism.causal_endpoint_semantics='Every slice is a coordinate of the embedding at the current requested cutoff t,not an embedding stamped at its bin endpoint. Shared whole-history uses only valid states at or before t. Later-in-window past may enter older anchored slices only within this endpoint embedding. Unchanged causal Transformer/backward relation bank and exact future-value AND future-mask invariance at all cutoffs are mandatory.'
$card.mechanism.same_export_weights_means='One single freshly initialized export tensor shared across the two pooling routes;unchanged shape/registration/draw order;never V9/V6 trained values'
$card.training.recipe_id='causal-v10-huber1-plus-one-joint-gaussian-live-mean-variance-v1'
$card.training.temporal_pooling_policy=$policy;$card.training.temporal_pooling_mix=@(0.5,0.5);$card.training.empty_local_bin_policy=$empty
$card.training.component_evidence='All2048 successful CSV rows record mean_Huber_F64,joint_Gaussian_NLL_F64,total_objective_F64 and separate Gaussian_NLL_F64 reporting copy,global norm/factor. Actual pre-step update1/2048 witnesses use row_mean_huber,row_joint_gaussian_nll,row_total_objective. Unchanged8192 temporal group/2048 gauge/2048 Hadamard observations and same12 actual arrays;no extra encoder forwards. Failed eligibility attempts remain blank CSV rows and affect actual attempted/raw-load counters.'
$card.quality.primary_methods=@('CPB-v3.market-01','CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01','CPB-v10.market-01.init','CPB-v10.market-01')
$card.quality.no_historical_encoder_rows_in_primary_tables=$false
$card.quality.primary_clean_control_rows_only=$true
$card.quality.clean_baseline_control=$controls.reuse_only
$card.quality.primary_report='Seven saved clean final controls plus own newly exported/measured initial0/final2048;four separate market task tables/four TEMPO panels,physical errors,full neural repetitions/query components/cost. Pending runtime producers stay pending until independent audits/pairings and a separate trust join. No automatic winner/application selection.'
$card.advancement.policy='All22 exact reviewed strongest V3–V9 conditions plus mandatory full trust/coverage;raw producer numeric/coverage and pending trust remain separate. No automatic winner or selection.'
$card.advancement.goal_protocol=$goalPin;$card.advancement.criteria=$goal.criteria;$card.advancement.criterion_count=22
$card.criteria.quality_interpretation='Exact reviewed strongest V3–V9 retention22 protocol221f;no V1/V2/RPB floors/no TRAIN NLL gate. Historical counts retain owncards. Joint mean-gradient route changes totalgradient/globalclipping/AdamW dynamics;no cause or improvement assumed.'
$card.criteria.establishment=$establishment
$card.criteria.trust_admission=@(
 'All76learned tensors/294752values and optimizer freshly constructed with identical V9 draw order;fresh4096TRAIN input/bank statistics,source_updates0,no old model/statistic import',
 '75 identifiable tensors have actual CUDA nonzero data-gradient and movement events;294751 is tensor capacity,not every-scalar learning. Scalarpool_score.bias gauge remains registered/optimized/observed and is the sole exemption;weightdecay is not data learning',
 'Four export8-row groups and four within-bin centered position groups each show data-gradient/movement events and final changes. Each of three Hadamard between-bin directions has data-gradient+movement events and overall final between-bin projection changes;no each-final-coefficient gate',
 'All128cutoffs/2350suffix/30internal temporal cases initialized/trained and actual saved0/2048;future values AND masks/nonfinite suffix cannot change endpoint native40;past-positive interventions',
 'All four temporal typed fields validated at inference root/payload and TRAIN root/encoder/encoder.payload,including missing/wrong policy/mix/empty/partition/dtype/shape/order. Actual typed512resume exact and controlled engineering next-step parity',
 'Exclusive independently reviewed prospective card/descriptor/source/input/compiler/binary closure before any build/test/CUDA;actual file/binary/TU counts derived from frozen owners,not assumed',
 'New nonvacuous joint-objective CUDA fixture with independent composite F64 finite differences;unchanged separate manual half-mixture/adjoint/trace/empty/common-gauge/relative-bin sensitivity/four-group/three-between-direction CUDA fixture',
 'Exactly13 DATA/statistic initial pairings versus V9: eight feature value matrices whole0:40 with whole valid/IDs/labels;five fresh-statistic archives whole numeric tensors. Content0:20 is paired;new own initial heads;no old model/score replay',
 'Independent saved-training/quality arithmetic audits,all fullcoverage finite metrics and separate final-trust evidence join before any advancement claim'
)
$card.input_manifest=Get-FilePin 'doc/causal_fresh_v10/input-manifest.json'
$card.baseline_preservation=Get-FilePin 'doc/causal_fresh_v9/final-preservation.json'
$card.pre_execution_preservation=Get-FilePin 'doc/causal_fresh_v10/preservation-before.json'
$card.saved_TRAIN_diagnosis=[ordered]@{
 card=(Get-FilePin 'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json');result=(Get-FilePin 'doc/causal_v8_v9_train_diagnosis/diagnosis.json')
 independent_audit=(Get-FilePin 'doc/causal_v8_v9_train_diagnosis/independent-audit.json');human_report=(Get-FilePin 'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md')
 completion=(Get-FilePin 'doc/causal_v8_v9_train_diagnosis/completion.json')
 human_report_root_review=(Get-FilePin 'doc/causal_v8_v9_train_diagnosis/human-report-root-review.json')
 scope='Completed independent savedTRAIN-only V8/V9 diagnosis;bounded scalar/adjoint arithmetic,zero fullencoder/librarymodel/head/refit calls;all old99checkpoints/source preserved'
 interpretation='Diagnosis selected no new model;root separately chose one live-joint-Gaussian objective-route hypothesis after diagnosis completion. No causal or counterfactual claim.'
}
$card.comparison_control=$controls;$card.scoped_reference='CPB-v9.market-01'
$card.scoped_references=@('CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01')
$card.retention_independent_review=$reviewPin;$card.retention_application_admission=$admissionPin
$card.geometry_override='Native40,unchanged D40 head recipes/seeds/counts and constructor shapes/draws. V9 forward/globalclip1 remain fixed;only the mean route of the single Gaussian objective term becomes live alongside Huber1.'
$card.prospective_descriptor='code/encoders/causal_patch_bottleneck_mae/versions/CPB-v10.json'
$card.initial_pairing.reference='CPB-v9.market-01 initial0 DATA/statistic artifacts only'
$card.initial_pairing.native_coordinates_compared=40;$card.initial_pairing.native_coordinate_slice=@(0,40)
$card.initial_pairing.unpaired_learned_content_slice=@();$card.initial_pairing.content20_equality_required=$true
$card.initial_pairing.whole_native40_equality_required=$true;$card.initial_pairing.statistic_numeric_equality_required=$true
$card.initial_pairing.initial_scores_reused=$false;$card.initial_pairing.new_own_initial_fixed_heads=$true
$v9Mirror=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/complete-mirror-attempt-001.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($reference in $card.initial_pairing.references){
 $container=$reference.reference_data_archive.path.Replace('causal-fresh-v8/','causal-fresh-v9/').Replace('cpb_v8-','cpb_v9-')
 $matches=@($v9Mirror.files|Where-Object { $_.container_path -eq $container })
 if($matches.Count-ne1){throw ('One exact V9 initial DATA pin required: '+$container)}
 $actual=Get-FilePin $matches[0].path
 if($actual.sha256-ne$matches[0].sha256-or$actual.bytes-ne$matches[0].bytes){throw 'V9 initial DATA reference changed'}
 $reference.reference_data_archive=[ordered]@{path=$container;bytes=$actual.bytes;sha256=$actual.sha256}
 $reference.candidate_relative_to_attempt=$reference.candidate_relative_to_attempt.Replace('cpb_v9-','cpb_v10-')
 $reference.comparison='All declared numeric fields compare whole logical tensors,including all40 feature coordinates and all fresh-statistic tensors;not whole-file metadata equality. No old model/head execution.'
 if($reference.Contains('feature_value_coordinate_slice')){$reference.feature_value_coordinate_slice=@(0,40)}
}
$card.trust_receipt_objective_fields.temporal_pooling_policy=$policy
$card.trust_receipt_objective_fields.temporal_pooling_mix=@(0.5,0.5)
$card.trust_receipt_objective_fields.empty_local_bin_policy=$empty
$card.trust_receipt_field_applicability=[ordered]@{encoder_gradient_kind='Main training summary only;not an invented engineering/savedproof/fresh selector';gradient_policy='Strict typed records/engineering/training/savedproofs;absent only in constructor fresh-initialization.json';direct_gradient_routes='Mandatory common objective JSON in all declared records including fresh constructor'}
$card.mechanism_proof_contract.PASS_prefix='PASS CPB-v10 huber1-mean-plus-joint-gaussian-nll-v1:'
$card.mechanism_proof_contract.Remove('direct_NLL_mean_gradient_zero')
foreach($key in @('direct_joint_NLL_mean_gradient_live','direct_joint_NLL_uncertainty_gradient_live','joint_Gaussian_variance_counted_once','total_equals_Huber_plus_one_joint_Gaussian','independent_composite_F64_finite_differences')){$card.mechanism_proof_contract[$key]=$true}
$card.mechanism_proof_contract.manual_parity_scope='Huber1 plus ONE live joint Gaussian,with F64 masked support reductions,manual native40 gradients and independent composite scalar finite differences;variance counted once'
$card.mechanism_proof_contract.log_tokens=@('manual composite-objective native40 gradient parity','joint Gaussian mean and uncertainty live','variance counted once','independent composite F64 finite differences','synthetic optimizer steps=1')
$temporal=$card.temporal_pooling_proof_contract
$temporal.status='DRAFT_source_owner_frozen_independent_review_before_any_execution'
$temporal.temporal_pooling_policy=$policy;$temporal.temporal_pooling_mix=@(0.5,0.5);$temporal.empty_local_bin_policy=$empty
$temporal.PASS_prefix='PASS CPB-v10 '+$policy+':'
$temporal.log_tokens=@('manual original-position half-mixture pooling parity','four 8-row projection gradient and movement','three between-bin Hadamard contrast gradient and movement','empty-bin finite zero','relative-bin global sensitivity','common softmax gauge invariance','synthetic optimizer steps=1')
$temporal.required_binary_count=$plan.static_build.binaries;$temporal.required_TU_count=$plan.static_build.translation_units
$proof=$temporal.required_proof_fields;$proof.model='CPB-v10';$proof.temporal_pooling_policy=$policy
$proof.temporal_pooling_mix=@(0.5,0.5);$proof.empty_local_bin_policy=$empty
foreach($key in @('softmax_gauge_invariance','fixture_backbone_Model_encode_calls','fixture_direct_pooling_kernel_calls','public_Encoder_calls')){$proof.Remove($key)}
$proof.common_softmax_gauge_invariance=$true;$proof.shared_global_half_mixture=$true
$proof.relative_bin_global_sensitivity=$true;$proof.three_between_bin_contrasts_gradient_and_movement=$true
$proof.backbone_encode_calls=1;$proof.public_encoder_calls=0;$proof.direct_pool_kernel_calls=5
$temporal.binary='test_cpb_v10_temporal_pooling'
$card.temporal_learning_receipt_contract.kind='cpb_v10_actual_temporal_pooling_learning_v1'
$card.temporal_learning_receipt_contract.temporal_pooling_policy=$policy
$card.temporal_learning_receipt_contract.temporal_pooling_mix=@(0.5,0.5)
$card.temporal_learning_receipt_contract.empty_local_bin_policy=$empty
$card.temporal_learning_receipt_contract.between_bin_rows=2048
$card.temporal_learning_receipt_contract.receipt_artifacts=@('temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv','temporal-between-bin-position-learning.csv')
$card.temporal_learning_receipt_contract.summary_positive_flags=@($card.temporal_learning_receipt_contract.summary_positive_flags|Select-Object -Unique)
$card.temporal_learning_receipt_contract.between_bin_positions=[ordered]@{
 basis='hadamard-full16-signs-div4-v1';basis_signs=@(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1));full_position_divisor=4;position_bin_width=4;rows=2048
 columns=@('update','between_bin_gradient_l2','between_bin_movement_l2','contrast_0_gradient','contrast_1_gradient','contrast_2_gradient','contrast_0_movement','contrast_1_movement','contrast_2_movement')
 gradient_definition='observed-post-globalclip-data-gradient';F64_before_projection_and_subtraction=$true;parseval_identity_required=$true
 each_direction_positive_data_gradient_and_movement_events=$true;overall_final_projection_change_l2_positive=$true
 each_final_signed_coefficient_nonzero_required=$false;movement_includes_AdamW_weight_decay=$true;weight_decay_counted_as_data_learning=$false;extra_encoder_calls=0
}
$card.fresh_initialization_JSON_exception.required_other_fields='All objective/delta/coefficients/direct_gradient_routes/detachfalse/reportingNLLfalse and temporalpolicy/partition/mix/empty fields plus75/76gauge fields remain mandatory;gradient_policy absent only in this constructor receipt'
$card.source_contract_status='DRAFT_NOT_ADMITTED_actual_owner_counts_and_independent_source_review_pending'
$card.source_admission='No AST/build/test/CUDA/model/head/body execution authorized by this draft. Exact final owner source counts/compiler/TU/binary closure and independent preflight must be frozen prospectively,without a source/card hash cycle.'
$card.source_pins_after_owner_freeze=$plan.owner_sources
foreach($key in @('independent_static_helper_review','independent_static_helper_review_amendment','source_review_scope')){$card.Remove($key)}
$card.source_counts=[ordered]@{family_inside=@($ownerFiles|Where-Object{$_.FullName.StartsWith((Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v10')+[IO.Path]::DirectorySeparatorChar)}).Count;family_external_header=1;research=@($ownerFiles|Where-Object{$_.FullName.StartsWith((Join-Path $root 'code/research/causal_fresh_v10')+[IO.Path]::DirectorySeparatorChar)}).Count;translation_units=$plan.static_build.translation_units;binaries=$plan.static_build.binaries;capsule_sources=$null;status='SOURCE_ONLY_actual_owner_path_set_bound_capsule_count_derived_at_freeze'}
$card.draft_copy_provenance=[ordered]@{
 kind='full_physical_V9_card_and_repaired_native_Docker_workflow_copies_then_owned_V10_edits'
 card=(Get-FilePin 'doc/CAUSAL_FRESH_V9_CARD.json');markdown=(Get-FilePin 'doc/CAUSAL_FRESH_V9_CARD.md')
 direct_orchestration_parent=(Get-FilePin 'doc/causal_fresh_v9/run-container.ps1')
 repaired_native_Docker_ancestry=(Get-FilePin 'doc/causal_fresh_v8/run-container-recovery-001.ps1')
 parent_copies=$copied;old_files_modified=$false;source_only=$true
}
$card.geometry.temporal_pooling_policy=$policy;$card.geometry.temporal_pooling_partition=@(4,16,32,4,8);$card.geometry.temporal_pooling_mix=@(0.5,0.5);$card.geometry.empty_local_bin_policy=$empty
$card.results=[ordered]@{status='unmeasured';quality=$null;trust=$null;winner_selection_made=$false;application_selection_changed=$false}
$card.parent_goal_preserved=Get-FilePin 'doc/clean_causal_encoder_search_goal_v8_retention.json'
$card.historical_V9_reporting_recovery=[ordered]@{card=(Get-FilePin 'doc/causal_fresh_v9/reporting-recovery-card-001.json');receipt=(Get-FilePin 'doc/causal_fresh_v9/reporting-recovery-receipt-001.json');failure_mirror=(Get-FilePin 'doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json');scope='HistoricalV9 reporting-only/no-science-replay evidence;not a V10 recovery or recipe change'}
$card.declaration_input_plan=[ordered]@{path=$planPath.Substring($root.Length+1).Replace([char]92,[char]47);bytes=(Get-Item -LiteralPath $planPath).Length;sha256=$InputPlanSHA256}
Write-JsonNew 'doc/CAUSAL_FRESH_V10_CARD.json' $card
$descriptor=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/CPB-v9.descriptor-draft.json') -Raw|ConvertFrom-Json -AsHashtable
$descriptor.version='CPB-v10';$descriptor.instance='CPB-v10.market-01';$descriptor.card=$null
$descriptor.physical_version_root='code/encoders/causal_patch_bottleneck_mae/versions/v10';$descriptor.public_entry='code/encoders/causal_patch_bottleneck_mae/versions/v10/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v10.h'
$descriptor.architecture_copy_parent='CPB-v9';$descriptor.objective_policy=$objective;$descriptor.gradient_policy=$gradient;$descriptor.direct_gradient_routes=$routes
$descriptor.variance_error_detached=$false;$descriptor.reporting_nll_backpropagated=$false;$descriptor.retention_protocol=$goalPin
$descriptor.mechanism='Same fresh V9 half-mixture forward/draw order/native40/globalclip1/heads;only lossgraph changes to Huber1 plus ONE live joint Gaussian,coefficients[1,1]. No old trained values imported.'
$descriptor.initial_pairing='Whole native40 eight DATA archives plus five fresh-statistic tensors against saved V9initial;own new initial heads'
Write-JsonNew 'doc/causal_fresh_v10/CPB-v10.descriptor-draft.json' $descriptor
[IO.File]::Copy((Join-Path $root $plan.card_markdown_template.path),(Join-Path $root 'doc/CAUSAL_FRESH_V10_CARD.md'),$false)
Write-JsonNew 'doc/causal_fresh_v10/draft-metadata-copy-receipt.json' ([ordered]@{status='draft_prepared_not_execution_admitted';controls=$controls;goal=$goalPin;review=$reviewPin;admission=$admissionPin;raw_controls=$raw;parent_copies=$copied;initial_DATA_references=13;checkpoint_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
Get-FilePin 'doc/CAUSAL_FRESH_V10_CARD.json'|ConvertTo-Json -Depth 4

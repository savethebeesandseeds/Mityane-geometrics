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
$draftPath=Join-Path $root 'doc/CAUSAL_FRESH_V11_CARD.json'
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
if($plan.kind-ne'cpb_v11_prospective_declaration_input_plan_v1'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true-or$plan.status-ne'root_reviewed_source_and_completed_context'){throw 'Only a root-reviewed completed-context plan can prepare a draft'}
foreach($required in @('doc/CAUSAL_FRESH_V10_CARD.json','doc/causal_fresh_v10/protocol-preflight.json','doc/causal_fresh_v10/complete-mirror-attempt-001.json','doc/CAUSAL_FRESH_V9_CARD.json','doc/CAUSAL_FRESH_V9_CARD.md','doc/causal_fresh_v10/final-preservation.json','doc/causal_fresh_v9/protocol-preflight.json','doc/causal_fresh_v9/complete-mirror-attempt-001.json','doc/causal_fresh_v9/reporting-recovery-card-001.json','doc/causal_fresh_v9/reporting-recovery-receipt-001.json','doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json','doc/clean_causal_encoder_search_goal_v9_retention.json','doc/causal_v9_retention_review/independent-review.json','doc/causal_v9_retention_review/applied-future-protocol.json','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json','doc/causal_v8_v9_train_diagnosis/diagnosis.json','doc/causal_v8_v9_train_diagnosis/independent-audit.json','doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md','doc/causal_v8_v9_train_diagnosis/completion.json','doc/causal_v8_v9_train_diagnosis/human-report-root-review.json','doc/clean_causal_encoder_research_ledger.json','code/encoders/causal_patch_bottleneck_mae/versions/current.json','doc/causal_fresh_v11/family-source-provenance.json','doc/causal_fresh_v11/evaluation-source-provenance.json','code/research/causal_fresh_v11/training-helper-provenance.json')){
 if(@($plan.protected_context|Where-Object{$_.path-eq$required}).Count-ne1){throw ('One exact reviewed context pin required: '+$required)}
}
foreach($pin in $plan.protected_context){Verify-Pin $pin}
if(@($plan.protected_context|Where-Object{$_.path-eq'doc/causal_fresh_v9/CPB-v9.descriptor-draft.json'}).Count-ne1){throw 'Exact physical V9 descriptor draft template must be externally plan-bound before parse'}
foreach($version in 3..10){if(@($plan.protected_context|Where-Object{$_.path-eq('doc/results/causal_fresh_v'+$version+'/table_rows.json')}).Count-ne1){throw 'Every saved historical final control must be externally plan-bound'}}
foreach($version in 6..10){if(@($plan.protected_context|Where-Object{$_.path-eq('doc/causal_fresh_v'+$version+'/final-establishment.json')}).Count-ne1){throw 'Every separate historical establishment must be externally plan-bound'}}
Verify-Pin $plan.independent_training_auditor_source_review
$trainingReview=Get-Content -LiteralPath (Join-Path $root $plan.independent_training_auditor_source_review.path) -Raw|ConvertFrom-Json -AsHashtable
if($trainingReview.kind-ne'cpb_v11_family_and_training_auditor_independent_static_review_v1'-or$trainingReview.status-ne'passed'){throw 'Actual independently passed V11 training-auditor source review required'}
if($plan.static_build.binaries-ne7-or$plan.static_build.translation_units-ne11){throw 'Final source owners declare seven binaries and eleven TUs; actual build admission remains separate'}
$orchestration=Get-FilePin 'doc/causal_fresh_v11/orchestration-provenance.json'
if($orchestration.sha256-ne$OrchestrationProvenanceSHA256){throw 'Exact independently reviewed orchestration provenance required'}
$workflow=Get-Content -LiteralPath (Join-Path $root $orchestration.path) -Raw|ConvertFrom-Json -AsHashtable
if($workflow.files.Count-ne10){throw 'Exactly ten owned orchestration helpers required'}
foreach($binding in $workflow.files){Verify-Pin $binding.parent;Verify-Pin $binding.owned}
$ownerFiles=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v11') -Recurse -File)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v11.h'))+@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v11') -Recurse -File)
$observedPaths=@($ownerFiles|ForEach-Object{$_.FullName.Substring($root.Length+1).Replace([char]92,[char]47)}|Sort-Object)
if(Compare-Object $observedPaths @($plan.owner_sources.path|Sort-Object)){throw 'Plan must bind the complete actual family/research path set'}
foreach($pin in $plan.owner_sources){Verify-Pin $pin}
foreach($output in @('doc/CAUSAL_FRESH_V11_CARD.md','doc/causal_fresh_v11/CPB-v11.descriptor-draft.json','doc/causal_fresh_v11/draft-metadata-copy-receipt.json','doc/causal_fresh_v11/preservation-before.json')){if(Test-Path -LiteralPath (Join-Path $root $output)){throw ('Preserve existing output: '+$output)}}
Verify-Pin $plan.card_markdown_template
if((Get-FilePin 'doc/causal_fresh_v10/final-preservation.json').sha256-ne'fa6eb72bd45b9cb43d8dc7ea523ed70908eee072cc1c2e046e45995cd3a4c24d'){throw 'Exact104-checkpoint V10 final preservation required before parse'}
$preservation=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v10/final-preservation.json') -Raw|ConvertFrom-Json -AsHashtable
if($preservation.status-ne'passed'-or$preservation.critical_checkpoint_pins.Count-ne104){throw 'Actual final V10 preservation with104 existing critical checkpoints required'}
foreach($pin in $preservation.critical_checkpoint_pins){$repo=$pin.repository;if(-not$repo){if(-not$pin.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')){throw 'Only five historical V5 pins may omit repository'};$repo='cuwacunu_embedding'};Verify-Pin $pin $repo}
$oldCard=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V9_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
if($oldCard.baseline_frozen_files.Count-ne47){throw '47 physical V3 source files required per repository'}
foreach($pin in $oldCard.baseline_frozen_files){foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){Verify-Pin $pin $repo}}
$oldSource=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/protocol-preflight.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($pin in $oldSource.sources){Verify-Pin $pin}
$latestPaths=[ordered]@{card='doc/CAUSAL_FRESH_V10_CARD.json';manifest='output/engineering/causal-fresh-v10/attempt-001/source-manifest.json';archive='output/engineering/causal-fresh-v10/attempt-001/source.zip';protocol_preflight='doc/causal_fresh_v10/protocol-preflight.json'}
foreach($role in $latestPaths.Keys){
 $binding=$plan.latest_frozen_source[$role]
 if($null-eq$binding-or$binding.path-cne$latestPaths[$role]){throw ('Exact latest frozen V10 source role required: '+$role)}
 $matches=@($plan.protected_context|Where-Object{$_.path-eq$binding.path})
 if($matches.Count-ne1-or$matches[0].bytes-ne$binding.bytes-or$matches[0].sha256-ne$binding.sha256){throw 'Latest source roles must be externally bound in the actual reviewed plan'}
 Verify-Pin $binding
}
$latestCard=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.card.path) -Raw|ConvertFrom-Json -AsHashtable
$latestManifest=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.manifest.path) -Raw|ConvertFrom-Json -AsHashtable
$latestPreflight=Get-Content -LiteralPath (Join-Path $root $plan.latest_frozen_source.protocol_preflight.path) -Raw|ConvertFrom-Json -AsHashtable
$latestCount=$latestCard.source_counts.capsule_sources
if(($latestCount-isnot[int]-and$latestCount-isnot[long])-or$latestCount-le0-or$latestCount-ne$preservation.V10_source_files-or$latestManifest.files.Count-ne$latestCount-or@($latestManifest.files.path|Sort-Object -Unique).Count-ne$latestCount-or($latestPreflight.sources.Count+1)-ne$latestCount){throw 'Latest source count must derive consistently from actual frozen V10 card/final preservation/preflight/manifest'}
if($latestCard.model-ne'CPB-v10'-or$latestCard.stage.Contains('DRAFT')-or-not$latestCard.frozen_utc-or$latestManifest.card_sha256-ne$plan.latest_frozen_source.card.sha256-or$latestPreflight.status-ne'passed'-or$latestPreflight.card_sha256-ne$plan.latest_frozen_source.card.sha256){throw 'Exact frozen V10 source identities required'}
$latestArchives=@($preservation.source_ZIPs|Where-Object{$_.manifest.path-eq$plan.latest_frozen_source.manifest.path})
if($latestArchives.Count-ne1-or$latestArchives[0].source_files-ne$latestCount-or$latestArchives[0].manifest.sha256-ne$plan.latest_frozen_source.manifest.sha256-or$latestArchives[0].archive.sha256-ne$plan.latest_frozen_source.archive.sha256){throw 'Latest manifest/whole ZIP must match independently passed final V10 preservation'}
$latestPathsExpected=@($latestPreflight.sources.path)+@($plan.latest_frozen_source.protocol_preflight.path)
if(Compare-Object (@($latestManifest.files.path)|Sort-Object) ($latestPathsExpected|Sort-Object)){throw 'Latest source manifest path set must equal the passed V10 preflight plus its own receipt'}
foreach($binding in $latestManifest.files){Verify-Pin $binding}
$latestFrozenSource=[ordered]@{card=$plan.latest_frozen_source.card;manifest=$plan.latest_frozen_source.manifest;archive=$plan.latest_frozen_source.archive;protocol_preflight=$plan.latest_frozen_source.protocol_preflight;source_files=$latestCount;source_entries_rehashed=$latestManifest.files.Count;scope='Host current-file hashes plus externally pinned whole ZIP/manifest/preflight;V9 scientific-parent rehash retained;no model or archive numeric read'}
Write-JsonNew 'doc/causal_fresh_v11/preservation-before.json' ([ordered]@{kind='cpb_v11_before_execution_file_preservation_v1';status='passed';created_utc=[DateTime]::UtcNow.ToString('o');input_plan=[ordered]@{path=$planPath.Substring($root.Length+1).Replace([char]92,[char]47);bytes=(Get-Item -LiteralPath $planPath).Length;sha256=$InputPlanSHA256};baseline_preservation=(Get-FilePin 'doc/causal_fresh_v10/final-preservation.json');critical_checkpoint_pins=$preservation.critical_checkpoint_pins;critical_checkpoint_files=104;baseline_sources_each_repository=47;baseline_source_pins=$oldCard.baseline_frozen_files;protected_context=$plan.protected_context;old_V9_source_files_verified=$oldSource.sources.Count;new_owner_sources=$plan.owner_sources;file_hash_inspection_only=$true;archive_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
$parentCopies='output/engineering/causal-fresh-v11-protocol-preparation/attempt-001/parent-copies'
$copied=@()
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V9_CARD.json' ($parentCopies+'/CAUSAL_FRESH_V9_CARD.json')
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V9_CARD.md' ($parentCopies+'/CAUSAL_FRESH_V9_CARD.md')
$card=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V9_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
$goal=Get-Content -LiteralPath (Join-Path $root 'doc/clean_causal_encoder_search_goal_v9_retention.json') -Raw|ConvertFrom-Json -AsHashtable
if((Get-FilePin 'doc/clean_causal_encoder_search_goal_v9_retention.json').sha256-ne'221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144'-or$goal.criteria.Count-ne22-or$goal.additional_train_NLL_gate-ne$false){throw 'Exact current22-condition/noTRAINNLL protocol required'}
$controls=[ordered]@{}
foreach($version in 3..10){$controls['v'+$version]=Copy-ExactMetadata ('doc/results/causal_fresh_v'+$version+'/table_rows.json') ('doc/causal_fresh_v11/baseline-control-v'+$version+'.json')}
foreach($version in 6..10){$controls['v'+$version+'_establishment']=Copy-ExactMetadata ('doc/causal_fresh_v'+$version+'/final-establishment.json') ('doc/causal_fresh_v11/baseline-v'+$version+'-establishment.json')}
$controls.reuse_only='Exact saved V3/V4/V5/V6/V7/V8/V9/V10 final tables and separate V6/V7/V8/V9/V10 full-trust receipts; preserve pending producer bytes; no old encoder/head execution'
$goalPin=Copy-ExactMetadata 'doc/clean_causal_encoder_search_goal_v9_retention.json' 'doc/causal_fresh_v11/goal-protocol.json'
$reviewPin=Copy-ExactMetadata 'doc/causal_v9_retention_review/independent-review.json' 'doc/causal_fresh_v11/goal-protocol-review.json'
$admissionPin=Copy-ExactMetadata 'doc/causal_v9_retention_review/applied-future-protocol.json' 'doc/causal_fresh_v11/goal-protocol-admission.json'
$raw=Copy-ExactMetadata 'doc/causal_fresh_v9/raw-controls.json' 'doc/causal_fresh_v11/raw-controls.json'
$input=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/input-manifest.json') -Raw|ConvertFrom-Json -AsHashtable
$input.kind='causal_fresh_v11_data_input_manifest_v1';$input.protocol='causal-fresh-v11';$input.model='CPB-v11.market-01'
Write-JsonNew 'doc/causal_fresh_v11/input-manifest.json' $input
$policy='original-position-four-bin-half-mixture-mean4-centered-variance4-v1'
$empty='zero-whole-export-slice-after-bias-v1'
$objective='huber1-mean-plus-detached-error-gaussian-variance-v1'
$gradient='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
$encoderGradient='Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1'
$routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
$establishment='Full trust including exact initial native20:40 DATA and whole-statistic pairings against saved V9, and independent training/quality audits, establishes a measured candidate. Advancement additionally requires full coverage and all22 declared numeric conditions. Winner and application selection remain separate and false.'
$card.kind='causal_fresh_v11_prospective_card_v1';$card.protocol='causal-fresh-v11'
$card.declared_utc=$null;$card.frozen_utc=$null;$card.stage='DRAFT_NOT_FROZEN_NOT_EXECUTION_ADMITTED'
$card.model='CPB-v11';$card.instance='CPB-v11.market-01';$card.initial_quality_label='CPB-v11.market-01.init'
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
$card.mechanism.gradient_interpretation='V9 objective unchanged: unweighted Huber1 mean route and detached-error Gaussian conditional-variance route;direct decoder paths remain separate. Shared globalclip1/singleAdamW and future residuals couple optimization. The new pooling centered-variance feature has live attention and mean dependence;it is not the decoder conditional variance/native uncertainty branch.'
$card.training.objective='R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v + log(v))];coefficients[1,1];supported operands sanitized and promoted separately to CUDAF64,unchanged hierarchical reduction. Only total_loss backpropagates;ordinary reporting NLL is a detached copy.'
$card.training.fixed_query_NLL='Ordinary Gaussian NLL from the detached reporting copy;separate Huber mean and detached-error variance objective components retain V9 routes'
$card.training.optimizer_coupling='One global norm2 clip1 and one AdamW;shared scaling,moments and future residuals couple optimization. No optimizer independence or validation-cause claim.'
$card.training.raw_load_counter_contract=[ordered]@{successful_updates=2048;attempts_min=2048;attempts_max=4096;calibration_loader_calls=512;initial_and_final_query_loader_calls=36;raw_load_calls_formula='512 + attempted_batches + 36';skipped_attempts='attempted_batches - new_causal_updates;derived,not a fabricated summary field'}
$card.mechanism.kind=$policy;$card.mechanism.temporal_pooling_policy=$policy
$card.mechanism.temporal_pooling_mix=@(0.5,0.5);$card.mechanism.empty_local_bin_policy=$empty
$card.mechanism.architecture='Both V9 temporal paths retain the actual half-local/half-global normalized attention mixture. Each nonempty original bin serves its preserved two-sum mean to first4 export rows and componentwise centered variance under that mixture to last4 rows. One fresh32x64 tensor/32bias retains registration/draws. Empty slices zero after bias;concat32 feeds unchanged global99->64->32 and shape20. Native40 stays content20+fixed12+mean4+uncertainty4.'
$card.mechanism.controlled_mechanism='Only the intermediate eight-row export input allocation changes versus fresh V9: mean4 plus centered-variance4 instead of mean8. The actual attention-dependent variance derivative remains live. V9 detached-error objective,all shapes/counts/draws/native40/globalclip1/AdamW/seeds/data/budget/D40heads remain unchanged;all learned state/statistics fresh.'
$card.mechanism.capacity_confounds='Same parameters/native40/D40heads and fresh constructor draws as V9;half the intermediate mean rows are reassigned to centered variance,changing scale,information allocation and trajectory. Native content20 may change;only20:40 DATA plus five whole statistic archives are paired. No TRAIN/validation causation or promised improvement. V3/V4 width/head-capacity confounds remain.'
$card.mechanism.allowed_typed_resume='Only fresh V11 lineage; objective/gradient/layout plus temporal_pooling_policy,Long[5] partition,Double[2] mix in anchored_local/whole_history order,and empty_local_bin_policy at inference root/payload and TRAIN root/encoder/encoder.payload. Reject missing,wrong value/dtype/shape/order,old V9/V6 kinds; preserve all optimizer moments/counters/order/data identities.'
$card.mechanism.fresh_initialization_order='Same V9 parameter registration/shapes/RNG draw order under943001; all294752 learned values and optimizer state fresh. A single fresh export tensor is shared by the two pooling routes; existing never means imported trained weights.'
$card.mechanism.empty_bin_policy='Original saved position membership. All-padding rows finite zero; all empty anchored bins zero the entire corresponding export slice after its one bias even when global history is nonempty; invalid hidden values cleared before pooling.'
$card.mechanism.causal_endpoint_semantics='Every slice is a coordinate of the embedding at the current requested cutoff t,not an embedding stamped at its bin endpoint. Shared whole-history uses only valid states at or before t. Later-in-window past may enter older anchored slices only within this endpoint embedding. Unchanged causal Transformer/backward relation bank and exact future-value AND future-mask invariance at all cutoffs are mandatory.'
$card.mechanism.same_export_weights_means='One single freshly initialized export tensor shared across the two pooling routes;unchanged shape/registration/draw order;never V9/V6 trained values'
$card.training.recipe_id='causal-v11-half-mixture-mean4-centered-variance4-v9-objective-v1'
$card.training.temporal_pooling_policy=$policy;$card.training.temporal_pooling_mix=@(0.5,0.5);$card.training.empty_local_bin_policy=$empty
$card.training.component_evidence='All2048 successful scalar rows,8192 whole-eight-row/within-bin,16384 mean4/centered-variance4 subgroup,2048 gauge and2048 Hadamard rows. Actual1/2048 component witnesses have23 base fields+12 parameter arrays+21 same-forward traces including two PRECLIP upstreams/two flags=56 fields. No additional encoder call. Failed eligibility attempts stay blank and affect attempted/rawload counters.'
$card.quality.primary_methods=@('CPB-v3.market-01','CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01','CPB-v10.market-01','CPB-v11.market-01.init','CPB-v11.market-01')
$card.quality.no_historical_encoder_rows_in_primary_tables=$false
$card.quality.primary_clean_control_rows_only=$true
$card.quality.clean_baseline_control=$controls.reuse_only
$card.quality.primary_report='Eight saved clean final controls plus own new initial/final heads;four market task tables/four TEMPO panels,physical errors,three neural repetitions/query components/cost. Pending producer trust stays unchanged until independent audits/pairings and separate join. No automatic winner/application selection.'
$card.advancement.policy='All22 exact reviewed strongest V3–V9 conditions plus mandatory full trust/coverage;raw producer numeric/coverage and pending trust remain separate. No automatic winner or selection.'
$card.advancement.goal_protocol=$goalPin;$card.advancement.criteria=$goal.criteria;$card.advancement.criterion_count=22
$card.criteria.quality_interpretation='All22 literal221f criterion objects unchanged;historical counts retain owncards;no TRAIN NLL gate. Generic ordered mean/dispersion retention hypothesis;no validation-cause or improvement claim.'
$card.criteria.establishment=$establishment
$card.criteria.trust_admission=@(
 'All76learned tensors/294752values and optimizer freshly constructed with identical V9 draw order;fresh4096TRAIN input/bank statistics,source_updates0,no old model/statistic import',
 '75 identifiable tensors have actual CUDA nonzero data-gradient and movement events;294751 is tensor capacity,not every-scalar learning. Scalarpool_score.bias gauge remains registered/optimized/observed and is the sole exemption;weightdecay is not data learning',
 'Four export8-row groups and four within-bin centered position groups each show data-gradient/movement events and final changes. Each of three Hadamard between-bin directions has data-gradient+movement events and overall final between-bin projection changes;no each-final-coefficient gate',
 'All128cutoffs/2350suffix/30internal temporal cases initialized/trained and actual saved0/2048;future values AND masks/nonfinite suffix cannot change endpoint native40;past-positive interventions',
 'All seven temporal typed fields validated at inference root/payload and TRAIN root/encoder/encoder.payload,including missing/wrong pooling/moment/layout/mix/empty/partition/dtype/shape/order. Actual typed512resume exact and controlled engineering next-step parity',
 'Exclusive independently reviewed prospective card/descriptor/source/input/compiler/binary closure before any build/test/CUDA;actual file/binary/TU counts derived from frozen owners,not assumed',
 'V9 partial-gradient objective CUDA fixture;new full-attention mean/centered-variance manual adjoints,F64 and actualF32 finite differences,trace/empty/constant/equal-mean-spread/common-gauge/three-relative-direction CUDA fixture',
 'Exactly13 DATA/statistic initial pairings versus V9: eight feature value matrices20:40 only with whole valid/IDs/labels;five fresh-statistic archives whole numeric tensors. Content0:20 is unpaired;new own initial heads;no old model/score replay',
 'Independent saved-training/quality arithmetic audits,all fullcoverage finite metrics and separate final-trust evidence join before any advancement claim'
)
$card.input_manifest=Get-FilePin 'doc/causal_fresh_v11/input-manifest.json'
$card.baseline_preservation=Get-FilePin 'doc/causal_fresh_v10/final-preservation.json'
$card.latest_frozen_source_preservation=$latestFrozenSource
$card.pre_execution_preservation=Get-FilePin 'doc/causal_fresh_v11/preservation-before.json'
$card.saved_TRAIN_diagnosis=[ordered]@{
 card=(Get-FilePin 'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json');result=(Get-FilePin 'doc/causal_v8_v9_train_diagnosis/diagnosis.json')
 independent_audit=(Get-FilePin 'doc/causal_v8_v9_train_diagnosis/independent-audit.json');human_report=(Get-FilePin 'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md')
 completion=(Get-FilePin 'doc/causal_v8_v9_train_diagnosis/completion.json')
 human_report_root_review=(Get-FilePin 'doc/causal_v8_v9_train_diagnosis/human-report-root-review.json')
 scope='Completed independent savedTRAIN-only V8/V9 diagnosis;bounded scalar/adjoint arithmetic,zero fullencoder/librarymodel/head/refit calls;all old99checkpoints/source preserved'
 interpretation='Diagnosis selected no new model;root separately chose one mean4/centered-variance4 pooling hypothesis after diagnosis completion. No causal or counterfactual claim.'
}
$card.comparison_control=$controls;$card.scoped_reference='CPB-v10.market-01'
$card.scientific_recipe_parent='CPB-v9 source only;fresh weights/optimizer/statistics,not trained ancestry'
$card.pairing_reference='CPB-v9.market-01 saved initial DATA/statistics only'
$card.scoped_references=@('CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01','CPB-v10.market-01')
$card.retention_independent_review=$reviewPin;$card.retention_application_admission=$admissionPin
$card.geometry_override='Native40/D40heads/counts/draw order unchanged;four mean plus four centered-variance rows per original bin replace V9 eight mean rows. V9 objective/globalclip1 unchanged. New content20 need not pair;remaining20 coordinates and five full stats pair against saved V9.'
$card.prospective_descriptor='code/encoders/causal_patch_bottleneck_mae/versions/CPB-v11.json'
$card.initial_pairing.reference='CPB-v9.market-01 initial0 DATA/statistic artifacts only'
$card.initial_pairing.native_coordinates_compared=20;$card.initial_pairing.native_coordinate_slice=@(20,40)
$card.initial_pairing.unpaired_learned_content_slice=@(0,20);$card.initial_pairing.content20_equality_required=$false
$card.initial_pairing.whole_native40_equality_required=$false;$card.initial_pairing.statistic_numeric_equality_required=$true
$card.initial_pairing.initial_scores_reused=$false;$card.initial_pairing.new_own_initial_fixed_heads=$true
$v9Mirror=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/complete-mirror-attempt-001.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($reference in $card.initial_pairing.references){
 $container=$reference.reference_data_archive.path.Replace('causal-fresh-v8/','causal-fresh-v9/').Replace('cpb_v8-','cpb_v9-')
 $matches=@($v9Mirror.files|Where-Object { $_.container_path -eq $container })
 if($matches.Count-ne1){throw ('One exact V9 initial DATA pin required: '+$container)}
 $actual=Get-FilePin $matches[0].path
 if($actual.sha256-ne$matches[0].sha256-or$actual.bytes-ne$matches[0].bytes){throw 'V9 initial DATA reference changed'}
 $reference.reference_data_archive=[ordered]@{path=$container;bytes=$actual.bytes;sha256=$actual.sha256}
 $reference.candidate_relative_to_attempt=$reference.candidate_relative_to_attempt.Replace('cpb_v9-','cpb_v11-')
 $reference.comparison='Feature value matrices compare20:40 only;valid/IDs/labels and five fresh-statistic archives compare whole declared numeric tensors. Content20 intentionally unpaired;new own initial heads,no old model/head execution.'
 if($reference.Contains('feature_value_coordinate_slice')){$reference.feature_value_coordinate_slice=@(20,40)}
}
$card.trust_receipt_objective_fields.temporal_pooling_policy=$policy
$card.trust_receipt_objective_fields.temporal_pooling_mix=@(0.5,0.5)
$card.trust_receipt_objective_fields.empty_local_bin_policy=$empty
$card.trust_receipt_field_applicability=[ordered]@{encoder_gradient_kind='Main training summary only;not an invented engineering/savedproof/fresh selector';gradient_policy='Strict typed records/engineering/training/savedproofs;absent only in constructor fresh-initialization.json';direct_gradient_routes='Mandatory common objective JSON in all declared records including fresh constructor'}
$card.mechanism_proof_contract.PASS_prefix='PASS CPB-v11 huber1-mean-plus-detached-error-gaussian-variance-v1:'
foreach($key in @('direct_joint_NLL_mean_gradient_live','direct_joint_NLL_uncertainty_gradient_live','joint_Gaussian_variance_counted_once','total_equals_Huber_plus_one_joint_Gaussian','independent_composite_F64_finite_differences')){$card.mechanism_proof_contract.Remove($key)}
$card.mechanism_proof_contract.direct_NLL_mean_gradient_zero=$true
$card.mechanism_proof_contract.manual_parity_scope='V9 Huber1 mean plus detached-error Gaussian conditional-variance objective;manual native40 partial adjoints,direct cross-path zeros,live head/projection and shared globalclip coupling'
$card.mechanism_proof_contract.log_tokens=@('manual composite-objective native40 gradient parity','detached-error NLL mean gradient zero','synthetic optimizer steps=1')
$temporal=$card.temporal_pooling_proof_contract
$temporal.status='DRAFT_source_owner_frozen_independent_review_before_any_execution'
$temporal.temporal_pooling_policy=$policy;$temporal.temporal_pooling_mix=@(0.5,0.5);$temporal.empty_local_bin_policy=$empty
$temporal.PASS_prefix='PASS CPB-v11 '+$policy+':'
$temporal.log_tokens=@('manual full-attention mean and centered-variance parity','independent full-expression F64 finite differences','actual CUDA F32 kernel finite differences','four mean4 and variance4 projection gradient and movement','equal mean different spread','present constant variance dtype tolerance','three between-bin Hadamard contrast gradient and movement','empty-bin finite zero','relative-bin global sensitivity','common softmax gauge invariance','synthetic optimizer steps=1')
$temporal.required_binary_count=$plan.static_build.binaries;$temporal.required_TU_count=$plan.static_build.translation_units
$proof=$temporal.required_proof_fields;$proof.model='CPB-v11';$proof.temporal_pooling_policy=$policy
$proof.temporal_pooling_mix=@(0.5,0.5);$proof.empty_local_bin_policy=$empty
foreach($key in @('softmax_gauge_invariance','fixture_backbone_Model_encode_calls','fixture_direct_pooling_kernel_calls','public_Encoder_calls')){$proof.Remove($key)}
$proof.common_softmax_gauge_invariance=$true;$proof.shared_global_half_mixture=$true
$proof.relative_bin_global_sensitivity=$true;$proof.three_between_bin_contrasts_gradient_and_movement=$true
$proof.backbone_encode_calls=1;$proof.public_encoder_calls=0;$proof.direct_pool_kernel_calls=44
foreach($field in @('mean4_centered_variance4_export_layout','full_attention_dependent_moment_adjoints','independent_full_expression_F64_finite_differences','actual_F32_kernel_finite_differences','equal_mean_different_variance','present_constant_variance_dtype_tolerance','eight_moment_subgroups_gradient_and_movement')){$proof[$field]=$true}
$proof.manual_F32_absolute_tolerance=3e-5;$proof.manual_F32_relative_tolerance=3e-5
$proof.F64_finite_difference_coordinates=18;$proof.F64_finite_difference_step=1e-5;$proof.F64_finite_difference_absolute_tolerance=2e-8;$proof.F64_finite_difference_relative_tolerance=2e-8
$proof.F32_finite_difference_coordinates=18;$proof.F32_finite_difference_step=2e-3;$proof.F32_finite_difference_absolute_tolerance=2e-3;$proof.F32_finite_difference_relative_tolerance=2e-3
$proof.present_constant_variance_bound='1e-10*(1+max_abs_hidden_squared)'
$temporal.binary='test_cpb_v11_temporal_pooling'
$card.temporal_learning_receipt_contract.kind='cpb_v11_actual_temporal_pooling_learning_v1'
$card.temporal_learning_receipt_contract.temporal_pooling_policy=$policy
$card.temporal_learning_receipt_contract.temporal_pooling_mix=@(0.5,0.5)
$card.temporal_learning_receipt_contract.empty_local_bin_policy=$empty
$card.temporal_learning_receipt_contract.between_bin_rows=2048
$card.temporal_learning_receipt_contract.receipt_artifacts=@('temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv','temporal-between-bin-position-learning.csv','temporal-moment-subgroup-learning.csv')
$card.temporal_learning_receipt_contract.moment_subgroup_rows=16384
$card.temporal_learning_receipt_contract.moment_subgroups=[ordered]@{count=8;moments=@('mean','centered_variance');rows_each=4;columns=@('update','bin','moment','export_row_begin','export_row_end','weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2','nonzero_export_gradient_values','moved_export_values');each_weight_and_bias_data_gradient_and_movement_events_positive=$true;each_final_weight_and_bias_changed=$true;every_scalar_or_update_nonzero_required=$false;pooling_intermediate_not_native_uncertainty_branch=$true}
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
$card.source_counts=[ordered]@{family_inside=@($ownerFiles|Where-Object{$_.FullName.StartsWith((Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v11')+[IO.Path]::DirectorySeparatorChar)}).Count;family_external_header=1;research=@($ownerFiles|Where-Object{$_.FullName.StartsWith((Join-Path $root 'code/research/causal_fresh_v11')+[IO.Path]::DirectorySeparatorChar)}).Count;translation_units=$plan.static_build.translation_units;binaries=$plan.static_build.binaries;capsule_sources=$null;status='SOURCE_ONLY_actual_owner_path_set_bound_capsule_count_derived_at_freeze'}
$card.draft_copy_provenance=[ordered]@{
 kind='full_physical_V9_scientific_card_and_normal_V10_native_workflow_then_owned_V11_architecture_edits'
 card=(Get-FilePin 'doc/CAUSAL_FRESH_V9_CARD.json');markdown=(Get-FilePin 'doc/CAUSAL_FRESH_V9_CARD.md')
 direct_orchestration_parent=(Get-FilePin 'doc/causal_fresh_v10/run-container.ps1')
 repaired_native_Docker_ancestry=(Get-FilePin 'doc/causal_fresh_v8/run-container-recovery-001.ps1')
 parent_copies=$copied;old_files_modified=$false;source_only=$true
}
$card.geometry.temporal_pooling_policy=$policy;$card.geometry.temporal_pooling_partition=@(4,16,32,4,8);$card.geometry.temporal_pooling_mix=@(0.5,0.5);$card.geometry.empty_local_bin_policy=$empty
foreach($record in @($card.geometry,$card.mechanism,$card.training,$card.trust_receipt_objective_fields,$card.temporal_pooling_proof_contract,$card.temporal_learning_receipt_contract,$proof)){
 $record.temporal_moment_policy='componentwise-centered-variance-two-sum-mean-live-attention-v1'
 $record.temporal_export_layout='per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'
 $record.temporal_moment_partition=@(4,4)
}
$card.actual_step_temporal_witness_contract=[ordered]@{updates=@(1,2048);base_fields=23;parameter_arrays=12;temporal_trace_fields=21;total_fields=56;extra_encoder_calls=0;upstream_stage='PRECLIP retained nonleaf;apply recorded clipfactor once when comparing POSTCLIP parameter gradients';unused_route='explicit Long scalar defined flag0 plus all-zero upstream fallback;no per-route activity gate';moment_F32_absolute_tolerance=3e-5;moment_F32_relative_tolerance=3e-5;local_parameter_adjoint_absolute_tolerance=1e-9;local_parameter_adjoint_relative_tolerance=3e-4;per_witness_export_weight_OR_bias_postclip_max_absolute_minimum=1e-6;common_score_bias='finite actual/reference/error descriptive only;no tight parity or nonzero gate';scope='independent scalar moments and tight export-weight/bias/pool-position local adjoints summed over actual routes;common-score-bias descriptive;not whole encoder gradients or norm reconstruction'}
$card.results=[ordered]@{status='unmeasured';quality=$null;trust=$null;winner_selection_made=$false;application_selection_changed=$false}
$card.parent_goal_preserved=Get-FilePin 'doc/clean_causal_encoder_search_goal_v8_retention.json'
$card.historical_V9_reporting_recovery=[ordered]@{card=(Get-FilePin 'doc/causal_fresh_v9/reporting-recovery-card-001.json');receipt=(Get-FilePin 'doc/causal_fresh_v9/reporting-recovery-receipt-001.json');failure_mirror=(Get-FilePin 'doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json');scope='HistoricalV9 reporting-only/no-science-replay evidence;not a V11 recovery or recipe change'}
$card.declaration_input_plan=[ordered]@{path=$planPath.Substring($root.Length+1).Replace([char]92,[char]47);bytes=(Get-Item -LiteralPath $planPath).Length;sha256=$InputPlanSHA256}
Write-JsonNew 'doc/CAUSAL_FRESH_V11_CARD.json' $card
$descriptor=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v9/CPB-v9.descriptor-draft.json') -Raw|ConvertFrom-Json -AsHashtable
$descriptor.version='CPB-v11';$descriptor.instance='CPB-v11.market-01';$descriptor.card=$null
$descriptor.physical_version_root='code/encoders/causal_patch_bottleneck_mae/versions/v11';$descriptor.public_entry='code/encoders/causal_patch_bottleneck_mae/versions/v11/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v11.h'
$descriptor.architecture_copy_parent='CPB-v9';$descriptor.objective_policy=$objective;$descriptor.gradient_policy=$gradient;$descriptor.direct_gradient_routes=$routes
$descriptor.variance_error_detached=$true;$descriptor.reporting_nll_backpropagated=$false;$descriptor.retention_protocol=$goalPin
$descriptor.mechanism='V9 unitHuber plus detached-error variance objective and half-mixture attention;only existing per-bin export inputs change from mean8 to mean4 plus centered-variance4. Same counts/draws/native40/globalclip1/heads;all learned state/statistics fresh.'
$descriptor.temporal_pooling_policy=$policy;$descriptor.temporal_moment_policy=$card.mechanism.temporal_moment_policy;$descriptor.temporal_export_layout=$card.mechanism.temporal_export_layout;$descriptor.temporal_moment_partition=@(4,4)
$descriptor.initial_pairing='Native20:40 eight DATA archives plus five whole fresh-statistic archives against saved V9initial;own new initial heads;content20 unpaired'
Write-JsonNew 'doc/causal_fresh_v11/CPB-v11.descriptor-draft.json' $descriptor
[IO.File]::Copy((Join-Path $root $plan.card_markdown_template.path),(Join-Path $root 'doc/CAUSAL_FRESH_V11_CARD.md'),$false)
Write-JsonNew 'doc/causal_fresh_v11/draft-metadata-copy-receipt.json' ([ordered]@{status='draft_prepared_not_execution_admitted';controls=$controls;goal=$goalPin;review=$reviewPin;admission=$admissionPin;raw_controls=$raw;parent_copies=$copied;initial_DATA_references=13;checkpoint_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
Get-FilePin 'doc/CAUSAL_FRESH_V11_CARD.json'|ConvertTo-Json -Depth 4

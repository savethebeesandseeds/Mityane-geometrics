# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host metadata/byte preparation only. This does not admit or execute a project phase.
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$draftPath=Join-Path $root 'doc/CAUSAL_FRESH_V9_CARD.json'
if(Test-Path -LiteralPath $draftPath){throw 'Preserve the existing draft card'}
function Get-FilePin([string]$relative){
 $file=Get-Item -LiteralPath (Join-Path $root $relative)
 if($file.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Physical metadata input required'}
 [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
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
$parentCopies='output/engineering/causal-fresh-v9-protocol-preparation/attempt-001/parent-copies'
$copied=@()
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V8_CARD.json' ($parentCopies+'/CAUSAL_FRESH_V8_CARD.json')
$copied+=Copy-ExactMetadata 'doc/CAUSAL_FRESH_V8_CARD.md' ($parentCopies+'/CAUSAL_FRESH_V8_CARD.md')
$card=Get-Content -LiteralPath (Join-Path $root 'doc/CAUSAL_FRESH_V8_CARD.json') -Raw|ConvertFrom-Json -AsHashtable
$goal=Get-Content -LiteralPath (Join-Path $root 'doc/clean_causal_encoder_search_goal_v8_retention.json') -Raw|ConvertFrom-Json -AsHashtable
$controls=[ordered]@{}
foreach($version in 3..8){$controls['v'+$version]=Copy-ExactMetadata ('doc/results/causal_fresh_v'+$version+'/table_rows.json') ('doc/causal_fresh_v9/baseline-control-v'+$version+'.json')}
foreach($version in 6..8){$controls['v'+$version+'_establishment']=Copy-ExactMetadata ('doc/causal_fresh_v'+$version+'/final-establishment.json') ('doc/causal_fresh_v9/baseline-v'+$version+'-establishment.json')}
$controls.reuse_only='Exact saved V3/V4/V5/V6/V7/V8 final tables and separate V6/V7/V8 full-trust receipts; preserve pending producer bytes; no old encoder/head execution'
$goalPin=Copy-ExactMetadata 'doc/clean_causal_encoder_search_goal_v8_retention.json' 'doc/causal_fresh_v9/goal-protocol.json'
$reviewPin=Copy-ExactMetadata 'doc/causal_v8_retention_review/independent-review.json' 'doc/causal_fresh_v9/goal-protocol-review.json'
$admissionPin=Copy-ExactMetadata 'doc/causal_v8_retention_review/applied-future-protocol.json' 'doc/causal_fresh_v9/goal-protocol-admission.json'
$raw=Copy-ExactMetadata 'doc/causal_fresh_v8/raw-controls.json' 'doc/causal_fresh_v9/raw-controls.json'
$input=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v8/input-manifest.json') -Raw|ConvertFrom-Json -AsHashtable
$input.kind='causal_fresh_v9_data_input_manifest_v1';$input.protocol='causal-fresh-v9';$input.model='CPB-v9.market-01'
Write-JsonNew 'doc/causal_fresh_v9/input-manifest.json' $input
$policy='original-position-four-bin-shared-global-half-mixture-v1'
$empty='zero-whole-export-slice-after-bias-v1'
$card.kind='causal_fresh_v9_prospective_card_v1';$card.protocol='causal-fresh-v9'
$card.declared_utc=$null;$card.frozen_utc=$null;$card.stage='DRAFT_NOT_FROZEN_NOT_EXECUTION_ADMITTED'
$card.model='CPB-v9';$card.instance='CPB-v9.market-01';$card.initial_quality_label='CPB-v9.market-01.init'
$card.execution_admitted=$false;$card.application_selection_changed=$false
$card.mechanism.kind=$policy;$card.mechanism.temporal_pooling_policy=$policy
$card.mechanism.temporal_pooling_mix=@(0.5,0.5);$card.mechanism.empty_local_bin_policy=$empty
$card.mechanism.architecture='V8 causal contextual patch states; each nonempty anchored original-position bin forms half its local masked-softmax hidden plus half the shared whole-history masked-softmax hidden, before one application of that bin''s eight rows plus bias from a single freshly initialized32x64 export tensor. Empty local bins zero the whole eight-coordinate slice after bias. Concatenate32, then unchanged global mixing and content20+literal12+mean4+uncertainty4 native40.'
$card.mechanism.controlled_mechanism='Only fixed temporal mixture changes versus freshly constructed V8. Both temporal routes use the same owned kernel and one fresh export tensor with unchanged parameter shapes/registration/RNG draw order. No trained values are loaded. Mix anchored_local=0.5,whole_history=0.5 is fixed prospectively; no rescue/tuning.'
$card.mechanism.capacity_confounds='Same native40/counts/constructor draw order/D40 heads as V8; existing downstream global mixing remains. The blend changes routing,activation scales and FLOPs. One trajectory on known development populations cannot identify the cause of a quality difference. Content0:20 may change initially;only20:40 plus fresh-statistic numeric tensors are paired versus V8.'
$card.mechanism.allowed_typed_resume='Only fresh V9 lineage; objective/gradient/layout plus temporal_pooling_policy,Long[5] partition,Double[2] mix in anchored_local/whole_history order,and empty_local_bin_policy at inference root/payload and TRAIN root/encoder/encoder.payload. Reject missing,wrong value/dtype/shape/order,old V8/V6 kinds; preserve all optimizer moments/counters/order/data identities.'
$card.mechanism.fresh_initialization_order='Same V8 parameter registration/shapes/RNG draw order under943001; all294752 learned values and optimizer state fresh. A single fresh export tensor is shared by the two pooling routes; existing never means imported trained weights.'
$card.mechanism.empty_bin_policy='Original saved position membership. All-padding rows finite zero; all empty anchored bins zero the entire corresponding export slice after its one bias even when global history is nonempty; invalid hidden values cleared before pooling.'
$card.mechanism.temporal_projection_learning+=' Three unit Hadamard between-bin directions each require at least one nonzero observed data-gradient event and movement event, plus positive overall final between-bin projection-change L2. Signed individual final coefficients are descriptive; they need not each be nonzero.'
$card.mechanism.causal_endpoint_semantics='Every slice is a coordinate of the embedding at the current requested cutoff t,not an embedding stamped at its bin endpoint. Shared whole-history uses only valid states at or before t. Later-in-window past may enter older anchored slices only within this endpoint embedding. Unchanged causal Transformer/backward relation bank and exact future-value AND future-mask invariance at all cutoffs are mandatory.'
$card.mechanism.same_export_weights_means='One single freshly initialized export tensor shared across the two pooling routes;unchanged shape/registration/draw order;never V8/V6 trained values'
$card.training.recipe_id='causal-v9-original-position-half-global-local-temporal-summary-v1'
$card.training.temporal_pooling_policy=$policy;$card.training.temporal_pooling_mix=@(0.5,0.5);$card.training.empty_local_bin_policy=$empty
$card.training.component_evidence+=' Add2048 observed between-bin Hadamard gradient/movement rows,derived in F64 from the same saved/current gradients and steps;no new forwards or actual-witness array fields.'
$card.quality.primary_methods=@('CPB-v3.market-01','CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01','CPB-v9.market-01.init','CPB-v9.market-01')
$card.quality.no_historical_encoder_rows_in_primary_tables=$false
$card.quality.primary_clean_control_rows_only=$true
$card.quality.clean_baseline_control=$controls.reuse_only
$card.quality.primary_report='Six saved clean final controls plus own newly exported/measured initial0/final2048;four separate market task tables/four TEMPO panels,physical errors,full neural repetitions/query components/cost. Pending runtime producers stay pending until independent audits/pairings and a separate trust join. No automatic winner/application selection.'
$card.advancement.policy='All22 exact reviewed strongest V3–V8 conditions plus mandatory full trust/coverage;raw producer numeric/coverage and pending trust remain separate. No automatic winner or selection.'
$card.advancement.goal_protocol=$goalPin;$card.advancement.criteria=$goal.criteria;$card.advancement.criterion_count=22
$card.criteria.quality_interpretation='Exact prospectively reviewed strongest V8-retention22 protocol e792;no V1/V2/RPB floors/no TRAIN NLL gate. Historical pass counts retain their own frozen cards. Blend changes routing/scales/FLOPs and cannot establish cause.'
$card.criteria.establishment='Full trust including both actual saved causality proofs,75 identifiable tensor learning/four row+local contrast groups/three between-bin directions,13 partial20:40/statistic pairings and independent training/quality audits establishes a measured candidate. Advancement additionally requires all22 prospective conditions and full coverage. Winner/application selection remains separate and false.'
$card.criteria.trust_admission=@(
 'All76learned tensors/294752values and optimizer freshly constructed with identical V8 draw order;fresh4096TRAIN input/bank statistics,source_updates0,no old model/statistic import',
 '75 identifiable tensors have actual CUDA nonzero data-gradient and movement events;294751 is tensor capacity,not every-scalar learning. Scalarpool_score.bias gauge remains registered/optimized/observed and is the sole exemption;weightdecay is not data learning',
 'Four export8-row groups and four within-bin centered position groups each show data-gradient/movement events and final changes. Each of three Hadamard between-bin directions has data-gradient+movement events and overall final between-bin projection changes;no each-final-coefficient gate',
 'All128cutoffs/2350suffix/30internal temporal cases initialized/trained and actual saved0/2048;future values AND masks/nonfinite suffix cannot change endpoint native40;past-positive interventions',
 'All four temporal typed fields validated at inference root/payload and TRAIN root/encoder/encoder.payload,including missing/wrong policy/mix/empty/partition/dtype/shape/order. Actual typed512resume exact and controlled engineering next-step parity',
 'Exclusive independently reviewed prospective card/descriptor/source/input/compiler/binary closure before any build/test/CUDA;actual file/binary/TU counts derived from frozen owners,not assumed',
 'Unchanged nonvacuous composite-objective CUDA fixture;separate manual half-mixture/adjoint/trace/empty/common-gauge/relative-bin sensitivity/four-group/three-between-direction CUDA fixture',
 'Exactly13 DATA/statistic initial pairings versus V8: eight feature value matrices only20:40 with whole valid/IDs/labels;five fresh-statistic archives whole numeric tensors. Content0:20 not paired;new own initial heads;no old model/score replay',
 'Independent saved-training/quality arithmetic audits,all fullcoverage finite metrics and separate final-trust evidence join before any advancement claim'
)
$card.input_manifest=Get-FilePin 'doc/causal_fresh_v9/input-manifest.json'
$card.baseline_preservation=Get-FilePin 'doc/causal_fresh_v8/final-preservation.json'
$card.pre_execution_preservation=Get-FilePin 'doc/causal_fresh_v9/preservation-before.json'
$card.saved_TRAIN_diagnosis=[ordered]@{
 card=(Get-FilePin 'doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.json');result=(Get-FilePin 'doc/causal_v6_v8_train_diagnosis/diagnosis.json')
 independent_audit=(Get-FilePin 'doc/causal_v6_v8_train_diagnosis/independent-audit.json');human_report=(Get-FilePin 'doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS.md')
 completion=(Get-FilePin 'doc/causal_v6_v8_train_diagnosis/completion.json')
 scope='Completed independent savedTRAIN-only V6/V8 diagnosis;bounded scalar/adjoint arithmetic,zero fullencoder/librarymodel/head/refit calls;all old94checkpoints/source preserved'
 interpretation='Diagnosis selected no new model;root separately chose fixed half-global/local hypothesis after diagnosis completion. No causal or counterfactual claim.'
}
$card.comparison_control=$controls;$card.scoped_reference='CPB-v8.market-01'
$card.scoped_references=@('CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01')
$card.retention_independent_review=$reviewPin;$card.retention_application_admission=$admissionPin
$card.geometry_override='Native40,unchanged D40 head recipes/seeds/counts and constructor shapes/draws. Fixed half-global/local blend is the sole selected forward mechanism;globalclip1/objective unchanged.'
$card.prospective_descriptor='code/encoders/causal_patch_bottleneck_mae/versions/CPB-v9.json'
$card.initial_pairing.reference='CPB-v8.market-01 initial0 DATA/statistic artifacts only'
$v8Mirror=Get-Content -LiteralPath (Join-Path $root 'doc/causal_fresh_v8/complete-mirror-attempt-001.json') -Raw|ConvertFrom-Json -AsHashtable
foreach($reference in $card.initial_pairing.references){
 $container=$reference.reference_data_archive.path.Replace('causal-fresh-v6/','causal-fresh-v8/').Replace('cpb_v6-','cpb_v8-')
 $matches=@($v8Mirror.files|Where-Object { $_.container_path -eq $container })
 if($matches.Count-ne1){throw ('One exact V8 initial DATA pin required: '+$container)}
 $actual=Get-FilePin $matches[0].path
 if($actual.sha256-ne$matches[0].sha256-or$actual.bytes-ne$matches[0].bytes){throw 'V8 initial DATA reference changed'}
 $reference.reference_data_archive=[ordered]@{path=$container;bytes=$actual.bytes;sha256=$actual.sha256}
 $reference.candidate_relative_to_attempt=$reference.candidate_relative_to_attempt.Replace('cpb_v8-','cpb_v9-')
}
$card.trust_receipt_objective_fields.temporal_pooling_policy=$policy
$card.trust_receipt_objective_fields.temporal_pooling_mix=@(0.5,0.5)
$card.trust_receipt_objective_fields.empty_local_bin_policy=$empty
$card.mechanism_proof_contract.PASS_prefix='PASS CPB-v9 huber1-mean-plus-detached-error-gaussian-variance-v1:'
$temporal=$card.temporal_pooling_proof_contract
$temporal.status='DRAFT_source_owner_frozen_independent_review_before_any_execution'
$temporal.temporal_pooling_policy=$policy;$temporal.temporal_pooling_mix=@(0.5,0.5);$temporal.empty_local_bin_policy=$empty
$temporal.PASS_prefix='PASS CPB-v9 '+$policy+':'
$temporal.log_tokens=@('manual original-position half-mixture pooling parity','four 8-row projection gradient and movement','three between-bin Hadamard contrast gradient and movement','empty-bin finite zero','relative-bin global sensitivity','common softmax gauge invariance','synthetic optimizer steps=1')
$temporal.required_binary_count=$null;$temporal.required_TU_count=$null
$proof=$temporal.required_proof_fields;$proof.model='CPB-v9';$proof.temporal_pooling_policy=$policy
$proof.temporal_pooling_mix=@(0.5,0.5);$proof.empty_local_bin_policy=$empty
foreach($key in @('softmax_gauge_invariance','fixture_backbone_Model_encode_calls','fixture_direct_pooling_kernel_calls','public_Encoder_calls')){$proof.Remove($key)}
$proof.common_softmax_gauge_invariance=$true;$proof.shared_global_half_mixture=$true
$proof.relative_bin_global_sensitivity=$true;$proof.three_between_bin_contrasts_gradient_and_movement=$true
$proof.backbone_encode_calls=1;$proof.public_encoder_calls=0;$proof.direct_pool_kernel_calls=5
$temporal.binary='test_cpb_v9_temporal_pooling'
$card.temporal_learning_receipt_contract.kind='cpb_v9_actual_temporal_pooling_learning_v1'
$card.temporal_learning_receipt_contract.temporal_pooling_policy=$policy
$card.temporal_learning_receipt_contract.temporal_pooling_mix=@(0.5,0.5)
$card.temporal_learning_receipt_contract.empty_local_bin_policy=$empty
$card.temporal_learning_receipt_contract.between_bin_rows=2048
$card.temporal_learning_receipt_contract.receipt_artifacts=@('temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv','temporal-between-bin-position-learning.csv')
$card.temporal_learning_receipt_contract.summary_positive_flags+=@('all_three_between_bin_position_directions_learned','final_between_bin_position_projection_changed')
$card.temporal_learning_receipt_contract.between_bin_positions=[ordered]@{
 basis='hadamard-full16-signs-div4-v1';basis_signs=@(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1));full_position_divisor=4;position_bin_width=4;rows=2048
 columns=@('update','between_bin_gradient_l2','between_bin_movement_l2','contrast_0_gradient','contrast_1_gradient','contrast_2_gradient','contrast_0_movement','contrast_1_movement','contrast_2_movement')
 gradient_definition='observed-post-globalclip-data-gradient';F64_before_projection_and_subtraction=$true;parseval_identity_required=$true
 each_direction_positive_data_gradient_and_movement_events=$true;overall_final_projection_change_l2_positive=$true
 each_final_signed_coefficient_nonzero_required=$false;movement_includes_AdamW_weight_decay=$true;weight_decay_counted_as_data_learning=$false;extra_encoder_calls=0
}
$card.fresh_initialization_JSON_exception.required_other_fields='All objective/delta/coefficients/detach/reportingNLL and temporalpolicy/partition/mix/empty fields plus75/76gauge fields remain mandatory;gradient_policy absent only in this constructor receipt'
$card.source_contract_status='DRAFT_NOT_ADMITTED_actual_owner_counts_and_independent_source_review_pending'
$card.source_admission='No AST/build/test/CUDA/model/head/body execution authorized by this draft. Exact final owner source counts/compiler/TU/binary closure and independent preflight must be frozen prospectively,without a source/card hash cycle.'
$card.source_pins_after_owner_freeze=@()
foreach($key in @('independent_static_helper_review','independent_static_helper_review_amendment','source_review_scope')){$card.Remove($key)}
$card.source_counts=[ordered]@{family_inside=$null;family_external_header=$null;research=$null;translation_units=$null;binaries=$null;capsule_sources=$null;status='PENDING_FINAL_OWNER_BINDINGS'}
$card.draft_copy_provenance=[ordered]@{
 kind='full_physical_V8_card_and_repaired_native_Docker_workflow_copies_then_owned_V9_edits'
 card=(Get-FilePin 'doc/CAUSAL_FRESH_V8_CARD.json');markdown=(Get-FilePin 'doc/CAUSAL_FRESH_V8_CARD.md')
 direct_orchestration_parent=(Get-FilePin 'doc/causal_fresh_v8/run-container-recovery-001.ps1')
 parent_copies=$copied;old_files_modified=$false;source_only=$true
}
Write-JsonNew 'doc/CAUSAL_FRESH_V9_CARD.json' $card
Write-JsonNew 'doc/causal_fresh_v9/draft-metadata-copy-receipt.json' ([ordered]@{status='draft_prepared_not_execution_admitted';controls=$controls;goal=$goalPin;review=$reviewPin;admission=$admissionPin;raw_controls=$raw;parent_copies=$copied;initial_DATA_references=13;checkpoint_numeric_reads=0;model_calls=0;head_fits=0;old_files_modified=$false})
Get-FilePin 'doc/CAUSAL_FRESH_V9_CARD.json'|ConvertTo-Json -Depth 4

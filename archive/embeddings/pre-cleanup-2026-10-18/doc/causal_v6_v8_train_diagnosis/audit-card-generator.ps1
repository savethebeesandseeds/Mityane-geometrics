# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host metadata/hash inspection only; refuses an absent or incomplete diagnosis.
param(
 [Parameter(Mandatory=$true)][switch]$RootReviewedAuditorSources,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainDiagnosisSha256
)
$ErrorActionPreference='Stop'
if(-not$RootReviewedAuditorSources){throw 'Independent source review and explicit root result admission required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$documentRoot='doc/causal_v6_v8_train_diagnosis/'
$destination=Join-Path $researchRoot ($documentRoot+'independent-audit-card.json')
if(Test-Path -LiteralPath $destination){throw 'Preserve the frozen independent audit card'}
function Pin([string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains('\')-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded repository path required'}
 $file=Join-Path $researchRoot $relative;$item=Get-Item -LiteralPath $file
 if($item.PSIsContainer-or($item.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Regular nonlinked metadata required'}
 [ordered]@{path=$relative;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$mainCard=Pin 'doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.json'
if($mainCard.bytes-ne81588-or$mainCard.sha256-ne'49b22f1360593f2f0dd6d8ebb7fd8b0d4baa4749fe01ad8875fa79e5fe529829'){throw 'Exact frozen main card required'}
$main=Get-Content -LiteralPath (Join-Path $researchRoot $mainCard.path) -Raw|ConvertFrom-Json
if($main.kind-ne'cpb_v6_v8_saved_train_diagnosis_card_v1'-or$main.stage-ne'prospective_before_diagnostic_arithmetic'-or$main.allowed_inputs.Count-ne52-or$main.preservation_pins.Count-ne33-or$main.preservation_pin_count-ne33-or$main.pre_execution_preservation.sha256-ne'919b4f11610e3ab61b63afbc3612145506e4c47a3be3c777f119d47c49e7126a'){throw 'Exact52/33/broad919b main declaration required'}
$diagnosis=Pin ($documentRoot+'diagnosis.json')
if($diagnosis.sha256-ne$MainDiagnosisSha256){throw 'Actual root-supplied completed diagnosis hash required before JSON parse'}
$result=Get-Content -LiteralPath (Join-Path $researchRoot $diagnosis.path) -Raw|ConvertFrom-Json
if($result.kind-ne'cpb_v6_v8_saved_TRAIN_diagnosis_v1'-or$result.status-ne'passed'-or$result.card.bytes-ne$mainCard.bytes-or$result.card.sha256-ne$mainCard.sha256){throw 'Completed exact main diagnosis required before audit-card freeze'}
foreach($counter in $result.counters.PSObject.Properties){if($counter.Value-ne0){throw 'Main diagnosis reports forbidden execution'}}
if(-not$result.paired_initial_native20_40_TRAIN_values_exact-or$result.whole_native40_equality_claimed-or$result.V8_only_temporal_learning_observations.group_rows-ne8192-or$result.V8_only_temporal_learning_observations.gauge_rows-ne2048){throw 'Actual partial initial pairing and saved temporal scope required'}
$expected=[ordered]@{'audit.py'='b528beb76716e9e2ffa2313309253213ce1fe977785bc06797c9cfb996732b86';'archive_reader.py'='1bbf6efad4e1222a9fc53a0dad46d0ff2b0ccad63e269af714968ebb6e738db9';'component_math.py'='18084fb7e78322c09946626661d853628bf063e0e45df9fd1fe053a0a684c514';'README.md'='13d308bdf7c917111967ec394975a4eaf74561b8fc9756a1bba45b229613b300';'provenance.json'='5a480e11425037a9f0550d8f2c804ff64e730e6fa782bab8ef8eb5aefb6371bf'}
$sources=@()
foreach($file in $expected.Keys){$p=Pin ('code/research/causal_v6_v8_train_diagnosis_audit/'+$file);if($p.sha256-ne$expected[$file]){throw 'Root-reviewed auditor source changed'};$sources +=[ordered]@{file=$file;path=$p.path;bytes=$p.bytes;sha256=$p.sha256}}
$helpers=@()
foreach($name in @('audit-card-generator.ps1','audit-stage.ps1','audit-execute.py','audit-mirror.ps1')){$helpers+=Pin ($documentRoot+$name)}
$card=[ordered]@{kind='cpb_v6_v8_TRAIN_diagnosis_independent_audit_card_v1';copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.';declared_utc=[DateTime]::UtcNow.ToString('o');stage='Frozen before independent numeric execution; main diagnosis completed once and immutable';main_card=$mainCard;main_diagnosis=$diagnosis;main_static_preflight=(Pin ($documentRoot+'preflight-review.json'));pre_execution_preservation=(Pin 'doc/causal_fresh_v8/final-preservation.json');auditor_sources=$sources;source_provenance=$sources[4];orchestration_helpers=$helpers;execution_wrapper=$helpers[2];numeric_input_scope='Exactly52 main-card roles with independent per-file field allowlists; all preservation hashes checked before storage values. No extra input or held-out selector.';all112_scope='Four tasks, three saved head repetitions, native40 block moments and exact saved ridge decomposition; saved Huber/variance/total rows in BOTH versions; no fit or forward';first8_scope='Four initial/final stage-matched query/decoder pairs; supported F64 kernels and local mean4/uncertainty4 adjoints only';temporal_bins=@(@(0,32),@(32,64),@(64,96),@(96,128));temporal_scope='Independent observed rawF64 and masked-error summaries only; separate support and bin-specific hierarchy, no labels/ablation/refit';actual_batch_scope='BOTH versions actual pre-step updates1/2048: support/components/artifact and CSV association only; no unmatched decoder weights/adjoints or replay';CSV_numeric_columns=[ordered]@{v6=@('update','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64','pre_clip_global_norm','clip_factor');v8=@('update','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64','pre_clip_global_norm','clip_factor')};CSV_successful_updates_each=2048;CSV_fixed_groups_each=5;CSV_other_columns_read=$false;V6_split_norm_inferred=$false;counterfactual_clip_computed=$false;gradient_limits='Both apply Huber1 mean plus detached-error variance. No cross-basis/model/stage cosine, full encoder/optimizer Jacobian, scalar-loss dominance or full norm reconstruction.';finite_differences=[ordered]@{checks=256;cells_per_stage=16;stages=4;kernels=@('Gaussian mean','Gaussian variance','MAE','Huber1');traversal='Canonical b,c,f,h first16 supported';positive_step_rules='Inherited unchanged neutral audit kernel';tolerance_absolute=1e-6;tolerance_relative=2e-4};saved_scalar_tolerance=[ordered]@{absolute=1e-10;relative=1e-9;recorded_component_identity_absolute=2e-12;recorded_component_identity_relative=2e-12};container=[ordered]@{name='cuwacunu';id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';output_root='/opt/cuwacunu/research/causal-v6-v8-train-diagnosis-audit/attempt-001'};model_calls=0;neural_forwards=0;head_fits=0;calibration_fits=0;training_updates=0;new_quality_scores=0;held_out_tensor_reads=0;old_encoder_replays=0;failures='Preserve every attempt/source/card/log. Additive correction only; never replay a producer/model/head.'}
$card.numeric_input_scope='Exactly52 main-card roles (common41+actual8+V8temporal3), exact33 preservation guards plus broad919b, all verified before storage. No extra input or held-out selector.'
$card.CSV_numeric_columns.v8=@('update','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64','pre_clip_global_norm','clip_factor')
$card.initial_pairing_coordinate_slice=@(20,40)
$card.whole_native40_equality_claimed=$false
$card.initial_mean_variance_equality_required=$false
$card.V8_split_norm_inferred=$false
$card.V8_temporal_observations=[ordered]@{group_rows=8192;gauge_rows=2048;actual_array_updates=@(1,2048);actual_array_fields_each=12;original_storage='F32';movement_arithmetic='F64 operands before subtraction/centering';summary_blocks='Overall and four fixed512 blocks';observed_post_globalclip_only=$true;V6_subgroup_counterfactual=$false;gauge_is_not_data_learning_proof=$true;checkpoint_encoder_weights_read=$false}
$card.gradient_limits='Both apply Huber1 mean plus detached-error variance and globalclip1. Bounded manual saved mean4 linear residual, variance affine/softplus-link and scalar FD arithmetic allowed. Full encoder/nonlinear parent/librarymodel/head execution, cross-basis cosine, parameter Jacobians, loss-ratio dominance, missing branch inference and optimizer replay forbidden.'
$card.execution_admitted=$false
$card.orchestration_provenance=Pin ($documentRoot+'audit-orchestration-provenance.json')
$card.root_supplied_main_diagnosis_sha256=$MainDiagnosisSha256
[IO.File]::WriteAllText($destination,(($card|ConvertTo-Json -Depth 15).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Pin ($documentRoot+'independent-audit-card.json')|ConvertTo-Json

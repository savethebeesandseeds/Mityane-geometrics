# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host metadata/hash inspection only; refuses an absent or incomplete diagnosis.
param(
 [Parameter(Mandatory=$true)][switch]$RootReviewedAuditorSources,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainDiagnosisSha256
)
$ErrorActionPreference='Stop'
if(-not$RootReviewedAuditorSources){throw 'Independent source review and explicit root result admission required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$documentRoot='doc/causal_v8_v9_train_diagnosis/'
$destination=Join-Path $researchRoot ($documentRoot+'independent-audit-card.json')
if(Test-Path -LiteralPath $destination){throw 'Preserve the frozen independent audit card'}
function Pin([string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains([char]92)-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded repository path required'}
 $file=Join-Path $researchRoot $relative;$item=Get-Item -LiteralPath $file
 if($item.PSIsContainer-or($item.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Regular nonlinked metadata required'}
 [ordered]@{path=$relative;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$mainCard=Pin 'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json'
if($mainCard.sha256-ne$MainCardSha256){throw 'External root-reviewed actual main card hash required before JSON parse'}
$main=Get-Content -LiteralPath (Join-Path $researchRoot $mainCard.path) -Raw|ConvertFrom-Json
if($main.kind-ne'cpb_v8_v9_saved_train_diagnosis_card_v1'-or$main.stage-ne'prospective_before_diagnostic_arithmetic'-or$main.allowed_inputs.Count-ne56-or$main.preservation_pins.Count-ne36-or$main.preservation_pin_count-ne36){throw 'Exact56/36/broad main declaration required'}
$guardPins=@($main.allowed_inputs)+@($main.preservation_pins)+@($main.pre_execution_preservation)
if($guardPins.Count-ne93-or@($guardPins.id|Sort-Object -Unique).Count-ne93){throw 'Exactly93 distinct declared file guards required'}
foreach($binding in $guardPins){
 $repository=switch($binding.host_repository){'cuwacunu_embedding'{$researchRoot};'cuwacunu_dev'{[IO.Path]::GetFullPath((Join-Path $researchRoot '../cuwacunu_dev'))};default{throw 'Unknown guarded repository'}}
 $relative=[string]$binding.host_path
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains([char]92)-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded file guard required'}
 $file=Get-Item -LiteralPath (Join-Path $repository $relative)
 if($file.PSIsContainer-or($file.Attributes-band[IO.FileAttributes]::ReparsePoint)-or$file.Length-ne$binding.bytes-or(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()-ne$binding.sha256){throw ('Changed main guard '+$binding.id)}
}
if($main.pre_execution_preservation.host_repository-ne'cuwacunu_embedding'-or$main.pre_execution_preservation.host_path-ne'doc/causal_fresh_v9/final-preservation.json'){throw 'Actual final99 preservation declaration required'}
$broad=Pin $main.pre_execution_preservation.host_path
$diagnosis=Pin ($documentRoot+'diagnosis.json')
if($diagnosis.sha256-ne$MainDiagnosisSha256){throw 'Actual root-supplied completed diagnosis hash required before JSON parse'}
$result=Get-Content -LiteralPath (Join-Path $researchRoot $diagnosis.path) -Raw|ConvertFrom-Json
if($result.kind-ne'cpb_v8_v9_saved_TRAIN_diagnosis_v1'-or$result.status-ne'passed'-or$result.card.bytes-ne$mainCard.bytes-or$result.card.sha256-ne$mainCard.sha256){throw 'Completed exact main diagnosis required before audit-card freeze'}
foreach($counter in $result.counters.PSObject.Properties){if($counter.Value-ne0){throw 'Main diagnosis reports forbidden execution'}}
if(-not$result.paired_initial_native20_40_TRAIN_values_exact-or$result.whole_native40_equality_claimed){throw 'Actual partial initial pairing required'}
foreach($version in @('v8','v9')){if($result.temporal_learning_observations.$version.group_rows-ne8192-or$result.temporal_learning_observations.$version.gauge_rows-ne2048){throw 'Both saved temporal scopes required'}}
if($result.temporal_learning_observations.v9.between_bin_positions.rows_checked-ne2048){throw 'Actual V9 saved between-bin scope required'}
$expected=[ordered]@{'audit.py'='af116a50e8d6418c8496883a522fef1c1d886db1b01d0009b12806aa6694d399';'archive_reader.py'='1bbf6efad4e1222a9fc53a0dad46d0ff2b0ccad63e269af714968ebb6e738db9';'component_math.py'='1ee9107397e1390bb60784e3fb61a0cd205fe87676fe070e90c0205ab8cd38b8';'README.md'='94272f43e1c0f02b70ebefd6ce10382d295be11694e4364441f61bc464ed841e';'provenance.json'='6518f01ad0112d7ed51d3975d116e99c0e64ffbe38e81753aa6944c9cd9fbc95'}
$sources=@()
foreach($file in $expected.Keys){$p=Pin ('code/research/causal_v8_v9_train_diagnosis_audit/'+$file);if($p.sha256-ne$expected[$file]){throw 'Root-reviewed auditor source changed'};$sources +=[ordered]@{file=$file;path=$p.path;bytes=$p.bytes;sha256=$p.sha256}}
$helpers=@()
foreach($name in @('audit-card-generator.ps1','audit-stage.ps1','audit-execute.py','audit-mirror.ps1')){$helpers+=Pin ($documentRoot+$name)}
$card=[ordered]@{
 kind='cpb_v8_v9_TRAIN_diagnosis_independent_audit_card_v1'
 copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 declared_utc=[DateTime]::UtcNow.ToString('o')
 stage='Frozen before independent numeric execution; actual main diagnosis completed once and immutable'
 main_card=$mainCard;main_diagnosis=$diagnosis
 main_static_preflight=(Pin ($documentRoot+'preflight-review.json'))
 pre_execution_preservation=$broad
 auditor_sources=$sources;source_provenance=$sources[4]
 orchestration_helpers=$helpers;execution_wrapper=$helpers[2]
 numeric_input_scope='Exactly56 main-card roles, exact36 preservation guards plus the actual final99 broad receipt; all93 hashes verified before storage. No extra input or held-out selector.'
 all112_scope='Four tasks, three saved head repetitions, native40 block moments and saved ridge decomposition; saved Huber/variance/total rows in BOTH versions; no fit or forward'
 first8_scope='Four initial/final stage-matched query/decoder pairs; supported F64 kernels and local mean4/uncertainty4 adjoints only'
 temporal_bins=@(@(0,32),@(32,64),@(64,96),@(96,128))
 temporal_scope='Independent observed rawF64 and masked-error summaries only; separate support and bin-specific hierarchy, no labels/ablation/refit'
 actual_batch_scope='BOTH versions actual pre-step updates1/2048: support/components/artifact and CSV association only; no unmatched decoder weights/adjoints or replay'
 CSV_numeric_columns=[ordered]@{
  v8=@('update','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64','pre_clip_global_norm','clip_factor')
  v9=@('update','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64','pre_clip_global_norm','clip_factor')
 }
 CSV_successful_updates_each=2048;CSV_fixed_groups_each=5;CSV_other_columns_read=$false
 V8_split_norm_inferred=$false;V9_split_norm_inferred=$false;counterfactual_clip_computed=$false
 gradient_limits='Both apply Huber1 mean plus detached-error variance and globalclip1. Bounded manual saved mean4 linear residual, variance affine/softplus-link and scalar FD arithmetic allowed. Full encoder/nonlinear parent/librarymodel/head execution, cross-basis cosine, parameter Jacobians, loss-ratio dominance, missing branch inference and optimizer replay forbidden.'
 finite_differences=[ordered]@{checks=256;cells_per_stage=16;stages=4;kernels=@('Gaussian mean','Gaussian variance','MAE','Huber1');traversal='Canonical b,c,f,h first16 supported';positive_step_rules='Inherited unchanged neutral audit kernel';tolerance_absolute=1e-6;tolerance_relative=2e-4}
 saved_scalar_tolerance=[ordered]@{absolute=1e-10;relative=1e-9;recorded_component_identity_absolute=2e-12;recorded_component_identity_relative=2e-12;new_between_bin_absolute=2e-12;new_between_bin_relative=2e-12}
 container=[ordered]@{name='cuwacunu';id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';output_root='/opt/cuwacunu/research/causal-v8-v9-train-diagnosis-audit/attempt-001'}
 model_calls=0;neural_forwards=0;head_fits=0;calibration_fits=0;training_updates=0;new_quality_scores=0;held_out_tensor_reads=0;old_encoder_replays=0
 failures='Preserve every new attempt/source/card/log. Additive correction only; never replay a producer/model/head. No historical failed-audit claim is made for this new workflow.'
 initial_pairing_coordinate_slice=@(20,40);whole_native40_equality_claimed=$false;initial_mean_variance_equality_required=$false
 temporal_observations=[ordered]@{versions=@('v8','v9');group_rows_each=8192;gauge_rows_each=2048;actual_array_updates=@(1,2048);actual_array_fields_each=12;original_storage='F32';movement_arithmetic='F64 operands before subtraction/centering';summary_blocks='Overall and four fixed512 blocks';observed_post_globalclip_only=$true;subgroup_counterfactuals=$false;gauge_is_not_data_learning_proof=$true;checkpoint_encoder_weights_read=$false}
 V9_between_bin_observations=[ordered]@{rows=2048;basis='hadamard-full16-signs-div4-v1';signs=@(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1));individual_final_coefficient_nonzero_gate=$false;actual_array_updates=@(1,2048);extra_tensor_arrays=0;full_trajectory_raw_gradients_reconstructed=$false}
 expected_input_roles=56;expected_preservation_roles=36;expected_data_and_preservation_pins=93
}
$card.execution_admitted=$false
$card.orchestration_provenance=Pin ($documentRoot+'audit-orchestration-provenance.json')
$card.root_supplied_main_card_sha256=$MainCardSha256
$card.root_supplied_main_diagnosis_sha256=$MainDiagnosisSha256
[IO.File]::WriteAllText($destination,(($card|ConvertTo-Json -Depth 15).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Pin ($documentRoot+'independent-audit-card.json')|ConvertTo-Json

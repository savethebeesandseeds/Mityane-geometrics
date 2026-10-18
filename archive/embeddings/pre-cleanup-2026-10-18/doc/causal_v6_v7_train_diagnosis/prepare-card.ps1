# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Owned adaptation of the frozen V5/V6 card pattern; file/hash metadata only.
# Preparing this helper does not declare a card or authorize numeric execution.
param(
 [switch]$RootReviewed,
 [string]$GoalProtocolPath='doc/clean_causal_encoder_search_goal_v7_retention.json',
 [ValidatePattern('^attempt-[0-9]{3}$')][string]$DiagnosisAttempt='attempt-001'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootReviewed){throw 'Wait for root review and completed final V7 metadata/preservation/retention context'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$cardPath='doc/CAUSAL_V6_V7_TRAIN_DIAGNOSIS_CARD.json';$markdownPath='doc/CAUSAL_V6_V7_TRAIN_DIAGNOSIS_CARD.md'
$futureSource='/opt/cuwacunu/research/causal-v6-v7-train-diagnosis/'+$DiagnosisAttempt+'/source'
function Require([bool]$ok,[string]$why){if(-not$ok){throw $why}}
function Rooted([string]$label,[string]$repo='cuwacunu_embedding'){
 Require ($repo-in@('cuwacunu_embedding','cuwacunu_dev')-and-not[IO.Path]::IsPathRooted($label)-and$label-notmatch':'-and@($label.Split('/')|Where-Object {$_-in@('','..','.')} ).Count-eq0) 'Bounded research/dev label required'
 $base=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repo));$path=[IO.Path]::GetFullPath((Join-Path $base $label))
 Require ($path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Host path escaped its repository'
 return $path
}
function Pin([string]$label,[string]$repo='cuwacunu_embedding'){
 $file=Get-Item -LiteralPath (Rooted $label $repo)
 Require (-not$file.PSIsContainer-and-not($file.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'Unlinked regular pinned file required'
 return [ordered]@{path=$label;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Read([string]$label){Get-Content -LiteralPath (Rooted $label) -Raw|ConvertFrom-Json}
function Exact($actual,$expected,[string]$why){Require ($actual.bytes-eq$expected.bytes-and$actual.sha256-eq$expected.sha256) $why}
Require (-not(Test-Path -LiteralPath (Rooted $cardPath))-and-not(Test-Path -LiteralPath (Rooted $markdownPath))) 'Preserve every prior card and refusal; this generator writes new cards exclusively'
$parentCard=Pin 'doc/CAUSAL_V5_V6_TRAIN_DIAGNOSIS_CARD.json'
Require ($parentCard.sha256-eq'bf4c92e4edff4ba055a83ba6d699ace5e20dd4de654d55fc20e279cb389efff5') 'Immutable direct parent card pattern'
$establishmentPath='doc/causal_fresh_v7/final-establishment.json';$establishment=Read $establishmentPath;$establishmentPin=Pin $establishmentPath
Require ($establishmentPin.sha256-eq'7678e84a676453c0294f0654af6117e06246f1639f4e67f9ff78e9cf51049512'-and$establishment.status-eq'passed'-and$establishment.candidate-eq'CPB-v7.market-01'-and$establishment.trust_admission_passed-and-not$establishment.winner_selection_made) 'Actual independent V7 establishment context'
$broadPath='doc/causal_fresh_v7/final-preservation.json';$broad=Read $broadPath;$broadPin=Pin $broadPath
Require ($broad.status-eq'passed'-and$broad.candidate-eq'CPB-v7.market-01'-and$broad.critical_checkpoint_total-eq89-and-not$broad.baseline_changed-and-not$broad.application_changed) 'Completed final V7 preservation required before declaring diagnostic inputs'
$ledgerPath='doc/clean_causal_encoder_research_ledger.json';$catalogPath='code/encoders/causal_patch_bottleneck_mae/versions/current.json'
Exact (Pin $ledgerPath) $broad.current_ledger 'Current post-V7 ledger must match final preservation'
Exact (Pin $catalogPath) $broad.current_catalog 'Current post-V7 catalog must match final preservation'
$ledger=Read $ledgerPath;$catalog=Read $catalogPath
Require ($ledger.baseline-eq'CPB-v3.market-01'-and@($ledger.candidates|Where-Object instance -EQ 'CPB-v7.market-01').Count-eq1-and$catalog.active_application-eq'CPB-v3.market-01'-and$catalog.fresh_research_candidate.instance-eq'CPB-v7.market-01') 'Applied V7 research metadata with selected V3 unchanged'
$goal=Read $GoalProtocolPath;$goalPin=Pin $GoalProtocolPath
Require ($goal.kind-eq'clean_causal_encoder_search_goal_v7_retention_protocol_v1'-and$goal.criterion_count-eq22-and$goal.additional_train_NLL_gate-eq$false) 'Reviewed strongest V3-V7 future context only; no extra TRAIN NLL gate'
$mirror6Path='doc/causal_fresh_v6/complete-mirror-proof-recovery-002-attempt-001.json';$mirror7Path='doc/causal_fresh_v7/complete-mirror-attempt-001.json'
Exact (Pin $mirror6Path) $broad.complete_V6_mirror 'Frozen complete V6 mirror binding'
Exact (Pin $mirror7Path) $broad.complete_V7_mirror 'Frozen complete V7 mirror binding'
$mirrors=@{v6=(Read $mirror6Path);v7=(Read $mirror7Path)}
foreach($version in @('v6','v7')){Require ($mirrors[$version].status-eq'passed'-and$mirrors[$version].scope-eq'Complete'-and$mirrors[$version].all_mirrored_pt_ignored) 'Existing completed main evidence mirror required'}
$script:allowed=@();$script:preservation=@();$script:metadataCopies=@()
function Mirrored([string]$version,[string]$relative){
 $label='output/runs/causal-fresh-'+$version+'/run-001/'+$relative
 $found=@($mirrors[$version].files|Where-Object path -EQ $label)
 Require ($found.Count-eq1) ('One canonical completed artifact required: '+$label)
 Exact (Pin $label) $found[0] ('Completed artifact changed: '+$label)
 return $found[0]
}
function Add-Input([string]$id,[string]$version,[string]$relative,[string[]]$fields,[string]$role='TRAIN_archive'){
 $pin=Mirrored $version $relative
 Require ($pin.container_path-notmatch'validation') 'Held-out numeric archive refused'
 $binding=[ordered]@{id=$id;host_repository='cuwacunu_embedding';host_path=$pin.path;container_path=$pin.container_path;bytes=$pin.bytes;sha256=$pin.sha256;role=$role}
 if($role-eq'TRAIN_archive'){$binding.allowed_tensor_fields=@($fields|Sort-Object -Unique)}
 elseif($role-eq'aggregate_JSON_TRAIN_key_only'){$binding.numeric_json_selector='train'}
 elseif($role-eq'TRAIN_CSV_2048_scalar_rows_only'){$binding.allowed_CSV_fields=$csvFields[$version];$binding.numeric_update_scope='all2048successfulupdates';$binding.fixed_update_blocks=@(@(1,512),@(513,1024),@(1025,1536),@(1537,2048))}
 $script:allowed+=$binding
}
function Add-Preserve([string]$id,[string]$label,[string]$containerPath='', [string]$repo='cuwacunu_embedding'){
 $pin=Pin $label $repo
 if(-not$containerPath){$containerPath=$futureSource+'/preservation/'+$id+[IO.Path]::GetExtension($label);$script:metadataCopies+=[ordered]@{host_repository=$repo;host_path=$label;container_path=$containerPath;bytes=$pin.bytes;sha256=$pin.sha256}}
 $script:preservation+=[ordered]@{id=$id;host_repository=$repo;host_path=$label;container_path=$containerPath;bytes=$pin.bytes;sha256=$pin.sha256;role='preservation_only_no_numeric_access'}
}
$featureFields=@('row_ids','values','valid')
$queryFields=@('row_ids','timestamps','target_mask','visible','observed','hidden','raw','targets','mean','variance','native','row_mae','row_nll','row_mean_huber','row_variance_nll','row_total_objective')
$headFields=@('row_ids','group_ids','targets','target_valid','valid','ridge_predictions','neural_predictions')
$fitFields=@('feature_mean','feature_scale','ridge_weights','target_scale','target_mean','ridge_intercept','actual_seed')
$checkpointFields=@('artifact_kind','model_tag','initialization_kind','pretrained_ancestor','source_updates','causal_updates','initialization_seed','channel_ids','payload.model.encoder.change_decoder','payload.model.variance_projection','payload.model.variance_bias','payload.link_constants_f32','payload.softplus_parameters_f64','gradient_policy','payload.gradient_policy')
foreach($field in @('native_layout','mean_width','uncertainty_width','native_width','objective_policy','mean_huber_delta_f64','loss_coefficients_f64','variance_error_detached','direct_gradient_routes','reporting_nll_backpropagated')){$checkpointFields+=@($field,('payload.'+$field))}
$clipTyped=@('gradient_clip_policy','gradient_clip_max_norms_f64','gradient_clip_norm_type_f64','gradient_clip_partition_v1','gradient_clip_parameter_names')
$actualFields=@('update','attempt','row_ids','timestamps','observed','hidden','context_deleted','visible','target_mask','eligible_channels','targets','mean','variance','native','row_mean_huber','row_variance_nll','row_total_objective','mean_Huber_F64','variance_NLL_F64','total_objective_F64','Gaussian_NLL_F64')
$clipFields=@{v6=@('pre_clip_global_norm','clip_factor');v7=@('mean_pre_clip_norm','uncertainty_pre_clip_norm','mean_clip_factor','uncertainty_clip_factor')}
$csvFields=@{};foreach($version in @('v6','v7')){$csvFields[$version]=@('update','Gaussian_NLL_F64','mean_Huber_F64','variance_NLL_F64','total_objective_F64')+@($clipFields[$version])}
$targetLabel='output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/train-targets.pt'
$targetPin=Pin $targetLabel 'cuwacunu_dev'
$inputManifest=Read 'doc/causal_fresh_v7/input-manifest.json';$targetReference=@($inputManifest.files|Where-Object path -EQ $targetLabel)
Require ($targetReference.Count-eq1) 'Exactly one existing shared TRAIN target declaration';Exact $targetPin $targetReference[0] 'Original shared TRAIN targets'
$script:allowed+=[ordered]@{id='train_targets';host_repository='cuwacunu_dev';host_path=$targetLabel;container_path=('/cuwacunu/'+$targetLabel);bytes=$targetPin.bytes;sha256=$targetPin.sha256;role='TRAIN_archive';allowed_tensor_fields=@('row_ids','group_ids','targets','valid')}
foreach($version in @('v6','v7')){
 $typed=@($checkpointFields);if($version-eq'v7'){foreach($field in $clipTyped){$typed+=@($field,('payload.'+$field))}}
 foreach($stage in @('initial','final')){
  $quality='quality-001/'+$stage+'/'
  Add-Input ($version+'_'+$stage+'_features') $version ('quality-001/features/'+$stage+'/market/cpb_'+$version+'-train.pt') $featureFields
  Add-Input ($version+'_'+$stage+'_query') $version ('witnesses/'+$stage+'-train-query.pt') $queryFields
  Add-Input ($version+'_'+$stage+'_query_summary') $version ('query-'+$stage+'.json') @() 'aggregate_JSON_TRAIN_key_only'
  $updates=if($stage-eq'initial'){'0'}else{'2048'}
  Add-Input ($version+'_'+$stage+'_checkpoint') $version ('checkpoints/inference-'+$updates+'.pt') $typed
  foreach($rep in @(2701,2802,2903)){
   $head=$quality+'market-readouts/models/cpb_'+$version+'/rep-'+$rep+'/'
   Add-Input ($version+'_'+$stage+'_head_'+$rep) $version ($head+'predictions_train.pt') $headFields
   Add-Input ($version+'_'+$stage+'_fit_'+$rep) $version ($head+'fit.pt') $fitFields
  }
 }
 foreach($update in @(1,2048)){Add-Input ($version+'_actual_update_'+$update) $version ('witnesses/training-update-'+$update+'-components.pt') (@($actualFields)+@($clipFields[$version]))}
 Add-Input ($version+'_actual_bindings') $version 'optimization-component-witnesses.json' @() 'actual_batch_binding_metadata'
 Add-Input ($version+'_training_csv') $version 'training.csv' @() 'TRAIN_CSV_2048_scalar_rows_only'
 foreach($name in @('training-512.pt','training-1024.pt','training-2048.pt','input-scaler.pt','bank-standardizer.pt')){
  $relative=if($name.StartsWith('training-')){'checkpoints/'+$name}else{'calibration/'+$name}
  $pin=Mirrored $version $relative;Add-Preserve ('preserve-'+$version+'-'+$name.Replace('.','-')) $pin.path $pin.container_path
 }
 foreach($ext in @('json','md')){Add-Preserve ('preserve-'+$version+'-card-'+$ext) ('doc/CAUSAL_FRESH_'+$version.ToUpperInvariant()+'_CARD.'+$ext) ('/opt/cuwacunu/research/causal-fresh-'+$version+'/attempt-001/source/doc/CAUSAL_FRESH_'+$version.ToUpperInvariant()+'_CARD.'+$ext)}
 Add-Preserve ('preserve-'+$version+'-establishment') ('doc/causal_fresh_'+$version+'/final-establishment.json')
 $mirrorLabel=if($version-eq'v6'){'doc/causal_fresh_v6/complete-mirror-proof-recovery-002-attempt-001.json'}else{'doc/causal_fresh_v7/complete-mirror-attempt-001.json'}
 Add-Preserve ('preserve-'+$version+'-complete-mirror') $mirrorLabel
 Add-Preserve ('preserve-'+$version+'-training-audit') ('doc/causal_fresh_'+$version+'/independent-training-audit.json')
 Add-Preserve ('preserve-'+$version+'-quality-audit') ('doc/causal_fresh_'+$version+'/quality-independent-audit.json')
 Add-Preserve ('preserve-'+$version+'-source-manifest') ('output/engineering/causal-fresh-'+$version+'/attempt-001/source-manifest.json') ('/opt/cuwacunu/research/causal-fresh-'+$version+'/attempt-001/source/CAPSULE_MANIFEST.json')
 Add-Preserve ('preserve-'+$version+'-source-ZIP') ('output/engineering/causal-fresh-'+$version+'/attempt-001/source.zip') ('/opt/cuwacunu/research/causal-fresh-'+$version+'/attempt-001/source.zip')
}
foreach($repo in @('cuwacunu_dev','cuwacunu_embedding')){Add-Preserve ('preserve-selected-v3-'+$repo) 'output/baselines/CPB-v3.market-01/inference.pt' '/cuwacunu/output/baselines/CPB-v3.market-01/inference.pt' $repo}
Add-Preserve 'preserve-current-clean-ledger' $ledgerPath
Add-Preserve 'preserve-current-catalog' $catalogPath
Add-Preserve 'preserve-future-v7-retention' $GoalProtocolPath
Require ($script:allowed.Count-eq49-and@($script:allowed.id|Sort-Object -Unique).Count-eq49-and$script:preservation.Count-eq31) 'Exactly49 numeric roles and31 preservation-only roles'
$owned=@(Get-ChildItem -LiteralPath (Rooted 'code/research/causal_v6_v7_train_diagnosis') -File|Sort-Object Name|ForEach-Object {Pin ('code/research/causal_v6_v7_train_diagnosis/'+$_.Name)})
Require ($owned.Count-eq5) 'Complete five owned source files'
$blocks=@([ordered]@{name='learned20';begin=0;end=20},[ordered]@{name='fixed_spectral12';begin=20;end=32},[ordered]@{name='mean4';begin=32;end=36},[ordered]@{name='uncertainty4';begin=36;end=40})
$policies=[ordered]@{v6='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1';v7='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;disjoint-mean-uncertainty-clip1-v1'}
$card=[ordered]@{
 copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.';kind='cpb_v6_v7_saved_train_diagnosis_card_v1';declared_utc=[DateTime]::UtcNow.ToString('o');stage='prospective_before_diagnostic_arithmetic'
 baseline='CPB-v3.market-01';comparison_references=@('CPB-v6.market-01','CPB-v7.market-01');versions=@('v6','v7');candidate_declared=$false;diagnosis_executed=$false;baseline_or_application_change=$false
 parent_card_pattern=$parentCard;goal_protocol=$goalPin;planning_handoff=(Pin 'doc/CAUSAL_V6_V7_TRAIN_DIAGNOSIS_PLAN.md');final_establishment_context=$establishmentPin;current_clean_ledger=(Pin $ledgerPath)
 dataset_id='MKT-HIST-2';designed_complexity_level=4;complexity_scale_max=5;recipe_id='causal-v6-v7-saved-train-diagnosis-v1';view='TRAIN112 saved features/heads/components;first8 fixed-query raw and adjoints;both actual1/2048;both2048 observed clipping CSV traces'
 train_rows=112;query_rows=8;stages=@('initial0','final2048');train_begin_ms=1704067259999;train_end_ms=1705881599999;canonical_channel_ids=@(0,1,2);head_repetitions=@(2701,2802,2903)
 native_widths=[ordered]@{v6=40;v7=40};native_blocks=[ordered]@{v6=$blocks;v7=$blocks};head_seeds_unsigned_decimal=[ordered]@{v6=@('6049515622977958931','8262941649267994858','2095371932330426401');v7=@('6049515622977958931','8262941649267994858','2095371932330426401')}
 cross_basis_cosine_policy='forbidden-between-independent-mean4-and-uncertainty4';gradient_objectives=@('Gaussian_NLL','MAE','Huber_delta1');Huber_delta=1.0;MAE_zero_subgradient=0.0
 temporal_bins=@(@(0,32),@(32,64),@(64,96),@(96,128));chronological_TRAIN_halves=@([ordered]@{name='early56';begin=0;end=56},[ordered]@{name='late56';begin=56;end=112});quantiles=@(0.0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1.0)
 allowed_inputs=$script:allowed;preservation_pins=$script:preservation;preservation_pin_count=31
 pre_execution_preservation=[ordered]@{id='final-v7-preservation';host_repository='cuwacunu_embedding';host_path=$broadPath;container_path=($futureSource+'/final-preservation.json');bytes=$broadPin.bytes;sha256=$broadPin.sha256}
 actual_batches=@(1,2048);actual_batch_decoder_adjoints='forbidden-no-matched-prestep-decoder-weights';full112_component_rows='saved-both-versions';actual_batch_tensors='saved-both-versions-no-replay';V6_split_norm_inference='forbidden'
 numeric_policy='Verify49 inputs,31 preservation-only pins and final V7 broad receipt before tensor storage;recheck source/card/allpins afterward. Inert stdlib only;no models/fits/heldoutnumeric.'
 mixed_summary_policy='Four JSON files contain both splits and are decoded;only train numeric subtree used. Head archives may contain other metadata;only declared fields read.'
 objective_policy='huber1-mean-plus-detached-error-gaussian-variance-v1';gradient_policies=$policies;mean_huber_delta=1.0;loss_coefficients=@(1.0,1.0);variance_error_detached=$true;reporting_nll_backpropagated=$false
 clipping_policy=[ordered]@{v6='shared-global-norm2-clip1';v7='disjoint-mean-uncertainty-norm2-clip1-v1';v7_caps=@(1.0,1.0);v7_partition=@(73,293537,3,1215,1,3888);combined_norm_confound='V7 bound sqrt2 versus V6 global1';optimizer_independence_claimed=$false}
 diagnostics=@('Observed2048 CSV objectives/norms/factors,overall/four512 blocks;firstlast actual component/support/CSV checks without decoder pairing','All112 native block variance/participation and existing ridge four-block contributions;all3 saved neural predictions,never forward/refit','First8 initial/final fixed queries:four32 rawF64 observed/reconstruction bins and matching local adjoints only')
 derivative_limits='Both applied mean adjoints are Huber1;meanNLL/MAE hypothetical. No cross-basis/model/stage cosine,fullJacobian,actualbatchdecoderadjoints or V6branch dominance.'
 raw_temporal_scope='Original saved rawF64 first8 only;observed means/std/within-bin adjacent differences,no task labels or fitted summary. Bin hierarchy has its own eligible support;not additive full loss.'
 tolerances=[ordered]@{saved_scalar_prediction_and_NLL_arithmetic='absolute1e-10+relative1e-9';saved_variance_link_consistency='absolute5e-6+relative5e-5 for saved F32 intermediate approximation';mask_row_support_and_paired_initial_data='exact'}
 owned_source_files=$owned;card_generator=(Pin 'doc/causal_v6_v7_train_diagnosis/prepare-card.ps1');metadata_copies_required_before_execution=$script:metadataCopies
 admission='Card generation is not execution admission. Root+independentpreflight must freeze five ownedsources/card/metadata copies/finalpreservation and authorize one exclusive existing-container pass;preserve every refusal.'
 model_calls=0;head_fits=0;training_updates=0;new_quality_scores=0;raw_cache_numeric_reads=0
 forbidden=@('Encoder/decoder/neural/head execution','New training/head/scaler/normalizer/statistic fits','V6split inference or V7counterfactualglobalclip','Actualbatch decoder pairing','Heldoutnumericfields/rawcache/PCA','New candidate/rescue/floors/TRAINNLLgate','Dev/publication/Git mutations')
 interpretation='Known TRAIN evidence only;no forecasting,heldout improvement,causal explanation or seed robustness established. Preserve every V3-V7 result/strength/history and selected V3.'
}
$json=($card|ConvertTo-Json -Depth 70).Replace([string][char]13,'')+"`n"
$stream=[IO.File]::Open((Rooted $cardPath),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes($json);$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
$cardPin=Pin $cardPath
$markdown=@"
<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Prospective saved TRAIN V6/V7 diagnosis

The machine authority is [the frozen card](CAUSAL_V6_V7_TRAIN_DIAGNOSIS_CARD.json),
SHA256 $($cardPin.sha256). Read [the scope and limits](CAUSAL_V6_V7_TRAIN_DIAGNOSIS_PLAN.md).
This is a saved TRAIN diagnosis declaration; no arithmetic/model/fit has run.
V3 remains the clean baseline and sole dev application encoder.

Dataset: **MKT-HIST-2**, past-window retention, **complexity4/5**. TRAIN112,
first8 detailed queries, both initial0/final2048 and actual pre-step1/2048.
The49 exact inputs and31 preservation-only pins plus final V7 broad preservation
must verify before numeric access and afterward. Source/card preflight and
explicit root admission are still required after this declaration.

Both versions use native40 and Huber1 mean plus detached-error variance NLL.
V6 global clipping and V7 disjoint clipping are observed separately; V7 joint
norm can reach sqrt2. No unavailable V6 split is inferred. The three panels are
existing2048-step scalar traces, saved112 native/readout contributions, and four
fixed32-step first8 raw/reconstruction bins with stage-matched local adjoints.
Both full112 component rows are saved; four mixed-split JSONs use train only.
No actual-batch decoder pairing, fits, neural/encoder/decoder forwards, training,
held-out tensors, raw caches, PCA, candidate or stopped-recipe rescue is admitted.
Future strongest V3-V7 floors are preservation context, never new diagnostic gates.
"@
$stream=[IO.File]::Open((Rooted $markdownPath),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes($markdown.Replace([string][char]13,'')+"`n");$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
[ordered]@{status='prospective_cards_declared_pending_independent_preflight';numeric_inputs=49;preservation_pins=31;metadata_copies=$script:metadataCopies.Count;card=$cardPin;markdown=(Pin $markdownPath);numeric_execution=$false}|ConvertTo-Json -Depth 5

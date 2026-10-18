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
$card=Read 'doc/CAUSAL_FRESH_V9_CARD.json'
Check ((Pin 'doc/CAUSAL_FRESH_V9_CARD.json').sha256-eq$CardSHA256) 'Exact prospective card'
Check (-not$card.stage.Contains('DRAFT')-and$null-ne$card.frozen_utc-and$card.execution_admitted) 'Frozen and explicitly admitted before any build/test/learning/quality'
Check ($card.model-eq'CPB-v9'-and$card.instance-eq'CPB-v9.market-01'-and$card.geometry.native_width-eq40) 'Named native40 candidate'
Check ($card.mechanism.kind-eq'original-position-four-bin-shared-global-half-mixture-v1'-and($card.mechanism.temporal_pooling_partition -join ',')-eq'4,16,32,4,8'-and($card.mechanism.temporal_pooling_mix -join ',')-eq'0.5,0.5'-and$card.mechanism.empty_local_bin_policy-eq'zero-whole-export-slice-after-bias-v1') 'Four endpoint half-local/half-global pools, empty slice zero after bias'
Check ($card.geometry.temporal_pooling_policy-eq$card.mechanism.temporal_pooling_policy-and($card.geometry.temporal_pooling_partition -join ',')-eq'4,16,32,4,8'-and($card.geometry.temporal_pooling_mix -join ',')-eq'0.5,0.5'-and$card.geometry.empty_local_bin_policy-eq$card.mechanism.empty_local_bin_policy) 'Geometry and mechanism temporal declarations exactly agree'
Check ($card.mechanism.gradient_policy-eq'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'-and$card.training.global_gradient_clip-eq1) 'Unchanged V6 global clip1'
Check ($card.training.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and($card.training.loss_coefficients -join ',')-eq'1,1'-and$card.training.variance_error_detached-and-not$card.training.reporting_nll_backpropagated) 'Unchanged V6 objective'
Check ($card.training.updates-eq2048-and$card.training.batch_size-eq8-and$card.training.learning_rate-eq0.001-and$card.training.weight_decay-eq0.0001-and$card.training.initialization_seed-eq943001-and$card.training.row_seed-eq943002-and$card.training.mask_seed-eq943003-and$card.training.context_seed-eq943004) 'One unchanged fresh trajectory'
Check ($card.mechanism.all_learned_values-eq294752-and$card.mechanism.all_learned_tensors-eq76-and$card.mechanism.mandatory_identifiable_learned_tensors-eq75-and$card.mechanism.data_gradient_gauge_exemption-eq'core.encoder.level.base.backbone.pool_score.bias'-and-not$card.mechanism.checkpoint_import_for_initialization) 'Same fresh capacity, explicit mathematical gauge'
foreach($p in @($card.input_pins)+@($card.historical_preservation_pins)){Verify $p $p.repository}
Verify $card.input_manifest
foreach($p in @($card.saved_TRAIN_diagnosis.card,$card.saved_TRAIN_diagnosis.result,$card.saved_TRAIN_diagnosis.independent_audit,$card.saved_TRAIN_diagnosis.human_report,$card.saved_TRAIN_diagnosis.completion,$card.pre_execution_preservation,$card.baseline_preservation,$card.comparison_control.v3,$card.comparison_control.v4,$card.comparison_control.v5,$card.comparison_control.v6,$card.comparison_control.v7,$card.comparison_control.v8,$card.comparison_control.v6_establishment,$card.comparison_control.v7_establishment,$card.comparison_control.v8_establishment,$card.advancement.goal_protocol,$card.retention_independent_review)){Verify $p}
$audit=Read $card.saved_TRAIN_diagnosis.independent_audit.path
Check ($audit.status-eq'passed'-and$audit.checks-eq642801-and$audit.mismatches.Count-eq0) 'Independent TRAIN diagnosis passed'
$protocol=Read $card.advancement.goal_protocol.path
Check ($card.advancement.goal_protocol.sha256-eq'e7920442fbc5528cbe7e67f71ada65101974a6a28ad4d39bc57dea461c477699'-and$card.advancement.criterion_count-eq22) 'Protect all V3–V8 scoped strengths'
Check (($card.advancement.criteria|ConvertTo-Json -Depth 30 -Compress)-ceq($protocol.criteria|ConvertTo-Json -Depth 30 -Compress)) 'Exact prospective conditions'
$review=Read $card.retention_independent_review.path
Check ($review.status-eq'passed') 'Independent retention review passed'
Verify $card.retention_application_admission
Check ($card.retention_application_admission.sha256-eq'8c2ebe5f240e43762bcf079bc61b09a0fe6c2ae5dd64aee2ad365ce418f0d622') 'Exact applied future floor receipt'
$admission=Read 'doc/causal_fresh_v9/goal-protocol-admission.json'
Check ($admission.status-eq'passed'-and$admission.prospective_future_use_eligible-and$admission.criteria-eq22-and-not$admission.historical_results_or_counts_recomputed) 'Applied future floor admission'
Check ($card.baseline_frozen_files.Count-eq47) '47 selected baseline files per repository'
foreach($p in $card.baseline_frozen_files){foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){Verify $p $repo}}
$preservation=Read $card.baseline_preservation.path
Check ($preservation.status-eq'passed'-and$preservation.critical_checkpoint_pins.Count-eq94) '94 protected checkpoint files;no uncreated V9 artifacts counted'
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
$old=Read 'doc/causal_fresh_v8/protocol-preflight.json'
foreach($p in $old.sources){Verify $p}
$family=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v9') -Recurse -File)
$research=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v9') -Recurse -File)
Check ($null-ne$card.source_counts.family_inside-and$null-ne$card.source_counts.research-and$family.Count-eq$card.source_counts.family_inside-and$research.Count-eq$card.source_counts.research) 'Actual final owner family/research counts required'
foreach($f in @($family)+@($research)){
 Check (-not($f.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'Physical source'
 if($f.Extension-in@('.h','.cpp','.mk','.py')-or$f.Name-eq'Makefile'){
  $s=Get-Content -LiteralPath $f.FullName -Raw
  Check ($s-notmatch'(?m)^\s*#\s*include[^\r\n]*(?:versions/v[1-8]/|detail/v[1-8]/|raw_patch_bottleneck_mae)') ('No historical implementation includes: '+$f.Name)
 }
}
$pool=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v9/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v9/temporal_pooling.h') -Raw
Check ($pool.Contains('pool_original_position_bins')-and$pool.Contains('finite_fallback')-and$pool.Contains('projection_weight.narrow')-and$pool.Contains('torch::where(present.unsqueeze(1),projected')-and$pool.Contains('global_weights')-and$pool.Contains('kTemporalPoolingLocalMix=.5')-and$pool.Contains('kTemporalPoolingGlobalMix=.5')) 'Owned masked safe fixed half-mixture pooling kernel'
$train=Get-Content -LiteralPath (Join-Path $root 'code/research/causal_fresh_v9/train_main.cpp') -Raw
Check ($train.Contains('clip_grad_norm_(')-and$train.Contains('total_loss.backward()')-and$train.Contains('typed-resume-512.json')-and$train.Contains('pool_score.bias')) 'Actual global clip, typed continuation, explicit gauge'
Check ($train.Contains('temporal-between-bin-position-learning.csv')-and$train.Contains('hadamard-full16-signs-div4-v1')-and$train.Contains('mandatory_identifiable_tensor_value_capacity')-and$train.Contains('per_scalar_nonzero_gradient_proof_claimed')) 'Observed between-bin evidence and honest tensor capacity labels'
$docs=@('LICENSE','doc/CAUSAL_FRESH_V9_CARD.json','doc/CAUSAL_FRESH_V9_CARD.md','code/encoders/causal_patch_bottleneck_mae/versions/CPB-v9.json',
 'doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.json','doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS.md','doc/causal_v6_v8_train_diagnosis/diagnosis.json','doc/causal_v6_v8_train_diagnosis/independent-audit.json','doc/causal_v6_v8_train_diagnosis/completion.json','doc/causal_fresh_v8/final-preservation.json')
$controls=@('preflight.ps1','stage-source.ps1','unpack-source.py','audit-build.py','execute-stage.py','run-quality.py','verify-mirror.ps1','run-container.ps1','input-manifest.json','raw-controls.json','draft-metadata-copy-receipt.json','preservation-before.json','orchestration-provenance.json','baseline-control-v3.json','baseline-control-v4.json','baseline-control-v5.json','baseline-control-v6.json','baseline-control-v7.json','baseline-control-v8.json','baseline-v6-establishment.json','baseline-v7-establishment.json','baseline-v8-establishment.json','goal-protocol.json','goal-protocol-review.json','goal-protocol-admission.json')
$all=@($family)+@($research)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v9.h'))+@($docs|ForEach-Object{Get-Item -LiteralPath (Join-Path $root $_)})+@($controls|ForEach-Object{Get-Item -LiteralPath (Join-Path $PSScriptRoot $_)})
$sources=@($all|Sort-Object FullName -Unique|ForEach-Object{Pin ($_.FullName.Substring($root.Length+1).Replace([char]92,[char]47))})
Check ($sources.Count-eq$all.Count-and($sources.Count+1)-eq$card.source_counts.capsule_sources) 'Unique actual reviewed paths plus generated preflight equal frozen capsule count'
$receipt=[ordered]@{kind='causal_fresh_v9_source_protocol_preflight_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');card_sha256=$CardSHA256;checks=$script:checks;family_files=$family.Count;extra_full_owned_public_header=1;owned_research_files=$research.Count;critical_checkpoints_verified=94;baseline_sources_verified_each_repo=47;old_v8_sources_verified=$old.sources.Count;sources=$sources;scope='Host file/hash/source/protocol inspection only;no build,test,model,numerical execution';goal_active=$true;application_changed=$false}
$receipt|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $destination -Encoding utf8NoBOM
[ordered]@{status='passed';checks=$script:checks;sources=$sources.Count;receipt=(Pin 'doc/causal_fresh_v9/protocol-preflight.json')}|ConvertTo-Json -Depth 4

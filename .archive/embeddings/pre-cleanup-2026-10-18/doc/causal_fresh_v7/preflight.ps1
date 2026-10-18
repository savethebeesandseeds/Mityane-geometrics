# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# File inspection only. All project execution belongs in managed cuwacunu.
param([Parameter(Mandatory=$true)][string]$CardSHA256)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspace=[IO.Path]::GetFullPath((Join-Path $root '..'))
$destination=Join-Path $PSScriptRoot 'protocol-preflight.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve existing preflight receipt'}
$script:checks=0
function Check([bool]$ok,[string]$why){$script:checks++;if(-not$ok){throw $why}}
function Pin([string]$relative){$f=Get-Item -LiteralPath (Join-Path $root $relative);Check (-not($f.Attributes-band[IO.FileAttributes]::ReparsePoint)) ('No source links: '+$relative);[ordered]@{path=$relative;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Verify($p,[string]$repo='cuwacunu_embedding'){
 $path=[IO.Path]::GetFullPath((Join-Path (Join-Path $workspace $repo) $p.path))
 $bound=[IO.Path]::GetFullPath((Join-Path $workspace $repo))+[IO.Path]::DirectorySeparatorChar
 Check ($path.StartsWith($bound,[StringComparison]::OrdinalIgnoreCase)) 'Pin stays inside repository'
 $f=Get-Item -LiteralPath $path
 Check (-not($f.Attributes-band[IO.FileAttributes]::ReparsePoint)) ('No input links: '+$path)
 Check ($f.Length-eq$p.bytes-and(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()-eq$p.sha256) ('Exact frozen input: '+$path)
}
function Read([string]$p){Get-Content -LiteralPath (Join-Path $root $p) -Raw|ConvertFrom-Json}
function Same($a,$b){($a|ConvertTo-Json -Depth 90 -Compress)-ceq($b|ConvertTo-Json -Depth 90 -Compress)}
$cardPath='doc/CAUSAL_FRESH_V7_CARD.json';$card=Read $cardPath
Check ((Pin $cardPath).sha256-eq$CardSHA256) 'Exact prospectively frozen card'
Check (-not$card.stage.Contains('DRAFT')-and$null-ne$card.frozen_utc) 'Draft cannot enter execution admission'
Check ($card.model-eq'CPB-v7'-and$card.instance-eq'CPB-v7.market-01'-and$card.geometry.native_width-eq40) 'Named native40 candidate'
Check ($card.mechanism.gradient_policy-eq'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;disjoint-mean-uncertainty-clip1-v1') 'Disjoint branch gradient policy'
Check ($card.training.objective_policy-eq'huber1-mean-plus-detached-error-gaussian-variance-v1'-and$card.training.mean_huber_delta-eq1-and$card.training.loss_coefficients.Count-eq2-and$card.training.loss_coefficients[0]-eq1-and$card.training.loss_coefficients[1]-eq1-and$card.training.variance_error_detached-and-not$card.training.reporting_nll_backpropagated) 'Unchanged V6 objective'
Check ($card.mechanism.kind-eq'disjoint-mean-uncertainty-clipping-v1'-and$card.training.clip_policy-eq'disjoint-mean-uncertainty-norm2-clip1-v1'-and$card.training.gradient_norm_type-eq2-and$card.training.branch_gradient_clips.mean-eq1-and$card.training.branch_gradient_clips.uncertainty-eq1) 'Independent norm2 clip1 groups declared prospectively'
Check ($card.training.clipping_group_contract.mean.learned_tensors-eq73-and$card.training.clipping_group_contract.mean.learned_values-eq293537-and$card.training.clipping_group_contract.uncertainty.learned_tensors-eq3-and$card.training.clipping_group_contract.uncertainty.learned_values-eq1215-and$card.training.clipping_group_contract.fixed_excluded.values-eq3888) 'Declared exhaustive partition and fixed exclusion'
Check ($card.training.updates-eq2048-and$card.training.batch_size-eq8-and$card.training.learning_rate-eq0.001-and$card.training.weight_decay-eq0.0001-and$card.training.initialization_seed-eq943001-and$card.training.row_seed-eq943002-and$card.training.mask_seed-eq943003-and$card.training.context_seed-eq943004) 'Unchanged one-trajectory training recipe'
Check ($card.mechanism.all_learned_values-eq294752-and$card.mechanism.all_learned_tensors-eq76-and$card.mechanism.fixed_literal_prior_values-eq3888-and$card.mechanism.total_parameters-eq298640-and-not$card.mechanism.checkpoint_import_for_initialization) 'Fresh unchanged capacity, no inherited trained state'
foreach($p in @($card.input_pins)+@($card.historical_preservation_pins)){Verify $p $p.repository}
Verify $card.input_manifest
foreach($p in @($card.saved_TRAIN_diagnosis.card,$card.saved_TRAIN_diagnosis.result,$card.saved_TRAIN_diagnosis.independent_audit,$card.saved_TRAIN_diagnosis.human_report,$card.baseline_preservation,$card.comparison_control.v3,$card.comparison_control.v4,$card.comparison_control.v5,$card.comparison_control.v6,$card.comparison_control.v6_establishment,$card.advancement.goal_protocol,$card.retention_independent_review)){Verify $p}
$diagnosisAudit=Read $card.saved_TRAIN_diagnosis.independent_audit.path
Check ($diagnosisAudit.status-eq'passed'-and$diagnosisAudit.checks-eq415326-and$diagnosisAudit.finite_difference_checks-eq256-and$diagnosisAudit.mismatches.Count-eq0) 'Independent TRAIN diagnosis admitted'
$protocol=Read $card.advancement.goal_protocol.path
Check ($card.advancement.goal_protocol.sha256-eq'0965842331056b61faee827a5ea782703205b23bd326298929ed3ecfca4604fb'-and$card.advancement.criterion_count-eq22) 'Strongest V3/V4/V5/V6 future floors'
Check (Same $card.advancement.criteria $protocol.criteria) 'Every criterion ID/rule/floor/source identical'
Check ((Pin 'doc/causal_fresh_v7/goal-protocol.json').sha256-eq$card.advancement.goal_protocol.sha256) 'Exact protocol copy'
$retention=Read $card.retention_independent_review.path
Check ($retention.status-eq'passed'-and$retention.checks-eq162-and$retention.protocol.sha256-eq$card.advancement.goal_protocol.sha256) 'Independent floor derivation verified'
Check ($card.baseline_frozen_files.Count-eq47) '47 selected baseline sources per repository'
foreach($p in $card.baseline_frozen_files){foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){Verify $p $repo}}
$preservation=Read 'doc/causal_fresh_v6/final-preservation.json'
Check ($preservation.status-eq'passed'-and$preservation.critical_checkpoint_pins.Count-eq84) '84 protected checkpoint pins'
foreach($p in $preservation.critical_checkpoint_pins){
 $repo=$p.repository
 if(-not$repo){Check ($p.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')-and$p.container_path.StartsWith('/opt/cuwacunu/research/causal-fresh-v5/attempt-003/run-001/checkpoints/')) 'The five repository-less V5 pins belong to the research repository';$repo='cuwacunu_embedding'}
 Verify $p $repo
}
foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){
 $paths=@($preservation.critical_checkpoint_pins|Where-Object{($_.repository-eq$repo)-or(-not$_.repository-and$repo-eq'cuwacunu_embedding')}|Select-Object -ExpandProperty path)
 $ignored=@(& git -c ('safe.directory='+($workspace+'/'+$repo).Replace([char]92,[char]47)) -C (Join-Path $workspace $repo) check-ignore -- @paths)
 Check ($LASTEXITCODE-eq0-and$ignored.Count-eq$paths.Count) ('Critical checkpoints remain ignored: '+$repo)
}
$v6Preflight=Read 'doc/causal_fresh_v6/protocol-preflight.json'
foreach($p in $v6Preflight.sources){Verify $p}
$family=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v7') -Recurse -File)
$research=@(Get-ChildItem -LiteralPath (Join-Path $root 'code/research/causal_fresh_v7') -Recurse -File)
Check ($family.Count-eq49-and$research.Count-eq30) 'Complete owned family/research copies'
foreach($file in @($family)+@($research)){
 Check (-not($file.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'Owned physical source'
 if($file.Extension-in@('.h','.cpp','.mk','.py')-or$file.Name-eq'Makefile'){
  $text=Get-Content -LiteralPath $file.FullName -Raw
  Check ($text-notmatch'(?m)^\s*#\s*include[^\r\n]*(?:versions/v[1-6]/|detail/v[1-6]/|raw_patch_bottleneck_mae)') ('No historical encoder includes: '+$file.Name)
 }
}
$familyHeader=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v7.h') -Raw
Check ($familyHeader.Length-gt30000-and$familyHeader.Contains('inline ClipResult clip_gradients(')-and$familyHeader.Contains('kClipPolicy=implementation::kLocationScaleDecoderClipPolicy')) 'Complete owned public implementation and clipping API'
$clipSource=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v7/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v7/location_scale_decoder_clipping.h') -Raw
$clipConfig=Get-Content -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v7/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v7/location_scale_decoder_config.h') -Raw
Check ($clipConfig.Contains('disjoint-mean-uncertainty-norm2-clip1-v1')-and$clipSource.Contains('core.uncertainty_projection')-and$clipSource.Contains('core.variance_projection')-and$clipSource.Contains('core.variance_bias')) 'Owned clipping implementation and exact uncertainty membership'
$train=Get-Content -LiteralPath (Join-Path $root 'code/research/causal_fresh_v7/train_main.cpp') -Raw
Check ($train.Contains('clip_gradients(')-and$train.Contains('total_loss.backward()')-and-not$train.Contains('clip_grad_norm_(')) 'Main trajectory applies declared objective and portable group clipping'
Check ($train.Contains('512')-and$train.Contains('typed-resume-512.json')) 'Actual typed512 continuation retained'
$docs=@('LICENSE','doc/CAUSAL_FRESH_V7_CARD.json','doc/CAUSAL_FRESH_V7_CARD.md',
 'doc/causal_fresh_v7/stage-source.ps1','doc/causal_fresh_v7/unpack-source.py','doc/causal_fresh_v7/audit-build.py','doc/causal_fresh_v7/run-quality.py','doc/causal_fresh_v7/execute-stage.py','doc/causal_fresh_v7/verify-mirror.ps1','doc/causal_fresh_v7/preflight.ps1',
 'doc/causal_fresh_v7/input-manifest.json','doc/causal_fresh_v7/raw-controls.json','doc/causal_fresh_v7/baseline-control-v3.json','doc/causal_fresh_v7/baseline-control-v4.json','doc/causal_fresh_v7/baseline-control-v5.json','doc/causal_fresh_v7/baseline-control-v6.json','doc/causal_fresh_v7/baseline-v6-establishment.json','doc/causal_fresh_v7/goal-protocol.json',
 'doc/CAUSAL_V5_V6_TRAIN_DIAGNOSIS_CARD.json','doc/causal_v5_v6_train_diagnosis/diagnosis.json','doc/causal_v5_v6_train_diagnosis/independent-audit.json','doc/causal_fresh_v6/final-preservation.json','doc/CAUSAL_V5_V6_TRAIN_DIAGNOSIS.md','doc/causal_v6_retention_review/independent-review.json','code/encoders/causal_patch_bottleneck_mae/versions/CPB-v7.json')
$all=@($family)+@($research)+@(Get-Item -LiteralPath (Join-Path $root 'code/encoders/causal_patch_bottleneck_mae/versions/v7.h'))+@($docs|ForEach-Object{Get-Item -LiteralPath (Join-Path $root $_)})
$sources=@($all|Sort-Object FullName -Unique|ForEach-Object{Pin ($_.FullName.Substring($root.Length+1).Replace([char]92,[char]47))})
$receipt=[ordered]@{kind='causal_fresh_v7_source_protocol_preflight_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');card_sha256=$CardSHA256;checks=$script:checks;family_files=$family.Count;owned_research_files=$research.Count;extra_full_owned_public_header=1;critical_checkpoints_verified=84;baseline_sources_verified_each_repo=47;old_v6_sources_verified=$v6Preflight.sources.Count;sources=$sources;scope='File/hash/prospective protocol and source-route inspection only; no model,numerical diagnosis,build,training or quality on host';baseline_changed=$false;application_changed=$false;goal_active=$true}
$receipt|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $destination -Encoding utf8NoBOM
[ordered]@{status='passed';checks=$script:checks;sources=$sources.Count;receipt=(Pin 'doc/causal_fresh_v7/protocol-preflight.json')}|ConvertTo-Json -Depth 4

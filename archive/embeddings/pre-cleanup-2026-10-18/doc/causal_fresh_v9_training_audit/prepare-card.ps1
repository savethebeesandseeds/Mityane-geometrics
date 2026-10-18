# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Owned metadata-only adaptation of the physically copied V9 quality-card generator.
# No Docker, project imports, AST, auditor body, model or numerical data algorithm.
param([Parameter(Mandatory=$true)][switch]$SourcesReviewed,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$TrainingMirrorSHA256,
 [Parameter(Mandatory=$true)][string]$SourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SourceReviewSHA256)
$ErrorActionPreference='Stop'
if(-not $SourcesReviewed){throw 'Independent static review must precede audit-card generation'}
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$output=Join-Path $PSScriptRoot 'audit-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve the existing audit declaration; use an admitted additive attempt'}
function Rooted([string]$relative){
 if([IO.Path]::IsPathRooted($relative)){throw 'Repository-relative metadata paths only'}
 $path=[IO.Path]::GetFullPath((Join-Path $repo $relative));$prefix=$repo.TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
 if(-not $path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Metadata path outside repository'}
 for($walk=$path;$walk-ne$repo;$walk=[IO.Path]::GetDirectoryName($walk)){$item=Get-Item -LiteralPath $walk;if(($item.Attributes-band[IO.FileAttributes]::ReparsePoint)-or-not$walk){throw 'Physical metadata ancestors required'}}
 $path
}
function Pin([string]$relative){$f=Get-Item -LiteralPath (Rooted $relative);if($f.PSIsContainer){throw 'Expected file'};[ordered]@{path=$relative.Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Exact($a,$b){if($a.bytes-ne$b.bytes-or$a.sha256-ne$b.sha256){throw 'Reviewed metadata/source hash mismatch'}}
$reviewPin=Pin $SourceReviewPath
if($reviewPin.sha256-ne$SourceReviewSHA256){throw 'External source-review hash required before parse'}
$review=Get-Content -Raw -LiteralPath (Rooted $SourceReviewPath)|ConvertFrom-Json
if($review.status-ne'passed'-or$review.blockers.Count-ne0){throw 'Passed independent source review required'}
$auditor=Pin 'code/research/causal_fresh_v9_training_audit/audit_training.py'
$readme=Pin 'code/research/causal_fresh_v9_training_audit/README.md'
$provenance=Pin 'code/research/causal_fresh_v9_training_audit/provenance.json'
Exact $auditor $review.source;Exact $readme $review.readme;Exact $provenance $review.provenance
$mainPin=Pin 'doc/CAUSAL_FRESH_V9_CARD.json'
if($mainPin.sha256-ne$MainCardSHA256){throw 'External main-card hash required before parse'}
$main=Get-Content -Raw -LiteralPath (Rooted $mainPin.path)|ConvertFrom-Json
$md=Pin 'doc/CAUSAL_FRESH_V9_CARD.md';$input=Pin 'doc/causal_fresh_v9/input-manifest.json'
if($main.model-ne'CPB-v9'-or$main.instance-ne'CPB-v9.market-01'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Actually frozen and admitted main card required'}
Exact $input $main.input_manifest
$manifestPin=Pin 'output/engineering/causal-fresh-v9/attempt-001/source-manifest.json'
if($manifestPin.sha256-ne$MainManifestSHA256){throw 'External main-manifest hash required before parse'}
$manifest=Get-Content -Raw -LiteralPath (Rooted $manifestPin.path)|ConvertFrom-Json
$zip=Pin 'output/engineering/causal-fresh-v9/attempt-001/source.zip'
$package=Get-Content -Raw -LiteralPath (Rooted 'output/engineering/causal-fresh-v9/attempt-001/package.json')|ConvertFrom-Json
if($manifest.card_sha256-ne$MainCardSHA256-or$manifest.files.Count-le0-or$package.source_files-ne$manifest.files.Count-or$package.manifest_sha256-ne$MainManifestSHA256-or$package.zip_sha256-ne$zip.sha256){throw 'Actual immutable main source package required'}
$mirrorPin=Pin 'doc/causal_fresh_v9/training-mirror-attempt-001.json'
if($mirrorPin.sha256-ne$TrainingMirrorSHA256){throw 'External completed Training mirror hash required before parse'}
$mirror=Get-Content -Raw -LiteralPath (Rooted $mirrorPin.path)|ConvertFrom-Json
if($mirror.status-ne'passed'-or$mirror.scope-ne'Training'-or$mirror.attempt-ne'attempt-001'-or$mirror.card_sha256-ne$MainCardSHA256-or$mirror.source_manifest_sha256-ne$MainManifestSHA256-or-not$mirror.all_mirrored_pt_ignored){throw 'Completed separately mirrored main training evidence required; no quality dependency'}
foreach($relative in @('output/engineering/causal-fresh-v9/attempt-001/training-stage.json','output/runs/causal-fresh-v9/run-001/training-summary.json')){
 $p=Pin $relative;$matches=@($mirror.files|Where-Object {$_.path-eq$relative});if($matches.Count-ne1){throw 'Exact completed training metadata role missing'};Exact $p $matches[0]
}
$stagePin=Pin 'output/engineering/causal-fresh-v9/attempt-001/training-stage.json'
$stage=Get-Content -Raw -LiteralPath (Rooted $stagePin.path)|ConvertFrom-Json
$summaryPin=Pin 'output/runs/causal-fresh-v9/run-001/training-summary.json'
$summary=Get-Content -Raw -LiteralPath (Rooted $summaryPin.path)|ConvertFrom-Json
if($stage.status-ne'passed'-or$stage.exit_code-ne0-or$summary.status-ne'passed'-or$summary.model-ne'CPB-v9.market-01'-or$summary.new_causal_updates-ne2048-or$summary.attempted_batches-lt2048-or$summary.attempted_batches-gt4096){throw 'Exactly2048 successful main updates within4096 attempts required'}
$policy='original-position-four-bin-shared-global-half-mixture-v1'
if($main.mechanism.temporal_pooling_policy-ne$policy-or($main.mechanism.temporal_pooling_partition -join ',')-ne'4,16,32,4,8'-or($main.mechanism.temporal_pooling_mix -join ',')-ne'0.5,0.5'-or$main.mechanism.empty_local_bin_policy-ne'zero-whole-export-slice-after-bias-v1'){throw 'Exact prospective half-mixture contract required'}
$parents=@(foreach($name in @('audit_training.py','README.md','provenance.json')){$p=Pin ('code/research/causal_fresh_v8_training_audit/'+$name);[ordered]@{path=$p.path;container_path='/opt/cuwacunu/research/causal-fresh-v8-training-audit/attempt-001/source/'+$name;bytes=$p.bytes;sha256=$p.sha256}})
$prefix='doc/causal_fresh_v9_training_audit/'
$card=[ordered]@{
 kind='cpb_v9_saved_training_independent_audit_card_v1';copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 stage='declared_before_independent_saved_data_execution';declared_utc=[DateTime]::UtcNow.ToString('o');frozen_utc=[DateTime]::UtcNow.ToString('o');execution_pending=$true
 model='CPB-v9.market-01';baseline='CPB-v3.market-01';outside_frozen_main_source=$true
 objective_policy='huber1-mean-plus-detached-error-gaussian-variance-v1';gradient_policy='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
 temporal_pooling_policy=$policy;temporal_pooling_partition=@(4,16,32,4,8);temporal_pooling_mix=@(0.5,0.5);empty_local_bin_policy='zero-whole-export-slice-after-bias-v1'
 frozen_training_card_sha256=$mainPin.sha256;frozen_training_markdown_sha256=$md.sha256;frozen_input_manifest_sha256=$input.sha256
 main_source_manifest_sha256=$manifestPin.sha256;main_source_zip_sha256=$zip.sha256;main_source_files=$manifest.files.Count
 main_protocol_preflight=(Pin 'doc/causal_fresh_v9/protocol-preflight.json');main_input_bindings=@($mainPin,$md,$input,$manifestPin,$zip)
 auditor_source=$auditor;auditor_support=@($readme,$provenance);preserved_parent_sources=$parents;independent_source_review=$reviewPin
 completed_training_mirror=$mirrorPin;completed_training_stage=$stagePin;completed_training_summary=$summaryPin
 expected_main_paths=[ordered]@{attempt='/opt/cuwacunu/research/causal-fresh-v9/attempt-001';run='/opt/cuwacunu/research/causal-fresh-v9/attempt-001/run-001';source='/opt/cuwacunu/research/causal-fresh-v9/attempt-001/source';build='/opt/cuwacunu/build/causal-fresh-v9-attempt001'}
 expected_audit_root='/opt/cuwacunu/research/causal-fresh-v9-training-audit/attempt-001'
 execution=[ordered]@{container='cuwacunu';immutable_id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';image='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7';pure_stdlib_saved_data_only=$true;host_numeric_execution=$false;new_containers=0;stage_and_run_separately_admitted=$true}
 scope=@('Exactly2048 successful B8 updates with2048..4096 attempted batches; every CSV attempt/skip row, actual seed streams, typed512/1024/2048 progress and actual1/2048 witness associations','FreshTRAIN4096 scaler/bank arithmetic; all77 parameter tensors,76 registered learned294752 capacity,3888 literal prior; actual512 exact-state continuation without training replay','Unit F64 Huber1 mean and detached-error Gaussian variance plus reportingNLL; observed joint clip1 factor; known112/32 query kernel arithmetic, no held-out mechanism selection','All8192 row-group/2048 gauge/2048 between-bin CSV observations; same12 arrays only actual1/2048; F64 operands BEFORE movement subtraction','Unit fullR16 Hadamard signs++--/+-+-/+--+ repeatedfour anddiv4; Parseval/event counts, descriptive signed final coefficients and positive OVERALL final projection only','75 mandatory identifiable tensors with294751 numel capacity, not every-scalar proof; one retained scalar pool-score bias gauge; decay not data learning','Seven admitted binaries/eleven TUs and separate unchanged objective/temporal actualCUDA fixtures, no fixture/model/head replay')
 expected=[ordered]@{updates=2048;attempted_batch_range=@(2048,4096);batch_size=8;native_width=40;parameter_tensors=77;learned_tensors=76;learned_values=294752;mandatory_nonzero_data_learning_tensors=75;mandatory_identifiable_tensor_value_capacity=294751;fixed_values=3888;global_clip_max_norm=1;binary_count=7;translation_units=11;temporal_group_rows=8192;gauge_rows=2048;between_bin_rows=2048;actual_temporal_raw_array_witness_updates=@(1,2048);arrays_per_witness=12}
 temporal_numeric_contract=[ordered]@{movement='Promote EACH F32 operand to F64 BEFORE subtraction/centering/contraction';between_basis='hadamard-full16-signs-div4-v1';between_signs=@(@(1,1,-1,-1),@(1,-1,1,-1),@(1,-1,-1,1));full_position_divisor=4;norm_absolute_tolerance=2e-12;norm_relative_tolerance=2e-12;Parseval_squared_absolute=1e-12;Parseval_squared_relative=1e-10;final_gate='Per-direction data-gradient/movement EVENTS and positive OVERALL final projection; no individual final-nonzero gate';unsaved2046_raw_gradients_reconstructed=$false}
 fresh_schema_contract='Only fresh constructor JSON lacks gradient_policy; require absence there and strict typed/other selectors.'
 module_counting='CORE plus five distinct modules: numeric_statistics,saved_tensor_state,manual_saved_Gaussian,actual_pre_step_composite_objective,temporal_pooling_saved_learning; any total is explicitly their sum.'
 stage_helper=(Pin ($prefix+'stage.ps1'));run_helper=(Pin ($prefix+'run.ps1'));execution_wrapper=(Pin ($prefix+'execute.py'));mirror_helper=(Pin ($prefix+'mirror.ps1'));card_generator=(Pin ($prefix+'prepare-card.ps1'));workflow_provenance=(Pin ($prefix+'provenance.json'))
 model_calls=0;head_fits=0;training_updates=0;AST_calls=0
 forbidden=@('Torch/NumPy/producer imports or executable tensor module deserialization','Encoder/decoder/head/optimizer replay or refits','Old model/statistic numeric reads','Claiming every scalar data-learns or reconstructing unsaved2046 gradients','Quality scores/heads/selection or mutation of frozen main/old evidence/dev')
}
[IO.File]::WriteAllText($output,($card|ConvertTo-Json -Depth 35)+[char]10,[Text.UTF8Encoding]::new($false))
Pin ($prefix+'audit-card.json')|ConvertTo-Json

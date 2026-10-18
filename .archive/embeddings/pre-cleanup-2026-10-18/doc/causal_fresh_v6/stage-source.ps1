# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host orchestration/file inspection only. All project execution remains in cuwacunu.
param([Parameter(Mandatory=$true)][ValidateRange(1,999)][int]$Attempt,
 [Alias('Preflight')][ValidatePattern('^protocol-preflight[-a-z0-9]*\.json$')][string]$PreflightReceiptName='protocol-preflight.json')
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$integrationRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '../cuwacunu_dev'))
$attemptName='attempt-{0:D3}' -f $Attempt
$attemptRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v6/$attemptName"
if(Test-Path -LiteralPath $attemptRoot){throw 'Preserve existing source attempt'}
$cardHash='7cbda7aaa756bf775e09e3e8bc2a2e8cc7c3a34703171ad95cdd5e291bea847d'
$inputHash='bdb9892e9a817ea470e8743b16a5075ea4127ab96bfea53b4c2d76e7ba7f3b7d'
$policy='huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
function File-Pin([string]$p){$f=Get-Item -LiteralPath $p;return [ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$p,$pin){$a=File-Pin $p;if($a.bytes-ne$pin.bytes -or $a.sha256-ne$pin.sha256){throw "Changed frozen input: $p"}}
$cardPath=Join-Path $researchRoot 'doc/CAUSAL_FRESH_V6_CARD.json'
if((File-Pin $cardPath).sha256-ne$cardHash){throw 'Prospective card changed'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
if($card.model-ne'CPB-v6' -or $card.mechanism.gradient_policy-ne$policy -or $card.mechanism.kind-ne'huber1-mean-plus-detached-error-gaussian-variance-v1' -or $card.geometry.native_width-ne40 -or $card.advancement.criterion_count-ne22){throw 'Unexpected v6 declaration'}
if($card.training.objective_policy-ne'huber1-mean-plus-detached-error-gaussian-variance-v1' -or $card.training.mean_huber_delta-ne1 -or ($card.training.loss_coefficients -join ',')-ne'1,1' -or -not $card.training.variance_error_detached -or $card.training.reporting_nll_backpropagated){throw 'Unexpected V6 objective contract'}
foreach($pin in @($card.input_pins)+@($card.historical_preservation_pins)){
 $pinRoot=if($pin.repository-eq'cuwacunu_dev'){$integrationRoot}else{$researchRoot}
 Require-Pin (Join-Path $pinRoot $pin.path) $pin
}
Require-Pin (Join-Path $researchRoot $card.input_manifest.path) $card.input_manifest
if($card.input_manifest.sha256-ne$inputHash){throw 'Input manifest changed'}
if($card.baseline_frozen_files.Count-ne47){throw 'Expected47 protected v3 source files'}
foreach($pin in $card.baseline_frozen_files){foreach($r in @($researchRoot,$integrationRoot)){Require-Pin (Join-Path $r $pin.path) $pin}}
$bound=@($card.saved_TRAIN_diagnosis.card,$card.saved_TRAIN_diagnosis.result,$card.saved_TRAIN_diagnosis.independent_audit,$card.baseline_preservation,$card.comparison_control.v3,$card.comparison_control.v4,$card.comparison_control.v5,$card.advancement.goal_protocol,$card.retention_independent_review)
foreach($pin in $bound){Require-Pin (Join-Path $researchRoot $pin.path) $pin}
$audit=Get-Content -LiteralPath (Join-Path $researchRoot $card.saved_TRAIN_diagnosis.independent_audit.path) -Raw|ConvertFrom-Json
if($audit.status-ne'passed'){throw 'Independent saved TRAIN diagnosis audit required'}
$preflightPath=Join-Path $PSScriptRoot $PreflightReceiptName
$preflight=Get-Content -LiteralPath $preflightPath -Raw|ConvertFrom-Json
if($preflight.status-ne'passed' -or $preflight.card_sha256-ne$cardHash){throw 'Independent source protocol preflight required'}
foreach($pin in $preflight.sources){Require-Pin (Join-Path $researchRoot $pin.path) $pin}
$family=@(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/encoders/causal_patch_bottleneck_mae/versions/v6') -Recurse -File)
$research=@(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/research/causal_fresh_v6') -Recurse -File)
if($family.Count-ne48 -or $research.Count-ne30){throw 'Expected exact48 family and30 research files'}
$docs=@('LICENSE','doc/CAUSAL_FRESH_V6_CARD.json','doc/CAUSAL_FRESH_V6_CARD.md',
 'doc/causal_fresh_v6/stage-source.ps1','doc/causal_fresh_v6/unpack-source.py','doc/causal_fresh_v6/audit-build.py','doc/causal_fresh_v6/run-quality.py','doc/causal_fresh_v6/execute-stage.py','doc/causal_fresh_v6/verify-mirror.ps1',
 'doc/causal_fresh_v6/input-manifest.json','doc/causal_fresh_v6/raw-controls.json','doc/causal_fresh_v6/baseline-control-v3.json','doc/causal_fresh_v6/baseline-control-v4.json','doc/causal_fresh_v6/baseline-control-v5.json','doc/causal_fresh_v6/goal-protocol.json',
 'doc/CAUSAL_V4_V5_TRAIN_DIAGNOSIS_CARD.json','doc/causal_v4_v5_train_diagnosis/diagnosis.json','doc/causal_v4_v5_train_diagnosis/independent-audit.json','doc/causal_fresh_v5/final-preservation.json',
 'doc/CAUSAL_V4_V5_TRAIN_DIAGNOSIS.md','doc/causal_v5_retention_review/independent-review.json',
 'code/encoders/causal_patch_bottleneck_mae/versions/CPB-v6.json'
 )
$sources=@($family)+@($research)+@(Get-Item -LiteralPath (Join-Path $researchRoot 'code/encoders/causal_patch_bottleneck_mae/versions/v6.h'))+@($docs|ForEach-Object{Get-Item -LiteralPath (Join-Path $researchRoot $_)})+@(Get-Item -LiteralPath $preflightPath)
$files=@($sources|Sort-Object FullName -Unique|ForEach-Object{
 if($_.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Source links forbidden'}
 [ordered]@{path=$_.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$selfReceiptRelative=$preflightPath.Substring($researchRoot.Length+1).Replace([char]92,[char]47)
$reviewed=@($preflight.sources.path|Sort-Object -Unique)
$packaged=@($files.path|Where-Object {$_ -ne $selfReceiptRelative}|Sort-Object -Unique)
if($reviewed.Count-ne$preflight.sources.Count -or $packaged.Count-ne$reviewed.Count -or (Compare-Object $packaged $reviewed)){throw 'Capsule must exactly match the independently reviewed source set plus its receipt'}
$manifest=[ordered]@{kind='causal_fresh_v6_source_capsule_v1';attempt=$attemptName;created_utc=[DateTime]::UtcNow.ToString('o');research_root=$researchRoot;container='cuwacunu';container_root="/opt/cuwacunu/research/causal-fresh-v6/$attemptName";card_sha256=$cardHash;card_markdown_sha256=(File-Pin (Join-Path $researchRoot 'doc/CAUSAL_FRESH_V6_CARD.md')).sha256;input_manifest_sha256=$inputHash;input_pins_verified=$card.input_pins.Count;baseline_v3_file_pairs_unchanged=47;gradient_policy=$policy;files=$files}
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($attemptRoot))|Out-Null
$null=New-Item -ItemType Directory -Path $attemptRoot -ErrorAction Stop
$manifestPath=Join-Path $attemptRoot 'source-manifest.json'
$manifest|ConvertTo-Json -Depth 14|Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $attemptRoot 'source.zip'
$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($e in $files){$p=Join-Path $researchRoot $e.path;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$p,$e.path,[IO.Compression.CompressionLevel]::Fastest)|Out-Null;Require-Pin $p $e};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$manifestPath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$zip.Dispose();$stream.Dispose()}
[ordered]@{status='packaged';attempt=$attemptName;source_files=$files.Count;zip_path=$zipPath;zip_sha256=(File-Pin $zipPath).sha256;manifest_sha256=(File-Pin $manifestPath).sha256;card_sha256=$cardHash;input_manifest_sha256=$inputHash}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $attemptRoot 'package.json') -Encoding utf8NoBOM
Get-Content -LiteralPath (Join-Path $attemptRoot 'package.json') -Raw

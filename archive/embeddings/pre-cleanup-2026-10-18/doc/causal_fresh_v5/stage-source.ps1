# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host orchestration/file inspection only. All project execution remains in cuwacunu.
param([Parameter(Mandatory=$true)][ValidateRange(1,999)][int]$Attempt,
 [Alias('Preflight')][ValidatePattern('^protocol-preflight[-a-z0-9]*\.json$')][string]$PreflightReceiptName='protocol-preflight.json')
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$integrationRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '../cuwacunu_dev'))
$attemptName='attempt-{0:D3}' -f $Attempt
$attemptRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v5/$attemptName"
if(Test-Path -LiteralPath $attemptRoot){throw 'Preserve existing source attempt'}
$cardHash='a6da3a9501dd4247c40e3e24e42faf2c2c46620fda7e0a0568709276d0e84d09'
$inputHash='e85b01306634a9a6b2aa76fb31e560fdba807ddbca60c533ffd56c3103148ee7'
$policy='separate-mean-and-uncertainty-bank-projections-live-gaussian-nll-v1'
function File-Pin([string]$p){$f=Get-Item -LiteralPath $p;return [ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$p,$pin){$a=File-Pin $p;if($a.bytes-ne$pin.bytes -or $a.sha256-ne$pin.sha256){throw "Changed frozen input: $p"}}
$cardPath=Join-Path $researchRoot 'doc/CAUSAL_FRESH_V5_CARD.json'
if((File-Pin $cardPath).sha256-ne$cardHash){throw 'Prospective card changed'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
if($card.model-ne'CPB-v5' -or $card.mechanism.kind-ne$policy -or $card.geometry.native_width-ne40 -or $card.advancement.criterion_count-ne22){throw 'Unexpected v5 declaration'}
foreach($pin in @($card.input_pins)+@($card.historical_preservation_pins)){
 $pinRoot=if($pin.repository-eq'cuwacunu_dev'){$integrationRoot}else{$researchRoot}
 Require-Pin (Join-Path $pinRoot $pin.path) $pin
}
Require-Pin (Join-Path $researchRoot $card.input_manifest.path) $card.input_manifest
if($card.input_manifest.sha256-ne$inputHash){throw 'Input manifest changed'}
if($card.baseline_frozen_files.Count-ne47){throw 'Expected47 protected v3 source files'}
foreach($pin in $card.baseline_frozen_files){foreach($r in @($researchRoot,$integrationRoot)){Require-Pin (Join-Path $r $pin.path) $pin}}
$bound=@($card.saved_TRAIN_diagnosis.card,$card.saved_TRAIN_diagnosis.result,$card.saved_TRAIN_diagnosis.independent_audit,$card.baseline_preservation,$card.comparison_control.v3,$card.comparison_control.v4,$card.advancement.goal_protocol)
foreach($pin in $bound){Require-Pin (Join-Path $researchRoot $pin.path) $pin}
$audit=Get-Content -LiteralPath (Join-Path $researchRoot $card.saved_TRAIN_diagnosis.independent_audit.path) -Raw|ConvertFrom-Json
if($audit.status-ne'passed'){throw 'Independent saved TRAIN diagnosis audit required'}
$preflightPath=Join-Path $PSScriptRoot $PreflightReceiptName
$preflight=Get-Content -LiteralPath $preflightPath -Raw|ConvertFrom-Json
if($preflight.status-ne'passed' -or $preflight.card_sha256-ne$cardHash){throw 'Independent source protocol preflight required'}
foreach($pin in $preflight.sources){Require-Pin (Join-Path $researchRoot $pin.path) $pin}
$family=@(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/encoders/causal_patch_bottleneck_mae/versions/v5') -Recurse -File)
$research=@(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/research/causal_fresh_v5') -Recurse -File)
if($family.Count-lt47 -or $research.Count-lt29){throw 'Incomplete owned v5 closure'}
$docs=@('LICENSE','doc/CAUSAL_FRESH_V5_CARD.json','doc/CAUSAL_FRESH_V5_CARD.md',
 'doc/causal_fresh_v5/stage-source.ps1','doc/causal_fresh_v5/unpack-source.py','doc/causal_fresh_v5/audit-build.py','doc/causal_fresh_v5/run-quality.py','doc/causal_fresh_v5/execute-stage.py',
 'doc/causal_fresh_v5/input-manifest.json','doc/causal_fresh_v5/raw-controls.json','doc/causal_fresh_v5/baseline-control-v3.json','doc/causal_fresh_v5/baseline-control-v4.json','doc/causal_fresh_v5/goal-protocol.json',
 'doc/CAUSAL_V3_V4_TRAIN_DIAGNOSIS_CARD.json','doc/causal_v3_v4_train_diagnosis/diagnosis.json','doc/causal_v3_v4_train_diagnosis/independent-audit.json','doc/causal_v3_v4_train_diagnosis/preservation-before.json'
 )
$sources=@($family)+@($research)+@(Get-Item -LiteralPath (Join-Path $researchRoot 'code/encoders/causal_patch_bottleneck_mae/versions/v5.h'))+@($docs|ForEach-Object{Get-Item -LiteralPath (Join-Path $researchRoot $_)})+@(Get-Item -LiteralPath $preflightPath)
$files=@($sources|Sort-Object FullName -Unique|ForEach-Object{
 if($_.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Source links forbidden'}
 [ordered]@{path=$_.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$manifest=[ordered]@{kind='causal_fresh_v5_source_capsule_v1';attempt=$attemptName;created_utc=[DateTime]::UtcNow.ToString('o');research_root=$researchRoot;container='cuwacunu';container_root="/opt/cuwacunu/research/causal-fresh-v5/$attemptName";card_sha256=$cardHash;card_markdown_sha256=(File-Pin (Join-Path $researchRoot 'doc/CAUSAL_FRESH_V5_CARD.md')).sha256;input_manifest_sha256=$inputHash;input_pins_verified=$card.input_pins.Count;baseline_v3_file_pairs_unchanged=47;gradient_policy=$policy;files=$files}
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

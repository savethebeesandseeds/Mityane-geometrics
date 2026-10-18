# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Owned derivative of doc/causal_fresh_v3/stage-source.ps1.
# Host-side source inspection/packaging only; all builds and model work are container-only.
param([Parameter(Mandatory=$true)][ValidateRange(1,999)][int]$Attempt)
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$integrationRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '../cuwacunu_dev'))
$attemptName='attempt-{0:D3}' -f $Attempt
$attemptRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v4/$attemptName"
if(Test-Path -LiteralPath $attemptRoot){throw 'Refusing to overwrite an existing source attempt'}
$cardPath=Join-Path $researchRoot 'doc/CAUSAL_FRESH_V4_CARD.json'
$cardHash='114026b2e6d11d84aee1855a8af673d84e782ab057eb45d398a5d66fa94b1dce'
$inputHash='a6bff90ffc9dd917c6a9c446659c52e0c76b5fb34a9f67201788eb3168948eb9'
function File-Pin([string]$path){$f=Get-Item -LiteralPath $path;return [ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$pin){$actual=File-Pin $path;if($actual.bytes-ne$pin.bytes -or $actual.sha256-ne$pin.sha256){throw "Changed pinned input/history: $path"}}
if((File-Pin $cardPath).sha256-ne$cardHash){throw 'Prospective machine card changed'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
if($card.model-ne'CPB-v4' -or $card.mechanism.kind-ne'variance_input_gradient_stop-v1' -or $card.input_pins.Count-ne20 -or $card.baseline_frozen_files.Count-ne47){throw 'Unexpected frozen v4 declaration'}
$cardMarkdown=Join-Path $researchRoot 'doc/CAUSAL_FRESH_V4_CARD.md'
# Bind the actual frozen v4 Markdown bytes; no historical Markdown hash is reused.
$cardMarkdownPin=File-Pin $cardMarkdown
foreach($pin in @($card.input_pins)+@($card.historical_preservation_pins)){
    $pinRoot=if($pin.repository-eq'cuwacunu_dev'){$integrationRoot}else{$researchRoot}
    Require-Pin (Join-Path $pinRoot $pin.path) $pin
}
$inputPath=Join-Path $researchRoot $card.input_manifest.path
Require-Pin $inputPath $card.input_manifest
if((File-Pin $inputPath).sha256-ne$inputHash){throw 'Frozen data-only manifest changed'}
foreach($pin in $card.baseline_frozen_files){
    if(!$pin.path.StartsWith('code/encoders/causal_patch_bottleneck_mae/versions/v3/')){throw 'Only clean v3 baseline source pins admitted'}
    foreach($pinRoot in @($researchRoot,$integrationRoot)){Require-Pin (Join-Path $pinRoot $pin.path) $pin}
}
foreach($pin in @($card.saved_TRAIN_diagnosis,$card.baseline_preservation)){
    if((File-Pin (Join-Path $researchRoot $pin.path)).sha256-ne$pin.sha256){throw "Changed diagnosis/preservation: $($pin.path)"}
}
Require-Pin (Join-Path $researchRoot $card.comparison_control.path) $card.comparison_control
$diagnosticAudit=Join-Path $researchRoot 'doc/causal_v3_train_diagnosis/independent-audit.json'
if((Get-Content -LiteralPath $diagnosticAudit -Raw|ConvertFrom-Json).status-ne'passed'){throw 'Saved TRAIN diagnosis audit required'}
$family=@(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/encoders/causal_patch_bottleneck_mae/versions/v4') -Recurse -File)
$research=@(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/research/causal_fresh_v4') -Recurse -File)
if($family.Count-ne48 -or $research.Count-ne27){throw 'Expected48 owned family files and26 research files plus compare_baseline.py'}
$null=Get-Item -LiteralPath (Join-Path $researchRoot 'code/research/causal_fresh_v4/evaluation/compare_baseline.py')
$sources=@($family)+@($research)+@(
    Get-Item -LiteralPath $cardPath
    Get-Item -LiteralPath $cardMarkdown
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v4/stage-source.ps1')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v4/unpack-source.py')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v4/audit-build.py')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v4/run-quality.py')
    Get-Item -LiteralPath $inputPath
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v4/raw-controls.json')
    Get-Item -LiteralPath (Join-Path $researchRoot $card.comparison_control.path)
    Get-Item -LiteralPath (Join-Path $researchRoot $card.saved_TRAIN_diagnosis.path)
    Get-Item -LiteralPath $diagnosticAudit
    Get-Item -LiteralPath (Join-Path $researchRoot $card.baseline_preservation.path)
    Get-Item -LiteralPath (Join-Path $researchRoot 'LICENSE')
)
$sources=@($sources|Sort-Object FullName -Unique)
$files=@($sources|ForEach-Object{
    if($_.Attributes-band[IO.FileAttributes]::ReparsePoint){throw "Source links forbidden: $($_.FullName)"}
    [ordered]@{path=$_.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$helperSources=@('stage-source.ps1','unpack-source.py','audit-build.py','run-quality.py')
$helperDerivation=@($helperSources|ForEach-Object{
    $path=Join-Path $researchRoot "doc/causal_fresh_v3/$_"
    [ordered]@{direct_source="doc/causal_fresh_v3/$_";bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$manifest=[ordered]@{
    kind='causal_fresh_v4_source_capsule_v1';attempt=$attemptName;created_utc=[DateTime]::UtcNow.ToString('o')
    research_root=$researchRoot;container='cuwacunu';container_root="/opt/cuwacunu/research/causal-fresh-v4/$attemptName"
    card_sha256=$cardHash;card_markdown_sha256=$cardMarkdownPin.sha256;input_manifest_sha256=$inputHash;input_pins_verified=20
    baseline_v3_file_pairs_unchanged=47;gradient_policy='variance_input_gradient_stop-v1'
    helper_derivation=[ordered]@{operation='Owned full v3 helper copies adapted for v4; v3 helpers never executed or imported';sources=$helperDerivation}
    files=$files
}
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($attemptRoot))|Out-Null
$null=New-Item -ItemType Directory -Path $attemptRoot -ErrorAction Stop
$manifestPath=Join-Path $attemptRoot 'source-manifest.json'
$manifest|ConvertTo-Json -Depth 14|Set-Content -LiteralPath $manifestPath -Encoding utf8
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $attemptRoot 'source.zip'
$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{
    foreach($entry in $files){
        $path=Join-Path $researchRoot $entry.path
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$entry.path,[IO.Compression.CompressionLevel]::Fastest)|Out-Null
        Require-Pin $path $entry
    }
    [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$manifestPath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null
}finally{$zip.Dispose();$stream.Dispose()}
[ordered]@{status='packaged';attempt=$attemptName;source_files=$files.Count;zip_path=$zipPath
    zip_sha256=(File-Pin $zipPath).sha256;manifest_sha256=(File-Pin $manifestPath).sha256
    card_sha256=$cardHash;card_markdown_sha256=$cardMarkdownPin.sha256;input_manifest_sha256=$inputHash
}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $attemptRoot 'package.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $attemptRoot 'package.json') -Raw

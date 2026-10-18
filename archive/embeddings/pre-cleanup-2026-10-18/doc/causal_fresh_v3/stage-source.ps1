# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host-side file inspection and source packaging only; compilation/execution is container-only.
param([Parameter(Mandatory=$true)][ValidateRange(1,999)][int]$Attempt)
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$integrationRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '../cuwacunu_dev'))
$attemptName='attempt-{0:D3}' -f $Attempt
$attemptRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v3/$attemptName"
if(Test-Path -LiteralPath $attemptRoot){throw 'Refusing to overwrite an existing source attempt'}
$cardPath=Join-Path $researchRoot 'doc/CAUSAL_FRESH_V3_CARD.json'
$cardHash='f88fa9751ff9da0e808805da9975fb0bf3afd4433688f3cfb420686392e6396f'
if((Get-FileHash -LiteralPath $cardPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $cardHash){throw 'Prospective machine card changed'}
$cardMarkdown=Join-Path $researchRoot 'doc/CAUSAL_FRESH_V3_CARD.md'
if((Get-FileHash -LiteralPath $cardMarkdown -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'd03c4f60f2eb270bc7d6ba77d575f998703c2c204a90b3df9d5ec0d938039446'){throw 'Prospective Markdown card changed'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
foreach($pin in @($card.input_pins)+@($card.historical_preservation_pins)){
    $pinRoot=if($pin.repository -eq 'cuwacunu_dev'){$integrationRoot}else{$researchRoot}
    $path=Join-Path $pinRoot $pin.path
    $file=Get-Item -LiteralPath $path
    if($file.Length-ne$pin.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$pin.sha256){throw "Changed input/history: $($pin.path)"}
}
$inputPath=Join-Path $researchRoot $card.input_manifest.path
if((Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()-ne$card.input_manifest.sha256){throw 'Frozen data-only input manifest changed'}
foreach($pin in $card.baseline_frozen_files){
    foreach($root in @($researchRoot,$integrationRoot)){
        $path=Join-Path $root $pin.path
        $file=Get-Item -LiteralPath $path
        if($file.Length-ne$pin.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$pin.sha256){throw "Baseline source changed: $path"}
    }
}
$sources=@(
    Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/encoders/causal_patch_bottleneck_mae/versions/v3') -Recurse -File
    Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/research/causal_fresh_v3') -Recurse -File
    Get-Item -LiteralPath $cardPath
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/CAUSAL_FRESH_V3_CARD.md')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v3/stage-source.ps1')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v3/unpack-source.py')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v3/audit-build.py')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v3/run-quality.py')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v3/input-manifest.json')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_fresh_v3/raw-controls.json')
    Get-Item -LiteralPath (Join-Path $researchRoot 'LICENSE')
)
$sources=@($sources|Sort-Object FullName -Unique)
$files=@($sources|ForEach-Object{
    if($_.Attributes-band[IO.FileAttributes]::ReparsePoint){throw "Source links are forbidden: $($_.FullName)"}
    [ordered]@{path=$_.FullName.Substring($researchRoot.Length+1).Replace('\','/');bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$manifest=[ordered]@{
    kind='causal_fresh_v3_source_capsule_v1';attempt=$attemptName;created_utc=[DateTime]::UtcNow.ToString('o');
    research_root=$researchRoot;container='cuwacunu';
    container_root="/opt/cuwacunu/research/causal-fresh-v3/$attemptName";
    card_sha256=$cardHash;input_pins_verified=$card.input_pins.Count;
    baseline_v1_file_pairs_unchanged=$card.baseline_frozen_files.Count;files=$files
}
[IO.Directory]::CreateDirectory($attemptRoot)|Out-Null
$manifestPath=Join-Path $attemptRoot 'source-manifest.json'
$manifest|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $manifestPath -Encoding utf8
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $attemptRoot 'source.zip'
$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{
    foreach($entry in $files){
        $path=Join-Path $researchRoot $entry.path
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$entry.path,[IO.Compression.CompressionLevel]::Fastest)|Out-Null
        if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$entry.sha256){throw "Source changed while packaging: $($entry.path)"}
    }
    [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$manifestPath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null
}finally{$zip.Dispose();$stream.Dispose()}
[ordered]@{status='packaged';attempt=$attemptName;source_files=$files.Count;zip_path=$zipPath;
    zip_sha256=(Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant();
    manifest_sha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant();
    card_sha256=$cardHash}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $attemptRoot 'package.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $attemptRoot 'package.json') -Raw

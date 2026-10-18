# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host-side file inspection and source packaging only; compilation/execution is container-only.
param([Parameter(Mandatory=$true)][ValidateRange(1,999)][int]$Attempt)
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$integrationRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '../cuwacunu_dev'))
$attemptName='attempt-{0:D3}' -f $Attempt
$attemptRoot=Join-Path $researchRoot "output/engineering/causal-finetune-v2/$attemptName"
if(Test-Path -LiteralPath $attemptRoot){throw 'Refusing to overwrite an existing source attempt'}
$cardPath=Join-Path $researchRoot 'doc/CAUSAL_FINETUNE_V2_CARD.json'
$cardHash='30aa70ce6f3999d9f6a991c1b298516db41caf3db276cacb75d91c9d3dcba222'
if((Get-FileHash -LiteralPath $cardPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $cardHash){throw 'Prospective machine card changed'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
foreach($pin in $card.input_pins){
    $path=Join-Path $integrationRoot $pin.path
    $file=Get-Item -LiteralPath $path
    if($file.Length-ne$pin.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$pin.sha256){throw "Changed input: $($pin.path)"}
}
foreach($pin in $card.baseline_frozen_files){
    foreach($root in @($researchRoot,$integrationRoot)){
        $path=Join-Path $root $pin.path
        $file=Get-Item -LiteralPath $path
        if($file.Length-ne$pin.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$pin.sha256){throw "Baseline source changed: $path"}
    }
}
$sources=@(
    Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/encoders/causal_patch_bottleneck_mae/versions/v2') -Recurse -File
    Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/research/causal_finetune_v2') -Recurse -File
    Get-Item -LiteralPath $cardPath
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/CAUSAL_FINETUNE_V2_CARD.md')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_finetune_v2/stage-source.ps1')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_finetune_v2/unpack-source.py')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_finetune_v2/audit-build.py')
    Get-Item -LiteralPath (Join-Path $researchRoot 'doc/causal_finetune_v2/run-quality.py')
    Get-Item -LiteralPath (Join-Path $researchRoot 'LICENSE')
)
$sources=@($sources|Sort-Object FullName -Unique)
$files=@($sources|ForEach-Object{
    if($_.Attributes-band[IO.FileAttributes]::ReparsePoint){throw "Source links are forbidden: $($_.FullName)"}
    [ordered]@{path=$_.FullName.Substring($researchRoot.Length+1).Replace('\','/');bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$manifest=[ordered]@{
    kind='causal_finetune_v2_source_capsule_v1';attempt=$attemptName;created_utc=[DateTime]::UtcNow.ToString('o');
    research_root=$researchRoot;container='cuwacunu';
    container_root="/opt/cuwacunu/research/causal-finetune-v2/$attemptName";
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

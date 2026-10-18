# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host byte-only packaging of exactly the reviewed source closure.
param([Parameter(Mandatory=$true)][ValidateRange(1,999)][int]$Attempt,[Parameter(Mandatory=$true)][string]$CardSHA256)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$label='attempt-{0:D3}' -f $Attempt
$destination=Join-Path $root ('output/engineering/causal-fresh-v9/'+$label)
if(Test-Path -LiteralPath $destination){throw 'Preserve existing attempt'}
function Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Physical files required'};[ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$p){$a=Pin $path;if($a.bytes-ne$p.bytes-or$a.sha256-ne$p.sha256){throw ('Changed reviewed input: '+$path)}}
$cardPath=Join-Path $root 'doc/CAUSAL_FRESH_V9_CARD.json'
if((Pin $cardPath).sha256-ne$CardSHA256){throw 'Frozen card hash mismatch'}
$card=Get-Content -LiteralPath $cardPath -Raw|ConvertFrom-Json
if($card.stage.Contains('DRAFT')-or$null-eq$card.frozen_utc-or$card.model-ne'CPB-v9'-or-not$card.execution_admitted){throw 'Frozen and explicitly admitted V9 card required'}
$preflightPath=Join-Path $PSScriptRoot 'protocol-preflight.json'
$preflight=Get-Content -LiteralPath $preflightPath -Raw|ConvertFrom-Json
if($preflight.status-ne'passed'-or$preflight.card_sha256-ne$CardSHA256){throw 'Exact passed preflight required'}
foreach($p in $preflight.sources){Require-Pin (Join-Path $root $p.path) $p}
$files=@($preflight.sources)+@([ordered]@{path='doc/causal_fresh_v9/protocol-preflight.json';bytes=(Pin $preflightPath).bytes;sha256=(Pin $preflightPath).sha256})
if(@($files.path|Sort-Object -Unique).Count-ne$files.Count){throw 'Duplicate capsule member'}
$manifest=[ordered]@{kind='causal_fresh_v9_source_capsule_v1';attempt=$label;created_utc=[DateTime]::UtcNow.ToString('o');research_root=$root;container='cuwacunu';container_root=('/opt/cuwacunu/research/causal-fresh-v9/'+$label);card_sha256=$CardSHA256;card_markdown_sha256=(Pin (Join-Path $root 'doc/CAUSAL_FRESH_V9_CARD.md')).sha256;input_manifest_sha256=$card.input_manifest.sha256;gradient_policy=$card.mechanism.gradient_policy;temporal_pooling_policy=$card.mechanism.temporal_pooling_policy;temporal_pooling_partition=$card.mechanism.temporal_pooling_partition;temporal_pooling_mix=$card.mechanism.temporal_pooling_mix;empty_local_bin_policy=$card.mechanism.empty_local_bin_policy;files=$files}
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))|Out-Null
$null=New-Item -ItemType Directory -Path $destination -ErrorAction Stop
$manifestPath=Join-Path $destination 'source-manifest.json'
$manifest|ConvertTo-Json -Depth 14|Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$archive=Join-Path $destination 'source.zip'
$stream=[IO.File]::Open($archive,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{
 foreach($p in $files){$path=Join-Path $root $p.path;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$p.path,[IO.Compression.CompressionLevel]::Fastest)|Out-Null;Require-Pin $path $p}
 [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$manifestPath,'CAPSULE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null
}finally{$zip.Dispose();$stream.Dispose()}
$receipt=[ordered]@{status='packaged';attempt=$label;source_files=$files.Count;zip_path=$archive;zip_sha256=(Pin $archive).sha256;manifest_sha256=(Pin $manifestPath).sha256;card_sha256=$CardSHA256;input_manifest_sha256=$card.input_manifest.sha256}
$receipt|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $destination 'package.json') -Encoding utf8NoBOM
$receipt|ConvertTo-Json -Depth 5

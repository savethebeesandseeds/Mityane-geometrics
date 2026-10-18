# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
param([Parameter(Mandatory=$true)][string]$Receipt)
$ErrorActionPreference='Stop'
$integration='C:\Work\cuwacunu\cuwacunu_dev'
$portable='C:\Work\cuwacunu\cuwacunu_embedding'
$members=@()
foreach($relativeRoot in @('code\encoders\causal_patch_bottleneck_mae','code\encoders\raw_patch_bottleneck_mae\versions')) {
  $leftRoot=Join-Path $integration $relativeRoot
  $rightRoot=Join-Path $portable $relativeRoot
  $left=@(Get-ChildItem -LiteralPath $leftRoot -File -Recurse | ForEach-Object {[IO.Path]::GetRelativePath($leftRoot,$_.FullName)} | Sort-Object)
  $right=@(Get-ChildItem -LiteralPath $rightRoot -File -Recurse | ForEach-Object {[IO.Path]::GetRelativePath($rightRoot,$_.FullName)} | Sort-Object)
  if(@(Compare-Object $left $right).Count -ne 0) {throw "Portable file inventory differs: $relativeRoot"}
  foreach($name in $left) {
    $relative=Join-Path $relativeRoot $name
    $a=Get-Item -LiteralPath (Join-Path $integration $relative)
    $b=Get-Item -LiteralPath (Join-Path $portable $relative)
    if(($a.Attributes -band [IO.FileAttributes]::ReparsePoint) -or ($b.Attributes -band [IO.FileAttributes]::ReparsePoint)) {throw "Owned copies must be files: $relative"}
    $digest=(Get-FileHash -LiteralPath $a.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if($digest -ne (Get-FileHash -LiteralPath $b.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -or $a.Length -ne $b.Length) {throw "Portable bytes differ: $relative"}
    $members+=@{path=$relative.Replace('\','/');sha256=$digest;bytes=$a.Length}
  }
}
$result=@{kind='causal_patch_bottleneck_mae_v1_portable_sync';status='passed';pairs_verified=$members.Count;all_byte_identical=$true;exact_family_inventories=$true;owned_files_not_links=$true;files=$members;model_execution=$false}
$target=[IO.Path]::GetFullPath((Join-Path $integration $Receipt))
if(-not $target.StartsWith((Join-Path $integration 'doc\results\causal_patch_bottleneck_mae_v1\'),[StringComparison]::OrdinalIgnoreCase)) {throw 'Receipt must be in the new causal evidence directory'}
$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($result | ConvertTo-Json -Depth 8)+"`n");$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
Write-Output "Verified $($members.Count) mirrored owned files"
Get-FileHash -LiteralPath $target -Algorithm SHA256

# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host byte inspection only; project execution remains inside cuwacunu.
$ErrorActionPreference='Stop'
$taskRoot='C:/Work/cuwacunu/cuwacunu_dev'
$taskPortable='C:/Work/cuwacunu/cuwacunu_embedding'
$taskOutput=Join-Path $taskRoot 'doc/results/market_parent_conditional_changes_v1/portable-sync.json'
if(Test-Path -LiteralPath $taskOutput){throw 'Preserve the existing portable receipt'}
function Record([string]$taskPath){
 $taskFile=Get-Item -LiteralPath $taskPath
 return [ordered]@{sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $taskPath).Hash.ToLowerInvariant();bytes=$taskFile.Length}
}
$taskPrior=Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'doc/results/market_parent_conditional_changes_v1/preflight.json')|ConvertFrom-Json
$taskPairs=@($taskPrior.portable_pairs)
$taskNames=@()
$taskHeaders=Join-Path $taskRoot 'code/encoders/rpb_conditional/include'
$taskHeaderPrefix=(Get-Item -LiteralPath $taskHeaders).FullName.TrimEnd('\')+'\'
foreach($taskFile in Get-ChildItem -LiteralPath $taskHeaders -Recurse -File){
 if(-not $taskFile.FullName.StartsWith($taskHeaderPrefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Unexpected header path outside module'}
 $taskRelative=$taskFile.FullName.Substring($taskHeaderPrefix.Length).Replace('\','/')
 $taskNames+=@{integration_path='code/encoders/rpb_conditional/include/'+$taskRelative;embedding_path='code/encoders/raw_patch_bottleneck_mae/include/'+$taskRelative}
}
$taskNames+=@{integration_path='code/encoders/rpb_conditional/tests/parent_conditional_change_test.cpp';embedding_path='code/encoders/raw_patch_bottleneck_mae/tests/parent_conditional_change_test.cpp'}
$taskNames+=@{integration_path='config/market_studies/encoder_parent_conditional_changes.conf';embedding_path='code/encoders/raw_patch_bottleneck_mae/config/parent_conditional_changes.conf'}
if($taskNames.Count -ne 9){throw 'Expected exactly9 new reusable pairs'}
foreach($taskName in $taskNames){
 $taskLeft=Record (Join-Path $taskRoot $taskName.integration_path)
 $taskRight=Record (Join-Path $taskPortable $taskName.embedding_path)
 if($taskLeft.sha256 -ne $taskRight.sha256 -or $taskLeft.bytes -ne $taskRight.bytes){throw ('New portable mismatch: '+$taskName.integration_path)}
 $taskPairs += [ordered]@{integration_path=$taskName.integration_path;embedding_path=$taskName.embedding_path;group='v27_conditioned';sha256=$taskLeft.sha256;bytes=$taskLeft.bytes;match=$true}
}
foreach($taskPair in $taskPairs){
 foreach($taskPath in @((Join-Path $taskRoot $taskPair.integration_path),(Join-Path $taskPortable $taskPair.embedding_path))){
  $taskActual=Record $taskPath
  if($taskActual.sha256 -ne $taskPair.sha256 -or $taskActual.bytes -ne $taskPair.bytes){throw ('Prior portable mismatch: '+$taskPath)}
 }
}
$taskAdapters=@()
foreach($taskSpec in @(@{root=$taskRoot;repository='integration';path='code/encoders/rpb_conditional/parent_conditional_change_test.mk'},@{root=$taskPortable;repository='embedding';path='code/encoders/raw_patch_bottleneck_mae/parent_conditional_change.mk'})){
 $taskHash=Record (Join-Path $taskSpec.root $taskSpec.path)
 $taskAdapters += [ordered]@{repository=$taskSpec.repository;path=$taskSpec.path;sha256=$taskHash.sha256;bytes=$taskHash.bytes;context_specific=$true;byte_identical_pair_claimed=$false}
}
$taskValue=[ordered]@{kind='market_parent_conditional_changes_portable_sync_v1';status='passed';prior_pairs=64;new_pairs=9;total_pairs=$taskPairs.Count;pairs=$taskPairs;build_adapters=$taskAdapters;primary_unchanged='RPB-v22';scope='Local byte verification only; no sibling execution or publication'}
$taskStream=[IO.File]::Open($taskOutput,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
try{$taskWriter.WriteLine(($taskValue|ConvertTo-Json -Depth 12))}finally{$taskWriter.Dispose()}
Write-Output ('PARENT_CONDITIONAL_PORTABLE_PAIRS_PASSED='+$taskPairs.Count)

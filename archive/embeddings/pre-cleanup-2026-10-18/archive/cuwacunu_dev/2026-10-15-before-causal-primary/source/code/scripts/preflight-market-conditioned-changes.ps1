# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host file inspection only; all project execution stays in the container.
$ErrorActionPreference='Stop'
$taskRoot='C:/Work/cuwacunu/cuwacunu_dev'
$taskPortable='C:/Work/cuwacunu/cuwacunu_embedding'
$taskOut=Join-Path $taskRoot 'doc/results/market_conditioned_changes_v1/preflight.json'
if(Test-Path -LiteralPath $taskOut){throw 'Preserve the existing preflight'}
function Record([string]$taskPath){
  $taskItem=Get-Item -LiteralPath $taskPath
  return [ordered]@{sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $taskPath).Hash.ToLowerInvariant();bytes=$taskItem.Length}
}
function Require-Record([string]$taskPath,$taskExpected){
  $taskNow=Record $taskPath
  if($taskNow.sha256 -ne $taskExpected.sha256 -or $taskNow.bytes -ne $taskExpected.bytes){throw ('Changed protected file: '+$taskPath)}
}
$taskPriorPath=Join-Path $taskRoot 'doc/results/market_change_scale_diagnosis_v1/preflight.json'
$taskPrior=Get-Content -Raw -LiteralPath $taskPriorPath | ConvertFrom-Json
$taskLedger=Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'doc/encoder_search_ledger.json') | ConvertFrom-Json
$taskPins=@{}
foreach($taskCandidate in $taskLedger.candidate_entries){
  foreach($taskPin in $taskCandidate.checkpoints){$taskPins[$taskPin.path]=$taskPin}
  foreach($taskProperty in @('card','report','completion','own_frozen_result_table')){
    $taskAsset=$taskCandidate.$taskProperty
    if($null -ne $taskAsset){Require-Record (Join-Path $taskRoot $taskAsset.path) $taskAsset}
  }
}
foreach($taskPin in $taskPrior.checkpoints){
  if($taskPins.ContainsKey($taskPin.path) -and $taskPins[$taskPin.path].sha256 -ne $taskPin.sha256){throw 'Conflicting historical checkpoint pins'}
  $taskPins[$taskPin.path]=$taskPin
}
$taskCheckpointRecords=@()
foreach($taskName in ($taskPins.Keys | Sort-Object)){
  $taskPin=$taskPins[$taskName]
  Require-Record (Join-Path $taskRoot $taskName) $taskPin
  $taskCheckpointRecords += [ordered]@{path=$taskName;sha256=$taskPin.sha256;bytes=$taskPin.bytes;git_ignored=$true;deserialized=$false}
}
$taskIgnored=@($taskPins.Keys | & git -c safe.directory=C:/Work/cuwacunu/cuwacunu_dev -C $taskRoot check-ignore --stdin)
if($LASTEXITCODE -ne 0 -or $taskIgnored.Count -ne $taskPins.Count){throw 'All preserved checkpoints must remain Git ignored'}
foreach($taskPair in $taskPrior.portable_pairs){
  Require-Record (Join-Path $taskRoot $taskPair.integration_path) $taskPair
  Require-Record (Join-Path $taskPortable $taskPair.embedding_path) $taskPair
}
$taskFrozen=@{
 'doc/MARKET_CONDITIONED_CHANGE_CARD.md'='66de5fbe159230c82c04bd45a0d48cdf0694e101a082319da2f8bf2bd8b10e84';
 'doc/MARKET_CHANGE_SCALE_DIAGNOSIS_RESULTS.md'='86b65a3bc245ff49b74852aa006a30ebe077ca0f87d6ce73f7db8d8ae9cd24ff';
 'doc/results/market_change_scale_diagnosis_v1/completion.json'='ad3df25fbc6929e97d1dbf3696e3379432340788648e92a89bcede9b53ca6072';
 'doc/MARKET_CHANGE_ENERGY_RESULTS.md'='42a50ac642600538a7ec1e365e337802e006d681b1bcf0e4b5aa67965eb38f08';
 'doc/LATENT_MARKET_DYNAMICS_ROADMAP.md'='56b0c0e284dae2c93318abe89c74be685e8034ee7a22a4e5325020124fbca617'
}
$taskFrozenRecords=@()
foreach($taskName in ($taskFrozen.Keys | Sort-Object)){
  $taskNow=Record (Join-Path $taskRoot $taskName)
  if($taskNow.sha256 -ne $taskFrozen[$taskName]){throw ('Frozen hash mismatch: '+$taskName)}
  $taskFrozenRecords += [ordered]@{path=$taskName;sha256=$taskNow.sha256;bytes=$taskNow.bytes}
}
$taskGit=@()
foreach($taskRepo in @($taskRoot,$taskPortable)){
 $taskHead=& git -c ('safe.directory='+$taskRepo) -C $taskRepo rev-parse HEAD
 if($LASTEXITCODE -ne 0){throw 'Git diagnostic failed'}
 $taskGit += [ordered]@{root=$taskRepo;head=$taskHead}
}
$taskValue=[ordered]@{
 kind='market_conditioned_changes_preflight_v1';status='passed';date='2026-10-15';
 scope='Metadata/SHA256/size/Git-ignore only, no tensor deserialization or project execution';
 card_sha256=$taskFrozen['doc/MARKET_CONDITIONED_CHANGE_CARD.md'];primary='RPB-v22';balanced_reference='RPB-v24.market-01.alt-01';
 checkpoints_verified=$taskCheckpointRecords.Count;checkpoints=$taskCheckpointRecords;
 all_checkpoints_available_hash_matched_and_ignored=$true;
 prior_portable_pairs_verified=$taskPrior.portable_pairs.Count;portable_pairs=$taskPrior.portable_pairs;
 all_prior_pairs_unchanged=$true;frozen_records=$taskFrozenRecords;
 previous_preflight=(Record $taskPriorPath);historical_preservation_reused=$taskPrior.historical_preservation_reused;
 broad_scan_repeated=$false;policy='Bind prior complete 77831-file scan; wrapper independently protects all preexisting integration output/results. Recheck broad two-repository manifest at search closure.';
 git_state=$taskGit;new_encoder_calls=0;new_head_fits=0;new_scaler_fits=0
}
$taskStream=[IO.File]::Open($taskOut,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
try{$taskWriter.WriteLine(($taskValue|ConvertTo-Json -Depth 30))}finally{$taskWriter.Dispose()}
Write-Output ('CONDITIONED_PREFLIGHT_PASSED checkpoints='+$taskCheckpointRecords.Count+' portable_pairs='+$taskPrior.portable_pairs.Count)

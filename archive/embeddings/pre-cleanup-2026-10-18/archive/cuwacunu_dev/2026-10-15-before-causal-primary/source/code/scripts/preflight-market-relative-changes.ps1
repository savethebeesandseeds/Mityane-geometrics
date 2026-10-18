# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host metadata/byte inspection only; no model or tensor execution.
$ErrorActionPreference='Stop'
$taskRoot='C:/Work/cuwacunu/cuwacunu_dev'
$taskPortable='C:/Work/cuwacunu/cuwacunu_embedding'
$taskDir=Join-Path $taskRoot 'doc/results/market_relative_changes_v1'
$taskOut=Join-Path $taskDir 'preflight.json'
New-Item -ItemType Directory -Force -Path $taskDir | Out-Null
if(Test-Path -LiteralPath $taskOut){throw 'Preserve existing preflight'}
function Record([string]$taskPath){
 $taskItem=Get-Item -LiteralPath $taskPath
 return [ordered]@{sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $taskPath).Hash.ToLowerInvariant();bytes=$taskItem.Length}
}
function Verify([string]$taskPath,$taskPin){
 $taskActual=Record $taskPath
 if($taskActual.sha256 -ne $taskPin.sha256 -or $taskActual.bytes -ne $taskPin.bytes){throw ('Protected mismatch: '+$taskPath)}
}
$taskLedgerPath=Join-Path $taskRoot 'doc/encoder_search_ledger.json'
$taskLedger=Get-Content -Raw -LiteralPath $taskLedgerPath|ConvertFrom-Json
if($taskLedger.candidate_entries.Count -ne 8){throw 'Expected eight preserved measured candidates'}
$taskCheckpoints=@()
foreach($taskCandidate in $taskLedger.candidate_entries){
 foreach($taskPin in $taskCandidate.checkpoints){
  Verify (Join-Path $taskRoot $taskPin.path) $taskPin
  $taskCheckpoints += [ordered]@{path=$taskPin.path;sha256=$taskPin.sha256;bytes=$taskPin.bytes;git_ignored=$true;deserialized=$false}
 }
 foreach($taskField in @('card','report','completion','own_frozen_result_table')){
  $taskPin=$taskCandidate.$taskField
  if($null -ne $taskPin){Verify (Join-Path $taskRoot $taskPin.path) $taskPin}
 }
}
if($taskCheckpoints.Count -ne 40){throw 'Expected40 checkpoint pins'}
$taskIgnored=@($taskCheckpoints.path | & git -c safe.directory=C:/Work/cuwacunu/cuwacunu_dev -C $taskRoot check-ignore --stdin)
if($LASTEXITCODE -ne 0 -or $taskIgnored.Count -ne 40){throw 'Checkpoint ignore check failed'}
$taskSyncPath=Join-Path $taskRoot 'doc/results/market_parent_conditional_changes_v1/portable-sync.json'
$taskSync=Get-Content -Raw -LiteralPath $taskSyncPath|ConvertFrom-Json
if($taskSync.status -ne 'passed' -or $taskSync.total_pairs -ne 73){throw 'Expected73 prior portable pairs'}
foreach($taskPair in $taskSync.pairs){
 Verify (Join-Path $taskRoot $taskPair.integration_path) $taskPair
 Verify (Join-Path $taskPortable $taskPair.embedding_path) $taskPair
}
$taskPins=@{
 'doc/MARKET_RELATIVE_CHANGE_CARD.md'='f4378d4a76185fc17b6c0adbadab5af27accd772ed66e5a8bc628ab7848f94a2';
 'doc/MARKET_PARENT_CONDITIONAL_CHANGE_RESULTS.md'='cb2d4548a8d35cf47b258e397161f50e1db04e224ae1b831309e6dbb9730ecb7';
 'doc/results/market_parent_conditional_changes_v1/completion.json'='ea8f754edfaf105de64dc55b8064606c497079f0dd9fdbb726b70c1b2a2f0076';
 'doc/results/market_parent_conditional_changes_v1/table_rows.json'='a0e6e91f577d377006201c9212025dda90a6e3553803e27f63334fa341f8beb0';
 'doc/results/market_parent_conditional_changes_v1/report-verification.json'='fe2b980a2b66453282b0c2a5267a371e98fee2af96a61ef596922695349b288d';
 'doc/LATENT_MARKET_DYNAMICS_ROADMAP.md'='56b0c0e284dae2c93318abe89c74be685e8034ee7a22a4e5325020124fbca617'
}
$taskFrozen=@()
foreach($taskName in ($taskPins.Keys|Sort-Object)){
 $taskActual=Record (Join-Path $taskRoot $taskName)
 if($taskActual.sha256 -ne $taskPins[$taskName]){throw ('Frozen pin changed: '+$taskName)}
 $taskFrozen += [ordered]@{path=$taskName;sha256=$taskActual.sha256;bytes=$taskActual.bytes}
}
$taskPriorPath=Join-Path $taskRoot 'doc/results/market_parent_conditional_changes_v1/preflight.json'
$taskPrior=Get-Content -Raw -LiteralPath $taskPriorPath|ConvertFrom-Json
$taskVerificationPath=Join-Path $taskRoot 'output/runs/market-parent-conditional-changes/run-1KSQSS/verification.json'
$taskVerification=Get-Content -Raw -LiteralPath $taskVerificationPath|ConvertFrom-Json
if($taskVerification.status -ne 'passed' -or $taskVerification.independent_arithmetic_checks -ne 2716060){throw 'Preceding protection incomplete'}
$taskResult=[ordered]@{kind='market_relative_changes_preflight_v1';status='passed';date='2026-10-15';
 scope='SHA256/size/availability/Git-ignore only;0 tensor deserialization, project execution, model calls or fits';
 card_sha256=$taskPins['doc/MARKET_RELATIVE_CHANGE_CARD.md'];primary='RPB-v22';balanced_reference='RPB-v24.market-01.alt-01';
 checkpoints_verified=40;checkpoints=$taskCheckpoints;all_checkpoints_available_hash_matched_and_ignored=$true;
 prior_portable_pairs_verified=73;portable_pairs=$taskSync.pairs;all_prior_pairs_unchanged=$true;
 ledger_before=(Record $taskLedgerPath);prior_portable_receipt=(Record $taskSyncPath);frozen_records=$taskFrozen;
 previous_preflight=(Record $taskPriorPath);previous_verified_run=(Record $taskVerificationPath);
 previous_integration_files_unchanged=$taskVerification.historical_integration_files_unchanged;
 historical_preservation_reused=$taskPrior.historical_preservation_reused;broad_scan_repeated=$false;
 new_calibration_encoder_calls=0;new_encoder_updates=0;new_head_fits=0;
 policy='Wrapper protects all preexisting integration output/results; original77831-file scan bound, broad two-repository scan repeated at goal closure.'}
$taskStream=[IO.File]::Open($taskOut,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
try{$taskWriter.WriteLine(($taskResult|ConvertTo-Json -Depth 30))}finally{$taskWriter.Dispose()}
foreach($taskName in @('encoder_search_ledger.json','ENCODER_SEARCH_LEDGER.md')){
 Copy-Item -LiteralPath (Join-Path $taskRoot ('doc/'+$taskName)) -Destination (Join-Path $taskDir ('search-ledger-before'+[IO.Path]::GetExtension($taskName)))
}
Write-Output 'RELATIVE_PREFLIGHT_PASSED checkpoints=40 prior_pairs=73'

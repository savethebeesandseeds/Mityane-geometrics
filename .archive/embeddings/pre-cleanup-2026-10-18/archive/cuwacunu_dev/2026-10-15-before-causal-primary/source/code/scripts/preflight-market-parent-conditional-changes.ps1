# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host metadata and byte inspection only; no project code or tensor execution.
$ErrorActionPreference='Stop'
$taskRoot='C:/Work/cuwacunu/cuwacunu_dev'
$taskPortable='C:/Work/cuwacunu/cuwacunu_embedding'
$taskOut=Join-Path $taskRoot 'doc/results/market_parent_conditional_changes_v1/preflight.json'
if(Test-Path -LiteralPath $taskOut){throw 'Preserve existing preflight'}
function Record([string]$taskPath){
 $taskItem=Get-Item -LiteralPath $taskPath
 return [ordered]@{sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $taskPath).Hash.ToLowerInvariant();bytes=$taskItem.Length}
}
function Verify([string]$taskPath,$taskPin){
 $taskActual=Record $taskPath
 if($taskActual.sha256 -ne $taskPin.sha256 -or $taskActual.bytes -ne $taskPin.bytes){throw ('Protected mismatch: '+$taskPath)}
}
$taskLedger=Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'doc/encoder_search_ledger.json')|ConvertFrom-Json
if($taskLedger.candidate_entries.Count -ne 7){throw 'Expected seven preserved measured candidates'}
$taskCheckpoints=@()
foreach($taskCandidate in $taskLedger.candidate_entries){
 foreach($taskPin in $taskCandidate.checkpoints){
  Verify (Join-Path $taskRoot $taskPin.path) $taskPin
  $taskCheckpoints += [ordered]@{path=$taskPin.path;sha256=$taskPin.sha256;bytes=$taskPin.bytes;git_ignored=$true;deserialized=$false}
 }
 foreach($taskField in @('card','report','completion')){
  $taskPin=$taskCandidate.$taskField
  if($null -ne $taskPin){Verify (Join-Path $taskRoot $taskPin.path) $taskPin}
 }
}
if($taskCheckpoints.Count -ne 35){throw 'Expected35 checkpoint pins'}
$taskIgnored=@($taskCheckpoints.path | & git -c safe.directory=C:/Work/cuwacunu/cuwacunu_dev -C $taskRoot check-ignore --stdin)
if($LASTEXITCODE -ne 0 -or $taskIgnored.Count -ne 35){throw 'Checkpoint ignore check failed'}
$taskSyncPath=Join-Path $taskRoot 'doc/results/market_conditioned_changes_v1/portable-sync.json'
$taskSync=Get-Content -Raw -LiteralPath $taskSyncPath|ConvertFrom-Json
if($taskSync.status -ne 'passed' -or $taskSync.total_pairs -ne 64){throw 'Expected64 prior portable pairs'}
foreach($taskPair in $taskSync.pairs){
 Verify (Join-Path $taskRoot $taskPair.integration_path) $taskPair
 Verify (Join-Path $taskPortable $taskPair.embedding_path) $taskPair
}
$taskPins=@{
 'doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md'='6e24e48bfdd05148f99446cf1421a713a7736d522f6c00159ba156e43b39c29b';
 'doc/MARKET_CONDITIONED_CHANGE_RESULTS.md'='89e959113fb3b38958ebb01b17d134b3421a43246058986f0aed0f9f103edd54';
 'doc/results/market_conditioned_changes_v1/completion.json'='884d119ad5bfdfd878434be782b7607c6f5486292ed63b0fba695cf05e5bea74';
 'doc/results/market_conditioned_changes_v1/table_rows.json'='56da31fe5256dca8c59ca3816bb9ff32e2a07a9075e353bc2d830cece11d678f';
 'doc/results/market_conditioned_changes_v1/report-verification.json'='09624b637eec6abcf411b4d6ddc994b5aa057a8fe4c51ed715ee4212d703d6cb';
 'output/runs/market-conditioned-changes/run-RjTTZZ/bank-standardizer.pt'='15e11373b14fe438fe920ccf13a72b86aa4719ea1c17dee8422249db1c185535';
 'output/runs/market-conditioned-changes/run-RjTTZZ/calibration-bank.pt'='e375dbbfc29cbb6a0d70def2de42d83a68bc5d376a8cef6daf88edbfd7d73c08';
 'output/runs/market-conditioned-changes/run-RjTTZZ/calibration-witness.json'='add7f7ed8826fdb7ef89b68155bfc25c570f1827c76aead07bc71e71ab523e52';
 'doc/LATENT_MARKET_DYNAMICS_ROADMAP.md'='56b0c0e284dae2c93318abe89c74be685e8034ee7a22a4e5325020124fbca617'
}
$taskFrozen=@()
foreach($taskName in ($taskPins.Keys|Sort-Object)){
 $taskActual=Record (Join-Path $taskRoot $taskName)
 if($taskActual.sha256 -ne $taskPins[$taskName]){throw ('Frozen pin changed: '+$taskName)}
 $taskFrozen += [ordered]@{path=$taskName;sha256=$taskActual.sha256;bytes=$taskActual.bytes}
}
$taskPrior=Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'doc/results/market_conditioned_changes_v1/preflight.json')|ConvertFrom-Json
$taskVerificationPath=Join-Path $taskRoot 'output/runs/market-conditioned-changes/run-RjTTZZ/verification.json'
$taskVerification=Get-Content -Raw -LiteralPath $taskVerificationPath|ConvertFrom-Json
if($taskVerification.status -ne 'passed' -or $taskVerification.historical_integration_files_unchanged -ne 1598){throw 'Preceding protection incomplete'}
$taskResult=[ordered]@{kind='market_parent_conditional_changes_preflight_v1';status='passed';date='2026-10-15';
 scope='SHA256/size/availability/Git-ignore only;0 tensor deserialization, project execution, model calls, fits or calibration forwards';
 card_sha256=$taskPins['doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md'];primary='RPB-v22';balanced_reference='RPB-v24.market-01.alt-01';
 checkpoints_verified=35;checkpoints=$taskCheckpoints;all_checkpoints_available_hash_matched_and_ignored=$true;
 prior_portable_pairs_verified=64;portable_pairs=$taskSync.pairs;all_prior_pairs_unchanged=$true;
 prior_portable_receipt=(Record $taskSyncPath);frozen_records=$taskFrozen;
 previous_verified_run=(Record $taskVerificationPath);previous_integration_files_unchanged=1598;
 historical_preservation_reused=$taskPrior.historical_preservation_reused;broad_scan_repeated=$false;
 new_calibration_encoder_calls=0;new_encoder_updates=0;new_head_fits=0;
 policy='New wrapper must protect all preexisting integration output/results; bind original77831-file scan honestly and recheck it at goal closure. New needed calibration forwards occur only after artificial GPU admission.'}
$taskStream=[IO.File]::Open($taskOut,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
try{$taskWriter.WriteLine(($taskResult|ConvertTo-Json -Depth 30))}finally{$taskWriter.Dispose()}
Write-Output 'PARENT_CONDITIONAL_PREFLIGHT_PASSED checkpoints=35 prior_pairs=64'

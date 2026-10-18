# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host metadata/byte inspection only; project execution remains in cuwacunu.
$ErrorActionPreference='Stop'
$taskRoot='C:/Work/cuwacunu/cuwacunu_dev'
$taskPortable='C:/Work/cuwacunu/cuwacunu_embedding'
$taskDir=Join-Path $taskRoot 'doc/results/market_location_scale_decoder_v1'
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
if($taskLedger.candidate_entries.Count -ne 10 -or (Record $taskLedgerPath).sha256 -ne 'bf0087116423b018ee2b11b042a9363939d492ae346034c51a2a97d3e03224a4'){throw 'Expected ten preserved candidates and sealed active declaration'}
foreach($taskName in @('encoder_search_ledger.json','ENCODER_SEARCH_LEDGER.md')){
 Verify (Join-Path $taskPortable ('doc/'+$taskName)) (Record (Join-Path $taskRoot ('doc/'+$taskName)))
}
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
if($taskCheckpoints.Count -ne 50){throw 'Expected50 checkpoint pins'}
$taskIgnored=@($taskCheckpoints.path | & git -c safe.directory=C:/Work/cuwacunu/cuwacunu_dev -C $taskRoot check-ignore --stdin)
if($LASTEXITCODE -ne 0 -or $taskIgnored.Count -ne 50){throw 'Checkpoint ignore check failed'}
$taskSyncPath=Join-Path $taskRoot 'doc/results/market_parent_modulated_decoder_v1/portable-sync.json'
$taskSync=Get-Content -Raw -LiteralPath $taskSyncPath|ConvertFrom-Json
if($taskSync.status -ne 'passed' -or $taskSync.total_pairs -ne 91){throw 'Expected91 prior portable pairs'}
foreach($taskPair in $taskSync.pairs){
 Verify (Join-Path $taskRoot $taskPair.integration_path) $taskPair
 Verify (Join-Path $taskPortable $taskPair.embedding_path) $taskPair
}
$taskPins=@{
 'doc/MARKET_LOCATION_SCALE_DECODER_CARD.md'='f620298e9ecf9f73a32ae61737b67295f8dcde709416aa3d6ed16feee613e45c';
 'doc/MARKET_PARENT_MODULATED_DECODER_RESULTS.md'='554d1d839eb08c0221d3e93134192a873d8f46d4df994695890c5d822c1f65b3';
 'doc/results/market_parent_modulated_decoder_v1/completion.json'='d4d4e8dbb0bd39ca7431d6aceb7d47bc92263621c8509bbe3faf2cab6cfc457c';
 'doc/results/market_parent_modulated_decoder_v1/table_rows.json'='a3c6bc69b9a0c1462d2d667fb2df215a486036283b900c1b8bf234a5eae5770d';
 'doc/results/market_parent_modulated_decoder_v1/report-verification.json'='a0beacf5d6726631d5e8b50eeb551056a2fad3a38104928df3405da98ecf3e21';
 'output/runs/market-relative-changes/run-vaPC4X/bank-standardizer.pt'='c3e4f9e850ebbb355ae58e8b434173e4db0d1ff98055c844af4fe16ba7879fbc';
 'output/runs/market-relative-changes/run-vaPC4X/bank-standardizer.json'='782128503036956beb1ec87e39ceb3c502395f00f1d0e8a6138e9d8e52d933ee';
 'output/runs/market-relative-changes/run-vaPC4X/features/initialized_v28-train.pt'='953c9967515f2de6c709e1651fab7a0b464410f1f62bd25e06c2a534d72cc593';
 'output/runs/market-relative-changes/run-vaPC4X/features/initialized_v28-validation.pt'='a774470823dc2d0aad0e5f622fee4acd52383f05aabf11e193071d88f5b8aea0';
 'output/runs/market-relative-changes/run-vaPC4X/readouts/report.json'='0f9199d88e7d3a90039d8a555873fb74399b8b79f6505798a58c3fcc5bae4156';
 'output/runs/market-relative-changes/run-vaPC4X/initial-query.json'='920b060da73114c5e45a84fe158054bc01b8adb49a0f94f67f1a7430f3597717';
 'output/runs/market-relative-changes/run-vaPC4X/independent-audit.json'='b8278365f60f95eb1a611b8ed34fe4a144a77be59de686e1891eda9e30b25d97';
 'output/runs/market-relative-changes/run-vaPC4X/raw-relative-bank-witness.json'='8abc4845192236def58b2727b02d60e816ecca21c501c4f742a1776b23273fa8';
 'output/runs/market-relative-changes/run-vaPC4X/raw-loader-record-witness.json'='804606752fd991e693cc558808dfab53ebd5a741891559e1ecc41dabdf4969cc';
 'doc/LATENT_MARKET_DYNAMICS_ROADMAP.md'='56b0c0e284dae2c93318abe89c74be685e8034ee7a22a4e5325020124fbca617'
}
$taskFrozen=@()
foreach($taskName in ($taskPins.Keys|Sort-Object)){
 $taskActual=Record (Join-Path $taskRoot $taskName)
 if($taskActual.sha256 -ne $taskPins[$taskName]){throw ('Frozen pin changed: '+$taskName)}
 $taskFrozen += [ordered]@{path=$taskName;sha256=$taskActual.sha256;bytes=$taskActual.bytes}
}
$taskPriorPath=Join-Path $taskRoot 'doc/results/market_parent_modulated_decoder_v1/preflight.json'
$taskPrior=Get-Content -Raw -LiteralPath $taskPriorPath|ConvertFrom-Json
$taskVerificationPath=Join-Path $taskRoot 'output/runs/market-parent-modulated-decoder/run-p4XMUY/verification.json'
$taskVerification=Get-Content -Raw -LiteralPath $taskVerificationPath|ConvertFrom-Json
if($taskVerification.status -ne 'passed' -or $taskVerification.independent_arithmetic_checks -ne 567804){throw 'Preceding protection incomplete'}
$taskResult=[ordered]@{kind='market_location_scale_decoder_preflight_v1';status='passed';date='2026-10-15';
 scope='SHA256/size/availability/Git-ignore only;0 tensor deserialization, project execution, model calls or fits';
 card_sha256=$taskPins['doc/MARKET_LOCATION_SCALE_DECODER_CARD.md'];primary='RPB-v22';balanced_reference='RPB-v24.market-01.alt-01';
 checkpoints_verified=50;checkpoints=$taskCheckpoints;all_checkpoints_available_hash_matched_and_ignored=$true;
 prior_portable_pairs_verified=91;portable_pairs=$taskSync.pairs;all_prior_pairs_unchanged=$true;
 ledger_before=(Record $taskLedgerPath);prior_portable_receipt=(Record $taskSyncPath);frozen_records=$taskFrozen;
 previous_preflight=(Record $taskPriorPath);previous_verified_run=(Record $taskVerificationPath);
 previous_integration_files_unchanged=$taskVerification.historical_integration_files_unchanged;
 historical_preservation_reused=$taskPrior.historical_preservation_reused;broad_scan_repeated=$false;
 new_calibration_encoder_calls=0;new_bank_normalizer_fits=0;new_encoder_updates=0;new_head_fits=0;
 policy='Reuse V28 bank calibration and initial scoring only after exact new full36 parity. Wrapper protects all preexisting integration output/results; original77831-file scan bound, broad two-repository scan repeated at goal closure.'}
$taskStream=[IO.File]::Open($taskOut,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
try{$taskWriter.WriteLine(($taskResult|ConvertTo-Json -Depth 30))}finally{$taskWriter.Dispose()}
foreach($taskName in @('encoder_search_ledger.json','ENCODER_SEARCH_LEDGER.md')){
 $taskSnapshot=Join-Path $taskDir ('search-ledger-active-before'+[IO.Path]::GetExtension($taskName))
 if(Test-Path -LiteralPath $taskSnapshot){throw 'Preserve active ledger snapshot'}
 Copy-Item -LiteralPath (Join-Path $taskRoot ('doc/'+$taskName)) -Destination $taskSnapshot
}
Write-Output 'LOCATION_SCALE_PREFLIGHT_PASSED checkpoints=50 prior_pairs=91'

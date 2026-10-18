# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host file/Git inspection only. No project/model/numerical execution.
$ErrorActionPreference='Stop'
$taskRoot='C:/Work/cuwacunu/cuwacunu_dev'
$taskPortable='C:/Work/cuwacunu/cuwacunu_embedding'
$taskDir=Join-Path $taskRoot 'doc/results/market_location_scale_synthetic_transfer_v1'
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
if($taskLedger.candidate_entries.Count -ne 11 -or (Record $taskLedgerPath).sha256 -ne 'b8d3ab33dae4937a0202ac64302f58a5b6c803c024b7277604f26855c76a4012'){throw 'Expected11 preserved candidates'}
if($taskLedger.live.primary_architecture -ne 'RPB-v22' -or $taskLedger.live.leading_balanced_research_reference -ne 'RPB-v24.market-01.alt-01'){throw 'Primary/balanced pointer changed before transfer'}
foreach($taskName in @('encoder_search_ledger.json','ENCODER_SEARCH_LEDGER.md')){
 Verify (Join-Path $taskPortable ('doc/'+$taskName)) (Record (Join-Path $taskRoot ('doc/'+$taskName)))
}
$taskCheckpoints=@();$taskFrozen=@()
foreach($taskCandidate in $taskLedger.candidate_entries){
 foreach($taskPin in $taskCandidate.checkpoints){
  Verify (Join-Path $taskRoot $taskPin.path) $taskPin
  $taskCheckpoints += [ordered]@{path=$taskPin.path;sha256=$taskPin.sha256;bytes=$taskPin.bytes;git_ignored=$true;deserialized=$false}
 }
 foreach($taskField in @('card','report','completion','own_frozen_result_table')){
  $taskPin=$taskCandidate.$taskField
  if($null -ne $taskPin){Verify (Join-Path $taskRoot $taskPin.path) $taskPin;$taskFrozen+=$taskPin}
 }
}
if($taskCheckpoints.Count -ne 55){throw 'Expected55 checkpoint pins'}
$taskIgnored=@($taskCheckpoints.path | & git -c safe.directory=C:/Work/cuwacunu/cuwacunu_dev -C $taskRoot check-ignore --stdin)
if($LASTEXITCODE -ne 0 -or $taskIgnored.Count -ne 55){throw 'Checkpoint ignore check failed'}
# check-ignore above includes only untracked ignored paths (no --no-index),
# so the same query also proves the checkpoint paths are not Git tracked.
$taskSyncPath=Join-Path $taskRoot 'doc/results/market_location_scale_decoder_v1/portable-sync.json'
$taskSync=Get-Content -Raw -LiteralPath $taskSyncPath|ConvertFrom-Json
if($taskSync.status -ne 'passed' -or $taskSync.total_pairs -ne 99){throw 'Expected99 portable pairs'}
foreach($taskPair in $taskSync.pairs){
 Verify (Join-Path $taskRoot $taskPair.integration_path) $taskPair
 Verify (Join-Path $taskPortable $taskPair.embedding_path) $taskPair
}
$taskManifestPath=Join-Path $taskRoot 'output/inputs/market-synthetic-transfer-v1/manifest.json'
if((Record $taskManifestPath).sha256 -ne 'fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb'){throw 'Staged manifest changed'}
$taskManifest=Get-Content -Raw -LiteralPath $taskManifestPath|ConvertFrom-Json
if($taskManifest.members.Count -ne 18){throw 'Expected18 staged members'}
foreach($taskPin in $taskManifest.members){Verify (Join-Path (Split-Path $taskManifestPath) $taskPin.path) $taskPin}
$taskPins=@{
 'doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md'='c3cfe0a04b174734599b2b01363c90c3f9b81fd5964fb30ffd4ce8a21976260e';
 'config/market_studies/encoder_location_scale_decoder.conf'='43320396447ca80be5fde4324d17c3a9ab129e116bf184455fcb6261aeb6923c';
 'doc/MARKET_SYNTHETIC_TRANSFER_CARD.md'='b0552d60cd50582a3bff213a2778e867ceabe1e20e52cd48e02aec4c34bea82b';
 'doc/MARKET_LOCATION_SCALE_DECODER_RESULTS.md'='f5519bc661fb5072a74f36ebe4258c25862ebfb5c13117056ab000533a296ff8';
 'doc/results/market_location_scale_decoder_v1/completion.json'='0b91cded8a9f6cd9829886724e240eb2ce770a6fd60c31ec04d5db8ae59d7602';
 'doc/results/market_location_scale_decoder_v1/report-verification.json'='a7510eb0fb8bd8f117347159a67aa64288efe8e29e3fd24e37b7ae78d2f4d517'
}
foreach($taskName in ($taskPins.Keys|Sort-Object)){
 $taskActual=Record (Join-Path $taskRoot $taskName)
 if($taskActual.sha256 -ne $taskPins[$taskName]){throw ('Frozen pin changed: '+$taskName)}
 $taskFrozen += [ordered]@{path=$taskName;sha256=$taskActual.sha256;bytes=$taskActual.bytes}
}
$taskResult=[ordered]@{kind='market_location_scale_synthetic_transfer_preflight_v1';status='passed';date='2026-10-15';
 scope='File SHA256/size/availability/Git-ignore only; no model calls, fits or tensor deserialization';
 checkpoints_verified=55;checkpoints=$taskCheckpoints;all_checkpoints_available_hash_matched_and_ignored=$true;
 portable_pairs_verified=99;portable_pairs=$taskSync.pairs;all_portable_pairs_match=$true;
 ledger_before=(Record $taskLedgerPath);portable_receipt=(Record $taskSyncPath);frozen_records=$taskFrozen;
 staged_manifest=(Record $taskManifestPath);staged_members=$taskManifest.members;
 primary='RPB-v22';balanced_reference='RPB-v24.market-01.alt-01';quality_started=$false;
 encoder_updates=0;head_fits=0;historical_broad_scan='Original77831-file two-repository baseline retained; fresh closure receipt required';
 unrelated_roadmap_observation=(Record (Join-Path $taskRoot 'doc/LATENT_MARKET_DYNAMICS_ROADMAP.md'));
 unrelated_roadmap_policy='Outside experiment inputs; externally changed since market run, preserved without mutation; see preflight-attempt-01.json'}
$taskStream=[IO.File]::Open($taskOut,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
try{$taskWriter.WriteLine(($taskResult|ConvertTo-Json -Depth 30))}finally{$taskWriter.Dispose()}
foreach($taskName in @('encoder_search_ledger.json','ENCODER_SEARCH_LEDGER.md')){
 $taskSnapshot=Join-Path $taskDir ('search-ledger-before'+[IO.Path]::GetExtension($taskName))
 if(Test-Path -LiteralPath $taskSnapshot){throw 'Preserve ledger snapshot'}
 Copy-Item -LiteralPath (Join-Path $taskRoot ('doc/'+$taskName)) -Destination $taskSnapshot
}
Write-Output 'LOCATION_SCALE_TRANSFER_PREFLIGHT_PASSED checkpoints=55 portable_pairs=99 staged_members=18'

# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host file inspection and document snapshotting only; no project execution.
$ErrorActionPreference='Stop'
$taskIntegration='C:/Work/cuwacunu/cuwacunu_dev'
$taskEmbedding='C:/Work/cuwacunu/cuwacunu_embedding'
$taskEvidence=Join-Path $taskIntegration 'doc/results/market_change_energy_v1'
function Hash([string]$taskPath) {(Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()}
function Record([string]$taskPath) {[ordered]@{path=$taskPath;sha256=(Hash $taskPath);bytes=(Get-Item -LiteralPath $taskPath).Length}}
$taskDestination=Join-Path $taskEvidence 'final-verification.json'
$taskSnapshot=Join-Path $taskEvidence 'live-final'
if ((Test-Path -LiteralPath $taskDestination) -or (Test-Path -LiteralPath $taskSnapshot)) {throw 'Preserve existing final verification'}
$taskAfterPath=Join-Path $taskEvidence 'preservation-after-summary.json'
$taskAfter=Get-Content -LiteralPath $taskAfterPath -Raw | ConvertFrom-Json
if ($taskAfter.status -ne 'passed' -or (@($taskAfter.roots.verified_files | Measure-Object -Sum).Sum) -ne 77831) {throw 'Historical preservation incomplete'}
$taskCompletionPath=Join-Path $taskEvidence 'completion.json'
$taskCompletion=Get-Content -LiteralPath $taskCompletionPath -Raw | ConvertFrom-Json
if ($taskCompletion.full_market_gate_passed -or -not $taskCompletion.stopped_without_rescue -or $taskCompletion.synthetic_transfer_measured -or $taskCompletion.primary_changed) {throw 'Unexpected experiment disposition'}
if ($taskCompletion.original_criteria_passed -ne 7 -or $taskCompletion.strict_criteria_passed -ne 6) {throw 'Unexpected criteria'}
$taskFrozen=@()
foreach ($taskBinding in @(@{path='doc/MARKET_CHANGE_ENERGY_RESULTS.md';hash=$taskCompletion.report_sha256},@{path='doc/results/market_change_energy_v1/table_rows.json';hash=$taskCompletion.table_rows_sha256},@{path='doc/results/market_change_energy_v1/asset_hashes.json';hash=$taskCompletion.asset_hashes_sha256})) {
  $taskPath=Join-Path $taskIntegration $taskBinding.path
  if ((Hash $taskPath) -ne $taskBinding.hash) {throw 'Frozen result changed'}
  $taskFrozen+=Record $taskPath
}
$taskReportAuditPath=Join-Path $taskEvidence 'report-verification.json'
$taskReportAudit=Get-Content -LiteralPath $taskReportAuditPath -Raw | ConvertFrom-Json
if ($taskReportAudit.status -ne 'passed' -or (Hash $taskReportAuditPath) -ne 'eb778a021a2a01189019511763aa335f1064d0ea20b5bd33d8938360209fd8ec') {throw 'Saved evidence audit missing or changed'}
$taskPortablePath=Join-Path $taskEvidence 'portable-sync-final.json'
$taskPortable=Get-Content -LiteralPath $taskPortablePath -Raw | ConvertFrom-Json
if ($taskPortable.portable_pairs.Count -ne 54) {throw 'Wrong portable pair inventory'}
foreach ($taskPair in $taskPortable.portable_pairs) {
  foreach ($taskPath in @((Join-Path $taskIntegration $taskPair.integration_path),(Join-Path $taskEmbedding $taskPair.embedding_path))) {
    if ((Hash $taskPath) -ne $taskPair.expected_sha256 -or (Get-Item -LiteralPath $taskPath).Length -ne $taskPair.bytes) {throw ('Portable file changed: '+$taskPath)}
  }
}
$taskCheckpoints=@()
foreach ($taskCheckpoint in $taskCompletion.checkpoint_assets) {
  $taskPath=Join-Path $taskIntegration $taskCheckpoint.source
  if ((Hash $taskPath) -ne $taskCheckpoint.sha256 -or (Get-Item -LiteralPath $taskPath).Length -ne $taskCheckpoint.bytes) {throw 'New checkpoint changed or unavailable'}
  & git -c safe.directory=C:/Work/cuwacunu/cuwacunu_dev -C $taskIntegration check-ignore --quiet -- $taskCheckpoint.source
  if ($LASTEXITCODE -ne 0) {throw 'New checkpoint is not ignored'}
  $taskCheckpoints+=Record $taskPath
}
if ($taskCheckpoints.Count -ne 5) {throw 'Five new immutable checkpoints required'}
$taskOld=Get-Content -LiteralPath (Join-Path $taskEvidence 'prequality-checkpoints.json') -Raw | ConvertFrom-Json
foreach ($taskCheckpoint in $taskOld.checkpoints) {if ((Hash (Join-Path $taskIntegration $taskCheckpoint.path)) -ne $taskCheckpoint.sha256) {throw 'Historical handoff checkpoint changed'}}
$taskRegistryPath=Join-Path $taskEmbedding 'doc/change_energy_embedding_versions.json'
$taskRegistry=Get-Content -LiteralPath $taskRegistryPath -Raw | ConvertFrom-Json
if ($taskRegistry.report_sha256 -ne $taskCompletion.report_sha256 -or $taskRegistry.completion_sha256 -ne (Hash $taskCompletionPath)) {throw 'Embedding registry binding mismatch'}
$taskHeads=@()
foreach ($taskSpec in @(@{name='integration';root=$taskIntegration;head='13a50e74d84b1ee2c75c1baa1a131dc47f6ec6cc'},@{name='embedding';root=$taskEmbedding;head='93d19dc17d6f87687ae65c589604843acd898d05'})) {
  $taskGitHead=(& git -c ('safe.directory='+$taskSpec.root) -C $taskSpec.root rev-parse HEAD).Trim()
  if ($LASTEXITCODE -ne 0 -or $taskGitHead -ne $taskSpec.head) {throw 'Unexpected new commit'}
  $taskHeads+=[ordered]@{repository=$taskSpec.name;head=$taskGitHead;new_commit=$false}
}
$taskDocuments=@()
foreach ($taskSpec in @(@{name='integration';root=$taskIntegration;files=@('AGENTS.md','Makefile','.gitattributes','doc/CONTINUATION.md','doc/ENCODER_BACKPORT.md','doc/MARKET_CHANGE_ENERGY_FOLLOWUP.md')},@{name='embedding';root=$taskEmbedding;files=@('AGENTS.md','Makefile','.gitattributes','doc/CURRENT_STATUS.md','doc/EMBEDDING_VERSIONS.md','doc/CHANGE_ENERGY_V25.md','doc/change_energy_embedding_versions.json')})) {
  foreach ($taskRelative in $taskSpec.files) {
    $taskSource=Join-Path $taskSpec.root $taskRelative
    $taskTarget=Join-Path (Join-Path $taskSnapshot $taskSpec.name) $taskRelative
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($taskTarget)) -Force | Out-Null
    Copy-Item -LiteralPath $taskSource -Destination $taskTarget
    if ((Hash $taskSource) -ne (Hash $taskTarget)) {throw 'Live document snapshot differs'}
    $taskDocuments+=[ordered]@{source=(Record $taskSource);snapshot=(Record $taskTarget)}
  }
}
if (@(Get-ChildItem -LiteralPath $taskEvidence -Recurse -File -Filter '*.pt').Count -ne 0) {throw 'Checkpoint tensors must not enter durable docs'}
$taskResult=[ordered]@{kind='market_change_energy_final_verification_v1';date='2026-10-15';status='passed';instance='RPB-v25.market-01';historical_files_unchanged=77831;historical_checkpoint_pins_unchanged=9;portable_pairs_unchanged=54;new_checkpoints_available_and_ignored=5;original_conditions_passed=7;strict_conditions_passed=6;stopped_without_rescue=$true;synthetic_transfer_measured=$false;primary='RPB-v22';leading_balanced_reference='RPB-v24.market-01.alt-01';new_commits_or_pushes=$false;git_heads=$taskHeads;completion=(Record $taskCompletionPath);report_verification=(Record $taskReportAuditPath);preservation=(Record $taskAfterPath);portable_sync=(Record $taskPortablePath);frozen_results=$taskFrozen;checkpoints=$taskCheckpoints;live_documents_and_snapshots=$taskDocuments;host_orchestrators=@((Record (Join-Path $taskIntegration 'code/scripts/preserve-market-energy.ps1')),(Record (Join-Path $taskIntegration 'code/scripts/verify-market-energy-preservation.ps1')),(Record $PSCommandPath));next_direction='Separate saved-TRAIN centering/price-level/relative-change diagnosis; no new diagnosis or encoder trajectory executed';excluded_unrelated_file='doc/LATENT_MARKET_DYNAMICS_ROADMAP.md'}
$taskStream=[IO.File]::Open($taskDestination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
try {$taskWriter.WriteLine(($taskResult | ConvertTo-Json -Depth 12))} finally {$taskWriter.Dispose()}
Write-Output ('FINAL_VERIFICATION_PASSED='+$taskDestination)

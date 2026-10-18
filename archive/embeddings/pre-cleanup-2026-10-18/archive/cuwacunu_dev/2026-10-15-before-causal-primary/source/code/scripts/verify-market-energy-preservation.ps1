# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host filesystem inspection only. Every baseline byte is rehashed; no deletion.
param([Parameter(Mandatory=$true)][string]$ManifestDirectory)
$ErrorActionPreference='Stop'
$taskRoot='C:/Work/cuwacunu/cuwacunu_dev'
$taskAllowed=[IO.Path]::GetFullPath((Join-Path $taskRoot 'output/runs/market-change-energy'))+[IO.Path]::DirectorySeparatorChar
$taskDirectory=[IO.Path]::GetFullPath($ManifestDirectory)
if (-not $taskDirectory.StartsWith($taskAllowed,[StringComparison]::OrdinalIgnoreCase)) { throw 'Wrong manifest directory' }
$taskSummaryPath=Join-Path $taskDirectory 'before-summary.json'
$taskSummary=Get-Content -LiteralPath $taskSummaryPath -Raw | ConvertFrom-Json
if ($taskSummary.kind -ne 'market_change_energy_preservation_v1' -or $taskSummary.roots.Count -ne 2) { throw 'Wrong baseline' }
if ((@($taskSummary.roots.repository | Sort-Object -Unique) -join ',') -ne 'embedding,integration') {throw 'Both distinct repository roots required'}
if ([IO.Path]::GetFullPath($taskSummary.local_manifest_directory) -ne $taskDirectory) {throw 'Baseline directory mismatch'}
$taskDurableBefore=Join-Path $taskRoot 'doc/results/market_change_energy_v1/preservation-before-summary.json'
$taskBeforeHash=(Get-FileHash -LiteralPath $taskSummaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ((Get-FileHash -LiteralPath $taskDurableBefore -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskBeforeHash) {throw 'Durable baseline differs'}
$taskDestinations=@((Join-Path $taskDirectory 'after-summary.json'),(Join-Path $taskRoot 'doc/results/market_change_energy_v1/preservation-after-summary.json'))
foreach ($taskDestination in $taskDestinations) {if (Test-Path -LiteralPath $taskDestination) {throw 'Preserve existing after receipt'}}
$taskResults=@()
foreach ($taskSpec in $taskSummary.roots) {
  $taskExpected = if ($taskSpec.repository -eq 'integration') {'C:/Work/cuwacunu/cuwacunu_dev'} elseif ($taskSpec.repository -eq 'embedding') {'C:/Work/cuwacunu/cuwacunu_embedding'} else {throw 'Unknown repository'}
  if ([IO.Path]::GetFullPath($taskSpec.root) -ne [IO.Path]::GetFullPath($taskExpected)) {throw 'Unexpected baseline root'}
  if ([IO.Path]::GetFullPath($taskSpec.manifest) -ne [IO.Path]::GetFullPath((Join-Path $taskDirectory ($taskSpec.repository+'-before.jsonl')))) {throw 'Unexpected manifest path'}
  if ((Get-FileHash -LiteralPath $taskSpec.manifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskSpec.sha256) {throw 'Baseline manifest changed'}
  $taskRepositoryRoot=$taskSpec.root
  $taskRepositoryName=$taskSpec.repository
  $taskVerified=0
  $taskFailures=[Collections.Generic.List[string]]::new()
  # Four read-only workers reduce per-file overhead; all entries and bytes are
  # still verified. Deliberate dangling links are verified as link metadata.
  [IO.File]::ReadLines($taskSpec.manifest) | ForEach-Object -Parallel {
    $ErrorActionPreference='Stop'
    try {
      $taskEntry=$_ | ConvertFrom-Json
      $taskPrefix=[IO.Path]::GetFullPath($using:taskRepositoryRoot)+[IO.Path]::DirectorySeparatorChar
      $taskPath=[IO.Path]::GetFullPath((Join-Path $using:taskRepositoryRoot $taskEntry.path))
      if (-not $taskPath.StartsWith($taskPrefix,[StringComparison]::OrdinalIgnoreCase)) {throw 'Manifest path escaped repository'}
      $taskFile=Get-Item -LiteralPath $taskPath
      if (($taskFile.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        $taskDump=(& fsutil reparsepoint query $taskPath 2>&1) -join "`n"
        if ($LASTEXITCODE -ne 0) {throw 'Cannot inspect reparse point'}
        $taskHasher=[Security.Cryptography.SHA256]::Create()
        try {$taskHash=[Convert]::ToHexString($taskHasher.ComputeHash([Text.UTF8Encoding]::new($false).GetBytes($taskDump))).ToLowerInvariant()}
        finally {$taskHasher.Dispose()}
        $taskKind='reparse-point-metadata'
      } else {
        $taskHash=(Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $taskKind='regular-file'
      }
      if ($taskHash -ne $taskEntry.sha256 -or $taskFile.Length -ne $taskEntry.bytes -or $taskKind -ne $taskEntry.kind) {throw 'Historical content changed'}
      [pscustomobject]@{ok=$true;path=$taskEntry.path;error=$null}
    } catch {
      [pscustomobject]@{ok=$false;path=$taskEntry.path;error=$_.Exception.Message}
    }
  } -ThrottleLimit 4 | ForEach-Object {
    if ($_.ok) {$taskVerified++} else {$taskFailures.Add(($_.path+': '+$_.error))}
    if ($taskVerified -gt 0 -and $taskVerified % 5000 -eq 0) {Write-Output "PRESERVATION_PROGRESS=$taskRepositoryName verified=$taskVerified"}
  }
  if ($taskFailures.Count -gt 0) {throw ($taskFailures -join "`n")}
  if ($taskVerified -ne $taskSpec.files) {throw 'Baseline count differs'}
  $taskResults += [ordered]@{repository=$taskRepositoryName;verified_files=$taskVerified;status='unchanged';before_manifest_sha256=$taskSpec.sha256}
}
if ((Get-FileHash -LiteralPath $taskSummaryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskBeforeHash) {throw 'Baseline summary changed during verification'}
$taskResult=[ordered]@{kind='market_change_energy_preservation_after_v1';status='passed';roots=$taskResults;before_summary_sha256=$taskBeforeHash;verification='Every baseline file SHA256/size/kind; four read-only workers, additions permitted';verifier_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
foreach ($taskDestination in $taskDestinations) {
  $taskStream=[IO.File]::Open($taskDestination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
  $taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
  try {$taskWriter.WriteLine(($taskResult | ConvertTo-Json -Depth 8))} finally {$taskWriter.Dispose()}
}
Write-Output "PRESERVATION_AFTER_PASSED=$taskDirectory"

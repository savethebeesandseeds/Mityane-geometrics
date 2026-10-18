# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host filesystem inspection only. No project execution, deletion or old receipt edits.
[CmdletBinding()]
param(
  [string]$ManifestDirectory='C:/Work/cuwacunu/cuwacunu_dev/output/runs/market-change-energy/preservation-dbcc75af6a394f52a3b08a792a8ef622',
  [string]$ReceiptPath
)
$ErrorActionPreference='Stop'
if ($PSVersionTable.PSVersion.Major -lt 7) {throw 'PowerShell 7 is required for four read-only workers'}
$taskStarted=[DateTime]::UtcNow
$taskTimer=[Diagnostics.Stopwatch]::StartNew()
$taskRoot='C:/Work/cuwacunu/cuwacunu_dev'
$taskEvidence=[IO.Path]::GetFullPath((Join-Path $taskRoot 'doc/results/encoder_search_goal_closure_v1'))
$taskAllowed=[IO.Path]::GetFullPath((Join-Path $taskRoot 'output/runs/market-change-energy'))+[IO.Path]::DirectorySeparatorChar
$taskDirectory=[IO.Path]::GetFullPath($ManifestDirectory)
if (-not $taskDirectory.StartsWith($taskAllowed,[StringComparison]::OrdinalIgnoreCase)) {throw 'Wrong original manifest directory'}
if (-not $ReceiptPath) {$ReceiptPath=Join-Path $taskEvidence ('preservation-after-'+[guid]::NewGuid().ToString('N')+'.json')}
$taskReceipt=[IO.Path]::GetFullPath($ReceiptPath)
if ([IO.Path]::GetDirectoryName($taskReceipt) -ne $taskEvidence -or [IO.Path]::GetExtension($taskReceipt) -ne '.json') {throw 'Fresh JSON receipt must be directly in the closure evidence directory'}
if (Test-Path -LiteralPath $taskReceipt) {throw 'Preserve existing closure receipt'}

function Hash-File([string]$taskPath) {
  return (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskBeforePath=Join-Path $taskDirectory 'before-summary.json'
$taskDurableBefore=Join-Path $taskRoot 'doc/results/market_change_energy_v1/preservation-before-summary.json'
$taskDurableAfter=Join-Path $taskRoot 'doc/results/market_change_energy_v1/preservation-after-summary.json'
$taskLocalAfter=Join-Path $taskDirectory 'after-summary.json'
$taskOldVerifier=Join-Path $taskRoot 'code/scripts/verify-market-energy-preservation.ps1'
$taskOldCreator=Join-Path $taskRoot 'code/scripts/preserve-market-energy.ps1'
$taskBeforeHash='80e23ccc93472d66a7d3af96298fdf63acb5fa46b7e8cf7a6872feb51ab4937d'
$taskOldVerifierHash='1d5dc23c8fb0debd8dac1ded494ecd4da5190dc343b0b66426aec23f60dc2d4a'
if ((Hash-File $taskBeforePath) -ne $taskBeforeHash -or (Hash-File $taskDurableBefore) -ne $taskBeforeHash) {throw 'Original baseline summary changed'}
if ((Hash-File $taskOldVerifier) -ne $taskOldVerifierHash) {throw 'Original verifier changed'}
$taskSummary=Get-Content -LiteralPath $taskBeforePath -Raw | ConvertFrom-Json
if ($taskSummary.kind -ne 'market_change_energy_preservation_v1' -or $taskSummary.roots.Count -ne 2) {throw 'Wrong original baseline kind/count'}
if ([IO.Path]::GetFullPath($taskSummary.local_manifest_directory) -ne $taskDirectory) {throw 'Baseline directory mismatch'}
if ((@($taskSummary.roots.repository | Sort-Object -Unique) -join ',') -ne 'embedding,integration') {throw 'Both distinct repository roots required'}
$taskOldAfter=Get-Content -LiteralPath $taskDurableAfter -Raw | ConvertFrom-Json
if ($taskOldAfter.kind -ne 'market_change_energy_preservation_after_v1' -or $taskOldAfter.status -ne 'passed' -or $taskOldAfter.before_summary_sha256 -ne $taskBeforeHash) {throw 'Original completed preservation receipt missing'}
if ((Hash-File $taskDurableAfter) -ne (Hash-File $taskLocalAfter)) {throw 'Original local/durable after receipts differ'}
$taskPinnedFiles=@($taskBeforePath,$taskDurableBefore,$taskDurableAfter,$taskLocalAfter,$taskOldVerifier,$taskOldCreator,$PSCommandPath)
$taskPins=@($taskPinnedFiles | ForEach-Object {[ordered]@{path=$_;sha256=(Hash-File $_);bytes=(Get-Item -LiteralPath $_).Length}})
$taskExpected=@{
  integration=@{root='C:/Work/cuwacunu/cuwacunu_dev';files=1500;total_bytes=[int64]415100170;manifest_bytes=317175;sha256='aa21344e18fc1bd1a4b3e3fb7199e7616d57aaf3423db83656ec042e271f885f'};
  embedding=@{root='C:/Work/cuwacunu/cuwacunu_embedding';files=76331;total_bytes=[int64]20169895215;manifest_bytes=20448781;sha256='14cd3bd147caf9e48a7c95867e9a18b33d9a640cd0e1d037d73eb08ce8b21203'}
}
$taskResults=@()
$taskFailures=[Collections.Generic.List[object]]::new()
foreach ($taskSpec in $taskSummary.roots) {
  $taskExpectedSpec=$taskExpected[$taskSpec.repository]
  if ($null -eq $taskExpectedSpec -or [IO.Path]::GetFullPath($taskSpec.root) -ne [IO.Path]::GetFullPath($taskExpectedSpec.root)) {throw 'Unexpected original repository root'}
  $taskManifest=Join-Path $taskDirectory ($taskSpec.repository+'-before.jsonl')
  if ([IO.Path]::GetFullPath($taskSpec.manifest) -ne [IO.Path]::GetFullPath($taskManifest)) {throw 'Unexpected original manifest path'}
  if ($taskSpec.files -ne $taskExpectedSpec.files -or $taskSpec.total_bytes -ne $taskExpectedSpec.total_bytes -or $taskSpec.bytes -ne $taskExpectedSpec.manifest_bytes -or $taskSpec.sha256 -ne $taskExpectedSpec.sha256) {throw 'Original manifest metadata changed'}
  if ((Hash-File $taskManifest) -ne $taskSpec.sha256 -or (Get-Item -LiteralPath $taskManifest).Length -ne $taskSpec.bytes) {throw 'Original manifest bytes changed'}
  # Prevalidate paths/counts before any file workers. Additions are deliberately
  # permitted; the full original list, not a new directory listing, is checked.
  $taskSeen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  $taskManifestCount=0
  $taskManifestBytes=[int64]0
  foreach ($taskLine in [IO.File]::ReadLines($taskManifest)) {
    $taskEntry=$taskLine | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($taskEntry.path) -or [IO.Path]::IsPathRooted($taskEntry.path) -or $taskEntry.path -match '(^|[\\/])\.\.([\\/]|$)' -or $taskEntry.path.Contains(':')) {throw 'Invalid baseline relative path'}
    if (-not $taskSeen.Add($taskEntry.path)) {throw 'Duplicate original manifest path'}
    if ($taskEntry.kind -notin @('regular-file','reparse-point-metadata') -or $taskEntry.sha256 -notmatch '^[0-9a-f]{64}$' -or $taskEntry.bytes -lt 0) {throw 'Invalid original manifest entry'}
    $taskManifestCount++
    $taskManifestBytes += [int64]$taskEntry.bytes
  }
  if ($taskManifestCount -ne $taskExpectedSpec.files -or $taskManifestBytes -ne $taskExpectedSpec.total_bytes) {throw 'Original complete manifest count/byte total differs'}
  $taskRepositoryRoot=[IO.Path]::GetFullPath($taskSpec.root)
  $taskRepositoryName=$taskSpec.repository
  $taskVerified=0
  $taskVerifiedBytes=[int64]0
  $taskRepositoryFailures=0
  [IO.File]::ReadLines($taskManifest) | ForEach-Object -Parallel {
    $ErrorActionPreference='Stop'
    $taskEntry=$null
    try {
      $taskEntry=$_ | ConvertFrom-Json
      $taskPrefix=$using:taskRepositoryRoot+[IO.Path]::DirectorySeparatorChar
      $taskPath=[IO.Path]::GetFullPath((Join-Path $using:taskRepositoryRoot $taskEntry.path))
      if (-not $taskPath.StartsWith($taskPrefix,[StringComparison]::OrdinalIgnoreCase)) {throw 'Baseline path escaped repository'}
      $taskFile=Get-Item -LiteralPath $taskPath
      if ($taskFile.PSIsContainer) {throw 'Expected a baseline file, found directory'}
      if (($taskFile.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        # This is exactly the original creator/verifier's UTF8 fsutil dump hash.
        # Deliberate dangling links are metadata, never followed as file content.
        $taskDump=(& fsutil reparsepoint query $taskPath 2>&1) -join "`n"
        if ($LASTEXITCODE -ne 0) {throw 'Cannot inspect original reparse metadata'}
        $taskHasher=[Security.Cryptography.SHA256]::Create()
        try {$taskHash=[Convert]::ToHexString($taskHasher.ComputeHash([Text.UTF8Encoding]::new($false).GetBytes($taskDump))).ToLowerInvariant()}
        finally {$taskHasher.Dispose()}
        $taskKind='reparse-point-metadata'
        if ($taskDump -cne $taskEntry.reparse_dump -or [int]$taskFile.Attributes -ne $taskEntry.attributes) {throw 'Original reparse dump/attributes changed'}
      } else {
        $taskHash=(Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $taskKind='regular-file'
      }
      if ($taskHash -ne $taskEntry.sha256 -or $taskFile.Length -ne $taskEntry.bytes -or $taskKind -ne $taskEntry.kind) {throw 'Historical SHA256/size/kind changed'}
      [pscustomobject]@{ok=$true;path=$taskEntry.path;bytes=[int64]$taskEntry.bytes;error=$null}
    } catch {
      [pscustomobject]@{ok=$false;path=$taskEntry.path;bytes=0;error=$_.Exception.Message}
    }
  } -ThrottleLimit 4 | ForEach-Object {
    if ($_.ok) {
      $taskVerified++
      $taskVerifiedBytes += $_.bytes
      if ($taskVerified % 5000 -eq 0) {Write-Output "CLOSURE_PRESERVATION_PROGRESS=$taskRepositoryName verified=$taskVerified"}
    } else {
      $taskRepositoryFailures++
      $taskFailures.Add([ordered]@{repository=$taskRepositoryName;path=$_.path;error=$_.error})
    }
  }
  if ((Hash-File $taskManifest) -ne $taskSpec.sha256) {throw 'Original manifest changed during verification'}
  if ($taskVerified+$taskRepositoryFailures -ne $taskExpectedSpec.files) {throw 'Worker accounting differs from complete original baseline'}
  if ($taskRepositoryFailures -eq 0 -and ($taskVerified -ne $taskExpectedSpec.files -or $taskVerifiedBytes -ne $taskExpectedSpec.total_bytes)) {throw 'Verified original count/bytes differ'}
  $taskResults += [ordered]@{repository=$taskRepositoryName;root=$taskRepositoryRoot;baseline_files=$taskExpectedSpec.files;verified_files=$taskVerified;verified_bytes=$taskVerifiedBytes;failed_files=$taskRepositoryFailures;status=$(if ($taskRepositoryFailures -eq 0) {'unchanged'} else {'failed'});before_manifest_sha256=$taskSpec.sha256;manifest_bytes=$taskSpec.bytes}
}
foreach ($taskPin in $taskPins) {
  if ((Hash-File $taskPin.path) -ne $taskPin.sha256 -or (Get-Item -LiteralPath $taskPin.path).Length -ne $taskPin.bytes) {throw 'Pinned original summary/receipt/verifier or current verifier changed during scan'}
}
$taskTimer.Stop()
$taskResult=[ordered]@{
  kind='encoder_search_goal_closure_preservation_v1';status=$(if ($taskFailures.Count -eq 0) {'passed'} else {'failed'});
  original_baseline_files=77831;verified_files=($taskResults | Measure-Object -Property verified_files -Sum).Sum;
  verified_bytes=($taskResults | Measure-Object -Property verified_bytes -Sum).Sum;failed_files=$taskFailures.Count;
  started_utc=$taskStarted.ToString('o');finished_utc=[DateTime]::UtcNow.ToString('o');wall_seconds=$taskTimer.Elapsed.TotalSeconds;
  roots=$taskResults;before_summary_sha256=$taskBeforeHash;original_baseline_policy=$taskSummary.policy;
  verification='All 77,831 original entries SHA256/size/kind and complete count/byte totals; original fsutil UTF8 reparse metadata, four read-only workers; additions permitted';
  exclusions='The original baseline excludes intentionally mutable live pointers and the then-new energy paths; this receipt does not claim those or later-added files were in that baseline';
  original_receipts_and_scripts_preserved=$true;pinned_files=$taskPins;failures=@($taskFailures);
  writes='Only this fresh closure JSON; no original manifest/receipt/source/output changed';receipt_path=$taskReceipt
}
New-Item -ItemType Directory -Path $taskEvidence -Force | Out-Null
$taskStream=[IO.File]::Open($taskReceipt,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false))
try {$taskWriter.WriteLine(($taskResult | ConvertTo-Json -Depth 12))} finally {$taskWriter.Dispose()}
Write-Output ('CLOSURE_PRESERVATION_RECEIPT='+$taskReceipt)
if ($taskFailures.Count -gt 0) {throw 'Original baseline preservation failed; see the fresh failure receipt'}
Write-Output 'CLOSURE_PRESERVATION_PASSED files=77831'

# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host file inspection/orchestration only; no project build or execution.
param(
  [Parameter(Mandatory=$true)][ValidateSet('Before','After')][string]$Action,
  [string]$ManifestDirectory
)
$ErrorActionPreference = 'Stop'
$taskIntegration = 'C:/Work/cuwacunu/cuwacunu_dev'
$taskEmbedding = 'C:/Work/cuwacunu/cuwacunu_embedding'
$taskEvidence = Join-Path $taskIntegration 'doc/results/market_change_energy_v1'
$taskEncoding = [Text.UTF8Encoding]::new($false)
function Write-NewJson([string]$taskPath, $taskValue) {
  $taskStream = [IO.File]::Open($taskPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
  $taskWriter = [IO.StreamWriter]::new($taskStream,$taskEncoding)
  try { $taskWriter.WriteLine(($taskValue | ConvertTo-Json -Depth 8)) } finally { $taskWriter.Dispose() }
}
function File-Record([string]$taskPath) {
  $taskFile = Get-Item -LiteralPath $taskPath
  if (($taskFile.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    # Preserve a link itself, including a deliberate dangling engineering link;
    # never follow it or invent nonexistent target contents.
    $taskDump = (& fsutil reparsepoint query $taskPath 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect reparse metadata: $taskPath" }
    $taskHash = [Security.Cryptography.SHA256]::Create()
    try { $taskDigest = [Convert]::ToHexString($taskHash.ComputeHash($taskEncoding.GetBytes($taskDump))).ToLowerInvariant() }
    finally { $taskHash.Dispose() }
    return [ordered]@{kind='reparse-point-metadata';sha256=$taskDigest;bytes=$taskFile.Length;attributes=[int]$taskFile.Attributes;reparse_dump=$taskDump}
  }
  return [ordered]@{kind='regular-file';sha256=(Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$taskFile.Length}
}
if ($Action -eq 'Before') {
  if ($ManifestDirectory) { throw 'Before chooses a fresh manifest directory' }
  New-Item -ItemType Directory -Path $taskEvidence -Force | Out-Null
  $taskLocalParent = Join-Path $taskIntegration 'output/runs/market-change-energy'
  New-Item -ItemType Directory -Path $taskLocalParent -Force | Out-Null
  $taskManifestRoot = Join-Path $taskLocalParent ('preservation-'+[guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Path $taskManifestRoot | Out-Null
  Write-Output ('PRESERVATION_DIRECTORY='+$taskManifestRoot)
  $taskSummaries = @()
  $taskMutable = @{
    integration=@('AGENTS.md','Makefile','.gitattributes','doc/CONTINUATION.md','doc/ENCODER_BACKPORT.md');
    embedding=@('AGENTS.md','Makefile','.gitattributes','doc/CURRENT_STATUS.md','doc/EMBEDDING_VERSIONS.md')
  }
  foreach ($taskSpec in @(@{name='integration';root=$taskIntegration},@{name='embedding';root=$taskEmbedding})) {
    $taskName=$taskSpec.name; $taskRoot=$taskSpec.root
    $taskSnapshot = Join-Path $taskEvidence ('mutable-before/'+$taskName)
    foreach ($taskRelative in $taskMutable[$taskName]) {
      $taskTarget=Join-Path $taskSnapshot $taskRelative
      New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($taskTarget)) -Force | Out-Null
      if (Test-Path -LiteralPath $taskTarget) {
        if ((File-Record $taskTarget).sha256 -ne (File-Record (Join-Path $taskRoot $taskRelative)).sha256) { throw 'Existing before snapshot differs' }
      } else { Copy-Item -LiteralPath (Join-Path $taskRoot $taskRelative) -Destination $taskTarget }
    }
    $taskFiles = @(foreach ($taskFolder in @('output','code','config','doc')) {
      $taskDirectory=Join-Path $taskRoot $taskFolder
      if (Test-Path -LiteralPath $taskDirectory) {
        Get-ChildItem -LiteralPath $taskDirectory -Recurse -File | Where-Object {
          $taskRelative=[IO.Path]::GetRelativePath($taskRoot,$_.FullName).Replace('\','/')
          $taskIsNew=$taskRelative -match '^(output/runs/market-change-energy/|doc/results/market_change_energy_v1/|code/encoders/rpb_energy/)' -or
            $taskRelative -match '(energy_change|change_energy|market-change-energy)' -or
            $taskRelative -eq 'doc/LATENT_MARKET_DYNAMICS_ROADMAP.md'
          -not $taskIsNew -and $taskRelative -notin $taskMutable[$taskName]
        }
      }
    }) | Sort-Object FullName
    $taskManifest=Join-Path $taskManifestRoot ($taskName+'-before.jsonl')
    $taskStream=[IO.File]::Open($taskManifest,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $taskWriter=[IO.StreamWriter]::new($taskStream,$taskEncoding)
    $taskCount=0; $taskBytes=[int64]0
    try {
      foreach ($taskFile in $taskFiles) {
        $taskRelative=[IO.Path]::GetRelativePath($taskRoot,$taskFile.FullName).Replace('\','/')
        $taskEntry=[ordered]@{path=$taskRelative}
        foreach ($taskField in (File-Record $taskFile.FullName).GetEnumerator()) { $taskEntry[$taskField.Key]=$taskField.Value }
        $taskWriter.WriteLine(($taskEntry | ConvertTo-Json -Compress))
        $taskCount++; $taskBytes+=$taskFile.Length
        if ($taskCount % 5000 -eq 0) { $taskWriter.Flush(); Write-Output ("PRESERVATION_PROGRESS=$taskName files=$taskCount") }
      }
    } finally { $taskWriter.Dispose() }
    $taskRecord=File-Record $taskManifest
    $taskSummaries += [ordered]@{repository=$taskName;root=$taskRoot;manifest=$taskManifest;files=$taskCount;total_bytes=$taskBytes;sha256=$taskRecord.sha256;bytes=$taskRecord.bytes}
    Write-Output ("PRESERVATION_BASELINE=$taskName files=$taskCount bytes=$taskBytes")
  }
  $taskSummary=[ordered]@{kind='market_change_energy_preservation_v1';stage='before_new_quality';client_date='2026-10-15';local_manifest_directory=$taskManifestRoot;roots=$taskSummaries;policy='All preexisting output/code/config/frozen docs in both repositories; additions allowed. New energy paths, unrelated roadmap and intentionally mutable live pointers excluded. Mutable before snapshots retained. Full manifests remain ignored local; summaries are durable.'}
  Write-NewJson (Join-Path $taskManifestRoot 'before-summary.json') $taskSummary
  Write-NewJson (Join-Path $taskEvidence 'preservation-before-summary.json') $taskSummary
  Write-Output ('PRESERVATION_BASELINE_COMPLETE='+$taskManifestRoot)
} else {
  if (-not $ManifestDirectory) { throw 'After requires existing ManifestDirectory' }
  $taskManifestRoot=[IO.Path]::GetFullPath($ManifestDirectory)
  $taskExpectedRoot=[IO.Path]::GetFullPath((Join-Path $taskIntegration 'output/runs/market-change-energy'))+[IO.Path]::DirectorySeparatorChar
  if (-not $taskManifestRoot.StartsWith($taskExpectedRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Wrong manifest root' }
  $taskSummary=Get-Content -LiteralPath (Join-Path $taskManifestRoot 'before-summary.json') -Raw | ConvertFrom-Json
  $taskResults=@()
  foreach ($taskSpec in $taskSummary.roots) {
    if ((File-Record $taskSpec.manifest).sha256 -ne $taskSpec.sha256) { throw 'Baseline manifest changed' }
    $taskCount=0
    foreach ($taskLine in [IO.File]::ReadLines($taskSpec.manifest)) {
      $taskEntry=$taskLine | ConvertFrom-Json
      $taskPath=Join-Path $taskSpec.root $taskEntry.path
      if (-not (Test-Path -LiteralPath $taskPath)) { throw "Missing historical file: $taskPath" }
      $taskNow=File-Record $taskPath
      if ($taskNow.sha256 -ne $taskEntry.sha256 -or $taskNow.bytes -ne $taskEntry.bytes -or $taskNow.kind -ne $taskEntry.kind) { throw "Historical file changed: $taskPath" }
      $taskCount++
      if ($taskCount % 5000 -eq 0) { Write-Output ("PRESERVATION_PROGRESS=$($taskSpec.repository) verified=$taskCount") }
    }
    if ($taskCount -ne $taskSpec.files) { throw 'Baseline entry count differs' }
    $taskResults += [ordered]@{repository=$taskSpec.repository;verified_files=$taskCount;status='unchanged';before_manifest_sha256=$taskSpec.sha256}
  }
  $taskResult=[ordered]@{kind='market_change_energy_preservation_after_v1';status='passed';roots=$taskResults;before_summary_sha256=(File-Record (Join-Path $taskManifestRoot 'before-summary.json')).sha256}
  Write-NewJson (Join-Path $taskManifestRoot 'after-summary.json') $taskResult
  Write-NewJson (Join-Path $taskEvidence 'preservation-after-summary.json') $taskResult
  Write-Output ('PRESERVATION_AFTER_PASSED='+$taskManifestRoot)
}

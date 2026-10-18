# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host-only file/hash/Git-ignore inspection. No tensor, model, build or head execution.
$ErrorActionPreference = 'Stop'
$integrationRoot = 'C:\Work\cuwacunu\cuwacunu_dev'
$embeddingRoot = 'C:\Work\cuwacunu\cuwacunu_embedding'
$evidenceRoot = Join-Path $integrationRoot 'doc\results\causal_patch_bottleneck_mae_v1'
$receiptPath = Join-Path $evidenceRoot 'preflight.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Preserve existing causal preflight receipt' }

function Get-Pin([string]$Path, [string]$RelativePath) {
    $item = Get-Item -LiteralPath $Path
    if ($item.PSIsContainer) { throw "Expected regular file: $Path" }
    return [ordered]@{path=$RelativePath; sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant(); bytes=$item.Length}
}
function Assert-Pin([string]$Root, $Pin, [string]$Digest) {
    $actual = Get-Pin (Join-Path $Root $Pin.path) $Pin.path
    if ($actual.sha256 -ne $Digest -or $actual.bytes -ne $Pin.bytes) { throw "Changed pinned file: $($Pin.path)" }
    return $actual
}
function Get-Inventory([string]$Root, [bool]$ExcludeNewVersions) {
    $files = @(Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object {
        -not $ExcludeNewVersions -or -not $_.FullName.StartsWith((Join-Path $Root 'versions') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
    })
    [string[]]$names = @($files | ForEach-Object {$_.FullName.Substring($Root.Length+1).Replace('\','/')})
    [Array]::Sort($names,[StringComparer]::Ordinal)
    $lines = [Text.StringBuilder]::new()
    $members = @()
    [long]$bytes = 0
    foreach ($name in $names) {
        $pin = Get-Pin (Join-Path $Root $name) $name
        $members += $pin
        [void]$lines.Append($name).Append("`t").Append($pin.bytes).Append("`t").Append($pin.sha256).Append("`n")
        $bytes += $pin.bytes
    }
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {$digest=[Convert]::ToHexString($algorithm.ComputeHash([Text.Encoding]::UTF8.GetBytes($lines.ToString()))).ToLowerInvariant()}
    finally {$algorithm.Dispose()}
    return [ordered]@{root=$Root;files=$names.Count;bytes=$bytes;inventory_recipe='ordinal relative POSIX path TAB decimal bytes TAB SHA256 LF, UTF8';inventory_sha256=$digest;members=$members}
}

$card = Get-Pin (Join-Path $integrationRoot 'doc\CAUSAL_PATCH_BOTTLENECK_MAE_V1_CARD.md') 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_CARD.md'
if ($card.sha256 -ne '3b059b84a60a7d193ffcb034480f54fa5aba532ed6cb6b541fa620c4ab584d5f') { throw 'Sealed CPB-v1 card changed' }
$ledgerPath = Join-Path $integrationRoot 'doc\encoder_search_ledger.json'
$ledger = Get-Content -Raw -LiteralPath $ledgerPath | ConvertFrom-Json
if (@($ledger.candidate_entries).Count -ne 11) { throw 'Preserve all eleven measured records' }
$checkpointPins = @($ledger.candidate_entries | ForEach-Object {$_.checkpoints})
if ($checkpointPins.Count -ne 55) { throw 'Expected55 historical checkpoint pins' }
$checkpoints = @()
foreach ($pin in $checkpointPins) {
    $actual = Assert-Pin $integrationRoot $pin $pin.sha256
    $actual.git_ignored = $true
    $actual.tensor_deserialized = $false
    $checkpoints += $actual
}
$ignored = @($checkpointPins.path | git -c safe.directory=C:/Work/cuwacunu/cuwacunu_dev -C $integrationRoot check-ignore --stdin)
if ($LASTEXITCODE -ne 0 -or $ignored.Count -ne 55) { throw 'Every old checkpoint must remain ignored' }
$tracked = @(git -c safe.directory=C:/Work/cuwacunu/cuwacunu_dev -C $integrationRoot ls-files -- $checkpointPins.path)
if ($LASTEXITCODE -ne 0 -or $tracked.Count -ne 0) { throw 'No checkpoint may be tracked' }
$syncPath = Join-Path $integrationRoot 'doc\results\market_location_scale_decoder_v1\portable-sync-final.json'
$sync = Get-Content -Raw -LiteralPath $syncPath | ConvertFrom-Json
if (@($sync.portable_pairs).Count -ne 99) { throw 'Expected99 old portable pairs' }
$pairs = @()
foreach ($pair in $sync.portable_pairs) {
    foreach ($side in @(@{root=$integrationRoot;field='integration_path'},@{root=$embeddingRoot;field='embedding_path'})) {
        $pin = [PSCustomObject]@{path=$pair.($side.field);bytes=$pair.bytes}
        $null = Assert-Pin $side.root $pin $pair.expected_sha256
    }
    $pairs += [ordered]@{integration_path=$pair.integration_path;embedding_path=$pair.embedding_path;sha256=$pair.expected_sha256;bytes=$pair.bytes}
}
$frozen = @()
foreach ($entry in $ledger.candidate_entries) {
    foreach ($field in @('card','report','completion','own_frozen_result_table')) {
        $pin = $entry.$field
        $actual = Assert-Pin $integrationRoot $pin $pin.sha256
        $actual.model = $entry.model
        $actual.role = $field
        $frozen += $actual
    }
}
if ($frozen.Count -ne 44) { throw 'Every old card/report/completion/table binding is required' }
$raw = Get-Inventory (Join-Path $embeddingRoot 'code\encoders\raw_patch_bottleneck_mae') $true
if ($raw.files -ne 275 -or $raw.bytes -ne 2990722 -or $raw.inventory_sha256 -ne '85e08c87812e80562cd4216355a0213d5c554c630d3877e0e107b5ba0249ca24') { throw 'Historical raw source tree changed' }
$stoppedRelative = 'output/runs/market-location-scale-synthetic-transfer/run-kh0f3ib3'
$stopped = Get-Inventory (Join-Path $integrationRoot $stoppedRelative) $false
if ($stopped.files -ne 23 -or $stopped.bytes -ne 8447717 -or $stopped.inventory_sha256 -ne '2157e0f932c925faeee3abdab6e36bd08e3040a456eacd37b6e6192c714ae199') { throw 'Interrupted V30 transfer artifacts changed' }
foreach ($name in @('readouts','independent-audit.json','verification.json')) {
    if (Test-Path -LiteralPath (Join-Path (Join-Path $integrationRoot $stoppedRelative) $name)) { throw 'Interrupted transfer disposition changed' }
}
$closureBase = 'doc/results/encoder_search_goal_closure_v1'
$originalPath = Join-Path $integrationRoot "$closureBase/preservation-after-01.json"
$summaryPath = Join-Path $integrationRoot "$closureBase/preservation-after-02-summary.json"
$original = Get-Content -Raw -LiteralPath $originalPath | ConvertFrom-Json
$summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
$originalPin = Get-Pin $originalPath "$closureBase/preservation-after-01.json"
if ($originalPin.sha256 -ne '22794bfc02b08137686beaa9f70b03a6a22836218f6d49b2bb6cb46d04020d25') { throw 'Original closure scan changed' }
if ($summary.status -ne 'passed' -or $summary.original_baseline_files -ne 77831 -or $summary.verified_files -ne 77831 -or $summary.failed_files -ne 0 -or -not $summary.original_receipts_and_scripts_preserved) { throw 'Historical closure correction incomplete' }
[long]$verifiedFiles=0;[long]$verifiedBytes=0
foreach ($rootRecord in $original.roots) {$verifiedFiles += $rootRecord.verified_files;$verifiedBytes += $rootRecord.verified_bytes;if($rootRecord.failed_files -ne 0 -or $rootRecord.status -ne 'unchanged'){throw 'Original closure root failed'}}
if ($verifiedFiles -ne 77831 -or $verifiedBytes -ne $summary.verified_bytes) { throw 'Closure corrected totals disagree with preserved root counts' }

$result = [ordered]@{
    kind='causal_patch_bottleneck_mae_v1_preflight';status='passed';date='2026-10-15';card=$card
    scope='Host file SHA256/size/availability/Git-ignore checks only; no tensor loads, model calls, builds, fits or source changes'
    checkpoints_verified=55;all_checkpoints_available_hash_matched_and_ignored=$true;tracked_checkpoints=0;checkpoints=$checkpoints
    portable_pairs_verified=99;all_portable_pairs_match=$true;portable_pairs=$pairs
    frozen_document_bindings=44;unique_frozen_documents=@($frozen.path|Sort-Object -Unique).Count;frozen_records=$frozen
    current_ledger=(Get-Pin $ledgerPath 'doc/encoder_search_ledger.json')
    current_ledger_markdown=(Get-Pin (Join-Path $integrationRoot 'doc\ENCODER_SEARCH_LEDGER.md') 'doc/ENCODER_SEARCH_LEDGER.md')
    old_portable_receipt=(Get-Pin $syncPath 'doc/results/market_location_scale_decoder_v1/portable-sync-final.json')
    historical_raw_source_inventory=$raw;historical_raw_versions_subdirectory_excluded_as_new=$true
    interrupted_v30_transfer=$stopped;interrupted_v30_transfer_scored=$false
    historical_closure_original=$originalPin;historical_closure_corrected=(Get-Pin $summaryPath "$closureBase/preservation-after-02-summary.json")
    historical_closure_files=$verifiedFiles;historical_closure_bytes=$verifiedBytes;new_broad_rescan=$false
    checkpoint_tensor_deserializations=0;encoder_calls=0;head_fits=0;new_encoder_updates=0
    script=(Get-Pin $PSCommandPath 'code/scripts/preflight-causal-patch-bottleneck-mae.ps1')
}
$null=New-Item -ItemType Directory -Path $evidenceRoot -Force
$payload=[Text.Encoding]::UTF8.GetBytes(($result|ConvertTo-Json -Depth 15)+"`n")
$stream=[IO.File]::Open($receiptPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try{$stream.Write($payload,0,$payload.Length)}finally{$stream.Dispose()}
Write-Output "CAUSAL_PREFLIGHT_PASSED=$receiptPath"
Write-Output ((Get-Pin $receiptPath 'doc/results/causal_patch_bottleneck_mae_v1/preflight.json')|ConvertTo-Json -Compress)

# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# File inspection only. Never rewrites historical receipts or runs a model.
$ErrorActionPreference = 'Stop'
$devRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$embeddingRoot = Join-Path (Split-Path $devRoot -Parent) 'cuwacunu_embedding'
$plan = Get-Content (Join-Path $PSScriptRoot 'rpb-layout-plan.json') -Raw | ConvertFrom-Json
$receiptPath = Join-Path $devRoot 'doc/rpb_layout_cleanup/relocation.json'
$receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
$before = Get-Content (Join-Path $devRoot 'doc/rpb_layout_cleanup/preservation-before.json') -Raw | ConvertFrom-Json
$utf8 = New-Object Text.UTF8Encoding($false)
function Hash($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Mapped($repo, $old) {
    $match = @($plan.($repo + '_moves') | Where-Object old_path -EQ $old)
    if ($match.Count -eq 1) { return $match[0].new_path }
    return $old
}
$pairs = @()
foreach ($pair in $before.portable_pairs) {
    $devPath = Mapped 'integration' $pair.integration_path
    $embeddingPath = Mapped 'embedding' $pair.embedding_path
    $a = Hash (Join-Path $devRoot $devPath)
    $b = Hash (Join-Path $embeddingRoot $embeddingPath)
    if ($a -ne $b -or $a -ne $pair.expected_sha256) { throw "Portable mismatch: $devPath" }
    $pairs += [ordered]@{ integration_old=$pair.integration_path; integration_current=$devPath; embedding_old=$pair.embedding_path; embedding_current=$embeddingPath; sha256=$a; historical_sha256=$pair.expected_sha256; match=$true }
}
$bodyCount = 0
$buildCount = 0
foreach ($repo in $receipt.repositories) {
    foreach ($row in $repo.moves) {
        $actual = Hash (Join-Path $repo.root $row.new_path)
        if (-not $row.new_path.EndsWith('.mk') -and -not $row.new_path.EndsWith('.sh') -and $actual -ne $row.before_sha256) { throw "Changed body: $($row.new_path)" }
        if ($row.new_path.EndsWith('.sh')) {
            $restored = [IO.File]::ReadAllText((Join-Path $repo.root $row.new_path)).Replace('/../../../../.."', '/../../../.."')
            $restoredHash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($utf8.GetBytes($restored))).Replace('-', '').ToLowerInvariant()
            if ($restoredHash -ne $row.before_sha256) { throw "Unexpected launcher difference: $($row.new_path)" }
        }
        if ($row.new_path.EndsWith('.mk') -or $row.new_path.EndsWith('.sh')) { $buildCount++ } else { $bodyCount++ }
        $row.after_sha256 = $actual
        $row.after_bytes = (Get-Item -LiteralPath (Join-Path $repo.root $row.new_path)).Length
        $row.byte_identical = $actual -eq $row.before_sha256
    }
    foreach ($row in $repo.build_changes) { $row.after_sha256 = Hash (Join-Path $repo.root $row.new_path) }
}
[IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 15) + "`n", $utf8)
$common = @()
$canonicalConfigs = @()
$aliases = Get-Content (Join-Path $devRoot 'doc/rpb_layout_cleanup/configuration-aliases.json') -Raw | ConvertFrom-Json
foreach ($alias in $aliases) {
    $a = Hash (Join-Path $devRoot $alias.canonical_path)
    $b = Hash (Join-Path $embeddingRoot $alias.canonical_path)
    $c = Hash (Join-Path $devRoot $alias.compatibility_path)
    if ($a -ne $b -or $a -ne $c -or $a -ne $alias.before_sha256) { throw "Canonical configuration mismatch: $($alias.canonical_path)" }
    $canonicalConfigs += [ordered]@{path=$alias.canonical_path;sha256=$a;compatibility_path=$alias.compatibility_path}
}
$family = 'code/encoders/raw_patch_bottleneck_mae'
foreach ($file in Get-ChildItem (Join-Path $devRoot $family) -Recurse -File -Filter '*.h') {
    $relative = $file.FullName.Substring($devRoot.Length + 1).Replace('\', '/')
    if ($relative -notmatch '/(shared|versions/v[0-9]+)/') { continue }
    $other = Join-Path $embeddingRoot $relative
    if (-not [IO.File]::Exists($other)) { continue }
    $a = Hash $file.FullName; $b = Hash $other
    if ($a -ne $b) { throw "Common canonical header mismatch: $relative" }
    $common += [ordered]@{path=$relative; sha256=$a}
}
$result = [ordered]@{
    kind='rpb_source_layout_sync_v1'; date='2026-10-15'; status='passed'
    unchanged_implementation_and_config_files=$bodyCount; relocated_build_or_launcher_files=$buildCount
    portable_pair_count=$pairs.Count; portable_pairs=$pairs
    common_canonical_header_count=$common.Count; common_canonical_headers=$common
    common_canonical_configuration_count=$canonicalConfigs.Count; common_canonical_configurations=$canonicalConfigs
    new_reusable_fixture=[ordered]@{
        path='code/encoders/raw_patch_bottleneck_mae/tests/layout_compatibility.cpp'
        sha256=(Hash (Join-Path $devRoot 'code/encoders/raw_patch_bottleneck_mae/tests/layout_compatibility.cpp'))
        description='New bounded no-training compatibility witness, additional to the 99 historical portable pairs.'
    }
    historical_records='Original 30 per-version JSON manifests and frozen receipts remain untouched; current navigation is additive in existing versions/index.json.'
}
$fixturePath = 'code/encoders/raw_patch_bottleneck_mae/tests/layout_compatibility.cpp'
if ((Hash (Join-Path $devRoot $fixturePath)) -ne (Hash (Join-Path $embeddingRoot $fixturePath))) { throw 'New reusable compatibility fixture differs' }
$json = ($result | ConvertTo-Json -Depth 12) + "`n"
foreach ($root in @($devRoot, $embeddingRoot)) {
    $destination = Join-Path $root 'doc/rpb_layout_cleanup/synchronization.json'
    [IO.Directory]::CreateDirectory((Split-Path $destination -Parent)) | Out-Null
    [IO.File]::WriteAllText($destination, $json, $utf8)
}
Write-Output "PASS: $bodyCount unchanged relocated bodies/configs, $($pairs.Count) historical portable pairs, $($common.Count) common canonical headers."

# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# One-time, source-only relocation. The plan and receipt are audit records,
# not model/version registries. Does not invoke builds, Git or research tools.
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
if (-not $Apply) { throw 'Inspect rpb-layout-plan.json, then pass -Apply.' }
$devRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot = Split-Path $devRoot -Parent
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'rpb-layout-plan.json') -Raw | ConvertFrom-Json
$utf8 = New-Object Text.UTF8Encoding($false)
$family = 'code/encoders/raw_patch_bottleneck_mae'
$receipts = @()

function Resolve-WorkspacePath($root, $relative) {
    $resolved = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $resolved.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escaped repository: $relative"
    }
    return $resolved
}
function Write-Utf8($path, $value) {
    [IO.Directory]::CreateDirectory((Split-Path $path -Parent)) | Out-Null
    [IO.File]::WriteAllText($path, $value, $utf8)
}
function Forward-Header($old, $new) {
    $parent = (Split-Path $old -Parent).Replace('\', '/') + '/'
    $relative = ([Uri]$parent).MakeRelativeUri([Uri]$new.Replace('\', '/')).ToString()
    Write-Utf8 $old ("// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.`n" +
        "// Compatibility include; implementation ownership is recorded in versions/README.md.`n#pragma once`n#include `"$relative`"`n")
}

foreach ($repository in @('integration', 'embedding')) {
    $root = if ($repository -eq 'integration') { $devRoot } else { Join-Path $workspaceRoot 'cuwacunu_embedding' }
    $moves = @($plan.($repository + '_moves'))
    if (-not $moves.Count) { throw "No planned moves for $repository" }
    $beforeBuild = @{}
    $buildFiles = @((Join-Path $root 'Makefile')) + @(Get-ChildItem (Join-Path $root 'code') -Recurse -File -Filter '*.mk' | ForEach-Object FullName)
    foreach ($file in $buildFiles) { $beforeBuild[$file] = [IO.File]::ReadAllText($file) }
    $pathMap = @{}
    $rows = @()
    $buildChanges = @()
    foreach ($move in $moves) {
        $old = Resolve-WorkspacePath $root $move.old_path
        $new = Resolve-WorkspacePath $root $move.new_path
        if (-not [IO.File]::Exists($old)) { throw "Missing source: $old" }
        if ([IO.File]::Exists($new)) { throw "Destination already exists: $new" }
        $hash = (Get-FileHash -LiteralPath $old -Algorithm SHA256).Hash.ToLowerInvariant()
        $bytes = (Get-Item -LiteralPath $old).Length
        [IO.Directory]::CreateDirectory((Split-Path $new -Parent)) | Out-Null
        [IO.File]::Move($old, $new)
        $kind = 'move'
        if ($old.EndsWith('.h')) { Forward-Header $old $new; $kind = 'move_with_compatibility_header' }
        if ($old.EndsWith('.conf')) { [IO.File]::Copy($new, $old); $kind = 'move_with_compatibility_config' }
        if ($old.EndsWith('.sh')) {
            # These two test launchers derive the repository root from their
            # own location; shared/tests adds exactly one directory level.
            $script = [IO.File]::ReadAllText($new)
            Write-Utf8 $new ($script.Replace('/../../../.."', '/../../../../.."'))
            $kind = 'move_with_repository_root_path_adjustment'
        }
        $pathMap[$move.old_path] = $move.new_path
        $rows += [ordered]@{ old_path=$move.old_path; new_path=$move.new_path; before_sha256=$hash; before_bytes=$bytes; operation=$kind }
    }

    # Keep all historic public include roots usable. A single family include root
    # also exposes every existing integration header, without copying model bodies.
    if ($repository -eq 'integration') {
        foreach ($move in $moves | Where-Object { $_.old_path -match '/include/' -and $_.old_path.EndsWith('.h') }) {
            $suffix = ($move.old_path -split '/include/', 2)[1]
            $alias = Resolve-WorkspacePath $root "$family/include/$suffix"
            if (-not [IO.File]::Exists($alias)) { Forward-Header $alias (Resolve-WorkspacePath $root $move.new_path) }
        }
    }

    # Expand only source/build/config path spellings. Public #include names and
    # all implementation bodies are left byte-identical.
    foreach ($entry in @($beforeBuild.GetEnumerator())) {
        $originalPath = $entry.Key.Substring($root.Length + 1).Replace('\', '/')
        $finalRelative = if ($pathMap.ContainsKey($originalPath)) { $pathMap[$originalPath] } else { $originalPath }
        $finalPath = Resolve-WorkspacePath $root $finalRelative
        $content = $entry.Value
        $roots = @{ 'RPB_ROOT'=$family; 'CODE_ROOT'='code' }
        foreach ($match in [regex]::Matches($content, '(?m)^([A-Z][A-Z0-9_]*ROOT)\s*:?=\s*([^\r\n]+)')) {
            $value = $match.Groups[2].Value.Trim().Replace('$(CODE_ROOT)', 'code').Replace('$(RPB_ROOT)', $family)
            if ($value.StartsWith('code/')) { $roots[$match.Groups[1].Value] = $value }
        }
        foreach ($move in $moves | Sort-Object { $_.old_path.Length } -Descending) {
            # Headers stay at old public paths as forwarding includes. Configs
            # keep compatibility bytes; live build checks use the canonical copy.
            if ($move.old_path.EndsWith('.h') -and $move.old_path -notmatch '/tests/') { continue }
            $content = $content.Replace($move.old_path, $move.new_path)
            foreach ($variable in @($roots.Keys)) {
                $prefix = $roots[$variable] + '/'
                if ($move.old_path.StartsWith($prefix)) {
                    $spelling = '$(' + $variable + ')/' + $move.old_path.Substring($prefix.Length)
                    $content = $content.Replace($spelling, $move.new_path)
                }
            }
        }
        if ($repository -eq 'embedding' -and $originalPath -eq 'Makefile') {
            $content = $content.Replace('$(RPB_ROOT)/src/*.cpp', '$(RPB_ROOT)/shared/src/*.cpp $(RPB_ROOT)/versions/*/src/*.cpp')
            $content = $content.Replace('$(RPB_ROOT)/config/*.conf', '$(RPB_ROOT)/shared/config/*.conf $(RPB_ROOT)/versions/*/config/*.conf')
            # Existing generic rules span multiple physical ownership folders.
            foreach ($spec in @(@('src','$(RPB_OBJECT_DIR)/%.o','%.cpp'), @('tests','$(RPB_TEST_DIR)/%_test.o','%_test.cpp'))) {
                $pattern = '(?m)^' + [regex]::Escape($spec[1] + ': $(RPB_ROOT)/' + $spec[0] + '/' + $spec[2]) + '\r?\n\tmkdir -p "\$\(@D\)"\r?\n\t[^\r\n]+'
                $found = [regex]::Match($content, $pattern)
                if (-not $found.Success) { throw "Missing expected generic $($spec[0]) build rule" }
                $folders = @($moves | Where-Object { $_.old_path -match "/$($spec[0])/" } | ForEach-Object {
                    ($_.new_path -split "/$($spec[0])/", 2)[0] + '/' + $spec[0]
                } | Sort-Object -Unique)
                $rules = @($folders | ForEach-Object {
                    $found.Value.Replace('$(RPB_ROOT)/' + $spec[0], $_)
                }) -join "`n`n"
                $content = $content.Substring(0, $found.Index) + $rules + $content.Substring($found.Index + $found.Length)
            }
            $canonicalHeaders = '$(wildcard $(RPB_ROOT)/shared/include/embedding/encoders/raw_patch_bottleneck_mae/*.h $(RPB_ROOT)/versions/*/include/embedding/encoders/raw_patch_bottleneck_mae/*.h)'
            $content = [regex]::Replace($content, '(?m)^(RPB_CORE_HEADERS :=[^\r\n]+)', '$1 ' + $canonicalHeaders)
        }
        if ($repository -eq 'integration' -and $originalPath -eq 'Makefile') {
            $content = $content.Replace('CPPFLAGS := ', "CPPFLAGS := -I$family/include ")
        }
        if ($content -cne $entry.Value) {
            $beforeHash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($utf8.GetBytes($entry.Value))).Replace('-', '').ToLowerInvariant()
            Write-Utf8 $finalPath $content
            $buildChanges += [ordered]@{
                old_path=$originalPath; new_path=$finalRelative; before_sha256=$beforeHash
                after_sha256=(Get-FileHash -LiteralPath $finalPath -Algorithm SHA256).Hash.ToLowerInvariant()
                review='Source/build/config path substitutions; expanded existing pattern rules; canonical-header provenance inputs; integration family include search root.'
            }
        }
    }
    foreach ($row in $rows) {
        $new = Resolve-WorkspacePath $root $row.new_path
        $row.after_sha256 = (Get-FileHash -LiteralPath $new -Algorithm SHA256).Hash.ToLowerInvariant()
        $row.after_bytes = (Get-Item -LiteralPath $new).Length
        $row.byte_identical = $row.before_sha256 -eq $row.after_sha256
        if (-not $row.byte_identical -and -not $new.EndsWith('.mk') -and -not $new.EndsWith('.sh')) { throw "Unexpected implementation difference: $new" }
        if ([IO.File]::Exists((Resolve-WorkspacePath $root $row.old_path))) {
            $row.compatibility_sha256 = (Get-FileHash -LiteralPath (Resolve-WorkspacePath $root $row.old_path) -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $receipts += [ordered]@{repository=$repository; root=$root; moves=$rows; build_changes=$buildChanges}
}
$auditRoot = Join-Path $devRoot 'doc/rpb_layout_cleanup'
[IO.Directory]::CreateDirectory($auditRoot) | Out-Null
Write-Utf8 (Join-Path $auditRoot 'relocation.json') (([ordered]@{
    kind='rpb_source_layout_relocation_v1'; date='2026-10-15'; repositories=$receipts
} | ConvertTo-Json -Depth 12) + "`n")
Write-Output ($receipts | ForEach-Object { "$($_.repository): $($_.moves.Count) relocated files" })

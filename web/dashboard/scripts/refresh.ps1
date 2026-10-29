# SPDX-License-Identifier: AGPL-3.0-only
# Host orchestration only. All project execution stays in the managed container.
param([switch]$Check, [switch]$Audit, [switch]$Test, [switch]$Branding)
$ErrorActionPreference = 'Stop'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$command = @('bash', 'web/dashboard/scripts/refresh.sh')
if ($Check) { $command += '--check' }
if ($Audit) { $command += '--audit' }
if ($Test) { $command += '--test' }
if ($Branding) { $command += '--branding' }
& (Join-Path $repositoryRoot 'container.ps1') -Action exec -Command $command
if ($LASTEXITCODE -ne 0) { throw 'Dashboard refresh failed; inspect the container output.' }

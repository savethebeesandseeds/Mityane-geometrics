# Exact additive archival copy only. No builds, research, overwrite or publication.
$ErrorActionPreference='Stop'
$repositoryRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$manifestPath=Join-Path $PSScriptRoot 'source-transfer.json'
$record=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$expected='C:\Work\Mityane-geometrics\archive\cuwacunu_dev\2026-10-18-risk-issuance-v1'
$destinationRoot=[IO.Path]::GetFullPath($record.destination)
if ($destinationRoot -ne $expected -or $record.sourceRepository.Replace('/','\') -ne $repositoryRoot) {throw 'Unexpected source or destination.'}
$prepared=@($record.files | ForEach-Object {
    $source=[IO.Path]::GetFullPath((Join-Path $repositoryRoot $_.source))
    $target=[IO.Path]::GetFullPath((Join-Path $destinationRoot $_.destination))
    if (-not $source.StartsWith($repositoryRoot+'\') -or -not $target.StartsWith($destinationRoot+'\')) {throw 'Path escaped fixed capsule roots.'}
    $bytes=[IO.File]::ReadAllBytes($source)
    $hash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
    if ($hash -ne $_.sha256 -or $bytes.Length -ne $_.bytes) {throw "Changed prepared source: $source"}
    if ((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash) {throw "Preserved different existing destination: $target"}
    [pscustomobject]@{entry=$_;target=$target;bytes=$bytes}
})
$manifestTarget=Join-Path $destinationRoot 'manifest.json'
$manifestBytes=[IO.File]::ReadAllBytes($manifestPath)
$manifestHash=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ((Test-Path -LiteralPath $manifestTarget) -and (Get-FileHash -LiteralPath $manifestTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifestHash) {throw 'Preserved different existing capsule manifest.'}
foreach ($item in $prepared) {
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($item.target))
    if (-not (Test-Path -LiteralPath $item.target)) {
        $stream=[IO.File]::Open($item.target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try {$stream.Write($item.bytes,0,$item.bytes.Length)} finally {$stream.Dispose()}
    }
    if ((Get-FileHash -LiteralPath $item.target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.entry.sha256) {throw 'Copied bytes differ.'}
}
if (-not (Test-Path -LiteralPath $manifestTarget)) {
    $stream=[IO.File]::Open($manifestTarget,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {$stream.Write($manifestBytes,0,$manifestBytes.Length)} finally {$stream.Dispose()}
}
if ((Get-FileHash -LiteralPath $manifestTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifestHash) {throw 'Copied capsule manifest differs.'}
$receipt=[ordered]@{schema='cuwacunu-risk-issuance-source-capsule-synchronization-v1';date='2026-10-18';
    destination=$record.destination;scope=$record.scope;sourceManifestSha256=$manifestHash;manifestDestination='manifest.json';files=$record.files}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'source-synchronization.json'),($receipt | ConvertTo-Json -Depth 8)+"`n",[Text.UTF8Encoding]::new($false))
Write-Output "Verified $($prepared.Count+1) additive source files including manifest in $destinationRoot"

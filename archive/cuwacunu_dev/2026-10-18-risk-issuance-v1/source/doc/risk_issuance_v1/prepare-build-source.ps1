# File inspection and packaging only; compilation remains in the managed container.
$ErrorActionPreference = 'Stop'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Set-Location -LiteralPath $repositoryRoot
$names = @(@(rg --files code) + @('Makefile','doc/risk_issuance_v1/policy.json')) | ForEach-Object { $_.Replace('\','/') } | Sort-Object -Unique
if ($LASTEXITCODE -ne 0) { throw 'Source inventory failed.' }
$entries = @($names | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    [ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$item.Length}
})
$record = [ordered]@{schema='byte-identical-container-build-source-v1';purpose='Avoid Windows-mount compiler stat latency; execute resulting binaries against live /cuwacunu source.';files=$entries}
$manifest = Join-Path $PSScriptRoot 'build-source-inputs.json'
[IO.File]::WriteAllText($manifest,($record | ConvertTo-Json -Depth 8)+"`n",[Text.UTF8Encoding]::new($false))
$archive = Join-Path $PSScriptRoot 'build-source.zip'
if (Test-Path -LiteralPath $archive) { throw 'Preserve the existing source archive; remove only that temporary archive explicitly before repackaging.' }
$zip = [IO.Compression.ZipFile]::Open($archive,[IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($entry in $entries) {
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,(Join-Path $repositoryRoot $entry.path),$entry.path)
    }
    [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$manifest,'build-source-inputs.json')
} finally { $zip.Dispose() }
[ordered]@{path=$archive;files=$entries.Count;sha256=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json

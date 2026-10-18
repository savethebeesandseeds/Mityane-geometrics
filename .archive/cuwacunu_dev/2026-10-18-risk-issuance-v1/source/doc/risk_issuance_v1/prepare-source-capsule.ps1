# File inspection and preservation handoff only; no build or research execution.
$ErrorActionPreference = 'Stop'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Set-Location -LiteralPath $repositoryRoot
$engineering = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'engineering.json') -Raw | ConvertFrom-Json
$snapshot = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'build-source-verification.json') -Raw | ConvertFrom-Json
foreach ($entry in @($engineering.sourceFiles)+@($snapshot.files)) {
    $actual=Get-FileHash -LiteralPath (Join-Path $repositoryRoot $entry.path) -Algorithm SHA256
    if ($actual.Hash.ToLowerInvariant() -ne $entry.sha256) {throw "Preserved evidence source changed: $($entry.path)"}
}
$readme = @'
Risk issuance v1: original source preservation capsule, 18 October 2026

This new archive preserves byte-identical Cuwacunu source, the prospective
policy and its bounded synthetic engineering receipt. source/ retains original
repository-relative paths and identities. It contains no checkpoints, raw data,
build products or credentials. Existing Mityane code and prior snapshots remain
unchanged. No Mityane build, research run, maintained port or publication is
established by copying this capsule.

Mityane's maintained tree is now flat. A future explicit port belongs under
code/operations/forecasting/risk_issuance/versions/v1, with the current operation
contracts, owning dependencies, new path-bound source identities and separate
managed-container verification. Never add these original nested paths to the
maintained code/embeddings tree or substitute them into historical evidence.

The saved engineering panel is artificial and has seven feature arms. The full
registered market panel, conventional/capacity/placebo controls, eligibility,
market utility and independent confirmation remain open. ObservableTargets is
caller-authenticated label authority; production adapters must authenticate the
original risk archive and exact origin join before calling the staged head.
'@
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'source-capsule-readme.txt'),$readme+"`n",[Text.UTF8Encoding]::new($false))
$pathMap = [ordered]@{
    schema='risk-issuance-original-source-path-map-v1'
    physicalRule='source/<original Cuwacunu repository-relative path>; no maintained source replacement'
    proposedFutureModuleOwner='code/operations/forecasting/risk_issuance/versions/v1'
    currentDependencyOwners=@('code/operations/forecasting','code/embeddings/state_representation','code/data/market','code/shared/artifacts','code/shared/settings')
    futurePortRequires=@('Explicit maintained-port scope','Current Operation Contract and Requirements-and-Evidence Record','New path-bound source identities','Separate managed Mityane container verification')
}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'source-capsule-path-map.json'),($pathMap | ConvertTo-Json -Depth 6)+"`n",[Text.UTF8Encoding]::new($false))
$names = @(@($snapshot.files.path) + @($engineering.sourceFiles.path) + @(
    'doc/risk_issuance_v1/engineering.json','LICENSE','LICENSES/NOTICE','LICENSES/MIT.txt',
    'doc/risk_issuance_v1/prepare-source-capsule.ps1','doc/risk_issuance_v1/sync-source-capsule.ps1',
    'doc/risk_issuance_v1/source-capsule-readme.txt','doc/risk_issuance_v1/source-capsule-path-map.json')) | Sort-Object -Unique
$files = @($names | ForEach-Object {
    $item=Get-Item -LiteralPath (Join-Path $repositoryRoot $_)
    $target = if ($_ -eq 'doc/risk_issuance_v1/source-capsule-readme.txt') {'README.txt'} elseif ($_ -eq 'doc/risk_issuance_v1/source-capsule-path-map.json') {'path-map.json'} else {'source/'+$_}
    [ordered]@{source=$_;destination=$target;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$record=[ordered]@{schema='cuwacunu-prepared-risk-issuance-source-capsule-v1';date='2026-10-18';
    sourceRepository=$repositoryRoot.Replace('\','/');destination='C:/Work/Mityane-geometrics/archive/cuwacunu_dev/2026-10-18-risk-issuance-v1';
    scope='Additive archival source preservation only; no maintained port, research execution, historical snapshot edit or publication.';manifestDestination='manifest.json';
    files=$files}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'source-transfer.json'),($record | ConvertTo-Json -Depth 8)+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{files=$files.Count+1;bytes=($files | ForEach-Object {$_.bytes} | Measure-Object -Sum).Sum;destination=$record.destination} | ConvertTo-Json

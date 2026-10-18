param([switch]$Render)
$ErrorActionPreference = 'Stop'
$paperDirectory = $PSScriptRoot
function Invoke-PaperDocker {
    param([string[]]$Arguments)
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Docker $($Arguments[0]) failed ($LASTEXITCODE)." }
}
$info = @(Invoke-PaperDocker -Arguments @('inspect', 'documents-latex') | ConvertFrom-Json)[0]
if ($info.Config.Labels.'org.local.documents-latex.managed' -ne 'true' -or
    $info.Config.Image -ne 'debian:12-slim' -or
    $info.State.Status -ne 'running' -or $info.State.Health.Status -ne 'healthy') {
    throw 'Use C:\Work\documents\cv.ps1 start to prepare the existing documents-latex environment.'
}
$containerId = $info.Id
$snapshot = '/tmp/cuwacunu-encoder-paper-' + [guid]::NewGuid().ToString('N')
Invoke-PaperDocker -Arguments @('exec', $containerId, 'mkdir', '-p', $snapshot)
foreach ($name in @('representation-encoder.tex', 'ieee-preamble.tex', 'references.bib', 'build.sh', 'vendor')) {
    Invoke-PaperDocker -Arguments @('cp', (Join-Path $paperDirectory $name), "${containerId}:${snapshot}/")
}
Invoke-PaperDocker -Arguments @('exec', $containerId, 'bash', "$snapshot/build.sh")
if ($Render) {
    Invoke-PaperDocker -Arguments @('exec', $containerId, 'mkdir', '-p', "$snapshot/build/preview")
    Invoke-PaperDocker -Arguments @('exec', $containerId, 'pdftoppm', '-r', '160', '-png', "$snapshot/build/representation-encoder.pdf", "$snapshot/build/preview/page")
}
$buildDirectory = Join-Path $paperDirectory 'build'
New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
Invoke-PaperDocker -Arguments @('cp', "${containerId}:${snapshot}/build/.", $buildDirectory)
Copy-Item -LiteralPath (Join-Path $buildDirectory 'representation-encoder.pdf') -Destination (Join-Path $paperDirectory 'representation-encoder.pdf')
$pdfInfo = Invoke-PaperDocker -Arguments @('exec', $containerId, 'pdfinfo', "$snapshot/build/representation-encoder.pdf")
if (-not ($pdfInfo -match '^Pages:\s+1\s*$')) { throw 'The PDF must be one page; inspect doc/build/preview and shorten the manuscript.' }
Write-Output "PDF: $(Join-Path $paperDirectory 'representation-encoder.pdf')"

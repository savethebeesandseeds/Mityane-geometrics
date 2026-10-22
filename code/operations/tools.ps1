# Host orchestration only. All builds and operation execution use container.ps1.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ToolArguments)
$ErrorActionPreference = 'Stop'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$forwarded = @()
$isPath = $false
foreach ($argument in $ToolArguments) {
    if ($isPath) {
        $absolute = [IO.Path]::GetFullPath($argument, $repositoryRoot)
        $relative = [IO.Path]::GetRelativePath($repositoryRoot, $absolute)
        if ($relative -eq '..' -or $relative.StartsWith('..\') -or [IO.Path]::IsPathRooted($relative)) {
            throw 'Tool request and output paths must be inside this repository.'
        }
        $forwarded += '/mityane/' + $relative.Replace('\', '/')
    } else { $forwarded += $argument }
    $isPath = $argument -in @('--request', '--spec', '--output', '--result', '--space', '--catalog', '--left', '--right', '--attempt')
}
if ($isPath) { throw 'A file or directory path is missing.' }
try {
    & (Join-Path $repositoryRoot 'container.ps1') -Action exec -Command (@('bash', 'code/operations/tools.sh') + $forwarded)
} catch {
    if ($LASTEXITCODE -in @(2, 3, 4)) { exit $LASTEXITCODE }
    throw
}
exit $LASTEXITCODE

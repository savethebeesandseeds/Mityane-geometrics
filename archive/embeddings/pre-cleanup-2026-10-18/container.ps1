[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('up', 'status', 'stop', 'shell', 'exec')]
    [string]$Action = 'status',
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Command
)
# Compatibility entry only. All lifecycle/configuration belongs to Mityane.
$launcher = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\container.ps1'))
& $launcher -Action $Action -Component embeddings -Command $Command
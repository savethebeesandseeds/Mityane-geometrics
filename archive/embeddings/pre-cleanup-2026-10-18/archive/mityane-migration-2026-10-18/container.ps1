param(
    [ValidateSet('up', 'status', 'stop', 'shell', 'exec')]
    [string]$Action = 'status',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$containerName = 'cuwacunu_embedding'
# Debian 12 image already used by the source project's documented environment.
$image = 'debian@sha256:6ebd97fa83deb272194a2cf015b3d26a4d538e9ad3a7a79d544c8af5b0a01443'
$manager = 'cuwacunu_embedding'

function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Docker failed (exit $LASTEXITCODE)." }
}

function Get-ManagedContainer {
    $ids = @(Invoke-Docker ps -aq --filter "name=^/$containerName`$")
    if ($ids.Count -eq 0) { return $null }
    $info = (Invoke-Docker inspect $ids[0] | ConvertFrom-Json)[0]
    $mounts = @($info.Mounts)
    $devices = @($info.HostConfig.DeviceRequests)
    $ports = @($info.HostConfig.PortBindings.PSObject.Properties)
    $expectedImageId = Invoke-Docker image inspect $image --format '{{.Id}}'
    $valid = $info.Config.Labels.'io.waajacu.managed-by' -eq $manager -and
        $info.Config.Labels.'io.waajacu.project-root' -eq $projectRoot -and
        $info.Config.Labels.'io.waajacu.configuration' -eq '1' -and
        $info.Image -eq $expectedImageId -and
        $info.Config.WorkingDir -eq '/embedding' -and
        (@($info.Config.Cmd) -join ' ') -eq '/bin/bash' -and
        -not $info.Config.Entrypoint -and
        $info.Config.OpenStdin -and $info.Config.Tty -and
        $mounts.Count -eq 1 -and $mounts[0].Type -eq 'bind' -and
        $mounts[0].Source -eq $projectRoot -and
        $mounts[0].Destination -eq '/embedding' -and $mounts[0].RW -and
        $info.HostConfig.ShmSize -eq 1073741824 -and
        $info.HostConfig.RestartPolicy.Name -eq 'no' -and
        -not $info.HostConfig.Privileged -and
        $devices.Count -eq 1 -and $devices[0].Count -eq -1 -and
        (@($devices[0].Capabilities[0]) -join ',') -eq 'gpu' -and
        -not $info.HostConfig.Devices -and $ports.Count -eq 0
    if (-not $valid) {
        throw "Container '$containerName' ($($info.Id)) has unmanaged or mismatched configuration; preserved without changes."
    }
    return $info
}

$container = Get-ManagedContainer
if ($Action -eq 'up') {
    if ($null -eq $container) {
        $available = @(Invoke-Docker image ls -q $image)
        if ($available.Count -eq 0) { Invoke-Docker pull $image }
        Invoke-Docker create --name $containerName --interactive --tty `
            --gpus all --shm-size 1g --restart no `
            --label "io.waajacu.managed-by=$manager" `
            --label "io.waajacu.project-root=$projectRoot" `
            --label 'io.waajacu.configuration=1' `
            --mount "type=bind,source=$projectRoot,target=/embedding" `
            --workdir /embedding $image /bin/bash | Out-Host
        $container = Get-ManagedContainer
    }
    if (-not $container.State.Running) { Invoke-Docker start $container.Id | Out-Host }
    $container = Get-ManagedContainer
    Write-Output "$containerName $($container.Id) $($container.State.Status)"
    exit 0
}
if ($null -eq $container) {
    if ($Action -eq 'status') { Write-Output "$containerName absent"; exit 0 }
    throw "Container is absent. Run .\container.ps1 up first."
}
switch ($Action) {
    'status' { Write-Output "$containerName $($container.Id) $($container.State.Status)" }
    'stop' { if ($container.State.Running) { Invoke-Docker stop $container.Id } }
    'shell' {
        if (-not $container.State.Running) { throw 'Container is stopped. Run .\container.ps1 up first.' }
        Invoke-Docker exec -it --workdir /embedding $container.Id /bin/bash
    }
    'exec' {
        if (-not $container.State.Running) { throw 'Container is stopped. Run .\container.ps1 up first.' }
        if (-not $Command) { throw 'Supply a command after exec.' }
        Invoke-Docker exec --workdir /embedding $container.Id @Command
    }
}

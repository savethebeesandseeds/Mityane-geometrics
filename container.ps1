param(
    [ValidateSet('up', 'status', 'stop', 'shell', 'exec')]
    [string]$Action = 'status',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$containerName = 'mityane-geometrics'
$manager = 'mityane-geometrics'
$imageRecordPath = Join-Path $projectRoot 'environment-image.json'
$hostPort = 4874

function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Docker failed (exit $LASTEXITCODE)." }
}

function Get-LocalImage {
    if (-not (Test-Path -LiteralPath $imageRecordPath -PathType Leaf)) {
        throw 'Missing environment-image.json. Restore the recorded local snapshot; no image is pulled or guessed.'
    }
    $record = Get-Content -LiteralPath $imageRecordPath -Raw | ConvertFrom-Json
    if ($record.image -ne 'mityane-geometrics:embedding-2026-10-12' -or
        $record.image_id -notmatch '^sha256:[0-9a-f]{64}$') {
        throw 'Invalid environment image tag or immutable ID; preserved without changes.'
    }
    $available = @(Invoke-Docker image ls --no-trunc -q $record.image)
    if ($available.Count -eq 0) {
        throw "Local image '$($record.image)' is absent. Restore the recorded snapshot as documented in doc/ENVIRONMENT.md."
    }
    $image = (Invoke-Docker image inspect $record.image | ConvertFrom-Json)[0]
    if ($image.Id -ne $record.image_id -or $image.Os -ne 'linux' -or $image.Architecture -ne 'amd64') {
        throw 'The local tag does not identify the recorded Linux amd64 image; preserved without changes.'
    }
    return $record
}

function Get-ManagedContainer($imageRecord) {
    $ids = @(Invoke-Docker ps -aq --filter "name=^/$containerName`$")
    if ($ids.Count -eq 0) { return $null }
    if ($ids.Count -ne 1) { throw "Ambiguous container '$containerName'; all containers were preserved." }
    $info = (Invoke-Docker inspect $ids[0] | ConvertFrom-Json)[0]
    $mounts = @($info.Mounts)
    $devices = @($info.HostConfig.DeviceRequests)
    $ports = @($info.HostConfig.PortBindings.PSObject.Properties)
    $binding = @($info.HostConfig.PortBindings.'4874/tcp')
    $valid = $info.Config.Labels.'io.waajacu.managed-by' -eq $manager -and
        $info.Config.Labels.'io.waajacu.project-root' -eq $projectRoot -and
        $info.Config.Labels.'io.waajacu.configuration' -eq '1' -and
        $info.Image -eq $imageRecord.image_id -and
        $info.Config.WorkingDir -eq '/mityane' -and
        (@($info.Config.Cmd) -join ' ') -eq '/bin/bash' -and
        -not $info.Config.Entrypoint -and $info.Config.OpenStdin -and $info.Config.Tty -and
        $mounts.Count -eq 1 -and $mounts[0].Type -eq 'bind' -and
        $mounts[0].Source -eq $projectRoot -and
        $mounts[0].Destination -eq '/mityane' -and $mounts[0].RW -and
        $info.HostConfig.ShmSize -eq 1073741824 -and
        $info.HostConfig.RestartPolicy.Name -eq 'no' -and
        -not $info.HostConfig.Privileged -and -not $info.HostConfig.PublishAllPorts -and
        $devices.Count -eq 1 -and $devices[0].Count -eq -1 -and
        (@($devices[0].Capabilities[0]) -join ',') -eq 'gpu' -and
        -not $devices[0].DeviceIDs -and -not $info.HostConfig.Devices -and
        $ports.Count -eq 1 -and $ports[0].Name -eq '4874/tcp' -and
        $binding.Count -eq 1 -and $binding[0].HostIp -eq '127.0.0.1' -and
        $binding[0].HostPort -eq [string]$hostPort
    if (-not $valid) {
        throw "Container '$containerName' ($($info.Id)) has unmanaged or mismatched configuration; preserved without changes."
    }
    return $info
}

function Assert-PortAvailable {
    # Include stopped containers: another project's reserved mapping is a conflict.
    $ids = @(Invoke-Docker ps -aq)
    foreach ($id in $ids) {
        $other = (Invoke-Docker inspect $id | ConvertFrom-Json)[0]
        if ($other.Name -eq "/$containerName") { continue }
        foreach ($entry in $other.HostConfig.PortBindings.PSObject.Properties) {
            if ($entry.Name -notlike '*/tcp') { continue }
            foreach ($binding in $entry.Value) {
                if ($binding.HostPort -eq [string]$hostPort) {
                    throw "TCP port $hostPort is reserved by $($other.Name) ($($other.Id)); preserved without changes."
                }
            }
        }
    }
    $socket = [Net.Sockets.Socket]::new(
        [Net.Sockets.AddressFamily]::InterNetwork,
        [Net.Sockets.SocketType]::Stream,
        [Net.Sockets.ProtocolType]::Tcp)
    try {
        $socket.ExclusiveAddressUse = $true
        $socket.Bind([Net.IPEndPoint]::new([Net.IPAddress]::Loopback, $hostPort))
    } catch {
        throw "Cannot reserve 127.0.0.1:$hostPort; a host listener or Windows port exclusion may conflict. No container was changed. $($_.Exception.Message)"
    } finally {
        $socket.Dispose()
    }
}

$imageRecord = Get-LocalImage
$container = Get-ManagedContainer $imageRecord
if ($Action -eq 'up') {
    if ($null -eq $container -or -not $container.State.Running) { Assert-PortAvailable }
    if ($null -eq $container) {
        Write-Output "Creating '$containerName' from $($imageRecord.image) ($($imageRecord.image_id)): /bin/bash; cwd /mityane; bind $projectRoot -> /mityane RW; no named volumes; GPUs all; shm 1 GiB; restart no; port 127.0.0.1:4874 -> 4874/tcp."
        Invoke-Docker create --pull never --name $containerName --interactive --tty `
            --gpus all --shm-size 1g --restart no `
            --label "io.waajacu.managed-by=$manager" `
            --label "io.waajacu.project-root=$projectRoot" `
            --label 'io.waajacu.configuration=1' `
            --mount "type=bind,source=$projectRoot,target=/mityane" `
            --publish '127.0.0.1:4874:4874/tcp' `
            --workdir /mityane $imageRecord.image_id /bin/bash | Out-Host
        $container = Get-ManagedContainer $imageRecord
    }
    if (-not $container.State.Running) { Invoke-Docker start $container.Id | Out-Host }
    $container = Get-ManagedContainer $imageRecord
    Write-Output "$containerName $($container.Id) $($container.State.Status)"
    exit 0
}
if ($null -eq $container) {
    if ($Action -eq 'status') { Write-Output "$containerName absent; local pinned image is available"; exit 0 }
    throw 'Container is absent. Run .\container.ps1 up first.'
}
switch ($Action) {
    'status' { Write-Output "$containerName $($container.Id) $($container.State.Status)" }
    'stop' { if ($container.State.Running) { Invoke-Docker stop $container.Id } }
    'shell' {
        if (-not $container.State.Running) { throw 'Container is stopped. Run .\container.ps1 up first.' }
        Invoke-Docker exec -it --workdir /mityane $container.Id /bin/bash
    }
    'exec' {
        if (-not $container.State.Running) { throw 'Container is stopped. Run .\container.ps1 up first.' }
        if (-not $Command) { throw 'Supply a command after exec.' }
        Invoke-Docker exec --workdir /mityane $container.Id @Command
    }
}

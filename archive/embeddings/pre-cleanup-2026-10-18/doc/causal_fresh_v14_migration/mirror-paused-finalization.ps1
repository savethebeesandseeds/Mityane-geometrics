# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Migration file inventory/copy/hash only. No project program is imported or run.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$containerRoot='/opt/cuwacunu/research/causal-fresh-v14-finalization/attempt-001'
$leaf=Join-Path $root 'output/engineering/causal-fresh-v14-finalization/attempt-001/migration-snapshot-001'
$payload=Join-Path $leaf 'payload'
$receipt=Join-Path $PSScriptRoot 'paused-finalization-mirror.json'
function Require([bool]$ok,[string]$message){if(-not $ok){throw $message}}
function Pin([string]$path){
 $item=Get-Item -LiteralPath $path
 Require (-not $item.PSIsContainer -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) 'Physical regular file required'
 [ordered]@{path=[IO.Path]::GetRelativePath($root,$item.FullName).Replace([char]92,[char]47);bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function NewJson([string]$path,$value){
 Require (-not (Test-Path -LiteralPath $path)) ('Preserve existing '+$path)
 [IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 30)+"`n",[Text.UTF8Encoding]::new($false))
}
Require (-not (Test-Path -LiteralPath $leaf) -and -not (Test-Path -LiteralPath $receipt)) 'Exclusive migration snapshot and receipt required'
$inspectLines=@(& docker.exe inspect cuwacunu)
Require ($LASTEXITCODE -eq 0) 'Read-only container inspection failed'
$inspect=($inspectLines -join "`n"|ConvertFrom-Json)[0]
Require ($inspect.Id -eq '82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e' -and $inspect.Image -eq 'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7' -and $inspect.State.Running) 'Exact existing running container required'
$null=[IO.Directory]::CreateDirectory($payload)
$environment=[ordered]@{observed_utc=[DateTime]::UtcNow.ToString('o');container='cuwacunu';id=$inspect.Id;image=$inspect.Image;path=$inspect.Path;args=@($inspect.Args);mounts=@($inspect.Mounts);ports=$inspect.NetworkSettings.Ports;restart=$inspect.HostConfig.RestartPolicy;devices=$inspect.HostConfig.DeviceRequests;state=$inspect.State.Status;note='Observed configuration only; no lifecycle or configuration changes'}
NewJson (Join-Path $leaf 'container-observation.json') $environment
$inventory=@'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
assert root==pathlib.Path('/opt/cuwacunu/research/causal-fresh-v14-finalization/attempt-001')
assert root.is_dir() and not root.is_symlink()
files=[]
for path in sorted(root.rglob('*')):
    assert not path.is_symlink(), 'Linked source forbidden'
    if path.is_file():
        data=path.read_bytes()
        files.append(dict(relative=path.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
assert files
print(json.dumps(dict(container_root=str(root),files=files),indent=2,sort_keys=True))
'@
try{
 $beforeLines=@($inventory|& docker.exe exec -i cuwacunu python3 - $containerRoot)
 Require ($LASTEXITCODE -eq 0) 'Read-only inventory failed'
 $beforePath=Join-Path $leaf 'inventory-before.json'
 [IO.File]::WriteAllText($beforePath,($beforeLines -join "`n")+"`n",[Text.UTF8Encoding]::new($false))
 $before=Get-Content -LiteralPath $beforePath -Raw|ConvertFrom-Json
 Require ('review-001.json' -notin @($before.files.relative)) 'Unexpected reviewer result: preserve state and reassess paused boundary'
 $copy=@(& docker.exe cp ('cuwacunu:'+$containerRoot+'/.') $payload 2>&1)
 Require ($LASTEXITCODE -eq 0) 'Copy failed; preserve partial snapshot'
 [IO.File]::WriteAllText((Join-Path $leaf 'copy.log'),($copy -join "`n")+"`n",[Text.UTF8Encoding]::new($false))
 $mapped=@()
 foreach($entry in $before.files){
  Require (-not [IO.Path]::IsPathRooted($entry.relative) -and $entry.relative -notmatch '(^|/)\.\.(/|$)' -and -not $entry.relative.Contains([string][char]92)) 'Bounded POSIX inventory path required'
  $target=[IO.Path]::GetFullPath((Join-Path $payload $entry.relative))
  Require ($target.StartsWith($payload+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Payload path escaped'
  $actual=Pin $target
  Require ($actual.bytes -eq $entry.bytes -and $actual.sha256 -eq $entry.sha256) ('Copy identity mismatch '+$entry.relative)
  $mapped += [ordered]@{relative=$entry.relative;container_path=$containerRoot+'/'+$entry.relative;host=$actual}
 }
 $actualNames=@(Get-ChildItem -LiteralPath $payload -Recurse -Force -File|ForEach-Object{[IO.Path]::GetRelativePath($payload,$_.FullName).Replace([char]92,[char]47)})
 Require ($actualNames.Count -eq $before.files.Count -and -not (Compare-Object ($actualNames|Sort-Object) ($before.files.relative|Sort-Object))) 'Exact whole directory member set required'
 Require (@(Get-ChildItem -LiteralPath $payload -Recurse -Force|Where-Object{$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count -eq 0) 'Linked payload forbidden'
 $afterLines=@($inventory|& docker.exe exec -i cuwacunu python3 - $containerRoot)
 Require ($LASTEXITCODE -eq 0) 'Read-only after inventory failed'
 $afterPath=Join-Path $leaf 'inventory-after.json'
 [IO.File]::WriteAllText($afterPath,($afterLines -join "`n")+"`n",[Text.UTF8Encoding]::new($false))
 Require ((Pin $beforePath).sha256 -eq (Pin $afterPath).sha256) 'Container bytes changed during snapshot'
 $card=Pin (Join-Path $payload 'card-source-001/finalization-card.json')
 Require ($card.bytes -eq 18917 -and $card.sha256 -eq '00d8b0c1a7c7aa2f13c1733a6160a5a1ac56f9b00e29e3e01836d89bf6575629') 'Exact reviewed metadata card required'
 $proposalPath=Join-Path $payload 'proposals-001/finalization-proposal.json'
 Require (Test-Path -LiteralPath $proposalPath) 'Completed proposal receipt required'
 $proposalOutputs=@(Get-ChildItem -LiteralPath (Join-Path $payload 'proposals-001') -Recurse -File|Where-Object{$_.Name -ne 'finalization-proposal.json'})
 Require ($proposalOutputs.Count -eq 3) 'Exactly three unapplied JSON proposal outputs required'
 $result=[ordered]@{kind='cpb_v14_paused_finalization_migration_mirror_v1';status='passed_file_preservation_only';utc=[DateTime]::UtcNow.ToString('o');goal_status='paused';container_root=$containerRoot;host_payload=[IO.Path]::GetRelativePath($root,$payload).Replace([char]92,[char]47);files=$mapped.Count;bytes=[long](($mapped|ForEach-Object{$_.host.bytes}|Measure-Object -Sum).Sum);mapped_files=$mapped;inventory_before=Pin $beforePath;inventory_after=Pin $afterPath;container_before_after_exact=$true;all_files_exact=$true;observed_environment=Pin (Join-Path $leaf 'container-observation.json');card=$card;proposal_generation=Pin $proposalPath;proposal_outputs=@($proposalOutputs|ForEach-Object{Pin $_.FullName});proposal_review_executed=$false;proposals_applied=$false;live_registry_files_written=0;model_calls=0;head_fits=0;training_updates=0;scientific_or_metadata_generators_replayed=0;container_lifecycle_changes=0;deletions=0;helper=Pin $PSCommandPath;limitations=@('Byte-preserving migration snapshot, not scientific or metadata proposal review.','Original finished-only mirror helper remains unused; no missing review is fabricated.','Frozen records retain original paths; remapping is future work, not achieved by moving a folder.')}
 NewJson $receipt $result
 [ordered]@{status=$result.status;files=$result.files;bytes=$result.bytes;receipt=Pin $receipt;proposal_outputs=$result.proposal_outputs}|ConvertTo-Json -Depth 8
}catch{
 $failure=Join-Path $leaf 'failure.json'
 if(-not(Test-Path -LiteralPath $failure)){NewJson $failure ([ordered]@{status='failed_preserved';message=$_.Exception.Message;deletions=0;generators_replayed=0;live_registry_files_written=0})}
 throw
}

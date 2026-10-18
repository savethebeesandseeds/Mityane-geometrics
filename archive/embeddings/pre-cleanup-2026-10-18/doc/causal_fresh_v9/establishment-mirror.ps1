# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Full physical V8 counterpart copy; exact metadata bytes only, no join/audit execution.
param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ManifestSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$StageSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExecutionSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InventorySha256,
 [string]$Manifest='doc/causal_fresh_v9/establishment-input-manifest.json')
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$containerRoot='/opt/cuwacunu/research/causal-fresh-v9-establishment/attempt-001'
$hostRoot=Join-Path $researchRoot 'output/engineering/causal-fresh-v9-establishment/attempt-001'
$mirror=Join-Path $hostRoot 'container-mirror'
if([IO.Path]::IsPathRooted($Manifest)-or$Manifest.Contains(':')-or$Manifest.Contains([char]92)-or$Manifest-match'(^|/)\.\.(/|$)'){throw 'Bounded forward-slash join-card repository path required'}
$manifestPath=[IO.Path]::GetFullPath((Join-Path $researchRoot $Manifest))
if(-not$manifestPath.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Join card escaped repository'}
if((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()-ne$ManifestSha256){throw 'Exact reviewed host join card required'}
if(-not(Test-Path -LiteralPath $hostRoot)-or(Test-Path -LiteralPath $mirror)){throw 'Existing stage and exclusive mirror required'}
$inspection=docker.exe inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$inspection.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$inspection.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$inspection.State.Running){throw 'Exact existing running container required'}
[IO.Directory]::CreateDirectory($mirror)|Out-Null
docker.exe cp "cuwacunu:${containerRoot}/." $mirror
if($LASTEXITCODE-ne0){throw 'Preserve partial mirror; copy failed'}
function Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No links'};[ordered]@{path=$f.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$binding){$p=Pin $path;if($p.bytes-ne$binding.bytes-or$p.sha256-ne$binding.sha256){throw "Bytes changed: $path"};$p}
foreach($pair in @(@('mirror-inventory.json',$InventorySha256),@('source-stage.json',$StageSha256),@('run-001/execution.json',$ExecutionSha256))){if((Pin (Join-Path $mirror $pair[0])).sha256-ne$pair[1]){throw ('Externally bound mirror bytes required before JSON parse: '+$pair[0])}}
if((Pin (Join-Path $hostRoot 'source-stage.json')).sha256-ne$StageSha256){throw 'Host Stage must match explicit external Stage digest'}
$inventory=Get-Content -Raw -LiteralPath (Join-Path $mirror 'mirror-inventory.json')|ConvertFrom-Json
if($inventory.kind-ne'cpb_v9_metadata_establishment_inventory_v1'){throw 'Exact inventory required'}
$files=@();$names=@();$prefix=[IO.Path]::GetFullPath($mirror).TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
foreach($entry in $inventory.files){$path=[IO.Path]::GetFullPath((Join-Path $mirror $entry.relative));if([IO.Path]::IsPathRooted($entry.relative)-or$entry.relative.Contains([char]92)-or$entry.relative-match'(^|/)\.\.(/|$)'-or-not$path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Unbounded mirror path'};$p=Require-Pin $path $entry;$names+=$entry.relative;$files +=[ordered]@{container=$entry.path;relative=$entry.relative;host=$p}}
$names+='mirror-inventory.json';$actual=@(Get-ChildItem -LiteralPath $mirror -Recurse -File|ForEach-Object{$_.FullName.Substring($mirror.Length+1).Replace([char]92,[char]47)})
if($actual.Count-ne$names.Count-or@($names|Sort-Object -Unique).Count-ne$names.Count-or(Compare-Object ($actual|Sort-Object) ($names|Sort-Object))){throw 'Missing,duplicate or unrecorded mirror file'}
$null=Require-Pin (Join-Path $mirror 'source.zip') (Pin (Join-Path $hostRoot 'source.zip'))
$null=Require-Pin (Join-Path $mirror 'source-stage.json') (Pin (Join-Path $hostRoot 'source-stage.json'))
$null=Require-Pin (Join-Path $mirror 'source/input-manifest.json') (Pin $manifestPath)
$null=Require-Pin (Join-Path $mirror 'source/CAPSULE_MANIFEST.json') (Pin (Join-Path $hostRoot 'source-manifest.json'))
$stage=Get-Content -Raw -LiteralPath (Join-Path $mirror 'source-stage.json')|ConvertFrom-Json
if($stage.join_card_sha256-ne$ManifestSha256-or$stage.status-ne'passed_AST_only'-or$stage.module_executions-ne0){throw 'Exact reviewed AST-only source stage required'}
if((Pin (Join-Path $mirror 'source/CAPSULE_MANIFEST.json')).sha256-ne$stage.manifest_sha256-or(Pin (Join-Path $mirror 'source.zip')).sha256-ne$stage.source_zip_sha256){throw 'Externally bound Stage must bind exact raw capsule manifest and ZIP'}
$capsule=Get-Content -Raw -LiteralPath (Join-Path $mirror 'source/CAPSULE_MANIFEST.json')|ConvertFrom-Json
if($capsule.join_card_sha256-ne$ManifestSha256-or$capsule.files.Count-ne$stage.files){throw 'Exact same externally bound reviewed card and source-member count required'}
$executionPath=Join-Path $mirror 'run-001/execution.json';$execution=Get-Content -Raw -LiteralPath $executionPath|ConvertFrom-Json
if($execution.status-ne$inventory.status){throw 'Disposition mismatch'}
$sourceExact=($execution.source_before|ConvertTo-Json -Depth 10 -Compress)-ceq($execution.source_after|ConvertTo-Json -Depth 10 -Compress)
if($execution.source_before.Count-ne$capsule.files.Count-or@($execution.source_before.path|Sort-Object -Unique).Count-ne$capsule.files.Count){$sourceExact=$false}
foreach($entry in $capsule.files){$expectedPath=$containerRoot+'/source/'+$entry.file;$bound=@($execution.source_before|Where-Object {$_.path-eq$expectedPath});if($bound.Count-ne1-or$bound[0].bytes-ne$entry.bytes-or$bound[0].sha256-ne$entry.sha256){$sourceExact=$false}}
if($execution.status-eq'passed'-and-not$sourceExact){throw 'Passing source-before/after equality required before canonical durable copies'}
$raw=Join-Path $mirror 'run-001/final-establishment.json'
$rawExists=Test-Path -LiteralPath $raw
if($rawExists){$null=Require-Pin $raw $execution.result}
$suffix=if($execution.status-eq'passed'){''}else{'.failed-attempt001'}
$durable=Join-Path $researchRoot ('doc/causal_fresh_v9/final-establishment'+$suffix+'.json')
$executionCopy=Join-Path $researchRoot ('doc/causal_fresh_v9/establishment-execution'+$suffix+'.json')
$receiptPath=Join-Path $researchRoot ('doc/causal_fresh_v9/establishment-mirror'+$suffix+'.json')
foreach($path in @($durable,$executionCopy,$receiptPath)){if(Test-Path -LiteralPath $path){throw 'Preserve prior durable receipt'}}
if($execution.status-eq'passed'-and-not$rawExists){throw 'Passed join must have a saved raw result'}
$durablePin=$null
if($rawExists){[IO.File]::Copy($raw,$durable,$false);$durablePin=Require-Pin $durable (Pin $raw)}
[IO.File]::Copy($executionPath,$executionCopy,$false);$executionPin=Require-Pin $executionCopy (Pin $executionPath)
$receipt=[ordered]@{kind='cpb_v9_metadata_establishment_mirror_v1';status='passed';join_status=$execution.status;files=$actual.Count;container_root=$containerRoot;host_root=$mirror;mapped_files=$files;inventory=(Pin (Join-Path $mirror 'mirror-inventory.json'));final_establishment=$durablePin;execution=$executionPin;helper=(Pin $PSCommandPath);join_card=(Pin $manifestPath);join_card_sha256=$ManifestSha256;source_before_after_exact=$sourceExact;model_calls=0;head_fits=0;training_updates=0;deletions=0}
[IO.File]::WriteAllText($receiptPath,($receipt|ConvertTo-Json -Depth 12)+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';join_status=$execution.status;files=$actual.Count;final_establishment=$durablePin;execution=$executionPin;mirror=(Pin $receiptPath)}|ConvertTo-Json -Depth 5

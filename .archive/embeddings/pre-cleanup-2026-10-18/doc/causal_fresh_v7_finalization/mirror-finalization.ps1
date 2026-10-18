# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Completed-file inventory/copy/hash only; no generator, reviewer or model replay.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$attempt=Join-Path $root 'output/engineering/causal-fresh-v7-finalization/attempt-001'
$leaf=Join-Path $attempt 'container-complete-001'
$payload=Join-Path $leaf 'payload'
$containerRoot='/opt/cuwacunu/research/causal-fresh-v7-finalization/attempt-001'
$proposalTarget=Join-Path $attempt 'proposals-001'
$durable=Join-Path $root 'doc/causal_fresh_v7_finalization'
function Pin([string]$p){
 $f=Get-Item -LiteralPath $p
 if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked file forbidden'}
 return [ordered]@{path=$f.FullName.Substring($root.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Match([string]$p,$expected){
 $a=Pin $p
 if($a.bytes-ne$expected.bytes-or$a.sha256-ne$expected.sha256){throw ('Changed completed bytes: '+$p)}
 return $a
}
function Json([string]$p,$v){
 if(Test-Path -LiteralPath $p){throw ('Preserve existing file: '+$p)}
 [IO.File]::WriteAllText($p,($v|ConvertTo-Json -Depth 20)+([string][char]10),[Text.UTF8Encoding]::new($false))
}
$copies=@(
 [ordered]@{source='card-source-001/finalization-card.json';destination='finalization-card.json'},
 [ordered]@{source='card-source-001/card-generation.json';destination='card-generation.json'},
 [ordered]@{source='proposals-001/finalization-proposal.json';destination='proposal-generation.json'},
 [ordered]@{source='review-001.json';destination='independent-proposal-review.json'}
)
if((Test-Path -LiteralPath $leaf)-or(Test-Path -LiteralPath $proposalTarget)){throw 'Exclusive full mirror and proposal copy required'}
foreach($c in $copies){if(Test-Path -LiteralPath (Join-Path $durable $c.destination)){throw 'Preserve existing durable receipt'}}
$observed=@(& docker inspect --format '{{.Id}}|{{.Image}}|{{.State.Running}}' cuwacunu)
if($LASTEXITCODE-ne0-or$observed.Count-ne1-or$observed[0]-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e|sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7|true'){throw 'Exact existing running managed container required'}
$null=New-Item -ItemType Directory -Path $leaf
$null=New-Item -ItemType Directory -Path $payload
$inventory=@'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
assert root==pathlib.Path('/opt/cuwacunu/research/causal-fresh-v7-finalization/attempt-001')
assert root.is_dir() and not root.is_symlink()
files=[]
for path in sorted(root.rglob('*')):
    assert not path.is_symlink(), 'Linked completed path'
    if path.is_file():
        data=path.read_bytes()
        files.append(dict(relative=path.relative_to(root).as_posix(),container_path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
assert files
print(json.dumps(dict(kind='cpb_v7_completed_finalization_file_inventory_v1',container_root=str(root),files=files),indent=2,sort_keys=True))
'@
[IO.File]::WriteAllText((Join-Path $leaf 'inventory.py'),$inventory+([string][char]10),[Text.UTF8Encoding]::new($false))
try{
 $beforeLines=@($inventory|& docker exec -i cuwacunu python3 - $containerRoot)
 if($LASTEXITCODE-ne0){throw 'Read-only inventory failed'}
 $beforePath=Join-Path $leaf 'inventory-before.json'
 $beforeLines|Set-Content -LiteralPath $beforePath -Encoding utf8NoBOM
 $before=Get-Content -LiteralPath $beforePath -Raw|ConvertFrom-Json
 $copy=@(& docker cp ('cuwacunu:'+($containerRoot+'/.')) $payload 2>&1)
 $copy|Set-Content -LiteralPath (Join-Path $leaf 'copy.log') -Encoding utf8NoBOM
 if($LASTEXITCODE-ne0){throw 'Full copy failed; preserve partial leaf'}
 $mapped=@()
 foreach($entry in $before.files){
  if([IO.Path]::IsPathRooted($entry.relative)-or$entry.relative-match'(^|/)\.\.(/|$)'-or$entry.relative.Contains([string][char]92)){throw 'Bounded POSIX inventory path required'}
  $target=[IO.Path]::GetFullPath((Join-Path $payload $entry.relative))
  if(-not$target.StartsWith($payload+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Inventory escaped new payload'}
  $mapped += [ordered]@{container_path=$entry.container_path;relative=$entry.relative;host=Match $target $entry}
 }
 $actual=@(Get-ChildItem -LiteralPath $payload -Recurse -Force -File|ForEach-Object{$_.FullName.Substring($payload.Length+1).Replace([char]92,[char]47)})
 if($actual.Count-ne$before.files.Count-or(Compare-Object ($actual|Sort-Object) ($before.files.relative|Sort-Object))){throw 'Missing or uninventoried payload file'}
 if(@(Get-ChildItem -LiteralPath $payload -Recurse -Force|Where-Object{$_.Attributes-band[IO.FileAttributes]::ReparsePoint}).Count){throw 'Linked payload path forbidden'}
 $afterLines=@($inventory|& docker exec -i cuwacunu python3 - $containerRoot)
 if($LASTEXITCODE-ne0){throw 'Read-only after-inventory failed'}
 $afterPath=Join-Path $leaf 'inventory-after.json'
 $afterLines|Set-Content -LiteralPath $afterPath -Encoding utf8NoBOM
 if((Pin $beforePath).sha256-ne(Pin $afterPath).sha256){throw 'Completed container bytes changed during mirror'}
 $review=Get-Content -LiteralPath (Join-Path $payload 'review-001.json') -Raw|ConvertFrom-Json
 if($review.status-ne'passed'-or$review.checks-ne725-or$review.numeric_passed-ne4-or$review.strict_new_quality_leaders-ne1){throw 'Actual independently reviewed finalization required'}
 $card=Pin (Join-Path $payload 'card-source-001/finalization-card.json')
 if($card.bytes-ne13437-or$card.sha256-ne'3e073a750072ae0f290a09cf1c6c25ef1683223dbeff8a0a7aeead7cef6ee879'){throw 'Actual root-admitted generated card required'}
 $proposalSource=Join-Path $payload 'proposals-001'
 $null=New-Item -ItemType Directory -Path $proposalTarget
 $proposalPins=@()
 foreach($f in Get-ChildItem -LiteralPath $proposalSource -Recurse -Force -File){
  $relative=$f.FullName.Substring($proposalSource.Length+1)
  $to=Join-Path $proposalTarget $relative
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($to))|Out-Null
  [IO.File]::Copy($f.FullName,$to,$false)
  $proposalPins += Match $to (Pin $f.FullName)
 }
 $docs=@()
 foreach($c in $copies){
  $from=Join-Path $payload $c.source
  $to=Join-Path $durable $c.destination
  [IO.File]::Copy($from,$to,$false)
  $docs += Match $to (Pin $from)
 }
 $receipt=[ordered]@{kind='cpb_v7_finalization_complete_container_mirror_v1';status='passed';container_root=$containerRoot;host_payload=$payload;
  files=$actual.Count;all_completed_files_exact=$true;container_before_after_exact=$true;
  inventory_before=Pin $beforePath;inventory_after=Pin $afterPath;mapped_files=$mapped;
  proposal_copies=$proposalPins;durable_receipts=$docs;actual_independent_review=Pin (Join-Path $payload 'review-001.json');
  helper=Pin $PSCommandPath;inventory_source=Pin (Join-Path $leaf 'inventory.py');model_calls=0;head_fits=0;training_updates=0;metadata_generators_replayed=0;live_files_written=0}
 Json (Join-Path $leaf 'mirror.json') $receipt
 Json (Join-Path $durable 'complete-mirror.json') $receipt
 [ordered]@{status='passed';files=$actual.Count;review_checks=$review.checks;proposal_copies=$proposalPins;durable_receipts=$docs;mirror=Pin (Join-Path $leaf 'mirror.json')}|ConvertTo-Json -Depth 8
}catch{
 if(-not(Test-Path -LiteralPath (Join-Path $leaf 'failure.json'))){Json (Join-Path $leaf 'failure.json') ([ordered]@{status='failed_preserved';error=$_.Exception.Message;deletions=0;generators_replayed=0;live_files_written=0})}
 throw
}

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# File/hash application only; never calculates research scores or runs project code.
param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ReviewSha256)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$attempt='output/engineering/causal-fresh-v7-finalization/attempt-001'
$receipt=Join-Path $PSScriptRoot 'applied-metadata.json'
function Require([bool]$ok,[string]$message){if(-not $ok){throw $message}}
function Rooted([string]$label){
 Require (-not [IO.Path]::IsPathRooted($label) -and $label -notmatch ':' -and @($label.Split('/')|Where-Object{$_ -in @('','..','.')}).Count -eq 0) 'Bounded research label required'
 $full=[IO.Path]::GetFullPath((Join-Path $researchRoot $label))
 Require ($full.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Research path escaped'
 return $full
}
function Hash([string]$path){return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Pin([string]$label){$path=Rooted $label;return [ordered]@{path=$label;bytes=(Get-Item -LiteralPath $path).Length;sha256=(Hash $path)}}
function CheckPin($pin,[string]$label){$actual=Pin $label;Require ($actual.bytes -eq $pin.bytes -and $actual.sha256 -eq $pin.sha256) ('Changed pinned file: '+$label)}
Require (-not (Test-Path -LiteralPath $receipt)) 'Preserve every previous metadata application receipt'
$reviewLabel='doc/causal_fresh_v7_finalization/independent-proposal-review.json'
$reviewPath=Rooted $reviewLabel
Require ((Hash $reviewPath) -eq $ReviewSha256) 'Explicitly reviewed proposal receipt required'
$review=Get-Content -LiteralPath $reviewPath -Raw|ConvertFrom-Json
Require ($review.kind -eq 'causal_fresh_v7_independent_metadata_proposal_review_v1' -and $review.status -eq 'passed' -and -not $review.baseline_changed -and -not $review.descriptor_changed -and -not $review.producer_changed) 'Independent saved metadata review must pass'
$snapshotLabel=$attempt+'/prior-live/manifest.json'
Require ((Hash (Rooted $snapshotLabel)) -eq '14a1d32feb36673436372808ffbd684de2975e5091209ad7a1b936dd19439da5') 'Original six-file prior-live manifest'
$snapshot=Get-Content -LiteralPath (Rooted $snapshotLabel) -Raw|ConvertFrom-Json
Require ($snapshot.files.Count -eq 6) 'Exactly six prior whole-file snapshots'
$byPath=@{}
foreach($old in $snapshot.files){CheckPin $old $old.path;CheckPin $old $old.snapshot;$byPath[$old.path]=$old}
$expected=@('doc/clean_causal_encoder_research_ledger.json','code/encoders/causal_patch_bottleneck_mae/versions/current.json','code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v7.market-01.json')
Require ($review.metadata_outputs.Count -eq 3 -and -not (Compare-Object $expected @($review.metadata_outputs.path))) 'Exactly three reviewed metadata outputs'
$changes=@()
foreach($pin in $review.metadata_outputs){
 $source=$attempt+'/proposals-001/'+$pin.path
 CheckPin $pin $source
 $old=$byPath[$pin.path]
 if(-not $old){Require (-not (Test-Path -LiteralPath (Rooted $pin.path))) 'New V7 instance must not replace any old file'}
 $changes += [ordered]@{path=$pin.path;source=$source;bytes=$pin.bytes;sha256=$pin.sha256;prior_sha256=$(if($old){$old.sha256}else{$null});snapshot=$(if($old){$old.snapshot}else{$null})}
}
$fronts=@('AGENTS.md','doc/CURRENT_STATUS.md','doc/EMBEDDING_VERSIONS.md','code/encoders/causal_patch_bottleneck_mae/README.md')
foreach($label in $fronts){
 $source=$attempt+'/front-note-proposals/'+$label
 $old=$byPath[$label]
 $data=[IO.File]::ReadAllBytes((Rooted $source))
 Require ($data.Length -gt $old.bytes) 'New front note plus whole prior text required'
 $tail=[byte[]]::new($old.bytes)
 [Array]::Copy($data,$data.Length-$old.bytes,$tail,0,$old.bytes)
 $hasher=[Security.Cryptography.SHA256]::Create()
 try{$tailHash=[Convert]::ToHexString($hasher.ComputeHash($tail)).ToLowerInvariant()}finally{$hasher.Dispose()}
 Require ($tailHash -eq $old.sha256) ('Prior whole text changed in prepend proposal: '+$label)
 $pin=Pin $source
 $changes += [ordered]@{path=$label;source=$source;bytes=$pin.bytes;sha256=$pin.sha256;prior_sha256=$old.sha256;snapshot=$old.snapshot}
}
Require ($changes.Count -eq 7) 'Only seven bounded metadata/document files'
$establish=Get-Content -LiteralPath (Rooted 'doc/causal_fresh_v7/final-establishment.json') -Raw|ConvertFrom-Json
Require ($establish.status -eq 'passed' -and $establish.trust_admission_passed -and $establish.full_coverage -and $establish.numeric_passed -eq 4 -and $establish.numeric_total -eq 22 -and -not $establish.overall_winner) 'Actual audited V7 outcome before application'
# All guards run before the first live write; the prior whole-file snapshots remain intact.
foreach($change in $changes){
 $destination=Rooted $change.path
 $null=[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))
 [IO.File]::WriteAllBytes($destination,[IO.File]::ReadAllBytes((Rooted $change.source)))
 CheckPin $change $change.path
}
$result=[ordered]@{kind='causal_fresh_v7_metadata_application_v1';status='passed';applied_utc=[DateTime]::UtcNow.ToString('o');files=$changes;prior_snapshot=$snapshotLabel;prior_snapshot_sha256='14a1d32feb36673436372808ffbd684de2975e5091209ad7a1b936dd19439da5';independent_proposal_review_sha256=$ReviewSha256;baseline='CPB-v3.market-01';baseline_changed=$false;descriptor_changed=$false;producer_changed=$false;goal_active=$true;application_changed=$false;model_calls=0;head_fits=0;training_updates=0;git_operations=0}
[IO.File]::WriteAllText($receipt,($result|ConvertTo-Json -Depth 30)+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';files=7;receipt=(Pin 'doc/causal_fresh_v7_finalization/applied-metadata.json')}|ConvertTo-Json -Depth 6

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# File/hash application only; never calculates research scores or runs project code.
param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ReviewSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SnapshotSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$FrontManifestSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$EstablishmentSha256)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$attempt='output/engineering/causal-fresh-v9-finalization/attempt-001'
$receipt=Join-Path $PSScriptRoot 'applied-metadata.json'
function Require([bool]$ok,[string]$message){if(-not $ok){throw $message}}
function Rooted([string]$label){
 Require (-not [IO.Path]::IsPathRooted($label) -and $label -notmatch ':' -and -not $label.Contains([string][char]92) -and @($label.Split('/')|Where-Object{$_ -in @('','..','.')}).Count -eq 0) 'Bounded research label required'
 $full=[IO.Path]::GetFullPath((Join-Path $researchRoot $label))
 Require ($full.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Research path escaped'
 $walk=$full
 while($walk -ne $researchRoot){
  if(Test-Path -LiteralPath $walk){Require (((Get-Item -LiteralPath $walk).Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) 'Linked metadata path forbidden'}
  $walk=[IO.Path]::GetDirectoryName($walk)
 }
 return $full
}
function Hash([string]$path){return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Pin([string]$label){$path=Rooted $label;return [ordered]@{path=$label;bytes=(Get-Item -LiteralPath $path).Length;sha256=(Hash $path)}}
function CheckPin($pin,[string]$label){$actual=Pin $label;Require ($actual.bytes -eq $pin.bytes -and $actual.sha256 -eq $pin.sha256) ('Changed pinned file: '+$label)}
Require (-not (Test-Path -LiteralPath $receipt)) 'Preserve every previous metadata application receipt'
$reviewLabel='doc/causal_fresh_v9_finalization/independent-proposal-review.json'
$reviewPath=Rooted $reviewLabel
Require ((Hash $reviewPath) -eq $ReviewSha256) 'Explicitly reviewed proposal receipt required'
$review=Get-Content -LiteralPath $reviewPath -Raw|ConvertFrom-Json
Require ($review.kind -eq 'causal_fresh_v9_independent_metadata_proposal_review_v1' -and $review.status -eq 'passed' -and -not $review.baseline_changed -and -not $review.descriptor_changed -and -not $review.producer_changed) 'Independent saved metadata review must pass'
$snapshotLabel='output/engineering/causal-fresh-v9-finalization/prior-live-001/manifest.json'
# SnapshotSha256 is external root-reviewed input; never infer a future value.
Require ((Hash (Rooted $snapshotLabel)) -eq $snapshotSha256) 'Original six-file prior-live manifest'
$snapshot=Get-Content -LiteralPath (Rooted $snapshotLabel) -Raw|ConvertFrom-Json
Require ($snapshot.files.Count -eq 6) 'Exactly six prior whole-file snapshots'
$byPath=@{}
foreach($old in $snapshot.files){CheckPin $old $old.path;CheckPin $old $old.snapshot;$byPath[$old.path]=$old}
$expected=@('doc/clean_causal_encoder_research_ledger.json','code/encoders/causal_patch_bottleneck_mae/versions/current.json','code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v9.market-01.json')
Require ($review.metadata_outputs.Count -eq 3 -and -not (Compare-Object $expected @($review.metadata_outputs.path))) 'Exactly three reviewed metadata outputs'
$changes=@()
foreach($pin in $review.metadata_outputs){
 $source=$attempt+'/proposals-001/'+$pin.path
 CheckPin $pin $source
 $old=$byPath[$pin.path]
 if(-not $old){Require (-not (Test-Path -LiteralPath (Rooted $pin.path))) 'New V9 instance must not replace any old file'}
 $changes += [ordered]@{path=$pin.path;source=$source;bytes=$pin.bytes;sha256=$pin.sha256;prior_sha256=$(if($old){$old.sha256}else{$null});snapshot=$(if($old){$old.snapshot}else{$null})}
}
$fronts=@('AGENTS.md','doc/CURRENT_STATUS.md','doc/EMBEDDING_VERSIONS.md','code/encoders/causal_patch_bottleneck_mae/README.md')
$frontManifestLabel='output/engineering/causal-fresh-v9-finalization/front-note-proposals-001/manifest.json'
# FrontManifestSha256 is external root-reviewed input, including all4proposal pins.
Require ((Hash (Rooted $frontManifestLabel)) -eq $frontManifestSha256) 'Exactly reviewed four prepend proposals required'
$frontManifest=Get-Content -LiteralPath (Rooted $frontManifestLabel) -Raw|ConvertFrom-Json
Require ($frontManifest.status -eq 'prepared_not_applied' -and $frontManifest.files.Count -eq 4 -and -not (Compare-Object $fronts @($frontManifest.files.path)) -and $frontManifest.prior_live_manifest.sha256 -eq $snapshotSha256) 'Exact four-note and prior snapshot scope'
CheckPin $frontManifest.final_establishment 'doc/causal_fresh_v9/final-establishment.json'
foreach($label in $fronts){
 $source='output/engineering/causal-fresh-v9-finalization/front-note-proposals-001/'+$label
 $frontPin=@($frontManifest.files|Where-Object{$_.path -eq $label})
 Require ($frontPin.Count -eq 1 -and $frontPin[0].proposal -eq $source) 'Exactly one bounded reviewed prepend proposal'
 CheckPin $frontPin[0] $source
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
Require ((Hash (Rooted 'doc/causal_fresh_v9/final-establishment.json'))-eq$EstablishmentSha256) 'Exact externally reviewed completed establishment required'
$establish=Get-Content -LiteralPath (Rooted 'doc/causal_fresh_v9/final-establishment.json') -Raw|ConvertFrom-Json
Require ($establish.kind -eq 'causal_fresh_v9_postmeasurement_establishment_v1' -and $establish.candidate -eq 'CPB-v9.market-01' -and $establish.status -eq 'passed' -and $establish.trust_admission_passed -and $establish.full_coverage -and $establish.numeric_passed -eq $review.numeric_passed -and $establish.numeric_total -eq 22 -and $review.numeric_total -eq 22 -and $establish.overall_establishment_passed -eq $review.overall_establishment_passed -and -not $establish.overall_winner -and -not $establish.winner_selection_made) 'Actual audited V9 outcome before application'
Require ($establish.initial_partial_native20_40_pairings_passed -and $establish.initial_pairing_count -eq 13 -and $establish.initial_pairing_feature_archives -eq 8 -and $establish.initial_pairing_statistic_archives -eq 5 -and -not $establish.whole_native40_initial_equality_claimed -and $establish.mandatory_identifiable_learned_tensors -eq 75 -and $establish.registered_learned_tensors -eq 76 -and $establish.all_four_export_row_groups_and_position_contrasts_verified -and -not $establish.weight_decay_counted_as_data_learning) 'Partial initial pairing and actual identifiable learning admission required'
Require ($establish.all_three_between_bin_position_directions_verified-and$establish.between_bin_overall_final_projection_verified-and-not$establish.individual_final_coefficient_nonzero_gate) 'All three direction events and overall final projection, no per-final-coefficient gate'
# All guards run before the first live write; the prior whole-file snapshots remain intact.
foreach($change in $changes){
 $destination=Rooted $change.path
 $null=[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))
 [IO.File]::WriteAllBytes($destination,[IO.File]::ReadAllBytes((Rooted $change.source)))
 CheckPin $change $change.path
}
$result=[ordered]@{kind='causal_fresh_v9_metadata_application_v1';status='passed';applied_utc=[DateTime]::UtcNow.ToString('o');files=$changes;prior_snapshot=$snapshotLabel;prior_snapshot_sha256=$snapshotSha256;independent_proposal_review_sha256=$ReviewSha256;front_note_proposal_manifest_sha256=$frontManifestSha256;numeric_passed=$establish.numeric_passed;numeric_total=$establish.numeric_total;baseline='CPB-v3.market-01';baseline_changed=$false;descriptor_changed=$false;producer_changed=$false;goal_active=$true;application_changed=$false;model_calls=0;head_fits=0;training_updates=0;git_operations=0}
[IO.File]::WriteAllText($receipt,($result|ConvertTo-Json -Depth 30)+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';files=7;receipt=(Pin 'doc/causal_fresh_v9_finalization/applied-metadata.json')}|ConvertTo-Json -Depth 6

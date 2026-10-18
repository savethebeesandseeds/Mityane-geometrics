# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
# Host file inspection only. Run after the authorized final metadata application.
param([switch]$FinalMetadataApplied)
$ErrorActionPreference='Stop'
if(-not $FinalMetadataApplied){throw 'Wait for the root final-metadata-applied signal'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$destination=Join-Path $researchRoot 'doc/causal_fresh_v6/final-preservation.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve the existing final receipt'}
$script:checks=0
$script:hashes=@{}
function Check([bool]$value,[string]$message){$script:checks++;if(-not $value){throw $message}}
function Hash([string]$path){
  $item=Get-Item -LiteralPath $path
  Check (-not($item.Attributes-band[IO.FileAttributes]::ReparsePoint)) ('Unexpected link: '+$path)
  $key=$item.FullName+'|'+$item.Length+'|'+$item.LastWriteTimeUtc.Ticks
  if(-not $script:hashes.ContainsKey($key)){$script:hashes[$key]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
  return $script:hashes[$key]
}
function Read([string]$path){Get-Content -LiteralPath (Join-Path $researchRoot $path) -Raw|ConvertFrom-Json}
function Pin([string]$path){$item=Get-Item -LiteralPath (Join-Path $researchRoot $path);[ordered]@{path=$path;bytes=$item.Length;sha256=(Hash $item.FullName)}}
function Verify($pin,[string]$repo='cuwacunu_embedding',[string]$override=''){
  $path=if($override){$override}elseif([IO.Path]::IsPathRooted($pin.path)){$pin.path}else{Join-Path (Join-Path $workspaceRoot $repo) $pin.path}
  $bound=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repo))+[IO.Path]::DirectorySeparatorChar
  Check ([IO.Path]::GetFullPath($path).StartsWith($bound,[StringComparison]::OrdinalIgnoreCase)) 'Pinned host path stays inside its repository'
  $item=Get-Item -LiteralPath $path
  Check ($item.Length-eq$pin.bytes-and(Hash $path)-eq$pin.sha256) ('Frozen pin changed: '+$repo+'/'+$pin.path)
}
function Equal($a,$b){($a|ConvertTo-Json -Depth 90 -Compress)-ceq($b|ConvertTo-Json -Depth 90 -Compress)}
$oldPath='doc/causal_v3_v4_train_diagnosis/preservation-before.json';$old=Read $oldPath
Check ((Hash (Join-Path $researchRoot $oldPath))-eq'18ef6f002103e3f2f920cb41678e8913104175f7b14afc291b3e37f32b11b748') 'Old1030 inventory hash'
Check ($old.pins.Count-eq1030) 'Exactly1030 old pins'
$v5Path='doc/causal_fresh_v5/final-preservation.json';$v5=Read $v5Path
Check ((Hash (Join-Path $researchRoot $v5Path))-eq'0cb3ec54789c580df741d68b5923459040948a38284584eca9691540c597873a') 'FrozenV5 final receipt'
Verify $v5.prior_live_snapshot
$snapshot5=Read $v5.prior_live_snapshot.path
foreach($p in $snapshot5.files){Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot $p.snapshot_path)}
$snapshot6Path='output/engineering/causal-fresh-v6-finalization/attempt-001/prior-live/manifest.json';$snapshot6=Read $snapshot6Path
Check ($snapshot6.status-eq'passed'-and$snapshot6.files.Count-eq6) 'Exactly six V6 prior-live snapshots'
foreach($p in $snapshot6.files){Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot $p.snapshot)}
$authorized=@{};foreach($p in $v5.authorized_live_metadata_changes){$authorized[$p.path]=$p}
$oldExact=0;$oldResolved=@()
foreach($p in $old.pins){
  $path=Join-Path (Join-Path $workspaceRoot $p.repository) $p.path;$item=Get-Item -LiteralPath $path
  if($item.Length-eq$p.bytes-and(Hash $path)-eq$p.sha256){$oldExact++;continue}
  Check ($p.repository-eq'cuwacunu_embedding'-and$authorized.ContainsKey($p.path)) ('Unauthorized old pin mutation: '+$p.repository+'/'+$p.path)
  $former=$authorized[$p.path]
  Check ($p.bytes-eq$former.old_bytes-and$p.sha256-eq$former.old_sha256) 'Exact original metadata identity'
  Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot $former.snapshot_path)
  $oldResolved += [ordered]@{path=$p.path;old_bytes=$p.bytes;old_sha256=$p.sha256;snapshot=$former.snapshot_path;current=(Pin $p.path)}
}
Check ($oldExact+$oldResolved.Count-eq1030-and$oldResolved.Count-eq5) 'All1030 retained with five authorized metadata displacements'
$metadataChain=@();foreach($p in $snapshot6.files){
  $v5Current=@($v5.authorized_live_metadata_changes|Where-Object path -EQ $p.path)
  if($v5Current.Count-eq1){Check ($p.bytes-eq$v5Current[0].current_bytes-and$p.sha256-eq$v5Current[0].current_sha256) 'Exact V5-current toV6-prior metadata chain'}
  else{Check ($p.path-eq$v5.supplemental_live_family_README.current.path-and$p.bytes-eq$v5.supplemental_live_family_README.current.bytes-and$p.sha256-eq$v5.supplemental_live_family_README.current.sha256) 'Exact sixth familyREADME metadata chain'}
  $metadataChain += [ordered]@{path=$p.path;prior_bytes=$p.bytes;prior_sha256=$p.sha256;prior_snapshot=$p.snapshot;current=(Pin $p.path)}
}
$applicationPath='doc/causal_fresh_v6_finalization/applied-metadata.json';$application=Read $applicationPath
Check ((Hash (Join-Path $researchRoot $applicationPath))-eq'89fffe2ae4e4a550c3dbae640b8e66925848b0425c20f4c56cde1ee18dbdf073') 'Exact authorized final application receipt'
Check ($application.status-eq'passed'-and$application.files.Count-eq7-and-not$application.baseline_changed-and-not$application.application_changed-and$application.goal_active) 'Exactly seven authorized metadata applications; goal active'
foreach($p in $application.files){Verify $p;Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot $p.source)}
$proposalReviewPath='doc/causal_fresh_v6_finalization/independent-proposal-review.json';$proposalReview=Read $proposalReviewPath
Check ($proposalReview.status-eq'passed'-and$proposalReview.checks-eq161-and(Hash (Join-Path $researchRoot $proposalReviewPath))-eq$application.independent_proposal_review_sha256) 'Bound independent161-check proposal review'
foreach($p in $proposalReview.metadata_outputs){Verify $p}
$generationPath='doc/causal_fresh_v6_finalization/proposal-generation.json';$generation=Read $generationPath
foreach($p in $generation.outputs){Verify $p}
$frontPath='output/engineering/causal-fresh-v6-finalization/attempt-001/front-note-proposals/manifest.json';$front=Read $frontPath
Check ($front.files.Count-eq4) 'Four guarded top-note prepends'
foreach($p in $front.files){
  Verify $p;Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot $p.proposal)
  $current=[IO.File]::ReadAllBytes((Join-Path $researchRoot $p.path));Check ($current.Length-ge$p.prior_bytes) 'Prior whole text retained behind new prepend'
  $tail=[byte[]]::new($p.prior_bytes);[Array]::Copy($current,$current.Length-$p.prior_bytes,$tail,0,$p.prior_bytes)
  $hasher=[Security.Cryptography.SHA256]::Create();try{$tailSHA=[Convert]::ToHexString($hasher.ComputeHash($tail)).ToLowerInvariant()}finally{$hasher.Dispose()}
  Check ($tailSHA-eq$p.prior_sha256) 'Complete prior live text byte-exact after prepend'
}
$metadataMirrorPath='output/engineering/causal-fresh-v6-finalization/attempt-001/metadata-output-mirror-001.json';$metadataMirror=Read $metadataMirrorPath
Check ($metadataMirror.status-eq'passed'-and$metadataMirror.file_count-eq53) 'All53 finalization metadata artifacts mirrored'
foreach($p in $metadataMirror.files){Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot ('output/engineering/causal-fresh-v6-finalization/attempt-001/'+$p.path))}
$preflight=Read 'doc/causal_fresh_v6/protocol-preflight.json'
Check ($preflight.status-eq'passed'-and$preflight.sources.Count-eq101) '101 reviewed sources +preflight102'
foreach($p in $preflight.sources){Verify $p}
foreach($repo in @('cuwacunu_dev','cuwacunu_embedding')){
  $source47=@($old.pins|Where-Object {$_.repository-eq$repo-and$_.path-like'code/encoders/causal_patch_bottleneck_mae/versions/v3/*'})
  Check ($source47.Count-eq47) ('47 selectedV3 sources: '+$repo)
  foreach($p in $source47){Verify $p $repo}
  Verify ([ordered]@{path='output/baselines/CPB-v3.market-01/inference.pt';bytes=1290625;sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'}) $repo
}
$mirror5Path='doc/causal_fresh_v5/complete-mirror-attempt-003.json';$mirror5=Read $mirror5Path;Verify $v5.complete_mirror
Check ($mirror5.file_count-eq355-and$mirror5.bytes-eq134015480) '355 frozenV5 saved files'
$mirror6Path='doc/causal_fresh_v6/complete-mirror-proof-recovery-002-attempt-001.json';$mirror6=Read $mirror6Path
Check ($mirror6.status-eq'passed'-and$mirror6.file_count-eq400-and$mirror6.bytes-eq218709469-and$mirror6.ignored_pt_count-eq307) '400 completeV6 saved files/307PT'
foreach($p in @($mirror5.files)+@($mirror6.files)){Verify $p}
$critical=@($preflight.preservation.critical_checkpoint_files)
$new5=@($mirror6.files|Where-Object {$_.path-like'output/runs/causal-fresh-v6/run-001/checkpoints/*.pt'})
Check ($critical.Count-eq79-and$new5.Count-eq5) '79 historical +five V6 main checkpoints'
foreach($p in $new5){$critical += [ordered]@{repository='cuwacunu_embedding';path=$p.path;bytes=$p.bytes;sha256=$p.sha256}}
foreach($p in $critical){$repo=if($p.repository){$p.repository}else{'cuwacunu_embedding'};Verify $p $repo}
$ignored=@()
foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){
  $paths=@($old.pins|Where-Object {$_.repository-eq$repo-and$_.path.EndsWith('.pt')}|ForEach-Object path)
  if($repo-eq'cuwacunu_embedding'){$paths+=@(@($mirror5.files)+@($mirror6.files)|Where-Object {$_.path.EndsWith('.pt')}|ForEach-Object path)}
  $paths+=@($critical|Where-Object {(-not$_.repository-and$repo-eq'cuwacunu_embedding')-or$_.repository-eq$repo}|ForEach-Object path)
  $paths=@($paths|Sort-Object -Unique);$returned=@();$repoRoot=(Join-Path $workspaceRoot $repo).Replace([char]92,[char]47)
  for($i=0;$i-lt$paths.Count;$i+=128){$end=[Math]::Min($i+127,$paths.Count-1);$batch=@($paths[$i..$end]);$returned+=@(git -c "safe.directory=$repoRoot" -C $repoRoot check-ignore -- @batch);Check ($LASTEXITCODE-eq0) 'Bounded ignored check'}
  Check ($returned.Count-eq$paths.Count-and-not(Compare-Object @($returned|Sort-Object -Unique) $paths)) ('Every pinnedPT ignored: '+$repo)
  $ignored += [ordered]@{repository=$repo;unique_pinned_PT=$paths.Count;all_ignored=$true}
}
# Source ZIPs, including failedV5build001, are checked against their full declared members.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$ZIPs=@()
foreach($spec in @(
  [ordered]@{label='V5 failed001';root='output/engineering/causal-fresh-v5/attempt-001';count=97},
  [ordered]@{label='V5 admitted003';root='output/engineering/causal-fresh-v5/attempt-003';count=97},
  [ordered]@{label='V6 admitted001';root='output/engineering/causal-fresh-v6/attempt-001';count=102}
)){
  $manifestPath=$spec.root+'/source-manifest.json';$manifest=Read $manifestPath;$zipPath=$spec.root+'/source.zip'
  Check ($manifest.files.Count-eq$spec.count) 'Exact source manifest count';$zip=[IO.Compression.ZipFile]::OpenRead((Join-Path $researchRoot $zipPath))
  try{
    Check ($zip.Entries.Count-eq$spec.count+1) 'Exact ZIP membership with manifest'
    foreach($p in $manifest.files){$entry=$zip.GetEntry($p.path);Check ($null-ne$entry-and$entry.Length-eq$p.bytes) 'ZIP source length';$stream=$entry.Open();$hasher=[Security.Cryptography.SHA256]::Create();try{$sha=[Convert]::ToHexString($hasher.ComputeHash($stream)).ToLowerInvariant()}finally{$hasher.Dispose();$stream.Dispose()};Check ($sha-eq$p.sha256) 'ZIP source SHA256'}
    $ZIPs += [ordered]@{label=$spec.label;source_files=$spec.count;ZIP_members=$zip.Entries.Count;archive=(Pin $zipPath);manifest=(Pin $manifestPath)}
  }finally{$zip.Dispose()}
}
$auxiliaryMirrors=@()
foreach($path in @('doc/causal_fresh_v6/quality-audit-attempt001-preservation.json','doc/causal_fresh_v6/quality-audit-mirror-attempt002.json','doc/causal_fresh_v6/training-audit-mirror.json')){
  $m=Read $path;Check ($m.status-eq'passed') 'Auxiliary mirror passed';foreach($p in $m.files){Verify $p};$auxiliaryMirrors += (Pin $path)
}
$durablePath='doc/causal_fresh_v6/durable-json-copies.json';$durable=Read $durablePath
foreach($p in $durable.files){Verify $p};Check ($durable.status-eq'passed') 'Durable16 evidence passed'
$reportRepair=Read 'output/engineering/causal-fresh-v6-preflight-repairs/attempt-001/repair.json';Verify $reportRepair.preserved_before
foreach($p in (Read $reportRepair.preserved_before.path).files){Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot $p.preserved)}
$provenanceRepairPath='output/engineering/causal-fresh-v6-preflight-repairs/attempt-001/before-provenance/manifest.json'
Verify $preflight.prior_provenance_snapshot
foreach($p in (Read $provenanceRepairPath).files){Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot $p.snapshot_path)}
$joinRoot='output/engineering/causal-fresh-v6-establishment/attempt-001';$joinManifest=Read ($joinRoot+'/package/manifest.json')
Verify $joinManifest.source
foreach($p in $joinManifest.inputs){Verify $p;Verify $p 'cuwacunu_embedding' (Join-Path $researchRoot ($joinRoot+'/package/'+$p.staged_path))}
$joinRecoveryPath='output/engineering/causal-fresh-v6-establishment/attempt-002/recovery-plan.json';$joinRecovery=Read $joinRecoveryPath
foreach($p in $joinRecovery.failed_preserved){Verify $p}
$preservedAuxiliary=@();foreach($path in @(
 'output/engineering/causal-fresh-v6-establishment/attempt-001/source.zip',
 'output/engineering/causal-fresh-v6-establishment/attempt-001/package/manifest.json',
 'output/engineering/causal-fresh-v6-establishment/attempt-002/recovery-plan.json',
 'output/engineering/causal-fresh-v6-establishment/attempt-002/execution.json',
 'output/engineering/causal-fresh-v6-establishment/attempt-002/execution.log',
 'output/engineering/causal-fresh-v6-quality-audit/attempt-002/pre-freeze-card-draft-invalid.json',
 'output/engineering/clean-causal-search-2026-10-16/v5-packaging-attempt002-failure.json'
)){$preservedAuxiliary += (Pin $path)}
$evidencePaths=@('doc/CAUSAL_FRESH_V6_CARD.json','doc/CAUSAL_FRESH_V6_CARD.md','doc/causal_fresh_v6/protocol-preflight.json','doc/causal_fresh_v6/independent-training-audit.json','doc/causal_fresh_v6/quality-independent-audit.json','doc/causal_fresh_v6/saved-proof-recovery-attempt002.json','doc/causal_fresh_v6/quality-audit-card.json','doc/causal_fresh_v6/quality-audit-card-attempt002.json','doc/causal_fresh_v6/verify-mirror.ps1','doc/causal_fresh_v6/verify-mirror-proof-recovery-002.ps1','doc/CAUSAL_FRESH_V6_RESULTS.md','doc/CAUSAL_FRESH_V6_DETAIL.md','doc/CAUSAL_FRESH_V6_REVIEW.md','doc/causal_fresh_v6/final-establishment.json')
$evidence=@();foreach($path in $evidencePaths){$evidence += (Pin $path)}
$quality=Read 'doc/causal_fresh_v6/quality-independent-audit.json';$training=Read 'doc/causal_fresh_v6/independent-training-audit.json';$establishment=Read 'doc/causal_fresh_v6/final-establishment.json'
Check ($quality.status-eq'passed'-and$quality.checks-eq12218-and$quality.mismatches.Count-eq0-and$quality.paired_initial_equality.reference_data_archives-eq13-and$quality.paired_initial_equality.all_declared_numeric_tensor_values_exact) 'Quality audit12218 +full40 pairings13'
$trainingTotal=[int64]$training.checks
foreach($name in @('numeric_statistics','saved_tensor_state','manual_saved_Gaussian','actual_pre_step_composite_objective')){Check ($training.$name.status-eq'passed') ('Training module passed: '+$name);$trainingTotal += [int64]$training.$name.checks}
Check ($training.status-eq'passed'-and$training.checks-eq21202-and$trainingTotal-eq270721-and$training.mismatches.Count-eq0) 'Training audit21202core +four modules =270721'
Check ($establishment.status-eq'passed'-and$establishment.trust_admission_passed-and$establishment.numeric_passed-eq12-and-not$establishment.overall_establishment_passed) 'Measured trust separated from12/22 quality'
$ledgerPath='doc/clean_causal_encoder_research_ledger.json';$ledger=Read $ledgerPath
$catalogPath='code/encoders/causal_patch_bottleneck_mae/versions/current.json';$catalog=Read $catalogPath
$oldLedger=Read 'output/engineering/causal-fresh-v6-finalization/attempt-001/prior-live/doc/clean_causal_encoder_research_ledger.json';$oldCatalog=Read 'output/engineering/causal-fresh-v6-finalization/attempt-001/prior-live/code/encoders/causal_patch_bottleneck_mae/versions/current.json'
Check ($ledger.baseline-eq'CPB-v3.market-01'-and-not$ledger.overall_winner-and-not$ledger.checkpoint_deletion_allowed) 'Protected V3 baseline ledger'
Check ($catalog.active_application-eq'CPB-v3.market-01'-and(Equal $oldCatalog.clean_causal_training_baseline $catalog.clean_causal_training_baseline)) 'Complete selected V3 catalog object unchanged'
foreach($field in @('candidates','candidate_status_history','stage_scorecards','leader_history','displaced_leaders','direction_history','stage_leader_update_history','reconstruction_stage_scorecards')){
  $before=@($oldLedger.$field);$after=@($ledger.$field);Check ($after.Count-ge$before.Count) ('History retained: '+$field)
  for($i=0;$i-lt$before.Count;$i++){Check (Equal $before[$i] $after[$i]) ('Exact historical object '+$field+':'+$i)}
}
Check ($ledger.candidates.Count-eq4-and$ledger.stage_scorecards.Count-eq8-and$ledger.reconstruction_stage_scorecards.Count-eq8) 'Four candidates/eight stage cards'
Check ($ledger.stage_leader_update.strictly_improved_scopes-eq8-and$ledger.stage_leader_update.earlier_scopes_retained-eq8) 'Eight new quality leaders/eight retained'
for($i=0;$i-lt16;$i++){Check (Equal $oldLedger.stage_leaders[$i] $ledger.leader_history[$oldLedger.leader_history.Count+$i].record) 'Every prior complete scoped leader retained'}
$devRoot=(Join-Path $workspaceRoot 'cuwacunu_dev').Replace([char]92,[char]47);$devStatus=@(git -c "safe.directory=$devRoot" -C $devRoot status --porcelain=v1 --untracked-files=all);Check ($LASTEXITCODE-eq0) 'Read-only concurrentDEV inventory'
$containerPath='output/engineering/causal-fresh-v6-preservation/attempt-001/container-byte-inspection.json';$container=Read $containerPath
Check ($container.status-eq'passed'-and$container.source_files_checked-eq199-and$container.binary_files_checked-eq12) 'Container97+102 sources/12 binaries exact'
$receipt=[ordered]@{
  kind='causal_fresh_v6_final_preservation_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks
  baseline='CPB-v3.market-01';candidate='CPB-v6.market-01';baseline_changed=$false;application_changed=$false
  prior_V5_final=(Pin $v5Path);old_before_inventory=(Pin $oldPath);old_pins=1030;old_pins_currently_exact=$oldExact;old_authorized_snapshots=$oldResolved
  V5_metadata6_chain=$metadataChain;V5_prior_snapshot=$v5.prior_live_snapshot;V6_prior_snapshot=(Pin $snapshot6Path)
  authorized_metadata_application=(Pin $applicationPath);independent_metadata_proposal_review=(Pin $proposalReviewPath);proposal_generation=(Pin $generationPath);finalization_card=(Pin 'doc/causal_fresh_v6_finalization/finalization-card.json');front_notes=(Pin $frontPath);metadata53_mirror=(Pin $metadataMirrorPath)
  context_cleanup=[ordered]@{historical_context_superseded_by='Guarded top prepends and retained history';old_full_text_preserved=$true;files_deleted=0;new_cleanup_action=0}
  critical_checkpoint_total=84;historical_critical_checkpoints=79;new_main_checkpoints=5;critical_checkpoint_pins=$critical
  selected_V3_source_files_each_repo=47;selected_V3_checkpoint_sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'
  saved_V5_files_verified=355;saved_V6_files_verified=400;V6_saved_bytes=218709469;V6_saved_PT=307;all_pinned_PT_ignored=$ignored
  PT_scope='PT counts include features/heads/witnesses/negative cases and checkpoints; not trained-model counts'
  source_ZIPs=$ZIPs;container_preservation=(Pin $containerPath);complete_V5_mirror=(Pin $mirror5Path);complete_V6_mirror=(Pin $mirror6Path)
  failed_original_proof_preserved=$true;corrected_final_proof_preserved=$true;both_quality_audit_attempts_preserved=$true;auxiliary_mirrors=$auxiliaryMirrors
  preflight_reporter_and_provenance_originals_preserved=$true;metadata_join_failed_and_corrected_invocations_preserved=$true;preserved_auxiliary=$preservedAuxiliary
  durable_evidence=(Pin $durablePath);final_evidence=$evidence;current_ledger=(Pin $ledgerPath);current_catalog=(Pin $catalogPath);current_instance=(Pin 'code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v6.market-01.json')
  numeric_passed=12;numeric_total=22;trust_admission_passed=$true;overall_establishment_passed=$false;overall_winner=$false
  independent_training_core_checks=21202;independent_training_core_plus_four_modules_checks=270721;independent_quality_checks=12218
  concurrency=[ordered]@{repository='cuwacunu_dev';policy='Unrelated concurrent Latent/artifact/Roadmap work inventoried only; no old integration-control equality assertion or edit/reset/delete';git_status=$devStatus}
  limitations=@('Rehashes all1030 old pins plus355V5/400V6 saved files; older bulk evidence outside these explicit inventories relies on immutable earlier receipts','Hashes saved GPU/quality evidence without any model/build/head/training execution')
  actions=[ordered]@{models=0;fits=0;builds=0;training_updates=0;Git_mutations=0;deletions=0;files_written=@('doc/causal_fresh_v6/final-preservation.json')}
}
[IO.File]::WriteAllText($destination,($receipt|ConvertTo-Json -Depth 90).Replace([string][char]13,'')+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';checks=$receipt.checks;bytes=(Get-Item -LiteralPath $destination).Length;sha256=(Hash $destination);critical84=$critical.Count;old1030=$oldExact+$oldResolved.Count;ignored=$ignored}|ConvertTo-Json -Depth 4

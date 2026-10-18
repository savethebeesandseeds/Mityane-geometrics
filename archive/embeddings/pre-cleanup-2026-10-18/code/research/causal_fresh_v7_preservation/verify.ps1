# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
# Owned V7 adaptation of causal_fresh_v6_preservation/verify.ps1 (b5d16d3d...).
# Host file/hash and JSON preservation inspection only. Never builds or loads models.
param(
 [switch]$FinalMetadataApplied,
 [string]$CompleteMirror='doc/causal_fresh_v7/complete-mirror-attempt-001.json',
 [string]$MetadataApplication='doc/causal_fresh_v7_finalization/applied-metadata.json',
 [string]$ProposalReview='doc/causal_fresh_v7_finalization/independent-proposal-review.json',
 [string]$FinalEstablishment='doc/causal_fresh_v7/final-establishment.json'
)
$ErrorActionPreference='Stop'
if(-not $FinalMetadataApplied){throw 'Wait for the explicit root FinalMetadataApplied signal'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$destination=Join-Path $researchRoot 'doc/causal_fresh_v7/final-preservation.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve the existing final receipt; use an additive reviewed repair'}
$script:checks=0;$script:hashes=@{}
function Check([bool]$ok,[string]$why){$script:checks++;if(-not $ok){throw $why}}
function Rooted([string]$path,[string]$repo='cuwacunu_embedding'){
 Check ($repo-in@('cuwacunu_embedding','cuwacunu_dev')) 'Only the two existing repositories are admitted'
 $repoRoot=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repo))
 $resolved=if([IO.Path]::IsPathRooted($path)){[IO.Path]::GetFullPath($path)}else{[IO.Path]::GetFullPath((Join-Path $repoRoot $path))}
 Check ($resolved.StartsWith($repoRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) ('Path escapes repository: '+$path)
 return $resolved
}
function Hash([string]$path){
 $item=Get-Item -LiteralPath $path
 Check (-not($item.Attributes-band[IO.FileAttributes]::ReparsePoint)) ('Unexpected link: '+$path)
 $key=$item.FullName+'|'+$item.Length+'|'+$item.LastWriteTimeUtc.Ticks
 if(-not$script:hashes.ContainsKey($key)){$script:hashes[$key]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
 return $script:hashes[$key]
}
function Read([string]$path){Get-Content -LiteralPath (Rooted $path) -Raw|ConvertFrom-Json}
function Pin([string]$path){$resolved=Rooted $path;$item=Get-Item -LiteralPath $resolved;[ordered]@{path=$path;bytes=$item.Length;sha256=(Hash $resolved)}}
function Verify($pin,[string]$repo='cuwacunu_embedding',[string]$override=''){
 $path=if($override){Rooted $override $repo}else{Rooted $pin.path $repo}
 $item=Get-Item -LiteralPath $path
 Check ($item.Length-eq$pin.bytes-and(Hash $path)-eq$pin.sha256) ('Frozen bytes changed: '+$repo+'/'+$pin.path)
}
function Equal($left,$right){($left|ConvertTo-Json -Depth 90 -Compress)-ceq($right|ConvertTo-Json -Depth 90 -Compress)}
function Prefix($before,$after,[string]$label){
 $oldItems=@($before);$newItems=@($after)
 Check ($newItems.Count-ge$oldItems.Count) ('History shrank: '+$label)
 for($i=0;$i-lt$oldItems.Count;$i++){Check (Equal $oldItems[$i] $newItems[$i]) ('Prior whole object changed: '+$label+':'+$i)}
}
function Verify-Historical($pin){
 $repo=if($pin.repository){$pin.repository}else{'cuwacunu_embedding'}
 if($repo-eq'cuwacunu_embedding'-and$script:prior7.ContainsKey($pin.path)){
  $former=$script:prior7[$pin.path]
  if($pin.bytes-eq$former.bytes-and$pin.sha256-eq$former.sha256){Verify $pin $repo $former.snapshot;return}
 }
 if($repo-eq'cuwacunu_embedding'-and$script:oldResolved.ContainsKey($pin.path)){
  $former=$script:oldResolved[$pin.path]
  if($pin.bytes-eq$former.old_bytes-and$pin.sha256-eq$former.old_sha256){Verify $pin $repo $former.snapshot;return}
 }
 Verify $pin $repo
}
function Verify-PinTree($value){
 if($null-eq$value-or$value-is[string]-or$value.GetType().IsValueType){return}
 if($value-is[Collections.IEnumerable]-and$value-isnot[pscustomobject]-and$value-isnot[Collections.IDictionary]){foreach($child in $value){Verify-PinTree $child};return}
 $names=@($value.PSObject.Properties.Name)
 if('path'-in$names-and'bytes'-in$names-and'sha256'-in$names){Verify-Historical $value;return}
 foreach($property in $value.PSObject.Properties){Verify-PinTree $property.Value}
}

# Preserve the existing receipt and resolve its original live metadata through exact snapshots.
$v6Path='doc/causal_fresh_v6/final-preservation.json'
Check ((Hash (Rooted $v6Path))-eq'aa99ac04f3c4bd013c617450c6fde2f99e96c041e6b4c8f5465e2d551abe42ce') 'Frozen V6 final preservation receipt'
$v6=Read $v6Path
Check ($v6.status-eq'passed'-and$v6.critical_checkpoint_pins.Count-eq84-and$v6.old_pins-eq1030) 'V6 preservation scope84/1030'
$snapshotPath='output/engineering/causal-fresh-v7-finalization/attempt-001/prior-live/manifest.json'
Check ((Hash (Rooted $snapshotPath))-eq'14a1d32feb36673436372808ffbd684de2975e5091209ad7a1b936dd19439da5') 'Frozen six-file pre-V7 metadata snapshot'
$snapshot=Read $snapshotPath
Check ($snapshot.status-eq'passed'-and$snapshot.files.Count-eq6) 'Six prior-live metadata files'
$script:prior7=@{};$script:oldResolved=@{}
foreach($pin in $snapshot.files){Check (-not$script:prior7.ContainsKey($pin.path)) 'Unique prior-live metadata path';$script:prior7[$pin.path]=$pin;Verify $pin 'cuwacunu_embedding' $pin.snapshot}
foreach($pin in $v6.old_authorized_snapshots){$script:oldResolved[$pin.path]=$pin}
foreach($pin in $v6.V5_metadata6_chain){
 Check ($script:prior7.ContainsKey($pin.path)) 'Every V6 current metadata object has a V7 prior snapshot'
 Check ($pin.current.bytes-eq$script:prior7[$pin.path].bytes-and$pin.current.sha256-eq$script:prior7[$pin.path].sha256) 'Exact V6-current to V7-prior metadata chain'
 Verify ([ordered]@{path=$pin.path;bytes=$pin.prior_bytes;sha256=$pin.prior_sha256}) 'cuwacunu_embedding' $pin.prior_snapshot
}
Verify-PinTree $v6
$oldPath=$v6.old_before_inventory.path;Verify $v6.old_before_inventory;$old=Read $oldPath
Check ($old.pins.Count-eq1030) 'All1030 original inventory entries'
foreach($pin in $old.pins){Verify-Historical $pin}

# Immutable V6 source/card/protocol and all completed saved artifacts remain available.
$v6Preflight=Read 'doc/causal_fresh_v6/protocol-preflight.json'
Check ($v6Preflight.status-eq'passed'-and$v6Preflight.sources.Count-eq101) '101 reviewed V6 sources'
foreach($pin in $v6Preflight.sources){Verify $pin}
$mirror6=Read $v6.complete_V6_mirror.path;Verify $v6.complete_V6_mirror
Check ($mirror6.status-eq'passed'-and$mirror6.files.Count-eq400-and$mirror6.file_count-eq400-and$mirror6.ignored_pt_count-eq307) '400 immutable V6 mirror files/307PT'
foreach($pin in $mirror6.files){Verify $pin}
foreach($pin in $v6.preserved_auxiliary){Verify $pin}
foreach($binding in $v6.auxiliary_mirrors){Verify $binding;$receipt=Read $binding.path;foreach($pin in $receipt.files){Verify $pin}}
Verify $v6Preflight.prior_provenance_snapshot
foreach($pin in (Read $v6Preflight.prior_provenance_snapshot.path).files){Verify $pin 'cuwacunu_embedding' $pin.snapshot_path}
$reportRepair=Read 'output/engineering/causal-fresh-v6-preflight-repairs/attempt-001/repair.json'
Verify $reportRepair.preserved_before
foreach($pin in (Read $reportRepair.preserved_before.path).files){Verify $pin 'cuwacunu_embedding' $pin.preserved}
$joinRecovery=Read 'output/engineering/causal-fresh-v6-establishment/attempt-002/recovery-plan.json'
foreach($pin in $joinRecovery.failed_preserved){Verify $pin}
foreach($binding in @($v6.metadata53_mirror,$v6.durable_evidence)){
 Verify $binding;$receipt=Read $binding.path
 foreach($pin in $receipt.files){
  if($binding.path-eq$v6.metadata53_mirror.path){Verify $pin 'cuwacunu_embedding' ('output/engineering/causal-fresh-v6-finalization/attempt-001/'+$pin.path)}else{Verify $pin}
 }
}

# Selected V3 is unchanged in both repositories, irrespective of concurrent dev integration work.
foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){
 $sources=@($old.pins|Where-Object {$_.repository-eq$repo-and$_.path-like'code/encoders/causal_patch_bottleneck_mae/versions/v3/*'})
 Check ($sources.Count-eq47) ('47 selected V3 sources: '+$repo)
 foreach($pin in $sources){Verify $pin $repo}
 Verify ([ordered]@{path='output/baselines/CPB-v3.market-01/inference.pt';bytes=1290625;sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'}) $repo
}

# The new Complete receipt supplies actual artifact hashes; no filenames or PT counts are invented.
$mirror=Read $CompleteMirror
Check ($mirror.status-eq'passed'-and$mirror.scope-eq'Complete'-and$mirror.attempt-eq'attempt-001'-and$mirror.source_files_verified-eq106-and$mirror.all_mirrored_pt_ignored) 'Completed V7 source106 artifact mirror'
Check ($mirror.file_count-eq$mirror.files.Count) 'Exact V7 mirror membership'
foreach($pin in $mirror.files){Verify $pin}
foreach($pin in $mirror.preserved_source_packages){Verify $pin}
$new5=@($mirror.files|Where-Object {$_.path-like'output/runs/causal-fresh-v7/run-001/checkpoints/*.pt'})
$checkpointNames=@('inference-0.pt','inference-2048.pt','training-512.pt','training-1024.pt','training-2048.pt')
Check ($new5.Count-eq5-and-not(Compare-Object @($new5|ForEach-Object {Split-Path -Leaf $_.path}|Sort-Object) @($checkpointNames|Sort-Object))) 'Exactly five V7 main checkpoints'
$critical=@();$legacyV5=@($checkpointNames|ForEach-Object {'output/runs/causal-fresh-v5/run-001/checkpoints/'+$_});$missingRepository=0
foreach($pin in $v6.critical_checkpoint_pins){
 $repo=$pin.repository
 if(-not$repo){Check ($pin.path-in$legacyV5) 'Missing repository allowed only on exact known V5 main checkpoint paths';$repo='cuwacunu_embedding';$missingRepository++}
 Verify $pin $repo
 $critical += [ordered]@{repository=$repo;path=$pin.path;bytes=$pin.bytes;sha256=$pin.sha256}
}
Check ($missingRepository-eq5) 'Exactly five known V5 legacy pins omit repository'
foreach($pin in $new5){$critical += [ordered]@{repository='cuwacunu_embedding';path=$pin.path;bytes=$pin.bytes;sha256=$pin.sha256}}
Check ($critical.Count-eq89-and@($critical|ForEach-Object {$_.repository+'/'+$_.path}|Sort-Object -Unique).Count-eq89) '89 unique protected checkpoints'
$ignored=@()
foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){
 $paths=@($old.pins|Where-Object {$_.repository-eq$repo-and$_.path.EndsWith('.pt')}|ForEach-Object path)
 $paths+=@($critical|Where-Object repository -EQ $repo|ForEach-Object path)
 if($repo-eq'cuwacunu_embedding'){$paths+=@(@($mirror6.files)+@($mirror.files)|Where-Object {$_.path.EndsWith('.pt')}|ForEach-Object path)}
 $paths=@($paths|Sort-Object -Unique);$returned=@();$repoRoot=(Join-Path $workspaceRoot $repo).Replace([char]92,[char]47)
 for($i=0;$i-lt$paths.Count;$i+=128){$end=[Math]::Min($i+127,$paths.Count-1);$batch=@($paths[$i..$end]);$returned+=@(& git -c ('safe.directory='+$repoRoot) -C $repoRoot check-ignore -- @batch);Check ($LASTEXITCODE-eq0) 'Bounded ignored-PT inspection'}
 Check ($returned.Count-eq$paths.Count-and-not(Compare-Object @($returned|Sort-Object -Unique) $paths)) ('Every pinned PT remains ignored: '+$repo)
 $ignored += [ordered]@{repository=$repo;unique_pinned_PT=$paths.Count;all_ignored=$true}
}
Check (@($mirror.files|Where-Object {$_.path.EndsWith('.pt')}).Count-eq$mirror.ignored_pt_count) 'Every V7 mirrored PT counted and ignored'

# Check ZIP contents in addition to whole archives, including the preserved failed V5 preparation.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipSpecs=@($v6.source_ZIPs)+@([ordered]@{label='V7 admitted001';source_files=106;archive=(Pin 'output/engineering/causal-fresh-v7/attempt-001/source.zip');manifest=(Pin 'output/engineering/causal-fresh-v7/attempt-001/source-manifest.json')})
$zipChecks=@()
foreach($spec in $zipSpecs){
 Verify $spec.archive;Verify $spec.manifest;$manifest=Read $spec.manifest.path
 Check ($manifest.files.Count-eq$spec.source_files) 'Exact source manifest membership'
 $zip=[IO.Compression.ZipFile]::OpenRead((Rooted $spec.archive.path))
 try{
  Check ($zip.Entries.Count-eq$spec.source_files+1-and@($zip.Entries.FullName|Sort-Object -Unique).Count-eq$zip.Entries.Count) 'ZIP exact unique source members plus manifest'
  foreach($pin in $manifest.files){
   $entry=$zip.GetEntry($pin.path);Check ($null-ne$entry-and$entry.Length-eq$pin.bytes) ('ZIP member bytes: '+$pin.path)
   $stream=$entry.Open();$hasher=[Security.Cryptography.SHA256]::Create()
   try{$actual=[Convert]::ToHexString($hasher.ComputeHash($stream)).ToLowerInvariant()}finally{$hasher.Dispose();$stream.Dispose()}
   Check ($actual-eq$pin.sha256) ('ZIP member hash: '+$pin.path)
  }
  $zipChecks += [ordered]@{label=$spec.label;source_files=$spec.source_files;archive=$spec.archive;manifest=$spec.manifest;all_member_hashes_exact=$true}
 }finally{$zip.Dispose()}
}
$mainManifest=Read 'output/engineering/causal-fresh-v7/attempt-001/source-manifest.json'
Check ((Hash (Rooted 'output/engineering/causal-fresh-v7/attempt-001/source-manifest.json'))-eq'09c151d24eb91ea4ce8e653601aeee9d5e42d098df2ecd7204af92e528d67f80'-and(Hash (Rooted 'output/engineering/causal-fresh-v7/attempt-001/source.zip'))-eq'ac43ef587383cbe3d610a5f73736a4d10cc852a64ebca249602f37db4b716999') 'Immutable V7 source106 package'
foreach($pin in $mainManifest.files){Verify $pin}
$buildPins=@($mirror.files|Where-Object {$_.path.EndsWith('/build-admission.json')})
Check ($buildPins.Count-eq1) 'One actual completed V7 build admission'
$build=Read $buildPins[0].path
Check ($build.status-eq'passed'-and$build.source_files-eq106-and$build.binaries.Count-eq6-and@($build.dependency_closures.PSObject.Properties).Count-eq10) 'Six binaries and ten translation units from source106'
Check ($build.source_manifest.sha256-eq'09c151d24eb91ea4ce8e653601aeee9d5e42d098df2ecd7204af92e528d67f80'-and$build.source_manifest.bytes-eq(Get-Item -LiteralPath (Rooted 'output/engineering/causal-fresh-v7/attempt-001/source-manifest.json')).Length) 'Build binds the exact source manifest bytes'
$expectedBinaries=@('test_causal_patch_bottleneck_mae_v7','test_cpb_v7_variance_gradient','causal_fresh_v7_train','causal_fresh_v7_extract','causal_fresh_v7_market_read','causal_fresh_v7_synthetic_read')
Check (-not(Compare-Object @($build.binaries|ForEach-Object {Split-Path -Leaf $_.path}|Sort-Object) @($expectedBinaries|Sort-Object))) 'Exact admitted binary set'

# Read-only availability/hash inspection in the existing container; no model executable is invoked.
$containerID='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$inspection=@(& docker inspect cuwacunu);Check ($LASTEXITCODE-eq0) 'Existing container can be inspected';$container=($inspection-join"`n"|ConvertFrom-Json)[0]
Check ($container.Id-eq$containerID-and$container.State.Running) 'Exact existing managed running container'
$containerInputs=@($mainManifest.files|ForEach-Object {[ordered]@{path=('/opt/cuwacunu/research/causal-fresh-v7/attempt-001/source/'+$_.path);bytes=$_.bytes;sha256=$_.sha256}})+@($build.source_manifest)+@($build.binaries)
$payload=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($containerInputs|ConvertTo-Json -Depth 6 -Compress)))
$containerScript=@'
import base64,hashlib,json
from pathlib import Path
items=json.loads(base64.b64decode('__INPUTS__'))
for item in items:
    p=Path(item['path'])
    if p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes']:
        raise RuntimeError('Missing or resized frozen file: '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    if h.hexdigest()!=item['sha256']: raise RuntimeError('Changed frozen file: '+str(p))
print(json.dumps({'status':'passed','source_files':106,'source_manifest_checked':True,'binary_files':6,'model_calls':0,'head_fits':0}))
'@
$containerOut=@($containerScript.Replace('__INPUTS__',$payload)|& docker exec -i cuwacunu python3 -B -)
Check ($LASTEXITCODE-eq0) 'Container source/binary byte inspection'
$containerBytes=($containerOut-join"`n"|ConvertFrom-Json)
Check ($containerBytes.status-eq'passed'-and$containerBytes.source_files-eq106-and$containerBytes.binary_files-eq6) 'All source106/six executable files remain available'

# Finalization is metadata-only. Require prior whole objects, not merely best-score scalars.
$application=Read $MetadataApplication;$review=Read $ProposalReview;$establishment=Read $FinalEstablishment
Check ($application.status-eq'passed'-and-not$application.baseline_changed-and-not$application.application_changed) 'Authorized metadata application preserves baseline/application'
foreach($pin in $application.files){Verify $pin;if($pin.source){Verify $pin 'cuwacunu_embedding' $pin.source}}
Check ($review.status-eq'passed') 'Independent proposal review passed'
foreach($pin in $review.metadata_outputs){Verify $pin}
Check ($establishment.status-eq'passed'-and$establishment.candidate-eq'CPB-v7.market-01'-and$establishment.measured_candidate_established-and$establishment.trust_admission_passed-and$establishment.full_coverage-and$establishment.initial_pairing_count-eq13-and$establishment.numeric_total-eq22) 'Measured trust and22 numeric outcomes remain separate'
$ledgerPath='doc/clean_causal_encoder_research_ledger.json';$catalogPath='code/encoders/causal_patch_bottleneck_mae/versions/current.json'
$ledger=Read $ledgerPath;$catalog=Read $catalogPath
$priorLedger=Read $script:prior7[$ledgerPath].snapshot;$priorCatalog=Read $script:prior7[$catalogPath].snapshot
Check ($ledger.baseline-eq'CPB-v3.market-01'-and-not$ledger.checkpoint_deletion_allowed-and$catalog.active_application-eq'CPB-v3.market-01') 'Selected V3 baseline/application and no checkpoint deletion'
Check (Equal $priorCatalog.clean_causal_training_baseline $catalog.clean_causal_training_baseline) 'Whole selected V3 catalog entry unchanged'
Prefix $priorCatalog.fresh_research_candidate_history $catalog.fresh_research_candidate_history 'catalog candidate history'
Check (@($catalog.fresh_research_candidate_history|Where-Object {Equal $_ $priorCatalog.fresh_research_candidate}).Count-eq1) 'Whole former V6 catalog entry retained'
Check ($catalog.fresh_research_candidate.instance-eq'CPB-v7.market-01') 'V7 measured candidate catalog entry'
foreach($field in @('candidates','candidate_status_history','stage_scorecards','leader_history','displaced_leaders','direction_history','stage_leader_update_history','reconstruction_stage_scorecards','reconstruction_leader_history')){Prefix $priorLedger.$field $ledger.$field $field}
Check ($ledger.candidates.Count-eq$priorLedger.candidates.Count+1-and$ledger.stage_scorecards.Count-eq$priorLedger.stage_scorecards.Count+2-and$ledger.reconstruction_stage_scorecards.Count-eq$priorLedger.reconstruction_stage_scorecards.Count+2) 'One candidate and both stage scorecards appended'
foreach($prior in $priorLedger.stage_leaders){Check (@($ledger.leader_history|Where-Object {Equal $_.record $prior}).Count-ge1) 'Every complete prior quality leader preserved in history'}
foreach($prior in $priorLedger.reconstruction_stage_leaders){
 $matches=@(foreach($entry in $ledger.reconstruction_leader_history){foreach($record in $entry.records){if(Equal $record $prior){$record}}})
 Check ($matches.Count-ge1) 'Every complete prior reconstruction leader preserved in history'
}
foreach($pin in $snapshot.files|Where-Object {$_.path.EndsWith('.md')}){
 $current=[IO.File]::ReadAllBytes((Rooted $pin.path));Check ($current.Length-ge$pin.bytes) 'Prior full text length retained'
 $tail=[byte[]]::new($pin.bytes);[Array]::Copy($current,$current.Length-$pin.bytes,$tail,0,$pin.bytes)
 $hasher=[Security.Cryptography.SHA256]::Create();try{$tailSHA=[Convert]::ToHexString($hasher.ComputeHash($tail)).ToLowerInvariant()}finally{$hasher.Dispose()}
 Check ($tailSHA-eq$pin.sha256) 'Whole prior front text retained behind new prepend'
}
$devRoot=(Join-Path $workspaceRoot 'cuwacunu_dev').Replace([char]92,[char]47)
$devStatus=@(& git -c ('safe.directory='+$devRoot) -C $devRoot status --porcelain=v1 --untracked-files=all);Check ($LASTEXITCODE-eq0) 'Read-only concurrent dev inventory'
$receipt=[ordered]@{
 kind='causal_fresh_v7_final_preservation_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks
 baseline='CPB-v3.market-01';candidate='CPB-v7.market-01';baseline_changed=$false;application_changed=$false
 prior_V6_final=(Pin $v6Path);old_before_inventory=(Pin $oldPath);old_pins=1030;old_authorized_snapshots=$v6.old_authorized_snapshots
 V7_prior_snapshot=(Pin $snapshotPath);prior_whole_metadata_chain_preserved=$true
 critical_checkpoint_total=89;historical_critical_checkpoints=84;new_main_checkpoints=5;critical_checkpoint_pins=$critical
 legacy_V5_repository_field_resolution='Only five exact expected V5 main checkpoint paths default to cuwacunu_embedding';legacy_V5_missing_repository_count=$missingRepository
 selected_V3_source_files_each_repo=47;selected_V3_checkpoint_sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'
 complete_V6_mirror=$v6.complete_V6_mirror;complete_V7_mirror=(Pin $CompleteMirror);saved_V6_files_verified=400;saved_V7_files_verified=$mirror.files.Count
 all_pinned_PT_ignored=$ignored;V7_saved_PT=$mirror.ignored_pt_count;PT_scope='Includes heads/features/witnesses/tests/checkpoints; not trained-encoder count'
 source_ZIPs=$zipChecks;V6_reviewed_sources=101;V7_source_files=106;V7_binary_files=6;container_preservation=$containerBytes;container_id=$containerID
 old_failed_preparations_preserved=$true;prior_V6_receipt_and_all_referenced_pins_verified=$true
 authorized_metadata_application=(Pin $MetadataApplication);independent_metadata_proposal_review=(Pin $ProposalReview);final_establishment=(Pin $FinalEstablishment)
 current_ledger=(Pin $ledgerPath);current_catalog=(Pin $catalogPath);current_instance=(Pin 'code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v7.market-01.json')
 numeric_passed=$establishment.numeric_passed;numeric_total=22;trust_admission_passed=$establishment.trust_admission_passed;overall_establishment_passed=$establishment.overall_establishment_passed
 source=[ordered]@{path='code/research/causal_fresh_v7_preservation/verify.ps1';bytes=(Get-Item -LiteralPath $PSCommandPath).Length;sha256=(Hash $PSCommandPath);parent='code/research/causal_fresh_v6_preservation/verify.ps1';parent_sha256='b5d16d3dafceecce9c4a883f6889504016fbbfc97b46bc554ad13239fb2eabd3'}
 concurrency=[ordered]@{repository='cuwacunu_dev';policy='Read-only status; no concurrent source/config reset or old integration equality claim';git_status=$devStatus}
 limitations=@('File/hash preservation and saved metadata object checks only; no quality metric recalculation','No model/build/head/encoder/optimizer execution; existing-container Python hashes bytes only')
 actions=[ordered]@{models=0;fits=0;builds=0;training_updates=0;Git_mutations=0;deletions=0;files_written=@('doc/causal_fresh_v7/final-preservation.json')}
}
[IO.File]::WriteAllText($destination,($receipt|ConvertTo-Json -Depth 90).Replace([string][char]13,'')+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';checks=$receipt.checks;critical89=$critical.Count;old1030=$old.pins.Count;receipt=(Pin 'doc/causal_fresh_v7/final-preservation.json')}|ConvertTo-Json -Depth 5

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
# Complete physical V8 verifier copy, with one additional preservation-chain level.
# File/hash/JSON inspection only; no model, encoder, optimizer, head or build execution.
param([Parameter(Mandatory=$true)][string]$PreservationCard,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PreservationCardSHA256,
 [switch]$FinalMetadataApplied)
$ErrorActionPreference='Stop'
if(-not $FinalMetadataApplied){throw 'Wait for the explicit root FinalMetadataApplied signal'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$destination=Join-Path $researchRoot 'doc/causal_fresh_v9/final-preservation.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve the prior receipt; declare additive repairs'}
$script:checks=0;$script:hashes=@{};$script:snapshots=@{};$script:concurrent=@{};$script:observedConcurrent=@{}
function Check([bool]$ok,[string]$why){$script:checks++;if(-not $ok){throw $why}}
function Rooted([string]$path,[string]$repo='cuwacunu_embedding'){
 Check ($repo-in@('cuwacunu_embedding','cuwacunu_dev')) 'Only two exact existing repositories'
 $base=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repo))
 $resolved=if([IO.Path]::IsPathRooted($path)){[IO.Path]::GetFullPath($path)}else{[IO.Path]::GetFullPath((Join-Path $base $path))}
 Check ($resolved.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) ('Unbounded repository path: '+$path)
 $resolved
}
function Hash([string]$path){
 $f=Get-Item -LiteralPath $path;Check (-not($f.Attributes-band[IO.FileAttributes]::ReparsePoint)) ('Unexpected file link: '+$path)
 $key=$f.FullName+'|'+$f.Length+'|'+$f.LastWriteTimeUtc.Ticks
 if(-not$script:hashes.ContainsKey($key)){$script:hashes[$key]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
 $script:hashes[$key]
}
function Read([string]$path){Get-Content -Raw -LiteralPath (Rooted $path)|ConvertFrom-Json}
function Pin([string]$path){$resolved=Rooted $path;$f=Get-Item -LiteralPath $resolved;[ordered]@{path=$path;bytes=$f.Length;sha256=(Hash $resolved)}}
function Verify($pin,[string]$repo='cuwacunu_embedding',[string]$override=''){
 $path=Rooted $(if($override){$override}else{$pin.path}) $repo;$f=Get-Item -LiteralPath $path
 Check ($f.Length-eq$pin.bytes-and(Hash $path)-eq$pin.sha256) ('Frozen bytes changed: '+$repo+'/'+$pin.path)
}
function Key($pin,[string]$repo){$repo+'|'+$pin.path+'|'+$pin.bytes+'|'+$pin.sha256}
function Add-Snapshot($pin,[string]$snapshot,[string]$repo='cuwacunu_embedding'){
 Verify $pin $repo $snapshot;$key=Key $pin $repo
 if(-not$script:snapshots.ContainsKey($key)){$script:snapshots[$key]=$snapshot}
}
function Verify-Historical($pin){
 $repo=if($pin.repository){$pin.repository}else{'cuwacunu_embedding'};$key=Key $pin $repo
 if($script:snapshots.ContainsKey($key)){Verify $pin $repo $script:snapshots[$key]}
 elseif($script:concurrent.ContainsKey($key)){
  $path=Rooted $pin.path $repo;$f=Get-Item -LiteralPath $path
  $script:observedConcurrent[$key]=[ordered]@{repository=$repo;path=$pin.path;historical_bytes=$pin.bytes;historical_sha256=$pin.sha256;current_bytes=$f.Length;current_sha256=(Hash $path);reason=$script:concurrent[$key].reason}
 }else{Verify $pin $repo}
}
function Verify-PinTree($value){
 if($null-eq$value-or$value-is[string]-or$value.GetType().IsValueType){return}
 if($value-is[Collections.IEnumerable]-and$value-isnot[pscustomobject]-and$value-isnot[Collections.IDictionary]){foreach($child in $value){Verify-PinTree $child};return}
 $names=@($value.PSObject.Properties.Name)
 if('path'-in$names-and'bytes'-in$names-and'sha256'-in$names){Verify-Historical $value;return}
 foreach($property in $value.PSObject.Properties){Verify-PinTree $property.Value}
}
function Equal($left,$right){($left|ConvertTo-Json -Depth 90 -Compress)-ceq($right|ConvertTo-Json -Depth 90 -Compress)}
function Prefix($before,$after,[string]$label){
 $old=@($before);$new=@($after);Check ($new.Count-ge$old.Count) ('History shrank: '+$label)
 for($i=0;$i-lt$old.Count;$i++){Check (Equal $old[$i] $new[$i]) ('Prior whole record changed: '+$label+':'+$i)}
}

# The root freezes this card after actual mirrors, audits, join and metadata application.
# Required pins: prior_V8_final,prior_live_snapshot,complete_mirror,source_manifest,
# source_zip,main_card,metadata_application,proposal_review,final_establishment,
# compiled_binary_mirror and its seven unique separately mirrored binary pins.
# Required recovery card/receipt/failure mirror retain the failed original
# quality/report stages and the complete separately mirrored recovery leaf.
# Optional explicit protected_pins and protected_inventories expand audit/failure leaves.
Check ((Hash (Rooted $PreservationCard))-eq$PreservationCardSHA256) 'Externally reviewed preservation card before parse'
$declaration=Read $PreservationCard
Check ($declaration.kind-eq'causal_fresh_v9_final_preservation_card_v1'-and$declaration.authorized-eq$true) 'Explicit frozen preservation admission'
Verify $declaration.verifier;Verify $declaration.generator;Verify $declaration.parent_template
Check ((Hash $PSCommandPath)-eq$declaration.verifier.sha256) 'Exact independently reviewed verifier source'
Verify $declaration.prior_V8_final
Check ($declaration.prior_V8_final.sha256-eq'919b4f11610e3ab61b63afbc3612145506e4c47a3be3c777f119d47c49e7126a') 'Exact prior V8 preservation receipt'
$v8=Read $declaration.prior_V8_final.path
Check ($v8.status-eq'passed'-and$v8.critical_checkpoint_pins.Count-eq94-and$v8.old_pins-eq1030) 'Actual94 inherited checkpoints and1030 historical pins; V9 not counted before existence'
Verify $v8.prior_V7_final
$v7=Read $v8.prior_V7_final.path
Check ($v7.status-eq'passed'-and$v7.critical_checkpoint_pins.Count-eq89-and$v7.old_pins-eq1030) 'Prior89 critical checkpoints and1030 historical pins'
Verify $v7.prior_V6_final;$v6=Read $v7.prior_V6_final.path
# Only explicitly declared concurrent DEV integration/Roadmap controls may differ;
# selected V3 sources/checkpoints are always independently verified without this bypass.
Check (@($declaration.concurrent_DEV_exclusions).Count-le6) 'At most six declared prior DEV integration controls'
foreach($p in $declaration.concurrent_DEV_exclusions){
 Check ($p.repository-eq'cuwacunu_dev'-and$p.reason-and-not$p.path.EndsWith('.pt')-and$p.path-notlike'code/encoders/causal_patch_bottleneck_mae/versions/v3/*') 'Bounded concurrent DEV exclusion; never baseline/model files'
 $script:concurrent[(Key $p $p.repository)]=$p
}
# Historical metadata aliases use full(path,size,hash), retaining every generation.
foreach($p in $v6.old_authorized_snapshots){Add-Snapshot ([ordered]@{path=$p.path;bytes=$p.old_bytes;sha256=$p.old_sha256}) $p.snapshot}
foreach($p in $v6.V5_metadata6_chain){Add-Snapshot ([ordered]@{path=$p.path;bytes=$p.prior_bytes;sha256=$p.prior_sha256}) $p.prior_snapshot}
Verify $v7.V7_prior_snapshot;$v7Snapshot=Read $v7.V7_prior_snapshot.path
foreach($p in $v7Snapshot.files){Add-Snapshot $p $p.snapshot}
Verify $v8.V8_prior_snapshot;$v8Snapshot=Read $v8.V8_prior_snapshot.path
foreach($p in $v8Snapshot.files){Add-Snapshot $p $p.snapshot}
Verify $declaration.prior_live_snapshot;$priorSnapshot=Read $declaration.prior_live_snapshot.path
Check ($priorSnapshot.status-eq'passed'-and$priorSnapshot.files.Count-eq6) 'Six whole pre-V9 live files preserved'
foreach($p in $priorSnapshot.files){Add-Snapshot $p $p.snapshot}
foreach($p in @($v8.current_ledger,$v8.current_catalog)){
 Check (@($priorSnapshot.files|Where-Object {$_.path-eq$p.path-and$_.bytes-eq$p.bytes-and$_.sha256-eq$p.sha256}).Count-eq1) 'Exact V8-current to pre-V9 snapshot chain'
}
Verify-PinTree $v6;Verify-PinTree $v7;Verify-PinTree $v8;Verify $v8.old_before_inventory
$old=Read $v7.old_before_inventory.path;Check ($old.pins.Count-eq1030) 'Every historical inventory pin retained'
foreach($p in $old.pins){Verify-Historical $p}
foreach($p in $declaration.concurrent_DEV_exclusions){Check (@($old.pins|Where-Object {$_.repository-eq$p.repository-and$_.path-eq$p.path-and$_.bytes-eq$p.bytes-and$_.sha256-eq$p.sha256}).Count-eq1) 'Concurrent exclusion matches exact historical inventory entry'}

# Expand every original V5/V6/V7 Complete mirror without inventing new artifact counts.
$oldMirrors=@($v6.complete_V5_mirror,$v6.complete_V6_mirror,$v7.complete_V7_mirror,$v8.complete_V8_mirror);$oldMirrorFiles=@();$oldMirrorSummary=@()
foreach($binding in $oldMirrors){
 Verify $binding;$receipt=Read $binding.path
 Check ($receipt.status-eq'passed'-and$receipt.file_count-eq$receipt.files.Count) 'Exact old Complete mirror membership'
 foreach($p in $receipt.files){Verify $p};$oldMirrorFiles+=@($receipt.files)
 $oldMirrorSummary+=[ordered]@{receipt=$binding;files=$receipt.files.Count;ignored_PT=$receipt.ignored_pt_count}
}
foreach($p in $v6.preserved_auxiliary){Verify $p}
foreach($binding in $v6.auxiliary_mirrors){Verify $binding;$r=Read $binding.path;foreach($p in $r.files){Verify $p}}
foreach($binding in @($v6.metadata53_mirror,$v6.durable_evidence)){
 Verify $binding;$r=Read $binding.path
 foreach($p in $r.files){if($binding.path-eq$v6.metadata53_mirror.path){Verify $p 'cuwacunu_embedding' ('output/engineering/causal-fresh-v6-finalization/attempt-001/'+$p.path)}else{Verify $p}}
}
foreach($path in @('doc/causal_fresh_v6/protocol-preflight.json','doc/causal_fresh_v7/protocol-preflight.json','doc/causal_fresh_v8/protocol-preflight.json')){
 $r=Read $path;Check ($r.status-eq'passed') 'Preserved prior admitted source closure';foreach($p in $r.sources){Verify-Historical $p}
}
# Each extra inventory is an explicit pinned JSON receipt plus an exact field,
# optional entry_pin_field (e.g.host) and optional bounded repository-relative prefix.
$expanded=@()
foreach($inventory in $declaration.protected_inventories){
 Verify $inventory.receipt;$r=Read $inventory.receipt.path;$entries=$r.($inventory.inventory_field)
 Check ($null-ne$entries) 'Declared saved inventory field exists'
 foreach($entry in $entries){
  $p=if($inventory.entry_pin_field){$entry.($inventory.entry_pin_field)}else{$entry}
  Check ($null-ne$p.path-and$null-ne$p.bytes-and$null-ne$p.sha256) 'Inventory member is an explicit file pin'
  $repo=if($p.repository){$p.repository}else{'cuwacunu_embedding'}
  if($inventory.prefix){Verify $p $repo ($inventory.prefix.TrimEnd('/')+'/'+$p.path)}else{Verify-Historical $p}
 }
 $expanded+=[ordered]@{receipt=$inventory.receipt;members=@($entries).Count;inventory_field=$inventory.inventory_field}
}
foreach($p in $declaration.protected_pins){Verify-Historical $p}

foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){
 $sources=@($old.pins|Where-Object {$_.repository-eq$repo-and$_.path-like'code/encoders/causal_patch_bottleneck_mae/versions/v3/*'})
 Check ($sources.Count-eq47) ('47 physical selected V3 sources: '+$repo);foreach($p in $sources){Verify $p $repo}
 Verify ([ordered]@{path='output/baselines/CPB-v3.market-01/inference.pt';bytes=1290625;sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'}) $repo
}
Verify $declaration.complete_mirror;$mirror=Read $declaration.complete_mirror.path
Check ($mirror.kind-eq'causal_fresh_v9_container_to_local_evidence_mirror_v1'-and$mirror.status-eq'passed'-and$mirror.scope-eq'Complete'-and$mirror.attempt-eq'attempt-001'-and$mirror.source_files_verified-eq$declaration.main_source_files-and$mirror.all_mirrored_pt_ignored) 'Actual Complete V9 mirror; source count from frozen manifest'
Check ($mirror.file_count-eq$mirror.files.Count) 'New artifact count comes from exact mirror receipt'
foreach($p in $mirror.files){Verify $p};foreach($p in $mirror.preserved_source_packages){Verify $p}
foreach($binding in @($declaration.reporting_recovery_card,$declaration.reporting_recovery_receipt,$declaration.failure_mirror)){Verify $binding}
Check ($declaration.reporting_recovery_card.sha256-eq'9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'-and$declaration.reporting_recovery_receipt.sha256-eq'5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72'-and$declaration.failure_mirror.sha256-eq'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532') 'Actual separately reviewed recovery and original failure mirror'
$recoveryCard=Read $declaration.reporting_recovery_card.path;$recovery=Read $declaration.reporting_recovery_receipt.path;$failure=Read $declaration.failure_mirror.path
Check ($recovery.kind-eq'cpb_v9_report_only_recovery_receipt_v1'-and$recovery.status-eq'passed'-and$recoveryCard.kind-eq'cpb_v9_report_only_recovery_card_v1'-and$recoveryCard.root_reviewed-and$recovery.card.sha256-eq$declaration.reporting_recovery_card.sha256-and$recovery.card.bytes-eq$declaration.reporting_recovery_card.bytes) 'Exact actual report-only receipt binds reviewed recovery declaration'
Check ($recovery.original_quality_stage_status-eq'failed'-and$recovery.completed_scientific_stages-eq5-and$recovery.science_stages_replayed-eq0-and$recovery.encoder_calls-eq0-and$recovery.head_fits-eq0-and$recovery.training_updates-eq0-and$recovery.frozen_scientific_source_unchanged-and$recovery.scientific_inputs_before_after_exact-and(Equal $recovery.scientific_inputs_before $recovery.scientific_inputs_after)) 'Reporting-only aggregation; failed original wrappers preserved, zero scientific replay'
Check ($recovery.main_source_identity.source_files-eq118-and$recovery.main_source_identity.card_sha256-eq$declaration.main_card.sha256-and$recovery.main_source_identity.manifest_sha256-eq$declaration.source_manifest.sha256-and$recovery.main_source_identity.archive_sha256-eq$declaration.source_zip.sha256) 'Recovery scientific118/card/manifest/ZIP remained immutable'
Check ($mirror.report_only_recovery.status-eq'passed'-and$mirror.report_only_recovery.card_sha256-eq$declaration.reporting_recovery_card.sha256-and$mirror.report_only_recovery.receipt_sha256-eq$declaration.reporting_recovery_receipt.sha256) 'Complete mirror explicitly supplements original failed quality with exact reporting recovery'
Check ($failure.status-eq'passed'-and$failure.files.Count-eq531-and$recovery.scientific_inputs_before.Count-eq531) 'All531 pre-recovery input files preserved'
foreach($p in $failure.files){Verify $p;$matches=@($recovery.scientific_inputs_before|Where-Object {$_.path-eq$p.container_path});Check ($matches.Count-eq1-and$matches[0].bytes-eq$p.bytes-and$matches[0].sha256-eq$p.sha256) 'Exact original file identity in recovery before/after evidence'}
$recoveryPrefix='output/engineering/causal-fresh-v9-report-recovery/attempt-001/'
$recoveryPins=@($declaration.protected_pins|Where-Object {$_.path.StartsWith($recoveryPrefix)})
$recoveryFiles=@(Get-ChildItem -LiteralPath (Rooted $recoveryPrefix.TrimEnd('/')) -Recurse -File)
Check ($recoveryPins.Count-gt0-and$recoveryPins.Count-eq$recoveryFiles.Count) 'Whole recovery leaf is frozen as explicit protected members'
foreach($p in $recoveryPins){$matches=@($mirror.files|Where-Object {$_.path-eq$p.path});Check ($matches.Count-eq1-and$matches[0].bytes-eq$p.bytes-and$matches[0].sha256-eq$p.sha256) 'Every whole recovery-leaf member is also in exact Complete mirror'}
foreach($name in @('temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv','temporal-between-bin-position-learning.csv','temporal-projection-learning.json')){Check (@($mirror.files|Where-Object {$_.path-eq('output/runs/causal-fresh-v9/run-001/'+$name)}).Count-eq1) ('Temporal evidence mirrored: '+$name)}
$new5=@($mirror.files|Where-Object {$_.path-like'output/runs/causal-fresh-v9/run-001/checkpoints/*.pt'})
$checkpointNames=@('inference-0.pt','inference-2048.pt','training-512.pt','training-1024.pt','training-2048.pt')
Check ($new5.Count-eq5-and-not(Compare-Object @($new5|ForEach-Object {Split-Path -Leaf $_.path}|Sort-Object) @($checkpointNames|Sort-Object))) 'Exactly five V9 main checkpoints'
$critical=@()
foreach($p in $v8.critical_checkpoint_pins){
 $repo=$p.repository
 if(-not$repo){Check ($p.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')) 'Only original five V5 checkpoint pins may lack repository';$repo='cuwacunu_embedding'}
 Verify $p $repo;$critical+=[ordered]@{repository=$repo;path=$p.path;bytes=$p.bytes;sha256=$p.sha256}
}
foreach($p in $new5){$critical+=[ordered]@{repository='cuwacunu_embedding';path=$p.path;bytes=$p.bytes;sha256=$p.sha256}}
Check ($critical.Count-eq99-and@($critical|ForEach-Object {$_.repository+'/'+$_.path}|Sort-Object -Unique).Count-eq99) '99 unique protected checkpoints only after actual five V9 files exist'
$ignored=@()
foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){
 $paths=@($old.pins|Where-Object {$_.repository-eq$repo-and$_.path.EndsWith('.pt')}|ForEach-Object {$_.path})
 $paths+=@($critical|Where-Object {$_.repository-eq$repo}|ForEach-Object {$_.path})
 if($repo-eq'cuwacunu_embedding'){$paths+=@(@($oldMirrorFiles)+@($mirror.files)|Where-Object {$_.path.EndsWith('.pt')}|ForEach-Object {$_.path})}
 $paths=@($paths|Sort-Object -Unique);$returned=@();$repoRoot=(Join-Path $workspaceRoot $repo).Replace([char]92,[char]47)
 for($i=0;$i-lt$paths.Count;$i+=128){$end=[Math]::Min($i+127,$paths.Count-1);$batch=@($paths[$i..$end]);$returned+=@(& git -c ('safe.directory='+$repoRoot) -C $repoRoot check-ignore -- @batch);Check ($LASTEXITCODE-eq0) 'Bounded Git ignore inspection'}
 Check ($returned.Count-eq$paths.Count-and-not(Compare-Object @($returned|Sort-Object -Unique) $paths)) ('All old/new/checkpoint PT ignored: '+$repo)
 $ignored+=[ordered]@{repository=$repo;unique_pinned_PT=$paths.Count;all_ignored=$true}
}
Check (@($mirror.files|Where-Object {$_.path.EndsWith('.pt')}).Count-eq$mirror.ignored_pt_count) 'New mirror PT count is exact'

Verify $declaration.source_manifest;Verify $declaration.source_zip;Verify $declaration.main_card
$manifest=Read $declaration.source_manifest.path
Check ($declaration.main_source_files-gt0-and$manifest.files.Count-eq$declaration.main_source_files-and$manifest.card_sha256-eq$declaration.main_card.sha256) 'Exact admitted manifest-derived source count/card membership'
foreach($p in $manifest.files){Verify $p}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipSpecs=@($v8.source_ZIPs)+@([ordered]@{label='V9 admitted001';source_files=$declaration.main_source_files;archive=$declaration.source_zip;manifest=$declaration.source_manifest});$zipChecks=@()
foreach($spec in $zipSpecs){
 Verify $spec.archive;Verify $spec.manifest;$m=Read $spec.manifest.path;Check ($m.files.Count-eq$spec.source_files) 'Exact source manifest member count'
 $zip=[IO.Compression.ZipFile]::OpenRead((Rooted $spec.archive.path))
 try{
  Check ($zip.Entries.Count-eq$spec.source_files+1-and@($zip.Entries.FullName|Sort-Object -Unique).Count-eq$zip.Entries.Count) 'ZIP unique exact source set plus manifest'
  foreach($p in $m.files){
   $entry=$zip.GetEntry($p.path);Check ($null-ne$entry-and$entry.Length-eq$p.bytes) ('ZIP member bytes: '+$p.path)
   $stream=$entry.Open();$hasher=[Security.Cryptography.SHA256]::Create()
   try{$digest=[Convert]::ToHexString($hasher.ComputeHash($stream)).ToLowerInvariant()}finally{$hasher.Dispose();$stream.Dispose()}
   Check ($digest-eq$p.sha256) ('ZIP member hash: '+$p.path)
  }
 }finally{$zip.Dispose()}
 $zipChecks+=[ordered]@{label=$spec.label;source_files=$spec.source_files;archive=$spec.archive;manifest=$spec.manifest;all_member_hashes_exact=$true}
}
$buildPins=@($mirror.files|Where-Object {$_.path.EndsWith('/build-admission.json')});Check ($buildPins.Count-eq1) 'One actual V9 build admission'
$build=Read $buildPins[0].path
Check ($build.status-eq'passed'-and$build.source_files-eq$declaration.main_source_files-and$build.binaries.Count-eq7-and@($build.dependency_closures.PSObject.Properties).Count-eq11) 'Manifest-derived sources/seven binaries/eleven translation units'
Check ($build.source_manifest.sha256-eq$declaration.source_manifest.sha256-and$build.card_sha256-eq$declaration.main_card.sha256) 'Build binds exact card/source manifest'
$binaryNames=@('test_causal_patch_bottleneck_mae_v9','test_cpb_v9_variance_gradient','test_cpb_v9_temporal_pooling','causal_fresh_v9_train','causal_fresh_v9_extract','causal_fresh_v9_market_read','causal_fresh_v9_synthetic_read')
Check (-not(Compare-Object @($build.binaries|ForEach-Object {Split-Path -Leaf $_.path}|Sort-Object) @($binaryNames|Sort-Object))) 'Exact seven binary identities'
Check ($declaration.compiled_binaries.Count-eq7) 'Seven separately mirrored compiled binaries are not Complete.files members'
Verify $declaration.compiled_binary_mirror
$binaryMirror=Read $declaration.compiled_binary_mirror.path
Check ($binaryMirror.kind-eq'cpb_v9_seven_build_binaries_mirror_v1'-and$binaryMirror.status-eq'passed'-and$binaryMirror.file_count-eq7-and$binaryMirror.files.Count-eq7-and$binaryMirror.included_in_main_Complete_files-eq$false-and$binaryMirror.container_files_preserved) 'Exact separate seven-binary mirror receipt'
Check ($binaryMirror.build_admission.path-eq$buildPins[0].path-and$binaryMirror.build_admission.bytes-eq$buildPins[0].bytes-and$binaryMirror.build_admission.sha256-eq$buildPins[0].sha256) 'Separate binary mirror binds same exact build admission'
foreach($set in @(@{items=@($declaration.compiled_binaries);label='host'},@{items=@($build.binaries);label='build'},@{items=@($binaryMirror.files);label='receipt'})){
 $names=@($set.items|ForEach-Object {Split-Path -Leaf $_.path})
 Check (@($set.items.path|Sort-Object -Unique).Count-eq7-and@($names|Sort-Object -Unique).Count-eq7-and-not(Compare-Object @($names|Sort-Object) @($binaryNames|Sort-Object))) ('Exact unique admitted seven-binary set: '+$set.label)
}
foreach($hostBinary in $declaration.compiled_binaries){
 Verify $hostBinary
 $matches=@($build.binaries|Where-Object {(Split-Path -Leaf $_.path)-eq(Split-Path -Leaf $hostBinary.path)})
 $mirrored=@($binaryMirror.files|Where-Object {$_.path-eq$hostBinary.path})
 Check ($matches.Count-eq1-and$matches[0].bytes-eq$hostBinary.bytes-and$matches[0].sha256-eq$hostBinary.sha256-and$mirrored.Count-eq1-and$mirrored[0].bytes-eq$hostBinary.bytes-and$mirrored[0].sha256-eq$hostBinary.sha256-and$mirrored[0].container_path-eq$matches[0].path) 'Exact host compiled binary matches separate receipt and container build-admission pin'
}

# The existing-container interpreter hashes bytes only; no model executable is invoked.
$containerID='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$observed=@(& docker.exe inspect cuwacunu);Check ($LASTEXITCODE-eq0) 'Inspect existing managed container';$container=($observed-join"`n"|ConvertFrom-Json)[0]
Check ($container.Id-eq$containerID-and$container.Image-eq'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-and$container.State.Running) 'Exact existing running container/image'
$containerRoot='/opt/cuwacunu/research/causal-fresh-v9/attempt-001'
$containerInputs=@($manifest.files|ForEach-Object {[ordered]@{path=($containerRoot+'/source/'+$_.path);bytes=$_.bytes;sha256=$_.sha256}})+@($build.source_manifest)+@($build.binaries)+@([ordered]@{path=$containerRoot+'/source.zip';bytes=$declaration.source_zip.bytes;sha256=$declaration.source_zip.sha256})
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
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    if h.hexdigest()!=item['sha256']:raise RuntimeError('Changed frozen file: '+str(p))
print(json.dumps({'status':'passed','files':len(items),'model_calls':0,'head_fits':0,'training_updates':0}))
'@
$checked=@($containerScript.Replace('__INPUTS__',$payload)|& docker.exe exec -i cuwacunu python3 -B -)
Check ($LASTEXITCODE-eq0) 'Container source/ZIP/seven-binary hash inspection';$containerBytes=($checked-join"`n"|ConvertFrom-Json)
Check ($containerBytes.status-eq'passed'-and$containerBytes.files-eq$containerInputs.Count) 'Every exact container source/binary/ZIP binding'

foreach($p in @($declaration.metadata_application,$declaration.proposal_review,$declaration.final_establishment)){Verify $p}
$application=Read $declaration.metadata_application.path;$review=Read $declaration.proposal_review.path;$establishment=Read $declaration.final_establishment.path
Check ($application.status-eq'passed'-and-not$application.baseline_changed-and-not$application.application_changed) 'Authorized metadata-only application'
foreach($p in $application.files){Verify $p;if($p.source){Verify $p 'cuwacunu_embedding' $p.source}}
Check ($review.status-eq'passed') 'Independent metadata proposal review';foreach($p in $review.metadata_outputs){Verify $p}
Check ($establishment.status-eq'passed'-and$establishment.candidate-eq'CPB-v9.market-01'-and$establishment.measured_candidate_established-and$establishment.trust_admission_passed-and$establishment.full_coverage-and$establishment.initial_pairing_count-eq13-and$establishment.numeric_total-eq22-and-not$establishment.overall_winner-and-not$establishment.winner_selection_made) 'Full trust and dynamic22 outcomes; no winner selection'
$ledgerPath='doc/clean_causal_encoder_research_ledger.json';$catalogPath='code/encoders/causal_patch_bottleneck_mae/versions/current.json'
$ledger=Read $ledgerPath;$catalog=Read $catalogPath
$priorLedgerPin=@($priorSnapshot.files|Where-Object {$_.path-eq$ledgerPath});$priorCatalogPin=@($priorSnapshot.files|Where-Object {$_.path-eq$catalogPath})
Check ($priorLedgerPin.Count-eq1-and$priorCatalogPin.Count-eq1) 'Whole prior ledger/catalog snapshots'
$priorLedger=Read $priorLedgerPin[0].snapshot;$priorCatalog=Read $priorCatalogPin[0].snapshot
Check ($ledger.baseline-eq'CPB-v3.market-01'-and-not$ledger.checkpoint_deletion_allowed-and$catalog.active_application-eq'CPB-v3.market-01') 'V3 selected; no checkpoint deletion'
Check (Equal $priorCatalog.clean_causal_training_baseline $catalog.clean_causal_training_baseline) 'Whole selected V3 catalog entry unchanged'
Prefix $priorCatalog.fresh_research_candidate_history $catalog.fresh_research_candidate_history 'catalog candidate history'
Check (@($catalog.fresh_research_candidate_history|Where-Object {Equal $_ $priorCatalog.fresh_research_candidate}).Count-eq1) 'Whole prior V8 catalog entry retained'
Check ($catalog.fresh_research_candidate.instance-eq'CPB-v9.market-01') 'New measured V9 catalog entry'
foreach($field in @('candidates','candidate_status_history','stage_scorecards','leader_history','displaced_leaders','direction_history','stage_leader_update_history','reconstruction_stage_scorecards','reconstruction_leader_history')){Prefix $priorLedger.$field $ledger.$field $field}
Check ($ledger.candidates.Count-eq$priorLedger.candidates.Count+1-and$ledger.stage_scorecards.Count-eq$priorLedger.stage_scorecards.Count+2-and$ledger.reconstruction_stage_scorecards.Count-eq$priorLedger.reconstruction_stage_scorecards.Count+2) 'One candidate/two full stages appended'
foreach($prior in $priorLedger.stage_leaders){Check (@($ledger.leader_history|Where-Object {Equal $_.record $prior}).Count-ge1) 'Whole prior quality leader preserved'}
foreach($prior in $priorLedger.reconstruction_stage_leaders){$matches=@(foreach($entry in $ledger.reconstruction_leader_history){foreach($record in $entry.records){if(Equal $record $prior){$record}}});Check ($matches.Count-ge1) 'Whole prior reconstruction leader preserved'}
foreach($p in $priorSnapshot.files|Where-Object {$_.path.EndsWith('.md')}){
 $current=[IO.File]::ReadAllBytes((Rooted $p.path));Check ($current.Length-ge$p.bytes) 'Prior whole text retained'
 $tail=[byte[]]::new($p.bytes);[Array]::Copy($current,$current.Length-$p.bytes,$tail,0,$p.bytes)
 $hasher=[Security.Cryptography.SHA256]::Create();try{$digest=[Convert]::ToHexString($hasher.ComputeHash($tail)).ToLowerInvariant()}finally{$hasher.Dispose()};Check ($digest-eq$p.sha256) 'Exact prior front-text tail behind new prepend'
}
$devRoot=(Join-Path $workspaceRoot 'cuwacunu_dev').Replace([char]92,[char]47)
$devStatus=@(& git -c ('safe.directory='+$devRoot) -C $devRoot status --porcelain=v1 --untracked-files=all);Check ($LASTEXITCODE-eq0) 'Read-only concurrent DEV inventory'
$receipt=[ordered]@{
 kind='causal_fresh_v9_final_preservation_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks
 baseline='CPB-v3.market-01';candidate='CPB-v9.market-01';baseline_changed=$false;application_changed=$false
 prior_V8_final=$declaration.prior_V8_final;old_before_inventory=$v8.old_before_inventory;old_pins=1030;old_authorized_snapshots=$v6.old_authorized_snapshots
 V9_prior_snapshot=$declaration.prior_live_snapshot;prior_whole_metadata_chain_preserved=$true
 critical_checkpoint_total=99;historical_critical_checkpoints=94;new_main_checkpoints=5;critical_checkpoint_pins=$critical
 selected_V3_source_files_each_repo=47;selected_V3_checkpoint_sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'
 old_complete_mirrors=$oldMirrorSummary;complete_V9_mirror=$declaration.complete_mirror;saved_V9_files_verified=$mirror.files.Count;V9_saved_PT=$mirror.ignored_pt_count
 all_pinned_PT_ignored=$ignored;PT_scope='Saved heads/features/witnesses/tests/checkpoints; not trained encoder count'
 source_ZIPs=$zipChecks;V9_source_files=$declaration.main_source_files;V9_binary_files=7;V9_build_binaries_mirror=$declaration.compiled_binary_mirror;V9_host_compiled_binaries=$declaration.compiled_binaries;V9_translation_units=11;container_preservation=$containerBytes;container_id=$containerID
 additional_protected_inventories=$expanded;additional_protected_pins=$declaration.protected_pins
 authorized_metadata_application=$declaration.metadata_application;independent_metadata_proposal_review=$declaration.proposal_review;final_establishment=$declaration.final_establishment
 reporting_recovery_card=$declaration.reporting_recovery_card;reporting_recovery_receipt=$declaration.reporting_recovery_receipt;before_recovery_failure_mirror=$declaration.failure_mirror;report_only_recovery_original_inputs_exact=531;science_stages_replayed=0
 current_ledger=(Pin $ledgerPath);current_catalog=(Pin $catalogPath);current_instance=(Pin 'code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v9.market-01.json')
 numeric_passed=$establishment.numeric_passed;numeric_total=22;trust_admission_passed=$establishment.trust_admission_passed;overall_establishment_passed=$establishment.overall_establishment_passed
 declaration=(Pin $PreservationCard)
 source=[ordered]@{path='code/research/causal_fresh_v9_preservation/verify.ps1';bytes=(Get-Item -LiteralPath $PSCommandPath).Length;sha256=(Hash $PSCommandPath);parent='code/research/causal_fresh_v8_preservation/verify.ps1';parent_sha256='5a66c68ef720b5c8aef988ef09e694f96e466e058f9042ed95b18ca6422b74db'}
 concurrency=[ordered]@{repository='cuwacunu_dev';policy='Only selected V3 sources/checkpoint fixed; no reset/equality claim for explicitly declared concurrent integration controls';observed_exclusions=@($script:observedConcurrent.Values);git_status=$devStatus}
 limitations=@('File/hash/metadata history only; no quality arithmetic/model deserialization','Prior bytes remain whole snapshots; superseded context uses prepends/history, zero deletion')
 actions=[ordered]@{models=0;fits=0;builds=0;training_updates=0;Git_mutations=0;deletions=0;files_written=@('doc/causal_fresh_v9/final-preservation.json')}
}
[IO.File]::WriteAllText($destination,($receipt|ConvertTo-Json -Depth 90).Replace([string][char]13,'')+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';checks=$receipt.checks;critical99=$critical.Count;old1030=$old.pins.Count;receipt=(Pin 'doc/causal_fresh_v9/final-preservation.json')}|ConvertTo-Json -Depth 5

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
# Complete physical FINAL V11 verifier copy, with one additional normal V12 chain level.
# File/hash/JSON inspection only; no model, encoder, optimizer, head or build execution.
param([Parameter(Mandatory=$true)][string]$PreservationCard,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PreservationCardSHA256,
 [switch]$FinalMetadataApplied)
$ErrorActionPreference='Stop'
if(-not $FinalMetadataApplied){throw 'Wait for the explicit root FinalMetadataApplied signal'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$destination=Join-Path $researchRoot 'doc/causal_fresh_v12/final-preservation.json'
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

# Root may freeze an actual card only after all mirrors/audits/trust/metadata complete.
# Nine normal roles, eight owned workflow assets and explicit five actual new pins.
# Historical V9 recovery and V10 whole40 pairings remain in their exact inherited chain.
Check ((Hash (Rooted $PreservationCard))-eq$PreservationCardSHA256) 'Externally reviewed preservation card before parse'
$declaration=Read $PreservationCard
Check ($declaration.kind-eq'causal_fresh_v12_final_preservation_card_v1'-and$declaration.authorized-is[bool]-and$declaration.authorized-eq$true-and$declaration.current_reporting_recovery_required-eq$false) 'Explicit normal frozen preservation Boolean admission'
Check ($declaration.historical_critical_checkpoints-eq109-and$declaration.new_actual_checkpoints-eq5-and$declaration.expected_final_critical_checkpoints-eq114) 'Conditional109 inherited plus five actual checkpoint scope'
Verify $declaration.verifier;Verify $declaration.generator;Verify $declaration.parent_template
Check ((Hash $PSCommandPath)-eq$declaration.verifier.sha256) 'Exact independently reviewed verifier source'
Check ($declaration.source_workflow.Count-eq8) 'Eight exact owned workflow assets'
foreach($binding in $declaration.source_workflow){Verify $binding}
Verify $declaration.input_plan
Check ($declaration.parent_template.bytes-eq1035137-and$declaration.parent_template.sha256-eq'516690f776d8cdb0306288135ba894c7611586044e4db2a9794e1aa828bc9325') 'Exact final V11 card DATA template;no old execution'
Verify $declaration.prior_V11_final
Check ($declaration.prior_V11_final.bytes-eq1066891-and$declaration.prior_V11_final.sha256-eq'805aa2183f14dbfd3a7bb521a67f5d01e33b22008b5244da45745baa8c51bab0') 'Exact final V11 receipt'
$v11=Read $declaration.prior_V11_final.path
Check ($v11.status-eq'passed'-and$v11.critical_checkpoint_pins.Count-eq109-and$v11.critical_checkpoint_total-eq109-and$v11.old_pins-eq1030-and$v11.declaration.sha256-eq$declaration.parent_template.sha256) 'Actual109 inherited checkpoints and exact V11 opaque declaration'
Verify $v11.prior_V10_final
$v10=Read $v11.prior_V10_final.path
Check ($v10.status-eq'passed'-and$v10.critical_checkpoint_pins.Count-eq104-and$v10.critical_checkpoint_total-eq104-and$v10.old_pins-eq1030) 'Actual104 inherited checkpoints and1030 historical pins'
Verify $v10.prior_V9_final
$v9=Read $v10.prior_V9_final.path
Check ($v9.status-eq'passed'-and$v9.critical_checkpoint_pins.Count-eq99-and$v9.critical_checkpoint_total-eq99-and$v9.old_pins-eq1030) 'Actual99 inherited checkpoints and1030 historical pins'
Verify $v9.prior_V8_final
$v8=Read $v9.prior_V8_final.path
Check ($v8.status-eq'passed'-and$v8.critical_checkpoint_pins.Count-eq94-and$v8.old_pins-eq1030) 'Prior94 critical checkpoints and1030 historical pins'
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
Verify $v9.V9_prior_snapshot;$v9Snapshot=Read $v9.V9_prior_snapshot.path
foreach($p in $v9Snapshot.files){Add-Snapshot $p $p.snapshot}
foreach($p in @($v8.current_ledger,$v8.current_catalog)){Check (@($v9Snapshot.files|Where-Object {$_.path-eq$p.path-and$_.bytes-eq$p.bytes-and$_.sha256-eq$p.sha256}).Count-eq1) 'Exact V8-current to pre-V9 snapshot chain'}
Verify $v10.V10_prior_snapshot;$v10Snapshot=Read $v10.V10_prior_snapshot.path
foreach($p in $v10Snapshot.files){Add-Snapshot $p $p.snapshot}
foreach($p in @($v9.current_ledger,$v9.current_catalog)){Check (@($v10Snapshot.files|Where-Object {$_.path-eq$p.path-and$_.bytes-eq$p.bytes-and$_.sha256-eq$p.sha256}).Count-eq1) 'Exact V9-current to pre-V10 snapshot chain'}
Verify $v11.V11_prior_snapshot;$v11Snapshot=Read $v11.V11_prior_snapshot.path
foreach($p in $v11Snapshot.files){Add-Snapshot $p $p.snapshot}
foreach($p in @($v10.current_ledger,$v10.current_catalog)){Check (@($v11Snapshot.files|Where-Object {$_.path-eq$p.path-and$_.bytes-eq$p.bytes-and$_.sha256-eq$p.sha256}).Count-eq1) 'Exact V10-current to pre-V11 snapshot chain'}
Verify $declaration.prior_live_snapshot;$priorSnapshot=Read $declaration.prior_live_snapshot.path
Check ($priorSnapshot.status-eq'passed'-and$priorSnapshot.files.Count-gt0-and@($priorSnapshot.files.path|Sort-Object -Unique).Count-eq$priorSnapshot.files.Count) 'Every whole pre-V12 live file preserved from actual unique snapshot inventory'
foreach($p in $priorSnapshot.files){Add-Snapshot $p $p.snapshot}
foreach($p in @($v11.current_ledger,$v11.current_catalog)){
 Check (@($priorSnapshot.files|Where-Object {$_.path-eq$p.path-and$_.bytes-eq$p.bytes-and$_.sha256-eq$p.sha256}).Count-eq1) 'Exact V11-current to pre-V12 snapshot chain'
}
Verify-PinTree $v6;Verify-PinTree $v7;Verify-PinTree $v8;Verify-PinTree $v9;Verify-PinTree $v10;Verify-PinTree $v11;Verify $v8.old_before_inventory
$old=Read $v7.old_before_inventory.path;Check ($old.pins.Count-eq1030) 'Every historical inventory pin retained'
foreach($p in $old.pins){Verify-Historical $p}
foreach($p in $declaration.concurrent_DEV_exclusions){Check (@($old.pins|Where-Object {$_.repository-eq$p.repository-and$_.path-eq$p.path-and$_.bytes-eq$p.bytes-and$_.sha256-eq$p.sha256}).Count-eq1) 'Concurrent exclusion matches exact historical inventory entry'}

# Expand every original V5..V11 Complete mirror without inventing artifact counts.
$oldMirrors=@($v6.complete_V5_mirror,$v6.complete_V6_mirror,$v7.complete_V7_mirror,$v8.complete_V8_mirror,$v9.complete_V9_mirror,$v10.complete_V10_mirror,$v11.complete_V11_mirror);$oldMirrorFiles=@();$oldMirrorSummary=@()
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
foreach($path in @('doc/causal_fresh_v6/protocol-preflight.json','doc/causal_fresh_v7/protocol-preflight.json','doc/causal_fresh_v8/protocol-preflight.json','doc/causal_fresh_v9/protocol-preflight.json','doc/causal_fresh_v10/protocol-preflight.json','doc/causal_fresh_v11/protocol-preflight.json')){
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
Check (-not(Compare-Object @($declaration.completed_evidence_roots.PSObject.Properties.Name|Sort-Object) @('training_audit','quality_audit','establishment','finalization'|Sort-Object))) 'Four actual completed host evidence roots'
foreach($root in $declaration.completed_evidence_roots.PSObject.Properties.Value){
 $directory=Rooted $root;$prefix=$root.TrimEnd('/')+'/'
 $members=@($declaration.protected_pins|Where-Object{$_.path.StartsWith($prefix)})
 $files=@(Get-ChildItem -LiteralPath $directory -Recurse -File)
 Check ($members.Count-gt0-and$members.Count-eq$files.Count) 'Whole completed host evidence root retained, including source/log/failure files outside container inventories'
 foreach($file in $files){$relative=[IO.Path]::GetRelativePath($researchRoot,$file.FullName).Replace([char]92,[char]47);Check (@($members|Where-Object{$_.path-eq$relative}).Count-eq1) 'Every actual completed host evidence member is explicitly pinned'}
}

foreach($repo in @('cuwacunu_embedding','cuwacunu_dev')){
 $sources=@($old.pins|Where-Object {$_.repository-eq$repo-and$_.path-like'code/encoders/causal_patch_bottleneck_mae/versions/v3/*'})
 Check ($sources.Count-eq47) ('47 physical selected V3 sources: '+$repo);foreach($p in $sources){Verify $p $repo}
 Verify ([ordered]@{path='output/baselines/CPB-v3.market-01/inference.pt';bytes=1290625;sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'}) $repo
}
Verify $declaration.complete_mirror;$mirror=Read $declaration.complete_mirror.path
Check ($mirror.kind-eq'causal_fresh_v12_container_to_local_evidence_mirror_v1'-and$mirror.status-eq'passed'-and$mirror.scope-eq'Complete'-and$mirror.attempt-eq$declaration.main_attempt-and$mirror.source_files_verified-eq$declaration.main_source_files-and$mirror.all_mirrored_pt_ignored) 'Actual normal Complete V12 mirror; source count from frozen manifest'
Check ($mirror.file_count-eq$mirror.files.Count) 'New artifact count comes from exact mirror receipt'
foreach($p in $mirror.files){Verify $p};foreach($p in $mirror.preserved_source_packages){Verify $p}
# Full prior V9 recovery/failure/source/report leaves were verified through its receipt, inherited inventories and Complete mirror above.
foreach($name in @('temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv','temporal-between-bin-position-learning.csv','temporal-moment-subgroup-learning.csv','temporal-projection-learning.json')){Check (@($mirror.files|Where-Object {$_.path-eq('output/runs/causal-fresh-v12/run-001/'+$name)}).Count-eq1) ('Temporal evidence mirrored: '+$name)}
$new5=@($mirror.files|Where-Object {$_.path-like'output/runs/causal-fresh-v12/run-001/checkpoints/*.pt'})
$checkpointNames=@('inference-0.pt','inference-2048.pt','training-512.pt','training-1024.pt','training-2048.pt')
Check ($new5.Count-eq5-and-not(Compare-Object @($new5|ForEach-Object {Split-Path -Leaf $_.path}|Sort-Object) @($checkpointNames|Sort-Object))) 'Exactly five actual V12 main checkpoints'
Check ($declaration.new_checkpoint_pins.Count-eq5) 'Five explicitly admitted new checkpoint pins'
foreach($p in $new5){Check (@($declaration.new_checkpoint_pins|Where-Object{$_.path-eq$p.path-and$_.container_path-eq$p.container_path-and$_.bytes-eq$p.bytes-and$_.sha256-eq$p.sha256}).Count-eq1) 'Actual checkpoint matches reviewed declaration and Complete'}
$critical=@()
foreach($p in $v11.critical_checkpoint_pins){
 $repo=$p.repository
 if(-not$repo){Check ($p.path.StartsWith('output/runs/causal-fresh-v5/run-001/checkpoints/')) 'Only original five V5 checkpoint pins may lack repository';$repo='cuwacunu_embedding'}
 Verify $p $repo;$critical+=[ordered]@{repository=$repo;path=$p.path;bytes=$p.bytes;sha256=$p.sha256}
}
foreach($p in $new5){$critical+=[ordered]@{repository='cuwacunu_embedding';path=$p.path;bytes=$p.bytes;sha256=$p.sha256}}
Check ($critical.Count-eq114-and@($critical|ForEach-Object {$_.repository+'/'+$_.path}|Sort-Object -Unique).Count-eq114) '114 unique protected checkpoints only after actual five V12 files exist'
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
Check ($declaration.main_source_files-eq137-and$manifest.files.Count-eq$declaration.main_source_files-and$manifest.card_sha256-eq$declaration.main_card.sha256) 'Exact admitted137-source/card membership'
Check ($manifest.attempt-eq$declaration.main_attempt-and$declaration.main_attempt-match'^attempt-[0-9]{3}$') 'Exact bounded admitted main attempt'
foreach($p in $manifest.files){Verify $p}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipSpecs=@($v11.source_ZIPs)+@([ordered]@{label=('V12 admitted '+$declaration.main_attempt);source_files=$declaration.main_source_files;archive=$declaration.source_zip;manifest=$declaration.source_manifest});$zipChecks=@()
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
$buildPins=@($mirror.files|Where-Object {$_.path.EndsWith('/build-admission.json')});Check ($buildPins.Count-eq1) 'One actual V12 build admission'
$build=Read $buildPins[0].path
Check ($build.status-eq'passed'-and$build.source_files-eq$declaration.main_source_files-and$build.binaries.Count-eq7-and@($build.dependency_closures.PSObject.Properties).Count-eq11) 'Manifest-derived sources/seven binaries/eleven translation units'
Check ($build.source_manifest.sha256-eq$declaration.source_manifest.sha256-and$build.card_sha256-eq$declaration.main_card.sha256) 'Build binds exact card/source manifest'
$binaryNames=@('test_causal_patch_bottleneck_mae_v12','test_cpb_v12_variance_gradient','test_cpb_v12_temporal_pooling','causal_fresh_v12_train','causal_fresh_v12_extract','causal_fresh_v12_market_read','causal_fresh_v12_synthetic_read')
Check (-not(Compare-Object @($build.binaries|ForEach-Object {Split-Path -Leaf $_.path}|Sort-Object) @($binaryNames|Sort-Object))) 'Exact seven binary identities'
Check ($declaration.compiled_binaries.Count-eq7) 'Seven separately mirrored compiled binaries are not Complete.files members'
Verify $declaration.compiled_binary_mirror
$binaryMirror=Read $declaration.compiled_binary_mirror.path
Check ($binaryMirror.kind-eq'cpb_v12_actual_build_binary_mirror_v1'-and$binaryMirror.status-eq'passed'-and$binaryMirror.source_files-eq$manifest.files.Count-and$binaryMirror.binaries-eq7-and$binaryMirror.translation_units-eq11-and$binaryMirror.files.Count-eq18) 'Actual build mirror: manifest-derived sources, seven binaries and eleven dependency files'
Check ($binaryMirror.build_admission.path-eq$buildPins[0].path-and$binaryMirror.build_admission.bytes-eq$buildPins[0].bytes-and$binaryMirror.build_admission.sha256-eq$buildPins[0].sha256) 'Separate binary mirror binds same exact build admission'
$mirroredBinaries=@($binaryMirror.files|Where-Object{-not$_.path.EndsWith('.d')})
$dependencies=@($binaryMirror.files|Where-Object{$_.path.EndsWith('.d')})
$dependencyPins=@($build.dependency_closures.PSObject.Properties|ForEach-Object{$_.Value.file})
Check ($mirroredBinaries.Count-eq7-and$dependencies.Count-eq11-and$dependencyPins.Count-eq11-and@($binaryMirror.files.path|Sort-Object -Unique).Count-eq18-and@($binaryMirror.files.container_path|Sort-Object -Unique).Count-eq18) 'Eighteen unique actual build members'
Check (Equal $declaration.compiled_dependency_files $dependencies) 'All eleven admitted dependency mirrors preserved in declaration'
foreach($p in $binaryMirror.files){Verify $p}
foreach($p in $dependencies){$matches=@($dependencyPins|Where-Object{$_.path-eq$p.container_path});Check ($matches.Count-eq1-and$matches[0].bytes-eq$p.bytes-and$matches[0].sha256-eq$p.sha256) 'Dependency mirror matches actual admitted translation-unit closure'}
foreach($set in @(@{items=@($declaration.compiled_binaries);label='host'},@{items=@($build.binaries);label='build'},@{items=$mirroredBinaries;label='receipt'})){
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
$containerRoot='/opt/cuwacunu/research/causal-fresh-v12/'+$declaration.main_attempt
$containerInputs=@($manifest.files|ForEach-Object {[ordered]@{path=($containerRoot+'/source/'+$_.path);bytes=$_.bytes;sha256=$_.sha256}})+@($build.source_manifest)+@($build.binaries)+@($dependencyPins)+@([ordered]@{path=$containerRoot+'/source.zip';bytes=$declaration.source_zip.bytes;sha256=$declaration.source_zip.sha256})
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
Check ($establishment.status-eq'passed'-and$establishment.candidate-eq'CPB-v12.market-01'-and$establishment.measured_candidate_established-and$establishment.trust_admission_passed-and$establishment.full_coverage-and$establishment.initial_partial_native20_40_pairings_passed-and($establishment.initial_pairing_native_coordinate_slice-join',')-eq'20,40'-and$establishment.whole_native40_initial_equality_claimed-eq$false-and$establishment.initial_pairing_count-eq13-and$establishment.numeric_total-eq22-and-not$establishment.overall_winner-and-not$establishment.winner_selection_made) 'Full trust/partial native20:40 and dynamic22 outcomes; no winner selection'
Check ($establishment.actual_component_archive_fields-eq60-and$establishment.same_forward_temporal_fields-eq25-and$establishment.new_uniform_dispersion_trace_fields-eq4-and$establishment.uniform_dispersion_direct_score_adjoint-eq0-and$establishment.local_hidden_adjoint_array_available-eq$false-and$establishment.score_to_hidden_route_reconstructed-eq$false-and$establishment.hidden_reference_is_observed_gradient_parity-eq$false-and$establishment.all_eight_temporal_moment_subgroups_verified-eq$true-and$establishment.whole8_two4_squared_norms_and_element_counts_verified-eq$true-and$establishment.actual_first_last_local_moment_adjoints_verified-eq$true-and$establishment.full_encoder_gradient_or_global_norm_reconstructed-eq$false-and$establishment.common_score_bias_gauge_tight_parity_required-eq$false) 'Saved60/25 dual-support/local-three-parameter adjoint scope;hidden reference descriptive only'
$dualSupportFlags=@('attention_mean_and_uniform_dispersion_full_expression_adjoints','uniform_dispersion_own_center','uniform_dispersion_score_independent','uniform_dispersion_hidden_gradient_live','attention_mean_score_gradient_live','dual_support_actual_trace')
foreach($flag in $dualSupportFlags){Check ($establishment.$flag-eq$true) ('Passed source-bound dual-support fixture flag: '+$flag)}
Check ($establishment.temporal_pooling_direct_kernel_calls-eq47-and$establishment.temporal_pooling_F64_finite_difference_coordinates-eq18-and$establishment.temporal_pooling_F32_finite_difference_coordinates-eq18) 'Exact source-bound47/18/18 fixture counts'
$main=Read $declaration.main_card.path
Check ($main.advancement.goal_protocol.sha256-eq'866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'-and$main.scientific_recipe_parent-eq'CPB-v11 source only;fresh weights/optimizer/statistics,not trained ancestry') 'Strongest866 criteria22 and V11 source-only fresh lineage'
Check ($establishment.normal_prospective_source_admission.normal_quality_all_stages_passed-eq$true-and$establishment.normal_prospective_source_admission.scientific_model_head_or_training_replay-eq$false) 'Normal completed quality without scientific replay'
Check ($declaration.current_protected_inventories.Count-eq4) 'Four current inventory roles separate from whole inherited role-tagged records'
foreach($role in @('training_audit','quality_audit','establishment','finalization')){$items=@($declaration.current_protected_inventories|Where-Object{$_.role-eq$role});Check ($items.Count-eq1) ('Completed current protected inventory: '+$role);Check (@($declaration.protected_inventories|Where-Object{Equal $_ $items[0]}).Count-eq1) 'Current inventory belongs to preserved combined inventory'}
Verify $declaration.preexecution_draft_amendment;Verify $declaration.preexecution_draft_amendment_before_snapshot
Check ($declaration.preexecution_draft_amendment_scope-eq'inherited_V11_documentary_history_only_not_current_V12'-and$declaration.preexecution_draft_amendment.path-eq'doc/causal_fresh_v11/draft-metadata-amendment-001.json') 'Historical V11 documentary amendment;no invented current V12 amendment'
$amendment=Read $declaration.preexecution_draft_amendment.path
Check (Equal $amendment.preserved_before $declaration.preexecution_draft_amendment_before_snapshot) 'Known draft tolerance amendment retains exact before-four-file snapshot'
$amendmentBefore=Read $amendment.preserved_before.path
Check ($amendmentBefore.files.Count-eq4) 'Whole four files before draft tolerance metadata refinement'
foreach($entry in $amendmentBefore.files){Verify $entry.preserved;Check ($entry.original.bytes-eq$entry.preserved.bytes-and$entry.original.sha256-eq$entry.preserved.sha256) 'Exact original bytes retained by amendment snapshot'}
$ledgerPath='doc/clean_causal_encoder_research_ledger.json';$catalogPath='code/encoders/causal_patch_bottleneck_mae/versions/current.json'
$ledger=Read $ledgerPath;$catalog=Read $catalogPath
$priorLedgerPin=@($priorSnapshot.files|Where-Object {$_.path-eq$ledgerPath});$priorCatalogPin=@($priorSnapshot.files|Where-Object {$_.path-eq$catalogPath})
Check ($priorLedgerPin.Count-eq1-and$priorCatalogPin.Count-eq1) 'Whole prior ledger/catalog snapshots'
$priorLedger=Read $priorLedgerPin[0].snapshot;$priorCatalog=Read $priorCatalogPin[0].snapshot
Check ($ledger.baseline-eq'CPB-v3.market-01'-and-not$ledger.checkpoint_deletion_allowed-and$catalog.active_application-eq'CPB-v3.market-01') 'V3 selected; no checkpoint deletion'
Check (Equal $priorCatalog.clean_causal_training_baseline $catalog.clean_causal_training_baseline) 'Whole selected V3 catalog entry unchanged'
Prefix $priorCatalog.fresh_research_candidate_history $catalog.fresh_research_candidate_history 'catalog candidate history'
Check (@($catalog.fresh_research_candidate_history|Where-Object {Equal $_ $priorCatalog.fresh_research_candidate}).Count-eq1) 'Whole prior V11 catalog entry retained'
Check ($catalog.fresh_research_candidate.instance-eq'CPB-v12.market-01') 'New measured V12 catalog entry'
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
 kind='causal_fresh_v12_final_preservation_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks
 baseline='CPB-v3.market-01';candidate='CPB-v12.market-01';baseline_changed=$false;application_changed=$false
 prior_V11_final=$declaration.prior_V11_final;old_before_inventory=$v11.old_before_inventory;old_pins=1030;old_authorized_snapshots=$v6.old_authorized_snapshots
 V12_prior_snapshot=$declaration.prior_live_snapshot;prior_whole_metadata_chain_preserved=$true
 critical_checkpoint_total=$critical.Count;historical_critical_checkpoints=$v11.critical_checkpoint_pins.Count;new_main_checkpoints=$new5.Count;critical_checkpoint_pins=$critical
 selected_V3_source_files_each_repo=47;selected_V3_checkpoint_sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'
 old_complete_mirrors=$oldMirrorSummary;complete_V12_mirror=$declaration.complete_mirror;saved_V12_files_verified=$mirror.files.Count;V12_saved_PT=$mirror.ignored_pt_count
 all_pinned_PT_ignored=$ignored;PT_scope='Saved heads/features/witnesses/tests/checkpoints; not trained encoder count'
 source_ZIPs=$zipChecks;V12_source_files=$declaration.main_source_files;V12_binary_files=7;V12_build_binaries_mirror=$declaration.compiled_binary_mirror;V12_host_compiled_binaries=$declaration.compiled_binaries;V12_host_dependency_files=$declaration.compiled_dependency_files;V12_translation_units=11;container_preservation=$containerBytes;container_id=$containerID
 additional_protected_inventories=$expanded;additional_protected_pins=$declaration.protected_pins
 authorized_metadata_application=$declaration.metadata_application;independent_metadata_proposal_review=$declaration.proposal_review;final_establishment=$declaration.final_establishment
 current_reporting_recovery_required=$false;historical_V9_reporting_recovery_preserved=$true;science_stages_replayed=0
 initial_pairing_native_coordinate_slice=@(20,40);whole_native40_initial_equality_claimed=$false;actual_component_archive_fields=60;same_forward_temporal_fields=25;new_uniform_dispersion_trace_fields=4
 uniform_dispersion_direct_score_adjoint=0;local_hidden_adjoint_array_available=$false;score_to_hidden_route_reconstructed=$false;hidden_reference_is_observed_gradient_parity=$false
 preexecution_draft_amendment=$declaration.preexecution_draft_amendment;preexecution_draft_amendment_before_snapshot=$declaration.preexecution_draft_amendment_before_snapshot;preexecution_draft_amendment_scope=$declaration.preexecution_draft_amendment_scope
 current_ledger=(Pin $ledgerPath);current_catalog=(Pin $catalogPath);current_instance=(Pin 'code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v12.market-01.json')
 numeric_passed=$establishment.numeric_passed;numeric_total=22;trust_admission_passed=$establishment.trust_admission_passed;overall_establishment_passed=$establishment.overall_establishment_passed
 declaration=(Pin $PreservationCard)
 source=[ordered]@{path='code/research/causal_fresh_v12_preservation/verify.ps1';bytes=(Get-Item -LiteralPath $PSCommandPath).Length;sha256=(Hash $PSCommandPath);parent='code/research/causal_fresh_v11_preservation/verify.ps1';parent_sha256='b6f36c3b8a190e47246cdc58ce99a6f96f891317726a4601bdc23862bf8412e8'}
 concurrency=[ordered]@{repository='cuwacunu_dev';policy='Only selected V3 sources/checkpoint fixed; no reset/equality claim for explicitly declared concurrent integration controls';observed_exclusions=@($script:observedConcurrent.Values);git_status=$devStatus}
 limitations=@('File/hash/metadata history only; no quality arithmetic/model deserialization','Prior bytes remain whole snapshots; superseded context uses prepends/history, zero deletion')
 actions=[ordered]@{models=0;fits=0;builds=0;training_updates=0;Git_mutations=0;deletions=0;files_written=@('doc/causal_fresh_v12/final-preservation.json')}
}
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($receipt|ConvertTo-Json -Depth 90).Replace([string][char]13,'')+"`n")
$stream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
[ordered]@{status='passed';checks=$receipt.checks;critical114=$critical.Count;old1030=$old.pins.Count;receipt=(Pin 'doc/causal_fresh_v12/final-preservation.json')}|ConvertTo-Json -Depth 5

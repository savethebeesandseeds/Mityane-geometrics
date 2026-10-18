# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Bounded host file/hash/JSON chain check; never replays diagnosis or audit.
param(
 [Parameter(Mandatory=$true)][switch]$RootReviewedFinalReport,
 [Parameter(Mandatory=$true)][string]$ReportPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ReportSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootReviewedFinalReport){throw 'Wait for the actual frozen human report and root review'}
$research=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspace=[IO.Path]::GetFullPath((Join-Path $research '..'))
$destination=Join-Path $PSScriptRoot 'completion.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve every prior completion receipt'}
$script:checks=0
function Check([bool]$value,[string]$label){$script:checks++;if(-not$value){throw $label}}
function Bounded([string]$label,[string]$repository='cuwacunu_embedding'){
 Check ($repository-in@('cuwacunu_embedding','cuwacunu_dev')-and-not[IO.Path]::IsPathRooted($label)-and$label-notmatch'[\\:]'-and@($label.Split('/')|Where-Object{$_-in@('','..','.')}).Count-eq0) 'Bounded existing-repository label'
 $base=Join-Path $workspace $repository;$path=[IO.Path]::GetFullPath((Join-Path $base $label))
 Check ($path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Inspection stays in repository'
 $walk=$path
 while($walk-ne$base){$item=Get-Item -LiteralPath $walk;Check (($item.Attributes-band[IO.FileAttributes]::ReparsePoint)-eq0) 'No linked preservation path';$walk=[IO.Path]::GetDirectoryName($walk)}
 return $path
}
function Pin([string]$label,[string]$repository='cuwacunu_embedding'){
 $path=Bounded $label $repository;$file=Get-Item -LiteralPath $path
 Check (-not$file.PSIsContainer) 'Regular evidence file'
 return [ordered]@{repository=$repository;path=$label;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Verify($binding,[string]$label,[string]$repository='cuwacunu_embedding'){
 $p=Pin $label $repository;Check ($p.bytes-eq$binding.bytes-and$p.sha256-ceq$binding.sha256) ('Exact bytes: '+$label);return $p
}
function Read([string]$label){Get-Content -LiteralPath (Bounded $label) -Raw|ConvertFrom-Json}
function Same($a,$b){($a|ConvertTo-Json -Depth 40 -Compress)-ceq($b|ConvertTo-Json -Depth 40 -Compress)}
$prefix='doc/causal_v6_v8_train_diagnosis/'
$mainCardPin=Pin 'doc/CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.json'
Check ($mainCardPin.sha256-ceq'49b22f1360593f2f0dd6d8ebb7fd8b0d4baa4749fe01ad8875fa79e5fe529829') 'Frozen main card unchanged'
$mainCard=Read $mainCardPin.path
Check ($mainCard.allowed_inputs.Count-eq52-and$mainCard.preservation_pins.Count-eq33) 'Original TRAIN52/context33 scope'
$all=@($mainCard.allowed_inputs)+@($mainCard.preservation_pins)+@($mainCard.pre_execution_preservation)
Check ($all.Count-eq86-and@($all.id|Sort-Object -Unique).Count-eq86) 'All86 distinct historical/data guards'
$guards=@();foreach($p in $all){$guards+=Verify $p $p.host_path $p.host_repository}
$broad=Read 'doc/causal_fresh_v8/final-preservation.json'
Check ($mainCard.pre_execution_preservation.sha256-ceq'919b4f11610e3ab61b63afbc3612145506e4c47a3be3c777f119d47c49e7126a'-and$broad.status-eq'passed'-and$broad.critical_checkpoint_pins.Count-eq94-and$broad.old_pins-eq1030) 'Preserved final94/old1030 receipt'
$critical=@();foreach($p in $broad.critical_checkpoint_pins){$critical+=Verify $p $p.path $p.repository}
$old=Read $broad.old_before_inventory.path;$null=Verify $broad.old_before_inventory $broad.old_before_inventory.path
$selected=@()
foreach($repository in @('cuwacunu_embedding','cuwacunu_dev')){
 $pins=@($old.pins|Where-Object{$_.repository-eq$repository-and$_.path-like'code/encoders/causal_patch_bottleneck_mae/versions/v3/*'})
 Check ($pins.Count-eq47) 'Exactly47 selected V3 physical sources in each repository'
 foreach($p in $pins){$selected+=Verify $p $p.path $repository}
 $baseline=Pin 'output/baselines/CPB-v3.market-01/inference.pt' $repository
 Check ($baseline.bytes-eq1290625-and$baseline.sha256-ceq'50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e') 'Exact selected V3 checkpoint'
}
$currentLedger=Verify $broad.current_ledger $broad.current_ledger.path
$currentCatalog=Verify $broad.current_catalog $broad.current_catalog.path
$catalog=Read $currentCatalog.path;$ledger=Read $currentLedger.path
Check ($catalog.active_application-eq'CPB-v3.market-01'-and$ledger.baseline-eq'CPB-v3.market-01'-and-not$ledger.checkpoint_deletion_allowed) 'No baseline change or deletion permission'
$mainSource=@();foreach($p in $mainCard.owned_source_files){$mainSource+=Verify $p $p.path}
$mainResultPin=Pin ($prefix+'diagnosis.json')
Check ($mainResultPin.bytes-eq4617944-and$mainResultPin.sha256-ceq'e48e0dc05a72cd2e514ad8a5480106ba2eba5c49e7903f9565cf1b9d1c552a43') 'Actual single main result exact'
$mainResult=Read $mainResultPin.path
Check ($mainResult.status-eq'passed'-and$mainResult.card.sha256-ceq$mainCardPin.sha256-and@($mainResult.counters.PSObject.Properties|Where-Object{$_.Value-ne0}).Count-eq0) 'Main result binds frozen card and zero forbidden operations'
$mainMirrorPin=Pin ($prefix+'complete-mirror.json');$mainMirror=Read $mainMirrorPin.path
Check ($mainMirror.status-eq'passed'-and$mainMirror.phase-eq'Run'-and$mainMirror.files.Count-eq35-and$mainMirror.source_capsule_files-eq21) 'Main35-file complete mirror'
$mainMirrorRoot=[IO.Path]::GetRelativePath($research,$mainMirror.host_root).Replace([char]92,[char]47)
$mainMapped=@();foreach($p in $mainMirror.files){$mainMapped+=Verify $p ($mainMirrorRoot+'/'+$p.path)}
Check (@(Get-ChildItem -LiteralPath (Bounded $mainMirrorRoot) -Recurse -Force -File).Count-eq35) 'Exact main mirror membership'
$auditCardPin=Pin ($prefix+'independent-audit-card.json');$auditCard=Read $auditCardPin.path
Check ($auditCardPin.sha256-ceq'ecb91be7577998db46e3643033667a24cac668070c32244bfbee3d2f1d8c5e98'-and$auditCard.main_card.sha256-ceq$mainCardPin.sha256-and$auditCard.main_diagnosis.sha256-ceq$mainResultPin.sha256) 'Audit card binds exact main result/card'
$auditSource=@();foreach($p in $auditCard.auditor_sources){$auditSource+=Verify $p $p.path}
$helpers=@();foreach($p in $auditCard.orchestration_helpers){$helpers+=Verify $p $p.path}
$auditPin=Pin ($prefix+'independent-audit.json');$audit=Read $auditPin.path
Check ($auditPin.bytes-eq80835-and$auditPin.sha256-ceq'3d085bc6d7f6403266752f8663a270eea9f00a5a7fde02a98b65e5c936c92f23') 'Actual once-only independent audit exact'
Check ($audit.status-eq'passed'-and$audit.checks-eq642801-and$audit.mismatches.Count-eq0-and$audit.finite_difference_checks-eq256-and$audit.card.sha256-ceq$mainCardPin.sha256-and$audit.diagnosis.sha256-ceq$mainResultPin.sha256-and$audit.audit_card.sha256-ceq$auditCardPin.sha256) 'Passed audit retains exact source/result chain'
$auditExecutionPin=Pin ($prefix+'independent-audit-execution.json');$execution=Read $auditExecutionPin.path
Check ($execution.status-eq'passed'-and$execution.exit_code-eq0-and$execution.result.sha256-ceq$auditPin.sha256-and(Same $execution.source_before $execution.source_after)-and$execution.source_before.Count-eq10) 'Audit one execution and exact10-file before-after state'
$auditMirrorPin=Pin ($prefix+'independent-audit-mirror.json');$auditMirror=Read $auditMirrorPin.path
Check ($auditMirror.status-eq'passed'-and$auditMirror.file_count-eq19-and$auditMirror.source_before_after_exact-and$auditMirror.all_hashes_exact-and$auditMirror.files.Count+1-eq19) 'Audit19-file exact full mirror'
$auditMapped=@();foreach($p in $auditMirror.files){$auditMapped+=Verify $p.host $p.host.path}
$auditMapped+=Verify $auditMirror.inventory $auditMirror.inventory.path
$auditMirrorRoot=[IO.Path]::GetRelativePath($research,$auditMirror.host_root).Replace([char]92,[char]47)
Check (@(Get-ChildItem -LiteralPath (Bounded $auditMirrorRoot) -Recurse -Force -File).Count-eq19) 'Exact audit mirror membership'
$failurePin=Pin ($prefix+'audit-mirror-invocation-failure-001.json');$failure=Read $failurePin.path
Check ($failure.status-eq'failed_before_helper_body'-and-not$failure.helper_body_invoked-and$failure.container_calls-eq0-and$failure.numeric_replays-eq0-and$failure.audit_result_sha256-ceq$auditPin.sha256) 'Preserved host parameter refusal caused no audit replay'
$reportPin=Pin $ReportPath
Check ($reportPin.sha256-ceq$ReportSha256) 'Exact actual root-reviewed final human report'
$receipt=[ordered]@{kind='cpb_v6_v8_saved_TRAIN_diagnosis_completion_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks;
 report=$reportPin;main_card=$mainCardPin;main_diagnosis=$mainResultPin;main_complete_mirror=$mainMirrorPin;main_mirrored_files=35;
 independent_audit_card=$auditCardPin;independent_audit=$auditPin;independent_audit_execution=$auditExecutionPin;independent_audit_mirror=$auditMirrorPin;audit_mirrored_files=19;
 audit_checks=$audit.checks;audit_mismatches=0;finite_difference_checks=$audit.finite_difference_checks;finite_difference_max_absolute_error=$audit.finite_difference_max_absolute_error;
 historical_preservation=$mainCard.pre_execution_preservation;historical_guard_file_bindings=$guards;critical_checkpoint_pins=$critical;critical_checkpoints=94;
 selected_V3_source_files_each_repo=47;selected_V3_sources=$selected;selected_V3_checkpoint_sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e';
 current_ledger=$currentLedger;current_catalog=$currentCatalog;baseline='CPB-v3.market-01';baseline_changed=$false;application_changed=$false;
 main_sources=$mainSource;auditor_sources=$auditSource;auditor_helpers=$helpers;main_mapped_files=$mainMapped;audit_mapped_files=$auditMapped;preserved_administrative_failure=$failurePin;
 interpretation='Metadata/file-byte chain only. Passed numerical assertions are copied from the frozen once-only independent audit; no arithmetic was replayed. The prior broad receipt remains unchanged; its old1030 bulk inventory is referenced, not reaudited here.';
 next_direction='No new candidate/model/quality/head/seed/cap/rate/bin setting or application promotion is selected by this completion receipt';
 actions=[ordered]@{project_runs=0;AST_calls=0;models=0;heads=0;fits=0;GPU_calls=0;Git_mutations=0;deletions=0;live_metadata_writes=0;files_written=@('doc/causal_v6_v8_train_diagnosis/completion.json')};
 verifier=(Pin 'doc/causal_v6_v8_train_diagnosis/verify-completion.ps1')}
[IO.File]::WriteAllText($destination,($receipt|ConvertTo-Json -Depth 45)+[char]10,[Text.UTF8Encoding]::new($false))
Pin 'doc/causal_v6_v8_train_diagnosis/completion.json'|ConvertTo-Json

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Bounded host file/hash/JSON chain check; never replays diagnosis or audit.
param(
 [Parameter(Mandatory=$true)][switch]$RootReviewedFinalReport,
 [Parameter(Mandatory=$true)][string]$ReportPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ReportSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainResultSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditResultSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainMirrorSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditMirrorSha256,
 [string]$AuditCardPath='doc/causal_v8_v9_train_diagnosis/independent-audit-card.json',
 [string[]]$OptionalFailurePaths=@(),
 [string[]]$OptionalFailureSha256=@()
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
 Check ($binding.sha256-is[string]-and$binding.sha256-cmatch'^[0-9a-f]{64}$'-and$binding.bytes-ge0) 'Typed file binding'
 $p=Pin $label $repository;Check ($p.bytes-eq$binding.bytes-and$p.sha256-ceq$binding.sha256) ('Exact bytes: '+$label);return $p
}
function Read([string]$label){Get-Content -LiteralPath (Bounded $label) -Raw|ConvertFrom-Json}
function Same($a,$b){($a|ConvertTo-Json -Depth 40 -Compress)-ceq($b|ConvertTo-Json -Depth 40 -Compress)}
function SamePin($a,$b,[string]$reason){Check ($a.bytes-eq$b.bytes-and$a.sha256-ceq$b.sha256) $reason}
function External([string]$label,[string]$sha){$pin=Pin $label;Check ($pin.sha256-ceq$sha) ('Externally reviewed SHA before JSON parse: '+$label);return $pin}
function Membership([string]$root,$names){
 $expected=@($names|Sort-Object);Check ($expected.Count-gt0-and@($expected|Sort-Object -Unique).Count-eq$expected.Count) 'Unique nonempty inventory membership'
 $base=Bounded $root
 $actual=@(Get-ChildItem -LiteralPath $base -Recurse -Force -File|ForEach-Object{[IO.Path]::GetRelativePath($base,$_.FullName).Replace([char]92,[char]47)}|Sort-Object)
 Check ($actual.Count-eq$expected.Count-and-not(Compare-Object $actual $expected)) 'Actual whole mirror membership exactly equals its inventory'
}
function MainMember([string]$name){$p=@($mainMirror.files|Where-Object{$_.path-ceq$name});Check ($p.Count-eq1) ('Unique main mirror member: '+$name);return Verify $p[0] ($mainMirrorRoot+'/'+$name)}
function AuditMember([string]$name){
 $p=@($auditMirror.files|Where-Object{$_.relative-ceq$name});Check ($p.Count-eq1) ('Unique audit mirror member: '+$name)
 Check ($p[0].host.path-ceq($auditMirrorRoot+'/'+$name)-and$p[0].container-ceq($auditMirror.container_root+'/'+$name)) 'Exact audit host/container mapping'
 return Verify $p[0].host $p[0].host.path
}
$prefix='doc/causal_v8_v9_train_diagnosis/'
$mainCardPin=External 'doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json' $MainCardSha256
$mainCard=Read $mainCardPin.path
Check ($mainCard.kind-eq'cpb_v8_v9_saved_train_diagnosis_card_v1'-and$mainCard.allowed_inputs.Count-eq56-and$mainCard.preservation_pins.Count-eq36-and$mainCard.preservation_pin_count-eq36) 'Frozen TRAIN56/context36 scope'
$all=@($mainCard.allowed_inputs)+@($mainCard.preservation_pins)+@($mainCard.pre_execution_preservation)
Check ($all.Count-eq93-and@($all.id|Sort-Object -Unique).Count-eq93) 'All93 distinct historical/data guards'
$guards=@();foreach($p in $all){$guards+=Verify $p $p.host_path $p.host_repository}
Check ($mainCard.pre_execution_preservation.host_path-ceq'doc/causal_fresh_v9/final-preservation.json') 'Actual V9 final99 guard, never inherited V8 receipt'
$broad=Read $mainCard.pre_execution_preservation.host_path
Check ($broad.kind-eq'causal_fresh_v9_final_preservation_v1'-and$broad.status-eq'passed'-and$broad.critical_checkpoint_total-eq99-and$broad.critical_checkpoint_pins.Count-eq99-and$broad.old_pins-eq1030-and$broad.prior_whole_metadata_chain_preserved-and$broad.selected_V3_source_files_each_repo-eq47) 'Preserved final99/old1030/selected47 receipt'
Check ($broad.all_pinned_PT_ignored.Count-eq2-and-not(Compare-Object @($broad.all_pinned_PT_ignored.repository|Sort-Object) @('cuwacunu_dev','cuwacunu_embedding'))-and@($broad.all_pinned_PT_ignored|Where-Object{-not$_.all_ignored}).Count-eq0) 'Two passed repository ignore records'
$critical=@();foreach($p in $broad.critical_checkpoint_pins){$critical+=Verify $p $p.path $p.repository}
Check (@($critical|ForEach-Object{$_.repository+'/'+$_.path}|Sort-Object -Unique).Count-eq99) 'Exactly99 unique checkpoint byte hashes; no deserialization'
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
Check ($mainSource.Count-eq5-and@($mainSource.path|Sort-Object -Unique).Count-eq5) 'Five unchanged owned main sources'
$mainResultPin=External ($prefix+'diagnosis.json') $MainResultSha256
$mainResult=Read $mainResultPin.path
Check ($mainResult.status-eq'passed'-and$mainResult.card.sha256-ceq$mainCardPin.sha256-and@($mainResult.counters.PSObject.Properties|Where-Object{$_.Value-ne0}).Count-eq0) 'Main result binds frozen card and zero forbidden operations'
Check ($mainResult.all_owned_source_files_unchanged-and$mainResult.verified_allowed_inputs.Count-eq56-and$mainResult.verified_preservation_pins.Count-eq36-and$mainResult.verified_pre_execution_preservation.sha256-ceq$mainCard.pre_execution_preservation.sha256) 'Main result records exact93 guards and unchanged source'
$mainMirrorPin=External ($prefix+'complete-mirror.json') $MainMirrorSha256;$mainMirror=Read $mainMirrorPin.path
Check ($mainMirror.kind-eq'cpb_v8_v9_TRAIN_diagnosis_exact_mirror_v1'-and$mainMirror.status-eq'passed'-and$mainMirror.phase-eq'Run'-and$mainMirror.source_capsule_files-eq24-and$mainMirror.all_sizes_sha256_exact-and$mainMirror.container_before_after_exact) 'Actual complete main mirror; no anticipated member count'
$mainMirrorRoot=[IO.Path]::GetRelativePath($research,$mainMirror.host_root).Replace([char]92,[char]47)
Check ($mainMirrorRoot.StartsWith('output/engineering/causal-v8-v9-train-diagnosis/attempt-001/',[StringComparison]::Ordinal)) 'Bounded main mirror namespace'
$mainMapped=@();foreach($p in $mainMirror.files){$mainMapped+=Verify $p ($mainMirrorRoot+'/'+$p.path)}
Membership $mainMirrorRoot @($mainMirror.files.path)
SamePin (MainMember 'source/CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json') $mainCardPin 'Mirrored main card exact';SamePin (MainMember 'run-001/diagnosis.json') $mainResultPin 'Mirrored main result exact'
$transportPin=MainMember 'manifest.json';$transport=Read $transportPin.path
Check ($transport.kind-eq'cpb_v8_v9_TRAIN_diagnosis_execution_capsule_v1'-and$transport.files.Count-eq24-and$transport.input_pins-eq93-and$transport.container_root-ceq$mainMirror.container_root) 'Main24-file93-pin capsule identity'
SamePin $transport.card $mainCardPin 'Capsule binds exact main card'
foreach($p in $transport.files){$null=Verify $p ($mainMirrorRoot+'/source/'+$p.path)}
Membership ($mainMirrorRoot+'/source') @($transport.files.path)
$null=Verify $transport.archive ($mainMirrorRoot+'/source.zip');$null=Verify $transport.wrapper ($mainMirrorRoot+'/phase-wrapper.py');$null=Verify $transport.helper $transport.helper.path
$preflightPin=Verify $transport.preflight ($prefix+'preflight-review.json');$preflight=Read $preflightPin.path
Check ($preflight.status-eq'passed'-and$preflight.card_sha256-ceq$MainCardSha256-and$preflight.allowed_input_pins-eq56-and$preflight.preservation_pins-eq36-and$preflight.metadata_copies-eq16) 'Actual passed preflight exact56/36/16 declaration'
$mainStagePin=MainMember 'source-stage.json';$mainSchemaPin=MainMember 'schema-stage.json';$mainExecutionPin=MainMember 'execution-stage.json'
foreach($p in @($mainStagePin,$mainSchemaPin,$mainExecutionPin)){$null=Verify $p ($prefix+[IO.Path]::GetFileName($p.path))}
$stage=Read $mainStagePin.path;$schema=Read $mainSchemaPin.path;$mainExecution=Read $mainExecutionPin.path
foreach($phase in @($stage,$schema,$mainExecution)){
 Check ($phase.status-eq'passed'-and$phase.input_pins_verified-eq93-and$phase.source_capsule_files-eq24) 'Main phases each bind93 guards/24 sources'
 SamePin $phase.manifest $transportPin 'Main phase capsule exact';SamePin $phase.card $mainCardPin 'Main phase card exact';SamePin $phase.wrapper $transport.wrapper 'Main phase wrapper exact'
}
Check ($stage.ast_sources-eq3-and$stage.numeric_storage_reads-eq0-and$stage.project_module_imports-eq0-and$schema.kind-eq'cpb_v8_v9_TRAIN_diagnosis_schema_v1'-and$schema.reviewed_stage_sha256-ceq$mainStagePin.sha256-and$mainExecution.reviewed_stage_sha256-ceq$mainStagePin.sha256-and$mainExecution.reviewed_schema_sha256-ceq$mainSchemaPin.sha256-and$mainExecution.exit_code-eq0-and$mainExecution.once_only) 'Actual Stage then Schema then separately admitted once-only Run'
SamePin $mainExecution.report $mainResultPin 'Actual Run result exact'
$auditCardPin=External $AuditCardPath $AuditCardSha256;$auditCard=Read $auditCardPin.path
Check ($auditCard.kind-eq'cpb_v8_v9_TRAIN_diagnosis_independent_audit_card_v1'-and$auditCard.main_card.sha256-ceq$mainCardPin.sha256-and$auditCard.main_diagnosis.sha256-ceq$mainResultPin.sha256) 'Audit card binds exact main result/card'
$auditSource=@();foreach($p in $auditCard.auditor_sources){$auditSource+=Verify $p $p.path}
Check ($auditSource.Count-eq5-and@($auditSource.path|Sort-Object -Unique).Count-eq5) 'Five unchanged independently owned auditor sources'
$helpers=@();foreach($p in $auditCard.orchestration_helpers){$helpers+=Verify $p $p.path}
$auditPin=External ($prefix+'independent-audit.json') $AuditResultSha256;$audit=Read $auditPin.path
Check ($audit.kind-eq'cpb_v8_v9_TRAIN_diagnosis_independent_audit_v1'-and$audit.status-eq'passed'-and$audit.checks-gt0-and$audit.mismatches-is[array]-and$audit.mismatches.Count-eq0-and$audit.finite_difference_checks-eq256-and$audit.card.sha256-ceq$mainCardPin.sha256-and$audit.diagnosis.sha256-ceq$mainResultPin.sha256-and$audit.audit_card.sha256-ceq$auditCardPin.sha256) 'Passed actual audit, zero mismatches,256 fixed finite differences; check total not anticipated'
Check ($audit.model_calls-eq0-and$audit.neural_forwards-eq0-and$audit.fits-eq0-and$audit.training_updates-eq0-and$audit.held_out_tensor_reads-eq0) 'Independent audit has zero forbidden operation counters'
$auditMirrorPin=External ($prefix+'independent-audit-mirror.json') $AuditMirrorSha256;$auditMirror=Read $auditMirrorPin.path
Check ($auditMirror.kind-eq'cpb_v8_v9_TRAIN_independent_audit_mirror_v1'-and$auditMirror.status-eq'passed'-and$auditMirror.source_before_after_exact-and$auditMirror.all_hashes_exact) 'Actual exact full audit mirror'
$auditMirrorRoot=[IO.Path]::GetRelativePath($research,$auditMirror.host_root).Replace([char]92,[char]47)
Check ($auditMirrorRoot.StartsWith('output/engineering/causal-v8-v9-train-diagnosis-audit/attempt-',[StringComparison]::Ordinal)) 'Bounded audit mirror namespace'
$auditMapped=@();foreach($p in $auditMirror.files){$auditMapped+=AuditMember $p.relative}
$inventoryPin=Verify $auditMirror.inventory $auditMirror.inventory.path;$auditMapped+=$inventoryPin
Check ($inventoryPin.path-ceq($auditMirrorRoot+'/mirror-inventory.json')) 'Inventory stays within exact mirror'
$inventory=Read $inventoryPin.path
Check ($inventory.status-eq'passed'-and$inventory.externally_reviewed_audit_card_sha256-ceq$AuditCardSha256-and$inventory.files.Count-eq$auditMirror.files.Count) 'Raw inventory binds actual card and mapped membership'
foreach($p in $inventory.files){Check ($p.path-ceq($auditMirror.container_root+'/'+$p.relative)) 'Exact raw inventory container mapping';SamePin $p (AuditMember $p.relative) 'Raw container inventory and mapped file bytes agree'}
Membership $auditMirrorRoot (@($auditMirror.files.relative)+@('mirror-inventory.json'))
Check ($auditMirror.file_count-eq$auditMapped.Count) 'Audit mirror count derives from exact observed membership'
SamePin (AuditMember 'source/MAIN_CARD.json') $mainCardPin 'Auditor main-card copy exact';SamePin (AuditMember 'source/diagnosis.json') $mainResultPin 'Auditor diagnosis copy exact';SamePin (AuditMember 'source/audit-card.json') $auditCardPin 'Audit declaration copy exact';SamePin (AuditMember 'run-001/audit.json') $auditPin 'Audit raw/durable result exact'
$auditExecutionPin=AuditMember 'run-001/execution.json';$null=Verify $auditExecutionPin ($prefix+'independent-audit-execution.json');$execution=Read $auditExecutionPin.path
Check ($execution.status-eq'passed'-and$execution.exit_code-eq0-and$execution.result.sha256-ceq$auditPin.sha256-and(Same $execution.source_before $execution.source_after)-and$execution.source_before.Count-gt0) 'Successful once-only audit and observed before-after source equality'
$auditStagePin=AuditMember 'source-stage.json';$auditStage=Read $auditStagePin.path
Check ($auditStage.status-eq'passed_AST_only'-and$auditStage.data_and_preservation_pins-eq93-and$auditStage.files-eq$execution.source_before.Count-and$execution.externally_reviewed_stage_sha256-ceq$auditStagePin.sha256-and$inventory.externally_reviewed_stage_sha256-ceq$auditStagePin.sha256-and$auditStage.audit_card_sha256-ceq$AuditCardSha256) 'Actual audit execution/inventory bind the same93-pin AST-only Stage'
$auditCapsulePin=AuditMember 'source/CAPSULE_MANIFEST.json'
Check ($auditStage.manifest_sha256-ceq$auditCapsulePin.sha256) 'Audit Stage binds its exact raw capsule manifest'
$auditCapsule=Read $auditCapsulePin.path
Check ($auditCapsule.kind-eq'cpb_v8_v9_TRAIN_independent_audit_source_capsule_v1'-and$auditCapsule.audit_card_sha256-ceq$AuditCardSha256-and$auditCapsule.container_root-ceq$auditMirror.container_root-and$auditCapsule.files.Count-eq$execution.source_before.Count) 'Actual audited source capsule membership and card'
foreach($p in $auditCapsule.files){SamePin $p (AuditMember ('source/'+$p.file)) 'Audit capsule manifest and mirrored source bytes exact'}
Membership ($auditMirrorRoot+'/source') (@($auditCapsule.files.file)+@('CAPSULE_MANIFEST.json'))
Check ($auditStage.source_zip_sha256-ceq(AuditMember 'source.zip').sha256) 'Audit Stage binds exact raw capsule ZIP'
foreach($p in $execution.source_before){Check ($p.path.StartsWith($auditMirror.container_root+'/source/',[StringComparison]::Ordinal)) 'Observed audit source stays in own source leaf';$relative=$p.path.Substring(($auditMirror.container_root+'/').Length);SamePin $p (AuditMember $relative) 'Observed execution source equals exact mirror'}
Check ($OptionalFailurePaths.Count-eq$OptionalFailureSha256.Count-and@($OptionalFailurePaths|Sort-Object -Unique).Count-eq$OptionalFailurePaths.Count) 'Optional actual failure labels and external hashes pair uniquely'
$failures=@()
for($i=0;$i-lt$OptionalFailurePaths.Count;$i++){
 Check ($OptionalFailureSha256[$i]-cmatch'^[0-9a-f]{64}$'-and$OptionalFailurePaths[$i].StartsWith($prefix,[StringComparison]::Ordinal)) 'Explicit bounded optional failure binding'
 $pin=External $OptionalFailurePaths[$i] $OptionalFailureSha256[$i];$failure=Read $pin.path
 Check ($failure.status-is[string]-and$failure.status.StartsWith('failed',[StringComparison]::Ordinal)) 'Only actual externally declared failed dispositions are included'
 $failures+=$pin
}
$reportPin=Pin $ReportPath
Check ($reportPin.sha256-ceq$ReportSha256) 'Exact actual root-reviewed final human report'
$receipt=[ordered]@{kind='cpb_v8_v9_saved_TRAIN_diagnosis_completion_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks;
 report=$reportPin;main_card=$mainCardPin;main_diagnosis=$mainResultPin;main_complete_mirror=$mainMirrorPin;main_mirrored_files=$mainMapped.Count;main_stage=$mainStagePin;main_schema=$mainSchemaPin;main_execution=$mainExecutionPin;
 independent_audit_card=$auditCardPin;independent_audit=$auditPin;independent_audit_execution=$auditExecutionPin;independent_audit_mirror=$auditMirrorPin;audit_mirrored_files=$auditMapped.Count;
 audit_checks=$audit.checks;audit_mismatches=0;finite_difference_checks=$audit.finite_difference_checks;finite_difference_max_absolute_error=$audit.finite_difference_max_absolute_error;
 historical_preservation=$mainCard.pre_execution_preservation;historical_guard_file_bindings=$guards;guard_identities=93;critical_checkpoint_pins=$critical;critical_checkpoints=99;
 selected_V3_source_files_each_repo=47;selected_V3_sources=$selected;selected_V3_checkpoint_sha256='50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e';
 current_ledger=$currentLedger;current_catalog=$currentCatalog;baseline='CPB-v3.market-01';baseline_changed=$false;application_changed=$false;
 main_sources=$mainSource;auditor_sources=$auditSource;auditor_helpers=$helpers;main_mapped_files=$mainMapped;audit_mapped_files=$auditMapped;preserved_declared_failures=$failures;
 interpretation='Metadata/file-byte chain only. Passed numerical assertions are copied from the frozen once-only independent audit; no arithmetic was replayed. The prior broad receipt remains unchanged; its old1030 bulk inventory is referenced, not reaudited here.';
 next_direction='No new candidate/model/quality/head/seed/cap/rate/bin setting or application promotion is selected by this completion receipt';
 actions=[ordered]@{project_runs=0;AST_calls=0;models=0;heads=0;fits=0;GPU_calls=0;Git_mutations=0;deletions=0;live_metadata_writes=0;files_written=@('doc/causal_v8_v9_train_diagnosis/completion.json')};
 verifier=(Pin 'doc/causal_v8_v9_train_diagnosis/verify-completion.ps1')}
$stream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($receipt|ConvertTo-Json -Depth 60).Replace([string][char]13,'')+"`n");$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Pin 'doc/causal_v8_v9_train_diagnosis/completion.json'|ConvertTo-Json

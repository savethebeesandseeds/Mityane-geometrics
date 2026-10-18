# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Bounded final hash/metadata join of completed saved-TRAIN evidence.
# No imports, Docker, archive decoding, tensor/scientific arithmetic or replay.
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
 [string[]]$OptionalFailurePaths=@(),
 [string[]]$OptionalFailureSha256=@()
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootReviewedFinalReport){throw 'Actual human report and exact completed chain require root review'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $researchRoot '..'))
$doc='doc/causal_v11_v13_train_diagnosis/'
$destination=Join-Path $researchRoot ($doc+'completion.json')
if(Test-Path -LiteralPath $destination){throw 'Preserve earlier completion; no overwrite'}
$script:checks=0
function Check([bool]$ok,[string]$why){$script:checks++;if(-not$ok){throw $why}}
function Bounded([string]$relative,[string]$repository='cuwacunu_embedding'){
 Check ($repository-in@('cuwacunu_embedding','cuwacunu_dev')-and-not[IO.Path]::IsPathRooted($relative)-and$relative-notmatch'[\\:]'-and@($relative.Split('/')|Where-Object{$_-in@('','..','.')} ).Count-eq0) 'Bounded explicit repository-relative file'
 $base=[IO.Path]::GetFullPath((Join-Path $workspaceRoot $repository));$p=[IO.Path]::GetFullPath((Join-Path $base $relative))
 Check ($p.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Within repository'
 $walk=$p
 while($walk-ne$base){$item=Get-Item -LiteralPath $walk;Check (-not($item.Attributes-band[IO.FileAttributes]::ReparsePoint)) 'Unlinked ancestors';$walk=[IO.Path]::GetDirectoryName($walk)}
 Check (-not(Get-Item -LiteralPath $p).PSIsContainer) 'Physical regular evidence file';return $p
}
function Pin([string]$relative,[string]$repository='cuwacunu_embedding'){
 $p=Bounded $relative $repository
 [ordered]@{path=$relative;bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Same($a,$b){$a.bytes-eq$b.bytes-and$a.sha256-ceq$b.sha256}
function Equal($a,$b){($a|ConvertTo-Json -Depth 80 -Compress)-ceq($b|ConvertTo-Json -Depth 80 -Compress)}
function External([string]$relative,[string]$hash){
 $p=Pin $relative;Check ($p.sha256-ceq$hash) ('External reviewed bytes before parse: '+$relative);return $p
}
function Read($pin){Get-Content -LiteralPath (Bounded $pin.path) -Raw|ConvertFrom-Json}
function Exact($observed,$expected,[string]$why){Check (@($observed).Count-eq@($expected).Count-and@($observed|Sort-Object -Unique).Count-eq@($expected).Count-and-not(Compare-Object @($observed|Sort-Object) @($expected|Sort-Object))) $why}
$reportPin=External $ReportPath $ReportSha256
$mainCardPin=External 'doc/CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json' $MainCardSha256
$mainPin=External ($doc+'diagnosis.json') $MainResultSha256
$auditCardPin=External ($doc+'independent-audit-card.ADMITTED.json') $AuditCardSha256
$auditPin=External ($doc+'independent-audit.json') $AuditResultSha256
$mainMirrorPin=External ($doc+'mirror.json') $MainMirrorSha256
$auditMirrorPin=External ($doc+'independent-audit-mirror.json') $AuditMirrorSha256
$mainCard=Read $mainCardPin;$main=Read $mainPin;$auditCard=Read $auditCardPin;$audit=Read $auditPin;$mainMirror=Read $mainMirrorPin;$auditMirror=Read $auditMirrorPin
Check ($mainCard.kind-eq'cpb_v11_v13_saved_train_diagnosis_card_v1'-and$mainCard.execution_admitted-is[bool]-and$mainCard.execution_admitted-and$mainCard.allowed_inputs.Count-eq25-and$mainCard.hash_only_guards.Count-eq35-and$mainCard.context_inputs.Count-eq14) 'Exact main admitted scope'
Check ($main.kind-eq'cpb_v11_v13_TRAIN_saved_information_diagnosis_v1'-and$main.status-eq'passed_readonly_saved_TRAIN_diagnosis'-and(Same $main.card $mainCardPin)-and$main.all_owned_source_files_unchanged-and$main.baseline-eq'CPB-v3.market-01') 'Actual completed V13 diagnosis'
Check ($auditCard.kind-eq'cpb_v11_v13_TRAIN_diagnosis_independent_audit_card_v1'-and$auditCard.execution_admitted-is[bool]-and$auditCard.execution_admitted-and(Same $auditCard.main_card $mainCardPin)-and(Same $auditCard.main_diagnosis $mainPin)-and(Same $auditCard.main_mirror $mainMirrorPin)) 'Exact admitted independent card/result/mirror'
Check ($audit.kind-eq'cpb_v11_v13_TRAIN_diagnosis_independent_audit_v1'-and$audit.status-eq'passed'-and$audit.checks-gt0-and@($audit.mismatches).Count-eq0-and(Same $audit.card $mainCardPin)-and(Same $audit.diagnosis $mainPin)-and(Same $audit.audit_card $auditCardPin)) 'Actual dynamic independent check count, zero mismatches'
$zero=@('model_calls','encoder_calls','decoder_calls','head_calls','module_forwards','neural_forwards','fits','head_fits','training_updates','held_out_tensor_reads','old_V11_raw_numeric_reads','old_V12_numeric_reads','old_V9_numeric_reads','old_V8_numeric_reads','local_adjoint_rechecks','learning_gauge_Hadamard_trust_rechecks')
foreach($record in @($main,$audit)){foreach($name in $zero){Check ($record.$name-is[ValueType]-and$record.$name-eq0) ('Forbidden body counter remains zero: '+$name)}}
Check ($main.new_encoder_experiment-is[bool]-and-not$main.new_encoder_experiment-and-not$main.candidate_declared-and-not$main.new_quality_scores-and-not$main.diagnostic_gate_or_next_encoder_selected-and-not$main.application_changed-and-not$audit.next_encoder_selected-and-not$audit.application_changed-and$audit.new_quality_scores-eq0) 'No new encoder, quality gate, application or selection'
Exact @($main.models.PSObject.Properties.Name) @('v13') 'V13-only new stage panel'
Exact @($main.models.v13.stages.PSObject.Properties.Name) @('initial','final') 'Two actual new V13 stages'
Exact @($main.cached_v11_TRAIN.PSObject.Properties.Name) @('stages','temporal_moment_observations') 'Exact cached reference container'
Exact @($main.cached_v11_TRAIN.stages.PSObject.Properties.Name) @('initial','final') 'Two cached V11 stages'
Exact @($main.cached_v11_TRAIN.temporal_moment_observations.PSObject.Properties.Name) @('csv','actual_witnesses') 'Only cached V11 actual moment panels'
$selectors=@('cached_v11_TRAIN.stages.initial','cached_v11_TRAIN.stages.final','cached_v11_TRAIN.temporal_moment_observations.csv','cached_v11_TRAIN.temporal_moment_observations.actual_witnesses')
Check ((Equal $main.cached_V11_numeric_selectors $selectors)-and(Equal $audit.cached_V11_selectors $selectors)-and(Equal $audit.computed_versions @('v13'))-and(Equal $audit.cached_reference_versions @('v11'))-and$main.V11_new_numerical_reductions-eq0) 'Only four opaque cached prior subtrees; no V11 reductions'
$families=@('ridge_total','ridge_block','mean4_decoder','temporal_mean4_export','temporal_regularized_stddev4_export')
$manual=$main.manual_arithmetic_counters.manual_saved_affine_contractions
Exact @($manual.PSObject.Properties.Name) $families 'Five disclosed manual affine families'
Exact @($audit.manual_saved_affine_contractions.PSObject.Properties.Name) $families 'Five independently counted affine families'
foreach($name in $families){
 foreach($unit in @('dot_outputs','multiply_add_terms')){Check ($manual.$name.$unit-is[ValueType]-and$manual.$name.$unit-gt0-and$manual.$name.$unit-eq$audit.manual_saved_affine_contractions.$name.$unit) 'Positive manual work independently counted, distinct from module zero'}
}
$scalar=$main.manual_arithmetic_counters.scalar_summary_function_calls
Exact @($scalar.PSObject.Properties.Name) @('mean','quantiles','variance','pearson','score') 'Disclosed scalar function counters'
foreach($p in $scalar.PSObject.Properties){Check ($p.Value-is[ValueType]-and$p.Value-gt0) 'Positive implementation-dependent scalar calls; no forced auditor equality'}
$slope=$main.manual_arithmetic_counters.captured_root_slope_scalar_divisions
Check ($slope-is[ValueType]-and$slope-ge0-and$slope-eq$audit.captured_root_slope_scalar_divisions) 'Independent present-coordinate captured-root slope division count, separate from affine families'
$ready=Pin $mainCard.scientific_source_review.path
Check (Same $ready $mainCard.scientific_source_review) 'Exact main-card-bound combined source review'
$science=Read $ready
Check ($science.kind-eq'causal_v11_v13_diagnosis_source_root_review_v1'-and$science.status-eq'passed_source_review'-and$science.execution_admitted-is[bool]-and-not$science.execution_admitted-and$science.scientific_files.Count-eq10-and$science.workflow_files.Count-eq10) 'Source-only root review of20 exact files'
foreach($row in @($science.scientific_files)+@($science.workflow_files)){Check (Same (Pin $row.path) $row) 'All twenty scientific/workflow sources unchanged'}
Check (Same $auditCard.scientific_source_review $ready) 'Main and independent cards bind the same reviewed current20 source closure'
foreach($row in @($science.design)+@($science.parent_preservation)){Check (Same (Pin $row.path) $row) 'Reviewed completed design and whole20-parent preservation unchanged'}
foreach($row in $auditCard.auditor_sources){Check (@($science.scientific_files|Where-Object{$_.path-eq$row.path-and(Same $_ $row)}).Count-eq1) 'Audit card sources belong directly to reviewed scientific10'}
foreach($row in $mainCard.orchestration_helpers){Check (Same (Pin $row.path) $row) 'All ten main-card-frozen workflow sources unchanged'}
foreach($row in $auditCard.auditor_sources){Check (Same (Pin $row.path) $row) 'Exact five independent frozen auditor sources'}
foreach($row in $auditCard.orchestration_helpers){Check (Same (Pin $row.path) $row) 'Audit helper closure unchanged'}
$all=@($mainCard.allowed_inputs)+@($mainCard.hash_only_guards)+@($mainCard.context_inputs)+@($mainCard.preservation_pins)+@($mainCard.pre_execution_preservation)
$hostFiles=@{};$runtime=@{}
foreach($row in $all){
 $key=$row.host_repository+'/'+$row.host_path
 if($hostFiles.ContainsKey($key)){Check (Same $hostFiles[$key] $row) 'Exact host alias still bound'}else{$p=Pin $row.host_path $row.host_repository;Check (Same $p $row) 'Every actual declared original input available/unchanged';$hostFiles[$key]=$p}
 if($runtime.ContainsKey($row.container_path)){Check (Same $runtime[$row.container_path] $row) 'Exact runtime alias retained'}else{$runtime[$row.container_path]=$row}
}
Check ($all.Count-eq$mainCard.logical_pin_occurrences-and$hostFiles.Count-eq$mainCard.unique_host_files-and$runtime.Count-eq$mainCard.unique_container_files-and$auditCard.main_input_binding_counts.occurrences-eq$all.Count-and$auditCard.main_input_binding_counts.unique_host_files-eq$hostFiles.Count-and$auditCard.main_input_binding_counts.unique_container_paths-eq$runtime.Count) 'Actual occurrence and unique physical counts match both cards'
Check ($mainMirror.status-eq'passed'-and$mainMirror.container_before_after_exact-and$mainMirror.all_sizes_sha256_exact-and(Same $mainMirror.canonical_diagnosis $mainPin)-and$auditMirror.status-eq'passed'-and$auditMirror.source_before_after_exact-and$auditMirror.all_hashes_exact-and(Same $auditMirror.independent_audit $auditPin)) 'Complete main/audit mirrors and exact canonical aliases'
$mainRoot=[IO.Path]::GetFullPath([string]$mainMirror.host_root)
$expectedRoot=Join-Path $researchRoot 'output/engineering/causal-v11-v13-train-diagnosis/attempt-001'
Check ($mainRoot.StartsWith($expectedRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Exact own main mirror root'
$mainFiles=@()
foreach($row in $mainMirror.files){
 Check (-not[IO.Path]::IsPathRooted($row.path)-and$row.path-notmatch'[\\:]'-and@($row.path.Split('/')|Where-Object{$_-in@('','..','.')} ).Count-eq0) 'Bounded mirrored member'
 $p=[IO.Path]::GetFullPath((Join-Path $mainRoot $row.path));Check ($p.StartsWith($mainRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Main member remains in mirror'
 $relative=$p.Substring($researchRoot.Length+1).Replace([char]92,[char]47);$pin=Pin $relative;Check (Same $pin $row) 'Every main mirror member exact';$mainFiles+=$pin
}
Check (@(Get-ChildItem -LiteralPath $mainRoot -Recurse -Force -File).Count-eq$mainMirror.files.Count) 'No unrecorded main mirror members'
foreach($row in $auditMirror.files){Check (Same (Pin $row.host.path) $row.host) 'Every independent audit mirror member exact'}
$manifestRelative=([IO.Path]::GetFullPath((Join-Path $mainRoot 'manifest.json'))).Substring($researchRoot.Length+1).Replace([char]92,[char]47)
$transport=Read (Pin $manifestRelative)
Check ($transport.source_capsule_files-eq$transport.files.Count-and$transport.source_capsule_files-eq$mainMirror.source_capsule_files-and$transport.input_pins-eq$all.Count-and$transport.card.sha256-eq$MainCardSha256) 'Dynamic main payload and declared role counts'
$executionPin=Pin ($doc+'execution.json');$execution=Read $executionPin
Check ($execution.status-eq'passed'-and$execution.phase-eq'Run'-and$execution.exit_code-eq0-and$execution.once_only-and(Same $execution.report $mainPin)-and(Same $execution.card $mainCardPin)) 'Once-only successful main execution binds exact result/card'
$independentExecution=Read $auditMirror.durable_execution
Check ($independentExecution.status-eq'passed'-and$independentExecution.exit_code-eq0-and(Same $independentExecution.result $auditPin)) 'Once-only successful independent execution binds exact result'
$contextMap=@{};foreach($row in $mainCard.context_inputs){$contextMap[$row.id]=$row}
$old=Get-Content -LiteralPath (Bounded $contextMap.old_audit.host_path) -Raw|ConvertFrom-Json
Check ($old.status-eq'passed'-and$old.checks-eq69952-and@($old.mismatches).Count-eq0-and(Same $main.cached_V11_independent_audit_reused $contextMap.old_audit)-and(Same $audit.cached_V11_original_audit $contextMap.old_audit)) 'Latest69952 cache audit reused as opaque exact context; upstream68264 remains separately bound'
Check (Same $audit.cached_V11_upstream_audit $old.cached_V11_original_audit) 'Independent audit preserves exact upstream V11 arithmetic-audit identity without replay'
$final=Get-Content -LiteralPath (Bounded $contextMap.v13_final_preservation.host_path) -Raw|ConvertFrom-Json
Check ($final.status-eq'passed'-and$final.critical_checkpoint_total-eq119-and$final.V13_source_files-eq143-and-not$final.current_reporting_recovery_required-and$final.baseline-eq'CPB-v3.market-01'-and-not$final.application_changed-and-not$final.baseline_changed) 'Completed119/143 preservation reused; no verifier replay'
$ledger=Get-Content -LiteralPath (Bounded $contextMap.current_ledger.host_path) -Raw|ConvertFrom-Json
$catalog=Get-Content -LiteralPath (Bounded $contextMap.current_catalog.host_path) -Raw|ConvertFrom-Json
Check ($ledger.baseline-eq'CPB-v3.market-01'-and$catalog.active_application-eq'CPB-v3.market-01') 'V3 remains operational baseline'
Check ($OptionalFailurePaths.Count-eq$OptionalFailureSha256.Count-and@($OptionalFailurePaths|Sort-Object -Unique).Count-eq$OptionalFailurePaths.Count) 'Explicit unique optional failure evidence pins'
$failures=@();for($i=0;$i-lt$OptionalFailurePaths.Count;$i++){$failures+=External $OptionalFailurePaths[$i] $OptionalFailureSha256[$i]}
$result=[ordered]@{
 kind='cpb_v11_v13_saved_TRAIN_diagnosis_completion_v1';status='passed';verified_utc=[DateTime]::UtcNow.ToString('o');checks=$script:checks
 report=$reportPin;main_card=$mainCardPin;main_diagnosis=$mainPin;independent_audit_card=$auditCardPin;independent_audit=$auditPin;main_mirror=$mainMirrorPin;independent_audit_mirror=$auditMirrorPin
 independent_checks=$audit.checks;independent_mismatches=0;scientific_source_review=$ready;logical_pin_occurrences=$all.Count;unique_host_files=$hostFiles.Count;unique_container_paths=$runtime.Count;main_mirrored_files=$mainMirror.files.Count;independent_mirrored_files=$auditMirror.file_count
 baseline='CPB-v3.market-01';application_changed=$false;new_candidate_declared=$false;new_quality_gate=$false;cached_old_audit=$contextMap.old_audit;cached_old_audit_checks=69952;cached_V11_upstream_audit=$audit.cached_V11_upstream_audit;old_audit_replayed=$false;actual_V13_trust_replayed=$false;completed_preservation_reused=$contextMap.v13_final_preservation;reused_critical_checkpoints=119;critical_checkpoints_rehashed_here=0
 manual_affine_counters=$manual;scalar_function_counters=$scalar;captured_root_slope_scalar_divisions=$slope;manual_counters_are_not_module_calls=$true;optional_failure_evidence=$failures
 limits=@('Completion verifies saved metadata/hash closure; independent body supplies arithmetic verification','No old V11 recomputation, finite differences or V13 local-adjoint/learning/gauge/Hadamard replay','TRAIN descriptors and saved readouts do not identify validation causes; missing captures remain missing')
 actions=[ordered]@{Docker=0;imports=0;AST=0;tensor_storage_reads=0;numerical_reductions=0;encoder_calls=0;module_calls=0;head_fits=0;training_updates=0;DEV_writes=0;historical_result_changes=0}
 source=(Pin ($doc+'verify-completion.ps1'))
}
$stream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try{$b=[Text.UTF8Encoding]::new($false).GetBytes(($result|ConvertTo-Json -Depth 60).Replace([string][char]13,'')+[char]10);$stream.Write($b,0,$b.Length)}finally{$stream.Dispose()}
Pin ($doc+'completion.json')|ConvertTo-Json

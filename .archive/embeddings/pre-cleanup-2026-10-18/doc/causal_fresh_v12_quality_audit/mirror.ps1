# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Exact file copying/hash inspection only; never executes an auditor or model.
param([ValidateSet('quality')][string]$Kind='quality',[ValidatePattern('^[0-9]{3}$')][string]$Attempt='001',
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$StageReceiptSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExecutionSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InventorySHA256,
 [Parameter(Mandatory=$true)][string]$WorkflowSourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WorkflowSourceReviewSHA256)
$ErrorActionPreference='Stop'
if($Attempt-ne'001'){throw 'This additive helper owns only the exclusive first audit attempt001'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$containerName='cuwacunu'
$containerRoot="/opt/cuwacunu/research/causal-fresh-v12-$Kind-audit/attempt-$Attempt"
$hostRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v12-$Kind-audit/attempt-$Attempt"
$hostMirror=Join-Path $hostRoot 'container-mirror'
function File-Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Reparse point forbidden'};[ordered]@{path=$f.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$entry){$p=File-Pin $path;if($p.bytes-ne$entry.bytes-or$p.sha256-ne$entry.sha256){throw "Mirror bytes differ: $path"};$p}
function Workflow-RepositoryPath([string]$value){
 $full=if([IO.Path]::IsPathRooted($value)){[IO.Path]::GetFullPath($value)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $value))}
 if(-not$full.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Repository-bounded workflow review required'}
 for($cursor=$full;$cursor.Length-ge$researchRoot.Length;$cursor=[IO.Path]::GetDirectoryName($cursor)){if((Get-Item -LiteralPath $cursor).Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked workflow/review path refused'};if($cursor-eq$researchRoot){break}}
 $full
}
function Workflow-Pin([string]$value){
 $full=Workflow-RepositoryPath $value;$file=Get-Item -LiteralPath $full
 [ordered]@{path=$full.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Read-IndependentWorkflowReview{
 $reviewPin=Workflow-Pin $WorkflowSourceReviewPath
 if($reviewPin.sha256-cne$WorkflowSourceReviewSHA256){throw 'External independent workflow-review SHA required before parse'}
 $review=Get-Content -LiteralPath (Workflow-RepositoryPath $reviewPin.path) -Raw|ConvertFrom-Json
 if($review.kind-cne'cpb_v12_quality_audit_workflow_independent_static_review_v1'-or$review.status-cne'passed'-or$review.source_only-isnot[bool]-or-not$review.source_only-or$review.independent_of_author-isnot[bool]-or-not$review.independent_of_author-or$review.scientific_execution_admitted-isnot[bool]-or$review.scientific_execution_admitted-ne$false){throw 'Independent SOURCE-ONLY workflow review with exact kind/status is required'}
 $prefix='doc/causal_fresh_v12_quality_audit/'
 $names=@('prepare-card.ps1','stage.ps1','execute.py','mirror.ps1','provenance.json')
 $paths=@($names|ForEach-Object{$prefix+$_})
 if($review.reviewed_sources.Count-ne$paths.Count-or(@($review.reviewed_sources.path|Sort-Object -Unique).Count)-ne$paths.Count-or(Compare-Object ($paths|Sort-Object) @($review.reviewed_sources.path|Sort-Object))){throw 'Review must bind exactly five owned workflow files including provenance'}
 foreach($entry in $review.reviewed_sources){$actual=Workflow-Pin $entry.path;if($actual.bytes-ne$entry.bytes-or$actual.sha256-cne$entry.sha256){throw ('Reviewed workflow source changed: '+$entry.path)}}
 $provenancePin=Workflow-Pin ($prefix+'provenance.json')
 $provenance=Get-Content -LiteralPath (Workflow-RepositoryPath $provenancePin.path) -Raw|ConvertFrom-Json
 $executablePaths=@($paths|Where-Object{$_-ne($prefix+'provenance.json')})
 if($provenance.kind-cne'cpb_v12_owned_quality_audit_orchestration_source_v1'-or$provenance.current_sources.Count-ne$executablePaths.Count-or(Compare-Object ($executablePaths|Sort-Object) @($provenance.current_sources.path|Sort-Object))){throw 'Exact complete four-helper provenance required'}
 foreach($entry in $provenance.current_sources){$matched=@($review.reviewed_sources|Where-Object{$_.path-ceq$entry.path});if($matched.Count-ne1-or$matched[0].bytes-ne$entry.bytes-or$matched[0].sha256-cne$entry.sha256){throw 'Independent review and current helper provenance disagree'}}
 [ordered]@{review_pin=$reviewPin;review=$review;provenance_pin=$provenancePin;provenance=$provenance}
}
$workflowGuard=Read-IndependentWorkflowReview
$cardPath=Join-Path $researchRoot 'doc/causal_fresh_v12_quality_audit/audit-card.json'
if((File-Pin $cardPath).sha256-ne$AuditCardSHA256){throw 'External audit card hash before parse required'}
$card=Get-Content -Raw -LiteralPath $cardPath|ConvertFrom-Json
$null=Require-Pin $PSCommandPath $card.mirror_helper
if($card.independent_workflow_source_review.sha256-cne$WorkflowSourceReviewSHA256-or$card.independent_workflow_source_review.bytes-ne$workflowGuard.review_pin.bytes-or$card.helpers_provenance.sha256-cne$workflowGuard.provenance_pin.sha256-or$card.helpers_provenance.bytes-ne$workflowGuard.provenance_pin.bytes){throw 'Exact externally reviewed current workflow required before any mirror operation'}
$hostStagePath=Join-Path $hostRoot 'source-stage.json'
if((File-Pin $hostStagePath).sha256-ne$StageReceiptSHA256){throw 'External host Stage hash before parse required'}
$stage=Get-Content -Raw -LiteralPath $hostStagePath|ConvertFrom-Json
if($stage.kind-ne'cpb_v12_quality_audit_source_stage_v1'-or$stage.status-ne'passed'-or$stage.audit_card_sha256-ne$AuditCardSHA256-or$stage.workflow_source_review_sha256-cne$WorkflowSourceReviewSHA256-or$stage.scientific_source_review_sha256-cne$card.independent_source_review.sha256){throw 'Exact passed source Stage and separate workflow/scientific authority required'}
if($stage.files-ne$card.staging_transport.payload_count-or$stage.ZIP_members-ne$card.staging_transport.ZIP_member_count-or$stage.python_AST_files-ne$card.staging_transport.python_AST_count){throw 'Source-derived payload/member/AST closure must match card and Stage'}
if(-not(Test-Path -LiteralPath $hostRoot)-or(Test-Path -LiteralPath $hostMirror)){throw 'Existing staging leaf and new mirror required; preserve every attempt'}
$observed=docker.exe inspect $containerName|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$observed.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$observed.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'){throw 'Exact existing managed container required'}
[IO.Directory]::CreateDirectory($hostMirror)|Out-Null
docker.exe cp "${containerName}:${containerRoot}/." $hostMirror
if($LASTEXITCODE-ne0){throw 'Audit mirror copy failed; retain partial copy'}
$inventoryPath=Join-Path $hostMirror 'mirror-inventory.json'
if((File-Pin $inventoryPath).sha256-ne$InventorySHA256){throw 'External inventory hash before parse required'}
$inventory=Get-Content -Raw -LiteralPath $inventoryPath|ConvertFrom-Json
if($inventory.kind-ne'cpb_v12_audit_complete_file_inventory_v1'){throw 'Exact completed wrapper inventory required'}
$expected=@();$mapped=@();$mirrorPrefix=[IO.Path]::GetFullPath($hostMirror).TrimEnd([char]92)+[IO.Path]::DirectorySeparatorChar
foreach($entry in $inventory.files){
 if($entry.relative-match'(^|/)\.\.(/|$)'-or[IO.Path]::IsPathRooted($entry.relative)){throw 'Unbounded inventory path'}
 $path=[IO.Path]::GetFullPath((Join-Path $hostMirror $entry.relative))
 if(-not$path.StartsWith($mirrorPrefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Inventory path outside new mirror'}
 $actual=Require-Pin $path $entry
 $expected+=$entry.relative;$mapped +=[ordered]@{container=$entry.path;relative=$entry.relative;host=$actual}
}
$expected+='mirror-inventory.json'
$actualFiles=@(Get-ChildItem -LiteralPath $hostMirror -Recurse -File|ForEach-Object{$_.FullName.Substring($hostMirror.Length+1).Replace([char]92,[char]47)})
if($expected.Count-ne$actualFiles.Count-or(Compare-Object ($expected|Sort-Object) ($actualFiles|Sort-Object))){throw 'Uninventoried or missing mirrored audit file'}
$null=Require-Pin (Join-Path $hostMirror 'source.zip') (File-Pin (Join-Path $hostRoot 'source.zip'))
$null=Require-Pin (Join-Path $hostMirror 'source-stage.json') (File-Pin (Join-Path $hostRoot 'source-stage.json'))
$null=Require-Pin (Join-Path $hostMirror 'source/audit-card.json') (File-Pin $cardPath)
$null=Require-Pin (Join-Path $hostMirror 'source/workflow-review.json') $workflowGuard.review_pin
$null=Require-Pin (Join-Path $hostMirror 'source/workflow-provenance.json') $workflowGuard.provenance_pin
$null=Require-Pin (Join-Path $hostMirror 'source/scientific-source-review.json') $card.independent_source_review
$manifestPath=Join-Path $hostMirror 'source/SOURCE_MANIFEST.json'
if((File-Pin $manifestPath).sha256-ne$stage.source_manifest_sha256-or(File-Pin (Join-Path $hostRoot 'source-manifest.json')).sha256-ne$stage.source_manifest_sha256-or(File-Pin (Join-Path $hostMirror 'source.zip')).sha256-ne$stage.source_zip_sha256){throw 'Raw source manifest/ZIP must bind passed Stage'}
$manifest=Get-Content -Raw -LiteralPath $manifestPath|ConvertFrom-Json
if($manifest.audit_card_sha256-ne$AuditCardSHA256-or$manifest.container_root-ne$containerRoot-or$manifest.workflow_source_review_sha256-cne$WorkflowSourceReviewSHA256-or$manifest.scientific_source_review_sha256-cne$card.independent_source_review.sha256){throw 'Exact card/source/review manifest context required'}
if($manifest.files.Count-ne$card.staging_transport.payload_count-or(Compare-Object @($manifest.files.file|Sort-Object) @($card.staging_transport.payload_files|Sort-Object))){throw 'Exactly the card-derived payload roles required'}
$executionPath=Join-Path $hostMirror 'run-001/execution.json'
if((File-Pin $executionPath).sha256-ne$ExecutionSHA256){throw 'External execution hash before parse required'}
$execution=Get-Content -Raw -LiteralPath $executionPath|ConvertFrom-Json
if($execution.kind-ne'causal_fresh_v12_quality_audit_execution_v1'-or$execution.status-ne$inventory.status-or$execution.audit_card.sha256-ne$AuditCardSHA256-or$execution.source_stage.sha256-ne$StageReceiptSHA256-or$execution.source_manifest.sha256-ne$stage.source_manifest_sha256){throw 'Execution/inventory/card/Stage disposition mismatch'}
if($execution.independent_workflow_source_review.sha256-cne$WorkflowSourceReviewSHA256-or$execution.independent_scientific_source_review.sha256-cne$card.independent_source_review.sha256){throw 'Execution must bind both externally held independent reviews'}
$sourceEquality=$true
if($execution.source_before.Count-ne$manifest.files.Count-or$execution.source_after.Count-ne$manifest.files.Count){$sourceEquality=$false}
foreach($entry in $manifest.files){
 $before=@($execution.source_before|Where-Object { $_.path -eq ($containerRoot+'/source/'+$entry.file) })
 $after=@($execution.source_after|Where-Object { $_.path -eq ($containerRoot+'/source/'+$entry.file) })
 $current=File-Pin (Join-Path $hostMirror ('source/'+$entry.file))
 if($before.Count-ne1-or$after.Count-ne1-or$before[0].bytes-ne$entry.bytes-or$after[0].bytes-ne$entry.bytes-or$before[0].sha256-ne$entry.sha256-or$after[0].sha256-ne$entry.sha256-or$current.bytes-ne$entry.bytes-or$current.sha256-ne$entry.sha256){$sourceEquality=$false}
}
if($execution.status-eq'passed'-and-not$sourceEquality){throw 'Passed durable audit requires actual source-before/after/manifest equality'}
$mainInputEquality=$true
if($execution.main_inputs_before.Count-ne4-or$execution.main_inputs_after.Count-ne4){$mainInputEquality=$false}
foreach($before in $execution.main_inputs_before){
 $after=@($execution.main_inputs_after|Where-Object{$_.path-eq$before.path})
 if($after.Count-ne1-or$after[0].bytes-ne$before.bytes-or$after[0].sha256-ne$before.sha256){$mainInputEquality=$false}
}
if($execution.status-eq'passed'-and-not$mainInputEquality){throw 'Passed durable audit requires main manifest/ZIP/completed quality stage/receipt before-after equality'}
$rawPath=Join-Path $hostMirror 'run-001/raw-result.json'
if(Test-Path -LiteralPath $rawPath){$null=Require-Pin $rawPath $execution.raw_result}
if($execution.status-eq'passed'){
 $raw=Get-Content -Raw -LiteralPath $rawPath|ConvertFrom-Json
 $normal=$raw.normal_quality_execution;$pairing=$raw.paired_initial_equality
 if($raw.kind-cne'causal_fresh_v12_quality_independent_audit_v1'-or$raw.temporal_pooling_policy-cne$card.temporal_pooling_policy-or$raw.temporal_moment_policy-cne$card.temporal_moment_policy-or$raw.clean_baseline_comparison.baseline-cne'CPB-v3.market-01'-or$raw.clean_baseline_comparison.scoped_reference-cne'CPB-v11.market-01'-or$raw.clean_baseline_comparison.numeric_total-ne22){throw 'Passed current V12 dual-support policy/sole baseline/scopedV11/22-condition identity required'}
 if($raw.status-ne'passed'-or$raw.model-ne'CPB-v12.market-01'-or$raw.current_quality_reporting_recovery_required-ne$false-or$normal.status-ne'passed'-or$normal.current_reporting_recovery_required-ne$false-or$normal.outer_stage.sha256-ne$card.completed_quality_stage.sha256-or$normal.quality_receipt.sha256-ne$card.completed_quality_receipt.sha256-or$normal.scientific_stages-ne5-or$normal.aggregation_stages-ne2-or$normal.stages.Count-ne7-or$pairing.reference_data_archives-ne13-or($pairing.feature_coordinate_slice-join',')-ne'20,40'-or$pairing.native_coordinates_compared-ne20-or$pairing.content0_20_equality_required-ne$false-or$pairing.whole_native40_equality_claimed-ne$false-or-not$pairing.all_declared_numeric_tensor_values_exact){throw 'Passed durable audit must bind normal completed5+2quality stages and partial20:40 thirteen-reference pairing'}
}
$docRoot=Join-Path $researchRoot 'doc/causal_fresh_v12'
$canonical='quality-independent-audit.json'
if($execution.status-ne'passed'){$canonical=$canonical.Replace('.json',".failed-attempt$Attempt.json")}
$durable=$null
if(Test-Path -LiteralPath $rawPath){$durablePath=Join-Path $docRoot $canonical;if(Test-Path -LiteralPath $durablePath){throw 'Preserve prior durable audit'};[IO.File]::Copy($rawPath,$durablePath,$false);$durable=Require-Pin $durablePath (File-Pin $rawPath)}
$executionCopy=Join-Path $docRoot "$Kind-audit-execution-attempt$Attempt.json"
if(Test-Path -LiteralPath $executionCopy){throw 'Preserve prior execution receipt'}
[IO.File]::Copy($executionPath,$executionCopy,$false)
$executionPin=Require-Pin $executionCopy (File-Pin $executionPath)
$receipt=[ordered]@{kind="cpb_v12_${Kind}_independent_audit_mirror_v1";status='passed';audit_status=$execution.status;container_root=$containerRoot;host_root=$hostMirror;files=$actualFiles.Count;exact_file_inventory=$true;mapped_files=$mapped;container_inventory=(File-Pin $inventoryPath);execution=$executionPin;durable_raw_audit=$durable;mirror_helper=(File-Pin $PSCommandPath);source_card=(File-Pin $cardPath);source_stage=(File-Pin $hostStagePath);source_before_after_manifest_equal=$sourceEquality;main_inputs_before_after_equal=$mainInputEquality;model_calls=0;head_fits=0;training_updates=0;interpretation='Pure byte copy/hash mapping; raw audit and every source/log/stage remain exact; no arithmetic replay'}
$receiptPath=Join-Path $docRoot "$Kind-audit-mirror-attempt$Attempt.json"
if(Test-Path -LiteralPath $receiptPath){throw 'Preserve prior audit mirror receipt'}
[IO.File]::WriteAllText($receiptPath,($receipt|ConvertTo-Json -Depth 12)+"`n",[Text.UTF8Encoding]::new($false))
$summary=[ordered]@{mirror='passed';audit=$execution.status;files=$actualFiles.Count;durable=$durable;execution=$executionPin;receipt=(File-Pin $receiptPath)}
$summary|ConvertTo-Json -Depth 5

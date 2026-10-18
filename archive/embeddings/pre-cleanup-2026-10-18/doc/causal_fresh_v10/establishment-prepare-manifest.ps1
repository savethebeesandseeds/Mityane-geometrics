# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# New owned host metadata generator for the physically copied final V9 join-card schema.
# File/hash/JSON inspection only. No Docker, AST, project body or tensor access.
param(
 [Parameter(Mandatory=$true)][string]$InputPlan,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InputPlanSHA256,
 [Parameter(Mandatory=$true)][switch]$RootReviewedCompletedInputs
)
$ErrorActionPreference='Stop'
if(-not $RootReviewedCompletedInputs){throw 'Actual completed quality/audit/Complete inputs require root review'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$destination='doc/causal_fresh_v10/establishment-input-manifest.json'
function Bounded([string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains([char]92)-or$relative-match'(^|/)\.\.(/|$)'){throw 'Forward-slash research-relative file path required'}
 $path=[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))
 if(-not $path.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Path escaped research root'}
 $path
}
function File-Pin([string]$relative){
 $path=Bounded $relative;$file=Get-Item -LiteralPath $path
 if($file.PSIsContainer-or($file.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Physical metadata file required'}
 [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Verify-Pin($expected){
 if($null-eq$expected.path-or$null-eq$expected.bytes-or$expected.sha256-notmatch'^[0-9a-f]{64}$'){throw 'Explicit path/bytes/SHA256 binding required'}
 $actual=File-Pin $expected.path
 if($actual.bytes-ne$expected.bytes-or$actual.sha256-ne$expected.sha256){throw ('Reviewed metadata changed: '+$expected.path)}
 $actual
}
function Read-Pinned($binding){$null=Verify-Pin $binding;Get-Content -Raw -LiteralPath (Bounded $binding.path)|ConvertFrom-Json}
if(Test-Path -LiteralPath (Bounded $destination)){throw 'Preserve the existing join card; additive attempts require a separate declaration'}
$planPin=File-Pin $InputPlan
if($planPin.sha256-ne$InputPlanSHA256){throw 'External input-plan hash required before JSON parse'}
$plan=Get-Content -Raw -LiteralPath (Bounded $InputPlan)|ConvertFrom-Json
if($plan.kind-ne'causal_fresh_v10_establishment_input_plan_v1'-or$plan.status-ne'root_reviewed_completed_quality_audits_and_mirror'-or$plan.authorized-isnot[bool]-or$plan.authorized-ne$true){throw 'Explicit actual completed-input plan admission required'}
$own=File-Pin ([IO.Path]::GetRelativePath($researchRoot,$PSCommandPath).Replace([char]92,[char]47))
if($plan.generator.path-ne$own.path-or$plan.generator.bytes-ne$own.bytes-or$plan.generator.sha256-ne$own.sha256){throw 'Exact root-reviewed generator source required'}
$roles=@('comparison','table_rows','quality_audit','training_audit','complete_mirror','card','quality_receipt','pairings','source_manifest','build_admission','temporal_pooling_proof','temporal_learning','prospective_freeze','protocol_preflight')
$ids=@($plan.inputs.id)
if(@($ids|Sort-Object -Unique).Count-ne$ids.Count-or$ids.Count-lt14-or$ids.Count-gt15-or@($roles|Where-Object {$_-notin$ids}).Count-or@($ids|Where-Object {$_-notin($roles+@('declaration_review'))}).Count){throw 'Exactly14 required JSON roles; only optional declaration_review'}
$requiredSources=@(
 'code/research/causal_fresh_v10_establishment/establish.py',
 'code/research/causal_fresh_v10_establishment/README.md',
 'code/research/causal_fresh_v10_establishment/provenance.json',
 'doc/causal_fresh_v10/establishment-stage.ps1',
 'doc/causal_fresh_v10/establishment-execute.py',
 'doc/causal_fresh_v10/establishment-mirror.ps1')
$reviewedSources=@($plan.reviewed_sources)
if($reviewedSources.Count-ne6-or@($reviewedSources.path|Sort-Object -Unique).Count-ne6-or(Compare-Object @($reviewedSources.path|Sort-Object) @($requiredSources|Sort-Object))){throw 'Exactly six independently reviewed join/support/transport source pins required'}
$sources=@{}
foreach($pin in $reviewedSources){$sources[$pin.path]=Verify-Pin $pin}
$inputs=@();$data=@{};$bindings=@{}
foreach($row in $plan.inputs){
 if(-not $row.path.EndsWith('.json')){throw 'Metadata inputs must be explicit JSON files'}
 $binding=Verify-Pin $row
 $data[$row.id]=Read-Pinned $binding;$bindings[$row.id]=$binding
 $inputs +=[ordered]@{id=$row.id;path=$binding.path;bytes=$binding.bytes;sha256=$binding.sha256;staged_path=('inputs/'+$row.id+'.json')}
}
if($bindings.pairings.path-ne$bindings.quality_audit.path-or$bindings.pairings.bytes-ne$bindings.quality_audit.bytes-or$bindings.pairings.sha256-ne$bindings.quality_audit.sha256){throw 'Pairings must be an exact alias of the complete quality audit receipt'}
$archive=File-Pin 'output/engineering/causal-fresh-v10/attempt-001/source.zip'
$identity=[ordered]@{card_sha256=$bindings.card.sha256;manifest_sha256=$bindings.source_manifest.sha256;archive_sha256=$archive.sha256;source_files=$data.source_manifest.files.Count;binary_count=$data.card.source_counts.binaries;translation_units=$data.card.source_counts.translation_units}
if($data.card.instance-ne'CPB-v10.market-01'-or$data.card.stage.Contains('DRAFT')-or-not$data.card.frozen_utc-or-not$data.card.execution_admitted-or$data.source_manifest.files.Count-le0-or$data.card.source_counts.capsule_sources-ne$identity.source_files-or$data.source_manifest.card_sha256-ne$identity.card_sha256){throw 'Actual frozen scientific declaration and dynamic exact source capsule required'}
if($data.quality_audit.kind-ne'causal_fresh_v10_quality_independent_audit_v1'-or$data.quality_audit.status-ne'passed'-or$data.training_audit.kind-ne'causal_fresh_v10_training_independent_audit_v1'-or$data.training_audit.status-ne'passed'){throw 'Both independently completed audits must pass before join-card declaration'}
if($data.complete_mirror.kind-ne'causal_fresh_v10_container_to_local_evidence_mirror_v1'-or$data.complete_mirror.status-ne'passed'-or$data.complete_mirror.scope-ne'Complete'-or$data.complete_mirror.attempt-ne'attempt-001'-or$data.complete_mirror.source_files_verified-ne$identity.source_files-or$data.complete_mirror.file_count-ne$data.complete_mirror.files.Count-or-not$data.complete_mirror.all_mirrored_pt_ignored){throw 'Actual exact normal Complete mirror required; counts derived from the receipt'}
if($data.complete_mirror.source_stage.status-ne'passed'-or$data.complete_mirror.source_stage.card_sha256-ne$identity.card_sha256-or$data.complete_mirror.source_stage.manifest_sha256-ne$identity.manifest_sha256-or$data.complete_mirror.source_stage.archive_sha256-ne$identity.archive_sha256){throw 'Complete mirror must bind the same immutable source transport'}
if($data.quality_receipt.status-ne'passed'-or$data.quality_receipt.card_sha256-ne$identity.card_sha256-or$data.build_admission.status-ne'passed'-or$data.build_admission.source_files-ne$identity.source_files-or$data.build_admission.binaries.Count-ne7-or$identity.binary_count-ne7-or$identity.translation_units-ne11-or@($data.build_admission.dependency_closures.PSObject.Properties).Count-ne11){throw 'Completed normal quality and actual seven-binary/eleven-TU build required'}
if($data.prospective_freeze.kind-ne'causal_fresh_v10_prospective_freeze_v1'-or$data.prospective_freeze.status-ne'passed'-or$data.prospective_freeze.card.sha256-ne$identity.card_sha256-or$data.prospective_freeze.declared_capsule_sources-ne$identity.source_files-or$data.protocol_preflight.status-ne'passed'-or$data.protocol_preflight.card_sha256-ne$identity.card_sha256-or($data.protocol_preflight.sources.Count+1)-ne$identity.source_files){throw 'Actual normal prospective freeze/preflight bindings required'}
$objective='huber1-mean-plus-joint-gaussian-nll-v1';$gradient='huber1-mean-live;joint-gaussian-mean-live-and-uncertainty4-live;shared-global-clip1-v1'
foreach($record in @($data.card.training,$data.table_rows,$data.comparison,$data.training_audit,$data.quality_audit)){if($record.objective_policy-ne$objective-or$record.gradient_policy-ne$gradient-or$record.variance_error_detached-ne$false-or$record.reporting_nll_backpropagated-ne$false){throw 'Actual one-joint-Gaussian objective receipt contract required'}}
if($data.card.advancement.goal_protocol.sha256-ne'221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144'){throw 'Exact current prospective22 protocol required'}
foreach($relative in @('training-stage.json','initial-saved-proof-stage.json','final-saved-proof-stage.json','quality-stage.json','quality-001/extraction-stage.json','quality-001/initial_market_heads-stage.json','quality-001/initial_synthetic_heads-stage.json','quality-001/final_market_heads-stage.json','quality-001/final_synthetic_heads-stage.json','quality-001/report-stage.json','quality-001/comparison-stage.json')){
 $containerPath='/opt/cuwacunu/research/causal-fresh-v10/attempt-001/'+$relative
 $matches=@($data.complete_mirror.files|Where-Object{$_.container_path-eq$containerPath});if($matches.Count-ne1){throw 'Every normal completed stage must belong to Complete mirror'}
 $stage=Read-Pinned $matches[0];if($stage.exit_code-ne0){throw 'Normal V10 quality/proofs must pass without an anticipated recovery'}
}
# These checks admit only a prospective JSON/hash join, never establish candidate
# trust or numerical advancement on the host. The separately reviewed join body
# validates the complete evidence graph inside the existing managed container.
$record=[ordered]@{
 kind='causal_fresh_v10_establishment_join_card_v1';stage='declared_after_completed_audits_before_metadata_join';declared_utc=[DateTime]::UtcNow.ToString('o');root_reviewed=$true;
 independent_actual_input_review_required_before_execution=$true;
 source=$sources['code/research/causal_fresh_v10_establishment/establish.py'];
 source_support=@($sources['code/research/causal_fresh_v10_establishment/README.md'],$sources['code/research/causal_fresh_v10_establishment/provenance.json']);
 execution_wrapper=$sources['doc/causal_fresh_v10/establishment-execute.py'];stage_helper=$sources['doc/causal_fresh_v10/establishment-stage.ps1'];mirror_helper=$sources['doc/causal_fresh_v10/establishment-mirror.ps1'];
 main_source_identity=$identity;inputs=$inputs;metadata_generator=$own;actual_input_plan=$planPin;
 source_static_review='Independent reviewed owned final V9-derived join and strengthened native-Docker transports; actual input-card review and root Stage/body admission remain separate';
 model_calls=0;head_fits=0;training_updates=0;scientific_metric_recomputations=0;Docker_calls=0;AST_calls=0;project_body_calls=0;
 raw_producer_pending_fields_must_remain_exact=$true;baseline='CPB-v3.market-01';application_changed=$false;goal_active=$true}
foreach($pin in @($reviewedSources)+@($plan.inputs)){$null=Verify-Pin $pin}
if((File-Pin $InputPlan).sha256-ne$InputPlanSHA256){throw 'Input plan changed during metadata generation'}
$outputBytes=[Text.UTF8Encoding]::new($false).GetBytes((($record|ConvertTo-Json -Depth 25).Replace("`r`n","`n")+"`n"))
$stream=[IO.File]::Open((Bounded $destination),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$stream.Write($outputBytes,0,$outputBytes.Length)}finally{$stream.Dispose()}
File-Pin $destination|ConvertTo-Json

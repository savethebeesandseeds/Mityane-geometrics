# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host file inspection only. No aggregate reduction, Docker, model or head calls.
param(
 [Parameter(Mandatory=$true)][switch]$RootSuppliedFinalMetadata,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PostV11LedgerSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$FinalPreservationSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootSuppliedFinalMetadata){throw 'Root must supply the post-V11 ledger/final109-preservation pins first'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$documentRoot=Join-Path $researchRoot 'doc/causal_v11_retention_review'
$destination=Join-Path $documentRoot 'derivation-manifest.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve the previous prospective manifest'}
function Pin([string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains([char]92)-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded repository-relative path required'}
 $path=[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))
 if(-not$path.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Path escaped repository'}
 $item=Get-Item -LiteralPath $path
 if($item.PSIsContainer-or($item.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Regular nonlinked metadata file required'}
 [ordered]@{path=$relative;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$sourceRoot='code/research/causal_v11_retention_review/'
$expected=[ordered]@{'derive.py'='7abf2c5bbd8911887173cf843335bfa1afb60e13e65a7b8d934e8f45350f20d4';'independent_review.py'='82836be7c2b00ea3bc4f4ef9640c1208acd6073c317a245214031aba115b297f';'README.md'='b0683606b576e1dfb170e485b3a1faed8795696950ad0417c981d05cc65fd3e2';'provenance.json'='fc3c6691487c377e64b99fde8d634ccd6adf023737a6eab74883ecef659cf01f'}
$sources=@()
foreach($name in $expected.Keys){$p=Pin ($sourceRoot+$name);if($p.sha256-ne$expected[$name]){throw 'Root-reviewed owned source changed'};$sources+=$p}
$paths=[ordered]@{parent='doc/clean_causal_encoder_search_goal_v9_retention.json';v3='doc/results/causal_fresh_v3/table_rows.json';v4='doc/results/causal_fresh_v4/table_rows.json';v5='doc/results/causal_fresh_v5/table_rows.json';v6='doc/results/causal_fresh_v6/table_rows.json';v7='doc/results/causal_fresh_v7/table_rows.json';v8='doc/results/causal_fresh_v8/table_rows.json';v9='doc/results/causal_fresh_v9/table_rows.json';v10='doc/results/causal_fresh_v10/table_rows.json';v11='doc/results/causal_fresh_v11/table_rows.json';ledger='doc/clean_causal_encoder_research_ledger.json';establishment='doc/causal_fresh_v11/final-establishment.json';complete_mirror='doc/causal_fresh_v11/complete-mirror-attempt-001.json';preservation='doc/causal_fresh_v11/final-preservation.json'}
$inputs=@()
foreach($id in $paths.Keys){$p=Pin $paths[$id];$inputs +=[ordered]@{id=$id;path=$p.path;staged_path="inputs/$id.json";bytes=$p.bytes;sha256=$p.sha256}}
if($inputs.Count-ne14-or$inputs[0].sha256-ne'221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144'){throw 'Fourteen roles and exact V9-retention parent required'}
if(@($inputs|Where-Object{$_.id-eq'ledger'})[0].sha256-ne$PostV11LedgerSha256-or@($inputs|Where-Object{$_.id-eq'preservation'})[0].sha256-ne$FinalPreservationSha256){throw 'Exact root-supplied post-V11 ledger/final preservation hashes required'}
$preservation=Get-Content -LiteralPath (Join-Path $researchRoot $paths.preservation) -Raw|ConvertFrom-Json
$establishment=Get-Content -LiteralPath (Join-Path $researchRoot $paths.establishment) -Raw|ConvertFrom-Json
$mirror=Get-Content -LiteralPath (Join-Path $researchRoot $paths.complete_mirror) -Raw|ConvertFrom-Json
if($preservation.kind-ne'causal_fresh_v11_final_preservation_v1'-or$preservation.status-ne'passed'-or$preservation.critical_checkpoint_total-ne109-or$establishment.kind-ne'causal_fresh_v11_postmeasurement_establishment_v1'-or$establishment.status-ne'passed'-or$establishment.candidate-ne'CPB-v11.market-01'-or$establishment.trust_admission_passed-isnot[bool]-or-not$establishment.trust_admission_passed-or$mirror.kind-ne'causal_fresh_v11_container_to_local_evidence_mirror_v1'-or$mirror.status-ne'passed'-or$mirror.scope-ne'Complete'-or$mirror.source_files_verified-ne132){throw 'Completed V11 preservation, separate trust and Complete mirror required'}
foreach($binding in @(@('current_ledger','ledger'),@('final_establishment','establishment'),@('complete_V11_mirror','complete_mirror'))){$row=@($inputs|Where-Object{$_.id-eq$binding[1]})[0];if($preservation.($binding[0]).sha256-ne$row.sha256-or$preservation.($binding[0]).bytes-ne$row.bytes){throw 'Final preservation must bind this exact post-V11 input set'}}
if($establishment.normal_prospective_source_admission.source_files-ne132-or$establishment.normal_prospective_source_admission.normal_quality_all_stages_passed-ne$true-or$establishment.normal_prospective_source_admission.scientific_model_head_or_training_replay-ne$false-or$preservation.current_reporting_recovery_required-ne$false-or$preservation.historical_V9_reporting_recovery_preserved-ne$true-or$preservation.science_stages_replayed-ne0){throw 'Normal V11 quality and preserved historical V9 recovery without replay required'}
$helpers=@()
foreach($name in @('prepare-manifest.ps1','stage-retention.ps1','execute-retention.py','mirror-retention.ps1')){$helpers+=Pin ('doc/causal_v11_retention_review/'+$name)}
$manifest=[ordered]@{copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.';kind='clean_causal_v11_retention_derivation_manifest_v1';state='prospective_source_and_pins_frozen_awaiting_root_review';client_date='2026-10-17';authorized_for_derivation=$false;container_id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';container_scope='/opt/cuwacunu/research/causal-v11-retention-review';source=$sources[0];independent_reviewer=$sources[1];owned_sources=$sources;execution_wrapper=$helpers[2];orchestration_helpers=$helpers;inputs=$inputs;prospective_contract=[ordered]@{criterion_count=22;quality_maximum_final_scopes=16;minimum_final_MAE_scopes=2;strict_minimum_final_VALIDATION_NLL_scopes=1;unchanged_strict_positivity_rules=3;expected_strengthened_floors='UNKNOWN;derive from actual saved aggregates';expected_strengthened_IDs=@();expected_unchanged_criteria='22 minus actual strengthened existing criteria';tie_policy='Chronological V3,V4,V5,V6,V7,V8,V9,V10,V11 exact ties retain earlier source';rules_tolerance_trust_coverage_preserved=$true;additional_TRAIN_NLL_gate=$false;initial_checkpoint_selection=$false;historical_counts_recomputed=$false;candidate_or_head_settings_declared=$false;baseline='CPB-v3.market-01';application_changed=$false};future_output_paths=@('doc/clean_causal_encoder_search_goal_v11_retention.json','doc/clean_causal_encoder_search_goal_v11_retention.md');execution_policy='Root reviews this exact manifest and admits a separate copy. Derivation once, independent review once, existing-container exclusive leaves only. Generated protocol remains a proposal until separately applied.';model_calls=0;head_fits=0;training_updates=0;numeric_derivations_executed=0;live_files_written=0}
$manifest.manifest_schema=Pin 'doc/causal_v11_retention_review/manifest-schema.json'
$manifest.orchestration_provenance=Pin 'doc/causal_v11_retention_review/orchestration-provenance.json'
$manifest.root_supplied_final_metadata=[ordered]@{post_V11_ledger_sha256=$PostV11LedgerSha256;final109_preservation_sha256=$FinalPreservationSha256}
[IO.Directory]::CreateDirectory($documentRoot)|Out-Null
[IO.File]::WriteAllText($destination,(($manifest|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Pin 'doc/causal_v11_retention_review/derivation-manifest.json'|ConvertTo-Json

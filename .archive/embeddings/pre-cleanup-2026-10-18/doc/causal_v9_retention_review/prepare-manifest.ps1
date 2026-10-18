# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host file inspection only. No aggregate reduction, Docker, model or head calls.
param(
 [Parameter(Mandatory=$true)][switch]$RootSuppliedFinalMetadata,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PostV9LedgerSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$FinalPreservationSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootSuppliedFinalMetadata){throw 'Root must supply the post-V9 ledger/final99-preservation pins first'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$documentRoot=Join-Path $researchRoot 'doc/causal_v9_retention_review'
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
$sourceRoot='code/research/causal_v9_retention_review/'
$expected=[ordered]@{'derive.py'='ca846a9fd709606270aec04b05e3002d59664388650f722d4b71370c4b187e76';'independent_review.py'='f8311524bc6819a7d9f574fb96351d1a00877ea58465c52b1f26975ce3a843d1';'README.md'='8c821a738503705d2cd9399c005719501cc19c8b5a820aa495d29110c2d883f7';'provenance.json'='93a7f6272cd8cade6564660a06725a1059da029176b43d65024a683cf97641cc'}
$sources=@()
foreach($name in $expected.Keys){$p=Pin ($sourceRoot+$name);if($p.sha256-ne$expected[$name]){throw 'Root-reviewed owned source changed'};$sources+=$p}
$paths=[ordered]@{parent='doc/clean_causal_encoder_search_goal_v8_retention.json';v3='doc/results/causal_fresh_v3/table_rows.json';v4='doc/results/causal_fresh_v4/table_rows.json';v5='doc/results/causal_fresh_v5/table_rows.json';v6='doc/results/causal_fresh_v6/table_rows.json';v7='doc/results/causal_fresh_v7/table_rows.json';v8='doc/results/causal_fresh_v8/table_rows.json';v9='doc/results/causal_fresh_v9/table_rows.json';ledger='doc/clean_causal_encoder_research_ledger.json';establishment='doc/causal_fresh_v9/final-establishment.json';complete_mirror='doc/causal_fresh_v9/complete-mirror-attempt-001.json';preservation='doc/causal_fresh_v9/final-preservation.json'}
$inputs=@()
foreach($id in $paths.Keys){$p=Pin $paths[$id];$inputs +=[ordered]@{id=$id;path=$p.path;staged_path="inputs/$id.json";bytes=$p.bytes;sha256=$p.sha256}}
if($inputs.Count-ne12-or$inputs[0].sha256-ne'e7920442fbc5528cbe7e67f71ada65101974a6a28ad4d39bc57dea461c477699'){throw 'Twelve roles and exact V8-retention parent required'}
if(@($inputs|Where-Object{$_.id-eq'ledger'})[0].sha256-ne$PostV9LedgerSha256-or@($inputs|Where-Object{$_.id-eq'preservation'})[0].sha256-ne$FinalPreservationSha256){throw 'Exact root-supplied post-V9 ledger/final preservation hashes required'}
$preservation=Get-Content -LiteralPath (Join-Path $researchRoot $paths.preservation) -Raw|ConvertFrom-Json
$establishment=Get-Content -LiteralPath (Join-Path $researchRoot $paths.establishment) -Raw|ConvertFrom-Json
$mirror=Get-Content -LiteralPath (Join-Path $researchRoot $paths.complete_mirror) -Raw|ConvertFrom-Json
if($preservation.kind-ne'causal_fresh_v9_final_preservation_v1'-or$preservation.status-ne'passed'-or$preservation.critical_checkpoint_total-ne99-or$establishment.kind-ne'causal_fresh_v9_postmeasurement_establishment_v1'-or$establishment.status-ne'passed'-or$establishment.candidate-ne'CPB-v9.market-01'-or$establishment.trust_admission_passed-isnot[bool]-or-not$establishment.trust_admission_passed-or$mirror.kind-ne'causal_fresh_v9_container_to_local_evidence_mirror_v1'-or$mirror.status-ne'passed'-or$mirror.scope-ne'Complete'-or$mirror.source_files_verified-ne118){throw 'Completed V9 preservation, separate trust and Complete mirror required'}
foreach($binding in @(@('current_ledger','ledger'),@('final_establishment','establishment'),@('complete_V9_mirror','complete_mirror'))){$row=@($inputs|Where-Object{$_.id-eq$binding[1]})[0];if($preservation.($binding[0]).sha256-ne$row.sha256-or$preservation.($binding[0]).bytes-ne$row.bytes){throw 'Final preservation must bind this exact post-V9 input set'}}
if($establishment.report_only_recovery.status-ne'passed'-or$mirror.report_only_recovery.status-ne'passed'-or$preservation.reporting_recovery_card.sha256-ne'9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'-or$preservation.reporting_recovery_receipt.sha256-ne'5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72'-or$preservation.science_stages_replayed-ne0){throw 'Separate report-only recovery and original failures must remain preserved'}
$helpers=@()
foreach($name in @('prepare-manifest.ps1','stage-retention.ps1','execute-retention.py','mirror-retention.ps1')){$helpers+=Pin ('doc/causal_v9_retention_review/'+$name)}
$manifest=[ordered]@{copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.';kind='clean_causal_v9_retention_derivation_manifest_v1';state='prospective_source_and_pins_frozen_awaiting_root_review';client_date='2026-10-16';authorized_for_derivation=$false;container_id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';container_scope='/opt/cuwacunu/research/causal-v9-retention-review';source=$sources[0];independent_reviewer=$sources[1];owned_sources=$sources;execution_wrapper=$helpers[2];orchestration_helpers=$helpers;inputs=$inputs;prospective_contract=[ordered]@{criterion_count=22;quality_maximum_final_scopes=16;minimum_final_MAE_scopes=2;strict_minimum_final_VALIDATION_NLL_scopes=1;unchanged_strict_positivity_rules=3;expected_strengthened_floors='UNKNOWN;derive from actual saved aggregates';expected_strengthened_IDs=@();expected_unchanged_criteria='22 minus actual strengthened existing criteria';tie_policy='Chronological V3,V4,V5,V6,V7,V8,V9 exact ties retain earlier source';rules_tolerance_trust_coverage_preserved=$true;additional_TRAIN_NLL_gate=$false;initial_checkpoint_selection=$false;historical_counts_recomputed=$false;candidate_or_head_settings_declared=$false;baseline='CPB-v3.market-01';application_changed=$false};future_output_paths=@('doc/clean_causal_encoder_search_goal_v9_retention.json','doc/clean_causal_encoder_search_goal_v9_retention.md');execution_policy='Root reviews this exact manifest and admits a separate copy. Derivation once, independent review once, existing-container exclusive leaves only. Generated protocol remains a proposal until separately applied.';model_calls=0;head_fits=0;training_updates=0;numeric_derivations_executed=0;live_files_written=0}
$manifest.manifest_schema=Pin 'doc/causal_v9_retention_review/manifest-schema.json'
$manifest.orchestration_provenance=Pin 'doc/causal_v9_retention_review/orchestration-provenance.json'
$manifest.root_supplied_final_metadata=[ordered]@{post_V9_ledger_sha256=$PostV9LedgerSha256;final99_preservation_sha256=$FinalPreservationSha256}
[IO.Directory]::CreateDirectory($documentRoot)|Out-Null
[IO.File]::WriteAllText($destination,(($manifest|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Pin 'doc/causal_v9_retention_review/derivation-manifest.json'|ConvertTo-Json

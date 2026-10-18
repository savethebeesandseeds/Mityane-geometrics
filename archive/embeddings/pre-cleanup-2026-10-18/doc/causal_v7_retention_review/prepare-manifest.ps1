# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host file inspection only. No aggregate reduction, Docker, model or head calls.
param()
$ErrorActionPreference='Stop'
$researchRoot='C:\Work\cuwacunu\cuwacunu_embedding'
$documentRoot=Join-Path $researchRoot 'doc/causal_v7_retention_review'
$destination=Join-Path $documentRoot 'derivation-manifest.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve the previous prospective manifest'}
function Pin([string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains('\')-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded repository-relative path required'}
 $path=[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))
 if(-not$path.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Path escaped repository'}
 $item=Get-Item -LiteralPath $path
 if($item.PSIsContainer-or($item.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Regular nonlinked metadata file required'}
 [ordered]@{path=$relative;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$sourceRoot='code/research/causal_v7_retention_review/'
$expected=[ordered]@{'derive.py'='6dbacf9541d2c111dfd6da4df7288cf2bd7f1af88f386941134852c423340366';'independent_review.py'='cacaeeaf7b019d731410e0f3d0900914765704d2b40c7b099e2a9f22e1228edc';'README.md'='759d1eafd50c86211b4fab92e5577b3c185f6f567a0f51b15ab84b884bfedc12';'provenance.json'='dfc19ca2abbb9212f394555d7f64518c561eb2ca020a98d6f1ac23fc63ecdbc1'}
$sources=@()
foreach($name in $expected.Keys){$p=Pin ($sourceRoot+$name);if($p.sha256-ne$expected[$name]){throw 'Root-reviewed owned source changed'};$sources+=$p}
$paths=[ordered]@{parent='doc/clean_causal_encoder_search_goal_v6_retention.json';v3='doc/results/causal_fresh_v3/table_rows.json';v4='doc/results/causal_fresh_v4/table_rows.json';v5='doc/results/causal_fresh_v5/table_rows.json';v6='doc/results/causal_fresh_v6/table_rows.json';v7='doc/results/causal_fresh_v7/table_rows.json';ledger='doc/clean_causal_encoder_research_ledger.json';establishment='doc/causal_fresh_v7/final-establishment.json';preservation='doc/causal_fresh_v7/final-preservation.json'}
$inputs=@()
foreach($id in $paths.Keys){$p=Pin $paths[$id];$inputs +=[ordered]@{id=$id;path=$p.path;staged_path="inputs/$id.json";bytes=$p.bytes;sha256=$p.sha256}}
if($inputs[0].sha256-ne'0965842331056b61faee827a5ea782703205b23bd326298929ed3ecfca4604fb'){throw 'Exact V6-retention parent required'}
$preservation=Get-Content -LiteralPath (Join-Path $researchRoot $paths.preservation) -Raw|ConvertFrom-Json
$establishment=Get-Content -LiteralPath (Join-Path $researchRoot $paths.establishment) -Raw|ConvertFrom-Json
if($preservation.status-ne'passed'-or$establishment.status-ne'passed'-or$establishment.candidate-ne'CPB-v7.market-01'-or$establishment.trust_admission_passed-isnot[bool]-or-not$establishment.trust_admission_passed){throw 'Completed V7 preservation and measured-candidate trust required'}
$helpers=@()
foreach($name in @('prepare-manifest.ps1','stage-retention.ps1','execute-retention.py','mirror-retention.ps1')){$helpers+=Pin ('doc/causal_v7_retention_review/'+$name)}
$manifest=[ordered]@{copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.';kind='clean_causal_v7_retention_derivation_manifest_v1';state='prospective_source_and_pins_frozen_awaiting_root_review';client_date='2026-10-16';authorized_for_derivation=$false;container_id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';container_scope='/opt/cuwacunu/research/causal-v7-retention-review';source=$sources[0];independent_reviewer=$sources[1];owned_sources=$sources;execution_wrapper=$helpers[2];orchestration_helpers=$helpers;inputs=$inputs;prospective_contract=[ordered]@{criterion_count=22;quality_maximum_final_scopes=16;minimum_final_MAE_scopes=2;strict_minimum_final_VALIDATION_NLL_scopes=1;unchanged_strict_positivity_rules=3;expected_strengthened_floors=2;expected_strengthened_IDs=@('past_volatility:neural_mean_r2','train:standardized_MAE');expected_unchanged_criteria=20;tie_policy='Chronological V3,V4,V5,V6,V7 exact ties retain earlier source';rules_tolerance_trust_coverage_preserved=$true;additional_TRAIN_NLL_gate=$false;initial_checkpoint_selection=$false;historical_counts_recomputed=$false;candidate_or_head_settings_declared=$false;baseline='CPB-v3.market-01';application_changed=$false};future_output_paths=@('doc/clean_causal_encoder_search_goal_v7_retention.json','doc/clean_causal_encoder_search_goal_v7_retention.md');execution_policy='Root reviews this exact manifest and admits a separate copy. Derivation once, independent review once, existing-container exclusive leaves only. Generated protocol remains a proposal until separately applied.';model_calls=0;head_fits=0;training_updates=0;numeric_derivations_executed=0;live_files_written=0}
[IO.Directory]::CreateDirectory($documentRoot)|Out-Null
[IO.File]::WriteAllText($destination,(($manifest|ConvertTo-Json -Depth 14).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Pin 'doc/causal_v7_retention_review/derivation-manifest.json'|ConvertTo-Json

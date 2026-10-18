# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host metadata pins only. Run after all final human reports and establishment exist.
param([Parameter(Mandatory=$true)][switch]$RootApprovedCompletedInputs,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PriorSnapshotSha256,
 [Parameter(Mandatory=$true)][string]$NextDirectionAction,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9]{4}-[0-9]{2}-[0-9]{2}$')][string]$ClientDate)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not$RootApprovedCompletedInputs){throw 'Root must approve completed human reports/establishment before plan generation'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$destination=Join-Path $PSScriptRoot 'input-plan.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve every prior input plan'}
$roles=@'
{
  "ledger": "doc/clean_causal_encoder_research_ledger.json",
  "catalog": "code/encoders/causal_patch_bottleneck_mae/versions/current.json",
  "prior_snapshot": "output/engineering/causal-fresh-v13-finalization/prior-live-001/manifest.json",
  "descriptor": "code/encoders/causal_patch_bottleneck_mae/versions/CPB-v13.json",
  "experiment_card": "doc/CAUSAL_FRESH_V13_CARD.json",
  "source_manifest": "output/engineering/causal-fresh-v13/attempt-001/source-manifest.json",
  "build": "output/engineering/causal-fresh-v13/attempt-001/build-admission.json",
  "training": "output/runs/causal-fresh-v13/run-001/training-summary.json",
  "resume": "output/runs/causal-fresh-v13/run-001/typed-resume-512.json",
  "initialization": "output/runs/causal-fresh-v13/run-001/fresh-initialization.json",
  "calibration": "output/runs/causal-fresh-v13/run-001/calibration/summary.json",
  "engineering": "output/engineering/causal-fresh-v13/attempt-001/engineering-proof/proof.json",
  "gradient": "output/engineering/causal-fresh-v13/attempt-001/variance-gradient-proof.json",
  "initial_proof": "output/engineering/causal-fresh-v13/attempt-001/initial-proof/proof.json",
  "final_proof": "output/engineering/causal-fresh-v13/attempt-001/final-proof/proof.json",
  "training_audit": "doc/causal_fresh_v13/independent-training-audit.json",
  "quality_audit": "doc/causal_fresh_v13/quality-independent-audit.json",
  "pairings": "doc/causal_fresh_v13/quality-independent-audit.json",
  "establishment": "doc/causal_fresh_v13/final-establishment.json",
  "mirror": "doc/causal_fresh_v13/complete-mirror-attempt-001.json",
  "quality_receipt": "output/runs/causal-fresh-v13/run-001/quality-001/quality-receipt.json",
  "comparison": "doc/results/causal_fresh_v13/comparison.json",
  "v3_table": "doc/results/causal_fresh_v3/table_rows.json",
  "v4_table": "doc/results/causal_fresh_v4/table_rows.json",
  "v5_table": "doc/results/causal_fresh_v5/table_rows.json",
  "v6_table": "doc/results/causal_fresh_v6/table_rows.json",
  "v7_table": "doc/results/causal_fresh_v7/table_rows.json",
  "v8_table": "doc/results/causal_fresh_v8/table_rows.json",
  "v9_table": "doc/results/causal_fresh_v9/table_rows.json",
  "v10_table": "doc/results/causal_fresh_v10/table_rows.json",
  "v11_table": "doc/results/causal_fresh_v11/table_rows.json",
  "v12_table": "doc/results/causal_fresh_v12/table_rows.json",
  "v13_table": "doc/results/causal_fresh_v13/table_rows.json",
  "v6_establishment": "doc/causal_fresh_v6/final-establishment.json",
  "v7_establishment": "doc/causal_fresh_v7/final-establishment.json",
  "v8_establishment": "doc/causal_fresh_v8/final-establishment.json",
  "v9_establishment": "doc/causal_fresh_v9/final-establishment.json",
  "v10_establishment": "doc/causal_fresh_v10/final-establishment.json",
  "v11_establishment": "doc/causal_fresh_v11/final-establishment.json",
  "v12_establishment": "doc/causal_fresh_v12/final-establishment.json",
  "temporal_proof": "doc/results/causal_fresh_v13/temporal-pooling-proof.json",
  "temporal_learning": "doc/results/causal_fresh_v13/temporal-projection-learning.json",
  "prospective_freeze": "doc/causal_fresh_v13/prospective-freeze.json",
  "protocol_preflight": "doc/causal_fresh_v13/protocol-preflight.json",
  "results": "doc/CAUSAL_FRESH_V13_RESULTS.md",
  "review": "doc/CAUSAL_FRESH_V13_REVIEW.md",
  "detail": "doc/CAUSAL_FRESH_V13_DETAIL.md"
}
'@|ConvertFrom-Json -AsHashtable
if(@($roles.Keys|Sort-Object -Unique).Count-ne$roles.Count){throw 'Unique exact source-declared metadata roles required'}
$snapshotPath=Join-Path $researchRoot $roles['prior_snapshot']
if((Get-FileHash -LiteralPath $snapshotPath -Algorithm SHA256).Hash.ToLowerInvariant()-ne$PriorSnapshotSha256){throw 'Exact externally reviewed outside-attempt prior six-file snapshot required'}
if([string]::IsNullOrWhiteSpace($NextDirectionAction)){throw 'Root must supply an explicit proposal-only next direction after measurement'}
$inputs=@()
foreach($id in $roles.Keys|Sort-Object){
 $label=$roles[$id]
 $path=Join-Path $researchRoot $label
 $item=Get-Item -LiteralPath $path
 if($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw ('Expected unlinked metadata file: '+$label)}
 $format=if($id -in @('results','review','detail')){'text'}else{'json'}
 if($format -eq 'json'){$null=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
 $extension=if($format -eq 'text'){'.txt'}else{'.json'}
 $inputs += [ordered]@{id=$id;source_path=('/opt/cuwacunu/research/causal-fresh-v13-finalization/attempt-001/source/inputs/'+$id+$extension);path=$label;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant();format=$format}
}
$assertions=[ordered]@{}
$assertions.pairing_status=[ordered]@{input='pairings';path=@('status');equals='passed'}
$assertions.pairing_count=[ordered]@{input='pairings';path=@('paired_initial_equality','reference_data_archives');equals=13}
$assertions.training_audit_status=[ordered]@{input='training_audit';path=@('status');equals='passed'}
$assertions.quality_audit_status=[ordered]@{input='quality_audit';path=@('status');equals='passed'}
foreach($entry in @(@('establishment_status','status','passed'),@('establishment_runtime','runtime_trust_admission_passed',$true),@('establishment_trust','trust_admission_passed',$true),@('establishment_coverage','full_coverage',$true),@('establishment_measured','measured_candidate_established',$true))){
 $assertions[$entry[0]]=[ordered]@{input='establishment';path=@($entry[1]);equals=$entry[2]}
}
$generator=Join-Path $researchRoot 'code/research/causal_fresh_v13_finalization/finalize.py'
$ownedSources=@(foreach($name in @('finalize.py','prepare_card.py','review_proposals.py','README.md','provenance.json')){
 $label='code/research/causal_fresh_v13_finalization/'+$name;$file=Get-Item -LiteralPath (Join-Path $researchRoot $label)
 if($file.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Physical owned source required'}
 [ordered]@{name=$name;path=$label;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$plan=[ordered]@{
 kind='cpb_v13_metadata_finalization_input_plan_v1';approved_for_metadata_card_generation=$true;client_date=$ClientDate;container_attempt='/opt/cuwacunu/research/causal-fresh-v13/attempt-001'
 generator=[ordered]@{path='code/research/causal_fresh_v13_finalization/finalize.py';bytes=(Get-Item -LiteralPath $generator).Length;sha256=(Get-FileHash -LiteralPath $generator -Algorithm SHA256).Hash.ToLowerInvariant()}
 owned_sources=$ownedSources;prior_snapshot_manifest_sha256=$PriorSnapshotSha256
 inputs=$inputs
 evidence_contract=[ordered]@{assertions=$assertions;training_audit_checks_path=@('checks');quality_audit_checks_path=@('checks');establishment_bindings_path=@('evidence');establishment_binding_keys=[ordered]@{experiment_card='card';comparison='comparison';v13_table='table_rows';pairings='pairings';training_audit='training_audit';quality_audit='quality_audit';source_manifest='source_manifest';build='build_admission';mirror='complete_mirror';quality_receipt='quality_receipt';temporal_proof='temporal_pooling_proof';temporal_learning='temporal_learning';prospective_freeze='prospective_freeze';protocol_preflight='protocol_preflight'}}
 next_direction=[ordered]@{status='proposal_only_after_completed_measurements_no_scientific_execution_admitted';action=$NextDirectionAction}
}
[IO.File]::WriteAllText($destination,($plan|ConvertTo-Json -Depth 40)+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='prepared_for_root_review';path='doc/causal_fresh_v13_finalization/input-plan.json';bytes=(Get-Item -LiteralPath $destination).Length;sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant();inputs=$inputs.Count}|ConvertTo-Json

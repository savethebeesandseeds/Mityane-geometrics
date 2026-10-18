# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host metadata pins only. Run after all final human reports and establishment exist.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$destination=Join-Path $PSScriptRoot 'input-plan.json'
if(Test-Path -LiteralPath $destination){throw 'Preserve every prior input plan'}
$roles=@'
{
  "ledger": "doc/clean_causal_encoder_research_ledger.json",
  "catalog": "code/encoders/causal_patch_bottleneck_mae/versions/current.json",
  "prior_snapshot": "output/engineering/causal-fresh-v7-finalization/attempt-001/prior-live/manifest.json",
  "descriptor": "code/encoders/causal_patch_bottleneck_mae/versions/CPB-v7.json",
  "experiment_card": "doc/CAUSAL_FRESH_V7_CARD.json",
  "source_manifest": "output/engineering/causal-fresh-v7/attempt-001/source-manifest.json",
  "build": "output/engineering/causal-fresh-v7/attempt-001/build-admission.json",
  "training": "output/runs/causal-fresh-v7/run-001/training-summary.json",
  "resume": "output/runs/causal-fresh-v7/run-001/typed-resume-512.json",
  "initialization": "output/runs/causal-fresh-v7/run-001/fresh-initialization.json",
  "calibration": "output/runs/causal-fresh-v7/run-001/calibration/summary.json",
  "engineering": "output/engineering/causal-fresh-v7/attempt-001/engineering-proof/proof.json",
  "gradient": "output/engineering/causal-fresh-v7/attempt-001/variance-gradient-proof.json",
  "initial_proof": "output/engineering/causal-fresh-v7/attempt-001/initial-proof/proof.json",
  "final_proof": "output/engineering/causal-fresh-v7/attempt-001/final-proof/proof.json",
  "training_audit": "doc/causal_fresh_v7/independent-training-audit.json",
  "quality_audit": "doc/causal_fresh_v7/quality-independent-audit.json",
  "pairings": "doc/causal_fresh_v7/quality-independent-audit.json",
  "establishment": "doc/causal_fresh_v7/final-establishment.json",
  "mirror": "doc/causal_fresh_v7/complete-mirror-attempt-001.json",
  "quality_receipt": "output/runs/causal-fresh-v7/run-001/quality-001/quality-receipt.json",
  "comparison": "doc/results/causal_fresh_v7/comparison.json",
  "v3_table": "doc/results/causal_fresh_v3/table_rows.json",
  "v4_table": "doc/results/causal_fresh_v4/table_rows.json",
  "v5_table": "doc/results/causal_fresh_v5/table_rows.json",
  "v6_table": "doc/results/causal_fresh_v6/table_rows.json",
  "v7_table": "doc/results/causal_fresh_v7/table_rows.json",
  "v6_establishment": "doc/causal_fresh_v6/final-establishment.json",
  "results": "doc/CAUSAL_FRESH_V7_RESULTS.md",
  "review": "doc/CAUSAL_FRESH_V7_REVIEW.md",
  "detail": "doc/CAUSAL_FRESH_V7_DETAIL.md"
}
'@|ConvertFrom-Json -AsHashtable
$inputs=@()
foreach($id in $roles.Keys|Sort-Object){
 $label=$roles[$id]
 $path=Join-Path $researchRoot $label
 $item=Get-Item -LiteralPath $path
 if($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw ('Expected unlinked metadata file: '+$label)}
 $format=if($id -in @('results','review','detail')){'text'}else{'json'}
 if($format -eq 'json'){$null=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
 $extension=if($format -eq 'text'){'.txt'}else{'.json'}
 $inputs += [ordered]@{id=$id;source_path=('/opt/cuwacunu/research/causal-fresh-v7-finalization/attempt-001/source/inputs/'+$id+$extension);path=$label;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant();format=$format}
}
$assertions=[ordered]@{}
$assertions.pairing_status=[ordered]@{input='pairings';path=@('status');equals='passed'}
$assertions.pairing_count=[ordered]@{input='pairings';path=@('paired_initial_equality','reference_data_archives');equals=13}
$assertions.training_audit_status=[ordered]@{input='training_audit';path=@('status');equals='passed'}
$assertions.quality_audit_status=[ordered]@{input='quality_audit';path=@('status');equals='passed'}
foreach($entry in @(@('establishment_status','status','passed'),@('establishment_runtime','runtime_trust_admission_passed',$true),@('establishment_trust','trust_admission_passed',$true),@('establishment_coverage','full_coverage',$true),@('establishment_measured','measured_candidate_established',$true))){
 $assertions[$entry[0]]=[ordered]@{input='establishment';path=@($entry[1]);equals=$entry[2]}
}
$generator=Join-Path $researchRoot 'code/research/causal_fresh_v7_finalization/finalize.py'
$plan=[ordered]@{
 kind='cpb_v7_metadata_finalization_input_plan_v1';approved_for_metadata_card_generation=$true;client_date='2026-10-16';container_attempt='/opt/cuwacunu/research/causal-fresh-v7/attempt-001'
 generator=[ordered]@{path='code/research/causal_fresh_v7_finalization/finalize.py';bytes=(Get-Item -LiteralPath $generator).Length;sha256=(Get-FileHash -LiteralPath $generator -Algorithm SHA256).Hash.ToLowerInvariant()}
 inputs=$inputs
 evidence_contract=[ordered]@{assertions=$assertions;training_audit_checks_path=@('checks');quality_audit_checks_path=@('checks');establishment_bindings_path=@('evidence');establishment_binding_keys=[ordered]@{experiment_card='card';comparison='comparison';v7_table='table_rows';pairings='pairings';training_audit='training_audit';quality_audit='quality_audit'}}
 next_direction=[ordered]@{status='proposal_only_saved_TRAIN_V6_V7_diagnosis_before_new_architecture';action='Separately declare a saved TRAIN-only V6/V7 diagnosis using existing features, fixed readouts and state-matched witnesses; no old model/head calls, fits or held-out tensor diagnosis. Use the result before declaring a broader causal temporal-summary architecture, protect all scoped V3-V7 gains prospectively, and stop the exact V7 recipe without seed/rate/cap/coefficient/head/budget rescue.'}
}
[IO.File]::WriteAllText($destination,($plan|ConvertTo-Json -Depth 40)+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='prepared_for_root_review';path='doc/causal_fresh_v7_finalization/input-plan.json';bytes=(Get-Item -LiteralPath $destination).Length;sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant();inputs=$inputs.Count}|ConvertTo-Json

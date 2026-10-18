# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host metadata pins only. Run after all final human reports and establishment exist.
param([Parameter(Mandatory=$true)][switch]$RootApprovedCompletedInputs)
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
  "prior_snapshot": "output/engineering/causal-fresh-v8-finalization/prior-live-001/manifest.json",
  "descriptor": "code/encoders/causal_patch_bottleneck_mae/versions/CPB-v8.json",
  "experiment_card": "doc/CAUSAL_FRESH_V8_CARD.json",
  "source_manifest": "output/engineering/causal-fresh-v8/attempt-001/source-manifest.json",
  "build": "output/engineering/causal-fresh-v8/attempt-001/build-admission.json",
  "training": "output/runs/causal-fresh-v8/run-001/training-summary.json",
  "resume": "output/runs/causal-fresh-v8/run-001/typed-resume-512.json",
  "initialization": "output/runs/causal-fresh-v8/run-001/fresh-initialization.json",
  "calibration": "output/runs/causal-fresh-v8/run-001/calibration/summary.json",
  "engineering": "output/engineering/causal-fresh-v8/attempt-001/engineering-proof/proof.json",
  "gradient": "output/engineering/causal-fresh-v8/attempt-001/variance-gradient-proof.json",
  "initial_proof": "output/engineering/causal-fresh-v8/attempt-001/initial-proof/proof.json",
  "final_proof": "output/engineering/causal-fresh-v8/attempt-001/final-proof/proof.json",
  "training_audit": "doc/causal_fresh_v8/independent-training-audit.json",
  "quality_audit": "doc/causal_fresh_v8/quality-independent-audit.json",
  "pairings": "doc/causal_fresh_v8/quality-independent-audit.json",
  "establishment": "doc/causal_fresh_v8/final-establishment.json",
  "mirror": "doc/causal_fresh_v8/complete-mirror-attempt-001.json",
  "quality_receipt": "output/runs/causal-fresh-v8/run-001/quality-001/quality-receipt.json",
  "comparison": "doc/results/causal_fresh_v8/comparison.json",
  "v3_table": "doc/results/causal_fresh_v3/table_rows.json",
  "v4_table": "doc/results/causal_fresh_v4/table_rows.json",
  "v5_table": "doc/results/causal_fresh_v5/table_rows.json",
  "v6_table": "doc/results/causal_fresh_v6/table_rows.json",
  "v7_table": "doc/results/causal_fresh_v7/table_rows.json",
  "v8_table": "doc/results/causal_fresh_v8/table_rows.json",
  "v6_establishment": "doc/causal_fresh_v6/final-establishment.json",
  "v7_establishment": "doc/causal_fresh_v7/final-establishment.json",
  "temporal_proof": "doc/results/causal_fresh_v8/temporal-pooling-proof.json",
  "temporal_learning": "doc/results/causal_fresh_v8/temporal-projection-learning.json",
  "host_recovery_card": "doc/causal_fresh_v8/host-orchestration-recovery-001.json",
  "host_recovery_review": "doc/causal_fresh_v8/recovery-independent-review-001.json",
  "results": "doc/CAUSAL_FRESH_V8_RESULTS.md",
  "review": "doc/CAUSAL_FRESH_V8_REVIEW.md",
  "detail": "doc/CAUSAL_FRESH_V8_DETAIL.md"
}
'@|ConvertFrom-Json -AsHashtable
if($roles.Count-ne37){throw 'Exactly37 prospective metadata roles'}
$snapshotPath=Join-Path $researchRoot $roles['prior_snapshot']
if((Get-FileHash -LiteralPath $snapshotPath -Algorithm SHA256).Hash.ToLowerInvariant()-ne'607ad940c9def57dd0390d7cbfb763c08fbaf769dbe911efc3d8eaf69f47ce7a'){throw 'Exact outside-attempt prior six-file snapshot required'}
$inputs=@()
foreach($id in $roles.Keys|Sort-Object){
 $label=$roles[$id]
 $path=Join-Path $researchRoot $label
 $item=Get-Item -LiteralPath $path
 if($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw ('Expected unlinked metadata file: '+$label)}
 $format=if($id -in @('results','review','detail')){'text'}else{'json'}
 if($format -eq 'json'){$null=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
 $extension=if($format -eq 'text'){'.txt'}else{'.json'}
 $inputs += [ordered]@{id=$id;source_path=('/opt/cuwacunu/research/causal-fresh-v8-finalization/attempt-001/source/inputs/'+$id+$extension);path=$label;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant();format=$format}
}
$assertions=[ordered]@{}
$assertions.pairing_status=[ordered]@{input='pairings';path=@('status');equals='passed'}
$assertions.pairing_count=[ordered]@{input='pairings';path=@('paired_initial_equality','reference_data_archives');equals=13}
$assertions.training_audit_status=[ordered]@{input='training_audit';path=@('status');equals='passed'}
$assertions.quality_audit_status=[ordered]@{input='quality_audit';path=@('status');equals='passed'}
foreach($entry in @(@('establishment_status','status','passed'),@('establishment_runtime','runtime_trust_admission_passed',$true),@('establishment_trust','trust_admission_passed',$true),@('establishment_coverage','full_coverage',$true),@('establishment_measured','measured_candidate_established',$true))){
 $assertions[$entry[0]]=[ordered]@{input='establishment';path=@($entry[1]);equals=$entry[2]}
}
$generator=Join-Path $researchRoot 'code/research/causal_fresh_v8_finalization/finalize.py'
$plan=[ordered]@{
 kind='cpb_v8_metadata_finalization_input_plan_v1';approved_for_metadata_card_generation=$true;client_date='2026-10-16';container_attempt='/opt/cuwacunu/research/causal-fresh-v8/attempt-001'
 generator=[ordered]@{path='code/research/causal_fresh_v8_finalization/finalize.py';bytes=(Get-Item -LiteralPath $generator).Length;sha256=(Get-FileHash -LiteralPath $generator -Algorithm SHA256).Hash.ToLowerInvariant()}
 inputs=$inputs
 evidence_contract=[ordered]@{assertions=$assertions;training_audit_checks_path=@('checks');quality_audit_checks_path=@('checks');establishment_bindings_path=@('evidence');establishment_binding_keys=[ordered]@{experiment_card='card';comparison='comparison';v8_table='table_rows';pairings='pairings';training_audit='training_audit';quality_audit='quality_audit';source_manifest='source_manifest';build='build_admission';mirror='complete_mirror';quality_receipt='quality_receipt';temporal_proof='temporal_pooling_proof';temporal_learning='temporal_learning';host_recovery_card='host_recovery_card';host_recovery_review='host_recovery_review'}}
 next_direction=[ordered]@{status='proposal_only_saved_TRAIN_V6_V8_diagnosis_before_any_new_mechanism';action='Separately declare bounded saved TRAIN-only diagnosis of the completed V8 versus preserved V6 recipe, using existing fixed readouts and stage-matched witnesses; no old model/head calls, fits or held-out tensor diagnosis. Protect every scoped V3-V8 gain and all historical results. No new encoder or seed/rate/cap/coefficient/head/budget rescue is authorized by metadata finalization.'}
}
[IO.File]::WriteAllText($destination,($plan|ConvertTo-Json -Depth 40)+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{status='prepared_for_root_review';path='doc/causal_fresh_v8_finalization/input-plan.json';bytes=(Get-Item -LiteralPath $destination).Length;sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant();inputs=$inputs.Count}|ConvertTo-Json

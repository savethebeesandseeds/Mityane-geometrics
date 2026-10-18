# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host file/hash/metadata declaration only; no report/scientific execution.
param([Parameter(Mandatory=$true)][switch]$RootReviewedRecoverySources)
$ErrorActionPreference='Stop'
if(-not$RootReviewedRecoverySources){throw 'Independent/root recovery source review required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$main='/opt/cuwacunu/research/causal-fresh-v9/attempt-001'
$runtime='/opt/cuwacunu/research/causal-fresh-v9-report-recovery/attempt-001'
$output=Join-Path $researchRoot 'doc/causal_fresh_v9/reporting-recovery-card-001.json'
if(Test-Path -LiteralPath $output){throw 'Preserve every recovery declaration'}
function Pin([string]$relative,[string]$containerPath){
 $path=[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))
 if(-not$path.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Bounded source required'}
 $file=Get-Item -LiteralPath $path
 if($file.PSIsContainer-or($file.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Physical file required'}
 [ordered]@{path=$containerPath;host_path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$failure=Pin 'doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json' ($runtime+'/source/failure-mirror.json')
if($failure.sha256-ne'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532'){throw 'Exact before-recovery failure mirror required'}
$mirror=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $failure.host_path)|ConvertFrom-Json
if($mirror.status-ne'passed'-or-not$mirror.failure_preserved-or$mirror.files.Count-ne531-or$mirror.source_files_verified-ne118){throw 'All original failed/completed scientific bytes must be preserved'}
$scientific=@(foreach($row in $mirror.files){
 $actual=Pin $row.path $row.container_path
 if($actual.bytes-ne$row.bytes-or$actual.sha256-ne$row.sha256){throw 'Preserved input changed'}
 $actual
})
$failedQuality=Pin 'output/engineering/causal-fresh-v9/attempt-001/quality-stage.json' ($main+'/quality-stage.json')
$failedReport=Pin 'output/runs/causal-fresh-v9/run-001/quality-001/report-stage.json' ($main+'/quality-001/report-stage.json')
$q=Get-Content -Raw (Join-Path $researchRoot $failedQuality.host_path)|ConvertFrom-Json
$r=Get-Content -Raw (Join-Path $researchRoot $failedReport.host_path)|ConvertFrom-Json
if($q.status-ne'failed'-or$q.exit_code-ne1-or$r.stage-ne'report'-or$r.exit_code-ne1){throw 'Original reporting failure must remain failed'}
$stages=[ordered]@{}
foreach($name in @('extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads')){
 $p=Pin ('output/runs/causal-fresh-v9/run-001/quality-001/'+$name+'-stage.json') ($main+'/quality-001/'+$name+'-stage.json')
 $value=Get-Content -Raw (Join-Path $researchRoot $p.host_path)|ConvertFrom-Json
 if($value.stage-ne$name-or$value.exit_code-ne0){throw 'All five scientific stages must already be passed'}
 $stages[$name]=$p
}
$card=Pin 'doc/CAUSAL_FRESH_V9_CARD.json' ($main+'/source/doc/CAUSAL_FRESH_V9_CARD.json')
$manifest=Pin 'output/engineering/causal-fresh-v9/attempt-001/source-manifest.json' ($main+'/source/CAPSULE_MANIFEST.json')
$archive=Pin 'output/engineering/causal-fresh-v9/attempt-001/source.zip' ($main+'/source.zip')
$identity=[ordered]@{card_sha256='94488bf2d42751b5a38c054eb5b6bbd7674e45d6e0bd444818b7f6fbcb34346d';manifest_sha256='43a44dbba0f7b885258766ab5741c9e806cc37ff0817237954b31d09de940750';archive_sha256='b54ffb00a1489dc4973efe7d93819196b4e5bcd445140851d977ba14732aa867';source_files=118;binary_count=7;translation_units=11}
if($card.sha256-ne$identity.card_sha256-or$manifest.sha256-ne$identity.manifest_sha256-or$archive.sha256-ne$identity.archive_sha256){throw 'Frozen scientific capsule cannot change'}
$mainCard=Get-Content -Raw (Join-Path $researchRoot $card.host_path)|ConvertFrom-Json
$value=[ordered]@{kind='cpb_v9_report_only_recovery_card_v1';root_reviewed=$true;declared_utc=[DateTime]::UtcNow.ToString('o');
 purpose='Reporting-only metadata guard recovery after completed extraction/heads; no scientific replay or metric/criterion change';
 main_source_identity=$identity;main_card=$card;source_manifest=$manifest;source_archive=$archive;failure_mirror=$failure;
 failed_quality_stage=$failedQuality;failed_report_stage=$failedReport;scientific_stages=$stages;scientific_inputs=$scientific;
 declared_establishment_text=$mainCard.criteria.establishment;
 reporter=(Pin 'code/research/causal_fresh_v9_report_recovery/report-recovery-001.py' ($runtime+'/source/report-recovery-001.py'));
 runner=(Pin 'code/research/causal_fresh_v9_report_recovery/run-recovery.py' ($runtime+'/source/run-recovery.py'));
 original_report=(Pin 'code/research/causal_fresh_v9/evaluation/report.py' ($main+'/source/code/research/causal_fresh_v9/evaluation/report.py'));
 comparator=(Pin 'code/research/causal_fresh_v9/evaluation/compare_baseline.py' ($main+'/source/code/research/causal_fresh_v9/evaluation/compare_baseline.py'));
 input_manifest=(Pin 'doc/causal_fresh_v9/input-manifest.json' ($main+'/source/doc/causal_fresh_v9/input-manifest.json'));
 raw_controls=(Pin 'doc/causal_fresh_v9/raw-controls.json' ($main+'/source/doc/causal_fresh_v9/raw-controls.json'));
 quality_receipt=(Pin 'output/runs/causal-fresh-v9/run-001/quality-001/quality-receipt.json' ($main+'/quality-001/quality-receipt.json'));
 outputs=[ordered]@{report=($main+'/quality-001/report');comparison=($main+'/quality-001/comparison')};
 receipt=($runtime+'/run-001/recovery.json');encoder_calls=0;head_fits=0;training_updates=0;science_stages_replayed=0}
foreach($name in @('report','comparison')){if(Test-Path -LiteralPath (Join-Path $researchRoot ('output/runs/causal-fresh-v9/run-001/quality-001/'+$name))){throw 'Recovery target must be absent'} }
[IO.File]::WriteAllText($output,(($value|ConvertTo-Json -Depth 20).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
[ordered]@{path='doc/causal_fresh_v9/reporting-recovery-card-001.json';bytes=(Get-Item $output).Length;sha256=(Get-FileHash $output -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json

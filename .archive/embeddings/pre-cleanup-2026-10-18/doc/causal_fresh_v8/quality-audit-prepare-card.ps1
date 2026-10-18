# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# File/JSON metadata only. Never executes an auditor or numerical project code.
param([Parameter(Mandatory=$true)][switch]$SourcesReviewed)
$ErrorActionPreference='Stop'
if(-not $SourcesReviewed){throw 'Independent source review must precede audit-card freeze'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$output=Join-Path $PSScriptRoot 'quality-audit-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve the prior prospective audit card'}
function Pin([string]$relative){
 $f=Get-Item -LiteralPath (Join-Path $researchRoot $relative)
 if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Physical source/evidence required'}
 [ordered]@{path=$relative;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$mainCardRelative='doc/CAUSAL_FRESH_V8_CARD.json'
$main=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $mainCardRelative)|ConvertFrom-Json
$mainPin=Pin $mainCardRelative
$inputPin=Pin 'doc/causal_fresh_v8/input-manifest.json'
if($main.model-ne'CPB-v8'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc){throw 'Frozen actual V8 prospective card required'}
if($inputPin.bytes-ne$main.input_manifest.bytes-or$inputPin.sha256-ne$main.input_manifest.sha256){throw 'Exact main-card-bound input manifest required'}
$packageRoot='output/engineering/causal-fresh-v8/attempt-001'
$manifest=Get-Content -Raw -LiteralPath (Join-Path $researchRoot "$packageRoot/source-manifest.json")|ConvertFrom-Json
$package=Get-Content -Raw -LiteralPath (Join-Path $researchRoot "$packageRoot/package.json")|ConvertFrom-Json
$manifestPin=Pin "$packageRoot/source-manifest.json"
$zipPin=Pin "$packageRoot/source.zip"
if($manifest.card_sha256-ne$mainPin.sha256-or$manifest.files.Count-ne113-or$package.source_files-ne113-or$package.manifest_sha256-ne$manifestPin.sha256-or$package.zip_sha256-ne$zipPin.sha256){throw 'Immutable113-file main package required'}
if(($main.initial_pairing.native_coordinate_slice -join ',')-ne'20,40'-or$main.initial_pairing.content20_equality_required-or$main.initial_pairing.references.Count-ne13){throw 'Declared partial native20:40 pairing only'}
$sources=@(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'code/research/causal_fresh_v8_quality_audit') -File|Sort-Object Name|ForEach-Object{
 $relative=$_.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47)
 $p=Pin $relative
 [ordered]@{file=$_.Name;bytes=$p.bytes;sha256=$p.sha256}
})
if(($sources.file -join ',')-ne'archive_reader.py,audit.py,provenance.json,README.md'){throw 'Exactly four reviewed owned auditor files required'}
$card=[ordered]@{
 kind='causal_fresh_v8_saved_quality_audit_card_v1'
 copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 declared_utc=[DateTime]::UtcNow.ToString('o')
 stage='declared_after_static_review_before_any_saved_numeric_audit'
 execution_pending=$true
 model='CPB-v8.market-01'
 model_calls=0;head_fits=0;training_updates=0;old_model_numeric_reads=0
 main_card=$mainPin
 input_manifest=$inputPin
 main_source_manifest_sha256=$manifestPin.sha256
 main_source_zip_sha256=$zipPin.sha256
 main_source_files=113
 main_attempt='attempt-001'
 outside_frozen_main_source=$true
 auditor_sources=$sources
 expected_main_paths=[ordered]@{training='/opt/cuwacunu/research/causal-fresh-v8/attempt-001/run-001';quality='/opt/cuwacunu/research/causal-fresh-v8/attempt-001/quality-001';card='/opt/cuwacunu/research/causal-fresh-v8/attempt-001/source/doc/CAUSAL_FRESH_V8_CARD.json'}
 paired_initial_reference_inputs=$main.initial_pairing.references
 native_coordinate_slice=@(20,40)
 whole_native40_equality_claimed=$false
 goal_protocol=(Pin 'doc/causal_fresh_v8/goal-protocol.json')
 source_parent=(Pin 'code/research/causal_fresh_v6_quality_audit/audit_attempt002.py')
 scope=@('Saved market predictions/physical errors/ridge reconstruction,112/32 row association,all repetitions and fixed D40 recipes','TEMPO5 saved count/group/interval/mean/coverage arithmetic;1572 recorded initial/final exports;no forwards or fits','Exactly13 references: eight partial20:40 DATA banks with whole remaining fields,five whole TRAIN-statistic archives;new own initial quality','Five clean controls,V6/V7 separately established trust and unchanged pending raw tables,c0a46522 max/min/positivity rules,chronological ties,no TRAIN NLL gate','Seven binaries/eleven translation units,separate objective and temporal CUDA proof,common temporal policy,75 identifiable learning plus sole observed registered gauge','Saved temporal8192/gauge2048 row counts/receipt pins and declared learning predicates;independent training audit owns detailed observed gradient-array/group arithmetic')
 fresh_schema_contract='Only MAIN fresh constructor JSON omits gradient_policy; require absence there and strict policy in all other actual objective-bearing trust records.'
 execution_wrapper=(Pin 'doc/causal_fresh_v8/quality-audit-execute.py')
 stage_helper=(Pin 'doc/causal_fresh_v8/quality-audit-stage.ps1')
 card_generator=(Pin 'doc/causal_fresh_v8/quality-audit-prepare-card.ps1')
 forbidden=@('Torch/NumPy or producer/comparator imports','Encoder/decoder/head forward or fitting','Old model tensor reads','Whole40 initial pairing claim','Post-encoder PCA/tuning/selection','Mutation of main113/card/source/results or previous evidence')
}
[IO.File]::WriteAllText($output,($card|ConvertTo-Json -Depth 30)+"`n",[Text.UTF8Encoding]::new($false))
Pin 'doc/causal_fresh_v8/quality-audit-card.json'|ConvertTo-Json

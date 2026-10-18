# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Full owned V9/V11 parent copy adapted to V12-only arithmetic/cachedV11 audit.
# Metadata/hash only. Creates one false PROPOSED card; root admission is separate.
param(
 [Parameter(Mandatory=$true)][switch]$RootReviewedAuditorSources,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainDiagnosisSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainMirrorSha256
)
$ErrorActionPreference='Stop'
if(-not$RootReviewedAuditorSources){throw 'Actual completed main result and reviewed auditor sources required'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$documentRoot='doc/causal_v11_v12_train_diagnosis/'
$destination=$documentRoot+'independent-audit-card.PROPOSED.json'
if(Test-Path -LiteralPath (Join-Path $researchRoot $destination)){throw 'Preserve all previous proposed audit cards'}
function Bounded([string]$root,[string]$relative){
 if([IO.Path]::IsPathRooted($relative)-or$relative.Contains(':')-or$relative.Contains([char]92)-or$relative-match'(^|/)\.\.(/|$)'){throw 'Bounded repository-relative path required'}
 $path=[IO.Path]::GetFullPath((Join-Path $root $relative))
 if(-not$path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Escaped repository'}
 $current=$path
 while($current-ne$root){$item=Get-Item -LiteralPath $current;if($item.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No linked file or parent'};$current=[IO.Path]::GetDirectoryName($current)}
 $path
}
function Pin([string]$relative){$file=Get-Item -LiteralPath (Bounded $researchRoot $relative);if($file.PSIsContainer){throw 'File required'};[ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Verify-Binding($binding){
 $repository=switch($binding.host_repository){'cuwacunu_embedding'{$researchRoot};'cuwacunu_dev'{[IO.Path]::GetFullPath((Join-Path $researchRoot '../cuwacunu_dev'))};default{throw 'Explicit known guarded repository required'}}
 $path=Bounded $repository ([string]$binding.host_path);$file=Get-Item -LiteralPath $path
 if($file.PSIsContainer-or$file.Length-ne$binding.bytes-or(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$binding.sha256){throw ('Changed guard '+$binding.id)}
 if(-not([string]$binding.container_path).StartsWith('/')-or([string]$binding.container_path).Contains([char]92)-or([string]$binding.container_path)-match'(^|/)\.\.(/|$)'){throw 'Explicit bounded absolute container path required'}
 $path.ToLowerInvariant()
}
$mainCard=Pin 'doc/CAUSAL_V11_V12_TRAIN_DIAGNOSIS_CARD.ADMITTED.json'
if($mainCard.sha256-ne$MainCardSha256){throw 'External exact main ADMITTED hash before parse'}
$main=Get-Content -LiteralPath (Join-Path $researchRoot $mainCard.path) -Raw|ConvertFrom-Json
if($main.kind-ne'cpb_v11_v12_saved_train_diagnosis_card_v1'-or$main.stage-ne'prospective_before_diagnostic_arithmetic'-or$main.execution_admitted-isnot[bool]-or$main.execution_admitted-ne$true-or$main.allowed_inputs.Count-ne25-or$main.hash_only_guards.Count-ne35-or$main.context_inputs.Count-ne14){throw 'Actual admitted25/35/14 main declaration required'}
if(($main.computed_versions-join',')-ne'v12'-or($main.cached_reference_versions-join',')-ne'v11'){throw 'New V12 arithmetic and cached V11 only'}
foreach($group in @(@{rows=@($main.allowed_inputs);count=25},@{rows=@($main.hash_only_guards);count=35},@{rows=@($main.context_inputs);count=14})){
 if($group.rows.Count-ne$group.count-or@($group.rows.id|Sort-Object -Unique).Count-ne$group.count){throw 'Exact distinct role IDs within each scope'}
}
$preservation=@();if($null-ne$main.preservation_pins){$preservation=@($main.preservation_pins)}
$guardPins=(@($main.allowed_inputs)+@($main.hash_only_guards)+@($main.context_inputs)+$preservation+@($main.pre_execution_preservation))
$hostFiles=@{};$containerPaths=@{}
foreach($binding in $guardPins){
 $key=Verify-Binding $binding;$identity=([string]$binding.bytes)+':'+$binding.sha256
 if($hostFiles.ContainsKey($key)-and$hostFiles[$key]-ne$identity){throw 'Host alias changed bytes'};$hostFiles[$key]=$identity
 $containerKey=[string]$binding.container_path
 if($containerPaths.ContainsKey($containerKey)-and$containerPaths[$containerKey]-ne$identity){throw 'Container alias changed bytes'};$containerPaths[$containerKey]=$identity
}
$counts=[ordered]@{numeric_roles=25;hash_only_guards=35;context_roles=14;preservation_roles=$preservation.Count;pre_execution_roles=1;occurrences=$guardPins.Count;unique_host_files=$hostFiles.Count;unique_container_paths=$containerPaths.Count}
$diagnosis=Pin ($documentRoot+'diagnosis.json');$mirror=Pin ($documentRoot+'mirror.json')
if($diagnosis.sha256-ne$MainDiagnosisSha256-or$mirror.sha256-ne$MainMirrorSha256){throw 'Exact actual main diagnosis/mirror hashes before parse'}
$result=Get-Content -LiteralPath (Join-Path $researchRoot $diagnosis.path) -Raw|ConvertFrom-Json
$mirrorRecord=Get-Content -LiteralPath (Join-Path $researchRoot $mirror.path) -Raw|ConvertFrom-Json
if($result.kind-ne'cpb_v11_v12_TRAIN_saved_information_diagnosis_v1'-or$result.status-ne'passed_readonly_saved_TRAIN_diagnosis'-or$result.card.bytes-ne$mainCard.bytes-or$result.card.sha256-ne$mainCard.sha256-or$mirrorRecord.status-ne'passed'-or-not$mirrorRecord.all_sizes_sha256_exact-or-not$mirrorRecord.container_before_after_exact-or$mirrorRecord.canonical_diagnosis.bytes-ne$diagnosis.bytes-or$mirrorRecord.canonical_diagnosis.sha256-cne$diagnosis.sha256){throw 'Exact completed V12-only diagnosis and full exact main mirror required'}
foreach($name in @('model_calls','encoder_calls','decoder_calls','head_calls','module_forwards','neural_forwards','fits','head_fits','training_updates','held_out_tensor_reads','old_V11_raw_numeric_reads','old_V9_numeric_reads','old_V8_numeric_reads','local_adjoint_rechecks','learning_gauge_Hadamard_trust_rechecks','V11_new_numerical_reductions')){if($result.$name-ne0){throw 'Forbidden execution counter'}}
if($result.new_encoder_experiment-ne$false-or$result.new_quality_scores-ne$false-or$result.diagnostic_gate_or_next_encoder_selected-ne$false-or$result.application_changed-ne$false-or($result.cached_V11_numeric_selectors-join'|')-ne'models.v11.stages.initial|models.v11.stages.final|temporal_moment_observations.csv|temporal_moment_observations.actual_witnesses'){throw 'Exact closed cached/new descriptor scope'}
$scienceReady=Pin $main.scientific_source_review.path
if($scienceReady.bytes-ne$main.scientific_source_review.bytes-or$scienceReady.sha256-cne$main.scientific_source_review.sha256){throw 'Exact main-card-bound combined source review'}
$science=Get-Content -LiteralPath (Join-Path $researchRoot $scienceReady.path) -Raw|ConvertFrom-Json
if($science.kind-ne'causal_v11_v12_diagnosis_source_root_review_v1'-or$science.status-ne'passed_source_review'-or$science.execution_admitted-isnot[bool]-or$science.execution_admitted-or$science.scientific_files.Count-ne10-or$science.workflow_files.Count-ne10){throw 'Combined source-only review of20 owned files'}
foreach($row in @($science.scientific_files)+@($science.workflow_files)){$p=Pin $row.path;if($p.bytes-ne$row.bytes-or$p.sha256-cne$row.sha256){throw 'Reviewed source changed'}}
$expected=@('audit.py','archive_reader.py','component_math.py','README.md','provenance.json');$sources=@()
foreach($name in $expected){
 $p=Pin ('code/research/causal_v11_v12_train_diagnosis_audit/'+$name);$binding=@($science.scientific_files|Where-Object{$_.path-eq$p.path})
 if($binding.Count-ne1-or$binding[0].bytes-ne$p.bytes-or$binding[0].sha256-ne$p.sha256){throw 'Reviewed scientific auditor source changed'}
 $sources +=[ordered]@{file=$name;path=$p.path;bytes=$p.bytes;sha256=$p.sha256}
}
$helpers=@(foreach($name in @('audit-card-generator.ps1','audit-stage.ps1','audit-execute.py','audit-mirror.ps1')){Pin ($documentRoot+$name)})
$card=[ordered]@{
 kind='cpb_v11_v12_TRAIN_diagnosis_independent_audit_card_v1';copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 declared_utc=[DateTime]::UtcNow.ToString('o');stage='PROPOSED_after_completed_new_V12_diagnosis_before_independent_audit';execution_admitted=$false
 main_card=$mainCard;main_diagnosis=$diagnosis;main_mirror=$mirror;scientific_source_review=$scienceReady
 auditor_sources=$sources;source_provenance=@($sources|Where-Object{$_.file-eq'provenance.json'})[0]
 orchestration_helpers=$helpers;execution_wrapper=@($helpers|Where-Object{$_.path.EndsWith('/audit-execute.py')})[0]
 orchestration_provenance=(Pin ($documentRoot+'audit-orchestration-provenance.json'))
 main_input_binding_counts=$counts
 numeric_input_scope='Exactly25 main roles,35 hash-only guards,14 metadata contexts plus actual declared preservation occurrences; aliases are preserved and unique host/container files counted separately.'
 independently_new_scope='V12-only native/readout/query/mean4/association/moment descriptors, initial/final,all4tasks/all3savedreadoutrepetitions,fixedTRAINhalves/all8subgroups/four512blocks and both actual1/2048 routes.'
 cached_reference_scope='Exactly4 closed cached V11 subtrees reuse old68264 audit; no V8 or old V11 raw numerical reduction.'
 manual_saved_affine_arithmetic='Explicit5families and dot-output/product counters; zero module/model/head calls does not mean zero scalar contractions.'
 forbidden='No local-adjoint/finite-difference/learning-gauge-Hadamard/globalnorm trust replay, heldout tensors, neuralweights/modules, fitting, oldencoder/head replay, new quality gate or next encoder.'
 saved_scalar_tolerance=[ordered]@{absolute=1e-10;relative=1e-9;scope='Existing independent scalar descriptor comparison; no changed V12 local-adjoint trust tolerance'}
 container=[ordered]@{name='cuwacunu';id='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e';output_root='/opt/cuwacunu/research/causal-v11-v12-train-diagnosis-audit/attempt-001'}
 root_supplied_main_card_sha256=$MainCardSha256;root_supplied_main_diagnosis_sha256=$MainDiagnosisSha256;root_supplied_main_mirror_sha256=$MainMirrorSha256
 model_calls=0;head_fits=0;training_updates=0;held_out_tensor_reads=0;old_V11_raw_numeric_reads=0;old_V9_numeric_reads=0;old_V8_numeric_reads=0;new_quality_scores=0
 failures='Preserve all new source/card/log/failed leaves; only separately reviewed additive correction, no producer/model/head replay.'
 admission='False PROPOSED bytes require actual peer/root review and separate true ADMITTED copy before Stage; actual Stage review and once-only body admission remain separate.'
}
foreach($binding in $guardPins){$null=Verify-Binding $binding}
if((Pin $mainCard.path).sha256-ne$MainCardSha256-or(Pin $diagnosis.path).sha256-ne$MainDiagnosisSha256-or(Pin $mirror.path).sha256-ne$MainMirrorSha256){throw 'Actual inputs changed during metadata declaration'}
$bytes=[Text.UTF8Encoding]::new($false).GetBytes((($card|ConvertTo-Json -Depth 20).Replace("`r`n","`n")+"`n"))
$stream=[IO.File]::Open((Join-Path $researchRoot $destination),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Pin $destination|ConvertTo-Json

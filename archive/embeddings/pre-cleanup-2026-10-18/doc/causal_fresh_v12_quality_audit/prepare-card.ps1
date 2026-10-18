# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Complete physical FINAL V11 normal helper copy, adapted to NORMAL V12 quality.
# Host file/hash/JSON metadata only. No Docker, AST, model or auditor execution.
param(
 [Parameter(Mandatory=$true)][switch]$SourcesAndCompletedEvidenceReviewed,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$MainZipSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QualityReceiptSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QualityStageSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$CompleteMirrorSHA256,
 [Parameter(Mandatory=$true)][string]$SourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SourceReviewSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$HelpersProvenanceSHA256,
 [Parameter(Mandatory=$true)][string]$WorkflowSourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WorkflowSourceReviewSHA256,
 [ValidateRange(1,999)][int]$MainAttempt=1
)
$ErrorActionPreference='Stop'
if(-not$SourcesAndCompletedEvidenceReviewed){throw 'Independent source and actual completed-quality review must precede audit-card freeze'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$helperRelative='doc/causal_fresh_v12_quality_audit'
$output=Join-Path $PSScriptRoot 'audit-card.json'
if(Test-Path -LiteralPath $output){throw 'Preserve any existing prospective audit card'}
function Rooted([string]$relative){
 $full=if([IO.Path]::IsPathRooted($relative)){[IO.Path]::GetFullPath($relative)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $relative))}
 if(-not$full.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Explicit repository-bounded file required'}
 for($cursor=$full;$cursor.Length-ge$researchRoot.Length;$cursor=[IO.Path]::GetDirectoryName($cursor)){
  if((Get-Item -LiteralPath $cursor).Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked path refused'}
  if($cursor-eq$researchRoot){break}
 }
 $full
}
function Pin([string]$relative){
 $path=Rooted $relative;$f=Get-Item -LiteralPath $path
 [ordered]@{path=$path.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Require-Pin($pin){$actual=Pin $pin.path;if($actual.bytes-ne$pin.bytes-or$actual.sha256-ne$pin.sha256){throw ('Exact file required: '+$pin.path)}}
function Workflow-RepositoryPath([string]$value){
 $full=if([IO.Path]::IsPathRooted($value)){[IO.Path]::GetFullPath($value)}else{[IO.Path]::GetFullPath((Join-Path $researchRoot $value))}
 if(-not$full.StartsWith($researchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Repository-bounded workflow review required'}
 for($cursor=$full;$cursor.Length-ge$researchRoot.Length;$cursor=[IO.Path]::GetDirectoryName($cursor)){if((Get-Item -LiteralPath $cursor).Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Linked workflow/review path refused'};if($cursor-eq$researchRoot){break}}
 $full
}
function Workflow-Pin([string]$value){
 $full=Workflow-RepositoryPath $value;$file=Get-Item -LiteralPath $full
 [ordered]@{path=$full.Substring($researchRoot.Length+1).Replace([char]92,[char]47);bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Read-IndependentWorkflowReview{
 $reviewPin=Workflow-Pin $WorkflowSourceReviewPath
 if($reviewPin.sha256-cne$WorkflowSourceReviewSHA256){throw 'External independent workflow-review SHA required before parse'}
 $review=Get-Content -LiteralPath (Workflow-RepositoryPath $reviewPin.path) -Raw|ConvertFrom-Json
 if($review.kind-cne'cpb_v12_quality_audit_workflow_independent_static_review_v1'-or$review.status-cne'passed'-or$review.source_only-isnot[bool]-or-not$review.source_only-or$review.independent_of_author-isnot[bool]-or-not$review.independent_of_author-or$review.scientific_execution_admitted-isnot[bool]-or$review.scientific_execution_admitted-ne$false){throw 'Independent SOURCE-ONLY workflow review with exact kind/status is required'}
 $prefix='doc/causal_fresh_v12_quality_audit/'
 $names=@('prepare-card.ps1','stage.ps1','execute.py','mirror.ps1','provenance.json')
 $paths=@($names|ForEach-Object{$prefix+$_})
 if($review.reviewed_sources.Count-ne$paths.Count-or(@($review.reviewed_sources.path|Sort-Object -Unique).Count)-ne$paths.Count-or(Compare-Object ($paths|Sort-Object) @($review.reviewed_sources.path|Sort-Object))){throw 'Review must bind exactly five owned workflow files including provenance'}
 foreach($entry in $review.reviewed_sources){$actual=Workflow-Pin $entry.path;if($actual.bytes-ne$entry.bytes-or$actual.sha256-cne$entry.sha256){throw ('Reviewed workflow source changed: '+$entry.path)}}
 $provenancePin=Workflow-Pin ($prefix+'provenance.json')
 $provenance=Get-Content -LiteralPath (Workflow-RepositoryPath $provenancePin.path) -Raw|ConvertFrom-Json
 $executablePaths=@($paths|Where-Object{$_-ne($prefix+'provenance.json')})
 if($provenance.kind-cne'cpb_v12_owned_quality_audit_orchestration_source_v1'-or$provenance.current_sources.Count-ne$executablePaths.Count-or(Compare-Object ($executablePaths|Sort-Object) @($provenance.current_sources.path|Sort-Object))){throw 'Exact complete four-helper provenance required'}
 foreach($entry in $provenance.current_sources){$matched=@($review.reviewed_sources|Where-Object{$_.path-ceq$entry.path});if($matched.Count-ne1-or$matched[0].bytes-ne$entry.bytes-or$matched[0].sha256-cne$entry.sha256){throw 'Independent review and current helper provenance disagree'}}
 [ordered]@{review_pin=$reviewPin;review=$review;provenance_pin=$provenancePin;provenance=$provenance}
}
$workflowGuard=Read-IndependentWorkflowReview
$workflowPin=$workflowGuard.provenance_pin
if($workflowPin.sha256-ne$HelpersProvenanceSHA256){throw 'External reviewed helper provenance SHA required before parse'}
$workflow=$workflowGuard.provenance
if($workflow.kind-ne'cpb_v12_owned_quality_audit_orchestration_source_v1'-or$workflow.current_sources.Count-ne4){throw 'Exact reviewed four-helper closure required'}
foreach($entry in $workflow.current_sources){Require-Pin $entry}
$self=@($workflow.current_sources|Where-Object{$_.path-eq($helperRelative+'/prepare-card.ps1')})
if($self.Count-ne1-or$self[0].sha256-ne(Pin $PSCommandPath).sha256){throw 'Generator self pin required'}
$reviewPin=Pin $SourceReviewPath
if($reviewPin.path-cne'doc/causal_fresh_v12/independent-quality-evaluation-source-review-001.json'-or$reviewPin.sha256-cne$SourceReviewSHA256-or$SourceReviewSHA256-cne'1fb03d37f56ee1c85fcc08829d6de866c74c553decab7604cf6442e3dcc7dc33'){throw 'Exact existing root independent scientific source review required before parse'}
$review=Get-Content -Raw -LiteralPath (Rooted $reviewPin.path)|ConvertFrom-Json
if($review.kind-cne'cpb_v12_quality_evaluation_independent_static_review_v1'-or$review.status-cne'passed'-or$review.owned_sources.Count-ne24-or$review.review_scope.evaluation_files-ne20-or$review.review_scope.quality_auditor_files-ne4-or$review.scientific_execution_admitted-ne$false){throw 'Actual root review covers20 evaluation plus4 quality sources;do not invent an old four-source review kind'}
$sourceRelative='code/research/causal_fresh_v12_quality_audit'
$sources=@(Get-ChildItem -LiteralPath (Rooted $sourceRelative) -File|Sort-Object Name|ForEach-Object{
 $relative=$_.FullName.Substring($researchRoot.Length+1).Replace([char]92,[char]47);$pin=Pin $relative
 $binding=@($review.owned_sources|Where-Object{$_.path-ceq$relative})
 if($binding.Count-ne1-or$binding[0].bytes-ne$pin.bytes-or$binding[0].sha256-ne$pin.sha256){throw 'Current auditor must match independent review exactly'}
 [ordered]@{file=$_.Name;bytes=$pin.bytes;sha256=$pin.sha256}
})
if(($sources.file-join',')-cne'archive_reader.py,audit.py,provenance.json,README.md'){throw 'Exactly four owned auditor sources required'}
$mainPin=Pin 'doc/CAUSAL_FRESH_V12_CARD.json'
if($mainPin.sha256-ne$MainCardSHA256){throw 'External actual main card SHA required before parse'}
$main=Get-Content -Raw -LiteralPath (Rooted $mainPin.path)|ConvertFrom-Json
$inputPin=Pin 'doc/causal_fresh_v12/input-manifest.json'
if($main.model-ne'CPB-v12'-or$main.stage.Contains('DRAFT')-or-not$main.frozen_utc-or-not$main.execution_admitted){throw 'Actual prospectively frozen V12 scientific card required'}
if($inputPin.bytes-ne$main.input_manifest.bytes-or$inputPin.sha256-ne$main.input_manifest.sha256){throw 'Exact main-card data-only input identity required'}
if(($main.initial_pairing.native_coordinate_slice-join',')-ne'20,40'-or$main.initial_pairing.content20_equality_required-ne$false-or$main.initial_pairing.native_coordinates_compared-ne20-or$main.initial_pairing.references.Count-ne13){throw 'Exact partial20:40 eight-DATA/five-whole-statistic pairing declaration required'}
if($main.initial_pairing.whole_native40_equality_required-ne$false-or$main.initial_pairing.initial_scores_reused-ne$false){throw 'Only saved V11 DATA tails/whole-statistic references;new own initial heads'}
foreach($reference in $main.initial_pairing.references){if(-not$reference.reference_data_archive.path.StartsWith('/opt/cuwacunu/research/causal-fresh-v11/attempt-001/')-or$reference.candidate_relative_to_attempt-match'cpb_v11-'){throw 'V12 references saved V11 DATA only'};if($reference.PSObject.Properties.Name-contains'feature_value_coordinate_slice'){if(($reference.feature_value_coordinate_slice-join',')-cne'20,40'){throw 'Only feature value coordinates20:40 may be paired'}}}
$policy='original-position-four-bin-attention-mean4-uniform-centered-variance4-v1'
$momentPolicy='componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1'
foreach($record in @($main.mechanism,$main.training)){if($record.temporal_pooling_policy-cne$policy-or$record.temporal_moment_policy-cne$momentPolicy-or($record.temporal_moment_partition-join',')-cne'4,4'-or($record.temporal_pooling_mix-join',')-cne'0.5,0.5'){throw 'Exact prospective V12 attention-mean/uniform-own-center dispersion declaration required'}}
$proofFlags=@('attention_mean_and_uniform_dispersion_full_expression_adjoints','uniform_dispersion_own_center','uniform_dispersion_score_independent','uniform_dispersion_hidden_gradient_live','attention_mean_score_gradient_live','dual_support_actual_trace')
foreach($field in $proofFlags){if($main.temporal_pooling_proof_contract.required_proof_fields.$field-isnot[bool]-or-not$main.temporal_pooling_proof_contract.required_proof_fields.$field){throw 'All six new V12 proof predicates must have been prospectively frozen'}}
if($main.temporal_pooling_proof_contract.required_proof_fields.direct_pool_kernel_calls-ne47-or$main.actual_step_temporal_witness_contract.total_fields-ne60-or$main.actual_step_temporal_witness_contract.temporal_trace_fields-ne25){throw 'Source-derived47 calls and60-field same-forward archive required'}
if($main.advancement.goal_protocol.sha256-cne'866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'-or$main.advancement.criterion_count-ne22-or($main.scoped_references-join',')-cne'CPB-v4.market-01,CPB-v5.market-01,CPB-v6.market-01,CPB-v7.market-01,CPB-v8.market-01,CPB-v9.market-01,CPB-v10.market-01,CPB-v11.market-01'){throw 'Nine saved controls and exact strongest22 future floors required'}
$label='attempt-{0:D3}' -f $MainAttempt
$packageRoot='output/engineering/causal-fresh-v12/'+$label
$runtimeMain='/opt/cuwacunu/research/causal-fresh-v12/'+$label
$manifestPin=Pin ($packageRoot+'/source-manifest.json')
$zipPin=Pin ($packageRoot+'/source.zip')
if($manifestPin.sha256-ne$MainManifestSHA256-or$zipPin.sha256-ne$MainZipSHA256){throw 'External actual main manifest/ZIP SHA guards before parse'}
$manifest=Get-Content -Raw -LiteralPath (Rooted $manifestPin.path)|ConvertFrom-Json
$packagePin=Pin ($packageRoot+'/package.json')
$package=Get-Content -Raw -LiteralPath (Rooted $packagePin.path)|ConvertFrom-Json
if($manifest.card_sha256-ne$MainCardSHA256-or$manifest.files.Count-le0-or$package.source_files-ne$manifest.files.Count-or$package.manifest_sha256-ne$MainManifestSHA256-or$package.zip_sha256-ne$MainZipSHA256){throw 'Immutable actual main package identity required'}
$qualityPin=Pin 'output/runs/causal-fresh-v12/run-001/quality-001/quality-receipt.json'
$qualityStageHost=Pin ($packageRoot+'/quality-stage.json')
$completePin=Pin ('doc/causal_fresh_v12/complete-mirror-'+$label+'.json')
if($qualityPin.sha256-ne$QualityReceiptSHA256-or$qualityStageHost.sha256-ne$QualityStageSHA256-or$completePin.sha256-ne$CompleteMirrorSHA256){throw 'External passed quality receipt/stage/Complete hashes required before parse'}
$quality=Get-Content -Raw -LiteralPath (Rooted $qualityPin.path)|ConvertFrom-Json
$qualityStage=Get-Content -Raw -LiteralPath (Rooted $qualityStageHost.path)|ConvertFrom-Json
$complete=Get-Content -Raw -LiteralPath (Rooted $completePin.path)|ConvertFrom-Json
if($quality.kind-ne'causal_fresh_v12_quality_execution_v1'-or$quality.status-ne'passed'-or$quality.model-ne'CPB-v12.market-01'-or$quality.card_sha256-ne$MainCardSHA256-or$quality.gradient_policy-ne'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'){throw 'Actual passed normal V12 quality receipt required'}
if($qualityStage.kind-ne'causal_fresh_v12_exclusive_stage_execution_v1'-or$qualityStage.status-ne'passed'-or$qualityStage.exit_code-ne0){throw 'Actual completed outer quality wrapper required,not a reporting-recovery assumption'}
if($complete.status-ne'passed'-or$complete.scope-ne'Complete'-or$complete.attempt-ne$label-or$complete.card_sha256-ne$MainCardSHA256-or$complete.source_manifest_sha256-ne$MainManifestSHA256-or-not$complete.all_mirrored_pt_ignored){throw 'Passed exact completed main mirror required'}
$sourcePackages=@($manifestPin,$zipPin,$packagePin)
foreach($entry in $sourcePackages){$preserved=@($complete.preserved_source_packages|Where-Object{$_.path-ceq$entry.path});if($preserved.Count-ne1-or$preserved[0].bytes-ne$entry.bytes-or$preserved[0].sha256-cne$entry.sha256){throw 'One exact preserved_source_packages host manifest/wholeZIP/package role required;these are not container payload files'};Require-Pin $preserved[0]}
function Require-CompletedFile([string]$relative,[string]$runtime){
 $matches=@($complete.files|Where-Object{$_.path-eq$relative-and$_.container_path-eq$runtime})
 if($matches.Count-ne1){throw ('One exact completed mirror member required: '+$relative)}
 Require-Pin $matches[0]
 $matches[0]
}
$null=Require-CompletedFile $qualityPin.path ($runtimeMain+'/quality-001/quality-receipt.json')
$null=Require-CompletedFile $qualityStageHost.path ($runtimeMain+'/quality-stage.json')
$temporalMatches=@($complete.files|Where-Object{$_.container_path-ceq$quality.temporal_pooling_proof.path})
if($temporalMatches.Count-ne1-or$temporalMatches[0].bytes-ne$quality.temporal_pooling_proof.bytes-or$temporalMatches[0].sha256-cne$quality.temporal_pooling_proof.sha256){throw 'One exact Complete-mirrored current temporal proof required'}
Require-Pin $temporalMatches[0]
$temporal=Get-Content -LiteralPath (Rooted $temporalMatches[0].path) -Raw|ConvertFrom-Json
if($temporal.kind-cne'cpb_v12_temporal_pooling_mechanism_proof_v1'-or$temporal.status-cne'passed'-or$temporal.model-cne'CPB-v12'-or$temporal.temporal_pooling_policy-cne$policy-or$temporal.temporal_moment_policy-cne$momentPolicy-or$temporal.direct_pool_kernel_calls-ne47){throw 'Actual saved47-call V12 dual-support proof must match the prospective recipe'}
foreach($field in $proofFlags){if($temporal.$field-isnot[bool]-or-not$temporal.$field){throw ('Actual new V12 temporal proof failed: '+$field)}}
$science=@('extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads')
if((@($quality.stage_seconds.PSObject.Properties.Name|Sort-Object)-join',')-cne(@($science|Sort-Object)-join',')){throw 'The receipt contains five scientific durations,before report/comparison'}
foreach($name in @($science)+@('report','comparison')){
 $relative='output/runs/causal-fresh-v12/run-001/quality-001/'+$name+'-stage.json'
 $null=Require-CompletedFile $relative ($runtimeMain+'/quality-001/'+$name+'-stage.json')
 $stage=Get-Content -Raw -LiteralPath (Rooted $relative)|ConvertFrom-Json
 if($stage.stage-ne$name-or$stage.exit_code-ne0){throw 'All seven normal local stages must have passed'}
}
foreach($relative in @('report/table_rows.json','report/completion.json','comparison/comparison.json')){
 $null=Require-CompletedFile ('output/runs/causal-fresh-v12/run-001/quality-001/'+$relative) ($runtimeMain+'/quality-001/'+$relative)
}
$qualityRuntimePin=[ordered]@{path=($runtimeMain+'/quality-stage.json');bytes=$qualityStageHost.bytes;sha256=$qualityStageHost.sha256}
$payloadFiles=@($sources.file)+@('audit-card.json','LICENSE','execute-audit.py','workflow-review.json','workflow-provenance.json','scientific-source-review.json')
if(@($payloadFiles|Sort-Object -Unique).Count-ne$payloadFiles.Count){throw 'Unique flat audit capsule role names required'}
$pythonFiles=@($payloadFiles|Where-Object{$_.EndsWith('.py')}|Sort-Object)
$card=[ordered]@{
 kind='causal_fresh_v12_saved_quality_audit_card_v1';copyright='Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.'
 declared_utc=[DateTime]::UtcNow.ToString('o');stage='declared_after_static_review_before_any_saved_numeric_audit'
 execution_pending=$true;model='CPB-v12.market-01';model_calls=0;head_fits=0;training_updates=0;old_model_numeric_reads=0
 main_card=$mainPin;input_manifest=$inputPin;main_source_manifest_sha256=$manifestPin.sha256;main_source_zip_sha256=$zipPin.sha256;main_source_files=$manifest.files.Count
 main_attempt=$label;audit_attempt='attempt-001';outside_frozen_main_source=$true;auditor_sources=$sources
 expected_main_paths=[ordered]@{training=($runtimeMain+'/run-001');quality=($runtimeMain+'/quality-001');card=($runtimeMain+'/source/doc/CAUSAL_FRESH_V12_CARD.json')}
 paired_initial_reference_inputs=$main.initial_pairing.references;native_coordinate_slice=@(20,40);native_coordinates_compared=20;content20_equality_required=$false;whole_native40_equality_claimed=$false
 goal_protocol=(Pin 'doc/causal_fresh_v12/goal-protocol.json')
 source_parent=(Pin 'code/research/causal_fresh_v11_quality_audit/audit.py')
 independent_source_review=$reviewPin;helpers_provenance=$workflowPin;independent_workflow_source_review=$workflowGuard.review_pin
 independent_source_review_scope='Root independent20evaluation+4quality review;exact quality4 subset bound separately from workflow review'
 staging_transport=[ordered]@{payload_files=$payloadFiles;payload_count=$payloadFiles.Count;ZIP_member_count=($payloadFiles.Count+1);python_AST_files=$pythonFiles;python_AST_count=$pythonFiles.Count;workflow_review_file='workflow-review.json';workflow_provenance_file='workflow-provenance.json';scientific_source_review_file='scientific-source-review.json';external_workflow_review_SHA_required_at_every_phase=$true}
 temporal_pooling_policy=$policy;temporal_moment_policy=$momentPolicy;required_new_temporal_flags=$proofFlags;temporal_fixture_kernel_calls=47;actual_step_archive_fields=60
 completed_quality_receipt=$qualityPin;completed_quality_stage=$qualityRuntimePin;completed_main_mirror=$completePin
 completed_main_source_packages=$sourcePackages
 report_only_recovery_required=$false
 scope=@('All112TRAIN/32knownVALIDATION saved market predictions,physical errors,ridge reconstruction,all3repetitions,fixedD40recipes;no fitting','TEMPO5 saved count/group/interval/mean/coverage arithmetic;1572 recorded initial/final exports;no model or head implementation forward','13savedV11 DATA references:eight initial native20:40 tails with whole IDs/support/labels plus five whole fresh-statistic archives;content0:20 unpaired;no imported weights/statistics or old score reuse','Nine saved clean V3–V11 controls,six separate V6–V11 trust joins;historicalV6/V7/V10FULL40 andV8/V9/V11PARTIAL semantics plus opaqueV9 recovery unchanged','866dec strongest22 exact ordered max/min/positivity conditions,earlier chronological ties,noTRAINNLLgate;no automatic winner/application selection','Sevenbinary11TU actualsource closure;unchanged V11 detached-error objective and new47-call attention-mean/uniform-own-center dispersion proof with6new flags;75mandatory tensors plus scalar gauge capacity','Actual passed normal outerquality wrapper/five scientific receipt durations/seven local exit0 stage files and report/comparison outputs;no current recovery','Four temporal CSV artifacts8192/2048/2048/16384 and60-field same-existing-forward actual1/2048 witnesses;independent TRAIN auditor owns detailed saved-array local adjoints;no every-scalar/full-hidden proof claim')
 fresh_schema_contract='Only main fresh-constructor JSON omits gradient_policy; requireabsence there and strict policies elsewhere.'
 execution_wrapper=(Pin ($helperRelative+'/execute.py'));stage_helper=(Pin ($helperRelative+'/stage.ps1'));mirror_helper=(Pin ($helperRelative+'/mirror.ps1'));card_generator=(Pin ($helperRelative+'/prepare-card.ps1'))
 forbidden=@('Torch/NumPy or producer imports','Encoder/decoder/head forward or fitting','Old trained checkpoint numeric reads','Whole40 current initial equality claim','Current reporting recovery arguments or prior-failure assumptions','PCA/tuning/selection','Mutation of source/cards/results/old evidence')
}
[IO.File]::WriteAllText($output,($card|ConvertTo-Json -Depth 30)+"`n",[Text.UTF8Encoding]::new($false))
Pin ($helperRelative+'/audit-card.json')|ConvertTo-Json -Depth 5

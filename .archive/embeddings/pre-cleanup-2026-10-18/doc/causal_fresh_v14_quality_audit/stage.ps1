# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
param([ValidateSet('quality')][string]$Kind='quality',[ValidatePattern('^[0-9]{3}$')][string]$Attempt='001',
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSHA256,
 [Parameter(Mandatory=$true)][string]$WorkflowSourceReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WorkflowSourceReviewSHA256,
 [Parameter(Mandatory=$true)][string]$ActualCardPeerReviewPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ActualCardPeerReviewSHA256)
$ErrorActionPreference='Stop'
if($Attempt-ne'001'){throw 'This additive helper owns only the exclusive first audit attempt001'}
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$containerName='cuwacunu'
$containerId='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$containerImage='sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'
function File-Pin([string]$path){$f=Get-Item -LiteralPath $path;if($f.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'No source links'};[ordered]@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
function Require-Pin([string]$path,$expected){$actual=File-Pin $path;if($actual.bytes-ne$expected.bytes-or$actual.sha256-ne$expected.sha256){throw "Frozen source changed: $path"}}
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
 if($review.kind-cne'cpb_v14_quality_audit_workflow_independent_static_review_v1'-or$review.status-cne'passed'-or$review.source_only-isnot[bool]-or-not$review.source_only-or$review.independent_of_author-isnot[bool]-or-not$review.independent_of_author-or$review.scientific_execution_admitted-isnot[bool]-or$review.scientific_execution_admitted-ne$false){throw 'Independent SOURCE-ONLY workflow review with exact kind/status is required'}
 $prefix='doc/causal_fresh_v14_quality_audit/'
 $names=@('prepare-card.ps1','stage.ps1','execute.py','mirror.ps1','provenance.json')
 $paths=@($names|ForEach-Object{$prefix+$_})
 if($review.reviewed_sources.Count-ne$paths.Count-or(@($review.reviewed_sources.path|Sort-Object -Unique).Count)-ne$paths.Count-or(Compare-Object ($paths|Sort-Object) @($review.reviewed_sources.path|Sort-Object))){throw 'Review must bind exactly five owned workflow files including provenance'}
 foreach($entry in $review.reviewed_sources){$actual=Workflow-Pin $entry.path;if($actual.bytes-ne$entry.bytes-or$actual.sha256-cne$entry.sha256){throw ('Reviewed workflow source changed: '+$entry.path)}}
 $provenancePin=Workflow-Pin ($prefix+'provenance.json')
 $provenance=Get-Content -LiteralPath (Workflow-RepositoryPath $provenancePin.path) -Raw|ConvertFrom-Json
 $executablePaths=@($paths|Where-Object{$_-ne($prefix+'provenance.json')})
 if($provenance.kind-cne'cpb_v14_owned_quality_audit_orchestration_source_v1'-or$provenance.current_sources.Count-ne$executablePaths.Count-or(Compare-Object ($executablePaths|Sort-Object) @($provenance.current_sources.path|Sort-Object))){throw 'Exact complete four-helper provenance required'}
 foreach($entry in $provenance.current_sources){$matched=@($review.reviewed_sources|Where-Object{$_.path-ceq$entry.path});if($matched.Count-ne1-or$matched[0].bytes-ne$entry.bytes-or$matched[0].sha256-cne$entry.sha256){throw 'Independent review and current helper provenance disagree'}}
 [ordered]@{review_pin=$reviewPin;review=$review;provenance_pin=$provenancePin;provenance=$provenance}
}
$workflowGuard=Read-IndependentWorkflowReview
$sourceRel="code/research/causal_fresh_v14_${Kind}_audit"
$cardRel='doc/causal_fresh_v14_quality_audit/audit-card.json'
$cardPath=Join-Path $researchRoot $cardRel
if((File-Pin $cardPath).sha256-ne$AuditCardSHA256){throw 'Externally reviewed audit card required before parse'}
$card=Get-Content -Raw -LiteralPath $cardPath|ConvertFrom-Json
if($card.stage-ne'declared_after_static_review_before_any_saved_numeric_audit'-or-not$card.execution_pending-or$card.model-ne'CPB-v14.market-01'){throw 'Independent static review/frozen declaration required'}
$cardPeerPin=Workflow-Pin $ActualCardPeerReviewPath
if($cardPeerPin.sha256-cne$ActualCardPeerReviewSHA256){throw 'External actual-card independent review SHA required before parse or staging'}
$cardPeer=Get-Content -LiteralPath (Workflow-RepositoryPath $cardPeerPin.path) -Raw|ConvertFrom-Json
if($cardPeer.kind-cne'cpb_v14_quality_audit_actual_card_independent_review_v1'-or$cardPeer.status-cne'passed'-or$cardPeer.independent_of_card_author-isnot[bool]-or-not$cardPeer.independent_of_card_author-or$cardPeer.actual_card.path-cne$cardRel-or$cardPeer.actual_card.sha256-cne$AuditCardSHA256-or$cardPeer.actual_card.bytes-ne(File-Pin $cardPath).bytes){throw 'Actual frozen audit-card nonself peer review must pass before AST-only Stage'}
Require-Pin $PSCommandPath $card.stage_helper
if($card.independent_workflow_source_review.sha256-cne$WorkflowSourceReviewSHA256-or$card.independent_workflow_source_review.bytes-ne$workflowGuard.review_pin.bytes-or$card.helpers_provenance.sha256-cne$workflowGuard.provenance_pin.sha256-or$card.helpers_provenance.bytes-ne$workflowGuard.provenance_pin.bytes){throw 'Card and external independent workflow review/provenance must agree'}
$expectedRoot104=$workflowGuard.provenance.scientific_source_authorities.combined_root104
if($card.independent_source_review.path-cne$expectedRoot104.path-or$card.independent_source_review.bytes-ne$expectedRoot104.bytes-or$card.independent_source_review.sha256-cne$expectedRoot104.sha256){throw 'Actual revised root104closure002 card/provenance binding required'}
foreach($authority in @($card.independent_source_review,$card.independent_quality_source_review,$card.independent_evaluation_source_review)){Require-Pin (Workflow-RepositoryPath $authority.path) $authority}
$scientificReview=Get-Content -LiteralPath (Workflow-RepositoryPath $card.independent_source_review.path) -Raw|ConvertFrom-Json
$qualityReview=Get-Content -LiteralPath (Workflow-RepositoryPath $card.independent_quality_source_review.path) -Raw|ConvertFrom-Json
$evaluationReview=Get-Content -LiteralPath (Workflow-RepositoryPath $card.independent_evaluation_source_review.path) -Raw|ConvertFrom-Json
if($scientificReview.kind-cne'cpb_v14_complete_104_owned_source_root_closure_v1'-or$scientificReview.status-cne'passed'-or$scientificReview.reviewed_owned_files.Count-ne104-or$qualityReview.kind-cne'cpb_v14_independent_quality_auditor_source_review_v1'-or$qualityReview.status-cne'passed'-or$qualityReview.independent_static_source_review-ne$true-or$qualityReview.reviewed_owned_files.Count-ne4-or$evaluationReview.kind-cne'cpb_v14_training_evaluation_and_training_auditor_independent_static_review_v1'-or$evaluationReview.status-cne'passed'-or$evaluationReview.independent_of_author-ne$true-or$evaluationReview.reviewed_owned_files.Count-ne32){throw 'Actual separate independent4+20 authorities and root104 source closure required'}
foreach($authority in @($card.independent_quality_source_review,$card.independent_evaluation_source_review)){$match=@($scientificReview.independent_reviews.PSObject.Properties.Value|Where-Object{$_.path-ceq$authority.path});if($match.Count-ne1-or$match[0].bytes-ne$authority.bytes-or$match[0].sha256-cne$authority.sha256){throw 'Root source closure must name the actual two independent authorities'}}
$sourceFiles=@('audit.py','archive_reader.py','README.md','provenance.json')
$declared=@($card.auditor_sources)
if($declared.Count-ne4-or($declared.file-join',')-cne'archive_reader.py,audit.py,provenance.json,README.md'-or$card.report_only_recovery_required-ne$false){throw 'Exact four-source normal V14quality audit declaration required'}
foreach($entry in $declared){$name=if($entry.path){[IO.Path]::GetFileName($entry.path)}else{$entry.file};Require-Pin (Join-Path $researchRoot "$sourceRel/$name") $entry;$bound=@($qualityReview.reviewed_owned_files|Where-Object{$_.path-ceq($sourceRel+'/'+$name)});if($bound.Count-ne1-or$bound[0].bytes-ne$entry.bytes-or$bound[0].sha256-cne$entry.sha256){throw 'Current4 auditor sources must match nonself independent quality review'}}
$wrapperPath=Join-Path $researchRoot 'doc/causal_fresh_v14_quality_audit/execute.py'
Require-Pin $wrapperPath $card.execution_wrapper
$payloadFiles=@($sourceFiles)+@('audit-card.json','LICENSE','execute-audit.py','workflow-review.json','workflow-provenance.json','scientific-source-review.json','quality-source-review.json','evaluation-source-review.json')
$pythonFiles=@($payloadFiles|Where-Object{$_.EndsWith('.py')}|Sort-Object)
if($card.staging_transport.payload_count-ne$payloadFiles.Count-or$card.staging_transport.ZIP_member_count-ne($payloadFiles.Count+1)-or(Compare-Object ($payloadFiles|Sort-Object) @($card.staging_transport.payload_files|Sort-Object))-or$card.staging_transport.python_AST_count-ne$pythonFiles.Count-or(Compare-Object $pythonFiles @($card.staging_transport.python_AST_files|Sort-Object))){throw 'Source-derived exact flat payload/member/Python-AST role set required'}
$hostRoot=Join-Path $researchRoot "output/engineering/causal-fresh-v14-$Kind-audit/attempt-$Attempt"
if(Test-Path -LiteralPath $hostRoot){throw 'Preserve existing audit attempt'}
$containerRoot="/opt/cuwacunu/research/causal-fresh-v14-$Kind-audit/attempt-$Attempt"
$observed=docker.exe inspect $containerName|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$observed.Id-ne$containerId-or$observed.Image-ne$containerImage-or-not$observed.State.Running){throw 'Exact existing running managed container required'}
[IO.Directory]::CreateDirectory($hostRoot)|Out-Null
$entries=@()
foreach($name in $sourceFiles){$path=Join-Path $researchRoot "$sourceRel/$name";$p=File-Pin $path;$entries +=[ordered]@{file=$name;repository_path="$sourceRel/$name";bytes=$p.bytes;sha256=$p.sha256}}
foreach($item in @([ordered]@{file='audit-card.json';path=$cardPath},[ordered]@{file='LICENSE';path=(Join-Path $researchRoot 'LICENSE')})){$p=File-Pin $item.path;$entries +=[ordered]@{file=$item.file;repository_path=$item.path;bytes=$p.bytes;sha256=$p.sha256}}
$wrapperPin=File-Pin $wrapperPath;$entries +=[ordered]@{file='execute-audit.py';repository_path=$wrapperPath;bytes=$wrapperPin.bytes;sha256=$wrapperPin.sha256}
foreach($item in @([ordered]@{file='workflow-review.json';path=(Workflow-RepositoryPath $workflowGuard.review_pin.path)},[ordered]@{file='workflow-provenance.json';path=(Workflow-RepositoryPath $workflowGuard.provenance_pin.path)},[ordered]@{file='scientific-source-review.json';path=(Workflow-RepositoryPath $card.independent_source_review.path)},[ordered]@{file='quality-source-review.json';path=(Workflow-RepositoryPath $card.independent_quality_source_review.path)},[ordered]@{file='evaluation-source-review.json';path=(Workflow-RepositoryPath $card.independent_evaluation_source_review.path)})){$p=File-Pin $item.path;$entries +=[ordered]@{file=$item.file;repository_path=$item.path;bytes=$p.bytes;sha256=$p.sha256}}
if($entries.Count-ne$payloadFiles.Count-or(Compare-Object ($payloadFiles|Sort-Object) @($entries.file|Sort-Object))){throw 'Payload files must exactly equal declared roles'}
$manifest=[ordered]@{kind="cpb_v14_${Kind}_independent_audit_source_manifest_v1";stage='AST_only_no_numeric';container_id=$containerId;container_image=$containerImage;container_root=$containerRoot;audit_card_sha256=$AuditCardSHA256;actual_card_peer_review=$cardPeerPin;workflow_source_review_sha256=$WorkflowSourceReviewSHA256;scientific_source_review_sha256=$card.independent_source_review.sha256;quality_source_review_sha256=$card.independent_quality_source_review.sha256;evaluation_source_review_sha256=$card.independent_evaluation_source_review.sha256;files=$entries}
$manifestPath=Join-Path $hostRoot 'source-manifest.json'
[IO.File]::WriteAllText($manifestPath,($manifest|ConvertTo-Json -Depth 10)+"`n",[Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zipPath=Join-Path $hostRoot 'source.zip'
$stream=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew)
$archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
try{foreach($entry in $entries){$path=if([IO.Path]::IsPathRooted($entry.repository_path)){$entry.repository_path}else{Join-Path $researchRoot $entry.repository_path};Require-Pin $path $entry;[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$path,$entry.file,[IO.Compression.CompressionLevel]::Fastest)|Out-Null};[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$manifestPath,'SOURCE_MANIFEST.json',[IO.Compression.CompressionLevel]::Fastest)|Out-Null}finally{$archive.Dispose();$stream.Dispose()}
$manifestPin=File-Pin $manifestPath;$zipPin=File-Pin $zipPath
$setup=@'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);assert str(p).startswith('/opt/cuwacunu/research/causal-fresh-v14-') and not p.exists()
p.mkdir(parents=True)
'@
docker.exe exec $containerName python3 -c $setup $containerRoot 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'setup.log')
if($LASTEXITCODE-ne0){throw 'Exclusive container audit leaf refused; preserve host capsule'}
docker.exe cp $zipPath "${containerName}:${containerRoot}/source.zip"
if($LASTEXITCODE-ne0){throw 'Audit capsule copy failed; preserve attempt'}
$stage=@'
import ast,datetime,hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]);kind,expected_zip,expected_manifest,expected_card,expected_workflow=sys.argv[2:]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(root/'source.zip')==expected_zip,'External ZIP hash before ZIP open'
with zipfile.ZipFile(root/'source.zip') as z:
 raw_manifest=z.read('SOURCE_MANIFEST.json')
 assert hashlib.sha256(raw_manifest).hexdigest()==expected_manifest,'External raw manifest hash before JSON parse'
 m=json.loads(raw_manifest);names=[e['file'] for e in m['files']]
 assert len(names)==len(set(names)) and len(z.namelist())==len(names)+1 and set(z.namelist())==set(names+['SOURCE_MANIFEST.json'])
 assert m['container_root']==str(root) and m['stage']=='AST_only_no_numeric' and m['audit_card_sha256']==expected_card
 assert hashlib.sha256(z.read('audit-card.json')).hexdigest()==expected_card,'External card before extraction/AST'
 assert hashlib.sha256(z.read('workflow-review.json')).hexdigest()==expected_workflow,'External workflow review before parse/AST'
 card=json.loads(z.read('audit-card.json'));workflow=json.loads(z.read('workflow-review.json'))
 assert workflow['kind']=='cpb_v14_quality_audit_workflow_independent_static_review_v1' and workflow['status']=='passed'
 assert workflow['source_only'] is True and workflow['independent_of_author'] is True and workflow['scientific_execution_admitted'] is False
 assert card['independent_workflow_source_review']['sha256']==expected_workflow==m['workflow_source_review_sha256']
 assert set(names)==set(card['staging_transport']['payload_files']) and len(names)==card['staging_transport']['payload_count']
 assert len(z.namelist())==card['staging_transport']['ZIP_member_count']
 assert hashlib.sha256(z.read('workflow-provenance.json')).hexdigest()==card['helpers_provenance']['sha256']
 assert hashlib.sha256(z.read('scientific-source-review.json')).hexdigest()==card['independent_source_review']['sha256']==m['scientific_source_review_sha256']
 assert hashlib.sha256(z.read('quality-source-review.json')).hexdigest()==card['independent_quality_source_review']['sha256']==m['quality_source_review_sha256']
 assert hashlib.sha256(z.read('evaluation-source-review.json')).hexdigest()==card['independent_evaluation_source_review']['sha256']==m['evaluation_source_review_sha256']
 combined=json.loads(z.read('scientific-source-review.json'));qr=json.loads(z.read('quality-source-review.json'));er=json.loads(z.read('evaluation-source-review.json'))
 assert combined['kind']=='cpb_v14_complete_104_owned_source_root_closure_v1' and combined['status']=='passed' and len(combined['reviewed_owned_files'])==104
 assert qr['kind']=='cpb_v14_independent_quality_auditor_source_review_v1' and qr['status']=='passed' and qr['independent_static_source_review'] is True and len(qr['reviewed_owned_files'])==4
 assert er['kind']=='cpb_v14_training_evaluation_and_training_auditor_independent_static_review_v1' and er['status']=='passed' and er['independent_of_author'] is True and len(er['reviewed_owned_files'])==32
 for authority in (card['independent_quality_source_review'],card['independent_evaluation_source_review']):
  refs=[p for p in combined['independent_reviews'].values() if p['path']==authority['path']]
  assert len(refs)==1 and all(refs[0][k]==authority[k] for k in ('bytes','sha256'))
 source=root/'source';source.mkdir()
 for e in m['files']:
  assert pathlib.PurePosixPath(e['file']).name==e['file']
  data=z.read(e['file']);assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
  (source/e['file']).write_bytes(data)
 (source/'SOURCE_MANIFEST.json').write_bytes(raw_manifest)
 python_files=sorted(source.glob('*.py'))
 assert [p.name for p in python_files]==sorted(card['staging_transport']['python_AST_files']) and len(python_files)==card['staging_transport']['python_AST_count']
 for p in python_files:
  tree=ast.parse(p.read_text(),filename=str(p))
  forbidden={'torch','numpy','subprocess','ctypes','importlib'}
  for node in ast.walk(tree):
   if isinstance(node,(ast.Import,ast.ImportFrom)):
    modules=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [(node.module or '').split('.')[0]]
    assert not(set(modules)&(forbidden-({'subprocess'} if p.name=='execute-audit.py' else set())))
 receipt={'kind':'cpb_v14_'+kind+'_audit_source_stage_v1','status':'passed','stage':'AST_only_no_numeric','files':len(names),'ZIP_members':len(names)+1,'python_AST_files':len(python_files),'source_root':str(source),'source_manifest_sha256':digest(source/'SOURCE_MANIFEST.json'),'source_zip_sha256':digest(root/'source.zip'),'audit_card_sha256':expected_card,'actual_card_peer_review':m['actual_card_peer_review'],'workflow_source_review_sha256':expected_workflow,'scientific_source_review_sha256':card['independent_source_review']['sha256'],'quality_source_review_sha256':card['independent_quality_source_review']['sha256'],'evaluation_source_review_sha256':card['independent_evaluation_source_review']['sha256'],'numeric_archive_reads':0,'model_calls':0,'head_fits':0,'training_updates':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 with (root/'source-stage.json').open('x') as output:output.write(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
'@
docker.exe exec $containerName python3 -c $stage $containerRoot $Kind $zipPin.sha256 $manifestPin.sha256 $AuditCardSHA256 $WorkflowSourceReviewSHA256 2>&1|Tee-Object -LiteralPath (Join-Path $hostRoot 'source-stage.log')
if($LASTEXITCODE-ne0){throw 'Source/AST admission failed; preserve every attempt'}
docker.exe cp "${containerName}:${containerRoot}/source-stage.json" (Join-Path $hostRoot 'source-stage.json')
if($LASTEXITCODE-ne0){throw 'Source-stage receipt mirror failed'}
Write-Output "Staged AST-only $Kind audit at $containerRoot; numerical execution remains separately gated."

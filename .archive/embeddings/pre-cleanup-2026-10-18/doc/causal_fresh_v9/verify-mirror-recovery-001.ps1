# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# External full physical copy of frozen V9 verify-mirror.ps1 (74777724...).
# Only explicitly declared report-only recovery can admit the failed quality wrapper.
# Host orchestration and byte inspection only; no project/model execution.
param(
    [ValidateRange(1,999)][int]$Attempt=1,
    [ValidateSet('Complete')][string]$Scope='Complete',
    [Parameter(Mandatory=$true)][string]$CardSHA256,
    [Parameter(Mandatory=$true)][string]$ManifestSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$RecoveryCardSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$RecoveryReceiptSHA256,
    [Parameter(Mandatory=$true)][string]$RecoveryReceiptContainerPath,
    [switch]$CopyMissing
)
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$label='attempt-{0:d3}' -f $Attempt
$receiptPath=Join-Path $PSScriptRoot ($Scope.ToLowerInvariant()+'-mirror-'+$label+'.json')
if(Test-Path -LiteralPath $receiptPath){throw 'Preserve the existing mirror receipt; do not rerun into its path'}
function Recovery-Pin([string]$relative){
    $file=Get-Item -LiteralPath (Join-Path $researchRoot $relative)
    if($file.PSIsContainer-or($file.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Physical recovery metadata required'}
    [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$recoveryCardPin=Recovery-Pin 'doc/causal_fresh_v9/reporting-recovery-card-001.json'
$recoveryReceiptPin=Recovery-Pin 'doc/causal_fresh_v9/reporting-recovery-receipt-001.json'
if($recoveryCardPin.sha256-ne$RecoveryCardSHA256-or$recoveryReceiptPin.sha256-ne$RecoveryReceiptSHA256){throw 'Externally reviewed actual reporting recovery card/receipt required before parse'}
$recoveryCard=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $recoveryCardPin.path)|ConvertFrom-Json
$recoveryReceipt=Get-Content -Raw -LiteralPath (Join-Path $researchRoot $recoveryReceiptPin.path)|ConvertFrom-Json
if($recoveryCard.kind-ne'cpb_v9_report_only_recovery_card_v1'-or$recoveryCard.root_reviewed-ne$true-or$recoveryReceipt.kind-ne'cpb_v9_report_only_recovery_receipt_v1'-or$recoveryReceipt.status-ne'passed'-or$recoveryReceipt.card.bytes-ne$recoveryCardPin.bytes-or$recoveryReceipt.card.sha256-ne$RecoveryCardSHA256){throw 'Explicit separately reviewed and actually passed report-only recovery required'}
$failureMirrorPin=Recovery-Pin 'doc/causal_fresh_v9/failure-mirror-before-report-recovery-001.json'
if($failureMirrorPin.sha256-ne'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532'-or$recoveryCard.failure_mirror.bytes-ne$failureMirrorPin.bytes-or$recoveryCard.failure_mirror.sha256-ne$failureMirrorPin.sha256){throw 'Exact pre-recovery failure preservation receipt required'}
if($RecoveryReceiptContainerPath-notlike'/opt/cuwacunu/research/causal-fresh-v9-report-recovery/attempt-001/*'-or$RecoveryReceiptContainerPath.Contains('..')){throw 'Exact exclusive recovery attempt receipt required'}
$expectedContainer='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$actualContainer=docker.exe inspect cuwacunu --format '{{.Id}} {{.State.Status}}'
if($LASTEXITCODE -ne 0 -or $actualContainer -ne ($expectedContainer+' running')){throw 'Existing managed cuwacunu identity/running guard failed'}
$inventoryCode=@'
import hashlib,json,pathlib,sys
label,scope,expected_card,expected_manifest,recovery_path,expected_recovery_card,expected_recovery_receipt=sys.argv[1:]
root=pathlib.Path('/opt/cuwacunu/research/causal-fresh-v9')/label
host='output/engineering/causal-fresh-v9/'+label
manifest_bytes=(root/'source/CAPSULE_MANIFEST.json').read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest()!=expected_manifest: raise RuntimeError('Externally reviewed capsule manifest changed')
manifest=json.loads(manifest_bytes.decode('utf-8-sig'))
if manifest['card_sha256']!=expected_card: raise RuntimeError('Externally reviewed scientific card changed')
def read(name):
    return json.loads((root/name).read_text())
for name in ('source-stage.json','build-stage.json','build-admission.json','variance-gradient-stage.json','variance-gradient-proof.json','temporal-pooling-stage.json','temporal-pooling-proof.json','engineering-stage.json','engineering-proof/proof.json'):
    if read(name)['status']!='passed': raise RuntimeError('Incomplete evidence: '+name)
stage=read('source-stage.json')
expected_count=read('build-admission.json')['source_files']
if stage['source_files']!=expected_count or len(manifest['files'])!=expected_count: raise RuntimeError('Expected exact admitted source capsule')
for pin in manifest['files']:
    path=root/'source'/pin['path']
    if path.is_symlink() or not path.is_file(): raise RuntimeError('Missing/linked source: '+str(path))
    data=path.read_bytes()
    if len(data)!=pin['bytes'] or hashlib.sha256(data).hexdigest()!=pin['sha256']: raise RuntimeError('Source hash mismatch: '+str(path))
fixed=('source-stage.json','build.log','build-stage.json','build-admission.json','variance-gradient.log','variance-gradient-stage.json','variance-gradient-proof.json','temporal-pooling.log','temporal-pooling-stage.json','temporal-pooling-proof.json','engineering.log','engineering-stage.json')
mapping={'engineering-proof':host+'/engineering-proof'}
if scope=='Training':
    if read('training-stage.json')['status']!='passed': raise RuntimeError('Incomplete main trajectory')
    if read('run-001/training-summary.json')['new_causal_updates']!=2048: raise RuntimeError('Expected completed2048 trajectory')
    fixed+=('training-stage.json','training.log')
    for subtree in ('calibration','checkpoints','witnesses'):
        mapping['run-001/'+subtree]='output/runs/causal-fresh-v9/run-001/'+subtree
if scope=='Complete':
    for name in ('training-stage.json','initial-saved-proof-stage.json','final-saved-proof-stage.json'):
        if read(name)['status']!='passed': raise RuntimeError('Incomplete main trajectory/proof: '+name)
    recovery_root=pathlib.Path('/opt/cuwacunu/research/causal-fresh-v9-report-recovery/attempt-001')
    recovery_path=pathlib.Path(recovery_path)
    if not recovery_path.resolve(strict=True).is_relative_to(recovery_root): raise RuntimeError('Recovery receipt escaped declared attempt')
    raw=recovery_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_recovery_receipt: raise RuntimeError('External recovery receipt hash before parse')
    recovery=json.loads(raw)
    recovery_card_path=pathlib.Path(recovery['card']['path'])
    if not recovery_card_path.resolve(strict=True).is_relative_to(recovery_root): raise RuntimeError('Recovery card escaped declared attempt')
    card_raw=recovery_card_path.read_bytes()
    if len(card_raw)!=recovery['card']['bytes'] or hashlib.sha256(card_raw).hexdigest()!=expected_recovery_card or recovery['card']['sha256']!=expected_recovery_card: raise RuntimeError('External recovery card hash before parse')
    recovery_card=json.loads(card_raw)
    if recovery['kind']!='cpb_v9_report_only_recovery_receipt_v1' or recovery['status']!='passed' or recovery_card['kind']!='cpb_v9_report_only_recovery_card_v1' or recovery_card['root_reviewed'] is not True: raise RuntimeError('Separate report-only recovery must pass')
    identity=recovery_card['main_source_identity']
    if identity!={'card_sha256':expected_card,'manifest_sha256':expected_manifest,'archive_sha256':stage['archive_sha256'],'source_files':expected_count,'binary_count':7,'translation_units':11}: raise RuntimeError('Recovery must bind unchanged admitted scientific source')
    for key in ('encoder_calls','head_fits','training_updates','science_stages_replayed'):
        if recovery[key]!=0: raise RuntimeError('Report recovery cannot replay scientific work: '+key)
    if recovery['scientific_inputs_before']!=recovery['scientific_inputs_after']: raise RuntimeError('All completed scientific artifacts must remain exact')
    def verify_recovery_pin(binding,expected_path):
        actual=pathlib.Path(expected_path)
        if actual.is_symlink() or not actual.is_file() or binding['path']!=str(actual): raise RuntimeError('Recovery evidence path mismatch')
        data=actual.read_bytes()
        if len(data)!=binding['bytes'] or hashlib.sha256(data).hexdigest()!=binding['sha256']: raise RuntimeError('Recovery evidence changed: '+str(actual))
    verify_recovery_pin(recovery['failed_quality_stage'],root/'quality-stage.json')
    verify_recovery_pin(recovery['failed_report_stage'],root/'quality-001/report-stage.json')
    if read('quality-stage.json')['status']!='failed' or read('quality-stage.json')['exit_code']==0 or read('quality-001/report-stage.json')['exit_code']==0: raise RuntimeError('Preserve original failed quality/report stages without relabelling')
    scientific_names=('extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads')
    if set(recovery['scientific_stages'])!=set(scientific_names): raise RuntimeError('Exactly five completed scientific stages')
    if recovery_card['scientific_stages']!=recovery['scientific_stages']: raise RuntimeError('Recovery declaration binds original completed stages')
    for name in scientific_names:
        verify_recovery_pin(recovery['scientific_stages'][name],root/'quality-001'/(name+'-stage.json'))
        if read('quality-001/'+name+'-stage.json')['exit_code']!=0: raise RuntimeError('A scientific stage did not pass: '+name)
    if read('quality-001/quality-receipt.json')['status']!='passed': raise RuntimeError('Incomplete quality')
    fixed=tuple(path.name for path in sorted(root.iterdir()) if path.is_file())
    mapping.update({'run-001':'output/runs/causal-fresh-v9/run-001','quality-001':'output/runs/causal-fresh-v9/run-001/quality-001','initial-proof':host+'/initial-proof','final-proof':host+'/final-proof'})
files=[]
def pin(path,destination):
    if path.is_symlink() or not path.is_file(): raise RuntimeError('Missing/linked evidence: '+str(path))
    data=path.read_bytes()
    files.append({'container_path':str(path),'path':destination,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
for name in fixed: pin(root/name,host+'/'+name)
if scope=='Training':
    # Only known completed main-run root files; never recurse through run-001.
    for name in ('training.csv','fresh-initialization.json','input-bindings.json','parameter-learning.json','query-initial.json','query-final.json','training-summary.json','typed-resume-512.json','optimization-component-witnesses.json','temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv','temporal-between-bin-position-learning.csv','temporal-projection-learning.json'):
        pin(root/'run-001'/name,'output/runs/causal-fresh-v9/run-001/'+name)
for subtree,destination in mapping.items():
    origin=root/subtree
    if not origin.is_dir(): raise RuntimeError('Missing completed subtree: '+subtree)
    for path in sorted(origin.rglob('*')):
        if scope=='Training' and any(part in ('initial-proof','final-proof','quality-001') for part in path.relative_to(root).parts):
            raise RuntimeError('Training mirror must exclude saved-proof/quality paths')
        if path.is_symlink(): raise RuntimeError('Unexpected evidence link')
        if path.is_file(): pin(path,destination+'/'+path.relative_to(origin).as_posix())
if scope=='Complete':
    for path in sorted(recovery_root.rglob('*')):
        if path.is_symlink(): raise RuntimeError('Unexpected recovery evidence link')
        if path.is_file(): pin(path,'output/engineering/causal-fresh-v9-report-recovery/attempt-001/'+path.relative_to(recovery_root).as_posix())
print(json.dumps({'source_files_verified':expected_count,'source_stage':stage,'files':files,
                 'report_only_recovery':{'kind':'cpb_v9_report_only_recovery_admission_v1','status':'passed',
                 'card_sha256':expected_recovery_card,'receipt_sha256':expected_recovery_receipt,
                 'original_quality_stage_status':'failed','original_report_stage_exit_nonzero':True,
                 'completed_scientific_stages':5,'science_stages_replayed':0,
                 'scientific_inputs_before_after_exact':True,'frozen_scientific_source_unchanged':True}}))
'@
$inventoryText=docker.exe exec cuwacunu python3 -c $inventoryCode $label $Scope $CardSHA256 $ManifestSHA256 $RecoveryReceiptContainerPath $RecoveryCardSHA256 $RecoveryReceiptSHA256
if($LASTEXITCODE -ne 0){throw 'Container byte-only inventory failed'}
$inventoryResult=$inventoryText|ConvertFrom-Json
$inventory=@($inventoryResult.files)
$totalBytes=[int64]0
foreach($pin in $inventory){
    $actualPath=[IO.Path]::GetFullPath((Join-Path $researchRoot $pin.path))
    $allowed=[IO.Path]::GetFullPath((Join-Path $researchRoot 'output'))+[IO.Path]::DirectorySeparatorChar
    if(-not $actualPath.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Mirror path escaped research output'}
    if(-not(Test-Path -LiteralPath $actualPath)){
        if(-not $CopyMissing){throw "Missing mirror: $($pin.path)"}
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($actualPath))|Out-Null
        docker.exe cp ('cuwacunu:'+$pin.container_path) $actualPath
        if($LASTEXITCODE -ne 0){throw "Copy failed: $($pin.path)"}
    }
    $file=Get-Item -LiteralPath $actualPath
    if($file.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Unexpected host link'}
    $hash=(Get-FileHash -LiteralPath $actualPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if($file.Length -ne $pin.bytes -or $hash -ne $pin.sha256){throw "Refuse mismatched existing destination: $($pin.path)"}
    $totalBytes+=$file.Length
}
$tensorPaths=@($inventory|Where-Object {$_.path.EndsWith('.pt')}|ForEach-Object {$_.path})
$ignored=@()
$safeResearchRoot=$researchRoot.Replace([char]92,[char]47)
for($start=0;$start-lt$tensorPaths.Count;$start+=128){
    $end=[Math]::Min($start+127,$tensorPaths.Count-1)
    $batch=@($tensorPaths[$start..$end])
    $ignored+=@(git -c "safe.directory=$safeResearchRoot" -C $researchRoot check-ignore -- @batch)
    if($LASTEXITCODE-ne0){throw 'Bounded ignored-PT verification failed'}
}
if($ignored.Count -ne $tensorPaths.Count -or (Compare-Object @($ignored|Sort-Object -Unique) @($tensorPaths|Sort-Object -Unique))){throw 'Every mirrored PT artifact must remain Git ignored'}
$packages=@()
foreach($old in @(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'output/engineering/causal-fresh-v9') -Directory|Where-Object {$_.Name -match '^attempt-[0-9]{3}$'}|ForEach-Object Name|Sort-Object)){
    foreach($name in @('source.zip','source-manifest.json','package.json')){
        $relative='output/engineering/causal-fresh-v9/'+$old+'/'+$name
        $path=Join-Path $researchRoot $relative
        if(-not(Test-Path -LiteralPath $path)){continue}
        $file=Get-Item -LiteralPath $path
        $packages += [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    }
}
$receipt=[ordered]@{
    kind='causal_fresh_v9_container_to_local_evidence_mirror_v1';status='passed';scope=$Scope;attempt=$label
    verified_utc=[DateTime]::UtcNow.ToString('o');source_files_verified=$inventoryResult.source_files_verified
    card_sha256=$CardSHA256;source_manifest_sha256=$ManifestSHA256
    source_stage=$inventoryResult.source_stage;file_count=$inventory.Count;bytes=$totalBytes;files=$inventory
    preserved_source_packages=$packages;prior_package_attempts=@($packages|ForEach-Object {($_.path -split '/')[3]}|Where-Object {$_ -ne $label}|Select-Object -Unique)
    all_mirrored_pt_ignored=$true;ignored_pt_count=$tensorPaths.Count
    report_only_recovery=$inventoryResult.report_only_recovery
    reporting_recovery_card=$recoveryCardPin;reporting_recovery_receipt=$recoveryReceiptPin
    pre_recovery_failure_mirror=$failureMirrorPin
    container='cuwacunu';container_id=$expectedContainer;container_copies_preserved=$true
    main_run_read=($Scope -ne 'Engineering');model_calls=0;head_fits=0;training_updates=0;deletions=0
    parent_helper=[ordered]@{path='doc/causal_fresh_v9/verify-mirror.ps1';bytes=10009;sha256='74777724d77b8033b349e838f7375959c66b0339b23fa3626f114d2f5bd3ddef'}
    helper=[ordered]@{path='doc/causal_fresh_v9/verify-mirror-recovery-001.ps1';bytes=(Get-Item -LiteralPath $PSCommandPath).Length;sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$json=($receipt|ConvertTo-Json -Depth 20).Replace([string][char]13,'')+[Environment]::NewLine
[IO.File]::WriteAllText($receiptPath,$json,[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';scope=$Scope;files=$inventory.Count;bytes=$totalBytes;ignored_pt=$tensorPaths.Count;receipt=$receiptPath}|ConvertTo-Json

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Additive full copy of the V6 mirror helper for declared saved-proof recovery002.
# Original V6 helper and its V5 ancestry remain preserved byte-exactly.
# Host orchestration and byte inspection only; no project/model execution.
param(
    [ValidateRange(1,999)][int]$Attempt=1,
    [ValidateSet('Engineering','Training','Complete')][string]$Scope='Engineering',
    [switch]$CopyMissing
)
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$label='attempt-{0:d3}' -f $Attempt
$receiptPath=Join-Path $PSScriptRoot ($Scope.ToLowerInvariant()+'-mirror-proof-recovery-002-'+$label+'.json')
if(Test-Path -LiteralPath $receiptPath){throw 'Preserve the existing mirror receipt; do not rerun into its path'}
$originalHelperPath=Join-Path $PSScriptRoot 'verify-mirror.ps1'
$originalHelperHash='7d2dc9eb971828b3839224107b0b570855492cc2440746ac00a819d913b3b10c'
if((Get-FileHash -LiteralPath $originalHelperPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $originalHelperHash){throw 'Original V6 helper changed'}
$recoveryPath=Join-Path $PSScriptRoot 'saved-proof-recovery-attempt002.json'
$recoveryHash='b9e790fa8d78b8ec3f4b0ffdad361a94d9147fe9f533619ec7c1c287a01dee99'
if((Get-FileHash -LiteralPath $recoveryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $recoveryHash){throw 'Declared recovery bytes changed'}
$recovery=Get-Content -LiteralPath $recoveryPath -Raw|ConvertFrom-Json
if($Attempt -ne 1 -or $recovery.kind -ne 'causal_fresh_v6_saved_proof_invocation_recovery_v1' -or
   $recovery.corrected_stage -ne 'final-saved-proof-attempt002' -or $recovery.model_source_or_weights_changed -ne $false){throw 'Only the declared attempt001 proof-path recovery is admitted'}
foreach($evidence in $recovery.evidence){
    $path=[IO.Path]::GetFullPath((Join-Path $researchRoot $evidence.path))
    if(-not $path.StartsWith(($researchRoot+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)){throw 'Recovery evidence escaped research checkout'}
    $file=Get-Item -LiteralPath $path
    if($file.Length -ne $evidence.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $evidence.sha256){throw 'Preserved failure/checkpoint/source evidence changed'}
}
$recoveryBase64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($recoveryPath))
$expectedContainer='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$actualContainer=docker inspect cuwacunu --format '{{.Id}} {{.State.Status}}'
if($LASTEXITCODE -ne 0 -or $actualContainer -ne ($expectedContainer+' running')){throw 'Existing managed cuwacunu identity/running guard failed'}
$inventoryCode=@'
import base64,hashlib,json,pathlib,sys
label,scope,recovery_base64,recovery_hash=sys.argv[1:]
recovery_bytes=base64.b64decode(recovery_base64)
if hashlib.sha256(recovery_bytes).hexdigest()!=recovery_hash: raise RuntimeError('Recovery declaration transport changed')
recovery=json.loads(recovery_bytes)
root=pathlib.Path('/opt/cuwacunu/research/causal-fresh-v6')/label
host='output/engineering/causal-fresh-v6/'+label
def read(name):
    return json.loads((root/name).read_text())
for name in ('source-stage.json','build-stage.json','build-admission.json','variance-gradient-stage.json','variance-gradient-proof.json','engineering-stage.json','engineering-proof/proof.json'):
    if read(name)['status']!='passed': raise RuntimeError('Incomplete evidence: '+name)
stage=read('source-stage.json')
manifest=read('source/CAPSULE_MANIFEST.json')
if manifest['card_sha256']!=recovery['card_sha256']: raise RuntimeError('Recovery must bind the unchanged prospective card')
expected_count=read('build-admission.json')['source_files']
if stage['source_files']!=expected_count or len(manifest['files'])!=expected_count: raise RuntimeError('Expected exact admitted source capsule')
for pin in manifest['files']:
    path=root/'source'/pin['path']
    if path.is_symlink() or not path.is_file(): raise RuntimeError('Missing/linked source: '+str(path))
    data=path.read_bytes()
    if len(data)!=pin['bytes'] or hashlib.sha256(data).hexdigest()!=pin['sha256']: raise RuntimeError('Source hash mismatch: '+str(path))
fixed=('source-stage.json','build.log','build-stage.json','build-admission.json','variance-gradient.log','variance-gradient-stage.json','variance-gradient-proof.json','engineering.log','engineering-stage.json')
mapping={'engineering-proof':host+'/engineering-proof'}
if scope=='Training':
    if read('training-stage.json')['status']!='passed': raise RuntimeError('Incomplete main trajectory')
    if read('run-001/training-summary.json')['new_causal_updates']!=2048: raise RuntimeError('Expected completed2048 trajectory')
    fixed+=('training-stage.json','training.log')
    for subtree in ('calibration','checkpoints','witnesses'):
        mapping['run-001/'+subtree]='output/runs/causal-fresh-v6/run-001/'+subtree
if scope=='Complete':
    for name in ('training-stage.json','initial-saved-proof-stage.json','final-saved-proof-attempt002-stage.json'):
        if read(name)['status']!='passed': raise RuntimeError('Incomplete main trajectory/proof: '+name)
    if read('final-saved-proof-stage.json')['status']!='failed': raise RuntimeError('Preserve the original failed invocation')
    corrected=read('final-saved-proof-attempt002-stage.json')
    if corrected['argv']!=recovery['corrected_argv'] or corrected['binary']['sha256']!=recovery['unchanged_binary_sha256']:
        raise RuntimeError('Corrected proof must use the declared same binary/settings/checkpoint/output')
    binary=pathlib.Path(corrected['binary']['path'])
    if str(binary)!=corrected['argv'][0] or not binary.is_file() or hashlib.sha256(binary.read_bytes()).hexdigest()!=recovery['unchanged_binary_sha256']:
        raise RuntimeError('Unchanged corrected-proof binary unavailable or different')
    expected_settings=root/'source/code/encoders/causal_patch_bottleneck_mae/versions/v6/config/v6.conf'
    if corrected['argv'][2]!=str(expected_settings) or corrected['argv'][4]!=str(root/'final-proof-attempt002'):
        raise RuntimeError('Corrected immutable settings/output path differs')
    final=read('final-proof-attempt002/proof.json')
    checkpoint=next(pin for pin in recovery['evidence'] if pin['path'].endswith('/checkpoints/inference-2048.pt'))
    if final['status']!='passed' or final['checkpoint']!=corrected['argv'][6] or final['checkpoint_sha256']!=checkpoint['sha256'] or final['causal_updates']!=2048:
        raise RuntimeError('Recovered proof must bind the unchanged actual final2048 checkpoint')
    if read('quality-001/quality-receipt.json')['status']!='passed': raise RuntimeError('Incomplete quality')
    fixed=tuple(path.name for path in sorted(root.iterdir()) if path.is_file())
    mapping.update({'run-001':'output/runs/causal-fresh-v6/run-001','quality-001':'output/runs/causal-fresh-v6/run-001/quality-001','initial-proof':host+'/initial-proof','final-proof-attempt002':host+'/final-proof-attempt002'})
files=[]
def pin(path,destination):
    if path.is_symlink() or not path.is_file(): raise RuntimeError('Missing/linked evidence: '+str(path))
    data=path.read_bytes()
    files.append({'container_path':str(path),'path':destination,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
for name in fixed: pin(root/name,host+'/'+name)
if scope=='Training':
    # Only known completed main-run root files; never recurse through run-001.
    for name in ('training.csv','fresh-initialization.json','input-bindings.json','parameter-learning.json','query-initial.json','query-final.json','training-summary.json','typed-resume-512.json','optimization-component-witnesses.json'):
        pin(root/'run-001'/name,'output/runs/causal-fresh-v6/run-001/'+name)
for subtree,destination in mapping.items():
    origin=root/subtree
    if not origin.is_dir(): raise RuntimeError('Missing completed subtree: '+subtree)
    for path in sorted(origin.rglob('*')):
        if scope=='Training' and any(part in ('initial-proof','final-proof','final-proof-attempt002','quality-001') for part in path.relative_to(root).parts):
            raise RuntimeError('Training mirror must exclude saved-proof/quality paths')
        if path.is_symlink(): raise RuntimeError('Unexpected evidence link')
        if path.is_file(): pin(path,destination+'/'+path.relative_to(origin).as_posix())
print(json.dumps({'source_files_verified':expected_count,'source_stage':stage,'files':files}))
'@
$inventoryText=docker exec cuwacunu python3 -c $inventoryCode $label $Scope $recoveryBase64 $recoveryHash
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
        docker cp ('cuwacunu:'+$pin.container_path) $actualPath
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
foreach($old in @(Get-ChildItem -LiteralPath (Join-Path $researchRoot 'output/engineering/causal-fresh-v6') -Directory|Where-Object {$_.Name -match '^attempt-[0-9]{3}$'}|ForEach-Object Name|Sort-Object)){
    foreach($name in @('source.zip','source-manifest.json','package.json')){
        $relative='output/engineering/causal-fresh-v6/'+$old+'/'+$name
        $path=Join-Path $researchRoot $relative
        if(-not(Test-Path -LiteralPath $path)){continue}
        $file=Get-Item -LiteralPath $path
        $packages += [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    }
}
$receipt=[ordered]@{
    kind='causal_fresh_v6_proof_recovery_container_to_local_evidence_mirror_v1';status='passed';scope=$Scope;attempt=$label
    verified_utc=[DateTime]::UtcNow.ToString('o');source_files_verified=$inventoryResult.source_files_verified
    source_stage=$inventoryResult.source_stage;file_count=$inventory.Count;bytes=$totalBytes;files=$inventory
    preserved_source_packages=$packages;prior_package_attempts=@($packages|ForEach-Object {($_.path -split '/')[3]}|Where-Object {$_ -ne $label}|Select-Object -Unique)
    all_mirrored_pt_ignored=$true;ignored_pt_count=$tensorPaths.Count
    container='cuwacunu';container_id=$expectedContainer;container_copies_preserved=$true
    main_run_read=($Scope -ne 'Engineering');model_calls=0;head_fits=0;training_updates=0;deletions=0
    parent_helper=[ordered]@{path='doc/causal_fresh_v5/verify-mirror.ps1';bytes=8563;sha256='5ffe17b5f9a5d8f1999b0bf7140158d676fc017884b1cc41e8deba6ae1ece0e3'}
    direct_parent_helper=[ordered]@{path='doc/causal_fresh_v6/verify-mirror.ps1';bytes=9113;sha256=$originalHelperHash}
    recovery_declaration=[ordered]@{path='doc/causal_fresh_v6/saved-proof-recovery-attempt002.json';bytes=2151;sha256=$recoveryHash}
    original_failed_stage_and_log_preserved=$true;corrected_final_proof_subtree='final-proof-attempt002'
    helper=[ordered]@{path='doc/causal_fresh_v6/verify-mirror-proof-recovery-002.ps1';bytes=(Get-Item -LiteralPath $PSCommandPath).Length;sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$json=($receipt|ConvertTo-Json -Depth 20).Replace([string][char]13,'')+[Environment]::NewLine
[IO.File]::WriteAllText($receiptPath,$json,[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';scope=$Scope;files=$inventory.Count;bytes=$totalBytes;ignored_pt=$tensorPaths.Count;receipt=$receiptPath}|ConvertTo-Json

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Owned adaptation of ../causal_fresh_v4/verify-mirror.ps1.
# Host orchestration and byte inspection only; no project/model execution.
param(
    [ValidateRange(1,999)][int]$Attempt=3,
    [ValidateSet('Engineering','Complete')][string]$Scope='Engineering',
    [switch]$CopyMissing
)
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$label='attempt-{0:d3}' -f $Attempt
$receiptPath=Join-Path $PSScriptRoot ($Scope.ToLowerInvariant()+'-mirror-'+$label+'.json')
if(Test-Path -LiteralPath $receiptPath){throw 'Preserve the existing mirror receipt; do not rerun into its path'}
$expectedContainer='82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'
$actualContainer=docker inspect cuwacunu --format '{{.Id}} {{.State.Status}}'
if($LASTEXITCODE -ne 0 -or $actualContainer -ne ($expectedContainer+' running')){throw 'Existing managed cuwacunu identity/running guard failed'}
$inventoryCode=@'
import hashlib,json,pathlib,sys
label,scope=sys.argv[1:]
root=pathlib.Path('/opt/cuwacunu/research/causal-fresh-v5')/label
host='output/engineering/causal-fresh-v5/'+label
def read(name):
    return json.loads((root/name).read_text())
for name in ('source-stage.json','build-stage.json','build-admission.json','variance-gradient-stage.json','variance-gradient-proof.json','engineering-stage.json','engineering-proof/proof.json'):
    if read(name)['status']!='passed': raise RuntimeError('Incomplete evidence: '+name)
stage=read('source-stage.json')
manifest=read('source/CAPSULE_MANIFEST.json')
if stage['source_files']!=97 or len(manifest['files'])!=97: raise RuntimeError('Expected exact97 source capsule')
for pin in manifest['files']:
    path=root/'source'/pin['path']
    if path.is_symlink() or not path.is_file(): raise RuntimeError('Missing/linked source: '+str(path))
    data=path.read_bytes()
    if len(data)!=pin['bytes'] or hashlib.sha256(data).hexdigest()!=pin['sha256']: raise RuntimeError('Source hash mismatch: '+str(path))
fixed=('source-stage.json','build.log','build-stage.json','build-admission.json','variance-gradient.log','variance-gradient-stage.json','variance-gradient-proof.json','engineering.log','engineering-stage.json')
mapping={'engineering-proof':host+'/engineering-proof'}
if scope=='Complete':
    for name in ('training-stage.json','initial-proof-stage.json','final-proof-stage.json'):
        if read(name)['status']!='passed': raise RuntimeError('Incomplete main trajectory/proof: '+name)
    if read('quality-001/quality-receipt.json')['status']!='passed': raise RuntimeError('Incomplete quality')
    fixed=tuple(path.name for path in sorted(root.iterdir()) if path.is_file())
    mapping.update({'run-001':'output/runs/causal-fresh-v5/run-001','quality-001':'output/runs/causal-fresh-v5/run-001/quality-001','initial-proof':host+'/initial-proof','final-proof':host+'/final-proof'})
files=[]
def pin(path,destination):
    if path.is_symlink() or not path.is_file(): raise RuntimeError('Missing/linked evidence: '+str(path))
    data=path.read_bytes()
    files.append({'container_path':str(path),'path':destination,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
for name in fixed: pin(root/name,host+'/'+name)
for subtree,destination in mapping.items():
    origin=root/subtree
    if not origin.is_dir(): raise RuntimeError('Missing completed subtree: '+subtree)
    for path in sorted(origin.rglob('*')):
        if path.is_symlink(): raise RuntimeError('Unexpected evidence link')
        if path.is_file(): pin(path,destination+'/'+path.relative_to(origin).as_posix())
print(json.dumps({'source_files_verified':97,'source_stage':stage,'files':files}))
'@
$inventoryText=docker exec cuwacunu python3 -c $inventoryCode $label $Scope
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
$ignored=@($tensorPaths|git -C $researchRoot check-ignore --stdin)
if($LASTEXITCODE -ne 0 -or $ignored.Count -ne $tensorPaths.Count){throw 'Every mirrored PT artifact must remain Git ignored'}
$packages=@()
foreach($old in @('attempt-001',$label)|Select-Object -Unique){
    foreach($name in @('source.zip','source-manifest.json','package.json')){
        $relative='output/engineering/causal-fresh-v5/'+$old+'/'+$name
        $path=Join-Path $researchRoot $relative
        $file=Get-Item -LiteralPath $path
        $packages += [ordered]@{path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    }
}
$receipt=[ordered]@{
    kind='causal_fresh_v5_container_to_local_evidence_mirror_v1';status='passed';scope=$Scope;attempt=$label
    verified_utc=[DateTime]::UtcNow.ToString('o');source_files_verified=97
    source_stage=$inventoryResult.source_stage;file_count=$inventory.Count;bytes=$totalBytes;files=$inventory
    preserved_source_packages=$packages;failed_attempts_preserved=@('attempt-001','attempt-002')
    failed_attempt002_note='Preserved container-only empty directories/unpack helper and root failure receipt; no source ZIP or model execution existed'
    all_mirrored_pt_ignored=$true;ignored_pt_count=$tensorPaths.Count
    container='cuwacunu';container_id=$expectedContainer;container_copies_preserved=$true
    main_run_read=($Scope -eq 'Complete');model_calls=0;head_fits=0;training_updates=0;deletions=0
    helper=[ordered]@{path='doc/causal_fresh_v5/verify-mirror.ps1';bytes=(Get-Item -LiteralPath $PSCommandPath).Length;sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$json=($receipt|ConvertTo-Json -Depth 20).Replace([string][char]13,'')+[Environment]::NewLine
[IO.File]::WriteAllText($receiptPath,$json,[Text.UTF8Encoding]::new($false))
[ordered]@{status='passed';scope=$Scope;files=$inventory.Count;bytes=$totalBytes;ignored_pt=$tensorPaths.Count;receipt=$receiptPath}|ConvertTo-Json

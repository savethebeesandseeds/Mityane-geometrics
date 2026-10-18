# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host orchestration/file inspection only. Numeric project code runs in cuwacunu.
$ErrorActionPreference='Stop'
$researchRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$receiptPath=Join-Path $PSScriptRoot 'mirror-verification.json'
if(Test-Path -LiteralPath $receiptPath){throw 'Preserve the existing mirror receipt'}
$inventoryCode=@'
import hashlib,json,pathlib
root=pathlib.Path('/opt/cuwacunu/research/causal-fresh-v4/attempt-001')
mapping={
 'run-001':'output/runs/causal-fresh-v4/run-001',
 'quality-001':'output/runs/causal-fresh-v4/run-001/quality-001',
 'engineering-proof':'output/engineering/causal-fresh-v4/attempt-001/engineering-proof',
 'initial-proof':'output/engineering/causal-fresh-v4/attempt-001/initial-proof',
 'final-proof':'output/engineering/causal-fresh-v4/attempt-001/final-proof',
}
files=[]
for subtree,destination in mapping.items():
    origin=root/subtree
    for path in sorted(origin.rglob('*')):
        if path.is_symlink(): raise RuntimeError('Unexpected evidence link')
        if path.is_file():
            content=path.read_bytes()
            files.append({'container_path':str(path),'path':destination+'/'+path.relative_to(origin).as_posix(),
                          'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
print(json.dumps(files))
'@
$inventoryText=docker exec cuwacunu python3 -c $inventoryCode
if($LASTEXITCODE-ne0){throw 'Container mirror inventory failed'}
$inventory=$inventoryText|ConvertFrom-Json
$totalBytes=[int64]0
foreach($pin in $inventory){
    $actualPath=Join-Path $researchRoot $pin.path
    $file=Get-Item -LiteralPath $actualPath
    $hash=(Get-FileHash -LiteralPath $actualPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if($file.Length-ne$pin.bytes -or $hash-ne$pin.sha256){throw "Mirror mismatch: $($pin.path)"}
    $totalBytes+=$file.Length
}
$receipt=[ordered]@{
    kind='causal_fresh_v4_container_to_local_evidence_mirror_v1';status='passed'
    verified_utc=[DateTime]::UtcNow.ToString('o');files=$inventory
    file_count=$inventory.Count;bytes=$totalBytes
    groups=@('main_run','single_quality_run_including_original_reporting_failure','engineering_proof','initial_saved_proof','final_saved_proof')
    container='cuwacunu';model_calls=0;head_fits=0;training_updates=0
    container_copies_preserved=$true;local_checkpoints_ignored_expected=$true
}
$receipt|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $receiptPath -Encoding utf8
[ordered]@{status='passed';files=$inventory.Count;bytes=$totalBytes}|ConvertTo-Json

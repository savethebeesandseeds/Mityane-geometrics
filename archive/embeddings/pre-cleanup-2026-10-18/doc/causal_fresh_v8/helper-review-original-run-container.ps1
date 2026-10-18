# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
# Host orchestration only; all builds/tests/numerical work execute in cuwacunu.
param([Parameter(Mandatory=$true)][ValidateSet('Stage','Build','Fixtures','Engineering','Training','InitialProof','FinalProof','Quality')][string]$Phase,
 [Parameter(Mandatory=$true)][string]$CardSHA256,[ValidateRange(1,999)][int]$Attempt=1)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$label='attempt-{0:D3}' -f $Attempt
$containerRoot='/opt/cuwacunu/research/causal-fresh-v8/'+$label
$source=$containerRoot+'/source'
$build='/opt/cuwacunu/build/causal-fresh-v8-attempt'+('{0:D3}' -f $Attempt)
$settings=$source+'/code/encoders/causal_patch_bottleneck_mae/versions/v8/config/v8.conf'
$recorder=$source+'/doc/causal_fresh_v8/execute-stage.py'
function Docker([string[]]$Arguments){& docker @Arguments;if($LASTEXITCODE-ne0){throw ('Container command failed: '+$Arguments[0])}}
$state=(& docker inspect cuwacunu --format '{{json .}}')|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$state.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$state.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$state.State.Running-or$state.Path-ne'/bin/bash'-or$state.Mounts.Count-ne1-or$state.Mounts[0].Destination-ne'/cuwacunu'-or$state.Mounts[0].Source-ne'C:\Work\cuwacunu\cuwacunu_dev'-or$state.HostConfig.RestartPolicy.Name-ne'no'-or$state.HostConfig.DeviceRequests[0].Count-ne-1){throw 'Exact existing managed GPU container required'}
$localCard=Join-Path $root 'doc/CAUSAL_FRESH_V8_CARD.json'
if((Get-FileHash -LiteralPath $localCard).Hash.ToLowerInvariant()-ne$CardSHA256){throw 'Frozen host card changed'}
$package=Get-Content -LiteralPath (Join-Path $root ('output/engineering/causal-fresh-v8/'+$label+'/package.json')) -Raw|ConvertFrom-Json
$manifestPath=Join-Path $root ('output/engineering/causal-fresh-v8/'+$label+'/source-manifest.json')
if($package.card_sha256-ne$CardSHA256-or(Get-FileHash -LiteralPath $manifestPath).Hash.ToLowerInvariant()-ne$package.manifest_sha256){throw 'Exact frozen host manifest required'}
$manifest=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json
if($manifest.card_sha256-ne$CardSHA256){throw 'Manifest/card mismatch'}
foreach($name in @('run-container.ps1','unpack-source.py')){
 $entry=@($manifest.files|Where-Object { $_.path -eq ('doc/causal_fresh_v8/'+$name) })
 $file=Get-Item -LiteralPath (Join-Path $PSScriptRoot $name)
 if($entry.Count-ne1-or$file.Length-ne$entry[0].bytes-or(Get-FileHash -LiteralPath $file.FullName).Hash.ToLowerInvariant()-ne$entry[0].sha256){throw ('Changed admitted orchestration helper: '+$name)}
}
$unpackPin=@($manifest.files|Where-Object { $_.path -eq 'doc/causal_fresh_v8/unpack-source.py' })[0]
if($Phase-eq'Stage'){
 if($package.card_sha256-ne$CardSHA256-or(Get-FileHash -LiteralPath $package.zip_path).Hash.ToLowerInvariant()-ne$package.zip_sha256){throw 'Exact frozen capsule required'}
 $prepare=@'
from pathlib import Path
import sys
root,build=map(Path,sys.argv[1:])
if build.exists(): raise RuntimeError('Preserve existing build leaf')
root.parent.mkdir(parents=True,exist_ok=True)
root.mkdir(exist_ok=False)
'@
 Docker @('exec','cuwacunu','python3','-c',$prepare,$containerRoot,$build)
 Docker @('cp',$package.zip_path,('cuwacunu:'+$containerRoot+'/source.zip'))
 Docker @('cp',(Join-Path $PSScriptRoot 'unpack-source.py'),('cuwacunu:'+$containerRoot+'/unpack-source.py'))
 $verify=@'
import hashlib,sys
from pathlib import Path
for path,digest in zip(sys.argv[1::2],sys.argv[2::2]):
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise RuntimeError('Frozen staging bytes changed')
'@
 Docker @('exec','cuwacunu','python3','-c',$verify,($containerRoot+'/source.zip'),$package.zip_sha256,($containerRoot+'/unpack-source.py'),$unpackPin.sha256)
 Docker @('exec','cuwacunu','python3',($containerRoot+'/unpack-source.py'),($containerRoot+'/source.zip'),$source)
 exit
}
$verifySource=@'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2]
manifest=json.loads((root/'source/CAPSULE_MANIFEST.json').read_text(encoding='utf-8-sig'))
if manifest['card_sha256']!=expected: raise RuntimeError('Frozen container card differs')
for item in manifest['files']:
 p=root/'source'/item['path'];data=p.read_bytes()
 if p.is_symlink() or len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']: raise RuntimeError('Admitted source changed: '+str(p))
'@
Docker @('exec','cuwacunu','python3','-c',$verifySource,$containerRoot,$CardSHA256)
function Recorded([string]$Name,[string[]]$Command,[string]$ProofFlag=''){
 $argv=@('exec','-w',$source,'cuwacunu','bash','-lc')
 # All path/flag words below are fixed project-controlled tokens without spaces.
 $words=@('python3',$recorder,'--root',$containerRoot,'--name',$Name)
 if($ProofFlag){$words+= $ProofFlag}
 $words+= @('--')+$Command
 Docker ($argv+@('. /etc/profile.d/embedding.sh && '+($words -join ' ')))
}
switch($Phase){
 'Build'{
  Recorded 'build' @('/usr/bin/make','-f','code/research/causal_fresh_v8/Makefile',('BUILD_DIR='+$build),'-j2')
  Docker @('exec','-w',$source,'cuwacunu','python3',($source+'/doc/causal_fresh_v8/audit-build.py'),$source,$build,($containerRoot+'/build-admission.json'))
 }
 'Fixtures'{
  Recorded 'variance-gradient' @(($build+'/test_cpb_v8_variance_gradient'),'--settings',$settings) '--gradient-proof'
  Recorded 'temporal-pooling' @(($build+'/test_cpb_v8_temporal_pooling'),'--settings',$settings) '--temporal-proof'
 }
 'Engineering'{Recorded 'engineering' @(($build+'/test_causal_patch_bottleneck_mae_v8'),'--settings',$settings,'--output',($containerRoot+'/engineering-proof'),'--seed','943001')}
 'Training'{Recorded 'training' @(($build+'/causal_fresh_v8_train'),'--settings',$settings,'--shared','/cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1','--cache','/cuwacunu/data/mkt-hist-1/csv/BTCUSDT-1m-2024-01.cache.bin','--cache','/cuwacunu/data/mkt-hist-1/csv/ETHUSDT-1m-2024-01.cache.bin','--cache','/cuwacunu/data/mkt-hist-1/csv/SOLUSDT-1m-2024-01.cache.bin','--output',($containerRoot+'/run-001'))}
 'InitialProof'{Recorded 'initial-saved-proof' @(($build+'/test_causal_patch_bottleneck_mae_v8'),'--settings',$settings,'--output',($containerRoot+'/initial-proof'),'--saved-checkpoint',($containerRoot+'/run-001/checkpoints/inference-0.pt'))}
 'FinalProof'{Recorded 'final-saved-proof' @(($build+'/test_causal_patch_bottleneck_mae_v8'),'--settings',$settings,'--output',($containerRoot+'/final-proof'),'--saved-checkpoint',($containerRoot+'/run-001/checkpoints/inference-2048.pt'))}
 'Quality'{
  $words=@('/usr/bin/python3',($source+'/doc/causal_fresh_v8/run-quality.py'),'--build',$build,'--source',$source,'--training',($containerRoot+'/run-001'),'--engineering-proof',($containerRoot+'/engineering-proof/proof.json'),'--initial-proof',($containerRoot+'/initial-proof/proof.json'),'--final-proof',($containerRoot+'/final-proof/proof.json'),'--admission',($containerRoot+'/build-admission.json'),'--gradient-proof',($containerRoot+'/variance-gradient-proof.json'),'--temporal-proof',($containerRoot+'/temporal-pooling-proof.json'),'--data-root','/cuwacunu','--output',($containerRoot+'/quality-001'))
  Recorded 'quality' $words
 }
}

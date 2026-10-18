# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Host-only once recorder. Numerical audit remains inside the existing container.
param([Parameter(Mandatory=$true)][switch]$RunAdmitted,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$AuditCardSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$StageReceiptSha256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ManifestSHA256,
 [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ZipSHA256)
$ErrorActionPreference='Stop'
if(-not$RunAdmitted){throw 'Numerical Run requires explicit separate admission'}
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$hostRoot=Join-Path $repo 'output/engineering/causal-fresh-v13-training-audit/attempt-001'
$containerRoot='/opt/cuwacunu/research/causal-fresh-v13-training-audit/attempt-001'
function Pin([string]$p){$f=Get-Item -LiteralPath $p;if($f.PSIsContainer-or($f.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Physical file required'};[ordered]@{path=$f.FullName;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}}
$cardPath=Join-Path $PSScriptRoot 'audit-card.json';$stagePath=Join-Path $hostRoot 'source-stage.json';$manifestPath=Join-Path $hostRoot 'source-manifest.json';$zipPath=Join-Path $hostRoot 'source.zip'
foreach($entry in @(@{path=$cardPath;hash=$AuditCardSHA256},@{path=$stagePath;hash=$StageReceiptSha256},@{path=$manifestPath;hash=$ManifestSHA256},@{path=$zipPath;hash=$ZipSHA256})){if((Pin $entry.path).sha256-ne$entry.hash){throw 'External reviewed card/transport hash required before parse'}}
$card=Get-Content -Raw -LiteralPath $cardPath|ConvertFrom-Json
$stage=Get-Content -Raw -LiteralPath $stagePath|ConvertFrom-Json
$manifest=Get-Content -Raw -LiteralPath $manifestPath|ConvertFrom-Json
$self=Pin $PSCommandPath
if($self.bytes-ne$card.run_helper.bytes-or$self.sha256-ne$card.run_helper.sha256){throw 'Exact separately reviewed host Run helper required'}
if($card.stage-ne'declared_before_independent_saved_data_execution'-or-not$card.execution_pending-or$card.expected_audit_root-ne$containerRoot-or$card.model-ne'CPB-v13.market-01'){throw 'Exact prospective training audit declaration required'}
if($stage.status-ne'passed'-or$stage.card_reviewed-ne$true-or$stage.kind-ne'cpb_v13_training_audit_source_stage_v1'-or$stage.source_root-ne($containerRoot+'/source')-or$stage.audit_card_sha256-ne$AuditCardSHA256-or$stage.source_manifest_sha256-ne$ManifestSHA256-or$stage.source_zip_sha256-ne$ZipSHA256-or$stage.files-ne6-or$stage.python_AST_files-ne2-or$stage.numeric_archive_reads-ne0){throw 'Exact admitted AST-only Stage required'}
if($manifest.container_root-ne$containerRoot-or$manifest.audit_card_sha256-ne$AuditCardSHA256-or$manifest.files.Count-ne6){throw 'Exact Stage transport context required'}
$marker=Join-Path $hostRoot 'run-invocation.json'
if(Test-Path -LiteralPath $marker){throw 'Preserve prior Run invocation, including pre-body failures'}
$observed=docker.exe inspect cuwacunu|ConvertFrom-Json
if($LASTEXITCODE-ne0-or$observed.Id-ne'82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'-or$observed.Image-ne'sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7'-or-not$observed.State.Running){throw 'Exact existing running managed container required'}
$bootstrap=@'
import hashlib,os,pathlib,sys
wrapper=pathlib.Path(sys.argv[1]);expected=sys.argv[2]
assert not wrapper.is_symlink() and hashlib.sha256(wrapper.read_bytes()).hexdigest()==expected,'External wrapper hash before source execution'
os.execv(sys.executable,[sys.executable,str(wrapper)]+sys.argv[3:])
'@
$argv=@('exec','cuwacunu','python3','-c',$bootstrap,($containerRoot+'/source/execute-audit.py'),$card.execution_wrapper.sha256,'--root',$containerRoot,'--kind','training','--execute-once','--audit-card-sha256',$AuditCardSHA256,'--stage-receipt-sha256',$StageReceiptSha256,'--manifest-sha256',$ManifestSHA256,'--zip-sha256',$ZipSHA256)
$invocation=[ordered]@{kind='cpb_v13_training_audit_host_run_invocation_v1';stage='separately_admitted_once';created_utc=[DateTime]::UtcNow.ToString('o');native_executable='docker.exe';argv=$argv;card=(Pin $cardPath);source_stage=(Pin $stagePath);source_manifest=(Pin $manifestPath);source_zip=(Pin $zipPath);helper=$self;host_numeric_execution=$false;new_containers=0}
[IO.File]::WriteAllText($marker,($invocation|ConvertTo-Json -Depth 15)+[char]10,[Text.UTF8Encoding]::new($false))
$clock=[Diagnostics.Stopwatch]::StartNew()
docker.exe @argv 1> (Join-Path $hostRoot 'run.stdout.log') 2> (Join-Path $hostRoot 'run.stderr.log')
$exit=$LASTEXITCODE;$clock.Stop()
$receipt=[ordered]@{kind='cpb_v13_training_audit_host_run_receipt_v1';status=$(if($exit-eq0){'passed'}else{'failed'});exit_code=$exit;elapsed_seconds=$clock.Elapsed.TotalSeconds;invocation=(Pin $marker);stdout=(Pin (Join-Path $hostRoot 'run.stdout.log'));stderr=(Pin (Join-Path $hostRoot 'run.stderr.log'));helper=$self;no_replay=$true;host_numeric_execution=$false}
$receiptPath=Join-Path $hostRoot 'run-receipt.json'
if(Test-Path -LiteralPath $receiptPath){throw 'Preserve prior host Run receipt'}
[IO.File]::WriteAllText($receiptPath,($receipt|ConvertTo-Json -Depth 15)+[char]10,[Text.UTF8Encoding]::new($false))
$receipt|ConvertTo-Json -Depth 10
if($exit-ne0){throw 'Audit invocation failed; preserve source/logs/receipts, never retry implicitly'}

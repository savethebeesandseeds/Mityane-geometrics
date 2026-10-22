// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Source/executable admission precedes the artificial CUDA fixture.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../../../..');
if(process.platform!=='linux'||root!=='/mityane'||!fs.existsSync('/.dockerenv'))throw Error('Managed container required');
const base=path.join(root,'.temp/embeddings/runs/cpb-v17-balance-001');
const family='code/embeddings/encoders/causal_patch_bottleneck_mae';
const executable='/opt/mityane-geometrics/embeddings/build/cpb-v17-balance-001/test_cpb_v17';
const pin=name=>{const p=path.isAbsolute(name)?name:path.join(root,name);const b=fs.readFileSync(p);return {path:p,bytes:b.length,sha256:crypto.createHash('sha256').update(b).digest('hex')};};
const unchanged=p=>{const actual=pin(p.path);if(actual.bytes!==p.bytes||actual.sha256!==p.sha256)throw Error(`Bound bytes changed: ${p.path}`);};
const save=(name,value)=>fs.writeFileSync(path.join(base,name),JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const earlier=JSON.parse(fs.readFileSync(path.join(base,'engineering-source-admission-v2.json')));
const compatibility=JSON.parse(fs.readFileSync(path.join(root,family,'family/evidence/compatibility-2026-10-22.json')));
const frozen=compatibility.sourceFiles.filter(p=>p.path.startsWith(`${family}/versions/v16/`)||p.path.startsWith(`${family}/family/include/`));
if(frozen.length!==52)throw Error('Expected49 V16 files and3 family headers');
frozen.forEach(unchanged);
const names=[...earlier.sources.map(p=>p.path),...frozen.map(p=>p.path),'.gitattributes',path.relative(root,fileURLToPath(import.meta.url))];
const sources=[...new Set(names)].sort().map(pin),binary=pin(executable);
const output=path.join(base,'engineering-001');if(fs.existsSync(output))throw Error('Preserve previous engineering output');
save('engineering-admission-v3.json',{kind:'cpb_v17_engineering_admission_v3',admitted_utc:new Date().toISOString(),sources,executable:binary,
  scope:'Artificial control parity, candidate resume/export/refusals across two configurations; no market observations or quality scoring.',
  amendments:'Preserve initial compile refusal and earlier admissions. Split mixed-type auto declaration; prospectively declare actual128/resume256 study schedule before any study execution. Raw source bytes retained.'});
const argv=['--settings',`${family}/versions/v17/config/research.conf`,'--alternative-settings',`${family}/family/config/paired.conf`,'--output',output];
save('engineering-command.json',{program:binary,argv,started_utc:new Date().toISOString()});
const out=fs.openSync(path.join(base,'engineering.stdout.log'),'wx'),err=fs.openSync(path.join(base,'engineering.stderr.log'),'wx');
let result;try{result=spawnSync(executable,argv,{cwd:root,stdio:['ignore',out,err]});}finally{fs.closeSync(out);fs.closeSync(err);}
let failure=result.error?.message??(result.status===0?null:`exit=${result.status}`);
try{sources.forEach(unchanged);unchanged(binary);}catch(error){failure=error.message;}
save('engineering-execution.json',{kind:'cpb_v17_engineering_execution_v1',status:failure?'failed':'passed',finished_utc:new Date().toISOString(),exit_code:result.status,error:failure,
  source_admission:pin(path.join(base,'engineering-admission-v3.json')),stdout:pin(path.join(base,'engineering.stdout.log')),stderr:pin(path.join(base,'engineering.stderr.log'))});
if(failure)throw Error(`Engineering fixture failed: ${failure}; inspect preserved stderr`);
const report=JSON.parse(fs.readFileSync(path.join(output,'summary.json')));if(report.status!=='passed')throw Error('Passed engineering receipt required');
console.log(JSON.stringify(report));

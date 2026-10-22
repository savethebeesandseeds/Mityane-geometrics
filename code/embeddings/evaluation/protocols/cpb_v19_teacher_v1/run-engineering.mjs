// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Admit exact source/executable bytes before artificial EMA-teacher CUDA checks.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../../../..');
if(process.platform!=='linux'||root!=='/mityane'||!fs.existsSync('/.dockerenv'))throw Error('Managed container required');
const family='code/embeddings/encoders/causal_patch_bottleneck_mae';
const protocol='code/embeddings/evaluation/protocols/cpb_v19_teacher_v1';
const base=path.join(root,'.temp/embeddings/runs/cpb-v19-teacher-001');
const executable='/opt/mityane-geometrics/embeddings/build/cpb-v19-teacher-001/test_cpb_v19';
const absolute=name=>path.isAbsolute(name)?name:path.join(root,name);
const pin=name=>{const p=absolute(name),b=fs.readFileSync(p);return {path:p,bytes:b.length,sha256:crypto.createHash('sha256').update(b).digest('hex')};};
const unchanged=p=>{const actual=pin(p.path);if(actual.bytes!==p.bytes||actual.sha256!==p.sha256)throw Error(`Bound bytes changed: ${p.path}`);};
const list=name=>fs.readdirSync(absolute(name),{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name)).flatMap(e=>{
  if(e.isSymbolicLink())throw Error(`Source symlink refused: ${name}/${e.name}`);return e.isDirectory()?list(`${name}/${e.name}`):[`${name}/${e.name}`];});
fs.mkdirSync(base,{recursive:true});
const save=(name,value)=>fs.writeFileSync(path.join(base,name),JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const compatibility=JSON.parse(fs.readFileSync(absolute(`${family}/family/evidence/compatibility-2026-10-22.json`)));
const frozen=compatibility.sourceFiles.filter(p=>p.path.startsWith(`${family}/versions/v16/`)||p.path.startsWith(`${family}/family/include/`));
if(frozen.length!==52)throw Error('Expected 49 V16 files and 3 family headers');
frozen.forEach(unchanged);
const sources=[...new Set([
  ...frozen.map(p=>p.path),...list(`${family}/versions/v19/include`),...list(`${family}/versions/v19/config`),...list(`${family}/versions/v19/tests`),
  ...list('code/data/observations/include'),...list('code/shared/settings/include'),
  `${protocol}/PROTOCOL.md`,`${protocol}/CONTRACT.md`,`${protocol}/run-engineering.mjs`,
  `${family}/family/config/paired.conf`,`${family}/versions/v19/v19.mk`,'code/embeddings/scripts/cpb_v19_train.cpp','Makefile','.gitattributes',
  'dependencies.lock','environment-image.json'
])].sort().map(pin);
const binary=pin(executable),output=path.join(base,'engineering-001');
if(fs.existsSync(output))throw Error('Preserve prior engineering output');
save('engineering-admission.json',{kind:'cpb_v19_engineering_admission_v1',admitted_utc:new Date().toISOString(),sources,executable:binary,
  scope:'Artificial shared construction/masks, learned-coordinate auxiliary gradients, teacher EMA/fixed-state/cutoff behavior, full resume/inference and refusals. No market data or quality scoring.'});
const argv=['--settings',`${family}/versions/v19/config/research.conf`,'--alternative-settings',`${family}/family/config/paired.conf`,'--output',output];
save('engineering-command.json',{program:binary,argv,started_utc:new Date().toISOString()});
const out=fs.openSync(path.join(base,'engineering.stdout.log'),'wx'),err=fs.openSync(path.join(base,'engineering.stderr.log'),'wx');
let result;try{result=spawnSync(executable,argv,{cwd:root,stdio:['ignore',out,err]});}finally{fs.closeSync(out);fs.closeSync(err);}
let failure=result.error?.message??(result.status===0?null:`exit=${result.status}`);
try{sources.forEach(unchanged);unchanged(binary);}catch(error){failure=error.message;}
save('engineering-execution.json',{kind:'cpb_v19_engineering_execution_v1',status:failure?'failed':'passed',finished_utc:new Date().toISOString(),exit_code:result.status,error:failure,
  source_admission:pin(path.join(base,'engineering-admission.json')),stdout:pin(path.join(base,'engineering.stdout.log')),stderr:pin(path.join(base,'engineering.stderr.log'))});
if(failure)throw Error(`Engineering fixture failed: ${failure}; inspect preserved stderr`);
const report=JSON.parse(fs.readFileSync(path.join(output,'summary.json')));if(report.status!=='passed'||report.quality_evaluated!==false)throw Error('Passed scoped engineering receipt required');
console.log(JSON.stringify(report));

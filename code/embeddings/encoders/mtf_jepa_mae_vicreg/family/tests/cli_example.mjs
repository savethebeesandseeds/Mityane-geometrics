// SPDX-License-Identifier: MIT
// Orchestration only. All numerical work runs in the supplied Linux C++ binary.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';

const args=new Map();
for(let i=2;i<process.argv.length;i+=2) {
  const key=process.argv[i],value=process.argv[i+1];
  if(!['--trainer','--config','--output','--source-sha256'].includes(key)||value===undefined||args.has(key))
    throw new Error(`Invalid option ${key}`);
  args.set(key,value);
}
for(const required of ['--trainer','--config','--output'])if(!args.has(required))throw new Error(`Required ${required}`);
const trainer=path.resolve(args.get('--trainer')),config=path.resolve(args.get('--config')),out=path.resolve(args.get('--output'));
if(fs.existsSync(out))throw new Error(`Output must be new: ${out}`);
fs.mkdirSync(out,{recursive:true});
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const source=args.get('--source-sha256')??sha(trainer);
if(!/^[a-f0-9]{64}$/.test(source))throw new Error('Source identity must be lowercase SHA256');
const p=name=>path.join(out,name),commands=[];
let physical=0;
function run(name,values) {
  const result=spawnSync(trainer,values,{encoding:'utf8',maxBuffer:16*1024*1024});
  fs.writeFileSync(p(`${name}.stdout.log`),result.stdout??'');
  fs.writeFileSync(p(`${name}.stderr.log`),result.stderr??'');
  commands.push({name,arguments:values,exit_status:result.status,stdout_sha256:sha(p(`${name}.stdout.log`)),stderr_sha256:sha(p(`${name}.stderr.log`))});
  if(result.error||result.status!==0)throw new Error(`${name}: ${result.error?.message??result.stderr}`);
  const receipt=JSON.parse(result.stdout.trim());
  physical+=receipt.physical_updates_this_invocation??0;
  if(physical>6)throw new Error('CLI example exceeded six physical updates');
  return receipt;
}
run('synthetic',['synthetic','--config',config,'--output',p('raw.pt'),'--rows','5']);
const prepared=run('prepare',['prepare','--config',config,'--input',p('raw.pt'),'--output',p('prepared.pt')]);
const common=['--config',config,'--source-sha256',source,'--input-source','artificial-analytic-B5-BCHF-v1',
  '--split','engineering-TRAIN-only-no-heldout','--recipe','portable-CLI-example-v1',
  '--sampling','replacement-step-seeded-v1','--budget','3'];
const first=run('first-update',['train',...common,'--input',p('raw.pt'),'--checkpoint',p('one.pt'),'--steps','1']);
const resumed=run('resume',['train',...common,'--input',p('prepared.pt'),'--resume',p('one.pt'),'--checkpoint',p('three.pt'),
  '--steps','2','--inference',p('inference.pt')]);
const whole=run('uninterrupted',['train',...common,'--input',p('raw.pt'),'--checkpoint',p('whole.pt'),'--steps','3']);
if(resumed.state_sha256!==whole.state_sha256||resumed.optimizer_sha256!==whole.optimizer_sha256)
  throw new Error('CLI uninterrupted/resumed state differs');
if(prepared.dataset_sha256!==first.dataset_sha256||first.dataset_sha256!==resumed.dataset_sha256)
  throw new Error('Identity preparation changed the declared prepared data');
run('encode',['encode','--config',config,'--input',p('raw.pt'),'--model',p('inference.pt'),
  '--source-sha256',source,'--state-sha256',resumed.state_sha256,'--output',p('embeddings.pt')]);
if(physical!==6)throw new Error('Unexpected physical update accounting');
const files=fs.readdirSync(out).filter(name=>fs.statSync(p(name)).isFile()).map(name=>({path:name,sha256:sha(p(name))}));
const receipt={schema:'mtf-family-cli-example-v1',status:'passed',scope:'artificial engineering; no quality result',
  source_sha256:source,source_binding_kind:args.has('--source-sha256')?'caller-supplied-source-closure':'executable-sha256-fallback',
  trainer_sha256:sha(trainer),configuration_sha256:sha(config),dataset_sha256:resumed.dataset_sha256,
  recipe_sha256:resumed.recipe_sha256,state_sha256:resumed.state_sha256,optimizer_sha256:resumed.optimizer_sha256,
  runtime:resumed.runtime,physical_updates:physical,completed_updates:3,
  assertion_coverage:['arbitrary-F-artificial-input','identity-prepare','fresh-update','save-resume','fixed-budget',
    'exact-uninterrupted-resumed-model-and-AdamW','inference-export-load-encode'],commands,files};
fs.writeFileSync(p('cli-result.json'),JSON.stringify(receipt,null,2)+'\n');
process.stdout.write(JSON.stringify(receipt)+'\n');

// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// CLI orchestration/metadata assertions; numerical work stays in native C++.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

const args = new Map();
for (let i=2; i<process.argv.length; i+=2) {
  const key=process.argv[i];
  assert(['--trainer','--fixture','--settings','--output'].includes(key) && process.argv[i+1] && !args.has(key), 'unique supported options required');
  args.set(key, process.argv[i+1]);
}
assert.equal(args.size,4,'usage: --trainer BIN --fixture BIN --settings FILE --output NEWLEAF');
const output=path.resolve(args.get('--output'));
assert(!fs.existsSync(output),'output leaf must be new');
fs.mkdirSync(path.dirname(output),{recursive:true});
fs.mkdirSync(output);
const trainer=path.resolve(args.get('--trainer')), fixture=path.resolve(args.get('--fixture'));
const settings=path.resolve(args.get('--settings'));
const sha=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
let commands=0,refusals=0,assertions=0;
function check(condition,message){assertions++;assert(condition,message);}
function run(binary,argv,label,expectedSuccess=true){
  commands++;
  const result=spawnSync(binary,argv,{encoding:'utf8',timeout:120000,maxBuffer:4*1024*1024});
  fs.writeFileSync(path.join(output,`${label}.stdout.log`),result.stdout??'');
  fs.writeFileSync(path.join(output,`${label}.stderr.log`),result.stderr??'');
  if(result.error)throw result.error;
  const normalExit=Number.isInteger(result.status)&&result.signal===null;
  check(normalExit&&(expectedSuccess ? result.status===0 : result.status===1),`${label}: unexpected exit ${result.status}, signal ${result.signal}: ${result.stderr}`);
  if(!expectedSuccess)refusals++;
  return result;
}
const dataset=path.join(output,'synthetic-TRAIN.pt');
const otherDataset=path.join(output,'other-synthetic-TRAIN.pt');
const datasetArgs=file=>['--output',file,'--channels','2','--history','40','--features','2','--rows','8','--sampling-interval','1000'];
run(fixture,datasetArgs(dataset),'fixture');
const changedRows=datasetArgs(otherDataset);changedRows[changedRows.indexOf('--rows')+1]='9';
run(fixture,changedRows,'other-fixture');
run(trainer,['--help'],'help');
const base=['--settings',settings,'--dataset',dataset,'--batch-size','2',
  '--learning-rate','0.001','--weight-decay','0.0001','--clip-norm','1',
  '--row-seed','22001','--mask-seed','22002','--context-seed','22003','--save-every','1'];
const fresh=path.join(output,'fresh-2'),resumed=path.join(output,'resumed-4'),direct=path.join(output,'uninterrupted-4');
run(trainer,[...base,'--seed','221101','--steps','2','--max-attempts','10','--output',fresh],'fresh');
const checkpoint=path.join(fresh,'training-final.pt');
check(fs.statSync(checkpoint).isFile(),'fresh checkpoint exists');
const checkpointHash=sha(checkpoint),datasetHash=sha(dataset);
run(trainer,[...base,'--resume',checkpoint,'--steps','4','--max-attempts','20','--output',resumed],'resumed');
run(trainer,[...base,'--seed','221101','--steps','4','--max-attempts','20','--output',direct],'uninterrupted');
const record=dir=>JSON.parse(fs.readFileSync(path.join(dir,'run.json'),'utf8'));
const start=record(fresh),a=record(resumed),b=record(direct);
for(const r of [start,a,b]){
  check(r.status==='passed','native run records success');
  check(r.native_width===15,'configured native15 rather than benchmark native36');
  check(r.initialization_seed===221101,'initialization seed retained');
}
for(const field of ['final_state_identity','optimizer_identity','scaler_identity','bank_identity','completed_updates','attempted_batches']){
  check(a[field]!==undefined && a[field]===b[field],`resumed/uninterrupted exact ${field}`);
}
check(start.completed_updates===2&&a.completed_updates===4,'steps means total completed target');
check(start.preprocessing_fitted===true&&b.preprocessing_fitted===true&&a.preprocessing_fitted===false,'resume performs no preprocessing refit');
check(start.physical_training_updates_this_invocation===2&&a.physical_training_updates_this_invocation===2&&b.physical_training_updates_this_invocation===4,'actual invocation update counts');
check(fs.statSync(path.join(fresh,'training-1.pt')).isFile()&&fs.statSync(path.join(resumed,'training-3.pt')).isFile(),'periodic typed checkpoints are published at completed updates');
const physicalUpdates=[start,a,b].reduce((sum,r)=>sum+r.physical_training_updates_this_invocation,0);
const resumeArgs=[...base,'--resume',checkpoint,'--steps','4','--max-attempts','20'];
function changed(argv,key,value){const copy=[...argv];const i=copy.indexOf(key);assert(i>=0);copy[i+1]=value;return copy;}
for(const [label,argv] of [
  ['wrong-dataset',changed(resumeArgs,'--dataset',otherDataset)],
  ['wrong-row-seed',changed(resumeArgs,'--row-seed','22009')],
  ['wrong-mask-seed',changed(resumeArgs,'--mask-seed','22009')],
  ['wrong-context-seed',changed(resumeArgs,'--context-seed','22009')],
  ['wrong-learning-rate',changed(resumeArgs,'--learning-rate','0.002')],
  ['wrong-clip-norm',changed(resumeArgs,'--clip-norm','0.5')],
])run(trainer,[...argv,'--output',path.join(output,label)],label,false);
const wrongSettings=path.join(output,'changed-settings.conf');
fs.writeFileSync(wrongSettings,fs.readFileSync(settings,'utf8').replace('EMBEDDING_SIZE = 15;','EMBEDDING_SIZE = 16;'));
run(trainer,[...changed(resumeArgs,'--settings',wrongSettings),'--output',path.join(output,'wrong-settings')],'wrong-settings',false);
run(trainer,[...base,'--resume',path.join(fresh,'inference-final.pt'),'--steps','4','--max-attempts','20','--output',path.join(output,'inference-as-resume')],'inference-as-resume',false);
run(trainer,[...base,'--seed','221101','--steps','2','--max-attempts','10','--output',fresh],'existing-output',false);
check(sha(checkpoint)===checkpointHash&&sha(dataset)===datasetHash,'reference input dataset/state preserved through all runs');
const summary={kind:'cpb_family_runnable_training_example_v1',status:'passed',synthetic_only:true,quality_evaluated:false,
  native_width:15,commands,assertions,refusals,physical_training_updates:physicalUpdates,preprocessing_fits:2,
  resumed_preprocessing_fits:0,exact_resumed_vs_uninterrupted_state_and_optimizer:true,
  source_checkpoint_preserved:true,dataset_sha256:datasetHash,trainer_sha256:sha(trainer),
  records:[fresh,resumed,direct].map(dir=>({path:path.relative('/mityane',path.join(dir,'run.json')),sha256:sha(path.join(dir,'run.json'))}))};
fs.writeFileSync(path.join(output,'summary.json'),JSON.stringify(summary,null,2)+'\n');
process.stdout.write(JSON.stringify(summary)+'\n');

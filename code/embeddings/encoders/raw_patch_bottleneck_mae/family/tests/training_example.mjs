// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Waajacu. See the repository LICENSE.
// Native C++ performs all numerical work; Node only orchestrates and checks receipts.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

const args=new Map();
for(let i=2;i<process.argv.length;i+=2){
  const key=process.argv[i];
  assert(['--trainer','--fixture','--variant','--config-directory','--output'].includes(key)&&process.argv[i+1]&&!args.has(key),'unique supported options required');
  args.set(key,process.argv[i+1]);
}
assert.equal(args.size,5,'--trainer BIN --fixture BIN --variant BIN --config-directory DIR --output NEWLEAF');
const output=path.resolve(args.get('--output'));assert(!fs.existsSync(output),'new output leaf required');
fs.mkdirSync(path.dirname(output),{recursive:true});fs.mkdirSync(output);
const trainer=path.resolve(args.get('--trainer')),fixture=path.resolve(args.get('--fixture')),variant=path.resolve(args.get('--variant'));
const configDirectory=path.resolve(args.get('--config-directory'));
const sha=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
let commands=0,refusals=0,assertions=0,physicalUpdates=0,physicalAttempts=0;
const records=[];
function check(condition,message){assertions++;assert(condition,message);}
function run(binary,argv,label,expected=0){
  commands++;const result=spawnSync(binary,argv,{encoding:'utf8',timeout:120000,maxBuffer:4*1024*1024});
  fs.writeFileSync(path.join(output,`${label}.stdout.log`),result.stdout??'');fs.writeFileSync(path.join(output,`${label}.stderr.log`),result.stderr??'');
  if(result.error)throw result.error;
  check(result.status===expected&&result.signal===null,`${label}: exit ${result.status}, signal ${result.signal}: ${result.stderr}`);
  if(expected===1)refusals++;return result;
}
const readRecord=dir=>JSON.parse(fs.readFileSync(path.join(dir,'run.json'),'utf8'));
const events=dir=>fs.readFileSync(path.join(dir,'steps.jsonl'),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
function changed(argv,key,value){const copy=[...argv];const index=copy.indexOf(key);assert(index>=0);copy[index+1]=value;return copy;}
function checkRecord(dir){
  const record=readRecord(dir);records.push(dir);
  physicalUpdates+=record.physical_training_updates_this_invocation;
  physicalAttempts+=record.attempted_batches-record.starting_attempted_batches;
  for(const item of [record.settings,record.dataset,record.executable,...Object.values(record.artifacts)]){
    check(fs.statSync(item.path).size===item.bytes&&sha(item.path)===item.sha256,'recorded artifact size and SHA256 independently checked');
  }
  check(record.final_inference_reload_verified===true&&record.embedding_export_reload_verified===true&&record.quality_evaluated===false,'actual inference/embedding exports verified without quality claim');
  check(record.declared_role==='TRAIN'&&record.split_membership_independently_verified===false,'caller role separated from real split admission');
  return record;
}
run(trainer,['--help'],'help');
for(const profile of [{name:'single',C:1,H:65,F:3,D:8},{name:'multi',C:3,H:129,F:9,D:32}]){
  const settings=path.join(configDirectory,`${profile.name}.conf`),dataset=path.join(output,`${profile.name}-TRAIN.pt`);
  const fixtureArgs=file=>['--output',file,'--channels',String(profile.C),'--history',String(profile.H+7),'--features',String(profile.F),'--rows','6','--sampling-interval','1'];
  run(fixture,fixtureArgs(dataset),`${profile.name}-fixture`);
  const base=['--settings',settings,'--dataset',dataset,'--batch-size','2','--initialization-seed','221601',
    '--row-seed','221611','--mask-seed','221612','--context-seed','221613','--save-every','1'];
  const fresh=path.join(output,`${profile.name}-fresh-2`),resumed=path.join(output,`${profile.name}-resumed-4`),direct=path.join(output,`${profile.name}-direct-4`);
  run(trainer,[...base,'--steps','2','--max-attempts','10','--output',fresh],`${profile.name}-fresh`);
  const start=checkRecord(fresh),fit=JSON.parse(fs.readFileSync(path.join(fresh,'preprocessing-fit.json'),'utf8'));
  check(start.scaler_identity===fit.scaler_identity,'independent preprocessing receipt supplies resume scaler identity');
  const checkpoint=path.join(fresh,'training-final.pt'),checkpointHash=sha(checkpoint),datasetHash=sha(dataset);
  const resumeArgs=[...base,'--resume',checkpoint,'--expected-scaler',fit.scaler_identity,'--steps','4','--max-attempts','20'];
  run(trainer,[...resumeArgs,'--output',resumed],`${profile.name}-resume`);
  run(trainer,[...base,'--steps','4','--max-attempts','20','--output',direct],`${profile.name}-direct`);
  const a=checkRecord(resumed),b=checkRecord(direct);
  for(const record of [start,a,b]){
    check(record.status==='passed'&&record.native_width===profile.D,'configuration-specific successful native width');
    check(record.initialization_seed===221601&&record.context_deletion==='none','initialization and context recipe retained');
  }
  for(const field of ['final_state_identity','optimizer_identity','scaler_identity','embeddings_identity','completed_updates','attempted_batches','recipe_identity']){
    check(a[field]!==undefined&&a[field]===b[field],`exact resumed/direct ${field}`);
  }
  check(start.preprocessing_fitted&&b.preprocessing_fitted&&!a.preprocessing_fitted&&a.preprocessing_fit_rows===0,'resume never refits TRAIN');
  check(start.preprocessing_fit_rows===6&&b.preprocessing_fit_rows===6,'fresh fit visits each TRAIN row once');
  check(start.physical_training_updates_this_invocation===2&&a.physical_training_updates_this_invocation===2&&b.physical_training_updates_this_invocation===4,'total-update target retains actual physical counts');
  check(fs.existsSync(path.join(fresh,'training-1.pt'))&&fs.existsSync(path.join(resumed,'training-3.pt')),'periodic states published');
  const startEvents=events(fresh),resumeEvents=events(resumed),directEvents=events(direct);
  check(JSON.stringify([...startEvents,...resumeEvents])===JSON.stringify(directEvents),'exact continued source rows/masks/preclip gradients/loss/norm/counters');
  check(directEvents.every(event=>event.updated&&event.target_cells>0&&event.eligible_examples>0),'actual eligible updates recorded');
  for(const [label,argv] of [
    ['wrong-initialization-seed',changed(resumeArgs,'--initialization-seed','221699')],
    ['wrong-row-seed',changed(resumeArgs,'--row-seed','221699')],
    ['wrong-mask-seed',changed(resumeArgs,'--mask-seed','221699')],
    ['wrong-context-seed',changed(resumeArgs,'--context-seed','221699')],
    ['wrong-scaler',changed(resumeArgs,'--expected-scaler',`${fit.scaler_identity}-wrong`)],
  ])run(trainer,[...argv,'--output',path.join(output,`${profile.name}-${label}`)],`${profile.name}-${label}`,1);
  const wrongSettings=path.join(output,`${profile.name}-changed-settings.conf`);
  fs.writeFileSync(wrongSettings,fs.readFileSync(settings,'utf8')+'\n# Changed bytes require a new explicit recipe.\n');
  run(trainer,[...changed(resumeArgs,'--settings',wrongSettings),'--output',path.join(output,`${profile.name}-wrong-settings`)],`${profile.name}-wrong-settings`,1);
  const otherDataset=path.join(output,`${profile.name}-other-TRAIN.pt`),otherArgs=fixtureArgs(otherDataset);otherArgs[otherArgs.indexOf('--rows')+1]='7';
  run(fixture,otherArgs,`${profile.name}-other-fixture`);
  run(trainer,[...changed(resumeArgs,'--dataset',otherDataset),'--output',path.join(output,`${profile.name}-wrong-dataset`)],`${profile.name}-wrong-dataset`,1);
  run(trainer,[...changed(resumeArgs,'--resume',path.join(fresh,'inference-final.pt')),'--output',path.join(output,`${profile.name}-inference-resume`)],`${profile.name}-inference-resume`,1);
  run(trainer,[...base,'--steps','2','--max-attempts','10','--output',fresh],`${profile.name}-existing-output`,1);
  for(const mode of ['TEST','allmissing','nonfinite','wrong-ids','ineligible']){
    const variantPath=path.join(output,`${profile.name}-${mode}.pt`);
    run(variant,['--input',dataset,'--output',variantPath,'--mode',mode],`${profile.name}-${mode}-fixture`);
    const argv=[...changed(base,'--dataset',variantPath),'--steps','1','--max-attempts','2','--output',path.join(output,`${profile.name}-${mode}`)];
    run(trainer,argv,`${profile.name}-${mode}`,mode==='ineligible'?2:1);
    if(mode==='ineligible'){
      const record=checkRecord(path.join(output,`${profile.name}-${mode}`));
      check(record.status==='attempt_budget_exhausted'&&record.completed_updates===0&&record.attempted_batches===2&&record.skipped_attempts_this_invocation===2,'exhaustion reports two skips and zero updates');
      check(record.starting_state_identity===record.final_state_identity&&record.starting_optimizer_identity===record.optimizer_identity,'skips apply no weight decay or optimizer actions');
      check(events(path.join(output,`${profile.name}-${mode}`)).every(event=>!event.updated&&event.loss===0),'ineligible attempts are explicit');
    }
  }
  check(sha(checkpoint)===checkpointHash&&sha(dataset)===datasetHash,'all negative probes preserve original state and TRAIN archive');
}
const summary={kind:'rpb_family_runnable_training_example_v1',status:'passed',synthetic_only:true,quality_evaluated:false,
  profiles:['C1-H65-F3-P8-D8-W16','C3-H129-F9-P8-D32-W32'],commands,assertions,refusals,physical_training_updates:physicalUpdates,
  physical_attempted_batches:physicalAttempts,preprocessing_fits:6,preprocessing_fit_count_semantics:'successful complete freezes; failed probes remain refusals',resumed_preprocessing_fits:0,exact_resumed_vs_direct_events_model_optimizer_scaler_embeddings:true,
  valid_TEST_role_refused:true,allmissing_fresh_fit_refused:true,attempt_exhaustion_with_zero_updates:true,
  trainer_sha256:sha(trainer),fixture_sha256:sha(fixture),variant_sha256:sha(variant),
  records:records.map(dir=>({path:path.relative('/mityane',path.join(dir,'run.json')),sha256:sha(path.join(dir,'run.json'))}))};
fs.writeFileSync(path.join(output,'summary.json'),JSON.stringify(summary,null,2)+'\n');process.stdout.write(JSON.stringify(summary)+'\n');

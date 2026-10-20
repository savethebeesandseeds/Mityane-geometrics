// Copyright (c) 2026 Waajacu. See LICENSE.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {canonicalParents,requireRunCard,requirePhaseReceipt,requireDevelopmentSeal,subrun,SEEDS,ARMS,ATTEMPT} from './run.mjs';
const pin=path=>({path,bytes:1,sha256:'a'.repeat(64)});
function card(){
  const shared='preserved/shared',synthetic_inputs='preserved/tempo';
  const input_files=['selections.pt','query-plan-train.pt','query-plan-validation.pt','features/train-targets.pt','features/validation-targets.pt'].map(name=>pin(`${shared}/${name}`));
  for(const master of [941101,941102,941103])for(const name of ['training.pt','validation.pt','validation-deleted.pt'])input_files.push(pin(`${synthetic_inputs}/master-${master}/${name}`));
  return {kind:'cpb_v15_objective_run_inputs_v1',root:'/mityane',attempt:ATTEMPT,seeds:[...SEEDS],stream_offsets:{row:1000,mask:2000,context:3000},training:{updates:2048,batch_size:8,global_gradient_clip:1,learning_rate:.001,weight_decay:.0001},native_width:36,mean_huber_delta:1,mean_loss_weight:1,variance_loss_weight:1,
    python_execution:false,old_encoder_replay:false,checkpoint_selection:false,shared,synthetic_inputs,input_files,
    january_caches:['BTCUSDT','ETHUSDT','SOLUSDT'].map(s=>pin(`preserved/data/mkt-hist-1/csv/${s}-1m-2024-01.cache.bin`)),settings:{control:pin('v4.conf'),candidate:pin('v15.conf')},gradient_policies:{control:'fixed-original',candidate:'fixed-new'},
    confirmation:{month:'2024-02',rows:32,training_use:false,head_refits:false,descriptive_reconstruction_only:true,query_geometry:pin(`${shared}/query-plan-validation.pt`)}};
}
test('the closed schedule has exactly six paired fresh subruns without extra seed/arm paths',()=>{
  const paths=SEEDS.flatMap(seed=>ARMS.map(arm=>subrun(seed,arm)));assert.equal(new Set(paths).size,6);
  assert.ok(paths.every(p=>p.startsWith(`${ATTEMPT}/subruns/`)));
  for(const [seed,arm] of [[945104,'control'],[945101,'old_encoder'],[943001,'candidate']])assert.throws(()=>subrun(seed,arm));
});
test('fixed recipe refuses budget/coefficient/native-width, seed or confirmation-fit changes',()=>{
  requireRunCard(card());
  for(const mutate of [c=>{c.seeds.reverse();},c=>{c.training.updates=4096;},c=>{c.training.global_gradient_clip=2;},c=>{c.variance_loss_weight=.5;},c=>{c.native_width=40;},c=>{c.confirmation.training_use=true;},c=>{c.confirmation.head_refits=true;},c=>{c.python_execution=true;},c=>{c.checkpoint_selection=true;}]){const c=card();mutate(c);assert.throws(()=>requireRunCard(c));}
});
test('runtime roles require every original shared input and all three unchanged TEMPO cohorts',()=>{
  for(const mutate of [c=>{c.input_files.pop();},c=>{c.january_caches.reverse();},c=>{c.confirmation.query_geometry=pin('new-mask.pt');},c=>{c.input_files.push(c.input_files[0]);}]){const c=card();mutate(c);assert.throws(()=>requireRunCard(c));}
});

test('a phase receipt cannot migrate a passed status across card, seed, arm or phase',()=>{
  const cardPin=pin('card.json'),value={kind:'cpb_v15_objective_subrun_phase_v1',status:'passed',passed:true,card:cardPin,phase:'train',seed:945101,arm:'control',outputs:[pin('checkpoint.pt')]};
  requirePhaseReceipt(value,cardPin,'train',945101,'control');
  for(const change of [{card:{...cardPin,path:'other-card.json'}},{card:{...cardPin,bytes:2}},{phase:'development'},{seed:945102},{arm:'candidate'},{outputs:[]},{passed:false}])assert.throws(()=>requirePhaseReceipt({...value,...change},cardPin,'train',945101,'control'));
});

test('confirmation seal requires all six final checkpoints, eighteen fits and receipt/audit bindings',()=>{
  const cardPin=pin('card.json'),value={kind:'cpb_v15_all_six_development_runs_sealed_v1',status:'passed',passed:true,card:cardPin,subruns:6,independent_numeric_audits:6,confirmation_numerical_outcomes_opened:false,records:[],audits:[],outputs:[]};
  for(const seed of SEEDS)for(const arm of ARMS){const base=subrun(seed,arm);value.records.push(...['train','development'].map(phase=>pin(`${base}/${phase}-receipt.json`)));value.audits.push(pin(`${base}/independent-audit.json`));value.outputs.push(pin(`${base}/training/checkpoints/inference-2048.pt`),...[2701,2802,2903].map(rep=>pin(`${base}/heads-final/market/models/native36/rep-${rep}/fit.pt`)));}
  requireDevelopmentSeal(value,cardPin);
  for(const change of [{card:{...cardPin,path:'other-card.json'}},{records:value.records.slice(1)},{audits:value.audits.slice(1)},{outputs:value.outputs.slice(1)},{outputs:[...value.outputs,value.outputs[0]]},{confirmation_numerical_outcomes_opened:true},{independent_numeric_audits:5}])assert.throws(()=>requireDevelopmentSeal({...value,...change},cardPin));
});
test('parent creation validates the existing ancestor before any mkdir through a redirect',t=>{
  const base=fs.mkdtempSync(path.join(os.tmpdir(),'cpb-v15-run-redirect-'));t.after(()=>fs.rmSync(base,{recursive:true,force:true}));
  const root=path.join(base,'root'),outside=path.join(base,'outside');fs.mkdirSync(root);fs.mkdirSync(outside);
  fs.symlinkSync(outside,path.join(root,'redirect'),'dir');
  assert.throws(()=>canonicalParents(path.join(root,'redirect','new','leaf'),root),/directory redirect|redirected directory ancestor/);
  assert.deepEqual(fs.readdirSync(outside),[]);
  const ordinary=path.join(root,'ordinary','new');assert.equal(canonicalParents(ordinary,root),ordinary);
  assert.ok(fs.statSync(ordinary).isDirectory());
});

// Copyright (c) 2026 Waajacu. See LICENSE.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {currentPin,requirePreparationCard,requirePrecisionReceipt,requireNumericPrecisionResult,prepareTargets} from './prepare.mjs';
import {jsonBytes,sha256,parseJSON} from '../../../../shared/artifacts/node/v1/files.mjs';
const attempt='.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001';
const pin=name=>({repository:'Mityane-geometrics',path:name,bytes:1,sha256:'a'.repeat(64)});
function card(){const shared='preserved/output/runs/market-capacity-study/shared-MKT-HIST-2-v1';return {kind:'cpb_v16_target_preparation_inputs_v1',root:'/mityane',attempt,phase:'target_precision_parity',shared,
  january_caches:['BTCUSDT','ETHUSDT','SOLUSDT'].map(symbol=>pin(`preserved/data/mkt-hist-1/csv/${symbol}-1m-2024-01.cache.bin`)),
  original_targets:['train','validation'].map(role=>pin(`${shared}/features/${role}-targets.pt`)),
  output:`${attempt}/target-parity-numeric-001.json`,log:`${attempt}/target-parity-001.log`,receipt:`${attempt}/target-parity.json`};}
test('target parity admits only the exact January roles and fixed attempt leaves',()=>{
  assert.equal(requirePreparationCard(card()).phase,'target_precision_parity');
  for(const mutate of [c=>{c.january_caches.reverse();},c=>{c.original_targets.reverse();},c=>{c.attempt+='-other';},c=>{c.output=c.log;},c=>{c.receipt='outside.json';},c=>{c.phase='train';}]){
    const c=card();mutate(c);assert.throws(()=>requirePreparationCard(c));
  }
});
test('confirmation targets require exactly the frozen ordered March caches and prior precision proof',()=>{
  const c=card();c.phase='confirmation_targets';c.output=`${attempt}/confirmation-targets.pt`;
  c.confirmation_caches=['BTCUSDT','ETHUSDT','SOLUSDT'].map(symbol=>pin(`data/${symbol}-1m-2024-03.cache.bin`));
  c.target_parity=pin(`${attempt}/target-parity.json`);assert.equal(requirePreparationCard(c).phase,'confirmation_targets');
  delete c.target_parity;assert.throws(()=>requirePreparationCard(c),/passed pinned precision/);
  c.target_parity=pin('parity.json');c.confirmation_caches[0].path='data/BTCUSDT-1m-2024-02.cache.bin';
  assert.throws(()=>requirePreparationCard(c),/frozen March/);
});
test('passed precision proof cannot omit exact numeric parity or claim prior March access',()=>{
  const result={kind:'cpb_v16_temporal_target_precision_parity_v1',status:'passed',passed:true,train_rows:112,validation_rows:32,target_precision:'legacy_f32_then_f64',all_target_values_bitwise_equal:true,all_masks_and_ids_exact:true,March_values_opened:false,encoder_calls:0,head_fits:0};
  requirePrecisionReceipt(result);
  for(const change of [{passed:false},{train_rows:111},{target_precision:'raw_f64'},{all_target_values_bitwise_equal:false},{March_values_opened:true},{head_fits:1}])assert.throws(()=>requirePrecisionReceipt({...result,...change}));
});

test('the attached numeric parity result independently requires112/32, legacy precision and zero fits',()=>{
  const result={kind:'cpb_v16_temporal_target_precision_parity_numeric_v1',status:'passed',passed:true,train_rows:112,validation_rows:32,target_precision:'legacy_f32_then_f64',all_target_values_bitwise_equal:true,all_masks_and_ids_exact:true,March_values_opened:false,encoder_calls:0,head_fits:0};
  requireNumericPrecisionResult(result);
  for(const change of [{kind:'unrelated_pass'},{validation_rows:31},{target_precision:'raw_f64'},{all_masks_and_ids_exact:false},{encoder_calls:1},{head_fits:1}])assert.throws(()=>requireNumericPrecisionResult({...result,...change}));
});
test('current source pins reject stale bytes, wrong repository and redirected locators',t=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'cpb-v16-prepare-'));t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  const bytes=Buffer.from('bound source');fs.writeFileSync(path.join(root,'source.cpp'),bytes);
  const p={repository:'Mityane-geometrics',path:'source.cpp',bytes:bytes.length,sha256:sha256(bytes)};
  assert.deepEqual(currentPin(root,p),p);
  assert.throws(()=>currentPin(root,{...p,repository:'cuwacunu_embedding'}),/physical pins/);
  assert.throws(()=>currentPin(root,{...p,path:'../source.cpp'}));
  fs.appendFileSync(path.join(root,p.path),'changed');assert.throws(()=>currentPin(root,p),/input changed/);
});
test('failed source review preserves a failed attempt without launching numerical preparation',t=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'cpb-v16-gate-'));t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  fs.mkdirSync(path.join(root,attempt),{recursive:true});
  const write=(name,value)=>{const bytes=jsonBytes(value);fs.writeFileSync(path.join(root,name),bytes);return {path:name,bytes:bytes.length,sha256:sha256(bytes)};};
  const c=card();c.source_review=write(`${attempt}/source-review.json`,{status:'failed',passed:false,files:[]});
  c.protocol_review=write(`${attempt}/protocol-review.json`,{status:'passed',passed:true,approved_phases:['target_precision_parity']});
  c.binary=pin('/opt/mityane-geometrics/embeddings/build/cpb-v16-temporal-001/never-launched');
  const input=`${attempt}/inputs.json`;write(input,c);
  assert.throws(()=>prepareTargets(root,input),/source review must have passed/);
  const failed=parseJSON(fs.readFileSync(path.join(root,c.receipt)));assert.equal(failed.passed,false);assert.equal(failed.scientific_result_claimed,false);
  assert.equal(fs.existsSync(path.join(root,c.log)),false);assert.equal(fs.existsSync(path.join(root,c.output)),false);
});
test('a redirected attempt is rejected before creating any output or failure receipt',t=>{
  const base=fs.mkdtempSync(path.join(os.tmpdir(),'cpb-v16-prepare-redirect-'));t.after(()=>fs.rmSync(base,{recursive:true,force:true}));
  const root=path.join(base,'root'),outside=path.join(base,'outside');fs.mkdirSync(root);fs.mkdirSync(outside);
  fs.mkdirSync(path.dirname(path.join(root,attempt)),{recursive:true});fs.symlinkSync(outside,path.join(root,attempt),'dir');
  const bytes=jsonBytes(card());fs.writeFileSync(path.join(root,'inputs.json'),bytes);
  assert.throws(()=>prepareTargets(root,'inputs.json'),/directory redirect|redirected directory ancestor/);
  assert.deepEqual(fs.readdirSync(outside),[]);
});

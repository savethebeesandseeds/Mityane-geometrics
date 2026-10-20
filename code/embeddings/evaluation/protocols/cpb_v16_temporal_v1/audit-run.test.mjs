// Copyright (c) 2026 Waajacu. See LICENSE.
// Artificial path metadata only: no dataset, checkpoint or tensor bytes opened.
import test from 'node:test';
import assert from 'node:assert/strict';
import {buildPlan,expectedDecodedPin,requireAuditBindings,requireFullDevelopmentSeal} from './audit-run.mjs';

const attempt='.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001';
const shared='artificial/original-January-shared';
const confirmation='artificial/blind-March/confirmation-targets.pt';
const card=()=>({shared,confirmation_targets:confirmation});
const base=(seed,arm)=>`${attempt}/subruns/seed-${seed}/${arm}`;
const reps=[2701,2802,2903];
const pin=path=>({repository:'Mityane-geometrics',path,bytes:1,sha256:'a'.repeat(64)});

test('all six independent development plans cover126 typed sets with no duplicate or checkpoint-decoding row',()=>{
  for(const seed of [946101,946102,946103])for(const arm of ['control','candidate']){
    const root=base(seed,arm),rows=buildPlan(card(),root,'development');
    assert.equal(rows.length,126);assert.equal(new Set(rows.map(row=>JSON.stringify(row))).size,126);
    const counts=Object.fromEntries([...new Set(rows.map(row=>row[0]))].map(mode=>[mode,rows.filter(row=>row[0]===mode).length]));
    assert.deepEqual(counts,{regression:12,classification:108,fixed_query:4,[arm==='candidate'?'objective_temporal':'objective_control']:2});
    for(const [mode,...files] of rows){
      assert.equal(files.length,mode==='regression'?6:mode==='classification'?4:mode==='objective_temporal'?2:1);
      assert.ok(files.every(file=>file.endsWith('.pt')&&!/[\t\r\n]/.test(file)));
      assert.ok(files.every(file=>!file.includes('/checkpoints/')&&!file.endsWith('/resume.pt')),'hash-only checkpoint must not enter arithmetic plan');
      assert.ok(files.every(file=>file.startsWith(`${root}/`)||file.startsWith(`${shared}/`)),'subrun and immutable shared inputs are the only development roles');
    }
  }
});

test('each market probe reuses that stage TRAIN features, original TRAIN labels and the exact same fitted head',()=>{
  const root=base(946101,'candidate'),rows=buildPlan(card(),root,'development').filter(row=>row[0]==='regression');
  for(const stage of ['initial','final'])for(const rep of reps){
    const fit=`${root}/heads-${stage}/market/models/native36/rep-${rep}/fit.pt`;
    const pair=rows.filter(row=>row[3]===fit);assert.equal(pair.length,2);
    assert.deepEqual(new Set(pair.map(row=>row[4])),new Set(['train','validation'].map(view=>`${root}/features/${stage}/market/cpb-objective-${view}.pt`)));
    for(const row of pair){
      assert.equal(row[1],`${root}/features/${stage}/market/cpb-objective-train.pt`);
      assert.equal(row[2],`${shared}/features/train-targets.pt`);
      const view=row[4].endsWith('-validation.pt')?'validation':'train';
      assert.equal(row[5],`${shared}/features/${view}-targets.pt`);
      assert.equal(row[6],`${root}/heads-${stage}/market/models/native36/rep-${rep}/predictions_${view}.pt`);
    }
  }
});

test('TEMPO readout plans retain all master, input-preparation, repetition and three-view identities',()=>{
  const root=base(946102,'control'),rows=buildPlan(card(),root,'development').filter(row=>row[0]==='classification');
  const groups=new Map();
  for(const row of rows){
    const match=new RegExp(`^${root}/heads-(initial|final)/synthetic/master-(941101|941102|941103)/(synthetic_train|market_scaler)/rep-(2701|2802|2903)/fit\\.pt$`).exec(row[2]);
    assert.ok(match,`producer fit role missing: ${row[2]}`);
    const method=row[2].slice(0,row[2].lastIndexOf('/rep-'));
    assert.equal(row[1],`${method}/training-features.pt`,'the fit population is always the matching TRAIN surface');
    const view=row[3].slice(method.length+1,-'-features.pt'.length);
    assert.ok(['training','validation-intact','validation-deleted'].includes(view));
    assert.equal(row[4],`${method}/rep-${match[4]}/${view}-predictions.pt`);
    const views=groups.get(row[2])??new Set();assert.ok(!views.has(view));views.add(view);groups.set(row[2],views);
  }
  assert.equal(groups.size,36);
  for(const views of groups.values())assert.deepEqual(views,new Set(['training','validation-intact','validation-deleted']));
});

test('objective witnesses cover only first/last actual update and query witnesses retain initial/final TRAIN/VALIDATION roles',()=>{
  const root=base(946103,'candidate'),rows=buildPlan(card(),root,'development');
  assert.deepEqual(rows.filter(row=>row[0]==='objective_temporal'),[
    ['objective_temporal',`${root}/training/witnesses/temporal-objective-step-1.pt`,`${root}/training/witnesses/objective-update-1.pt`],
    ['objective_temporal',`${root}/training/witnesses/temporal-objective-step-2048.pt`,`${root}/training/witnesses/objective-update-2048.pt`],
  ]);
  const control=base(946103,'control');
  assert.deepEqual(buildPlan(card(),control,'development').filter(row=>row[0]==='objective_control'),[
    ['objective_control',`${control}/training/witnesses/objective-update-1.pt`],
    ['objective_control',`${control}/training/witnesses/objective-update-2048.pt`],
  ]);
  assert.ok(rows.every(row=>row[0]!=='objective_candidate'),'V15 Huber objective mode must never be assigned to a V16 candidate');
  assert.deepEqual(rows.filter(row=>row[0]==='fixed_query'),[
    ['fixed_query',`${root}/training/witnesses/initial-train-query.pt`],
    ['fixed_query',`${root}/training/witnesses/initial-validation-query.pt`],
    ['fixed_query',`${root}/training/witnesses/final-train-query.pt`],
    ['fixed_query',`${root}/training/witnesses/final-validation-query.pt`],
  ]);
});

test('confirmation audits use three sealed final TRAIN fits and March targets without any fit or training-objective job',()=>{
  const root=base(946101,'control'),rows=buildPlan(card(),root,'confirmation');
  assert.equal(rows.length,4);
  for(const [index,rep] of reps.entries())assert.deepEqual(rows[index],[
    'regression',`${root}/features/final/market/cpb-objective-train.pt`,`${shared}/features/train-targets.pt`,
    `${root}/heads-final/market/models/native36/rep-${rep}/fit.pt`,
    `${root}/confirmation-features/confirmation-features.pt`,confirmation,
    `${root}/confirmation-readouts/rep-${rep}/predictions_confirmation.pt`,
  ]);
  assert.deepEqual(rows[3],['fixed_query',`${root}/confirmation-features/confirmation-query-witness.pt`]);
  assert.ok(rows.every(row=>!row[0].startsWith('objective_')&&row[0]!=='classification'));
  assert.throws(()=>buildPlan(card(),root,'train'),/only development\/confirmation audit/);
  assert.throws(()=>buildPlan(card(),root,'confirmation-retry'),/only development\/confirmation audit/);
});

test('decoded arrays require one exact phase-output or frozen external study role, never a newly observed replacement',()=>{
  const root=base(946101,'candidate'),local=pin(`${root}/heads-final/market/models/native36/rep-2701/fit.pt`),target=pin(`${shared}/features/train-targets.pt`),february=pin(confirmation);
  const options={base:root,phase_outputs:[local],study_inputs:[target],confirmation_target:february};
  assert.strictEqual(expectedDecodedPin(local.path,options),local);
  assert.strictEqual(expectedDecodedPin(target.path,options),target);
  assert.strictEqual(expectedDecodedPin(february.path,options),february);
  assert.throws(()=>expectedDecodedPin(`${root}/missing-current-array.pt`,options),/absent or duplicated/);
  assert.throws(()=>expectedDecodedPin('new-external-targets.pt',options),/frozen study inputs/);
  assert.throws(()=>expectedDecodedPin(local.path,{...options,phase_outputs:[],study_inputs:[local,target]}),/actual phase outputs/,'a subrun archive cannot bypass phase ownership through the study input list');
  assert.throws(()=>expectedDecodedPin(target.path,{...options,study_inputs:[target,{...target,sha256:'b'.repeat(64)}]}),/duplicated/);
  assert.throws(()=>expectedDecodedPin(february.path,{...options,confirmation_target:null}),/frozen study inputs/);
  assert.equal(expectedDecodedPin(target.path,options).sha256,'a'.repeat(64),'caller must verify the frozen expected pin, rather than construct a replacement identity');
});

test('auditor source/build bindings include the shared parser and both drivers, and reject passed but unrelated receipts',()=>{
  const prefix='code/embeddings/evaluation/protocols/cpb_v16_temporal_v1';
  const cfg={source_review:pin('audit-source-review.json'),binary:pin('/opt/mityane-geometrics/embeddings/build/cpb-v16-temporal-001/cpb_v16_independent_audit')};
  const source={kind:'cpb_v16_independent_source_review_v1',status:'passed',passed:true,
    files:[...['audit.cpp','include/temporal_audit.h','audit-run.mjs','audit-run.test.mjs','audit_fixture.cpp','audit_test.mjs','audit.mk'].map(name=>pin(`${prefix}/${name}`)),pin('code/shared/artifacts/node/v1/files.mjs')]};
  const build={kind:'cpb_v16_independent_audit_build_v1',status:'passed',passed:true,source_review:cfg.source_review,binaries:[{...cfg.binary,role:'independent_audit'}]};
  requireAuditBindings(source,build,cfg);
  for(const change of [{kind:'unrelated_source_review'},{passed:false},{files:source.files.slice(0,-1)},{files:source.files.filter(p=>!p.path.endsWith('/audit-run.test.mjs'))},{files:source.files.filter(p=>!p.path.endsWith('/temporal_audit.h'))},{files:[...source.files,source.files[0]]}])assert.throws(()=>requireAuditBindings({...source,...change},build,cfg));
  for(const change of [{kind:'unrelated_build'},{status:'failed'},{source_review:{...cfg.source_review,path:'other-review.json'}},{source_review:{...cfg.source_review,repository:'cuwacunu_embedding'}},{binaries:[]},{binaries:[...build.binaries,{...cfg.binary,role:'independent_audit',sha256:'b'.repeat(64)}]}])assert.throws(()=>requireAuditBindings(source,{...build,...change},cfg));
});

test('plans reject old-study seeds, unrelated arms and predecessor attempt roots',()=>{
  for(const root of [base(945101,'candidate'),base(946101,'other'),'.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-946101/candidate']){
    assert.throws(()=>buildPlan(card(),root,'development'),/exact V16 declared seed\/arm/);
  }
});

test('confirmation requires the typed blind six-run seal with every phase receipt, audit, final checkpoint and fitted head',()=>{
  const cardPin=pin('study-card.json'),seal={kind:'cpb_v16_all_six_development_runs_sealed_v1',status:'passed',passed:true,card:cardPin,subruns:6,independent_numeric_audits:6,confirmation_numerical_outcomes_opened:false,records:[],audits:[],outputs:[]};
  for(const seed of [946101,946102,946103])for(const arm of ['control','candidate']){
    const root=base(seed,arm);seal.records.push(...['train','development'].map(phase=>pin(`${root}/${phase}-receipt.json`)));
    seal.audits.push(pin(`${root}/independent-audit.json`));seal.outputs.push(pin(`${root}/training/checkpoints/inference-2048.pt`),...reps.map(rep=>pin(`${root}/heads-final/market/models/native36/rep-${rep}/fit.pt`)));
  }
  assert.strictEqual(requireFullDevelopmentSeal(seal,cardPin),seal);
  for(const change of [{kind:'some_other_passed_seal'},{card:{...cardPin,sha256:'b'.repeat(64)}},{card:{...cardPin,repository:'cuwacunu_embedding'}},{subruns:5},{confirmation_numerical_outcomes_opened:true},{records:seal.records.slice(1)},{records:[...seal.records.slice(1),seal.records[1]]},{audits:seal.audits.slice(1)},{outputs:seal.outputs.slice(1)},{outputs:[...seal.outputs,seal.outputs[0]]},{outputs:seal.outputs.filter(p=>!p.path.endsWith('/rep-2802/fit.pt'))}])assert.throws(()=>requireFullDevelopmentSeal({...seal,...change},cardPin));
});

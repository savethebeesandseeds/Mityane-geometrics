// Copyright (c) 2026 Waajacu. See LICENSE.
// Artificial scalar fixtures only; no data/checkpoint access or model execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import * as original from './report.mjs';
import {compareCriterion,aggregateMarket,aggregateTempo,historicalRetention,pairedScreen,requireReportCard,requireTrainingSummary,requireAuditTuple,TASKS} from './report_final_unique_001.mjs';
import {ORIGINAL_REPORT_PROGRAM,planFinalVerification,verifyFinalBarrier} from './report_final_unique_001.mjs';

const reps=[2701,2802,2903],seeds=['13088411117524848790','10018485049408248344','9083751155097190455'];
function metric(r2){return {total:32,target_eligible:32,valid:32,coverage:1,mae:.1,rmse:.2,r2};}
function marketRows(neural=[.1,.4,.7]){return reps.map((repetition,i)=>({repetition,actual_seed_decimal:seeds[i],tasks:TASKS.map(()=>({ridge:metric(.8),neural:metric(neural[i])}))}));}
function tempoRows(){return [941101,941102,941103].flatMap((master,j)=>['synthetic_train','market_scaler'].map(input_policy=>({master,input_policy,repetitions:reps.map((repetition,i)=>({repetition,actual_seed_decimal:seeds[i],...Object.fromEntries(['training','validation_intact','validation_deleted'].map(view=>[view,{total:view==='training'?512:256,valid:view==='training'?512:256,coverage:1,ridge_accuracy:.9+j*.01,neural_accuracy:.8+i*.01}]))}))})));}
function goal(){const criteria=[];for(const task of TASKS)for(const metric of ['linear_r2','neural_mean_r2'])criteria.push({id:`${task}:${metric}`,category:'market_retention',floor:0,rule:'candidate + 1e-12 >= floor'});
  for(const mode of ['synthetic_train','market_scaler'])for(const view of ['validation_intact','validation_deleted'])for(const metric of ['linear_head_accuracy','neural_head_accuracy'])criteria.push({id:`${mode}:${view}:${metric}`,category:'tempo5_retention',floor:.8,rule:'candidate + 1e-12 >= floor'});
  for(const role of ['train','validation'])criteria.push({id:`${role}:standardized_MAE`,category:'fixed_query_MAE',floor:.3,rule:'candidate <= floor + 1e-12'});
  criteria.push({id:'known_validation:Gaussian_NLL_F64',category:'variance_quality',floor:0,rule:'candidate < floor - 1e-12'},{id:'past_trend:all3_neural_R2_positive',category:'neural_trend_robustness',floor:0,rule:'all three final neural trend R2 values > 0'},{id:'past_relation:linear_r2:positive',category:'relation_usefulness',floor:0,rule:'candidate > 0'},{id:'past_relation:neural_mean_r2:positive',category:'relation_usefulness',floor:0,rule:'candidate > 0'});return {criterion_count:22,criteria};}
function pairs(){return [945101,945102,945103].map(seed=>({seed,control:{confirmation:Object.fromEntries(TASKS.map(t=>[t,{linear_r2:.4,neural_mean_r2:.3,coverage:1}]))},candidate:{hard_timing_passed:true,confirmation:Object.fromEntries(TASKS.map(t=>[t,{linear_r2:t==='past_trend'?.4:.5,neural_mean_r2:.35,coverage:1}]))}}));}

test('historical rule comparisons preserve inclusive tolerances, strict NLL and strict positivity',()=>{
  const check=(rule,value,floor=1)=>compareCriterion({id:'fixture',rule,floor},value).passed;
  assert.equal(check('candidate + 1e-12 >= floor',1-5e-13),true);
  assert.equal(check('candidate + 1e-12 >= floor',1-2e-12),false);
  assert.equal(check('candidate <= floor + 1e-12',1+5e-13),true);
  assert.equal(check('candidate < floor - 1e-12',1),false);
  assert.equal(check('candidate < floor - 1e-12',1-2e-12),true);
  assert.equal(check('candidate > 0',0,0),false);assert.equal(check('candidate > 0',Number.MIN_VALUE,0),true);
  assert.equal(check('all three final neural trend R2 values > 0',[.1,0,.2],0),false);
  assert.equal(check('all three final neural trend R2 values > 0',[.1,.1,.2],0),true);
  assert.equal(check('candidate + 1e-12 >= floor',null),false);
  assert.throws(()=>check('almost exceeds',1),/unknown frozen criterion/);
});

test('market aggregation retains every typed head seed and averages neural repetitions without adding encoder samples',()=>{
  const result=aggregateMarket(marketRows());assert.equal(result.past_activity.linear_r2,.8);assert.ok(Math.abs(result.past_activity.neural_mean_r2-.4)<1e-15);
  assert.deepEqual(result.past_activity.neural_r2_each,[.1,.4,.7]);assert.equal(result.past_activity.repetitions.length,3);
  assert.equal(result.past_activity.repetitions[0].actual_seed_decimal,'13088411117524848790');
  assert.throws(()=>aggregateMarket(marketRows().slice(1)),/three D36/);
  const changed=marketRows();changed[1].tasks[0].ridge.r2=.7;assert.throws(()=>aggregateMarket(changed),/deterministic/);
});

test('undefined neural R2 is retained as undefined mean and cannot silently become zero',()=>{
  const result=aggregateMarket(marketRows([.1,null,.7]));assert.equal(result.past_trend.neural_mean_r2,null);assert.deepEqual(result.past_trend.neural_r2_each,[.1,null,.7]);
});

test('TEMPO aggregation retains six cohorts, three repetitions and both views using original master means',()=>{
  const result=aggregateTempo(tempoRows());assert.equal(result.cohorts.length,6);
  assert.ok(Math.abs(result.scores.synthetic_train.validation_intact.linear_head_accuracy-.91)<1e-15);
  assert.ok(Math.abs(result.scores.synthetic_train.validation_deleted.neural_head_accuracy-.81)<1e-15);
  assert.deepEqual(result.scores.market_scaler.validation_intact.linear_master_means,[.9,.91,.92]);
  assert.throws(()=>aggregateTempo(tempoRows().slice(1)),/three masters/);
  const repeated=tempoRows();repeated[5]=repeated[0];assert.throws(()=>aggregateTempo(repeated),/unique declared/);
});

test('the historical report keeps all22 objects, eight timing checks and three-head positivity quantifier',()=>{
  const original=goal(),copy=structuredClone(original),market=aggregateMarket(marketRows()),tempo=aggregateTempo(tempoRows()),queries={train:{standardized_fixed_query_MAE:.2,valid_rows:112},validation:{standardized_fixed_query_MAE:.2,Gaussian_NLL_F64:-.1,valid_rows:32}};
  const result=historicalRetention(original,market,tempo,queries);assert.equal(result.total,22);assert.equal(result.passed,22);assert.equal(result.hard_timing_passed,true);assert.deepEqual(original,copy);
  market.past_trend.neural_r2_each[1]=0;const failed=historicalRetention(original,market,tempo,queries);assert.equal(failed.passed,21);assert.equal(failed.criteria.find(c=>c.id==='past_trend:all3_neural_R2_positive').passed,false);
  queries.validation.Gaussian_NLL_F64=0;assert.equal(historicalRetention(original,market,tempo,queries).passed,20);
  assert.equal(result.TRAIN_NLL_additional_gate,false);
});

test('February screen uses three paired encoder differences with primary medians, trend tolerance and all candidate timing floors',()=>{
  const rows=pairs(),result=pairedScreen(rows);assert.equal(result.verdict,'promising');assert.equal(result.encoder_pairs,3);assert.equal(result.overall_winner,false);assert.equal(result.Mityane_application_selection,null);
  rows[0].candidate.confirmation.past_activity.linear_r2=.3;assert.equal(pairedScreen(rows).promising,true);
  rows[1].candidate.confirmation.past_activity.linear_r2=.3;assert.equal(pairedScreen(rows).promising,false);
  const trend=pairs();for(const p of trend)p.candidate.confirmation.past_trend.linear_r2=.389;assert.equal(pairedScreen(trend).trend_tolerance_passed,false);
  const timing=pairs();timing[2].candidate.hard_timing_passed=false;assert.equal(pairedScreen(timing).promising,false);
});

test('missing pairs, support or defined R2 produce inconclusive outcomes without dropping a seed',()=>{
  assert.equal(pairedScreen(pairs().slice(1)).verdict,'inconclusive');
  const rows=pairs();rows[0].candidate.confirmation.past_volatility.linear_r2=null;assert.equal(pairedScreen(rows).verdict,'inconclusive');
  const support=pairs();support[2].control.confirmation.past_activity.coverage=.9;assert.equal(pairedScreen(support).verdict,'inconclusive');
  const developmentSupport=pairs();developmentSupport[1].candidate.full_declared_coverage=false;assert.equal(pairedScreen(developmentSupport).verdict,'inconclusive');
  const secondary=pairs();secondary[1].candidate.confirmation.past_relation.neural_mean_r2=null;assert.equal(pairedScreen(secondary).verdict,'inconclusive');
  const negative=pairs();for(const p of negative)for(const t of ['past_volatility','past_activity'])p.candidate.confirmation[t].linear_r2=.3;assert.equal(pairedScreen(negative).verdict,'negative');
});

test('report card requires exact immutable historical goal and all six declared roles',()=>{
  const original={repository:'cuwacunu_embedding',path:'doc/causal_fresh_v14/goal-protocol.json',bytes:17536,sha256:'4ace8b79a496ff24440ef1a269b63b45b746624f51dbd734e1b9c013b8a93245'},card={kind:'cpb_v15_objective_report_inputs_v1',root:'/mityane',attempt:'.temp/embeddings/runs/cpb-v15-objective-001/attempt-001',output:'.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-001.json',subruns:[945101,945102,945103].flatMap(seed=>['control','candidate'].map(arm=>({seed,arm}))),historical_goal:{original,current:{...original,repository:'Mityane-geometrics',path:'archive/goal.json'}},historical_sources:[]};
  requireReportCard(card);
  assert.throws(()=>requireReportCard({...card,subruns:card.subruns.slice(1)}),/six/);
  assert.throws(()=>requireReportCard({...card,historical_goal:{...card.historical_goal,original:{...original,sha256:'a'.repeat(64)}}}),/historical22/);
  assert.throws(()=>requireReportCard({...card,output:'current.json'}),/exclusive/);
});

test('selector arms bind frozen compiled training identities rather than equating different schema labels',()=>{
  const summary={kind:'cpb_v15_objective_paired_training_summary_v1',status:'passed',initialization_seed:945101,arm:'fresh_v4_partial_nll_control',gradient_policy:'frozen-control-policy',new_causal_updates:2048,batch_size:8,typed_resume_512_exact:true};
  requireTrainingSummary(summary,945101,'control','frozen-control-policy');
  requireTrainingSummary({...summary,arm:'unit_huber1_plus_variance_nll_v15',gradient_policy:'frozen-candidate-policy'},945101,'candidate','frozen-candidate-policy');
  assert.throws(()=>requireTrainingSummary({...summary,arm:'control'},945101,'control','frozen-control-policy'),/compiled arm/);
  assert.throws(()=>requireTrainingSummary(summary,945101,'candidate','frozen-control-policy'),/compiled arm/);
});

test('arithmetic association binds every plan role including TRAIN/probe features, target files and checkpoint stage',()=>{
  const expected=['/mityane/final-train-features.pt','/mityane/january-train-targets.pt','/mityane/final-fit.pt','/mityane/confirmation-features.pt','/mityane/february-targets.pt','/mityane/confirmation-predictions.pt'];
  requireAuditTuple({mode:'regression',paths:[...expected]},'regression',expected);
  for(const [index,value] of [[0,'/mityane/initial-train-features.pt'],[1,'/mityane/validation-targets.pt'],[3,'/mityane/january-validation-features.pt'],[4,'/mityane/january-targets.pt']]){const wrong=[...expected];wrong[index]=value;assert.throws(()=>requireAuditTuple({mode:'regression',paths:wrong},'regression',expected),/complete exact/);}
  const classification=['/mityane/training-surface.pt','/mityane/fit.pt','/mityane/deleted-surface.pt','/mityane/deleted-predictions.pt'];
  requireAuditTuple({mode:'classification',paths:classification},'classification',classification);
  assert.throws(()=>requireAuditTuple({mode:'classification',paths:[classification[2],...classification.slice(1)]},'classification',classification),/complete exact/);
});

const fixturePin=(name='a.txt',bytes=3,sha256='a'.repeat(64))=>({repository:'Mityane-geometrics',path:name,bytes,sha256});
const binaryPin=(name='audit',bytes=4,sha256='b'.repeat(64))=>fixturePin(`/opt/mityane-geometrics/embeddings/build/cpb-v15-objective-001/${name}`,bytes,sha256);

test('final barrier collapses only complete identical tuples and retains their last occurrence order',()=>{
  const a=fixturePin(),b=fixturePin('b.txt'),c=fixturePin('c.txt'),lastA={...a},lastB={...b};
  const input=[a,b,lastA,c,lastB],snapshot=structuredClone(input),plan=planFinalVerification(input,[]);
  assert.deepEqual(plan.files.map(p=>p.path),['a.txt','c.txt','b.txt']);
  assert.equal(plan.files[0],lastA);assert.equal(plan.files[2],lastB);
  assert.deepEqual(input,snapshot);assert.equal(plan.summary.file_pin_occurrences,5);
  assert.equal(plan.summary.unique_file_fulltuples,3);assert.equal(plan.summary.duplicate_file_occurrences,2);
  assert.equal(plan.summary.declared_file_occurrence_bytes_decimal,'15');assert.equal(plan.summary.declared_unique_file_bytes_decimal,'9');
  assert.equal(plan.summary.earlier_barrier_observations_reused,false);assert.equal(plan.summary.retention_order,'last occurrence');
  assert.deepEqual(plan.summary.original_program,ORIGINAL_REPORT_PROGRAM);
});

test('one physical path with conflicting bytes or SHA is rejected rather than overwritten',()=>{
  const a=fixturePin();
  assert.throws(()=>planFinalVerification([a,{...a,bytes:4}],[]),/conflicting final-barrier/);
  assert.throws(()=>planFinalVerification([a,{...a,sha256:'c'.repeat(64)}],[]),/conflicting final-barrier/);
  const unqualified={...a};delete unqualified.repository;
  assert.equal(planFinalVerification([a,unqualified],[]).files.length,1);
  assert.throws(()=>planFinalVerification([a,{...unqualified,bytes:4}],[]),/conflicting final-barrier/);
});

test('every occurrence is validated before a final hash including malformed duplicate-shaped pins',()=>{
  const a=fixturePin();
  for(const bad of [null,[],{...a,bytes:-1},{...a,bytes:Number.MAX_SAFE_INTEGER+1},{...a,bytes:3.5},{...a,sha256:'A'.repeat(64)},{...a,sha256:['a'.repeat(64)]},{...a,repository:'cuwacunu_embedding'},...['../a','/a','a//b','a/./b','a\\b','a:b','a\tb','a\nb',''].map(path=>({...a,path}))]){
    assert.throws(()=>planFinalVerification([a,bad],[]));
  }
  // No first-pin filesystem access is allowed to mask a later shape failure.
  assert.throws(()=>verifyFinalBarrier([fixturePin('not-created-anywhere.txt'),{...a,bytes:-1}],[],{root:'/opt/mityane-geometrics/does-not-exist'}),/exact final-barrier file pin/);
  assert.throws(()=>verifyFinalBarrier([a,{...a,bytes:4}],[],{root:'/opt/mityane-geometrics/does-not-exist'}),/conflicting final-barrier/);
  assert.throws(()=>planFinalVerification(null,[]),/pin arrays/);
  assert.throws(()=>planFinalVerification([],{}),/pin arrays/);
});

test('binary and file physical modes have distinct complete identities and strict path contracts',()=>{
  const file=fixturePin('audit'),binary=binaryPin(),plan=planFinalVerification([file,file],[binary,{...binary}]);
  assert.deepEqual(plan.files,[file]);assert.deepEqual(plan.binaries,[binary]);
  assert.equal(plan.summary.unique_file_fulltuples,1);assert.equal(plan.summary.unique_binary_fulltuples,1);
  assert.equal(plan.summary.binary_pin_occurrences,2);assert.equal(plan.summary.duplicate_binary_occurrences,1);
  assert.throws(()=>planFinalVerification([binary],[]),/safe exact relative/);
  assert.throws(()=>planFinalVerification([],[file]),/named-build binary/);
  for(const path of ['/opt/mityane-geometrics/embeddings/build/other/audit',binary.path+'/nested',binary.path+'/',binary.path+'/../audit',binary.path+'\n'])assert.throws(()=>planFinalVerification([],[{...binary,path}]),/named-build binary/);
  assert.throws(()=>planFinalVerification([],[binary,{...binary,sha256:'c'.repeat(64)}]),/conflicting final-barrier/);
});

test('declared-byte accounting remains exact above the JSON safe integer range',()=>{
  const a=fixturePin('a',Number.MAX_SAFE_INTEGER),b=fixturePin('b',Number.MAX_SAFE_INTEGER);
  const plan=planFinalVerification([a,b,a],[]);
  assert.equal(plan.summary.declared_file_occurrence_bytes_decimal,'27021597764222973');
  assert.equal(plan.summary.declared_unique_file_bytes_decimal,'18014398509481982');
});

test('unique final files receive a fresh unchanged physical hash on every independent barrier',()=>{
  assert.equal(process.platform,'linux');assert.equal(fs.existsSync('/.dockerenv'),true);
  const root=fs.mkdtempSync('/opt/mityane-geometrics/report-final-unique-fixture-');
  const target=path.join(root,'fixture.txt'),content=Buffer.from('artificial metadata fixture\n');
  fs.writeFileSync(target,content,{flag:'wx'});
  const pin=fixturePin('fixture.txt',content.length,createHash('sha256').update(content).digest('hex'));
  const passed=verifyFinalBarrier([pin,{...pin}],[],{root});
  assert.equal(passed.status,'passed');assert.equal(passed.unique_file_fulltuples,1);
  const changed=Buffer.from(content);changed[0]^=1;fs.writeFileSync(target,changed,{flag:'r+'});
  assert.throws(()=>verifyFinalBarrier([pin,pin],[],{root}),/bound preparation input changed/);
  // Fixture leaf is intentionally retained in the container, separate from study outputs.
});

test('the scientific core and metric function bodies remain identical to the frozen reporting generation',()=>{
  const oldText=fs.readFileSync(new URL('./report.mjs',import.meta.url),'utf8');
  const newText=fs.readFileSync(new URL('./report_final_unique_001.mjs',import.meta.url),'utf8');
  assert.equal(createHash('sha256').update(oldText).digest('hex'),ORIGINAL_REPORT_PROGRAM.sha256);
  assert.equal(Buffer.byteLength(oldText),ORIGINAL_REPORT_PROGRAM.bytes);
  assert.equal(newText.split('export function requireReportCard')[0],oldText.split('export function requireReportCard')[0]);
  for(const [name,value] of Object.entries({compareCriterion,aggregateMarket,aggregateTempo,historicalRetention,pairedScreen,requireTrainingSummary,requireAuditTuple}))assert.equal(value.toString(),original[name].toString(),name);
  const market=marketRows([.11,.37,.91]),tempo=tempoRows(),queries={train:{standardized_fixed_query_MAE:.2,valid_rows:112},validation:{standardized_fixed_query_MAE:.2,Gaussian_NLL_F64:-.1,valid_rows:32}};
  assert.deepEqual(aggregateMarket(market),original.aggregateMarket(market));
  assert.deepEqual(aggregateTempo(tempo),original.aggregateTempo(tempo));
  assert.deepEqual(historicalRetention(goal(),aggregateMarket(market),aggregateTempo(tempo),queries),original.historicalRetention(goal(),original.aggregateMarket(market),original.aggregateTempo(tempo),queries));
  for(const rows of [pairs(),pairs().slice(1)])assert.deepEqual(pairedScreen(rows),original.pairedScreen(rows));
  assert.ok(newText.includes('for(const pin of pins)currentPin(ROOT,pin);\n    header='));
  assert.ok(newText.includes('const verification_final_barrier=verifyFinalBarrier(pins,binaryPins);'));
});

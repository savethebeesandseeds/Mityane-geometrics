// Copyright (c) 2026 Waajacu. See LICENSE.
// Source-reviewed, bounded target preparation. Does not train or decode tensors.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {admitFiles, hashAdmitted, readAdmitted, parseJSON, jsonBytes, safeRelative, canonicalDirectory,
  requireCondition as require, writeNew} from '../../../../shared/artifacts/node/v1/files.mjs';

const PREFIX='code/embeddings/evaluation/protocols/cpb_v15_objective_v1';
const BUILD='/opt/mityane-geometrics/embeddings/build/cpb-v15-objective-001';
const ATTEMPT='.temp/embeddings/runs/cpb-v15-objective-001/attempt-001';
const REQUIRED=[`${PREFIX}/prepare.mjs`,`${PREFIX}/target_prepare_main.cpp`,
  `${PREFIX}/include/market_data.h`,`${PREFIX}/deps/src/fixed_regression.cpp`,
  `${PREFIX}/deps/include/cuwacunu/evaluation/fixed_regression.h`,'code/shared/artifacts/node/v1/files.mjs'];
const passed=(value,message)=>require(value?.passed===true&&value.status==='passed',message);
const samePin=(a,b)=>a?.path===b?.path&&a.bytes===b.bytes&&a.sha256===b.sha256;
export function currentPin(root,expected,{binary=false}={}){
  require(expected&&Number.isSafeInteger(expected.bytes)&&expected.bytes>=0&&/^[0-9a-f]{64}$/.test(expected.sha256??''),'exact file pin required');
  require(expected.repository===undefined||expected.repository==='Mityane-geometrics','current physical pins require Mityane repository context');
  let absolute;
  if(binary){require(typeof expected.path==='string'&&path.dirname(expected.path)===BUILD,'one actual named-build binary leaf required');absolute=expected.path;}
  else {safeRelative(expected.path);absolute=path.join(root,expected.path);}
  const row=admitFiles([absolute],{root:binary?BUILD:root,singleLink:true})[0];
  require(row.bytes===expected.bytes&&hashAdmitted(row)===expected.sha256,`bound preparation input changed: ${expected.path}`);
  return {repository:'Mityane-geometrics',path:expected.path,bytes:expected.bytes,sha256:expected.sha256};
}
function pinFile(root,relative){const row=admitFiles([path.join(root,safeRelative(relative))],{root,singleLink:true})[0];return {repository:'Mityane-geometrics',path:relative,bytes:row.bytes,sha256:hashAdmitted(row)};}
function readPinned(root,pin){currentPin(root,pin);return parseJSON(readAdmitted(admitFiles([path.join(root,pin.path)],{root,singleLink:true})[0]));}
export function requirePreparationCard(card){
  require(card?.kind==='cpb_v15_target_preparation_inputs_v1'&&card.root==='/mityane'&&card.attempt===ATTEMPT,'exact target-preparation card/root/attempt required');
  require(['target_precision_parity','confirmation_targets'].includes(card.phase),'only target parity or blind confirmation targets admitted here');
  require(Array.isArray(card.january_caches)&&card.january_caches.length===3&&Array.isArray(card.original_targets)&&card.original_targets.length===2,'ordered three January caches and original112/32 target pins required');
  for(const [index,symbol] of ['BTCUSDT','ETHUSDT','SOLUSDT'].entries())require(card.january_caches[index].path.endsWith(`/data/mkt-hist-1/csv/${symbol}-1m-2024-01.cache.bin`),'exact ordered January source roles');
  for(const [index,role] of ['train','validation'].entries())require(card.original_targets[index].path===`${card.shared}/features/${role}-targets.pt`,'original target roles and shared root must agree');
  safeRelative(card.shared);
  const outputs=[card.output,card.log,card.receipt];
  require(outputs.every(value=>typeof value==='string'&&path.posix.dirname(value)===ATTEMPT)&&new Set(outputs).size===3,'three different exclusive output leaves in exact attempt required');
  require(card.log.endsWith('.log')&&card.receipt.endsWith('.json')&&card.output.endsWith(card.phase==='target_precision_parity'?'.json':'.pt'),'correct preparation output formats');
  if(card.phase==='confirmation_targets'){
    require(Array.isArray(card.confirmation_caches)&&card.confirmation_caches.length===3&&card.target_parity,'blind February target preparation requires a passed pinned precision receipt');
    for(const [index,symbol] of ['BTCUSDT','ETHUSDT','SOLUSDT'].entries())require(card.confirmation_caches[index].path.endsWith(`/${symbol}-1m-2024-02.cache.bin`),'exact ordered frozen February month');
  }
  return card;
}
export function requirePrecisionReceipt(value){
  passed(value,'actual January precision parity must have passed');
  require(value.kind==='cpb_v15_target_precision_parity_v1'&&value.train_rows===112&&value.validation_rows===32&&
    value.target_precision==='legacy_f32_then_f64'&&value.all_target_values_bitwise_equal===true&&
    value.all_masks_and_ids_exact===true&&value.February_values_opened===false&&value.encoder_calls===0&&value.head_fits===0,
    'complete actual legacy target precision proof required');
}
export function requireNumericPrecisionResult(value){
  passed(value,'actual numeric parity result must have passed');
  require(value.kind==='cpb_v15_target_precision_parity_numeric_v1'&&value.train_rows===112&&value.validation_rows===32&&
    value.target_precision==='legacy_f32_then_f64'&&value.all_target_values_bitwise_equal===true&&
    value.all_masks_and_ids_exact===true&&value.February_values_opened===false&&value.encoder_calls===0&&value.head_fits===0,
    'numeric proof must establish exact pre-February legacy precision parity');
}
function bindPrecisionChain(root,pin,card){
  const parity=readPinned(root,pin);requirePrecisionReceipt(parity);
  require(Array.isArray(parity.sources)&&parity.sources.length>0&&Array.isArray(parity.january_caches)&&parity.january_caches.length===3&&Array.isArray(parity.original_targets)&&parity.original_targets.length===2,'precision proof must retain complete source/raw/target closure');
  for(const key of ['source_review','protocol_review','protocol','build_admission'])require(samePin(parity[key],card[key]),`precision proof must bind current preparation ${key}`);
  require(samePin(parity.binary,card.binary),'precision proof must bind the current reviewed target preparation binary');
  require(parity.january_caches.every((p,i)=>samePin(p,card.january_caches[i]))&&parity.original_targets.every((p,i)=>samePin(p,card.original_targets[i])),'precision proof must establish the exact January inputs for this preparation');
  const pins=[pin,...parity.sources,...parity.january_caches,...parity.original_targets,parity.inputs,parity.admission,parity.source_review,parity.protocol_review,parity.protocol,parity.build_admission,parity.numeric_result,parity.log];
  for(const p of pins)currentPin(root,p);
  currentPin(root,parity.binary,{binary:true});
  const source=readPinned(root,parity.source_review),review=readPinned(root,parity.protocol_review),build=readPinned(root,parity.build_admission),admission=readPinned(root,parity.admission);
  passed(source,'precision source review must have passed');passed(review,'precision protocol review must have passed');passed(build,'precision build admission must have passed');
  require(Array.isArray(source.files)&&source.files.length===parity.sources.length&&parity.sources.every(p=>source.files.filter(s=>samePin(s,p)).length===1)&&REQUIRED.every(name=>source.files.some(p=>p.path===name)),'precision source closure must equal its passed source review');
  require(samePin(review.source_review,parity.source_review)&&samePin(review.protocol,parity.protocol)&&review.approved_phases?.includes('target_precision_parity'),'precision protocol/source generation mismatch');
  require(samePin(build.source_review,parity.source_review)&&build.binaries?.some(p=>p.role==='target_prepare'&&samePin(p,parity.binary)),'precision binary/source generation mismatch');
  require(samePin(admission.protocol,parity.protocol)&&samePin(admission.source_review,parity.source_review)&&admission.admitted_phases?.includes('target_precision_parity'),'precision admission must bind exact passed phase/source');
  requireNumericPrecisionResult(readPinned(root,parity.numeric_result));
  return {pins,binary:parity.binary};
}
export function prepareTargets(root,inputsPath){
  const inputs=pinFile(root,inputsPath),card=requirePreparationCard(readPinned(root,inputs));
  canonicalDirectory(path.join(root,ATTEMPT),{root});
  const receiptPath=path.join(root,card.receipt),logPath=path.join(root,card.log);
  for(const value of [card.output,card.log,card.receipt])require(!fs.existsSync(path.join(root,value)),`preserve earlier preparation output: ${value}`);
  const bindings={inputs};let sources=[],result=null,log=null,binary=null;
  try{
    const sourceReview=readPinned(root,card.source_review),protocolReview=readPinned(root,card.protocol_review);
    passed(sourceReview,'actual preparation source review must have passed');passed(protocolReview,'actual protocol preparation review must have passed');
    const build=readPinned(root,card.build_admission),admission=readPinned(root,card.admission);
    passed(build,'actual preparation build admission must have passed');
    require(samePin(protocolReview.protocol,card.protocol)&&samePin(protocolReview.source_review,card.source_review),'exact declaration/source generation in protocol review required');
    require(samePin(build.source_review,card.source_review)&&Array.isArray(build.binaries)&&build.binaries.some(row=>row.role==='target_prepare'&&samePin(row,card.binary)),'preparation executable must bind actual reviewed source build');
    require(Array.isArray(protocolReview.approved_phases)&&protocolReview.approved_phases.includes(card.phase),'protocol review must explicitly admit this exact preparation phase');
    require(samePin(admission.protocol,card.protocol)&&samePin(admission.source_review,card.source_review)&&Array.isArray(admission.admitted_phases)&&admission.admitted_phases.includes(card.phase),'actual preparation admission must bind exact declaration/source and phase');
    require(Array.isArray(sourceReview.files),'source closure required');
    sources=sourceReview.files.map(pin=>currentPin(root,pin));
    require(REQUIRED.every(name=>sources.some(pin=>pin.path===name)),'reviewed preparation implementation closure incomplete');
    bindings.source_review=currentPin(root,card.source_review);bindings.protocol_review=currentPin(root,card.protocol_review);
    bindings.protocol=currentPin(root,card.protocol);bindings.build_admission=currentPin(root,card.build_admission);
    bindings.admission=currentPin(root,card.admission);
    binary=currentPin(root,card.binary,{binary:true});bindings.binary=binary;
    const january=card.january_caches.map(pin=>currentPin(root,pin)),originalTargets=card.original_targets.map(pin=>currentPin(root,pin));
    // A proof failure must occur before even hashing February cache bytes here.
    const precision=card.phase==='confirmation_targets'?bindPrecisionChain(root,card.target_parity,card):null;
    const caches=card.phase==='target_precision_parity'?january:card.confirmation_caches.map(pin=>currentPin(root,pin));
    const args=[card.phase==='target_precision_parity'?'parity':'confirmation'];
    for(const [index,symbol] of ['btc','eth','sol'].entries())args.push(`--cache-${symbol}`,path.join(root,caches[index].path));
    args.push('--output',path.join(root,card.output));
    if(card.phase==='target_precision_parity')args.push('--shared',path.join(root,card.shared));
    else {
      bindings.target_parity=currentPin(root,card.target_parity);
      args.push('--target-parity',path.join(root,card.target_parity.path),'--target-parity-sha256',card.target_parity.sha256);
    }
    const descriptor=fs.openSync(logPath,'wx');let processResult;
    try{processResult=spawnSync(binary.path,args,{cwd:root,env:{...process.env,OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1'},stdio:['ignore',descriptor,descriptor],timeout:120000});}
    finally{fs.closeSync(descriptor);}
    log=pinFile(root,card.log);bindings.log=log;
    require(!processResult.error&&processResult.status===0,`target preparation failed: ${processResult.error?.message??`exit ${processResult.status}`}`);
    result=pinFile(root,card.output);
    let numeric={};
    if(card.phase==='target_precision_parity'){
      numeric=readPinned(root,result);requireNumericPrecisionResult(numeric);
    }
    for(const pin of [inputs,card.admission,card.source_review,card.protocol_review,card.protocol,card.build_admission,...sources,...january,...originalTargets,...caches,...(precision?.pins??[])])currentPin(root,pin);
    if(precision)currentPin(root,precision.binary,{binary:true});
    currentPin(root,binary,{binary:true});
    const receipt={...numeric,kind:card.phase==='target_precision_parity'?'cpb_v15_target_precision_parity_v1':'cpb_v15_confirmation_targets_prepared_v1',status:'passed',passed:true,
      created_utc:new Date().toISOString(),...bindings,binary,sources,january_caches:january,original_targets:originalTargets,
      ...(card.phase==='target_precision_parity'?{numeric_result:result}:{targets:result,confirmation_caches:caches,blind:true,numerical_outcomes_exposed:false}),
      encoder_calls:0,head_fits:0,python_executed:false};
    writeNew(receiptPath,jsonBytes(receipt));return receipt;
  }catch(error){
    writeNew(receiptPath,jsonBytes({kind:'cpb_v15_target_preparation_failure_v1',status:'failed',passed:false,created_utc:new Date().toISOString(),...bindings,sources,binary,numeric_result:result,log,failures:[String(error.message)],scientific_result_claimed:false}));throw error;
  }
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  try{
    require(process.platform==='linux'&&fs.existsSync('/.dockerenv')&&process.argv.length===4&&process.argv[2]==='--inputs','managed Linux preparation: --inputs relative-card.json');
    safeRelative(process.argv[3]);const result=prepareTargets('/mityane',process.argv[3]);process.stdout.write(jsonBytes({status:result.status,kind:result.kind}));
  }catch(error){process.stderr.write(`${error.stack}\n`);process.exitCode=1;}
}

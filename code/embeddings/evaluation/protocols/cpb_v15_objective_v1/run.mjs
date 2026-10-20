// Copyright (c) 2026 Waajacu. See LICENSE.
// One declared paired study. Completed leaves are immutable; no retry or rescue.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {admitFiles, hashAdmitted, readAdmitted, parseJSON, jsonBytes, safeRelative,
  canonicalDirectory, requireCondition as require, writeNew} from '../../../../shared/artifacts/node/v1/files.mjs';
import {currentPin} from './prepare.mjs';

const ROOT='/mityane';
export const ATTEMPT='.temp/embeddings/runs/cpb-v15-objective-001/attempt-001';
export const SEEDS=Object.freeze([945101,945102,945103]);
export const ARMS=Object.freeze(['control','candidate']);
const PREFIX='code/embeddings/evaluation/protocols/cpb_v15_objective_v1';
const REPS=[2701,2802,2903],MASTERS=[941101,941102,941103];
const passed=(v,message)=>require(v?.passed===true&&v.status==='passed',message);
const samePin=(a,b)=>a?.path===b?.path&&a.bytes===b.bytes&&a.sha256===b.sha256;
const local=relative=>path.join(ROOT,safeRelative(relative));
function observed(relative){const row=admitFiles([local(relative)],{root:ROOT,singleLink:true})[0];return {repository:'Mityane-geometrics',path:relative,bytes:row.bytes,sha256:hashAdmitted(row)};}
function read(pin){currentPin(ROOT,pin);return parseJSON(readAdmitted(admitFiles([local(pin.path)],{root:ROOT,singleLink:true})[0]));}
function leaf(relative){require(relative.startsWith(`${ATTEMPT}/`),'only admitted attempt outputs');return local(relative);}
export function canonicalParents(parent,root=ROOT){
  canonicalDirectory(root);
  const missing=[];let existing=parent;
  while(!fs.existsSync(existing)){missing.unshift(existing);existing=path.dirname(existing);}
  canonicalDirectory(existing,{root});
  for(const directory of missing){fs.mkdirSync(directory);canonicalDirectory(directory,{root});}
  return canonicalDirectory(parent,{root});
}
function newParents(relative){canonicalParents(path.dirname(leaf(relative)));}
function collect(relative){
  const entries=[];
  for(const row of fs.readdirSync(leaf(relative),{withFileTypes:true})){
    const name=`${relative}/${row.name}`;require(!row.isSymbolicLink(),'output symlinks refused');
    if(row.isDirectory())entries.push(...collect(name));else {require(row.isFile(),'regular output files only');entries.push(observed(name));}
  }
  return entries.sort((a,b)=>a.path.localeCompare(b.path));
}
export function subrun(seed,arm){require(SEEDS.includes(seed)&&ARMS.includes(arm),'exact three paired seeds and two arms');return `${ATTEMPT}/subruns/seed-${seed}/${arm}`;}
export function requireRunCard(card){
  require(card?.kind==='cpb_v15_objective_run_inputs_v1'&&card.root===ROOT&&card.attempt===ATTEMPT,'actual declared paired-study root/card required');
  require(Array.isArray(card.seeds)&&card.seeds.length===3&&card.seeds.every((v,i)=>v===SEEDS[i]),'unchanged three declared initialization seeds');
  require(card.stream_offsets?.row===1000&&card.stream_offsets.mask===2000&&card.stream_offsets.context===3000,'unchanged paired independent stream families');
  require(card.training?.updates===2048&&card.training.batch_size===8&&card.training.global_gradient_clip===1&&card.training.learning_rate===.001&&card.training.weight_decay===.0001,'fixed2048 B8 singleclip1 AdamW recipe');
  require(card.native_width===36&&card.mean_huber_delta===1&&card.mean_loss_weight===1&&card.variance_loss_weight===1,'fixed native36/unit composite mechanism');
  require(card.confirmation?.month==='2024-02'&&card.confirmation.rows===32&&card.confirmation.training_use===false&&card.confirmation.head_refits===false&&card.confirmation.descriptive_reconstruction_only===true,'untouched fixed32 February confirmation with no fits or reconstruction gate');
  require(card.python_execution===false&&card.old_encoder_replay===false&&card.checkpoint_selection===false,'no old replay, Python or checkpoint selection');
  require(Array.isArray(card.january_caches)&&card.january_caches.length===3&&Array.isArray(card.input_files)&&card.input_files.length>=14,'explicit raw/shared/TEMPO input closure');
  safeRelative(card.shared);safeRelative(card.synthetic_inputs);
  for(const [i,symbol] of ['BTCUSDT','ETHUSDT','SOLUSDT'].entries())require(card.january_caches[i].path.endsWith(`/data/mkt-hist-1/csv/${symbol}-1m-2024-01.cache.bin`),'ordered unchanged January raw sources');
  const requiredInputs=['selections.pt','query-plan-train.pt','query-plan-validation.pt','features/train-targets.pt','features/validation-targets.pt'].map(name=>`${card.shared}/${name}`);
  for(const master of MASTERS)for(const role of ['training.pt','validation.pt','validation-deleted.pt'])requiredInputs.push(`${card.synthetic_inputs}/master-${master}/${role}`);
  require(requiredInputs.every(name=>card.input_files.filter(pin=>pin.path===name).length===1),'complete exact shared and three TEMPO cohort roles required');
  require(card.confirmation.query_geometry?.path===`${card.shared}/query-plan-validation.pt`,'confirmation reuses only original known-VALIDATION Boolean geometry');
  for(const arm of ARMS)require(card.settings?.[arm]&&typeof card.gradient_policies?.[arm]==='string','settings and exact compiled gradient policy per arm');
  return card;
}
function bindStudy(card,phase){
  const source=read(card.source_review),protocol=read(card.protocol_review),engineering=read(card.engineering_review),build=read(card.build_admission),admission=read(card.admission);
  passed(source,'actual independent source review must have passed');passed(protocol,'actual independent protocol review must have passed');passed(engineering,'actual CUDA engineering review must have passed');passed(build,'actual reviewed build admission must have passed');
  require(samePin(protocol.protocol,card.protocol)&&samePin(protocol.source_review,card.source_review),'protocol review must bind this exact declaration and source generation');
  require(samePin(build.source_review,card.source_review)&&samePin(engineering.source_review,card.source_review)&&samePin(engineering.build_admission,card.build_admission),'actual executable build and CUDA engineering must bind reviewed current source');
  require(samePin(admission.protocol,card.protocol)&&Array.isArray(admission.admitted_phases)&&admission.admitted_phases.includes(phase),'actual human admission must bind this declaration and exact phase');
  require(Array.isArray(build.binaries),'actual build binary role bindings required');
  for(const [role,pin] of Object.entries(card.binaries)){require(build.binaries.filter(p=>p.role===role&&samePin(p,pin)).length===1,`binary must be bound by reviewed actual build: ${role}`);currentPin(ROOT,pin,{binary:true});}
  require(engineering.actual_CUDA===true&&engineering.fresh_pair_initialization_equal===true&&engineering.objective_and_gradient_routes_verified===true,'nonvacuous actual CUDA paired initialization/objective engineering required');
  require(Array.isArray(protocol.approved_phases)&&protocol.approved_phases.includes(phase),'exact execution phase must be admitted by reviewed protocol');
  require(Array.isArray(source.files)&&['run.mjs','prepare.mjs','train_main.cpp','extract_main.cpp','confirmation_extract_main.cpp','confirmation_read_main.cpp','market_read_main.cpp','synthetic_read_main.cpp','deps/src/fixed_regression.cpp','deps/src/fixed_feature_readouts.cpp','deps/src/feature_harness.cpp'].every(name=>source.files.some(p=>p.path===`${PREFIX}/${name}`)),'reviewed runner/producer/reader source closure required');
  const pins=[card.admission,card.resumption,card.protocol,card.input_manifest,card.source_review,card.protocol_review,card.engineering_review,card.build_admission,...source.files,...card.january_caches,...card.input_files,...Object.values(card.settings)];
  for(const pin of pins)currentPin(ROOT,pin);
  const inputManifest=read(card.input_manifest);
  require(inputManifest.kind==='cpb_v15_objective_input_manifest_v1'&&Array.isArray(inputManifest.inputs)&&[...card.january_caches,...card.input_files,...Object.values(card.settings)].every(pin=>inputManifest.inputs.filter(p=>samePin(p,pin)).length===1),'exact frozen input manifest must bind every runtime data/settings role');
  return {pins,source};
}
function binary(card,role){require(card.binaries?.[role],`declared binary role missing: ${role}`);return currentPin(ROOT,card.binaries[role],{binary:true});}
function child(card,role,args,log){
  const executable=binary(card,role);newParents(log);const fd=fs.openSync(leaf(log),'wx');let result;
  process.stdout.write(jsonBytes({phase:'starting',role,log}));
  try{result=spawnSync(executable.path,args,{cwd:ROOT,env:{...process.env,OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1'},stdio:['ignore',fd,fd],timeout:1800000});}
  finally{fs.closeSync(fd);}
  currentPin(ROOT,executable,{binary:true});
  const call={binary:executable,arguments:args,log:observed(log),exit_code:result.status,process_error:result.error?.message??null};
  if(result.error||result.status!==0){const error=new Error(`${role} failed; preserve ${log}: ${result.error?.message??`exit ${result.status}`}`);error.failed_call=call;throw error;}
  return call;
}
function machine(pin){const value=read(pin);require(value.status==='passed','actual producer/proof record must pass');return value;}
function saveReceipt(relative,value){writeNew(leaf(relative),jsonBytes(value));return observed(relative);}
function phaseReceipt(base,phase){return `${base}/${phase}-receipt.json`;}
function finalFitPaths(base){return REPS.map(rep=>`${base}/heads-final/market/models/native36/rep-${rep}/fit.pt`);}
function checkExistingReceipt(relative){const pin=observed(relative),data=read(pin);passed(data,`actual ${relative} must have passed`);require(Array.isArray(data.outputs)&&data.outputs.length>0,'actual receipt must retain complete nonempty outputs');for(const p of data.outputs)currentPin(ROOT,p);return {pin,data};}
export function requirePhaseReceipt(value,cardPin,phase,seed,arm){
  passed(value,'actual subrun phase must have passed');
  require(value.kind==='cpb_v15_objective_subrun_phase_v1'&&samePin(value.card,cardPin)&&value.phase===phase&&value.seed===seed&&value.arm===arm&&Array.isArray(value.outputs)&&value.outputs.length>0,'subrun receipt must bind exact card, phase, seed, arm and outputs');
}
export function requireDevelopmentSeal(value,cardPin){
  passed(value,'actual all-six development seal must have passed');
  require(value.kind==='cpb_v15_all_six_development_runs_sealed_v1'&&value.subruns===6&&samePin(value.card,cardPin)&&value.independent_numeric_audits===6&&value.confirmation_numerical_outcomes_opened===false,'all six development runs, checkpoints and heads must be sealed before any confirmation output');
  require(Array.isArray(value.records)&&value.records.length===12&&Array.isArray(value.audits)&&value.audits.length===6&&Array.isArray(value.outputs)&&value.outputs.length>0,'seal requires all twelve phase receipts, six audits and actual output closure');
  const records=SEEDS.flatMap(seed=>ARMS.flatMap(arm=>['train','development'].map(phase=>phaseReceipt(subrun(seed,arm),phase)))),audits=SEEDS.flatMap(seed=>ARMS.map(arm=>`${subrun(seed,arm)}/independent-audit.json`));
  require(records.every(name=>value.records.filter(p=>p.path===name).length===1)&&audits.every(name=>value.audits.filter(p=>p.path===name).length===1)&&new Set(value.outputs.map(p=>p.path)).size===value.outputs.length,'seal must cover exactly the declared six subruns with unique outputs');
  const required=SEEDS.flatMap(seed=>ARMS.flatMap(arm=>{const base=subrun(seed,arm);return [`${base}/training/checkpoints/inference-2048.pt`,...finalFitPaths(base)];}));
  require(required.every(name=>value.outputs.filter(p=>p.path===name).length===1),'all six final checkpoints and eighteen original TRAIN fits must be inside the seal');
  return value;
}
function allTraining(cardPin){
  const rows=[];for(const seed of SEEDS)for(const arm of ARMS){const receipt=checkExistingReceipt(phaseReceipt(subrun(seed,arm),'train'));requirePhaseReceipt(receipt.data,cardPin,'train',seed,arm);rows.push(receipt);}
  return rows;
}
function sealBindings(seal,cardPin){
  requireDevelopmentSeal(seal.data,cardPin);
  for(const pin of [...seal.data.records,...seal.data.audits])currentPin(ROOT,pin);
  return [...seal.data.records,...seal.data.audits,...seal.data.outputs];
}

export function executeSubrun(card,cardPin,phase,seed,arm){
  const base=subrun(seed,arm),receipt=phaseReceipt(base,phase);newParents(receipt);
  require(!fs.existsSync(leaf(receipt)),'preserve earlier subrun receipt');
  const calls=[],outputs=[],dependencies=[];let binding;
  try{
    binding=bindStudy(card,phase);
    const settings=local(card.settings[arm].path);
    if(phase==='train'){
      calls.push(child(card,`${arm}_train`,['--settings',settings,'--shared',local(card.shared),
        ...card.january_caches.flatMap(p=>['--cache',local(p.path)]),'--seed',String(seed),'--output',leaf(`${base}/training`)],`${base}/training.log`));
      const summary=machine(observed(`${base}/training/training-summary.json`));
      require(summary.new_causal_updates===2048&&summary.batch_size===8&&summary.initialization_seed===seed&&summary.row_seed===seed+1000&&summary.mask_seed===seed+2000&&summary.context_seed===seed+3000&&summary.gradient_policy===card.gradient_policies[arm]&&summary.typed_resume_512_exact===true&&summary.old_encoder_calls===0&&summary.head_fits===0,'actual declared paired fresh training/streams/resume required');
      outputs.push(...collect(`${base}/training`));
    }else if(phase==='development'){
      // All training is completed before the first development extraction/head fit.
      for(const row of allTraining(cardPin))dependencies.push(row.pin,...row.data.outputs);
      const training=checkExistingReceipt(phaseReceipt(base,'train'));
      requirePhaseReceipt(training.data,cardPin,'train',seed,arm);
      for(const [stage,updates] of [['initial',0],['final',2048]]){
        const checkpoint=`${base}/training/checkpoints/inference-${updates}.pt`,cp=observed(checkpoint);
        calls.push(child(card,`${arm}_proof`,['--settings',settings,'--output',leaf(`${base}/proof-${stage}`),'--saved-checkpoint',leaf(checkpoint)],`${base}/proof-${stage}.log`));
        const proof=machine(observed(`${base}/proof-${stage}/proof.json`));
        require(proof.checkpoint_sha256===cp.sha256&&proof.initialization_seed===seed&&proof.source_updates===0&&proof.causal_updates===updates&&proof.new_training_updates===0&&proof.gradient_policy===card.gradient_policies[arm]&&proof.all_cutoffs===128&&proof.suffix_cases===2350&&proof.internal_temporal_cases===30&&proof.state_unchanged===true,'actual initial/final checkpoint causal proof must bind scored seed/policy/bytes');
      }
      calls.push(child(card,`${arm}_extract`,['--settings',settings,'--initial-checkpoint',leaf(`${base}/training/checkpoints/inference-0.pt`),'--final-checkpoint',leaf(`${base}/training/checkpoints/inference-2048.pt`),'--output',leaf(`${base}/features`),'--shared',local(card.shared),'--synthetic-inputs',local(card.synthetic_inputs),...['btc','eth','sol'].flatMap((s,i)=>[`--cache-${s}`,local(card.january_caches[i].path)])],`${base}/extraction.log`));
      for(const stage of ['initial','final']){
        const headBase=`${base}/heads-${stage}`;fs.mkdirSync(leaf(headBase));
        calls.push(child(card,'market_read',['--train-targets',local(`${card.shared}/features/train-targets.pt`),'--validation-targets',local(`${card.shared}/features/validation-targets.pt`),'--method','native36',leaf(`${base}/features/${stage}/market/cpb-objective-train.pt`),leaf(`${base}/features/${stage}/market/cpb-objective-validation.pt`),'--output-directory',leaf(`${headBase}/market`),'--card-sha256',cardPin.sha256,'--input-manifest-sha256',card.input_manifest.sha256,'--dataset-id','MKT-HIST-2'],`${base}/market-${stage}.log`));
        calls.push(child(card,'synthetic_read',['--features',leaf(`${base}/features/${stage}/synthetic`),'--output',leaf(`${headBase}/synthetic`),'--stage',stage],`${base}/synthetic-${stage}.log`));
      }
      for(const name of ['proof-initial','proof-final','features','heads-initial','heads-final'])outputs.push(...collect(`${base}/${name}`));
    }else if(phase==='confirmation'){
      const seal=checkExistingReceipt(`${ATTEMPT}/development-seal.json`);
      dependencies.push(seal.pin,...sealBindings(seal,cardPin));
      const prepared=read(card.confirmation.targets_receipt);passed(prepared,'actual blind confirmation targets preparation required');
      require(prepared.kind==='cpb_v15_confirmation_targets_prepared_v1'&&prepared.blind===true&&prepared.numerical_outcomes_exposed===false,'confirmation targets must retain blind preparation boundary');
      const caches=card.confirmation.caches;require(Array.isArray(caches)&&caches.length===3,'ordered February caches');
      require(Array.isArray(prepared.confirmation_caches)&&prepared.confirmation_caches.length===3&&caches.every((p,i)=>samePin(p,prepared.confirmation_caches[i])),'prepared targets and confirmation encoder must bind the same ordered February cache generation');
      for(const [i,symbol] of ['BTCUSDT','ETHUSDT','SOLUSDT'].entries())require(caches[i].path.endsWith(`/${symbol}-1m-2024-02.cache.bin`),'fixed ordered February confirmation data');
      dependencies.push(card.confirmation.targets_receipt,prepared.targets,...caches,card.confirmation.query_geometry);
      for(const pin of [...caches,prepared.targets,card.confirmation.query_geometry])currentPin(ROOT,pin);
      const checkpoint=observed(`${base}/training/checkpoints/inference-2048.pt`);
      require(seal.data.outputs.some(pin=>samePin(pin,checkpoint)),'confirmation checkpoint must be the exact sealed final bytes');
      calls.push(child(card,`${arm}_confirmation_extract`,['--settings',settings,'--checkpoint',leaf(checkpoint.path),'--checkpoint-sha256',checkpoint.sha256,'--targets',local(prepared.targets.path),'--query-geometry',local(card.confirmation.query_geometry.path),...['btc','eth','sol'].flatMap((s,i)=>[`--cache-${s}`,local(caches[i].path)]),'--seal',leaf(seal.pin.path),'--seal-sha256',seal.pin.sha256,'--output',leaf(`${base}/confirmation-features`)],`${base}/confirmation-extraction.log`));
      fs.mkdirSync(leaf(`${base}/confirmation-readouts`));
      for(const [i,fitPath] of finalFitPaths(base).entries()){
        const fit=observed(fitPath);
        require(seal.data.outputs.some(pin=>samePin(pin,fit)),'confirmation fit must be the exact sealed TRAIN-fitted bytes');
        calls.push(child(card,'confirmation_read',['--fit',leaf(fitPath),'--features',leaf(`${base}/confirmation-features/confirmation-features.pt`),'--targets',local(prepared.targets.path),'--output',leaf(`${base}/confirmation-readouts/rep-${REPS[i]}`),'--seal',leaf(seal.pin.path),'--seal-sha256',seal.pin.sha256,'--fit-sha256',fit.sha256],`${base}/confirmation-readout-${REPS[i]}.log`));
      }
      outputs.push(...collect(`${base}/confirmation-features`),...collect(`${base}/confirmation-readouts`));
    }else require(false,'unknown subrun phase');
    outputs.push(...calls.map(call=>call.log));
    for(const pin of [cardPin,...binding.pins,...dependencies,...outputs])currentPin(ROOT,pin);
    return saveReceipt(receipt,{kind:'cpb_v15_objective_subrun_phase_v1',status:'passed',passed:true,created_utc:new Date().toISOString(),card:cardPin,phase,seed,arm,calls,dependencies,outputs,python_executed:false});
  }catch(error){if(error.failed_call)calls.push(error.failed_call);saveReceipt(receipt,{kind:'cpb_v15_objective_subrun_phase_v1',status:'failed',passed:false,created_utc:new Date().toISOString(),card:cardPin,phase,seed,arm,calls,outputs,failures:[String(error.message)],existing_partial_outputs_preserved:true});throw error;}
}
export function sealDevelopment(card,cardPin){
  const bindings=bindStudy(card,'seal'),records=[],outputs=[],audits=[];
  for(const seed of SEEDS)for(const arm of ARMS){
    const base=subrun(seed,arm),training=checkExistingReceipt(phaseReceipt(base,'train')),development=checkExistingReceipt(phaseReceipt(base,'development'));
    requirePhaseReceipt(training.data,cardPin,'train',seed,arm);requirePhaseReceipt(development.data,cardPin,'development',seed,arm);
    const auditPin=observed(`${base}/independent-audit.json`),audit={pin:auditPin,data:read(auditPin)};passed(audit.data,'actual independent numeric audit must have passed');
    require(audit.data.kind==='cpb_v15_independent_numeric_audit_v1'&&audit.data.independent_arithmetic===true&&audit.data.producer_module_forwards===0&&audit.data.producer_arithmetic_called===false&&Array.isArray(audit.data.checked_files),'actual independent saved-result arithmetic review per subrun required');
    const expected=[...training.data.outputs,...development.data.outputs];
    for(const pin of expected)require(audit.data.checked_files.some(p=>p.path===pin.path&&p.bytes===pin.bytes&&p.sha256===pin.sha256),'independent audit must bind complete actual subrun outputs');
    for(const pin of audit.data.checked_files)currentPin(ROOT,pin);
    records.push(training.pin,development.pin);audits.push(audit.pin);outputs.push(...expected);
  }
  const unique=[...new Map(outputs.map(p=>[p.path,p])).values()].sort((a,b)=>a.path.localeCompare(b.path));
  for(const pin of [...bindings.pins,...records,...audits,...unique,cardPin])currentPin(ROOT,pin);
  const value={kind:'cpb_v15_all_six_development_runs_sealed_v1',status:'passed',passed:true,created_utc:new Date().toISOString(),card:cardPin,subruns:6,independent_numeric_audits:6,records,audits,outputs:unique,confirmation_numerical_outcomes_opened:false};
  requireDevelopmentSeal(value,cardPin);return saveReceipt(`${ATTEMPT}/development-seal.json`,value);
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  try{
    require(process.platform==='linux'&&fs.existsSync('/.dockerenv'),'managed Linux /mityane container only');
    const flags={};for(let i=2;i<process.argv.length;i+=2){require(['--inputs','--phase','--seed','--arm'].includes(process.argv[i])&&process.argv[i+1]&&!Object.hasOwn(flags,process.argv[i]),'exact paired study flags');flags[process.argv[i]]=process.argv[i+1];}
    require(flags['--inputs']&&flags['--phase'],'--inputs and --phase required');safeRelative(flags['--inputs']);
    const cardPin=observed(flags['--inputs']),card=requireRunCard(read(cardPin));
    const pin=flags['--phase']==='seal'?(require(Object.keys(flags).length===2,'seal takes no subrun selector'),sealDevelopment(card,cardPin)):(require(Object.keys(flags).length===4,'seed and arm required'),executeSubrun(card,cardPin,flags['--phase'],Number(flags['--seed']),flags['--arm']));
    process.stdout.write(jsonBytes({status:'passed',receipt:pin}));
  }catch(error){process.stderr.write(`${error.stack}\n`);process.exitCode=1;}
}

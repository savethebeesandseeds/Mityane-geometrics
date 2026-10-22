// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// One prospective mask-policy comparison; native execution stays in the managed container.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

export const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../../../..');
export const BASE='.temp/embeddings/runs/cpb-v18-span-001';
export const ATTEMPT=`${BASE}/attempt-001`;
export const OLD='.temp/embeddings/runs/cpb-v17-balance-001/attempt-001';
export const FAMILY='code/embeddings/encoders/causal_patch_bottleneck_mae';
export const PROTOCOL='code/embeddings/evaluation/protocols/cpb_v18_span_v1';
export const SETTINGS=`${FAMILY}/versions/v17/config/research.conf`;
export const BUILD='/opt/mityane-geometrics/embeddings/build/cpb-v18-span-001';
export const OLD_BUILD='/opt/mityane-geometrics/embeddings/build/cpb-v17-balance-001';
export const SEEDS=[947101,947102,947103];
export const TASKS=['past_volatility','past_trend','past_activity','past_relation'];
export const SCHEDULE=[[947101,'control'],[947101,'candidate'],[947102,'candidate'],[947102,'control'],[947103,'control'],[947103,'candidate']]
  .map(([seed,arm])=>({seed,arm,id:`seed-${seed}-${arm}`,mask_mode:arm==='control'?'scattered':'contiguous',temporal_weight:.1}));
export const assert=(condition,message)=>{if(!condition)throw Error(message);};
export const absolute=name=>path.isAbsolute(name)?name:path.join(ROOT,name);
export const relative=name=>name.startsWith(`${ROOT}/`)?path.relative(ROOT,name).split(path.sep).join('/'):name;
export const read=name=>JSON.parse(fs.readFileSync(absolute(name),'utf8'));
export const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
export function pin(name){const p=absolute(name),s=fs.lstatSync(p);assert(s.isFile()&&!s.isSymbolicLink(),`regular nonsymlink file required: ${name}`);
  const bytes=fs.readFileSync(p);assert(bytes.length===s.size,`file changed while reading: ${name}`);return {path:relative(p),bytes:bytes.length,sha256:hash(bytes)};}
export function check(expected){const actual=pin(expected.path);assert(actual.bytes===expected.bytes&&actual.sha256===expected.sha256,`bound bytes changed: ${expected.path}`);return actual;}
export function save(name,value){fs.mkdirSync(path.dirname(absolute(name)),{recursive:true});const fd=fs.openSync(absolute(name),'wx');
  try{fs.writeFileSync(fd,JSON.stringify(value,null,2)+'\n');fs.fsyncSync(fd);}finally{fs.closeSync(fd);}}
export function filesUnder(name){return fs.readdirSync(absolute(name),{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name)).flatMap(e=>{
  assert(!e.isSymbolicLink(),`source symlink refused: ${name}/${e.name}`);return e.isDirectory()?filesUnder(`${name}/${e.name}`):[`${name}/${e.name}`];});}
const allPins=names=>[...new Set(names)].sort().map(pin);
export function nativeGuard(){assert(process.platform==='linux'&&ROOT==='/mityane'&&fs.existsSync('/.dockerenv'),'Use the managed mityane-geometrics container at /mityane.');}
const directory=t=>`${ATTEMPT}/trajectories/${t.id}`;
const prefixDirectory=seed=>`${OLD}/trajectories/seed-${seed}-control`;
const OLD_RECORD='code/embeddings/evaluation/records/CPB-v17.balance-001.json';
const OLD_RECORD_SHA='2030251a86c174da78ee402a77599f282e160be23f3be603fca528fa753f5a8e';
const report=(t,stage)=>read(`${directory(t)}/${stage}/run.json`);
const stages=t=>t.arm==='control'?[['continue-640',640,256],['resume-1024',1024,640]]:[['fresh-512',512,0],['resume-1024',1024,512]];
export const trainingDataset=()=>pin(`${OLD}/data-train/calibration-TRAIN.pt`);
export function targetPins(){const inputs=read('.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/run-inputs-001.json');
  const selected=leaf=>{const found=inputs.input_files.filter(p=>p.path.endsWith(`/features/${leaf}`));assert(found.length===1,`one original target artifact required: ${leaf}`);return check(found[0]);};
  return {train:selected('train-targets.pt'),validation:selected('validation-targets.pt')};}
function baseline(){const receipt=read(`${FAMILY}/family/evidence/compatibility-2026-10-22.json`);
  assert(receipt.status==='passed','passed configurable-family compatibility receipt required');
  const pins=receipt.sourceFiles.filter(p=>p.path.startsWith(`${FAMILY}/versions/v16/`)||p.path.startsWith(`${FAMILY}/family/include/`));
  assert(pins.filter(p=>p.path.startsWith(`${FAMILY}/versions/v16/`)).length===49&&pins.filter(p=>p.path.startsWith(`${FAMILY}/family/include/`)).length===3,'49 frozen V16 files and three family numerical headers required');pins.forEach(check);return pins;}
function preservedControlSource(){
  const capsule=`${OLD}/preserved-source-at-closeout`,capture=read(`${capsule}/capture-receipt.json`),manifest=pin(capture.original_manifest.path);
  assert(capture.kind==='cpb_v17_exact_closeout_source_preservation_v1'&&capture.files===108&&capture.sources.length===108,'complete preserved V17 closeout source capsule required');
  assert(manifest.sha256===capture.original_manifest.sha256,'preserved-source original manifest changed');
  const original=read(manifest.path).sources,lookup=new Map(original.map(p=>[p.path,p]));
  const preserved=capture.sources.map(p=>{const expected=lookup.get(p.path);assert(expected&&expected.bytes===p.bytes&&expected.sha256===p.sha256,'capsule member differs from original V17 source pin');
    const copied=check({...p,path:`${capsule}/${p.path}`});return {origin:p,preserved:copied};});
  const current=original.filter(p=>p.path.startsWith(`${FAMILY}/versions/v16/`)||p.path.startsWith(`${FAMILY}/family/include/`)||p.path.startsWith(`${FAMILY}/versions/v17/include/`)
    ||p.path.startsWith('code/data/observations/include/')||p.path.startsWith('code/shared/settings/include/')||p.path.startsWith('code/shared/representation/include/')
    ||p.path.startsWith('code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/')
    ||p.path==='code/embeddings/evaluation/protocols/cpb_v17_balance_v1/audit.cpp'
    ||p.path===SETTINGS||p.path==='code/embeddings/scripts/cpb_v17_train.cpp');
  current.forEach(check);const train=read(`${OLD}/train.admission.json`);train.sources.forEach(check);train.executables.forEach(check);
  assert(train.executables.length===1&&train.executables[0].path===`${OLD_BUILD}/cpb_v17_train`,'exact original control trainer required');
  return {capture:pin(`${capsule}/capture-receipt.json`),original_manifest:manifest,preserved,current_numerical_sources:current,original_train_phase:pin(`${OLD}/train.admission.json`),executable:train.executables[0]};
}
function evaluationEquivalence(){
  const oldPhase=read(`${OLD}/readout.admission.json`),audit=oldPhase.executables.find(p=>p.path===`${OLD_BUILD}/cpb_v17_audit`);
  assert(audit,'original independent scalar audit pin required');check(audit);
  const origin=`${OLD}/preserved-source-at-closeout/code/embeddings/evaluation/protocols/cpb_v17_balance_v1/readout.cpp`,candidate=`${PROTOCOL}/readout.cpp`;
  const text=name=>fs.readFileSync(absolute(name),'utf8').replaceAll('\r\n','\n').replaceAll('[V18 readout]','[V17 readout]');
  const original=text(origin),adapted=text(candidate),regions=[];
  for(const [name,start,end] of [['feature_and_target_functions','torch::Tensor features','int run('],['TRAIN_normalization_ridge_prediction_scoring','  const auto train=ow::read','  const fs::path out=']]){
    const block=source=>{const begin=source.indexOf(start),finish=source.indexOf(end,begin+start.length);assert(begin>=0&&finish>begin,`readout source region required: ${name}`);return source.slice(begin,finish);};
    const left=block(original),right=block(adapted);assert(left===right,`fixed evaluation arithmetic changed: ${name}`);regions.push({name,normalized_source_sha256:hash(Buffer.from(left))});
  }
  return {status:'passed',basis:'Exact source-region equality after EOL and diagnostic-message version-prefix normalization only; typed loaders, update1024 admission and metadata differ outside these regions.',
    source_files:[pin(origin),pin(candidate)],regions,original_audit_phase:pin(`${OLD}/readout.admission.json`),unchanged_audit_executable:audit,
    witness_codec:'cpb_v17_ridge_witness_v1',witness_codec_meaning:'generic ridge tensor structure, not encoder identity'};
}
function prefix(seed){
  assert(pin(OLD_RECORD).sha256===OLD_RECORD_SHA,'original V17 compact closeout record changed');
  const bound=read(OLD_RECORD).trajectories.find(t=>t.seed===seed&&t.arm==='control');assert(bound,'original control trajectory record required');
  const reports=['fresh-128','resume-256'].map(stage=>{const p=`${prefixDirectory(seed)}/${stage}/run.json`,expected=bound.training.find(r=>r.stage===stage)?.report;
    assert(expected&&expected.path===p,'original prefix report binding required');check(expected);const r=read(p);
    assert(r.status==='passed'&&r.initialization_seed===seed&&r.temporal_weight===.1&&r.native_width===36&&r.final_inference_reload_verified===true,'valid original scattered0.1 control prefix required');
    check(r.executable);check(r.settings);check(r.dataset);Object.values(r.artifacts).forEach(check);return {stage,report:pin(p),data:r};});
  const [first,final]=reports.map(p=>p.data);
  assert(first.completed_updates===128&&first.attempted_batches===128&&final.completed_updates===256&&final.attempted_batches===256,'exact previously completed256 prefix required');
  assert(first.preprocessing_fitted===true&&first.preprocessing_fit_windows===1024&&final.preprocessing_fitted===false&&final.preprocessing_fit_windows===0,'preserved TRAIN fit required');
  assert(final.artifacts.resume_input.sha256===first.artifacts.training_state.sha256,'old prefix resume identity mismatch');
  for(const r of [first,final])assert(r.row_seed===seed+1000&&r.mask_seed===seed+2000&&r.context_seed===seed+3000&&r.context_deletion==='none','original paired stream identities required');
  for(const key of ['recipe_identity','initial_state_identity','scaler_identity','bank_identity'])assert(first[key]===final[key],`old prefix changed ${key}`);
  const events=reports.flatMap(r=>eventRows(r.data.artifacts.steps.path));assert(events.length===256,'old control prefix must contain256 actual events');
  events.forEach((e,i)=>assert(e.attempt===i&&e.updated===true&&e.completed_updates===i+1,'old control prefix event/progress mismatch'));
  return {seed,reports:reports.map(({stage,report})=>({stage,report})),initial_inference:first.artifacts.initial_inference,training_state:final.artifacts.training_state,
    final_inference:final.artifacts.inference,recipe_identity:final.recipe_identity,initial_state_identity:final.initial_state_identity,scaler_identity:final.scaler_identity,bank_identity:final.bank_identity,
    settings:final.settings,dataset:final.dataset,executable:final.executable,completed_updates:256,attempted_batches:256,reused_physical_updates:256};
}
export function eventRows(name){const text=fs.readFileSync(absolute(name),'utf8').trim();return text?text.split('\n').map(JSON.parse):[];}
function baseSources(){return allPins(['Makefile','.gitattributes','code/scripts/task.sh','code/scripts/node.sh','code/scripts/node-runtime.sh','dependencies.lock','environment-image.json',
  `${PROTOCOL}/PROTOCOL.md`,`${PROTOCOL}/CONTRACT.md`,`${PROTOCOL}/run-study.mjs`,`${FAMILY}/versions/v18/v18.mk`,`${FAMILY}/family/family.mk`,SETTINGS,
  `${FAMILY}/versions/v18/tests/engineering_test.cpp`,`${FAMILY}/family/config/paired.conf`,...filesUnder(`${FAMILY}/versions/v16`),...filesUnder(`${FAMILY}/family/include`),
  ...filesUnder(`${FAMILY}/versions/v17/include`),...filesUnder(`${FAMILY}/versions/v18/include`),...filesUnder('code/shared/settings/include'),...filesUnder('code/shared/representation/include'),...filesUnder('code/data/observations/include')]);}
export function admission(){const a=read(`${ATTEMPT}/admission.json`);assert(a.kind==='cpb_v18_span_study_admission_v1','wrong V18 admission');
  [...a.sources,...a.inputs,...a.frozen_numerical_baseline,...a.engineering].forEach(check);check(a.protocol_copy);check(a.control_source.capture);check(a.control_source.original_manifest);
  a.control_source.current_numerical_sources.forEach(check);check(a.control_source.executable);
  a.evaluation_equivalence.source_files.forEach(check);check(a.evaluation_equivalence.original_audit_phase);check(a.evaluation_equivalence.unchanged_audit_executable);
  for(const p of a.control_prefixes){p.reports.forEach(r=>check(r.report));[p.initial_inference,p.training_state,p.final_inference,p.settings,p.dataset,p.executable].forEach(check);}
  return a;}
function phase(name,executables,sources){admission();const filename=`${ATTEMPT}/${name}.admission.json`;
  if(fs.existsSync(absolute(filename))){const p=read(filename);[...p.sources,...p.executables].forEach(check);return p;}
  const p={kind:'cpb_v18_span_phase_admission_v1',phase:name,admitted_utc:new Date().toISOString(),study_admission:pin(`${ATTEMPT}/admission.json`),sources:allPins(sources),executables:allPins(executables)};
  if(name==='readout')p.independent_audit={absolute_tolerance:1e-9,relative_tolerance:1e-8,witness_codec:'cpb_v17_ridge_witness_v1',codec_meaning:'generic-ridge-tensor-codec; not an encoder identity; actual kind/checkpoint SHA are separate'};
  save(filename,p);return p;}
const phaseCheck=p=>{admission();check(p.study_admission);[...p.sources,...p.executables].forEach(check);};
export function completed(label){const p=`${ATTEMPT}/receipts/${label}.json`;if(!fs.existsSync(absolute(p)))return null;const r=read(p);
  assert(r.status==='passed'&&r.exit_code===0,`preserved failed step needs explicit review: ${label}`);[...r.inputs,...r.outputs].forEach(check);check(r.stdout);check(r.stderr);check(r.command);return r;}
function run(label,program,args,output,inputs,p,validate){phaseCheck(p);if(completed(label)){console.log(`Verified ${label}; no replay.`);return;}
  assert(!fs.existsSync(absolute(output)),`preserve partial output; no automatic replay: ${output}`);const command=`${ATTEMPT}/receipts/${label}.command.json`;
  assert(!fs.existsSync(absolute(command)),`launched step without passed receipt needs review: ${label}`);inputs.forEach(check);const binary=pin(program);
  assert(p.executables.some(e=>e.path===binary.path&&e.sha256===binary.sha256),'binary not in phase admission');
  fs.mkdirSync(absolute(`${ATTEMPT}/logs`),{recursive:true});save(command,{kind:'cpb_v18_span_native_command_v1',label,started_utc:new Date().toISOString(),program:binary,argv:args,inputs,output,phase_admission:pin(`${ATTEMPT}/${p.phase}.admission.json`)});
  const stdout=`${ATTEMPT}/logs/${label}.stdout.log`,stderr=`${ATTEMPT}/logs/${label}.stderr.log`,out=fs.openSync(absolute(stdout),'wx'),err=fs.openSync(absolute(stderr),'wx');let result,error;
  console.log(`Starting ${label}`);try{result=spawnSync(absolute(binary.path),args,{cwd:ROOT,env:process.env,stdio:['ignore',out,err]});}catch(e){result={status:null,signal:null,error:e};}finally{fs.closeSync(out);fs.closeSync(err);}
  try{assert(!result.error&&result.status===0,`native exit=${result.status}; ${result.error?.message??''}`);phaseCheck(p);inputs.forEach(check);validate();}catch(e){error=e.message;}
  const outputs=fs.existsSync(absolute(output))?filesUnder(output).map(pin):[];
  save(`${ATTEMPT}/receipts/${label}.json`,{kind:'cpb_v18_span_native_receipt_v1',label,status:error?'failed':'passed',finished_utc:new Date().toISOString(),exit_code:result.status,signal:result.signal??null,error:error??null,inputs,outputs,stdout:pin(stdout),stderr:pin(stderr),command:pin(command)});
  assert(!error,`${label}: ${error}; prior command/output/logs preserved`);console.log(`Completed ${label}`);}
export function validateTrain(t,stage){const definition=stages(t).find(s=>s[0]===stage);assert(definition,'declared training stage required');const [,target,start]=definition,r=report(t,stage),resuming=start>0;
  assert(r.status==='passed'&&r.resume===resuming&&r.completed_updates===target&&r.target_total_updates===target&&r.starting_completed_updates===start,'training progress/budget mismatch');
  assert(r.maximum_total_attempts===2048&&r.attempted_batches<=2048&&r.batch_size===8&&r.save_every_updates===128,'attempt/batch/save policy mismatch');
  assert(r.initialization_seed===t.seed&&r.row_seed===t.seed+1000&&r.mask_seed===t.seed+2000&&r.context_seed===t.seed+3000,'paired initialization/streams mismatch');
  assert(r.temporal_weight===.1&&r.native_width===36&&r.context_deletion==='none'&&r.learning_rate===.001&&r.weight_decay===.0001&&r.clip_norm===1,'shared numerical recipe mismatch');
  assert(r.preprocessing_fitted===!resuming&&r.preprocessing_fit_windows===(resuming?0:1024)&&r.physical_training_updates_this_invocation===target-start,'fit/physical update accounting mismatch');
  assert(r.dataset.sha256===trainingDataset().sha256&&r.settings.sha256===pin(SETTINGS).sha256&&r.final_inference_reload_verified===true&&r.quality_evaluated===false,'admitted data/settings/export mismatch');
  Object.values(r.artifacts).forEach(check);const prior=t.arm==='control'&&start===256?read(`${prefixDirectory(t.seed)}/resume-256/run.json`):start?report(t,stages(t)[0][0]):null;
  if(prior){assert(r.artifacts.resume_input.sha256===prior.artifacts.training_state.sha256&&r.starting_attempted_batches===prior.attempted_batches,'exact own saved state/counters required');
    for(const key of ['recipe_identity','initial_state_identity','scaler_identity','bank_identity'])assert(r[key]===prior[key],`resume changed ${key}`);}
  if(t.arm==='candidate')assert(r.mask_mode==='contiguous','candidate contiguous training recipe required');
  const old=read(`${prefixDirectory(t.seed)}/fresh-128/run.json`);for(const key of ['initial_state_identity','scaler_identity','bank_identity'])assert(r[key]===old[key],`base initialization/TRAIN fit differs within pair: ${key}`);
  const events=eventRows(r.artifacts.steps.path);assert(events.filter(e=>e.updated).length===target-start,'event/update accounting mismatch');
  assert(events.length===r.attempted_batches-r.starting_attempted_batches,'event/attempt accounting mismatch');
  events.forEach((e,i)=>{assert(e.attempt===r.starting_attempted_batches+i,'absolute event attempt order mismatch');
    if(t.arm==='candidate'){assert(Array.isArray(e.temporal_pair_counts)&&e.temporal_pair_counts.length===3&&e.temporal_pair_counts.every(n=>Number.isSafeInteger(n)&&n>=0),'candidate actual per-lag pair counts required');
      for(const key of ['selected_patch_count','unsupported_channels','no_legal_span_channels','insufficient_support_channels','target_cells'])assert(Number.isSafeInteger(e[key])&&e[key]>=0,`candidate actual mask/support counter required: ${key}`);
      assert(typeof e.mask_counter_seed==='string'&&/^[0-9]+$/.test(e.mask_counter_seed)&&BigInt(e.mask_counter_seed)<=9223372036854775807n,'exact decimal-string mask counter seed required');
      for(const key of ['mask_starts','legal_span_counts','mask_quotas'])assert(Array.isArray(e[key])&&e[key].length===24&&e[key].every(Number.isSafeInteger),`dense B8/C3 mask witness required: ${key}`);
      assert(e.mask_starts.every(n=>n>=0&&n<=12)&&e.legal_span_counts.every(n=>n===13)&&e.mask_quotas.every(n=>n===4),'dense original16-patch/four-hidden contiguous span law failed');
      assert(e.selected_patch_count===96&&e.target_cells===6912&&e.unsupported_channels===0&&e.no_legal_span_channels===0&&e.insufficient_support_channels===0,'dense admitted training mask/support count law failed');
      assert(JSON.stringify(e.temporal_pair_counts)===JSON.stringify([6696,6048,3456]),'actual dense contiguous temporal pair counts differ from declared geometry');}});
  return r;}
function train(){const a=admission(),p=phase('train',[`${OLD_BUILD}/cpb_v17_train`,`${BUILD}/cpb_v18_train`],['code/embeddings/scripts/cpb_v17_train.cpp','code/embeddings/scripts/cpb_v18_train.cpp']);
  const data=trainingDataset();for(const t of SCHEDULE){fs.mkdirSync(absolute(directory(t)),{recursive:true});for(const [stage,target,start] of stages(t)){
    const output=`${directory(t)}/${stage}`,program=t.arm==='control'?`${OLD_BUILD}/cpb_v17_train`:`${BUILD}/cpb_v18_train`;
    const args=t.arm==='control'?['--temporal-weight','0.1']:['--mask-mode','contiguous'];
    args.push('--settings',absolute(SETTINGS),'--dataset',absolute(data.path),'--output',absolute(output),'--steps',String(target),'--max-attempts','2048','--batch-size','8','--save-every','128',
      '--row-seed',String(t.seed+1000),'--mask-seed',String(t.seed+2000),'--context-seed',String(t.seed+3000),'--learning-rate','0.001','--weight-decay','0.0001','--clip-norm','1');
    const inputs=[pin(SETTINGS),data];if(start===0)args.push('--seed',String(t.seed));else{const state=start===256?a.control_prefixes.find(q=>q.seed===t.seed).training_state:pin(`${directory(t)}/${stages(t)[0][0]}/training-final.pt`);inputs.push(state);args.push('--resume',absolute(state.path));}
    run(`${t.id}-${stage}`,program,args,output,inputs,p,()=>validateTrain(t,stage));validateTrain(t,stage);
  }}seal();}
export function fullEvents(t){const past=t.arm==='control'?['fresh-128','resume-256'].flatMap(stage=>eventRows(`${prefixDirectory(t.seed)}/${stage}/steps.jsonl`)):[];
  return [...past,...stages(t).flatMap(([stage])=>eventRows(`${directory(t)}/${stage}/steps.jsonl`))];}
function candidateMaskCounters(t){const events=fullEvents(t),fields=['selected_patch_count','unsupported_channels','no_legal_span_channels','insufficient_support_channels','target_cells'];
  const starts=events.flatMap(e=>e.mask_starts),startHistogram=Array.from({length:13},(_,start)=>starts.filter(n=>n===start).length);
  return {events:events.length,temporal_pair_counts:{lags:[1,4,16],total:[0,1,2].map(j=>events.reduce((n,e)=>n+e.temporal_pair_counts[j],0)),minimum_per_attempt:[0,1,2].map(j=>Math.min(...events.map(e=>e.temporal_pair_counts[j]))),maximum_per_attempt:[0,1,2].map(j=>Math.max(...events.map(e=>e.temporal_pair_counts[j])))},
    totals:Object.fromEntries(fields.map(f=>[f,events.reduce((n,e)=>n+e[f],0)])),mask_witness:{flattened_channel_draws:starts.length,start_positions:Array.from({length:13},(_,i)=>i),start_histogram:startHistogram,all_legal_span_counts_13:true,all_quotas_4:true,exact_decimal_counter_seeds_saved:true},
    limits:'Actual returned candidate mask/support witnesses only; no second mask replay. Old control logs did not record detailed mask/lag fields.'};}
function seal(){if(fs.existsSync(absolute(`${ATTEMPT}/training-seal.json`))){verifySeal();return;}const a=admission(),trajectories=[];
  for(const t of SCHEDULE){const reports=stages(t).map(([stage])=>{assert(completed(`${t.id}-${stage}`),'all planned training commands required');return {stage,report:pin(`${directory(t)}/${stage}/run.json`),data:validateTrain(t,stage)};});
    const final=reports.at(-1).data,events=fullEvents(t);assert(events.length===final.attempted_batches,'complete logical attempt history required');let completedUpdates=0;
    events.forEach((e,i)=>{assert(e.attempt===i,'absolute full-trajectory attempt order mismatch');if(e.updated)++completedUpdates;assert(e.completed_updates===completedUpdates,'full-trajectory update/event accounting mismatch');});
    assert(completedUpdates===1024,'1024 successful updates required within the declared attempt ceiling');
    trajectories.push({...t,reused_prefix_updates:t.arm==='control'?256:0,new_physical_updates:reports.reduce((n,r)=>n+r.data.physical_training_updates_this_invocation,0),completed_updates:1024,attempted_batches:final.attempted_batches,
      initial_state_identity:final.initial_state_identity,scaler_identity:final.scaler_identity,bank_identity:final.bank_identity,training_reports:reports.map(({stage,report})=>({stage,report})),
      final_inference:check(final.artifacts.inference),final_training_state:check(final.artifacts.training_state),candidate_mask_counters:t.arm==='candidate'?candidateMaskCounters(t):null});}
  const pairs=SEEDS.map(seed=>{const control=trajectories.find(t=>t.seed===seed&&t.arm==='control'),candidate=trajectories.find(t=>t.seed===seed&&t.arm==='candidate');
    for(const key of ['initial_state_identity','scaler_identity','bank_identity'])assert(control[key]===candidate[key],`paired numerical state identity mismatch: ${key}`);
    const left=fullEvents(control),right=fullEvents(candidate);assert(left.length===right.length,'paired attempt counts differ');left.forEach((e,i)=>assert(e.attempt===right[i].attempt&&e.updated===right[i].updated&&JSON.stringify(e.source_row_ids)===JSON.stringify(right[i].source_row_ids),'paired realized row/attempt/support mismatch'));
    return {seed,matching_base_initialization_scaler_bank_numerical_identities:true,matching_realized_row_ids_attempts_and_update_flags:true,mask_algorithm_intentionally_different:true,identity_limit:'FNV tensor-byte identities are engineering witnesses, not cryptographic equality proofs'};});
  assert(trajectories.reduce((n,t)=>n+t.new_physical_updates,0)===5376,'new physical work count mismatch');
  save(`${ATTEMPT}/training-seal.json`,{kind:'cpb_v18_span_all_training_sealed_v1',status:'passed',sealed_utc:new Date().toISOString(),all_six_complete_before_any_head:true,logical_updates:6144,new_physical_updates:5376,reused_control_prefix_updates:768,control_prefix_admission:pin(`${ATTEMPT}/admission.json`),trajectories,pairs});
  console.log('Sealed six final1024 states:5376 new physical updates plus768 preserved control-prefix updates.');}
export function verifySeal(){admission();const s=read(`${ATTEMPT}/training-seal.json`);assert(s.status==='passed'&&s.trajectories.length===6&&s.logical_updates===6144&&s.new_physical_updates===5376&&s.reused_control_prefix_updates===768,'complete six-model1024 seal required');
  for(const [i,t] of s.trajectories.entries()){assert(t.id===SCHEDULE[i].id&&t.mask_mode===SCHEDULE[i].mask_mode,'sealed schedule/mask mismatch');t.training_reports.forEach(r=>{check(r.report);assert(completed(`${t.id}-${r.stage}`),'sealed command receipt missing');validateTrain(t,r.stage);});check(t.final_inference);check(t.final_training_state);}return s;}
function readout(){const seal=verifySeal(),p=phase('readout',[`${BUILD}/cpb_v18_readout`,`${OLD_BUILD}/cpb_v17_audit`],[`${PROTOCOL}/readout.cpp`,'code/embeddings/evaluation/protocols/cpb_v17_balance_v1/audit.cpp',
    ...filesUnder('code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps')]);
  const train=pin(`${OLD}/data-probes/probe-TRAIN.pt`),validation=pin(`${OLD}/data-probes/probe-VALIDATION.pt`),targets=targetPins();
  for(const t of seal.trajectories){const output=`${ATTEMPT}/readouts/${t.id}`,inputs=[pin(SETTINGS),t.final_inference,train,validation,targets.train,targets.validation];
    fs.mkdirSync(absolute(`${ATTEMPT}/readouts`),{recursive:true});fs.mkdirSync(absolute(`${ATTEMPT}/audits`),{recursive:true});
    run(`${t.id}-readout`,`${BUILD}/cpb_v18_readout`,['--settings',absolute(SETTINGS),'--checkpoint',absolute(t.final_inference.path),'--weight','0.1','--encoder-kind',t.arm==='control'?'v17':'v18',
      '--train',absolute(train.path),'--validation',absolute(validation.path),'--expected-train-targets',absolute(targets.train.path),'--expected-validation-targets',absolute(targets.validation.path),'--output',absolute(output),'--expected-updates','1024'],output,inputs,p,()=>{
      const r=read(`${output}/summary.json`);assert(r.updates===1024&&r.temporal_weight===.1&&r.mask_mode===t.mask_mode&&r.encoder_kind===(t.arm==='control'?'v17':'v18')&&r.target_precision_parity===true&&r.encoder_state_unchanged===true&&r.head_fits===4,'fixed readout/typed loader/target parity failed');
      assert(r.witness_codec==='cpb_v17_ridge_witness_v1'&&r.metrics.length===4&&r.metrics.every(m=>Number.isFinite(m.r2)&&m.coverage===1&&m.valid===32),'complete unchanged ridge witness required');});
    const witness=pin(`${output}/ridge-witness.pt`),auditOutput=`${ATTEMPT}/audits/${t.id}`;
    run(`${t.id}-audit`,`${OLD_BUILD}/cpb_v17_audit`,['--witness',absolute(witness.path),'--witness-sha256',witness.sha256,'--expected-train-targets',absolute(targets.train.path),'--expected-train-targets-sha256',targets.train.sha256,
      '--expected-validation-targets',absolute(targets.validation.path),'--expected-validation-targets-sha256',targets.validation.sha256,'--output',absolute(auditOutput)],auditOutput,[witness,targets.train,targets.validation],p,()=>{
        const r=read(`${auditOutput}/summary.json`);assert(r.status==='passed'&&r.absolute_tolerance===1e-9&&r.relative_tolerance===1e-8&&r.original_legacy_target_bits_equal===true&&r.original_row_group_association_equal===true,'unchanged independent scalar audit failed');});
  }}
export function differences(measurements){const paired=SEEDS.map(seed=>{const a=measurements.find(t=>t.seed===seed&&t.arm==='control'),b=measurements.find(t=>t.seed===seed&&t.arm==='candidate');
  assert(a&&b,'complete seed pair required');return {seed,...Object.fromEntries(TASKS.map(task=>{const metric=t=>t.metrics.find(m=>m.task===task)?.r2;assert(Number.isFinite(metric(a))&&Number.isFinite(metric(b)),`missing finite R2: ${task}`);return [task,metric(b)-metric(a)];}))};});
  const aggregate=Object.fromEntries(TASKS.map(task=>{const v=paired.map(p=>p[task]);return [task,{median:[...v].sort((a,b)=>a-b)[1],minimum:Math.min(...v),maximum:Math.max(...v),nonnegative_pairs:v.filter(x=>x>=0).length,positive_pairs:v.filter(x=>x>0).length}];}));
  const conditions={trend_median_nonnegative:aggregate.past_trend.median>=0,trend_at_least_two_nonnegative_pairs:aggregate.past_trend.nonnegative_pairs>=2,activity_median_nonnegative:aggregate.past_activity.median>=0,activity_at_least_two_nonnegative_pairs:aggregate.past_activity.nonnegative_pairs>=2,volatility_median_floor:aggregate.past_volatility.median>=-.01};
  return {paired,aggregate,conditions,passed:Object.values(conditions).every(Boolean),observed_gain:paired.some(p=>TASKS.some(task=>p[task]>0)),strict_trend_or_activity_gain_observed:paired.some(p=>p.past_trend>0||p.past_activity>0)};}
function summary(){const seal=verifySeal(),measurements=seal.trajectories.map(t=>{assert(completed(`${t.id}-readout`)&&completed(`${t.id}-audit`),'all six readouts/audits required');return {...t,metrics:read(`${ATTEMPT}/readouts/${t.id}/summary.json`).metrics,readout:pin(`${ATTEMPT}/readouts/${t.id}/summary.json`),audit:pin(`${ATTEMPT}/audits/${t.id}/summary.json`)};});
  const d=differences(measurements),record={kind:'cpb_v18_span_screen_summary_v1',status:'complete',created_utc:new Date().toISOString(),training_seal:pin(`${ATTEMPT}/training-seal.json`),primary_updates:1024,logical_updates:6144,new_physical_updates:5376,reused_control_prefix_updates:768,
    head_fits:24,independent_audits:6,paired_candidate_minus_control_r2:d.paired,aggregate:d.aggregate,predeclared_conditions:d.conditions,predeclared_continuation_criterion_passed:d.passed,observed_gain:d.observed_gain,strict_trend_or_activity_gain_observed:d.strict_trend_or_activity_gain_observed,
    automatic_promotion:false,new_best_encoder_established:false,fresh_confirmation:false,measurements,limitations:['Known January development only; no fresh population is opened.','Previously executed scattered controls retain their own typed256 ancestry;5376 updates are new physical work, not6144.','All final1024 states were sealed before scoring; no intermediate checkpoint gate or outcome-dependent budget extension.','Same target arithmetic,112/32 rows, TRAIN normalizers, ridge1 and independent scalar audit as V17.','No neural, TEMPO, fixed-query reconstruction, full2048-budget comparison or application utility is measured.']};
  const filename=`${ATTEMPT}/screen-summary.json`;if(fs.existsSync(absolute(filename))){const old=read(filename);assert(JSON.stringify(old.paired_candidate_minus_control_r2)===JSON.stringify(d.paired),'preserved summary differs');console.log(JSON.stringify(old,null,2));return;}
  save(filename,record);console.log(JSON.stringify(record,null,2));}
function admit(engineeringPath){assert(!fs.existsSync(absolute(ATTEMPT)),'preserve existing attempt; use its stage commands');const frozen=baseline(),control_source=preservedControlSource(),control_prefixes=SEEDS.map(prefix),evaluation_equivalence=evaluationEquivalence(),expected=targetPins();
  const oldAdmission=read(`${OLD}/admission.json`),oldData=read(`${OLD}/data-train/preparation.json`),oldProbes=read(`${OLD}/data-probes/preparation.json`);
  assert(oldData.status==='passed'&&oldProbes.status==='passed'&&oldData.actual_role_membership_verified&&oldProbes.actual_role_membership_verified,'previous admitted dataset preparation required');
  for(const prep of [oldData,oldProbes]){prep.inputs.forEach(check);check(prep.executable);}
  const archivePins=[['train',oldData],['probes',oldProbes]].flatMap(([mode,prep])=>prep.artifacts.map(p=>check({path:`${OLD}/data-${mode}/${p.file}`,bytes:p.bytes,sha256:p.sha256})));
  assert(oldData.artifacts.length===1&&oldData.artifacts[0].role==='TRAIN'&&oldData.artifacts[0].rows===1024,'original1024 TRAIN archive binding required');
  assert(oldProbes.artifacts.length===2&&oldProbes.artifacts.some(p=>p.role==='TRAIN'&&p.rows===112)&&oldProbes.artifacts.some(p=>p.role==='VALIDATION'&&p.rows===32),'original112/32 probe archive bindings required');
  const dataset=trainingDataset(),settings=pin(SETTINGS);control_prefixes.forEach(p=>assert(p.settings.sha256===settings.sha256&&p.dataset.sha256===dataset.sha256&&p.executable.sha256===control_source.executable.sha256,'all controls must retain exact original executable/data/settings'));
  const engineering=pin(engineeringPath),e=read(engineeringPath);assert(e.status==='passed'&&e.quality_evaluated===false,'passed scoped V18 engineering fixture required');
  const engineeringPins=[engineering],executionPath=`${BASE}/engineering-execution.json`;
  if(fs.existsSync(absolute(executionPath))){const execution=read(executionPath);assert(execution.status==='passed'&&execution.exit_code===0,'valid engineering execution required');engineeringPins.push(pin(executionPath));for(const field of ['source_admission','stdout','stderr'])if(execution[field])engineeringPins.push(check(execution[field]));
    if(execution.source_admission){const a=read(execution.source_admission.path);a.sources?.forEach(check);if(a.executable)check(a.executable);}}
  const inputNames=[OLD_RECORD,`${OLD}/admission.json`,`${OLD}/training-seal.json`,`${OLD}/closeout-input-manifest.json`,`${OLD}/data-train/preparation.json`,`${OLD}/data-probes/preparation.json`,`${OLD}/data-probes/probe-TRAIN.pt`,`${OLD}/data-probes/probe-VALIDATION.pt`,
    `${FAMILY}/research-reference.json`,`${FAMILY}/family/evidence/compatibility-2026-10-22.json`];
  fs.mkdirSync(absolute(ATTEMPT),{recursive:true});const copy=`${ATTEMPT}/protocol-admitted.md`;fs.copyFileSync(absolute(`${PROTOCOL}/PROTOCOL.md`),absolute(copy),fs.constants.COPYFILE_EXCL);
  save(`${ATTEMPT}/admission.json`,{kind:'cpb_v18_span_study_admission_v1',status:'admitted',admitted_utc:new Date().toISOString(),scope:'One contiguous-versus-scattered mask comparison at fixed1024; unchanged development evaluation; no global promotion.',
    sources:baseSources(),frozen_numerical_baseline:frozen,inputs:[...allPins(inputNames),...archivePins,dataset,expected.train,expected.validation,...oldAdmission.inputs.filter(p=>p.path.includes('.cache.bin')).map(check)],engineering:engineeringPins,protocol_copy:pin(copy),control_source,control_prefixes,evaluation_equivalence,schedule:SCHEDULE,
    reused_data:{training:dataset,probes:[pin(`${OLD}/data-probes/probe-TRAIN.pt`),pin(`${OLD}/data-probes/probe-VALIDATION.pt`)],preparation_reexecuted:false,legacy_paths_are_dataset_identities_not_encoder_ancestry:true},
    shared_recipe:{coefficient:.1,configuration:settings,encoder_seeds:SEEDS,row_seed_offset:1000,mask_seed_offset:2000,context_seed_offset:3000,TRAIN_fit_windows:1024,TRAIN_fit_policy:'all-TRAIN-source-order-batch-chunks-Chan-v1;batch=8',batch_size:8,context:'none',learning_rate:.001,weight_decay:.0001,clip_norm:1,primary_updates:1024,max_total_attempts:2048,save_every:128},
    work_plan:{controls:'Reuse each exact256 state; continue256->640->1024 through original V17 CLI/executable/recipe without refitting.',candidates:'Fresh0->512; typed resume512->1024 through new V18 contiguous-mask recipe.',logical_updates:6144,new_physical_updates:5376,reused_control_prefix_updates:768,planned_native_training_processes:12},
    equivalence:{basis:'Byte-verified preserved original control trainer/core/preprocessing/mask source, exact original executable and typed state; scoped V18 engineering receipt verifies shared construction/objective and mask mechanism.',architecture_unchanged:true,objective_weight_unchanged:true,mask_algorithm_is_the_intended_change:true,original_root_build_file_need_not_match:'The closed V17 root build/attributes are retained in its exact108-file capsule; the new study uses a newly admitted root build.'}});
  console.log(`Admitted ${ATTEMPT}; old control prefixes and source capsule preserved.`);}
function main(){const [command,...extra]=process.argv.slice(2);if(command==='--help'||!command){console.log('Commands: admit [--engineering FILE], train, readout, summary. No preparation replay; failed/partial outputs are preserved.');return;}nativeGuard();
  if(command==='admit'){assert(extra.length===0||(extra.length===2&&extra[0]==='--engineering'),'only --engineering FILE allowed for admission');admit(extra[1]??`${BASE}/engineering-001/summary.json`);return;}
  assert(extra.length===0,'unexpected arguments');if(command==='train')train();else if(command==='readout')readout();else if(command==='summary')summary();else throw Error(`unknown command: ${command}`);}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){try{main();}catch(error){console.error(error.stack??String(error));process.exitCode=1;}}

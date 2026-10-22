// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Prospective, bounded native study orchestration. Node standard library only.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../../../..');
const STUDY = path.join(ROOT, '.temp/embeddings/runs/cpb-v17-balance-001');
const ATTEMPT = path.join(STUDY, 'attempt-001');
const BUILD = '/opt/mityane-geometrics/embeddings/build/cpb-v17-balance-001';
const FAMILY = 'code/embeddings/encoders/causal_patch_bottleneck_mae';
const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v17_balance_v1';
const SETTINGS = `${FAMILY}/versions/v17/config/research.conf`;
const COMPAT = `${FAMILY}/family/evidence/compatibility-2026-10-22.json`;
const V16_INPUTS = '.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/run-inputs-001.json';
const REFERENCE = '.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/training/checkpoints/inference-2048.pt';
const REFERENCE_SHA = 'b6f3cbe711235307f362bf2f999571d65d00c85825f8a4a5c2fbc4fa4242e51e';
const CACHE_BASE = '.temp/embeddings/closeout/cpb-v14/attempt-001/portable-external-evidence/data/mkt-hist-1/csv';
const CACHES = [
  ['BTCUSDT', '67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58'],
  ['ETHUSDT', '0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa'],
  ['SOLUSDT', '364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7'],
].map(([symbol, sha256]) => ({path: `${CACHE_BASE}/${symbol}-1m-2024-01.cache.bin`, bytes:3928320, sha256}));
const SCHEDULE = [
  [947101,'control',0.1], [947101,'candidate',0.03],
  [947102,'candidate',0.03], [947102,'control',0.1],
  [947103,'control',0.1], [947103,'candidate',0.03],
].map(([seed, arm, weight]) => ({seed, arm, weight, id:`seed-${seed}-${arm}`}));
const assert = (ok, why) => {if (!ok) throw Error(why);};
const absolute = name => path.isAbsolute(name) ? name : path.join(ROOT, name);
const display = name => name.startsWith(`${ROOT}/`) ? path.relative(ROOT,name).split(path.sep).join('/') : name;
const read = name => JSON.parse(fs.readFileSync(absolute(name),'utf8'));
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
function pin(name) {
  const p=absolute(name), s=fs.lstatSync(p);
  assert(s.isFile()&&!s.isSymbolicLink(), `regular nonsymlink file required: ${name}`);
  return {path:display(p), bytes:s.size, sha256:sha(fs.readFileSync(p))};
}
function check(p) {
  const actual=pin(p.path);
  assert(actual.bytes===p.bytes&&actual.sha256===p.sha256, `bound bytes changed: ${p.path}`);
  return actual;
}
function save(name, value) {
  const p=absolute(name); fs.mkdirSync(path.dirname(p),{recursive:true});
  const fd=fs.openSync(p,'wx');
  try {fs.writeFileSync(fd,JSON.stringify(value,null,2)+'\n');fs.fsyncSync(fd);} finally {fs.closeSync(fd);}
}
function filesUnder(name) {
  const p=absolute(name); assert(fs.existsSync(p),`missing source directory: ${name}`);
  return fs.readdirSync(p,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name)).flatMap(e=>{
    assert(!e.isSymbolicLink(),`source symlink forbidden: ${name}/${e.name}`);
    return e.isDirectory()?filesUnder(`${name}/${e.name}`):[`${name}/${e.name}`];
  });
}
function allPins(names) {return [...new Set(names)].sort().map(pin);}
function nativeGuard() {
  assert(process.platform==='linux'&&fs.existsSync('/.dockerenv')&&ROOT==='/mityane',
    'Run this study only inside the managed mityane-geometrics container at /mityane.');
}
function frozenPins() {
  const receipt=read(COMPAT); assert(receipt.status==='passed','passed family compatibility receipt required');
  const pins=receipt.sourceFiles.filter(p=>p.path.startsWith(`${FAMILY}/versions/v16/`)||p.path.startsWith(`${FAMILY}/family/include/`));
  assert(pins.filter(p=>p.path.startsWith(`${FAMILY}/versions/v16/`)).length===49,'expected49 frozen V16 source records');
  assert(pins.filter(p=>p.path.startsWith(`${FAMILY}/family/include/`)).length===3,'expected3 frozen family numerical headers');
  pins.forEach(check);return pins;
}
function targets() {
  const inputs=read(V16_INPUTS);
  const select=leaf=>{
    const found=inputs.input_files.filter(p=>p.path.endsWith(`/features/${leaf}`));
    assert(found.length===1,`exact original target pin required: ${leaf}`);return check(found[0]);
  };
  return {train:select('train-targets.pt'),validation:select('validation-targets.pt')};
}
function baseSources() {
  return allPins([
    'Makefile','code/scripts/task.sh','code/scripts/node.sh','code/scripts/node-runtime.sh','dependencies.lock','environment-image.json',
    `${PROTOCOL}/PROTOCOL.md`,`${PROTOCOL}/run-study.mjs`,
    `${FAMILY}/versions/v17/v17.mk`,`${FAMILY}/family/family.mk`,SETTINGS,
    `${FAMILY}/versions/v17/tests/engineering_test.cpp`,`${FAMILY}/family/config/paired.conf`,
    ...filesUnder(`${FAMILY}/versions/v16`),
    ...filesUnder(`${FAMILY}/family/include`),
    ...filesUnder(`${FAMILY}/versions/v17/include`),
    ...filesUnder('code/shared/settings/include'),
    ...filesUnder('code/shared/representation/include'),
    ...filesUnder('code/shared/artifacts/versions/v1/include'),
    ...filesUnder('code/data/observations/include'),
  ]);
}
function admission() {
  const a=read(path.join(ATTEMPT,'admission.json'));
  assert(a.kind==='cpb_v17_balance_study_admission_v1','wrong study admission');
  [...a.sources,...a.frozen_numerical_baseline,...a.inputs,...a.engineering].forEach(check);
  check(a.protocol_copy);
  for(const name of fs.readdirSync(ATTEMPT).filter(n=>n.endsWith('.admission.json'))){
    const admittedPhase=read(path.join(ATTEMPT,name));
    [...admittedPhase.sources,...admittedPhase.executables].forEach(check);
  }
  return a;
}
function phase(label, binaryNames, sources=[]) {
  admission();const filename=path.join(ATTEMPT,`${label}.admission.json`);
  if(fs.existsSync(filename)) {
    const existing=read(filename);[...existing.sources,...existing.executables].forEach(check);return existing;
  }
  const record={kind:'cpb_v17_phase_admission_v1',phase:label,admitted_utc:new Date().toISOString(),
    study_admission:pin(path.join(ATTEMPT,'admission.json')),
    sources:allPins(sources),executables:allPins(binaryNames.map(n=>`${BUILD}/${n}`))};
  if(label==='readout')record.independent_audit_tolerance={absolute:1e-9,relative:1e-8};
  save(filename,record);return record;
}
function phaseCheck(p) {admission();[...p.sources,...p.executables].forEach(check);}
function completeStep(label) {
  const receipt=path.join(ATTEMPT,'receipts',`${label}.json`);
  if(!fs.existsSync(receipt))return null;
  const record=read(receipt);assert(record.status==='passed',`preserved failed step requires review: ${label}`);
  record.inputs.forEach(check);record.outputs.forEach(check);return record;
}
function run(label,binary,args,output,inputs,p,validate) {
  phaseCheck(p);
  if(completeStep(label)){console.log(`Verified completed step ${label}; no replay.`);return;}
  assert(!fs.existsSync(output),`preserve existing partial output; no replay: ${display(output)}`);
  const receiptDir=path.join(ATTEMPT,'receipts');fs.mkdirSync(receiptDir,{recursive:true});
  const logs=path.join(ATTEMPT,'logs');fs.mkdirSync(logs,{recursive:true});
  const commandFile=path.join(receiptDir,`${label}.command.json`);
  assert(!fs.existsSync(commandFile),`prior launched step has no successful receipt; review required: ${label}`);
  inputs.forEach(check);const program=pin(`${BUILD}/${binary}`);
  assert(p.executables.some(b=>b.path===program.path&&b.sha256===program.sha256),'program absent from phase admission');
  save(commandFile,{kind:'cpb_v17_native_command_v1',label,started_utc:new Date().toISOString(),program,argv:args,inputs,output:display(output),phase_admission:pin(path.join(ATTEMPT,`${p.phase}.admission.json`))});
  const out=path.join(logs,`${label}.stdout.log`),err=path.join(logs,`${label}.stderr.log`);
  const outfd=fs.openSync(out,'wx'),errfd=fs.openSync(err,'wx');let result,error;
  console.log(`Starting ${label}`);
  try {result=spawnSync(absolute(program.path),args,{cwd:ROOT,env:process.env,stdio:['ignore',outfd,errfd]});}
  catch(e){result={status:null,signal:null,error:e};}
  finally {fs.closeSync(outfd);fs.closeSync(errfd);}
  try {
    assert(!result.error&&result.status===0,`native command failed: exit=${result.status}; ${result.error?.message??''}`);
    phaseCheck(p);inputs.forEach(check);validate();
  } catch(e){error=e.message;}
  const outputPins=fs.existsSync(output)?filesUnder(display(output)).map(pin):[];
  const record={kind:'cpb_v17_native_command_receipt_v1',label,status:error?'failed':'passed',finished_utc:new Date().toISOString(),exit_code:result.status,signal:result.signal??null,error:error??null,
    inputs,outputs:outputPins,stdout:pin(out),stderr:pin(err),command:pin(commandFile)};
  save(path.join(receiptDir,`${label}.json`),record);
  assert(!error,`${label}: ${error}; preserved receipt/logs at ${display(receiptDir)}`);
  console.log(`Completed ${label}`);
}
const dataDir=mode=>path.join(ATTEMPT,`data-${mode}`);
const dataset=(mode,name)=>pin(path.join(dataDir(mode),name));
const trajectoryDir=t=>path.join(ATTEMPT,'trajectories',t.id);
const trainReport=(t,stage)=>read(path.join(trajectoryDir(t),stage,'run.json'));
function prepare(mode) {
  if(mode==='train')requireDiagnostic();
  if(mode==='probes')verifySeal();
  const p=phase('windows',['cpb_v17_windows'],['code/data/market/experiments/cpb_v17_balance_v1/windows.cpp']);
  const output=dataDir(mode);const args=['--mode',mode];
  for(const c of CACHES)args.push('--cache',absolute(c.path));
  for(const c of CACHES)args.push('--cache-sha256',c.sha256);
  args.push('--output',output);
  run(`prepare-${mode}`,'cpb_v17_windows',args,output,CACHES,p,()=>{
    const receipt=read(path.join(output,'preparation.json'));
    assert(receipt.status==='passed'&&receipt.mode===mode&&receipt.actual_role_membership_verified===true,'native producer membership admission failed');
    const expected=mode==='diagnostic'?[['TRAIN',64]]:mode==='train'?[['TRAIN',1024]]:[['TRAIN',112],['VALIDATION',32]];
    assert(receipt.artifacts.length===expected.length,'unexpected exported dataset count');
    expected.forEach(([role,rows],i)=>assert(receipt.artifacts[i].role===role&&receipt.artifacts[i].rows===rows,'wrong source-admitted role/count'));
  });
}
function requireDiagnostic() {
  assert(completeStep('diagnostic'),'complete predeclared zero-update diagnostic required first');
  const r=read(path.join(ATTEMPT,'diagnostic','summary.json'));
  assert(r.status==='passed'&&r.physical_training_updates===0&&r.preprocessing_fits===0&&r.heldout_readouts===0&&r.original_and_adapted_state_unchanged===true,'diagnostic boundary failed');
}
function diagnostic() {
  assert(completeStep('prepare-diagnostic'),'prepare diagnostic dataset first');
  const p=phase('diagnostic',['cpb_v17_diagnostic'],[`${PROTOCOL}/diagnostic.cpp`]);
  const d=dataset('diagnostic','diagnostic-TRAIN.pt'),ref=pin(REFERENCE),output=path.join(ATTEMPT,'diagnostic');
  run('diagnostic','cpb_v17_diagnostic',['--settings',absolute(SETTINGS),'--dataset',absolute(d.path),'--dataset-sha256',d.sha256,
    '--checkpoint',absolute(ref.path),'--checkpoint-sha256',ref.sha256,'--mask-seed','949103','--output',output],output,[pin(SETTINGS),d,ref],p,()=>{
      const r=read(path.join(output,'summary.json'));
      assert(r.status==='passed'&&r.batches===8&&r.rows===64&&r.physical_training_updates===0&&r.original_and_adapted_state_unchanged===true,'zero-update diagnostic failed');
    });
}
function validateTrain(t,stage,d) {
  const r=trainReport(t,stage),resuming=stage==='resume-256',target=resuming?256:128;
  assert(r.status==='passed'&&r.resume===resuming&&r.completed_updates===target&&r.target_total_updates===target,'wrong training completion');
  assert(r.maximum_total_attempts===512&&r.attempted_batches<=512&&r.batch_size===8&&r.save_every_updates===64,'training budget mismatch');
  assert(r.row_seed===t.seed+1000&&r.mask_seed===t.seed+2000&&r.context_seed===t.seed+3000&&r.initialization_seed===t.seed,'stream/initialization seed mismatch');
  assert(r.temporal_weight===t.weight&&r.native_width===36&&r.context_deletion==='none'&&r.learning_rate===0.001&&r.weight_decay===0.0001&&r.clip_norm===1,'numerical recipe mismatch');
  assert(r.dataset.sha256===d.sha256&&r.final_inference_reload_verified===true,'dataset/export mismatch');
  assert(r.preprocessing_fitted===!resuming&&r.preprocessing_fit_windows===(resuming?0:1024),'unexpected preprocessing fit');
  assert(r.starting_completed_updates===(resuming?128:0)&&r.physical_training_updates_this_invocation===128,'physical-update accounting mismatch');
  assert(typeof r.initial_state_identity==='string'&&r.initial_state_identity.length>0,'initial tensor identity missing');
  Object.values(r.artifacts).forEach(check);
  if(resuming){
    const first=trainReport(t,'fresh-128');
    assert(r.artifacts.resume_input.sha256===first.artifacts.training_state.sha256&&r.starting_attempted_batches===first.attempted_batches,'resume input/progress mismatch');
    for(const key of ['recipe_identity','initial_state_identity','scaler_identity','bank_identity'])assert(r[key]===first[key],`resume changed ${key}`);
  }
  return r;
}
function train() {
  requireDiagnostic();assert(completeStep('prepare-train'),'prepare training dataset first');
  const p=phase('train',['cpb_v17_train'],['code/embeddings/scripts/cpb_v17_train.cpp']);
  const d=dataset('train','calibration-TRAIN.pt');
  for(const t of SCHEDULE){
    fs.mkdirSync(trajectoryDir(t),{recursive:true});
    for(const [stage,target] of [['fresh-128',128],['resume-256',256]]){
      const output=path.join(trajectoryDir(t),stage);
      const args=['--temporal-weight',String(t.weight),'--settings',absolute(SETTINGS),'--dataset',absolute(d.path),'--output',output,
        '--steps',String(target),'--max-attempts','512','--batch-size','8','--save-every','64',
        '--row-seed',String(t.seed+1000),'--mask-seed',String(t.seed+2000),'--context-seed',String(t.seed+3000),
        '--learning-rate','0.001','--weight-decay','0.0001','--clip-norm','1'];
      const inputs=[pin(SETTINGS),d];
      if(stage==='fresh-128')args.push('--seed',String(t.seed));
      else {const state=pin(path.join(trajectoryDir(t),'fresh-128','training-final.pt'));inputs.push(state);args.push('--resume',absolute(state.path));}
      run(`${t.id}-${stage}`,'cpb_v17_train',args,output,inputs,p,()=>validateTrain(t,stage,d));
      validateTrain(t,stage,d);
    }
  }
  seal();
}
function seal() {
  const filename=path.join(ATTEMPT,'training-seal.json');if(fs.existsSync(filename)){verifySeal();return;}
  admission();const d=dataset('train','calibration-TRAIN.pt');const trajectories=[];
  for(const t of SCHEDULE){
    for(const stage of ['fresh-128','resume-256'])assert(completeStep(`${t.id}-${stage}`),`incomplete trajectory: ${t.id}/${stage}`);
    const first=validateTrain(t,'fresh-128',d),final=validateTrain(t,'resume-256',d);
    trajectories.push({...t,initial_state_identity:first.initial_state_identity,scaler_identity:first.scaler_identity,bank_identity:first.bank_identity,
      fresh_report:pin(path.join(trajectoryDir(t),'fresh-128','run.json')),final_report:pin(path.join(trajectoryDir(t),'resume-256','run.json')),
      final_inference:check(final.artifacts.inference),completed_updates:final.completed_updates,attempted_batches:final.attempted_batches,
      physical_updates:first.physical_training_updates_this_invocation+final.physical_training_updates_this_invocation});
  }
  const pairs=[];
  for(const seed of [947101,947102,947103]){
    const a=trajectories.find(t=>t.seed===seed&&t.arm==='control'),b=trajectories.find(t=>t.seed===seed&&t.arm==='candidate');
    for(const key of ['initial_state_identity','scaler_identity','bank_identity'])assert(a[key]===b[key],`paired tensor identity mismatch for ${seed}/${key}`);
    const events=t=>['fresh-128','resume-256'].flatMap(stage=>fs.readFileSync(path.join(trajectoryDir(t),stage,'steps.jsonl'),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse));
    const ae=events(a),be=events(b);assert(ae.length===be.length,'paired attempt count differs');
    for(let i=0;i<ae.length;i++)assert(ae[i].attempt===be[i].attempt&&ae[i].updated===be[i].updated&&JSON.stringify(ae[i].source_row_ids)===JSON.stringify(be[i].source_row_ids),'paired realized sampler/support mismatch');
    pairs.push({seed,exact_initial_scaler_bank_numerical_identity_match:true,paired_attempt_row_id_match:true,
      identity_limit:'tensor-byte FNV numerical identities are explicit engineering witnesses, not cryptographic equality proofs'});
  }
  save(filename,{kind:'cpb_v17_all_training_sealed_v1',status:'passed',sealed_utc:new Date().toISOString(),
    no_probe_or_head_execution_before_seal:true,logical_updates:1536,recorded_physical_updates:trajectories.reduce((n,t)=>n+t.physical_updates,0),trajectories,pairs});
  console.log('Sealed all six final256 trajectories before probes/readouts.');
}
function verifySeal() {
  admission();const seal=read(path.join(ATTEMPT,'training-seal.json'));
  assert(seal.status==='passed'&&seal.trajectories.length===6&&seal.logical_updates===1536,'six-instance training seal required');
  const d=dataset('train','calibration-TRAIN.pt');
  for(const [i,t] of seal.trajectories.entries()){
    assert(t.id===SCHEDULE[i].id&&t.seed===SCHEDULE[i].seed&&t.weight===SCHEDULE[i].weight,'sealed schedule mismatch');
    check(t.fresh_report);check(t.final_report);check(t.final_inference);
    for(const stage of ['fresh-128','resume-256']){
      assert(completeStep(`${t.id}-${stage}`),'sealed command receipt missing');validateTrain(t,stage,d);
    }
  }
  return seal;
}
function readout() {
  const seal=verifySeal();assert(completeStep('prepare-probes'),'prepare sealed probe datasets first');
  const p=phase('readout',['cpb_v17_readout','cpb_v17_audit'],[`${PROTOCOL}/readout.cpp`,`${PROTOCOL}/audit.cpp`,
    ...filesUnder('code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps')]);
  const td=dataset('probes','probe-TRAIN.pt'),vd=dataset('probes','probe-VALIDATION.pt');const expected=targets();
  fs.mkdirSync(path.join(ATTEMPT,'readouts'),{recursive:true});fs.mkdirSync(path.join(ATTEMPT,'audits'),{recursive:true});
  for(const t of seal.trajectories){
    const output=path.join(ATTEMPT,'readouts',t.id),inputs=[pin(SETTINGS),t.final_inference,td,vd,expected.train,expected.validation];
    run(`${t.id}-readout`,'cpb_v17_readout',['--settings',absolute(SETTINGS),'--checkpoint',absolute(t.final_inference.path),'--weight',String(t.weight),
      '--train',absolute(td.path),'--validation',absolute(vd.path),'--expected-train-targets',absolute(expected.train.path),
      '--expected-validation-targets',absolute(expected.validation.path),'--output',output,'--expected-updates','256'],output,inputs,p,()=>{
        const r=read(path.join(output,'summary.json'));
        assert(r.updates===256&&r.temporal_weight===t.weight&&r.target_precision_parity===true&&r.encoder_state_unchanged===true&&r.head_fits===4&&r.metrics.length===4,'development readout receipt failed');
        for(const m of r.metrics)assert(Number.isFinite(m.r2)&&m.coverage===1,'complete finite R2 required');
      });
    const witness=pin(path.join(output,'ridge-witness.pt')),auditOutput=path.join(ATTEMPT,'audits',t.id);
    run(`${t.id}-audit`,'cpb_v17_audit',['--witness',absolute(witness.path),'--witness-sha256',witness.sha256,
      '--expected-train-targets',absolute(expected.train.path),'--expected-train-targets-sha256',expected.train.sha256,
      '--expected-validation-targets',absolute(expected.validation.path),'--expected-validation-targets-sha256',expected.validation.sha256,
      '--output',auditOutput],auditOutput,[witness,expected.train,expected.validation],p,()=>{
        const r=read(path.join(auditOutput,'summary.json'));
        assert(r.status==='passed'&&r.absolute_tolerance===1e-9&&r.relative_tolerance===1e-8&&r.original_legacy_target_bits_equal===true&&r.original_row_group_association_equal===true,'independent readout audit failed');
      });
  }
}
function summary() {
  const seal=verifySeal();const measurements=[];
  for(const t of seal.trajectories){
    assert(completeStep(`${t.id}-readout`)&&completeStep(`${t.id}-audit`),'all six readouts and independent audits required');
    const r=read(path.join(ATTEMPT,'readouts',t.id,'summary.json'));
    measurements.push({...t,metrics:r.metrics,readout:pin(path.join(ATTEMPT,'readouts',t.id,'summary.json')),audit:pin(path.join(ATTEMPT,'audits',t.id,'summary.json'))});
  }
  const tasks=['past_volatility','past_trend','past_activity','past_relation'];
  const deltas=[947101,947102,947103].map(seed=>{
    const pair=measurements.filter(t=>t.seed===seed),a=pair.find(t=>t.arm==='control'),b=pair.find(t=>t.arm==='candidate');
    return {seed,...Object.fromEntries(tasks.map(task=>{const metric=t=>t.metrics.find(m=>m.task===task)?.r2;assert(Number.isFinite(metric(a))&&Number.isFinite(metric(b)),`missing R2: ${task}`);return[task,metric(b)-metric(a)];}))};
  });
  const aggregate=Object.fromEntries(tasks.map(task=>{const values=deltas.map(d=>d[task]);return[task,{median:[...values].sort((a,b)=>a-b)[1],minimum:Math.min(...values),maximum:Math.max(...values),nonnegative_pairs:values.filter(v=>v>=0).length,positive_pairs:values.filter(v=>v>0).length}];}));
  const trend=aggregate.past_trend,activity=aggregate.past_activity,volatility=aggregate.past_volatility;
  const criterionPass=trend.median>=0&&trend.nonnegative_pairs>=2&&activity.median>=0&&activity.nonnegative_pairs>=2&&volatility.median>=-0.01;
  const strictGain=deltas.some(d=>d.past_trend>0||d.past_activity>0);
  const record={kind:'cpb_v17_bounded_screen_summary_v1',status:'complete',created_utc:new Date().toISOString(),training_seal:pin(path.join(ATTEMPT,'training-seal.json')),
    logical_updates:1536,recorded_physical_updates:seal.recorded_physical_updates,head_fits:24,independent_audits:6,paired_candidate_minus_control_r2:deltas,aggregate,
    predeclared_conditions:{trend_median_nonnegative:trend.median>=0,trend_at_least_two_nonnegative_pairs:trend.nonnegative_pairs>=2,
      activity_median_nonnegative:activity.median>=0,activity_at_least_two_nonnegative_pairs:activity.nonnegative_pairs>=2,volatility_median_floor:volatility.median>=-0.01},
    predeclared_continuation_criterion_passed:criterionPass,observed_gain:deltas.some(d=>tasks.some(task=>d[task]>0)),
    strict_trend_or_activity_gain_observed:strictGain,automatic_promotion:false,new_best_encoder_established:false,fresh_confirmation:false,
    limitations:['Known January development population and one256-update screen only.','Fresh controls share new fitter/calibration/sampling/context recipe and do not reproduce historical V16.','A positive continuation criterion only supports proposing further study; it does not authorize execution or replace the selected V16 reference.','Historical timing guards and TEMPO transfer were not rerun; fresh quality confirmation remains deferred.'],measurements};
  const output=path.join(ATTEMPT,'screen-summary.json');
  if(fs.existsSync(output)){const old=read(output);assert(JSON.stringify(old.paired_candidate_minus_control_r2)===JSON.stringify(deltas),'existing summary differs');console.log(JSON.stringify(old,null,2));return;}
  save(output,record);console.log(JSON.stringify(record,null,2));
}
function admit(engineeringPath) {
  assert(!fs.existsSync(ATTEMPT),'attempt-001 already exists; preserve it and use subsequent stage commands');
  const frozen=frozenPins(),expected=targets();CACHES.forEach(check);
  const reference=pin(REFERENCE);assert(reference.sha256===REFERENCE_SHA&&reference.bytes===1295473,'selected V16 reference identity mismatch');
  const engineering=pin(engineeringPath),engineeringReport=read(engineeringPath);
  assert(engineeringReport.status==='passed','passed V17 engineering fixture receipt required before study admission');
  const sources=baseSources();const inputPins=[...CACHES,reference,expected.train,expected.validation,pin(V16_INPUTS),pin(COMPAT),pin(`${FAMILY}/research-reference.json`)];
  fs.mkdirSync(STUDY,{recursive:true});fs.mkdirSync(ATTEMPT);
  const copy=path.join(ATTEMPT,'protocol-admitted.md');fs.copyFileSync(absolute(`${PROTOCOL}/PROTOCOL.md`),copy,fs.constants.COPYFILE_EXCL);
  save(path.join(ATTEMPT,'admission.json'),{kind:'cpb_v17_balance_study_admission_v1',status:'admitted',admitted_utc:new Date().toISOString(),
    managed_container:'mityane-geometrics',root:ROOT,build_directory:BUILD,attempt:display(ATTEMPT),protocol_copy:pin(copy),sources,
    frozen_numerical_baseline:frozen,inputs:inputPins,engineering:[engineering],schedule:SCHEDULE,
    fixed_recipe:{fresh_stop:128,resume_total_stop:256,total_attempt_ceiling:512,batch_size:8,save_every:64,row_seed_offset:1000,mask_seed_offset:2000,context_seed_offset:3000,
      calibration_windows:1024,calibration_fitter:'rounded-parent-Chan',context:'none',learning_rate:0.001,weight_decay:0.0001,clip_norm:1},
    phase_admissions:'Native executable and phase-only source pins are persisted before each dependent phase. Source mutation after admission stops the phase.',
    scope:'One zero-update TRAIN diagnostic and six fresh paired256-update trajectories, followed by sealed January development linear readouts and independent audits. No full study/promotion/publication.'});
  console.log(`Admitted prospective study: ${display(ATTEMPT)}`);
}
function main() {
  const [command,...extra]=process.argv.slice(2);
  if(command==='--help'||!command){console.log('Commands: admit [--engineering FILE], prepare-diagnostic, diagnostic, prepare-train, train, prepare-probes, readout, summary. Run in order, inside managed container. Completed steps are verified without replay; partial/failed outputs are preserved and block automatic retry.');return;}
  nativeGuard();
  if(command==='admit'){
    assert(extra.length===0||(extra.length===2&&extra[0]==='--engineering'),'admit accepts only --engineering FILE');
    admit(extra[1]??path.join(STUDY,'engineering-001','summary.json'));return;
  }
  assert(extra.length===0,'unexpected arguments');
  switch(command){
    case 'prepare-diagnostic':prepare('diagnostic');break;
    case 'diagnostic':diagnostic();break;
    case 'prepare-train':prepare('train');break;
    case 'train':train();break;
    case 'prepare-probes':prepare('probes');break;
    case 'readout':readout();break;
    case 'summary':summary();break;
    default:throw Error(`unknown command: ${command}`);
  }
}
try{main();}catch(error){console.error(error.stack??String(error));process.exitCode=1;}

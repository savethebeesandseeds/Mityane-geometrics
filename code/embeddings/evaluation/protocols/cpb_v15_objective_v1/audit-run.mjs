// Copyright (c) 2026 Waajacu. See LICENSE.
// Metadata orchestration only. The isolated native auditor owns array arithmetic.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {admitFiles,canonicalDirectory,hashAdmitted,readAdmitted,parseJSON,jsonBytes,
  safeRelative,writeNew,requireCondition as require} from '../../../../shared/artifacts/node/v1/files.mjs';

const ROOT='/mityane',BUILD='/opt/mityane-geometrics/embeddings/build/cpb-v15-objective-001';
const ATTEMPT='.temp/embeddings/runs/cpb-v15-objective-001/attempt-001';
const PREFIX='code/embeddings/evaluation/protocols/cpb_v15_objective_v1';
const SEEDS=[945101,945102,945103],ARMS=['control','candidate'],REPS=[2701,2802,2903],MASTERS=[941101,941102,941103];
const same=(a,b)=>a?.path===b?.path&&a?.bytes===b?.bytes&&a?.sha256===b?.sha256&&(a?.repository??'Mityane-geometrics')===(b?.repository??'Mityane-geometrics');
const passed=(value,why)=>require(value?.status==='passed'&&value?.passed===true,why);
function absolute(name,binary=false){
  if(binary){require(path.dirname(name)===BUILD&&path.isAbsolute(name),'isolated named-build auditor required');return name;}
  return path.join(ROOT,safeRelative(name));
}
function observed(name,binary=false){const file=absolute(name,binary),row=admitFiles([file],{root:binary?BUILD:ROOT,singleLink:true})[0];return {repository:'Mityane-geometrics',path:name,bytes:row.bytes,sha256:hashAdmitted(row)};}
function current(pin,binary=false){require(pin?.repository===undefined||pin.repository==='Mityane-geometrics','current repository required');const actual=observed(pin.path,binary);require(same(actual,pin),`audit binding changed: ${pin.path}`);return actual;}
function read(pin){current(pin);return parseJSON(readAdmitted(admitFiles([absolute(pin.path)],{root:ROOT,singleLink:true})[0]));}
function exactPhase(pin,cardPin,phase,seed,arm){const value=read(pin);passed(value,'actual producer phase must pass');require(value.kind==='cpb_v15_objective_subrun_phase_v1'&&same(value.card,cardPin)&&value.phase===phase&&value.seed===seed&&value.arm===arm&&Array.isArray(value.outputs),'exact actual producer phase/card/seed/arm required');for(const item of value.outputs)current(item);return value;}
function unique(pins){const values=new Map();for(const pin of pins){if(values.has(pin.path))require(same(values.get(pin.path),pin),'conflicting duplicate audit identity');values.set(pin.path,pin);}return [...values.values()].sort((a,b)=>a.path.localeCompare(b.path));}

export function requireAuditBindings(source,build,cfg){
  passed(source,'independent auditor source review required');passed(build,'actual reviewed auditor build required');
  require(source.kind==='cpb_v15_independent_source_review_v1'&&build.kind==='cpb_v15_independent_audit_build_v1','exact independent source/build review kinds required');
  const requiredSources=['audit.cpp','audit-run.mjs','audit-run.test.mjs','audit_fixture.cpp','audit_test.mjs','audit.mk'].map(name=>`${PREFIX}/${name}`);
  requiredSources.push('code/shared/artifacts/node/v1/files.mjs');
  require(Array.isArray(source.files)&&requiredSources.every(name=>source.files.filter(pin=>pin.path===name).length===1),'complete independent native audit source closure required');
  const binaries=Array.isArray(build.binaries)?build.binaries.filter(pin=>pin.role==='independent_audit'):[];
  require(same(build.source_review,cfg.source_review)&&binaries.length===1&&same(binaries[0],cfg.binary),'actual auditor binary/source build binding required');
  require(cfg.binary.path===`${BUILD}/cpb_v15_independent_audit`,'isolated auditor binary role');
  return {source,build};
}

export function requireFullDevelopmentSeal(seal,cardPin){
  passed(seal,'actual complete development seal required');
  require(seal.kind==='cpb_v15_all_six_development_runs_sealed_v1'&&seal.subruns===6&&seal.independent_numeric_audits===6&&same(seal.card,cardPin)&&seal.confirmation_numerical_outcomes_opened===false,'same actual six-run blind development seal required');
  require(Array.isArray(seal.records)&&seal.records.length===12&&Array.isArray(seal.audits)&&seal.audits.length===6&&Array.isArray(seal.outputs)&&seal.outputs.length>0,'all twelve phase receipts, six audits and actual output closure required');
  const roots=SEEDS.flatMap(seed=>ARMS.map(arm=>`${ATTEMPT}/subruns/seed-${seed}/${arm}`));
  const records=roots.flatMap(base=>['train','development'].map(phase=>`${base}/${phase}-receipt.json`));
  const audits=roots.map(base=>`${base}/independent-audit.json`);
  require(records.every(name=>seal.records.filter(pin=>pin.path===name).length===1)&&audits.every(name=>seal.audits.filter(pin=>pin.path===name).length===1)&&new Set(seal.outputs.map(pin=>pin.path)).size===seal.outputs.length,'exact unique six-run seal record/audit/output identities required');
  const requiredOutputs=roots.flatMap(base=>[`${base}/training/checkpoints/inference-2048.pt`,...REPS.map(rep=>`${base}/heads-final/market/models/native36/rep-${rep}/fit.pt`)]);
  require(requiredOutputs.every(name=>seal.outputs.filter(pin=>pin.path===name).length===1),'all six final checkpoints and eighteen original TRAIN fits required inside seal');
  return seal;
}

export function expectedDecodedPin(name,{base,phase_outputs,study_inputs,confirmation_target=null}){
  const local=name.startsWith(`${base}/`);
  require(Array.isArray(local?phase_outputs:study_inputs),'explicit decoded source identities required');
  const candidates=local?phase_outputs:[...(study_inputs??[]),...(confirmation_target?[confirmation_target]:[])];
  require(Array.isArray(candidates),'explicit decoded source identities required');
  const matches=candidates.filter(pin=>pin.path===name);
  require(matches.length===1,local?'decoded numerical array absent or duplicated in actual phase outputs':'decoded external numerical array absent or duplicated in frozen study inputs');
  return matches[0];
}

export function buildPlan(card,base,phase){
  const rows=[];
  const add=(mode,...files)=>rows.push([mode,...files]);
  const trainingTargets=`${card.shared}/features/train-targets.pt`;
  if(phase==='development'){
    for(const stage of ['initial','final']){
      const features=`${base}/features/${stage}/market`,heads=`${base}/heads-${stage}`;
      for(const rep of REPS)for(const view of ['train','validation'])add('regression',`${features}/cpb-objective-train.pt`,trainingTargets,
        `${heads}/market/models/native36/rep-${rep}/fit.pt`,`${features}/cpb-objective-${view}.pt`,`${card.shared}/features/${view}-targets.pt`,`${heads}/market/models/native36/rep-${rep}/predictions_${view}.pt`);
      for(const master of MASTERS)for(const policy of ['synthetic_train','market_scaler']){
        const method=`${heads}/synthetic/master-${master}/${policy}`;
        for(const rep of REPS)for(const view of ['training','validation-intact','validation-deleted'])add('classification',`${method}/training-features.pt`,`${method}/rep-${rep}/fit.pt`,`${method}/${view}-features.pt`,`${method}/rep-${rep}/${view}-predictions.pt`);
      }
      for(const view of ['train','validation'])add('fixed_query',`${base}/training/witnesses/${stage}-${view}-query.pt`);
    }
    for(const update of [1,2048])add(`objective_${base.endsWith('/candidate')?'candidate':'control'}`,`${base}/training/witnesses/objective-update-${update}.pt`);
    require(rows.length===126,'complete12market/108synthetic/4query/2objective set count');
  }else{
    require(phase==='confirmation','only development/confirmation audit');
    for(const rep of REPS)add('regression',`${base}/features/final/market/cpb-objective-train.pt`,trainingTargets,
      `${base}/heads-final/market/models/native36/rep-${rep}/fit.pt`,`${base}/confirmation-features/confirmation-features.pt`,card.confirmation_targets,`${base}/confirmation-readouts/rep-${rep}/predictions_confirmation.pt`);
    add('fixed_query',`${base}/confirmation-features/confirmation-query-witness.pt`);
    require(rows.length===4,'all3confirmationheads and full32query witness');
  }
  return rows;
}

export function executeAudit(inputsPath){
  require(process.platform==='linux'&&fs.existsSync('/.dockerenv'),'managed Linux container required');canonicalDirectory(ROOT);
  const inputs=observed(inputsPath),cfg=read(inputs);
  require(cfg.kind==='cpb_v15_independent_audit_inputs_v1'&&cfg.root===ROOT&&cfg.attempt===ATTEMPT&&SEEDS.includes(cfg.seed)&&ARMS.includes(cfg.arm)&&['development','confirmation'].includes(cfg.phase),'actual one-subrun independent audit card required');
  const base=`${ATTEMPT}/subruns/seed-${cfg.seed}/${cfg.arm}`;
  canonicalDirectory(absolute(base),{root:ROOT});
  const stem=cfg.phase==='development'?'independent':'confirmation';
  const leaves={plan:`${base}/${stem}-plan.tsv`,numeric:`${base}/${stem}-numeric.json`,log:`${base}/${stem}-audit.log`,receipt:`${base}/${cfg.phase==='development'?'independent-audit':'confirmation-audit'}.json`};
  for(const name of Object.values(leaves))require(!fs.existsSync(absolute(name)),`existing audit output retained: ${name}`);
  let planPin=null,numericPin=null,logPin=null;const checked=[],decoded=[],bindings=[];
  try{
    const source=read(cfg.source_review),build=read(cfg.build_admission);requireAuditBindings(source,build,cfg);current(cfg.binary,true);
    bindings.push(inputs,cfg.source_review,cfg.build_admission,cfg.study_card,...source.files);for(const pin of bindings)current(pin);
    const card=read(cfg.study_card);require(card.kind==='cpb_v15_objective_run_inputs_v1'&&card.root===ROOT&&card.attempt===ATTEMPT,'exact study card');
    const trainPin=observed(`${base}/train-receipt.json`),devPin=observed(`${base}/development-receipt.json`);
    const train=exactPhase(trainPin,cfg.study_card,'train',cfg.seed,cfg.arm),development=exactPhase(devPin,cfg.study_card,'development',cfg.seed,cfg.arm);
    bindings.push(trainPin,devPin);checked.push(...train.outputs,...development.outputs);
    let confirmationTarget=null;
    if(cfg.phase==='confirmation'){
      const confirmPin=observed(`${base}/confirmation-receipt.json`),confirmation=exactPhase(confirmPin,cfg.study_card,'confirmation',cfg.seed,cfg.arm);
      const sealPin=observed(`${ATTEMPT}/development-seal.json`),seal=requireFullDevelopmentSeal(read(sealPin),cfg.study_card);
      const sealReferents=[...seal.records,...seal.audits,...seal.outputs];
      for(const pin of sealReferents)current(pin);bindings.push(...sealReferents);
      const targets=read(card.confirmation.targets_receipt);passed(targets,'actual blind prepared confirmation targets');require(targets.kind==='cpb_v15_confirmation_targets_prepared_v1'&&targets.blind===true&&targets.numerical_outcomes_exposed===false,'confirmation targets preparation boundary');
      confirmationTarget=targets.targets;card.confirmation_targets=confirmationTarget.path;bindings.push(confirmPin,sealPin,card.confirmation.targets_receipt,confirmationTarget);checked.push(...confirmation.outputs);
    }
    const rows=buildPlan(card,base,cfg.phase);
    for(const row of rows)for(const name of row.slice(1)){
      const expected=expectedDecodedPin(name,{base,phase_outputs:checked,study_inputs:card.input_files,confirmation_target:confirmationTarget});
      const pin=current(expected);require(same(pin,{...expected,path:name}),'decoded numerical array identity changed');decoded.push(pin);
    }
    const plan=rows.map(row=>[row[0],...row.slice(1).map(name=>absolute(name))].join('\t')).join('\n')+'\n';
    writeNew(absolute(leaves.plan),Buffer.from(plan));planPin=observed(leaves.plan);
    const descriptor=fs.openSync(absolute(leaves.log),'wx');let result;
    try{result=spawnSync(cfg.binary.path,['--plan',absolute(leaves.plan),'--output',absolute(leaves.numeric)],{cwd:ROOT,env:{...process.env,OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1'},stdio:['ignore',descriptor,descriptor],timeout:1800000});}finally{fs.closeSync(descriptor);}
    logPin=observed(leaves.log);if(fs.existsSync(absolute(leaves.numeric)))numericPin=observed(leaves.numeric);
    require(!result.error&&result.status===0,`independent native arithmetic failed: ${result.error?.message??`exit ${result.status}`}`);
    const numeric=read(numericPin);passed(numeric,'actual native arithmetic result required');require(numeric.kind==='cpb_v15_independent_saved_fit_audit_v1'&&numeric.array_sets===rows.length&&numeric.model_forwards===0&&numeric.head_fits===0&&numeric.optimizer_replays===0,'complete independent saved-array result');
    for(const pin of unique([...checked,...decoded,...bindings,planPin,numericPin,logPin]))current(pin);current(cfg.binary,true);
    const receipt={kind:'cpb_v15_independent_numeric_audit_v1',status:'passed',passed:true,created_utc:new Date().toISOString(),seed:cfg.seed,arm:cfg.arm,phase:cfg.phase,inputs,study_card:cfg.study_card,source_review:cfg.source_review,build_admission:cfg.build_admission,binary:cfg.binary,plan:planPin,numeric_result:numericPin,log:logPin,
      checked_files:unique(checked),decoded_files:unique(decoded),independent_arithmetic:true,producer_module_forwards:0,producer_arithmetic_called:false,python_executed:false,
      limitations:['Checkpoint and remaining phase outputs are hashed; their full model/optimizer arrays are not independently recomputed.','January query waveforms cover first8 rows perpopulation; full mean checks use producer savedrowmetrics.','No optimizer replay, model forward or bootstrap-interval verification; objective arithmetic does not establish gradient routes.']};
    writeNew(absolute(leaves.receipt),jsonBytes(receipt));return receipt;
  }catch(error){writeNew(absolute(leaves.receipt),jsonBytes({kind:'cpb_v15_independent_numeric_audit_v1',status:'failed',passed:false,created_utc:new Date().toISOString(),inputs,seed:cfg.seed,arm:cfg.arm,phase:cfg.phase,source_review:cfg.source_review,build_admission:cfg.build_admission,binary:cfg.binary,plan:planPin,numeric_result:numericPin,log:logPin,failures:[String(error.message)],scientific_result_claimed:false}));throw error;}
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  try{require(process.argv.length===4&&process.argv[2]==='--inputs','audit-run --inputs CARD');safeRelative(process.argv[3]);const result=executeAudit(process.argv[3]);process.stdout.write(jsonBytes({status:result.status,phase:result.phase,seed:result.seed,arm:result.arm}));}
  catch(error){process.stderr.write(`${error.stack}\n`);process.exitCode=1;}
}

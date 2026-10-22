// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Metadata-only closeout: no process spawning, model calls or tensor decoding.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const HERE=path.dirname(fileURLToPath(import.meta.url));
const ROOT=path.resolve(HERE,'../../../../..');
const BASE='.temp/embeddings/runs/cpb-v17-balance-001';
const ATTEMPT=`${BASE}/attempt-001`;
const FAMILY='code/embeddings/encoders/causal_patch_bottleneck_mae';
const RECORD='code/embeddings/evaluation/records/CPB-v17.balance-001.json';
const RESULTS='code/embeddings/evaluation/protocols/cpb_v17_balance_v1/RESULTS.md';
const CONTRACT='code/embeddings/evaluation/protocols/cpb_v17_balance_v1/CONTRACT.md';
const CONTRACT_ADMISSION=`${ATTEMPT}/readout-contract-admission.json`;
const MANIFEST=`${ATTEMPT}/closeout-input-manifest.json`;
const TASKS=['past_volatility','past_trend','past_activity','past_relation'];
const SEEDS=[947101,947102,947103];
const EXPECTED_REFERENCE_SHA='b6f3cbe711235307f362bf2f999571d65d00c85825f8a4a5c2fbc4fa4242e51e';
const assert=(condition,message)=>{if(!condition)throw Error(message);};
const absolute=name=>path.isAbsolute(name)?name:path.join(ROOT,name);
const relative=name=>name.startsWith(`${ROOT}/`)?path.relative(ROOT,name).split(path.sep).join('/'):name;
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const read=name=>JSON.parse(fs.readFileSync(absolute(name),'utf8'));
const canonical=p=>({...p,path:relative(absolute(p.path))});
function pin(name){
  const p=absolute(name),s=fs.lstatSync(p);assert(s.isFile()&&!s.isSymbolicLink(),`regular nonsymlink artifact required: ${name}`);
  const bytes=fs.readFileSync(p);assert(bytes.length===s.size,`artifact changed while hashing: ${name}`);
  return {path:relative(p),bytes:bytes.length,sha256:hash(bytes)};
}
const checked=new Map();
function check(expected){
  assert(expected&&typeof expected.path==='string'&&Number.isSafeInteger(expected.bytes)&&/^[a-f0-9]{64}$/.test(expected.sha256),'complete artifact pin required');
  const key=absolute(expected.path),prior=checked.get(key);
  if(prior){assert(prior.bytes===expected.bytes&&prior.sha256===expected.sha256,`conflicting evidence pins: ${expected.path}`);return canonical(expected);}
  const actual=pin(expected.path);assert(actual.bytes===expected.bytes&&actual.sha256===expected.sha256,`bound evidence changed: ${expected.path}`);
  checked.set(key,actual);return actual;
}
function uniquePins(pins){
  const out=new Map();for(const raw of pins){const p=canonical(raw),old=out.get(p.path);assert(!old||(old.sha256===p.sha256&&old.bytes===p.bytes),`source identity conflict: ${p.path}`);out.set(p.path,p);}
  return [...out.values()].sort((a,b)=>a.path.localeCompare(b.path));
}
function textPin(name,text){const bytes=Buffer.from(text);return {path:name,bytes:bytes.length,sha256:hash(bytes)};}
function saveNew(name,text){
  const fd=fs.openSync(absolute(name),'wx');try{fs.writeFileSync(fd,text);fs.fsyncSync(fd);}finally{fs.closeSync(fd);}
}
function quantile(sorted,p){const index=(sorted.length-1)*p,lo=Math.floor(index),fraction=index-lo;return sorted[lo]*(1-fraction)+sorted[Math.min(lo+1,sorted.length-1)]*fraction;}
function distribution(values){
  assert(values.every(v=>v===null||Number.isFinite(v)),'finite or explicit null diagnostic metric required');
  const sorted=values.filter(v=>v!==null).sort((a,b)=>a-b);
  return {batches:values.length,finite_batches:sorted.length,null_batches:values.length-sorted.length,
    minimum:sorted.length?sorted[0]:null,q25:sorted.length?quantile(sorted,.25):null,
    median:sorted.length?quantile(sorted,.5):null,q75:sorted.length?quantile(sorted,.75):null,maximum:sorted.length?sorted.at(-1):null};
}
const same=(a,b,message)=>assert(JSON.stringify(a)===JSON.stringify(b),message);
const integer=(n,label)=>{assert(Number.isSafeInteger(n)&&n>=0,`nonnegative exact count required: ${label}`);return n;};

function collect(buildHistoryPath){
  assert(process.platform==='linux'&&ROOT==='/mityane'&&fs.existsSync('/.dockerenv'),'Run closeout only in the managed mityane-geometrics container.');
  for(const name of [RECORD,RESULTS,MANIFEST])assert(!fs.existsSync(absolute(name)),`preserve existing closeout output: ${name}`);
  const admission=read(`${ATTEMPT}/admission.json`),admissionPin=pin(`${ATTEMPT}/admission.json`);
  assert(admission.kind==='cpb_v17_balance_study_admission_v1'&&admission.status==='admitted','prospective study admission required');
  [...admission.sources,...admission.inputs,...admission.engineering,...admission.frozen_numerical_baseline].forEach(check);check(admission.protocol_copy);
  assert(admission.frozen_numerical_baseline.filter(p=>p.path.startsWith(`${FAMILY}/versions/v16/`)).length===49,'49 frozen V16 source pins required');
  assert(admission.frozen_numerical_baseline.filter(p=>p.path.startsWith(`${FAMILY}/family/include/`)).length===3,'three frozen family headers required');
  const phases={};for(const name of ['windows','diagnostic','train','readout']){
    const p=`${ATTEMPT}/${name}.admission.json`,r=read(p);assert(r.phase===name,'wrong phase admission');
    check(r.study_admission);[...r.sources,...r.executables].forEach(check);phases[name]={artifact:pin(p),data:r};
  }
  const receiptFiles=fs.readdirSync(absolute(`${ATTEMPT}/receipts`)).filter(n=>n.endsWith('.json')&&!n.endsWith('.command.json')).sort();
  const receipts=[];for(const file of receiptFiles){
    const name=`${ATTEMPT}/receipts/${file}`,r=read(name);assert(r.status==='passed'&&r.exit_code===0,`failed/incomplete study step must not be hidden: ${file}`);
    [...r.inputs,...r.outputs].forEach(check);check(r.stdout);check(r.stderr);check(r.command);receipts.push({label:r.label,artifact:pin(name)});
  }
  assert(receipts.length===28,'expected3 preparation +1 diagnostic +12 training +6 readout +6 audit receipts');
  const engineeringExecutionPath=`${BASE}/engineering-execution.json`,engineeringExecution=read(engineeringExecutionPath);
  assert(engineeringExecution.status==='passed'&&engineeringExecution.exit_code===0,'passed artificial engineering execution required');
  [engineeringExecution.source_admission,engineeringExecution.stdout,engineeringExecution.stderr].forEach(check);
  const engineeringAdmission=read(engineeringExecution.source_admission.path);engineeringAdmission.sources.forEach(check);check(engineeringAdmission.executable);
  const engineeringPath=`${BASE}/engineering-001/summary.json`,engineering=read(engineeringPath);
  assert(engineering.status==='passed'&&engineering.quality_evaluated===false,'artificial engineering receipt required');
  const buildHistory=read(buildHistoryPath);assert(buildHistory.kind==='cpb_v17_pre_execution_build_history_v1','explicit pre-execution build history required');
  assert(buildHistory.first_build.shell_exit_code!==0&&buildHistory.first_build.optimizer_updates===0&&buildHistory.first_build.model_executions===0&&buildHistory.corrected_build.shell_exit_code===0,'initial compile refusal must precede all model execution');
  assert(/transcription/i.test(buildHistory.record_source)&&/not.*raw/i.test(buildHistory.record_source),'build-history transcription limitation must remain explicit');
  const engineeringEvidence={execution:pin(engineeringExecutionPath),command:pin(`${BASE}/engineering-command.json`),summary:pin(engineeringPath),
    admissions:['engineering-source-admission.json','engineering-source-admission-v2.json','engineering-admission-v3.json'].map(n=>pin(`${BASE}/${n}`)),build_history:pin(buildHistoryPath)};
  const preparation={};for(const mode of ['diagnostic','train','probes']){
    const p=`${ATTEMPT}/data-${mode}/preparation.json`,r=read(p);assert(r.status==='passed'&&r.actual_role_membership_verified===true&&r.mode===mode,'typed dataset admission failed');
    r.inputs.forEach(check);check(r.executable);
    preparation[mode]={receipt:pin(p),source_identity:r.source_identity,artifacts:r.artifacts.map(a=>({role:a.role,rows:a.rows,split_identity:a.split_identity,
      retained_history:a.retained_history,selection_history:a.selection_history,role_begin:a.role_begin,role_end:a.role_end,
      archive:check({path:`${ATTEMPT}/data-${mode}/${a.file}`,bytes:a.bytes,sha256:a.sha256})}))};
  }
  const diagnosticPath=`${ATTEMPT}/diagnostic/summary.json`,diagnostic=read(diagnosticPath);check(diagnostic.batches_artifact);
  assert(diagnostic.status==='passed'&&diagnostic.batches===8&&diagnostic.rows===64&&diagnostic.parameter_gradient_evaluations===32&&diagnostic.physical_training_updates===0&&diagnostic.preprocessing_fits===0&&diagnostic.original_and_adapted_state_unchanged===true,'zero-update diagnostic boundary failed');
  const batches=fs.readFileSync(absolute(diagnostic.batches_artifact.path),'utf8').trim().split('\n').map(JSON.parse);assert(batches.length===8,'eight diagnostic batches required');
  batches.forEach((b,i)=>assert(b.attempt===i&&b.row_ids.length===8,'fixed diagnostic batching changed'));
  const diagnosticFields=['gaussian_l2_f64','temporal_l2_f64','dot_f64','cosine_f64','weighted_temporal_to_gaussian_ratio_0_1','weighted_temporal_to_gaussian_ratio_0_03',
    'actual_combined_0_1_l2_f64','actual_combined_0_03_l2_f64','actual_combined_0_1_l2_model_dtype','actual_combined_0_03_l2_model_dtype','hypothetical_clip1_scale_0_1','hypothetical_clip1_scale_0_03'];
  const groupSummaries={};for(const group of Object.keys(diagnostic.group_definitions)){
    const values=batches.map(b=>b.gradients[group]);assert(values.every(Boolean),`missing diagnostic group: ${group}`);
    groupSummaries[group]={definition:diagnostic.group_definitions[group],parameter_tensors:values[0].parameter_tensors,parameter_values:values[0].parameter_values,
      distributions:Object.fromEntries(diagnosticFields.map(field=>[field,distribution(values.map(v=>v[field]))]))};
  }
  const diagnosticSummary={receipt:pin(diagnosticPath),batches:check(diagnostic.batches_artifact),checkpoint:check(diagnostic.checkpoint),rows:64,batch_size:8,
    mask_seed:diagnostic.mask_seed,parameter_gradient_evaluations:32,physical_training_updates:0,preprocessing_fits:0,heldout_readouts:0,
    checkpoint_updates:diagnostic.checkpoint_updates,checkpoint_initialization_seed:diagnostic.checkpoint_initialization_seed,
    original_and_adapted_state_unchanged:true,groups:groupSummaries,limitations:diagnostic.limitations};
  diagnosticSummary.local_observations={
    all_learned_global_positive_cosine_batches:batches.filter(b=>b.gradients.all_learned.cosine_f64>0).length,
    all_learned_global_negative_cosine_batches:batches.filter(b=>b.gradients.all_learned.cosine_f64<0).length,
    backbone_negative_cosine_batches:batches.filter(b=>b.gradients.backbone_without_waveform_decoder.cosine_f64<0).length,
    variance_temporal_gradient_zero_batches:batches.filter(b=>b.gradients.variance_head.temporal_l2_f64===0).length,
    interpretation_scope:'These points can show global alignment alongside conflict in a parameter subgroup. They do not establish gradient dominance or its absence throughout fresh training, or identify the cause of historical regressions.'};
  const sealPath=`${ATTEMPT}/training-seal.json`,seal=read(sealPath);assert(seal.status==='passed'&&seal.trajectories.length===6&&seal.logical_updates===1536&&seal.recorded_physical_updates===1536,'complete1536-update training seal required');
  const contractAdmission=read(CONTRACT_ADMISSION),contractPin=check(contractAdmission.contract);
  assert(contractPin.path===CONTRACT,'readout supplement must bind the actual maintained operation contract');
  [contractAdmission.protocol,contractAdmission.study_admission,contractAdmission.training_seal].forEach(check);
  assert(contractAdmission.protocol.sha256===admission.protocol_copy.sha256&&contractAdmission.study_admission.sha256===admissionPin.sha256&&contractAdmission.training_seal.sha256===pin(sealPath).sha256,'operation-contract supplement changed the admitted protocol/study/seal association');
  const contractTime=Date.parse(contractAdmission.admitted_utc);
  assert(Number.isFinite(contractTime)&&contractTime>=Date.parse(seal.sealed_utc),'operation-contract supplement must follow the completed training seal');
  const contractAdmissionPin=check(pin(CONTRACT_ADMISSION));
  const screenPath=`${ATTEMPT}/screen-summary.json`,screen=read(screenPath);assert(screen.status==='complete'&&screen.logical_updates===1536&&screen.recorded_physical_updates===1536&&screen.head_fits===24&&screen.independent_audits===6,'complete bounded screen summary required');check(screen.training_seal);
  const trajectories=[],trainingReports=[];let physicalUpdates=0,attempts=0,headFits=0,audits=0,fittingPasses=0;
  for(const [index,t] of seal.trajectories.entries()){
    const scheduled=admission.schedule[index];same([t.id,t.seed,t.arm,t.weight],[scheduled.id,scheduled.seed,scheduled.arm,scheduled.weight],'trajectory order differs from prospective schedule');
    const reports=[];for(const [stage,target] of [['fresh-128',128],['resume-256',256]]){
      const p=`${ATTEMPT}/trajectories/${t.id}/${stage}/run.json`,r=read(p),resuming=target===256;
      assert(r.status==='passed'&&r.resume===resuming&&r.completed_updates===target&&r.starting_completed_updates===target-128&&r.physical_training_updates_this_invocation===128,'fresh128/resume256 accounting failed');
      assert(r.temporal_weight===t.weight&&r.initialization_seed===t.seed&&r.row_seed===t.seed+1000&&r.mask_seed===t.seed+2000&&r.context_seed===t.seed+3000,'paired recipe/stream mismatch');
      assert(r.maximum_total_attempts===512&&r.attempted_batches<=512&&r.preprocessing_fitted===!resuming&&r.preprocessing_fit_windows===(resuming?0:1024),'bounded attempt/fitting policy failed');
      assert(r.final_inference_reload_verified===true&&r.quality_evaluated===false,'training export verification required');Object.values(r.artifacts).forEach(check);
      physicalUpdates+=integer(r.physical_training_updates_this_invocation,'physical updates');if(r.preprocessing_fitted)++fittingPasses;
      const compact={stage,report:pin(p),completed_updates:r.completed_updates,attempted_batches:r.attempted_batches,starting_updates:r.starting_completed_updates,
        starting_attempts:r.starting_attempted_batches,physical_updates:r.physical_training_updates_this_invocation,skipped_attempts:r.skipped_attempts_this_invocation,
        preprocessing_fitted:r.preprocessing_fitted,preprocessing_fit_windows:r.preprocessing_fit_windows,recipe_identity:r.recipe_identity,
        initial_state_identity:r.initial_state_identity,final_state_identity:r.final_state_identity,optimizer_identity:r.optimizer_identity,scaler_identity:r.scaler_identity,bank_identity:r.bank_identity,
        training_state:check(r.artifacts.training_state),inference:check(r.artifacts.inference),resume_input:resuming?check(r.artifacts.resume_input):null};
      trainingReports.push(compact.report);reports.push(compact);
    }
    const [fresh,final]=reports;assert(final.resume_input.sha256===fresh.training_state.sha256&&final.starting_attempts===fresh.attempted_batches,'resume bound to wrong saved state');
    for(const key of ['recipe_identity','initial_state_identity','scaler_identity','bank_identity'])assert(fresh[key]===final[key],`resume changed ${key}`);
    attempts+=final.attempted_batches;check(t.fresh_report);check(t.final_report);check(t.final_inference);
    const readoutPath=`${ATTEMPT}/readouts/${t.id}/summary.json`,r=read(readoutPath),auditPath=`${ATTEMPT}/audits/${t.id}/summary.json`,a=read(auditPath);
    const headCommand=read(`${ATTEMPT}/receipts/${t.id}-readout.command.json`);
    assert(contractTime<=Date.parse(headCommand.started_utc),'operation contract was not admitted before head execution');
    assert(r.updates===256&&r.temporal_weight===t.weight&&r.head_fits===4&&r.encoder_updates===0&&r.preprocessing_fits===0&&r.encoder_state_unchanged===true&&r.target_precision_parity===true,'readout boundary/parity failed');
    assert(a.status==='passed'&&a.targets===4&&a.absolute_tolerance===1e-9&&a.relative_tolerance===1e-8&&a.original_legacy_target_bits_equal===true&&a.original_row_group_association_equal===true,'independent audit failed');
    for(const task of TASKS){const metric=r.metrics.find(m=>m.task===task),independent=a.independent_metrics.find(m=>m.task===task);assert(metric&&independent&&metric.coverage===1&&independent.coverage===1,'complete target coverage required');
      for(const key of ['r2','mae','rmse','coverage'])assert(Number.isFinite(metric[key])&&Math.abs(metric[key]-independent[key])<=1e-9+1e-8*Math.max(Math.abs(metric[key]),Math.abs(independent[key])),`independent metric mismatch: ${task}/${key}`);}
    headFits+=r.head_fits;++audits;
    trajectories.push({id:t.id,seed:t.seed,arm:t.arm,temporal_weight:t.weight,training:reports,metrics:r.metrics,
      readout:pin(readoutPath),witness:check(r.witness),audit:pin(auditPath),audit_metrics:a.independent_metrics});
  }
  assert(physicalUpdates===1536&&headFits===24&&audits===6&&fittingPasses===6,'actual work counts differ from bounded execution');
  for(const seed of SEEDS){const pair=trajectories.filter(t=>t.seed===seed),a=pair.find(t=>t.arm==='control'),b=pair.find(t=>t.arm==='candidate');
    for(const key of ['initial_state_identity','scaler_identity','bank_identity'])assert(a.training[0][key]===b.training[0][key],`pair initialization/fitting identity mismatch: ${seed}/${key}`);}
  const paired=SEEDS.map(seed=>{const pair=trajectories.filter(t=>t.seed===seed),a=pair.find(t=>t.arm==='control'),b=pair.find(t=>t.arm==='candidate');return {seed,...Object.fromEntries(TASKS.map(task=>[task,b.metrics.find(m=>m.task===task).r2-a.metrics.find(m=>m.task===task).r2]))};});
  const aggregate=Object.fromEntries(TASKS.map(task=>{const v=paired.map(p=>p[task]);return [task,{median:[...v].sort((a,b)=>a-b)[1],minimum:Math.min(...v),maximum:Math.max(...v),nonnegative_pairs:v.filter(x=>x>=0).length,positive_pairs:v.filter(x=>x>0).length}];}));
  same(paired,screen.paired_candidate_minus_control_r2,'screen paired differences do not match actual readouts');same(aggregate,screen.aggregate,'screen aggregates do not match actual readouts');
  const conditions={trend_median_nonnegative:aggregate.past_trend.median>=0,trend_at_least_two_nonnegative_pairs:aggregate.past_trend.nonnegative_pairs>=2,
    activity_median_nonnegative:aggregate.past_activity.median>=0,activity_at_least_two_nonnegative_pairs:aggregate.past_activity.nonnegative_pairs>=2,volatility_median_floor:aggregate.past_volatility.median>=-.01};
  same(conditions,screen.predeclared_conditions,'screen criteria do not match actual readouts');
  const passed=Object.values(conditions).every(Boolean);assert(passed===screen.predeclared_continuation_criterion_passed,'screen verdict mismatch');
  assert(screen.observed_gain===paired.some(p=>TASKS.some(task=>p[task]>0))&&screen.strict_trend_or_activity_gain_observed===paired.some(p=>p.past_trend>0||p.past_activity>0),'screen observed-gain flags differ from measured pairs');
  const referenceDocument=read(`${FAMILY}/research-reference.json`),reference=referenceDocument.reference;check(reference.checkpoint);
  assert(reference.version==='CPB-v16'&&reference.encoder_seed===946103&&reference.logical_checkpoint_update===2048&&reference.checkpoint.sha256===EXPECTED_REFERENCE_SHA,'selected research reference unexpectedly changed');
  check(reference.TRAIN_input_scaler);check(reference.TRAIN_bank_standardizer);
  const sources=uniquePins([...admission.sources,...Object.values(phases).flatMap(p=>p.data.sources),...engineeringAdmission.sources,contractPin]);
  const executables=uniquePins([...Object.values(phases).flatMap(p=>p.data.executables),engineeringAdmission.executable]);
  const sourceDigest=hash(Buffer.from(JSON.stringify(sources))),executableDigest=hash(Buffer.from(JSON.stringify(executables)));
  const manifest={kind:'cpb_v17_closeout_verified_input_manifest_v1',created_utc:new Date().toISOString(),generator:pin(fileURLToPath(import.meta.url)),
    checked_artifacts:[...checked.values()].sort((a,b)=>a.path.localeCompare(b.path)),sources,source_closure_sha256:sourceDigest,executables,executable_set_sha256:executableDigest,
    admissions:{study:admissionPin,phases:Object.fromEntries(Object.entries(phases).map(([name,p])=>[name,p.artifact]))},command_receipts:receipts};
  const manifestText=JSON.stringify(manifest,null,2)+'\n';
  const record={kind:'cpb_v17_balance_study_closeout_v1',study_id:'cpb-v17-balance-001',created_utc:manifest.created_utc,work_status:'complete',study_valid_and_complete:true,
    question:'At fixed 256 updates, does temporal strength 0.03 improve the balance of trend/activity and volatility against fresh 0.1 controls under the same portable recipe?',
    decision:{predeclared_continuation_criterion_passed:passed,predeclared_conditions:conditions,observed_gain:screen.observed_gain,strict_trend_or_activity_gain_observed:screen.strict_trend_or_activity_gain_observed,
      automatic_promotion:false,new_best_encoder_established:false,research_reference_replaced:false,application_selection_changed:false,
      interpretation:passed?'The bounded development criterion passed; this supports consideration of a separately declared longer study.':'The bounded development criterion failed; these results do not support advancing this strength ablation under the declared rule.'},
    comparison:{control_temporal_weight:.1,candidate_temporal_weight:.03,encoder_seed_pairs:3,initialization:'fresh within each arm; exact same-seed numerical initialization/scaler/bank identities',
      geometry:'C3/H128/F9/P8/native36; channel IDs0,1,2',calibration:'1024 evenly spaced TRAIN windows; rounded-parent Chan fitter, B8 source order',
      context_deletion:'none',fixed_recipe:admission.fixed_recipe,
      historical_control_limit:'Fresh 0.1 controls share the new fitter/calibration/sampler/context recipe; they are not historical V16 trajectory reproductions.',
      January_readout:'112 TRAIN and 32 known January VALIDATION H256 nonoverlap anchor geometry; recent 128 encoder input; exact legacy F32-to-F64 target parity; TRAIN-standardized ridge 1.'},
    paired_candidate_minus_control_r2:paired,aggregate,trajectories,
    work_counts:{scientific:{fresh_encoder_trajectories:6,planned_native_training_processes:12,logical_optimizer_updates:1536,recorded_physical_optimizer_updates:physicalUpdates,
      attempted_batches:attempts,TRAIN_preprocessing_fits:fittingPasses,TRAIN_windows_per_fit:1024,linear_target_head_fits:headFits,independent_scalar_audits:audits,
      independent_scalar_target_refits_for_verification:audits*4,
      zero_update_diagnostic_batches:8,zero_update_diagnostic_parameter_derivative_evaluations:32,diagnostic_optimizer_updates:0},
      engineering:{assertions:integer(engineering.assertions,'engineering assertions'),named_gradient_tensors_compared:integer(engineering.named_gradient_tensors_compared,'engineering gradient tensors'),
        physical_synthetic_optimizer_updates:integer(engineering.physical_synthetic_updates,'engineering updates'),preprocessing_fits:integer(engineering.preprocessing_fits,'engineering fits'),
        deliberate_runtime_refusals:integer(engineering.refusals,'engineering refusals'),quality_evaluated:false},
      pre_execution_compile_refusal:{count:1,optimizer_updates:0,model_executions:0,source:buildHistory.record_source,first_build:buildHistory.first_build,
        corrected_build:buildHistory.corrected_build,evidence:engineeringEvidence.build_history},
      distinction:'Scientific 1536 updates, artificial engineering updates and a failed compile are separate counts. No replayed scientific trajectory is included.'},
    diagnostic:diagnosticSummary,
    current_reference:{version:'CPB-v16',seed:946103,logical_update:2048,checkpoint:check(reference.checkpoint),TRAIN_input_scaler:check(reference.TRAIN_input_scaler),TRAIN_bank_standardizer:check(reference.TRAIN_bank_standardizer),
      declaration:pin(`${FAMILY}/research-reference.json`),unchanged:true},
    measured:{final256_January_linear_readouts:true,exact_legacy_target_ID_group_support_parity:true,independent_scalar_ridge_arithmetic:true,TRAIN_gradient_diagnostic:true,planned_actual_128_to_256_resume:true,typed_final_inference_reload:true,
      fixed_query_reconstruction:false,TEMPO_timing:false,neural_readouts:false,fresh_population_confirmation:false,full_budget_superiority:false,forecasting:false,application_utility:false},
    limits:['January is exposed development evidence; January/February/March 2024 have informed prior work and are consumed for subsequent choices.',
      'Three fresh seed pairs and 256 updates do not establish a universal or full-budget best encoder.',
      'The zero-update diagnostic describes gradients at 64 TRAIN windows from the selected trained V16 checkpoint; it does not establish the cause of historical regressions.',
      'Null cosines/ratios retain zero-denominator cases. Clip scales are hypothetical and no diagnostic optimizer step occurred.',
      'Raw gradient norms and cosines do not measure AdamW-preconditioned update contributions.',
      'Matching FNV tensor-byte numerical identities are engineering witnesses, not cryptographic equality proofs.',
      'Original strongest timing protections remain unchanged but were not rerun. Fixed-query reconstruction, neural heads, fresh confirmation and application docking were not measured.',
      'A longer study requires separate prospective scope, bindings and authorization.'],
    evidence:{study_admission:admissionPin,protocol_copy:check(admission.protocol_copy),operation_contract:contractPin,contract_admission:contractAdmissionPin,
      phase_admissions:Object.fromEntries(Object.entries(phases).map(([name,p])=>[name,p.artifact])),
      engineering:engineeringEvidence,preparation,training_seal:pin(sealPath),training_reports:trainingReports,screen_summary:pin(screenPath),
      command_receipts:receipts,verified_input_manifest:textPin(MANIFEST,manifestText),
      source_closure:{unique_files:sources.length,canonical_sorted_pin_array_sha256:sourceDigest,complete_pins_in:MANIFEST},
      executables,executable_set_sha256:executableDigest,frozen_V16_source_files_unchanged:49,frozen_family_numerical_headers_unchanged:3},
    reproduction:{generator:pin(fileURLToPath(import.meta.url)),model_calls:0,external_processes:0,tensor_archives_decoded:0,publication_payload:'Compact records and source only; raw datasets, model states, full generated evidence and logs remain ignored local artifacts.'}};
  return {record,manifestText};
}

const number=n=>n===null?'null':Number.isInteger(n)?String(n):n.toPrecision(7);
const sign=n=>`${n>=0?'+':''}${number(n)}`;
function markdown(record){
  const lines=['# CPB V17 temporal-strength bounded study','',record.decision.interpretation,'',
    '**The selected research reference remains V16 seed 946103/final 2048.** This 256-update development screen does not establish a new best encoder.','',
    'The candidate uses temporal weight 0.03; the fresh control uses 0.1. Both share the corrected configurable native36 component, 1024-window Chan TRAIN fitting, context deletion NONE, B8, the same seed streams and one AdamW/global clip 1. Each trajectory starts fresh, stops at 128 and resumes its exact typed state to 256. These controls do not reproduce the historical V16 training trajectory.','',
    '## Final256 January development readouts','',
    'R2 below uses 112 TRAIN/32 known-validation probe anchors, original legacy target arithmetic and TRAIN-standardized ridge 1. All six readout runs have complete support and independent arithmetic audits. Full-precision R2/MAE/RMSE/coverage and checkpoint identities are retained in the compact JSON record.','',
    '| Seed | Arm | Volatility R2 | Trend R2 | Activity R2 | Relation R2 |','| --- | --- | ---: | ---: | ---: | ---: |'];
  for(const t of record.trajectories)lines.push(`| ${t.seed} | ${t.arm} (${t.temporal_weight}) | ${TASKS.map(task=>number(t.metrics.find(m=>m.task===task).r2)).join(' | ')} |`);
  lines.push('','Candidate minus same-seed control:','','| Task | Seed 947101 | Seed 947102 | Seed 947103 | Median | Minimum | Maximum | Nonnegative pairs |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
  for(const task of TASKS){const a=record.aggregate[task];lines.push(`| ${task} | ${record.paired_candidate_minus_control_r2.map(p=>sign(p[task])).join(' | ')} | ${sign(a.median)} | ${sign(a.minimum)} | ${sign(a.maximum)} | ${a.nonnegative_pairs}/3 |`);}
  lines.push('','| Predeclared condition | Result |','| --- | --- |');
  for(const [condition,passed] of Object.entries(record.decision.predeclared_conditions))lines.push(`| ${condition.replaceAll('_',' ')} | ${passed?'PASS':'FAIL'} |`);
  lines.push('',`The declared continuation criterion **${record.decision.predeclared_continuation_criterion_passed?'passed':'failed'}**. An observed positive trend/activity pair difference ${record.decision.strict_trend_or_activity_gain_observed?'occurred':'did not occur'}. Equal scores satisfy nonnegative conditions but are not an improvement. No automatic promotion or longer execution follows.`,'',
    '## TRAIN-only zero-update diagnostic','',
    'Eight fixed B8 batches from 64 TRAIN windows were evaluated using the selected frozen V16/final 2048 checkpoint and its original fitted statistics. The 32 parameter-derivative evaluations performed no preprocessing fit, optimizer step or artifact export. These local gradients do not identify the cause of earlier quality changes.','',
    'Distributions below are medians [minimum, maximum] over the eight batches; null denominator cases remain null in the JSON record. The coefficient-weighted ratio is temporal-gradient norm divided by Gaussian-gradient norm. Clipping scales are hypothetical.','',
    '| Parameter group | Gaussian L2 | Temporal L2 | Cosine | Weighted ratio 0.1 | Weighted ratio 0.03 | Clip scale 0.1 | Clip scale 0.03 |','| --- | --- | --- | --- | --- | --- | --- | --- |');
  const fields=['gaussian_l2_f64','temporal_l2_f64','cosine_f64','weighted_temporal_to_gaussian_ratio_0_1','weighted_temporal_to_gaussian_ratio_0_03','hypothetical_clip1_scale_0_1','hypothetical_clip1_scale_0_03'];
  const range=d=>d.finite_batches?`${number(d.median)} [${number(d.minimum)}, ${number(d.maximum)}]${d.null_batches?`; ${d.null_batches} null`:''}`:`null (${d.null_batches}/8)`;
  for(const [group,data] of Object.entries(record.diagnostic.groups))lines.push(`| ${group.replaceAll('_',' ')} | ${fields.map(f=>range(data.distributions[f])).join(' | ')} |`);
  const local=record.diagnostic.local_observations;
  lines.push('',`Global learned-gradient cosines were positive in ${local.all_learned_global_positive_cosine_batches}/8 batches and negative in ${local.all_learned_global_negative_cosine_batches}/8. The backbone subgroup had ${local.backbone_negative_cosine_batches}/8 negative cosines. Global alignment and subgroup conflict are different observations; neither establishes behavior at every fresh-training point. Raw gradient norms and cosines do not measure AdamW-preconditioned update contributions. The variance-head temporal gradient was zero in ${local.variance_temporal_gradient_zero_batches}/8 batches.`);
  const work=record.work_counts;lines.push('','## Actual work and preserved failures','',
    `Scientific work: six fresh trajectories, 12 planned training processes, ${work.scientific.recorded_physical_optimizer_updates} physical/logical optimizer updates, ${work.scientific.attempted_batches} attempts, six TRAIN preprocessing fits, 24 linear target-head fits and six independent scalar audits. Those audits separately recomputed 24 target fits for arithmetic verification. The diagnostic added 32 derivative evaluations and zero training updates.`,'',
    `Artificial engineering work is separate: ${work.engineering.assertions} assertions, ${work.engineering.named_gradient_tensors_compared} named gradient tensors compared, ${work.engineering.physical_synthetic_optimizer_updates} synthetic optimizer updates, ${work.engineering.preprocessing_fits} preprocessing fits and ${work.engineering.deliberate_runtime_refusals} deliberate runtime refusals. The initial failed compile executed zero models/updates. Its preserved record is a transcription of the actual Codex tool output; a separately captured raw compiler log is unavailable. Earlier source admissions and the corrected build remain linked.`,'',
    '## Scope and evidence','',
    'Fixed-query reconstruction, TEMPO timing, neural heads, fresh-population confirmation, full-budget superiority, forecasting and application utility were not measured. January/February/March 2024 are consumed evidence for later selection. V16’s historical mixed verdict and timing limitations remain in its original records.','',
    'The reusable component retains configurable construction, TRAIN fitting, the complete training objective, typed resume/export and the training source. A research checkpoint and its benchmark shape do not define the family’s full supported interface.','',
    'The [OR-05 Operation Contract and Requirements-and-Evidence Record](CONTRACT.md) restates this admitted readout scope. Its supplemental admission binds the unchanged original protocol, study admission and completed training seal before any head execution.','',
    `The [compact closeout record](../../records/CPB-v17.balance-001.json) links the full prospective admission, phase source/executable pins, engineering record, three preparation receipts, diagnostic batches, training seal, 12 training reports, six readouts, six independent audits and final summary. Full ignored evidence is preserved at \`${ATTEMPT}/\`; the preserved \`closeout-input-manifest.json\` there retains verified source and artifact pins. Frozen 49 V16 files, three family numerical headers and the selected reference checkpoint/statistics were checked unchanged at closeout.`,
    '',`Generated from actual saved evidence by \`closeout-study.mjs\` at ${record.created_utc}; no model execution or tensor decoding.`, '');
  return lines.join('\n');
}

function main(){
  const args=process.argv.slice(2);if(args.length===1&&args[0]==='--help'){console.log('closeout-study.mjs [--build-history FILE] — after screen-summary.json exists; new-only compact JSON, RESULTS.md and ignored verified-input manifest. No model/process calls.');return;}
  assert(args.length===0||(args.length===2&&args[0]==='--build-history'),'only optional --build-history FILE is accepted');
  const {record,manifestText}=collect(args[1]??`${BASE}/engineering-build-history.json`);
  const resultsText=markdown(record);record.evidence.results=textPin(RESULTS,resultsText);
  const recordText=JSON.stringify(record,null,2)+'\n';
  saveNew(MANIFEST,manifestText);saveNew(RESULTS,resultsText);saveNew(RECORD,recordText);
  console.log(JSON.stringify({record:pin(RECORD),results:pin(RESULTS),verified_input_manifest:pin(MANIFEST),status:'saved',model_calls:0},null,2));
}
try{main();}catch(error){console.error(error.stack??String(error));process.exitCode=1;}

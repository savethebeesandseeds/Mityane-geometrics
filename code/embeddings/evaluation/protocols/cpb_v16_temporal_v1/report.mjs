// Copyright (c) 2026 Waajacu. See LICENSE.
// Reporting only: saved audited arrays, unchanged historical floors, no fitting.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {admitFiles,hashAdmitted,readAdmitted,parseJSON,jsonBytes,safeRelative,
  requireCondition as require,writeNew} from '../../../../shared/artifacts/node/v1/files.mjs';
import {currentPin} from './prepare.mjs';
import {ATTEMPT,SEEDS,ARMS,subrun,requireRunCard,requirePhaseReceipt,requireDevelopmentSeal} from './run.mjs';

const ROOT='/mityane',PREFIX='code/embeddings/evaluation/protocols/cpb_v16_temporal_v1';
export const TASKS=Object.freeze(['past_volatility','past_trend','past_activity','past_relation']);
const REPS=[2701,2802,2903],HEAD_SEEDS=['13088411117524848790','10018485049408248344','9083751155097190455'];
const MASTERS=[941101,941102,941103],MODES=['synthetic_train','market_scaler'];
const VIEWS=['training','validation_intact','validation_deleted'];
const GOAL={repository:'cuwacunu_embedding',path:'doc/causal_fresh_v14/goal-protocol.json',bytes:17536,sha256:'4ace8b79a496ff24440ef1a269b63b45b746624f51dbd734e1b9c013b8a93245'};
const COMPILED_ARMS={control:'fresh_v4_partial_nll_control',candidate:'v4_nll_plus_hidden_temporal_huber_v16'};
export const TEMPORAL_CONTRACT=Object.freeze({huber_delta:1,weight:.1,base_nll_weight:1,lags:Object.freeze([1,4,16]),endpoint_scale:'sqrt2',support:'both-original-hidden-target-endpoints',lag_reduction:'fixed-one-third-including-empty'});
const passed=(value,message)=>require(value?.passed===true&&value.status==='passed',message);
const samePin=(a,b)=>a?.path===b?.path&&a.bytes===b.bytes&&a.sha256===b.sha256;
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const mean=values=>values.every(finite)&&values.length?values.reduce((sum,value)=>sum+value,0)/values.length:null;
const median=values=>{if(values.length!==3||!values.every(finite))return null;return [...values].sort((a,b)=>a-b)[1];};
function close(actual,expected,message){require(finite(actual)&&finite(expected)&&Math.abs(actual-expected)<=2e-9+2e-8*Math.abs(expected),message);}
function compareMetric(actual,expected,message){if(actual===null||expected===null)require(actual===expected,message);else close(actual,expected,message);}
function exactlyOne(rows,predicate,message){const selected=rows.filter(predicate);require(selected.length===1,message);return selected[0];}
function observed(relative){const row=admitFiles([path.join(ROOT,safeRelative(relative))],{root:ROOT,singleLink:true})[0];return {repository:'Mityane-geometrics',path:relative,bytes:row.bytes,sha256:hashAdmitted(row)};}
function read(pin){currentPin(ROOT,pin);return parseJSON(readAdmitted(admitFiles([path.join(ROOT,pin.path)],{root:ROOT,singleLink:true})[0]));}
function absolute(relative){return path.posix.join(ROOT,safeRelative(relative));}
export function requireTrainingSummary(value,seed,arm,gradientPolicy){
  require(value.kind==='cpb_v16_temporal_paired_training_summary_v1'&&value.status==='passed'&&value.initialization_seed===seed&&value.arm===COMPILED_ARMS[arm]&&value.gradient_policy===gradientPolicy&&value.new_causal_updates===2048&&value.batch_size===8&&value.typed_resume_512_exact===true,'actual completed paired fresh trajectory and compiled arm identity');
}
export function requireTemporalContract(value){
  require(value&&Object.entries(TEMPORAL_CONTRACT).every(([key,expected])=>Array.isArray(expected)?Array.isArray(value[key])&&value[key].length===expected.length&&value[key].every((v,i)=>v===expected[i]):value[key]===expected),'exact V4 NLL plus fixed hidden temporal Huber contract required');
  return value;
}
export function requireTemporalAuditEntry(value,update){
  require(value?.kind==='objective_temporal'&&value.update===update&&[1,2048].includes(update)&&Number.isSafeInteger(value.attempt)&&value.attempt>=update-1&&value.attempt<4096,'actual declared temporal witness step/zero-based attempt required');
  require(Array.isArray(value.lags)&&value.lags.length===3&&value.lags.every((v,i)=>v===TEMPORAL_CONTRACT.lags[i])&&value.outer_weight===.1&&value.huber_delta===1&&value.difference_divisor==='sqrt2'&&Array.isArray(value.lag_weights)&&value.lag_weights.length===3&&value.lag_weights.every(v=>v===1/3),'exact temporal witness lags, weights, scale and Huber delta required');
  require(Array.isArray(value.lag_pair_totals)&&value.lag_pair_totals.length===3&&value.lag_pair_totals.every(v=>Number.isSafeInteger(v)&&v>=0)&&Array.isArray(value.lag_supported)&&value.lag_supported.length===3&&value.lag_supported.every((v,i)=>typeof v==='boolean'&&v===(value.lag_pair_totals[i]>0))&&Array.isArray(value.lag_losses)&&value.lag_losses.length===3&&value.lag_losses.every((v,i)=>finite(v)&&v>=0&&(value.lag_supported[i]||v===0)),'temporal lag support, empty zeros and loss counts required');
  require(value.same_step_operands_bitwise_equal===true&&value.actual_preclip_output_gradients_audited===true&&value.arithmetic_only===true&&value.gradient_reconstruction==='direct_output_adjoints_only'&&value.encoder_jacobian_reconstructed===false&&value.optimizer_replayed===false&&value.global_clipping_reconstructed===false,'bounded same-step temporal arithmetic and direct-output-adjoint audit required');
  require(finite(value.base_nll)&&value.gradient_absolute_tolerance===2e-7&&value.gradient_relative_tolerance===2e-5,'finite base NLL and frozen direct-output-gradient audit tolerances required');
  close(value.temporal_loss,value.lag_losses.reduce((sum,loss)=>sum+loss,0)/3,'temporal objective retains fixed one-third reduction including empty lags');
  close(value.total_loss,value.base_nll+.1*value.temporal_loss,'V4 base Gaussian NLL plus weighted temporal objective');
  return value;
}

export function compareCriterion(criterion,value){
  require(finite(criterion.floor),'finite frozen criterion floor required');
  let supported=finite(value),result=false;
  switch(criterion.rule){
    case 'candidate + 1e-12 >= floor':if(supported)result=value+1e-12>=criterion.floor;break;
    case 'candidate <= floor + 1e-12':if(supported)result=value<=criterion.floor+1e-12;break;
    case 'candidate < floor - 1e-12':if(supported)result=value<criterion.floor-1e-12;break;
    case 'candidate > 0':require(criterion.floor===0,'unchanged strict positivity floor');if(supported)result=value>0;break;
    case 'all three final neural trend R2 values > 0':require(criterion.floor===0,'unchanged three-head positivity floor');supported=Array.isArray(value)&&value.length===3&&value.every(finite);if(supported)result=value.every(number=>number>0);break;
    default:require(false,`unknown frozen criterion rule: ${criterion.rule}`);
  }
  return {...criterion,candidate:value,supported,passed:supported&&result};
}

export function aggregateMarket(repetitions){
  require(Array.isArray(repetitions)&&repetitions.length===3&&repetitions.every((r,i)=>r.repetition===REPS[i]&&r.actual_seed_decimal===HEAD_SEEDS[i]&&r.tasks?.length===4),'exact three D36 repetitions and four target roles');
  const rows={};
  for(const [index,task] of TASKS.entries()){
    const targets=repetitions.map(rep=>rep.tasks[index]);
    for(const value of targets)for(const head of ['ridge','neural']){
      const score=value[head];require(score&&Number.isSafeInteger(score.total)&&Number.isSafeInteger(score.valid)&&Number.isSafeInteger(score.target_eligible)&&score.total===32&&score.target_eligible===32&&score.valid>=0&&score.valid<=32,'declared32 target population required');
      close(score.coverage,score.valid/32,'market support fraction mismatch');
      for(const metric of ['mae','rmse','r2'])require(score[metric]===null||finite(score[metric]),'finite or explicitly undefined audited market metric');
    }
    const ridge=targets[0].ridge;
    for(const other of targets.slice(1))for(const metric of ['mae','rmse','r2','coverage'])compareMetric(other.ridge[metric],ridge[metric],'linear ridge is deterministic and reused across head repetitions');
    const row={linear_r2:ridge.r2,linear_mae:ridge.mae,linear_rmse:ridge.rmse,
      neural_r2_each:targets.map(t=>t.neural.r2),neural_mae_each:targets.map(t=>t.neural.mae),neural_rmse_each:targets.map(t=>t.neural.rmse),
      coverage:Math.min(...targets.flatMap(t=>[t.ridge.coverage,t.neural.coverage])),valid:Math.min(...targets.flatMap(t=>[t.ridge.valid,t.neural.valid])),total:32,
      repetitions:repetitions.map((rep,i)=>({repetition:rep.repetition,actual_seed_decimal:rep.actual_seed_decimal,...targets[i]}))};
    for(const metric of ['r2','mae','rmse'])row[`neural_mean_${metric}`]=mean(row[`neural_${metric}_each`]);
    rows[task]=row;
  }
  return rows;
}

export function aggregateTempo(cohorts){
  require(Array.isArray(cohorts)&&cohorts.length===6,'all three masters and two input policies required');
  const scores={};
  for(const mode of MODES){scores[mode]={};
    const selected=MASTERS.map(master=>exactlyOne(cohorts,c=>c.master===master&&c.input_policy===mode,'unique declared master/policy cohort'));
    for(const cohort of selected)require(cohort.repetitions?.length===3&&cohort.repetitions.every((r,i)=>r.repetition===REPS[i]&&r.actual_seed_decimal===HEAD_SEEDS[i]),'unchanged D36 TEMPO head repetitions');
    for(const view of VIEWS){
      const total=view==='training'?512:256;
      for(const cohort of selected)for(const rep of cohort.repetitions){const s=rep[view];require(s&&s.total===total&&Number.isSafeInteger(s.valid)&&s.valid>=0&&s.valid<=total,'complete original TEMPO population identity');close(s.coverage,s.valid/total,'TEMPO coverage arithmetic');for(const key of ['ridge_accuracy','neural_accuracy'])require(s[key]===null||(finite(s[key])&&s[key]>=0&&s[key]<=1),'bounded TEMPO accuracy');}
      const ridgeMasters=selected.map(c=>mean(c.repetitions.map(r=>r[view].ridge_accuracy))),neuralMasters=selected.map(c=>mean(c.repetitions.map(r=>r[view].neural_accuracy)));
      scores[mode][view]={linear_head_accuracy:mean(ridgeMasters),neural_head_accuracy:mean(neuralMasters),linear_master_means:ridgeMasters,neural_master_means:neuralMasters,
        coverage:Math.min(...selected.flatMap(c=>c.repetitions.map(r=>r[view].coverage))),
        aggregation:'arithmetic mean of three master means; each master mean averages three fixed head repetitions'};
    }
  }
  return {scores,cohorts};
}

export function historicalRetention(goal,market,tempo,queries){
  const values={};
  for(const task of TASKS)for(const metric of ['linear_r2','neural_mean_r2'])values[`${task}:${metric}`]=market[task][metric];
  for(const mode of MODES)for(const view of ['validation_intact','validation_deleted'])for(const metric of ['linear_head_accuracy','neural_head_accuracy'])values[`${mode}:${view}:${metric}`]=tempo.scores[mode][view][metric];
  for(const role of ['train','validation'])values[`${role}:standardized_MAE`]=queries[role].standardized_fixed_query_MAE;
  values['known_validation:Gaussian_NLL_F64']=queries.validation.Gaussian_NLL_F64;
  values['past_trend:all3_neural_R2_positive']=market.past_trend.neural_r2_each;
  values['past_relation:linear_r2:positive']=market.past_relation.linear_r2;
  values['past_relation:neural_mean_r2:positive']=market.past_relation.neural_mean_r2;
  require(goal.criterion_count===22&&goal.criteria?.length===22&&new Set(goal.criteria.map(c=>c.id)).size===22&&Object.keys(values).every(id=>goal.criteria.some(c=>c.id===id)),'exact unchanged22 historical criterion identities');
  const criteria=goal.criteria.map(criterion=>compareCriterion(criterion,values[criterion.id]));
  const timing=criteria.filter(c=>c.category==='tempo5_retention');require(timing.length===8,'exact eight historical hard timing conditions');
  const full_coverage=TASKS.every(t=>market[t].coverage===1)&&MODES.every(m=>VIEWS.every(v=>tempo.scores[m][v].coverage===1))&&queries.train.valid_rows===112&&queries.validation.valid_rows===32;
  return {criteria,passed:criteria.filter(c=>c.passed).length,total:22,full_coverage,all22_passed:full_coverage&&criteria.every(c=>c.passed),hard_timing_passed:full_coverage&&timing.every(c=>c.passed),historical_floor_objects_unchanged:true,TRAIN_NLL_additional_gate:false};
}

export function pairedScreen(pairs){
  const complete=Array.isArray(pairs)&&pairs.length===3&&pairs.every((p,i)=>p.seed===SEEDS[i]&&p.control&&p.candidate&&typeof p.candidate.hard_timing_passed==='boolean');
  if(!complete)return {verdict:'inconclusive',promising:false,complete:false,reasons:['all three paired encoder runs and eight timing checks are required']};
  const differences={};
  for(const task of ['past_trend','past_activity','past_volatility'])differences[task]=pairs.map(p=>{const a=p.control.confirmation?.[task]?.linear_r2,b=p.candidate.confirmation?.[task]?.linear_r2;return finite(a)&&finite(b)?b-a:null;});
  const full=pairs.every(p=>[p.control,p.candidate].every(r=>r.full_declared_coverage===true&&TASKS.every(t=>r.confirmation?.[t]?.coverage===1)));
  const supported=full&&Object.values(differences).every(values=>values.every(finite))&&pairs.every(p=>[p.control,p.candidate].every(r=>TASKS.every(t=>finite(r.confirmation?.[t]?.linear_r2)&&finite(r.confirmation?.[t]?.neural_mean_r2))));
  const targets=Object.fromEntries(Object.entries(differences).map(([task,values])=>[task,{paired_differences:values,median:median(values),range:values.every(finite)?[Math.min(...values),Math.max(...values)]:null,positive_pairs:values.filter(v=>finite(v)&&v>0).length}]));
  const primary=['past_trend','past_activity'].every(task=>targets[task].median>0&&targets[task].positive_pairs>=2);
  const volatility=targets.past_volatility.median!==null&&targets.past_volatility.median>=-.01,timing=pairs.every(p=>p.candidate.hard_timing_passed);
  const promising=supported&&primary&&volatility&&timing;
  const reasons=[];if(!supported)reasons.push('undefined R2 or incomplete declared confirmation support');
  for(const task of ['past_trend','past_activity'])if(!(targets[task].median>0&&targets[task].positive_pairs>=2))reasons.push(`${task} requires positive median and at least two improving pairs`);
  if(!volatility)reasons.push('March linear volatility median difference is below -0.01 or undefined');if(!timing)reasons.push('at least one candidate fails a frozen hard timing condition');
  const negative=supported&&['past_trend','past_activity'].every(task=>targets[task].median<=0);
  return {verdict:!supported?'inconclusive':promising?'promising':negative?'negative':'mixed',promising,complete:true,targets,primary_targets:['past_trend','past_activity'],primary_targets_passed:primary,volatility_tolerance_passed:volatility,all_candidates_hard_timing_passed:timing,reasons,
    encoder_pairs:3,head_repetitions_are_not_encoder_replicates:true,overall_winner:false,Mityane_application_selection:null,automatic_promotion:false};
}

function planRows(pin){
  currentPin(ROOT,pin);const text=readAdmitted(admitFiles([path.join(ROOT,pin.path)],{root:ROOT,singleLink:true})[0]).toString('utf8');
  require(text.length>0&&text.length<=16000000&&!text.includes('\r')&&!text.includes('\0')&&text.endsWith('\n'),'bounded actual LF audit plan required');
  return text.slice(0,-1).split('\n').map(line=>{const parts=line.split('\t'),mode=parts.shift();const count={regression:6,classification:4,objective_control:1,objective_temporal:2,fixed_query:1}[mode];require(parts.length===count&&parts.every(p=>p.startsWith(`${ROOT}/`)&&path.posix.normalize(p)===p),'exact explicit numeric audit plan rows');return {mode,paths:parts};});
}
function bindAudit(pin,expectedOutputs,allPins,binaryPins){
  const wrapper=read(pin);passed(wrapper,'actual independent arithmetic wrapper required');
  require(wrapper.kind==='cpb_v16_independent_numeric_audit_v1'&&wrapper.independent_arithmetic===true&&wrapper.producer_module_forwards===0&&wrapper.producer_arithmetic_called===false&&Array.isArray(wrapper.checked_files)&&Array.isArray(wrapper.decoded_files),'actual independent scope and hash/decoded distinction required');
  for(const output of expectedOutputs)require(wrapper.checked_files.some(p=>samePin(p,output)),'arithmetic wrapper must bind every actual phase output');
  const proofPins=[pin,wrapper.numeric_result,wrapper.plan,wrapper.source_review,wrapper.build_admission,wrapper.log,...wrapper.checked_files,...wrapper.decoded_files];
  for(const p of proofPins)currentPin(ROOT,p);currentPin(ROOT,wrapper.binary,{binary:true});
  const source=read(wrapper.source_review),build=read(wrapper.build_admission);passed(source,'actual audit source review required');passed(build,'actual audit build admission required');
  require(samePin(build.source_review,wrapper.source_review)&&build.binaries?.some(p=>samePin(p,wrapper.binary)),'audit binary must bind actual reviewed source build');
  for(const p of source.files??[])currentPin(ROOT,p);
  require(source.files?.some(p=>p.path===`${PREFIX}/audit.cpp`),'independent arithmetic source must be reviewed');
  const numeric=read(wrapper.numeric_result);passed(numeric,'actual independent numeric result required');
  require(numeric.kind==='cpb_v16_independent_saved_fit_audit_v1'&&numeric.model_forwards===0&&numeric.head_fits===0&&numeric.optimizer_replays===0&&numeric.bootstrap_intervals_audited===false&&Array.isArray(numeric.entries),'exact independent numeric result schema required');
  const rows=planRows(wrapper.plan);require(numeric.array_sets===rows.length&&numeric.entries.length===rows.length,'numeric result must cover every explicit plan row');
  for(const [i,row] of rows.entries()){
    const entry=numeric.entries[i];require(entry.kind===row.mode,'audit mode association must retain plan order');
    if(row.mode==='regression')require(entry.fit===row.paths[2]&&entry.predictions===row.paths[5],'audit regression association must match its bound plan');
    else if(row.mode==='classification')require(entry.fit===row.paths[1]&&entry.predictions===row.paths[3],'audit classification association must match its bound plan');
    else {require(entry.witness===row.paths[0],'waveform witness association must match actual plan');if(row.mode==='objective_temporal')require(entry.original_objective_witness===row.paths[1],'temporal evidence must bind original same-step objective witness');}
    for(const p of row.paths){const relative=p.slice(ROOT.length+1);require(wrapper.decoded_files.some(pin=>pin.path===relative),'every numeric-plan file must be in the declared decoded subset');}
  }
  allPins.push(...proofPins,...(source.files??[]));binaryPins.push(wrapper.binary);return {wrapper,numeric,rows};
}
export function requireAuditTuple(row,mode,expected){
  require(row?.mode===mode&&Array.isArray(row.paths)&&row.paths.length===expected.length&&row.paths.every((p,i)=>p===expected[i]),'numeric audit must bind the complete exact stage/data/target/fit/prediction input tuple');
}
function auditEntry(audit,mode,key,value,expected){
  const entry=exactlyOne(audit.numeric.entries,e=>e.kind===mode&&e[key]===absolute(value),'required unique numerical audit entry absent or duplicated');
  if(expected)requireAuditTuple(audit.rows[audit.numeric.entries.indexOf(entry)],mode,expected.map(absolute));
  return entry;
}
function producer(pin,outputPins,allPins){require(outputPins.some(p=>samePin(p,pin)),'producer JSON must be inside an actual phase output receipt');allPins.push(pin);return read(pin);}
function marketStage(base,stage,phaseOutputs,audit,studyPin,study,allPins){
  const report=producer(observed(`${base}/heads-${stage}/market/report.json`),phaseOutputs,allPins);
  require(report.kind==='fixed_past_window_regression_v1'&&report.dataset_id==='MKT-HIST-2'&&report.card_sha256===studyPin.sha256&&report.input_manifest_sha256===study.input_manifest.sha256&&report.training_rows===112&&report.validation_rows===32,'unchanged January market population/card');
  require(report.heads?.ridge_penalty===1&&report.heads.neural_hidden===16&&report.heads.neural_learning_rate===.01&&report.heads.neural_updates===100&&report.heads.outputs===4,'unchanged fixed market head recipe');
  require(report.methods?.length===1&&report.methods[0].name==='native36'&&report.methods[0].size===36,'only declared native36 head input');
  const method=report.methods[0];require(method.training_feature_rows===112&&method.feature_training_supported===true&&method.repetitions?.length===3,'complete market TRAIN fit support');
  const reps=method.repetitions.map((rep,index)=>{
    require(rep.repetition===REPS[index]&&String(rep.actual_seed)===HEAD_SEEDS[index]&&rep.neural_updates_completed===100&&rep.ridge_fit_reused===(index!==0)&&rep.targets?.length===4,'unchanged typed market seeds, three head fits and ridge reuse');
    const fit=`${base}/heads-${stage}/market/models/native36/rep-${REPS[index]}/fit.pt`,prediction=`${base}/heads-${stage}/market/models/native36/rep-${REPS[index]}/predictions_validation.pt`;
    const entry=auditEntry(audit,'regression','predictions',prediction,[`${base}/features/${stage}/market/cpb-objective-train.pt`,`${study.shared}/features/train-targets.pt`,fit,`${base}/features/${stage}/market/cpb-objective-validation.pt`,`${study.shared}/features/validation-targets.pt`,prediction]);
    require(entry.actual_seed_decimal===HEAD_SEEDS[index]&&entry.fit===absolute(`${base}/heads-${stage}/market/models/native36/rep-${REPS[index]}/fit.pt`)&&entry.tasks?.length===4,'audited market fit/seed association');
    for(const [i,target] of rep.targets.entries()){
      require(target.name===TASKS[i]&&target.train_target_eligible===112&&target.train_target_supported===true&&target.method_training_supported===true,'fixed four market target order and fit support');
      for(const head of ['ridge','neural'])for(const metric of ['r2','mae','rmse','coverage'])compareMetric(target[head][metric],entry.tasks[i][head][metric],'producer market score differs from independent arithmetic');
    }
    return {repetition:rep.repetition,actual_seed_decimal:entry.actual_seed_decimal,tasks:entry.tasks,producer_targets:rep.targets};
  });
  return aggregateMarket(reps);
}
function confirmationMarket(base,outputs,audit,study,preparedTargets,allPins){
  const reps=REPS.map((rep,index)=>{
    const report=producer(observed(`${base}/confirmation-readouts/rep-${rep}/report.json`),outputs,allPins);
    require(report.kind==='cpb_v16_saved_fit_confirmation_v1'&&report.status==='passed'&&report.rows===32&&report.head_fits===0&&report.normalizer_fits===0&&report.encoder_calls===0&&report.TRAIN_fitted_arrays_reused===true&&report.actual_seed_decimal===HEAD_SEEDS[index]&&report.targets?.length===4,'confirmation uses exact frozen TRAIN fits and no new fit');
    const fit=observed(`${base}/heads-final/market/models/native36/rep-${rep}/fit.pt`);allPins.push(fit);
    require(report.fit_path===absolute(fit.path)&&report.fit_sha256===fit.sha256,'confirmation report must bind original final TRAIN fit');
    const prediction=`${base}/confirmation-readouts/rep-${rep}/predictions_confirmation.pt`;
    const entry=auditEntry(audit,'regression','predictions',prediction,[`${base}/features/final/market/cpb-objective-train.pt`,`${study.shared}/features/train-targets.pt`,fit.path,`${base}/confirmation-features/confirmation-features.pt`,preparedTargets.path,prediction]);
    require(entry.fit===absolute(fit.path)&&entry.actual_seed_decimal===HEAD_SEEDS[index]&&entry.tasks?.length===4,'confirmation audit binds same original fit/seed');
    for(const [i,target] of report.targets.entries()){require(target.name===TASKS[i],'frozen confirmation target order');for(const head of ['ridge','neural'])for(const metric of ['r2','mae','rmse','coverage'])compareMetric(target[head][metric],entry.tasks[i][head][metric],'confirmation producer differs from independent arithmetic');}
    return {repetition:rep,actual_seed_decimal:entry.actual_seed_decimal,tasks:entry.tasks};
  });return aggregateMarket(reps);
}
function tempoStage(base,stage,outputs,audit,allPins){
  const cohorts=[];
  for(const master of MASTERS){
    const report=producer(observed(`${base}/heads-${stage}/synthetic/master-${master}/report.json`),outputs,allPins);
    require(report.protocol==='fixed-feature-readouts-v1'&&String(report.master_seed)===String(master)&&report.methods?.length===2,'fixed three TEMPO cohorts and both policies');
    const recipe=report.recipe;require(recipe?.ridge_penalty===1&&recipe.tiny_hidden===16&&recipe.tiny_steps===100&&recipe.tiny_learning_rate===.01&&recipe.validation_fits===0&&recipe.encoder_calls===0&&recipe.pca_fits===0,'unchanged TEMPO fixed head recipe');
    for(const mode of MODES){
      const method=exactlyOne(report.methods,m=>m.method===mode,'unique TEMPO policy');require(method.size===36&&method.status==='measured'&&method.repetitions?.length===3,'native36 measured TEMPO cohort');
      const repetitions=method.repetitions.map((rep,index)=>{
        require(rep.id===`rep-${REPS[index]}`&&rep.actual_probe_seed_decimal===HEAD_SEEDS[index]&&rep.status==='measured','unchanged TEMPO fixed repetition seeds');
        const row={repetition:REPS[index],actual_seed_decimal:rep.actual_probe_seed_decimal,conditional_intervals_from_producer:{},conditional_intervals_independently_audited:false};
        for(const view of VIEWS){
          const file=view==='training'?'training':view==='validation_intact'?'validation-intact':'validation-deleted';
          const folder=`${base}/heads-${stage}/synthetic/master-${master}/${mode}`,prediction=`${folder}/rep-${REPS[index]}/${file}-predictions.pt`;
          const entry=auditEntry(audit,'classification','predictions',prediction,[`${folder}/training-features.pt`,`${folder}/rep-${REPS[index]}/fit.pt`,`${folder}/${file}-features.pt`,prediction]);
          require(entry.fit===absolute(`${base}/heads-${stage}/synthetic/master-${master}/${mode}/rep-${REPS[index]}/fit.pt`)&&entry.actual_seed_decimal===HEAD_SEEDS[index],'TEMPO audit fit/seed association');
          const score=rep[view],n=view==='training'?512:256,groups=n/2;
          require(score?.population?.total_rows===n&&score.population.total_source_groups===groups&&score.population.valid_rows===entry.valid&&score.population.valid_source_groups===groups,'TEMPO exact rows and paired source groups');
          close(score.ridge.accuracy,entry.ridge_accuracy,'TEMPO producer ridge accuracy mismatch');close(score.tiny_secondary.accuracy,entry.neural_accuracy,'TEMPO producer neural accuracy mismatch');
          row[view]=entry;
          if(view!=='training')row.conditional_intervals_from_producer[view]={ridge:score.ridge_grouped_interval,neural:score.tiny_grouped_interval};
        }
        return row;
      });cohorts.push({master,input_policy:mode,repetitions});
    }
  }
  return aggregateTempo(cohorts);
}
function queryRows(value,rows){
  require(value.valid_rows===rows&&value.row_mae?.length===rows&&value.row_nll?.length===rows&&value.row_target_cells?.length===rows&&value.row_target_cells.every(n=>Number.isSafeInteger(n)&&n>0),'complete fixed-query saved row support');
  close(mean(value.row_mae),value.standardized_fixed_query_MAE,'whole-population query MAE mean');close(mean(value.row_nll),value.Gaussian_NLL_F64,'whole-population query NLL mean');
  require(value.target_cells===value.row_target_cells.reduce((a,b)=>a+b,0),'whole-population query target count');
  return value;
}
function querySummary(base,summary,audit){
  const result={};for(const stage of ['initial','final']){result[stage]={};for(const [role,rows] of [['train',112],['validation',32]]){
    const value=queryRows(summary[`fixed_query_${stage}`][role],rows),proof=auditEntry(audit,'fixed_query','witness',`${base}/training/witnesses/${stage}-${role}-query.pt`);
    require(proof.examples===8&&proof.valid_examples===8,'inherited training query witness audits first batch only');
    close(proof.mae,mean(value.row_mae.slice(0,8)),'first-eight query MAE arithmetic');close(proof.gaussian_nll,mean(value.row_nll.slice(0,8)),'first-eight query NLL arithmetic');
    result[stage][role]={...value,independent_waveform_examples:8,full_saved_row_reductions_recomputed:true,full_population_waveform_reconstructed:false};
  }}return result;
}
function pairedChanges(runs){
  return SEEDS.map(seed=>{
    const control=exactlyOne(runs,r=>r.seed===seed&&r.arm==='control','paired control'),candidate=exactlyOne(runs,r=>r.seed===seed&&r.arm==='candidate','paired candidate');
    const populations={};for(const population of ['january_initial','january_final','confirmation']){populations[population]={};for(const task of TASKS){populations[population][task]={};for(const metric of ['linear_r2','neural_mean_r2','linear_mae','linear_rmse','neural_mean_mae','neural_mean_rmse']){const a=control[population][task][metric],b=candidate[population][task][metric];populations[population][task][metric]={control:a,candidate:b,candidate_minus_control:finite(a)&&finite(b)?b-a:null};}}}
    return {seed,control,candidate,populations};
  });
}

export function requireReportCard(card){
  require(card?.kind==='cpb_v16_temporal_report_inputs_v1'&&card.root===ROOT&&card.attempt===ATTEMPT,'exact bounded reporting card required');
  require(Array.isArray(card.subruns)&&card.subruns.length===6&&SEEDS.every(seed=>ARMS.every(arm=>card.subruns.filter(r=>r.seed===seed&&r.arm===arm).length===1)),'exact declared six reporting subruns');
  require(card.output===`${ATTEMPT}/report.json`,'one exclusive final report leaf');
  require(card.historical_goal?.original?.repository===GOAL.repository&&samePin(card.historical_goal.original,GOAL)&&samePin({...card.historical_goal.current,path:GOAL.path},GOAL),'unchanged exact historical22 goal bytes and original identity');
  require(Array.isArray(card.historical_sources),'explicit original/current historical protection bindings required');
  return card;
}
const FINAL_BUILD='/opt/mityane-geometrics/embeddings/build/cpb-v16-temporal-001';
function finalPinIdentity(pin,binary){
  require(pin&&typeof pin==='object'&&!Array.isArray(pin)&&Number.isSafeInteger(pin.bytes)&&pin.bytes>=0&&typeof pin.sha256==='string'&&/^[0-9a-f]{64}$/.test(pin.sha256),'exact final-barrier file pin required');
  require(pin.repository===undefined||pin.repository==='Mityane-geometrics','current physical pins require Mityane repository context');
  if(binary)require(typeof pin.path==='string'&&path.posix.dirname(pin.path)===FINAL_BUILD&&path.posix.normalize(pin.path)===pin.path&&!pin.path.endsWith('/')&&!/[\0\t\r\n]/.test(pin.path),'one actual named-build binary leaf required');
  else safeRelative(pin.path);
  const repository=pin.repository??'Mityane-geometrics',mode=binary?'binary':'file';
  return {pin,binary,location:JSON.stringify([mode,repository,pin.path]),tuple:JSON.stringify([mode,repository,pin.path,pin.bytes,pin.sha256])};
}
export function planFinalVerification(pins,binaryPins){
  require(Array.isArray(pins)&&Array.isArray(binaryPins),'complete final-barrier pin arrays required');
  // Validate EVERY occurrence, including duplicates, before any filesystem read.
  const files=pins.map(pin=>finalPinIdentity(pin,false)),binaries=binaryPins.map(pin=>finalPinIdentity(pin,true));
  const locations=new Map();
  for(const item of [...files,...binaries]){
    const previous=locations.get(item.location);
    require(previous===undefined||previous===item.tuple,'conflicting final-barrier physical path identity');
    locations.set(item.location,item.tuple);
  }
  const retainLast=items=>{const retained=new Map();for(const item of items){retained.delete(item.tuple);retained.set(item.tuple,item.pin);}return [...retained.values()];};
  const uniqueFiles=retainLast(files),uniqueBinaries=retainLast(binaries);
  const bytes=items=>items.reduce((total,pin)=>total+BigInt(pin.bytes),0n).toString();
  return {files:uniqueFiles,binaries:uniqueBinaries,summary:{
    kind:'cpb_v16_final_barrier_unique_verification_v1',scope:'final_verification_barrier_only',
    file_pin_occurrences:pins.length,unique_file_fulltuples:uniqueFiles.length,duplicate_file_occurrences:pins.length-uniqueFiles.length,
    binary_pin_occurrences:binaryPins.length,unique_binary_fulltuples:uniqueBinaries.length,duplicate_binary_occurrences:binaryPins.length-uniqueBinaries.length,
    declared_file_occurrence_bytes_decimal:bytes(pins),declared_unique_file_bytes_decimal:bytes(uniqueFiles),
    full_tuple_fields:['physical_mode','repository','path','bytes','sha256'],undefined_repository_context:'Mityane-geometrics (unchanged currentPin default)',
    validation_before_final_hashes:true,conflicting_physical_path_pins_rejected:true,retention_order:'last occurrence',
    fresh_full_hash_per_retained_identity:true,unchanged_currentPin_guards:true,earlier_barrier_observations_reused:false,
    concurrent_writer_limitation:'Exact duplicate checks are coalesced within this final barrier on completed sealed artifacts; this is not observationally identical to repeated checks with concurrent writers. Initial and per-read checks remain separate and fresh; unique final files retain unchanged canonical, single-link and full-hash checks.'
  }};
}
export function verifyFinalBarrier(pins,binaryPins,{root=ROOT}={}){
  const plan=planFinalVerification(pins,binaryPins);
  for(const pin of plan.files)currentPin(root,pin);
  for(const pin of plan.binaries)currentPin(root,pin,{binary:true});
  return {...plan.summary,passed:true,status:'passed'};
}
export function makeReport(inputsPath){
  const inputs=observed(inputsPath),card=requireReportCard(read(inputs)),destination=path.join(ROOT,card.output);
  require(!fs.existsSync(destination),'preserve existing final report');
  const pins=[inputs],binaryPins=[],runs=[],failures=[];let header;
  try{
    const source=read(card.source_review),review=read(card.protocol_review),admission=read(card.admission);
    passed(source,'actual reporting source review required');passed(review,'actual reporting protocol review required');
    require(source.files&&[`${PREFIX}/report.mjs`,`${PREFIX}/report.test.mjs`,`${PREFIX}/run.mjs`,`${PREFIX}/prepare.mjs`,'code/shared/artifacts/node/v1/files.mjs'].every(name=>source.files.some(p=>p.path===name)),'reviewed report/imported-guard/parser implementation closure required');
    require(samePin(review.source_review,card.source_review)&&samePin(review.protocol,card.protocol)&&review.approved_phases?.includes('report')&&samePin(admission.protocol,card.protocol)&&samePin(admission.source_review,card.source_review)&&samePin(admission.protocol_review,card.protocol_review)&&admission.admitted_phases?.includes('report'),'exact reporting source/protocol/admission phase required');
    pins.push(card.source_review,card.protocol_review,card.protocol,card.admission,card.study_card,...source.files);
    const study=requireRunCard(read(card.study_card)),seal=read(card.development_seal);requireTemporalContract(study.temporal);requireDevelopmentSeal(seal,card.study_card);pins.push(card.development_seal,...seal.records,...seal.audits,...seal.outputs);
    pins.push(study.source_review,study.protocol_review,study.build_admission,study.engineering_review,study.admission,study.protocol,study.input_manifest,study.resumption);
    const studySource=read(study.source_review);passed(studySource,'actual scientific source review must remain bound');require(Array.isArray(studySource.files),'scientific source closure required');pins.push(...studySource.files);
    const prepared=read(study.confirmation.targets_receipt);passed(prepared,'actual blind confirmation target receipt required');
    require(prepared.kind==='cpb_v16_temporal_confirmation_targets_prepared_v1'&&prepared.blind===true&&prepared.numerical_outcomes_exposed===false,'exact blind confirmation target preparation boundary');pins.push(study.confirmation.targets_receipt,prepared.targets);
    const goal=read(card.historical_goal.current);pins.push(card.historical_goal.current);
    for(const row of card.historical_sources){require(typeof row.original?.repository==='string'&&typeof row.original.path==='string'&&row.current?.bytes===row.original.bytes&&row.current.sha256===row.original.sha256,'each historical locator must retain its full original identity and unchanged bytes');pins.push(row.current);}
    const historical=[goal.parent_protocol,...Object.values(goal.evidence)];
    for(const original of historical){const row=exactlyOne(card.historical_sources,r=>r.original?.repository===original.repository&&samePin(r.original,original),'every frozen protection source needs explicit full original/current binding');require(row.current.bytes===original.bytes&&row.current.sha256===original.sha256,'historical source bytes must be preserved');pins.push(row.current);}
    for(const pin of pins)currentPin(ROOT,pin);
    header={study_card:card.study_card,protocol:card.protocol,source_review:card.source_review,protocol_review:card.protocol_review,admission:card.admission,development_seal:card.development_seal,historical_goal:card.historical_goal,historical_sources:card.historical_sources,
      objective_contract:{control:'V4 value Gaussian NLL',candidate:'V4 value Gaussian NLL + 0.1 * fixed-one-third hidden temporal Huber1',temporal:study.temporal,variance_head_input_detached:true,single_AdamW:true,shared_global_clip:1,reporting_Gaussian_NLL_is_not_candidate_training_objective:true}};
    for(const seed of SEEDS)for(const arm of ARMS){
      const role=exactlyOne(card.subruns,r=>r.seed===seed&&r.arm===arm,'unique reporting subrun'),base=subrun(seed,arm);
      try{
        const phases={};for(const [phase,key] of [['train','training_receipt'],['development','development_receipt'],['confirmation','confirmation_receipt']]){
          require(role[key]?.path===`${base}/${phase}-receipt.json`,'actual phase receipt must retain declared subrun location');
          const value=read(role[key]);requirePhaseReceipt(value,card.study_card,phase,seed,arm);phases[phase]=value;pins.push(role[key],...value.outputs,...(value.dependencies??[]));
        }
        require(seal.records.some(p=>samePin(p,role.training_receipt))&&seal.records.some(p=>samePin(p,role.development_receipt))&&seal.audits.some(p=>samePin(p,role.development_audit)),'development inputs must be the exact all-six sealed receipts/audit');
        require(role.development_audit.path===`${base}/independent-audit.json`&&role.confirmation_audit.path===`${base}/confirmation-audit.json`,'exact root-owned independent arithmetic wrapper roles');
        const developmentAudit=bindAudit(role.development_audit,[...phases.train.outputs,...phases.development.outputs],pins,binaryPins),confirmationAudit=bindAudit(role.confirmation_audit,phases.confirmation.outputs,pins,binaryPins);
        const training=producer(observed(`${base}/training/training-summary.json`),phases.train.outputs,pins);
        requireTrainingSummary(training,seed,arm,study.gradient_policies[arm]);
        const queries=querySummary(base,training,developmentAudit),january_initial=marketStage(base,'initial',phases.development.outputs,developmentAudit,card.study_card,study,pins),january_final=marketStage(base,'final',phases.development.outputs,developmentAudit,card.study_card,study,pins);
        const tempo_initial=tempoStage(base,'initial',phases.development.outputs,developmentAudit,pins),tempo_final=tempoStage(base,'final',phases.development.outputs,developmentAudit,pins),confirmation=confirmationMarket(base,phases.confirmation.outputs,confirmationAudit,study,prepared.targets,pins);
        const confirmationQuery=producer(observed(`${base}/confirmation-features/report.json`),phases.confirmation.outputs,pins),waveform=auditEntry(confirmationAudit,'fixed_query','witness',`${base}/confirmation-features/confirmation-query-witness.pt`);
        require(confirmationQuery.kind==='cpb_v16_confirmation_extraction_v1'&&confirmationQuery.status==='passed'&&confirmationQuery.rows===32&&confirmationQuery.head_fits===0&&confirmationQuery.normalizer_fits===0&&confirmationQuery.encoder_updates===0&&confirmationQuery.descriptive_only_no_gate_or_selection===true&&waveform.examples===32&&waveform.valid_examples===32,'full32 descriptive confirmation query audit with no fit');
        close(confirmationQuery.standardized_MAE,waveform.mae,'confirmation query MAE arithmetic');close(confirmationQuery.Gaussian_NLL_F64,waveform.gaussian_nll,'confirmation query NLL arithmetic');
        const objectives=[1,2048].map(update=>{
          const original=`${base}/training/witnesses/objective-update-${update}.pt`;
          if(arm==='control')return auditEntry(developmentAudit,'objective_control','witness',original,[original]);
          const witness=`${base}/training/witnesses/temporal-objective-step-${update}.pt`;
          return requireTemporalAuditEntry(auditEntry(developmentAudit,'objective_temporal','witness',witness,[witness,original]),update);
        });
        let temporal_objective=null;
        if(arm==='candidate'){
          const training_log=observed(`${base}/training/temporal-training.csv`);require(phases.train.outputs.some(p=>samePin(p,training_log)),'temporal training log must be inside the actual training output receipt');pins.push(training_log);
          temporal_objective={contract:study.temporal,training_log,audited_same_step_witnesses:objectives,scope:'Independent arithmetic/support/direct-output-adjoint checks at updates1/2048; whole log bytes are sealed, without independently reconstructing every remaining update or global clipping.'};
        }
        const retention=historicalRetention(goal,january_final,tempo_final,queries.final);
        const januaryCoverage=[january_initial,january_final].every(market=>TASKS.every(task=>market[task].coverage===1))&&[tempo_initial,tempo_final].every(tempo=>MODES.every(mode=>VIEWS.every(view=>tempo.scores[mode][view].coverage===1)));
        retention.full_coverage=retention.full_coverage&&januaryCoverage;
        retention.all22_passed=retention.full_coverage&&retention.criteria.every(c=>c.passed);
        retention.hard_timing_passed=retention.full_coverage&&retention.criteria.filter(c=>c.category==='tempo5_retention').every(c=>c.passed);
        const full_declared_coverage=retention.full_coverage&&TASKS.every(task=>confirmation[task].coverage===1);
        const extraction=producer(observed(`${base}/features/extraction-summary.json`),phases.development.outputs,pins),synthetic_readers={};
        for(const stage of ['initial','final'])synthetic_readers[stage]=producer(observed(`${base}/heads-${stage}/synthetic/reader-summary.json`),phases.development.outputs,pins);
        runs.push({seed,arm,full_declared_coverage,january_initial,january_final,confirmation,tempo_initial,tempo_final,fixed_queries:queries,confirmation_query:{...confirmationQuery,independent_waveform_examples:32,full_population_waveform_reconstructed:true},historical_retention:retention,hard_timing_passed:retention.hard_timing_passed,
          objective_witnesses:objectives,temporal_objective,training_summary:training,extraction_summary:extraction,synthetic_reader_summaries:synthetic_readers,
          compute_cost:{GPU_step_seconds_synchronized:training.GPU_step_seconds_synchronized,GPU_step_timing_scope:training.GPU_step_timing_scope,loader_mask_seconds:training.loader_mask_seconds,checkpoint_resume_logging_seconds:training.checkpoint_resume_logging_seconds,parameter_gradient_evidence_seconds:training.parameter_gradient_evidence_seconds,training_trajectory_wall_seconds:training.training_trajectory_wall_seconds,calibration_seconds:training.calibration_seconds,development_extraction_seconds_including_IO:extraction.elapsed_seconds_including_all_input_and_state_IO,synthetic_heads_initial_seconds_including_IO:synthetic_readers.initial.head_fitting_and_scoring_seconds_including_io,synthetic_heads_final_seconds_including_IO:synthetic_readers.final.head_fitting_and_scoring_seconds_including_io,market_head_cpu_seconds:null,confirmation_wall_seconds:null,unrecorded_timings_not_estimated:true,overlapping_timings_not_summed:true},
          audited_numeric_results:{development:role.development_audit,confirmation:role.confirmation_audit},
          initial_to_final: Object.fromEntries(TASKS.map(task=>[task,Object.fromEntries(['linear_r2','neural_mean_r2','linear_mae','linear_rmse','neural_mean_mae','neural_mean_rmse'].map(metric=>{const a=january_initial[task][metric],b=january_final[task][metric];return [metric,{initial:a,final:b,difference:finite(a)&&finite(b)?b-a:null}];}))]))});
      }catch(error){failures.push({seed,arm,error:String(error.message)});}
    }
    const verification_final_barrier=verifyFinalBarrier(pins,binaryPins);
    const complete=runs.length===6&&failures.length===0,pairs=complete?pairedChanges(runs):[],screen=complete?pairedScreen(pairs):{verdict:'inconclusive',promising:false,complete:false,reasons:['all six fully audited runs are required']};
    const report={kind:'cpb_v16_temporal_comparison_report_v1',status:complete?'complete':'incomplete',passed:complete,created_utc:new Date().toISOString(),inputs,...header,verification_final_barrier,expected_subruns:6,complete_subruns:runs.length,runs,paired_comparisons:pairs.map(({control,candidate,...row})=>row),prospective_screen:screen,failures,
      historical22_interpretation:'Each seed is checked separately on final January development; original criterion objects, master/repetition means, strict inequalities and positivity quantifiers are unchanged. Counts are not comparable to V4 historical19/22 under its earlier floors.',
      historical_goal_context_is_not_current_authorization:true,baseline:'CPB-v3.market-01',historical_broad_clean_leader:'CPB-v4.market-01',historical_balanced_encoder:'RPB-v24.market-01.alt-01 native36 (later causal eligibility diagnosis applies)',overall_winner:false,Mityane_application_selection:null,automatic_promotion:false,
      limitations:['Three paired encoder seeds on one additional month constitute a screen, not general-purpose superiority.','January was repeatedly used and remains development evidence; March comparisons are within the new paired arms.','February was already consumed by V15 and is not fresh confirmation for V16; March becomes observed evidence after this declared study.','Head repetitions are not independent encoder or market-population replications.','Saved-fit arithmetic is independently reconstructed; encoder forwards and optimizer trajectories are not independently reimplemented.','Temporal witnesses audit only updates1/2048 and direct output adjoints, not the full encoder Jacobian, every remaining training update or full global-clip norm.','Value/temporal direct routes share one global clip1 and AdamW; objective components are not optimization-independent.','Training fixed-query waveform witnesses cover first8 examples only; full112/32 saved row reductions are checked without claiming full waveform reconstruction.','Conditional bootstrap intervals from the frozen producer are retained but are not independently audited.','Past-window readout, reconstruction and TEMPO do not establish forecasting, interventions or downstream decision utility.'],
      fresh_bindings:[...new Map(pins.map(p=>[p.path,p])).values()],fresh_audit_binaries:[...new Map(binaryPins.map(p=>[p.path,p])).values()],python_executed:false,new_training_updates:0,new_head_fits:0};
    writeNew(destination,jsonBytes(report));return report;
  }catch(error){const failure={kind:'cpb_v16_temporal_comparison_report_v1',status:'incomplete',passed:false,created_utc:new Date().toISOString(),inputs,...header,expected_subruns:6,complete_subruns:runs.length,runs,prospective_screen:{verdict:'inconclusive',promising:false,complete:false},failures:[...failures,{error:String(error.message)}],overall_winner:false,Mityane_application_selection:null,scientific_result_claimed:false};writeNew(destination,jsonBytes(failure));return failure;}
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  try{require(process.platform==='linux'&&fs.existsSync('/.dockerenv')&&process.argv.length===4&&process.argv[2]==='--inputs','managed Linux report: --inputs relative-card.json');safeRelative(process.argv[3]);const result=makeReport(process.argv[3]);process.stdout.write(jsonBytes({status:result.status,complete_subruns:result.complete_subruns,verdict:result.prospective_screen.verdict}));if(result.status!=='complete')process.exitCode=1;}
  catch(error){process.stderr.write(`${error.stack}\n`);process.exitCode=1;}
}

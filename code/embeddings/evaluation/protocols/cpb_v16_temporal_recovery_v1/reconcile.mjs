// Copyright (c) 2026 Waajacu. See LICENSE.
// Bounded external-process-interruption recovery. No tensor decoding or models.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {admitFiles, hashAdmitted, readAdmitted, parseJSON, jsonBytes, safeRelative,
  canonicalDirectory, createExclusiveDirectory, requireCondition as require, writeNew,
  sha256} from '../../../../shared/artifacts/node/v1/files.mjs';
import {currentPin} from '../cpb_v16_temporal_v1/prepare.mjs';
import {requireRunCard, requirePhaseReceipt} from '../cpb_v16_temporal_v1/run.mjs';

export const ROOT='/mityane';
export const A='.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001';
export const BASE=`${A}/subruns/seed-946103/candidate`;
export const SNAPSHOT=`${A}/interruption/snapshot001`;
export const RECOVERY=`${A}/recovery001`;
export const PREFIX='code/embeddings/evaluation/protocols/cpb_v16_temporal_recovery_v1';
const BUILD='/opt/mityane-geometrics/embeddings/build/cpb-v16-temporal-001';
const CSV_HEADERS=Object.freeze({
  'training.csv':'attempt,update,training_objective_F64,eligible_examples,target_cells,nonzero_gradient_tensors,changed_tensors,row_seed,mask_seed,anchors',
  'temporal-training.csv':'attempt,update,base_nll,temporal_loss,weighted_temporal,total_loss,lag1_loss,lag4_loss,lag16_loss,lag1_pairs,lag4_pairs,lag16_pairs,lag1_supported,lag4_supported,lag16_supported'
});
const ORIGINAL_NAMES=Object.freeze([
  'fresh-initialization.json','input-bindings.json','query-initial.json','temporal-training.csv',
  'training.csv','typed-resume-512.json','calibration/bank-standardizer.pt',
  'calibration/first-TRAIN-batch-witness.pt','calibration/input-scaler.pt','calibration/summary.json',
  'checkpoints/inference-0.pt','checkpoints/training-512.pt','witnesses/initial-train-query.pt',
  'witnesses/initial-validation-query.pt','witnesses/objective-update-1.pt',
  'witnesses/temporal-objective-step-1.pt'
]);
const NEW_NAMES=Object.freeze([
  'checkpoints/training-1024.pt','checkpoints/training-2048.pt','checkpoints/inference-2048.pt',
  'witnesses/objective-update-2048.pt','witnesses/temporal-objective-step-2048.pt',
  'witnesses/final-train-query.pt','witnesses/final-validation-query.pt','query-final.json',
  'parameter-learning.json','training-summary.json','recovery-bindings.json'
]);
const SAME=(a,b)=>a?.path===b?.path&&a.bytes===b.bytes&&a.sha256===b.sha256;
const PASSED=(v,message)=>require(v?.status==='passed'&&v.passed===true,message);
const local=relative=>path.join(ROOT,safeRelative(relative));
const now=()=>new Date().toISOString();
function pin(relative){const row=admitFiles([local(relative)],{root:ROOT,singleLink:true})[0];return {repository:'Mityane-geometrics',path:relative,bytes:row.bytes,sha256:hashAdmitted(row)};}
function bytes(expected){currentPin(ROOT,expected);return readAdmitted(admitFiles([local(expected.path)],{root:ROOT,singleLink:true})[0]);}
function read(expected){return parseJSON(bytes(expected),{duplicates:'reject'});}
function save(relative,value){writeNew(local(relative),jsonBytes(value));return pin(relative);}
function parents(relative){let parent=path.dirname(local(relative));const missing=[];while(!fs.existsSync(parent)){missing.unshift(parent);parent=path.dirname(parent);}canonicalDirectory(parent,{root:ROOT});for(const name of missing){fs.mkdirSync(name);canonicalDirectory(name,{root:ROOT});}}
function absent(relative){require(!fs.existsSync(local(relative)),`preserve existing leaf: ${relative}`);}
function collect(relative){const result=[];canonicalDirectory(local(relative),{root:ROOT});for(const entry of fs.readdirSync(local(relative),{withFileTypes:true})){require(!entry.isSymbolicLink(),'no output links');const name=`${relative}/${entry.name}`;if(entry.isDirectory())result.push(...collect(name));else{require(entry.isFile(),'regular files only');result.push(pin(name));}}return result.sort((a,b)=>a.path.localeCompare(b.path));}
function exactPaths(pins,expected,label){require(Array.isArray(pins)&&pins.length===expected.length&&new Set(pins.map(v=>v.path)).size===pins.length&&expected.every(v=>pins.some(p=>p.path===v)),`exact ${label} file closure`);}
function unchanged(pins){for(const p of pins)currentPin(ROOT,p);}
function noDevelopment(){
  absent(`${A}/development-seal.json`);
  for(const seed of [946101,946102,946103])for(const arm of ['control','candidate']){
    const base=`${A}/subruns/seed-${seed}/${arm}`;
    for(const name of ['development-receipt.json','confirmation-receipt.json','features','proof-initial','proof-final','heads-initial','heads-final','confirmation-features','confirmation-readouts'])absent(`${base}/${name}`);
  }
}
function fiveTraining(cardPin){
  const result=[];
  for(const seed of [946101,946102,946103])for(const arm of ['control','candidate']){
    if(seed===946103&&arm==='candidate')continue;
    const receipt=pin(`${A}/subruns/seed-${seed}/${arm}/train-receipt.json`),value=read(receipt);
    requirePhaseReceipt(value,cardPin,'train',seed,arm);unchanged(value.outputs);
    result.push({receipt,outputs:value.outputs});
  }
  return result;
}
function originalState(cardPin){
  const card=requireRunCard(read(cardPin));
  require(cardPin.path===`${A}/run-inputs-001.json`,'original single declared study card');
  absent(`${BASE}/train-receipt.json`);noDevelopment();
  const training=collect(`${BASE}/training`);
  exactPaths(training,ORIGINAL_NAMES.map(v=>`${BASE}/training/${v}`),'interrupted original training');
  return {card,originals:[...training,pin(`${BASE}/training.log`)],five:fiveTraining(cardPin)};
}

// This parser never rounds stream identities. Complete original lines and their
// replayed partners must retain exact lexemes AND identical binary64 floats.
export function parseCSV(raw,name,{partial=false,first=1,last}={}){
  require(Object.hasOwn(CSV_HEADERS,name),'one declared CSV role');
  const text=new TextDecoder('utf-8',{fatal:true}).decode(raw);
  require(!text.includes('\r')&&!text.includes('\0'),'unchanged LF UTF-8 CSV encoding');
  const lines=text.split('\n'),fragment=lines.pop();
  require(partial||fragment==='','suffix CSV must end with newline');
  require(lines.shift()===CSV_HEADERS[name],'unchanged exact CSV header');
  const rows=lines.map((line,index)=>{
    require(line.length>0,'no blank CSV rows');
    const fields=name==='training.csv'?line.match(/^([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),"([0-9;]+)"$/)?.slice(1):line.split(',');
    require(fields?.length===(name==='training.csv'?10:15),'exact complete CSV columns');
    const update=first+index;
    require(fields[0]===String(update-1)&&fields[1]===String(update),'no skipped, repeated or displaced attempt/update');
    const floats=name==='training.csv'?[2]:[2,3,4,5,6,7,8];
    for(let i=0;i<fields.length;i++){
      if(floats.includes(i))require(/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$/.test(fields[i])&&Number.isFinite(Number(fields[i])),'finite roundtrip binary64 scalar');
      else if(name==='training.csv'&&i===9)require(fields[i].split(';').length===8&&fields[i].split(';').every(v=>/^[0-9]{13}$/.test(v)),'eight exact anchor strings');
      else require(/^(?:0|[1-9][0-9]*)$/.test(fields[i])&&BigInt(fields[i])<=18446744073709551615n,'exact nonnegative U64 counter/stream/support');
    }
    if(name==='training.csv')require(fields[3]==='8'&&fields[4]==='6912'&&Number(fields[5])<=75&&Number(fields[6])<=75,'fixed B8 target support and model tensor bounds');
    else for(const i of [12,13,14])require(['0','1'].includes(fields[i]),'Boolean lag support');
    return {update,line,fields,floats};
  });
  require(last===undefined||rows.at(-1)?.update===last,'exact final complete CSV update');
  if(fragment)require(fragment.startsWith(`${first+rows.length-1},${first+rows.length},`),'single trailing fragment has next logical counters');
  return {header:CSV_HEADERS[name],rows,fragment,first,last:rows.at(-1)?.update??null};
}
function sameF64(left,right){const a=Buffer.alloc(8),b=Buffer.alloc(8);a.writeDoubleLE(Number(left));b.writeDoubleLE(Number(right));return a.equals(b);}
export function mergeCSV(originalRaw,suffixRaw,name){
  const original=parseCSV(originalRaw,name,{partial:true,first:1}),suffix=parseCSV(suffixRaw,name,{first:513,last:2048});
  require(original.rows.length>=512&&original.rows.length<2048&&suffix.rows.length===1536,'one preserved 512 prefix and 1536 continuation rows');
  const compared=[];
  for(const old of original.rows.filter(row=>row.update>512)){
    const replay=suffix.rows[old.update-513];
    require(replay&&old.line===replay.line,`replayed complete ${name} row ${old.update} differs; no rescue`);
    for(const i of old.floats)require(sameF64(old.fields[i],replay.fields[i]),`replayed ${name} scalar bits differ at update ${old.update}`);
    compared.push(old.update);
  }
  if(original.fragment){const next=suffix.rows[original.rows.length+1-513];require(next&&next.line.startsWith(original.fragment),`replayed ${name} trailing fragment differs; no rescue`);}
  const merged=Buffer.from(`${original.header}\n${original.rows.slice(0,512).map(row=>row.line).join('\n')}\n${suffix.rows.map(row=>row.line).join('\n')}\n`);
  parseCSV(merged,name,{first:1,last:2048});
  return {merged,comparison:{name,complete_original_rows:original.rows.length,original_last_complete_update:original.last,
    preserved_prefix_rows:512,suffix_rows:1536,complete_replayed_overlap_rows:compared.length,
    replayed_overlap_first:compared[0]??null,replayed_overlap_last:compared.at(-1)??null,
    raw_lexemes_equal:true,binary64_float_bits_equal:true,stream_counter_anchor_strings_equal:true,
    tolerance_used:false,original_fragment_bytes:Buffer.byteLength(original.fragment),
    original_fragment_sha256:sha256(Buffer.from(original.fragment)),fragment_exact_replayed_line_prefix:original.fragment?true:null,
    fragment_certifies_completed_update:false,merged_rows:2048}};
}

export function snapshot(cardRelative){
  const cardPin=pin(cardRelative),state=originalState(cardPin);
  const incident=pin(`${A}/interruption-observation-001.json`),incidentValue=read(incident);
  require(incidentValue.kind==='cpb_v16_external_interruption_observation_v1'&&incidentValue.status==='interrupted'&&incidentValue.passed===false&&SAME(incidentValue.original_card,cardPin),'actual retained external interruption observation');
  absent(`${SNAPSHOT}/snapshot.json`);absent(SNAPSHOT);
  const helper=pin(`${PREFIX}/reconcile.mjs`);
  // Admit the full source matrix before copying any member payload.
  const sourceRows=admitFiles(state.originals.map(v=>local(v.path)),{root:ROOT,singleLink:true});
  createExclusiveDirectory(local(SNAPSHOT),{root:ROOT});
  const mappings=[];
  for(let i=0;i<sourceRows.length;i++){
    const original=state.originals[i],relative=original.path.slice(`${BASE}/`.length),retained=`${SNAPSHOT}/files/${relative}`;
    parents(retained);writeNew(local(retained),readAdmitted(sourceRows[i]));
    const retainedPin=pin(retained);require(retainedPin.bytes===original.bytes&&retainedPin.sha256===original.sha256,'retained bytes exact at explicit NEW locator');mappings.push({original,retained:retainedPin});
  }
  const tails=Object.keys(CSV_HEADERS).map(name=>{
    const value=parseCSV(bytes(state.originals.find(v=>v.path===`${BASE}/training/${name}`)),name,{partial:true});
    return {name,complete_rows:value.rows.length,last_complete_update:value.last,trailing_fragment_bytes:Buffer.byteLength(value.fragment),fragment_certifies_completed_update:false};
  });
  require(tails.every(v=>v.complete_rows>=512)&&tails.find(v=>v.name==='temporal-training.csv').complete_rows>=565,'retained interruption tail is not shorter than already observed');
  unchanged([cardPin,helper,incident,...state.originals,...state.five.flatMap(v=>[v.receipt,...v.outputs])]);
  return save(`${SNAPSHOT}/snapshot.json`,{kind:'cpb_v16_external_interruption_snapshot_v1',status:'passed',passed:true,created_utc:now(),
    original_card:cardPin,helper,incident,seed:946103,arm:'candidate',resume_checkpoint_update:512,training_files:16,sibling_logs:1,total_original_files:17,
    mappings,five_passed_training:state.five,tails,typed_checkpoint:state.originals.find(v=>v.path===`${BASE}/training/checkpoints/training-512.pt`),
    tensor_payloads_decoded:false,model_calls:0,canonical_files_changed:false,original_completed_updates_exact:null,physical_repeated_updates_exact:null});
}

function readSnapshot(expected,{afterPublication=false}={}){
  const value=read(expected);PASSED(value,'actual retained interruption snapshot required');
  require(expected.path===`${SNAPSHOT}/snapshot.json`&&value.kind==='cpb_v16_external_interruption_snapshot_v1'&&value.total_original_files===17&&value.training_files===16&&value.sibling_logs===1,'exact original17file interruption snapshot');
  require(value.incident?.path===`${A}/interruption-observation-001.json`,'actual interruption observation bound by snapshot');currentPin(ROOT,value.incident);
  exactPaths(value.mappings.map(v=>v.original),[...ORIGINAL_NAMES.map(v=>`${BASE}/training/${v}`),`${BASE}/training.log`],'snapshot originals');
  for(const pair of value.mappings){
    require(pair.retained.path===`${SNAPSHOT}/files/${pair.original.path.slice(`${BASE}/`.length)}`&&pair.original.bytes===pair.retained.bytes&&pair.original.sha256===pair.retained.sha256,'explicit exact-byte retained locator mapping');
    currentPin(ROOT,pair.retained);
    if(!afterPublication||!Object.keys(CSV_HEADERS).some(name=>pair.original.path===`${BASE}/training/${name}`))currentPin(ROOT,pair.original);
  }
  for(const item of value.five_passed_training){currentPin(ROOT,item.receipt);unchanged(item.outputs);}
  require(value.five_passed_training.length===5,'five completed trajectories preserved');
  return value;
}
function bindingGraph(cardRelative,phase,{afterPublication=false}={}){
  const inputPin=pin(cardRelative),input=read(inputPin);
  require(input.kind==='cpb_v16_external_recovery_bridge_inputs_v1'&&input.root===ROOT,'exact bounded recovery bridge card');
  const fields=['original_card','snapshot','amendment','source_review','build_admission','admission','suffix_execution'];
  require(fields.every(name=>input[name]),'all actual full-pin recovery graph roles required');
  const original=requireRunCard(read(input.original_card)),snapshotValue=readSnapshot(input.snapshot,{afterPublication}),source=read(input.source_review),build=read(input.build_admission),admission=read(input.admission),execution=read(input.suffix_execution);
  bytes(input.amendment);
  require(SAME(snapshotValue.original_card,input.original_card),'snapshot exact original study context');
  for(const value of [source,build,admission,execution]){PASSED(value,'actual reviewed recovery source/build/admission/execution must have passed');require(SAME(value.original_card,input.original_card)&&SAME(value.amendment,input.amendment),'recovery record must bind original study and explicit amendment');}
  require(source.kind==='cpb_v16_external_recovery_source_review_v1'&&SAME(source.original_source_review,original.source_review)&&source.original_runtime_sources_unchanged===true&&source.production_model_unchanged===true&&Array.isArray(source.files)&&source.files.some(p=>p.path===`${PREFIX}/reconcile.mjs`),'explicit reviewed operational source extension, unchanged original model closure');
  require(build.kind==='cpb_v16_external_recovery_build_admission_v1'&&SAME(build.source_review,input.source_review)&&SAME(build.original_build_admission,original.build_admission)&&build.original_17_binary_tuples_unchanged===true&&build.actual_build_completed===true,'actual recovery compile admission with original binary preservation');
  require(admission.kind==='cpb_v16_external_recovery_admission_v1'&&SAME(admission.source_review,input.source_review)&&SAME(admission.build_admission,input.build_admission)&&admission.admitted_phases?.includes(phase),'explicit recovery phase admission');
  const oldSource=read(original.source_review),oldBuild=read(original.build_admission);
  unchanged([...oldSource.files,...source.files]);
  for(const binary of oldBuild.binaries)currentPin(ROOT,binary,{binary:true});
  require(oldBuild.binaries.length===17,'all17 original runtime/test binary roles preserved');
  const recoveryBinary=build.binaries?.find(p=>p.role==='candidate_recovery_train');
  require(recoveryBinary?.path===`${BUILD}/cpb_v16_temporal_recovery_candidate_train`&&build.binaries.filter(p=>p.role==='candidate_recovery_train').length===1,'actual separately compiled recovery candidate binary');
  currentPin(ROOT,recoveryBinary,{binary:true});
  require(execution.kind==='cpb_v16_external_recovery_suffix_execution_v1'&&SAME(execution.source_review,input.source_review)&&SAME(execution.build_admission,input.build_admission)&&SAME(execution.snapshot,input.snapshot)&&execution.seed===946103&&execution.arm==='candidate'&&execution.output_directory===`${RECOVERY}/training`&&execution.physical_repeated_updates_exact===null&&execution.old_encoder_calls===0&&execution.head_fits===0,'actual bounded suffix execution, no fictitious original call completion');
  require(Array.isArray(execution.calls)&&execution.calls.length===1&&SAME(execution.calls[0].binary,recoveryBinary)&&execution.calls[0].exit_code===0&&execution.calls[0].process_error===null&&execution.calls[0].log?.path===`${RECOVERY}/training.log`,'one actual passed recovery binary call and new log');
  const args=execution.calls[0].arguments;
  const requiredArgs=['--settings',local(original.settings.candidate.path),'--shared',local(original.shared),...original.january_caches.flatMap(p=>['--cache',local(p.path)]),'--seed','946103','--resume-state',local(`${BASE}/training/checkpoints/training-512.pt`),'--source-output',local(`${BASE}/training`),'--output',local(`${RECOVERY}/training`)];
  require(JSON.stringify(args)===JSON.stringify(requiredArgs),'exact original input/seed plus explicit512 resume and distinct suffix output CLI');
  exactPaths(execution.outputs,[...NEW_NAMES,...Object.keys(CSV_HEADERS)].map(name=>`${RECOVERY}/training/${name}`).concat(`${RECOVERY}/training.log`),'actual recovery suffix output');
  unchanged(execution.outputs);
  const actualSuffix=collect(`${RECOVERY}/training`);exactPaths(actualSuffix,[...NEW_NAMES,...Object.keys(CSV_HEADERS)].map(name=>`${RECOVERY}/training/${name}`),'suffix directory');
  require(actualSuffix.every(p=>execution.outputs.some(v=>SAME(p,v))),'executed suffix exact saved output bytes');
  for(const field of fields)currentPin(ROOT,input[field]);currentPin(ROOT,inputPin);
  return {inputPin,input,original,snapshot:snapshotValue,source,build,admission,execution,recoveryBinary};
}
function requireSummary(graph){
  const summary=read(graph.execution.outputs.find(p=>p.path===`${RECOVERY}/training/training-summary.json`));
  const op=summary.operational_recovery;
  require(summary.kind==='cpb_v16_temporal_paired_training_summary_v1'&&summary.status==='passed'&&summary.initialization_seed===946103&&summary.new_causal_updates===2048&&summary.batch_size===8&&summary.row_seed===947103&&summary.mask_seed===948103&&summary.context_seed===949103&&summary.gradient_policy===graph.original.gradient_policies.candidate&&summary.typed_resume_512_exact===true&&summary.training_replays===null,'honest recovered logical full-trajectory summary with exact original policy');
  require(op?.external_process_interruption===true&&op.resume_checkpoint_update===512&&op.suffix_first_update===513&&op.suffix_last_update===2048&&op.suffix_successful_updates===1536&&op.original_completed_updates_exact===null&&op.physical_repeated_updates_exact===null&&op.observed_original_complete_temporal_last_update===565&&op.whole_trajectory_cost_complete===false&&op.whole_trajectory_per_parameter_counters_complete===false&&op.scientific_recipe_changed===false,'summary explicitly scopes suffix evidence and missing physical cost/counters');
  for(const name of ['GPU_step_seconds_synchronized','loader_mask_seconds','checkpoint_resume_logging_seconds','parameter_gradient_evidence_seconds','training_trajectory_wall_seconds','raw_load_calls','direct_F64_record_cells_checked','genuine_F64_differs_from_F32_cells'])require(summary[name]===null,`unavailable whole-trajectory evidence must be null: ${name}`);
  require(summary.recovery_new_scaler_fits===0&&summary.recovery_new_normalizer_fits===0&&summary.recovery_own_trajectory_training_states_opened===1&&summary.old_encoder_calls===0&&summary.head_fits===0,'own typed state restored without refits, older encoders or heads');
  require(summary.recovery_segment?.scope==='actual-logical-updates513..2048-plus-final-queries;excludes-original-prefix-and-discarded-interrupted-work','actual suffix timing and loader evidence has bounded scope');
  const bindings=read(graph.execution.outputs.find(p=>p.path===`${RECOVERY}/training/recovery-bindings.json`));
  require(bindings.kind==='cpb_v16_operational_recovery_bindings_v1'&&bindings.status==='passed'&&bindings.mode==='suffix-training'&&bindings.source_output===local(`${BASE}/training`)&&bindings.resume_state===local(`${BASE}/training/checkpoints/training-512.pt`)&&bindings.resume_checkpoint_update===512&&bindings.resumed_attempt_counter===512&&bindings.row_seed===947103&&bindings.mask_seed===948103&&bindings.context_seed===949103&&typeof bindings.model_state_identity_at512==='string'&&typeof bindings.optimizer_state_identity_at512==='string','actual recovery loader metadata binds source512 and exact absolute seed/counter streams');
  const shared=['selections.pt','query-plan-train.pt','query-plan-validation.pt'].map(name=>graph.original.input_files.find(p=>p.path===`${graph.original.shared}/${name}`));
  require(shared.every(Boolean),'original shared selection/query source bindings');
  const expected=[graph.original.settings.candidate,...shared,...graph.original.january_caches,...graph.snapshot.mappings.filter(p=>p.original.path.startsWith(`${BASE}/training/`)).map(p=>p.original)].map(p=>({...p,path:local(p.path)}));
  exactPaths(bindings.inputs,expected.map(p=>p.path),'actual23input recovery loader');
  require(expected.every(p=>bindings.inputs.some(v=>SAME(p,v))),'actual recovery source bytes equal original card and retained interrupted snapshot');
  return summary;
}
export function prepare(cardRelative){
  const graph=bindingGraph(cardRelative,'reconcile');noDevelopment();requireSummary(graph);
  absent(`${BASE}/train-receipt.json`);absent(`${RECOVERY}/merge.json`);absent(`${RECOVERY}/merged`);
  const result=Object.keys(CSV_HEADERS).map(name=>{
    const original=graph.snapshot.mappings.find(v=>v.original.path===`${BASE}/training/${name}`);
    const suffix=graph.execution.outputs.find(p=>p.path===`${RECOVERY}/training/${name}`);
    return {name,original,suffix,...mergeCSV(bytes(original.retained),bytes(suffix),name)};
  });
  // Neither complete rows nor fragments can be discarded to rescue divergence.
  require(result.find(v=>v.name==='temporal-training.csv').comparison.complete_replayed_overlap_rows>=53,'all already observed temporal tail checks present');
  createExclusiveDirectory(local(`${RECOVERY}/merged`),{root:ROOT});
  const replacements=[];
  for(const value of result){const relative=`${RECOVERY}/merged/${value.name}`;writeNew(local(relative),value.merged);replacements.push({canonical:value.original.original,retained:value.original.retained,replacement:pin(relative),comparison:value.comparison});}
  const additions=NEW_NAMES.map(name=>({source:graph.execution.outputs.find(p=>p.path===`${RECOVERY}/training/${name}`),destination:`${BASE}/training/${name}`}));
  for(const item of additions)absent(item.destination);
  readSnapshot(graph.input.snapshot);unchanged([graph.inputPin,...graph.execution.outputs]);
  return save(`${RECOVERY}/merge.json`,{kind:'cpb_v16_external_interruption_merge_v1',status:'passed',passed:true,created_utc:now(),
    bridge_inputs:graph.inputPin,original_card:graph.input.original_card,snapshot:graph.input.snapshot,amendment:graph.input.amendment,
    source_review:graph.input.source_review,build_admission:graph.input.build_admission,suffix_execution:graph.input.suffix_execution,
    replacements,additions,merged_logical_updates:2048,original_prefix_updates:512,suffix_updates:1536,
    canonical_files_changed:false,original_call_completed:false,physical_repeated_updates_exact:null,
    all_available_complete_overlap_and_fragments_checked:true,no_tolerance_or_retry_rescue:true,model_calls:0,tensor_payloads_decoded:false});
}

// Publication is intentionally a bounded multi-file operation, not an atomic
// transaction. The exclusive plan survives any failure for explicit inspection;
// a failed partial publication never creates a passed phase receipt.
export function publish(cardRelative){
  const graph=bindingGraph(cardRelative,'publish');noDevelopment();requireSummary(graph);
  const mergePin=pin(`${RECOVERY}/merge.json`),merge=read(mergePin);PASSED(merge,'actual exact overlap/merge verification required');
  require(merge.kind==='cpb_v16_external_interruption_merge_v1'&&SAME(merge.bridge_inputs,graph.inputPin)&&SAME(merge.suffix_execution,graph.input.suffix_execution)&&SAME(merge.snapshot,graph.input.snapshot)&&merge.all_available_complete_overlap_and_fragments_checked===true&&merge.no_tolerance_or_retry_rescue===true,'same actual merge graph with no rescued mismatch');
  exactPaths(merge.replacements.map(p=>p.canonical),Object.keys(CSV_HEADERS).map(name=>`${BASE}/training/${name}`),'two canonical CSV replacements');
  exactPaths(merge.additions.map(p=>({path:p.destination})),NEW_NAMES.map(name=>`${BASE}/training/${name}`),'missing-only additions');
  for(const row of merge.replacements){
    const name=path.posix.basename(row.canonical.path),original=graph.snapshot.mappings.find(p=>SAME(p.original,row.canonical));
    require(original&&SAME(original.retained,row.retained)&&row.replacement.path===`${RECOVERY}/merged/${name}`,'replacement exact retained original and explicit merged locator');
    const suffix=graph.execution.outputs.find(p=>p.path===`${RECOVERY}/training/${name}`),checked=mergeCSV(bytes(row.retained),bytes(suffix),name);
    require(bytes(row.replacement).equals(checked.merged),'fresh exact recomputed merge equals reviewed replacement');
  }
  for(const row of merge.additions){require(row.source.path===`${RECOVERY}/training/${row.destination.slice(`${BASE}/training/`.length)}`&&graph.execution.outputs.some(p=>SAME(p,row.source)),'addition exact executed suffix origin');absent(row.destination);}
  absent(`${BASE}/train-receipt.json`);absent(`${RECOVERY}/publication-plan.json`);absent(`${RECOVERY}/publication.json`);
  readSnapshot(graph.input.snapshot);unchanged([graph.inputPin,mergePin,...merge.replacements.map(p=>p.replacement),...merge.additions.map(p=>p.source)]);
  const planPin=save(`${RECOVERY}/publication-plan.json`,{kind:'cpb_v16_external_interruption_publication_plan_v1',status:'planned',passed:false,created_utc:now(),bridge_inputs:graph.inputPin,merge:mergePin,snapshot:graph.input.snapshot,replacements:merge.replacements,additions:merge.additions,canonical_original_bytes_retained:true,multi_file_atomic:false});
  // Stage every future payload exclusively before the first canonical change.
  const staging=`${RECOVERY}/publish-staging`;absent(staging);createExclusiveDirectory(local(staging),{root:ROOT});
  const staged=merge.replacements.map((row,i)=>{const relative=`${staging}/replacement-${i}.csv`;writeNew(local(relative),bytes(row.replacement));return {relative,row};});
  const additions=merge.additions.map((row,i)=>{const relative=`${staging}/addition-${i}`;writeNew(local(relative),bytes(row.source));return {relative,row};});
  // Final fresh tuple barrier checks ALL originals, not only the two CSV leaves.
  readSnapshot(graph.input.snapshot);unchanged([graph.inputPin,mergePin,planPin,...graph.execution.outputs]);
  for(const {relative,row} of staged){currentPin(ROOT,row.canonical);const stagedPin=pin(relative);require(stagedPin.bytes===row.replacement.bytes&&stagedPin.sha256===row.replacement.sha256,'fresh staged CSV exact reviewed payload');canonicalDirectory(path.dirname(local(row.canonical.path)),{root:ROOT});fs.renameSync(local(relative),local(row.canonical.path));require(pin(row.canonical.path).sha256===row.replacement.sha256,'canonical CSV installs exact merged bytes');}
  for(const {relative,row} of additions){absent(row.destination);parents(row.destination);writeNew(local(row.destination),bytes(pin(relative)));}
  const canonical=collect(`${BASE}/training`);
  exactPaths(canonical,[...ORIGINAL_NAMES,...NEW_NAMES].map(name=>`${BASE}/training/${name}`),'published complete logical training closure');
  readSnapshot(graph.input.snapshot,{afterPublication:true});
  for(const row of merge.replacements){const actual=canonical.find(p=>p.path===row.canonical.path);require(actual.bytes===row.replacement.bytes&&actual.sha256===row.replacement.sha256,'published CSV new path-bound identity');}
  for(const row of merge.additions){const actual=canonical.find(p=>p.path===row.destination);require(actual.bytes===row.source.bytes&&actual.sha256===row.source.sha256,'published missing artifact exact new locator mapping');}
  unchanged([graph.inputPin,mergePin,planPin,...graph.execution.outputs]);currentPin(ROOT,graph.recoveryBinary,{binary:true});
  const publication=save(`${RECOVERY}/publication.json`,{kind:'cpb_v16_external_interruption_publication_v1',status:'passed',passed:true,created_utc:now(),
    bridge_inputs:graph.inputPin,original_card:graph.input.original_card,snapshot:graph.input.snapshot,amendment:graph.input.amendment,
    source_review:graph.input.source_review,build_admission:graph.input.build_admission,admission:graph.input.admission,suffix_execution:graph.input.suffix_execution,
    merge:mergePin,plan:planPin,outputs:canonical,replaced_originals:merge.replacements.map(p=>({original:p.canonical,retained:p.retained,current:canonical.find(v=>v.path===p.canonical.path)})),
    added_paths:merge.additions.map(p=>p.destination),other_original_bytes_unchanged:true,five_completed_trajectories_unchanged:true,sibling_original_training_log_unchanged:true,
    canonical_csv_identity_changed:true,logical_updates:2048,original_call_completed:false,actual_new_binary_call:true,physical_repeated_updates_exact:null,multi_file_atomic:false});
  const operational_recovery={kind:'cpb_v16_external_interruption_train_extension_v1',amendment:graph.input.amendment,source_review:graph.input.source_review,build_admission:graph.input.build_admission,admission:graph.input.admission,snapshot:graph.input.snapshot,suffix_execution:graph.input.suffix_execution,merge:mergePin,publication,
    original_study_context:graph.input.original_card,actual_new_binary_call:true,original_call_completed:false,resume_checkpoint_update:512,suffix_successful_updates:1536,
    original_completed_updates_exact:null,physical_repeated_updates_exact:null,observed_post_checkpoint_complete_temporal_updates:merge.replacements.find(p=>p.comparison.name==='temporal-training.csv').comparison.complete_replayed_overlap_rows,no_replay_claim:false,
    whole_trajectory_cost_complete:false,whole_trajectory_per_parameter_counters_complete:false,source_extension_independently_admitted:true};
  const dependencies=[graph.inputPin,...['snapshot','amendment','source_review','build_admission','admission','suffix_execution'].map(name=>graph.input[name]),mergePin,planPin,publication];
  const outputs=[...canonical,pin(`${BASE}/training.log`),...graph.execution.outputs];
  unchanged([...dependencies,...outputs,graph.input.original_card]);
  return save(`${BASE}/train-receipt.json`,{kind:'cpb_v16_temporal_subrun_phase_v1',status:'passed',passed:true,created_utc:now(),card:graph.input.original_card,phase:'train',seed:946103,arm:'candidate',calls:graph.execution.calls,dependencies,outputs,operational_recovery,python_executed:false});
}

// The recovery-aware orchestration/report wrapper MUST invoke this before each
// downstream phase. The frozen runner's permissive receipt check is insufficient.
export function verifyPublished(cardRelative,phase){
  require(['development','seal','confirmation','report'].includes(phase),'explicit admitted downstream phase');
  const graph=bindingGraph(cardRelative,phase,{afterPublication:true});requireSummary(graph);
  const publicationPin=pin(`${RECOVERY}/publication.json`),publication=read(publicationPin),mergePin=pin(`${RECOVERY}/merge.json`),merge=read(mergePin),receiptPin=pin(`${BASE}/train-receipt.json`),receipt=read(receiptPin);
  PASSED(publication,'actual completed recovery publication required');PASSED(merge,'actual completed exact merge required');requirePhaseReceipt(receipt,graph.input.original_card,'train',946103,'candidate');
  require(publication.kind==='cpb_v16_external_interruption_publication_v1'&&SAME(publication.bridge_inputs,graph.inputPin)&&SAME(publication.snapshot,graph.input.snapshot)&&SAME(publication.source_review,graph.input.source_review)&&SAME(publication.build_admission,graph.input.build_admission)&&SAME(publication.suffix_execution,graph.input.suffix_execution)&&SAME(publication.merge,mergePin)&&publication.actual_new_binary_call===true&&publication.original_call_completed===false&&publication.canonical_csv_identity_changed===true&&publication.other_original_bytes_unchanged===true,'mandatory actual new-source/binary publication graph');
  require(SAME(merge.bridge_inputs,graph.inputPin)&&SAME(merge.snapshot,graph.input.snapshot)&&SAME(merge.suffix_execution,graph.input.suffix_execution)&&merge.all_available_complete_overlap_and_fragments_checked===true&&merge.no_tolerance_or_retry_rescue===true,'mandatory saved exact-overlap merge graph');
  const op=receipt.operational_recovery;
  require(op?.kind==='cpb_v16_external_interruption_train_extension_v1'&&SAME(op.original_study_context,graph.input.original_card)&&SAME(op.publication,publicationPin)&&SAME(op.merge,mergePin)&&SAME(op.snapshot,graph.input.snapshot)&&SAME(op.amendment,graph.input.amendment)&&SAME(op.source_review,graph.input.source_review)&&SAME(op.build_admission,graph.input.build_admission)&&SAME(op.admission,graph.input.admission)&&SAME(op.suffix_execution,graph.input.suffix_execution)&&op.actual_new_binary_call===true&&op.original_call_completed===false&&op.physical_repeated_updates_exact===null&&op.no_replay_claim===false&&op.whole_trajectory_cost_complete===false&&op.whole_trajectory_per_parameter_counters_complete===false,'normal train receipt must disclose and bind recovery extension');
  require(JSON.stringify(receipt.calls)===JSON.stringify(graph.execution.calls),'phase call is actual recovery call, never original compiled-binary substitution');
  const canonical=collect(`${BASE}/training`);
  exactPaths(canonical,[...ORIGINAL_NAMES,...NEW_NAMES].map(name=>`${BASE}/training/${name}`),'current recovered training closure');
  require(canonical.every(p=>publication.outputs.some(v=>SAME(p,v)))&&publication.outputs.length===canonical.length,'publication exact complete current canonical closure');
  const expectedOutputs=[...canonical,pin(`${BASE}/training.log`),...graph.execution.outputs];
  exactPaths(receipt.outputs,expectedOutputs.map(p=>p.path),'recovered phase complete canonical and suffix outputs');
  require(expectedOutputs.every(p=>receipt.outputs.some(v=>SAME(v,p))),'recovered phase complete exact output tuples');
  for(const name of Object.keys(CSV_HEADERS)){
    const old=graph.snapshot.mappings.find(p=>p.original.path===`${BASE}/training/${name}`),suffix=graph.execution.outputs.find(p=>p.path===`${RECOVERY}/training/${name}`),actual=canonical.find(p=>p.path===`${BASE}/training/${name}`);
    require(bytes(actual).equals(mergeCSV(bytes(old.retained),bytes(suffix),name).merged),'fresh downstream exact reconstructed CSV verification');
  }
  unchanged([publicationPin,mergePin,receiptPin,...receipt.dependencies,...receipt.outputs]);
  return {kind:'cpb_v16_external_recovery_downstream_compatibility_v1',status:'passed',passed:true,phase,bridge_inputs:graph.inputPin,publication:publicationPin,training_receipt:receiptPin,actual_new_binary_call:true,original_call_completed:false,source_extension_independently_admitted:true};
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  try{
    require(process.platform==='linux'&&fs.existsSync('/.dockerenv'),'managed Linux container only');
    const flags={};for(let i=2;i<process.argv.length;i+=2){require(['--mode','--card','--inputs'].includes(process.argv[i])&&process.argv[i+1]&&!Object.hasOwn(flags,process.argv[i]),'exact bridge flags');flags[process.argv[i]]=process.argv[i+1];}
    require(Object.keys(flags).length===2,'one mode and one exact input required');
    let receipt;if(flags['--mode']==='snapshot'){require(flags['--card']&&!flags['--inputs'],'snapshot takes original study card');receipt=snapshot(flags['--card']);}
    else{require(flags['--inputs']&&!flags['--card']&&['prepare','publish'].includes(flags['--mode']),'prepare or publish takes complete recovery bridge card');receipt=flags['--mode']==='prepare'?prepare(flags['--inputs']):publish(flags['--inputs']);}
    process.stdout.write(jsonBytes({status:'passed',receipt}));
  }catch(error){process.stderr.write(`${error.stack}\n`);process.exitCode=1;}
}

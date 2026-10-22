// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Metadata only: imports the study's readers/checkers without invoking its command dispatcher.
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {ROOT,BASE,ATTEMPT,OLD,FAMILY,PROTOCOL,SCHEDULE,SEEDS,TASKS,absolute,relative,read,pin,check,save,hash,nativeGuard,admission,completed,verifySeal,validateTrain,fullEvents,differences} from './run-study.mjs';

const RECORD='code/embeddings/evaluation/records/CPB-v18.span-001.json';
const RESULTS=`${PROTOCOL}/RESULTS.md`;
const MANIFEST=`${ATTEMPT}/closeout-input-manifest.json`;
const same=(left,right,message)=>{if(JSON.stringify(left)!==JSON.stringify(right))throw Error(message);};
const require=(condition,message)=>{if(!condition)throw Error(message);};
const canonical=p=>({...p,path:relative(absolute(p.path))});
function unique(pins){const map=new Map();for(const raw of pins){const p=canonical(raw),old=map.get(p.path);require(!old||(old.sha256===p.sha256&&old.bytes===p.bytes),`conflicting source identity: ${p.path}`);map.set(p.path,p);}return [...map.values()].sort((a,b)=>a.path.localeCompare(b.path));}
function collect(){
  nativeGuard();for(const name of [RECORD,RESULTS,MANIFEST])require(!fs.existsSync(absolute(name)),`preserve previous closeout output: ${name}`);
  const a=admission(),aPin=pin(`${ATTEMPT}/admission.json`),seal=verifySeal(),sealPin=pin(`${ATTEMPT}/training-seal.json`),screen=read(`${ATTEMPT}/screen-summary.json`);
  require(screen.status==='complete'&&screen.primary_updates===1024&&screen.logical_updates===6144&&screen.new_physical_updates===5376&&screen.reused_control_prefix_updates===768,'complete1024 comparison summary required');check(screen.training_seal);
  const contract=a.sources.find(p=>p.path===`${PROTOCOL}/CONTRACT.md`),protocol=a.sources.find(p=>p.path===`${PROTOCOL}/PROTOCOL.md`);
  require(contract&&protocol&&protocol.sha256===a.protocol_copy.sha256,'operation contract and original protocol must be admitted before execution');check(contract);check(protocol);
  require(Date.parse(a.admitted_utc)<=Date.parse(seal.sealed_utc),'training seal must follow prospective admission');
  const phases={};for(const phase of ['train','readout']){const name=`${ATTEMPT}/${phase}.admission.json`,r=read(name);require(r.phase===phase,'phase identity mismatch');check(r.study_admission);[...r.sources,...r.executables].forEach(check);phases[phase]={artifact:pin(name),data:r};}
  const receiptNames=fs.readdirSync(absolute(`${ATTEMPT}/receipts`)).filter(n=>n.endsWith('.json')&&!n.endsWith('.command.json')).sort();
  require(receiptNames.length===24,'12 new training +6 readout +6 audit receipts required; preparation is reused');
  const commandReceipts=receiptNames.map(leaf=>{const p=`${ATTEMPT}/receipts/${leaf}`,r=read(p);require(completed(r.label),'complete successful native command required');return {label:r.label,receipt:pin(p)};});
  const engineeringExecution=read(`${BASE}/engineering-execution.json`);require(engineeringExecution.status==='passed'&&engineeringExecution.exit_code===0,'passed source-bound V18 engineering execution required');
  [engineeringExecution.source_admission,engineeringExecution.stdout,engineeringExecution.stderr].forEach(check);
  const engineeringAdmission=read(engineeringExecution.source_admission.path);engineeringAdmission.sources.forEach(check);check(engineeringAdmission.executable);
  const engineeringPath=`${BASE}/engineering-001/summary.json`,engineering=read(engineeringPath);require(engineering.status==='passed'&&engineering.quality_evaluated===false,'engineering fixture must remain distinct from research quality');
  const capture=read(a.control_source.capture.path);check(a.control_source.capture);check(a.control_source.original_manifest);
  require(capture.files===108&&a.control_source.preserved.length===108,'full original control-source preservation required');
  a.control_source.preserved.forEach(member=>check(member.preserved));
  const referencePath=`${FAMILY}/research-reference.json`,referenceDoc=read(referencePath),reference=referenceDoc.reference;
  require(reference.version==='CPB-v16'&&reference.encoder_seed===946103&&reference.logical_checkpoint_update===2048&&reference.checkpoint.sha256==='b6f3cbe711235307f362bf2f999571d65d00c85825f8a4a5c2fbc4fa4242e51e','selected research reference changed during the comparison');
  [reference.checkpoint,reference.TRAIN_input_scaler,reference.TRAIN_bank_standardizer].forEach(check);
  const prefixRecords=[];let reusedUpdates=0,reusedAttempts=0,reusedFits=0;
  for(const p of a.control_prefixes){require(p.completed_updates===256&&p.attempted_batches===256,'admitted completed control prefix required');
    const reports=p.reports.map(s=>{check(s.report);const r=read(s.report.path);Object.values(r.artifacts).forEach(check);check(r.executable);check(r.settings);check(r.dataset);
      reusedUpdates+=r.physical_training_updates_this_invocation;reusedAttempts+=r.attempted_batches-r.starting_attempted_batches;if(r.preprocessing_fitted)++reusedFits;
      return {stage:s.stage,report:s.report,physical_optimizer_updates:r.physical_training_updates_this_invocation,attempts:r.attempted_batches-r.starting_attempted_batches,preprocessing_fit_windows:r.preprocessing_fit_windows};});
    prefixRecords.push({...p,reports});
  }
  require(reusedUpdates===768&&reusedAttempts===768&&reusedFits===3,'old completed prefixes must be counted honestly and not replayed');
  const trajectories=[];let newUpdates=0,newAttempts=0,newFits=0,headFits=0,auditCount=0;
  for(const [index,t] of seal.trajectories.entries()){
    same([t.id,t.seed,t.arm,t.mask_mode],[SCHEDULE[index].id,SCHEDULE[index].seed,SCHEDULE[index].arm,SCHEDULE[index].mask_mode],'sealed schedule differs from prospective order');
    const training=t.training_reports.map(stage=>{check(stage.report);const r=validateTrain(t,stage.stage);Object.values(r.artifacts).forEach(check);
      newUpdates+=r.physical_training_updates_this_invocation;newAttempts+=r.attempted_batches-r.starting_attempted_batches;if(r.preprocessing_fitted)++newFits;
      return {stage:stage.stage,report:stage.report,completed_updates:r.completed_updates,attempted_batches:r.attempted_batches,starting_updates:r.starting_completed_updates,starting_attempts:r.starting_attempted_batches,
        new_physical_updates:r.physical_training_updates_this_invocation,skipped_attempts:r.skipped_attempts_this_invocation,preprocessing_fitted:r.preprocessing_fitted,preprocessing_fit_windows:r.preprocessing_fit_windows,
        recipe_identity:r.recipe_identity,initial_state_identity:r.initial_state_identity,final_state_identity:r.final_state_identity,optimizer_identity:r.optimizer_identity,scaler_identity:r.scaler_identity,bank_identity:r.bank_identity,
        training_state:check(r.artifacts.training_state),inference:check(r.artifacts.inference),resume_input:r.artifacts.resume_input?check(r.artifacts.resume_input):null};});
    const allEvents=fullEvents(t);let updates=0;allEvents.forEach((event,attempt)=>{require(event.attempt===attempt,'complete absolute attempt stream required');if(event.updated)++updates;require(event.completed_updates===updates,'saved complete progress/event stream mismatch');});
    require(updates===1024&&allEvents.length===t.attempted_batches,'complete1024 logical trajectory required');
    const rPath=`${ATTEMPT}/readouts/${t.id}/summary.json`,r=read(rPath),auditPath=`${ATTEMPT}/audits/${t.id}/summary.json`,audit=read(auditPath),command=read(`${ATTEMPT}/receipts/${t.id}-readout.command.json`);
    require(Date.parse(command.started_utc)>=Date.parse(seal.sealed_utc),'all six final states must be sealed before any new head execution');
    require(r.updates===1024&&r.temporal_weight===.1&&r.mask_mode===t.mask_mode&&r.head_fits===4&&r.encoder_updates===0&&r.preprocessing_fits===0&&r.target_precision_parity&&r.encoder_state_unchanged,'unchanged fixed readout scope failed');
    require(r.witness_codec==='cpb_v17_ridge_witness_v1'&&r.witness_codec_meaning==='generic-ridge-tensor-codec-not-encoder-identity','generic structural ridge witness must not impersonate encoder identity');
    Object.values(r.inputs).forEach(check);check(r.executable);check(r.witness);
    require(audit.status==='passed'&&audit.targets===4&&audit.absolute_tolerance===1e-9&&audit.relative_tolerance===1e-8&&audit.original_legacy_target_bits_equal&&audit.original_row_group_association_equal,'unchanged independent scalar audit required');
    Object.values(audit.inputs).forEach(check);check(audit.executable);
    require(audit.executable.sha256===phases.readout.data.executables.find(p=>p.path.endsWith('/cpb_v17_audit')).sha256,'actual audit executable must be the admitted preserved audit');
    for(const task of TASKS){const value=r.metrics.find(m=>m.task===task),independent=audit.independent_metrics.find(m=>m.task===task);
      require(value&&independent&&value.coverage===1&&value.valid===32&&independent.coverage===1,'finite complete32/32 scoring required');
      for(const key of ['r2','mae','rmse','coverage'])require(Number.isFinite(value[key])&&Math.abs(value[key]-independent[key])<=1e-9+1e-8*Math.max(Math.abs(value[key]),Math.abs(independent[key])),`independent metric mismatch: ${t.id}/${task}/${key}`);}
    headFits+=r.head_fits;++auditCount;
    trajectories.push({...t,training,metrics:r.metrics,readout:pin(rPath),witness:check(r.witness),audit:pin(auditPath),independent_metrics:audit.independent_metrics,
      detailed_control_mask_witness:t.arm==='control'?{available:false,reason:'The preserved original control CLI did not log temporal pair counts, selected starts or span quotas.'}:null});
  }
  require(newUpdates===5376&&newFits===3&&headFits===24&&auditCount===6,'actual new physical work/fit/readout accounting mismatch');
  const d=differences(trajectories);same(d.paired,screen.paired_candidate_minus_control_r2,'paired summary differs from actual readouts');same(d.aggregate,screen.aggregate,'aggregate summary differs');same(d.conditions,screen.predeclared_conditions,'screen conditions differ');require(d.passed===screen.predeclared_continuation_criterion_passed,'declared verdict differs');
  require(screen.observed_gain===d.observed_gain&&screen.strict_trend_or_activity_gain_observed===d.strict_trend_or_activity_gain_observed,'observed-gain flags differ');
  const currentSources=unique([...a.sources,...Object.values(phases).flatMap(p=>p.data.sources),...engineeringAdmission.sources]),executables=unique([...Object.values(phases).flatMap(p=>p.data.executables),engineeringAdmission.executable]);
  const sourceDigest=hash(Buffer.from(JSON.stringify(currentSources))),executableDigest=hash(Buffer.from(JSON.stringify(executables)));
  const generatedUtc=new Date().toISOString(),manifest={kind:'cpb_v18_span_verified_closeout_manifest_v1',created_utc:generatedUtc,generator:pin(fileURLToPath(import.meta.url)),study_admission:aPin,
    phase_admissions:Object.fromEntries(Object.entries(phases).map(([key,p])=>[key,p.artifact])),current_sources:currentSources,current_source_closure_sha256:sourceDigest,executables,executable_set_sha256:executableDigest,
    original_control_source_capsule:a.control_source,evaluation_source_equivalence:a.evaluation_equivalence,old_control_prefixes:prefixRecords,native_command_receipts:commandReceipts};
  const manifestText=JSON.stringify(manifest,null,2)+'\n',manifestBytes=Buffer.from(manifestText),manifestPin={path:MANIFEST,bytes:manifestBytes.length,sha256:hash(manifestBytes)};
  const record={kind:'cpb_v18_span_study_closeout_v1',study_id:'cpb-v18-span-001',created_utc:generatedUtc,work_status:'complete',study_valid_and_complete:true,
    question:'Does replacing scattered hidden patches with one contiguous four-patch span at fixed1024 updates improve the declared linear readout balance under the same encoder/objective and evaluation?',
    decision:{predeclared_continuation_criterion_passed:d.passed,predeclared_conditions:d.conditions,observed_gain:d.observed_gain,strict_trend_or_activity_gain_observed:d.strict_trend_or_activity_gain_observed,
      automatic_promotion:false,new_best_encoder_established:false,research_reference_replaced:false,application_selection_changed:false,
      interpretation:d.passed?'The fixed1024 development continuation criterion passed; broader superiority and reference promotion remain unestablished.':'The fixed1024 development continuation criterion failed; this declared mask-policy comparison does not support advancement.'},
    comparison:{control:'Original scattered-mask coefficient0.1 recipe, resumed from its own exact completed256 state.',candidate:'Fresh contiguous-span-mask coefficient0.1 recipe, with exact typed512-to1024 resume.',
      intended_change:'Training mask arrangement only; no architecture, parameter count, latent layout, Gaussian/temporal objective, target reduction, preprocessing or fixed evaluation change.',
      configuration:a.shared_recipe.configuration,geometry:'C3/H128/F9/P8/native36; IDs0,1,2; original16 patch positions, four adjacent hidden patches for dense candidate channels',shared_recipe:a.shared_recipe,
      paired_state_identity_limit:'Shared initialization/scaler/bank FNV tensor-byte identities and realized row/attempt checks are engineering witnesses, not cryptographic tensor equality proofs.',
      evaluation:'Unchanged112 TRAIN/32 known January VALIDATION H256 anchors retaining128; legacy F32-to-F64 targets; TRAIN-standardized ridge1; original independent scalar audit and tolerance.',evaluation_source_equivalence:a.evaluation_equivalence,
      historical_control_limit:'Controls retain original portable recipe and actual256 prefix ancestry; they are not historical selected V16 trajectory reproductions.'},
    paired_candidate_minus_control_r2:d.paired,aggregate:d.aggregate,trajectories,
    work_counts:{scientific:{logical_optimizer_updates:6144,new_physical_optimizer_updates:newUpdates,previously_executed_control_prefix_updates:reusedUpdates,
      new_attempted_batches:newAttempts,previous_control_prefix_attempts:reusedAttempts,total_logical_attempts:newAttempts+reusedAttempts,
      new_native_training_processes:12,new_candidate_fresh_processes:3,new_control_continuation_processes:6,new_candidate_resume_processes:3,
      new_TRAIN_preprocessing_fits:newFits,reused_control_TRAIN_preprocessing_fits:reusedFits,TRAIN_windows_per_fit:1024,head_fits:headFits,independent_scalar_audits:auditCount,independent_scalar_target_refits:auditCount*4,
      preparation_reexecutions:0,old_control_training_prefix_replays:0,new_gradient_diagnostic_executions:0},
      engineering:{summary:pin(engineeringPath),details:engineering,quality_evaluated:false},
      distinction:'6144 is logical training across six1024 histories; only5376 optimizer updates are new work in this study. The old768 control-prefix updates are preserved and never rerun.'},
    current_reference:{version:'CPB-v16',seed:946103,logical_update:2048,unchanged:true,declaration:pin(referencePath),checkpoint:check(reference.checkpoint),TRAIN_input_scaler:check(reference.TRAIN_input_scaler),TRAIN_bank_standardizer:check(reference.TRAIN_bank_standardizer)},
    measured:{final1024_January_linear_readouts:true,legacy_target_ID_group_support_parity:true,independent_scalar_ridge_arithmetic:true,actual_candidate_span_and_lag_witnesses:true,control_actual_detailed_mask_witnesses:false,
      actual_control256_to640_to1024_continuation:true,actual_candidate0_to512_to1024_fresh_resume:true,typed_final_inference_reload:true,
      fixed_query_reconstruction:false,TEMPO_timing:false,neural_readouts:false,fresh_population_confirmation:false,full2048_budget_superiority:false,forecasting:false,application_utility:false},
    limits:['January development is exposed; January/February/March2024 are consumed evidence for later selection.','Three seed pairs and a fixed1024 budget do not establish universal or full2048-budget encoder superiority.',
      'Contiguous masking changes training difficulty/support arrangement; no evaluation masks, target definitions or head recipe were changed.','Actual detailed mask/lag witnesses are available for candidate runs only; unavailable control fields are not reconstructed or invented.',
      'The original V17 structural ridge witness kind is reused strictly as a tensor codec; actual encoder kind, mask policy, checkpoint hash and full update count remain separately bound.',
      'No intermediate adaptive score gate, rescue seed, budget extension, neural/timing/fresh confirmation or automatic application update is performed.'],
    evidence:{study_admission:aPin,protocol_copy:check(a.protocol_copy),operation_contract:check(contract),phase_admissions:Object.fromEntries(Object.entries(phases).map(([key,p])=>[key,p.artifact])),training_seal:sealPin,
      screen_summary:pin(`${ATTEMPT}/screen-summary.json`),engineering_execution:pin(`${BASE}/engineering-execution.json`),engineering_admission:check(engineeringExecution.source_admission),engineering_summary:pin(engineeringPath),
      original_control_source_capsule:a.control_source.capture,original_control_source_capsule_files:108,reused_control_prefixes:prefixRecords,reused_data:a.reused_data,native_command_receipts:commandReceipts,
      verified_input_manifest:manifestPin,current_source_closure:{unique_files:currentSources.length,canonical_sorted_pin_array_sha256:sourceDigest,complete_pins_in:MANIFEST},executables,executable_set_sha256:executableDigest,
      frozen_V16_files_unchanged:49,frozen_family_headers_unchanged:3},
    reproduction:{generator:pin(fileURLToPath(import.meta.url)),model_calls:0,external_processes_invoked:0,tensor_archives_decoded:0,
      original_control_build_preservation:'The108 exact V17 source files preserve the old root build and attributes separately from the newly admitted V18 root build; origin and preserved paths retain explicit mapping.',
      publication_payload:'Compact records and source only. Data, checkpoints, optimizer states, full receipts and source capsules remain ignored local artifacts.'}};
  return {record,manifestText};
}
const number=n=>Number.isInteger(n)?String(n):n.toPrecision(8);
const signed=n=>`${n>=0?'+':''}${number(n)}`;
function markdown(r){const lines=['# CPB V18 contiguous-mask comparison','',r.decision.interpretation,'',
  '**V16 seed946103/final2048 remains the selected research reference.** This known-development comparison does not automatically establish a new best encoder.','',
  'The candidate changes training from scattered hidden patches to one contiguous four-patch span per eligible dense channel. Architecture/native36, temporal coefficient0.1, Gaussian/temporal objective, TRAIN fitting, optimizer and evaluation stay shared. All six final1024 states were sealed before any new readout.','',
  'Controls continue their own already completed256 states through640 and1024 using the exact original executable/recipe, with no refit. Candidates start fresh, stop at512 and resume their typed state to1024. Existing controls are reused evidence, not new prefix retrainings.','',
  '## Fixed January linear readouts','',
  'The unchanged evaluation uses112 TRAIN/32 known-validation probe anchors, legacy target arithmetic and TRAIN-standardized ridge1. All six runs have complete support and independent scalar arithmetic audits.','',
  '| Seed | Arm / mask | Volatility R2 | Trend R2 | Activity R2 | Relation R2 |','| --- | --- | ---: | ---: | ---: | ---: |'];
  for(const t of r.trajectories)lines.push(`| ${t.seed} | ${t.arm} / ${t.mask_mode} | ${TASKS.map(task=>number(t.metrics.find(m=>m.task===task).r2)).join(' | ')} |`);
  lines.push('','Candidate minus same-seed control:','','| Task | Seed947101 | Seed947102 | Seed947103 | Median | Minimum | Maximum | Nonnegative pairs |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
  for(const task of TASKS){const a=r.aggregate[task];lines.push(`| ${task} | ${r.paired_candidate_minus_control_r2.map(p=>signed(p[task])).join(' | ')} | ${signed(a.median)} | ${signed(a.minimum)} | ${signed(a.maximum)} | ${a.nonnegative_pairs}/3 |`);}
  lines.push('','| Predeclared condition | Result |','| --- | --- |');for(const [condition,passed] of Object.entries(r.decision.predeclared_conditions))lines.push(`| ${condition.replaceAll('_',' ')} | ${passed?'PASS':'FAIL'} |`);
  const trendSeeds=r.paired_candidate_minus_control_r2.filter(p=>p.past_trend>0).map(p=>p.seed),activitySeeds=r.paired_candidate_minus_control_r2.filter(p=>p.past_activity>0).map(p=>p.seed);
  lines.push('',`The continuation criterion **${r.decision.predeclared_continuation_criterion_passed?'passed':'failed'}**. Trend improved in ${trendSeeds.length}/3 pairs${trendSeeds.length?` (${trendSeeds.join(', ')})`:''}; activity improved in ${activitySeeds.length}/3 pairs${activitySeeds.length?` (${activitySeeds.join(', ')})`:''}. All gains and regressions remain visible; a tie is not a gain. Relation is descriptive and does not replace the declared primary conditions.`,
    '','## Actual work and reused evidence','',
    `Six complete1024 histories contain **6144 logical optimizer updates**. This study executed **5376 new physical updates**:2304 control continuation updates and3072 fresh candidate updates. The remaining **768 control-prefix updates** were already executed in V17 and were preserved without replay. Twelve new training processes implement the planned continuation boundaries. New TRAIN preprocessing fits:3; preserved control fits:3.`,
    '',`Actual attempts: ${r.work_counts.scientific.new_attempted_batches} new plus${r.work_counts.scientific.previous_control_prefix_attempts} preserved. New scientific readout work:24 target-head fits, six independent scalar audits and24 separate scalar target refits for verification. Dataset preparation and the old TRAIN gradient diagnostic were not rerun. Artificial engineering work remains separately counted in the compact record.`,
    '','## Actual contiguous-mask witnesses','',
    'Candidate logs retain returned starts, legal-span counts, quotas, exact decimal-string counter seeds, selected patch/target counts, unsupported reasons and all three temporal pair counts. On the declared dense B8/C3/H128/P8 geometry, each of24 channel draws has quota4 and13 legal starts0..12; each update selects96 patches,6912 target cells and lag1/4/16 pair counts6696/6048/3456. These are actual returned values checked per event; no second mask replay manufactured them. The original control CLI did not log detailed mask/lag fields, so those remain unavailable.','',
    '| Candidate seed | Saved attempts | Flattened channel draws | Lag1 pairs | Lag4 pairs | Lag16 pairs | Unsupported channels |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |');
  for(const t of r.trajectories.filter(t=>t.arm==='candidate')){const c=t.candidate_mask_counters;lines.push(`| ${t.seed} | ${c.events} | ${c.mask_witness.flattened_channel_draws} | ${c.temporal_pair_counts.total.join(' | ')} | ${c.totals.unsupported_channels} |`);}
  lines.push('','## Evidence and limits','',
    'Known January development data and one1024 budget do not establish fresh-period, universal or full2048-budget superiority. Neural heads, TEMPO timing, fixed-query reconstruction, forecasting and application utility were not measured. January/February/March2024 remain consumed evidence for later choices.',
    '', 'The preserved `cpb_v17_ridge_witness_v1` is a generic structural tensor codec reused by the unchanged audit. It is not the encoder identity: actual v17-scattered/v18-contiguous kind, mask policy, checkpoint SHA and1024 update count are separately bound.',
    '',`The [compact record](../../records/CPB-v18.span-001.json) links the admitted [protocol](PROTOCOL.md), [Operation Contract](CONTRACT.md), engineering proof, original108-file control source capsule, exact reused prefix states, twelve new training reports, all-six seal, six readouts/audits and verified source/input manifest. Full generated evidence remains under \`${ATTEMPT}/\`. The49 frozen V16 files, three family headers and selected reference checkpoint/statistics were checked unchanged.`,
    '', 'The reusable component retains configurable construction, TRAIN fitting, complete mask/objective/optimizer methods and typed training/resume/inference export. A benchmark geometry or checkpoint does not define the full family interface.',
    '',`Generated at ${r.created_utc} by metadata-only closeout; zero model/process calls and no tensor archive decoding.`, '');return lines.join('\n');}
function main(){if(process.argv.length===3&&process.argv[2]==='--help'){console.log('After the complete1024 screen: verify and write new-only compact JSON, RESULTS.md and ignored closeout-input-manifest.json. Metadata only.');return;}
  require(process.argv.length===2,'no arguments accepted except --help');const {record,manifestText}=collect(),results=markdown(record),bytes=Buffer.from(results);
  record.evidence.results={path:RESULTS,bytes:bytes.length,sha256:hash(bytes)};
  const fd=fs.openSync(absolute(MANIFEST),'wx');try{fs.writeFileSync(fd,manifestText);fs.fsyncSync(fd);}finally{fs.closeSync(fd);}
  const resultfd=fs.openSync(absolute(RESULTS),'wx');try{fs.writeFileSync(resultfd,results);fs.fsyncSync(resultfd);}finally{fs.closeSync(resultfd);}
  save(RECORD,record);console.log(JSON.stringify({status:'saved',record:pin(RECORD),results:pin(RESULTS),manifest:pin(MANIFEST),model_calls:0},null,2));}
if(process.argv[1]&&absolute(process.argv[1])===fileURLToPath(import.meta.url)){try{main();}catch(error){console.error(error.stack??String(error));process.exitCode=1;}}

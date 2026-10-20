// Metadata-only protocol tools. New receipts use a named encoding; frozen
// Python records and their original byte identities are never rewritten.
import fs from 'node:fs';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {fileURLToPath} from 'node:url';
import {requireCondition as check, safeRelative, canonicalDirectory, admitFiles,
  hashAdmitted, readAdmitted, parseJSON, jsonBytes, writeNew, writeJSON,
  createExclusiveDirectory, sha256, JSON_ENCODING} from '../../../../../shared/artifacts/node/v1/files.mjs';

export const SDK = '/opt/cuwacunu_embedding/setup/sdk-20261008T234205Z-6a74defe83e3/proof.json';
export const SDK_SHA = 'e370b91a50e264739c284c0fc5b461436b60a50735b36a2777750a6603b3a0bb';
export const INFO_SHA = '04fda53a38d8ead378dec5b276c62eba322ec5ea0361daab789a02ee3c03569f';
export const KNOWN = Object.freeze([75272,76373,77474,78575,79676]);
export const FRESH = Object.freeze([80787,81888,82989,84090,85191]);
export const FAMILIES = Object.freeze({
  architecture_screen_v1: {protocol:'architecture-screen-v1',masters:KNOWN,screen:[75272,76373],title:'Temporal architecture screen'},
  fixed_prior_relation_screen_v1: {protocol:'fixed-prior-relation-screen-v1',masters:KNOWN,screen:[75272,76373],title:'Fixed-prior temporal relation screen'},
  partitioned_relation_screen_v1: {protocol:'partitioned-relation-screen-v1',masters:KNOWN,screen:[75272,76373],title:'Partitioned temporal relation screen'},
  unit_relation_screen_v1: {protocol:'unit-relation-screen-v1',masters:FRESH,screen:[80787,84090],title:'Unit temporal relation screen',unit:true},
  fixed_prior_fresh_confirmation_v1: {protocol:'fixed-prior-fresh-confirmation-v1',masters:FRESH,fresh:true,title:'RPB-v17 fresh-source confirmation'},
  two_component_information_v1: {protocol:'two-component-information-v1',masters:[910901,910902],information:true},
  multiband_screen_v1: {protocol:'multiband-screen-v1',masters:[920903,920904],tasks:['slow_lag_sign','component_balance'],candidates:['v18','v19'],run:'rpb-multiband-screen'},
  spectral_confirmation_v1: {protocol:'spectral-confirmation-v1',masters:[930905,930906,930907],tasks:['slow_lag_sign','component_balance'],candidates:['v19'],run:'rpb-spectral-confirmation'},
});
const sharedSources = ['code/embeddings/evaluation/protocols/node/v1/metadata.mjs',
  'code/shared/artifacts/node/v1/files.mjs','code/scripts/node.sh','code/scripts/node-runtime.sh'];
const decode = row => parseJSON(readAdmitted(row), {allowBOM:false,duplicates:'last',integers:'bigint'});
const hash = file => hashAdmitted(admitFiles([file],{singleLink:true})[0]);
const read = file => decode(admitFiles([file],{singleLink:true})[0]);
const equal = (a,b) => jsonBytes(a).equals(jsonBytes(b));
const text = row => new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(readAdmitted(row));
const stamp = {language:'node',receipt_encoding:JSON_ENCODING,equivalence:'not-certified'};
function mkdir(directory) {
  const missing=[]; let current=directory;
  while (!fs.existsSync(current)) {missing.unshift(current);current=path.dirname(current);}
  canonicalDirectory(current);
  for (const target of missing) {fs.mkdirSync(target);canonicalDirectory(target);}
}
function copy(row,target) {mkdir(path.dirname(target));writeNew(target,readAdmitted(row));}
function rows(names,root) {return admitFiles(names.map(n=>path.join(root,safeRelative(n))),{root,singleLink:true});}
function manifest(names,admitted) {return names.map((n,i)=>`${hashAdmitted(admitted[i])}  ${n}\n`).join('');}
export function inputNames(masters,tasks=['lag_sign']) {
  return masters.flatMap(m=>tasks.flatMap(t=>['training','validation','validation-deleted'].map(v=>
    `results/seed-${m}-${t}/controlled-${v}.pt`)));
}
function sourceCapture(family,repo,sourceList) {
  const listRow=admitFiles([sourceList],{singleLink:true})[0];
  const list=text(listRow).split(/\r?\n/);if(list.at(-1)==='')list.pop();
  const declared=[...new Set(list)].sort();
  check(declared.length>0,'closed nonempty source list required');declared.forEach(safeRelative);
  const wrapper=`code/embeddings/evaluation/protocols/${family}/`;
  const names=[...new Set([...declared,...sharedSources,...fs.readdirSync(path.join(repo,wrapper))
    .filter(n=>n.endsWith('.mjs')).map(n=>wrapper+n)])].sort();
  const admitted=rows(names,repo),body=manifest(names,admitted);
  return {names,admitted,body,fingerprint:sha256(body),listRow};
}
function emitSource(capture,output) {
  writeNew(path.join(output,'sources.sha256'),capture.body);
  for(let i=0;i<capture.names.length;i++) copy(capture.admitted[i],path.join(output,'SOURCE',capture.names[i]));
  check(manifest(capture.names,capture.admitted)===capture.body,'source changed while capturing');
}
function verifySources(repo,output) {
  const body=text(admitFiles([path.join(output,'sources.sha256')],{singleLink:true})[0]);
  const lines=body.trimEnd().split('\n');
  const names=lines.map(line=>{check(/^[a-f0-9]{64}  /.test(line),'canonical source manifest');return safeRelative(line.slice(66));});
  check(equal(names,[...new Set(names)].sort()),'unique sorted source closure');
  for(const root of [repo,path.join(output,'SOURCE')]) check(manifest(names,rows(names,root))===body,'frozen source changed');
}
export function prepare(family,args) {
  const policy=FAMILIES[family];check(policy,'known metadata protocol');
  const repo=canonicalDirectory(args['repo-root']??'/mityane');
  const output=args.output;check(path.isAbsolute(output)&&path.normalize(output)===output,'exact output path');
  check(output.startsWith(path.join(repo,'.temp/embeddings/runs')+'/'),'new outputs stay in current ignored run root');
  if(args['bind-input-root']) return bindInputs(family,repo,output,args['bind-input-root']);
  check(args['source-list'],'source list required');
  const capture=sourceCapture(family,repo,args['source-list']);
  const card=path.join(repo,'code/embeddings/evaluation/cards',family+'.md');
  const metadata=admitFiles([card,args['sdk-proof']??SDK,...(policy.tasks?[path.join(repo,'.archive/embeddings/pre-cleanup-2026-10-18/doc/results/two_component_information_v1.json')]:[])],{singleLink:true});
  const cardSHA=hashAdmitted(metadata[0]);check(hashAdmitted(metadata[1])===SDK_SHA,'existing verified SDK identity changed');
  if(policy.tasks)check(hashAdmitted(metadata[2])===INFO_SHA,'verified information admission changed');
  let masters=policy.masters;
  if(!policy.fresh&&!policy.information&&!policy.tasks) {
    masters=(args.masters??'').split(',').map(x=>Number(x));
    check(masters.length>0&&masters.every(x=>Number.isSafeInteger(x)&&policy.masters.includes(x))&&new Set(masters).size===masters.length,'declared retained masters only');
  }
  let inputManifest=null,dataRow=null,inputAdmissions=[],inputRoles=[];
  if(args['input-root']) {
    canonicalDirectory(args['input-root']);
    const names=inputNames(masters);
    if(policy.fresh) {
      dataRow=admitFiles([path.join(args['input-root'],'data-report.json')],{singleLink:true})[0];
      const data=decode(dataRow);
      check(data.protocol===policy.protocol&&equal(data.masters,FRESH)&&data.source_fingerprint===capture.fingerprint&&data.card_sha256===cardSHA,'complete fresh writer must match source/card/cohorts');
      check(data.files.length===15&&new Set(data.files.map(r=>r.path)).size===15,'exact complete fresh file matrix');
      check(equal(data.files.map(r=>r.path).sort(),[...names].sort()),'exact fresh file role names');
      const admitted=rows(names,args['input-root']);
      const actual=new Map(admitted.map((r,i)=>[names[i],hashAdmitted(r)]));
      check(data.files.every(r=>actual.get(r.path)===r.sha256),'fresh archive differs from writer report');
      inputManifest=manifest(names,admitted);
      inputAdmissions=admitted;inputRoles=names;
    } else {inputAdmissions=rows(names,args['input-root']);inputRoles=names;inputManifest=manifest(names,inputAdmissions);}
  } else check(policy.fresh||policy.information||policy.tasks,'input root required');
  if(policy.tasks) {
    canonicalDirectory(path.dirname(output));
    check(path.basename(output)==='freeze'&&path.basename(path.dirname(output)).startsWith('admission-')&&
      path.dirname(path.dirname(output))===path.join(repo,'.temp/embeddings/runs',policy.run),'direct named admission freeze required');
  }
  createExclusiveDirectory(output);
  emitSource(capture,output);copy(metadata[0],path.join(output,'card.md'));copy(metadata[1],path.join(output,'sdk-proof.json'));
  const record={protocol:policy.protocol,masters,source_files:capture.names.length,source_fingerprint:capture.fingerprint,
    card_sha256:cardSHA,sdk_proof_sha256:SDK_SHA,model_execution:false,encoder_training:false,head_fitting:false,
    old_artifacts_replaced:false,migration_implementation:stamp};
  if(policy.information) Object.assign(record,{engineering_masters:masters,datasets:['TEMPO-4','AMP-2'],designed_complexity_level_each:5,complexity_scale_max:5,quality_generation:false});
  if(policy.fresh) record.instance_bundle='RPB-v17.alt-01';
  if(policy.tasks)Object.assign(record,{tasks:policy.tasks,candidates:policy.candidates,information_summary_sha256:INFO_SHA});
  else record.input_root=args['input-root']??null;
  if(inputManifest!==null) {writeNew(path.join(output,'inputs.sha256'),inputManifest);record.inputs_sha256=sha256(inputManifest);}
  if(dataRow) {copy(dataRow,path.join(output,'data-report.json'));record.data_report_sha256=hashAdmitted(dataRow);}
  if(inputManifest!==null)check(manifest(inputRoles,inputAdmissions)===inputManifest,'inputs changed while source freeze was written');
  writeJSON(path.join(output,policy.tasks?'engineering-freeze.json':'freeze.json'),record,{sortKeys:false});return record;
}
function bindInputs(family,repo,output,inputRoot) {
  const policy=FAMILIES[family];check(policy.tasks,'binding only for declared generated-data protocols');
  canonicalDirectory(output);canonicalDirectory(inputRoot);
  check(path.basename(output)==='freeze'&&path.dirname(path.dirname(output))===path.join(repo,'.temp/embeddings/runs',policy.run)&&
    path.basename(path.dirname(output)).startsWith('admission-')&&inputRoot===path.join(path.dirname(output),'quality-data'),'same-admission quality data');
  verifySources(repo,output);
  const meta=admitFiles([path.join(output,'engineering-freeze.json'),path.join(inputRoot,'data-report.json'),path.join(inputRoot,'data-complete.json')],{singleLink:true});
  const [record,data,complete]=meta.map(decode),count=policy.masters.length*6;
  check(complete.status==='complete'&&complete.protocol===data.protocol&&data.protocol===policy.protocol&&data.data_recipe==='two-component-v1'&&
    complete.source_fingerprint===data.source_fingerprint&&complete.human_card_sha256===data.human_card_sha256&&equal(complete.masters,data.masters)&&
    complete.archive_count===data.archive_count&&complete.data_report_sha256===hashAdmitted(meta[1])&&data.source_fingerprint===record.source_fingerprint&&
    data.human_card_sha256===record.card_sha256&&equal(data.masters,policy.masters)&&equal(data.tasks,policy.tasks)&&data.generator_calls===policy.masters.length*2&&
    data.archive_count===count&&data.encoder_calls===0&&data.information_fit_calls===0,'complete once-generated data with exact source/card required');
  const names=inputNames(policy.masters,policy.tasks);check(equal(data.files.map(r=>r.path),names),'exact ordered generated-file role matrix');
  const admitted=rows(names,inputRoot);
  check(admitted.every((row,i)=>row.bytes===data.files[i].bytes),'data writer byte sizes changed');
  check(admitted.every((row,i)=>hashAdmitted(row)===data.files[i].sha256),'data writer bytes changed');
  const body=manifest(names,admitted);writeNew(path.join(output,'inputs.sha256'),body);
  const result={...record,input_root:inputRoot,inputs_sha256:sha256(body),input_files:count,data_report_sha256:hashAdmitted(meta[1])};
  writeJSON(path.join(output,'freeze.json'),result,{sortKeys:false});return result;
}
export function captureMetadata(family,args) {
  const repo=canonicalDirectory(args['repo-root']??'/mityane'),admission=canonicalDirectory(args.admission);
  check(admission.startsWith(path.join(repo,'.temp/embeddings/runs')+'/'),'capture only new current capsules; historical evidence stays immutable');
  const prefix=`code/embeddings/evaluation/protocols/${family}/`;
  const names=[...new Set([...sharedSources,...fs.readdirSync(path.join(repo,prefix)).filter(n=>/\.(?:mjs|sh)$/.test(n)).map(n=>prefix+n)])].sort();
  const admitted=rows(names,repo),destination=path.join(admission,'METADATA_SOURCE');createExclusiveDirectory(destination);
  for(let i=0;i<names.length;i++)copy(admitted[i],path.join(destination,names[i]));
  writeNew(path.join(admission,'metadata-sources.sha256'),manifest(names,admitted));
  const paths=[];
  function walk(directory) {for(const entry of fs.readdirSync(directory,{withFileTypes:true})) {
    check(!entry.isSymbolicLink(),'no redirected capsule entries');const child=path.join(directory,entry.name);
    if(entry.isDirectory()){canonicalDirectory(child);walk(child);}else{check(entry.isFile(),'regular capsule entries');paths.push(child);}
  }}
  walk(admission);paths.sort();const all=admitFiles(paths,{root:admission,singleLink:true});
  writeNew(path.join(admission,'artifact-inventory.sha256'),manifest(paths.map(p=>path.relative(admission,p)),all));
  return {metadata_tools:names.length,artifacts:paths.length,inventory_sha256:hash(path.join(admission,'artifact-inventory.sha256'))};
}
export function cli(family,action,argv=process.argv.slice(2)) {
  check(process.platform==='linux'&&fs.existsSync('/.dockerenv')&&process.cwd()==='/mityane','use managed development container');
  const options={};for(const name of ['repo-root','source-list','input-root','output','masters','sdk-proof','bind-input-root','admission','summary','report'])options[name]={type:'string'};
  options.run={type:'string',multiple:true};
  const {values}=parseArgs({args:argv,options,strict:true});
  const result=action==='prepare'?prepare(family,values):action==='capture'?captureMetadata(family,values):publish(family,values);
  process.stdout.write(jsonBytes(result,{indent:0,sortKeys:false}));return result;
}
export const isMain = url => process.argv[1]&&fileURLToPath(url)===path.resolve(process.argv[1]);

// Python statistics.mean uses exact rational accumulation of binary floats.
// Preserve that computation for metadata scores, including cancellation cases.
export function mean(values) {
  check(Array.isArray(values)&&values.length>0,'nonempty mean');if(values.includes(null))return null;
  const view=new DataView(new ArrayBuffer(8));let numerator=0n;
  for(const number of values){check(typeof number==='number'&&Number.isFinite(number),'finite metadata mean');
    view.setFloat64(0,number,false);const bits=view.getBigUint64(0,false),sign=bits>>63n?-1n:1n;
    const exp=Number((bits>>52n)&2047n),fraction=bits&((1n<<52n)-1n);
    numerator+=sign*(exp===0?fraction:((1n<<52n)|fraction)<<BigInt(exp-1));}
  return rationalFloat(numerator,BigInt(values.length)*(1n<<1074n));
}
function rationalFloat(n,d) {
  if(n===0n)return 0;const negative=n<0n;if(negative)n=-n;
  let exponent=n.toString(2).length-d.toString(2).length;
  if(exponent>=0?(n<d<<BigInt(exponent)):(n<<BigInt(-exponent)<d))exponent--;
  const shift=Math.max(-1074,exponent-52),a=shift<0?n<<BigInt(-shift):n,b=shift>0?d<<BigInt(shift):d;
  let q=a/b;const r=a%b;if(r*2n>b||(r*2n===b&&(q&1n)))q++;
  return (negative?-1:1)*Number(q)*2**shift;
}
export function rowsFor(cohorts,updates,view) {
  return cohorts.map(cohort=>{const points=cohort.points.filter(p=>p.updates===updates);check(points.length===1,'one retained update point');
    const methods=points[0].readouts.methods;check(methods.length===1,'one fixed method');const method=methods[0],reps=method.repetitions,supported=method.status==='measured';
    return {master:cohort.master,ridge:supported?mean(reps.map(r=>r[view].ridge.accuracy)):null,
      tiny_secondary:supported?mean(reps.map(r=>r[view].tiny_secondary.accuracy)):null,
      coverage:supported?mean(reps.map(r=>r[view].ridge.coverage)):method[view+'_population'].coverage,
      fit_status:method.status,reason:method.reason,repetitions:reps.map(r=>Object.fromEntries(['id','actual_probe_seed_decimal',view].filter(k=>Object.hasOwn(r,k)).map(k=>[k,r[k]])))};});
}
export function aggregate(label,rows,size=32,reused=false) {
  check(rows.length>0,'nonempty declared cohort population');const values=key=>rows.map(r=>r[key]);
  return {label,size,evidence:reused?'reused':'new',per_master:rows,ridge_mean:mean(values('ridge')),
    tiny_secondary_mean:mean(values('tiny_secondary')),coverage_mean:mean(values('coverage')),
    ridge_worst:values('ridge').includes(null)?null:Math.min(...values('ridge')),
    tiny_secondary_worst:values('tiny_secondary').includes(null)?null:Math.min(...values('tiny_secondary'))};
}
function table(rows) {const display=(v,scale=100)=>v===null?'—':(scale*v).toFixed(2);
  return ['| Method | Size | Linear head % | Neural head % | Coverage % |','| --- | ---: | ---: | ---: | ---: |',
    ...rows.map(r=>`| ${r.label} | ${r.size} | ${display(r.ridge_mean)} | ${display(r.tiny_secondary_mean)} | ${display(r.coverage_mean)} |`)].join('\n');}
function training(cohorts,runs,label) {
  return {label,updates:512,train_error:mean(cohorts.map(c=>c.training_reconstruction.standardized_mae)),
    validation_error:mean(cohorts.map(c=>c.validation_reconstruction.standardized_mae)),GPU_training_seconds:mean(cohorts.map(c=>c.progress.training_seconds)),
    per_master:cohorts.map(c=>{const progress=path.join(runs.get(`${c.model_tag}:${c.master}`),`seed-${c.master}/progress.json`);
      return {master:c.master,training_reconstruction:c.training_reconstruction,validation_reconstruction:c.validation_reconstruction,costs:c.costs,
        progress:Object.fromEntries(Object.entries(c.progress).filter(([key])=>key!=='losses')),full_progress_artifact:progress,full_progress_sha256:hash(progress)};})};
}
function trainingTable(rows) {
  const display=(value,digits)=>value===null?'—':value.toFixed(digits);
  return ['| Encoder | Updates | Train error | Validation error | GPU training seconds |',
    '| --- | ---: | ---: | ---: | ---: |',...rows.map(r=>
      `| ${r.label} | ${r.updates} | ${display(r.train_error,6)} | ${display(r.validation_error,6)} | ${display(r.GPU_training_seconds,2)} |`)].join('\n');
}
export function publish(family,args) {
  const policy=FAMILIES[family];check(policy?.screen||policy?.fresh,'metadata summary protocol required');
  const repo=canonicalDirectory(args['repo-root']??'/mityane'),runs=new Map(),candidates=new Map(),inputs=[];
  check(Array.isArray(args.run)&&args.run.length>0&&args.summary&&args.report,'runs and exclusive summary/report required');
  for(const output of [args.summary,args.report])check(path.isAbsolute(output)&&path.normalize(output)===output&&
    output.startsWith(path.join(repo,'.temp/embeddings/runs')+'/'),
  'new publication output must stay in the current ignored run root');
  for(const run of args.run) {canonicalDirectory(run);const admitted=admitFiles([path.join(run,'report.json'),path.join(run,'saved-check.json')],{singleLink:true});
    const [report,saved]=admitted.map(decode);check(saved.status==='passed'&&saved.report_sha256===hashAdmitted(admitted[0]),'passed saved check bound to exact report');
    if(policy.fresh)check(report.protocol===policy.protocol&&equal(report.masters,FRESH),'complete fresh confirmation required');
    for(const c of report.cohorts){const key=`${c.model_tag}:${c.master}`;check(!runs.has(key),'duplicate candidate/cohort');runs.set(key,run);
      if(!candidates.has(c.model_tag))candidates.set(c.model_tag,[]);candidates.get(c.model_tag).push(c);}
    inputs.push({run,report_sha256:hashAdmitted(admitted[0]),saved_check_sha256:hashAdmitted(admitted[1]),source_fingerprint:report.source_fingerprint,saved_check:saved});
  }
  const summary={protocol:policy.protocol,dataset_id:'TEMPO-3',task:'lag_sign',recipe_id:'structured-hard-timing-v1',designed_complexity_level:4,
    complexity_scale_max:5,promotion:false,old_instances_replaced:false,testing_accessed:false,migration_implementation:stamp};
  const lines=[`# ${policy.title}`,'','Metadata summary of hash-bound saved reports. TRAIN/VALIDATION only; all retained failures are included.','',
    'Native32; Ridge1 and tanh16/Adam .01/100, three head repetitions. Untrained scores describe the built-in prior; they do not establish learning.',''];
  if(policy.fresh) publishFresh(args,summary,lines,candidates,runs,inputs);
  else publishScreen(policy,repo,args,summary,lines,candidates,runs,inputs);
  lines.push('No automatic promotion or reference replacement. Saved baselines are reused without fitting or model execution.','',
    'Receipt encoding: '+JSON_ENCODING+'. This migrated publisher has not yet been certified equivalent to the frozen Python publication.','');
  mkdir(path.dirname(args.summary));mkdir(path.dirname(args.report));
  writeJSON(args.summary,summary,{sortKeys:false});writeNew(args.report,lines.join('\n')+'\n');
  return {summary_sha256:hash(args.summary),report_sha256:hash(args.report),...(summary.advancement?{advancement:Object.fromEntries(Object.entries(summary.advancement).map(([k,v])=>[k,v.passed]))}:
    {confirmation:Object.fromEntries(Object.entries(summary.confirmation).filter(([,v])=>typeof v==='object').map(([k,v])=>[k,v.passed]))})};
}
function publishScreen(policy,repo,args,summary,lines,candidates,runs,inputs) {
  const parentName=policy.unit?'fixed_prior_fresh_confirmation_v1.json':'structured_hard_timing_comparison_v2.json';
  const parentPath=path.join(repo,'.archive/embeddings/pre-cleanup-2026-10-18/doc/results',parentName);
  check(hash(parentPath)===(policy.unit?'6255530c52f710dbb5311e385d7529f878ba3637ce5f5c9101b93864a43bab11':'1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4'),'protected baseline metadata changed');
  const parent=read(parentPath);Object.assign(summary,{screen_masters:policy.screen,inputs,stages:{},advancement:{}});
  if(args.admission)summary.captured_evidence={admission:args.admission,metadata_sources_manifest_sha256:hash(path.join(args.admission,'metadata-sources.sha256')),artifact_inventory_sha256:hash(path.join(args.admission,'artifact-inventory.sha256'))};
  for(const [tag,cohorts] of candidates) {const screen=cohorts.filter(c=>policy.screen.includes(c.master));
    check(equal(screen.map(c=>c.master).sort((a,b)=>a-b),policy.screen),'both declared screen cohorts required');const conditions=[];
    for(const cohort of screen)for(const view of ['validation_intact','validation_deleted']) {const r=rowsFor([cohort],512,view)[0];
      const base={master:cohort.master,view,ridge:r.ridge,tiny_secondary:r.tiny_secondary,coverage:r.coverage};
      let passed=r.ridge!==null&&r.tiny_secondary!==null&&Math.min(r.ridge,r.tiny_secondary)>=.75&&r.coverage===1;
      if(policy.unit){const old=parent.quality[view].per_master.filter(x=>x.master===cohort.master);check(old.length===1,'one matched saved v17 cohort');base.saved_v17_neural=old[0].tiny_secondary;passed=passed&&r.tiny_secondary+1e-15>=old[0].tiny_secondary;}
      conditions.push({...base,passed});}
    if(policy.unit){const candidate=mean(rowsFor(screen,512,'validation_deleted').map(r=>r.ridge)),old=mean(parent.quality.validation_deleted.per_master.filter(r=>policy.screen.includes(r.master)).map(r=>r.ridge));
      const gain=candidate===null?null:candidate-old;conditions.push({scope:'two-cohort mean deleted Linear gain',saved_v17_mean:old,candidate_mean:candidate,gain,required_gain:.05,passed:gain!==null&&gain+1e-15>=.05});}
    summary.advancement[tag]={passed:conditions.every(r=>r.passed),conditions};}
  for(const stage of ['screen','all_known']) {const selected=[...candidates].filter(([,cs])=>stage==='screen'||cs.length===5).map(([tag,cs])=>[tag,cs.filter(c=>stage!=='screen'||policy.screen.includes(c.master))]);if(!selected.length)continue;
    const masters=[...new Set(selected.flatMap(([,cs])=>cs.map(c=>c.master)))].sort((a,b)=>a-b),data={masters,quality:{},untrained:{},training:[]};
    lines.push(`## ${stage==='screen'?'Two-cohort screen':'Five known cohorts'}`,'',`Masters: ${masters.join(', ')}.`,'');
    for(const view of ['validation_intact','validation_deleted']) {const old=policy.unit?[parent.quality[view]]:parent.quality[view].filter(r=>['raw','pca_only','native_early'].includes(r.method));
      data.quality[view]=[...old.map(r=>aggregate(r.label,r.per_master.filter(x=>masters.includes(x.master)),r.size,true)),...selected.map(([tag,cs])=>aggregate(tag,rowsFor(cs,512,view)))];
      data.untrained[view]=selected.map(([tag,cs])=>aggregate(tag,rowsFor(cs,0,view)));lines.push(`TEMPO-3 · complexity 4/5 · ${view} · trained512`,'',table(data.quality[view]),'',`Untrained0 · ${view}`,'',table(data.untrained[view]),'');}
    data.training=selected.map(([tag,cs])=>training(cs,runs,tag));
    lines.push('TEMPO-3 · fixed masked TRAIN/intact VALIDATION queries · standardized MAE','',trainingTable(data.training),'');
    summary.stages[stage]=data;}
  for(const [tag,gate]of Object.entries(summary.advancement))lines.push(`- ${tag}: ${gate.passed?'passes':'fails'} the predeclared continuation gate.`);
}
function publishFresh(args,summary,lines,candidates,runs,inputs) {
  check(args.run.length===1&&args.admission&&candidates.size===1&&candidates.has('RPB-v17.alt-01'),'one complete declared confirmation');
  const cohorts=candidates.get('RPB-v17.alt-01');check(equal(cohorts.map(c=>c.master),FRESH),'all five declared instances retained');
  const report=read(path.join(args.run[0],'report.json')),freeze=read(path.join(args.admission,'freeze/freeze.json'));
  check(freeze.source_fingerprint===report.source_fingerprint&&freeze.inputs_sha256===report.inputs_sha256,'admission/source/data freeze matches quality');
  const hashes={engineering_log_sha256:'engineering.log',data_report_sha256:'fresh-data/data-report.json',freeze_json_sha256:'freeze/freeze.json',metadata_sources_manifest_sha256:'metadata-sources.sha256',artifact_inventory_sha256:'artifact-inventory.sha256'};
  const input={...inputs[0],card_sha256:report.human_card_sha256,inputs_sha256:report.inputs_sha256,sdk_proof_sha256:report.sdk_proof_sha256,admission:args.admission,compiled_source_files:freeze.source_files};
  for(const[key,name]of Object.entries(hashes))input[key]=hash(path.join(args.admission,name));
  Object.assign(summary,{instance_bundle:'RPB-v17.alt-01',model_design:'RPB-v17',masters:FRESH,input,quality:{},untrained:{},execution_counts:report.counts,stress_accessed:false,selection:false});
  const continuity=[],capability=[];
  for(const view of ['validation_intact','validation_deleted']) {const trained=aggregate('RPB-v17.alt-01',rowsFor(cohorts,512,view));summary.quality[view]=trained;summary.untrained[view]=aggregate('RPB-v17.alt-01',rowsFor(cohorts,0,view));
    lines.push(`TEMPO-3 · complexity 4/5 · ${view} · trained512`,'',table([trained]),'',`Untrained0 · ${view}`,'',table([summary.untrained[view]]),'');
    for(const row of trained.per_master){const supported=row.ridge!==null&&row.tiny_secondary!==null,base={master:row.master,view,ridge:row.ridge,tiny_secondary:row.tiny_secondary,coverage:row.coverage};
      continuity.push({...base,passed:supported&&Math.min(row.ridge,row.tiny_secondary)>=.75&&row.coverage===1});const threshold=view.endsWith('intact')?.95:.75;
      capability.push({...base,linear_threshold:threshold,neural_threshold:.95,passed:supported&&row.ridge>=threshold&&row.tiny_secondary>=.95&&row.coverage===1});}}
  summary.confirmation={continuity:{passed:continuity.every(c=>c.passed),conditions:continuity},strong_capability:{passed:capability.every(c=>c.passed),conditions:capability},all_five_retained:true,intermediate_score_selection:false};
  summary.training=training(cohorts,runs,'RPB-v17.alt-01');
  lines.push('TEMPO-3 · fixed hidden-target TRAIN/intact VALIDATION queries · standardized MAE','',trainingTable([summary.training]),'',
    'Training seconds are synchronized loop wall time including CPU batch preparation/transfers; native extraction, head fitting, queries and checkpoint I/O are excluded.','');
  lines.push(`Continuity: ${summary.confirmation.continuity.passed?'passes':'fails'}.`,`Strong capability: ${summary.confirmation.strong_capability.passed?'passes':'fails'}.`,'',
    'Credit the fixed 12-coordinate prior. Timing accuracy alone does not establish learned improvement or broad representation quality.');
}

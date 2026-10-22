// Admission, exclusive artifacts and phase sealing; numerical work is Linux C++.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {spawn, spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {admitFiles, canonicalDirectory, createExclusiveDirectory, fileSHA256, inside,
  jsonBytes, parseJSON, readAdmitted, readJSON, requireCondition, sha256, writeJSON, writeNew}
  from '../../../../shared/artifacts/node/v1/files.mjs';

const ROOT = '/mityane';
const COMPONENT = 'code/operations/counterfactuals';
const STUDY = `${COMPONENT}/evaluation/representation_contribution_v1`;
const STORE = `${ROOT}/.temp/operations/counterfactuals/representation-contribution-v1`;
const BUILD = '/opt/mityane-geometrics';
const SOURCE_FILES = ['AGENTS.md', 'Makefile', 'environment-image.json', 'dependencies.lock',
  'container.ps1', 'doc/operations_over_representations_design_spec_v0.1.0.md',
  'code/scripts/task.sh', 'code/scripts/maintenance.mk', 'code/scripts/node.sh', 'code/scripts/node-runtime.sh',
  'code/shared/artifacts/node/v1/files.mjs', `${COMPONENT}/CONTRACT.md`, `${COMPONENT}/EVIDENCE.md`,
  `${COMPONENT}/counterfactuals.mk`, `${COMPONENT}/evaluation/synthetic_response_v1/configuration.json`,
  `${COMPONENT}/evaluation/synthetic_response_v1/INTERFACE.md`,
  ...['configuration.json', 'PROTOCOL.md', 'RER.md', 'contribution.cpp', 'run.mjs', 'run.test.mjs', 'audit.py']
    .map(name => `${STUDY}/${name}`)].sort();

export class Generator {
  constructor(seed, usedB = new Set()) { this.state = seed >>> 0; this.usedB = usedB; }
  next() { return this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0; }
  rows(split, count) {
    const rows = [];
    while (rows.length < count) {
      const a = this.next() % 6145 - 3072, b = this.next() % 8193 - 4096;
      if (this.usedB.has(b)) continue;
      this.usedB.add(b);
      rows.push([`${split}-${rows.length}`, a, b, this.next() >>> 31 ? 1 : -1]);
    }
    return rows;
  }
}
export function priorExclusions(configuration, includeEngineering = true) {
  const result = new Set();
  for (const seed of configuration.prior_exposure.seeds) {
    const generator = new Generator(seed);
    for (const split of ['training', 'development', 'assessment'])
      generator.rows(split, configuration.prior_exposure.counts[split]);
    for (const b of generator.usedB) result.add(b);
  }
  const originalOnly = new Set(result);
  // Preserve the accidentally generated preflight candidate population as exposed.
  const candidates = configuration.preflight_exposure;
  const generators = candidates.seeds.map(seed => new Generator(seed, result));
  for (const generator of generators)
    for (const split of ['training', 'development']) generator.rows(split, candidates.counts[split]);
  for (const generator of generators) generator.rows('assessment', candidates.counts.assessment);
  if (includeEngineering) {
    const base = new Set(result), engineering = configuration.engineering_exposure;
    for (const reserved of [originalOnly, base]) {
      const generator = new Generator(engineering.seed, reserved);
      for (const split of ['training','development','assessment']) generator.rows(split,engineering.counts[split]);
      for (const b of reserved) result.add(b);
    }
    for (let i=0;i<96;i++) result.add((i*79+17)%8193-4096);
    result.add(0);result.add(1);
  }
  return [...result].sort((a,b) => a-b);
}
function tsv(rows) { return `unit\ta_num\tb_num\tnoise_sign\n${rows.map(row => row.join('\t')).join('\n')}\n`; }
function pin(filename, relativeTo) {
  const record = admitFiles([filename])[0];
  return {path: relativeTo ? path.relative(relativeTo, filename) : filename,
    bytes: record.bytes, sha256: sha256(readAdmitted(record))};
}
function sources() { return SOURCE_FILES.map(name => ({...pin(path.join(ROOT, name)), path: name})); }
function checkSources(freeze) {
  requireCondition(sha256(jsonBytes(sources())) === freeze.source_id, 'Frozen source changed; preserve attempt.');
  requireCondition(fileSHA256(freeze.binary.path) === freeze.binary.sha256, 'Sealed native binary changed.');
}
function checkPins(attempt, pins) {
  for (const row of pins) requireCondition(fileSHA256(path.join(attempt,row.path)) === row.sha256, `Sealed artifact changed: ${row.path}`);
}
function append(index,event) {
  const fd = fs.openSync(index, fs.constants.O_WRONLY|fs.constants.O_CREAT|fs.constants.O_APPEND|fs.constants.O_NOFOLLOW, 0o600);
  try { fs.writeFileSync(fd, JSON.stringify({...event, timestamp:new Date().toISOString()})+'\n'); fs.fsyncSync(fd); }
  finally { fs.closeSync(fd); }
}
function processStart(pid) {
  try { const text=fs.readFileSync(`/proc/${pid}/stat`,'utf8'); return text.slice(text.lastIndexOf(')')+2).split(' ')[19]; }
  catch { return null; }
}
async function child(command,args,log,timeout,control) {
  requireCondition(!control.signal,`Interrupted: ${control.signal}`);
  await new Promise((resolve,reject) => {
    const proc=spawn(command,args,{cwd:ROOT,stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONDONTWRITEBYTECODE:'1',OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1'}});
    control.child=proc;
    if(proc.pid) writeJSON(`${log}.process.json`,{pid:proc.pid,process_start:processStart(proc.pid),command,args});
    let output='',problem;
    const stop=message=>{ problem=message; proc.kill('SIGTERM'); };
    const timer=setTimeout(()=>stop('stage timeout'),timeout);
    proc.stdout.on('data',data=>{output+=data; if(output.length>16*1024*1024) stop('log limit');});
    proc.stderr.on('data',data=>{output+=data; if(output.length>16*1024*1024) stop('log limit');});
    proc.on('error',error=>{problem=error.message;});
    proc.on('close',(code,signal)=>{
      clearTimeout(timer);control.child=null;
      try {
        writeNew(log,output); writeJSON(`${log}.exit.json`,{code,signal,problem:problem??null});
        requireCondition(!problem&&!control.signal&&code===0&&!signal,`Stage failed: ${problem??control.signal??signal??code}; ${log}`);
        resolve();
      } catch(error) {reject(error);}
    });
  });
}
function filesBelow(directory,base=directory) {
  const rows=[];
  for(const entry of fs.readdirSync(directory,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))) {
    const filename=path.join(directory,entry.name);
    requireCondition(!entry.isSymbolicLink(),`Artifact redirect: ${filename}`);
    if(entry.isDirectory()) rows.push(...filesBelow(filename,base));
    else {requireCondition(entry.isFile(),`Unexpected artifact: ${filename}`); rows.push(pin(filename,base));}
  }
  return rows;
}
function argumentsFor(args) {
  const options={};
  for(let i=0;i<args.length;i++) {
    if(['--self-check','--recover'].includes(args[i])) options[args[i].slice(2)]=true;
    else if(['--binary','--attempt'].includes(args[i])) {
      const key=args[i].slice(2);requireCondition(!options[key]&&args[i+1],'Missing or duplicate explicit argument');options[key]=args[++i];
    } else throw new Error(`Unknown argument: ${args[i]}`);
  }
  requireCondition(options.attempt&&(options.binary||options.recover),'Explicit --attempt and --binary required');
  return options;
}
function recover(attempt) {
  canonicalDirectory(attempt);
  requireCondition(!['final.json','failure.json','recovery.json'].some(name=>fs.existsSync(path.join(attempt,name))),'Attempt already finalized');
  const runtime=readJSON(path.join(attempt,'runtime.json'));
  requireCondition(processStart(runtime.pid)!==runtime.process_start,'Original process is active');
  for(const file of filesBelow(attempt).filter(row=>row.path.endsWith('.process.json'))) {
    if(fs.existsSync(path.join(attempt,file.path.replace(/\.process\.json$/,'.exit.json')))) continue;
    const active=readJSON(path.join(attempt,file.path));
    requireCondition(processStart(active.pid)===null||(active.process_start!==null&&processStart(active.pid)!==active.process_start),'Original child is active');
  }
  const record={study_id:'OR06-REP-CONTRIB-001',attempt,disposition:'interrupted',assessment_opened:fs.existsSync(path.join(attempt,'assessment-open.json')),reason:'Original processes absent; preserved without replay.'};
  writeJSON(path.join(attempt,'recovery.json'),record);append(path.join(path.dirname(attempt),'index.jsonl'),record);
  console.log(JSON.stringify(record));return 0;
}
export async function run(args) {
  requireCondition(process.platform==='linux'&&fs.existsSync('/.dockerenv')&&process.cwd()===ROOT,'Use the managed container via container.ps1');
  const options=argumentsFor(args),engineering=options['self-check']===true;
  requireCondition(path.isAbsolute(options.attempt)&&path.normalize(options.attempt)===options.attempt,'Exact absolute attempt path required');
  requireCondition(engineering?inside(BUILD,options.attempt):path.dirname(options.attempt)===STORE,'Attempt outside permitted root');
  if(options.recover) return recover(options.attempt);
  requireCondition(path.isAbsolute(options.binary)&&inside(BUILD,options.binary),'Binary must remain in managed build root');
  const attempt=createExclusiveDirectory(options.attempt),index=path.join(path.dirname(attempt),'index.jsonl');
  const purpose=engineering?'engineering_self_check':'representation_contribution_evaluation';
  const event=(disposition,detail={})=>append(index,{study_id:'OR06-REP-CONTRIB-001',attempt,scope:purpose,disposition,...detail});
  let admitted=false,opened=false;
  const started=process.hrtime.bigint(),control={signal:null,child:null};
  const interrupt=()=>{control.signal='SIGINT';control.child?.kill('SIGTERM');};
  const terminate=()=>{control.signal='SIGTERM';control.child?.kill('SIGTERM');};
  process.once('SIGINT',interrupt);process.once('SIGTERM',terminate);
  try {
    writeJSON(path.join(attempt,'runtime.json'),{pid:process.pid,process_start:processStart(process.pid),hostname:os.hostname()});event('admission_pending');
    const configuration=readJSON(path.join(ROOT,STUDY,'configuration.json'),{duplicates:'reject',integers:'number'});
    requireCondition(configuration.study_id==='OR06-REP-CONTRIB-001'&&configuration.version==='1.0.0','Unknown study');
    if(engineering) {configuration.seeds=[40031];configuration.counts={training:64,development:16,assessment:24};configuration.bootstrap.samples=100;}
    const prior=fs.readFileSync(index,'utf8').trim().split('\n').map(line=>parseJSON(line,{duplicates:'reject'})).filter(row=>row.attempt!==attempt);
    if(!engineering) requireCondition(!prior.some(row=>row.scope===purpose&&(row.disposition==='assessment_opened'||row.disposition==='completed'||row.assessment_opened)),'Prior assessment exposure exists; fresh replay refused');
    const sourceRows=sources(),sourceId=sha256(jsonBytes(sourceRows));
    for(const row of sourceRows) {
      const destination=path.join(attempt,'source',row.path);fs.mkdirSync(path.dirname(destination),{recursive:true});
      const bytes=readAdmitted(admitFiles([path.join(ROOT,row.path)])[0]);requireCondition(sha256(bytes)===row.sha256,'Source mutated during capture');writeNew(destination,bytes);
    }
    const configurationSha=writeJSON(path.join(attempt,'configuration.json'),configuration);
    const permissionsSha=writeJSON(path.join(attempt,'permissions.json'),{date:'2026-10-20',user_request:'Complete the goals already assigned to Hx06, including the entire remaining Resolve representation contribution goal. Continue through evaluation, reporting, the readout decision and evidence-backed dashboard updates.',selected_scope:configuration.study_id,purpose,limits:['Synthetic CPU only','No external services/data/models','No inherited or GPU research','No overwriting or score-based replay']});
    const excluded=priorExclusions(configuration,!engineering);
    requireCondition(excluded.length+configuration.seeds.length*Object.values(configuration.counts).reduce((a,b)=>a+b,0)<=8193,'Insufficient reserved coordinate capacity');
    const exposureSha=writeJSON(path.join(attempt,'prior_exposure.json'),{original_study:configuration.prior_exposure.study_id,original_config_sha256:fileSHA256(path.join(ROOT,COMPONENT,'evaluation/synthetic_response_v1/configuration.json')),excluded_b:excluded,policy:configuration.prior_exposure.policy,preflight_exposure:configuration.preflight_exposure,engineering_exposure:configuration.engineering_exposure,engineering_excluded:!engineering,task_formula_and_old_result_already_exposed:true});
    const compiler=spawnSync('g++',['-dumpfullversion'],{encoding:'utf8'}),python=spawnSync('python3',['--version'],{encoding:'utf8'});
    requireCondition(compiler.status===0&&python.status===0,'Existing compiler/auditor runtime unavailable');
    const built=pin(options.binary),sealedDirectory=createExclusiveDirectory(path.join(BUILD,'operations/counterfactuals/contribution-sealed',sha256(attempt+sourceId)));
    const sealedBinary=path.join(sealedDirectory,'representation_contribution');writeNew(sealedBinary,readAdmitted(admitFiles([options.binary])[0]),{mode:0o755});
    requireCondition(fileSHA256(sealedBinary)===built.sha256,'Build snapshot changed');
    const freeze={study_id:configuration.study_id,version:configuration.version,purpose,source_id:sourceId,sources:sourceRows,configuration_sha256:configurationSha,permissions_sha256:permissionsSha,exposure_sha256:exposureSha,binary:pin(sealedBinary),build_binary:built,node:process.version,compiler:compiler.stdout.trim(),python:python.stdout.trim(),environment_image:readJSON(path.join(ROOT,'environment-image.json')),assessment_opened:false,metrics_observed:false,prior_attempt_events:prior};
    const freezeSha=writeJSON(path.join(attempt,'freeze.json'),freeze);checkSources(freeze);
    if(!engineering) writeJSON(path.join(STORE,'scientific-admission.json'),{study_id:configuration.study_id,attempt,freeze_sha256:freezeSha,policy:'Single scientific admission; never replay as fresh confirmation'});
    admitted=true;event('registered',{freeze_sha256:freezeSha});
    const globalB=new Set(excluded),generators=new Map(),training=[],models=[];
    for(const seed of configuration.seeds) {
      checkSources(freeze);
      const directory=createExclusiveDirectory(path.join(attempt,`seed-${seed}`)),generator=new Generator(seed,globalB);generators.set(seed,generator);
      for(const split of ['training','development']) {const file=path.join(directory,`${split}.tsv`);writeNew(file,tsv(generator.rows(split,configuration.counts[split])));training.push(pin(file,attempt));}
      writeJSON(path.join(directory,'pre-assessment-generator.json'),{state:generator.state,global_reserved_b:[...globalB].sort((a,b)=>a-b)});
      await child(sealedBinary,['fit',path.join(directory,'training.tsv'),path.join(directory,'model.tsv')],path.join(directory,'fit.log'),configuration.budget.native_timeout_ms,control);
      models.push({seed,...pin(path.join(directory,'model.tsv'),attempt)});
    }
    const pins=[...training,...models];
    const sealSha=writeJSON(path.join(attempt,'seal.json'),{freeze_sha256:freezeSha,models,training,sealed_at:new Date().toISOString(),assessment_opened:false});event('models_sealed',{seal_sha256:sealSha});
    for(const seed of configuration.seeds) {
      checkSources(freeze);checkPins(attempt,pins);
      const dir=path.join(attempt,`seed-${seed}`);await child(sealedBinary,['evaluate',path.join(dir,'development.tsv'),path.join(dir,'model.tsv'),path.join(dir,'development-pairs.tsv')],path.join(dir,'development.log'),configuration.budget.native_timeout_ms,control);
    }
    checkSources(freeze);checkPins(attempt,pins);
    writeJSON(path.join(attempt,'assessment-open.json'),{freeze_sha256:freezeSha,seal_sha256:sealSha,all_models_sealed:true,assessment_previously_generated:false,selection_policy:configuration.selection_policy,opened_at:new Date().toISOString()});event('assessment_opened',{seal_sha256:sealSha});opened=true;
    for(const seed of configuration.seeds) {
      checkSources(freeze);checkPins(attempt,pins);
      const dir=path.join(attempt,`seed-${seed}`),generator=generators.get(seed);writeNew(path.join(dir,'assessment.tsv'),tsv(generator.rows('assessment',configuration.counts.assessment)));
      writeJSON(path.join(dir,'final-generator.json'),{state:generator.state,global_reserved_b:[...globalB].sort((a,b)=>a-b)});
      await child(sealedBinary,['evaluate',path.join(dir,'assessment.tsv'),path.join(dir,'model.tsv'),path.join(dir,'assessment-pairs.tsv')],path.join(dir,'assessment.log'),configuration.budget.native_timeout_ms,control);
    }
    checkSources(freeze);checkPins(attempt,pins);
    // Execute the frozen auditor capsule, not a subsequently edited working file.
    await child('python3',['-B',path.join(attempt,'source',STUDY,'audit.py'),attempt,path.join(attempt,'audit.json')],path.join(attempt,'audit.log'),configuration.budget.audit_timeout_ms,control);
    const audit=readJSON(path.join(attempt,'audit.json'),{duplicates:'reject',integers:'number'});requireCondition(audit.valid===true,'Independent audit invalid');
    checkSources(freeze);checkPins(attempt,pins);
    const auditSha=fileSHA256(path.join(attempt,'audit.json')),manifest=filesBelow(attempt),manifestSha=writeJSON(path.join(attempt,'file-manifest.json'),{files:manifest});
    const final={study_id:configuration.study_id,disposition:'completed',verdict:audit.synthetic_screen?.passed?'supported_in_scope':'negative_or_inconclusive_in_scope',assessment_opened:opened,source_unchanged:true,freeze_sha256:freezeSha,seal_sha256:sealSha,audit_sha256:auditSha,file_manifest_sha256:manifestSha,elapsed_seconds:Number(process.hrtime.bigint()-started)/1e9,limits:'Fixed synthetic representation/readout compatibility only; no learned encoder, observational or causal utility claim.'};
    writeJSON(path.join(attempt,'final.json'),final);event('completed',{verdict:final.verdict,audit_sha256:auditSha});console.log(JSON.stringify({attempt,...final}));return 0;
  } catch(error) {
    const failure={study_id:'OR06-REP-CONTRIB-001',disposition:control.signal?'interrupted':admitted?'failed':'admission_rejected',assessment_opened:opened,error:error.stack??String(error),elapsed_seconds:Number(process.hrtime.bigint()-started)/1e9};
    writeJSON(path.join(attempt,'failure.json'),failure);event(failure.disposition,{assessment_opened:opened,error:error.message});throw error;
  } finally {process.removeListener('SIGINT',interrupt);process.removeListener('SIGTERM',terminate);}
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) run(process.argv.slice(2)).then(code=>{process.exitCode=code;}).catch(error=>{console.error(error.stack??error);process.exitCode=1;});

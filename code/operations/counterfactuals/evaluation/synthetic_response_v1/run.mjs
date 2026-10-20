// Operation-local prospective study orchestration; run only in the managed container.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {spawn, spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {
  admitFiles, canonicalDirectory, createExclusiveDirectory, fileSHA256,
  inside, jsonBytes, parseJSON, readAdmitted, readJSON, requireCondition,
  sha256, writeJSON, writeNew,
} from '../../../../shared/artifacts/node/v1/files.mjs';
import {auditAttempt} from './audit.mjs';

const ROOT = '/mityane';
const COMPONENT = 'code/operations/counterfactuals';
const STUDY = `${COMPONENT}/evaluation/synthetic_response_v1`;
const ARTIFACT_ROOT = `${ROOT}/.temp/operations/counterfactuals`;
const STUDY_ARTIFACT_ROOT = `${ARTIFACT_ROOT}/synthetic-response-v1`;
const BUILD_ROOT = '/opt/mityane-geometrics';
const SOURCE_FILES = [
  'Makefile', 'AGENTS.md', 'environment-image.json', 'dependencies.lock',
  'doc/operations_over_representations_design_spec_v0.1.0.md',
  'code/scripts/task.sh', 'code/scripts/node.sh', 'code/scripts/node-runtime.sh',
  'code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/sidecar.h',
  'code/shared/artifacts/node/v1/files.mjs',
  `${COMPONENT}/CONTRACT.md`, `${COMPONENT}/EVIDENCE.md`, `${COMPONENT}/counterfactuals.mk`,
  `${COMPONENT}/include/mityane/operations/counterfactuals/paired_response.h`,
  `${COMPONENT}/src/paired_response.cpp`,
  ...['configuration.json', 'INTERFACE.md', 'PROTOCOL.md', 'RER.md',
      'run.mjs', 'audit.mjs', 'synthetic_response.cpp'].map(name => `${STUDY}/${name}`),
].sort();

export class Generator {
  constructor(seed) { this.state = seed >>> 0; this.usedB = new Set(); }
  next() { this.state = (Math.imul(1664525, this.state) + 1013904223) >>> 0; return this.state; }
  rows(split, count) {
    const rows = [];
    while (rows.length < count) {
      const a = this.next() % 6145 - 3072;
      const b = this.next() % 8193 - 4096;
      if (this.usedB.has(b)) continue;
      this.usedB.add(b);
      const noise = (this.next() >>> 31) ? 1 : -1;
      rows.push([`${split}-${rows.length}`, a, b, noise]);
    }
    return rows;
  }
  snapshot() { return {state: this.state, used_b: [...this.usedB].sort((a, b) => a - b)}; }
}

function tsv(rows) {
  return `unit\ta_num\tb_num\tnoise_sign\n${rows.map(row => row.join('\t')).join('\n')}\n`;
}
function sources() {
  const admitted = admitFiles(SOURCE_FILES.map(name => path.join(ROOT, name)), {root: ROOT});
  return admitted.map((row, i) => ({path: SOURCE_FILES[i], bytes: row.bytes, sha256: sha256(readAdmitted(row))}));
}
function checkSources(frozen) {
  requireCondition(sha256(jsonBytes(sources())) === frozen.source_id, 'Source changed after prospective freeze; preserve attempt.');
  requireCondition(fileSHA256(frozen.binary.path) === frozen.binary.sha256, 'Native binary changed after freeze.');
}
function pin(filename) {
  const row = admitFiles([filename])[0];
  return {path: filename, bytes: row.bytes, sha256: sha256(readAdmitted(row))};
}
function checkPins(rows) {
  for (const row of rows)
    requireCondition(fileSHA256(row.path) === row.sha256, `Sealed artifact changed: ${row.path}`);
}
function append(index, event) {
  const line = JSON.stringify({...event, timestamp: new Date().toISOString()}) + '\n';
  const fd = fs.openSync(index, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_APPEND | fs.constants.O_NOFOLLOW, 0o600);
  try { fs.writeFileSync(fd, line); fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
}
async function native(binary, args, log, timeout, control) {
  requireCondition(!control.signal, `Interrupted by ${control.signal}`);
  await new Promise((resolve, reject) => {
    const child = spawn(binary, args, {cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe']});
    control.child = child;
    if (child.pid) writeJSON(`${log}.process.json`, {pid: child.pid, process_start: processStart(child.pid), binary, args});
    let stdout = '', stderr = '', problem;
    const stop = message => { problem = message; child.kill('SIGTERM'); };
    const timer = setTimeout(() => stop('native timeout'), timeout);
    child.stdout.on('data', chunk => { stdout += chunk; if (stdout.length + stderr.length > 16 * 1024 * 1024) stop('native log limit'); });
    child.stderr.on('data', chunk => { stderr += chunk; if (stdout.length + stderr.length > 16 * 1024 * 1024) stop('native log limit'); });
    child.on('error', error => { problem = error.message; });
    child.on('close', (code, signal) => {
      clearTimeout(timer); control.child = null;
      try {
        writeJSON(`${log}.exit.json`, {code, signal, problem: problem ?? null});
        writeNew(log, `stdout:\n${stdout}\nstderr:\n${stderr}`);
        requireCondition(!problem && !control.signal && code === 0 && !signal,
          `Native stage failed: ${problem ?? control.signal ?? signal ?? code}; log ${log}`);
        resolve();
      } catch (error) { reject(error); }
    });
  });
}
function filesBelow(directory) {
  const rows = [];
  for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
    const filename = path.join(directory, entry.name);
    requireCondition(!entry.isSymbolicLink(), `Artifact redirect: ${filename}`);
    if (entry.isDirectory()) rows.push(...filesBelow(filename));
    else { requireCondition(entry.isFile(), `Unexpected artifact type: ${filename}`); rows.push(pin(filename)); }
  }
  return rows;
}
function parsedArguments(args) {
  const result = {selfCheck: false, recover: false};
  for (let i = 0; i < args.length; ++i) {
    if (args[i] === '--self-check') result.selfCheck = true;
    else if (args[i] === '--recover') result.recover = true;
    else if (args[i] === '--binary' || args[i] === '--attempt') {
      requireCondition(i + 1 < args.length, 'Missing explicit argument value.');
      const name = args[i].slice(2);
      requireCondition(result[name] === undefined, `Duplicate ${name}`);
      result[name] = args[++i];
    } else throw new Error(`Unknown argument: ${args[i]}`);
  }
  requireCondition(result.attempt && (result.recover || result.binary), 'Explicit --attempt and, for execution, --binary are required.');
  return result;
}
function processStart(pid) {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19];
  } catch { return null; }
}
function recover(attempt) {
  canonicalDirectory(attempt);
  requireCondition(!['final.json', 'failure.json', 'recovery.json'].some(name => fs.existsSync(path.join(attempt, name))),
    'Attempt already finalized; preserve its disposition.');
  const runtime = readJSON(path.join(attempt, 'runtime.json'), {duplicates: 'reject'});
  requireCondition(processStart(runtime.pid) !== runtime.process_start, 'Original study process is still active.');
  const childFiles = filesBelow(attempt).filter(row => row.path.endsWith('.log.process.json'));
  for (const row of childFiles) {
    const exit = row.path.replace(/\.process\.json$/, '.exit.json');
    if (fs.existsSync(exit)) continue;
    const child = readJSON(row.path, {duplicates: 'reject'});
    const current = processStart(child.pid);
    requireCondition(current === null || (child.process_start !== null && current !== child.process_start),
      'A native child of the interrupted wrapper is still active.');
  }
  const freezePath = path.join(attempt, 'freeze.json');
  const freeze = fs.existsSync(freezePath) ? readJSON(freezePath, {duplicates: 'reject'}) : null;
  const record = {study_id: 'OR06-SYNTH-RESP-001', disposition: 'interrupted', attempt,
    assessment_opened: fs.existsSync(path.join(attempt, 'assessment-open.json')),
    original_runtime: runtime, original_freeze_sha256: freeze ? fileSHA256(freezePath) : null,
    reason: 'Original process is absent; no experiment replay or result inference performed.'};
  writeJSON(path.join(attempt, 'recovery.json'), record);
  append(path.join(path.dirname(attempt), 'index.jsonl'), {...record, scope: freeze?.purpose ?? 'admission_incomplete'});
  console.log(JSON.stringify(record));
  return 0;
}

export async function run(args) {
  requireCondition(process.platform === 'linux' && fs.existsSync('/.dockerenv') && process.cwd() === ROOT,
    'Use container.ps1 in the managed mityane-geometrics container.');
  const options = parsedArguments(args);
  requireCondition(path.isAbsolute(options.attempt) && path.normalize(options.attempt) === options.attempt,
    'Attempt must be an exact absolute path.');
  const allowedRoot = options.selfCheck ? BUILD_ROOT : ARTIFACT_ROOT;
  requireCondition(inside(allowedRoot, options.attempt) && options.attempt !== allowedRoot,
    'Attempt escapes its permitted artifact root.');
  if (!options.selfCheck) requireCondition(path.dirname(options.attempt) === STUDY_ARTIFACT_ROOT,
    'Scientific attempts must share the fixed synthetic-response-v1 ledger directory.');
  if (options.recover) return recover(options.attempt);
  requireCondition(path.isAbsolute(options.binary) && path.normalize(options.binary) === options.binary &&
    inside(BUILD_ROOT, options.binary), 'Native binary must be an exact managed-container build path.');
  canonicalDirectory(path.dirname(options.binary));
  const configurationPath = path.join(ROOT, STUDY, 'configuration.json');
  const attempt = createExclusiveDirectory(options.attempt);
  const configuration = {study_id: 'OR06-SYNTH-RESP-001'};
  const index = path.join(path.dirname(attempt), 'index.jsonl');
  const event = (disposition, details = {}) => append(index,
    {study_id: configuration.study_id, attempt, scope: options.selfCheck ? 'engineering_self_check' : 'synthetic_operation_evaluation', disposition, ...details});
  let admitted = false;
  let assessmentOpened = false;
  const control = {signal: null, child: null};
  const signalHandler = signal => { control.signal = signal; control.child?.kill('SIGTERM'); };
  const interrupt = () => signalHandler('SIGINT'), terminate = () => signalHandler('SIGTERM');
  process.once('SIGINT', interrupt); process.once('SIGTERM', terminate);
  const started = process.hrtime.bigint();
  try {
    writeJSON(path.join(attempt, 'runtime.json'), {pid: process.pid, process_start: processStart(process.pid), hostname: os.hostname()});
    event('admission_pending');
    Object.assign(configuration, readJSON(configurationPath, {duplicates: 'reject', integers: 'number'}));
    requireCondition(configuration.study_id === 'OR06-SYNTH-RESP-001' && configuration.version === '1.0.0' &&
      configuration.claim_level === 'controlled_model_response', 'Unsupported prospective configuration.');
    if (options.selfCheck) {
      configuration.seeds = [4001]; configuration.counts = {training: 64, development: 16, assessment: 16};
      configuration.bootstrap.samples = 100;
    }
    const prior = fs.existsSync(index) ? fs.readFileSync(index, 'utf8').trim().split('\n').filter(Boolean)
      .map(line => parseJSON(line, {duplicates: 'reject'})).filter(row => row.attempt !== attempt) : [];
    if (!options.selfCheck) {
      requireCondition(!prior.some(row => row.scope === 'synthetic_operation_evaluation' &&
        (row.disposition === 'assessment_opened' || row.disposition === 'completed' || row.assessment_opened)),
      'Previous scientific assessment exposure exists; a fresh prospective replay is refused.');
    }
    const sourceRows = sources();
    const sourceId = sha256(jsonBytes(sourceRows));
    for (const row of sourceRows) {
      const destination = path.join(attempt, 'source', row.path);
      fs.mkdirSync(path.dirname(destination), {recursive: true});
      const bytes = readAdmitted(admitFiles([path.join(ROOT, row.path)], {root: ROOT})[0]);
      requireCondition(sha256(bytes) === row.sha256, `Source changed while capturing capsule: ${row.path}`);
      writeNew(destination, bytes);
    }
    const configSha = writeJSON(path.join(attempt, 'configuration.json'), configuration);
    const permissionsSha = writeJSON(path.join(attempt, 'permissions.json'), {
      user_request: "that is good, ok, let's continue the next goal",
      context: 'Continue Hx06 evaluation after the component and supporting-file boundaries were explained.',
      interpretation: 'A bounded local synthetic CPU evaluation selected to implement that request; no external provider or another Hero is admitted.',
      selected_scope: configuration.study_id,
      purpose: options.selfCheck ? 'Engineering preflight only; not scientific evidence.' : 'Prospective synthetic controlled-model-response study.',
      limits: ['No inherited encoders/checkpoints', 'No external data/services/downloads', 'No GPU', 'One sequential CPU process', 'Exclusive local artifacts; no overwrites'],
    });
    const compiler = spawnSync('g++', ['-dumpfullversion'], {encoding: 'utf8'});
    requireCondition(compiler.status === 0, 'Cannot identify existing compiler.');
    const builtBinary = pin(options.binary);
    const binaryDirectory = createExclusiveDirectory(path.join(BUILD_ROOT, 'operations/counterfactuals/sealed', sha256(attempt + sourceId)));
    const sealedBinary = path.join(binaryDirectory, 'synthetic_response');
    const binaryBytes = readAdmitted(admitFiles([options.binary])[0]);
    requireCondition(sha256(binaryBytes) === builtBinary.sha256, 'Native build changed before snapshot.');
    writeNew(sealedBinary, binaryBytes, {mode: 0o755});
    const freeze = {
      study_id: configuration.study_id, version: configuration.version,
      purpose: options.selfCheck ? 'engineering_self_check' : 'synthetic_operation_evaluation',
      claim_level: configuration.claim_level, source_id: sourceId, sources: sourceRows,
      configuration_sha256: configSha, permissions_sha256: permissionsSha,
      binary: pin(sealedBinary), build_binary: builtBinary, node: process.version, compiler: compiler.stdout.trim(),
      managed_launcher: 'container.ps1', hostname: os.hostname(),
      environment_image: readJSON(path.join(ROOT, 'environment-image.json'), {duplicates: 'reject'}),
      assessment_opened: false, metrics_observed: false,
      prior_attempt_events: prior,
      exposure: 'No scientific study data/model/results existed in this attempt before freeze; separate seed-4001 engineering preflight is excluded.',
    };
    const freezeSha = writeJSON(path.join(attempt, 'freeze.json'), freeze);
    checkSources(freeze);
    if (!options.selfCheck) writeJSON(path.join(path.dirname(attempt), 'scientific-admission.json'), {
      study_id: configuration.study_id, attempt, freeze_sha256: freezeSha,
      policy: 'One prospective scientific admission per study/version; subsequent work needs explicit new protocol/exposure scope.',
    });
    admitted = true;
    event('registered', {freeze_sha256: freezeSha});

    const generators = new Map();
    const seals = [];
    for (const seed of configuration.seeds) {
      checkSources(freeze);
      const directory = createExclusiveDirectory(path.join(attempt, `seed-${seed}`));
      const generator = new Generator(seed);
      generators.set(seed, generator);
      const train = path.join(directory, 'train.tsv');
      const development = path.join(directory, 'development.tsv');
      const model = path.join(directory, 'model.tsv');
      writeNew(train, tsv(generator.rows('training', configuration.counts.training)));
      writeNew(development, tsv(generator.rows('development', configuration.counts.development)));
      writeJSON(path.join(directory, 'pre-assessment-generator.json'), generator.snapshot());
      await native(sealedBinary, ['fit', train, model, String(seed)], path.join(directory, 'fit.log'), configuration.budget.native_timeout_ms, control);
      seals.push(...[train, development, model, `${model}.metadata.tsv`].map(pin));
    }
    const sealSha = writeJSON(path.join(attempt, 'model-seal.json'), {freeze_sha256: freezeSha, models_and_fit_inputs: seals, assessment_opened: false});
    event('models_sealed', {model_seal_sha256: sealSha});
    for (const seed of configuration.seeds) {
      checkSources(freeze); checkPins(seals);
      const directory = path.join(attempt, `seed-${seed}`);
      const output = path.join(directory, 'development'); // native owns exclusive output creation
      await native(sealedBinary, ['evaluate', path.join(directory, 'development.tsv'), path.join(directory, 'model.tsv'), output,
        String(seed), sourceId, configSha], path.join(directory, 'development.log'), configuration.budget.native_timeout_ms, control);
    }
    checkSources(freeze); checkPins(seals);
    writeJSON(path.join(attempt, 'assessment-open.json'), {freeze_sha256: freezeSha, model_seal_sha256: sealSha,
      all_models_sealed: true, selection_policy: configuration.selection_policy, assessment_previously_generated: false});
    event('assessment_opened', {model_seal_sha256: sealSha});
    assessmentOpened = true;
    for (const seed of configuration.seeds) {
      checkSources(freeze); checkPins(seals);
      const directory = path.join(attempt, `seed-${seed}`);
      const input = path.join(directory, 'assessment.tsv');
      const generator = generators.get(seed);
      writeNew(input, tsv(generator.rows('assessment', configuration.counts.assessment)));
      writeJSON(path.join(directory, 'final-generator.json'), generator.snapshot());
      const output = path.join(directory, 'assessment'); // native owns exclusive output creation
      await native(sealedBinary, ['evaluate', input, path.join(directory, 'model.tsv'), output,
        String(seed), sourceId, configSha], path.join(directory, 'assessment.log'), configuration.budget.native_timeout_ms, control);
    }
    checkSources(freeze); checkPins(seals);
    const audit = await auditAttempt(attempt, configuration);
    requireCondition(!control.signal, `Interrupted by ${control.signal}`);
    const bindingValid = audit.bindings?.source_sha256 === sourceId && audit.bindings?.configuration_sha256 === configSha;
    const configurationValid = options.selfCheck || audit.scientific_configuration_matches_frozen_protocol === true;
    audit.wrapper_binding_check = {valid: bindingValid && configurationValid, configuration_matches: configurationValid,
      expected_source_sha256: sourceId, expected_configuration_sha256: configSha};
    const auditSha = writeJSON(path.join(attempt, 'audit.json'), audit);
    checkSources(freeze); checkPins(seals);
    const fileManifest = filesBelow(attempt);
    const filesSha = writeJSON(path.join(attempt, 'file-manifest.json'), {files: fileManifest});
    const valid = audit.valid && bindingValid && configurationValid;
    const disposition = valid ? 'completed' : 'invalid';
    const verdict = !valid ? 'invalid' : audit.supported ? 'supported_in_scope' : 'refuted_in_scope';
    writeJSON(path.join(attempt, 'final.json'), {
      study_id: configuration.study_id, scope: freeze.purpose, disposition, verdict,
      freeze_sha256: freezeSha, model_seal_sha256: sealSha, audit_sha256: auditSha,
      file_manifest_sha256: filesSha, elapsed_seconds: Number(process.hrtime.bigint() - started) / 1e9,
      assessment_opened: assessmentOpened, source_unchanged: true,
      limits: 'Synthetic controlled-model-response pipeline only. No real-system causal or inherited-encoder claim.',
    });
    event(disposition, {verdict, audit_sha256: auditSha});
    console.log(JSON.stringify({attempt, disposition, verdict, audit_valid: audit.valid, supported: audit.supported, metrics: audit.metrics?.pooled}));
    return valid ? 0 : 1;
  } catch (error) {
    const disposition = control.signal ? 'interrupted' : admitted ? 'failed' : 'admission_rejected';
    const failure = {study_id: configuration.study_id, disposition, assessment_opened: assessmentOpened,
      error: error.stack ?? String(error), elapsed_seconds: Number(process.hrtime.bigint() - started) / 1e9};
    if (!fs.existsSync(path.join(attempt, 'failure.json'))) writeJSON(path.join(attempt, 'failure.json'), failure);
    event(disposition, {assessment_opened: assessmentOpened, error: error.message});
    throw error;
  } finally {
    process.removeListener('SIGINT', interrupt); process.removeListener('SIGTERM', terminate);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run(process.argv.slice(2)).then(code => { process.exitCode = code; }).catch(error => { console.error(error.stack ?? String(error)); process.exitCode = 1; });
}

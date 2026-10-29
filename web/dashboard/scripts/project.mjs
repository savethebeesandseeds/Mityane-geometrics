// SPDX-License-Identifier: AGPL-3.0-only
import {readFile, readdir, readlink, realpath, stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {resolve, dirname, basename, relative, isAbsolute} from 'node:path';
import {validateManifest, validateHero, validateProject, validateReadyMilestones, flattenMilestones} from '../src/model.js';

export const dashboardDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MAX_VERIFICATION_BYTES = 4 * 1024 * 1024;
const insist = (condition, message) => { if (!condition) throw new Error(message); };
const digest = data => createHash('sha256').update(data).digest('hex');
const missing = error => ['ENOENT', 'ENOTDIR'].includes(error.code);

function repositoryFile(value, kind, allowFragment = false) {
  insist(typeof value === 'string' && value.length > 0 && value.length <= 500, `${kind} path must be bounded text`);
  const [file, fragment, ...extra] = value.split('#');
  insist(extra.length === 0 && (fragment === undefined || (allowFragment && /^[A-Za-z0-9_-]+$/.test(fragment))), `${kind} path has an invalid fragment: ${value}`);
  insist(/^[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*$/.test(file) && file.split('/').every(part => part !== '.' && part !== '..'), `${kind} path must be repository-relative without traversal: ${value}`);
  return file;
}

async function resolvedPath(target, depth = 0) {
  insist(depth < 40, 'Repository path contains too many symlink or missing path components');
  try { return {path: await realpath(target), exists: true}; }
  catch (error) { if (!missing(error)) throw error; }
  // Follow dangling symlinks as well as existing parents, matching non-strict
  // resolution without permitting missing targets to bypass containment.
  try { return resolvedPath(resolve(dirname(target), await readlink(target)), depth + 1); }
  catch (error) { if (!missing(error) && error.code !== 'EINVAL') throw error; }
  const parent = dirname(target);
  insist(parent !== target, `Cannot resolve repository path: ${target}`);
  return {path: resolve((await resolvedPath(parent, depth + 1)).path, basename(target)), exists: false};
}

async function repositoryTarget(root, file, kind) {
  const resolved = await resolvedPath(resolve(root, file));
  const path = relative(root, resolved.path);
  insist(path !== '..' && !path.startsWith('../') && !path.startsWith('..\\') && !isAbsolute(path), `${kind} path escapes repository: ${file}`);
  return resolved.exists ? resolved.path : null;
}

async function verificationInput(root, file, inputs) {
  file = repositoryFile(file, 'Verification');
  const target = await repositoryTarget(root, file, 'Verification');
  let data = null;
  if (target !== null) {
    try {
      const information = await stat(target);
      if (information.isFile()) {
        insist(information.size <= MAX_VERIFICATION_BYTES, `Verification input exceeds 4 MiB: ${file}`);
        data = await readFile(target);
        insist(data.length <= MAX_VERIFICATION_BYTES, `Verification input exceeds 4 MiB: ${file}`);
      }
    } catch (error) { if (!missing(error)) throw error; }
  }
  if (inputs.has(file)) {
    const previous = inputs.get(file);
    insist(previous === null ? data === null : data !== null && previous.equals(data), `Verification input changed during build: ${file}`);
  }
  inputs.set(file, data);
  return data;
}

async function verificationAssessments(heroes, root, inputs) {
  const assessments = {};
  for (const {milestone} of flattenMilestones(heroes)) {
    const authored = milestone.verification;
    if (!authored) {
      assessments[milestone.id] = {status: 'unverified', scope: 'Not assessed', reason: 'No current evidence assessment is recorded.'};
      continue;
    }
    const assessment = {status: authored.status, scope: authored.scope, reason: authored.reason};
    assessments[milestone.id] = assessment;
    if (authored.status !== 'current') continue;
    const stale = reason => Object.assign(assessment, {status: 'stale', reason});
    const data = await verificationInput(root, authored.receipt, inputs);
    if (data === null) { stale('The pinned verification receipt is missing.'); continue; }
    if (digest(data) !== authored.receiptSha256) { stale('The verification receipt no longer matches its recorded SHA256.'); continue; }
    try {
      const receipt = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(data).replace(/^\uFEFF/, ''));
      insist(receipt !== null && typeof receipt === 'object' && !Array.isArray(receipt), 'Verification receipt must be an object');
      const sources = receipt.sourceFiles;
      insist(Array.isArray(sources) && sources.length <= 128, 'Verification sourceFiles must be an array of at most 128 entries');
      insist(sources.length > 0, 'Verification sourceFiles must not be empty');
      const seen = new Set();
      for (const source of sources) {
        insist(source !== null && typeof source === 'object' && !Array.isArray(source) && Object.keys(source).length === 2 && Object.hasOwn(source, 'path') && Object.hasOwn(source, 'sha256'), 'Verification source must contain only path and sha256');
        const file = repositoryFile(source.path, 'Verification');
        await repositoryTarget(root, file, 'Verification');
        insist(typeof source.sha256 === 'string' && /^[a-f0-9]{64}$/.test(source.sha256), 'Verification source SHA256 must be a lowercase SHA256');
        insist(!seen.has(file), 'Verification receipt repeats a source path');
        seen.add(file);
      }
      const mismatches = [];
      for (const source of sources) {
        const current = await verificationInput(root, source.path, inputs);
        if (current === null || digest(current) !== source.sha256) mismatches.push(source.path);
      }
      if (mismatches.length) stale('Source bytes differ from the bounded receipt: ' + mismatches.join(', '));
    } catch (error) { stale(`The pinned receipt cannot establish current evidence: ${error.message}`); }
  }
  return assessments;
}

function recordedAssessments(heroes) {
  const assessments = {};
  for (const {milestone} of flattenMilestones(heroes)) {
    const authored = milestone.verification;
    assessments[milestone.id] = authored
      ? {status: authored.status, scope: authored.scope, reason: authored.reason}
      : {status: 'unverified', scope: 'Not assessed', reason: 'No current evidence assessment is recorded.'};
  }
  return assessments;
}

async function loadProject(directory, inputs, {audit = false} = {}) {
  const json = async file => JSON.parse((await readFile(resolve(directory, file), 'utf8')).replace(/^\uFEFF/, ''));
  const manifest = validateManifest(await json('manifest.json'));
  const heroes = await Promise.all(manifest.heroes.map(async entry => validateHero(await json(entry.file))));
  validateProject(manifest, heroes);
  const milestones = flattenMilestones(heroes);
  const record = {manifest, heroes};
  const assesses = milestones.some(({milestone}) => Object.hasOwn(milestone, 'verification') || Object.hasOwn(milestone, 'requiresCurrentEvidence'));
  if (audit) {
    const root = await realpath(resolve(directory, '../..'));
    for (const {milestone} of milestones) {
      for (const evidence of milestone.evidence || []) {
        const file = repositoryFile(evidence.path, 'Evidence', true);
        const target = await repositoryTarget(root, file, 'Evidence');
        insist(target !== null && (await stat(target)).isFile(), `Evidence must resolve to a regular file: ${file}`);
      }
      if (milestone.verification?.receipt) {
        const file = repositoryFile(milestone.verification.receipt, 'Verification');
        await repositoryTarget(root, file, 'Verification');
      }
    }
    if (assesses) record.verificationAssessments = await verificationAssessments(heroes, root, inputs);
  } else if (assesses) {
    record.verificationAssessments = recordedAssessments(heroes);
  }
  validateReadyMilestones(heroes, record.verificationAssessments || {});
  return record;
}

export async function readProject(directory = dashboardDirectory) {
  return loadProject(directory, new Map(), {audit: true});
}

export async function readPublishedProject(directory = dashboardDirectory) {
  return loadProject(directory, new Map());
}

export async function sourceFingerprint(directory = dashboardDirectory) {
  const files = ['manifest.json', 'package.json'];
  async function collect(folder) {
    for (const entry of await readdir(resolve(directory, folder), {withFileTypes: true})) {
      const file = `${folder}/${entry.name}`;
      if (entry.isDirectory()) await collect(file);
      else if (entry.isFile()) files.push(file);
      else throw new Error(`Dashboard inputs must be ordinary files: ${file}`);
    }
  }
  for (const folder of ['src', 'scripts', 'heroes', 'schemas', 'assets', 'tests']) await collect(folder);
  const hash = createHash('sha256');
  for (const file of files.sort()) hash.update(file).update('\0').update(await readFile(resolve(directory, file))).update('\0');
  const inputs = new Map();
  await loadProject(directory, inputs, {audit: true});
  for (const [file, data] of [...inputs].sort(([left], [right]) => left.localeCompare(right))) {
    hash.update('repository/').update(file).update('\0');
    hash.update(data === null ? 'missing' : String(data.length)).update('\0');
    if (data !== null) hash.update(data);
    hash.update('\0');
  }
  return hash.digest('hex');
}

async function collectFiles(directory, folder, files) {
  let entries;
  try { entries = await readdir(resolve(directory, folder), {withFileTypes: true}); }
  catch (error) { if (missing(error)) return; throw error; }
  for (const entry of entries) {
    const file = `${folder}/${entry.name}`;
    if (entry.isDirectory()) await collectFiles(directory, file, files);
    else if (entry.isFile()) files.push(file);
    else throw new Error(`Dashboard inputs must be ordinary files: ${file}`);
  }
}

export async function publishFingerprint(directory = dashboardDirectory) {
  const files = ['manifest.json'];
  await collectFiles(directory, 'heroes', files);
  await collectFiles(directory, 'src', files);
  await collectFiles(directory, 'assets/branding', files);
  const hash = createHash('sha256');
  for (const file of files.sort()) hash.update(file).update('\0').update(await readFile(resolve(directory, file))).update('\0');
  return hash.digest('hex');
}

#!/usr/bin/env node
/** Source/capsule preparation only. No model, fitter, subprocess or experiment runs here.
 * Frozen history preserves original Python helpers. Current source closures include
 * this implementation and receive new fingerprints; old receipts are never re-sealed.
 */
import fs from 'node:fs';
import path from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { pathToFileURL } from 'node:url';
import { JSON_ENCODING, sha256 as digest, canonicalDirectory, admitFiles, readAdmitted, hashAdmitted, fileSHA256,
  parseJSON, jsonBytes, writeNew as writeExclusive } from '../../shared/artifacts/node/v1/files.mjs';

export const BASE_SOURCE = 'code/embeddings/scripts/prepare-fresh-decoder-replication.mjs';
export const FILES_SOURCE = 'code/shared/artifacts/node/v1/files.mjs';
export const PARENT_IDENTITY = 'output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b';
export const PARENT_SHA = 'a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90';
export const require = (condition, message) => { if (!condition) throw new Error(message); };
export const equal = isDeepStrictEqual;
export const sha256 = digest;
export const utc = () => new Date().toISOString();
export const isSha = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
export const within = (root, file) => file === root || (file.startsWith(root + path.sep));
export const sortedUnique = values => [...new Set(values)].sort();
export const manifestBytes = records => Buffer.from(records.map(row => `${row.sha256}  ${row.path}\n`).join(''), 'utf8');
const readBytes = (file, admitted) => readAdmitted(admitted ?? admitFiles([file])[0]);
export const readText = (file, admitted) => new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(readBytes(file, admitted)).replace(/\r\n?/g, '\n');

export function regularDirectory(directory) {
  return canonicalDirectory(directory);
}

/** No file byte is read here. Callers admit a complete matrix before hashing. */
export function admitMatrix(files, root, { singleLink = true } = {}) {
  regularDirectory(root);
  require(equal(files, sortedUnique(files)), 'unique sorted whole matrix required');
  return admitFiles(files, { root, singleLink });
}

export function walkFiles(root) {
  regularDirectory(root);
  const result = [];
  function visit(directory) {
    for (const name of fs.readdirSync(directory)) {
      const file = path.join(directory, name), stat = fs.lstatSync(file);
      if (stat.isDirectory()) visit(file);
      else { require(stat.isFile() || stat.isSymbolicLink(), 'unsupported capsule filesystem entry'); result.push(file); }
    }
  }
  visit(root);
  return result.sort();
}

export function fileSha(file) {
  return fileSHA256(file);
}

export function writeNew(file, content) {
  writeExclusive(file, content);
}

// New JSON bytes have their own identity. This does not claim Python serializer parity.
export function writeJson(file, value) {
  writeNew(file, jsonBytes(value, { sortKeys: false, ensureAscii: true }));
}

export function readJson(file, { strictLinks = true, admitted } = {}) {
  require(path.extname(file) === '.json', 'JSON metadata role only');
  const row = admitted ?? admitMatrix([file], path.dirname(file), { singleLink: strictLinks })[0];
  return parseJSON(readText(file, row), { duplicates: 'last', integers: 'number', allowBOM: false });
}

export function relativeName(name) {
  require(typeof name === 'string' && name.length > 0 && !/[\\\s]/u.test(name), 'safe relative source name');
  require(!path.posix.isAbsolute(name) && !name.split('/').some(part => part === '.' || part === '..' || !part)
    && path.posix.normalize(name) === name, 'normalized relative source name');
  require(name === 'Makefile' || ['.h', '.cpp', '.py', '.mjs', '.sh', '.md', '.conf', '.lock'].includes(path.posix.extname(name)),
    'source-only suffix required');
  return name;
}

function frozen(value) {
  if (value && typeof value === 'object') { for (const child of Object.values(value)) frozen(child); Object.freeze(value); }
  return value;
}

function clonePolicy(value) {
  if (Array.isArray(value)) return value.map(clonePolicy);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, clonePolicy(child)]));
  return value;
}

/** Policy is explicit and immutable; adapters never mutate another module's globals. */
export function createPreparation(configuration, hooks = {}) {
  const config = frozen(clonePolicy(configuration));
  const hash = hooks.fileSha ?? ((file, admitted) => admitted ? hashAdmitted(admitted) : fileSha(file));
  const { protocol, card } = config;
  const sources = sortedUnique([...config.requiredSources, FILES_SOURCE, BASE_SOURCE, config.script]);
  const read = file => readJson(file, { strictLinks: config.strictMetadata ?? true });
  const now = hooks.utc ?? utc;

  function checkPlan(plan) {
    require(plan && typeof plan === 'object', 'recipe object required');
    for (const [key, value] of Object.entries(config.plan)) require(equal(plan[key], value), 'fixed recipe: ' + key);
    if (config.planCardBinding) require(plan.human_card_sha256 === config.cardSha && isSha(plan.source_fingerprint),
      'plan frozen card and compiled source identity');
  }

  function sourceAdmission(repo, names) {
    require(Array.isArray(names) && names.length && equal(names, sortedUnique(names)), 'unique sorted explicit source list');
    require(sources.every(name => names.includes(name)), 'required maintained source closure missing');
    return admitMatrix(names.map(name => path.join(repo, relativeName(name))), repo,
      { singleLink: config.singleLinkSources ?? false }).map((row, i) => ({ ...row, name: names[i] }));
  }

  function recordsFor(admitted, cardSha) {
    const rows = admitted.map(row => ({ path: row.name, bytes: row.bytes, sha256: hash(row.path, row) }));
    require(rows.find(row => row.path === card)?.sha256 === cardSha, 'frozen human card differs');
    return rows;
  }

  function sourceRecords(repo, names, cardSha) { return recordsFor(sourceAdmission(repo, names), cardSha); }

  function copySources(repo, target, records) {
    const admitted = sourceAdmission(repo, records.map(row => row.path));
    const source = path.join(target, 'source'); fs.mkdirSync(source);
    for (const [i, row] of records.entries()) {
      const dest = path.join(source, row.path); fs.mkdirSync(path.dirname(dest), { recursive: true });
      require(admitted[i].bytes === row.bytes, 'source changed before capture');
      writeNew(dest, readAdmitted(admitted[i]));
      require(fs.statSync(dest).size === row.bytes && hash(dest) === row.sha256, 'source copy mutation');
    }
    const manifest = manifestBytes(records), identity = sha256(manifest);
    writeNew(path.join(target, 'source-inputs.sha256'), manifest);
    writeNew(path.join(target, 'source-fingerprint.txt'), identity + '\n');
    writeJson(path.join(target, 'source-manifest.json'), { protocol, algorithm: 'sha256-source-manifest-v1',
      human_card: card, source_fingerprint: identity, sources: records,
      preparation_implementation: config.script, historical_source_identity_reused: false, json_encoding: JSON_ENCODING });
    return identity;
  }

  function sourceState(repo, target, cardSha) {
    const sidecars = new Map(admitMatrix(['source-fingerprint.txt', 'source-inputs.sha256', 'source-manifest.json'].map(name => path.join(target, name)),
      target, { singleLink: config.strictMetadata ?? true }).map(row => [row.path, row]));
    const manifestPath = path.join(target, 'source-manifest.json'), metadata = readJson(manifestPath, { admitted: sidecars.get(manifestPath) });
    require(metadata.protocol === protocol && metadata.human_card === card && metadata.preparation_implementation === config.script
      && metadata.historical_source_identity_reused === false && metadata.json_encoding === JSON_ENCODING, 'source metadata protocol/implementation');
    const records = metadata.sources, names = records.map(row => row.path), source = path.join(target, 'source');
    // Both whole matrices, including copied extras, are admitted before either is hashed.
    const live = sourceAdmission(repo, names);
    require(equal(walkFiles(source).map(file => path.relative(source, file)).sort(), names), 'closed captured source matrix');
    const captured = sourceAdmission(source, names);
    require(equal(recordsFor(live, cardSha), records), 'current sources changed');
    records.forEach((row, i) => require(row.bytes === captured[i].bytes && row.sha256 === hash(captured[i].path, captured[i]), 'captured source changed'));
    const manifest = manifestBytes(records), identity = sha256(manifest);
    const inputsPath = path.join(target, 'source-inputs.sha256'), fingerprintPath = path.join(target, 'source-fingerprint.txt');
    require(readBytes(inputsPath, sidecars.get(inputsPath)).equals(manifest), 'canonical source manifest changed');
    require(metadata.source_fingerprint === identity && readText(fingerprintPath, sidecars.get(fingerprintPath)).trim() === identity,
      'source aggregate identity');
    return { metadata, identity };
  }

  function directory(repo, target, admission = false) {
    regularDirectory(target);
    const parent = path.join(repo, config.outputRoot, ...(admission ? ['admission'] : []));
    require(path.dirname(target) === parent && path.basename(target).startsWith(admission ? 'admission-' : config.prefix),
      'new .temp protocol directory bounds');
    return target;
  }

  function binaryRecord(binary) {
    require(typeof binary === 'string' && path.isAbsolute(binary) && path.normalize(binary) === binary &&
      within('/opt/mityane-geometrics/embeddings/build', binary) && path.basename(binary) === config.binary,
      'named container binary path required');
    const [row] = admitMatrix([binary], '/opt/mityane-geometrics/embeddings/build', { singleLink: false });
    return { path: binary, bytes: row.bytes, sha256: hash(binary, row) };
  }

  function admissionPaths(target, sealed = false) {
    const metadata = read(path.join(target, 'source-manifest.json'));
    const expected = metadata.sources.map(row => 'source/' + relativeName(row.path));
    expected.push('source-inputs.sha256', 'source-fingerprint.txt', 'source-manifest.json', 'admission-plan.json', 'build-and-tests.log');
    if (sealed) expected.push('source-preserved-after.json', 'passed.json');
    const files = walkFiles(target);
    require(equal(files.map(file => path.relative(target, file)).sort(), expected.sort()), 'closed admission source/metadata/log matrix');
    admitMatrix(files, target, { singleLink: config.strictMetadata ?? true });
    for (const file of files) require(!['.pt', '.bin', '.npz'].includes(path.extname(file)), 'no model/tensor admission copies');
    return files;
  }

  function freezeAdmission(repo, target, names, cardSha) {
    require(fs.readdirSync(target).length === 0, 'new admission must be empty');
    const identity = copySources(repo, target, sourceRecords(repo, names, cardSha));
    writeJson(path.join(target, 'admission-plan.json'), { protocol, created_utc: now(), human_card: card,
      human_card_sha256: cardSha, source_fingerprint: identity, quality_generated: false,
      historical_quality_inputs: [], historical_payload_roles: 0,
      scope: 'engineering admission only; no declared fresh quality generation' });
    sourceState(repo, target, cardSha);
    return { source_fingerprint: identity };
  }

  function admissionRecord(repo, target, cardSha, identity, binary, sealed = false) {
    const admitted = new Map(admitMatrix(admissionPaths(target, sealed), target,
      { singleLink: config.strictMetadata ?? true }).map(row => [row.path, row]));
    require(sourceState(repo, target, cardSha).identity === identity, 'actual compiled/source identity');
    const planPath = path.join(target, 'admission-plan.json'), plan = readJson(planPath, { admitted: admitted.get(planPath) });
    require(plan.source_fingerprint === identity && plan.human_card_sha256 === cardSha && plan.quality_generated === false
      && equal(plan.historical_quality_inputs, []), 'source-only engineering admission plan');
    const log = path.join(target, 'build-and-tests.log'), text = readText(log, admitted.get(log));
    for (const marker of [...config.markers, identity]) require(text.includes(marker), 'actual source-bound admission marker: ' + marker);
    if (config.sdkGate) require(!text.includes('/embedding/.external/libtorch') && !text.includes('not found'), 'internal pinned SDK resolution');
    return { protocol, status: 'passed', ...(config.admissionTimestamp ? { created_utc: now() } : {}),
      source_fingerprint: identity, human_card_sha256: cardSha, source_preserved: true, log_sha256: hash(log, admitted.get(log)),
      compiled_binary: binaryRecord(binary), quality_generated: false, historical_payload_roles: 0,
      ...(config.headFits !== undefined ? { head_fits: config.headFits } : {}), gates: config.gates };
  }

  function compareAdmission(passed, current) {
    const { created_utc: before, ...stablePassed } = passed, { created_utc: after, ...stableCurrent } = current;
    require(equal(stablePassed, stableCurrent), 'passed admission evidence mismatch');
  }

  function copyAdmission(repo, admission, target, args) {
    const current = admissionRecord(repo, admission, args.card_sha256, args.compiled_source_id, args.binary, true);
    const passed = read(path.join(admission, 'passed.json')); compareAdmission(passed, current);
    const copied = path.join(target, 'admission'); fs.mkdirSync(copied);
    const admitted = admitMatrix(admissionPaths(admission, true), admission, { singleLink: config.strictMetadata ?? true });
    for (const row of admitted) {
      const dest = path.join(copied, path.relative(admission, row.path)); fs.mkdirSync(path.dirname(dest), { recursive: true });
      const bytes = readAdmitted(row); writeNew(dest, bytes); require(hash(dest) === sha256(bytes), 'admission copy differs');
    }
    require(hash(path.join(copied, 'build-and-tests.log')) === passed.log_sha256 && equal(read(path.join(copied, 'passed.json')), passed),
      'admission changed between verification and capture');
    return passed;
  }

  function readerPreflight(repo, args) {
    require(args.reader_source && args.reader_fixtures && args.reader_seal, 'explicit independent reader/source fixtures/seal required');
    const originals = [args.reader_source, args.reader_fixtures, args.reader_seal];
    const rows = admitMatrix(sortedUnique(originals), path.join(repo, config.outputRoot, 'audit-tools'));
    const indexed = new Map(rows.map(row => [row.path, row]));
    require(new Set(originals).size === 3, 'distinct reader evidence roles required');
    require(path.extname(args.reader_source) === '.py',
      'replacement independent reader identity has not been accepted; historical Python source-only seal cannot certify Node/C++ bytes');
    const seal = readJson(args.reader_seal, { admitted: indexed.get(args.reader_seal) });
    require(seal.status === 'passed' && seal.source_sha256 === hash(args.reader_source, indexed.get(args.reader_source))
      && seal.fixtures_sha256 === hash(args.reader_fixtures, indexed.get(args.reader_fixtures)),
      'presealed independent source-only reader');
    if (config.readerBlockedFlags) {
      const text = readText(args.reader_source, indexed.get(args.reader_source)),
        fixtures = readJson(args.reader_fixtures, { admitted: indexed.get(args.reader_fixtures) });
      require(/^REVIEWED_SCHEMA = False$/m.test(text) && /^MEASURED_IMPLEMENTATION = False$/m.test(text),
        'reviewed source-only reader requires both execution flags FALSE');
      require(fixtures.status === 'passed' && fixtures.measured_execution_enabled === false,
        'passed source-only fixtures remain prospectively blocked');
    }
    return { paths: originals, rows };
  }

  function bindReader(repo, target, args, preflight = readerPreflight(repo, args)) {
    const copied = path.join(target, 'independent-reader'); fs.mkdirSync(copied);
    const indexed = new Map(preflight.rows.map(row => [row.path, row]));
    return preflight.paths.map((file, i) => {
      const dest = path.join(copied, ['source.py', 'source-fixtures.json', 'reader-seal.json'][i]);
      const bytes = readAdmitted(indexed.get(file)); writeNew(dest, bytes);
      return { path: file, captured: dest, bytes: bytes.length, sha256: sha256(bytes) };
    });
  }

  function historicalParentCapsule(args) {
    require(args.historicalParentCapsule,
      'explicit --historical-parent-capsule directory mapping required; old output paths are evidence identities');
    return regularDirectory(args.historicalParentCapsule);
  }

  function inputRecords(args, rows) {
    const parent = historicalParentCapsule(args), paths = sortedUnique(rows.flatMap(row => row.slice(config.parent.roleOffset)));
    require(paths.length === config.parent.roles, 'exact closed historical role count');
    const inventory = path.join(parent, 'artifact-integrity.json');
    const admitted = admitMatrix(sortedUnique([inventory, ...paths]), parent);
    const indexedAdmission = new Map(admitted.map(row => [row.path, row]));
    require(hash(inventory, indexedAdmission.get(inventory)) === PARENT_SHA, 'pinned historical parent inventory');
    const metadata = readJson(inventory, { admitted: indexedAdmission.get(inventory) });
    require(metadata.protocol === 'fresh-decoder-replication-v1' && metadata.file_count === 948 && metadata.files.length === 948,
      'parent inventory shape');
    const index = new Map(metadata.files.map(row => [row.path, row])); require(index.size === 948, 'duplicate parent inventory role');
    const records = [];
    for (const row of admitted) {
      if (row.path === inventory) continue;
      const pinned = index.get(path.relative(parent, row.path));
      require(pinned && pinned.bytes === row.bytes && isSha(pinned.sha256), 'whole pinned role/size/hash association');
      records.push({ path: row.path, bytes: row.bytes, sha256: pinned.sha256 });
    }
    // No historical role payload is hashed until the entire metadata association passed.
    for (const row of records) require(hash(row.path, indexedAdmission.get(row.path)) === row.sha256, 'pinned parent role bytes');
    return { records, inventory, inventoryAdmission: indexedAdmission.get(inventory),
      mapping: { historical_identity: PARENT_IDENTITY, mapped_capsule: parent, inventory_sha256: PARENT_SHA } };
  }

  function freezeRun(repo, target, args) {
    require(equal(fs.readdirSync(target).sort(), ['recipe-plan.json']), 'exclusive new plan-only capsule');
    const recipePath = path.join(target, 'recipe-plan.json'), recipeAdmission = admitMatrix([recipePath], target)[0];
    const plan = readJson(recipePath, { admitted: recipeAdmission }); checkPlan(plan);
    if (config.planCardBinding) require(plan.source_fingerprint === args.compiled_source_id, 'recipe compiled-source binding');
    const records = sourceRecords(repo, args.sources, args.card_sha256);
    require(sha256(manifestBytes(records)) === args.compiled_source_id, 'run compiled/source/card identity');
    // Reader metadata is checked before any new target write or historical role access.
    const originals = config.reader ? readerPreflight(repo, args) : undefined;
    const admission = directory(repo, args.admission, true);
    const passed = copyAdmission(repo, admission, target, args);
    require(copySources(repo, target, records) === args.compiled_source_id, 'source capture identity');
    const reader = config.reader ? bindReader(repo, target, args, originals) : undefined;
    let roles, mapping;
    if (config.parent) {
      require(within(repo, historicalParentCapsule(args)), 'mapped historical parent capsule remains inside this repository');
      const rows = config.parent.rows(historicalParentCapsule(args));
      const inputs = inputRecords(args, rows); roles = inputs.records; mapping = inputs.mapping;
      writeNew(path.join(target, 'parent-inventory.json'), readAdmitted(inputs.inventoryAdmission));
      writeNew(path.join(target, 'instances.tsv'), config.parent.header.join('\t') + '\n' + rows.map(row => row.join('\t') + '\n').join(''));
      writeNew(path.join(target, 'input-checksums.sha256'), roles.map(row => `${row.sha256}  ${row.path}\n`).join(''));
      writeJson(path.join(target, 'input-role-plan.json'), { protocol, roles, [config.parent.countKey]: config.parent.roles,
        parent_inventory_sha256: PARENT_SHA, [config.parent.heldoutKey]: 0, prior_mixed_audit_payload_read: false, parent_path_mapping: mapping });
    } else writeJson(path.join(target, 'input-role-plan.json'), { protocol, historical_payload_roles: 0, prior_quality_input_roles: [],
      fresh_quality_inputs: config.freshInputDescription ?? 'generated only after frozen source/admission/reader launch' });
    const copied = path.join(target, 'admission');
    const argv = [args.binary, ...(roles ? ['--instances', path.join(target, 'instances.tsv'), '--checksums', path.join(target, 'input-checksums.sha256'),
      '--input-root', repo] : []), '--output', path.join(target, 'results'), '--admission-log', path.join(copied, 'build-and-tests.log'),
      '--admission-sha256', passed.log_sha256, ...(config.launchCard ? ['--card', path.join(target, 'source', card), '--card-sha256', args.card_sha256] : [])];
    writeJson(path.join(target, 'launch-plan.json'), { protocol, created_utc: now(), status: config.launchStatus ?? 'frozen_before_quality_generation',
      source_fingerprint: args.compiled_source_id, human_card_sha256: args.card_sha256, json_encoding: JSON_ENCODING,
      ...(config.baseProtocol ? { human_card: card, recipe: plan } : {}),
      compiled_binary: passed.compiled_binary, admission_passed_sha256: hash(path.join(copied, 'passed.json')),
      admission_log_sha256: passed.log_sha256, ...(reader ? { reader } : {}), ...(roles ? { roles, parent_path_mapping: mapping,
        instances_sha256: hash(path.join(target, 'instances.tsv')), checksums_sha256: hash(path.join(target, 'input-checksums.sha256')) } : { historical_payload_roles: 0 }),
      [config.baseProtocol ? 'recipe_plan_sha256' : 'recipe_sha256']: hash(recipePath, recipeAdmission), argv, ...config.launchFields });
    sourceState(repo, target, args.card_sha256);
    return { protocol, source_fingerprint: args.compiled_source_id, ...(config.parent ? { [config.parent.countKey]: config.parent.roles } : { historical_payload_roles: 0 }) };
  }

  function checkComplete(complete) {
    require(complete.protocol === protocol && complete.status === 'complete', 'completed measurement protocol/status');
    for (const [key, value] of Object.entries(config.complete ?? {})) require(equal(complete[key], value), 'completed measurement: ' + key);
    for (const [key, bounds] of Object.entries(config.completeRanges ?? {})) require(typeof complete[key] === 'number'
      && complete[key] >= bounds[0] && complete[key] <= bounds[1], 'completed measurement range: ' + key);
    if (config.pairedHeads) require(complete.individual_heads === 2 * complete.head_pipelines, 'completed paired heads');
  }

  function finalize(repo, target, args) {
    require(sourceState(repo, target, args.card_sha256).identity === args.compiled_source_id, 'live/captured source preserved');
    const launch = read(path.join(target, 'launch-plan.json'));
    require(launch.protocol === protocol && launch.source_fingerprint === args.compiled_source_id && launch.human_card_sha256 === args.card_sha256,
      'launch source/card boundary');
    require(equal(binaryRecord(launch.argv[0]), launch.compiled_binary), 'tested binary bytes preserved');
    require(hash(path.join(target, 'recipe-plan.json')) === launch[config.baseProtocol ? 'recipe_plan_sha256' : 'recipe_sha256'], 'recipe preserved');
    checkPlan(read(path.join(target, 'recipe-plan.json')));
    require(hash(path.join(target, 'admission/passed.json')) === launch.admission_passed_sha256 &&
      hash(path.join(target, 'admission/build-and-tests.log')) === launch.admission_log_sha256, 'admission preserved');
    if (config.baseProtocol) {
      admissionPaths(path.join(target, 'admission'), true);
      require(equal(read(path.join(target, 'admission/passed.json')).compiled_binary, launch.compiled_binary), 'copied tested binary identity');
      require(read(path.join(target, 'results/report.json')).protocol === protocol, 'measurement report protocol');
    }
    if (config.parent) {
      require(equal(launch.parent_path_mapping, { historical_identity: PARENT_IDENTITY, mapped_capsule: historicalParentCapsule(args), inventory_sha256: PARENT_SHA }),
        'explicit historical parent mapping preserved');
      for (const [name, key] of [['instances.tsv', 'instances_sha256'], ['input-checksums.sha256', 'checksums_sha256']])
        require(hash(path.join(target, name)) === launch[key], 'run role metadata preserved');
      const admitted = new Map(admitMatrix(launch.roles.map(row => row.path).sort(), historicalParentCapsule(args)).map(row => [row.path, row]));
      for (const row of launch.roles) require(admitted.get(row.path).bytes === row.bytes
        && hash(row.path, admitted.get(row.path)) === row.sha256, 'historical input preserved');
    }
    if (config.reader) {
      const readerFiles = launch.reader.flatMap(row => [row.path, row.captured]);
      const admitted = new Map(admitMatrix(readerFiles.sort(), repo).map(row => [row.path, row]));
      for (const row of launch.reader) require(hash(row.path, admitted.get(row.path)) === row.sha256
        && hash(row.captured, admitted.get(row.captured)) === row.sha256, 'reader evidence preserved');
    }
    checkComplete(read(path.join(target, 'results/complete.json')));
    writeJson(path.join(target, 'source-preserved-after.json'), { protocol, source_fingerprint: args.compiled_source_id, source_preserved: true,
      ...(config.parent ? { inputs_preserved: true } : { historical_payload_roles: 0 }), ...(config.reader ? { reader_preserved: true } : {}),
      ...(config.baseProtocol ? { copied_admission_preserved: true, recipe_preserved: true } : {}), created_utc: now() });
    const files = admitMatrix(walkFiles(target), target).map(row => ({ path: path.relative(target, row.path), bytes: row.bytes, sha256: hash(row.path, row) }));
    const total = files.reduce((sum, row) => sum + row.bytes, 0);
    require(Number.isSafeInteger(total), 'inventory total exact integer range');
    writeJson(path.join(target, 'artifact-integrity.json'), { schema_version: 1, protocol, json_encoding: JSON_ENCODING, created_utc: now(), inventory_excludes_itself: true,
      checksum_algorithm: 'sha256-file-bytes', file_count: files.length, total_bytes: total, files });
    return { protocol, status: 'complete', source_fingerprint: args.compiled_source_id,
      inventory_sha256: hash(path.join(target, 'artifact-integrity.json')), file_count: files.length,
      ...(config.baseProtocol ? { total_bytes: total } : {}) };
  }

  function run(args) {
    require(fs.existsSync('/.dockerenv') && process.platform === 'linux', 'existing managed Linux container required');
    require(['role-plan', 'freeze-admission', 'admit', 'freeze-run', 'finalize'].includes(args.mode), 'supported preparation mode required; source tests use node --test');
    if (args.mode === 'role-plan') { require(config.rolePlan, 'role-plan mode is declared only by the fresh decoder protocol'); return { protocol, ...config.rolePlan }; }
    require(args.target && isSha(args.card_sha256) && (!config.cardSha || args.card_sha256 === config.cardSha), 'explicit target/frozen card SHA required');
    const repo = fs.realpathSync(args.repo_root ?? '.'); regularDirectory(repo);
    const target = directory(repo, args.target, ['freeze-admission', 'admit'].includes(args.mode));
    if (args.mode !== 'freeze-admission') require(isSha(args.compiled_source_id), 'compiled source SHA required');
    if (args.mode === 'freeze-admission') return freezeAdmission(repo, target, args.sources, args.card_sha256);
    if (args.mode === 'admit') {
      const record = admissionRecord(repo, target, args.card_sha256, args.compiled_source_id, args.binary);
      writeJson(path.join(target, 'source-preserved-after.json'), { protocol, source_fingerprint: args.compiled_source_id,
        source_preserved: true, ...(config.baseProtocol ? { created_utc: now() } : {}) });
      writeJson(path.join(target, 'passed.json'), record); return record;
    }
    if (args.mode === 'freeze-run') return freezeRun(repo, target, args);
    return finalize(repo, target, args);
  }

  return Object.freeze({ config, requiredSources: Object.freeze(sources), checkPlan, sourceAdmission, sourceRecords, sourceState, copySources,
    directory, binaryRecord, admissionPaths, admissionRecord, freezeAdmission, readerPreflight, bindReader, inputRecords, freezeRun, checkComplete, finalize, run });
}

export function parseArgs(argv) {
  const args = { mode: argv[0], repo_root: '.' }, scalar = new Set(['repo-root', 'target', 'card-sha256', 'compiled-source-id',
    'admission', 'binary', 'reader-source', 'reader-fixtures', 'reader-seal', 'historical-parent-capsule']);
  const seen = new Set();
  for (let i = 1; i < argv.length; i++) {
    const key = argv[i]; require(key.startsWith('--'), 'named argument required');
    const name = key === '--historical-parent-capsule' ? 'historicalParentCapsule' : key.slice(2).replaceAll('-', '_');
    require(!seen.has(name), 'duplicate CLI argument'); seen.add(name);
    if (key === '--sources') {
      const values = []; while (i + 1 < argv.length && !argv[i + 1].startsWith('--')) values.push(argv[++i]);
      require(values.length > 0, '--sources requires values'); args.sources = values;
    } else {
      require(scalar.has(key.slice(2)) && i + 1 < argv.length && !argv[i + 1].startsWith('--'), 'unknown or missing CLI argument: ' + key);
      args[name] = argv[++i];
    }
  }
  return args;
}

export function main(preparation, argv = process.argv.slice(2)) {
  try { process.stdout.write(JSON.stringify(preparation.run(parseArgs(argv)), null, 2) + '\n'); }
  catch (error) { process.stderr.write(error.message + '\n'); process.exitCode = 1; }
}
export const isMain = url => process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === url;

export const freshDecoderConfig = {
  protocol: 'fresh-decoder-replication-v1', card: 'code/embeddings/evaluation/cards/fresh_decoder_replication_v1.md',
  script: BASE_SOURCE, outputRoot: '.temp/embeddings/runs/rpb-fresh-decoder-replication', prefix: 'fresh-decoder-replication-',
  binary: 'embedding_fresh_decoder_replication', baseProtocol: true, strictMetadata: false, admissionTimestamp: true, headFits: 0,
  markers: ['Fresh decoder replication CUDA admission passed', 'V7 decoder calibration CUDA admission passed', 'Fixed feature readout tests passed'],
  requiredSources: ['code/embeddings/evaluation/cards/fresh_decoder_replication_v1.md', 'Makefile', 'dependencies.lock',
    'code/embeddings/evaluation/src/fresh_decoder_replication_main.cpp', 'code/embeddings/scripts/check-fresh-decoder-replication.sh',
    'code/embeddings/scripts/evaluate-fresh-decoder-replication.sh', 'code/embeddings/scripts/prepare-fresh-decoder-replication.mjs',
    'code/shared/representation/include/embedding/shared/fixed_feature_readouts.h', 'code/shared/representation/src/fixed_feature_readouts.cpp',
    'code/shared/representation/tests/fixed_feature_readouts_test.cpp',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/include/embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/src/frozen_decoder_calibration.cpp',
    'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v7/tests/frozen_decoder_calibration_test.cpp'],
  plan: { protocol: 'fresh-decoder-replication-v1', master_seeds: [9109, 10210, 11311, 12412, 13513], tags: ['RPB-v4', 'RPB-v7'],
    encoder_updates: 512, decoder_updates: 128, train_pairs: 128, validation_pairs: 64, test_pairs: 0,
    prior_quality_input_roles: [], quality_generated: false, testing_accessed: false, stress_accessed: false },
  rolePlan: { master_seeds: [9109, 10210, 11311, 12412, 13513], historical_payload_roles: 0, prior_quality_input_roles: [],
    quality_generated: false, testing_accessed: false, stress_accessed: false },
  complete: { testing_accessed: false, stress_accessed: false },
  launchFields: { testing_accessed: false, stress_accessed: false, quality_generated: false },
  freshInputDescription: 'generated only after launch plan inside measured main',
  gates: ['actual CUDA paired encoder/frozen-decoder lifecycle contract on separate engineering fixtures',
    'declared quality recipe is encoder512 plus fresh decoder128; engineering admission does not train those quality cohorts',
    'frozen encoder/scaler/native outputs; original query and direct/split fresh decoder AdamW parity',
    'legacy v7 typed decoder adapter preserved', 'fresh initialization/data/streams, immutable snapshots and bounded counters'],
};

export const freshDecoder = createPreparation(freshDecoderConfig);
if (isMain(import.meta.url)) main(freshDecoder);

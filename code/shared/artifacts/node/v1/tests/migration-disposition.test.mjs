// Source and policy inspection through the managed-container Node test target.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../../..');
const read = name => JSON.parse(fs.readFileSync(path.join(root, 'doc', name), 'utf8'));
const baseline = read('PYTHON_MIGRATION_INVENTORY_2026-10-18.json');
const disposition = read('PYTHON_MIGRATION_DISPOSITION_2026-10-18.json');
const removal = read('PYTHON_MIGRATION_REMOVAL_2026-10-18.json');
const sourceKey = row => row.path;
const invocationKey = row => `${row.path}:${row.baselineLine ?? row.line}`;
const blockKey = row => `${row.path}:${row.baselineInvocationLine ?? row.invocationLine}`;

function sameClosedKeys(actual, expected, key) {
  const keys = actual.map(key);
  assert.equal(new Set(keys).size, keys.length, 'no duplicate disposition row');
  assert.deepEqual(keys.slice().sort(), expected.map(key).sort());
}

test('every baseline Python source has an explicit disposition and original identity', () => {
  sameClosedKeys(disposition.files, baseline.pythonFiles, sourceKey);
  const original = new Map(baseline.pythonFiles.map(row => [row.path, row]));
  const allowed = new Set(['retain-scientific-python', 'replaced-bookkeeping-reference',
    'replaced-retired-refusal', 'replaced-maintenance-reference']);
  for (const row of disposition.files) {
    assert.ok(allowed.has(row.disposition), row.path);
    assert.equal(row.baselineSha256, original.get(row.path).sha256);
    assert.equal(row.baselinePhysicalLines, original.get(row.path).physicalLines);
    assert.ok(row.rationale.length > 30, row.path);
    if (row.disposition === 'retain-scientific-python') {
      assert.ok(row.runtimeDependencies.includes('managed-container-python3'), row.path);
    } else {
      assert.ok(row.replacement.endsWith('.mjs'), row.path);
      assert.ok(fs.statSync(path.join(root, row.replacement)).isFile(), row.replacement);
      assert.deepEqual(row.runtimeDependencies, []);
    }
  }
});

test('obsolete working copies are absent and their retired source identities remain preserved', () => {
  const replaced = disposition.files.filter(row => row.disposition !== 'retain-scientific-python');
  sameClosedKeys(removal.files, replaced, sourceKey);
  for (const row of removal.files) {
    const policy = replaced.find(item => item.path === row.path);
    assert.equal(row.baselineSha256, policy.baselineSha256, row.path);
    assert.equal(row.replacement, policy.replacement, row.path);
    assert.equal(fs.existsSync(path.join(root, row.path)), false, `obsolete working copy: ${row.path}`);
    assert.ok(row.preservation.path.startsWith('.archive/'), row.path);
    assert.equal(row.preservation.sha256, row.removedSha256, row.path);
    assert.equal(row.preservation.bytes, row.removedBytes, row.path);
    const preserved = fs.readFileSync(path.join(root, row.preservation.path));
    assert.equal(preserved.length, row.removedBytes, row.path);
    assert.equal(crypto.createHash('sha256').update(preserved).digest('hex'), row.removedSha256, row.path);
    assert.equal(policy.physicalStatus, 'removed', row.path);
  }

  const actual = [];
  const excluded = new Set(['code/embeddings/output', 'code/embeddings/.build', 'code/embeddings/.external', 'web/dashboard/public']);
  function visit(relative) {
    if (excluded.has(relative)) return;
    for (const entry of fs.readdirSync(path.join(root, relative), { withFileTypes: true })) {
      const name = `${relative}/${entry.name}`;
      assert.ok(!entry.isSymbolicLink(), `source scan redirect: ${name}`);
      if (entry.isDirectory()) visit(name);
      else if (entry.isFile() && entry.name.endsWith('.py')) actual.push(name);
    }
  }
  visit('code'); visit('web');
  const retained = disposition.files.filter(row => row.disposition === 'retain-scientific-python');
  assert.deepEqual(actual.sort(), retained.map(row => row.path).sort(), 'only declared scientific Python remains in maintained source');
  assert.ok(retained.every(row => row.physicalStatus === 'retained'));
});

test('every original caller, invocation and embedded block is accounted for', () => {
  sameClosedKeys(disposition.callers, baseline.pythonCallers, sourceKey);
  sameClosedKeys(disposition.invocations, baseline.invocations, invocationKey);
  sameClosedKeys(disposition.inlineBlocks, baseline.heredocBlocks, blockKey);
  const calls = new Set(disposition.invocations.map(invocationKey));
  for (const caller of disposition.callers) {
    assert.ok(caller.invocationKeys.length > 0);
    for (const key of caller.invocationKeys) assert.ok(calls.has(key), key);
  }
  assert.ok(disposition.inlineBlocks.every(row => row.disposition === 'replace-with-node-metadata'));
});

test('mixed capacity bind and pinned scientific paths remain explicit', () => {
  const capacity = disposition.files.find(row => row.path.endsWith('/configurable_capacity_v1/capsule.py'));
  assert.equal(capacity.disposition, 'retain-scientific-python');
  assert.deepEqual(capacity.retainedModes, ['bind']);
  assert.deepEqual(capacity.replacedModes, ['freeze', 'verify-admission', 'finish']);
  assert.equal(disposition.runtimeDependencies.explicitPathMapping.frozenCodecSha256,
    '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d');
  assert.equal(disposition.execution.researchPayloadReads, false);
  assert.equal(disposition.policy.researchPermission, 'Migration does not resume paused experiments, fits, training, held-out access or V14 metadata work.');
});

function sourceFiles() {
  const names = ['Makefile', 'setup.sh', 'container.ps1'];
  const excluded = new Set(['code/embeddings/output', 'code/embeddings/.build', 'code/embeddings/.external', 'web/dashboard/public']);
  function visit(relative) {
    if (excluded.has(relative)) return;
    for (const entry of fs.readdirSync(path.join(root, relative), { withFileTypes: true })) {
      const name = `${relative}/${entry.name}`;
      assert.ok(!entry.isSymbolicLink(), `source scan redirect: ${name}`);
      if (entry.isDirectory()) visit(name);
      else if (entry.isFile() && (entry.name === 'Makefile' || /\.(?:sh|mk|ps1|mjs)$/.test(entry.name))) names.push(name);
    }
  }
  visit('code'); visit('web'); return names;
}

function exactScientificCall(filename, source, line, command) {
  const allowed = new Map(disposition.files.filter(row => row.disposition === 'retain-scientific-python').map(row => [row.path, row]));
  let tail = command.trim();
  while (/^-[BO]+(?:\s|$)/.test(tail)) tail = tail.replace(/^-[BO]+\s*/, '');
  assert.ok(!/^-(?:c|m)(?:\s|$)/.test(tail), `${filename}: inline/module Python remains: ${line}`);
  const match = tail.match(/^(?:"([^"]+)"|'([^']+)'|([^\s]+))(?:\s|$)/);
  assert.ok(match, `${filename}: explicit Python target required`);
  let target = match[1] ?? match[2] ?? match[3];
  if (target === '$$fixture') {
    assert.equal(filename, 'code/scripts/python-migration.mk', 'only the closed scientific fixture loop may select a variable target');
    const loop = source.match(/for fixture in\s+\\([\s\S]*?); do/);
    assert.ok(loop, 'closed scientific fixture loop declaration');
    const items = loop[1].replaceAll('\\', ' ').trim().split(/\s+/);
    assert.ok(items.length > 0 && items.every(name => allowed.has(name)), 'every loop member is an explicitly retained scientific file');
    return;
  }
  const bindings = new Map([...source.matchAll(/^([A-Za-z_][A-Za-z0-9_]*)\s*:?=\s*['"]?(code\/[^\s'";]+)['"]?\s*$/gm)].map(row => [row[1], row[2]]));
  target = target.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$\(([A-Za-z_][A-Za-z0-9_]*)\)|\$([A-Za-z_][A-Za-z0-9_]*)/g,
    (_match, braced, make, plain) => bindings.get(braced ?? make ?? plain) ?? '<unresolved>');
  target = target.replace(/^\/mityane\//, '');
  assert.ok(allowed.has(target), `${filename}: Python call outside scientific disposition: ${target}`);
  const policy = allowed.get(target);
  if (policy.retainedModes) {
    const remaining = tail.slice(match[0].length).trim().split(/\s+/)[0];
    assert.ok(policy.retainedModes.includes(remaining), `${filename}: Python mode outside retained scientific scope: ${remaining}`);
  }
}

test('current source calls Python only for explicitly retained scientific tools', () => {
  const calls = [];
  for (const filename of sourceFiles()) {
    const source = fs.readFileSync(path.join(root, filename), 'utf8');
    if (filename.endsWith('.mjs')) {
      // Node metadata must not silently delegate back to a Python executable.
      assert.ok(!/\b(?:spawnSync|spawn|execFileSync|execFile)\s*\(\s*['"](?:[^'"\n]*\/)?python(?:3(?:\.\d+)?)?['"]/.test(source), filename);
      continue;
    }
    const lines = source.split(/\r?\n/);
    assert.ok(!/^#!.*\bpython/.test(lines[0] ?? ''), `non-Python file has Python shebang: ${filename}`);
    lines.forEach((line, index) => {
      if (/^\s*#/.test(line)) return;
      for (const match of line.matchAll(/(?:^|[\s;])(?:[^\s;"']*\/)?python(?:3(?:\.\d+)?)?\s+([^;\n]*)/g)) {
        assert.ok(!match[1].includes('<<'), `${filename}:${index + 1}: Python heredoc remains`);
        exactScientificCall(filename, source, index + 1, match[1]); calls.push(`${filename}:${index + 1}`);
      }
    });
  }
  assert.ok(calls.length > 0, 'scanner observes the deliberately retained scientific calls');
});

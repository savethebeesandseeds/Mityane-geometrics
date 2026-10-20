// Artificial receipt fixtures only; not saved-data or protocol parity evidence.
// Source authored during Docker recovery; validation uses the managed container.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createCapsule, MASTERS, PROTOCOL } from './capsule.mjs';
import { sha, implementationReceipt } from '../node/v1/capsule_support.mjs';

function fixture(t) {
  assert.equal(fs.existsSync('/.dockerenv'), true, 'run fixtures only in the managed Linux container');
  const base = '/opt/mityane-geometrics/python-migration-fixtures';
  fs.mkdirSync(base, { recursive: true });
  const repositoryRoot = fs.mkdtempSync(path.join(base, 'capacity-receipts-'));
  t.after(() => fs.rmSync(repositoryRoot, { recursive: true, force: true }));
  const root = path.join(repositoryRoot, '.temp/embeddings/runs/rpb-v20-capacity/admission-fixture');
  fs.mkdirSync(root, { recursive: true });
  const write = (relative, value) => {
    const file = path.join(root, relative); fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, typeof value === 'string' ? value : `${JSON.stringify(value)}\n`);
  };
  const capsule = createCapsule({ repositoryRoot, buildDirectory: path.join(repositoryRoot, 'build') });
  for (const name of ['sources.sha256', 'card.md', 'config.conf', 'inputs.sha256']) write(`freeze/${name}`, 'artificial receipt fixture\n');
  const receipt = { protocol: PROTOCOL, source_id: sha(path.join(root, 'freeze/sources.sha256')),
    card_sha256: sha(path.join(root, 'freeze/card.md')), config_sha256: sha(path.join(root, 'freeze/config.conf')) };
  receipt.migration_implementation = implementationReceipt(receipt.source_id, []);
  write('freeze/receipt.json', receipt);
  write('admission.json', { protocol: PROTOCOL, status: 'admitted', quality_encoder_runs: 0 });
  write('runner-engineering/complete.json', { status: 'passed', protocol: PROTOCOL, master: 941000,
    profiles: ['A', 'B', 'C'], updates_each: 2, CPU_encoder_calls: 0 });
  const bindings = { source_fingerprint: receipt.source_id, card_sha: receipt.card_sha256,
    config_sha: receipt.config_sha256, input_manifest_sha: sha(path.join(root, 'freeze/inputs.sha256')) };
  for (const profile of ['A', 'B', 'C']) for (const master of MASTERS) {
    const run = `profiles/${profile}/master-${master}`;
    write(`${run}/report.json`, { fixture: 'opaque counters; no tensor payloads' });
    for (const point of [0, 512]) for (const name of ['checkpoint.pt', 'features.pt',
      'reconstruction-training.pt', 'reconstruction-validation.pt']) write(`${run}/point-${point}/${name}`, 'opaque fixture');
    write(`${run}/complete.json`, { ...bindings, status: 'complete', protocol: PROTOCOL, profile, master, updates: 512,
      report_sha256: sha(path.join(root, run, 'report.json')) });
  }
  write('readouts/report.json', { fixture: 'opaque receipt fixture' });
  write('readouts/complete.json', { status: 'complete', protocol: PROTOCOL, saved_arithmetic_fits_replayed: 81,
    report_sha256: sha(path.join(root, 'readouts/report.json')) });
  write('summary.json', { ...bindings, protocol: PROTOCOL, saved_arithmetic_fits_replayed: 81 });
  write('report.md', 'Artificial receipt fixture; no measured results.\n');
  const read = relative => JSON.parse(fs.readFileSync(path.join(root, relative), 'utf8'));
  return { root, write, read, capsule };
}

test('quality admission requires all profile CUDA engineering and unchanged settings', t => {
  const f = fixture(t); f.capsule.verifyAdmission(f.root);
  f.write('runner-engineering/complete.json', { ...f.read('runner-engineering/complete.json'), profiles: ['A', 'B'] });
  assert.throws(() => f.capsule.verifyAdmission(f.root), /all three profiles/);
  f.write('runner-engineering/complete.json', { ...f.read('runner-engineering/complete.json'), profiles: ['A', 'B', 'C'] });
  f.write('freeze/config.conf', 'changed settings');
  assert.throws(() => f.capsule.verifyAdmission(f.root), /receipt differs/);
});
test('unbound archive reader fails before any observation/report access', t => {
  const f = fixture(t);
  assert.equal(fs.existsSync(path.join(f.root, 'quality-data')), false);
  assert.throws(() => f.capsule.bind(f.root), /no fallback is bound/);
  assert.equal(fs.existsSync(path.join(f.root, 'freeze/information.sha256')), false);
});
for (const [name, relative, patch] of [
  ['incomplete producer', 'profiles/A/master-941101/complete.json', { updates: 511 }],
  ['wrong cohort', 'profiles/B/master-941102/complete.json', { master: 941101 }],
  ['incomplete reader', 'readouts/complete.json', { saved_arithmetic_fits_replayed: 80 }],
  ['wrong publisher input', 'summary.json', { input_manifest_sha: '0'.repeat(64) }],
]) test(`${name} cannot publish completion`, t => {
  const f = fixture(t); f.write(relative, { ...f.read(relative), ...patch });
  assert.throws(() => f.capsule.finish(f.root));
  assert.equal(fs.existsSync(path.join(f.root, 'artifacts.sha256')), false);
  assert.equal(fs.existsSync(path.join(f.root, 'complete.json')), false);
});
test('changed report and linked retained artifact cannot publish completion', t => {
  const f = fixture(t); f.write('readouts/report.json', 'changed report');
  assert.throws(() => f.capsule.finish(f.root), /saved-reader/);
  const g = fixture(t);
  fs.linkSync(path.join(g.root, 'report.md'), path.join(g.root, 'report-alias.md'));
  assert.throws(() => g.capsule.finish(g.root), /single-link/);
  assert.equal(fs.existsSync(path.join(g.root, 'complete.json')), false);
});
test('new receipt source identity is required and completion is immutable', t => {
  const f = fixture(t);
  const legacy = f.read('freeze/receipt.json'); delete legacy.migration_implementation;
  f.write('freeze/receipt.json', legacy);
  assert.throws(() => f.capsule.finish(f.root), /new source identity/);
  const g = fixture(t); g.capsule.finish(g.root);
  const before = fs.readFileSync(path.join(g.root, 'complete.json'));
  const completion = g.read('complete.json');
  assert.equal(completion.saved_head_fits, 81);
  assert.equal(completion.encoder_trajectories, 9);
  assert.equal(completion.artifacts_manifest_sha256, sha(path.join(g.root, 'artifacts.sha256')));
  assert.throws(() => g.capsule.finish(g.root), /finalized again/);
  assert.deepEqual(fs.readFileSync(path.join(g.root, 'complete.json')), before);
});

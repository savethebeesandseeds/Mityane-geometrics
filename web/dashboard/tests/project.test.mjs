// SPDX-License-Identifier: AGPL-3.0-only
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, writeFile, rm, symlink} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {publishFingerprint, readProject, readPublishedProject, sourceFingerprint} from '../scripts/project.mjs';

const digest = value => createHash('sha256').update(value).digest('hex');

async function fixture(t) {
  const temporary = await mkdtemp(join(tmpdir(), 'dashboard-evidence-'));
  t.after(() => rm(temporary, {recursive: true, force: true}));
  const root = join(temporary, 'repository'), directory = join(root, 'web/dashboard');
  for (const folder of ['src', 'scripts', 'heroes', 'schemas', 'assets', 'tests']) await mkdir(join(directory, folder), {recursive: true});
  await mkdir(join(root, 'docs'), {recursive: true});
  await writeFile(join(directory, 'package.json'), '{"type":"module"}');
  await writeFile(join(root, 'docs/result.md'), 'Confirmed bounded result.\n');
  const milestone = {
    id: 'sample-completed', title: 'Completed scoped work', status: 'done',
    acceptance: ['The scoped result is recorded.'],
    evidence: [{label: 'Recorded result', path: 'docs/result.md#result'}],
  };
  const hero = {schemaVersion: 2, id: 'sample', number: 1, title: 'Sample project', summary: 'Synthetic loader records.', updatedAt: '2026-10-20', goals: [
    {id: 'sample-goal', title: 'Confirmed scope', milestones: [milestone]},
  ]};
  const manifest = {schemaVersion: 2, title: 'Synthetic dashboard', description: 'Evidence loader fixtures.', heroes: [{id: 'sample', file: 'heroes/01-sample.json'}]};
  const save = async () => {
    await writeFile(join(directory, 'manifest.json'), JSON.stringify(manifest));
    await writeFile(join(directory, 'heroes/01-sample.json'), JSON.stringify(hero));
  };
  await save();
  return {temporary, root, directory, hero, milestone, save};
}

async function pinCurrent(value, sources = [{path: 'docs/result.md', sha256: digest('Confirmed bounded result.\n')}]) {
  const receipt = JSON.stringify({sourceFiles: sources});
  await writeFile(join(value.root, 'docs/receipt.json'), receipt);
  value.milestone.verification = {
    status: 'current', scope: 'The bounded recorded result.', reason: 'Receipt and source hashes match.',
    receipt: 'docs/receipt.json', receiptSha256: digest(receipt),
  };
  await value.save();
  return receipt;
}

test('ordinary records retain their existing shape and evidence directories are rejected', async t => {
  const value = await fixture(t);
  assert.deepEqual(Object.keys(await readProject(value.directory)).sort(), ['heroes', 'manifest']);
  value.milestone.evidence[0].path = 'docs';
  await value.save();
  await assert.rejects(readProject(value.directory), /regular file/);
});

test('evidence accepts internal file symlinks and rejects repository escapes', async t => {
  const value = await fixture(t);
  await symlink(join(value.root, 'docs/result.md'), join(value.root, 'docs/internal.md'));
  value.milestone.evidence[0].path = 'docs/internal.md';
  await value.save();
  await readProject(value.directory);
  const outside = join(value.temporary, 'outside');
  await mkdir(outside);
  await writeFile(join(outside, 'result.md'), 'Outside the repository.');
  await symlink(outside, join(value.root, 'docs/external'));
  value.milestone.evidence[0].path = 'docs/external/result.md';
  await value.save();
  await assert.rejects(readProject(value.directory), /escapes repository/);
});

test('verification assessments preserve completion while changed sources and receipts become stale', async t => {
  const value = await fixture(t), receipt = await pinCurrent(value);
  let record = await readProject(value.directory);
  assert.equal(record.verificationAssessments[value.milestone.id].status, 'current');
  const original = await sourceFingerprint(value.directory);
  await writeFile(join(value.root, 'docs/result.md'), 'Changed source.\n');
  record = await readProject(value.directory);
  assert.equal(record.heroes[0].goals[0].milestones[0].status, 'done');
  assert.equal(record.verificationAssessments[value.milestone.id].status, 'stale');
  assert.match(record.verificationAssessments[value.milestone.id].reason, /Source bytes differ/);
  assert.notEqual(await sourceFingerprint(value.directory), original);
  await writeFile(join(value.root, 'docs/result.md'), 'Confirmed bounded result.\n');
  await writeFile(join(value.root, 'docs/receipt.json'), receipt + '\n');
  record = await readProject(value.directory);
  assert.match(record.verificationAssessments[value.milestone.id].reason, /receipt no longer matches/);
  assert.notEqual(await sourceFingerprint(value.directory), original);
  await rm(join(value.root, 'docs/receipt.json'));
  record = await readProject(value.directory);
  assert.match(record.verificationAssessments[value.milestone.id].reason, /receipt is missing/);
  assert.notEqual(await sourceFingerprint(value.directory), original);
});

test('missing assessments are unverified and authored historical or unverified assessments stay intact', async t => {
  const value = await fixture(t);
  value.hero.goals[0].milestones.push({
    id: 'sample-next', title: 'Evidence-dependent next step', status: 'proposed',
    dependsOn: [value.milestone.id], requiresCurrentEvidence: [value.milestone.id],
  });
  await value.save();
  assert.equal((await readProject(value.directory)).verificationAssessments[value.milestone.id].status, 'unverified');
  for (const status of ['historical', 'unverified']) {
    value.milestone.verification = {status, scope: 'Recorded prototype only.', reason: 'No current applicability claim.'};
    await value.save();
    assert.deepEqual((await readProject(value.directory)).verificationAssessments[value.milestone.id], value.milestone.verification);
  }
});

test('malformed or unsafe bounded receipts cannot establish current evidence', async t => {
  const value = await fixture(t);
  for (const sources of [[], Array.from({length: 129}, (_, number) => ({path: `docs/source-${number}.md`, sha256: '0'.repeat(64)})),
    [{path: '../outside.md', sha256: '0'.repeat(64)}], [{path: 'docs/result.md#result', sha256: '0'.repeat(64)}],
    [{path: 'docs/result.md', sha256: '0'.repeat(64)}, {path: 'docs/result.md', sha256: '0'.repeat(64)}]]) {
    await pinCurrent(value, sources);
    const assessment = (await readProject(value.directory)).verificationAssessments[value.milestone.id];
    assert.equal(assessment.status, 'stale');
    assert.match(assessment.reason, /cannot establish current evidence/);
  }
});

test('verification paths cannot escape through a symlink even when their final file is missing', async t => {
  const value = await fixture(t), outside = join(value.temporary, 'outside');
  await mkdir(outside);
  await symlink(outside, join(value.root, 'docs/external'));
  await pinCurrent(value, [{path: 'docs/external/missing.md', sha256: '0'.repeat(64)}]);
  const assessment = (await readProject(value.directory)).verificationAssessments[value.milestone.id];
  assert.equal(assessment.status, 'stale');
  assert.match(assessment.reason, /escapes repository/);
  value.milestone.verification.receipt = 'docs/external/missing.json';
  await value.save();
  await assert.rejects(readProject(value.directory), /escapes repository/);
  await symlink(join(outside, 'missing.json'), join(value.root, 'docs/dangling.json'));
  value.milestone.verification.receipt = 'docs/dangling.json';
  await value.save();
  await assert.rejects(readProject(value.directory), /escapes repository/);
});

test('ready gates require the recorded outcome and freshly checked evidence while completed history is preserved', async t => {
  const value = await fixture(t);
  await pinCurrent(value);
  value.milestone.outcome = 'negative';
  const gate = {
    id: 'sample-next', title: 'Begin the next supported step', status: 'ready',
    dependsOn: [value.milestone.id], requiresCurrentEvidence: [value.milestone.id],
    requiresOutcomes: [{milestone: value.milestone.id, outcomes: ['supported']}],
  };
  value.hero.goals[0].milestones.push(gate);
  await value.save();
  await assert.rejects(readProject(value.directory), /ready/i);
  value.milestone.outcome = 'supported';
  await value.save();
  assert.equal((await readProject(value.directory)).heroes[0].goals[0].milestones[1].status, 'ready');
  await writeFile(join(value.root, 'docs/result.md'), 'Source changed after the receipt.\n');
  await assert.rejects(readProject(value.directory), /ready/i);
  gate.status = 'proposed';
  await value.save();
  const record = await readProject(value.directory);
  assert.equal(record.heroes[0].goals[0].milestones[0].status, 'done');
  assert.equal(record.verificationAssessments[value.milestone.id].status, 'stale');
});

test('publishing reads the Hero JSON without opening evidence targets', async t => {
  const value = await fixture(t);
  value.milestone.evidence[0].path = 'docs/absent.md';
  value.milestone.verification = {
    status: 'current', scope: 'The bounded recorded result.', reason: 'Recorded in the Hero JSON.',
    receipt: 'docs/absent-receipt.json', receiptSha256: 'ab'.repeat(32),
  };
  await value.save();
  const published = await readPublishedProject(value.directory);
  assert.equal(published.verificationAssessments[value.milestone.id].status, 'current');
  assert.equal(published.verificationAssessments[value.milestone.id].reason, 'Recorded in the Hero JSON.');
  const before = await publishFingerprint(value.directory);
  await writeFile(join(value.root, 'docs/result.md'), 'Changed evidence bytes.\n');
  assert.equal(await publishFingerprint(value.directory), before);
  await assert.rejects(readProject(value.directory), /regular file/);
});

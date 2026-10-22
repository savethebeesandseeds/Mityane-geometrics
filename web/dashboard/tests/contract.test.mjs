// SPDX-License-Identifier: AGPL-3.0-only
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {assertManagedContainer} from '../scripts/environment.mjs';
assertManagedContainer();
import {filterHeroes, heroLabel, milestoneReadiness, validateProject, validateReadyMilestones} from '../src/model.js';

const fixture = JSON.parse(await readFile(new URL('./shared-contract.json', import.meta.url), 'utf8'));
function projectFor({patches = []} = {}) {
  const project = structuredClone(fixture.project);
  for (const patch of patches) {
    const target = patch.path.slice(0, -1).reduce((value, key) => value[key], project);
    const key = patch.path.at(-1);
    if (patch.remove) delete target[key];
    else target[key] = structuredClone(patch.value);
  }
  return project;
}

for (const contractCase of fixture.validationCases) {
  test(`shared contract: ${contractCase.name}`, () => {
    const {manifest, heroes} = projectFor(contractCase);
    if (contractCase.valid) assert.doesNotThrow(() => validateProject(manifest, heroes));
    else assert.throws(() => validateProject(manifest, heroes), error => error.message.includes(contractCase.error));
  });
}

test('shared contract: canonical and displayed Hero references find the same whole scope', () => {
  const {heroes} = projectFor();
  assert.equal(heroLabel(heroes[0]), 'Hx01');
  for (const query of fixture.searchQueries) {
    const matches = filterHeroes(heroes, {query});
    assert.deepEqual(matches, heroes, query);
  }
  assert.deepEqual(filterHeroes(heroes, {query: 'Hx02'}), []);
});

test('shared contract: evidence applicability and outcomes gate advancement independently of completion', () => {
  const {heroes} = projectFor();
  const [decision, followUp] = heroes[0].goals[0].milestones;
  assert.equal(milestoneReadiness(followUp, heroes).ready, false);
  assert.equal(milestoneReadiness(followUp, heroes, fixture.verificationAssessments).ready, true);
  decision.outcome = 'negative';
  assert.equal(milestoneReadiness(followUp, heroes, fixture.verificationAssessments).ready, false);
  assert.equal(milestoneReadiness(decision, heroes, fixture.verificationAssessments).complete, true);
  assert.equal(decision.status, 'done');
  assert.equal(followUp.status, 'proposed');
});

test('shared contract: ready states require current assessed gates', () => {
  const {heroes} = projectFor();
  heroes[0].goals[0].milestones[1].status = 'ready';
  assert.doesNotThrow(() => validateReadyMilestones(heroes, fixture.verificationAssessments));
  assert.throws(() => validateReadyMilestones(heroes), /Ready Milestone shared-next/);
  const stale = structuredClone(fixture.verificationAssessments);
  stale['shared-decision'].status = 'stale';
  assert.throws(() => validateReadyMilestones(heroes, stale), /Ready Milestone shared-next/);
});

test('shared contract: ready descendants retain all ancestor outcome and evidence gates', () => {
  const {heroes} = projectFor();
  const [decision, intermediate] = heroes[0].goals[0].milestones;
  Object.assign(intermediate, {status: 'done', acceptance: ['The bounded step is recorded.'],
    evidence: [{label: 'Synthetic evidence', path: 'docs/shared-evidence.md'}]});
  heroes[0].goals[0].milestones.push({id: 'shared-tail', title: 'Advance after the bounded step',
    status: 'ready', dependsOn: [intermediate.id]});
  assert.doesNotThrow(() => validateReadyMilestones(heroes, fixture.verificationAssessments));
  decision.outcome = 'negative';
  assert.throws(() => validateReadyMilestones(heroes, fixture.verificationAssessments), /Ready Milestone shared-tail/);
  decision.outcome = 'supported';
  const stale = structuredClone(fixture.verificationAssessments);
  stale[decision.id].status = 'stale';
  assert.throws(() => validateReadyMilestones(heroes, stale), /Ready Milestone shared-tail/);
  assert.equal(decision.status, 'done');
  assert.equal(intermediate.status, 'done');
});

test('shared contract: gate drift preserves historical completed work', () => {
  const {heroes} = projectFor();
  const [decision, followUp] = heroes[0].goals[0].milestones;
  Object.assign(followUp, {status: 'done', acceptance: ['The bounded step is recorded.'],
    evidence: [{label: 'Synthetic evidence', path: 'docs/shared-evidence.md'}]});
  decision.outcome = 'negative';
  const stale = structuredClone(fixture.verificationAssessments);
  stale[decision.id].status = 'stale';
  const before = structuredClone(heroes);
  assert.doesNotThrow(() => validateReadyMilestones(heroes, stale));
  assert.deepEqual(heroes, before);
});

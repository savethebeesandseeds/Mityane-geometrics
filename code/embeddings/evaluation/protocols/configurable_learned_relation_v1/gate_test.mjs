// Artificial fixture source only. Execution is deferred until Docker recovery.
import test from 'node:test';
import assert from 'node:assert/strict';
import { VIEWS, METHODS, assess } from './gate.mjs';
import { assess as assessContent } from '../configurable_content_relation_v1/gate.mjs';

function fixture(masters = [941101, 941102]) {
  const scores = Object.fromEntries(METHODS.map(method => [method,
    Object.fromEntries(masters.map(master => [master, Object.fromEntries(VIEWS.map(view =>
      [view, { ridge: .95, tiny_secondary: .94, coverage: 1 }]))]))]));
  for (const method of ['candidate_learned', 'initial_learned', 'reference_learned']) {
    for (const master of masters) for (const view of VIEWS) Object.assign(scores[method][master][view],
      { ridge: method === 'candidate_learned' ? .75 : .5,
        tiny_secondary: method === 'candidate_learned' ? .73 : .5 });
  }
  const queries = Object.fromEntries(masters.map(master => [master, Object.fromEntries(
    ['training', 'validation'].map(role => [role, { candidate: .79, reference: .8, coverage: 1 }]))]));
  return { scores, queries, masters };
}
const run = fixture => assess(fixture.scores, fixture.queries, fixture.masters);

test('closed complete populations pass with no promotion and family-identical conditions', () => {
  const f = fixture(); const out = run(f);
  assert.equal(out.passed, true); assert.equal(out.promotion, false);
  assert.equal(out.conditions.length, 54);
  assert.deepEqual(out, assessContent(f.scores, f.queries, f.masters));
});
test('exact .60 accuracy and .10 gain boundaries retain declared slack', () => {
  const f = fixture();
  for (const master of f.masters) for (const view of VIEWS)
    Object.assign(f.scores.candidate_learned[master][view], { ridge: .60, tiny_secondary: .60 });
  assert.equal(run(f).passed, true);
  for (const master of f.masters) f.scores.candidate_learned[master].validation_deleted.ridge = .60 - 2e-12;
  assert.equal(run(f).passed, false);
});
test('an omitted cohort, extra view, nonfinite value, or unsupported master is rejected', () => {
  const f = fixture(); delete f.scores.reference_learned[941102]; assert.throws(() => run(f));
  const g = fixture(); g.scores.candidate_learned[941101].validation_intact.ridge = NaN;
  assert.throws(() => run(g));
  const h = fixture(); h.queries[941101].test = { candidate: 0, reference: 1, coverage: 1 };
  assert.throws(() => run(h));
  assert.throws(() => run(fixture([941104])));
});
test('a single head/view failure and worst cohort are never hidden by a mean', () => {
  const f = fixture();
  for (const master of f.masters) f.scores.candidate_learned[master].validation_deleted.tiny_secondary = .59;
  assert.ok(run(f).failed_conditions.includes('learned_accuracy/validation_deleted/tiny_secondary'));
  const g = fixture();
  g.scores.candidate_full[941101].validation_intact.ridge = .88;
  g.scores.candidate_full[941102].validation_intact.ridge = 1;
  assert.ok(run(g).conditions.some(c => c.condition === 'complete_cohort_loss/validation_intact/ridge' &&
    c.master === 941101 && !c.passed));
});
test('initial score, coverage, and query role protection remain independent', () => {
  const f = fixture(); f.scores.initial_learned = structuredClone(f.scores.candidate_learned);
  assert.ok(run(f).failed_conditions.some(name => name.startsWith('learned_training_gain/')));
  const g = fixture(); g.scores.candidate_learned[941101].validation_deleted.coverage = .99;
  assert.equal(run(g).passed, false);
  const h = fixture(); h.queries[941101].training.candidate = 1;
  assert.ok(run(h).failed_conditions.includes('query_cohort_ratio/training'));
});
test('third development stage stays distinct and overflow fails closed', () => {
  const f = fixture([941103]); assert.equal(run(f).stage, 'third-master-development');
  const g = fixture();
  for (const master of g.masters) for (const role of ['training', 'validation'])
    Object.assign(g.queries[master][role], { candidate: Number.MAX_VALUE, reference: Number.MAX_VALUE });
  assert.throws(() => run(g), /overflow/);
});

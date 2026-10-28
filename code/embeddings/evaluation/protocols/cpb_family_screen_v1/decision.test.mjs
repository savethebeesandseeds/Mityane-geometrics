// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {CONSUMED, MINIMUM_EFFECT, RECIPE_PREFIX, REFERENCE_BUDGET, cli, decide, median} from './decision.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const confirmationPath = path.join(here, 'confirmation.json');
const stored = JSON.parse(fs.readFileSync(confirmationPath, 'utf8'));

const v17 = [
  {seed: 947101, past_volatility: -0.01575917, past_trend: -0.006259664, past_activity: -0.02294518, past_relation: -0.03412885},
  {seed: 947102, past_volatility: -0.002418553, past_trend: 0.1220014, past_activity: -0.003551464, past_relation: 0.01620092},
  {seed: 947103, past_volatility: 0.0002848377, past_trend: -0.003029098, past_activity: -0.008995412, past_relation: 0.003185947},
];
const v18 = [
  {seed: 947101, past_volatility: 0.010075214, past_trend: -0.22184589, past_activity: -0.10393168, past_relation: 0.015733337},
  {seed: 947102, past_volatility: -0.081093928, past_trend: -0.26636724, past_activity: -0.090619824, past_relation: -0.0018954941},
  {seed: 947103, past_volatility: -0.057857421, past_trend: -0.092922906, past_activity: 0.014212272, past_relation: 0.16720569},
];
const v19 = [
  {seed: 947101, past_volatility: 0.053184188, past_trend: -0.072451761, past_activity: -0.097578568, past_relation: -0.29490823},
  {seed: 947102, past_volatility: 0.029557094, past_trend: -0.029320871, past_activity: 0.050901682, past_relation: 0.10958329},
  {seed: 947103, past_volatility: 0.14628854, past_trend: 0.037644915, past_activity: -0.053270555, past_relation: 0.19066920},
];

const shipping = `${RECIPE_PREFIX}executable=${'a'.repeat(64)};settings=${'b'.repeat(64)};dataset=${'c'.repeat(64)};batch=8;rows=mt19937_64-counter-seed-mod-N-with-replacement-v1;context=none;calibration=all-TRAIN-source-order-batch-chunks-Chan-v1;row-seed=1;mask-seed=2;context-seed=3;AdamW-lr=0.0005;wd=0.0002;global-clip=0.7`;

function passing(seedCount = 3) {
  return Array.from({length: seedCount}, (_, seed) => ({
    seed,
    past_volatility: 0.02,
    past_trend: 0.03,
    past_activity: 0.04,
    past_relation: 0.05,
  }));
}

function sealed(id = 'sealed-later-window-fixture', consumedPopulations = stored.consumed_populations) {
  return {
    kind: 'cpb_family_sealed_confirmation_v1',
    status: 'sealed',
    bound: {id, consumed: false},
    consumed_populations: consumedPopulations,
    rule: stored.rule,
  };
}

function screen(overrides = {}) {
  return decide({
    population: {id: 'sealed-later-window-fixture', role: 'confirmation'},
    pairs: passing(),
    completedUpdates: REFERENCE_BUDGET,
    recipe: shipping,
    confirmation: sealed(),
    ...overrides,
  });
}

test('stored confirmation is unbound and names the consumed populations', () => {
  assert.equal(stored.kind, 'cpb_family_sealed_confirmation_v1');
  assert.equal(stored.status, 'unbound');
  assert.equal(stored.bound, null);
  assert.deepEqual(stored.consumed_populations.map(population => population.id), CONSUMED);
  assert.deepEqual(stored.consumed_populations.map(population => population.role), ['development', 'confirmation', 'confirmation']);
  assert.equal(MINIMUM_EFFECT, 0.01);
  assert.equal(REFERENCE_BUDGET, 2048);
  assert.equal(RECIPE_PREFIX, 'cpb-family-train-example-v1;');
});

test('median is the middle order statistic and requires an odd count', () => {
  assert.equal(median([-0.006259664, 0.1220014, -0.003029098]), -0.003029098);
  assert.throws(() => median([0.01, 0.02]), /odd nonempty/);
  assert.throws(() => median([]), /odd nonempty/);
});

test('published V17 pairs are inconclusive and keep the historical nonnegative counts', () => {
  const decision = decide({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    pairs: v17,
    completedUpdates: 256,
    recipe: 'cpb-v17-portable-train-v1',
    confirmation: stored,
  });
  assert.equal(decision.outcome, 'inconclusive');
  assert.equal(decision.promotesReference, false);
  assert.equal(decision.trendActivity.trend.median, -0.003029098);
  assert.equal(decision.trendActivity.trend.nonnegative, 1);
  assert.equal(decision.trendActivity.activity.median, -0.008995412);
  assert.equal(decision.trendActivity.activity.nonnegative, 0);
  assert.equal(decision.volatility.median, -0.002418553);
  assert.equal(decision.volatility.nonnegative, 1);
  assert.equal(decision.relation.median, 0.003185947);
  assert.equal(decision.relation.nonnegative, 2);
  assert.match(decision.reasons.join('\n'), /does not kill or pass/);
});

test('published V18 relation gain extends without erasing the trend failure', () => {
  const decision = decide({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    pairs: v18,
    completedUpdates: 1024,
    recipe: 'cpb-v18-span-v1',
    confirmation: stored,
  });
  assert.equal(decision.outcome, 'extend_for_volatility_or_relation');
  assert.equal(decision.promotesReference, false);
  assert.equal(decision.relation.median, 0.015733337);
  assert.equal(decision.relation.buysExtension, true);
  assert.equal(decision.trendActivity.trend.median, -0.22184589);
  assert.match(decision.reasons.join('\n'), /trend median -0.22184589/);
  assert.match(decision.reasons.join('\n'), /failure stays on the record/);
});

test('published V19 volatility gain extends and does not promote', () => {
  const decision = decide({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    pairs: v19,
    completedUpdates: 1024,
    recipe: 'cpb-v19-teacher-v1',
    confirmation: stored,
  });
  assert.equal(decision.outcome, 'extend_for_volatility_or_relation');
  assert.equal(decision.promotesReference, false);
  assert.equal(decision.volatility.median, 0.053184188);
  assert.equal(decision.volatility.buysExtension, true);
  assert.equal(decision.relation.median, 0.10958329);
  assert.equal(decision.trendActivity.trend.median, -0.029320871);
  assert.equal(decision.trendActivity.activity.median, -0.053270555);
  assert.match(decision.reasons.join('\n'), /volatility median 0.053184188/);
  assert.match(decision.reasons.join('\n'), /activity median -0.053270555/);
});

test('development gains continue to the reference budget and cannot promote', () => {
  const decision = screen({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    confirmation: sealed('sealed-later-window-fixture'),
  });
  assert.equal(decision.outcome, 'continue_at_reference_budget');
  assert.equal(decision.promotesReference, false);
  assert.match(decision.reasons.join('\n'), /development cannot promote/);
});

test('a sealed family screen at the reference budget promotes', () => {
  const decision = screen();
  assert.equal(decision.outcome, 'promote');
  assert.equal(decision.promotesReference, true);
  const single = screen({pairs: passing(1)});
  assert.equal(single.outcome, 'promote');
  assert.equal(single.trendActivity.trend.pairs, 1);
});

test('consumed, unbound, historical-recipe and short-budget confirmation scores refuse promotion', () => {
  const march = screen({
    population: {id: 'march-2024', role: 'confirmation'},
    confirmation: sealed('march-2024'),
  });
  assert.equal(march.outcome, 'refuse_promotion');
  assert.equal(march.promotesReference, false);
  assert.match(march.reasons.join('\n'), /march-2024 is consumed/);

  const unbound = screen({confirmation: stored});
  assert.equal(unbound.outcome, 'refuse_promotion');
  assert.match(unbound.reasons.join('\n'), /unbound/);

  const historical = screen({recipe: 'cpb-v17-portable-train-v1', completedUpdates: 256, confirmation: stored});
  assert.equal(historical.outcome, 'refuse_promotion');
  const text = historical.reasons.join('\n');
  assert.match(text, /cpb-v17-portable-train-v1/);
  assert.match(text, /256/);
  assert.match(text, /unbound/);

  const short = screen({completedUpdates: 2047});
  assert.equal(short.outcome, 'refuse_promotion');
  assert.match(short.reasons.join('\n'), /2047/);

  const forged = screen({recipe: 'cpb-family'});
  assert.equal(forged.outcome, 'refuse_promotion');
  assert.match(forged.reasons.join('\n'), /does not begin with cpb-family-train-example-v1;/);
  const barePrefix = screen({recipe: 'cpb-family-train-example-v1'});
  assert.equal(barePrefix.outcome, 'refuse_promotion');

  const bypass = screen({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'confirmation'},
    confirmation: sealed('january-2024-mkt-hist-2-days-22-28', []),
  });
  assert.equal(bypass.outcome, 'refuse_promotion');
  assert.match(bypass.reasons.join('\n'), /consumed/);
});

test('failed confirmation scores stay stop, extend or inconclusive', () => {
  const stopped = screen({
    pairs: passing().map(row => ({...row, past_trend: -0.02, past_activity: -0.03, past_volatility: 0, past_relation: 0})),
  });
  assert.equal(stopped.outcome, 'stop');
  assert.equal(stopped.promotesReference, false);
  const extended = screen({pairs: v19});
  assert.equal(extended.outcome, 'extend_for_volatility_or_relation');
  const quiet = screen({pairs: v17});
  assert.equal(quiet.outcome, 'inconclusive');
});

test('boundaries and specialty priority follow the 0.01 rule', () => {
  const exact = screen({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    pairs: passing().map(row => ({...row, past_trend: 0.01, past_activity: 0.01, past_volatility: -0.01, past_relation: 0})),
  });
  assert.equal(exact.outcome, 'continue_at_reference_budget');
  const floorMiss = screen({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    pairs: passing().map(row => ({...row, past_volatility: -0.02, past_relation: 0.02})),
  });
  assert.equal(floorMiss.outcome, 'extend_for_volatility_or_relation');
  const killed = screen({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    pairs: passing().map(row => ({...row, past_trend: -0.02, past_activity: -0.02, past_volatility: 0, past_relation: 0})),
  });
  assert.equal(killed.outcome, 'stop');
  const bought = screen({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    pairs: passing().map(row => ({...row, past_trend: -0.2, past_activity: -0.2, past_volatility: 0, past_relation: 0.01})),
  });
  assert.equal(bought.outcome, 'extend_for_volatility_or_relation');
  const edge = screen({
    population: {id: 'january-2024-mkt-hist-2-days-22-28', role: 'development'},
    pairs: passing().map(row => ({...row, past_trend: -0.01, past_activity: -0.01, past_volatility: 0, past_relation: 0})),
  });
  assert.equal(edge.outcome, 'inconclusive');
});

test('the command line reads pairs and refuses to bind', () => {
  const before = fs.readFileSync(confirmationPath);
  assert.throws(() => cli(['--owner-bind', '--confirmation', confirmationPath]), /does not bind/);
  assert.deepEqual(fs.readFileSync(confirmationPath), before);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'cpb-family-screen-'));
  try {
    const pairs = path.join(directory, 'pairs.json');
    fs.writeFileSync(pairs, JSON.stringify(v17));
    const decision = cli([
      '--pairs', pairs,
      '--population', 'january-2024-mkt-hist-2-days-22-28',
      '--role', 'development',
      '--recipe', 'cpb-v17-portable-train-v1',
      '--completed-updates', '256',
      '--confirmation', confirmationPath,
    ]);
    assert.equal(decision.outcome, 'inconclusive');
    assert.throws(() => cli([
      '--pairs', pairs,
      '--population', 'january-2024-mkt-hist-2-days-22-28',
      '--role', 'development',
      '--recipe', 'cpb-family',
      '--completed-updates', '256',
      '--completed-updates', '2048',
      '--confirmation', confirmationPath,
    ]), /duplicate option/);
    fs.writeFileSync(pairs, JSON.stringify(passing()));
    const parsed = cli([
      '--pairs', pairs,
      '--population', 'sealed-later-window-fixture',
      '--role', 'confirmation',
      '--recipe', shipping,
      '--completed-updates', '2048',
      '--confirmation', confirmationPath,
    ]);
    assert.equal(parsed.outcome, 'refuse_promotion');
    assert.match(parsed.reasons.join('\n'), /unbound/);
    assert.throws(() => cli([
      '--pairs', pairs,
      '--population', 'sealed-later-window-fixture',
      '--role', 'confirmation',
      '--recipe', 'cpb-family train',
      '--completed-updates', '2048',
      '--confirmation', confirmationPath,
    ]), /recipe token invalid/);
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
});

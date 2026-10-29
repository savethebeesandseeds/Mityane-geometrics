// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Standing family-screen decision. Development cannot promote.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const MINIMUM_EFFECT = 0.01;
export const REFERENCE_BUDGET = 2048;
// The shipping trainer writes this prefix, then executable, settings, dataset,
// batch, seeds and optimizer fields. The short token "cpb-family" is not a recipe.
export const RECIPE_PREFIX = 'cpb-family-train-example-v1;';
// Published three-instrument probe names. anchor_volatility, anchor_trend,
// anchor_activity and cross_resolution are a different contract and are not
// accepted by this rule.
export const TASKS = ['past_volatility', 'past_trend', 'past_activity', 'past_relation'];
export const CONSUMED = [
  'january-2024-mkt-hist-2-days-22-28',
  'february-2024',
  'march-2024',
];
const CONFIRMATION_KIND = 'cpb_family_sealed_confirmation_v1';

export function median(values) {
  if (!Array.isArray(values) || values.length === 0 || values.length % 2 === 0) {
    throw new Error('odd nonempty pair count required so the median is unique');
  }
  for (const value of values) {
    if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('finite pair deltas required');
  }
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[(sorted.length - 1) / 2];
}

export function summarize(values) {
  const middle = median(values);
  return {
    median: middle,
    minimum: Math.min(...values),
    maximum: Math.max(...values),
    nonnegative: values.filter(value => value >= 0).length,
    pairs: values.length,
  };
}

function finiteTask(row, task) {
  const value = row?.[task];
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`finite ${task} delta required`);
  return value;
}

function taskColumn(pairs, task) {
  return pairs.map(row => finiteTask(row, task));
}

function confirmationShape(confirmation) {
  if (!confirmation || confirmation.kind !== CONFIRMATION_KIND) throw new Error('cpb_family_sealed_confirmation_v1 required');
  if (confirmation.status !== 'unbound' && confirmation.status !== 'sealed') throw new Error('confirmation status must be unbound or sealed');
  if (!Array.isArray(confirmation.consumed_populations)) throw new Error('consumed_populations required');
  for (const population of confirmation.consumed_populations) {
    if (!population || typeof population.id !== 'string' || population.id === '') throw new Error('consumed population id required');
    if (population.role !== 'development' && population.role !== 'confirmation') throw new Error('consumed population role invalid');
  }
  if (confirmation.status === 'unbound') {
    if (confirmation.bound !== null) throw new Error('unbound confirmation has no bound population');
    return;
  }
  const bound = confirmation.bound;
  if (!bound || typeof bound.id !== 'string' || bound.id === '' || typeof bound.consumed !== 'boolean') {
    throw new Error('sealed confirmation requires bound.id and bound.consumed');
  }
}

function countLine(trend, activity, volatility, relation) {
  const part = (name, summary) => `${name} ${summary.nonnegative}/${summary.pairs}`;
  return `nonnegative pairs: ${part('trend', trend)}, ${part('activity', activity)}, ${part('volatility', volatility)}, ${part('relation', relation)}`;
}

export function shippingRecipe(recipe) {
  return recipe.startsWith(RECIPE_PREFIX) && recipe.length > RECIPE_PREFIX.length && recipe.length <= 4096 && !recipe.includes('\0');
}

function promotionGate(population, recipe, completedUpdates, confirmation) {
  const reasons = [];
  const consumed = new Set([...CONSUMED, ...confirmation.consumed_populations.map(item => item.id)]);
  if (confirmation.status !== 'sealed' || confirmation.bound === null) reasons.push('confirmation is unbound');
  if (confirmation.status === 'sealed' && confirmation.bound && confirmation.bound.id !== population.id) {
    reasons.push(`sealed population ${confirmation.bound.id} is not ${population.id}`);
  }
  if (confirmation.status === 'sealed' && confirmation.bound && confirmation.bound.consumed !== false) {
    reasons.push('sealed population is already marked consumed');
  }
  if (consumed.has(population.id)) reasons.push(`population ${population.id} is consumed`);
  if (!shippingRecipe(recipe)) reasons.push(`recipe ${recipe} does not begin with ${RECIPE_PREFIX}`);
  if (completedUpdates < REFERENCE_BUDGET) reasons.push(`completed updates ${completedUpdates} are below reference budget ${REFERENCE_BUDGET}`);
  return reasons;
}

export function decide({population, pairs, completedUpdates, recipe, confirmation}) {
  if (!population || (population.role !== 'development' && population.role !== 'confirmation')) {
    throw new Error('population role must be development or confirmation');
  }
  if (typeof population.id !== 'string' || population.id === '') throw new Error('population id required');
  if (!Array.isArray(pairs) || pairs.length === 0) throw new Error('pairs required');
  const seeds = new Set();
  for (const row of pairs) {
    if (!row || (typeof row.seed !== 'number' && typeof row.seed !== 'string')) throw new Error('pair seed required');
    if (seeds.has(row.seed)) throw new Error(`duplicate pair seed ${row.seed}`);
    seeds.add(row.seed);
  }
  if (typeof recipe !== 'string' || recipe === '') throw new Error('recipe required');
  if (!Number.isInteger(completedUpdates) || completedUpdates < 0) throw new Error('nonnegative integer completed updates required');
  confirmationShape(confirmation);

  const trend = summarize(taskColumn(pairs, 'past_trend'));
  const activity = summarize(taskColumn(pairs, 'past_activity'));
  const volatilitySummary = summarize(taskColumn(pairs, 'past_volatility'));
  const relationSummary = summarize(taskColumn(pairs, 'past_relation'));
  const trendActivity = {trend, activity, passed: trend.median >= MINIMUM_EFFECT && activity.median >= MINIMUM_EFFECT};
  const volatility = {
    ...volatilitySummary,
    floorHeld: volatilitySummary.median >= -MINIMUM_EFFECT,
    buysExtension: volatilitySummary.median >= MINIMUM_EFFECT,
  };
  const relation = {...relationSummary, buysExtension: relationSummary.median >= MINIMUM_EFFECT};
  const reported = countLine(trend, activity, volatility, relation);
  const base = {trendActivity, volatility, relation, promotesReference: false};
  const scoresPass = trendActivity.passed && volatility.floorHeld;
  const specialty = volatility.buysExtension || relation.buysExtension;

  if (population.role === 'confirmation' && scoresPass) {
    const blocked = promotionGate(population, recipe, completedUpdates, confirmation);
    if (blocked.length === 0) {
      return {
        ...base,
        outcome: 'promote',
        promotesReference: true,
        reasons: [
          'sealed confirmation scores pass the trend, activity and volatility-floor gates at the reference budget for a cpb-family-train-example-v1 training state',
          reported,
        ],
      };
    }
    return {...base, outcome: 'refuse_promotion', reasons: [...blocked, reported]};
  }

  if (population.role === 'development' && scoresPass) {
    const reasons = [
      'development cannot promote',
      'trend and activity pass and volatility holds the floor; continue to the reference budget on this development population',
    ];
    if (!shippingRecipe(recipe)) reasons.push(`recipe ${recipe} does not begin with ${RECIPE_PREFIX}; this tool cannot promote it`);
    reasons.push(reported);
    return {...base, outcome: 'continue_at_reference_budget', reasons};
  }

  if (specialty) {
    const reasons = [];
    if (volatility.buysExtension) reasons.push(`volatility median ${volatility.median} buys a full-budget development extension`);
    if (relation.buysExtension) reasons.push(`relation median ${relation.median} buys a full-budget development extension`);
    if (!trendActivity.passed) {
      reasons.push(`trend median ${trend.median} and activity median ${activity.median} do not both clear ${MINIMUM_EFFECT}; that failure stays on the record`);
    }
    if (trendActivity.passed && !volatility.floorHeld) {
      reasons.push(`volatility median ${volatility.median} misses the ${-MINIMUM_EFFECT} floor, so this is an extension rather than reference-budget continuation`);
    }
    reasons.push('development cannot promote');
    reasons.push(reported);
    return {...base, outcome: 'extend_for_volatility_or_relation', reasons};
  }

  if (trend.median < -MINIMUM_EFFECT && activity.median < -MINIMUM_EFFECT) {
    return {
      ...base,
      outcome: 'stop',
      reasons: [
        `trend median ${trend.median} and activity median ${activity.median} are both below ${-MINIMUM_EFFECT}`,
        'neither volatility nor relation buys an extension',
        reported,
      ],
    };
  }

  const inside = [trend, activity, volatility, relation].every(summary => summary.median >= -MINIMUM_EFFECT && summary.median < MINIMUM_EFFECT);
  return {
    ...base,
    outcome: 'inconclusive',
    reasons: [
      inside
        ? 'all task medians lie in [-0.01, 0.01); this does not kill or pass'
        : 'the screen is mixed: trend and activity do not continue, volatility and relation do not buy an extension, and trend and activity are not both below -0.01',
      reported,
    ],
  };
}

function flagValue(args, name) {
  const index = args.indexOf(name);
  if (index < 0 || index + 1 >= args.length || args[index + 1].startsWith('--')) throw new Error(`missing option: ${name}`);
  return args[index + 1];
}

export function cli(argv) {
  if (argv.includes('--owner-bind')) {
    throw new Error('binding a confirmation is an owner edit of confirmation.json; this tool does not bind or write research-reference.json');
  }
  const known = new Set(['--pairs', '--population', '--role', '--recipe', '--completed-updates', '--confirmation']);
  const seen = new Set();
  for (let index = 0; index < argv.length; index += 2) {
    if (!known.has(argv[index])) throw new Error(`unknown option: ${argv[index]}`);
    if (seen.has(argv[index])) throw new Error(`duplicate option: ${argv[index]}`);
    seen.add(argv[index]);
    if (index + 1 >= argv.length) throw new Error(`missing value: ${argv[index]}`);
  }
  for (const name of known) if (!seen.has(name)) throw new Error(`missing option: ${name}`);
  const populationId = flagValue(argv, '--population');
  const role = flagValue(argv, '--role');
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(populationId)) throw new Error('population id must be lowercase hyphenated');
  if (role !== 'development' && role !== 'confirmation') throw new Error('role must be development or confirmation');
  const recipe = flagValue(argv, '--recipe');
  if (!/^[A-Za-z0-9][A-Za-z0-9._:;=+\-]{0,4095}$/.test(recipe)) throw new Error('recipe token invalid');
  const updatesText = flagValue(argv, '--completed-updates');
  if (!/^(0|[1-9][0-9]*)$/.test(updatesText)) throw new Error('completed updates must be a nonnegative integer');
  const completedUpdates = Number(updatesText);
  if (!Number.isSafeInteger(completedUpdates)) throw new Error('completed updates exceed the safe integer range');
  const pairs = JSON.parse(fs.readFileSync(flagValue(argv, '--pairs'), 'utf8'));
  const confirmation = JSON.parse(fs.readFileSync(flagValue(argv, '--confirmation'), 'utf8'));
  return decide({population: {id: populationId, role}, pairs, completedUpdates, recipe, confirmation});
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  try {
    process.stdout.write(`${JSON.stringify(cli(process.argv.slice(2)), null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}

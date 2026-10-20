// Prospective numerical gate; scores must already be independently audited.
export const VIEWS = Object.freeze(['validation_intact', 'validation_deleted']);
export const HEADS = Object.freeze(['ridge', 'tiny_secondary']);
export const METHODS = Object.freeze(['candidate_full', 'candidate_learned', 'initial_learned',
  'reference_full', 'reference_learned']);
const check = (ok, why) => { if (!ok) throw new Error(why); };
const numeric = value => (typeof value === 'number' || typeof value === 'boolean') && Number.isFinite(Number(value));
const keys = (value, expected) => value !== null && typeof value === 'object' && !Array.isArray(value) &&
  JSON.stringify(Object.keys(value).sort()) === JSON.stringify(expected.map(String).sort());

function mean(values) {
  // The closed prospective stages contain exactly one or two cohorts. Each
  // population is finite and nonnegative. A single binary64 addition is the
  // correctly rounded exact sum for two terms, matching math.fsum here. Do not
  // extend the stage population or replace this with a multi-term reduction.
  check((values.length === 1 || values.length === 2) && values.every(value => numeric(value) && value >= 0),
    'finite nonempty closed gate population');
  const total = values.length === 1 ? Number(values[0]) : Number(values[0]) + Number(values[1]);
  check(Number.isFinite(total), 'gate sum overflow');
  return total === 0 ? 0 : total / values.length;
}

export function assess(scores, queries, masters) {
  check(JSON.stringify(masters) === '[941101,941102]' || JSON.stringify(masters) === '[941103]',
    'closed prospective stage');
  check(keys(scores, METHODS) && keys(queries, masters), 'complete declared gate methods/cohorts');
  const conditions = [];
  function record(condition, value, bound, rule, master = null) {
    check(numeric(value) && numeric(bound), 'finite gate values');
    const passed = rule === 'at_least' ? value >= bound - 1e-12 : value <= bound + 1e-12;
    conditions.push({ condition, master, value, bound, rule, passed });
  }
  for (const method of METHODS) {
    check(keys(scores[method], masters), 'no omitted or extra method cohort');
    for (const master of masters) {
      check(keys(scores[method][master], VIEWS), 'exact two scored views');
      for (const view of VIEWS) {
        const panel = scores[method][master][view];
        check(keys(panel, [...HEADS, 'coverage']) && Object.values(panel).every(value =>
          numeric(value) && value >= 0 && value <= 1), 'finite bounded scores and coverage');
        record(`${method}/${view}/coverage`, panel.coverage, 1, 'at_least', master);
      }
    }
  }
  for (const view of VIEWS) for (const head of HEADS) {
    const average = Object.fromEntries(METHODS.map(method =>
      [method, mean(masters.map(master => scores[method][master][view][head]))]));
    const name = `${view}/${head}`;
    record(`learned_accuracy/${name}`, average.candidate_learned, .60, 'at_least');
    record(`learned_training_gain/${name}`, average.candidate_learned - average.initial_learned, .10, 'at_least');
    record(`learned_reference_gain/${name}`, average.candidate_learned - average.reference_learned, .10, 'at_least');
    record(`complete_mean_loss/${name}`, average.reference_full - average.candidate_full, .02, 'at_most');
    for (const master of masters) record(`complete_cohort_loss/${name}`,
      scores.reference_full[master][view][head] - scores.candidate_full[master][view][head], .05, 'at_most', master);
  }
  for (const role of ['training', 'validation']) {
    const candidate = [], reference = [];
    for (const master of masters) {
      check(keys(queries[master], ['training', 'validation']), 'exact query roles');
      const panel = queries[master][role];
      check(keys(panel, ['candidate', 'reference', 'coverage']) && Object.values(panel).every(numeric) &&
        panel.candidate >= 0 && panel.reference > 0 && panel.coverage >= 0 && panel.coverage <= 1,
      'finite supported query errors');
      record(`query_coverage/${role}`, panel.coverage, 1, 'at_least', master);
      record(`query_cohort_ratio/${role}`, panel.candidate / panel.reference, 1.05, 'at_most', master);
      candidate.push(panel.candidate); reference.push(panel.reference);
    }
    record(`query_mean_ratio/${role}`, mean(candidate) / mean(reference), 1.02, 'at_most');
  }
  return { passed: conditions.every(condition => condition.passed), conditions,
    failed_conditions: conditions.filter(condition => !condition.passed).map(condition => condition.condition),
    stage: masters.length === 2 ? 'two-master-screen' : 'third-master-development', masters: [...masters],
    units: 'accuracy fractions; MAE ratios', promotion: false };
}

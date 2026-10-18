"""Prospective card's numerical gate, using independently audited saved scores."""
import math

VIEWS = ('validation_intact', 'validation_deleted')
HEADS = ('ridge', 'tiny_secondary')
METHODS = ('candidate_full', 'candidate_learned', 'initial_learned',
           'reference_full', 'reference_learned')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def mean(values):
    check(bool(values) and all(isinstance(v, (float, int)) and math.isfinite(v)
                              for v in values), 'finite nonempty gate population')
    return math.fsum(values) / len(values)


def assess(scores, queries, masters):
    """Scores are equal-repetition cohort means, in fractions, not percents.

    Every cohort must already have all three declared repetitions audited.
    Each method maps master -> view -> {ridge, tiny_secondary, coverage}.
    Queries map master -> role -> {candidate, reference, coverage}.
    Integrity is a prerequisite owned by the publisher, never a numeric score.
    """
    check(tuple(masters) in ((941101, 941102), (941103,)), 'closed prospective stage')
    check(set(scores) == set(METHODS) and set(queries) == set(masters),
          'complete declared gate methods/cohorts')
    conditions = []

    def record(name, value, bound, rule, cohort=None):
        check(math.isfinite(value) and math.isfinite(bound), 'finite gate values')
        # Binary roundoff must not turn an exact declared boundary into a fail.
        passed = value >= bound - 1e-12 if rule == 'at_least' else value <= bound + 1e-12
        conditions.append(dict(condition=name, master=cohort, value=value,
                               bound=bound, rule=rule, passed=passed))

    for method in METHODS:
        check(set(scores[method]) == set(masters), 'no omitted or extra method cohort')
        for master in masters:
            check(set(scores[method][master]) == set(VIEWS), 'exact two scored views')
            for view in VIEWS:
                panel = scores[method][master][view]
                check(set(panel) == set(HEADS) | {'coverage'}
                      and all(isinstance(v, (int, float)) and math.isfinite(v)
                              and 0 <= v <= 1 for v in panel.values()),
                      'finite bounded scores and coverage')
                record(method + '/' + view + '/coverage', panel['coverage'], 1.,
                       'at_least', master)
    for view in VIEWS:
        for head in HEADS:
            average = {method: mean([scores[method][m][view][head] for m in masters])
                       for method in METHODS}
            name = view + '/' + head
            record('learned_accuracy/' + name, average['candidate_learned'], .60, 'at_least')
            record('learned_training_gain/' + name,
                   average['candidate_learned'] - average['initial_learned'], .10, 'at_least')
            record('learned_reference_gain/' + name,
                   average['candidate_learned'] - average['reference_learned'], .10, 'at_least')
            record('complete_mean_loss/' + name,
                   average['reference_full'] - average['candidate_full'], .02, 'at_most')
            for master in masters:
                record('complete_cohort_loss/' + name,
                       scores['reference_full'][master][view][head]
                       - scores['candidate_full'][master][view][head], .05, 'at_most', master)
    for role in ('training', 'validation'):
        candidate, reference = [], []
        for master in masters:
            check(set(queries[master]) == {'training', 'validation'}, 'exact query roles')
            panel = queries[master][role]
            check(set(panel) == {'candidate', 'reference', 'coverage'}
                  and all(isinstance(v, (float, int)) and math.isfinite(v)
                          for v in panel.values())
                  and panel['candidate'] >= 0 and panel['reference'] > 0
                  and 0 <= panel['coverage'] <= 1, 'finite supported query errors')
            record('query_coverage/' + role, panel['coverage'], 1., 'at_least', master)
            record('query_cohort_ratio/' + role, panel['candidate'] / panel['reference'],
                   1.05, 'at_most', master)
            candidate.append(panel['candidate'])
            reference.append(panel['reference'])
        record('query_mean_ratio/' + role, mean(candidate) / mean(reference), 1.02, 'at_most')
    return dict(passed=all(c['passed'] for c in conditions), conditions=conditions,
                failed_conditions=[c['condition'] for c in conditions if not c['passed']],
                stage='two-master-screen' if len(masters) == 2 else 'third-master-development',
                masters=list(masters), units='accuracy fractions; MAE ratios', promotion=False)

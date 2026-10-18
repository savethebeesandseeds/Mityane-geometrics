#!/usr/bin/env python3
"""Meaningful gate mutations; no tensors, quality input, model or classifier."""
import copy
import unittest
import gate


def fixture(masters=(941101, 941102)):
    scores = {method: {master: {view: dict(ridge=.95, tiny_secondary=.94, coverage=1.)
                              for view in gate.VIEWS} for master in masters}
              for method in gate.METHODS}
    for name in ('candidate_learned', 'initial_learned', 'reference_learned'):
        for master in masters:
            for view in gate.VIEWS:
                scores[name][master][view].update(
                    ridge=.75 if name == 'candidate_learned' else .5,
                    tiny_secondary=.73 if name == 'candidate_learned' else .5)
    queries = {master: {role: dict(candidate=.79, reference=.8, coverage=1.)
                       for role in ('training', 'validation')} for master in masters}
    return scores, queries


class GateTest(unittest.TestCase):
    def test_complete_candidate_passes(self):
        s, q = fixture()
        self.assertTrue(gate.assess(s, q, (941101, 941102))['passed'])

    def test_exact_accuracy_and_gain_boundaries(self):
        s, q = fixture()
        for master in s['candidate_learned']:
            for view in gate.VIEWS:
                s['candidate_learned'][master][view].update(ridge=.60, tiny_secondary=.60)
        self.assertTrue(gate.assess(s, q, (941101, 941102))['passed'])

    def test_unseen_confirmation_is_not_allocated(self):
        s, q = fixture((941103,))
        out = gate.assess(s, q, (941103,))
        self.assertTrue(out['passed'])
        self.assertEqual(out['stage'], 'third-master-development')
        with self.assertRaises(ValueError):
            gate.assess(s, q, (941104,))

    def test_one_head_one_view_failure_is_retained(self):
        s, q = fixture()
        for master in s['candidate_learned']:
            s['candidate_learned'][master]['validation_deleted']['tiny_secondary'] = .59
        out = gate.assess(s, q, (941101, 941102))
        self.assertFalse(out['passed'])
        self.assertIn('learned_accuracy/validation_deleted/tiny_secondary', out['failed_conditions'])

    def test_high_initial_score_cannot_be_called_learning(self):
        s, q = fixture()
        s['initial_learned'] = copy.deepcopy(s['candidate_learned'])
        out = gate.assess(s, q, (941101, 941102))
        self.assertFalse(out['passed'])
        self.assertTrue(any(n.startswith('learned_training_gain/') for n in out['failed_conditions']))

    def test_worst_cohort_not_hidden_by_mean(self):
        s, q = fixture()
        s['candidate_full'][941101]['validation_intact']['ridge'] = .88
        s['candidate_full'][941102]['validation_intact']['ridge'] = 1.
        out = gate.assess(s, q, (941101, 941102))
        self.assertFalse(out['passed'])
        self.assertTrue(any(c['condition'] == 'complete_cohort_loss/validation_intact/ridge'
                            and c['master'] == 941101 and not c['passed'] for c in out['conditions']))

    def test_missing_coverage_fails(self):
        s, q = fixture()
        s['candidate_learned'][941101]['validation_deleted']['coverage'] = .99
        self.assertFalse(gate.assess(s, q, (941101, 941102))['passed'])

    def test_query_guard_separates_roles(self):
        s, q = fixture()
        q[941101]['training']['candidate'] = 1.
        out = gate.assess(s, q, (941101, 941102))
        self.assertFalse(out['passed'])
        self.assertIn('query_cohort_ratio/training', out['failed_conditions'])

    def test_missing_or_nonfinite_populations_rejected(self):
        s, q = fixture()
        s['candidate_learned'][941101]['validation_intact']['ridge'] = float('nan')
        with self.assertRaises(ValueError):
            gate.assess(s, q, (941101, 941102))
        s, q = fixture()
        del s['reference_learned'][941102]
        with self.assertRaises(ValueError):
            gate.assess(s, q, (941101, 941102))


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Artificial mutation witnesses; no quality inputs, model execution or fitting."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('branch_publisher_tests', Path(__file__).with_name('publish.py'))
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def item(dtype, shape, values):
    return dict(dtype=dtype, shape=shape, values=list(values))


def text(value):
    return item('ByteStorage', [len(value.encode())], value.encode())


def integer(value):
    return item('LongStorage', [], [value])


class BranchAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.width, self.master = 4, 12, 941000
        self.labels = [0, 1, 0, 1]
        # Real shuffled split roles retain sorted, noncontiguous source IDs;
        # TRAIN does not start at zero and VALIDATION does not start at256.
        self.packed = [(self.master << 31) + index * 2 for index in (1, 1, 3, 3)]
        ids = json.dumps(['TEMPO-5/' + str(value) for value in self.packed], separators=(',', ':'))
        self.original = dict(observed=item('DoubleStorage', [4, 3, 128, 9], [0.] * (4 * 3 * 128 * 9)),
                             feature_mask=item('BoolStorage', [4, 3, 128, 9], [1] * (4 * 3 * 128 * 9)),
                             labels_scoring_only=item('LongStorage', [4], self.labels), source_ids_json=text(ids),
                             source_ids=item('LongStorage', [4], self.packed))
        all_values = [((b * 32 + d) % 17 - 8) / 8 for b in range(4) for d in range(32)]
        self.export = dict(training_values=item('DoubleStorage', [4, 32], all_values),
                           training_valid=item('BoolStorage', [4], [1] * 4))
        values = [all_values[b * 32 + d] for b in range(4) for d in range(20, 32)]
        self.surface = dict(features=item('DoubleStorage', [4, 12], values), valid=item('BoolStorage', [4], [1] * 4),
                            labels_scoring_only=self.original['labels_scoring_only'], source_ids_json=self.original['source_ids_json'],
                            provenance=text('artificial branch audit fixture'))
        self.info = self.audit_surface()
        mean, scale = publisher.moments(values, 4, 12)
        prepared = publisher.normalize(values, mean, scale, 12)
        probe_mean, probe_scale = publisher.moments(prepared, 4, 12)
        weights = [(i % 5 - 2) * .02 for i in range(24)]
        w1 = [(i % 7 - 3) * .01 for i in range(12 * 16)]
        w2 = [(i % 3 - 1) * .03 for i in range(16 * 2)]
        self.fit = dict(feature_mean=item('DoubleStorage', [12], mean), feature_scale=item('DoubleStorage', [12], scale),
                        outer_normalizer_applied=item('BoolStorage', [], [1]), outer_fitted_rows=integer(4), fitted_rows=integer(4),
                        ridge_mean=item('DoubleStorage', [12], probe_mean), ridge_scale=item('DoubleStorage', [12], probe_scale),
                        ridge_weights=item('DoubleStorage', [12, 2], weights), ridge_intercept=item('DoubleStorage', [2], [.01, -.01]),
                        tiny_mean=item('DoubleStorage', [12], probe_mean), tiny_scale=item('DoubleStorage', [12], probe_scale),
                        tiny_w1=item('DoubleStorage', [12, 16], w1), tiny_b1=item('DoubleStorage', [16], [.01] * 16),
                        tiny_w2=item('DoubleStorage', [16, 2], w2), tiny_b2=item('DoubleStorage', [2], [.005, -.005]),
                        actual_probe_seed_decimal=text(str(publisher.stream_seed(2701, 12))), training_source_ids_json=text(ids),
                        ridge_penalty=item('DoubleStorage', [], [1.]), tiny_hidden=integer(16), tiny_steps=integer(100),
                        tiny_learning_rate=item('DoubleStorage', [], [.01]))
        normalized = publisher.normalize(prepared, probe_mean, probe_scale, 12)
        ridge = publisher.linear(normalized, 4, 12, weights, 2, [.01, -.01])
        hidden = publisher.linear(normalized, 4, 12, w1, 16, [.01] * 16)
        tiny = publisher.linear([publisher.math.tanh(v) for v in hidden], 4, 16, w2, 2, [.005, -.005])
        self.prediction = dict(probe_input_features=item('DoubleStorage', [4, 12], prepared),
                               ridge_logits=item('DoubleStorage', [4, 2], ridge),
                               tiny_hidden_preactivation=item('DoubleStorage', [4, 16], hidden),
                               tiny_logits=item('DoubleStorage', [4, 2], tiny),
                               ridge=item('LongStorage', [4], publisher.decisions(ridge, 4)),
                               tiny_secondary=item('LongStorage', [4], publisher.decisions(tiny, 4)),
                               labels_scoring_only=item('LongStorage', [4], self.labels), valid=item('BoolStorage', [4], [1] * 4),
                               source_ids_json=text(ids))
        self.metadata = dict(population=self.info['population'],
                             **{h: publisher.score(self.prediction[h]['values'], self.labels, [1] * 4)
                                for h in ('ridge', 'tiny_secondary')})

    def audit_surface(self):
        return publisher.audit_surface(self.surface, self.original, self.export, 'training', 4, 12, 20, self.master)

    def audit_prediction(self):
        return publisher.audit_prediction(self.prediction, self.info, self.fit, self.metadata)

    def test_exact_branch_and_persisted_arithmetic(self):
        publisher.audit_fit(self.fit, self.info, 2701)
        self.assertEqual(self.audit_prediction()['coverage'], 1)

    def test_noncontiguous_validation_source_indices(self):
        packed = [(self.master << 31) + index * 2 for index in (0, 0, 12, 12)]
        self.original['source_ids'] = item('LongStorage', [4], packed)
        self.original['source_ids_json'] = text(json.dumps(['TEMPO-5/' + str(value) for value in packed], separators=(',', ':')))
        self.assertEqual(publisher.source_rows(self.original, 4, self.master)[0], self.labels)

    def test_source_text_disagrees_with_packed_identity(self):
        self.original['source_ids_json'] = text('["TEMPO-5/wrong0","TEMPO-5/wrong0","TEMPO-5/wrong1","TEMPO-5/wrong1"]')
        with self.assertRaisesRegex(ValueError, 'source text matches packed identity'):
            publisher.source_rows(self.original, 4, self.master)

    def test_original_source_pair_order_rejected(self):
        self.original['source_ids'] = item('LongStorage', [4], list(reversed(self.packed)))
        with self.assertRaisesRegex(ValueError, 'ordered noncontiguous original source domain'):
            publisher.source_rows(self.original, 4, self.master)

    def test_shifted_branch_values_rejected(self):
        self.surface['features']['values'][3] += .25
        with self.assertRaisesRegex(ValueError, 'exact original native branch slice'):
            self.audit_surface()

    def test_wrong_offset_same_width_rejected(self):
        values = self.export['training_values']['values']
        self.surface['features']['values'] = [values[b * 32 + d] for b in range(4) for d in range(19, 31)]
        with self.assertRaisesRegex(ValueError, 'exact original native branch slice'):
            self.audit_surface()

    def test_source_order_rejected(self):
        self.surface['source_ids_json'] = text(json.dumps(['engineering/train-1'] * 2 + ['engineering/train-0'] * 2))
        with self.assertRaisesRegex(ValueError, 'branch source labels/order'):
            self.audit_surface()

    def test_undeclared_latent_surface_field_rejected(self):
        self.surface['phase_truth'] = item('DoubleStorage', [4], [0.] * 4)
        with self.assertRaisesRegex(ValueError, 'closed branch surface schema'):
            self.audit_surface()

    def test_odd_packed_source_identity_rejected(self):
        self.original['source_ids']['values'][0] += 1
        with self.assertRaisesRegex(ValueError, 'cohort packed source identity'):
            self.audit_surface()

    def test_changed_support_rejected(self):
        self.surface['valid']['values'][0] = 0
        with self.assertRaisesRegex(ValueError, 'unchanged original branch support'):
            self.audit_surface()

    def test_wrong_train_normalizer_rejected(self):
        self.fit['feature_mean']['values'][0] += .5
        with self.assertRaisesRegex(ValueError, 'TRAIN outer mean'):
            publisher.audit_fit(self.fit, self.info, 2701)

    def test_wrong_width_paired_seed_rejected(self):
        self.fit['actual_probe_seed_decimal'] = text(str(publisher.stream_seed(2701, 32)))
        with self.assertRaisesRegex(ValueError, 'declared width-paired probe seed'):
            publisher.audit_fit(self.fit, self.info, 2701)

    def test_validation_sources_in_fit_rejected(self):
        self.fit['training_source_ids_json'] = text('["engineering/validation-0"]')
        with self.assertRaisesRegex(ValueError, 'TRAIN-only fit source association'):
            publisher.audit_fit(self.fit, self.info, 2701)

    def test_changed_hidden_arithmetic_rejected(self):
        self.prediction['tiny_hidden_preactivation']['values'][0] += .001
        with self.assertRaisesRegex(ValueError, 'independent neural hidden arithmetic'):
            self.audit_prediction()

    def test_changed_decision_rejected_even_with_matching_score(self):
        self.prediction['ridge']['values'][0] ^= 1
        self.metadata['ridge'] = publisher.score(self.prediction['ridge']['values'], self.labels, [1] * 4)
        with self.assertRaisesRegex(ValueError, 'independent saved head decisions'):
            self.audit_prediction()

    def test_wrong_report_score_rejected(self):
        self.metadata['ridge']['accuracy'] += .25
        with self.assertRaisesRegex(ValueError, 'reported head score accuracy'):
            self.audit_prediction()

    def test_nonfinite_fit_value_rejected(self):
        self.fit['tiny_w1']['values'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite saved tensor tiny_w1'):
            publisher.audit_fit(self.fit, self.info, 2701)


class PairedAuditTests(unittest.TestCase):
    def setUp(self):
        labels, valid = [i % 2 for i in range(256)], [1] * 256
        ids = ['engineering/source-' + str(i // 2) for i in range(256)]
        pop = publisher.population(labels, ids, valid)
        self.pairs, self.replay, self.predictions = [], {}, {}
        for rep in ('rep-2701', 'rep-2802', 'rep-2903'):
            for view in ('validation_intact', 'validation_deleted'):
                interval = dict(source_groups=128, replicates=1000, confidence=.95, estimate=.5, lower=.45, upper=.55)
                pair = dict(status='measured', id='learned_20_512_minus_learned_20_0', repetition=rep,
                            view=view, bootstrap_seed_decimal='123', common_population=pop,
                            ridge=copy.deepcopy(interval), tiny_secondary=copy.deepcopy(interval))
                self.pairs.append(pair)
                key = (941000, pair['id'], rep, view)
                self.replay[key] = dict(valid_rows=256, bootstrap_seed_decimal='123',
                                        ridge=copy.deepcopy(interval), tiny_secondary=copy.deepcopy(interval))
                for part, predictions in [('learned_20_0', [0] * 256), ('learned_20_512', labels)]:
                    self.predictions[part, rep, view] = dict(labels=labels, valid=valid, population=pop,
                                                           ridge=predictions, tiny_secondary=predictions)

    def test_all_declared_effects(self):
        checked = set()
        publisher.audit_pairs(self.pairs, 941000, self.replay, checked, self.predictions)
        self.assertEqual(checked, set(self.replay))

    def test_changed_paired_estimate_rejected(self):
        self.pairs[0]['ridge']['estimate'] = .9
        with self.assertRaisesRegex(ValueError, 'paired learned training effect'):
            publisher.audit_pairs(self.pairs, 941000, self.replay, set(), self.predictions)

    def test_changed_paired_interval_rejected(self):
        self.pairs[0]['ridge']['upper'] = .99
        with self.assertRaisesRegex(ValueError, 'independently replayed paired interval upper'):
            publisher.audit_pairs(self.pairs, 941000, self.replay, set(), self.predictions)

    def test_duplicate_comparison_rejected(self):
        self.pairs[1] = copy.deepcopy(self.pairs[0])
        with self.assertRaisesRegex(ValueError, 'declared unique learned training comparison'):
            publisher.audit_pairs(self.pairs, 941000, self.replay, set(), self.predictions)


class SourcePartitionTests(unittest.TestCase):
    def setUp(self):
        base = 941000 << 31
        # Noncontiguous role membership covers the common source domain. Zero
        # belongs to VALIDATION and the largest source belongs to TRAIN.
        training = list(range(1, 256)) + [383]
        validation = [0] + list(range(256, 383))
        self.training = [base + index * 2 for index in training for _ in range(2)]
        self.validation = [base + index * 2 for index in validation for _ in range(2)]
        self.deleted = list(self.validation)

    def test_shuffled_disjoint_roles_are_accepted(self):
        publisher.audit_source_partition(self.training, self.validation, self.deleted)

    def test_overlapping_roles_rejected(self):
        self.validation[:2] = self.training[:2]
        self.deleted[:2] = self.training[:2]
        with self.assertRaisesRegex(ValueError, 'shuffled source roles disjoint'):
            publisher.audit_source_partition(self.training, self.validation, self.deleted)

    def test_deleted_row_order_rejected(self):
        self.deleted = list(reversed(self.deleted))
        with self.assertRaisesRegex(ValueError, 'shuffled source roles disjoint'):
            publisher.audit_source_partition(self.training, self.validation, self.deleted)

    def test_swapped_complete_roles_rejected(self):
        with self.assertRaisesRegex(ValueError, 'declared source role sizes'):
            publisher.audit_source_partition(self.validation, self.training, self.training)

    def test_swapped_original_file_bytes_rejected(self):
        training, validation = Path('/original/training.pt'), Path('/original/validation.pt')
        inventory = {training: 'a' * 64, validation: 'b' * 64}
        publisher.historical_digest(training, 'a' * 64, inventory)
        with self.assertRaisesRegex(ValueError, 'original archived role bytes unchanged'):
            publisher.historical_digest(training, 'b' * 64, inventory)


if __name__ == '__main__':
    unittest.main()

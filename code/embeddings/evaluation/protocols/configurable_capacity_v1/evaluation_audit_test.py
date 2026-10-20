#!/usr/bin/env python3
"""Artificial saved-query mutation tests; no models, heads or quality inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).resolve().parent
publisher = module('capacity_publisher_for_test', HERE / 'publish.py')
publisher.SCALER_AUDIT = module('capacity_scaler_for_test', HERE / 'saved_audit.py')


def item(dtype, shape, values):
    return dict(dtype=dtype, shape=shape, values=values)


def text(value):
    return item('ByteStorage', [len(value)], list(value.encode()))


class QueryAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows, cells = 8, 8 * 3 * 128 * 9
        self.binding = dict(protocol='configurable-capacity-v1', dataset_id='TEMPO-5',
                            profile='A', model_tag='RPB-v20.alt-01', master=941000,
                            native_width=32, source_fingerprint='engineering-only',
                            card_sha='engineering-only', input_manifest_sha='engineering-only',
                            config_sha='engineering-only', scaler_identity='engineering-only', point=512)
        ids = list(range(self.rows))
        ids_json = json.dumps(['engineering/' + str(i) for i in ids], separators=(',', ':'))
        self.original = dict(observed=item('DoubleStorage', [8, 3, 128, 9], [1.5] * cells),
                             feature_mask=item('BoolStorage', [8, 3, 128, 9], [1] * cells),
                             source_ids=item('LongStorage', [8], ids), source_ids_json=text(ids_json))
        self.query = {key: text(value) if isinstance(value, str) else item('LongStorage', [], [value])
                      for key, value in self.binding.items()}
        self.query.update(artifact_kind=text('configurable_capacity_compact_original_query_bank_v1'),
                          standardized_prediction=item('DoubleStorage', [8, 3, 128, 9], [.625] * cells),
                          standardized_target=item('DoubleStorage', [8, 3, 128, 9], [.5] * cells),
                          target_mask=item('BoolStorage', [8, 3, 128, 9], [1] * cells),
                          original_observed_mask=self.original['feature_mask'],
                          requested_observed_target_mask=self.original['feature_mask'],
                          source_ids=self.original['source_ids'], source_ids_json=self.original['source_ids_json'],
                          trial_channel_eligible=item('BoolStorage', [16, 8, 3], [1] * (16 * 8 * 3)),
                          trial_index_for_time=item('LongStorage', [128], [h // 8 for h in range(128)]),
                          patch_length=item('LongStorage', [], [8]),
                          channel_target_counts=item('LongStorage', [8, 3], [1152] * 24),
                          example_standardized_mae=item('DoubleStorage', [8], [.125] * 8))
        self.metadata = dict(standardized_mae=.125, coverage=1, valid_target_cells=cells,
                             CPU_encoder_calls=0, CUDA_forward_calls=16)

    def audit(self):
        query, original = self.query, self.original
        class Codec:
            @staticmethod
            def load(path):
                return query if str(path) == 'query' else original
            @staticmethod
            def text(value):
                return bytes(value['values']).decode()
        old = publisher.remember
        publisher.remember = lambda path: None
        try:
            return publisher.query_error(Codec, Path('query'), Path('original'), self.metadata,
                                         512, 8, ([1.] * 27, [1.] * 27), self.binding)
        finally:
            publisher.remember = old

    def test_legal_original_targets(self):
        self.assertEqual(self.audit(), .125)

    def test_coordinated_wrong_target_prediction(self):
        for key in ('standardized_target', 'standardized_prediction'):
            self.query[key]['values'] = [v + .25 for v in self.query[key]['values']]
        # The error stays exactly .125; only original-target derivation catches it.
        with self.assertRaisesRegex(ValueError, 'query target comes from original'):
            self.audit()

    def test_wrong_cohort_binding(self):
        self.query['master'] = item('LongStorage', [], [941001])
        with self.assertRaisesRegex(ValueError, 'fixed query binding master'):
            self.audit()

    def test_wrong_source_rows(self):
        self.query['source_ids'] = item('LongStorage', [8], list(reversed(range(8))))
        with self.assertRaisesRegex(ValueError, 'original query context/source/point binding'):
            self.audit()

    def test_exact_float32_cast_is_required(self):
        # Stored original values are double; model targets must follow the
        # explicit float32 conversion, not unrounded double normalization.
        self.original['observed']['values'][0] = 1.50000001
        self.assertEqual(self.audit(), .125)  # rounds to exactly .5
        self.query['standardized_target']['values'][0] = .50000001
        with self.assertRaisesRegex(ValueError, 'query target comes from original'):
            self.audit()


class PairedAuditTests(unittest.TestCase):
    def setUp(self):
        interval = dict(source_groups=128, estimate=.01, lower=-.01, upper=.03)
        self.pair = dict(status='measured', id='native_B_512_minus_native_A_512',
                         repetition='rep-2701', view='validation_intact',
                         bootstrap_seed_decimal='123', common_population=dict(valid_rows=256),
                         ridge=copy.deepcopy(interval), tiny_secondary=copy.deepcopy(interval))
        self.key = (941000, self.pair['id'], self.pair['repetition'], self.pair['view'])
        self.replay = {self.key: dict(bootstrap_seed_decimal='123', valid_rows=256,
                                     ridge=copy.deepcopy(interval), tiny_secondary=copy.deepcopy(interval))}

    def test_legal_comparison(self):
        checked = set()
        publisher.audit_pairs([self.pair], 941000, self.replay, checked)
        self.assertEqual(checked, {self.key})

    def test_wrong_difference_interval(self):
        self.pair['ridge']['upper'] = .99
        with self.assertRaisesRegex(ValueError, 'paired comparison replay upper'):
            publisher.audit_pairs([self.pair], 941000, self.replay, set())

    def test_missing_or_duplicate_replay(self):
        with self.assertRaisesRegex(ValueError, 'every declared paired comparison replayed once'):
            publisher.audit_pairs([self.pair], 941000, {}, set())
        with self.assertRaisesRegex(ValueError, 'every declared paired comparison replayed once'):
            publisher.audit_pairs([self.pair, self.pair], 941000, self.replay, set())


class CheckpointMetadataTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = dict(
            artifact_kind=text('rpb_configurable_inference_checkpoint_v1'), model_tag=text('RPB-v20'),
            architecture_id=text('early-configurable-shape-plus-fixed-two-band-complex-relations-v1'),
            native_layout=text('learned-D-minus-four-per-semantic-pair;fixed-four-per-semantic-pair;'
                               'canonical-all-channel-pairs;ReLow-ImLow-ReHigh-ImHigh;'
                               'q1-3-low-q4-8-high;original-H;feature-pair-mean;pair-unit-L2-floor1e-12'),
            dtype=text('float32'), training_device=text('cuda:0'),
            training_updates=item('LongStorage', [], [512]),
            config_integer_v1=item('LongStorage', [15], [3, 128, 9, 8, 64, 32, 3, 4, 1, 1, 2, 0, 0, 256, 128]),
            config_number_v1=item('DoubleStorage', [6], [0., 1e-5, .25, 1., 1e-6, 1.]),
            channel_ids=item('LongStorage', [3], [0, 1, 2]))

    def audit(self, checkpoint=None):
        class Codec:
            @staticmethod
            def text(value):
                return bytes(value['values']).decode()
        publisher.checkpoint_settings(Codec, checkpoint or self.checkpoint, 'A', 32, 512)

    def test_exact_declared_checkpoint_metadata(self):
        self.audit()

    def test_every_numeric_setting_is_bound(self):
        for coordinate in range(6):
            with self.subTest(coordinate=coordinate):
                changed = copy.deepcopy(self.checkpoint)
                changed['config_number_v1']['values'][coordinate] += .01
                with self.assertRaisesRegex(ValueError, 'frozen checkpoint dropout epsilon'):
                    self.audit(changed)

    def test_every_schema_text_is_bound(self):
        for key in ('artifact_kind', 'model_tag', 'architecture_id', 'native_layout', 'dtype', 'training_device'):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.checkpoint)
                changed[key] = text('wrong-schema')
                with self.assertRaisesRegex(ValueError, 'frozen checkpoint schema text ' + key):
                    self.audit(changed)

    def test_config_float_vector_schema_is_bound(self):
        self.checkpoint['config_number_v1']['dtype'] = 'FloatStorage'
        with self.assertRaisesRegex(ValueError, 'saved tensor schema config_number_v1'):
            self.audit()


class InformationAdmissionTests(unittest.TestCase):
    """Opaque artificial files exercise the scientific pre-decoding boundary."""
    def setUp(self):
        self.audit = module('capacity_information_admission_tests', HERE / 'capsule.py')
        base = Path('/opt/mityane-geometrics/migration/scientific-fixtures/capacity-information')
        if not Path('/.dockerenv').is_file():
            raise RuntimeError('Use the managed Linux container for scientific fixtures')
        base.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='case-', dir=base)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = [self.root / f'master-{master}/{view}{suffix}.pt'
                      for master in self.audit.MASTERS
                      for view in ('training', 'validation', 'validation-deleted')
                      for suffix in ('', '-information')]
        for path in self.paths:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'opaque engineering fixture; never decoded')

    def test_missing_last_role_is_rejected_without_payload_opens(self):
        self.paths[-1].unlink()
        with patch.object(Path, 'open', side_effect=AssertionError('payload opened')):
            with self.assertRaises(FileNotFoundError):
                self.audit._admit_information_files(self.root)

    def test_redirect_and_alias_roles_are_rejected_before_decoding(self):
        last = self.paths[-1]
        last.unlink()
        last.symlink_to(self.paths[0])
        with self.assertRaisesRegex(ValueError, 'Direct absolute'):
            self.audit._admit_information_files(self.root)
        last.unlink()
        last.hardlink_to(self.paths[0])
        with self.assertRaisesRegex(ValueError, 'Single-link'):
            self.audit._admit_information_files(self.root)

    def test_full_matrix_is_metadata_only_and_detects_same_byte_replacement(self):
        with patch.object(Path, 'open', side_effect=AssertionError('payload opened')):
            rows = self.audit._admit_information_files(self.root)
        self.assertEqual(set(rows), set(self.paths))
        self.assertEqual(len(rows), 18)
        selected = self.paths[0]
        payload = selected.read_bytes()
        selected.rename(self.root / 'prior-opaque-role')
        selected.write_bytes(payload)
        self.assertNotEqual(rows[selected], self.audit._information_signature(selected))


if __name__ == '__main__':
    unittest.main()

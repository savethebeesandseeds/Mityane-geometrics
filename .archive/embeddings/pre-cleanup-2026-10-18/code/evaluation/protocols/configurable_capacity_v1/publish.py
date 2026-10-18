#!/usr/bin/env python3
"""Audit persisted scores and emit the agreed capacity tables; never fit or run an encoder.

The producer runs exclusively on CUDA. The separate C++ reader fits ordinary
CPU readouts and independently replays all saved arithmetic, TRAIN maps, raw
PCA transforms and whole-source intervals. This script checks those witnesses
against the shared report, retaining every master and repetition in aggregates.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/embedding')
CODEC = ROOT / 'code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py'
CODEC_SHA = '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d'
READS = {}


def check(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remember(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(), 'direct saved path')
    digest = sha(path)
    check(path not in READS or READS[path] == digest, 'saved file changed while auditing')
    READS[path] = digest
    return digest


def document(path):
    remember(path)
    return json.loads(path.read_text())


def close(actual, expected, why):
    if expected is None:
        check(actual is None, why)
    else:
        check(actual is not None and math.isfinite(actual) and abs(actual - expected) <= 2e-9 * max(1, abs(expected)), why)


def average(values):
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None


def pct(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def mean_spread(values):
    value = average(values)
    return '—' if value is None else f'{100*value:.1f}% ({100*min(values):.1f}–{100*max(values):.1f})'


def tensor(archive, name, dtype, shape):
    item = archive[name]
    check(item['dtype'] == dtype and item['shape'] == shape, 'saved tensor schema ' + name)
    return list(item['values'])


def audit_pairs(pairs, master, replay, checked):
    for pair in pairs:
        if pair['status'] == 'unsupported_fit':
            continue
        key = (master, pair['id'], pair['repetition'], pair['view'])
        check(key in replay and key not in checked, 'every declared paired comparison replayed once')
        checked.add(key)
        actual = replay[key]
        check(pair['bootstrap_seed_decimal'] == actual['bootstrap_seed_decimal']
              and pair['common_population']['valid_rows'] == actual['valid_rows'], 'paired bootstrap seed/common support')
        for head in ('ridge', 'tiny_secondary'):
            check(pair[head]['source_groups'] == actual[head]['source_groups'], 'paired source-group population')
            for field in ('estimate', 'lower', 'upper'):
                close(pair[head][field], actual[head][field], 'paired comparison replay ' + field)


def checkpoint_settings(codec, checkpoint, profile, width, point):
    expected_text = dict(
        artifact_kind='rpb_configurable_inference_checkpoint_v1', model_tag='RPB-v20',
        architecture_id='early-configurable-shape-plus-fixed-two-band-complex-relations-v1',
        native_layout=('learned-D-minus-four-per-semantic-pair;fixed-four-per-semantic-pair;'
                       'canonical-all-channel-pairs;ReLow-ImLow-ReHigh-ImHigh;'
                       'q1-3-low-q4-8-high;original-H;feature-pair-mean;pair-unit-L2-floor1e-12'),
        dtype='float32', training_device='cuda:0')
    for key, expected in expected_text.items():
        check(codec.text(checkpoint[key]) == expected, 'frozen checkpoint schema text ' + key)
    check(tensor(checkpoint, 'training_updates', 'LongStorage', []) == [point]
          and tensor(checkpoint, 'config_integer_v1', 'LongStorage', [15]) ==
              [3, 128, 9, 8, 128 if profile == 'B' else 64, width, 3, 4, 1, 1, 2, 0, 0, 256, 128],
          'frozen checkpoint point and resolved integer geometry')
    check(tensor(checkpoint, 'config_number_v1', 'DoubleStorage', [6]) ==
              [0., 1e-5, .25, 1., 1e-6, 1.],
          'frozen checkpoint dropout epsilon masking Huber scaler floor and sampling interval')
    check(tensor(checkpoint, 'channel_ids', 'LongStorage', [3]) == [0, 1, 2],
          'frozen checkpoint semantic channel identity')


def query_error(codec, path, input_path, metadata, point, rows, scaler, binding):
    remember(path)
    remember(input_path)
    query, original = codec.load(path), codec.load(input_path)
    for key, expected in binding.items():
        if isinstance(expected, str):
            check(codec.text(query[key]) == expected, 'fixed query binding ' + key)
        else:
            check(tensor(query, key, 'LongStorage', []) == [expected], 'fixed query binding ' + key)
    check(codec.text(query['artifact_kind']) == 'configurable_capacity_compact_original_query_bank_v1',
          'fixed query artifact schema')
    prediction = tensor(query, 'standardized_prediction', 'DoubleStorage', [rows, 3, 128, 9])
    target = tensor(query, 'standardized_target', 'DoubleStorage', [rows, 3, 128, 9])
    mask = tensor(query, 'target_mask', 'BoolStorage', [rows, 3, 128, 9])
    original_mask = tensor(original, 'feature_mask', 'BoolStorage', [rows, 3, 128, 9])
    original_values = tensor(original, 'observed', 'DoubleStorage', [rows, 3, 128, 9])
    means, scales = scaler
    # The input scaler performs CPU float64 arithmetic and then casts to CUDA
    # float32. Targets are that float32 value exported back as float64. Require
    # this exact cast, rather than widening the reconstruction error tolerance.
    for i, (actual, supported) in enumerate(zip(target, mask)):
        coordinate = (i // 1152 % 3) * 9 + i % 9
        expected = SCALER_AUDIT.float32((original_values[i] - means[coordinate]) / scales[coordinate]) if supported else 0
        check(actual == expected, 'query target comes from original observation and frozen TRAIN scaler')
    check(tensor(query, 'original_observed_mask', 'BoolStorage', [rows, 3, 128, 9]) == original_mask
          and tensor(query, 'requested_observed_target_mask', 'BoolStorage', [rows, 3, 128, 9]) == original_mask,
          'fixed query belongs to exact original legal observations')
    check(codec.text(query['source_ids_json']) == codec.text(original['source_ids_json'])
          and tensor(query, 'source_ids', 'LongStorage', [rows]) == tensor(original, 'source_ids', 'LongStorage', [rows])
          and tensor(query, 'point', 'LongStorage', []) == [point]
          and tensor(query, 'trial_index_for_time', 'LongStorage', [128]) == [h // 8 for h in range(128)]
          and tensor(query, 'patch_length', 'LongStorage', []) == [8], 'original query context/source/point binding')
    eligibility = tensor(query, 'trial_channel_eligible', 'BoolStorage', [16, rows, 3])
    wanted, channel_counts = [], []
    for b in range(rows):
        channels = []
        for c in range(3):
            start, end = (b * 3 + c) * 1152, (b * 3 + c + 1) * 1152
            observed_patches = [any(original_mask[start + p * 72:start + (p + 1) * 72]) for p in range(16)]
            for p in range(16):
                expected = bool(observed_patches[p] and sum(observed_patches) - observed_patches[p] >= 2)
                check(bool(eligibility[(p * rows + b) * 3 + c]) == expected, 'two visible patches and one target patch query eligibility')
                check(mask[start + p * 72:start + (p + 1) * 72] == [int(bool(v) and expected) for v in original_mask[start + p * 72:start + (p + 1) * 72]], 'exact fixed query target cells')
            count = sum(mask[start:end])
            channel_counts.append(count)
            error = math.fsum(abs(p - t) for p, t, valid in zip(prediction[start:end], target[start:end], mask[start:end]) if valid)
            channels.append(error / count if count else None)
        wanted.append(average([v for v in channels if v is not None]))
    check(tensor(query, 'channel_target_counts', 'LongStorage', [rows, 3]) == channel_counts, 'original query target populations')
    check(all(math.isfinite(p) and math.isfinite(t) and (bool(v) or p == t == 0)
              for p, t, v in zip(prediction, target, mask)), 'finite original query and zero untargeted storage')
    saved = tensor(query, 'example_standardized_mae', 'DoubleStorage', [rows])
    for actual, expected in zip(saved, wanted):
        close(actual, expected if expected is not None else 0, 'original query hierarchical standardized MAE')
    valid_errors = [v for v in wanted if v is not None]
    close(metadata['standardized_mae'], average(valid_errors), 'reported standardized fixed query MAE')
    close(metadata['coverage'], len(valid_errors) / rows, 'fixed query coverage')
    check(metadata['valid_target_cells'] == sum(channel_counts) and metadata['CPU_encoder_calls'] == 0
          and metadata['CUDA_forward_calls'] == rows // 8 * 16, 'declared query populations and CUDA-only inference calls')
    return metadata['standardized_mae']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capsule', required=True, type=Path)
    parser.add_argument('--output-directory', type=Path,
                        help='new direct directory for a saved-only review; preserve original capsule outputs')
    parser.add_argument('--arithmetic-audit', type=Path,
                        help='new saved-only C++ replay including paired intervals')
    args = parser.parse_args()
    check(args.capsule.is_absolute() and args.capsule.resolve(strict=True) == args.capsule
          and args.capsule.is_relative_to(ROOT / 'output/runs/rpb-v20-capacity') and not args.capsule.is_symlink(), 'direct capacity capsule')
    args.readouts = args.capsule / 'readouts'
    args.producer_root = args.capsule
    destination = args.capsule
    if args.output_directory:
        destination = args.output_directory
        check(destination.is_absolute() and destination == destination.resolve(strict=False)
              and destination.parent.resolve(strict=True) == destination.parent
              and not destination.exists() and not destination.is_symlink(), 'new direct review directory')
        destination.mkdir()
    args.output = destination / 'summary.json'
    args.markdown = destination / 'report.md'
    for path in (args.output, args.markdown):
        check(path.is_absolute() and not path.exists() and path.parent.resolve(strict=True) == path.parent, 'new direct output')
    check(remember(CODEC) == CODEC_SHA, 'unchanged independently reviewed saved archive codec')
    spec = importlib.util.spec_from_file_location('capacity_saved_codec', CODEC)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    helper_path = Path(__file__).resolve().with_name('saved_audit.py')
    remember(helper_path)
    helper_spec = importlib.util.spec_from_file_location('capacity_saved_scaler_audit', helper_path)
    global SCALER_AUDIT
    SCALER_AUDIT = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(SCALER_AUDIT)
    report = document(args.readouts / 'report.json')
    complete = document(args.readouts / 'complete.json')
    audit = document(args.arithmetic_audit or args.readouts / 'arithmetic-audit.json')
    check(report['protocol'] == complete['protocol'] == 'configurable-capacity-v1'
          and complete['status'] == 'complete' and complete['report_sha256'] == sha(args.readouts / 'report.json'), 'completed report binding')
    check(report['dataset'] == 'TEMPO-5' and report['masters'] == [941101, 941102, 941103]
          and report['encoder_calls'] == report['native_PCA_fits'] == 0, 'closed challenge and saved-feature scope')
    check(audit['encoder_calls'] == audit['head_fits'] == 0
          and audit['fitted_pairs_replayed'] == complete['saved_arithmetic_fits_replayed'], 'saved arithmetic replay scope')
    replay = {r['prediction']: r for r in audit['records']}
    check(len(replay) == len(audit['records']), 'unique complete replay roles')
    pair_replay = {(r['master'], r['id'], r['repetition'], r['view']): r for r in audit['pairs']}
    check(len(pair_replay) == len(audit['pairs']), 'unique paired comparison replay roles')
    checked_pairs = set()
    expected_recipe = dict(ridge_penalty=1, tiny_hidden=16, tiny_steps=100, tiny_learning_rate=.01,
                          probe_seed_policy='stream_seed(repetition,width);same_equal_width_methods',
                          validation_fits=0, encoder_calls=0, pca_fits=0)
    collected = {}
    checked_predictions = 0
    for cohort in report['cohorts']:
        master = cohort['master']
        check(master in report['masters'], 'declared master')
        base = args.readouts / f'master-{master}'
        originals = {}
        for view, role in [('training', 'training'), ('validation_intact', 'validation'), ('validation_deleted', 'validation-deleted')]:
            original_path = args.capsule / 'quality-data' / f'master-{master}' / (role + '.pt')
            remember(original_path)
            originals[view] = codec.load(original_path)
        sections = [('raw', base / 'baselines/raw/readouts', cohort['raw']),
                    ('pca32', base / 'baselines/pca32/readouts', cohort['pca32']),
                    ('pca64', base / 'baselines/pca64/readouts', cohort['pca64']),
                    ('models', base / 'models', cohort['models'])]
        for section, directory, metadata in sections:
            check(document(directory / 'report.json') == metadata and metadata['recipe'] == expected_recipe
                  and metadata['master_seed'] == str(master), 'shared unchanged fixed recipe')
            for method in metadata['methods']:
                key = method['method'] if section == 'models' else section
                values = dict(master=master, width=method['size'], status=method['status'], views={})
                check(method['status'] in ('measured', 'unsupported_fit'), 'measured or retained unsupported fit')
                check(len(method['repetitions']) == 3 and [r['id'] for r in method['repetitions']] == ['rep-2701', 'rep-2802', 'rep-2903'], 'all fixed repetitions')
                method_root = directory / method['method']
                for view, population, rows in [('training', 'training_population', 512),
                                               ('validation_intact', 'validation_intact_population', 256),
                                               ('validation_deleted', 'validation_deleted_population', 256)]:
                    surface_path = method_root / (view.replace('_', '-') + '-features.pt')
                    remember(surface_path)
                    surface = codec.load(surface_path)
                    original = originals[view]
                    check(tensor(surface, 'labels_scoring_only', 'LongStorage', [rows])
                          == tensor(original, 'labels_scoring_only', 'LongStorage', [rows])
                          and codec.text(surface['source_ids_json']) == codec.text(original['source_ids_json']),
                          'readout surface scoring labels/source rows belong to original role')
                    if section == 'models':
                        _, profile, point_text = method['method'].split('_')
                        point = int(point_text)
                        producer_path = args.capsule / 'profiles' / profile / f'master-{master}' / f'point-{point}' / 'features.pt'
                        remember(producer_path)
                        producer_features = codec.load(producer_path)
                        prefix = dict(training='training', validation_intact='validation', validation_deleted='deleted')[view]
                        check(tensor(surface, 'features', 'DoubleStorage', [rows, method['size']])
                              == tensor(producer_features, prefix + '_values', 'DoubleStorage', [rows, method['size']])
                              and tensor(surface, 'valid', 'BoolStorage', [rows])
                              == tensor(producer_features, prefix + '_valid', 'BoolStorage', [rows]),
                              'readout native surface exact saved CUDA row association')
                        if prefix + '_source_ids' in producer_features:
                            check(tensor(producer_features, prefix + '_source_ids', 'LongStorage', [rows])
                                  == tensor(original, 'source_ids', 'LongStorage', [rows])
                                  and codec.text(producer_features[prefix + '_source_ids_json']) == codec.text(original['source_ids_json']),
                                  'explicit native export source row witness')
                    values['views'][view] = dict(coverage=method[population]['coverage'])
                    scores = {'ridge': [], 'tiny_secondary': []}
                    for repetition in method['repetitions']:
                        if method['status'] != 'measured':
                            scores['ridge'].append(None)
                            scores['tiny_secondary'].append(None)
                            continue
                        filename = view.replace('_', '-') + '-predictions.pt'
                        prediction_path = method_root / repetition['id'] / filename
                        relative = str(prediction_path.relative_to(args.readouts))
                        check(relative in replay, 'every saved prediction has arithmetic witness')
                        remember(prediction_path)
                        saved = codec.load(prediction_path)
                        labels = tensor(saved, 'labels_scoring_only', 'LongStorage', [rows])
                        valid = tensor(saved, 'valid', 'BoolStorage', [rows])
                        ids = json.loads(codec.text(saved['source_ids_json']))
                        check(labels == tensor(original, 'labels_scoring_only', 'LongStorage', [rows])
                              and codec.text(saved['source_ids_json']) == codec.text(original['source_ids_json'])
                              and valid == tensor(surface, 'valid', 'BoolStorage', [rows]),
                              'saved decisions original role labels/source/support association')
                        groups = set(ids)
                        check(len(groups) == rows // 2 and len(ids) == rows, 'source grouped scored population')
                        classes = [sum(bool(ok) and label == k for ok, label in zip(valid, labels)) for k in (0, 1)]
                        population_actual = dict(total_rows=rows, valid_rows=sum(valid), class_valid_rows=classes,
                                                 total_source_groups=len(groups), valid_source_groups=len({i for i, ok in zip(ids, valid) if ok}),
                                                 coverage=sum(valid) / rows)
                        check(method[population] == repetition[view]['population'] == population_actual, 'saved population counts/coverage')
                        for head in scores:
                            predictions = tensor(saved, head, 'LongStorage', [rows])
                            correct = sum(bool(ok) and p == y for p, y, ok in zip(predictions, labels, valid))
                            accuracy = correct / sum(valid) if sum(valid) else None
                            expected = dict(total=rows, valid=sum(valid), correct=correct, accuracy=accuracy,
                                            coverage=sum(valid)/rows, full_population_correctness=correct/rows)
                            for name, value in expected.items():
                                close(repetition[view][head][name], value, 'saved shared score ' + name)
                                close(replay[relative][head][name], value, 'arithmetic replay saved score ' + name)
                            scores[head].append(accuracy)
                            if view != 'training':
                                interval_key = 'ridge_grouped_interval' if head == 'ridge' else 'tiny_grouped_interval'
                                check(repetition[view]['bootstrap_seed_decimal'] == replay[relative]['bootstrap_seed_decimal'], 'whole source interval seed')
                                declared, actual = repetition[view][interval_key], replay[relative][interval_key]
                                check(declared['replicates'] == 1000 and declared['confidence'] == .95
                                      and declared['source_groups'] == actual['source_groups'], 'whole source interval population')
                                for name in ('estimate', 'lower', 'upper'):
                                    close(declared[name], actual[name], 'replayed whole source interval ' + name)
                        checked_predictions += 1
                    for head, scores_for_head in scores.items():
                        values['views'][view][head] = average(scores_for_head)
                        values['views'][view][head + '_repetitions'] = scores_for_head
                collected.setdefault(key, []).append(values)
            audit_pairs(metadata['pairs'], master, pair_replay, checked_pairs)
    check(checked_predictions == 3 * complete['saved_arithmetic_fits_replayed'] == len(replay), 'entire persisted prediction matrix audited')
    check(checked_pairs == set(pair_replay), 'entire saved paired comparison matrix audited')
    for feature in report['saved_feature_inputs']:
        check(remember(Path(feature['path'])) == feature['sha256'], 'saved CUDA feature bytes unchanged')
    summaries = []
    for key in ['raw', 'pca32', 'pca64', 'native_A_0', 'native_A_512', 'native_B_0', 'native_B_512', 'native_C_0', 'native_C_512']:
        panels = collected[key]
        check([p['master'] for p in panels] == report['masters'], 'every master retained in aggregate')
        summary = dict(method=key, width=panels[0]['width'], cohorts=panels, views={})
        for view in ('training', 'validation_intact', 'validation_deleted'):
            summary['views'][view] = {}
            for metric in ('ridge', 'tiny_secondary', 'coverage'):
                values = [p['views'][view][metric] for p in panels]
                summary['views'][view][metric] = dict(mean=average(values), master_means=values,
                    minimum=min(values) if all(v is not None for v in values) else None,
                    maximum=max(values) if all(v is not None for v in values) else None)
        summaries.append(summary)
    training = []
    for profile, number in [('A', 1), ('B', 2), ('C', 3)]:
        cohorts = []
        for master in report['masters']:
            base = args.producer_root / 'profiles' / profile / f'master-{master}'
            producer = document(base / 'report.json')
            progress = document(base / 'progress.json')
            done = document(base / 'complete.json')
            check(producer['progress'] == progress and done['status'] == 'complete'
                  and done['report_sha256'] == sha(base / 'report.json'), 'CUDA completed report/progress binding')
            for key in ('source_fingerprint', 'card_sha', 'input_manifest_sha'):
                check(producer[key] == progress[key] == report[key], 'CUDA/readout provenance ' + key)
            width = 64 if profile == 'C' else 32
            initial_path, final_path = [base / f'point-{point}' / 'features.pt' for point in (0, 512)]
            remember(initial_path)
            remember(final_path)
            initial, final = codec.load(initial_path), codec.load(final_path)
            expected_binding = dict(protocol='configurable-capacity-v1', dataset_id='TEMPO-5', profile=profile,
                model_tag=f'RPB-v20.alt-0{number}', master=master, native_width=width,
                source_fingerprint=report['source_fingerprint'], card_sha=report['card_sha'],
                input_manifest_sha=report['input_manifest_sha'], config_sha=producer['config_sha'])
            for point, features in [(0, initial), (512, final)]:
                for key, expected in dict(expected_binding, point=point).items():
                    if isinstance(expected, str):
                        check(codec.text(features[key]) == expected, 'native feature binding ' + key)
                    else:
                        check(tensor(features, key, 'LongStorage', []) == [expected], 'native feature binding ' + key)
            for prefix, rows in [('training', 512), ('validation', 256), ('deleted', 256)]:
                initial_view = tensor(initial, prefix + '_values', 'DoubleStorage', [rows, width])
                final_view = tensor(final, prefix + '_values', 'DoubleStorage', [rows, width])
                check(all(initial_view[b * width + d] == final_view[b * width + d]
                          for b in range(rows) for d in range(width - 12, width))
                      and tensor(initial, prefix + '_valid', 'BoolStorage', [rows]) == tensor(final, prefix + '_valid', 'BoolStorage', [rows]),
                      'independent unchanged fixed spectral coordinates/support for every view')
            initial_values = tensor(initial, 'training_values', 'DoubleStorage', [512, width])
            final_values = tensor(final, 'training_values', 'DoubleStorage', [512, width])
            differences = [abs(final_values[b * width + d] - initial_values[b * width + d])
                           for b in range(512) for d in range(width - 12)]
            movement = producer['learned_training_coordinate_movement']
            check(movement['changed_cells'] == sum(value > 0 for value in differences)
                  and movement['changed_cells'] > 0, 'saved learned native coordinates changed after training')
            close(movement['max_absolute_change'], max(differences), 'saved learned native movement maximum')
            check(progress['attempted'] == progress['completed'] == 512 and progress['skipped_attempts'] == 0
                  and progress['sampled_rows'] == 4096 and progress['training_device'] == 'cuda:0'
                  and progress['CPU_encoder_calls'] == 0 and progress['cuda_parameter_count'] == progress['parameter_count']
                  and progress['weights_changed'] and len(progress['losses']) == 512, 'all declared CUDA updates executed')
            for i, row in enumerate(progress['losses']):
                check(row['attempted'] == row['completed'] == i + 1 and row['target_cells'] > 0
                      and math.isfinite(row['total_loss']) and row['total_loss'] == row['waveform_loss']
                      and row['dynamics_loss'] == 0 and math.isfinite(row['gradient_norm']) and row['gradient_norm'] > 0
                      and row['context_requested'] == row['context_actual'] + row['context_restored'], 'finite contiguous original CUDA trace')
            errors = {}
            training_path = args.capsule / 'quality-data' / f'master-{master}' / 'training.pt'
            remember(training_path)
            original_training = codec.load(training_path)
            for point in (0, 512):
                checkpoint_path = base / f'point-{point}' / 'checkpoint.pt'
                remember(checkpoint_path)
                checkpoint, checkpoint_scaler = SCALER_AUDIT.checkpoint_scaler(codec, checkpoint_path)
                checkpoint_settings(codec, checkpoint, profile, width, point)
                identity = SCALER_AUDIT.scaler_identity(checkpoint_scaler)
                check(identity == codec.text(checkpoint['scaler_identity']) == progress['scaler_identity'], 'frozen checkpoint scaler content identity')
                scaler = SCALER_AUDIT.audit_train_scaler(codec, checkpoint_scaler, original_training, check, close, tensor)
                expected_binding['scaler_identity'] = identity
                witness_path = base / f'point-{point}' / 'scaler-witness.pt'
                if witness_path.exists():
                    remember(witness_path)
                    witness = codec.load(witness_path)
                    for key, expected in dict(expected_binding, point=point).items():
                        if isinstance(expected, str):
                            check(codec.text(witness[key]) == expected, 'CPU scaler witness binding ' + key)
                        else:
                            check(tensor(witness, key, 'LongStorage', []) == [expected], 'CPU scaler witness binding ' + key)
                    for key in ('mean', 'scale', 'count', 'channel_ids', 'floor_applied', 'scale_floor'):
                        check(witness[key]['dtype'] == checkpoint_scaler[key]['dtype']
                              and witness[key]['shape'] == checkpoint_scaler[key]['shape']
                              and witness[key]['raw'] == checkpoint_scaler[key]['raw'], 'CPU scaler witness exact checkpoint statistics')
                    check(codec.text(witness['training_source_ids_json']) == codec.text(original_training['source_ids_json'])
                          and tensor(witness, 'training_source_ids', 'LongStorage', [512]) == tensor(original_training, 'source_ids', 'LongStorage', [512]),
                          'CPU scaler witness original TRAIN source population')
                for role, rows in [('training', 512), ('validation', 256)]:
                    prefix = 'initial_' if point == 0 else ''
                    errors[prefix + role] = query_error(codec,
                        base / f'point-{point}' / ('reconstruction-' + role + '.pt'),
                        args.capsule / 'quality-data' / f'master-{master}' / (role + '.pt'),
                        producer[prefix + role + '_reconstruction'], point, rows, scaler,
                        dict(expected_binding, point=point))
            cohorts.append(dict(master=master, updates=512, train_error=errors['training'], validation_error=errors['validation'],
                                GPU_training_seconds=progress['training_seconds'],
                                peak_allocated_MiB=progress['training_peak_allocated_bytes'] / 1048576,
                                peak_reserved_MiB=progress['training_peak_reserved_bytes'] / 1048576,
                                parameter_count=progress['parameter_count'], trainable_parameter_count=progress['trainable_parameter_count'],
                                initial_train_error=errors['initial_training'], initial_validation_error=errors['initial_validation'],
                                first_minibatch_loss=progress['losses'][0]['total_loss'],
                                last_minibatch_loss=progress['losses'][-1]['total_loss'],
                                final32_minibatch_mean_loss=average([r['total_loss'] for r in progress['losses'][-32:]])))
        training.append(dict(model_tag=f'RPB-v20.alt-0{number}', profile=profile, cohorts=cohorts,
                            **{key: average([c[key] for c in cohorts]) for key in
                               ('train_error', 'validation_error', 'GPU_training_seconds', 'peak_allocated_MiB', 'final32_minibatch_mean_loss')}))
    result = dict(protocol=report['protocol'], dataset='TEMPO-5', designed_complexity=5, complexity_max=5,
                  stage='development', TEST_access=False, post_encoder_PCA=False, encoder_calls=0,
                  aggregation='mean of three fixed head repetitions within each master, then equal mean across all three masters; no discarded unsupported values',
                  predictions_checked=checked_predictions, saved_arithmetic_fits_replayed=complete['saved_arithmetic_fits_replayed'],
                  paired_comparisons_checked=len(checked_pairs), review_arithmetic_audit=str(args.arithmetic_audit) if args.arithmetic_audit else None,
                  source_fingerprint=report['source_fingerprint'], card_sha=report['card_sha'], input_manifest_sha=report['input_manifest_sha'],
                  summaries=summaries, training=training,
                  intervals='Full within-master, conditional-on-fitted-head source-group95% intervals retained in readouts and independently replayed; master mean ranges in compact table are not confidence intervals',
                  read_files=[dict(path=str(p), sha256=d) for p, d in sorted(READS.items())])
    legend = 'TEMPO-5 — complexity **5/5** (designed): three streams, 128 candles, nine structured features; infer slow lead/lag beneath faster patterns and unrelated structured signals. Natural missing values and contiguous candle gaps. TRAIN: 256 source pairs; VALIDATION: 128 source pairs per master. Three masters, three fixed heads per method.'
    lines = [legend, '',
             'RPB-v20.alt-01: A, internal width64, native32. RPB-v20.alt-02: B, internal width128, native32. RPB-v20.alt-03: C, internal width64, native64. Initial means zero encoder updates; trained means512. All encoder operations ran on CUDA.', '',
             'Linear head = ridge classifier. Neural head = fixed tanh16 classifier. Accuracy is conditional on valid rows. Coverage is the fraction scored. Each percentage is the mean across three masters and three fixed head repetitions; parentheses show the range of master means. Raw data includes visibility masks: 6,912 numbers, with13,826 linear-head or110,642 neural-head parameters. Equal-size methods use identical head initialization seeds.', '',
             'Intact VALIDATION, trained encoders:', '',
             '| Method | Size | Linear head % | Neural head % | Coverage % |',
             '|---|---:|---:|---:|---:|']
    names = dict(raw='Raw data — no encoder', pca32='PCA only — no encoder', pca64='PCA only — no encoder')
    for profile, number in [('A', 1), ('B', 2), ('C', 3)]:
        names[f'native_{profile}_0'] = f'RPB-v20.alt-0{number}'
        names[f'native_{profile}_512'] = f'RPB-v20.alt-0{number}'
    for s in summaries:
        if s['method'].endswith('_0'):
            continue
        v = s['views']['validation_intact']
        lines.append('| ' + ' | '.join([names[s['method']], str(s['width']), mean_spread(v['ridge']['master_means']), mean_spread(v['tiny_secondary']['master_means']), pct(v['coverage']['mean'])]) + ' |')
    lines += ['', legend, '', 'VALIDATION with an additional30% coordinate deletion; the intact TRAIN-fitted heads remain fixed:', '', '| Method | Size | Linear head % | Neural head % | Coverage % |', '|---|---:|---:|---:|---:|']
    for s in summaries:
        if s['method'].endswith('_0'):
            continue
        v = s['views']['validation_deleted']
        lines.append('| ' + ' | '.join([names[s['method']], str(s['width']), mean_spread(v['ridge']['master_means']), mean_spread(v['tiny_secondary']['master_means']), pct(v['coverage']['mean'])]) + ' |')
    lines += ['', legend, '', 'Initial encoder exports, zero updates, intact VALIDATION:', '', '| Method | Size | Linear head % | Neural head % | Coverage % |', '|---|---:|---:|---:|---:|']
    for s in summaries:
        if not s['method'].endswith('_0'):
            continue
        v = s['views']['validation_intact']
        lines.append('| ' + ' | '.join([names[s['method']], str(s['width']), mean_spread(v['ridge']['master_means']), mean_spread(v['tiny_secondary']['master_means']), pct(v['coverage']['mean'])]) + ' |')
    lines += ['', legend, '', 'Fixed leave-one-patch-out reconstruction error is standardized MAE, averaged within examples and then over examples. Values below are means across three masters. Training used4096 sampled rows /512 available rows =8 equivalent presentations; sampling is with replacement, so these are not complete epochs.', '', '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |', '|---|---:|---:|---:|---:|']
    for t in training:
        lines.append(f'| {t["model_tag"]} | 512 | {t["train_error"]:.4f} | {t["validation_error"]:.4f} | {t["GPU_training_seconds"]:.1f} |')
    lines += ['', 'Optimization used waveform hierarchical Huber, AdamW0.001, batch8, patch masking25% plus context deletion15%; no dynamics auxiliary. Loss traces, initial/final minibatch losses and the final32-update mean are retained separately from the fixed reconstruction error. GPU memory and registered/trainable parameter counts remain in the JSON report.', '']
    lines += ['', 'The compact ranges describe variation between the three dataset masters. Source-group confidence intervals and paired architecture/training differences remain in the saved reports. This screen uses VALIDATION only; it does not establish held-out TEST or market performance.', '']
    for path, content in [(args.output, json.dumps(result, indent=2) + '\n'), (args.markdown, '\n'.join(lines))]:
        with path.open('x') as stream:
            stream.write(content)
    for path, digest in READS.items():
        check(sha(path) == digest, 'saved bytes preserved at audit completion')
    print(f'Capacity saved scores/populations/group intervals audited: {checked_predictions} predictions; no encoder calls or fitting.')


if __name__ == '__main__':
    main()

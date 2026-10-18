#!/usr/bin/env python3
"""Independent saved-array branch audit; stdlib only, no encoder or head fits."""
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
ORIGINAL = ROOT / 'output/runs/rpb-v20-capacity/admission-0gN4uD'
ORIGINAL_SOURCE = '8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1'
ORIGINAL_CARD = '562e685c6e1283a7bc22bad7ddcf9fb138c75221cb3dbbd8cd576f8edcfe144e'
ORIGINAL_INPUT = '6121773c0b97e5490f191a9f8ff013e90f27a9a557376ac8256bdbb2d6af6091'
ORIGINAL_ARTIFACTS = '62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83'
MASTERS = [941101, 941102, 941103]
REPETITIONS = [2701, 2802, 2903]
VIEWS = [('training', 'training', 512), ('validation_intact', 'validation', 256),
         ('validation_deleted', 'deleted', 256)]
PARTS = [('fixed_12', 12, 0, 20), ('learned_20_0', 20, 0, 0), ('learned_20_512', 20, 512, 0)]
RECIPE = dict(ridge_penalty=1, tiny_hidden=16, tiny_steps=100, tiny_learning_rate=.01,
              probe_seed_policy='stream_seed(repetition,width);same_equal_width_methods',
              validation_fits=0, encoder_calls=0, pca_fits=0)
SURFACE_KEYS = {'features', 'valid', 'labels_scoring_only', 'source_ids_json', 'provenance'}
PREDICTION_KEYS = {'ridge', 'tiny_secondary', 'valid', 'probe_input_features', 'ridge_logits',
                   'tiny_hidden_preactivation', 'tiny_logits', 'labels_scoring_only', 'source_ids_json'}
FIT_KEYS = {'feature_mean', 'feature_scale', 'outer_normalizer_applied', 'outer_fitted_rows',
            'fitted_rows', 'ridge_mean', 'ridge_scale', 'ridge_weights', 'ridge_intercept',
            'tiny_mean', 'tiny_scale', 'tiny_w1', 'tiny_b1', 'tiny_w2', 'tiny_b2',
            'actual_probe_seed_decimal', 'training_source_ids_json', 'ridge_penalty',
            'tiny_hidden', 'tiny_steps', 'tiny_learning_rate'}
READS = {}
CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remember(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
          and path.is_file() and path.stat().st_nlink == 1, 'direct single-link saved file')
    digest = sha(path)
    check(path not in READS or READS[path] == digest, 'saved file changed during audit')
    READS[path] = digest
    return digest


def document(path):
    remember(path)
    return json.loads(path.read_text())


def tensor(archive, key, dtype, shape):
    check(key in archive and archive[key]['dtype'] == dtype and archive[key]['shape'] == shape,
          'saved tensor schema ' + key)
    values = list(archive[key]['values'])
    check(len(values) == math.prod(shape), 'saved tensor size ' + key)
    if dtype in ('DoubleStorage', 'FloatStorage'):
        check(all(math.isfinite(value) for value in values), 'finite saved tensor ' + key)
    if dtype == 'BoolStorage':
        check(set(values) <= {0, 1}, 'boolean saved tensor ' + key)
    return values


def text(archive, key):
    check(key in archive and archive[key]['dtype'] == 'ByteStorage'
          and len(archive[key]['shape']) == 1, 'saved text schema ' + key)
    return bytes(archive[key]['values']).decode('utf-8')


def scalar(archive, key, dtype='LongStorage'):
    return tensor(archive, key, dtype, [])[0]


def close(actual, expected, message):
    if expected is None:
        check(actual is None, message)
    else:
        check(actual is not None and math.isfinite(actual)
              and abs(actual - expected) <= 2e-9 * max(1., abs(expected)), message)


def close_values(actual, expected, message):
    check(len(actual) == len(expected), message + ' size')
    for a, e in zip(actual, expected):
        close(a, e, message)


def stream_seed(seed, width):
    mask = (1 << 64) - 1
    value = (seed + 0x9e3779b97f4a7c15 * width) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return value ^ (value >> 31)


def row_ids(archive, rows):
    raw = text(archive, 'source_ids_json')
    ids = json.loads(raw)
    check(len(ids) == rows and all(isinstance(value, str) for value in ids)
          and len(set(ids)) == rows // 2
          and all(ids[i] == ids[i + 1] for i in range(0, rows, 2)), 'paired source row schema')
    return raw, ids


def population(labels, ids, valid):
    check(len(labels) == len(ids) == len(valid) and set(labels) <= {0, 1}
          and set(valid) <= {0, 1}, 'binary scoring row schema')
    return dict(total_rows=len(labels), valid_rows=sum(valid),
                class_valid_rows=[sum(bool(v) and y == k for y, v in zip(labels, valid)) for k in (0, 1)],
                total_source_groups=len(set(ids)), valid_source_groups=len({i for i, v in zip(ids, valid) if v}),
                coverage=sum(valid) / len(labels))


def score(predictions, labels, valid):
    check(len(predictions) == len(labels) == len(valid) and set(predictions) <= {0, 1},
          'binary saved decisions')
    correct = sum(bool(v) and p == y for p, y, v in zip(predictions, labels, valid))
    return dict(total=len(labels), valid=sum(valid), correct=correct,
                accuracy=correct / sum(valid) if sum(valid) else None,
                coverage=sum(valid) / len(labels), full_population_correctness=correct / len(labels))


def source_rows(original, rows, master):
    check(set(original) == {'observed', 'feature_mask', 'labels_scoring_only', 'source_ids', 'source_ids_json'},
          'closed original observation schema')
    labels = tensor(original, 'labels_scoring_only', 'LongStorage', [rows])
    packed = tensor(original, 'source_ids', 'LongStorage', [rows])
    raw_ids, ids = row_ids(original, rows)
    check(all(labels[i] != labels[i + 1] for i in range(0, rows, 2)), 'opposite-label source pairs')
    check(all(packed[i] == packed[i + 1] and packed[i] >= 0 and packed[i] & 1 == 0
              and packed[i] >> 31 == master for i in range(0, rows, 2)), 'cohort packed source identity')
    return labels, raw_ids, ids


def audit_surface(surface, original, export, prefix, rows, width, offset, master):
    check(set(surface) == SURFACE_KEYS, 'closed branch surface schema')
    labels, original_json, ids = source_rows(original, rows, master)
    check(tensor(surface, 'labels_scoring_only', 'LongStorage', [rows]) == labels
          and text(surface, 'source_ids_json') == original_json, 'branch source labels/order')
    all_values = tensor(export, prefix + '_values', 'DoubleStorage', [rows, 32])
    expected = [all_values[b * 32 + d] for b in range(rows) for d in range(offset, offset + width)]
    values = tensor(surface, 'features', 'DoubleStorage', [rows, width])
    valid = tensor(surface, 'valid', 'BoolStorage', [rows])
    check(values == expected, 'exact original native branch slice')
    check(valid == tensor(export, prefix + '_valid', 'BoolStorage', [rows]) and all(valid),
          'unchanged original branch support')
    return dict(values=values, valid=valid, labels=labels, source_json=original_json, ids=ids,
                rows=rows, width=width, population=population(labels, ids, valid))


def moments(values, rows, width):
    mean = [math.fsum(values[b * width + d] for b in range(rows)) / rows for d in range(width)]
    scale = [max(1e-8, math.sqrt(math.fsum((values[b * width + d] - mean[d]) ** 2
                                        for b in range(rows)) / rows)) for d in range(width)]
    return mean, scale


def normalize(values, mean, scale, width):
    return [(value - mean[i % width]) / scale[i % width] for i, value in enumerate(values)]


def audit_fit(fit, train, repetition):
    check(set(fit) == FIT_KEYS, 'closed saved fit schema')
    width, rows = train['width'], train['rows']
    check(scalar(fit, 'outer_normalizer_applied', 'BoolStorage') == 1
          and scalar(fit, 'outer_fitted_rows') == scalar(fit, 'fitted_rows') == rows,
          'TRAIN-only outer/probe fitting population')
    check(text(fit, 'training_source_ids_json') == train['source_json'], 'TRAIN-only fit source association')
    check(text(fit, 'actual_probe_seed_decimal') == str(stream_seed(repetition, width)), 'declared width-paired probe seed')
    check(scalar(fit, 'ridge_penalty', 'DoubleStorage') == 1
          and scalar(fit, 'tiny_hidden') == 16 and scalar(fit, 'tiny_steps') == 100
          and scalar(fit, 'tiny_learning_rate', 'DoubleStorage') == .01, 'fixed ridge/tanh16 head recipe')
    mean, scale = moments(train['values'], rows, width)
    close_values(tensor(fit, 'feature_mean', 'DoubleStorage', [width]), mean, 'TRAIN outer mean')
    close_values(tensor(fit, 'feature_scale', 'DoubleStorage', [width]), scale, 'TRAIN outer scale')
    # Replay persisted preprocessing using persisted coefficients, then verify
    # their TRAIN moments independently. No fitted head is trained a second time.
    prepared = normalize(train['values'], tensor(fit, 'feature_mean', 'DoubleStorage', [width]),
                         tensor(fit, 'feature_scale', 'DoubleStorage', [width]), width)
    mean, scale = moments(prepared, rows, width)
    for head in ('ridge', 'tiny'):
        close_values(tensor(fit, head + '_mean', 'DoubleStorage', [width]), mean, 'TRAIN probe mean')
        close_values(tensor(fit, head + '_scale', 'DoubleStorage', [width]), scale, 'TRAIN probe scale')
    for key, shape in [('ridge_weights', [width, 2]), ('ridge_intercept', [2]),
                       ('tiny_w1', [width, 16]), ('tiny_b1', [16]), ('tiny_w2', [16, 2]), ('tiny_b2', [2])]:
        tensor(fit, key, 'DoubleStorage', shape)


def linear(values, rows, width, weights, outputs, bias):
    return [math.fsum(values[b * width + d] * weights[d * outputs + k] for d in range(width)) + bias[k]
            for b in range(rows) for k in range(outputs)]


def decisions(logits, rows):
    return [int(logits[2 * b + 1] > logits[2 * b]) for b in range(rows)]


def audit_prediction(saved, surface, fit, metadata, replay=None):
    check(set(saved) == PREDICTION_KEYS, 'closed saved prediction schema')
    rows, width = surface['rows'], surface['width']
    labels = tensor(saved, 'labels_scoring_only', 'LongStorage', [rows])
    valid = tensor(saved, 'valid', 'BoolStorage', [rows])
    check(labels == surface['labels'] and valid == surface['valid']
          and text(saved, 'source_ids_json') == surface['source_json'], 'prediction original source association')
    check(metadata['population'] == surface['population'], 'reported prediction population')
    if fit is not None:
        prepared = normalize(surface['values'], tensor(fit, 'feature_mean', 'DoubleStorage', [width]),
                             tensor(fit, 'feature_scale', 'DoubleStorage', [width]), width)
        close_values(tensor(saved, 'probe_input_features', 'DoubleStorage', [rows, width]), prepared, 'saved TRAIN outer map')
        ridge_input = normalize(prepared, tensor(fit, 'ridge_mean', 'DoubleStorage', [width]),
                                tensor(fit, 'ridge_scale', 'DoubleStorage', [width]), width)
        ridge = linear(ridge_input, rows, width, tensor(fit, 'ridge_weights', 'DoubleStorage', [width, 2]),
                       2, tensor(fit, 'ridge_intercept', 'DoubleStorage', [2]))
        tiny_input = normalize(prepared, tensor(fit, 'tiny_mean', 'DoubleStorage', [width]),
                               tensor(fit, 'tiny_scale', 'DoubleStorage', [width]), width)
        hidden = linear(tiny_input, rows, width, tensor(fit, 'tiny_w1', 'DoubleStorage', [width, 16]),
                        16, tensor(fit, 'tiny_b1', 'DoubleStorage', [16]))
        tiny = linear([math.tanh(v) for v in hidden], rows, 16,
                      tensor(fit, 'tiny_w2', 'DoubleStorage', [16, 2]), 2,
                      tensor(fit, 'tiny_b2', 'DoubleStorage', [2]))
        close_values(tensor(saved, 'ridge_logits', 'DoubleStorage', [rows, 2]), ridge, 'independent ridge logits')
        close_values(tensor(saved, 'tiny_hidden_preactivation', 'DoubleStorage', [rows, 16]), hidden,
                     'independent neural hidden arithmetic')
        close_values(tensor(saved, 'tiny_logits', 'DoubleStorage', [rows, 2]), tiny, 'independent neural logits')
        check(tensor(saved, 'ridge', 'LongStorage', [rows]) == decisions(ridge, rows)
              and tensor(saved, 'tiny_secondary', 'LongStorage', [rows]) == decisions(tiny, rows),
              'independent saved head decisions')
    result = {}
    for head in ('ridge', 'tiny_secondary'):
        actual = score(tensor(saved, head, 'LongStorage', [rows]), labels, valid)
        for key, value in actual.items():
            close(metadata[head][key], value, 'reported head score ' + key)
            if replay is not None:
                close(replay[head][key], value, 'C++ replay head score ' + key)
        result[head] = actual['accuracy']
    result['coverage'] = surface['population']['coverage']
    return result


def average(values):
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None


def summarize(method, width, cohorts):
    check([c['master'] for c in cohorts] == MASTERS, 'all three cohorts retained')
    return dict(method=method, width=width, cohorts=cohorts,
                views={view: {metric: dict(mean=average([c['views'][view][metric] for c in cohorts]),
                                          master_means=[c['views'][view][metric] for c in cohorts])
                              for metric in ('ridge', 'tiny_secondary', 'coverage')}
                       for view, _, _ in VIEWS})


def audit_pairs(pairs, master, replay, checked, predictions):
    check(len(pairs) == 6, 'all learned0/512 comparisons retained')
    for pair in pairs:
        key = (master, pair['id'], pair['repetition'], pair['view'])
        check(pair['status'] == 'measured' and pair['id'] == 'learned_20_512_minus_learned_20_0'
              and key in replay and key not in checked, 'declared unique learned training comparison')
        checked.add(key)
        actual = replay[key]
        left = predictions['learned_20_512', pair['repetition'], pair['view']]
        right = predictions['learned_20_0', pair['repetition'], pair['view']]
        check(left['labels'] == right['labels'] and left['valid'] == right['valid'], 'paired original scoring population')
        check(pair['common_population'] == left['population']
              and actual['valid_rows'] == left['population']['valid_rows']
              and actual['bootstrap_seed_decimal'] == pair['bootstrap_seed_decimal'], 'paired source population and seed')
        for head in ('ridge', 'tiny_secondary'):
            expected = score(left[head], left['labels'], left['valid'])['accuracy'] - score(right[head], right['labels'], right['valid'])['accuracy']
            close(pair[head]['estimate'], expected, 'paired learned training effect')
            check(pair[head]['source_groups'] == actual[head]['source_groups'] == 128
                  and pair[head]['replicates'] == 1000 and pair[head]['confidence'] == .95, 'paired bootstrap population/budget')
            for field in ('estimate', 'lower', 'upper'):
                close(pair[head][field], actual[head][field], 'independently replayed paired interval ' + field)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capsule', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--card', required=True, type=Path)
    parser.add_argument('--sources-sha256', type=Path,
                        help='defaults to the diagnostic capsule freeze/sources.sha256')
    parser.add_argument('--output-directory', required=True, type=Path)
    args = parser.parse_args()
    args.sources_sha256 = args.sources_sha256 or args.results.parent / 'freeze/sources.sha256'
    check(args.capsule == ORIGINAL, 'declared original capacity capsule')
    check(args.results.is_absolute() and args.results.resolve(strict=True) == args.results,
          'direct branch readout directory')
    destination = args.output_directory
    check(destination.is_absolute() and destination == destination.resolve(strict=False)
          and destination.parent.resolve(strict=True) == destination.parent
          and not destination.exists() and not destination.is_symlink(), 'new direct publisher directory')
    check(not args.results.is_relative_to(ORIGINAL) and not destination.is_relative_to(ORIGINAL),
          'new branch audit outputs stay outside original capsule')
    # No output is created until every saved-data check passes.
    check(remember(CODEC) == CODEC_SHA, 'pinned independent saved-array codec')
    spec = importlib.util.spec_from_file_location('branch_saved_codec', CODEC)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    original_manifest = args.capsule / 'artifacts.sha256'
    check(remember(original_manifest) == ORIGINAL_ARTIFACTS, 'frozen original artifact inventory')
    inventory = {}
    for line in original_manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = args.capsule / name
        check(len(digest) == 64 and path.is_relative_to(args.capsule) and path not in inventory,
              'closed original artifact inventory paths')
        inventory[path] = digest
    check(len(inventory) == 815, 'all 815 original artifacts declared')

    def load(path, historical=False):
        digest = remember(path)
        if historical:
            check(path in inventory and digest == inventory[path], 'original archived bytes unchanged')
        return codec.load(path)

    def old_document(path):
        value = document(path)
        check(path in inventory and READS[path] == inventory[path], 'original report bytes unchanged')
        return value

    complete = document(args.capsule / 'complete.json')
    check(complete['status'] == 'complete' and complete['artifacts_manifest_sha256'] == ORIGINAL_ARTIFACTS,
          'completed original quality evidence')
    old_report = old_document(args.capsule / 'readouts/report.json')
    old_summary = old_document(args.capsule / 'summary.json')
    report = document(args.results / 'report.json')
    done = document(args.results / 'complete.json')
    arithmetic = document(args.results / 'arithmetic-audit.json')
    branch_card_sha = remember(args.card)
    source_sha = remember(args.sources_sha256)
    freeze = document(args.results.parent / 'freeze/receipt.json')
    check(freeze['source_fingerprint'] == source_sha and freeze['card_sha256'] == branch_card_sha
          and freeze['original_source_fingerprint'] == ORIGINAL_SOURCE
          and freeze['original_artifacts_manifest_sha256'] == ORIGINAL_ARTIFACTS,
          'frozen new source/card and original evidence binding')
    for key, expected in dict(protocol='configurable-branch-v1', dataset='TEMPO-5',
                              original_source_fingerprint=ORIGINAL_SOURCE, original_card_sha=ORIGINAL_CARD,
                              original_input_manifest_sha=ORIGINAL_INPUT, original_artifact_manifest_sha=ORIGINAL_ARTIFACTS,
                              source_fingerprint=source_sha, card_sha=branch_card_sha, encoder_calls=0,
                              head_fitted_pairs=27, head_fits=54, validation_fits=0, pca_fits=0,
                              raw_fits=0, full32_fits=0).items():
        check(report[key] == expected, 'closed branch report binding ' + key)
    check(old_report['source_fingerprint'] == old_summary['source_fingerprint'] == ORIGINAL_SOURCE
          and old_report['card_sha'] == old_summary['card_sha'] == ORIGINAL_CARD
          and old_report['input_manifest_sha'] == old_summary['input_manifest_sha'] == ORIGINAL_INPUT
          and old_report['masters'] == MASTERS, 'frozen original quality protocol identity')
    check(done['status'] == 'complete' and done['protocol'] == report['protocol']
          and done['report_sha256'] == READS[args.results / 'report.json']
          and done['arithmetic_audit_sha256'] == READS[args.results / 'arithmetic-audit.json']
          and done['saved_arithmetic_fits_replayed'] == done['head_fitted_pairs'] == 27,
          'completed new branch report/arithmetic binding')
    check(arithmetic['protocol'] == 'saved-branch-arithmetic-v1'
          and arithmetic['encoder_calls'] == arithmetic['head_fits'] == 0
          and arithmetic['fitted_pairs_replayed'] == 27 and arithmetic['prediction_archives'] == 81
          and arithmetic['paired_comparisons'] == 18, 'saved C++ arithmetic replay only')
    replay = {record['prediction']: record for record in arithmetic['records']}
    check(len(replay) == len(arithmetic['records']) == 81, 'complete unique branch prediction replay matrix')
    checked_replay = set()
    pair_replay = {(r['master'], r['id'], r['repetition'], r['view']): r for r in arithmetic['pairs']}
    check(len(pair_replay) == len(arithmetic['pairs']) == 18, 'complete unique learned comparison replay matrix')
    checked_pairs = set()
    collected = {name: [] for name, _, _, _ in PARTS}
    collected.update(native_A_0=[], native_A_512=[], raw=[])
    fixed_cells = 0
    predictions_checked = 0
    reused_predictions_checked = 0
    for cohort in report['cohorts']:
        master = cohort['master']
        check(master in MASTERS, 'declared branch cohort')
        old_cohort = next(c for c in old_report['cohorts'] if c['master'] == master)
        inputs, exports = {}, {}
        for view, prefix, rows in VIEWS:
            role = dict(training='training', validation_intact='validation', validation_deleted='validation-deleted')[view]
            inputs[view] = load(args.capsule / 'quality-data' / f'master-{master}' / (role + '.pt'), True)
        for point in (0, 512):
            exports[point] = load(args.capsule / 'profiles/A' / f'master-{master}' / f'point-{point}/features.pt', True)
            export = exports[point]
            for key, expected in dict(protocol='configurable-capacity-v1', dataset_id='TEMPO-5', profile='A',
                                      model_tag='RPB-v20.alt-01', source_fingerprint=ORIGINAL_SOURCE,
                                      card_sha=ORIGINAL_CARD, input_manifest_sha=ORIGINAL_INPUT).items():
                check(text(export, key) == expected, 'exact original native identity ' + key)
            check(scalar(export, 'master') == master and scalar(export, 'point') == point
                  and scalar(export, 'native_width') == 32, 'original native cohort/point/width')
        for view, prefix, rows in VIEWS:
            initial = tensor(exports[0], prefix + '_values', 'DoubleStorage', [rows, 32])
            final = tensor(exports[512], prefix + '_values', 'DoubleStorage', [rows, 32])
            check(all(initial[b * 32 + d] == final[b * 32 + d] for b in range(rows) for d in range(20, 32))
                  and tensor(exports[0], prefix + '_valid', 'BoolStorage', [rows])
                  == tensor(exports[512], prefix + '_valid', 'BoolStorage', [rows]), 'fixed12 exact across both points/all views')
            fixed_cells += rows * 12
        metadata = cohort['readouts']
        readout_dir = args.results / f'master-{master}/readouts'
        check(document(readout_dir / 'report.json') == metadata and metadata['recipe'] == RECIPE
              and metadata['master_seed'] == str(master)
              and [m['method'] for m in metadata['methods']] == [p[0] for p in PARTS], 'closed shared branch readout recipe')
        check(metadata['fit_counts'] == dict(outer_train_normalizer_fits=3, ridge_fits=9, tiny_fits=9,
                                             validation_fits=0, encoder_calls=0, pca_fits=0), 'all branch fits once on TRAIN')
        cohort_predictions = {}
        for (name, width, point, offset), method in zip(PARTS, metadata['methods']):
            check(method['size'] == width and method['status'] == 'measured'
                  and method['outer_train_normalizer_fits'] == 1
                  and [r['id'] for r in method['repetitions']] == [f'rep-{r}' for r in REPETITIONS], 'declared branch methods/repetitions')
            method_dir = readout_dir / name
            surfaces = {}
            for view, prefix, rows in VIEWS:
                surface = load(method_dir / (view.replace('_', '-') + '-features.pt'))
                surfaces[view] = audit_surface(surface, inputs[view], exports[point], prefix, rows, width, offset, master)
                check(method[view + '_population'] == surfaces[view]['population'], 'branch method population')
            values = dict(master=master, width=width, views={view: {h: [] for h in ('ridge', 'tiny_secondary', 'coverage')}
                                                          for view, _, _ in VIEWS})
            for repetition, rep_seed in zip(method['repetitions'], REPETITIONS):
                check(repetition['actual_probe_seed_decimal'] == str(stream_seed(rep_seed, width))
                      and repetition['status'] == 'measured'
                      and repetition['ridge_parameters'] == 2 * width + 2
                      and repetition['neural_parameters'] == 16 * (width + 3) + 2, 'fixed width-dependent head identity')
                rep_dir = method_dir / repetition['id']
                fit = load(rep_dir / 'fit.pt')
                audit_fit(fit, surfaces['training'], rep_seed)
                for view, _, _ in VIEWS:
                    path = rep_dir / (view.replace('_', '-') + '-predictions.pt')
                    relative = str(path.relative_to(args.results))
                    check(relative in replay and relative not in checked_replay, 'branch predictions replayed once')
                    checked_replay.add(relative)
                    saved_prediction = load(path)
                    actual = audit_prediction(saved_prediction, surfaces[view], fit, repetition[view], replay[relative])
                    cohort_predictions[name, repetition['id'], view] = dict(
                        labels=surfaces[view]['labels'], valid=surfaces[view]['valid'],
                        population=surfaces[view]['population'],
                        **{h: tensor(saved_prediction, h, 'LongStorage', [surfaces[view]['rows']])
                           for h in ('ridge', 'tiny_secondary')})
                    for metric, value in actual.items():
                        values['views'][view][metric].append(value)
                    if view != 'training':
                        check(repetition[view]['bootstrap_seed_decimal'] == replay[relative]['bootstrap_seed_decimal'],
                              'fixed source-group interval seed')
                        for interval_key in ('ridge_grouped_interval', 'tiny_grouped_interval'):
                            declared, verified = repetition[view][interval_key], replay[relative][interval_key]
                            check(declared['source_groups'] == verified['source_groups'] == 128
                                  and declared['replicates'] == 1000 and declared['confidence'] == .95,
                                  'fixed source-group interval population/budget')
                            for key in ('estimate', 'lower', 'upper'):
                                close(declared[key], verified[key], 'independently replayed source-group interval ' + key)
                    predictions_checked += 1
            for view, _, _ in VIEWS:
                values['views'][view] = {metric: average(v) for metric, v in values['views'][view].items()}
            collected[name].append(values)
        audit_pairs(metadata['pairs'], master, pair_replay, checked_pairs, cohort_predictions)
        # Reuse all original complete32 and raw fitted decisions. Their fits and
        # arithmetic audits remain historical; no original classifier is refit.
        for name, section, method_name in [('native_A_0', 'models', 'native_A_0'),
                                           ('native_A_512', 'models', 'native_A_512'), ('raw', 'raw', 'raw')]:
            method = next(m for m in old_cohort[section]['methods'] if m['method'] == method_name)
            width = method['size']
            old_dir = args.capsule / 'readouts' / f'master-{master}'
            old_dir /= 'models' if section == 'models' else 'baselines/raw/readouts'
            old_dir /= method_name
            values = dict(master=master, width=width, views={view: {h: [] for h in ('ridge', 'tiny_secondary', 'coverage')}
                                                          for view, _, _ in VIEWS})
            for view, prefix, rows in VIEWS:
                surface = load(old_dir / (view.replace('_', '-') + '-features.pt'), True)
                if section == 'models':
                    info = audit_surface(surface, inputs[view], exports[int(name.rsplit('_', 1)[1])],
                                         prefix, rows, 32, 0, master)
                else:
                    labels, raw_ids, ids = source_rows(inputs[view], rows, master)
                    valid = tensor(surface, 'valid', 'BoolStorage', [rows])
                    check(tensor(surface, 'labels_scoring_only', 'LongStorage', [rows]) == labels
                          and text(surface, 'source_ids_json') == raw_ids and all(valid), 'reused raw population identity')
                    info = dict(rows=rows, width=width, labels=labels, source_json=raw_ids, valid=valid,
                                population=population(labels, ids, valid))
                for repetition in method['repetitions']:
                    prediction = load(old_dir / repetition['id'] / (view.replace('_', '-') + '-predictions.pt'), True)
                    actual = audit_prediction(prediction, info, None, repetition[view])
                    for metric, value in actual.items():
                        values['views'][view][metric].append(value)
                    reused_predictions_checked += 1
            for view, _, _ in VIEWS:
                values['views'][view] = {metric: average(v) for metric, v in values['views'][view].items()}
            collected[name].append(values)
    check(checked_replay == set(replay) and predictions_checked == reused_predictions_checked == 81,
          'complete branch/reused prediction matrices audited')
    check(checked_pairs == set(pair_replay) and report['fixed_coordinates_exact'] ==
          dict(passed=True, cells=fixed_cells, views=9, fixed_heads_refitted_at_512=False),
          'entire learned comparison/fixed coordinate matrix audited')
    summaries = [summarize(name, width, collected[name]) for name, width in
                 [('native_A_0', 32), ('native_A_512', 32), ('fixed_12', 12),
                  ('learned_20_0', 20), ('learned_20_512', 20), ('raw', 6912)]]
    for summary in summaries:
        if summary['method'] not in ('native_A_0', 'native_A_512', 'raw'):
            continue
        prior = next(s for s in old_summary['summaries'] if s['method'] == summary['method'])
        for view, _, _ in VIEWS:
            for metric in ('ridge', 'tiny_secondary', 'coverage'):
                check(summary['views'][view][metric] == {k: prior['views'][view][metric][k] for k in ('mean', 'master_means')},
                      'reused original summary exactly unchanged')
    for entry in report['saved_input_files']:
        check(remember(Path(entry['path'])) == entry['sha256'], 'branch input ledger unchanged')
    for path, digest in list(READS.items()):
        check(sha(path) == digest, 'all consumed saved bytes unchanged after audit')
    result = dict(protocol='configurable-branch-v1', dataset='TEMPO-5', designed_complexity=5, complexity_max=5,
                  stage='known development VALIDATION saved-feature diagnostic', TEST_access=False,
                  encoder_calls=0, encoder_updates=0, post_encoder_PCA=False,
                  new_head_fit_pairs=27, new_classifiers=54, reused_complete32_fitted_pairs=18,
                  reused_raw_fitted_pairs=9, new_prediction_archives_checked=predictions_checked,
                  reused_prediction_archives_checked=reused_predictions_checked, fixed_coordinates_exact_cells=fixed_cells,
                  source_fingerprint=source_sha, card_sha256=branch_card_sha,
                  original_source_fingerprint=ORIGINAL_SOURCE, original_card_sha=ORIGINAL_CARD,
                  original_input_manifest_sha=ORIGINAL_INPUT, original_artifact_manifest_sha=ORIGINAL_ARTIFACTS,
                  original_report_sha=READS[args.capsule / 'readouts/report.json'],
                  original_summary_sha=READS[args.capsule / 'summary.json'],
                  branch_report_sha=READS[args.results / 'report.json'], paired_comparisons_checked=18, checks=CHECKS,
                  summaries=summaries, training=[v for v in old_summary['training'] if v['profile'] == 'A'],
                  read_files=[dict(path=str(p), sha256=d) for p, d in sorted(READS.items())])
    legend = ('TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; '
              'slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. '
              'TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.')
    lines = [legend, '',
             'New saved-feature diagnostic; zero encoder execution or updates. RPB-v20.alt-01 is unchanged. '
             'Size32 means its complete export; size20 means the learned coordinates; size12 means the fixed spectral coordinates. '
             'The fixed12 arrays are exactly identical at0/512, so their nine TRAIN-fitted head pairs are shared across both points. '
             'Learned20 uses18 new pairs. Complete32 and raw results reuse original fitted decisions without new fits.', '',
             'Linear head = ridge1; Neural head = tanh16,100 Adam steps at0.01. Accuracy is conditional on valid rows; '
             'coverage is the fraction scored. Means retain all three cohorts and all three head repetitions. '
             'Changing branch width changes head parameter counts and initialization seeds; this tests branch sufficiency, '
             'rather than providing a perfectly controlled attribution of each coordinate.', '']
    by_name = {s['method']: s for s in summaries}
    for point in (0, 512):
        for view in ('validation_intact', 'validation_deleted'):
            lines += [legend, '',
                      f'{"Initial, zero updates" if point == 0 else "Trained,512 updates"}; '
                      f'{"intact VALIDATION" if view == "validation_intact" else "VALIDATION with an additional30% coordinate deletion"}. '
                      'The TRAIN-fitted head is reused unchanged across views.', '',
                      '| Method | Size | Linear head % | Neural head % | Coverage % |',
                      '|---|---:|---:|---:|---:|']
            for name in (f'native_A_{point}', f'learned_20_{point}', 'fixed_12', 'raw'):
                summary, label = by_name[name], 'Raw data — no encoder' if name == 'raw' else 'RPB-v20.alt-01'
                values = summary['views'][view]
                metrics = ['—' if values[m]['mean'] is None else f'{100 * values[m]["mean"]:.2f}'
                           for m in ('ridge', 'tiny_secondary', 'coverage')]
                lines.append('| ' + ' | '.join([label, str(summary['width'])] + metrics) + ' |')
            lines += ['']
    lines += ['Reused original training record; no training occurred in this diagnostic.', '',
              '| Encoder | Updates | Train error | Validation error | GPU training seconds |',
              '|---|---:|---:|---:|---:|']
    t = result['training'][0]
    lines += [f'| RPB-v20.alt-01 |512|{t["train_error"]:.4f}|{t["validation_error"]:.4f}|{t["GPU_training_seconds"]:.1f}|', '',
              'Training errors are original fixed-query standardized MAE; the source training usedCUDA float32,batch8. '
              'These known VALIDATION results do not establish unseen TEST or market quality.', '']
    destination.mkdir()
    for filename, content in [('summary.json', json.dumps(result, indent=2) + '\n'), ('report.md', '\n'.join(lines))]:
        with (destination / filename).open('x', encoding='utf-8') as stream:
            stream.write(content)
    print(json.dumps(dict(status='passed', output=str(destination), new_prediction_archives=81,
                          reused_prediction_archives=81, checks=CHECKS)))


if __name__ == '__main__':
    main()

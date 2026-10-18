"""Independent, read-only phase-1 archive QA. Python stdlib; run in container."""
import argparse
import array
import csv
import datetime
import hashlib
import io
import itertools
import json
import math
import pickle
import statistics
import sys
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

ROOT = Path('/embedding')
MASK64 = (1 << 64) - 1
CHECKS = 0


def check(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise AssertionError(message)


class Module:
    pass


def rebuild(storage, offset, size, stride, *rest):
    kind, raw = storage
    fmt = {'DoubleStorage': 'd', 'FloatStorage': 'f', 'LongStorage': 'q',
           'BoolStorage': 'B', 'ByteStorage': 'B'}[kind]
    values = array.array(fmt)
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    expected, step = [], 1
    for length in reversed(size):
        expected.insert(0, step)
        step *= length
    if tuple(stride) == tuple(expected):
        part = values[offset:offset + math.prod(size)]
    else:
        # LibTorch may serialize column-major linalg outputs without copying.
        # Decode logical row-major values using the saved offset/strides.
        part = array.array(fmt, (values[offset + sum(index * step for index, step in zip(indices, stride))]
                                for indices in itertools.product(*(range(length) for length in size))))
    check(len(part) == math.prod(size), 'truncated tensor storage')
    return {'dtype': kind, 'shape': list(size), 'values': part, 'raw': part.tobytes()}


class Reader(pickle.Unpickler):
    def __init__(self, buffer, archive, prefix):
        super().__init__(buffer)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return rebuild
        if module == 'torch' and name in ('DoubleStorage', 'FloatStorage', 'LongStorage',
                                          'BoolStorage', 'ByteStorage'):
            return name
        if module.startswith('__torch__') and name == 'Module':
            return Module
        raise ValueError('disallowed pickle class: ' + module + '.' + name)

    def persistent_load(self, pid):
        check(pid[0] == 'storage' and pid[3] == 'cpu', 'audited archive contains non-CPU storage')
        return pid[1], self.archive.read(self.prefix + 'data/' + pid[2])


def load(path):
    with zipfile.ZipFile(path) as archive:
        key = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        return vars(Reader(io.BytesIO(archive.read(key)), archive, key[:-len('data.pkl')]).load())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(tensor):
    check(tensor['dtype'] == 'ByteStorage' and len(tensor['shape']) == 1, 'text tensor type')
    return tensor['raw'].decode('utf-8')


def same(left, right, message):
    check(left['dtype'] == right['dtype'] and left['shape'] == right['shape']
          and left['raw'] == right['raw'], message)


def stream_seed(seed, stream):
    value = (seed + 0x9e3779b97f4a7c15 * stream) & MASK64
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK64
    return value ^ (value >> 31)


def scores(predictions, labels, valid):
    check(len(predictions) == len(labels) == len(valid), 'prediction/label/support row equality')
    check(set(predictions).issubset({0, 1}) and set(labels).issubset({0, 1}), 'binary predictions/labels')
    count = sum(valid)
    correct = sum(p == y and v for p, y, v in zip(predictions, labels, valid))
    return {'total': len(labels), 'valid': count, 'correct': correct,
            'abstained': len(labels) - count, 'coverage': count / len(labels),
            'accuracy': correct / count if count else None,
            'full_population_correctness': correct / len(labels)}


def compare_score(actual, declared):
    for key, value in actual.items():
        check(key in declared, 'missing score field ' + key)
        if value is None:
            check(declared[key] is None, 'unsupported score must be null')
        else:
            check(abs(declared[key] - value) <= 1e-14, 'score mismatch ' + key)


def population(labels, ids, valid):
    groups, classes = {}, [0, 0]
    for label, source, supported in zip(labels, ids, valid):
        group = groups.setdefault(source, [0, 0])
        group[0] += 1
        if supported:
            group[1] += 1
            classes[label] += 1
    return {'total_rows': len(labels), 'valid_rows': sum(classes), 'class_valid_rows': classes,
            'total_source_groups': len(groups), 'valid_source_groups': sum(g[1] > 0 for g in groups.values()),
            'complete_source_pairs': sum(g[0] == g[1] for g in groups.values()),
            'coverage': sum(classes) / len(labels)}


def compare_population(actual, declared):
    for key, value in actual.items():
        check(declared[key] == value, 'population mismatch ' + key)


def source_fingerprint():
    paths = sorted(set([str(p.relative_to(ROOT)) for pattern in
        ('code/shared/include/embedding/shared/*.h', 'code/shared/src/*.cpp') for p in ROOT.glob(pattern)] +
        ['code/evaluation/src/archive_readout_main.cpp', 'code/evaluation/cards/archive_readout_v1.md',
         'Makefile', 'dependencies.lock']))
    lines = ''.join(sha(ROOT / p) + '  ' + p + '\n' for p in paths)
    return hashlib.sha256(lines.encode()).hexdigest(), paths


def evaluate(result_dir, plan_dir, input_audit):
    plan = json.loads((plan_dir / 'experiment-plan.json').read_text())
    baseline = json.loads(input_audit.read_text())
    check(sha(input_audit) == plan['input_audit_sha256'], 'original input audit SHA')
    check(sha(plan_dir / 'inputs.tsv') == plan['input_manifest_sha256'], 'prescore TSV SHA')
    with (plan_dir / 'inputs.tsv').open(newline='', encoding='utf-8-sig') as source:
        tsv = {row['id']: row for row in csv.DictReader(source, delimiter='\t')}
    report = json.loads((result_dir / 'report.json').read_text())
    card = json.loads((result_dir / report['card_file']).read_text())
    inputs = json.loads((result_dir / report['input_manifest_file']).read_text())
    outputs = json.loads((result_dir / report['output_manifest_file']).read_text())
    fingerprint, source_paths = source_fingerprint()
    check(report['source_fingerprint'] == card['source_fingerprint'] == fingerprint,
          'exact compiled/report/current frozen evaluator fingerprint')
    for value in (report, card):
        check(value['protocol'] == plan['protocol'] and value['policy_version'] == '1.2', 'new protocol/policy')
        check(value['validation_only'] is True and value['test_access'] is False
              and value['encoder_training'] is False and value['post_encoder_pca'] is False, 'declared scope')
    check(report['original_inputs_byte_preserved'] is True, 'engine preservation assertion')
    check(card['shape'] == plan['shape'] and card['compact_width'] == 32 and card['threads'] == 1, 'frozen geometry/threads')
    check([int(rep['probe_seed']) for rep in card['repetitions']] == plan['probe_seeds'], 'all frozen repetition seeds')
    recipe = card['recipe']
    check(recipe['ridge']['penalty'] == 1 and recipe['tiny_secondary']['hidden'] == 16
          and recipe['tiny_secondary']['updates'] == 100 and recipe['tiny_secondary']['learning_rate'] == 0.01,
          'fixed head recipe')
    check(recipe['confidence'] == 0.95 and recipe['bootstrap_replicates'] == 1000, 'interval recipe')
    check(len(report['inputs']) == len(card['inputs']) == len(tsv) == 12, 'all twelve cohorts retained')
    old_files = {str(ROOT / f['path']): f for f in baseline['files']}
    check(len(inputs['files']) == len(old_files) == 48, 'all 48 declared archives retained')
    for entry in inputs['files']:
        path = str(Path(entry['path']).resolve())
        check(path in old_files, 'unexpected input archive')
        check(entry['sha256'] == old_files[path]['sha256'] == sha(path), 'input SHA preservation')
        check(entry['bytes'] == Path(path).stat().st_size, 'input manifest file size')
    for entry in outputs['files']:
        path = (result_dir / entry['path']).resolve()
        check(path.is_relative_to(result_dir.resolve()), 'output manifest escapes results')
        check(path.is_file() and sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'output artifact SHA/size')
    cohorts = {(c['master'], c['task']): c for c in baseline['cohorts']}
    all_training_sources, all_validation_sources = set(), set()
    method_count = rep_count = score_count = prediction_count = paired_count = 0
    observations, accuracies = [], {}
    for cohort in report['inputs']:
        identity = cohort['input']; item = tsv[identity['id']]
        master, task = int(identity['master_seed']), identity['task']
        prior = cohorts[(master, task)]
        check(identity == next(c for c in card['inputs'] if c['id'] == identity['id']), 'report/card input identity')
        for key, value in item.items():
            if key in ('master_seed', 'checkpoint_steps'):
                check(int(identity[key]) == int(value), 'frozen TSV integer')
            elif key.endswith('observations') or key.endswith('features'):
                expected = Path(value) if Path(value).is_absolute() else plan_dir / value
                check(Path(identity[key]).resolve() == expected.resolve(), 'frozen TSV archive path')
            else:
                check(identity[key] == value, 'frozen TSV identity field ' + key)
        check(identity['tag'] == 'RPB-v4' and identity['checkpoint_steps'] == 512, 'registered design/milestone')
        check(identity['producer_source_fingerprint'] == prior['historical_training_producer_source_fingerprint']
              and identity['producer_source_fingerprint'] != fingerprint, 'distinct historical encoder/new head producer')
        check(cohort['native_provenance'] == prior['expected_feature_provenance'], 'exact native provenance')
        check(cohort['raw_pca_numerical_rank'] is not None and cohort['raw_pca_numerical_rank'] >= 32, 'supported raw PCA rank')
        split_values = {}
        for split, rows in (('training', 256), ('validation', 128)):
            original = load(Path(identity[split + '_observations']))
            native = load(Path(identity[split + '_features']))
            copied = load(result_dir / identity['id'] / ('native-' + split + '.pt'))
            same(native['features'], copied['features'], 'exact native values preserved')
            same(native['valid'], copied['valid'], 'exact native support preserved')
            check(text(copied['provenance']) == text(native['provenance']), 'exact copied native provenance')
            same(original['labels_scoring_only'], copied['labels_scoring_only'], 'copied native labels/order')
            check(text(original['source_ids_json']) == text(copied['source_ids_json']), 'copied native source row order')
            labels = original['labels_scoring_only']['values']; ids = json.loads(text(original['source_ids_json']))
            valid = native['valid']['values']
            check(len(labels) == rows and sum(valid) == rows, 'declared full-support population')
            pop = population(labels, ids, valid)
            compare_population(pop, cohort['native_' + split + '_population'])
            compare_population(pop, cohort['raw_' + split + '_population'])
            (all_training_sources if split == 'training' else all_validation_sources).update(ids)
            split_values[split] = labels, ids, valid
        assets = result_dir / identity['id']
        normalizer = load(assets / 'native-normalizer.pt')
        check(normalizer['feature_mean']['shape'] == [32] and normalizer['feature_scale']['shape'] == [32], 'native normalization only')
        check(normalizer['fitted_rows']['values'][0] == 256, 'native normalizer TRAIN fit count')
        raw_scale = load(assets / 'raw-scaler.pt')
        check(raw_scale['mean']['shape'] == raw_scale['scale']['shape'] == raw_scale['counts']['shape'] == [3, 3], 'raw C/F scaler')
        raw_norm = load(assets / 'raw-normalizer.pt')
        check(raw_norm['feature_mean']['shape'] == [576] and raw_norm['fitted_rows']['values'][0] == 256, 'raw outer fit count')
        pca = load(assets / 'pca-only.pt')
        check(pca['pca_mean']['shape'] == [576] and pca['pca_components']['shape'] == [576, 32], 'PCA acts only on raw576')
        check(pca['fitted_rows']['values'][0] == 256 and pca['pca_numerical_rank']['values'][0] == cohort['raw_pca_numerical_rank'], 'PCA fit count/rank')
        check(len(cohort['repetitions']) == 3, 'all repetitions retained')
        for repetition, frozen_rep in zip(cohort['repetitions'], card['repetitions']):
            rep_count += 1
            check(repetition['id'] == frozen_rep['id'] and len(repetition['methods']) == 3, 'repetition identity/method count')
            check([m['method'] for m in repetition['methods']] == ['raw', 'pca_only', 'native'], 'fixed methods/order')
            prediction_cache = {}
            for method in repetition['methods']:
                method_count += 1
                width = 576 if method['method'] == 'raw' else 32
                check(method['status'] == 'measured' and method['size'] == width, 'all actual methods measured')
                seed = stream_seed(int(frozen_rep['probe_seed']), width)
                check(int(method['actual_probe_seed']) == seed, 'exact width-paired head seed')
                check(method['ridge_parameters'] == 2 * width + 2 and method['neural_parameters'] == 16 * (width + 3) + 2, 'head parameter counts')
                fit = load(result_dir / method['fit_file'])
                check(text(fit['actual_probe_seed_decimal']) == str(seed), 'saved fit seed')
                check(fit['fitted_rows']['values'][0] == 256, 'readout outer fit TRAIN count')
                outer_width = 576 if method['method'] == 'pca_only' else width
                check(fit['feature_mean']['shape'] == fit['feature_scale']['shape'] == [outer_width], 'correct preprocessing asset width')
                for name, shape in (('ridge_mean', [width]), ('ridge_scale', [width]), ('ridge_weights', [width, 2]), ('ridge_intercept', [2]),
                                    ('tiny_mean', [width]), ('tiny_scale', [width]), ('tiny_w1', [width, 16]), ('tiny_b1', [16]), ('tiny_w2', [16, 2]), ('tiny_b2', [2])):
                    check(fit[name]['shape'] == shape and all(math.isfinite(x) for x in fit[name]['values']), 'fit tensor shape/finiteness ' + name)
                for split in ('training', 'validation'):
                    saved = load(result_dir / method[split + '_predictions_file'])
                    prediction_count += 1
                    labels, ids, support = split_values[split]
                    check(saved['labels_scoring_only']['values'] == labels and json.loads(text(saved['source_ids_json'])) == ids, 'prediction label/source order')
                    check(saved['valid']['values'] == support and saved['probe_features']['shape'] == [len(labels), width], 'prediction support/feature width')
                    compare_population(population(labels, ids, support), method[split + '_population'])
                    for head, archive_key in (('ridge', 'ridge'), ('tiny_secondary', 'tiny_secondary')):
                        actual = scores(saved[archive_key]['values'], labels, support)
                        compare_score(actual, method[head][split]); score_count += 1
                        if split == 'validation':
                            interval = method[head]['validation_interval']
                            check(interval['source_groups'] == 64 and interval['replicates'] == 1000 and interval['confidence'] == .95, 'interval group/budget')
                            check(abs(interval['estimate'] - actual['accuracy']) <= 1e-14 and 0 <= interval['lower'] <= interval['upper'] <= 1, 'interval score/range')
                            prediction_cache[method['method'], head] = saved[archive_key]['values']
                            accuracies.setdefault((task, method['method'], head), []).append(actual['accuracy'])
            pairs = repetition['paired_comparisons']
            check(len(pairs) == 3, 'all paired comparisons retained')
            labels, ids, support = split_values['validation']
            for pair in pairs:
                paired_count += 1
                check(pair['status'] == 'measured', 'paired fit supported')
                compare_population(population(labels, ids, support), pair['common_population'])
                for head in ('ridge', 'tiny_secondary'):
                    candidate = scores(prediction_cache[pair['candidate'], head], labels, support)
                    comparator = scores(prediction_cache[pair['comparator'], head], labels, support)
                    compare_score(candidate, pair[head]['candidate']); compare_score(comparator, pair[head]['comparator'])
                    interval = pair[head]['candidate_minus_comparator_interval']
                    check(interval['source_groups'] == 64 and interval['replicates'] == 1000 and interval['confidence'] == .95, 'paired interval population/budget')
                    check(abs(interval['estimate'] - (candidate['accuracy'] - comparator['accuracy'])) <= 1e-14
                          and -1 <= interval['lower'] <= interval['upper'] <= 1, 'paired effect exactness/range')
    check(method_count == 108 and rep_count == 36 and prediction_count == 216 and score_count == 432 and paired_count == 108, 'complete result matrix')
    check(not all_training_sources & all_validation_sources and len(all_training_sources | all_validation_sources) == 2304, 'all source roles globally disjoint')
    return {'method_entries': method_count, 'repetitions': rep_count, 'cohorts': 12, 'prediction_archives': prediction_count,
            'recomputed_head_scores': score_count, 'paired_comparisons': paired_count, 'preserved_input_archives': 48,
            'verified_output_files': len(outputs['files']), 'source_groups': 2304, 'native_exports_exact': True,
            'native_pca_applied': False, 'encoder_training': False, 'test_artifact_content_reads': 0,
            'scope_evidence': 'Explicit TRAIN/VALIDATION input ledger, source code path review and artifact roles; no OS syscall trace.',
            'evaluator_source_fingerprint': fingerprint, 'source_inputs': source_paths,
            'report_sha256': sha(result_dir / 'report.json'), 'card_sha256': sha(result_dir / 'archive-readout-card.json'),
            'input_manifest_sha256': sha(result_dir / 'input-manifest.json'), 'output_manifest_sha256': sha(result_dir / 'output-manifest.json'),
            'validation_means': [{'task': task, 'method': method, 'head': head, 'entries': len(values),
                                  'accuracy_mean': statistics.mean(values), 'scope': 'three archived encoder masters x three head fits; validation diagnosis'}
                                 for (task, method, head), values in sorted(accuracies.items())]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--input-audit', required=True, type=Path)
    args = parser.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    destination = ROOT / 'output/runs/archive-controls' / ('phase1-validation-' + now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    result = {'version': 1, 'created_utc': now.isoformat(), 'auditor': 'independent allowlisted Python stdlib tensor/artifact audit; no fits or Torch runtime',
              'auditor_script_sha256': sha(Path(__file__)), 'results_directory': str(args.results), 'input_audit': str(args.input_audit), 'plan_directory': str(args.plan)}
    try:
        result.update(evaluate(args.results, args.plan, args.input_audit))
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=str(error), error_type=type(error).__name__)
    result['checks'] = CHECKS
    path = destination / 'validation.json'
    with path.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({'path': str(path), 'status': result['status'], 'checks': CHECKS,
                      'error': result.get('error'), 'sha256': sha(path)}, separators=(',', ':')))
    sys.exit(0 if result['status'] == 'passed' else 1)

#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Report the single admitted frozen V30 transfer from saved scalars; stdlib only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time

ROOT = Path('/cuwacunu')
PROTOCOL = 'market-location-scale-synthetic-transfer-v1'
MODEL = 'RPB-v30.market-01'
NATIVE36 = 'RPB-v24.market-01.alt-01'
REFERENCE = 'RPB-v22.alt-01'
MASTERS = (941101, 941102, 941103)
MODES = ('synthetic_train', 'market_scaler')
VIEWS = ('validation_intact', 'validation_deleted')
HEADS = ('ridge', 'tiny_secondary')
REPETITIONS = (2701, 2802, 2903)
CARD = 'c3cfe0a04b174734599b2b01363c90c3f9b81fd5964fb30ffd4ce8a21976260e'
INPUT = 'fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb'
CHECKPOINT = 'c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb'
SETTINGS = '43320396447ca80be5fde4324d17c3a9ab129e116bf184455fcb6261aeb6923c'
OLD_TABLE = '2e1a3561127f4fe01b1f57834f10c0b90af671fc568b76fc9b27fe66e5e008cf'
MARKET_COMPLETION = '0b91cded8a9f6cd9829886724e240eb2ce770a6fd60c31ec04d5db8ae59d7602'
NORMALIZER = 'rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476'
FLOORS = {
    'synthetic_train': {'validation_intact': (.9895833333333334, .9774305555555556),
                        'validation_deleted': (.93359375, .9253472222222222)},
    'market_scaler': {'validation_intact': (1., .9952256944444444),
                      'validation_deleted': (.9609375, .9505208333333334)},
}
DEPS = {
    'include/embedding/shared/types.h': '3c7987b9df5d936e009da619474af7ec08630083721281efb5523492200e2371',
    'include/embedding/shared/data.h': 'b9a7b80f1df93f541d05453ebc6e6abde5de1a2144f8d5e03ad5f32a068cb92e',
    'include/embedding/shared/evaluation.h': '63b7064fec06d3e33ccca7e32b20ed604c7e97550a79117ff9bbf6b93283a270',
    'include/embedding/shared/feature_harness.h': '0823c7a8e72c55e9907a4d89b076393d6356cb495122c23c78a3086ba9062657',
    'include/embedding/shared/fixed_feature_readouts.h': 'aea837bf3129044e2921695c4770082f07ae2f4ec7508b1ffeef535dfd52ce79',
    'src/data.cpp': '2a5b1b19baa0f67fac9020d91cd2883a35f7448a3e2fedb3d430b5928800043c',
    'src/feature_harness.cpp': '8cb2aaa6c2b0d6582441fe51fe04445479e15824a44003cf59b97a8f8a67abbc',
    'src/fixed_feature_readouts.cpp': '1de2800e99c52a3df6b2bc168cd7b4c900888162e0056d1c681c2e74c3396b2c',
}
LEGEND = ('Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped '
          'features and structured distractors · **complexity 5/5**.')
TEXT = {'.json', '.log', '.md', '.mk', '.h', '.cpp', '.sha256', '.py', '.sh', '.conf', '.txt', '.d'}
LOCAL_ONLY = {'preprocessing-witnesses.json', 'parent-parity.json'}
CHECKS = 0


def require(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify_records(records):
    for name, pin in records.items():
        path = Path(name)
        require(path.is_file() and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                'Pinned file changed: ' + name)


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def save_json(path, value):
    save(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def close(actual, expected, why):
    require(math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= 1e-12, why)


def seed(repetition, width):
    mask = (1 << 64) - 1
    value = (repetition + 0x9e3779b97f4a7c15 * width) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return str(value ^ (value >> 31))


def validate_recipe(report, master):
    require(report['protocol'] == 'fixed-feature-readouts-v1' and int(report['master_seed']) == master,
            'Fixed-head protocol/master changed')
    for key, value in {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100,
                       'tiny_learning_rate': .01, 'validation_fits': 0, 'encoder_calls': 0,
                       'pca_fits': 0, 'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods'}.items():
        require(report['recipe'][key] == value, 'Fixed head recipe: ' + key)


def validate_method(method, width):
    require(method['size'] == width and method['status'] == 'measured' and
            method['inputs_train_prepared'] is False and method['outer_train_normalizer_fits'] == 1,
            'Measured native width/TRAIN normalization')
    require(len(method['repetitions']) == 3, 'All fixed repetitions retained')
    for rep, repetition in zip(method['repetitions'], REPETITIONS):
        require(rep['id'] == 'rep-' + str(repetition) and rep['status'] == 'measured' and
                rep['actual_probe_seed_decimal'] == seed(repetition, width), 'Fixed fit/seed')
        require(rep['ridge_parameters'] == 2 * width + 2 and
                rep['neural_parameters'] == 16 * (width + 3) + 2, 'Dimension-dependent head budget')
        for view in ('training',) + VIEWS:
            row = rep[view]
            rows, groups = (512, 256) if view == 'training' else (256, 128)
            p = row['population']
            require(p['total_rows'] == rows and p['total_source_groups'] == groups and
                    0 < p['valid_rows'] <= rows and 0 < p['valid_source_groups'] <= groups,
                    'Declared row/source population')
            close(p['coverage'], p['valid_rows'] / rows, 'Population coverage arithmetic')
            for head in HEADS:
                score = row[head]
                require(score['total'] == rows and score['valid'] == p['valid_rows'] and
                        isinstance(score['correct'], int) and 0 <= score['correct'] <= score['valid'],
                        'Saved head correct/valid/total counts')
                close(score['coverage'], score['valid'] / rows, 'Head coverage arithmetic')
                close(score['accuracy'], score['correct'] / score['valid'], 'Saved accuracy arithmetic')
                close(score['full_population_correctness'], score['correct'] / rows, 'Full-population correctness')
            if view in VIEWS:
                for head, key in (('ridge', 'ridge_grouped_interval'), ('tiny_secondary', 'tiny_grouped_interval')):
                    interval = row[key]
                    require(interval['replicates'] == 1000 and interval['confidence'] == .95 and
                            interval['source_groups'] == groups and
                            0 <= interval['lower'] <= interval['upper'] <= 1, 'Conditional group interval recipe')
                    close(interval['estimate'], row[head]['accuracy'], 'Interval point estimate')


def aggregate(methods):
    result = {}
    for view in VIEWS:
        result[view] = {}
        for head in HEADS:
            means = [statistics.mean(rep[view][head]['accuracy'] for rep in method['repetitions']) for method in methods]
            result[view][head] = {'mean': statistics.mean(means), 'master_means': means,
                                  'minimum': min(means), 'maximum': max(means)}
        means = [statistics.mean(rep[view]['population']['coverage'] for rep in method['repetitions']) for method in methods]
        result[view]['coverage'] = {'mean': statistics.mean(means), 'master_means': means,
                                    'minimum': min(means), 'maximum': max(means)}
    return result


def full_coverage(methods):
    return all(rep[view][head]['valid'] == (512 if view == 'training' else 256)
               and rep[view][head]['coverage'] == 1
               and rep[view]['population']['valid_rows'] == rep[view]['population']['total_rows']
               and rep[view]['population']['valid_source_groups'] == rep[view]['population']['total_source_groups']
               for method in methods for rep in method['repetitions']
               for view in ('training',) + VIEWS for head in HEADS)


def pct(value):
    return f'{100 * value:.2f}'


def repetitions(title, methods):
    content = [title, '', LEGEND + ' Known intact/deleted VALIDATION; all saved fixed fits.', '',
               '| Cohort | Fit | View | Linear % [95% interval] | Neural % [95% interval] | Coverage % |',
               '| ---: | ---: | --- | ---: | ---: | ---: |']
    for master, method in zip(MASTERS, methods):
        for rep in method['repetitions']:
            for view in VIEWS:
                row = rep[view]
                linear, neural = row['ridge_grouped_interval'], row['tiny_grouped_interval']
                content.append(f"| {master} | {rep['id'][4:]} | {'intact' if view == VIEWS[0] else 'deleted'} | "
                               f"{pct(row['ridge']['accuracy'])} [{pct(linear['lower'])}, {pct(linear['upper'])}] | "
                               f"{pct(row['tiny_secondary']['accuracy'])} [{pct(neural['lower'])}, {pct(neural['upper'])}] | "
                               f"{pct(row['population']['coverage'])} |")
    return content


def snapshot(source, target):
    require(source.is_file() and not source.is_symlink(), 'Snapshot regular file')
    if (source.name in LOCAL_ONLY or
            (source.suffix == '.json' and 'witness' in source.name.lower()) or source.suffix == '.pt'):
        return False
    if source.suffix not in TEXT and source.name != 'Makefile':
        return False
    require(not target.exists(), 'Preserve snapshot output: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(sha(source) == sha(target), 'Snapshot byte identity')
    return True


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('doc/results/market_location_scale_synthetic_transfer_v1'))
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Existing cuwacunu container only')
    run = args.run.resolve(strict=True)
    require(run.is_relative_to(ROOT / 'output/runs/market-location-scale-synthetic-transfer') and
            run.name.startswith('run-'), 'Declared one-run output root')
    durable = args.output_dir.resolve()
    require(durable.is_relative_to(ROOT / 'doc/results'), 'Durable evidence stays in doc/results')
    link_root = str(durable.relative_to(ROOT / 'doc'))
    durable.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_RESULTS.md'
    for path in (report_path, durable / 'capture', durable / 'table_rows.json', durable / 'completion.json',
                 durable / 'asset_hashes.json', durable / 'durable_files.json'):
        require(not path.exists(), 'Exclusive report output already exists: ' + str(path))
    inputs = ROOT / 'output/inputs/market-synthetic-transfer-v1'
    evidence = ROOT / 'doc/results/market_location_scale_synthetic_transfer_v1'
    card = ROOT / 'doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md'
    checkpoint = ROOT / 'output/runs/market-location-scale-decoder/run-MAmvTJ/checkpoints/inference-2048.pt'
    settings = ROOT / 'config/market_studies/encoder_location_scale_decoder.conf'
    old_table_path = ROOT / 'doc/results/market_synthetic_transfer_v1/table_rows.json'
    market_completion_path = ROOT / 'doc/results/market_location_scale_decoder_v1/completion.json'
    for path, expected in ((card, CARD), (checkpoint, CHECKPOINT), (settings, SETTINGS),
                           (inputs / 'manifest.json', INPUT), (old_table_path, OLD_TABLE),
                           (market_completion_path, MARKET_COMPLETION)):
        require(sha(path) == expected, 'Frozen source/reference pin: ' + str(path))
    market = load(market_completion_path)
    require(market['full_market_gate_passed'] and market['original_criteria_passed'] == 9 and
            market['strict_criteria_passed'] == 8, 'Independent market pass remains frozen')
    manifest = load(inputs / 'manifest.json')
    require(tuple(manifest['masters']) == MASTERS and len(manifest['members']) == 18, 'Exact staged cohorts/input members')
    input_assets = []
    for member in manifest['members']:
        path = inputs / member['path']
        require(path.resolve().is_relative_to(inputs) and path.stat().st_size == member['bytes'] and
                sha(path) == member['sha256'], 'Unchanged staged input: ' + member['path'])
        input_assets.append(record(path))
    execution = load(run / 'verification.json')
    audit = load(run / 'independent-audit.json')
    admission = load(evidence / 'admission-review.json')
    require(execution['status'] == audit['status'] == admission['status'] == 'passed', 'Admission/execution/independent audit required')
    require(execution['card_sha256'] == audit['card_sha256'] == admission['card_sha256'] == CARD and
            execution['checkpoint_sha256'] == audit['checkpoint_sha256'] == CHECKPOINT and
            execution['admission_sha256'] == sha(evidence / 'admission-review.json'), 'Exact admission/run/audit bindings')
    require(audit['protocol'] == PROTOCOL and audit['encoder_updates'] == 0 and
            audit['auditor_encoder_calls'] == audit['auditor_head_fits'] == 0, 'Saved-only independent audit')
    for key in ('encoder_updates', 'old_encoder_calls', 'old_head_fits', 'bank_calibration_fits', 'decoder_calls'):
        require(execution[key] == 0, 'Forbidden execution work: ' + key)
    source_records = load(run / 'sources-before.json')
    feature_records = load(run / 'features-before-heads.json')
    require(len(source_records) == execution['source_files_unchanged'] and
            len(feature_records) == execution['feature_files_unchanged'], 'Source/feature protection counts')
    verify_records(source_records)
    verify_records(feature_records)
    verify_records(admission['protected'])
    require(str(Path(__file__).resolve()) in admission['protected'], 'Reporter source prospectively admitted')
    require(execution['new_full_cuda_encode_calls'] == 768 and
            execution['new_same_checkpoint_parent_correctness_calls'] == 6, 'Execution call budget')
    features = load(run / 'features/summary.json')
    reader = load(run / 'readouts/reader-summary.json')
    source_before = run / 'features/sources-before.sha256'
    require(source_before.read_bytes() == (run / 'features/sources-after.sha256').read_bytes(),
            'Extraction source ledgers remain exact')
    proc_entries = 0
    for line in source_before.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split('  ', 1)
        require(len(digest) == 64 and all(x in '0123456789abcdef' for x in digest), 'Source digest syntax')
        if name.startswith('/proc/') and name.endswith('/exe'):
            proc_entries += 1
            require(digest == features['binary_sha256'], 'Actual running extractor binary identity')
        else:
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            require(sha(path) == digest, 'Extraction source/input/old-feature pin: ' + name)
    extract_binary = Path('/opt/cuwacunu/build/market-location-scale-transfer/market_location_scale_synthetic_transfer_extract')
    require(proc_entries == 1 and sha(extract_binary) == features['binary_sha256'] and
            str(extract_binary) in admission['protected'], 'Persistent admitted extraction executable')
    require(features['checkpoint_sha256'] == CHECKPOINT and features['card_sha256'] == CARD and
            features['settings_sha256'] == SETTINGS and features['bank_normalizer_identity'] == NORMALIZER,
            'Typed frozen extraction identities')
    require(features['native_width'] == 36 and features['batch_size'] == 8 and features['device'] == 'cuda:0' and
            features['full_model_encode_calls'] == features['total_CUDA_encoder_calls'] == 768 and
            features['parent_correctness_calls'] == 6 and features['parent_correctness_rows'] == 48 and
            features['total_parent_forwards'] == 774 and features['total_rows'] == 6144, 'Actual extraction device/call/row contract')
    for key in ('encoder_updates', 'decoder_calls', 'head_calls', 'CPU_encoder_calls', 'scaler_refits',
                'bank_normalizer_fits', 'calibration_encoder_calls'):
        require(features[key] == 0, 'Forbidden extractor work: ' + key)
    for key in ('all_parameters_and_buffers_bitwise_unchanged', 'all_parameter_gradients_undefined',
                'masked_sentinel_input_invariance', 'source_hashes_unchanged', 'normalizer_unchanged'):
        require(features[key] is True, 'Frozen extraction state/mask: ' + key)
    require(features['labels_passed_to_encoder'] is False and features['resampling'] is False, 'No label/resampling input')
    require(len(features['runs']) == 6 and {(x['mode'], x['master']) for x in features['runs']} ==
            {(mode, master) for mode in MODES for master in MASTERS}, 'Both policies/all cohorts')
    for captured in features['runs']:
        leaf = run / 'features' / captured['mode'] / ('master-' + str(captured['master'])) / 'features.pt'
        require(sha(leaf) == captured['features_sha256'], 'Saved native feature hash')
        require({x['role'] for x in captured['roles']} == {'training', 'validation', 'deleted'} and
                len(captured['roles']) == 3, 'All extraction roles retained')
        for role in captured['roles']:
            rows = 512 if role['role'] == 'training' else 256
            require(role['rows'] == role['valid'] == rows and role['CUDA_encoder_calls'] == rows // 8,
                    'Full extraction support and actual role budget')
    for name, count, key in (('preprocessing-receipts.json', 768, 'receipts'),
                              ('preprocessing-witnesses.json', 18, 'cases'), ('parent-parity.json', 6, 'cases')):
        require(len(load(run / 'features' / name)[key]) == count, 'Declared arithmetic witness count: ' + name)
    require(audit['preprocessing_receipts'] == 768 and audit['bounded_preprocessing_cases'] == 18 and
            audit['parent_prefix_cases'] == 6 and audit['exact_parent_prefix_cells'] == 1536,
            'Independent exact prefix/bounded preprocessing evidence')
    for name in ('frozen-state.json', 'saved-prefix-comparison.json'):
        require((run / 'features' / name).is_file(), 'Frozen state/descriptive precision receipt: ' + name)
    deps_root = ROOT / 'code/evaluation/market_synthetic_transfer_v1/deps'
    dependencies = []
    for name, digest in DEPS.items():
        path = deps_root / name
        require(sha(path) == digest, 'Eight byte-identical fixed evaluator dependencies: ' + name)
        dependencies.append(record(path))
    require(reader['protocol'] == PROTOCOL and reader['model'] == MODEL and reader['native_width'] == 36 and
            reader['device'] == 'cpu' and reader['primary_input_mode'] == MODES[0] and
            reader['secondary_input_mode'] == MODES[1], 'Fixed reader scope')
    for key in ('encoder_calls', 'encoder_updates', 'old_encoder_calls', 'old_head_fits', 'post_encoder_pca_fits'):
        require(reader[key] == 0, 'Forbidden reader work: ' + key)
    old_table = load(old_table_path)
    require(old_table['status'] == 'complete' and old_table['protocol'] == 'market-synthetic-transfer-v1', 'Saved native36 transfer')
    old_reports = old_table['all_saved_repetitions_and_intervals']['new']
    synthetic_reports = old_table['all_saved_repetitions_and_intervals']['reused_reference']
    require(len(old_reports) == len(synthetic_reports) == 3, 'Three saved native36 and synthetic reference cohorts')
    old_methods, new_methods, new_reports = {m: [] for m in MODES}, {m: [] for m in MODES}, []
    reference_methods = []
    for master, old_report, synthetic_report in zip(MASTERS, old_reports, synthetic_reports):
        new_report = load(run / 'readouts' / ('master-' + str(master)) / 'report.json')
        new_reports.append(new_report)
        for report in (old_report, synthetic_report, new_report):
            validate_recipe(report, master)
        reference = next(x for x in synthetic_report['methods'] if x['method'] == 'native_512')
        validate_method(reference, 32)
        reference_methods.append(reference)
        require({x['method'] for x in new_report['methods']} == set(MODES) and len(new_report['methods']) == 2 and
                not new_report['pairs'], 'Exactly two new scaler-policy methods')
        require(new_report['fit_counts'] == {'outer_train_normalizer_fits': 2, 'ridge_fits': 6, 'tiny_fits': 6,
                                            'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'Fixed per-cohort fit budget')
        for mode in MODES:
            for report, destination in ((old_report, old_methods), (new_report, new_methods)):
                method = next(x for x in report['methods'] if x['method'] == mode)
                validate_method(method, 36)
                destination[mode].append(method)
            for rep in new_methods[mode][-1]['repetitions']:
                leaf = run / 'readouts' / ('master-' + str(master)) / mode / rep['id']
                for name in ('fit.pt', 'training-predictions.pt', 'validation-intact-predictions.pt', 'validation-deleted-predictions.pt'):
                    require((leaf / name).is_file(), 'Saved fit/prediction availability')
    reference = aggregate(reference_methods)
    old_scores = {mode: aggregate(old_methods[mode]) for mode in MODES}
    scores = {mode: aggregate(new_methods[mode]) for mode in MODES}
    for mode in MODES:
        for view in VIEWS:
            for head in HEADS + ('coverage',):
                close(old_scores[mode][view][head]['mean'], old_table['new_scores'][mode][view][head]['mean'],
                      'Original native36 aggregation unchanged')
    gates = []
    for mode in MODES:
        for view in VIEWS:
            for head, floor in zip(HEADS, FLOORS[mode][view]):
                close(old_scores[mode][view][head]['mean'], floor, 'Frozen native36 retention floor')
                value = scores[mode][view][head]['mean']
                gates.append({'mode': mode, 'view': view, 'head': head, 'mean': value, 'floor': floor,
                              'cohort_means': scores[mode][view][head]['master_means'],
                              'difference': value - floor, 'arithmetic_tolerance': 1e-12, 'passed': value + 1e-12 >= floor})
    coverage = all(full_coverage(new_methods[mode]) for mode in MODES)
    balanced_pass = coverage and all(x['passed'] for x in gates)
    require(audit['full_coverage'] == coverage and audit['balanced_transfer_gate_passed'] == balanced_pass,
            'Independent transfer coverage/gate agreement')
    for gate in gates:
        other = next(x for x in audit['retention_conditions'] if
                     (x['mode'], x['view'], x['head']) == (gate['mode'], gate['view'], gate['head']))
        close(gate['mean'], other['mean'], 'Independent eight-gate mean')
        require(gate['floor'] == other['floor'] and gate['passed'] == other['passed'], 'Independent exact floor decision')
    disposition = 'verified_transfer_pass_selection_pending' if balanced_pass else 'verified_transfer_failure_native36_balanced'
    closure_path = ROOT / 'doc/results/encoder_search_goal_closure_v1/preservation-after-02-summary.json'
    original_closure_path = ROOT / 'doc/results/encoder_search_goal_closure_v1/preservation-after-01.json'
    closure_pin = None
    if closure_path.is_file() and load(closure_path).get('status') == 'passed':
        closure = load(closure_path)
        require(closure['original_baseline_files'] == closure['verified_files'] == 77831 and
                closure['failed_files'] == 0 and closure['original_receipts_and_scripts_preserved'],
                'Completed original two-repository preservation closure')
        require(sha(original_closure_path) ==
                '22794bfc02b08137686beaa9f70b03a6a22836218f6d49b2bb6cb46d04020d25',
                'Preserved original closure scan receipt identity')
        original_closure = load(original_closure_path)
        require(original_closure['status'] == 'passed' and
                original_closure['verified_files'] is None and original_closure['verified_bytes'] is None and
                sum(row['verified_files'] for row in original_closure['roots']) == closure['verified_files'] and
                sum(row['verified_bytes'] for row in original_closure['roots']) == closure['verified_bytes'] and
                all(row['failed_files'] == 0 and row['status'] == 'unchanged' for row in original_closure['roots']),
                'Additive closure totals correction reproduces preserved per-root scan')
        closure_pin = record(closure_path)
        closure_pin['original_scan_receipt'] = record(original_closure_path)
        closure_pin['aggregate_metadata_correction_only'] = True
    precision = load(run / 'features/preprocessing-receipts.json')['receipts']
    precision_summary = []
    for mode in MODES:
        values = [x for x in precision if x['mode'] == mode]
        precision_summary.append({'mode': mode, 'batch_receipts': len(values),
                                  'observed_cells': sum(x['observed_cells'] for x in values),
                                  'raw_rounding_difference_cells': sum(x['raw_rounding_difference_cells'] for x in values),
                                  'normalized_difference_cells': sum(x['normalized_difference_cells'] for x in values),
                                  'maximum_raw_rounding_delta': max(x['max_raw_rounding_delta'] for x in values),
                                  'maximum_normalized_delta': max(x['max_normalized_delta'] for x in values),
                                  'hidden_zero_all_batches': all(x['hidden_zero'] for x in values),
                                  'mask_metadata_exact_all_batches': all(x['mask_metadata_exact'] for x in values)})
    timings = {key: features[key] for key in ('full_API_inference_seconds', 'external_preprocessing_seconds',
               'parent_correctness_seconds', 'feature_and_receipt_io_seconds', 'state_verification_seconds')}
    panels = []
    for mode in MODES:
        for view in VIEWS:
            panels.append({'dataset_id': 'TEMPO-5', 'task': 'lag_sign', 'recipe_id': 'tempo-5-slow-fast-nine-feature-v1',
                           'designed_complexity_level': 5, 'complexity_scale_max': 5, 'view': view,
                           'input_policy': mode, 'head_fitting_dataset_id': 'TEMPO-5', 'scoring_dataset_id': 'TEMPO-5',
                           'rows': [{'method': model, 'size': width, 'evidence': stage, 'encoder_fitting_dataset_id': fit,
                                     'linear_head_accuracy': data[view]['ridge']['mean'],
                                     'neural_head_accuracy': data[view]['tiny_secondary']['mean'],
                                     'coverage': data[view]['coverage']['mean']}
                                    for model, width, stage, fit, data in
                                    ((REFERENCE, 32, 'reused_saved_synthetic_scores', 'TEMPO-5', reference),
                                     (NATIVE36, 36, 'reused_saved_native36_transfer', 'MKT-HIST-2', old_scores[mode]),
                                     (MODEL, 36, 'new_frozen_checkpoint_transfer', 'MKT-HIST-2', scores[mode]))]})
    capture = durable / 'capture'
    capture.mkdir()
    # Hash all assets; never export tensors or full raw/native witness arrays into docs.
    assets = [record(path) for path in sorted(run.rglob('*')) if path.is_file()]
    for path in sorted(run.rglob('*')):
        if path.is_file():
            snapshot(path, capture / 'run' / path.relative_to(run))
    admitted = set(admission['protected']) | set(source_records)
    for name in sorted(admitted):
        path = Path(name)
        if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'output'):
            snapshot(path, capture / 'admitted' / path.relative_to(ROOT))
    for path in (old_table_path, market_completion_path, card, inputs / 'manifest.json') + ((closure_path, original_closure_path) if closure_pin else ()):
        snapshot(path, capture / 'references' / path.relative_to(ROOT))
    for path in sorted(evidence.rglob('*')):
        if (path.is_file() and not path.is_relative_to(capture) and path.name not in
                {'table_rows.json', 'completion.json', 'asset_hashes.json', 'durable_files.json'}):
            snapshot(path, capture / 'admission' / path.relative_to(evidence))
    table = {'kind': 'market_location_scale_synthetic_transfer_results_v1', 'protocol': PROTOCOL,
             'status': 'complete', 'disposition': disposition, 'panels': panels, 'new_scores': scores,
             'reused_native36_scores': old_scores, 'reused_synthetic_reference_scores': reference,
             'retention_conditions': gates, 'retention_conditions_passed': sum(x['passed'] for x in gates),
             'retention_conditions_total': 8, 'full_coverage_all_roles_modes_heads': coverage,
             'balanced_transfer_gate_passed': balanced_pass, 'all_saved_repetitions_and_intervals':
             {'new': new_reports, 'native36': old_reports, 'synthetic_reference': synthetic_reports},
             'precision_summary': precision_summary, 'timings': timings, 'encoder_updates': 0,
             'original_two_repository_preservation_closure': closure_pin,
             'selection_updated': False, 'primary_changed': False}
    save_json(durable / 'table_rows.json', table)
    save_json(durable / 'asset_hashes.json', {'kind': 'market_location_scale_synthetic_transfer_assets_v1',
              'run_root': str(run), 'run_assets': assets, 'input_assets': input_assets,
              'admitted_assets': admission['protected'], 'dependency_pins': dependencies,
              'card': record(card), 'checkpoint': record(checkpoint), 'settings': record(settings),
              'old_native36_table': record(old_table_path), 'market_completion': record(market_completion_path),
              'reporter': record(Path(__file__).resolve()), 'independent_audit': record(run / 'independent-audit.json'),
              'wrapper_verification': record(run / 'verification.json'),
              'original_two_repository_preservation_closure': closure_pin,
              'local_only_array_receipts': [record(run / 'features' / name) for name in sorted(LOCAL_ONLY)]})
    content = ['# Frozen V30 market encoder on TEMPO-5', '',
               'Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.', '',
               ('**V30 passes all eight declared transfer floors and full coverage.** Selection remains pending the separate verified decision.'
                if balanced_pass else '**V30 fails the complete declared transfer protection gate.** Keep native36 balanced and preserve V30 as the verified market9/9 reference.'), '',
               f"{sum(x['passed'] for x in gates)}/8 exact native36 head/view floors pass; coverage across every role/cohort/policy/head is {'100%' if coverage else 'below100%'}. "
               'The completed market result remains9/9 original and8/8 strict independently of this transfer. RPB-v22 remains primary; this report changes no leader pointer or ledger.', '',
               'This is new frozen-checkpoint extraction/scoring of RPB-v30.market-01, with zero encoder updates after its2048 parent plus2048 branch updates. '
               'The exact final checkpoint is identical across three known synthetic cohorts. Saved native36 market-transfer and synthetic-trained v22 controls are reused without encoder replay/head refits. '
               'The v22 control retains its own synthetic normalization pipeline and is contextual in both policy panels; native36 is the matched-policy retention reference.', '',
               'Each cohort941101/941102/941103 reuses512 TRAIN examples/256 paired source groups and256 known VALIDATION examples/128 disjoint groups, with the exact saved extra30% erasure. '
               'C3/H128/F9 observations, cohorts and masks are fixed. Three cohorts are data cohorts, not newly trained encoders; three readout fits are head variation. '
               'Native36 enters fixed classifiers directly, without PCA. TRAIN-only outer feature scaling, ridge1 and neural16-tanh/Adam.01/100 updates retain repetitions2701/2802/2903. '
               'D36 binary heads have74/626 parameters and uint64 seeds13088411117524848790/10018485049408248344/9083751155097190455; D32 controls retain66/562 and their own fixed seeds. '
               'Six outer TRAIN normalizers,18 ridge and18 neural fits are new; each fixed fit scores TRAIN/intact/deleted without refitting.', '']
    for panel in panels:
        content += [LEGEND + (' Known intact VALIDATION.' if panel['view'] == VIEWS[0] else ' Known VALIDATION with saved additional30% coordinate erasure.'), '',
                    'Input policy: `' + panel['input_policy'] + '`. Equal three-cohort means; neural means include all three fixed fits per cohort.', '',
                    '| Method | Size | Linear head % | Neural head % | Coverage % |',
                    '| --- | ---: | ---: | ---: | ---: |']
        for row in panel['rows']:
            content.append(f"| {row['method']} | {row['size']} | {pct(row['linear_head_accuracy'])} | {pct(row['neural_head_accuracy'])} | {pct(row['coverage'])} |")
        content += ['']
    content += ['## Fixed retention decisions', '', LEGEND + ' Both scaler policies; intact/deleted known VALIDATION.', '',
                '| Policy | View | Head | Exact mean | Frozen floor | Pass |',
                '| --- | --- | --- | ---: | ---: | --- |']
    for gate in gates:
        content.append(f"| {gate['mode']} | {gate['view']} | {gate['head']} | {gate['mean']!r} | {gate['floor']!r} | {'pass' if gate['passed'] else 'fail'} |")
    content += ['', 'All eight floors were declared before extraction; tolerance1e-12 is arithmetic only, with no accuracy allowance. '
                'The old primary intact linear mean has a one-ULP aggregation difference0.9895833333333333/0.9895833333333334; use the prospective floor and stated tolerance. '
                'No post-hoc per-cohort or individual-neural-fit floor is added. All repetitions, including failures, remain visible.', '',
                '## Precision, frozen state and actual work', '',
                'V30 receives genuine CPU-F64 observations. Its frozen parent pipeline is `F32((F64(F32(raw64))-mean64)/scale64)` on observed cells, hidden0. '
                'The old native36 transfer used `F32((raw64-mean64)/scale64)`. Their rounding orders can differ; the adapter/API is frozen and no alternative precision arm is selected. '
                'Both policies reuse the exact saved scaler statistics. The bank keeps its embedded V28 normalizer unchanged, with zero calibration fits/forwards.', '',
                'There are exactly768 full V30 CUDA calls/6144 rows and6 same-loaded-checkpoint parent-only correctness calls/48 repeated rows, hence774 parent forwards. '
                'All1536 compared prefix cells are bitwise equal on the same V30-normalized inputs. Saved old native36 prefix differences are descriptive across different preprocessing, not equality gates. '
                'There are zero CPU encodes, encoder updates, decoder calls, gradient/optimizer work, new masks, TEST inputs, resampling or old-head fits. '
                'All parameters/buffers/flags and unregistered bank-normalizer/scaler state are checked across each role and exit; gradients stay undefined.', '',
                'All768 preprocessing receipts and18 bounded first-role witnesses are independently checked, with exact sanitized hidden zeros/masks/metadata and18 hidden-poison preprocessing checks. '
                'Raw F32 rounding is bitwise checked; CPU-F64 normalized arithmetic permits at most two F32 ULP as declared by the auditor. '
                'The six actual same-input parent-prefix checks are exact. Raw/native arrays remain ignored local assets; their hashes are preserved.', '',
                LEGEND + ' All extraction roles; repeated inputs counted once per actual batch call.', '',
                '| Policy | Batch receipts | Raw-rounding differing cells | Normalized differing cells | Max normalized delta |',
                '| --- | ---: | ---: | ---: | ---: |']
    for row in precision_summary:
        content.append(f"| {row['mode']} | {row['batch_receipts']} | {row['raw_rounding_difference_cells']} | {row['normalized_difference_cells']} | {row['maximum_normalized_delta']!r} |")
    content += ['', 'Full raw delta maxima, observed counts and saved-prefix difference summaries are retained in captured compact JSON. '
                'The new extraction/scoring report is saved-output arithmetic, not an encoder training experiment; train error and GPU training seconds are not measured for this frozen stage.', '',
                '## Costs and evidence', '',
                'Full API inference time includes internal parent preprocessing/device transfers; external preprocessing, six parent correctness calls, feature/receipt I/O and frozen-state verification are separately recorded. '
                'These are descriptive costs, not a pure network-speed benchmark. Wrapper wall time also includes provenance hashing and preservation work.', '',
                LEGEND + ' Frozen extraction across all cohorts/policies and CPU fixed heads.', '',
                '| Stage | Seconds |', '| --- | ---: |']
    for key, value in timings.items():
        content.append(f'| {key} | {value!r} |')
    content += [f"| fixed head fitting/scoring/output I/O | {reader['head_fitting_and_scoring_seconds_including_io']!r} |",
                f"| saved scalar audit wall | {execution['saved_arithmetic_audit_wall_seconds']!r} |", '',
                f"Independent audit passes{audit['checks']} checks; this reporter performs independent scalar count/mean/coverage/seed/interval-point/gate checks as well. "
                'Per-row predictions are saved as ignored tensors, not JSON/CSV. No prediction tensor replay or bootstrap-quantile recomputation occurs in the stdlib audit/report. '
                'Full prediction correctness and bootstrap quantiles rely on the unchanged eight-file evaluator, exact recipes/compiler dependency capsule and saved artifact hashes. '
                'Intervals are95% source-group percentile bootstrap1000 replicates conditional on each fitted readout, not uncertainty across trained encoders or unseen domains.', '',
                'Admission reuses V30’s passed artificial CUDA fixture, verifies55 checkpoint pins and99 reusable source pairs, captures both binaries and actual compiler dependencies, '
                'and seals the one-run extraction. Preflight attempt01 stopped before quality on an externally changed unrelated roadmap pin; its receipt/source is preserved and the corrected contract excludes that unrelated input. '
                'No claim is made that the roadmap remained unchanged since V30. Historical output/result/card/report/current-ledger protections pass before/after; '
                + ('the original77,831-file two-repository preservation closure also passes. Its original scan receipt01 and additive totals summary02 are both bound. '
                   'The original PowerShell Measure-Object aggregation over ordered hashtables left only the top-level file/byte totals null; '
                   'its preserved per-root SHA/size/kind scan passed with1500+76331 files and415100170+20169895215 bytes. '
                   'Summary02 corrects those scalar totals without a rescan or changing the original receipt/verifier. '
                   if closure_pin else 'the broader original two-repository preservation check remains pending goal closure. ')
                + 'No source or quality retry is silently discarded.', '',
                'Checkpoint SHA256: `' + CHECKPOINT + '`. Card SHA256: `' + CARD + '`. '
                f'See [exact tables/repetitions]({link_root}/table_rows.json), '
                f'[asset inventory]({link_root}/asset_hashes.json) and '
                f'[completion]({link_root}/completion.json). '
                'All weights/embeddings/predictions/raw witness arrays remain local and ignored; compact receipts and admitted sources are captured.', '',
                '## Every cohort, fit and conditional interval', '']
    for mode in MODES:
        content += repetitions(MODEL + ' — ' + mode + ' — new frozen transfer', new_methods[mode]) + ['']
        content += repetitions(NATIVE36 + ' — ' + mode + ' — reused saved transfer', old_methods[mode]) + ['']
    content += repetitions(REFERENCE + ' — reused synthetic-trained reference', reference_methods) + ['', '## Next direction', '']
    content += [('The eight transfer means and coverage pass. Root must independently verify the frozen report and record any balanced-reference selection as a separate additive event; this report does not change the live ledger.'
                 if balanced_pass else 'Retain V30 as the verified market9/9 reference and native36 as balanced. Preserve this transfer failure without scaler/head/seed/precision rescue or new encoder training under this card.'), '',
                'Retain every old/new checkpoint and partial task leader. RPB-v22 stays primary. TEMPO-5 is known-cohort development evidence for this compatible C3/H128/F9 family; TEMPO-4/AMP-2 geometry remains unmeasured. '
                'No universal ranking, primary promotion, publication, commit, push or deployment follows.']
    save(report_path, '\n'.join(content).rstrip() + '\n')
    completion = {'kind': 'market_location_scale_synthetic_transfer_completion_v1', 'protocol': PROTOCOL,
                  'status': 'passed', 'report_stage': 'complete', 'disposition': disposition, 'model_tag': MODEL,
                  'dataset_id': 'TEMPO-5', 'designed_complexity_level': 5, 'complexity_scale_max': 5,
                  'masters': list(MASTERS), 'new_encoder_updates': 0, 'inherited_parent_updates': 2048,
                  'inherited_change_branch_updates': 2048, 'full_CUDA_encode_calls': 768,
                  'same_checkpoint_parent_correctness_calls': 6, 'total_parent_forwards': 774,
                  'CPU_encoder_calls': 0, 'new_bank_calibration_fits': 0, 'new_ridge_fits': 18,
                  'new_neural_fits': 18, 'outer_TRAIN_normalizer_fits': 6, 'old_control_head_fits': 0,
                  'retention_conditions_passed': sum(x['passed'] for x in gates), 'retention_conditions_total': 8,
                  'retention_conditions': gates,
                  'full_coverage_all_roles_modes_heads': coverage, 'balanced_transfer_gate_passed': balanced_pass,
                  'market_gate_passed_separately': True, 'market_completion': record(market_completion_path),
                  'primary_changed': False, 'ledger_or_selection_updated': False,
                  'card_sha256': CARD, 'checkpoint_sha256': CHECKPOINT, 'staged_manifest_sha256': INPUT,
                  'old_native36_table_sha256': OLD_TABLE, 'market_completion_sha256': MARKET_COMPLETION,
                  'admission_sha256': sha(evidence / 'admission-review.json'), 'independent_audit_sha256': sha(run / 'independent-audit.json'),
                  'wrapper_verification_sha256': sha(run / 'verification.json'), 'reporter_sha256': sha(Path(__file__).resolve()),
                  'original_two_repository_preservation_closure': closure_pin,
                  'report_sha256': sha(report_path), 'table_rows_sha256': sha(durable / 'table_rows.json'),
                  'asset_hashes_sha256': sha(durable / 'asset_hashes.json'), 'timings': timings,
                  'head_fitting_scoring_IO_seconds': reader['head_fitting_and_scoring_seconds_including_io'],
                  'reporting_hash_capture_seconds': time.monotonic() - started, 'reporter_scalar_checks': CHECKS,
                  'independent_audit_checks': audit['checks'], 'tensor_deserializations_by_reporter': 0,
                  'encoder_or_head_calls_by_reporter': 0, 'prediction_replay_or_bootstrap_quantile_recompute': False,
                  'local_array_witnesses_exported': False, 'all_reused_control_scores_preserved': True}
    save_json(durable / 'completion.json', completion)
    verify_records(admission['protected'])
    verify_records(feature_records)
    # This manifest is exclusive and self-excluding; later independent verification is additive.
    members = [{'path': str(path.relative_to(durable)), 'sha256': sha(path), 'bytes': path.stat().st_size}
               for path in sorted(durable.rglob('*')) if path.is_file()]
    save_json(durable / 'durable_files.json', members)
    print(json.dumps(completion, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

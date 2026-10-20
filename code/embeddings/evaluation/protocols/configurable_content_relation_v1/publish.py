#!/usr/bin/env python3
"""Independent saved-array publication; no torch, encoder execution or head fits."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import gate

sys.dont_write_bytecode = True
ROOT = Path('/mityane')
ORIGINAL = ROOT / 'code/embeddings/output/runs/rpb-v20-capacity/admission-0gN4uD'
BRANCH = ROOT / 'code/embeddings/output/runs/rpb-v20-branches/diagnostic-YA3Nws'
ORIGINAL_MANIFEST = '62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83'
BRANCH_MANIFEST = 'cefa413c560bdf2ab1eee5be2e3f34feca6e2f723a99627f9eeae683c4c11d93'
ORIGINAL_SOURCE = '8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1'
ORIGINAL_INPUTS = '6121773c0b97e5490f191a9f8ff013e90f27a9a557376ac8256bdbb2d6af6091'
PROTOCOL = 'configurable-content-relation-v1'
VIEWS = [('training', 'training', 'training', 512),
         ('validation_intact', 'validation', 'validation', 256),
         ('validation_deleted', 'deleted', 'validation-deleted', 256)]
REPETITIONS = ('rep-2701', 'rep-2802', 'rep-2903')
PAIR_IDS = ('candidate_native_512_minus_native_A_0', 'candidate_native_512_minus_native_A_512',
            'candidate_learned_20_512_minus_learned_20_0', 'candidate_learned_20_512_minus_learned_20_512')
READS = {}
CHECKS = 0


def check(ok, why):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise ValueError(why)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def remember(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
          and path.is_file() and path.stat().st_nlink == 1, 'direct single-link file: ' + str(path))
    digest = sha(path)
    check(path not in READS or READS[path] == digest, 'file changed during saved audit')
    READS[path] = digest
    return digest


def document(path):
    remember(path)
    return json.loads(path.read_text())


def module(path, name):
    remember(path)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def inventory(root, expected):
    check(remember(root / 'artifacts.sha256') == expected, 'pinned original artifact inventory')
    done = document(root / 'complete.json')
    check(done['status'] == 'complete' and done['artifacts_manifest_sha256'] == expected,
          'completed original evidence')
    entries = {}
    for line in (root / 'artifacts.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = Path(name)
        check(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
              and not relative.is_absolute() and '..' not in relative.parts
              and relative.as_posix() == name and name not in entries, 'canonical historical artifact path')
        entries[root / relative] = digest
    return entries


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def average(values):
    check(bool(values) and all(v is not None and math.isfinite(v) for v in values), 'complete finite aggregate')
    return math.fsum(values) / len(values)


def validate_stage(stage, masters):
    check(stage in ('screen', 'third') and masters ==
          ([941101, 941102] if stage == 'screen' else [941103]), 'stage directory binds allocated cohorts')


def validate_pairs(pairs, masters):
    expected = {(master, name, rep, view) for master in masters for name in PAIR_IDS
                for rep in REPETITIONS for view in gate.VIEWS}
    actual = [(p['master'], p['id'], p['repetition'], p['view']) for p in pairs]
    check(len(actual) == len(set(actual)) == len(expected) and set(actual) == expected,
          'exact declared paired comparison population, identities and views')


def merge_query_reads(external):
    for path, digest in external.items():
        check(remember(path) == digest, 'imported query arithmetic input hash binding')


def prediction_key(master, method, repetition, view, is_new):
    return master, 'candidate_' + method if is_new else method, repetition, view


def validate_checkpoint_metadata(checkpoint, codec, arithmetic, point):
    expected_text = dict(
        artifact_kind='rpb_content_ordered_relation_inference_checkpoint_v1',
        model_tag='RPB-v22',
        architecture_id='configurable-fixed-spectral-plus-zero-start-content-ordered-lag-v1',
        native_layout=('learned-base-shape-plus-zero-start-content-ordered-lag-projection;'
            'unchanged-fixed-four-per-semantic-pair;'
            'lag-bank-odd-even-no-additive-support;canonical-pair-feature-i-feature-j-lag;'
            'original-tick-lags1-2-4-8;four-endpoint-common-support;'
            'joint-endpoint-centering;min-common2;energy-floor1e-12'),
        dtype='float32', training_device='cuda:0')
    check(point in (0, 512) and all(codec.text(checkpoint[key]) == value for key, value in expected_text.items())
          and arithmetic.tensor(checkpoint, 'training_updates', 'LongStorage', []) == [point]
          and arithmetic.tensor(checkpoint, 'config_integer_v1', 'LongStorage', [15]) ==
              [3, 128, 9, 8, 64, 32, 3, 4, 1, 1, 2, 0, 0, 256, 128]
          and arithmetic.tensor(checkpoint, 'config_number_v1', 'DoubleStorage', [6]) ==
              [0., 1e-5, .25, 1., 1e-6, 1.], 'candidate typed checkpoint tag/updates/resolved geometry')
    check(arithmetic.tensor(checkpoint, 'channel_ids', 'LongStorage', [3]) == [0, 1, 2]
          and arithmetic.tensor(checkpoint, 'relation_lags_v1', 'LongStorage', [4]) == [1, 2, 4, 8]
          and arithmetic.tensor(checkpoint, 'relation_minimum_common', 'LongStorage', []) == [2]
          and arithmetic.tensor(checkpoint, 'relation_energy_floor', 'DoubleStorage', []) == [1e-12],
          'candidate typed semantic and relation contracts')


def publish(capsule, readouts, destination):
    check(Path('/.dockerenv').exists() and Path.cwd() == ROOT, 'managed embedding container required')
    check(capsule.is_absolute() and capsule.resolve(strict=True) == capsule
          and capsule.parent == ROOT / '.temp/embeddings/runs/rpb-v22-content-relation'
          and readouts.parent.parent == capsule and readouts.parent.name in ('screen', 'third')
          and not capsule.is_relative_to(ORIGINAL)
          and not capsule.is_relative_to(BRANCH), 'separate direct candidate capsule')
    check(destination.parent == readouts.parent and not destination.exists() and not destination.is_symlink(),
          'new capsule publication leaf')
    original_inventory = inventory(ORIGINAL, ORIGINAL_MANIFEST)
    branch_inventory = inventory(BRANCH, BRANCH_MANIFEST)
    check(len(original_inventory) == 815, 'complete original artifact population')
    frozen = document(capsule / 'freeze/receipt.json')
    source_path = capsule / 'freeze/sources.sha256'
    source = remember(source_path)
    card = remember(capsule / 'freeze/card.md')
    check(frozen['source_fingerprint'] == source and frozen['card_sha256'] == card,
          'exact frozen candidate source and card')
    captured = {}
    for line in source_path.read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = Path(name)
        check(not relative.is_absolute() and '..' not in relative.parts and name not in captured,
              'closed candidate source closure')
        captured[name] = digest
        check(remember(ROOT / relative) == remember(capsule / 'freeze/source' / relative) == digest,
              'executed source remains byte-identical to capture')

    def imported(relative, name):
        check(relative in captured, 'arithmetic helper included in frozen source closure')
        return module(ROOT / relative, name)

    codec = imported('.archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py', 'relation_codec')
    check(captured['.archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py'] ==
          '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d', 'pinned independent codec')
    arithmetic = imported('code/embeddings/evaluation/protocols/configurable_branch_v1/publish.py', 'relation_math')
    query_math = imported('code/embeddings/evaluation/protocols/configurable_capacity_v1/publish.py', 'relation_query_math')
    scaler_math = imported('code/embeddings/evaluation/protocols/configurable_capacity_v1/saved_audit.py', 'relation_scaler_math')
    query_math.SCALER_AUDIT = scaler_math
    query_math.check = check

    def load(path):
        digest = remember(path)
        historical = original_inventory if path.is_relative_to(ORIGINAL) else branch_inventory if path.is_relative_to(BRANCH) else None
        if historical is not None:
            check(historical.get(path) == digest, 'historical consumed archive unchanged')
        return codec.load(path)

    def old_document(path):
        value = document(path)
        historical = original_inventory if path.is_relative_to(ORIGINAL) else branch_inventory
        check(historical.get(path) == READS[path], 'historical report unchanged')
        return value

    report = document(readouts / 'report.json')
    done = document(readouts / 'complete.json')
    replay = document(readouts / 'arithmetic-audit.json')
    masters = report['masters']
    validate_stage(readouts.parent.name, masters)
    count = len(masters)
    for key, expected in dict(protocol=PROTOCOL, dataset='TEMPO-5', source_fingerprint=source,
                              card_sha=card, input_manifest_sha=ORIGINAL_INPUTS,
                              original_source_fingerprint=ORIGINAL_SOURCE,
                              original_artifact_manifest_sha=ORIGINAL_MANIFEST,
                              branch_artifact_manifest_sha=BRANCH_MANIFEST,
                              encoder_calls=0, encoder_updates=0, control_head_fits=0,
                              validation_fits=0, pca_fits=0, raw_fits=0,
                              config_sha=captured['code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/config/content_ordered_relation.conf'],
                              head_fitted_pairs=6 * count, head_fits=12 * count).items():
        check(report[key] == expected, 'closed readout report binding: ' + key)
    check(done['status'] == 'complete' and done['protocol'] == PROTOCOL
          and done['encoder_calls'] == 0 and done['head_fitted_pairs'] == done['saved_arithmetic_fits_replayed'] == 6 * count
          and done['report_sha256'] == READS[readouts / 'report.json']
          and done['arithmetic_audit_sha256'] == READS[readouts / 'arithmetic-audit.json'],
          'completed saved readout arithmetic bindings')
    check(replay['protocol'] == 'saved-content-relation-arithmetic-v1'
          and replay['encoder_calls'] == replay['head_fits'] == 0
          and replay['fitted_pairs_replayed'] == 6 * count
          and replay['prediction_archives'] == 18 * count,
          'complete candidate arithmetic replay scope')
    replay_predictions = {r['prediction']: r for r in replay['records']}
    check(len(replay_predictions) == len(replay['records']) == 18 * count, 'unique complete prediction witnesses')
    checked_replay = set()
    predictions = {}
    scores = {name: {} for name in gate.METHODS}
    training = []
    queries = {}
    fixed_cells = initial_cells = new_pairs = reused_pairs = 0

    def inspect_method(directory, metadata, method_name, master, originals, exports, point, width, offset, is_new):
        nonlocal new_pairs, reused_pairs
        method = next(m for m in metadata['methods'] if m['method'] == method_name)
        check(method['status'] == 'measured' and method['size'] == width
              and not method['inputs_train_prepared'] and method['outer_train_normalizer_fits'] == 1,
              'declared native/subset normalization and measured method')
        check([r['id'] for r in method['repetitions']] == list(REPETITIONS), 'all three fixed head repetitions')
        surfaces = {}
        for view, prefix, _, rows in VIEWS:
            surfaces[view] = arithmetic.audit_surface(
                load(directory / method_name / (view.replace('_', '-') + '-features.pt')),
                originals[view], exports[point], prefix, rows, width, offset, master)
        values = {view: dict(coverage=surfaces[view]['population']['coverage']) for view, _, _, _ in VIEWS}
        fits = {}
        for rep in method['repetitions']:
            path = directory / method_name / rep['id'] / 'fit.pt'
            check(rep['fit_artifact'] == str(path.relative_to(directory)), 'declared fit artifact association')
            fit = load(path)
            arithmetic.audit_fit(fit, surfaces['training'], int(rep['id'].split('-')[1]))
            fits[rep['id']] = fit
            for view, _, _, _ in VIEWS:
                prediction_path = path.parent / (view.replace('_', '-') + '-predictions.pt')
                saved = load(prediction_path)
                witness = None
                if is_new:
                    relative = str(prediction_path.relative_to(readouts))
                    check(relative in replay_predictions and relative not in checked_replay, 'unique saved candidate prediction replay')
                    checked_replay.add(relative)
                    witness = replay_predictions[relative]
                actual = arithmetic.audit_prediction(saved, surfaces[view], fit, rep[view], witness)
                for head in gate.HEADS:
                    values[view].setdefault(head + '_repetitions', []).append(actual[head])
                predictions[prediction_key(master, method_name, rep['id'], view, is_new)] = (saved, surfaces[view], prediction_path)
            if is_new:
                new_pairs += 1
            else:
                reused_pairs += 1
        for view in values:
            for head in gate.HEADS:
                values[view][head] = average(values[view][head + '_repetitions'])
        return values

    check([c['master'] for c in report['cohorts']] == masters, 'every allocated cohort retained in order')
    for cohort in report['cohorts']:
        master = cohort['master']
        originals = {view: load(ORIGINAL / 'quality-data' / f'master-{master}' / (role + '.pt'))
                     for view, _, role, _ in VIEWS}
        arithmetic.audit_source_partition(
            arithmetic.tensor(originals['training'], 'source_ids', 'LongStorage', [512]),
            arithmetic.tensor(originals['validation_intact'], 'source_ids', 'LongStorage', [256]),
            arithmetic.tensor(originals['validation_deleted'], 'source_ids', 'LongStorage', [256]))
        reference_exports = {p: load(ORIGINAL / 'profiles/A' / f'master-{master}' / f'point-{p}/features.pt') for p in (0, 512)}
        candidate_root = capsule / 'candidates' / f'master-{master}'
        candidate_exports = {p: load(candidate_root / f'point-{p}/features.pt') for p in (0, 512)}
        producer = document(candidate_root / 'report.json')
        producer_done = document(candidate_root / 'complete.json')
        progress = document(candidate_root / 'progress.json')
        check(producer['progress'] == progress, 'complete producer/standalone update witness equality')
        check(producer_done['status'] == 'complete'
              and producer_done['report_sha256'] == READS[candidate_root / 'report.json'], 'completed CUDA producer binding')
        for key, expected in dict(protocol=PROTOCOL, model_tag='RPB-v22.alt-01', master=master,
                                  source_fingerprint=source, card_sha=card, input_manifest_sha=ORIGINAL_INPUTS).items():
            check(producer[key] == progress[key] == expected, 'CUDA producer binding: ' + key)
        check(producer['config_sha'] == progress['config_sha'] ==
              captured['code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/config/content_ordered_relation.conf'],
              'producer and progress exact frozen candidate settings')
        check(progress['attempted'] == progress['completed'] == 512 and progress['skipped_attempts'] == 0
              and progress['CPU_encoder_calls'] == 0 and progress['training_device'] == 'cuda:0'
              and progress['weights_changed'] and progress['parameter_count'] == progress['cuda_parameter_count'],
              'actual completed CUDA training scope')
        check(len(progress['losses']) == 512 and progress['sampled_rows'] == 4096,
              'complete unskipped batch8 update trace')
        original_progress = old_document(ORIGINAL / 'profiles/A' / f'master-{master}' / 'progress.json')
        check(progress['initialization_seed'] == original_progress['initialization_seed'], 'unchanged encoder initialization seed')
        check(progress['relation_gradient_updates'] > 0
              and math.isfinite(progress['relation_weight_max_absolute_change'])
              and progress['relation_weight_max_absolute_change'] > 0,
              'new relation weights receive gradients and actually change')
        positive_relation_updates = 0
        for index, trace in enumerate(progress['losses'], 1):
            check(trace['attempted'] == trace['completed'] == index
                  and trace['target_cells'] > 0 and trace['gradient_norm'] > 0
                  and math.isfinite(trace['gradient_norm']) and math.isfinite(trace['waveform_loss'])
                  and trace['dynamics_loss'] == 0 and trace['total_loss'] == trace['waveform_loss']
                  and len(trace['sampled_indices']) == 8, 'finite waveform-only ordered CUDA update trace')
            control_trace = original_progress['losses'][index - 1]
            for field in ('attempted', 'completed', 'target_cells', 'mask_seed', 'torch_seed',
                          'hidden_sha256', 'visible_sha256', 'sampled_indices',
                          'context_requested', 'context_actual', 'context_restored'):
                check(trace[field] == control_trace[field], 'matched saved training rows/masks/streams: ' + field)
            check(math.isfinite(trace['relation_gradient_norm']) and trace['relation_gradient_norm'] >= 0,
                  'finite per-update new projection gradient')
            positive_relation_updates += trace['relation_gradient_norm'] > 0
        check(positive_relation_updates == progress['relation_gradient_updates'], 'new projection gradient counter')
        for point, export in candidate_exports.items():
            check(arithmetic.text(export, 'protocol') == PROTOCOL
                  and arithmetic.text(export, 'artifact_kind') == 'configurable_content_relation_native_feature_views_v1'
                  and arithmetic.text(export, 'model_tag') == 'RPB-v22.alt-01'
                  and arithmetic.text(export, 'source_fingerprint') == source
                  and arithmetic.text(export, 'card_sha') == card
                  and arithmetic.text(export, 'input_manifest_sha') == ORIGINAL_INPUTS
                  and arithmetic.text(export, 'config_sha') == producer['config_sha']
                  and arithmetic.text(export, 'scaler_identity') == progress['scaler_identity']
                  and arithmetic.scalar(export, 'master') == master
                  and arithmetic.scalar(export, 'point') == point
                  and arithmetic.scalar(export, 'native_width') == 32, 'saved CUDA feature metadata')
            for view, prefix, _, rows in VIEWS:
                candidate = arithmetic.tensor(export, prefix + '_values', 'DoubleStorage', [rows, 32])
                reference = arithmetic.tensor(reference_exports[point], prefix + '_values', 'DoubleStorage', [rows, 32])
                check(arithmetic.tensor(export, prefix + '_valid', 'BoolStorage', [rows])
                      == arithmetic.tensor(reference_exports[point], prefix + '_valid', 'BoolStorage', [rows]),
                      'candidate/reference export validity equality')
                if point == 0:
                    check(export[prefix + '_values']['raw'] == reference_exports[point][prefix + '_values']['raw'],
                          'bit-exact initial candidate/control values before head reuse')
                    initial_cells += rows * 32
                check(all(candidate[b * 32 + d] == reference[b * 32 + d]
                          for b in range(rows) for d in range(20, 32)), 'exact fixed12 preservation')
                fixed_cells += rows * 12
                check(arithmetic.tensor(export, prefix + '_source_ids', 'LongStorage', [rows])
                      == arithmetic.tensor(originals[view], 'source_ids', 'LongStorage', [rows])
                      and arithmetic.text(export, prefix + '_source_ids_json')
                      == arithmetic.text(originals[view], 'source_ids_json'), 'explicit candidate source row association')

        candidate_directory = readouts / f'master-{master}' / 'models'
        candidate_metadata = document(candidate_directory / 'report.json')
        check(candidate_metadata == cohort['candidate_heads'] and candidate_metadata['recipe'] == arithmetic.RECIPE
              and candidate_metadata['master_seed'] == str(master)
              and [m['method'] for m in candidate_metadata['methods']] == ['native_512', 'learned_20_512'],
              'exact two new methods under fixed shared head recipes')
        check(candidate_metadata['fit_counts'] == dict(outer_train_normalizer_fits=2,
              ridge_fits=6, tiny_fits=6, validation_fits=0, encoder_calls=0, pca_fits=0),
              'complete shared two-method fitting scope')
        reference_directory = ORIGINAL / 'readouts' / f'master-{master}' / 'models'
        reference_metadata = old_document(reference_directory / 'report.json')
        branch_directory = BRANCH / 'readouts' / f'master-{master}' / 'readouts'
        branch_metadata = old_document(branch_directory / 'report.json')
        reused = cohort['reuse']
        check(len(reused['methods']) == 5 and {m['method'] for m in reused['methods']} ==
              {'native_A_0', 'native_A_512', 'learned_20_0', 'learned_20_512', 'fixed_12'},
              'complete declared saved-control reuse ledger')
        for item in reused['methods']:
            expected_directory = reference_directory if item['method'].startswith('native_') else branch_directory
            check(item['report_path'] == str(expected_directory / 'report.json')
                  and item['root'] == str(expected_directory / item['method'])
                  and item['head_fits_reused'] == 3, 'saved-control method identity and fit counter')
        for item in reused['files']:
            path = Path(item['path'])
            digest = remember(path)
            historical = original_inventory if path.is_relative_to(ORIGINAL) else branch_inventory
            check(digest == item['sha256'] == historical.get(path), 'reuse receipt original artifact bytes')
        specifications = [
            ('candidate_full', candidate_directory, candidate_metadata, 'native_512', candidate_exports, 512, 32, True),
            ('candidate_learned', candidate_directory, candidate_metadata, 'learned_20_512', candidate_exports, 512, 20, True),
            ('initial_learned', branch_directory, branch_metadata, 'learned_20_0', reference_exports, 0, 20, False),
            ('reference_full', reference_directory, reference_metadata, 'native_A_512', reference_exports, 512, 32, False),
            ('reference_learned', branch_directory, branch_metadata, 'learned_20_512', reference_exports, 512, 20, False),
        ]
        for name, directory, metadata, method, exports, point, width, new in specifications:
            panel = inspect_method(directory, metadata, method, master, originals, exports, point, width, 0, new)
            scores[name][master] = {view: {key: panel[view][key] for key in (*gate.HEADS, 'coverage')}
                                    for view in gate.VIEWS}
        # Initial complete32 is another reused method, retained for initial/full
        # training comparisons even though its scores do not enter the gate.
        inspect_method(reference_directory, reference_metadata, 'native_A_0', master,
                       originals, reference_exports, 0, 32, 0, False)
        inspect_method(branch_directory, branch_metadata, 'fixed_12', master,
                       originals, reference_exports, 0, 12, 20, False)

        queries[master] = {}
        query_reports = {}
        for point in (0, 512):
            point_root = candidate_root / f'point-{point}'
            point_report = document(point_root / 'query-report.json')
            checkpoint_path = point_root / 'checkpoint.pt'
            remember(checkpoint_path)
            checkpoint, scaler = scaler_math.checkpoint_scaler(codec, checkpoint_path)
            validate_checkpoint_metadata(checkpoint, codec, arithmetic, point)
            projection = arithmetic.tensor(vars(vars(checkpoint['model'])['relation_projection']),
                                           'weight', 'FloatStorage', [20, 1944])
            check(all(v == 0 for v in projection) if point == 0 else any(v != 0 for v in projection),
                  'checkpoint new projection zero-start versus learned movement')
            if point == 512:
                arithmetic.close(max(abs(v) for v in projection), progress['relation_weight_max_absolute_change'],
                                 'saved relation weights reconcile reported actual movement')
            fitted_scaler = scaler_math.audit_train_scaler(codec, scaler, originals['training'], check,
                                                          arithmetic.close, arithmetic.tensor)
            identity = scaler_math.scaler_identity(scaler)
            check(identity == codec.text(checkpoint['scaler_identity']) == progress['scaler_identity'],
                  'candidate scaler content identity')
            errors = {}
            for role, rows in (('training', 512), ('validation', 256)):
                binding = dict(protocol=PROTOCOL, dataset_id='TEMPO-5', model_tag='RPB-v22.alt-01', master=master,
                               source_fingerprint=source, card_sha=card, input_manifest_sha=ORIGINAL_INPUTS,
                               config_sha=producer['config_sha'])
                metadata = point_report[role + '_reconstruction']
                value = query_math.query_error(codec, point_root / ('reconstruction-' + role + '.pt'),
                    ORIGINAL / 'quality-data' / f'master-{master}' / (role + '.pt'),
                    metadata, point, rows, fitted_scaler, binding)
                errors[role] = value
                if point == 512:
                    reference_query = old_document(ORIGINAL / 'profiles/A' / f'master-{master}' / 'point-512/query-report.json')
                    queries[master][role] = dict(candidate=value,
                        reference=reference_query[role + '_reconstruction']['standardized_mae'],
                        coverage=metadata['coverage'])
            query_reports[point] = errors
        training.append(dict(master=master, updates=512, train_error=query_reports[512]['training'],
                             validation_error=query_reports[512]['validation'], initial_errors=query_reports[0],
                             GPU_training_seconds=progress['training_seconds'],
                             parameter_count=progress['parameter_count'], costs=producer['costs']))

    check(checked_replay == set(replay_predictions) and new_pairs == 6 * count and reused_pairs == 15 * count,
          'complete new and reused saved fit arithmetic inventory')
    # Paired intervals are independently replayed by the C++ saved reader;
    # this publisher verifies the paired population and estimate from original
    # decisions, without fitting heads or invoking encoders.
    check(len(replay['pairs']) == replay['paired_comparisons'] == 24 * count, 'complete paired comparison matrix')
    validate_pairs(replay['pairs'], masters)
    pair_keys = set()
    for pair in replay['pairs']:
        key = (pair['master'], pair['id'], pair['repetition'], pair['view'])
        check(key not in pair_keys and pair['master'] in masters and pair['repetition'] in REPETITIONS
              and pair['view'] in gate.VIEWS, 'unique declared paired comparison')
        pair_keys.add(key)
        candidate_method, reference_method = pair['id'].split('_minus_', 1)
        left = predictions[pair['master'], candidate_method, pair['repetition'], pair['view']]
        right = predictions[pair['master'], reference_method, pair['repetition'], pair['view']]
        check(pair['candidate_prediction'] == str(left[2]) and pair['reference_prediction'] == str(right[2]),
              'paired candidate/reference exact saved artifact paths')
        check(left[1]['labels'] == right[1]['labels'] and left[1]['ids'] == right[1]['ids']
              and left[1]['valid'] == right[1]['valid'] and pair['valid_rows'] == 256,
              'paired original population/source order')
        for head in gate.HEADS:
            actual = arithmetic.score(arithmetic.tensor(left[0], head, 'LongStorage', [256]),
                                     left[1]['labels'], left[1]['valid'])['accuracy']
            control = arithmetic.score(arithmetic.tensor(right[0], head, 'LongStorage', [256]),
                                      right[1]['labels'], right[1]['valid'])['accuracy']
            arithmetic.close(pair[head]['estimate'], actual - control, 'paired training/control effect')
            check(pair[head]['source_groups'] == 128
                  and -1 <= pair[head]['lower'] <= pair[head]['upper'] <= 1, 'paired conditional interval range/population')
    for item in report['saved_input_files']:
        check(remember(Path(item['path'])) == item['sha256'], 'bound readout input unchanged')
    merge_query_reads(query_math.READS)
    for path, digest in list(READS.items()):
        check(sha(path) == digest, 'all consumed evidence unchanged after arithmetic')

    result = dict(protocol=PROTOCOL, dataset_id='TEMPO-5', task='slow timing',
                  recipe_id='tempo-5-slow-fast-nine-feature-v1', designed_complexity_level=5,
                  complexity_scale_max=5, evidence_stage='known-validation development', model_tag='RPB-v22.alt-01',
                  masters=masters, source_fingerprint=source, card_sha256=card,
                  input_manifest_sha256=ORIGINAL_INPUTS, scores=scores, training=training,
                  queries=queries, gate=gate.assess(scores, queries, masters),
                  new_head_fit_pairs=new_pairs, new_classifiers=2 * new_pairs,
                  reused_fit_pairs_arithmetic_checked=reused_pairs,
                  candidate_prediction_archives=18 * count, encoder_updates=512 * count,
                  encoder_trajectories=count, CPU_encoder_calls=0, publisher_encoder_calls=0,
                  publisher_head_fits=0, initial_native_values_exact=initial_cells,
                  fixed_values_exact=fixed_cells, paired_comparisons=24 * count,
                  checks=CHECKS + arithmetic.CHECKS + query_math.CHECKS if hasattr(query_math, 'CHECKS') else CHECKS + arithmetic.CHECKS,
                  consumed_files=len(READS), testing_accessed=False, stress_accessed=False, promotion=False)
    destination.mkdir()
    write_new(destination / 'summary.json', result)
    write_new(destination / 'consumed-files.sha256', ''.join(f'{digest}  {path}\n' for path, digest in sorted(READS.items())))
    write_new(destination / 'gate.json', result['gate'])
    lines = ['# RPB-v22 content-only relation development evidence', '',
             'New candidate training; saved controls reused without refitting. RPB-v20 remains preserved.', '',
             'Dataset: **TEMPO-5** · complexity **5/5** · slow timing beneath fast rhythms and nine grouped features.', '',
             'RPB-v22.alt-01 learns ordered content relations without additive mask-density inputs. '
             'RPB-v20.alt-01 is the saved complete32 control.', '']
    for view, label in (('validation_intact', 'Intact VALIDATION'), ('validation_deleted', 'Extra30% deletion VALIDATION')):
        lines += [label + '. Dataset: **TEMPO-5**, complexity **5/5**, slow timing with structured distractors.', '',
                  '| Method | Size | Linear head % | Neural head % | Coverage % |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for name, tag, width in [('reference_full', 'RPB-v20.alt-01', 32), ('candidate_full', 'RPB-v22.alt-01', 32),
                                 ('reference_learned', 'RPB-v20.alt-01', 20), ('candidate_learned', 'RPB-v22.alt-01', 20)]:
            panel = {k: 100 * average([scores[name][m][view][k] for m in masters]) for k in (*gate.HEADS, 'coverage')}
            lines.append(f'| {tag} | {width} | {panel["ridge"]:.2f} | {panel["tiny_secondary"]:.2f} | {panel["coverage"]:.2f} |')
        lines += ['', 'Size32 is the complete native export; size20 is the first20 learned coordinates, a diagnostic subset.', '']
    lines += ['Dataset: **TEMPO-5**, complexity **5/5**. Fixed original hidden-query standardized MAE; lower is better.', '',
              '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |',
              '| --- | ---: | ---: | ---: | ---: |',
              f'| RPB-v22.alt-01 | 512 | {average([p["train_error"] for p in training]):.5f} | {average([p["validation_error"] for p in training]):.5f} | {average([p["GPU_training_seconds"] for p in training]):.1f} |', '',
              'Prospective joint gate: **' + ('PASS' if result['gate']['passed'] else 'FAIL') + '**.', '',
              'Every allocated cohort and fixed head repetition is retained. No TEST, market measurement or default replacement.', '']
    write_new(destination / 'report.md', '\n'.join(lines))
    print(json.dumps(dict(status='published', masters=masters, gate_passed=result['gate']['passed'],
                          checks=result['checks'], consumed_files=len(READS))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', required=True, type=Path)
    parser.add_argument('--readouts', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    args = parser.parse_args()
    publish(args.capsule, args.readouts, args.output_directory)


if __name__ == '__main__':
    main()

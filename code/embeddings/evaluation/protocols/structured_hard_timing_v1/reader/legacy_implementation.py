"""Unchanged saved-contract validators from reviewed B114 SOURCE.
The new entry supplies admitted archive/metadata functions and codec; no old run or inputs are imported.
"""
from pathlib import Path
import sys
from saved_cpu_math import *
IMPLEMENTATION_PROTOCOL = 'early-mixer-reliability-v1'
FIT_PROTOCOL = IMPLEMENTATION_PROTOCOL + '/lag_sign'
COUNTER_POLICY = 'splitmix64-counter-rows-masks-torch-attempt-v1'
POLICY = 'rpb-training-context-deletion-015-v1'
F32_ATOL, F32_RTOL = 2e-6, 2e-5
HEAD_REPETITIONS = (2701,2802,2903)
VIEWS = ('training','validation-intact','validation-deleted')
METHODS = {'raw':576,'mask_metadata':288,'pca_only':32,'untrained_late':32,'untrained_early':32,'native_late':32,'native_early':32}
ARCHITECTURES = ('aligned-mixer-after-temporal-v1', 'aligned-mixer-before-temporal-v1')


OUTPUT_SEMANTICS = ('local_observed_contextual_aligned_global_semantic_mlp_bottleneck_v1',
                    'local_observed_contextual_pretemporal_aligned_global_semantic_mlp_bottleneck_v1')


RECONSTRUCTION_SEMANTICS = ('exact_contextual_observed_global_semantic_mlp_export_v1',
                          'exact_pretemporal_contextual_observed_global_semantic_mlp_export_v1')


CONTEXT_LITERALS = {
    'training_policy_id': POLICY,
    'context_deletion_ratio': '0.15',
    'context_deletion_stream': '0x6374782d64726f70',
    'context_deletion_rng_policy': 'splitmix64-counter-base-plus-semantic-canonical-BCHF-ordinal;top53-u01;independent-ctx-drop-stream-v1',
    'context_deletion_repair_policy': 'eligible-channels-only;restore-earliest-erased-original-patch-first-original-visible-coordinate-until-two-groups-v1',
    'context_deletion_visibility_policy': 'V=(O&~A)&~E;E-subset-original-visible;repair-only-E;original-O-A-Q-eligibility-Huber-unchanged-v1',
    'context_deletion_count_policy': 'cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only',
    'context_deletion_resume_policy': 'fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API'
}


def control_rows(root, training, validation, deleted):
    values, mask, ids, labels = training
    asset = load(root / 'raw-scaler.pt')
    check(set(asset) == {'mean', 'scale', 'counts'}, 'one task TRAIN raw ObservationScaler')
    mean = tensor(asset['mean'], 'DoubleStorage', [3, 3])
    scale = tensor(asset['scale'], 'DoubleStorage', [3, 3])
    counts = tensor(asset['counts'], 'DoubleStorage', [3, 3])
    for c in range(3):
        for f in range(3):
            selected = [values[(b * 3 + c) * 96 + h * 3 + f] for b in range(256) for h in range(32)
                        if mask[(b * 3 + c) * 96 + h * 3 + f]]
            d = c * 3 + f
            check(counts[d] == len(selected), 'ordinary raw observed TRAIN counts')
            avg = math.fsum(selected) / len(selected) if selected else 0.
            std = math.sqrt(math.fsum((x - avg) ** 2 for x in selected) / len(selected)) if selected else 0.
            close(mean[d], avg, 'ordinary raw observed TRAIN mean')
            close(scale[d], max(1e-8, std), 'ordinary raw population scale/floor')
    unprepared, masks, valids = [], [], []
    for split in (training, validation, deleted):
        data, support, _, _ = split
        rows, flags, valid = [], [], []
        for b in range(len(split[2])):
            observed = list(support[b * 288:(b + 1) * 288])
            rows.append([(data[b * 288 + j] - mean[(j // 96) * 3 + j % 3]) / scale[(j // 96) * 3 + j % 3]
                         if observed[j] else 0. for j in range(288)] + [float(x) for x in observed])
            flags.append([float(x) for x in observed]); valid.append(any(observed))
        unprepared.append(rows); masks.append(flags); valids.append(valid)
    outer = load(root / 'raw-outer-normalizer.pt')
    check(set(outer) == {'feature_mean', 'feature_scale', 'fitted_rows', 'fit_count', 'training_source_ids_json'},
          'one shared driver raw TRAIN outer map')
    fm = tensor(outer['feature_mean'], 'DoubleStorage', [576])
    fs = tensor(outer['feature_scale'], 'DoubleStorage', [576])
    check(scalar(outer, 'fitted_rows') == sum(valids[0]), 'one shared raw TRAIN fit population')
    check(scalar(outer, 'fit_count') == 1 and json.loads(text(outer, 'training_source_ids_json')) == ids, 'one exact TRAIN raw outer fit and source order')
    check(len(ids) == 256, 'one original TRAIN map from the bound 256-row source order; constructor/count is source-bound')
    map_statistics(unprepared[0], valids[0], fm, fs)
    prepared = [normalize(rows, keep, fm, fs) for rows, keep in zip(unprepared, valids)]
    result = {'raw': prepared, 'mask_metadata': masks, 'pca_only': [], '_raw_unprepared': unprepared}
    status = read_json(root / 'pca-status.json')
    check(status['width'] == 32 and not status['native_PCA'], 'raw standalone PCA only, no native PCA')
    if status['status'] == 'unsupported_fit':
        check(not (root / 'pca-preprocessing.pt').exists() and isinstance(status['reason'], str) and
              any(x in status['reason'] for x in ('PCA dimensions exceed numerical training rank',
                   'PCA dimensions exceed valid centered training-row bound', 'feature fit requires two valid training rows')),
              'explicit rank/support failure; no invented PCA map')
        result['pca_only'] = [[[0.] * 32 for _ in rows] for rows in prepared]
        return result, status
    check(status == {'status': 'measured', 'width': 32, 'native_PCA': False}, 'measured standalone PCA status')
    pc = load(root / 'pca-preprocessing.pt')
    check(set(pc) == {'feature_mean', 'feature_scale', 'fitted_rows', 'pca_mean', 'pca_components',
                     'pca_singular_values', 'pca_numerical_rank'}, 'PCA writer schema, no second raw normalizer')
    for key in ('feature_mean', 'feature_scale', 'fitted_rows'):
        exact(pc[key], outer[key], 'PCA shares byte-exact one driver raw map ' + key)
    n = sum(valids[0]); pm = tensor(pc['pca_mean'], 'DoubleStorage', [576])
    components = tensor(pc['pca_components'], 'DoubleStorage', [576, 32])
    singular = tensor(pc['pca_singular_values'], 'DoubleStorage', [min(n, 576)])
    finite(components, 'finite PCA columns'); finite(singular, 'finite PCA spectrum')
    check(all(singular[i] >= singular[i + 1] >= 0 for i in range(len(singular) - 1)) and singular[0] > 0,
          'ordered PCA spectrum')
    rank = sum(x > 576 * sys.float_info.epsilon * singular[0] for x in singular)
    check(scalar(pc, 'pca_numerical_rank') == rank and rank >= 32, 'supported explicit TRAIN numerical rank')
    for d in range(576):
        close(pm[d], math.fsum(row[d] for row, keep in zip(prepared[0], valids[0]) if keep) / n,
              'PCA mean uses shared prepared TRAIN only')
    for i in range(32):
        for j in range(i + 1):
            close(math.fsum(components[d * 32 + i] * components[d * 32 + j] for d in range(576)), float(i == j),
                  'PCA orthonormal columns')
    for rows, keep in zip(prepared, valids):
        result['pca_only'].append([[math.fsum((row[d] - pm[d]) * components[d * 32 + j] for d in range(576))
                                    for j in range(32)] if ok else [0.] * 32 for row, ok in zip(rows, keep)])
    pca_covariance_replay([row for row, ok in zip(unprepared[0], valids[0]) if ok], pc)
    return result, status


def named_group(value, expected_total=None):
    value = group(value)
    count = scalar(value, 'count')
    check(count >= 0 and set(value) == {'count'} | {f'tensor_{i}' for i in range(count)}, 'complete numbered named tensor group')
    result = {}
    for i in range(count):
        item = group(value[f'tensor_{i}'])
        check(set(item) == {'parameter_name', 'value'}, 'named tensor item schema')
        name = text(item, 'parameter_name')
        tensor_value = item['value']
        check(name and name not in result and tensor_value['dtype'] in ('FloatStorage', 'DoubleStorage', 'LongStorage', 'BoolStorage'),
              'unique CPU named tensor type')
        check(len(tensor_value['values']) == math.prod(tensor_value['shape']), 'named tensor geometry')
        finite(tensor_value['values'], 'finite captured named tensor values')
        result[name] = tensor_value
    if expected_total is not None:
        check(sum(len(x['values']) for x in result.values()) == expected_total, 'fixed parameter total')
    return result


def same_named(left, right, message):
    check(set(left) == set(right), message + ': names')
    for name in left:
        exact(left[name], right[name], message + ': ' + name)


def initial_witness(asset):
    check(set(asset) == {'late_parameters', 'early_parameters', 'late_buffers', 'early_buffers', 'late_scaler', 'early_scaler'},
          'complete paired initialization writer schema')
    late = named_group(asset['late_parameters'], 225805)
    early = named_group(asset['early_parameters'], 225805)
    lb, eb = named_group(asset['late_buffers']), named_group(asset['early_buffers'])
    check(list(late)==list(early) and list(lb)==list(eb),'same paired parameter/buffer registration order')
    same_named(late, early, 'same registered initial parameters')
    same_named(lb, eb, 'same initial buffers')
    ls, es = group(asset['late_scaler']), group(asset['early_scaler'])
    check(set(ls) == set(es), 'paired initialization scaler schema')
    for key in ls:
        exact(ls[key], es[key], 'paired initialization scaler ' + key)
    _, _, identity = scaler(ls)
    return late, lb, ls, identity


def parse_settings(value, placement, master):
    result = {}
    for line in value.splitlines():
        check('=' in line, 'canonical settings line')
        key, val = line.split('=', 1)
        check(key not in result, 'unique resolved settings')
        result[key] = val
    expected = {'channel_count': 3, 'history_length': 32, 'input_width': 3, 'patch_length': 8,
                'encoder_width': 64, 'export_width': 32, 'num_layers': 3, 'num_heads': 4,
                'feedforward_width': 256, 'decoder_hidden_width': 128, 'channel_mixer_layers': 1,
                'global_bottleneck_mode': 2, 'dropout': 0, 'huber_delta': 1, 'sampling_interval': 1,
                'batch_size': 8, 'threads': 1, 'learning_rate': .001, 'weight_decay': .0001,
                'gradient_clip_norm': 1, 'steps': 512, 'attempt_limit': 1024, 'log_every': 1, 'seed': master,
                'layer_norm_epsilon': 1e-5, 'mask_ratio': .25, 'scale_floor': 1e-6}
    for key, val in expected.items():
        check(key in result and float(result[key]) == val, 'frozen resolved setting ' + key)
    check(result['device'] in ('cuda', 'cuda:0'), 'actual CUDA setting, no CPU model')
    check(result['channel_ids'] in ('', '0,1,2'), 'configured canonical channel IDs; empty resolves to range(C)')
    check(('channel_mixer_placement' not in result) if placement == 0 else result.get('channel_mixer_placement') == '1',
          'default placement omitted, explicit nondefault placement')
    check('global_pool_input_source' not in result,'unchanged projected-D global route; default field omitted')
    common = dict(result)
    common.pop('channel_mixer_placement', None)
    common['device'] = 'cuda'
    return common


def parent_audit(audit, master, placement, point, ids):
    check(text(audit, 'artifact_kind') == 'rpb_learning_curve_training_audit_v1' and
          text(audit, 'protocol_id') == FIT_PROTOCOL and text(audit, 'actual_training_seed') == str(master) and
          text(audit, 'initialization_seed') == str(mixed(master ^ 0x7270622d696e6974)) and
          text(audit, 'fit_source_manifest') == source_manifest(ids), 'exact label-free fit namespace/data/init seed')
    check(text(audit, 'model_tag') == ('RPB-v10' if placement else 'RPB-v7') and
          text(audit, 'architecture_id') == ARCHITECTURES[placement] and
          text(audit, 'channel_mixer_placement') == str(placement) and
          scalar(audit, 'channel_mixer_placement_value') == placement and
          text(audit, 'output_semantics') == OUTPUT_SEMANTICS[placement] and
          text(audit, 'reconstruction_export_semantics') == RECONSTRUCTION_SEMANTICS[placement], 'typed placement and architecture semantics')
    check(scalar(audit, 'attempted_steps') == scalar(audit, 'completed_steps') == point and
          scalar(audit, 'sampled_rows') == point * 8 and text(audit, 'model_weight_update_budget') == str(point),
          'absolute unskipped point counters')
    check(list(tensor(audit['channel_order'], 'LongStorage', [3])) == [0, 1, 2] and
          scalar(audit, 'sampling_interval', 'DoubleStorage') == 1 and scalar(audit, 'endpoint', 'DoubleStorage') == 31,
          'typed canonical semantic/time metadata')
    for key, expected in CONTEXT_LITERALS.items():
        check(text(audit, key) == expected, 'unchanged coordinate15 policy ' + key)
    check(scalar(audit, 'context_deletion_ratio_value', 'DoubleStorage') == .15 and
          scalar(audit, 'context_deletion_stream_value') == 0x6374782d64726f70, 'typed context recipe')
    check(not any(key.startswith('context_deletion_schedule') or key in ('context_ordinary_attempts', 'context_deletion_attempts') for key in audit),
          'no balanced or alternative recipe')
    check(not any(key.startswith('source_gain_') or key.startswith('pooled_') or key=='global_pool_input_source_value' for key in audit),
          'delegated ordinary coordinate15 audit has no gain or pooled-route additions')
    counts = [scalar(audit, key) for key in ('context_requested_deleted_coordinates', 'context_actual_deleted_coordinates', 'context_restored_coordinates')]
    check(0 <= counts[1] <= counts[0] <= point * 8 * 288 and counts[2] == counts[0] - counts[1], 'context counts requested/actual/restored')
    seconds = scalar(audit, 'training_seconds', 'DoubleStorage')
    finite_timer(seconds, 'CUDA update loop')
    check(scalar(audit, 'weights_changed', 'BoolStorage') == bool(point) and
          scalar(audit, 'finite_gradients', 'BoolStorage') == bool(point) and
          ((seconds == 0) if point == 0 else (seconds > 0)), 'actual training/point0 scalar witnesses')
    check(text(audit, 'rng_policy') == COUNTER_POLICY and
          text(audit, 'sampling_policy') == 'with_replacement_counter_rows;sampled_rows_includes_no_update_attempts' and
          text(audit, 'optimizer_policy') == 'one_continuous_AdamW_state;absolute_completed_update_budgets', 'unchanged absolute sampling/Torch/optimizer streams')
    common = parse_settings(text(audit, 'resolved_settings'), placement, master)
    return {'counts': counts, 'training_seconds': seconds, 'common_settings': common,
            'dataset_id': text(audit, 'training_dataset_id'), 'scaler_id': text(audit, 'preprocessing_id'),
            'core_source': text(audit, 'core_writer_source_fingerprint'),
            'training_source': text(audit, 'training_producer_source_fingerprint'),'parameter_count':225805}


def progress(value, audit_info):
    check(value['attempted'] == value['completed'] == 512 and value['sampled_rows'] == 4096 and
          value['parameter_count'] == value['cuda_parameter_count'] == 225805 and
          value['training_device'] in ('cuda', 'cuda:0') and
          all(value[k] is True for k in ('last_input_cuda', 'last_loss_cuda', 'finite_gradients', 'weights_changed')),
          'complete actual CUDA encoder trajectory')
    close(value['training_seconds'], audit_info['training_seconds'], 'same saved update-loop timer')
    check(value['training_dataset_id'] == audit_info['dataset_id'] and value['preprocessing_id'] == audit_info['scaler_id'], 'progress fit/scaler association')
    trace = value['losses']
    check(len(trace) == 512, 'all 512 trace entries, not selected rows')
    for i, row in enumerate(trace):
        check(len(row) == 5 and row[0] == row[1] == i + 1 and type(row[2]) is int and row[2] > 0 and
              math.isfinite(row[3]) and row[3] >= 0 and math.isfinite(row[4]) and row[4] >= 0, 'ordered exact trace prefix/targets/finite loss and gradient')
    return trace


def query_archive(asset, split, frozen_scaler):
    data, mask, ids, labels = split
    rows = len(ids)
    expected = {'standardized_prediction', 'standardized_target', 'target_mask', 'requested_observed_target_mask',
                'visible_mask', 'trial_channel_eligible', 'channel_target_counts', 'channel_valid',
                'channel_standardized_mae', 'channel_standardized_huber', 'example_valid', 'example_standardized_mae',
                'example_standardized_huber', 'source_ids_json'}
    check(set(asset) == expected, 'unchanged fixed query archive schema')
    shape = [4, rows, 3, 32, 3]
    p = tensor(asset['standardized_prediction'], 'DoubleStorage', shape)
    t = tensor(asset['standardized_target'], 'DoubleStorage', shape)
    q = list(tensor(asset['target_mask'], 'BoolStorage', shape))
    rq = list(tensor(asset['requested_observed_target_mask'], 'BoolStorage', shape))
    v = list(tensor(asset['visible_mask'], 'BoolStorage', shape))
    e = list(tensor(asset['trial_channel_eligible'], 'BoolStorage', [4, rows, 3]))
    wanted_rq, wanted_v, wanted_q, wanted_e = query_masks(list(mask), rows)
    check(rq == wanted_rq and v == wanted_v and q == wanted_q and e == wanted_e, 'ordinary original-Q visibility/eligibility/support, no E augmentation')
    check(json.loads(text(asset, 'source_ids_json')) == ids, 'fixed query exact source order')
    mean, scale, _ = scaler(frozen_scaler)
    for i, keep in enumerate(q):
        if keep:
            original = i % (rows * 288)
            channel = (original // 96) % 3
            feature = original % 3
            want = f32((data[original] - mean[channel * 3 + feature]) / scale[channel * 3 + feature])
            check(abs(t[i] - want) <= F32_ATOL + F32_RTOL * abs(want), 'frozen TRAIN scaler/F32 target arithmetic')
    result = reductions(p, t, q, rows)
    shapes = {'channel_target_counts': ('LongStorage', [rows, 3], result['counts']),
              'channel_valid': ('BoolStorage', [rows, 3], [bool(x) for x in result['counts']]),
              'channel_standardized_mae': ('DoubleStorage', [rows, 3], result['channel_mae']),
              'channel_standardized_huber': ('DoubleStorage', [rows, 3], result['channel_huber']),
              'example_valid': ('BoolStorage', [rows], result['example_valid']),
              'example_standardized_mae': ('DoubleStorage', [rows], result['example_mae']),
              'example_standardized_huber': ('DoubleStorage', [rows], result['example_huber'])}
    for key, (dtype, dims, values) in shapes.items():
        saved = tensor(asset[key], dtype, dims)
        if dtype in ('BoolStorage', 'LongStorage'):
            check(list(saved) == values, 'exact fixed query counts/support ' + key)
        else:
            for actual, wanted in zip(saved, values):
                close(actual, wanted, 'hierarchical query reduction ' + key)
    result['requested_observed_target_cells'] = sum(rq)
    result['valid_target_cells'] = sum(result['counts'])
    result['valid_examples'] = sum(result['example_valid'])
    result['total_examples'] = rows
    return result


def audit_readouts(directory, data, controls, pca_status, provider_records, master, task):
    train, val, deleted = data; root = directory / 'readouts'; report = read_json(root / 'report.json')
    methods = methods_for_task(task)
    check(report['protocol'] == 'fixed-feature-readouts-v1' and report['master_seed'] == str(master), 'actual fixed readout writer')
    check(report['recipe'] == {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100, 'tiny_learning_rate': .01,
          'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods', 'validation_fits': 0,
          'encoder_calls': 0, 'pca_fits': 0}, 'unchanged fixed heads, no view or encoder fitting')
    check([x['method'] for x in report['methods']] == list(methods), 'all unique task methods, controls fit once')
    stored = {}; summaries = []; fits = outer_fits = 0
    for item in report['methods']:
        name = item['method']; width = methods[name]; prepared = name in ('raw', 'pca_only'); path = root / name
        check(item['size'] == width and item['inputs_train_prepared'] is prepared,
              'prepared raw/PCA bypass only; native outer stage unchanged')
        surfaces = []
        for view, split, popkey in zip(VIEWS, data, ('training_population', 'validation_intact_population', 'validation_deleted_population')):
            feature = load(path / (view + '-features.pt'))
            rows, valid = feature_archive(feature, len(split[2]), width, split[2], split[3],
                                         'DoubleStorage' if name in controls else 'FloatStorage')
            support = [any(split[1][b * 288:(b + 1) * 288]) for b in range(len(split[2]))]
            if name in controls:
                check(valid == support, 'raw/PCA/mask observation-any support independent of native')
                for a, b in zip(rows, controls[name][len(surfaces)]):
                    for x, y in zip(a, b): close(x, y, 'saved prepared legal control ' + name)
            else:
                check(all(not ok or support[b] for b, ok in enumerate(valid)), 'native support subset of original observations')
                check(text(feature, 'provenance') == provider_records[name]['extractor_provenance'], 'same retained immutable CUDA surface')
            check(item[popkey] == population(valid, split[3], split[2]), 'explicit per-method view support')
            surfaces.append((rows, valid))
        rows, valid = surfaces[0]
        supported = sum(valid) >= 2 and {train[3][i] for i, ok in enumerate(valid) if ok} == {0, 1}
        if name == 'pca_only' and pca_status['status'] == 'unsupported_fit': supported = False
        check(item['status'] == ('measured' if supported else 'unsupported_fit'), 'independent ordinary TRAIN fit support')
        count = {'outer_train_normalizer_fits': int(supported and not prepared), 'ridge_fits': 3 if supported else 0,
                 'tiny_fits': 3 if supported else 0, 'validation_fits': 0}
        check(read_json(path / 'fit-counts.json') == count and item['outer_train_normalizer_fits'] == count['outer_train_normalizer_fits'],
              'one native/mask outer map; driver prepared raw/PCA not refitted')
        outer_fits += count['outer_train_normalizer_fits']
        method_summary = {'method': name, 'size': width, 'status': item['status'], 'repetitions': []}
        check(len(item['repetitions']) == 3, 'three fixed head repetitions including unsupported')
        for rep, item_rep in zip(HEAD_REPETITIONS, item['repetitions']):
            repid = f'rep-{rep}'; repdir = path / repid
            check(item_rep['id'] == repid and item_rep['status'] == item['status'] and
                  item_rep['actual_probe_seed_decimal'] == str(stream_seed(rep, width)) and
                  item_rep['ridge_parameters'] == 2 * width + 2 and item_rep['neural_parameters'] == 16 * (width + 3) + 2,
                  'actual width-paired seed and fixed head parameter counts')
            if not supported:
                check(not repdir.exists() and not any(k in item_rep for k in ('fit_artifact', 'training', 'validation_intact', 'validation_deleted')),
                      'unsupported fit invents no asset/predictions')
                continue
            fits += 1; fit = load(repdir / 'fit.pt'); fitted = fit_schema(fit, width, train[2], rep, prepared, rows, valid)
            check(item_rep['fit_artifact'] == f'{name}/{repid}/fit.pt', 'one ordinary TRAIN asset per unique pipeline')
            predictions = []; summary = {'repetition': repid, 'actual_probe_seed_decimal': str(stream_seed(rep, width))}
            for view, split, surface, key in zip(VIEWS, data, surfaces, ('training', 'validation_intact', 'validation_deleted')):
                feature_rows, keep = surface; prediction = load(repdir / (view + '-predictions.pt'))
                ridge, tiny, _ = infer_saved_fit(fit, feature_rows, keep, split[3], split[2], prediction, prepared, width, fitted)
                declared = item_rep[key]
                check(declared['population'] == population(keep, split[3], split[2]), 'saved prediction support/source denominator')
                scores = {'ridge': score(ridge, split[3], keep), 'tiny_secondary': score(tiny, split[3], keep)}
                for head in scores: check_saved_score(scores[head], declared[head])
                if view != 'training':
                    seed = stream_seed(master, fnv64(name + '/' + repid + '/' + view))
                    check(declared['bootstrap_seed_decimal'] == str(seed), 'unchanged per-method interval seed')
                    for head, preds, field in [('ridge', ridge, 'ridge_grouped_interval'), ('tiny_secondary', tiny, 'tiny_grouped_interval')]:
                        ci = bootstrap_groups(accuracy_groups(preds, split[3], keep, split[2]), seed, 1000)
                        check_interval(ci, declared[field]); scores[head]['grouped_interval'] = ci
                summary[key] = scores; predictions.append((ridge, tiny, keep))
            stored[name, rep] = predictions; method_summary['repetitions'].append(summary)
        summaries.append(method_summary)
    check(report['fit_counts'] == {'outer_train_normalizer_fits': outer_fits, 'ridge_fits': fits, 'tiny_fits': fits,
                                  'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'actual supported unique fit counts')
    budgets = (512,)
    expected_ids = {'native_early_minus_native_late'}
    check(len(report['pairs']) == len(budgets) * 6, 'one declared comparison per budget/rep/view')
    pairs = []
    for record in report['pairs']:
        check(record['id'] in expected_ids and record['repetition'] in [f'rep-{r}' for r in HEAD_REPETITIONS] and
              record['view'] in ('validation_intact', 'validation_deleted'), 'explicit budget-specific vector comparison')
        candidate = 'native_early'; reference = 'native_late'
        rep = int(record['repetition'][4:]); view = 1 if record['view'] == 'validation_intact' else 2
        if (candidate, rep) not in stored or (reference, rep) not in stored:
            check(record['status'] == 'unsupported_fit', 'unsupported pair remains explicit'); pairs.append(record); continue
        a, b = stored[candidate, rep][view], stored[reference, rep][view]
        common = [bool(x and y) for x, y in zip(a[2], b[2])]
        check(record['status'] == ('measured' if any(common) else 'unsupported_zero_common') and
              record['common_population'] == population(common, val[3], val[2]), 'per-budget common supported population')
        seed = stream_seed(master, fnv64(record['id'] + '/' + record['repetition'] + '/' + record['view']))
        check(record['bootstrap_seed_decimal'] == str(seed), 'generic unchanged pair bootstrap seed')
        for head, index in [('ridge', 0), ('tiny_secondary', 1)]:
            effect = paired_group_effect(a[index], b[index], val[3], a[2], b[2], val[2])
            check_interval(bootstrap_groups(effect['groups'], seed, 1000), record[head])
        pairs.append(record)
    check(len({(p['id'], p['repetition'], p['view']) for p in pairs}) == len(budgets) * 6,
          'all vector comparison records unique, no selected points/heads')
    return {'data_master': master, 'methods': summaries, 'pairs': pairs, 'fit_counts': report['fit_counts'],
            'driver_raw_outer_fits': 1}, report


def replay_parent(audit, raw, scaler_asset, controlled, master, placement, point, scopes):
    data, mask, ids, labels = controlled
    info = parent_audit(audit, master, placement, point, ids)
    check(info['core_source'] == scopes['core_writer'] and info['training_source'] == scopes['curve_training'],
          'actual captured core/continuous-trainer compile scopes')
    check(text(audit, 'source_fingerprint_scope') ==
          'ordinary_checkpoint_field=core_writer;companion_audit_training_producer=evaluation_source', 'parent source scope literal')
    check(text(raw, 'artifact_kind') == 'rpb_raw_uniform_history_v1' and scalar(raw, 'format_version') == 1,
          'typed legal timing raw companion')
    exact(raw['data'], fake('DoubleStorage', [256,3,32,3], data), 'raw fitting data exactly controlled timing TRAIN')
    exact(raw['observed'], fake('BoolStorage', [256,3,32,3], mask), 'raw fitting mask exactly controlled timing TRAIN')
    check(list(tensor(raw['channel_ids'], 'LongStorage', [3])) == [0,1,2] and
          list(tensor(raw['endpoints'], 'DoubleStorage', [256])) == [31.] * 256 and
          scalar(raw, 'sampling_interval', 'DoubleStorage') == 1., 'raw semantic/time metadata')
    check(text(raw,'feature_units') == text(audit,'feature_units') == 'unitless,unitless,unitless', 'same ordinary units')
    check(text(scaler_asset,'artifact_kind') == 'rpb_frozen_training_scaler_v1' and
          scalar(scaler_asset,'format_version') == 1, 'typed scaler companion')
    frozen = group(scaler_asset['scaler'])
    mean, scale, identity = scaler(frozen)
    check(identity == info['scaler_id'] == text(scaler_asset,'preprocessing_id'), 'exact frozen scaler identity')
    check(text(raw,'schema_id') == text(scaler_asset,'schema_id') and
          text(raw,'dataset_id') == text(scaler_asset,'fit_dataset_id') == info['dataset_id'] == text(audit,'scaler_fit_dataset_id'),
          'timing-only fitting schema/data association')
    count = tensor(frozen['count'],'LongStorage',[3,3]); floors = tensor(frozen['floor_applied'],'BoolStorage',[3,3])
    for c in range(3):
        for f in range(3):
            values = [data[b*288+c*96+h*3+f] for b in range(256) for h in range(32) if mask[b*288+c*96+h*3+f]]
            i = c*3+f
            check(len(values) == count[i] and values, 'ordinary observed TRAIN scaler count')
            wanted_mean = math.fsum(values)/len(values)
            sd = math.sqrt(math.fsum((x-wanted_mean)**2 for x in values)/len(values))
            close(mean[i],wanted_mean,'ordinary TRAIN scaler mean')
            close(scale[i],max(1e-6,sd),'ordinary TRAIN population scale')
            check(bool(floors[i]) == (sd<1e-6), 'frozen scaler floor condition')
    info.update({'schema_id':text(raw,'schema_id'),'frozen_scaler':frozen})
    return info


def snapshot_binding(asset, audit, info, cp, placement, point, scopes, initial):
    check(text(asset,'artifact_kind') == 'rpb_early_mixer_cuda_snapshot_v1', 'strict new immutable CUDA snapshot kind')
    expected = {
        'protocol_id':IMPLEMENTATION_PROTOCOL, 'original_training_protocol_id':FIT_PROTOCOL,
        'parent_checkpoint_path':str(cp), 'parent_writer_source_fingerprint':scopes['core_writer'],
        'parent_training_producer_source_fingerprint':scopes['curve_training'],
        'snapshot_loader_source_fingerprint':scopes['early_adapter'],
        'source_fingerprint_scope':'parent=core_writer_and_training_producer;loader=new_early_mixer_adapter',
        'original_encoder_attempted':str(point), 'original_encoder_completed':str(point),
        'training_schema_id':info['schema_id'], 'training_dataset_id':info['dataset_id'],
        'preprocessing_id':info['scaler_id'], 'parameter_count':'225805',
        'encoder_updates':'0','decoder_updates':'0','head_refits':'0','no_optimizer_created':'true',
        'snapshot_policy':'independent_CUDA_model;eval_no_grad;frozen_timing_TRAIN_scaler;ordinary_query_no_context_erasure;no_refit'
    }
    for key, value in expected.items():
        check(text(asset,key) == value, 'snapshot lineage/immutability '+key)
    check(text(asset,'inference_device') in ('cuda','cuda:0'), 'snapshot actual CUDA inference')
    for key in ('channel_mixer_placement_value','original_encoder_attempted_value','original_encoder_completed_value'):
        check(scalar(asset,key) == (placement if key.startswith('channel') else point), 'typed original snapshot '+key)
    overrides = set(expected) | {'artifact_kind'}
    for key,value in audit.items():
        if value.get('dtype') == 'ByteStorage' and key not in overrides:
            check(text(asset,key) == text(audit,key), 'snapshot preserves original typed audit text '+key)
    for key, value in zip(('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'),info['counts']):
        check(text(asset,key) == str(value), 'snapshot saved point context count '+key)
    for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt'):
        # Whole CUDA checkpoint bytes are hashed, never CPU-decoded.
        check(text(asset,'parent_content_id'+suffix) == 'fnv1a64-runtime-content-v1-'+fnv(Path(str(cp)+suffix).read_bytes()), 'same original point byte/FNV binding '+suffix)
    saved_scaler = group(asset['scaler'])
    check(set(saved_scaler) == set(info['frozen_scaler']), 'snapshot scaler schema')
    for key in saved_scaler: exact(saved_scaler[key],info['frozen_scaler'][key],'snapshot frozen TRAIN scaler '+key)
    params = named_group(asset['model_parameters'],225805); buffers = named_group(asset['model_buffers'])
    check(list(params) == sorted(params) and list(buffers) == sorted(buffers), 'snapshot saved named tensors are lexicographic')
    base_params, base_buffers, base_scaler, base_id = initial
    check(set(params) == set(base_params) and set(buffers) == set(base_buffers), 'unchanged registered modules/dimensions')
    for name in params:
        check(params[name]['dtype'] == base_params[name]['dtype'] and params[name]['shape'] == base_params[name]['shape'], 'same parameter architecture '+name)
    if point == 0:
        same_named(params,base_params,'snapshot exact paired initial parameters')
        same_named(buffers,base_buffers,'snapshot exact initial buffers')
    else:
        check(any(bytes_of(params[name]) != bytes_of(base_params[name]) for name in params), 'trained snapshot actually changes weights')
    for key in base_scaler: exact(saved_scaler[key],base_scaler[key],'all retained points keep ordinary TRAIN scaler '+key)
    base = ('RPB-v10' if placement else 'RPB-v7') + '; immutable timing-protocol checkpoint; exact contextual-global32 CUDA serving; ordinary original-Q decoder'
    return {'extractor_provenance':base+'; '+base,'placement':placement,'point':point,
            'architecture_id':ARCHITECTURES[placement],'scaler_identity':info['scaler_id'],
            'parent_checkpoint_sha256':sha(cp),'checkpoint_body_decoded':False,
            'actual_encoder_execution':'source/admission-bound; no CPU model rerun'}


def query_summary(saved, replay, asset, artifact):
    n = replay['total_examples']; valid = replay['valid_examples']; targets = replay['valid_target_cells']; requested = replay['requested_observed_target_cells']
    expected = {'status':'measured' if valid else 'unsupported_zero_support','units':'frozen training-scaler standardized',
                'total_examples':n,'valid_examples':valid,'valid_channels':sum(bool(x) for x in replay['counts']),
                'valid_target_cells':targets,'requested_observed_target_cells':requested,'artifact':artifact,
                'query_checksum_fnv1a64':fnv(bytes_of(asset['target_mask']))}
    for key,value in expected.items(): check(saved[key] == value,'saved query report '+key)
    for key,value in [('example_coverage',valid/n),('target_coverage',targets/requested if requested else None),
                      ('standardized_mae',replay['mae']),('standardized_huber',replay['huber'])]:
        if value is None: check(saved[key] is None,'unsupported query null '+key)
        else: close(saved[key],value,'reported hierarchical query '+key)


def methods_for_task(task):
    check(task in ('lag_sign', 'amplitude'), 'two separately declared tasks')
    return METHODS


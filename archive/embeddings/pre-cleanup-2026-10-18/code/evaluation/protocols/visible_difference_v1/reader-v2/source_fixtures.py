#!/usr/bin/env python3
"""Artificial SOURCE fixtures; never opens a capsule or a tensor archive.

The production validator is supplied by an explicit reviewed SHA. All saved
tensor representations, fits and logits below are constructed in memory. No
model or optimizer is instantiated and no readout is fitted.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
CARD_SHA = '75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc'
V = M = None
CHECKS = 0
NEGATIVES = []
LOADS = []


def require(ok, why):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise AssertionError(why)


def reject(label, callback):
    try:
        callback()
    except (AssertionError, ValueError, KeyError, TypeError, IndexError):
        NEGATIVES.append(label)
        return
    raise AssertionError('negative fixture accepted: ' + label)


def f(dtype, shape, values):
    return M.fake(dtype, shape, values)


def s(value, dtype='LongStorage'):
    return f(dtype, [], [value])


def text(value):
    return M.fake_text(value)


def replace(record, index, value):
    values = list(record['values'])
    values[index] = value
    return f(record['dtype'], record['shape'], values)


def named(values):
    return {'count': s(len(values)), **{
        'tensor_' + str(i): {'parameter_name': text(name), 'value': value}
        for i, (name, value) in enumerate(values.items())}}


def frozen_scaler():
    return {'mean': f('DoubleStorage', [3, 3], [0.] * 9),
            'scale': f('DoubleStorage', [3, 3], [1.] * 9),
            'count': f('LongStorage', [3, 3], [32] * 9),
            'channel_ids': f('LongStorage', [3], [0, 1, 2]),
            'floor_applied': f('BoolStorage', [3, 3], [False] * 9),
            'scale_floor': s(1e-6, 'DoubleStorage')}


def original_initial():
    # Generic named-state admission is tested without constructing an encoder.
    params = named({'common.weight': f('FloatStorage', [225805], [0.] * 225805)})
    buffers = named({'static.index': f('LongStorage', [2], [0, 1])})
    return {'late_parameters': params, 'early_parameters': copy.deepcopy(params),
            'late_buffers': buffers, 'early_buffers': copy.deepcopy(buffers),
            'late_scaler': frozen_scaler(), 'early_scaler': frozen_scaler()}


def continuation(point, initial):
    old = V.named(initial['early_parameters'])
    initial_values = {**copy.deepcopy(old),
                      'visible_difference_projection.weight': f('FloatStorage', [64, 48], [0.] * 3072)}
    current = copy.deepcopy(initial_values)
    if point:
        current['common.weight'] = replace(current['common.weight'], 0, .25)
        current['visible_difference_projection.weight'] = replace(current['visible_difference_projection.weight'], 0, .125)
    optimizer = {'parameter_count': s(len(current)), 'active_state_count': s(len(current) if point else 0)}
    for i, (name, value) in enumerate(current.items()):
        item = {'parameter_name': text(name), 'parameter_shape': f('LongStorage', [len(value['shape'])], value['shape']),
                'has_state': s(bool(point), 'BoolStorage')}
        if point:
            item.update(step=s(point), exp_avg=f('FloatStorage', value['shape'], [.01] * len(value['values'])),
                        exp_avg_sq=f('FloatStorage', value['shape'], [.001] * len(value['values'])))
        optimizer['parameter_' + str(i)] = item
    return {'artifact_kind': text('rpb_visible_difference_continuation_state_v1'),
            'attempted_steps': s(point), 'completed_steps': s(point), 'sampled_rows': s(8 * point),
            'temporal_difference_input_value': s(1), 'common_parameter_values': s(225805),
            'copied_parameter_values': s(0), 'difference_parameter_values': s(3072),
            'model_parameters': named(current), 'model_buffers': copy.deepcopy(initial['early_buffers']),
            'initial_model_parameters': named(initial_values), 'initial_model_buffers': copy.deepcopy(initial['early_buffers']),
            'scaler': frozen_scaler(), 'optimizer_state': optimizer,
            'loss_trace_counters': f('LongStorage', [point, 3], [v for i in range(point) for v in (i + 1, i + 1, 192)]),
            'loss_trace_values': f('DoubleStorage', [point, 2], [.5, .1] * point)}


def controlled(prefix='train'):
    return {'observations': f('DoubleStorage', [4, 3, 32, 3], [0.] * 1152),
            'feature_mask': f('BoolStorage', [4, 3, 32, 3], [True] * 1152),
            'labels_scoring_only': f('LongStorage', [4], [0, 1, 0, 1]),
            'source_ids_json': text(json.dumps([prefix + '/a'] * 2 + [prefix + '/b'] * 2))}


def features(split, valid=None):
    valid = [True] * 4 if valid is None else valid
    rows = [[(-1. if i % 2 == 0 else 1.) if keep else 0.] + [0.] * 31 for i, keep in enumerate(valid)]
    return {'features': f('FloatStorage', [4, 32], [x for row in rows for x in row]),
            'valid': f('BoolStorage', [4], valid), 'labels_scoring_only': f('LongStorage', [4], split[3]),
            'source_ids_json': text(json.dumps(split[2])), 'provenance': text('synthetic-source-only-native32')}


def raw_scaler_case(split):
    frozen = frozen_scaler()
    identity = M.scaler(frozen)[2]
    envelope = {'encoder_id': text('raw_patch_bottleneck_mae_v1'), 'format_version': s(1),
                'artifact_kind': text('rpb_frozen_training_scaler_v1'),
                'schema_id': text('synthetic-schema'), 'fit_dataset_id': text('synthetic-data'),
                'preprocessing_id': text(identity), 'scaler': copy.deepcopy(frozen)}
    raw = {'encoder_id': text('raw_patch_bottleneck_mae_v1'), 'format_version': s(1),
           'artifact_kind': text('rpb_raw_uniform_history_v1'),
           'data': f('DoubleStorage', [4, 3, 32, 3], split[0]),
           'observed': f('BoolStorage', [4, 3, 32, 3], split[1]),
           'channel_ids': f('LongStorage', [3], [0, 1, 2]),
           'endpoints': f('DoubleStorage', [4], [31.] * 4),
           'sampling_interval': s(1., 'DoubleStorage'), 'feature_units': text('unitless,unitless,unitless'),
           'schema_id': text('synthetic-schema'), 'dataset_id': text('synthetic-data')}
    return raw, envelope, frozen


def point_metadata_case(point, ids):
    cp, old_cp = Path('/synthetic/candidate/point-' + str(point) + '/checkpoint.pt'), Path('/synthetic/original/point-0/checkpoint.pt')
    scopes = {'core_writer': 'synthetic-core', 'curve_training': 'synthetic-training', 'visible_adapter': 'synthetic-adapter'}
    raw_info = ('synthetic-schema', 'synthetic-data', M.scaler(frozen_scaler())[2])
    # These declarations are fixture inputs, not fields copied from a validator output.
    declarations = dict(V.L.CONTEXT_LITERALS)
    declarations.update(protocol_id='visible-difference-v1/lag_sign', model_tag='RPB-v13',
        architecture_id='aligned-mixer-before-temporal-visible-first-difference-v1',
        provider_id='raw_patch_bottleneck_mae_v1', adapter_id='rpb_continuous_cuda_learning_curve_v1',
        fit_policy='permitted_training_observations_only;scaler_fit_once',
        output_semantics='local_observed_visible_first_difference_contextual_pretemporal_aligned_global_semantic_mlp_bottleneck_v1',
        reconstruction_export_semantics='exact_visible_first_difference_pretemporal_contextual_observed_global_semantic_mlp_export_v1',
        temporal_difference_input='1', temporal_difference_input_semantics='actual_visible_original_adjacent_backward_difference_and_pair_visibility_additive_projection_v1',
        actual_training_seed='75272', initialization_seed=str(M.mixed(75272 ^ 0x7270622d696e6974)),
        parameter_count='228877', cuda_parameter_count='228877', fit_source_manifest=M.source_manifest(ids),
        fit_source_manifest_id='rpb-fit-source-manifest-fnv1a-v1-' + M.fnv(M.source_manifest(ids).encode()),
        training_observation_rows=str(len(ids)), training_source_groups=str(len(set(ids))),
        training_dataset_id=raw_info[1], scaler_fit_dataset_id=raw_info[1], preprocessing_id=raw_info[2],
        feature_units='unitless,unitless,unitless', channel_mixer_placement='1',
        core_writer_source_fingerprint=scopes['core_writer'], training_producer_source_fingerprint=scopes['curve_training'],
        source_fingerprint_scope='ordinary_checkpoint_field=core_writer;companion_audit_training_producer=evaluation_source',
        rng_policy='splitmix64-counter-rows-masks-torch-attempt-v1',
        optimizer_policy='one_continuous_AdamW_state;absolute_completed_update_budgets',
        sampling_policy='with_replacement_counter_rows;sampled_rows_includes_no_update_attempts',
        trace_policy='first_completed_update;every_log_every_updates;queried_milestone_endpoints_persist_as_cumulative_prefix',
        continuation_state_artifact_kind='rpb_visible_difference_continuation_state_v1', continuation_state_suffix='.continuation.pt',
        continuation_state_policy='live_named_CPU_model_buffers_scaler_AdamW_and_complete_trace_v1',
        curve_skip_policy='abort_ineligible_attempt;no_skipped_update_prefix_permitted',
        visible_difference_initialization_policy='independent-original-common-state-exact;last-zero-branch-no-RNG;assert-before-AdamW;no-copy',
        visible_difference_common_parameter_values='225805', visible_difference_added_parameter_values='3072', visible_difference_copied_parameter_values='0',
        visible_difference_parent_point0_path=str(old_cp), visible_difference_parent_core_source_fingerprint=V.PARENT_CORE,
        visible_difference_parent_training_producer_source_fingerprint=V.PARENT_TRAINING,
        visible_difference_parent_fit_protocol='early-mixer-reliability-v1/lag_sign',
        visible_difference_binding_artifact_kind='rpb_visible_difference_training_binding_v1', visible_difference_binding_suffix='.visible-difference.pt')
    settings = {'channel_count': 3, 'history_length': 32, 'input_width': 3, 'patch_length': 8,
        'encoder_width': 64, 'export_width': 32, 'num_layers': 3, 'num_heads': 4, 'feedforward_width': 256,
        'decoder_hidden_width': 128, 'channel_mixer_layers': 1, 'global_bottleneck_mode': 2, 'dropout': 0,
        'huber_delta': 1, 'sampling_interval': 1, 'batch_size': 8, 'threads': 1, 'learning_rate': .001,
        'weight_decay': .0001, 'gradient_clip_norm': 1, 'steps': 512, 'attempt_limit': 1024, 'log_every': 1,
        'seed': 75272, 'layer_norm_epsilon': 1e-5, 'mask_ratio': .25, 'scale_floor': 1e-6,
        'device': 'cuda', 'channel_ids': '0,1,2', 'channel_mixer_placement': 1, 'temporal_difference_input': 1}
    common = {k: text(v) for k, v in declarations.items()}
    common.update(attempted_steps=s(point), completed_steps=s(point), sampled_rows=s(point * 8),
                  temporal_difference_input_value=s(1), resolved_settings=text(''.join(str(k) + '=' + str(v) + '\n' for k, v in settings.items())))
    assets = []
    for kind in ('rpb_learning_curve_training_audit_v1', 'rpb_visible_difference_training_binding_v1', 'rpb_visible_difference_continuation_state_v1'):
        item = copy.deepcopy(common); item['artifact_kind'] = text(kind); assets.append(item)
    audit, binding, live = assets
    audit.update(model_weight_update_budget=text(str(point)), training_device=text('cuda'), channel_mixer_placement_value=s(1),
        channel_order=f('LongStorage', [3], [0, 1, 2]), sampling_interval=s(1., 'DoubleStorage'), endpoint=s(31., 'DoubleStorage'),
        context_deletion_ratio_value=s(.15, 'DoubleStorage'), context_deletion_stream_value=s(0x6374782d64726f70),
        weights_changed=s(bool(point), 'BoolStorage'), finite_gradients=s(bool(point), 'BoolStorage'))
    live['channel_mixer_placement_value'] = s(1)
    for item in (audit, live):
        item.update(context_requested_deleted_coordinates=s(point * 10), context_actual_deleted_coordinates=s(point * 8),
                    context_restored_coordinates=s(point * 2), training_seconds=s(.125 if point else 0., 'DoubleStorage'))
    for item in (binding, live):
        item['checkpoint_path'] = text(str(cp))
    snapshot = {k: text(v) for k, v in declarations.items()}
    snapshot.update({k: text(v) for k, v in {
        'artifact_kind': 'rpb_visible_difference_cuda_snapshot_v1', 'protocol_id': 'visible-difference-v1',
        'original_training_protocol_id': 'visible-difference-v1/lag_sign',
        'source_fingerprint_scope': 'parent=core_writer_and_training_producer;loader=new_visible_difference_adapter',
        'checkpoint_path': str(cp), 'parent_checkpoint_path': str(cp), 'snapshot_loader_source_fingerprint': scopes['visible_adapter'],
        'parent_writer_source_fingerprint': scopes['core_writer'], 'parent_training_producer_source_fingerprint': scopes['curve_training'],
        'original_encoder_attempted': str(point), 'original_encoder_completed': str(point), 'training_schema_id': raw_info[0],
        'encoder_updates': '0', 'decoder_updates': '0', 'head_refits': '0', 'no_optimizer_created': 'true',
        'snapshot_policy': 'immutable_CUDA_model;original_TRAIN_scaler;encoder_decoder_updates0;no_optimizer_load',
        'attempted_steps': str(point), 'completed_steps': str(point), 'model_weight_update_budget': str(point),
        'inference_device': 'cuda:0'}.items()})
    snapshot.update(temporal_difference_input_value=s(1), original_encoder_attempted_value=s(point), original_encoder_completed_value=s(point))
    def content(path):
        # A fixed synthetic pathname association, without reading a file.
        return 'synthetic-fnv-' + M.fnv(str(path).encode())
    for suffix in ('', '.audit.pt', '.scaler.pt', '.training-raw.pt'):
        for item in (*assets, snapshot):
            item['visible_difference_parent_content_id' + suffix] = text(content(Path(str(old_cp) + suffix)))
    for suffix in ('', '.audit.pt', '.scaler.pt', '.training-raw.pt', '.continuation.pt'):
        binding['checkpoint_content_id' + suffix] = text(content(Path(str(cp) + suffix)))
    for suffix in ('', '.audit.pt', '.scaler.pt', '.training-raw.pt', '.continuation.pt', '.visible-difference.pt'):
        snapshot['parent_content_id' + suffix] = text(content(Path(str(cp) + suffix)))
    return (*assets, snapshot, point, 75272, ids, cp, old_cp, scopes, raw_info, content)


def head_fit(split, rep):
    fit = {key: f('DoubleStorage', [32], [0.] * 32) for key in ('feature_mean', 'ridge_mean', 'tiny_mean')}
    fit.update({key: f('DoubleStorage', [32], [1.] + [1e-8] * 31) for key in ('feature_scale', 'ridge_scale', 'tiny_scale')})
    fit.update({'outer_normalizer_applied': s(True, 'BoolStorage'), 'outer_fitted_rows': s(4), 'fitted_rows': s(4),
                'ridge_weights': f('DoubleStorage', [32, 2], [-1., 1.] + [0.] * 62),
                'ridge_intercept': f('DoubleStorage', [2], [0., 0.]),
                'tiny_w1': f('DoubleStorage', [32, 16], [1.] + [0.] * 511),
                'tiny_b1': f('DoubleStorage', [16], [0.] * 16),
                'tiny_w2': f('DoubleStorage', [16, 2], [-1., 1.] + [0.] * 30),
                'tiny_b2': f('DoubleStorage', [2], [0., 0.]),
                'actual_probe_seed_decimal': text(str(M.stream_seed(rep, 32))),
                'training_source_ids_json': text(json.dumps(split[2])), 'ridge_penalty': s(1., 'DoubleStorage'),
                'tiny_hidden': s(16), 'tiny_steps': s(100), 'tiny_learning_rate': s(.01, 'DoubleStorage')})
    return fit


def prediction(split, asset):
    rows, valid = M.feature_archive(asset, 4, 32, split[2], split[3], 'FloatStorage')
    classes = [int(row[0] > 0) for row in rows]
    return {'ridge': f('LongStorage', [4], classes), 'tiny_secondary': f('LongStorage', [4], classes),
            'valid': f('BoolStorage', [4], valid), 'probe_input_features': f('DoubleStorage', [4, 32], [x for row in rows for x in row]),
            'ridge_logits': f('DoubleStorage', [4, 2], [x for row in rows for x in (-row[0], row[0])]),
            'tiny_hidden_preactivation': f('DoubleStorage', [4, 16], [x for row in rows for x in ([row[0]] + [0.] * 15)]),
            'tiny_logits': f('DoubleStorage', [4, 2], [x for row in rows for x in (-math.tanh(row[0]), math.tanh(row[0]))]),
            'labels_scoring_only': f('LongStorage', [4], split[3]), 'source_ids_json': text(json.dumps(split[2]))}


def saved_head_case(unsupported=False, null_view=False):
    data = [V.controlled(controlled(p), 4) for p in ('train', 'validation', 'validation')]
    directory = Path('/synthetic/readouts/native_visible_difference')
    store = {}
    method = {'method': 'native_visible_difference', 'size': 32, 'inputs_train_prepared': False,
              'outer_train_normalizer_fits': int(not unsupported), 'status': 'unsupported_fit' if unsupported else 'measured',
              'reason': 'valid TRAIN fitting rows require at least 2 rows and both classes' if unsupported else '', 'repetitions': []}
    assets = []
    for i, (name, split) in enumerate(zip(('training', 'validation-intact', 'validation-deleted'), data)):
        valid = [False] * 4 if (unsupported and i == 0) or (null_view and i == 2) else [True] * 4
        asset = features(split, valid); assets.append(asset); store[directory / (name + '-features.pt')] = asset
        method[name.replace('-', '_') + '_population'] = M.population(valid, split[3], split[2])
    for rep in V.REPS:
        declared = {'id': 'rep-' + str(rep), 'actual_probe_seed_decimal': str(M.stream_seed(rep, 32)),
                    'status': method['status'], 'ridge_parameters': 66, 'neural_parameters': 562}
        if not unsupported:
            folder = directory / ('rep-' + str(rep)); store[folder / 'fit.pt'] = head_fit(data[0], rep)
            declared['fit_artifact'] = 'native_visible_difference/rep-' + str(rep) + '/fit.pt'
            for i, (name, split, asset) in enumerate(zip(('training', 'validation-intact', 'validation-deleted'), data, assets)):
                pred = prediction(split, asset); store[folder / (name + '-predictions.pt')] = pred
                valid = list(asset['valid']['values']); classes = list(pred['ridge']['values'])
                score = M.score(classes, split[3], valid); score['full_population_correctness'] = score['correct'] / score['total']
                result = {'population': M.population(valid, split[3], split[2]), 'ridge': score, 'tiny_secondary': copy.deepcopy(score)}
                if i:
                    seed = M.stream_seed(75272, M.fnv64('native_visible_difference/rep-' + str(rep) + '/' + name))
                    result['bootstrap_seed_decimal'] = str(seed)
                    selected = {source: [0, 0] for source, keep in zip(split[2], valid) if keep}
                    for label, cls, keep, source in zip(split[3], classes, valid, split[2]):
                        if keep:
                            selected[source][0] += int(label == cls); selected[source][1] += 1
                    # Every supported synthetic source is perfect: exact degenerate bounds.
                    interval = {'replicates': 1000, 'confidence': .95, 'source_groups': len(selected),
                                'estimate': 1. if selected else None, 'lower': 1. if len(selected) >= 2 else None,
                                'upper': 1. if len(selected) >= 2 else None}
                    result['ridge_grouped_interval'] = interval; result['tiny_grouped_interval'] = copy.deepcopy(interval)
                declared[name.replace('-', '_')] = result
        method['repetitions'].append(declared)
    report = {'protocol': 'fixed-feature-readouts-v1', 'master_seed': '75272',
              'recipe': {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100, 'tiny_learning_rate': .01,
                         'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods',
                         'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0},
              'methods': [method], 'pairs': [], 'fit_counts': {'outer_train_normalizer_fits': int(not unsupported),
                         'ridge_fits': 3 * int(not unsupported), 'tiny_fits': 3 * int(not unsupported),
                         'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}}
    return store, directory, data, report


def head_call(case):
    store, directory, data, report = case
    def load(path):
        require(path in store, 'synthetic-only admitted dictionary lookup')
        LOADS.append(str(path))
        return store[path]
    return V.saved_heads(load, directory, data, report)


def fixtures(original_reader, actual_reader, actual_reader_sha):
    base = controlled(); split = V.controlled(base, 4)
    require(len(split[0]) == 1152 and split[3] == [0, 1, 0, 1], 'actual controlled positive geometry')
    deleted = copy.deepcopy(base); deleted['requested_erasure'] = f('BoolStorage', [4, 3, 32, 3], [False] * 1152)
    V.controlled(deleted, 4)
    for label, mutate in (
        ('controlled wrong dtype', lambda a: a.update(observations=f('FloatStorage', [4, 3, 32, 3], [0.] * 1152))),
        ('controlled wrong shape', lambda a: a.update(observations=f('DoubleStorage', [4, 3, 3, 32], [0.] * 1152))),
        ('controlled extra field', lambda a: a.update(hidden_phase=s(0))),
        ('controlled malformed labels', lambda a: a.update(labels_scoring_only=f('LongStorage', [4], [0, 0, 0, 1]))),
        ('controlled source mismatch', lambda a: a.update(source_ids_json=text('["a","b","c","c"]'))),
        ('controlled paired masks differ', lambda a: a.update(feature_mask=replace(a['feature_mask'], 0, False))),
        ('controlled nonfinite observed value', lambda a: a.update(observations=replace(a['observations'], 0, float('nan'))))):
        bad = copy.deepcopy(base); mutate(bad); reject(label, lambda bad=bad: V.controlled(bad, 4))
    bad = copy.deepcopy(base); mask = list(bad['feature_mask']['values']); mask[0] = mask[288] = False
    bad['feature_mask'] = f('BoolStorage', [4, 3, 32, 3], mask); bad['observations'] = replace(bad['observations'], 0, 1.)
    reject('controlled hidden nonzero', lambda: V.controlled(bad, 4))
    raw, envelope, frozen = raw_scaler_case(split)
    require(V.raw_scaler_envelope(raw, envelope, split, frozen) ==
            ('synthetic-schema', 'synthetic-data', M.scaler(frozen)[2]), 'actual nested raw/scaler envelope positive')
    for label, mutate in (
        ('raw wrong original values', lambda r, e: r.update(data=replace(r['data'], 0, 1.))),
        ('raw wrong original mask', lambda r, e: r.update(observed=replace(r['observed'], 0, False))),
        ('raw wrong schema association', lambda r, e: r.update(schema_id=text('other-schema'))),
        ('raw wrong original dataset', lambda r, e: e.update(fit_dataset_id=text('other-data'))),
        ('raw wrong sampling interval', lambda r, e: r.update(sampling_interval=s(2., 'DoubleStorage'))),
        ('raw wrong original channel ordering', lambda r, e: r.update(channel_ids=f('LongStorage', [3], [1, 0, 2]))),
        ('raw changed feature units', lambda r, e: r.update(feature_units=text('other-units'))),
        ('raw mistyped format version', lambda r, e: r.update(format_version=s(1., 'DoubleStorage'))),
        ('scaler wrong nested kind', lambda r, e: e.update(artifact_kind=text('ordinary_scaler'))),
        ('scaler missing nested field', lambda r, e: e['scaler'].pop('count')),
        ('scaler changed nested original mean', lambda r, e: e['scaler'].update(mean=replace(e['scaler']['mean'], 0, 1.))),
        ('scaler wrong preprocessing identity', lambda r, e: e.update(preprocessing_id=text('other-preprocessing')))):
        bad_raw, bad_envelope = copy.deepcopy(raw), copy.deepcopy(envelope)
        mutate(bad_raw, bad_envelope)
        reject(label, lambda r=bad_raw, e=bad_envelope: V.raw_scaler_envelope(r, e, split, frozen))
    group = named({'a': f('FloatStorage', [2], [1., 2.]), 'b': f('LongStorage', [1], [3])})
    require(list(V.named(group, 3)) == ['a', 'b'], 'closed numbered named group positive')
    bad = copy.deepcopy(group); bad['tensor_1']['parameter_name'] = text('a'); reject('duplicate named tensor', lambda: V.named(bad, 3))
    bad = copy.deepcopy(group); del bad['tensor_1']; reject('missing named tensor', lambda: V.named(bad, 3))
    reject('wrong named total', lambda: V.named(group, 4))
    initial = original_initial(); zero = continuation(0, initial); trained = continuation(2, initial)
    require(len(V.state(zero, 0, initial)[3]) == 2, 'actual zero-state full228877 positive')
    require(len(V.state(trained, 2, initial)[1]) == 6, 'actual trained state/AdamW/trace positive')
    for label, key, value in (
        ('state wrong kind', 'artifact_kind', text('ordinary_training_state')),
        ('state wrong attempted', 'attempted_steps', s(3)),
        ('state mistyped counter', 'completed_steps', s(2., 'DoubleStorage')),
        ('state wrong branch discriminator', 'temporal_difference_input_value', s(0)),
        ('state invented copies', 'copied_parameter_values', s(225805)),
        ('state reordered trace', 'loss_trace_counters', f('LongStorage', [2, 3], [2, 2, 192, 1, 1, 192])),
        ('state empty target support', 'loss_trace_counters', f('LongStorage', [2, 3], [1, 1, 0, 2, 2, 192])),
        ('state negative objective', 'loss_trace_values', f('DoubleStorage', [2, 2], [-.5, .1, .5, .1])),
        ('state nonfinite gradient', 'loss_trace_values', f('DoubleStorage', [2, 2], [.5, float('nan'), .5, .1]))):
        bad = copy.deepcopy(trained); bad[key] = value; reject(label, lambda bad=bad: V.state(bad, 2, initial))
    for label, mutate in (
        ('state wrong initial common bytes', lambda a: a['initial_model_parameters']['tensor_0'].update(value=replace(a['initial_model_parameters']['tensor_0']['value'], 0, 1.))),
        ('state nonzero initial branch', lambda a: a['initial_model_parameters']['tensor_1'].update(value=replace(a['initial_model_parameters']['tensor_1']['value'], 0, 1.))),
        ('state changed current name', lambda a: a['model_parameters']['tensor_0'].update(parameter_name=text('other.weight'))),
        ('state changed current shape', lambda a: a['model_parameters']['tensor_0'].update(value=f('FloatStorage', [5, 45161], a['model_parameters']['tensor_0']['value']['values']))),
        ('state changed current dtype', lambda a: a['model_parameters']['tensor_0'].update(value=f('DoubleStorage', [225805], a['model_parameters']['tensor_0']['value']['values']))),
        ('state changed static buffer', lambda a: a['model_buffers']['tensor_0'].update(value=f('LongStorage', [2], [1, 0]))),
        ('state changed original scaler', lambda a: a['scaler'].update(mean=replace(a['scaler']['mean'], 0, 1.))),
        ('state duplicate optimizer name', lambda a: a['optimizer_state']['parameter_1'].update(parameter_name=text('common.weight'))),
        ('state inconsistent active count', lambda a: a['optimizer_state'].update(active_state_count=s(1))),
        ('state negative second moment', lambda a: a['optimizer_state']['parameter_1'].update(exp_avg_sq=replace(a['optimizer_state']['parameter_1']['exp_avg_sq'], 0, -.001))),
        ('state branch incomplete optimizer step', lambda a: a['optimizer_state']['parameter_1'].update(step=s(1)))):
        bad = copy.deepcopy(trained); mutate(bad); reject(label, lambda bad=bad: V.state(bad, 2, initial))
    bad = copy.deepcopy(zero); bad['optimizer_state']['parameter_1']['has_state'] = s(True, 'BoolStorage')
    reject('point0 invented optimizer state', lambda: V.state(bad, 0, initial))
    metadata = point_metadata_case(2, split[2])
    require(V.point_metadata(*metadata) == ([20, 16, 4], .125), 'actual trained binding and immutable snapshot scope positive')
    require(V.point_metadata(*point_metadata_case(0, split[2])) == ([0, 0, 0], 0.), 'actual point0 binding and immutable snapshot scope positive')
    for label, index, key, value in (
        ('binding wrong typed kind', 1, 'artifact_kind', text('rpb_visible_difference_binding_v1')),
        ('snapshot wrong fit-scoped protocol', 3, 'protocol_id', text('visible-difference-v1/lag_sign')),
        ('snapshot wrong original fit namespace', 3, 'original_training_protocol_id', text('early-mixer-reliability-v1/lag_sign')),
        ('binding stale original parent bytes', 1, 'visible_difference_parent_content_id', text('stale')),
        ('binding stale candidate continuation bytes', 1, 'checkpoint_content_id.continuation.pt', text('stale')),
        ('snapshot stale sixth sidecar bytes', 3, 'parent_content_id.visible-difference.pt', text('stale')),
        ('snapshot wrong loader source', 3, 'snapshot_loader_source_fingerprint', text('other-source'))):
        bad = list(copy.deepcopy(metadata)); bad[index][key] = value
        reject(label, lambda bad=bad: V.point_metadata(*bad))
    feature = features(split); V.parity(feature, copy.deepcopy(feature), split)
    bad = copy.deepcopy(feature); bad['features'] = replace(bad['features'], 0, -1.000001)
    reject('initial feature byte mismatch', lambda: V.parity(bad, feature, split))
    bad = copy.deepcopy(feature); bad['source_ids_json'] = text('["wrong","wrong","train/b","train/b"]')
    reject('initial feature source mismatch', lambda: V.parity(bad, feature, split))
    bad = copy.deepcopy(feature); bad['valid'] = f('BoolStorage', [4], [False, True, True, True])
    reject('invalid feature invented payload', lambda: V.parity(bad, feature, split))
    measured = saved_head_case(); require(len(head_call(measured)) == 3, 'all fixed synthetic head repetitions positive')
    null_case = saved_head_case(null_view=True)
    null_heads = head_call(null_case)
    require(all(r['views']['validation_deleted']['scores']['ridge']['accuracy'] is None for r in null_heads), 'zero valid view remains null after supported TRAIN fit')
    unsupported = saved_head_case(unsupported=True); before = len(LOADS); unsupported_heads = head_call(unsupported)
    require(all(r['status'] == 'unsupported_fit' and r['views'] == {} for r in unsupported_heads), 'unsupported fit retained')
    require(len(LOADS) - before == 3, 'unsupported fit never requests invented fit/predictions')
    for label, mutate in (
        ('head wrong master', lambda a: a[3].update(master_seed='1')),
        ('head changed recipe', lambda a: a[3]['recipe'].update(tiny_steps=200)),
        ('head omitted repetition', lambda a: a[3]['methods'][0]['repetitions'].pop()),
        ('head wrong width-paired seed', lambda a: a[3]['methods'][0]['repetitions'][0].update(actual_probe_seed_decimal='0')),
        ('head invented population', lambda a: a[3]['methods'][0]['validation_deleted_population'].update(valid_rows=3)),
        ('head wrong fitting artifact', lambda a: a[3]['methods'][0]['repetitions'][0].update(fit_artifact='other/fit.pt')),
        ('head changed conditional interval', lambda a: a[3]['methods'][0]['repetitions'][0]['validation_intact']['ridge_grouped_interval'].update(lower=.5))):
        bad = copy.deepcopy(measured); mutate(bad); reject(label, lambda bad=bad: head_call(bad))
    fit_path = measured[1] / 'rep-2701/fit.pt'; prediction_path = measured[1] / 'rep-2701/training-predictions.pt'
    bad = copy.deepcopy(measured); bad[0][fit_path]['actual_probe_seed_decimal'] = text('0')
    reject('saved fit changed seed', lambda: head_call(bad))
    bad = copy.deepcopy(measured); bad[0][prediction_path]['ridge'] = replace(bad[0][prediction_path]['ridge'], 0, 1)
    reject('saved classes differ from own logits', lambda: head_call(bad))
    bad = copy.deepcopy(measured); bad[0][fit_path]['pca_components'] = f('DoubleStorage', [1], [1.])
    reject('post-encoder PCA state rejected', lambda: head_call(bad))
    bad = copy.deepcopy(unsupported); bad[3]['methods'][0]['reason'] = ''
    reject('unsupported fit missing reason', lambda: head_call(bad))
    # Execute only the actual run() query-summary call with in-memory records.
    # The archive path remains a synthetic admitted-role stand-in; no file opens.
    p = Path('/synthetic/candidate/point-512/training-reconstruction.pt')
    asset = {'target_mask': f('BoolStorage', [4], [True, True, True, False])}
    replay = {'total_examples': 4, 'valid_examples': 4, 'valid_target_cells': 3,
              'requested_observed_target_cells': 4, 'counts': [1, 0, 2], 'mae': .25, 'huber': .1}
    declared = {'status': 'measured', 'units': 'frozen training-scaler standardized',
                'total_examples': 4, 'valid_examples': 4, 'valid_channels': 2,
                'valid_target_cells': 3, 'requested_observed_target_cells': 4,
                'artifact': p.name, 'query_checksum_fnv1a64': M.fnv(M.bytes_of(asset['target_mask'])),
                'example_coverage': 1., 'target_coverage': .75, 'standardized_mae': .25, 'standardized_huber': .1}
    run_ast = next(n for n in ast.parse(actual_reader).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    calls = [n for n in ast.walk(run_ast) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'L' and n.func.attr == 'query_summary']
    require(len(calls) == 1 and ast.unparse(calls[0].args[-1]) == 'p.name', 'exact actual repaired query call')
    statement = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])], type_ignores=[]))
    exec(compile(statement, '<synthetic-actual-query-summary-call>', 'exec'),
         {'L': V.L, 'declared': declared, 'replay': replay, 'asset': asset, 'p': p})
    reject('query report absolute path is not basename', lambda: V.L.query_summary(declared, replay, asset, str(p)))
    wrong = dict(declared, artifact='validation-reconstruction.pt')
    reject('query report wrong basename', lambda: V.L.query_summary(wrong, replay, asset, p.name))
    proof = V.reader_repair_provenance(original_reader, actual_reader, Path('/synthetic/captured-reader.py'),
                                     Path('/synthetic/repaired-reader.py'), actual_reader_sha)
    require(proof['exact_original_byte_derivation'] and proof['numeric_modules_unchanged'], 'full original SOURCE byte derivation')
    changed = actual_reader.replace(b'head_pipelines <= 15', b'head_pipelines <= 16', 1)
    require(changed != actual_reader, 'extra numeric edit fixture changes actual code')
    reject('unapproved extra numerical SOURCE change', lambda: V.reader_repair_provenance(
        original_reader, changed, p, p, hashlib.sha256(changed).hexdigest()))
    reject('changed captured original SOURCE', lambda: V.reader_repair_provenance(
        original_reader[:-1] + b'X', actual_reader, p, p, actual_reader_sha))
    require(V.DECODES == 0, 'real validator archive decodes remain zero')


def main():
    global V, M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validator-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'existing managed container only')
    root = Path(__file__).resolve().parent
    validator = root / 'validate_visible_difference.py'
    original_validator = root.parent / 'reader/validate_visible_difference.py'
    card = Path('/embedding/code/evaluation/cards/visible_difference_v1.md')
    source_paths = [Path(__file__).resolve(), validator, card,
                    *[root / p for p in ('saved_cpu_math.py', 'archive_codec.py', 'legacy_implementation.py')], original_validator]
    seen = set()
    for path in source_paths:
        require(path.resolve(strict=True) == path and all(not q.is_symlink() for q in (path, *path.parents)), 'canonical SOURCE file')
        st = path.stat()
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and (st.st_dev, st.st_ino) not in seen, 'unique regular SOURCE')
        seen.add((st.st_dev, st.st_ino))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    require(hashes[str(validator)] == args.validator_sha256 and hashes[str(card)] == CARD_SHA, 'reviewed validator/card source pins')
    output = Path(args.output)
    require(output.is_absolute() and not output.exists() and output.parent.resolve(strict=True) == output.parent and
            all(not p.is_symlink() for p in (output.parent, *output.parent.parents)) and
            output.name == 'source-fixtures.json' and output.parent == root, 'exclusive prescribed SOURCE fixture record')
    spec = importlib.util.spec_from_file_location('visible_difference_fixture_validator', validator)
    V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V); V.modules(); M = V.M
    def forbidden(*args, **kwargs):
        raise AssertionError('actual archive/capsule execution forbidden in SOURCE fixtures')
    V.R.load = forbidden; V.run = forbidden
    fixtures(original_validator.read_bytes(), validator.read_bytes(), args.validator_sha256)
    require(not any(name in sys.modules for name in ('torch', 'numpy', 'sklearn')), 'no model/numeric-runtime imports')
    for path in source_paths:
        require(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)], 'SOURCE input bytes preserved')
    result = {'status': 'passed', 'source_only': True, 'validator_sha256': hashes[str(validator)],
              'original_validator_sha256': hashes[str(original_validator)],
              'fixture_script_sha256': hashes[str(Path(__file__).resolve())], 'card_sha256': CARD_SHA,
              'module_sha256': V.MODULE_PINS, 'checks': CHECKS + V.CHECKS + M.CHECKS + V.R.CHECKS + V.L.CHECKS,
              'negative_fixtures': len(NEGATIVES), 'negative_cases': NEGATIVES,
              'positive_branches': ['controlled', 'raw_scaler_envelope', 'named', 'point0_state', 'trained_state', 'point0_metadata', 'trained_metadata', 'initial_parity',
                                    'all_three_saved_heads', 'undefined_validation_view', 'unsupported_fit',
                                    'actual_query_basename_call', 'exact_original_reader_byte_derivation'],
              'synthetic_dictionary_loads': len(LOADS), 'actual_archive_reads': 0, 'actual_capsule_reads': 0,
              'model_forward': 0, 'optimizer_execution': 0, 'head_fits': 0, 'PCA_fits': 0,
              'synthetic_saved_head_arithmetic': True, 'all_source_inputs_preserved': True}
    with output.open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

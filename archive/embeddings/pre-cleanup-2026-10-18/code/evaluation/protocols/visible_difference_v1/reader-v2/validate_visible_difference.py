#!/usr/bin/env python3
"""Independent saved CPU arithmetic for one candidate and retained controls.

No model, optimizer, classifier, PCA, dataset generator or CUDA execution.
The prospective card and this source must be frozen before quality training.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import stat
import sys
import time

sys.dont_write_bytecode = True
PROTOCOL = 'visible-difference-v1'
MASTERS = [75272, 76373, 77474, 78575, 79676]
REPS = [2701, 2802, 2903]
CARD_SHA = '75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc'
PARENT_PINS = {'artifact-integrity.json': '9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa',
               'parent-validation.json': 'cf79abdca511f22778622d2b08e62b74b592c10618d1090ea2bd53eefd84539c',
               'parent-summary.json': '1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4'}
PARENT_CORE = '6a8a526087a7cf14fdd9f9f4d0e35218349f766bc24c7a7c58f4f388681bb7b1'
PARENT_TRAINING = '63eae8c8b907885f14a9848d5f203ad6aef09c281489c34bc0b07cf3c7cfd1de'
ARCHITECTURE = 'aligned-mixer-before-temporal-visible-first-difference-v1'
INPUT_SEMANTICS = 'actual_visible_original_adjacent_backward_difference_and_pair_visibility_additive_projection_v1'
# BEGIN query basename repair v2 declarations
import re
ORIGINAL_READER_SHA = 'ea60cd616fe033b90d5b6bdfd04b791cd63e3a002e77d8bd3bbb5f9bc89de657'
ORIGINAL_READER_SEAL_SHA = '9a68e9fe41e44d55469373e038c4ee584709d6b74a9530ec4917f315072aaf47'
ORIGINAL_FAILURE_PATH = '/embedding/output/runs/rpb-visible-difference/audit-tools/run-wnUUCa-v1/validation.json'
ORIGINAL_FAILURE_SHA = '173eaceca4ef7e782d03f4e8aaeca5fbea2e05dfd0387e32aaa340518f4fe2d0'
# END query basename repair v2 declarations
MODULE_PINS = {
    'saved_cpu_math.py': '1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9',
    'archive_codec.py': '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d',
    'legacy_implementation.py': 'f0c64b9fe67a90a9d95196433e05dec527515778d1635bdc909d1703fa278b68',
}
CHECKS = DECODES = 0
M = R = L = None
AUDIT_OUTPUT = None

def check(ok, why):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise AssertionError(why)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def admit(paths):
    seen = set()
    for path in paths:
        check(path.is_absolute() and path.resolve(strict=True) == path and all(not p.is_symlink() for p in (path, *path.parents)), 'direct canonical input role')
        s = path.stat()
        check(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and (s.st_dev, s.st_ino) not in seen, 'unique regular non-aliased file role')
        seen.add((s.st_dev, s.st_ino))

# BEGIN query basename repair v2 provenance
def reader_repair_provenance(original, actual, original_path, actual_path, actual_sha):
    check(hashlib.sha256(original).hexdigest() == ORIGINAL_READER_SHA and
          hashlib.sha256(actual).hexdigest() == actual_sha, 'original and actual reader SOURCE pins')
    beforeimage = actual
    for label in ('declarations', 'provenance', 'capture', 'result'):
        pattern = (rb'(?m)^[ \t]*# BEGIN query basename repair v2 ' + label.encode() +
                   rb'\n[\s\S]*?^[ \t]*# END query basename repair v2 ' + label.encode() + rb'\n')
        check(len(re.findall(pattern, beforeimage)) == 1, 'one exact repair metadata block ' + label)
        beforeimage = re.sub(pattern, b'', beforeimage)
    replacements = [
        (b"    check(before['independent-reader/validate_visible_difference.py'] == ORIGINAL_READER_SHA, 'exact prequality captured validator SOURCE')\n",
         b"    check(before['independent-reader/validate_visible_difference.py'] == args.source_sha256, 'exact prequality captured validator SOURCE')\n"),
        (b'            L.query_summary(declared, replay, asset, p.name)\n',
         b'            L.query_summary(declared, replay, asset, str(p))\n')]
    for replacement, old in replacements:
        check(beforeimage.count(replacement) == 1, 'one exact anchored repaired call/binding')
        beforeimage = beforeimage.replace(replacement, old, 1)
    check(beforeimage == original, 'full exact original reader byte derivation; no other changes')
    return {'protocol': 'visible-difference-reader-query-basename-repair-v2',
            'evidence_stage': 'metadata_path_repair_after_completed_quality',
            'original_reader_path': str(original_path), 'original_reader_sha256': ORIGINAL_READER_SHA,
            'original_reader_seal_sha256': ORIGINAL_READER_SEAL_SHA,
            'actual_reader_path': str(actual_path), 'actual_reader_sha256': actual_sha,
            'original_failed_audit': {'path': ORIGINAL_FAILURE_PATH, 'sha256': ORIGINAL_FAILURE_SHA},
            'narrow_replacements': [{'beforeimage': old.decode().strip(), 'replacement': replacement.decode().strip()}
                                    for replacement, old in replacements],
            'exact_original_byte_derivation': True, 'numeric_modules_unchanged': True,
            'capsule_modified': False, 'query_artifact_contract': 'producer filename basename; full archive path separately admitted and inventory bound'}
# END query basename repair v2 provenance
def safe_name(value):
    check(isinstance(value, str) and value and '\\' not in value and
          PurePosixPath(value).as_posix() == value and not PurePosixPath(value).is_absolute() and
          all(x not in ('.', '..', '') for x in PurePosixPath(value).parts), 'normalized relative role')
    return value

def retained_role_names():
    names = []
    for master in MASTERS:
        b = f'results/seed-{master}-lag_sign/'
        names += [b + n for n in ('controlled-training.pt', 'controlled-validation.pt', 'controlled-validation-deleted.pt', 'initial-pair.pt')]
        names += [b + 'early/point-0/checkpoint.pt' + s for s in ('', '.audit.pt', '.scaler.pt', '.training-raw.pt', '.structured-timing.pt')]
        names += [b + 'lag_sign/readouts/untrained_early/' + v + '-features.pt' for v in ('training', 'validation-intact', 'validation-deleted')]
        names += [b + f'lag_sign/readouts/{c}/rep-{r}/{v}-predictions.pt' for c in ('native_late', 'native_early')
                  for r in REPS for v in ('validation-intact', 'validation-deleted')]
        names += [b + 'early/point-512/' + v + '-reconstruction.pt' for v in ('training', 'validation')]
    check(len(names) == len(set(names)) == 130, 'literal closed retained role matrix')
    return sorted(names)

def source_binding(capsule, meta, before, report, card_sha):
    value = meta(capsule / 'source-manifest.json')
    rows = value['files']; names = [safe_name(r['path']) for r in rows]
    check(names == sorted(set(names)) and names and all(n in ('Makefile', 'setup.sh', 'dependencies.lock') or n.startswith('code/') for n in names),
          'canonical captured producer SOURCE namespace')
    check({'source/' + n for n in names} == {n for n in before if n.startswith('source/')}, 'complete captured SOURCE role matrix')
    for r in rows:
        check(set(r) == {'path', 'bytes', 'sha256'} and (capsule / 'source' / r['path']).stat().st_size == r['bytes'] and
              before['source/' + r['path']] == r['sha256'], 'exact captured producer SOURCE byte identity')
    body = ''.join(r['sha256'] + '  ' + r['path'] + '\n' for r in rows)
    fp = hashlib.sha256(body.encode()).hexdigest()
    check(value['protocol'] == PROTOCOL and value['source_sha256'] == fp == report['source_fingerprint'] and
          (capsule / 'source-inputs.sha256').read_text() == body and (capsule / 'source-fingerprint.txt').read_text() == fp + '\n',
          'compiled enclosing SOURCE identity')
    check(next(r['sha256'] for r in rows if r['path'] == 'code/evaluation/cards/visible_difference_v1.md') == card_sha,
          'card bound in producer SOURCE closure')
    scopes = meta(capsule / 'admission/source-scopes.json')
    check(set(scopes) == {'core_writer', 'curve_training', 'early_adapter', 'visible_adapter'}, 'exact four compilation scopes')
    index = {r['path']: r for r in rows}; ids = {}
    for name, scope in scopes.items():
        paths = [safe_name(r['path']) for r in scope['files']]
        check(paths == sorted(set(paths)) and paths and set(paths) <= set(index) and scope['files'] == [index[p] for p in paths],
              'whole canonical compilation source subset')
        identity = hashlib.sha256(''.join(index[p]['sha256'] + '  ' + p + '\n' for p in paths).encode()).hexdigest()
        check(scope['source_sha256'] == identity, 'canonical compile scope identity')
        ids[name] = identity
    check(scopes['visible_adapter']['files'] == rows and ids['visible_adapter'] == fp, 'new adapter scope is full enclosing closure')
    log_path = capsule / 'admission/build-and-tests.log'; log = log_path.read_text()
    check('/embedding/.external/libtorch' not in log and 'not found' not in log and all(s in log for s in
          ('Visible difference model CUDA admission passed', 'Visible difference adapter CUDA admission passed',
           'Fixed feature readout tests passed', 'Frozen role guard checks passed', 'Container SDK proof:')), 'actual CUDA/shared/internal SDK admission')
    binary = '/opt/cuwacunu_embedding/build/rpb-visible-difference/embedding_visible_difference'
    old = 'code/encoders/raw_patch_bottleneck_mae/src/'
    od = '/opt/cuwacunu_embedding/build/rpb-visible-difference/code/encoders/raw_patch_bottleneck_mae/'
    nd = '/opt/cuwacunu_embedding/build/rpb-visible-difference/code/protocols/visible_difference_v1/'
    requirements = [(old + 'workflow.cpp', od + 'workflow.o', 'RPB_SOURCE_ID', 'core_writer'),
                    (old + 'learning_curve_adapter.cpp', od + 'learning_curve_adapter.o', 'EVALUATION_SOURCE_ID', 'curve_training'),
                    (old + 'early_mixer_adapter.cpp', od + 'early_mixer_adapter.o', 'EARLY_MIXER_ADAPTER_SOURCE_ID', 'early_adapter'),
                    (old + 'visible_difference_adapter.cpp', nd + 'visible_difference_adapter.o', 'VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID', 'visible_adapter'),
                    ('code/evaluation/protocols/visible_difference_v1/visible_difference_main.cpp', nd + 'visible_difference_main.o', 'VISIBLE_DIFFERENCE_SOURCE_ID', 'visible_adapter')]
    commands = []
    for line in log.splitlines():
        try:
            commands.append(shlex.split(line))
        except ValueError:
            pass
    for src, obj, macro, scope in requirements:
        found = [a for a in commands if '-c' in a and '-o' in a and a[a.index('-c') + 1] == src and a[a.index('-o') + 1] == obj]
        check(len(found) == 1 and [v.split('=', 1)[1].strip('"') for v in found[0] if v.startswith('-D' + macro + '=')] == [ids[scope]],
              'one exact current producer compile/macro')
    links = [a for a in commands if '-c' not in a and '-o' in a and a[a.index('-o') + 1] == binary]
    check(len(links) == 1 and all(links[0].count(obj) == 1 for _, obj, _, _ in requirements), 'same five current objects linked exactly once')
    passed = meta(capsule / 'admission/passed.json'); launch = meta(capsule / 'launch-plan.json')
    check(passed['status'] == 'passed' and passed['protocol'] == launch['protocol'] == PROTOCOL and
          passed['source_sha256'] == launch['source_sha256'] == fp and passed['card_sha256'] == launch['card_sha256'] == card_sha and
          passed['binary'] == launch['binary'] and passed['log'] == launch['log'] and launch['log']['sha256'] == before['admission/build-and-tests.log'] and
          launch['admission_sha256'] == before['admission/passed.json'] and passed['quality_generated'] is False,
          'actual admission/capsule compiler and binary association')
    return fp, ids

def modules():
    global M, R, L
    root = Path(__file__).resolve().parent
    paths = [root / name for name in MODULE_PINS]
    admit(paths)
    for path, name in zip(paths, MODULE_PINS):
        check(sha(path) == MODULE_PINS[name], 'unchanged independent saved arithmetic module')
    loaded = []
    for path in paths:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    M, R, L = loaded
    M.R = R

def controlled(asset, rows):
    check(set(asset) in ({'observations', 'feature_mask', 'labels_scoring_only', 'source_ids_json'},
                         {'observations', 'feature_mask', 'labels_scoring_only', 'source_ids_json', 'requested_erasure'}), 'literal retained controlled observation schema')
    values = M.tensor(asset['observations'], 'DoubleStorage', [rows, 3, 32, 3])
    mask = list(M.tensor(asset['feature_mask'], 'BoolStorage', [rows, 3, 32, 3]))
    labels = list(M.tensor(asset['labels_scoring_only'], 'LongStorage', [rows]))
    ids = json.loads(M.text(asset, 'source_ids_json'))
    check(len(ids) == rows and all(isinstance(x, str) and x for x in ids), 'controlled row association')
    M.finite(values, 'finite legal observations')
    check(all(v or x == 0 for x, v in zip(values, mask)), 'absent storage exactly zero')
    check(len(M.validate_pairs(labels, ids, [mask[i * 288:(i + 1) * 288] for i in range(rows)])) == rows // 2, 'whole paired controlled sources')
    return values, mask, ids, labels

def named(value, total=None):
    return L.named_group(value, total)

def state(asset, point, initial):
    check(M.text(asset, 'artifact_kind') == 'rpb_visible_difference_continuation_state_v1', 'candidate live CPU state kind')
    check(M.scalar(asset, 'attempted_steps') == M.scalar(asset, 'completed_steps') == point and M.scalar(asset, 'sampled_rows') == 8 * point, 'complete candidate update prefix')
    check(M.scalar(asset, 'temporal_difference_input_value') == 1 and M.scalar(asset, 'common_parameter_values') == 225805 and
          M.scalar(asset, 'copied_parameter_values') == 0 and M.scalar(asset, 'difference_parameter_values') == 3072, 'typed independent initialization route')
    params = named(asset['model_parameters'], 228877)
    buffers = named(asset['model_buffers'])
    initial_params = named(asset['initial_model_parameters'], 228877)
    initial_buffers = named(asset['initial_model_buffers'])
    check(list(params) == sorted(params) and list(buffers) == sorted(buffers) and
          list(initial_params) == sorted(initial_params) and list(initial_buffers) == sorted(initial_buffers),
          'literal lexical live-state writer order')
    check(set(params) == set(initial_params) and set(buffers) == set(initial_buffers), 'current/initial exact named state sets')
    for name, value in params.items():
        check(value['dtype'] == initial_params[name]['dtype'] == 'FloatStorage' and
              value['shape'] == initial_params[name]['shape'], 'current/initial parameter geometry ' + name)
    L.same_named(buffers, initial_buffers, 'static buffers stay identical through training')
    old_params, old_buffers, old_scaler, identity = L.initial_witness(initial)
    extra = 'visible_difference_projection.weight'
    check(set(initial_params) == set(old_params) | {extra}, 'one extra named parameter')
    for name in old_params:
        M.exact(initial_params[name], old_params[name], 'same original common initialization ' + name)
    L.same_named(initial_buffers, old_buffers, 'same original initial buffers')
    check(initial_params[extra]['shape'] == [64, 48] and initial_params[extra]['dtype'] == 'FloatStorage' and not any(initial_params[extra]['values']), 'zero bias-free difference branch')
    current_scaler = M.group(asset['scaler'])
    check(set(current_scaler) == set(old_scaler), 'same scaler schema')
    for key in old_scaler:
        M.exact(current_scaler[key], old_scaler[key], 'same original TRAIN scaler ' + key)
    M.scaler(current_scaler)
    if point == 0:
        L.same_named(params, initial_params, 'point0 current/initial exact state')
        L.same_named(buffers, initial_buffers, 'point0 current/initial exact buffers')
    else:
        check(any(params[extra]['values']), 'new difference weight actually updated')
    counters = M.tensor(asset['loss_trace_counters'], 'LongStorage', [point, 3])
    losses = M.tensor(asset['loss_trace_values'], 'DoubleStorage', [point, 2])
    M.finite(losses, 'finite candidate objective trace')
    for i in range(point):
        check(counters[3 * i] == counters[3 * i + 1] == i + 1 and counters[3 * i + 2] > 0, 'ordered unskipped target trace')
        check(losses[2 * i] >= 0 and losses[2 * i + 1] >= 0, 'nonnegative objective/gradient norm')
    opt = M.group(asset['optimizer_state'])
    count = M.scalar(opt, 'parameter_count')
    check(count == len(params) and set(opt) == {'parameter_count', 'active_state_count'} | {f'parameter_{i}' for i in range(count)}, 'closed named AdamW witness')
    names, active = [], 0
    for i in range(count):
        entry = M.group(opt[f'parameter_{i}']); name = M.text(entry, 'parameter_name'); names.append(name)
        check(name in params and list(M.tensor(entry['parameter_shape'], 'LongStorage', [len(params[name]['shape'])])) == params[name]['shape'], 'AdamW parameter shape association')
        has = M.scalar(entry, 'has_state', 'BoolStorage')
        check(set(entry) == ({'parameter_name', 'parameter_shape', 'has_state', 'step', 'exp_avg', 'exp_avg_sq'} if has else {'parameter_name', 'parameter_shape', 'has_state'}), 'closed AdamW entry')
        if has:
            active += 1
            check(0 < M.scalar(entry, 'step') <= point, 'live optimizer step bound')
            for key in ('exp_avg', 'exp_avg_sq'):
                values = M.tensor(entry[key], params[name]['dtype'], params[name]['shape']); M.finite(values, 'finite AdamW moments')
                if key == 'exp_avg_sq':
                    check(all(x >= 0 for x in values), 'nonnegative second moment')
    check(names == sorted(params) and active == M.scalar(opt, 'active_state_count') and (point != 0 or active == 0), 'exact named active optimizer state')
    if point:
        branch = next(M.group(opt[f'parameter_{i}']) for i in range(count) if names[i] == extra)
        check(M.scalar(branch, 'has_state', 'BoolStorage') == 1 and M.scalar(branch, 'step') == point, 'difference branch updated through full trajectory')
    return current_scaler, counters, losses, params, buffers

def parity(asset, retained, split):
    rows = len(split[2])
    a, av = M.feature_archive(asset, rows, 32, split[2], split[3], 'FloatStorage')
    b, bv = M.feature_archive(retained, rows, 32, split[2], split[3], 'FloatStorage')
    check(av == bv and a == b, 'candidate point0 feature parity with saved untrained early')
    M.exact(asset['features'], retained['features'], 'exact initial native F32 bytes')

def raw_scaler_envelope(raw, scaler_asset, split, frozen):
    check(set(raw) == {'encoder_id', 'format_version', 'artifact_kind', 'data', 'observed', 'channel_ids',
          'endpoints', 'sampling_interval', 'feature_units', 'schema_id', 'dataset_id'}, 'closed ordinary raw envelope')
    check(set(scaler_asset) == {'encoder_id', 'format_version', 'artifact_kind', 'schema_id',
          'preprocessing_id', 'fit_dataset_id', 'scaler'}, 'closed ordinary scaler envelope')
    for asset, kind in ((raw, 'rpb_raw_uniform_history_v1'), (scaler_asset, 'rpb_frozen_training_scaler_v1')):
        check(M.text(asset, 'encoder_id') == 'raw_patch_bottleneck_mae_v1' and
              M.scalar(asset, 'format_version') == 1 and M.text(asset, 'artifact_kind') == kind, 'typed versioned ordinary companion')
    values, mask, ids, _ = split
    M.exact(raw['data'], M.fake('DoubleStorage', [len(ids), 3, 32, 3], values), 'original TRAIN raw values')
    M.exact(raw['observed'], M.fake('BoolStorage', [len(ids), 3, 32, 3], mask), 'original TRAIN raw visibility')
    check(list(M.tensor(raw['channel_ids'], 'LongStorage', [3])) == [0, 1, 2] and
          list(M.tensor(raw['endpoints'], 'DoubleStorage', [len(ids)])) == [31.] * len(ids) and
          M.scalar(raw, 'sampling_interval', 'DoubleStorage') == 1 and M.text(raw, 'feature_units') == 'unitless,unitless,unitless',
          'raw semantic IDs, endpoints and units')
    saved_scaler = M.group(scaler_asset['scaler'])
    check(set(saved_scaler) == set(frozen), 'same full frozen scaler keys')
    for key in frozen:
        M.exact(saved_scaler[key], frozen[key], 'same frozen original TRAIN scaler ' + key)
    _, _, identity = M.scaler(saved_scaler)
    check(M.text(raw, 'schema_id') == M.text(scaler_asset, 'schema_id') and
          M.text(raw, 'dataset_id') == M.text(scaler_asset, 'fit_dataset_id') and
          M.text(scaler_asset, 'preprocessing_id') == identity, 'raw/scaler identity association')
    return M.text(raw, 'schema_id'), M.text(raw, 'dataset_id'), identity

def point_metadata(audit, binding, continuation, snapshot, point, master, ids, cp, old_cp, scopes, raw_info, file_content):
    schema_id, dataset_id, scaler_id = raw_info
    expected = dict(L.CONTEXT_LITERALS)
    expected.update(protocol_id=PROTOCOL + '/lag_sign', model_tag='RPB-v13', architecture_id=ARCHITECTURE,
        provider_id='raw_patch_bottleneck_mae_v1', adapter_id='rpb_continuous_cuda_learning_curve_v1',
        fit_policy='permitted_training_observations_only;scaler_fit_once',
        output_semantics='local_observed_visible_first_difference_contextual_pretemporal_aligned_global_semantic_mlp_bottleneck_v1',
        reconstruction_export_semantics='exact_visible_first_difference_pretemporal_contextual_observed_global_semantic_mlp_export_v1',
        temporal_difference_input='1', temporal_difference_input_semantics=INPUT_SEMANTICS,
        actual_training_seed=str(master), initialization_seed=str(M.mixed(master ^ 0x7270622d696e6974)),
        parameter_count='228877', cuda_parameter_count='228877', fit_source_manifest=M.source_manifest(ids),
        fit_source_manifest_id='rpb-fit-source-manifest-fnv1a-v1-' + M.fnv(M.source_manifest(ids).encode()),
        training_observation_rows=str(len(ids)), training_source_groups=str(len(set(ids))),
        training_dataset_id=dataset_id, scaler_fit_dataset_id=dataset_id, preprocessing_id=scaler_id,
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
        visible_difference_parent_point0_path=str(old_cp), visible_difference_parent_core_source_fingerprint=PARENT_CORE,
        visible_difference_parent_training_producer_source_fingerprint=PARENT_TRAINING,
        visible_difference_parent_fit_protocol='early-mixer-reliability-v1/lag_sign',
        visible_difference_binding_artifact_kind='rpb_visible_difference_training_binding_v1', visible_difference_binding_suffix='.visible-difference.pt')
    for asset, kind in ((audit, 'rpb_learning_curve_training_audit_v1'),
                        (binding, 'rpb_visible_difference_training_binding_v1'),
                        (continuation, 'rpb_visible_difference_continuation_state_v1')):
        check(M.text(asset, 'artifact_kind') == kind, 'exact candidate companion kind')
        check(not any(k.startswith(('source_gain_', 'pooled_')) or k == 'global_pool_input_source_value' for k in asset),
              'no gain/pooled additions in candidate metadata')
        for key, value in expected.items():
            check(M.text(asset, key) == value, 'candidate companion text association ' + key)
        check(M.scalar(asset, 'attempted_steps') == M.scalar(asset, 'completed_steps') == point and
              M.scalar(asset, 'sampled_rows') == point * 8 and M.scalar(asset, 'temporal_difference_input_value') == 1,
              'typed candidate absolute counters and input flag')
        settings_text = M.text(asset, 'resolved_settings')
        settings = settings_text.splitlines()
        check(settings.count('temporal_difference_input=1') == 1 and not any(k.startswith('temporal_difference_input=') and k != 'temporal_difference_input=1' for k in settings),
              'explicit sole difference setting')
        L.parse_settings('\n'.join(k for k in settings if not k.startswith('temporal_difference_input=')) + '\n', 1, master)
        check(M.text(asset, 'resolved_settings') == M.text(audit, 'resolved_settings'), 'identical companion settings')
    for asset in (binding, continuation):
        check(M.text(asset, 'checkpoint_path') == str(cp), 'exact point pathname association')
    check(M.text(audit, 'model_weight_update_budget') == str(point) and
          M.text(audit, 'training_device') in ('cuda', 'cuda:0') and
          M.scalar(audit, 'channel_mixer_placement_value') == M.scalar(continuation, 'channel_mixer_placement_value') == 1 and
          list(M.tensor(audit['channel_order'], 'LongStorage', [3])) == [0, 1, 2] and
          M.scalar(audit, 'sampling_interval', 'DoubleStorage') == 1 and M.scalar(audit, 'endpoint', 'DoubleStorage') == 31,
          'typed architecture/channel/time metadata')
    check(M.scalar(audit, 'context_deletion_ratio_value', 'DoubleStorage') == .15 and
          M.scalar(audit, 'context_deletion_stream_value') == 0x6374782d64726f70, 'typed unchanged coordinate15 policy')
    counts = [M.scalar(audit, k) for k in ('context_requested_deleted_coordinates', 'context_actual_deleted_coordinates', 'context_restored_coordinates')]
    check(0 <= counts[1] <= counts[0] <= point * 8 * 288 and counts[2] == counts[0] - counts[1], 'exact context count bounds')
    for key, value in zip(('context_requested_deleted_coordinates', 'context_actual_deleted_coordinates', 'context_restored_coordinates'), counts):
        check(M.scalar(continuation, key) == value, 'same live/audit context total')
    seconds = M.scalar(audit, 'training_seconds', 'DoubleStorage')
    M.finite_timer(seconds, 'candidate CUDA training loop')
    M.close(M.scalar(continuation, 'training_seconds', 'DoubleStorage'), seconds, 'same saved live/audit timer')
    check(M.scalar(audit, 'weights_changed', 'BoolStorage') == M.scalar(audit, 'finite_gradients', 'BoolStorage') == bool(point) and
          (seconds > 0 if point else seconds == 0), 'actual update and point0 training witnesses')
    for suffix in ('', '.audit.pt', '.scaler.pt', '.training-raw.pt'):
        key = 'visible_difference_parent_content_id' + suffix
        value = file_content(Path(str(old_cp) + suffix))
        for asset in (audit, binding, continuation, snapshot):
            check(M.text(asset, key) == value, 'original point0 byte association')
    for suffix in ('', '.audit.pt', '.scaler.pt', '.training-raw.pt', '.continuation.pt'):
        check(M.text(binding, 'checkpoint_content_id' + suffix) == file_content(Path(str(cp) + suffix)), 'five parent roles bind sixth sidecar')
    check(M.text(snapshot, 'artifact_kind') == 'rpb_visible_difference_cuda_snapshot_v1', 'new immutable snapshot kind')
    snapshot_expected = dict(expected)
    snapshot_expected.update(protocol_id=PROTOCOL, original_training_protocol_id=PROTOCOL + '/lag_sign',
        source_fingerprint_scope='parent=core_writer_and_training_producer;loader=new_visible_difference_adapter',
        checkpoint_path=str(cp), parent_checkpoint_path=str(cp), snapshot_loader_source_fingerprint=scopes['visible_adapter'],
        parent_writer_source_fingerprint=scopes['core_writer'], parent_training_producer_source_fingerprint=scopes['curve_training'],
        original_encoder_attempted=str(point), original_encoder_completed=str(point), training_schema_id=schema_id,
        encoder_updates='0', decoder_updates='0', head_refits='0', no_optimizer_created='true',
        snapshot_policy='immutable_CUDA_model;original_TRAIN_scaler;encoder_decoder_updates0;no_optimizer_load',
        attempted_steps=str(point), completed_steps=str(point), model_weight_update_budget=str(point))
    for key, value in snapshot_expected.items():
        check(M.text(snapshot, key) == value, 'immutable snapshot metadata ' + key)
    check(M.scalar(snapshot, 'temporal_difference_input_value') == 1 and
          M.scalar(snapshot, 'original_encoder_attempted_value') == M.scalar(snapshot, 'original_encoder_completed_value') == point and
          M.text(snapshot, 'inference_device') == 'cuda:0', 'typed immutable CUDA serving fields')
    for suffix in ('', '.audit.pt', '.scaler.pt', '.training-raw.pt', '.continuation.pt', '.visible-difference.pt'):
        check(M.text(snapshot, 'parent_content_id' + suffix) == file_content(Path(str(cp) + suffix)), 'immutable snapshot six-file content')
    return counts, seconds

def saved_heads(load, directory, data, report):
    check(set(report) == {'protocol', 'master_seed', 'recipe', 'methods', 'pairs', 'fit_counts'} and
          report['protocol'] == 'fixed-feature-readouts-v1' and len(report['methods']) == 1 and
          report['methods'][0]['method'] == 'native_visible_difference' and report['pairs'] == [], 'single native candidate method')
    master = int(report['master_seed'])
    check(report['master_seed'] == str(master) and master in MASTERS, 'exact fixed master readout seed')
    check(report['recipe'] == {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100, 'tiny_learning_rate': .01,
          'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods', 'validation_fits': 0,
          'encoder_calls': 0, 'pca_fits': 0}, 'unchanged fixed readout recipe')
    saved_method = report['methods'][0]
    measured = saved_method['status'] == 'measured'
    check(saved_method['status'] in ('measured', 'unsupported_fit') and saved_method['size'] == 32 and
          saved_method['inputs_train_prepared'] is False and saved_method['outer_train_normalizer_fits'] == int(measured), 'explicit native fit disposition')
    check(report['fit_counts'] == {'outer_train_normalizer_fits': int(measured), 'ridge_fits': 3 * int(measured), 'tiny_fits': 3 * int(measured), 'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'exact candidate-only fixed fitting counts')
    rows, valid = [], []
    for name, split in zip(('training', 'validation-intact', 'validation-deleted'), data):
        x, v = M.feature_archive(load(directory / (name + '-features.pt')), len(split[2]), 32, split[2], split[3], 'FloatStorage')
        rows.append(x); valid.append(v)
        actual_population = M.population(v, split[3], split[2])
        check(saved_method[name.replace('-', '_') + '_population'] == actual_population, 'complete saved fitting/scoring population')
    selected = [label for label, keep in zip(data[0][3], valid[0]) if keep]
    check(measured == (len(selected) >= 2 and set(selected) == {0, 1}), 'actual TRAIN support determines native fit disposition')
    check(saved_method['reason'] == ('' if measured else 'valid TRAIN fitting rows require at least 2 rows and both classes'), 'literal native unsupported reason')
    output = []
    check([r['id'] for r in saved_method['repetitions']] == ['rep-' + str(r) for r in REPS], 'all fixed repetitions')
    for rep, declared in zip(REPS, saved_method['repetitions']):
        check(declared['actual_probe_seed_decimal'] == str(M.stream_seed(rep, 32)) and
              declared['ridge_parameters'] == 66 and declared['neural_parameters'] == 562,
              'width-paired saved seed and fixed head sizes')
        if not measured:
            check(set(declared) == {'id', 'actual_probe_seed_decimal', 'status', 'ridge_parameters', 'neural_parameters'} and
                  declared['status'] == 'unsupported_fit', 'retain unsupported fit without invented scores')
            output.append({'repetition': rep, 'status': 'unsupported_fit', 'reason': saved_method['reason'], 'views': {}})
            continue
        folder = directory / ('rep-' + str(rep)); fit = load(folder / 'fit.pt')
        check(declared['status'] == 'measured' and declared['fit_artifact'] == 'native_visible_difference/rep-' + str(rep) + '/fit.pt', 'exact saved fitting artifact')
        fitted = M.fit_schema(fit, 32, data[0][2], rep, False, rows[0], valid[0])
        views = {}
        for index, (name, split) in enumerate(zip(('training', 'validation-intact', 'validation-deleted'), data)):
            prediction = load(folder / (name + '-predictions.pt'))
            ridge, tiny, _ = M.infer_saved_fit(fit, rows[index], valid[index], split[3], split[2], prediction, False, 32, fitted)
            scores = {head: M.score(classes, split[3], valid[index]) for head, classes in (('ridge', ridge), ('tiny_secondary', tiny))}
            key = name.replace('-', '_')
            check(declared[key]['population'] == M.population(valid[index], split[3], split[2]), 'every reported view population')
            for head in scores:
                M.check_saved_score(scores[head], declared[key][head])
            if index:
                seed = M.stream_seed(master, M.fnv64('native_visible_difference/rep-' + str(rep) + '/' + name))
                check(declared[key]['bootstrap_seed_decimal'] == str(seed), 'declared marginal source bootstrap seed')
                for head, classes, interval_key in (('ridge', ridge, 'ridge_grouped_interval'),
                                                    ('tiny_secondary', tiny, 'tiny_grouped_interval')):
                    interval = M.bootstrap_groups(M.accuracy_groups(classes, split[3], valid[index], split[2]), seed, 1000)
                    M.check_interval(interval, declared[key][interval_key])
            views[key] = {'scores': scores, 'ridge': ridge, 'tiny_secondary': tiny, 'valid': valid[index]}
        output.append({'repetition': rep, 'status': 'measured', 'views': views})
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--capsule'); parser.add_argument('--output'); parser.add_argument('--source-sha256'); parser.add_argument('--card-sha256')
    parser.add_argument('--inventory-sha256')
    args = parser.parse_args()
    check(Path('/.dockerenv').is_file(), 'managed container only')
    modules()
    if args.self_test:
        check(M.score([0, 1, 0, 1], [0, 1, 1, 0], [True] * 4)['accuracy'] == .5, 'literal saved score fixture')
        check(M.paired_group_effect([0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [True] * 4, [True] * 4, ['a', 'a', 'b', 'b'])['estimate'] == .5, 'whole source paired effect fixture')
        print(json.dumps({'status': 'passed', 'source_sha256': sha(Path(__file__)), 'checks': CHECKS + M.CHECKS, 'archive_reads': 0, 'module_sha256': MODULE_PINS}))
        return
    check(all((args.capsule, args.output, args.source_sha256, args.card_sha256, args.inventory_sha256)), 'explicit prospective source/card/inventory/capsule boundary')
    try:
        run(args)
    except Exception as error:
        if AUDIT_OUTPUT is not None:
            failure = {'status': 'failed', 'protocol': PROTOCOL, 'capsule': args.capsule,
                       'source_sha256': args.source_sha256, 'card_sha256': args.card_sha256,
                       'inventory_sha256': args.inventory_sha256, 'checks': CHECKS + M.CHECKS + R.CHECKS,
                       'archive_decodes': DECODES, 'error': type(error).__name__ + ': ' + str(error)}
            with (AUDIT_OUTPUT / 'validation.json').open('x', encoding='utf-8', newline='\n') as out:
                out.write(json.dumps(failure, indent=2, allow_nan=False) + '\n')
        raise

def run(args):
    global DECODES, AUDIT_OUTPUT
    started = time.monotonic()
    capsule = Path(args.capsule)
    output = Path(args.output)
    check(capsule.is_absolute() and capsule.resolve(strict=True) == capsule and capsule.parent == Path('/embedding/output/runs/rpb-visible-difference') and capsule.name.startswith('visible-difference-'), 'closed completed capsule')
    check(output.is_absolute() and output.parent.resolve(strict=True) == output.parent and
          all(not p.is_symlink() for p in (output.parent, *output.parent.parents)) and
          not output.exists() and not output.is_symlink() and output.parent in (capsule.parent, capsule.parent / 'audit-tools'),
          'new exclusive audit leaf outside capsule')
    output.mkdir()
    AUDIT_OUTPUT = output
    inventory_path = capsule / 'artifact-integrity.json'
    card = capsule / 'source/code/evaluation/cards/visible_difference_v1.md'
    metadata = [inventory_path, card, Path(__file__).resolve()]
    admit(metadata)
    inventory_sha = sha(inventory_path)
    check(inventory_sha == args.inventory_sha256 and sha(metadata[2]) == args.source_sha256 and
          sha(card) == args.card_sha256 == CARD_SHA, 'reviewed reader/card/completed inventory SOURCE pins')
    inventory = json.loads(inventory_path.read_text())
    check(inventory['status'] == 'complete' and inventory['protocol'] == PROTOCOL and
          inventory['excludes'] == ['artifact-integrity.json'], 'completed immutable inventory')
    records = inventory['files']
    names = [safe_name(r['path']) for r in records]
    check(names == sorted(set(names)) and len(names) == inventory['files_count'] and
          sum(r['bytes'] for r in records) == inventory['total_bytes'], 'unique canonical string-ordered inventory totals')
    for r in records:
        check(set(r) == {'path', 'bytes', 'sha256'} and type(r['bytes']) is int and r['bytes'] >= 0 and
              isinstance(r['sha256'], str) and len(r['sha256']) == 64 and all(c in '0123456789abcdef' for c in r['sha256']),
              'typed completed file identity')
    actual_names = []
    for path in capsule.rglob('*'):
        check(not path.is_symlink(), 'no hidden capsule redirect')
        if path.is_file() and path != inventory_path:
            actual_names.append(path.relative_to(capsule).as_posix())
    check(sorted(actual_names) == names, 'complete actual closed capsule file matrix')
    paths = [capsule / r['path'] for r in records]
    admit(paths)
    for p, r in zip(paths, records):
        check(p.stat().st_size == r['bytes'], 'complete capsule size matrix before hashes')
    before = {r['path']: sha(p) for p, r in zip(paths, records)}
    check(all(before[r['path']] == r['sha256'] for r in records), 'all completed capsule bytes')
    admitted = set(paths)
    check(before['independent-reader/validate_visible_difference.py'] == ORIGINAL_READER_SHA, 'exact prequality captured validator SOURCE')
# BEGIN query basename repair v2 capture
    check(before['independent-reader/reader-seal.json'] == ORIGINAL_READER_SEAL_SHA, 'exact unchanged prequality reader seal')
    repair = reader_repair_provenance((capsule / 'independent-reader/validate_visible_difference.py').read_bytes(),
                                    Path(__file__).resolve().read_bytes(), capsule / 'independent-reader/validate_visible_difference.py',
                                    Path(__file__).resolve(), args.source_sha256)
# END query basename repair v2 capture
    for name, pin in MODULE_PINS.items():
        check(before['independent-reader/' + name] == pin, 'prequality captured unchanged arithmetic module')
    cache = {}
    def load(path):
        global DECODES
        check(path in admitted and not (path.name == 'checkpoint.pt'), 'only admitted CPU witness, never CUDA checkpoint body')
        if path not in cache:
            cache[path] = R.load(path); DECODES += 1
        return cache[path]
    def meta(path):
        check(path in admitted, 'admitted metadata role')
        return json.loads(path.read_text())
    parent_inventory = meta(capsule / 'parent-metadata/artifact-integrity.json')
    parent_validation = meta(capsule / 'parent-metadata/parent-validation.json')
    parent_summary = meta(capsule / 'parent-metadata/parent-summary.json')
    for name, pin in PARENT_PINS.items():
        check(sha(capsule / 'parent-metadata' / name) == pin, 'original audited metadata pin')
    check(parent_validation['status'] == 'passed', 'passed retained audit')
    old_records = {r['path']: r for r in parent_inventory['files']}
    role_manifest = meta(capsule / 'source/code/evaluation/protocols/visible_difference_v1/retained-roles.json')
    capture_manifest = meta(capsule / 'retained-inputs-manifest.json')
    expected_roles = retained_role_names()
    check(role_manifest == capture_manifest and role_manifest['protocol'] == PROTOCOL and
          role_manifest['parent_inventory_sha256'] == PARENT_PINS['artifact-integrity.json'] and
          [r['path'] for r in role_manifest['files']] == expected_roles and
          role_manifest['files'] == [old_records[n] for n in expected_roles], 'exact SOURCE-declared inventory-associated retained130 matrix')
    retained = {p.relative_to(capsule / 'retained-inputs').as_posix(): p for p in paths if p.is_relative_to(capsule / 'retained-inputs')}
    check(sorted(retained) == expected_roles, 'closed retained130 actual payload names')
    for name, p in retained.items():
        check(name in old_records and p.stat().st_size == old_records[name]['bytes'] and before[str(p.relative_to(capsule))] == old_records[name]['sha256'], 'retained role matches immutable parent')
    report = meta(capsule / 'results/report.json'); complete = meta(capsule / 'results/complete.json')
    producer_fp, scopes = source_binding(capsule, meta, before, report, args.card_sha256)
    check(report['dataset_id'] == 'TEMPO-3' and report['recipe_id'] == 'structured-hard-timing-v1' and
          report['task'] == 'lag_sign' and report['parent_summary_sha256'] == PARENT_PINS['parent-summary.json'] and
          all(report[k] is False for k in ('selection', 'promotion', 'testing_accessed', 'stress_accessed')), 'unchanged dataset/no selection/promotion namespace')
    check(report['protocol'] == complete['protocol'] == PROTOCOL and complete['status'] == 'complete', 'completed candidate protocol')
    check([c['timing_master'] for c in report['cohorts']] == MASTERS, 'all five fixed cohorts')
    for key, value in {'encoder_trajectories': 5, 'encoder_updates_each': 512, 'sampled_rows': 20480, 'retained_points': 10,
                       'cohorts': 5, 'skipped_attempts': 0, 'checkpoint_roles_per_point': 6, 'planned_pipelines': 15, 'planned_heads': 30,
                       'retained_payload_roles': 130, 'initial_parity_exports': 15, 'quality_native_exports': 15, 'full_native_export_calls': 30,
                       'query_writer_calls': 10, 'necessary_query_forwards': 40, 'baseline_or_control_refits': 0, 'parent_model_forwards': 0,
                       'generator_calls': 0, 'PCA_fits': 0, 'CPU_encoder_training': False, 'CPU_encoder_forward': False,
                       'extra_decoder_calibration_updates': 0, 'selection': False, 'promotion': False, 'testing_accessed': False, 'stress_accessed': False}.items():
        check(complete[key] == value, 'exact declared workload ' + key)
    wanted_archives = {'retained-inputs/' + n for n in expected_roles}
    for master in MASTERS:
        b = f'results/seed-{master}-lag_sign/'
        for point in (0, 512):
            wanted_archives.update(b + f'candidate/point-{point}/checkpoint.pt' + s for s in
                ('', '.audit.pt', '.scaler.pt', '.training-raw.pt', '.continuation.pt', '.visible-difference.pt'))
            wanted_archives.add(b + f'candidate/point-{point}/snapshot-assets/visible-difference-snapshot-audit.pt')
        wanted_archives.update(b + 'initial-parity/' + v + '-features.pt' for v in ('training', 'validation-intact', 'validation-deleted'))
        wanted_archives.update(b + 'candidate/point-512/' + v + '-reconstruction.pt' for v in ('training', 'validation'))
        root_heads = capsule / b / 'lag_sign/readouts'
        heads = meta(root_heads / 'report.json')
        check(len(heads['methods']) == 1 and heads['methods'][0]['method'] == 'native_visible_difference' and
              heads['methods'][0]['status'] in ('measured', 'unsupported_fit'), 'single declared conditional native head matrix')
        h = b + 'lag_sign/readouts/native_visible_difference/'
        wanted_archives.update(h + v + '-features.pt' for v in ('training', 'validation-intact', 'validation-deleted'))
        if heads['methods'][0]['status'] == 'measured':
            for rep in REPS:
                wanted_archives.add(h + f'rep-{rep}/fit.pt')
                wanted_archives.update(h + f'rep-{rep}/' + v + '-predictions.pt' for v in ('training', 'validation-intact', 'validation-deleted'))
    check({n for n in names if n.endswith('.pt')} == wanted_archives, 'exact retained/new conditional archive role matrix before any decode')
    tsv = ['master\tattempted\tcompleted\tsampled_rows\trequested\tactual\trestored\ttarget_cells']
    head_pipelines = outer_fits = 0
    cohorts = []
    for cohort in report['cohorts']:
        master = cohort['timing_master']; base = f'results/seed-{master}-lag_sign'
        old = capsule / 'retained-inputs' / base; root = capsule / base
        data = [controlled(load(old / name), n) for name, n in [('controlled-training.pt', 256), ('controlled-validation.pt', 128), ('controlled-validation-deleted.pt', 128)]]
        check(data[1][2:] == data[2][2:] and not set(data[0][2]) & set(data[1][2]), 'original source split/validation association')
        erased = list(M.tensor(load(old / 'controlled-validation-deleted.pt')['requested_erasure'], 'BoolStorage', [128, 3, 32, 3]))
        values, masks = M.deletion_view(data[1][0], data[1][1], erased)
        check(list(values) == list(data[2][0]) and masks == data[2][1], 'exact retained deletion values/support')
        initial = load(old / 'initial-pair.pt')
        old_cp = old / 'early/point-0/checkpoint.pt'
        old_raw = load(Path(str(old_cp) + '.training-raw.pt'))
        old_scaler = load(Path(str(old_cp) + '.scaler.pt'))
        original_info = L.replay_parent(load(Path(str(old_cp) + '.audit.pt')), old_raw, old_scaler, data[0], master, 1, 0,
                                       {'core_writer': PARENT_CORE, 'curve_training': PARENT_TRAINING})
        parent_cohort = next(c for c in parent_summary['cohorts'] if c['timing_master'] == master)
        parent_point = next(p for p in parent_cohort['encoder_points'] if p['model_tag'] == 'RPB-v10.alt-05')['encoder_progress']
        original = next(c for c in parent_validation['per_master'] if c['timing_master'] == master)
        parent_encoder = next(e for e in original['encoders'] if e['role'] == 'early')
        targets = [v[2] for v in parent_point['losses']]
        check(len(targets) == 512 and parent_point['attempted'] == parent_point['completed'] == 512 and
              parent_point['sampled_rows'] == 4096 and parent_encoder['completed'] == parent_encoder['attempted'] == 512 and
              parent_encoder['sampled_rows'] == 4096, 'exact unchanged original completed prefix metadata')
        tsv.append('\t'.join(map(str, [master, parent_point['attempted'], parent_point['completed'], parent_point['sampled_rows'], *parent_encoder['context_counts']])) + '\t' + ','.join(map(str, targets)))
        point_states = {}
        for point in (0, 512):
            cp = root / f'candidate/point-{point}/checkpoint.pt'
            continuation = load(Path(str(cp) + '.continuation.pt'))
            frozen, counters, losses, params, buffers = state(continuation, point, initial)
            audit = load(Path(str(cp) + '.audit.pt'))
            binding = load(Path(str(cp) + '.visible-difference.pt'))
            snapshot = load(cp.parent / 'snapshot-assets/visible-difference-snapshot-audit.pt')
            L.same_named(named(snapshot['model_parameters'], 228877), params, 'snapshot captured parameters equal live saved point')
            L.same_named(named(snapshot['model_buffers']), buffers, 'snapshot captured buffers equal live saved point')
            for key in frozen:
                M.exact(M.group(snapshot['scaler'])[key], frozen[key], 'snapshot frozen TRAIN scaler')
            scaler_asset = load(Path(str(cp) + '.scaler.pt'))
            raw = load(Path(str(cp) + '.training-raw.pt'))
            raw_info = raw_scaler_envelope(raw, scaler_asset, data[0], frozen)
            check(raw_info == (original_info['schema_id'], original_info['dataset_id'], original_info['scaler_id']), 'same original schema/data/scaler identities')
            check(set(raw) == set(old_raw) and set(scaler_asset) == set(old_scaler), 'exact ordinary companion keys')
            for key in raw:
                M.exact(raw[key], old_raw[key], 'same original raw companion field ' + key)
            def content_id(path):
                check(path in admitted, 'bound admitted file bytes only')
                return 'fnv1a64-runtime-content-v1-' + M.fnv(path.read_bytes())
            counts, seconds = point_metadata(audit, binding, continuation, snapshot, point, master, data[0][2], cp, old_cp, scopes, raw_info, content_id)
            check(counts == ([0, 0, 0] if point == 0 else parent_encoder['context_counts']), 'same cumulative original erasure counts')
            if point:
                check([counters[3 * i + 2] for i in range(point)] == targets, 'same every-step target-cell count')
            point_states[point] = frozen
            if point:
                point_audit = audit
                point_seconds = seconds
        for name, split in zip(('training', 'validation-intact', 'validation-deleted'), data):
            parity(load(root / 'initial-parity' / (name + '-features.pt')), load(old / 'lag_sign/readouts/untrained_early' / (name + '-features.pt')), split)
        queries = {}
        for name, split in zip(('training', 'validation'), data[:2]):
            p = root / f'candidate/point-512/{name}-reconstruction.pt'; asset = load(p)
            original_query = load(old / f'early/point-512/{name}-reconstruction.pt')
            for key in ('standardized_target', 'target_mask', 'requested_observed_target_mask', 'visible_mask', 'trial_channel_eligible', 'channel_target_counts', 'channel_valid', 'example_valid', 'source_ids_json'):
                M.exact(asset[key], original_query[key], 'same original query target/support/source ' + key)
            replay = L.query_archive(asset, split, point_states[512])
            declared = cohort['encoder_points'][0][name + '_reconstruction']
            L.query_summary(declared, replay, asset, p.name)
            queries[name] = {k: replay[k] for k in ('mae', 'huber', 'valid_examples', 'total_examples', 'valid_target_cells', 'requested_observed_target_cells')}
        readouts = root / 'lag_sign/readouts'; head_report = meta(readouts / 'report.json')
        check(head_report == cohort['tasks'][0]['readouts'] and len(cohort['tasks']) == 1 and
              cohort['tasks'][0]['task'] == 'lag_sign' and cohort['tasks'][0]['data_master'] == master and
              head_report['master_seed'] == str(master), 'same reported whole single-task readout record')
        head_data = saved_heads(load, readouts / 'native_visible_difference', data, head_report)
        head_pipelines += head_report['fit_counts']['ridge_fits']
        outer_fits += head_report['fit_counts']['outer_train_normalizer_fits']
        check(meta(readouts / 'native_visible_difference/fit-counts.json') ==
              {k: head_report['fit_counts'][k] for k in ('outer_train_normalizer_fits', 'ridge_fits', 'tiny_fits', 'validation_fits')},
              'exact per-method fitting count receipt')
        paired = meta(root / 'lag_sign/reused-paired-comparisons.json')
        check(paired == cohort['tasks'][0]['reused_paired_comparisons'] and
              [(p['reference'], p['repetition'], p['view']) for p in paired] ==
              [(c, 'rep-' + str(r), v) for c in ('native_early', 'native_late') for r in REPS for v in ('validation_intact', 'validation_deleted')],
              'all unique two controls/three reps/two validation views in literal order')
        for item in paired:
            rep = int(item['repetition'][4:]); view = item['view']; control = item['reference']
            check(item['id'] == 'native_visible_difference_minus_' + control and item['candidate'] == 'native_visible_difference', 'exact paired method identity')
            candidate = next(r for r in head_data if r['repetition'] == rep)
            if candidate['status'] != 'measured':
                check(set(item) == {'id', 'reference', 'candidate', 'repetition', 'view', 'status'} and
                      item['status'] == 'unsupported_fit', 'retain unsupported paired comparison without invented effect')
                continue
            check(control in ('native_early', 'native_late') and view in ('validation_intact', 'validation_deleted'), 'closed retained comparison role')
            split = data[1 if view == 'validation_intact' else 2]
            old_prediction = load(old / f'lag_sign/readouts/{control}/rep-{rep}/{view.replace("_", "-")}-predictions.pt')
            check(set(old_prediction) == {'ridge', 'tiny_secondary', 'valid', 'probe_input_features', 'ridge_logits',
                  'tiny_hidden_preactivation', 'tiny_logits', 'labels_scoring_only', 'source_ids_json'}, 'literal retained fixed prediction schema')
            check(json.loads(M.text(old_prediction, 'source_ids_json')) == split[2] and list(M.tensor(old_prediction['labels_scoring_only'], 'LongStorage', [128])) == split[3], 'retained scoring association')
            old_valid = list(M.tensor(old_prediction['valid'], 'BoolStorage', [128]))
            current = candidate['views'][view]
            common = sum(bool(a and b) for a, b in zip(current['valid'], old_valid))
            check(item['total_rows'] == 128 and item['common_valid_rows'] == common and
                  item['status'] == ('measured' if common else 'unsupported_zero_common'), 'exact paired common population/status')
            seed = M.stream_seed(master, M.fnv64(item['id'] + '/' + item['repetition'] + '/' + view))
            check(item['bootstrap_seed_decimal'] == str(seed), 'unchanged declared paired seed policy')
            for head, key in [('ridge', 'ridge_logits'), ('tiny_secondary', 'tiny_logits')]:
                logits = M.tensor(old_prediction[key], 'DoubleStorage', [128, 2]); classes = list(M.tensor(old_prediction[head], 'LongStorage', [128]))
                M.finite(logits, 'finite retained own logits')
                check(classes == [int(logits[2 * i + 1] > logits[2 * i]) for i in range(128)], 'retained exact own-logit classes')
                effect = M.paired_group_effect(current[head], classes, split[3], current['valid'], old_valid, split[2])
                interval = M.bootstrap_groups(effect['groups'], seed, 1000)
                M.check_interval(interval, item[head])
        p = cohort['encoder_points'][0]['encoder_progress']
        check(len(cohort['encoder_points']) == 1 and cohort['encoder_points'][0]['model_tag'] == 'RPB-v13' and
              cohort['encoder_points'][0]['placement'] == cohort['encoder_points'][0]['temporal_difference_input'] == 1,
              'one full-budget candidate encoder point')
        check(p['attempted'] == p['completed'] == 512 and p['sampled_rows'] == 4096 and p['parameter_count'] == p['cuda_parameter_count'] == 228877 and
              all(p[k] is True for k in ('last_input_cuda', 'last_loss_cuda', 'finite_gradients', 'weights_changed')) and
              p['training_device'] in ('cuda', 'cuda:0') and len(p['losses']) == 512 and
              p['training_dataset_id'] == M.text(point_audit, 'training_dataset_id') and
              p['preprocessing_id'] == M.text(point_audit, 'preprocessing_id'), 'actual CUDA training evidence')
        M.close(p['training_seconds'], point_seconds, 'reported saved training-loop timer')
        for i, row in enumerate(p['losses']):
            check(row[:3] == [i + 1, i + 1, targets[i]], 'reported full trace counters')
            M.close(row[3], losses[2 * i], 'reported original objective trace'); M.close(row[4], losses[2 * i + 1], 'reported gradient norm trace')
        check(cohort['initial_parity'] == {'exports': 3, 'exact_saved_untrained_early_features': True, 'before_heads': True}, 'per-cohort declared initial parity')
        check(set(cohort['costs']) == {'parent_archive_verification_and_load_seconds', 'binding_checkpoint_and_snapshot_IO_seconds',
              'CUDA_initial_parity_transfer_and_verification_seconds', 'CUDA_query_transfer_verification_IO_seconds',
              'CUDA_quality_native_transfer_seconds', 'CPU_candidate_heads_and_paired_bootstrap_IO_seconds'}, 'complete mixed-cost scopes')
        for scope, seconds in cohort['costs'].items():
            M.finite_timer(seconds, scope)
        compact_heads = [{'repetition': r['repetition'], 'status': r['status'], 'views': {v: x['scores'] for v, x in r['views'].items()}} for r in head_data]
        cohorts.append({'timing_master': master, 'model_tag': 'RPB-v13', 'query': queries, 'readouts': compact_heads,
                        'paired': paired, 'training_seconds': p['training_seconds'], 'initialization_exact': True,
                        'encoder_progress': p, 'costs': cohort['costs']})
    check(complete['head_pipelines'] == head_pipelines <= 15 and complete['individual_heads'] == 2 * head_pipelines and
          complete['helper_outer_train_fits'] == outer_fits <= 5, 'all five actual supported fitting counts')
    check((capsule / 'parent-progress.tsv').read_text() == '\n'.join(tsv) + '\n', 'independently rederived parent counter TSV')
    check(all(sha(p) == before[r['path']] for p, r in zip(paths, records)), 'all saved capsule bytes preserved after arithmetic')
    check(sha(inventory_path) == inventory_sha, 'exact original inventory bytes preserved')
    check(sha(Path(__file__).resolve()) == args.source_sha256 and all(sha(Path(__file__).resolve().parent / n) == pin for n, pin in MODULE_PINS.items()),
          'all executing reader SOURCE bytes preserved')
    result = {'status': 'passed', 'protocol': PROTOCOL, 'source_sha256': args.source_sha256, 'card_sha256': args.card_sha256,
              'capsule': str(capsule), 'inventory_sha256': inventory_sha, 'checks': CHECKS + M.CHECKS + R.CHECKS + L.CHECKS,
              'archive_decodes': DECODES, 'per_master': cohorts, 'elapsed_seconds': time.monotonic() - started,
              'complete': complete,
              'source': {'reader_sha256': args.source_sha256, 'source_fingerprint': producer_fp, 'actual_compile_scopes': scopes,
                         'card_sha256': args.card_sha256, 'inventory_sha256': inventory_sha, 'modules': MODULE_PINS},
              'validated_metadata': {'report_sha256': before['results/report.json'], 'complete_sha256': before['results/complete.json'],
                                     'parent_summary_sha256': PARENT_PINS['parent-summary.json']},
              'limits': {'model_forward': False, 'encoder_updates': 0, 'head_refits': 0, 'PCA_fits': 0, 'CUDA_checkpoint_body_decodes': 0,
                         'all_inputs_preserved': True, 'TEST_or_stress': False, 'promotion': False}}
# BEGIN query basename repair v2 result
    result['source']['reader_repair'] = repair
# END query basename repair v2 result
    with (output / 'validation.json').open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'checks', 'archive_decodes', 'elapsed_seconds')}))

if __name__ == '__main__':
    main()

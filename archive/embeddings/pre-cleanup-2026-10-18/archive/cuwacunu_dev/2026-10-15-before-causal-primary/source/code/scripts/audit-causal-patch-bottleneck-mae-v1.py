#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Validate saved CPB-v1 engineering receipts; no tensor/model loads or fits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import struct
import time

ROOT = Path('/cuwacunu')
VERSION = Path('code/encoders/causal_patch_bottleneck_mae/versions/v1')
APP = Path('code/evaluation/causal_patch_bottleneck_mae_v1')
BUILD = Path('/opt/cuwacunu/build/causal-patch-v1')
SOURCE = 'c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb'
CARD = '3b059b84a60a7d193ffcb034480f54fa5aba532ed6cb6b541fa620c4ab584d5f'
ADDENDUM = 'ae0a8adae317a7b2daee3f58a645c9000faab130cde17ba6662a66fae06047dd'
POLICY = 'crop-before-validation;rolling-last-H;masked-left-pad;original-patch-end-causal-attention;prefix-only-banks-pooling-v1'
BANK = 'rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476'
SCALER = 'rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969'
OLD_RECEIPT = Path('output/engineering/location-scale-transfer-build/attempt-01/build-receipt.json')
OLD_RECEIPT_SHA = 'd83285807756316b49ba1278de81fa9e4333b3d8c31553c7058f1bd6472f8111'
DEPS = (
    'include/embedding/shared/types.h', 'include/embedding/shared/data.h',
    'include/embedding/shared/evaluation.h', 'include/embedding/shared/feature_harness.h',
    'include/embedding/shared/fixed_feature_readouts.h', 'src/data.cpp',
    'src/feature_harness.cpp', 'src/fixed_feature_readouts.cpp',
)
CHECKS = 0


def require(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = resolve(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def verify(item):
    path = resolve(item['path'])
    require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
            'Preserved binding: ' + str(path))
    return path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fnv(text):
    return isinstance(text, str) and re.fullmatch(r'[0-9a-f]{16}', text) is not None


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def dependency_paths(path):
    text = path.read_text().replace('\\\n', ' ')
    require(': ' in text, 'Make dependency target syntax')
    result = set()
    for word in shlex.split(text.split(': ', 1)[1]):
        if word.endswith(':'):
            continue
        candidate = resolve(word).resolve()
        if candidate.is_relative_to(ROOT):
            result.add(candidate)
    return result


def admission(leaf):
    value = load(leaf / 'admission.json')
    require(value['kind'] == 'causal_patch_bottleneck_mae_v1_build_admission_v1' and
            value['status'] == 'built_pending_source_review_and_cuda_proof' and value['encoder_updates'] == 0,
            'Original build admission before quality')
    source = {}
    capsule = resolve(value['source_capsule'])
    require(capsule.resolve().is_relative_to(leaf), 'Compact capsule inside this engineering attempt')
    for item in value['source']:
        path = verify(item)
        require(path.is_relative_to(ROOT), 'Source is repository-local')
        source[path] = item
        saved = capsule / path.relative_to(ROOT)
        require(saved.is_file() and saved.stat().st_size == item['bytes'] and sha(saved) == item['sha256'],
                'Actual admitted source capsule matches current file')
    manifest = load(leaf / 'source-manifest.json')
    require(manifest == value['source'], 'Build manifest/admission same exact source list')
    for item in value['pins']:
        verify(item)
    for item in value['binaries'] + value['reused_fixed_head_objects']:
        verify(item)
    binary = BUILD / 'test_causal_patch_bottleneck_mae_v1'
    require(any(resolve(item['path']) == binary for item in value['binaries']), 'Admitted actual fixture binary')
    deps = value['dependencies']
    require(set(deps) == {'core', 'app', 'reader'}, 'Three separate compiler dependency closures')
    core = set()
    for item in deps['core']:
        path = verify(item)
        require(path.is_relative_to(ROOT / VERSION), 'Standalone core depends only on its physical version')
        require(path in source, 'Every core compiler dependency captured')
        core.add(path)
    require(ROOT / VERSION / 'tests/v1_test.cpp' in core and
            ROOT / VERSION / 'include/embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h' in core,
            'Fixture includes one public version entry')
    require(len([p for p in core if p.suffix == '.h']) == 42, 'Complete independently owned42-header closure')
    require(dependency_paths(BUILD / 'causal_patch_bottleneck_mae_v1_test.d') == core,
            'Actual compiler dependency file equals admitted standalone closure')
    for kind in ('app', 'reader'):
        for item in deps[kind]:
            path = verify(item)
            allowed = path.is_relative_to(ROOT / APP)
            if kind == 'app':
                allowed |= path.is_relative_to(ROOT / VERSION) or path.is_relative_to(ROOT / 'code/data/include')
            else:
                allowed |= path.is_relative_to(ROOT / 'code/evaluation/market_synthetic_transfer_v1/deps')
            require(allowed, 'App/head closure boundary: ' + str(path))
    review = load(leaf / 'source-review.json')
    require(review['status'] == 'passed', 'Independent pre-GPU static review')
    tools = load(leaf / 'quality-tools-admission.json')
    require(tools['status'] == 'passed' and isinstance(tools['sources'], list), 'Pre-quality saved-output tools admission')
    tool_paths = {verify(item) for item in tools['sources']}
    require(Path(__file__).resolve() in tool_paths and
            ROOT / 'code/scripts/report-causal-patch-bottleneck-mae-v1.py' in tool_paths,
            'Exact auditor and reporter source bytes admitted')
    require(sha(ROOT / 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_CARD.md') == CARD and
            sha(ROOT / 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_LAYOUT_ADDENDUM.md') == ADDENDUM,
            'Prospective card and physical-version layout binding')
    return value, core, tool_paths


def fixed_heads(value, tool_paths):
    require(sha(ROOT / OLD_RECEIPT) == OLD_RECEIPT_SHA, 'Preserved V30 transfer classifier build receipt')
    old = load(ROOT / OLD_RECEIPT)
    require(old['status'] == 'passed' and old['exit_code'] == 0 and old['warnings'] == 0, 'Original warning-free fixed classifier build')
    previous = old['compiled_repository_dependencies']
    root = ROOT / 'code/evaluation/market_synthetic_transfer_v1/deps'
    all_paths = set()
    for name in DEPS:
        path = root / name
        require(str(path) in previous, 'All eight original classifier dependencies explicitly admitted')
        verify(dict(path=str(path), **previous[str(path)]))
        require(path in tool_paths, 'Old fixed-head source prospectively protected')
        all_paths.add(path)
    union = set()
    for item in value['reused_fixed_head_objects']:
        path = verify(item)
        require(path.name in {'market_location_scale_synthetic_transfer_data.o',
                              'market_location_scale_synthetic_transfer_feature_harness.o',
                              'market_location_scale_synthetic_transfer_fixed_feature_readouts.o'},
                'Only the three unchanged inherited classifier objects reused')
        dep = path.with_suffix('.d')
        require(dep in tool_paths, 'Actual inherited object dependency file prospectively protected')
        dependencies = dependency_paths(dep)
        require(dependencies and dependencies <= all_paths, 'Inherited object closure contains only the eight fixed dependencies')
        union.update(dependencies)
    require(union == all_paths and len(value['reused_fixed_head_objects']) == 3,
            'Three reused objects cover exactly all eight unchanged dependencies')
    old_reader = ROOT / 'code/evaluation/market_location_scale_synthetic_transfer_v1/read_main.cpp'
    require(sha(old_reader) == '37556b80d8dac3b8f387a9f70e1e8a46cc90be8630297d88b0c10a0e57b9c1dd',
            'Original metadata adapter remains frozen')
    expected = old_reader.read_text().replace('RPB-v30.market-01', 'CPB-v1.market-01')
    expected = expected.replace('market-location-scale-synthetic-transfer-v1', 'causal-patch-bottleneck-mae-v1')
    expected = expected.replace('frozen-market-v30-native36-on-tempo5-v1', 'frozen-cpb-v1-native36-on-tempo5-v1')
    expected = expected.replace('market_location_scale_synthetic_transfer_read', 'causal_patch_bottleneck_mae_v1_read')
    require((ROOT / APP / 'read_main.cpp').read_text() == expected, 'Reader differs only by four declared metadata substitutions')
    return len(all_paths)


def proof(leaf):
    stage = load(leaf / 'proof-stage.json')
    require(stage['status'] == 'passed', 'Completed pre-quality GPU proof stage')
    path = verify(stage['proof'])
    checkpoint = verify(stage['causal_checkpoint'])
    require(path == leaf / 'proof/proof.json' and checkpoint == leaf / 'proof/inference.pt', 'Single causal proof and inference artifact')
    p = load(path)
    require(p['kind'] == 'causal_patch_bottleneck_mae_v1_cuda_proof_v1' and p['status'] == 'passed' and
            p['model'] == 'CPB-v1' and p['source_checkpoint_sha256'] == SOURCE and p['causal_policy'] == POLICY,
            'Exact proof family/source/causal policy')
    for name in ('causal_updates', 'encoder_updates', 'CPU_encoder_calls', 'head_fits', 'decoder_calls'):
        require(p[name] == 0, 'Forbidden proof work: ' + name)
    require(p['source_updates'] == 2048 and p['device'] == 'cuda:0' and p['model_dtype'] == 'float32', 'Frozen actual GPU source state')
    for name in ('weights_imported_exact', 'typed_save_load_exact', 'ordinary_loader_rejects_RPB',
                 'future_values_and_masks_invariant', 'observed_future_NaN_Inf_overflow_ignored',
                 'hidden_prefix_poison_ignored', 'observed_prefix_poison_rejected', 'both_temporal_paths_causal',
                 'legal_past_value_positive_control', 'legal_past_mask_positive_control', 'all_state_exact', 'undefined_gradients'):
        require(p[name] is True, 'Mandatory correctness proof: ' + name)
    require(p['all_cutoffs_tested'] == 128 and p['cutoff_indices'] == list(range(128)) and
            p['suffix_invariance_cases'] == 2350 and p['suffix_intervention_variants'] == 9 and p['suffix_layouts'] == 2 and
            p['rolling_history_length_tested'] == 193 and p['mixed_cutoff_plans'] == 5 and
            p['internal_temporal_cases'] == 30 and p['bounded_witness_cases'] == 20 and p['typed_negative_cases'] == 14,
            'Exhaustive declared cutoff/suffix/internal/typed proof counts')
    require(p['source_parameter_tensors_exact'] == 76 and p['source_parameter_count_exact'] == 297560 and
            p['source_tensor_leaves_exact'] >= 87 and p['checks'] > 10000 and p['retained_cells_checked'] > 0,
            'Nontrivial complete imported state and assertion population')
    require(p['CUDA_encoder_calls'] == 3687 and p['public_API_attempts'] == 3700,
            'Exact successful proof calls and13 rejected raw API attempts; no training')
    for name in ('import_seconds_including_state_and_io', 'proof_seconds_including_state_checks',
                 'artifact_and_typed_check_seconds'):
        require(finite(p[name]) and p[name] > 0, 'Separated measured proof costs')
    for index in range(1, 11):
        require((leaf / 'proof' / f'refusal-{index}.pt').is_file(), 'Preserved malformed typed refusal artifact')
    return p


def imported_and_frozen(leaf, p):
    imported = load(leaf / 'proof/import-parity.json')
    require(imported['kind'] == 'cpb_v1_source_import_parity_v1' and imported['source_checkpoint_sha256'] == SOURCE and
            imported['all_parameters_buffers_scaler_normalizer_exact'] is True and
            imported['source_tensor_leaves'] == p['source_tensor_leaves_exact'], 'Source-data import parity receipt')
    params = imported['parameter_tensors']
    require(len(params) == 76 and len({v['name'] for v in params}) == 76, 'All76 imported parameter leaves')
    for value in params:
        require(value['name'].startswith('core.') and value['source_key'] == 'model.' + value['name'][5:] and
                value['category'] == 'parameter' and fnv(value['content_fnv1a64']), 'Exact one-prefix source parameter mapping')
    state = load(leaf / 'proof/frozen-state.json')
    require(state['kind'] == 'cpb_v1_frozen_state_v1' and state['bitwise_exact'] is True and
            state['before'] == state['after'], 'All registered/unregistered state exactly unchanged')
    before = state['before']
    require(before['parameter_count'] == 297560 and before['normalizer_metadata'].startswith(BANK + ';') and
            SCALER in before['normalizer_metadata'] and before['scaler_metadata'].split(';')[0] == SCALER and
            float(before['scaler_metadata'].split(';')[-1]) == 1e-6, 'Frozen normalizer/scaler recipe metadata')
    tensors = before['tensors']
    require(len(tensors) == 87 and len({v['name'] for v in tensors}) == 87, '76parameters+6unregistered-bank+5scaler tensor states')
    expected = {v['name']: v for v in params}
    total = 0
    for value in tensors:
        require(isinstance(value['requires_grad'], bool) and value['gradient_defined'] is False and
                isinstance(value['shape'], list) and all(isinstance(n, int) and n > 0 for n in value['shape']) and
                fnv(value['content_fnv1a64']), 'Immutable shape/gradient/tensor-byte fingerprint')
        name = value['name']
        if name.startswith('parameter/'):
            key = name[len('parameter/'):]
            require(key in expected and value['device'] == 'cuda:0' and value['dtype'] == 'Float' and
                    value['content_fnv1a64'] == expected[key]['content_fnv1a64'], 'Actual CUDA imported parameter unchanged')
            total += math.prod(value['shape'])
        else:
            require(name.startswith(('normalizer/', 'scaler/')) and value['device'] == 'cpu' and
                    value['requires_grad'] is False and value['dtype'] in ('Double', 'Long', 'Bool'),
                    'Complete unregistered CPU frozen constants')
    require(total == 297560, 'All imported parameter coordinates counted')


def observed(b, channel, t, f):
    valid = not (17 <= t <= 23 or 61 <= t <= 70 or (t + 3 * f + channel) % 23 == 0)
    if b == 1 and channel == 2:
        valid = False
    if b == 2 and t < 81:
        valid = False
    return valid


def witnesses(leaf):
    value = load(leaf / 'proof/proof-witnesses.json')
    require(value['kind'] == 'cpb_v1_bounded_causal_witnesses_v1' and len(value['cases']) == 20, 'Bounded witnesses only')
    indices = {0, 1, 7, 8, 15, 16, 63, 64, 126, 127}
    seen = set()
    for case in value['cases']:
        t, layout = case['cutoff_index'], case['permuted']
        require(t in indices and isinstance(layout, bool) and (t, layout) not in seen, 'Unique declared bounded cutoff/layout')
        seen.add((t, layout))
        ids = [2, 0, 1] if layout else [0, 1, 2]
        cutoff = t * 60000
        require(case['channel_ids'] == ids and case['cutoffs'] == [cutoff] * 3, 'Exact semantic/time metadata')
        require(case['retained_timestamps'] == [[cutoff - (127 - i) * 60000 for i in range(128)]] * 3,
                'Rolling fixedH timestamps including unavailable left-pad')
        support_count, expected_positions, valid_rows = 0, [], []
        for b in range(3):
            valid_rows.append(any(observed(b, channel, time_index, f)
                                  for channel in ids for time_index in range(t + 1) for f in range(9)))
            for channel in ids:
                slots = set()
                for time_index in range(t + 1):
                    for f in range(9):
                        if observed(b, channel, time_index, f):
                            support_count += 1
                            slots.add((127 - t + time_index) // 8)
                if slots:
                    expected_positions.append(sorted(slots))
        maximum = max(map(len, expected_positions), default=0)
        positions = [row + [-1] * (maximum - len(row)) for row in expected_positions]
        masks = [[slot != -1 for slot in row] for row in positions]
        require(case['retained_observed_count'] == support_count and case['patch_positions'] == positions and
                case['patch_valid'] == masks, 'Independent scalar support/packing/left-pad oracle')
        times = [[cutoff - (128 - (slot + 1) * 8) * 60000 if slot >= 0 else 0 for slot in row] for row in positions]
        require(case['token_end_times'] == times and all(end <= cutoff for row, mask in zip(times, masks)
                                                       for end, active in zip(row, mask) if active), 'Every active token ends by its cutoff')
        require(case['valid'] == valid_rows and len(case['native']) == 3, 'Prefix-only sample validity')
        for row, valid in zip(case['native'], valid_rows):
            require(len(row) == 36 and all(finite(x) and f32(x) == x for x in row), 'Finite actual CUDA-F32 native witness')
            require(valid or all(x == 0 for x in row), 'Empty prefix witness native exact0')
        for name in ('content_values_fnv1a64', 'spectral_values_fnv1a64', 'change_values_fnv1a64'):
            require(fnv(case[name]), 'Actual bank trace fingerprint')
        require(len(case['observed_raw_samples']) == 27, 'Bounded original-raw samples')
        for cell in case['observed_raw_samples']:
            b, c, f = cell['b'], cell['c'], cell['f']
            require(b in range(3) and c in range(3) and f in (0, 3, 8), 'Raw scalar witness coordinates')
            channel = ids[c]
            raw = (10. + channel * 3. + f * .4 + .01 * t +
                   .8 * math.sin(.11 * t + .21 * channel + .17 * f + .23 * b) +
                   .2 * math.cos(.73 * t + .31 * f))
            require(finite(cell['raw64']) and abs(cell['raw64'] - raw) <= 1e-12 and
                    cell['observed'] is observed(b, channel, t, f), 'Independent artificial raw value/mask oracle')
    require(seen == {(t, layout) for t in indices for layout in (False, True)}, 'All20 bounded causal witness cases')


def main():
    started = time.monotonic()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Run only inside existing cuwacunu container')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engineering', required=True, type=Path)
    args = parser.parse_args()
    leaf = args.engineering.resolve(strict=True)
    require(leaf.is_relative_to(ROOT / 'output/engineering/causal-patch-bottleneck-mae-v1'), 'Declared engineering attempt')
    output = leaf / 'independent-audit.json'
    require(not output.exists(), 'Exclusive audit receipt; preserve every prior attempt')
    admitted, core, tools = admission(leaf)
    fixed_count = fixed_heads(admitted, tools)
    p = proof(leaf)
    imported_and_frozen(leaf, p)
    witnesses(leaf)
    proof_files = [binding(path) for path in sorted((leaf / 'proof').rglob('*')) if path.is_file()]
    evidence = [binding(leaf / name) for name in ('admission.json', 'source-manifest.json', 'source-review.json',
                'quality-tools-admission.json', 'proof-stage.json', 'proof.log')]
    result = dict(kind='causal_patch_bottleneck_mae_v1_independent_saved_receipt_audit_v1', status='passed',
                  model='CPB-v1', card_sha256=CARD, layout_addendum_sha256=ADDENDUM,
                  source_checkpoint_sha256=SOURCE, engineering=str(leaf), checks=CHECKS,
                  causal_policy=POLICY, core_dependency_files=len(core), independent_first_party_headers=42,
                  fixed_head_dependencies_verified=fixed_count, source_parameter_tensors_exact=76,
                  source_parameter_count_exact=297560, all_cutoffs_tested=128, suffix_invariance_cases=2350,
                  internal_temporal_cases=30, suffix_layouts=2, bounded_witness_cases=20,
                  encoder_updates=0, auditor_encoder_calls=0, auditor_head_fits=0, tensor_archives_deserialized=0,
                  proof_files=proof_files, evidence=evidence, script=binding(Path(__file__)),
                  seconds=time.monotonic() - started,
                  scope='Saved admission/source/compile bindings and recorded CUDA proof plus independent scalar bounded-witness checks; no model-output or causal-invariance recomputation')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('CPB_V1_INDEPENDENT_RECEIPT_AUDIT_PASSED checks=' + str(CHECKS), flush=True)


if __name__ == '__main__':
    main()

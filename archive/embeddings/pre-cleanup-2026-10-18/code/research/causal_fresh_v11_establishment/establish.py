# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Owned full final V10 copy: normal V11 JSON/hash evidence join; no model or producer imports."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

TAG, BASE = 'CPB-v11.market-01', 'CPB-v3.market-01'
# Actual scientific identities come only from the externally reviewed join card.


POLICY = 'original-position-four-bin-half-mixture-mean4-centered-variance4-v1'
PARTITION = [4, 16, 32, 4, 8]
MOMENT_POLICY = 'componentwise-centered-variance-two-sum-mean-live-attention-v1'
MOMENT_LAYOUT = 'per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1'
MOMENT_PARTITION = [4, 4]
GRADIENT = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
OBJECTIVE = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
ROUTES = 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
GAUGE = 'core.encoder.level.base.backbone.pool_score.bias'
INPUT_IDS = frozenset(('comparison', 'table_rows', 'quality_audit', 'training_audit',
                      'complete_mirror', 'card', 'quality_receipt', 'pairings',
                      'source_manifest', 'build_admission', 'temporal_pooling_proof', 'temporal_learning',
                      'prospective_freeze', 'protocol_preflight'))
OPTIONAL_IDS = frozenset(('declaration_review',))
TRAIN_MODULES = ('numeric_statistics', 'saved_tensor_state', 'manual_saved_Gaussian',
                 'actual_pre_step_composite_objective', 'temporal_pooling_saved_learning')
PENDING = {'initial_native20_40_and_statistics_pairings': 'pending',
           'independent_training_audit': 'pending', 'independent_quality_audit': 'pending'}
BINARIES = frozenset(('test_causal_patch_bottleneck_mae_v11', 'test_cpb_v11_variance_gradient',
                      'test_cpb_v11_temporal_pooling', 'causal_fresh_v11_train',
                      'causal_fresh_v11_extract', 'causal_fresh_v11_market_read', 'causal_fresh_v11_synthetic_read'))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def decode(data):
    return json.loads(data.decode('utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def matches(left, right):
    """Artifact identity, allowing separate staged and durable path labels."""
    return left['bytes'] == right['bytes'] and left['sha256'] == right['sha256']


def positive_count(value, name):
    require(type(value) is int and value > 0, 'Positive saved integer count: '+name)
    return value


def audited_scalar_identity(left, right):
    """Match the existing auditor tolerance, not Torch/Python reduction bit order."""
    return (type(left) in (int, float) and type(right) in (int, float) and
            math.isfinite(left) and math.isfinite(right) and
            abs(left-right) <= 2e-12 + 2e-12*abs(right))


def temporal(record, name):
    require(record['temporal_pooling_policy'] == POLICY and
            record['temporal_pooling_partition'] == PARTITION and
            record['temporal_pooling_mix'] == [0.5, 0.5] and
            record['empty_local_bin_policy'] == 'zero-whole-export-slice-after-bias-v1' and
            record['temporal_moment_policy'] == MOMENT_POLICY and
            record['temporal_export_layout'] == MOMENT_LAYOUT and
            record['temporal_moment_partition'] == MOMENT_PARTITION,
            'Exact seven temporal pooling/moment/export-layout fields: '+name)


def has_pin(items, binding):
    return any(matches(item, binding) for item in items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'Existing project container required')
    require(not args.output.exists() and
            args.output.resolve().is_relative_to(Path('/opt/cuwacunu/research/causal-fresh-v11-establishment')),
            'Exclusive bounded establishment output;preserve every previous attempt')
    manifest_bytes = args.manifest.read_bytes()
    require(sha(manifest_bytes) == args.manifest_sha256, 'Exact externally admitted join-card SHA256')
    manifest = decode(manifest_bytes)
    require(manifest['kind'] == 'causal_fresh_v11_establishment_join_card_v1' and
            manifest['stage'] == 'declared_after_completed_audits_before_metadata_join' and
            manifest['root_reviewed'] is True, 'External exact reviewed evidence-join declaration')
    own = Path(__file__).read_bytes()
    require(len(own) == manifest['source']['bytes'] and sha(own) == manifest['source']['sha256'],
            'Frozen owned join source; no unknown future evidence hashes embedded')
    identity = manifest['main_source_identity']
    require(set(identity) == {'card_sha256','manifest_sha256','archive_sha256',
                              'source_files','binary_count','translation_units'} and
            all(type(identity[key]) is str and len(identity[key]) == 64 and
                set(identity[key]) <= set('0123456789abcdef')
                for key in ('card_sha256','manifest_sha256','archive_sha256')) and
            type(identity['source_files']) is int and identity['source_files'] > 0 and
            identity['binary_count'] == 7 and identity['translation_units'] == 11,
            'Externally reviewed actual immutable main source identities')
    CARD_SHA, SOURCE_SHA, ARCHIVE_SHA = (identity[key] for key in
                                      ('card_sha256','manifest_sha256','archive_sha256'))
    source_count = identity['source_files']

    values, bindings, staged = {}, {}, {}
    base = args.manifest.parent.resolve(strict=True)
    for item in manifest['inputs']:
        identifier = item['id']
        require(identifier in INPUT_IDS | OPTIONAL_IDS and identifier not in values, 'Exact unique evidence ID: '+identifier)
        relative = PurePosixPath(item['staged_path'])
        require(not relative.is_absolute() and '..' not in relative.parts and
                '\\' not in item['staged_path'] and relative.suffix == '.json', 'Bounded JSON-only staged evidence')
        path = base / str(relative)
        require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(base), 'Physical bounded evidence')
        data = path.read_bytes()
        require(len(data) == item['bytes'] and sha(data) == item['sha256'], 'Evidence bytes: '+identifier)
        require(all(key in item for key in ('path', 'bytes', 'sha256')), 'Durable evidence identity: '+identifier)
        values[identifier], bindings[identifier], staged[identifier] = decode(data), item, path
    require(INPUT_IDS <= set(values) <= INPUT_IDS | OPTIONAL_IDS and len(manifest['inputs']) == len(values),
            'All14 normal JSON roles; only optional declaration review may be added')

    card, source, build = (values[name] for name in ('card', 'source_manifest', 'build_admission'))
    require(bindings['card']['sha256'] == CARD_SHA and bindings['source_manifest']['sha256'] == SOURCE_SHA,
            'Exact scientific card/source manifest')
    require(card['kind'] == 'causal_fresh_v11_prospective_card_v1' and card['instance'] == TAG and
            'DRAFT' not in card['stage'] and card['frozen_utc'] and card['execution_admitted'] is True and
            card['baseline'] == BASE and card['baseline_remains_selected'] and
            (card['results'] is None or
             (card['results']['status'] == 'unmeasured' and
              card['results']['winner_selection_made'] is False)),
            'Immutable prospective declaration/baseline')
    require(card['mechanism']['source_updates'] == 0 and card['geometry']['native_width'] == 40 and
            not card['mechanism']['checkpoint_import_for_initialization'], 'Fresh native40 lineage')
    temporal(card['mechanism'], 'prospective architecture')
    require(card['scientific_recipe_parent'] == 'CPB-v9 source only;fresh weights/optimizer/statistics,not trained ancestry' and
            card['scoped_reference'] == 'CPB-v10.market-01',
            'V9 scientific source recipe and latest measured V10 are distinct;no inherited learned values')
    require(card['scoped_references'] == ['CPB-v4.market-01', 'CPB-v5.market-01',
                                         'CPB-v6.market-01', 'CPB-v7.market-01',
                                         'CPB-v8.market-01', 'CPB-v9.market-01', 'CPB-v10.market-01'], 'Eight protected clean references')
    require(source['card_sha256'] == CARD_SHA and source['gradient_policy'] == GRADIENT and
            len(source['files']) == len({item['path'] for item in source['files']}) == source_count ==
            card['source_counts']['capsule_sources'], 'Actual unique card-bound scientific closure')
    source_items = {item['path']: item for item in source['files']}
    require(matches(source_items['doc/CAUSAL_FRESH_V11_CARD.json'], bindings['card']), 'Closure binds card')
    owner_count = (card['source_counts']['family_inside'] + card['source_counts']['family_external_header'] +
                   card['source_counts']['research'])
    require(len(card['source_pins_after_owner_freeze']) ==
            len({item['path'] for item in card['source_pins_after_owner_freeze']}) == owner_count and
            all(item['path'] in source_items and matches(item, source_items[item['path']])
                for item in card['source_pins_after_owner_freeze']), 'All actual declared owner pins admitted')
    require(build['kind'] == 'causal_fresh_v11_build_admission_v1' and build['status'] == 'passed' and
            build['card_sha256'] == CARD_SHA and build['gradient_policy'] == GRADIENT and
            build['source_files'] == source_count and matches(build['source_manifest'], bindings['source_manifest']) and
            len(build['binaries']) == 7 and {Path(item['path']).name for item in build['binaries']} == BINARIES and
            len(build['dependency_closures']) == 11, 'Actual source/seven-binary/eleven-TU admission')
    freeze, preflight = (values[name] for name in ('prospective_freeze', 'protocol_preflight'))
    require(freeze['kind'] == 'causal_fresh_v11_prospective_freeze_v1' and freeze['status'] == 'passed' and
            matches(freeze['card'], bindings['card']) and freeze['owner_source_files'] == owner_count and
            freeze['reviewed_source_paths_before_preflight'] == source_count-1 and
            freeze['declared_capsule_sources'] == source_count and
            freeze['critical_checkpoints_verified'] == 104 and freeze['V3_sources_verified_each_repo'] == 47 and
            all(freeze[key] == 0 for key in ('AST_calls','Docker_calls','project_body_calls','model_calls','head_fits','training_updates')),
            'Fresh declaration froze before all project execution; prior104 checkpoints preserved')
    require(preflight['kind'] == 'causal_fresh_v11_source_protocol_preflight_v1' and preflight['status'] == 'passed' and
            preflight['card_sha256'] == CARD_SHA and
            preflight['family_files'] == card['source_counts']['family_inside'] and
            preflight['extra_full_owned_public_header'] == card['source_counts']['family_external_header'] and
            preflight['owned_research_files'] == card['source_counts']['research'] and
            len(preflight['sources']) == source_count-1 and
            all(item['path'] in source_items and matches(item, source_items[item['path']]) for item in preflight['sources']) and
            matches(source_items['doc/causal_fresh_v11/protocol-preflight.json'], bindings['protocol_preflight']),
            'Actual independently reviewed sources plus generated preflight close the exact capsule')
    if 'declaration_review' in values:
        declaration_review = values['declaration_review']
        require(declaration_review['status'] == 'passed' and not declaration_review['mismatches'] and
                matches(declaration_review['card'], bindings['card']), 'Exact optional static declaration review')

    comparison, table, quality, training = (values[name] for name in
                                           ('comparison', 'table_rows', 'quality_audit', 'training_audit'))
    require(all(record['gradient_policy'] == GRADIENT and record['objective_policy'] == OBJECTIVE
                and record['direct_gradient_routes'] == ROUTES and record['loss_coefficients'] == [1,1]
                and record['variance_error_detached'] is True and record['reporting_nll_backpropagated'] is False
                for record in (comparison, table, quality, training)) and
            card['training']['gradient_policy'] == GRADIENT and card['training']['objective_policy'] == OBJECTIVE and
            card['training']['direct_gradient_routes'] == ROUTES and
            card['advancement']['goal_protocol']['sha256'] ==
            '221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144',
            'All producer/audit records bind V9 detached-error objective/V11 moment routes/current22 protocol')
    require(matches(bindings['pairings'], bindings['quality_audit']) and values['pairings'] == quality,
            'Pairings is an explicit alias of exact independent quality audit')
    require(quality['kind'] == 'causal_fresh_v11_quality_independent_audit_v1' and
            training['kind'] == 'causal_fresh_v11_training_independent_audit_v1' and
            quality['status'] == training['status'] == 'passed' and
            not quality['mismatches'] and not training['mismatches'], 'Both independent audits pass/zero mismatches')
    require(training['core_saved_arithmetic_status'] == 'passed' and
            training['source_updates'] == 0 and training['pretrained_ancestor'] == 'none', 'Fresh TRAIN admission')
    require(all(training[key] == 0 for key in ('encoder_calls', 'decoder_calls', 'head_fits', 'encoder_updates')),
            'Independent saved TRAIN audit did not replay scientific execution')
    core_checks = positive_count(training['checks'], 'training core')
    module_counts = {}
    for name in TRAIN_MODULES:
        module = training[name]
        require(module['status'] == 'passed' and not module['mismatches'], 'Passed training module: '+name)
        module_counts[name] = positive_count(module['checks'], name)
    training_check_total = core_checks + sum(module_counts.values())
    positive_count(quality['checks'], 'quality audit')
    require(training['parameter_tensors'] == 77 and training['learned_tensors'] == 76 and
            training['learned_values'] == 294752 and training['fixed_values'] == 3888 and
            training['mandatory_nonzero_data_learning_tensors'] == 75 and
            training['mandatory_identifiable_tensor_value_capacity'] == 294751 and
            training['gauge_exemption'] == GAUGE and training['gauge_decay_not_data_learning'] is True,
            '76 fresh tensors;75 mandatory identifiable tensors;gauge observed separately')
    require(training['successful_CSV_rows'] == training['independent_actual_CSV_stream_pairs'] == 2048 and
            training['CSV_rows'] == training['attempted_batches'] and
            2048 <= training['attempted_batches'] <= 4096 and
            training['skipped_ineligible_attempts'] == training['attempted_batches']-2048 and
            training['shared_clipping']['rows_checked'] == 2048 and
            training['shared_clipping']['global_max_norm'] == 1 and
            training['shared_clipping']['observed_norm_independently_recomputed'] is False,
            'Actual shared globalclip1;no reconstructed upstream norm claim')
    require(has_pin(training['evidence_pins'], bindings['build_admission']) and
            has_pin(training['evidence_pins'], bindings['temporal_learning']) and
            matches(training['source_bound_temporal_pooling_proof'], bindings['temporal_pooling_proof']),
            'TRAIN audit binds exact build/temporal proof/learning')
    temporal(training, 'training audit')
    temporal(quality, 'quality audit')
    require(quality['postmeasurement_quality_and_pairing_audit_passed'] is True and
            quality['source_updates'] == 0 and quality['initial_and_final_quality_new'] is True and
            quality['registered_learned_tensors'] == 76 and quality['mandatory_identifiable_learned_tensors'] == 75 and
            quality['data_gradient_gauge_exemption'] == GAUGE, 'Independent new-quality/fresh/gauge admission')

    pairs = quality['paired_initial_equality']
    require(pairs['reference_data_archives'] == len(pairs['pairs']) == 13 and
            pairs['feature_archives'] == 8 and pairs['statistic_archives'] == 5 and
            pairs['feature_coordinate_slice'] == [20, 40] and pairs['content0_20_equality_required'] is False and
            pairs['native_coordinates_compared'] == 20 and
            pairs['whole_native40_equality_claimed'] is False and pairs['old_model_numeric_reads'] == 0,
            'Eight partial native20:40 DATA plus five whole-statistic pairings against saved V9')
    require(pairs['all_declared_numeric_tensor_values_exact'] is True and
            all(item['all_declared_numeric_tensor_values_exact'] is True for item in pairs['pairs']),
            'All13 declared tail20/whole-statistic pairings exact')
    feature_fields = 0
    for pair in pairs['pairs']:
        require(all(item['tensor_bytes_equal'] is True for item in pair['numeric_tensors']), 'All declared logical tensor fields exact')
        for item in pair['numeric_tensors']:
            if item['native_coordinate_slice'] is not None:
                require(item['native_coordinate_slice'] == [20, 40] and
                        item['comparison'] == 'native columns20:40 only', 'Only unchanged tail20 coordinates compare')
                feature_fields += 1
    require(feature_fields == 20, 'Two market matrices plus three views in six synthetic DATA archives')

    proof, learning = (values[name] for name in ('temporal_pooling_proof', 'temporal_learning'))
    temporal(proof, 'temporal fixture')
    temporal(learning, 'actual temporal learning')
    for key, expected_value in card['temporal_pooling_proof_contract']['required_proof_fields'].items():
        require(proof[key] == expected_value, 'Exact audited fixture predicate: '+key)
    require(proof['kind'] == 'cpb_v11_temporal_pooling_mechanism_proof_v1' and
            Path(proof['binary']['path']).name == 'test_cpb_v11_temporal_pooling' and
            any(matches(proof['binary'], item) for item in build['binaries']), 'Admitted seventh temporal binary')
    require(learning['kind'] == 'cpb_v11_actual_temporal_pooling_learning_v1' and learning['status'] == 'passed' and
            learning['updates'] == 2048 and learning['group_rows'] == 8192 and learning['extra_encoder_calls'] == 0 and
            learning['weight_decay_counted_as_data_gradient'] is False and
            learning['movement_includes_AdamW_weight_decay'] is True, 'Actual bounded postclip learning receipt')
    require(len(learning['groups']) == 4 and [group['bin'] for group in learning['groups']] == [0, 1, 2, 3],
            'All four ordered original-position groups')
    event_names = ('weight_nonzero_data_gradient_steps', 'bias_nonzero_data_gradient_steps',
                   'weight_movement_steps', 'bias_movement_steps',
                   'position_centered_data_gradient_steps', 'position_centered_movement_steps')
    for index, group in enumerate(learning['groups']):
        require([group[key] for key in ('export_row_begin', 'export_row_end', 'position_begin', 'position_end')] ==
                [8*index, 8*(index+1), 4*index, 4*(index+1)] and
                all(type(group[key]) is int and 0 < group[key] <= 2048 for key in event_names) and
                group['final_weight_changed'] is True and group['final_bias_changed'] is True,
                'Observed group event/final-change predicates;no float norm recalculation')
    gauge = learning['gauge']
    require(gauge['name'] == GAUGE and gauge['data_gradient_requirement_exempt'] is True and
            gauge['registered_in_same_AdamW'] is True and gauge['movement_is_not_data_learning_proof'] is True,
            'Registered gauge;decay movement is not data learning')
    checked_temporal = training['temporal_pooling_saved_learning']
    require(checked_temporal['group_rows_checked'] == 8192 and checked_temporal['gauge_rows_checked'] == 2048 and
            checked_temporal['final_projection_rows_and_position_contrasts_verified'] is True and
            matches(checked_temporal['receipt'], bindings['temporal_learning']), 'Independent group/contrast/gauge arithmetic')
    checked_moments = checked_temporal['temporal_moment_saved_learning']
    require(checked_moments['status'] == 'passed' and not checked_moments['mismatches'] and
            positive_count(checked_moments['checks'], 'nested moment audit') > 0 and
            checked_moments['moment_subgroup_rows_checked'] == learning['moment_subgroup_rows'] == 16384 and
            checked_moments['whole_group_two_moment_partition_rows_checked'] == 8192 and
            checked_moments['whole_group_partition_squared_norms_and_counts_verified'] is True and
            checked_moments['manual_F32_absolute_tolerance'] == checked_moments['manual_F32_relative_tolerance'] == 3e-5 and
            checked_moments['per_scalar_gradient_proof_claimed'] is False and
            checked_moments['other2046_raw_gradients_reconstructed'] is False and
            checked_moments['pool_centered_variance_is_native_uncertainty'] is False,
            'Passed nested moment/subgroup/whole8-two4 saved arithmetic,with honest reconstruction limits')
    subgroup_events = ('weight_nonzero_data_gradient_steps', 'bias_nonzero_data_gradient_steps',
                       'weight_movement_steps', 'bias_movement_steps')
    require(len(learning['moment_subgroups']) == len(checked_moments['moment_subgroup_event_counts']) == 8,
            'Exactly eight observed mean4/centered-variance4 subgroups')
    for slot, group in enumerate(learning['moment_subgroups']):
        original_bin, half = divmod(slot, 2)
        require(group['bin'] == original_bin and group['moment'] == ('mean' if half == 0 else 'centered_variance') and
                group['export_row_begin'] == original_bin*8+half*4 and
                group['export_row_end'] == original_bin*8+(half+1)*4 and
                all(type(group[key]) is int and 0 < group[key] <= 2048 and
                    group[key] == checked_moments['moment_subgroup_event_counts'][slot][key]
                    for key in subgroup_events) and
                group['final_weight_changed'] is True and group['final_bias_changed'] is True,
                'Independent four-row subgroup EVENT/final-change admission;whole8 events are OR,not additive booleans')
    local_witnesses = checked_moments['actual_same_forward_moment_witnesses']
    require(len(local_witnesses) == 2 and [item['update'] for item in local_witnesses] == [1,2048],
            'Only actual first/last same-forward moment/local-adjoint witnesses')
    local_parameters = {'temporal_export_weight', 'temporal_export_bias', 'temporal_pool_positions'}
    for item in local_witnesses:
        require(item['same_forward_only'] is True and item['total_component_archive_fields'] == 56 and
                item['new_temporal_trace_fields'] == 21 and item['parameter_array_fields'] == 12 and
                item['upstream_stage'] == 'PRECLIP-post-empty-export' and
                len(item['upstream_gradient_defined_by_route']) == 2 and
                all(type(flag) is bool for flag in item['upstream_gradient_defined_by_route']) and
                type(item['recorded_clip_factor_applied_once']) in (int,float) and
                math.isfinite(item['recorded_clip_factor_applied_once']) and
                0 < item['recorded_clip_factor_applied_once'] <= 1 and
                item['local_export_and_positions_adjoint_verified'] is True and
                item['local_adjoint_absolute_tolerance'] == 1e-9 and item['local_adjoint_relative_tolerance'] == 3e-4 and
                item['postclip_export_nonvacuity_threshold'] == 1e-6 and
                math.isfinite(item['postclip_export_expected_maximum']) and
                item['postclip_export_expected_maximum'] > 1e-6 and
                item['full_hidden_encoder_or_global_norm_reconstruction'] is False and
                item['matched_pre_step_decoder_adjoint_replay'] is False,
                'Actual56-field local-adjoint scope,PRECLIP flags,oneclip/tight tolerance/nonvacuity;no model replay')
        errors = item['local_parameter_gradient_error_summaries']
        require(set(errors) == local_parameters, 'Exactly three actual local parameter parity gates')
        for name, observed in errors.items():
            require(set(observed) == {'maximum_expected_absolute','maximum_actual_absolute',
                                     'maximum_absolute_error','maximum_scaled_error'} and
                    all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in observed.values()) and
                    observed['maximum_scaled_error'] <= 1,
                    'Passed independent saved local-gradient error summary: '+name)
        gauge_observed = item['common_score_bias_gauge_descriptive']
        require(gauge_observed['parity_gate'] is False and gauge_observed['nonzero_gradient_required'] is False and
                all(type(gauge_observed[key]) in (int,float) and math.isfinite(gauge_observed[key])
                    for key in ('actual','reference','absolute_error')),
                'Common scalar gauge finite descriptive evidence,never a tight parity/nonzero gate')
        component_witnesses = training['actual_pre_step_composite_objective']['witnesses']
        require(has_pin([entry['witness'] for entry in component_witnesses if entry['update'] == item['update']], item['artifact']) and
                has_pin([entry['artifact'] for entry in checked_temporal['actual_raw_array_witnesses']
                         if entry['update'] == item['update']], item['artifact']),
                'Same exact actual first/last archive binds component,parameter-array and local-moment audits')
    require(matches(quality['temporal_mechanism_and_learning_bindings']['proof'], bindings['temporal_pooling_proof']) and
            matches(quality['temporal_mechanism_and_learning_bindings']['learning_receipt'], bindings['temporal_learning']),
            'Quality/TRAIN audits bind same temporal fixture/trajectory receipt')
    between = learning['between_bin_positions']
    checked_between = checked_temporal['between_bin_positions']
    require(between['basis'] == checked_between['basis'] == 'hadamard-full16-signs-div4-v1' and
            between['basis_signs'] == checked_between['basis_signs'] == [[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]] and
            between['full_position_divisor'] == checked_between['full_position_divisor'] == 4 and
            between['rows'] == checked_between['rows_checked'] == 2048 and
            between['gradient_steps'] == checked_between['gradient_event_counts'] and
            between['movement_steps'] == checked_between['movement_event_counts'] and
            all(type(n) is int and 0 < n <= 2048 for n in between['gradient_steps']+between['movement_steps']) and
            checked_between['individual_final_coefficient_nonzero_gate'] is False and
            len(between['final_contrast_changes']) == len(checked_between['final_signed_contrast_changes']) == 3 and
            all(audited_scalar_identity(left,right) for left,right in
                zip(between['final_contrast_changes'],checked_between['final_signed_contrast_changes'])) and
            audited_scalar_identity(between['final_projection_change_l2'],checked_between['final_projection_change_l2']) and
            between['final_projection_change_l2'] > 0 and checked_between['final_projection_change_l2'] > 0 and
            checked_between['actual_raw_array_updates'] == [1,2048] and
            checked_between['other2046_raw_gradients_reconstructed'] is False,
            'All three independent between-bin events and actual two-witness arithmetic, no per-final-coefficient gate')
    require(len(learning['artifacts']) == 4 and
            {Path(item['path']).name for item in learning['artifacts']} ==
            {'temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv','temporal-between-bin-position-learning.csv','temporal-moment-subgroup-learning.csv'},
            'Four observed CSV artifacts bound by independent saved arithmetic')

    mirror = values['complete_mirror']
    require(mirror['kind'] == 'causal_fresh_v11_container_to_local_evidence_mirror_v1' and
            mirror['status'] == 'passed' and mirror['scope'] == 'Complete' and mirror['attempt'] == 'attempt-001' and
            mirror['source_files_verified'] == source_count and mirror['all_mirrored_pt_ignored'] is True and
            mirror['source_stage']['status'] == 'passed' and
            mirror['source_stage']['card_sha256'] == CARD_SHA and
            mirror['source_stage']['manifest_sha256'] == SOURCE_SHA and
            mirror['source_stage']['archive_sha256'] == ARCHIVE_SHA, 'Complete unchanged-source local mirror')
    attempt = Path('/opt/cuwacunu/research/causal-fresh-v11/attempt-001')
    mirrored = {item['container_path']:item for item in mirror['files']}

    def completed_stage(path, root_stage):
        require(str(path) in mirrored and path.is_file() and not path.is_symlink(),
                'Every normal quality/proof stage is an exact Complete mirror member')
        data = path.read_bytes()
        require(len(data) == mirrored[str(path)]['bytes'] and sha(data) == mirrored[str(path)]['sha256'],
                'Actual normal completed stage bytes remain mirrored')
        record = decode(data)
        require(record['exit_code'] == 0 and (not root_stage or record['status'] == 'passed'),
                'Normal V11 stages all pass;no new recovery is assumed')
        return record

    for name in ('training','initial-saved-proof','final-saved-proof','quality'):
        completed_stage(attempt/(name+'-stage.json'), True)
    for name in ('extraction','initial_market_heads','initial_synthetic_heads',
                 'final_market_heads','final_synthetic_heads','report','comparison'):
        completed_stage(attempt/'quality-001'/(name+'-stage.json'), False)
    require(comparison['candidate'] == quality['model'] == training['model'] == table['model'] == TAG and
            comparison['baseline'] == BASE and comparison['baseline_unchanged'] and table['clean_baseline_unchanged'],
            'Same candidate andprotected clean baseline')
    require(quality['full_coverage'] and comparison['full_coverage'] and table['full_coverage'] and
            comparison['runtime_trust_admission_passed'] and table['runtime_trust_admission_passed'], 'Coverage/runtime admission')
    require(comparison['mandatory_postmeasurement_trust'] == table['mandatory_postmeasurement_trust'] == PENDING and
            comparison['trust_admission_passed'] is False and table['trust_protocol_admission_passed'] is False and
            comparison['overall_establishment_passed'] is False and table['overall_establishment_passed'] is False,
            'Original frozen producer pending-trust bytes remain unchanged')
    conditions = comparison['numeric_conditions']
    require(comparison['numeric_total'] == len(conditions) == 22 and
            [item['id'] for item in conditions] == [item['id'] for item in card['advancement']['criteria']] and
            all(type(item['passed']) is bool for item in conditions), 'All22 frozen decision identities')
    count = sum(item['passed'] for item in conditions)
    require(count == comparison['numeric_passed'] == quality['clean_baseline_comparison']['numeric_passed'] and
            quality['clean_baseline_comparison']['numeric_total'] == 22, 'Independently verified numeric condition count')
    require(comparison['numeric_conditions_passed'] is (count == 22) and
            comparison['numeric_and_coverage_conditions_passed'] is (count == 22), 'Numeric/coverage distinct from pending trust')
    require(not comparison['overall_winner_declared'] and not table['overall_winner_declared'] and
            not comparison['automatic_promotion'] and not comparison['application_changed'], 'No winner/application selection')
    for identifier in ('comparison', 'table_rows', 'card', 'quality_receipt', 'build_admission',
                       'temporal_pooling_proof', 'temporal_learning'):
        require(has_pin(quality['verified_artifacts'], bindings[identifier]), 'Quality audit binds exact input: '+identifier)
    execution = values['quality_receipt']
    require(execution['status'] == 'passed' and execution['card_sha256'] == CARD_SHA and
            matches(execution['build_admission'], bindings['build_admission']) and
            matches(execution['temporal_pooling_proof'], bindings['temporal_pooling_proof']) and
            matches(execution['temporal_learning']['receipt'], bindings['temporal_learning']), 'Source/proof/quality execution joins')
    for identifier, path in staged.items():
        data = path.read_bytes()
        require(len(data) == bindings[identifier]['bytes'] and sha(data) == bindings[identifier]['sha256'],
                'Evidence unchanged throughout JSON join: '+identifier)
    require(args.manifest.read_bytes() == manifest_bytes and Path(__file__).read_bytes() == own, 'Join source/card unchanged')

    overall = count == 22
    result = {
        'kind': 'causal_fresh_v11_postmeasurement_establishment_v1', 'status': 'passed',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidate': TAG, 'baseline': BASE, 'baseline_changed': False,
        'measured_candidate_established': True, 'runtime_trust_admission_passed': True,
        'trust_admission_passed': True, 'initial_partial_native20_40_pairings_passed': True,
        'initial_pairing_count': 13, 'initial_pairing_feature_archives': 8, 'initial_pairing_statistic_archives': 5,
        'initial_pairing_native_coordinate_slice': [20, 40], 'whole_native40_initial_equality_claimed': False,
        'independent_training_audit_passed': True, 'independent_quality_audit_passed': True,
        'training_audit_core_checks': core_checks, 'training_audit_five_module_checks': module_counts,
        'training_audit_total_checks': training_check_total,
        'training_audit_count_definition': 'core checks plus five modules;no scientific replay',
        'quality_audit_checks': quality['checks'], 'temporal_pooling_policy': POLICY,
        'temporal_pooling_partition': PARTITION, 'temporal_pooling_mix': [0.5,0.5],
        'temporal_moment_policy': MOMENT_POLICY, 'temporal_export_layout': MOMENT_LAYOUT,
        'temporal_moment_partition': MOMENT_PARTITION,
        'scientific_recipe_parent': card['scientific_recipe_parent'], 'scoped_reference': card['scoped_reference'],
        'empty_local_bin_policy': 'zero-whole-export-slice-after-bias-v1', 'registered_learned_tensors': 76,
        'mandatory_identifiable_learned_tensors': 75, 'mandatory_identifiable_tensor_value_capacity': 294751,
        'data_gradient_gauge_exemption': GAUGE, 'weight_decay_counted_as_data_learning': False,
        'all_four_export_row_groups_and_position_contrasts_verified': True,
        'all_eight_temporal_moment_subgroups_verified': True,
        'whole8_two4_squared_norms_and_element_counts_verified': True,
        'actual_first_last_local_moment_adjoints_verified': True,
        'actual_local_moment_adjoint_parameter_tensors': sorted(local_parameters),
        'actual_local_moment_adjoint_absolute_tolerance': 1e-9,
        'actual_local_moment_adjoint_relative_tolerance': 3e-4,
        'each_actual_local_export_nonvacuity_threshold': 1e-6,
        'common_score_bias_gauge_tight_parity_required': False,
        'actual_component_archive_fields': 56, 'same_forward_temporal_fields': 21,
        'full_encoder_gradient_or_global_norm_reconstructed': False,
        'all_three_between_bin_position_directions_verified': True,
        'between_bin_overall_final_projection_verified': True,
        'individual_final_coefficient_nonzero_gate': False,
        'full_coverage': True, 'numeric_passed': count, 'numeric_total': 22,
        'numeric_conditions_passed': overall, 'numeric_and_coverage_conditions_passed': overall,
        'overall_establishment_passed': overall, 'advancement_conditions_passed': overall,
        'overall_winner': False, 'winner_selection_made': False,
        'frozen_producer_pending_trust_preserved': True, 'application_changed': False, 'automatic_promotion': False,
        'objective_policy': OBJECTIVE, 'gradient_policy': GRADIENT, 'direct_gradient_routes': ROUTES,
        'variance_error_detached': True, 'reporting_nll_backpropagated': False,
        'normal_prospective_source_admission': {'source_files':source_count, 'owner_source_files':owner_count,
            'freeze_and_preflight_bound':True, 'normal_quality_all_stages_passed':True,
            'scientific_model_head_or_training_replay':False},
        'historical_V9_reporting_recovery_preserved': card['historical_V9_reporting_recovery'],
        'evidence': {identifier: {key: item[key] for key in ('path', 'bytes', 'sha256')}
                     for identifier, item in bindings.items()},
        'join_card': {'path': str(args.manifest), 'bytes': len(manifest_bytes), 'sha256': sha(manifest_bytes)},
        'source': manifest['source'], 'model_calls': 0, 'head_fits': 0, 'training_updates': 0,
        'scientific_metric_recomputations': 0, 'archive_tensor_numeric_reads': 0,
        'saved_float_identity_tolerance': {'absolute':2e-12,'relative':2e-12,
            'scope':'Existing training-audit tolerance on saved producer/independent final contrasts/norm only;no metric recomputation'},
        'scope': 'Postmeasurement JSON/hash join only;trustworthy measurements,criterion achievement andhuman winner/application selection are separate.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'numeric_passed': count, 'numeric_total': 22,
                      'training_audit_total_checks': training_check_total,
                      'advancement_conditions_passed': overall, 'overall_winner': False}))


if __name__ == '__main__':
    main()


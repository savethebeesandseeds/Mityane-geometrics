# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Owned V7-derived V8 JSON/hash evidence join; no model or producer imports."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath

TAG, BASE = 'CPB-v8.market-01', 'CPB-v3.market-01'
CARD_SHA = '6acfaadfe758c0d252c8138fc2830807d40444f9a9adfa61f85ec13dde7c63a6'
SOURCE_SHA = '8daba935ee45332b892225c6994a00b59ad95b2e4da4fe7348f6c6518266cf2e'
ARCHIVE_SHA = 'bc883e6e14365f68e02e7f5b0e5c3df9eeca204cbdb7140d34de775768d14591'
POLICY = 'original-position-four-bin-disjoint-export-rows-v1'
PARTITION = [4, 16, 32, 4, 8]
GRADIENT = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
OBJECTIVE = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
GAUGE = 'core.encoder.level.base.backbone.pool_score.bias'
INPUT_IDS = frozenset(('comparison', 'table_rows', 'quality_audit', 'training_audit',
                      'complete_mirror', 'card', 'quality_receipt', 'pairings',
                      'source_manifest', 'build_admission', 'temporal_pooling_proof', 'temporal_learning',
                      'host_recovery_card', 'host_recovery_review'))
OPTIONAL_IDS = frozenset(('declaration_review',))
TRAIN_MODULES = ('numeric_statistics', 'saved_tensor_state', 'manual_saved_Gaussian',
                 'actual_pre_step_composite_objective', 'temporal_pooling_saved_learning')
PENDING = {'initial_native20_40_and_statistics_pairings': 'pending',
           'independent_training_audit': 'pending', 'independent_quality_audit': 'pending'}
BINARIES = frozenset(('test_causal_patch_bottleneck_mae_v8', 'test_cpb_v8_variance_gradient',
                      'test_cpb_v8_temporal_pooling', 'causal_fresh_v8_train',
                      'causal_fresh_v8_extract', 'causal_fresh_v8_market_read', 'causal_fresh_v8_synthetic_read'))


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


def temporal(record, name):
    require(record['temporal_pooling_policy'] == POLICY and
            record['temporal_pooling_partition'] == PARTITION, 'Exact temporal policy/partition: '+name)


def has_pin(items, binding):
    return any(matches(item, binding) for item in items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'Existing project container required')
    require(not args.output.exists(), 'Preserve every previous establishment attempt')
    manifest_bytes = args.manifest.read_bytes()
    require(sha(manifest_bytes) == args.manifest_sha256, 'Exact externally admitted join-card SHA256')
    manifest = decode(manifest_bytes)
    require(manifest['kind'] == 'causal_fresh_v8_establishment_join_card_v1' and
            manifest['stage'] == 'declared_after_completed_audits_before_metadata_join' and
            manifest['root_reviewed'] is True, 'External exact reviewed evidence-join declaration')
    own = Path(__file__).read_bytes()
    require(len(own) == manifest['source']['bytes'] and sha(own) == manifest['source']['sha256'],
            'Frozen owned join source; no unknown future evidence hashes embedded')
    require(manifest['main_source_identity'] == {
        'card_sha256': CARD_SHA, 'manifest_sha256': SOURCE_SHA, 'archive_sha256': ARCHIVE_SHA,
        'source_files': 113, 'binary_count': 7, 'translation_units': 11}, 'Actual immutable V8 source identities')

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
            'All14 required JSON roles; only optional declaration review may be added')

    card, source, build = (values[name] for name in ('card', 'source_manifest', 'build_admission'))
    require(bindings['card']['sha256'] == CARD_SHA and bindings['source_manifest']['sha256'] == SOURCE_SHA,
            'Exact scientific card/source manifest')
    require(card['kind'] == 'causal_fresh_v8_prospective_card_v1' and card['instance'] == TAG and
            card['baseline'] == BASE and card['baseline_remains_selected'] and card['results'] is None,
            'Immutable prospective declaration/baseline')
    require(card['mechanism']['source_updates'] == 0 and card['geometry']['native_width'] == 40 and
            not card['mechanism']['checkpoint_import_for_initialization'], 'Fresh native40 lineage')
    require(card['scoped_references'] == ['CPB-v4.market-01', 'CPB-v5.market-01',
                                         'CPB-v6.market-01', 'CPB-v7.market-01'], 'Five protected clean references')
    require(source['card_sha256'] == CARD_SHA and source['gradient_policy'] == GRADIENT and
            len(source['files']) == len({item['path'] for item in source['files']}) == 113,
            'Actual unique113-file scientific closure')
    source_items = {item['path']: item for item in source['files']}
    require(matches(source_items['doc/CAUSAL_FRESH_V8_CARD.json'], bindings['card']), 'Closure binds card')
    require(len(card['source_pins_after_owner_freeze']) == 90 and
            all(item['path'] in source_items and matches(item, source_items[item['path']])
                for item in card['source_pins_after_owner_freeze']), 'All90 declared owner pins admitted')
    require(build['kind'] == 'causal_fresh_v8_build_admission_v1' and build['status'] == 'passed' and
            build['card_sha256'] == CARD_SHA and build['gradient_policy'] == GRADIENT and
            build['source_files'] == 113 and matches(build['source_manifest'], bindings['source_manifest']) and
            len(build['binaries']) == 7 and {Path(item['path']).name for item in build['binaries']} == BINARIES and
            len(build['dependency_closures']) == 11, 'Actual113-source/seven-binary/eleven-TU admission')
    recovery, recovery_review = (values[name] for name in ('host_recovery_card', 'host_recovery_review'))
    require(bindings['host_recovery_card']['sha256'] == '526e51c1357d6f89b289cd96e202a7853ad54c17e991f01c8bcb293e3cbd28b9' and
            bindings['host_recovery_review']['sha256'] == '8557b4e386ebe43a7f8cfd1bf50d3222a0e3371f8fde66306114d99bc1fcd67f',
            'Exact prospective external host-repair declaration and independent review')
    require(recovery['kind'] == 'causal_fresh_v8_host_orchestration_recovery_v1' and
            recovery['status'] == 'prospectively_reviewed_host_only_repair' and
            recovery_review['kind'] == 'causal_fresh_v8_host_recovery_independent_static_review_v1' and
            recovery_review['status'] == 'passed' and not recovery_review['mismatches'] and
            matches(recovery_review['recovery_card'], bindings['host_recovery_card']), 'Reviewed additive host-only recovery')
    require(recovery['original_source_and_scientific_semantics_unchanged'] is True and recovery['old_files_modified'] is False and
            matches(recovery['encoder_card'], bindings['card']) and
            matches(recovery['original_manifest'], bindings['source_manifest']) and
            recovery['original_archive']['sha256'] == ARCHIVE_SHA and
            recovery['new_helper']['sha256'] == '0e9b471cb89bc7cecd363551a684c61f4f894409bd0c774cf4b4a5b37920366d' and
            matches(recovery_review['new_helper'], recovery['new_helper']) and
            matches(recovery['frozen_original_helper'], source_items['doc/causal_fresh_v8/run-container.ps1']),
            'Original113 retains unused Windows helper; actual external full-copy repair executes same scientific source')
    failure = recovery['failure']
    require(failure['phase'] == 'Stage' and failure['before_project_build_or_test'] is True and
            'before the initial inspect' in failure['reason'] and
            all(failure[key] == 0 for key in ('builds', 'model_calls', 'head_fits', 'training_updates')) and
            failure['container_candidate_path_exists'] is False and failure['container_build_path_exists'] is False,
            'One preserved host Stage failure before Docker/project operations; no scientific attempt repeated')
    if 'declaration_review' in values:
        declaration_review = values['declaration_review']
        require(bindings['declaration_review']['sha256'] == 'b84c0308f4a0f0cc87a9d87e22660eb331c4da68ae5486593b6f416c5fd3fdb6' and
                declaration_review['status'] == 'passed' and not declaration_review['mismatches'] and
                matches(declaration_review['card'], bindings['card']), 'Exact optional static declaration review')

    comparison, table, quality, training = (values[name] for name in
                                           ('comparison', 'table_rows', 'quality_audit', 'training_audit'))
    require(all(record['gradient_policy'] == GRADIENT and record['objective_policy'] == OBJECTIVE
                for record in (comparison, table, quality, training)) and
            card['training']['gradient_policy'] == GRADIENT and card['training']['objective_policy'] == OBJECTIVE,
            'All producer/audit records preserve the V6 objective and global-clipping policy')
    require(matches(bindings['pairings'], bindings['quality_audit']) and values['pairings'] == quality,
            'Pairings is an explicit alias of exact independent quality audit')
    require(quality['kind'] == 'causal_fresh_v8_quality_independent_audit_v1' and
            training['kind'] == 'causal_fresh_v8_training_independent_audit_v1' and
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
            training['mandatory_nonzero_data_learning_values'] == 294751 and
            training['gauge_exemption'] == GAUGE and training['gauge_decay_not_data_learning'] is True,
            '76 fresh tensors;75 mandatory identifiable tensors;gauge observed separately')
    require(training['CSV_rows'] == training['independent_actual_CSV_stream_pairs'] == 2048 and
            training['shared_clipping']['rows_checked'] == 2048 and
            training['shared_clipping']['global_max_norm'] == 1 and
            training['shared_clipping']['observed_norm_independently_recomputed'] is False,
            'Actual V6 shared clipping;no reconstructed norm claim')
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
            pairs['whole_native40_equality_claimed'] is False and pairs['old_model_numeric_reads'] == 0,
            'Eight partial native20:40 DATA plusfive whole-statistic pairings;never whole40')
    require(pairs['all_declared_numeric_tensor_values_exact'] is True and
            all(item['all_declared_numeric_tensor_values_exact'] is True for item in pairs['pairs']),
            'All13 declared partial/whole-statistic pairings exact')
    feature_fields = 0
    for pair in pairs['pairs']:
        require(all(item['tensor_bytes_equal'] is True for item in pair['numeric_tensors']), 'All declared logical tensor fields exact')
        for item in pair['numeric_tensors']:
            if item['native_coordinate_slice'] is not None:
                require(item['native_coordinate_slice'] == [20, 40] and
                        item['comparison'] == 'native columns20:40 only', 'No learned content0:20 equality gate')
                feature_fields += 1
    require(feature_fields == 20, 'Two market matrices plus three views in six synthetic DATA archives')

    proof, learning = (values[name] for name in ('temporal_pooling_proof', 'temporal_learning'))
    temporal(proof, 'temporal fixture')
    temporal(learning, 'actual temporal learning')
    for key, expected_value in card['temporal_pooling_proof_contract']['required_proof_fields'].items():
        require(proof[key] == expected_value, 'Exact audited fixture predicate: '+key)
    require(proof['kind'] == 'cpb_v8_temporal_pooling_mechanism_proof_v1' and
            Path(proof['binary']['path']).name == 'test_cpb_v8_temporal_pooling' and
            any(matches(proof['binary'], item) for item in build['binaries']), 'Admitted seventh temporal binary')
    require(learning['kind'] == 'cpb_v8_actual_temporal_pooling_learning_v1' and learning['status'] == 'passed' and
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
    require(matches(quality['temporal_mechanism_and_learning_bindings']['proof'], bindings['temporal_pooling_proof']) and
            matches(quality['temporal_mechanism_and_learning_bindings']['learning_receipt'], bindings['temporal_learning']),
            'Quality/TRAIN audits bind same temporal fixture/trajectory receipt')

    mirror = values['complete_mirror']
    require(mirror['kind'] == 'causal_fresh_v8_container_to_local_evidence_mirror_v1' and
            mirror['status'] == 'passed' and mirror['scope'] == 'Complete' and mirror['attempt'] == 'attempt-001' and
            mirror['source_files_verified'] == 113 and mirror['all_mirrored_pt_ignored'] is True and
            mirror['source_stage']['status'] == 'passed' and
            mirror['source_stage']['card_sha256'] == CARD_SHA and
            mirror['source_stage']['manifest_sha256'] == SOURCE_SHA and
            mirror['source_stage']['archive_sha256'] == ARCHIVE_SHA, 'Complete unchanged-source local mirror')
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
        'kind': 'causal_fresh_v8_postmeasurement_establishment_v1', 'status': 'passed',
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
        'temporal_pooling_partition': PARTITION, 'registered_learned_tensors': 76,
        'mandatory_identifiable_learned_tensors': 75, 'mandatory_identifiable_tensor_value_capacity': 294751,
        'data_gradient_gauge_exemption': GAUGE, 'weight_decay_counted_as_data_learning': False,
        'all_four_export_row_groups_and_position_contrasts_verified': True,
        'full_coverage': True, 'numeric_passed': count, 'numeric_total': 22,
        'numeric_conditions_passed': overall, 'numeric_and_coverage_conditions_passed': overall,
        'overall_establishment_passed': overall, 'advancement_conditions_passed': overall,
        'overall_winner': False, 'winner_selection_made': False,
        'frozen_producer_pending_trust_preserved': True, 'application_changed': False, 'automatic_promotion': False,
        'host_orchestration_recovery': {
            'preserved_host_Stage_failures': 1, 'failure_before_Docker_or_project_operations': True,
            'successful_exclusive_container_source_Stage_passes': 1,
            'scientific_card_or_source_revision': False, 'scientific_model_head_or_training_replay': False,
            'original_scientific_source_files': 113, 'actual_external_helper_sha256': recovery['new_helper']['sha256'],
            'scope': 'Full-copy host wrapper repair only; original113 source/binaries and all failures preserved.',
        },
        'evidence': {identifier: {key: item[key] for key in ('path', 'bytes', 'sha256')}
                     for identifier, item in bindings.items()},
        'join_card': {'path': str(args.manifest), 'bytes': len(manifest_bytes), 'sha256': sha(manifest_bytes)},
        'source': manifest['source'], 'model_calls': 0, 'head_fits': 0, 'training_updates': 0,
        'scientific_float_arithmetic': 0, 'archive_tensor_numeric_reads': 0,
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


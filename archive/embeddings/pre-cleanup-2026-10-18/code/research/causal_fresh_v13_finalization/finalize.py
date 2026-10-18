# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Owned V13 metadata-only proposal generator, copied completely from final V12.

Only saved JSON/text and artifact hashes are read. No models, heads, array
libraries, live metadata application, descriptor rewrite or publication.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import pathlib
import sys

FAMILY = 'code/encoders/causal_patch_bottleneck_mae/versions/'
METHOD = 'CPB-v13.market-01'
BASELINE = 'CPB-v3.market-01'
POLICY = 'Strictly higher saved final score updates only; exact ties retain earlier leader. Initial stages are descriptive, never checkpoint selection or overall promotion'
OBJECTIVE = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
GRADIENT = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
ENCODER_GRADIENT = 'Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1'
OBJECTIVE_FIELDS = dict(objective_policy=OBJECTIVE, mean_huber_delta=1,
                        loss_coefficients=[1, 1], variance_error_detached=True,
                        reporting_nll_backpropagated=False,
                        direct_gradient_routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1')
TEMPORAL_FIELDS = dict(temporal_pooling_policy='original-position-four-bin-half-mixture-mean4-regularized-stddev4-v1',
    temporal_pooling_partition=[4,16,32,4,8],temporal_pooling_mix=[0.5,0.5],
    temporal_moment_policy='componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1',
    temporal_export_layout='per-original-bin-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1',
    temporal_moment_partition=[4,4],
    temporal_dispersion_transform_policy='sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1',
    temporal_dispersion_regularizer=0.0009765625,
    empty_local_bin_policy='zero-whole-export-slice-after-bias-v1',registered_learned_tensors=76,
    mandatory_identifiable_learned_tensors=75,
    data_gradient_gauge_exemption='core.encoder.level.base.backbone.pool_score.bias')
PRIOR_VERSIONS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
VERSIONS = PRIOR_VERSIONS + (13,)
REQUIRED_INPUTS = frozenset(('ledger', 'catalog', 'prior_snapshot', 'descriptor',
    'experiment_card', 'source_manifest', 'build', 'training', 'resume',
    'initialization', 'calibration', 'engineering', 'gradient', 'initial_proof',
    'final_proof', 'training_audit',
    'quality_audit', 'pairings', 'establishment', 'mirror', 'quality_receipt',
    'comparison', 'v3_table', 'v4_table', 'v5_table', 'v6_table', 'v7_table', 'v8_table', 'v9_table', 'v10_table', 'v11_table', 'v12_table', 'v13_table',
    'v6_establishment', 'v7_establishment', 'v8_establishment', 'v9_establishment', 'v10_establishment', 'v11_establishment', 'v12_establishment', 'temporal_proof', 'temporal_learning',
    'prospective_freeze', 'protocol_preflight',
    'results', 'review', 'detail'))
FINALIZATION_ROOT = pathlib.Path('/opt/cuwacunu/research/causal-fresh-v13-finalization')
ASSERTIONS = dict(pairing_status='passed', pairing_count=13,
    training_audit_status='passed', quality_audit_status='passed',
    establishment_status='passed', establishment_runtime=True,
    establishment_trust=True, establishment_coverage=True,
    establishment_measured=True)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def pin(path, label=None):
    path = pathlib.Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=label or str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Finite scalar required')
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bounded(root, relative):
    relative = pathlib.PurePosixPath(relative)
    require(not relative.is_absolute() and '..' not in relative.parts and relative.parts,
            'Bounded metadata path required')
    result = root / relative
    require(result.resolve().is_relative_to(root.resolve()), 'Metadata path escaped source')
    return result


def at(record, path):
    require(isinstance(path, list) and path, 'Explicit evidence field path required')
    for key in path:
        record = record[key]
    return record


def objective(record, label, require_gradient_policy=True):
    for key, expected in OBJECTIVE_FIELDS.items():
        require(record[key] == expected, label + ' exact objective field: ' + key)
    if require_gradient_policy:
        require(record['gradient_policy'] == GRADIENT, label + ' gradient policy')
    else:
        require('gradient_policy' not in record, label + ' documented absent gradient selector; no fabricated field')


def temporal(record, label):
    for key, expected in TEMPORAL_FIELDS.items():
        require(record[key] == expected, label + ' exact temporal/identifiable field: ' + key)


def stagecards(table, version, source):
    method = 'CPB-v%d.market-01' % version
    require(table['new_market_rows'] == 8, 'Eight measured market rows')
    measured = [r for r in table['market_rows'] if r['method'] in (method, method + '.init')]
    require(len(measured) == 8 and len(table['synthetic']['panels']) == 4,
            'Two complete measured stages')
    result = []
    for stage, label in (('initial', method + '.init'), ('final', method)):
        market = [copy.deepcopy(r) for r in measured if r['method'] == label]
        panels = []
        for panel in table['synthetic']['panels']:
            rows = [r for r in panel['rows'] if r['method'] == label]
            require(len(rows) == 1, 'Exactly one measured synthetic stage row')
            item = {k: copy.deepcopy(v) for k, v in panel.items() if k != 'rows'}
            item['row'] = copy.deepcopy(rows[0])
            panels.append(item)
        require(len(market) == 4, 'Exactly four measured market tasks per stage')
        result.append(dict(method=label, instance=method, checkpoint_stage=stage,
            source_updates=0, causal_updates=0 if stage == 'initial' else 2048,
            measurement_reused_for_v13_comparison=False, table_source=source,
            market_rows=market, synthetic_panels=panels, selection_policy=POLICY))
    return result


def quality_row(table, version, leader, stage):
    label = 'CPB-v%d.market-01' % version + ('.init' if stage == 'initial' else '')
    if leader['dataset_id'] == 'MKT-HIST-2':
        rows = [r for r in table['market_rows'] if r['method'] == label and r['task'] == leader['task']]
    else:
        require(leader['dataset_id'] == 'TEMPO-5', 'Only declared quality populations')
        panels = [p for p in table['synthetic']['panels']
                  if p['view'] == leader['view'] and p['input_policy'] == leader['input_policy']]
        require(len(panels) == 1, 'Unique saved timing panel')
        rows = [r for r in panels[0]['rows'] if r['method'] == label]
    require(len(rows) == 1 and rows[0]['checkpoint_stage'] == stage, 'Unique saved quality stage')
    finite(rows[0][leader['metric']])
    return rows[0]


def reconstruction_card(table, version, stage, source):
    queries = {}
    for split in ('train', 'validation'):
        query = table['fixed_queries'][stage][split]
        keys = ('standardized_fixed_query_MAE', 'Gaussian_NLL_F64', 'valid_rows', 'target_cells')
        queries[split] = {key: finite(query[key]) for key in keys}
        for key in ('mean_Huber_F64', 'variance_NLL_F64', 'joint_Gaussian_NLL_F64', 'total_objective_F64'):
            if key in query:
                queries[split][key] = finite(query[key])
    return dict(instance='CPB-v%d.market-01' % version, checkpoint_stage=stage,
        causal_updates=0 if stage == 'initial' else 2048, source_updates=0,
        native_width=40 if version >= 5 else 36, table_source=source, queries=queries,
        selection_policy='All queries retained; descriptive final-only scope leader, no checkpoint selection')


def moment_admission(audit, learning, established):
    """Join already audited saved scalars; never read moment/gradient tensors."""
    checked = audit['temporal_pooling_saved_learning']['temporal_moment_saved_learning']
    require(checked['status'] == 'passed' and not checked['mismatches']
            and type(checked['checks']) is int and checked['checks'] > 0
            and checked['moment_subgroup_rows_checked'] == learning['moment_subgroup_rows'] == 16384
            and checked['whole_group_two_moment_partition_rows_checked'] == 8192
            and checked['whole_group_partition_squared_norms_and_counts_verified'] is True
            and checked['manual_F32_absolute_tolerance'] == checked['manual_F32_relative_tolerance'] == 3e-5
            and checked['per_scalar_gradient_proof_claimed'] is False
            and checked['other2046_raw_gradients_reconstructed'] is False
            and checked['pool_centered_variance_is_native_uncertainty'] is False,
            'Nested saved moment audit: intermediate mean/dispersion; no full-gradient or native-uncertainty claim')
    events = ('weight_nonzero_data_gradient_steps', 'bias_nonzero_data_gradient_steps',
              'weight_movement_steps', 'bias_movement_steps')
    require(len(learning['moment_subgroups']) == len(checked['moment_subgroup_event_counts']) == 8,
            'Eight ordered moment subgroup records')
    for slot, group in enumerate(learning['moment_subgroups']):
        original_bin, half = divmod(slot, 2)
        require(group['bin'] == original_bin and group['moment'] == ('mean' if half == 0 else 'regularized_stddev')
                and [group['export_row_begin'], group['export_row_end']] ==
                [8*original_bin+4*half, 8*original_bin+4*(half+1)]
                and all(type(group[key]) is int and 0 < group[key] <= 2048
                        and group[key] == checked['moment_subgroup_event_counts'][slot][key] for key in events)
                and group['final_weight_changed'] is True and group['final_bias_changed'] is True,
                'Four-row subgroup events/final changes; whole8 event flags use OR, not summed booleans')
    witnesses = checked['actual_same_forward_moment_witnesses']
    require(len(witnesses) == 2 and [item['update'] for item in witnesses] == [1,2048],
            'Only actual first/last same-forward moment/local-adjoint witnesses')
    parameters = {'temporal_export_weight', 'temporal_export_bias', 'temporal_pool_positions'}
    components = audit['actual_pre_step_composite_objective']['witnesses']
    raw = audit['temporal_pooling_saved_learning']['actual_raw_array_witnesses']
    for index, item in enumerate(witnesses):
        require(item['same_forward_only'] is True and item['total_component_archive_fields'] == 60
                and item['new_temporal_trace_fields'] == 25 and item['parameter_array_fields'] == 12
                and item['new_regularized_dispersion_trace_fields'] == 4
                and item['regularized_sqrt_uses_actual_saved_root'] is True
                and item['attention_dispersion_score_and_position_routes_live'] is True
                and item['upstream_stage'] == 'PRECLIP-post-empty-export'
                and len(item['upstream_gradient_defined_by_route']) == 2
                and all(type(flag) is bool for flag in item['upstream_gradient_defined_by_route'])
                and 0 < finite(item['recorded_clip_factor_applied_once']) <= 1
                and item['local_export_and_positions_adjoint_verified'] is True
                and item['local_adjoint_absolute_tolerance'] == 1e-9
                and item['local_adjoint_relative_tolerance'] == 3e-4
                and item['postclip_export_nonvacuity_threshold'] == 1e-6
                and finite(item['postclip_export_expected_maximum']) > 1e-6
                and item['full_hidden_encoder_or_global_norm_reconstruction'] is False
                and item['matched_pre_step_decoder_adjoint_replay'] is False,
                'Actual60-field PRECLIP/oneclip three-parameter adjoint admission and declared nonvacuity')
        require(all(item['artifact'][key] == components[index]['witness'][key] == raw[index]['artifact'][key]
                    for key in ('bytes','sha256')), 'Same actual archive across three saved audit modules')
        errors = item['local_parameter_gradient_error_summaries']
        require(set(errors) == parameters, 'Exactly three tight actual local parameter parity gates')
        for name, row in errors.items():
            require(set(row) == {'maximum_expected_absolute','maximum_actual_absolute',
                                 'maximum_absolute_error','maximum_scaled_error'}
                    and all(type(value) in (int,float) and math.isfinite(value) and value >= 0 for value in row.values())
                    and row['maximum_scaled_error'] <= 1, 'Passed saved local-gradient summary: '+name)
        gauge = item['common_score_bias_gauge_descriptive']
        require(gauge['parity_gate'] is False and gauge['nonzero_gradient_required'] is False
                and all(type(gauge[key]) in (int,float) and math.isfinite(gauge[key])
                        for key in ('actual','reference','absolute_error')),
                'Common score bias is finite descriptive gauge evidence, not a tight parity/nonzero gate')
    summary = dict(all_eight_temporal_moment_subgroups_verified=True,
        whole8_two4_squared_norms_and_element_counts_verified=True,
        actual_first_last_local_moment_adjoints_verified=True,
        actual_local_moment_adjoint_parameter_tensors=sorted(parameters),
        actual_local_moment_adjoint_absolute_tolerance=1e-9, actual_local_moment_adjoint_relative_tolerance=3e-4,
        each_actual_local_export_nonvacuity_threshold=1e-6, common_score_bias_gauge_tight_parity_required=False,
        actual_component_archive_fields=60, same_forward_temporal_fields=25,
        new_regularized_dispersion_trace_fields=4, regularized_sqrt_uses_actual_saved_root=True,
        attention_dispersion_score_and_position_routes_live=True, raw_root_absent_offset_exception=True,
        local_hidden_adjoint_array_available=False,
        score_to_hidden_route_reconstructed=False, hidden_reference_is_observed_gradient_parity=False,
        full_encoder_gradient_or_global_norm_reconstructed=False)
    require(all(established[key] == value for key,value in summary.items()),
            'Separate independent establishment agrees with bounded moment evidence')
    return checked


def run(args):
    require(sys.platform.startswith('linux') and pathlib.Path('/opt/cuwacunu/research').is_dir(),
            'Existing Linux research container only')
    require(args.output.is_absolute() and args.output.resolve().is_relative_to(FINALIZATION_ROOT)
            and args.output.resolve() != FINALIZATION_ROOT, 'New exclusive finalization leaf only')
    card = json.loads(args.card.read_text(encoding='utf-8-sig'))
    require(card['kind'] == 'cpb_v13_metadata_only_finalization_card_v1', 'Own finalization card required')
    source = args.card.parent
    own = pin(pathlib.Path(__file__))
    require(own['bytes'] == card['generator']['bytes'] and own['sha256'] == card['generator']['sha256'],
            'Frozen generator identity')
    inputs, bindings, by_id = {}, [], {}
    for row in card['inputs']:
        require(row['id'] not in inputs, 'Unique metadata input IDs')
        path = bounded(source, row['staged_path'])
        actual = pin(path, row['path'])
        require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'Input hash mismatch: ' + row['id'])
        inputs[row['id']] = (json.loads(path.read_text(encoding='utf-8-sig'))
                             if row.get('format', 'json') == 'json' else path.read_text(encoding='utf-8-sig'))
        bindings.append(actual)
        by_id[row['id']] = actual
    require(set(inputs) == REQUIRED_INPUTS, 'Exact prospective metadata input set')
    require(by_id['prior_snapshot']['sha256'] == card['prior_snapshot_manifest_sha256'], 'Frozen prior-live manifest')
    require(by_id['experiment_card']['sha256'] == card['prospective_encoder_card_sha256'], 'Prospective experiment unchanged')
    require(by_id['descriptor']['sha256'] == card['prospective_descriptor_sha256'], 'Prospective descriptor immutable')
    snapshot_files = inputs['prior_snapshot']['files']
    require(len(snapshot_files) == 6 and {p['path'].replace('\\', '/') for p in snapshot_files} == {
        'AGENTS.md', 'doc/CURRENT_STATUS.md', 'doc/EMBEDDING_VERSIONS.md',
        'doc/clean_causal_encoder_research_ledger.json',
        'code/encoders/causal_patch_bottleneck_mae/README.md', FAMILY+'current.json'},
        'New six-file prior-live snapshot preserves ledger/catalog and whole four front-note tails')
    for input_id, suffix in (('ledger', 'doc/clean_causal_encoder_research_ledger.json'),
                             ('catalog', FAMILY + 'current.json')):
        matches = [p for p in snapshot_files if p['path'].replace('\\', '/').endswith(suffix)]
        require(len(matches) == 1 and all(matches[0][k] == by_id[input_id][k] for k in ('bytes', 'sha256')),
                'Prior-live snapshot binds ' + input_id)
    old, catalog = inputs['ledger'], inputs['catalog']
    require(old['baseline'] == BASELINE and catalog['active_application'] == BASELINE,
            'Protected V3 baseline/application')
    require(catalog['fresh_research_candidate']['instance'] == 'CPB-v12.market-01', 'Whole V12 current entry available')
    require(not any(c['instance'] == METHOD for c in old['candidates']), 'V13 not already finalized')
    tables = {version: inputs['v%d_table' % version] for version in VERSIONS}
    for version in (6,7,8,9,10,11,12):
        established = inputs['v%d_establishment' % version]
        require(established['kind'] == 'causal_fresh_v%d_postmeasurement_establishment_v1' % version
                and established['status'] == 'passed' and established['measured_candidate_established'] is True
                and established['trust_admission_passed'] is True and established['full_coverage'] is True
                and tables[version]['runtime_trust_admission_passed'] is True
                and tables[version]['trust_protocol_admission_passed'] is False,
                'Preserved V6/V7/V8/V9/V10/V11/V12 raw pending trust with separate completed establishment')
        require(all(established['evidence']['table_rows'][k] == by_id['v%d_table' % version][k]
                    for k in ('bytes','sha256')), 'Separate establishment binds exact reused clean table')
        if version in (8,9,11,12):
            require(established['initial_partial_native20_40_pairings_passed'] is True
                    and established['initial_pairing_native_coordinate_slice'] == [20,40]
                    and established['whole_native40_initial_equality_claimed'] is False,
                    'V8/V9/V11/V12 historical partial20:40 pairing semantics retained')
        else:
            require(established['initial_full40_pairings_passed'] is True,
                    'V6/V7/V10 historical whole40 pairing semantics retained')
        require(established['initial_pairing_count'] == 13, 'Historical13 declared references retained')
    for version in VERSIONS:
        table = tables[version]
        require(table['model'] == 'CPB-v%d.market-01' % version and table['status'] == 'complete'
                and table['source_updates'] == 0 and table['full_coverage'] is True,
                'Complete exact clean saved table identity')
    for input_id, control in [('v%d_table' % v, inputs['experiment_card']['comparison_control']['v%d' % v])
                              for v in PRIOR_VERSIONS] + [
                                  ('v6_establishment', inputs['experiment_card']['comparison_control']['v6_establishment']),
                                  ('v7_establishment', inputs['experiment_card']['comparison_control']['v7_establishment']),
                                  ('v8_establishment', inputs['experiment_card']['comparison_control']['v8_establishment']),
                                  ('v9_establishment', inputs['experiment_card']['comparison_control']['v9_establishment']),
                                  ('v10_establishment', inputs['experiment_card']['comparison_control']['v10_establishment']),
                                  ('v11_establishment', inputs['experiment_card']['comparison_control']['v11_establishment']),
                                  ('v12_establishment', inputs['experiment_card']['comparison_control']['v12_establishment'])]:
        require(all(by_id[input_id][k] == control[k] for k in ('bytes', 'sha256')),
                'Exact prospective preserved comparison-control binding: ' + input_id)
    comparison, training = inputs['comparison'], inputs['training']
    conditions = comparison['numeric_conditions']
    require(len(conditions) == comparison['numeric_total'] == 22 and len({r['id'] for r in conditions}) == 22,
            'Original unique22 numeric conditions, no added floor')
    declared_conditions = inputs['experiment_card']['advancement']['criteria']
    require(len(declared_conditions) == 22 and
            [{key: row[key] for key in declared} for row, declared in zip(conditions, declared_conditions)]
            == declared_conditions, 'Exact prospective condition identities, floors, sources and rules')
    goal_pin = inputs['experiment_card']['advancement']['goal_protocol']
    require(all(comparison['goal_protocol'][k] == goal_pin[k] for k in ('bytes', 'sha256')),
            'Exact reviewed strongest-V3-through-V11 retention protocol')
    require(goal_pin['sha256'] == '866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b',
            'Current prospective22 thresholds retained; no retrospective new floor')
    require(all(type(row['passed']) is bool for row in conditions), 'Boolean numeric outcomes')
    numeric_passed = sum(row['passed'] for row in conditions)
    require(comparison['numeric_passed'] == numeric_passed, 'Actual numeric count derived from22 records')
    require(comparison['candidate'] == METHOD and comparison['runtime_trust_admission_passed'] is True
            and comparison['trust_admission_passed'] is False and comparison['overall_establishment_passed'] is False
            and comparison['advancement_conditions_passed'] is False, 'Producer runtime-only pending trust preserved')
    require(all(value == 'pending' for value in comparison['mandatory_postmeasurement_trust'].values())
            and set(comparison['mandatory_postmeasurement_trust']) == {'initial_native20_40_and_statistics_pairings', 'independent_training_audit', 'independent_quality_audit'},
            'Producer mandatory postmeasurement trust remains pending')
    objective(comparison, 'Producer comparison')
    objective(training, 'Training summary')
    temporal(comparison, 'Producer comparison')
    temporal(training, 'Training summary')
    require(inputs['experiment_card']['scientific_recipe_parent'] ==
            'CPB-v11 source only;fresh weights/optimizer/statistics,not trained ancestry'
            and inputs['experiment_card']['scoped_reference'] == 'CPB-v11.market-01'
            and comparison['scientific_recipe_parent'] == 'CPB-v11.market-01'
            and comparison['scientific_recipe_parent_is_source_only'] is True
            and comparison['scoped_reference'] == 'CPB-v11.market-01',
            'Fresh V11 source recipe and scoped V11 SOURCE; latest measured V12 are distinct from selected V3')
    require(training['new_causal_updates'] == 2048
            and type(training['attempted_batches']) is int and 2048 <= training['attempted_batches'] <= 4096
            and training['source_updates'] == 0 and training['batch_size'] == 8
            and training['raw_load_calls'] == 512 + training['attempted_batches'] + 36,
            'One fresh2048/B8 trajectory; calibration+actual attempted loads+fixed queries')
    require(training['parameters'] == 298640 and 294751 <= training['learned_tensor_value_capacity_with_recorded_movement_events'] <= 294752
            and training['fixed_prior_parameters'] == 3888 and training['native_width'] == 40, 'Actual native40 learned/fixed partition')
    require(training['typed_resume_512_exact'] and training['fixed_prior_scaler_normalizer_unchanged']
            and training['backward_tensor'] == 'objective.total_loss'
            and training['encoder_gradient_kind'] == ENCODER_GRADIENT
            and training['shared_global_clip'] is True and training['optimizer_independence_claimed'] is False
            and training['all_mandatory_identifiable_tensors_learned'] is True
            and training['mandatory_identifiable_tensor_value_capacity'] == 294751
            and training['mandatory_identifiable_count_semantics'] == 'tensor numel capacity, not every-scalar gradient/movement proof'
            and training['per_scalar_nonzero_gradient_proof_claimed'] is False,
            'Actual typed continuation/frozen state/composite backward/V6 global clip/75 mandatory tensors')
    require(training['all_four_temporal_export_row_groups_learned'] is True
            and training['all_eight_temporal_moment_subgroups_learned'] is True
            and training['all_four_pool_position_contrasts_learned'] is True
            and training['all_three_between_bin_position_directions_learned'] is True
            and training['final_between_bin_position_projection_changed'] is True
            and all(training['temporal_pooling_learning'][k] == by_id['temporal_learning'][k] for k in ('bytes','sha256')),
            'Source-bound actual four-group/centered-contrast learning receipt')
    objective(inputs['initialization'], 'Fresh initialization', require_gradient_policy=False)
    temporal(inputs['initialization'], 'Fresh initialization')
    for name in ('engineering', 'initial_proof', 'final_proof'):
        objective(inputs[name], name)
        temporal(inputs[name], name)
    require(inputs['gradient']['status'] == 'passed' and inputs['engineering']['status'] == 'passed',
            'Completed actualCUDA mechanism and full-family engineering evidence')
    require(inputs['engineering']['typed_negative_cases'] ==
            inputs['experiment_card']['engineering_typed_negative_cases'] == 636,
            'Actual compiled strict String transform and tau Double[1] five-node negative cases')
    objective(inputs['gradient'], 'Mechanism proof')
    for name, updates in (('initial_proof', 0), ('final_proof', 2048)):
        proof = inputs[name]
        require(proof['status'] == 'passed' and proof['all_cutoffs'] == 128 and proof['suffix_cases'] == 2350
                and proof['internal_temporal_cases'] == 30 and proof['state_unchanged']
                and proof['causal_updates'] == updates and proof['source_updates'] == 0,
                'Complete saved-stage causal proof bound to actual0/2048 and source0')
    require(inputs['gradient']['shared_clip_coupling_witness'] is True,
            'Actual nonzero CUDA shared-clip composite-objective fixture')
    for key, expected in inputs['experiment_card']['mechanism_proof_contract'].items():
        if type(expected) is bool:
            require(inputs['gradient'][key] is expected, 'Actual source-bound unchanged V11 objective mechanism predicate: '+key)
    temporal_fixture = inputs['temporal_proof']
    require(temporal_fixture['direct_pool_kernel_calls'] == 51
            and temporal_fixture['F64_finite_difference_coordinates'] == temporal_fixture['F32_finite_difference_coordinates'] == 18
            and temporal_fixture['isolated_autograd_grad_calls'] == 4
            and temporal_fixture['transform_derivative_upper_bound'] == 16
            and all(temporal_fixture[key] is True for key in (
                'mean4_regularized_stddev4_export_layout',
                'full_attention_dependent_regularized_moment_adjoints',
                'live_dispersion_score_gradient',
                'raw_variance_root_dispersion_trace',
                'zero_tiny_large_analytic_dispersion_adjoints',
                'supported_nonfinite_and_variance_overflow_refused',
                'absent_local_huge_singleton_sanitized',
                'captured_F32_residual_correction',
                'raw_root_absent_offset_exception')), 'Exact attention-coupled regularized full-expression51/18/18/4 proof scope')
    for key, expected in inputs['experiment_card']['temporal_pooling_proof_contract']['required_proof_fields'].items():
        require(temporal_fixture[key] == expected, 'Actual source-bound temporal mechanism predicate: '+key)
    require(all(inputs['establishment'][key] is True for key in (
                'mean4_regularized_stddev4_export_layout',
                'full_attention_dependent_regularized_moment_adjoints',
                'live_dispersion_score_gradient',
                'raw_variance_root_dispersion_trace',
                'zero_tiny_large_analytic_dispersion_adjoints',
                'supported_nonfinite_and_variance_overflow_refused',
                'absent_local_huge_singleton_sanitized',
                'captured_F32_residual_correction',
                'raw_root_absent_offset_exception')), 'Separate establishment preserves all nine actual regularized-dispersion proof flags')
    require(inputs['establishment']['temporal_pooling_direct_kernel_calls'] == 51
            and inputs['establishment']['temporal_pooling_F64_finite_difference_coordinates'] == 18
            and inputs['establishment']['temporal_pooling_F32_finite_difference_coordinates'] == 18
            and inputs['establishment']['temporal_pooling_isolated_autograd_grad_calls'] == 4
            and inputs['establishment']['engineering_typed_negative_cases'] == 636,
            'Separate establishment preserves actual51/18/18/4 fixture scope')
    learning = inputs['temporal_learning']
    require(learning['status'] == 'passed' and learning['updates'] == 2048 and learning['group_rows'] == 8192
            and learning['extra_encoder_calls'] == 0 and learning['weight_decay_counted_as_data_gradient'] is False
            and learning['gauge']['name'] == TEMPORAL_FIELDS['data_gradient_gauge_exemption'],
            'Actual four temporal groups and separately observed gauge')
    between = learning['between_bin_positions']
    require(between['basis'] == 'hadamard-full16-signs-div4-v1'
            and between['basis_signs'] == [[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]
            and between['full_position_divisor'] == between['position_bin_width'] == 4
            and between['rows'] == 2048 and len(between['gradient_steps']) == len(between['movement_steps']) == 3
            and all(type(n) is int and 0 < n <= 2048 for n in between['gradient_steps']+between['movement_steps'])
            and len(between['final_contrast_changes']) == 3
            and all(type(value) in (int,float) and math.isfinite(value) for value in between['final_contrast_changes'])
            and finite(between['final_projection_change_l2']) > 0
            and between['weight_decay_counted_as_data_learning'] is False and between['extra_encoder_calls'] == 0,
            'Three direction gradient/movement EVENTS; signed final coefficients descriptive, positive overall final projection')
    contract = card['evidence_contract']
    require(set(contract['assertions']) == set(ASSERTIONS), 'Explicit complete postmeasurement trust assertions')
    for name, expected in ASSERTIONS.items():
        assertion = contract['assertions'][name]
        expected_input = ('pairings' if name.startswith('pairing_') else
                          'training_audit' if name.startswith('training_audit_') else
                          'quality_audit' if name.startswith('quality_audit_') else 'establishment')
        require(assertion['input'] == expected_input, 'Assertion bound to its actual independent receipt')
        require(assertion['equals'] == expected and type(assertion['equals']) is type(expected), 'Frozen assertion value')
        actual = at(inputs[assertion['input']], assertion['path'])
        require(actual == expected and type(actual) is type(expected), 'Required independent evidence: ' + name)
    establishment_bindings = at(inputs['establishment'], contract['establishment_bindings_path'])
    establishment_ids = {'experiment_card', 'comparison', 'v13_table', 'pairings', 'training_audit', 'quality_audit',
        'source_manifest', 'build', 'mirror', 'quality_receipt', 'temporal_proof', 'temporal_learning',
        'prospective_freeze', 'protocol_preflight'}
    require(set(contract['establishment_binding_keys']) == establishment_ids, 'Exact independent establishment hash aliases')
    for input_id in sorted(establishment_ids):
        bound = establishment_bindings[contract['establishment_binding_keys'][input_id]]
        require(all(bound[k] == by_id[input_id][k] for k in ('bytes', 'sha256')),
                'Separate establishment receipt binds ' + input_id)
    audit_counts = {}
    for name in ('training_audit', 'quality_audit'):
        count = at(inputs[name], contract[name + '_checks_path'])
        require(type(count) is int and count > 0, 'Actual independent audit check count')
        audit_counts[name] = count
    training_audit_modules = {}
    for name in ('numeric_statistics', 'saved_tensor_state', 'manual_saved_Gaussian',
                 'actual_pre_step_composite_objective', 'temporal_pooling_saved_learning'):
        module = inputs['training_audit'][name]
        count = module['checks']
        require(module['status'] == 'passed' and type(count) is int and count > 0 and not module['mismatches'],
                'Actual named independent training-audit module checks')
        training_audit_modules[name] = count
    training_audit_total = audit_counts['training_audit'] + sum(training_audit_modules.values())
    require(not inputs['training_audit']['mismatches'] and not inputs['quality_audit']['mismatches'],
            'No independent core/quality audit mismatches')
    audited_training = inputs['training_audit']
    require(audited_training['successful_CSV_rows'] == audited_training['independent_actual_CSV_stream_pairs'] == 2048
            and audited_training['CSV_rows'] == audited_training['attempted_batches'] == training['attempted_batches']
            and audited_training['skipped_ineligible_attempts'] == training['attempted_batches']-2048
            and audited_training['checkpoint_attempt_counters']['2048'] == training['attempted_batches'],
            'Actual independent successful/attempt/skipped accounting; no assumed zero skips')
    full_coverage = comparison['full_coverage'] is True
    require(full_coverage, 'Complete finite saved quality coverage required')
    require(inputs['quality_receipt']['status'] == 'passed', 'Completed saved quality execution')
    numeric_success = numeric_passed == 22
    advancement = numeric_success and full_coverage  # All independent trust assertions passed above.
    establishment = inputs['establishment']
    require(establishment['measured_candidate_established'] is True
            and establishment['saved_initial_final_endpoint_proofs_bound'] is True
            and establishment['historical_saved_control_count'] == 10
            and establishment['historical_separate_trust_join_count'] == 7
            and establishment['comparison_method_count'] == 12
            and establishment['comparison_market_rows'] == 48
            and establishment['initial_partial_native20_40_pairings_passed'] is True
            and establishment['initial_pairing_count'] == 13
            and establishment['initial_pairing_feature_archives'] == 8
            and establishment['initial_pairing_statistic_archives'] == 5
            and establishment['initial_pairing_native_coordinate_slice'] == [20,40]
            and establishment['whole_native40_initial_equality_claimed'] is False
            and establishment['independent_training_audit_passed'] is True
            and establishment['independent_quality_audit_passed'] is True,
            'Measured trusted candidate established separately from numeric advancement')
    require(establishment['numeric_passed'] == numeric_passed and establishment['numeric_total'] == 22
            and establishment['numeric_conditions_passed'] is numeric_success
            and establishment['overall_establishment_passed'] is advancement
            and establishment['advancement_conditions_passed'] is advancement
            and establishment['overall_winner'] is False
            and establishment['winner_selection_made'] is False,
            'Separate receipt agrees with actual criterion achievement; no winner or application selection')
    require(establishment['training_audit_core_checks'] == audit_counts['training_audit']
            and establishment['training_audit_five_module_checks'] == training_audit_modules
            and establishment['training_audit_total_checks'] == training_audit_total
            and establishment['quality_audit_checks'] == audit_counts['quality_audit'],
            'Separate join reconciles core plus five named training modules and quality checks')
    temporal(establishment, 'Separate completed establishment')
    moment_evidence = moment_admission(inputs['training_audit'], learning, establishment)
    require(establishment['mandatory_identifiable_tensor_value_capacity'] == 294751
            and establishment['weight_decay_counted_as_data_learning'] is False
            and establishment['all_four_export_row_groups_and_position_contrasts_verified'] is True
            and establishment['all_three_between_bin_position_directions_verified'] is True
            and establishment['between_bin_overall_final_projection_verified'] is True
            and establishment['individual_final_coefficient_nonzero_gate'] is False,
            'Tensor-capacity/gauge and four local/export plus three between-bin event semantics')
    admission = establishment['normal_prospective_source_admission']
    source_counts = inputs['experiment_card']['source_counts']
    source_count = source_counts['capsule_sources']
    owner_count = source_counts['family_inside'] + source_counts['family_external_header'] + source_counts['research']
    require(source_count == 143 and source_counts['binaries'] == 7
            and source_counts['translation_units'] == 11,
            'Current actual143-source/seven-binary/eleven-TU declaration;not historicalV12/137 closure')
    require(admission == dict(source_files=source_count, owner_source_files=owner_count,
            freeze_and_preflight_bound=True, normal_quality_all_stages_passed=True,
            scientific_model_head_or_training_replay=False),
            'Normal completed V13 quality, frozen source and no scientific replay')
    historical_recovery = inputs['experiment_card']['historical_V9_reporting_recovery']
    require(establishment['historical_V9_reporting_recovery_preserved'] == historical_recovery,
            'Historical V9 reporting recovery remains an opaque preserved context record')
    stage = inputs['source_manifest']
    require(len(stage['files']) == card['expected_source_capsule_files'] == source_count
            and len(inputs['build']['binaries']) == card['expected_binaries'] == source_counts['binaries']
            and len(inputs['build']['dependency_closures']) == card['expected_translation_units'] == source_counts['translation_units'],
            'Actual card-bound source/binary/translation-unit closure')
    require(stage['kind'] == 'causal_fresh_v13_source_capsule_v1'
            and stage['card_sha256'] == by_id['experiment_card']['sha256']
            and stage['container_root'] == card['container_attempt'], 'Exact admitted source/card/attempt identity')
    require(inputs['build']['source_files'] == source_count
            and all(inputs['build']['source_manifest'][k] == by_id['source_manifest'][k] for k in ('bytes','sha256')),
            'Actual build binds exact source manifest')
    freeze, preflight = inputs['prospective_freeze'], inputs['protocol_preflight']
    require(freeze['kind'] == 'causal_fresh_v13_prospective_freeze_v1' and freeze['status'] == 'passed'
            and all(freeze['card'][k] == by_id['experiment_card'][k] for k in ('bytes','sha256'))
            and freeze['owner_source_files'] == owner_count
            and freeze['critical_checkpoints_verified'] == 114
            and freeze['latest_v12_sources_verified'] == 137,
            'Actual prospective freeze binds same card and owner closure')
    require(preflight['kind'] == 'causal_fresh_v13_source_protocol_preflight_v1' and preflight['status'] == 'passed'
            and len(preflight['sources']) + 1 == source_count,
            'Actual prospective preflight uniquely closes capsule including its own receipt')
    require(inputs['mirror']['scope'] == 'Complete' and inputs['mirror']['source_files_verified'] == source_count
            and inputs['mirror']['all_mirrored_pt_ignored'] is True,
            'Actual completed normal-quality mirror and ignored PT evidence')
    capsule = pathlib.Path(card['container_attempt']) / 'source'
    for row in stage['files']:
        actual = pin(bounded(capsule, row['path']))
        require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Immutable source binding')
    for row in inputs['build']['binaries']:
        actual = pin(row['path'])
        require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Saved binary binding; never executed')
    checkpoint_pins = []
    for row in inputs['mirror']['files']:
        if '/run-001/checkpoints/' in row['container_path']:
            actual = pin(row['container_path'], row['path'])
            require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Saved canonical checkpoint availability')
            checkpoint_pins.append(actual)
    require(len(checkpoint_pins) == 5 and inputs['mirror']['all_mirrored_pt_ignored'], 'Five saved locally ignored checkpoints')
    require({pathlib.PurePosixPath(p['path']).name for p in checkpoint_pins} == {
        'inference-0.pt', 'inference-2048.pt', 'training-512.pt', 'training-1024.pt', 'training-2048.pt'},
        'Exact five checkpoint roles')
    descriptor = inputs['descriptor']
    require(descriptor['version'] == 'CPB-v13' and descriptor['instance'] == METHOD
            and descriptor['created_after_measurement'] is False
            and descriptor['created_before_build_training_or_quality'] is True
            and descriptor['immutable_prospective_semantics'] is True
            and all(descriptor['card'][k] == by_id['experiment_card'][k] for k in ('bytes', 'sha256')),
            'Existing prospective descriptor and exact frozen card, never regenerated')
    ledger = copy.deepcopy(old)
    new_stagecards = stagecards(tables[13], 13, by_id['v13_table'])
    require(len(old['stage_scorecards']) == 2*len(PRIOR_VERSIONS) and len(old['stage_leaders']) == 16, 'All chronological prior stages/16quality scopes')
    require({row['method'] for row in old['stage_scorecards']} == {
        'CPB-v%d.market-01' % v + suffix for v in PRIOR_VERSIONS for suffix in ('', '.init')}, 'All prior stage labels')
    replacements = []
    for leader in ledger['stage_leaders']:
        original = copy.deepcopy(leader)
        finals = []
        for version in VERSIONS:
            row = quality_row(tables[version], version, leader, 'final')
            finals.append(dict(method=row['method'], score=row[leader['metric']], checkpoint_stage='final',
                               table_source=by_id['v%d_table' % version]))
        old_best = max(finals[:-1], key=lambda row: row['score'])
        require((old_best['method'], old_best['score']) == (original['method'], original['score']),
                'Prior complete leader matches all ten protected saved final score tables')
        best = max(finals, key=lambda row: row['score'])  # Stable chronological tie keeps earlier.
        additions = [dict(method=row['method'], score=row[leader['metric']], checkpoint_stage=stage_name)
                     for stage_name in ('initial', 'final')
                     for row in (quality_row(tables[13], 13, leader, stage_name),)]
        leader.setdefault('stage_scores', []).extend(additions)
        better = best['score'] > original['score']
        ledger.setdefault('leader_history', []).append(dict(preserved_before=METHOD + ' measured',
            record=original, displaced=better, successor_method=best['method'], successor_score=best['score']))
        if better:
            change = dict(dataset_id=leader['dataset_id'], task=leader['task'], view=leader['view'],
                input_policy=leader.get('input_policy'), metric=leader['metric'], former_method=original['method'],
                former_score=original['score'], new_method=best['method'], new_score=best['score'],
                strictly_better=True, former_complete_record=original)
            ledger.setdefault('displaced_leaders', []).append(change)
            replacements.append(change)
            leader.update(method=best['method'], score=best['score'], initial_and_final=copy.deepcopy(additions))
            selected = quality_row(tables[13], 13, leader, 'final')
            if 'recipe_id' in selected:
                leader['recipe_id'] = selected['recipe_id']
        leader['all_final_scores'] = finals
        leader['selection_policy'] = POLICY
    ledger['stage_scorecards'].extend(new_stagecards)
    reconstruction = copy.deepcopy(old['reconstruction_stage_scorecards'])
    require(len(reconstruction) == 2*len(PRIOR_VERSIONS), 'All prior reconstruction stagecards retained')
    require([(r['instance'], r['checkpoint_stage']) for r in reconstruction] == [
        ('CPB-v%d.market-01' % v, s) for v in PRIOR_VERSIONS for s in ('initial', 'final')],
        'Chronological reconstruction order preserves earlier exact ties')
    for version in PRIOR_VERSIONS:
        for stage_name in ('initial', 'final'):
            expected = reconstruction_card(tables[version], version, stage_name, by_id['v%d_table' % version])
            actual = [r for r in reconstruction if r['instance'] == expected['instance'] and r['checkpoint_stage'] == stage_name]
            require(len(actual) == 1 and actual[0]['queries'] == expected['queries'], 'Prior reconstruction data exact')
    reconstruction.extend(reconstruction_card(tables[13], 13, s, by_id['v13_table']) for s in ('initial', 'final'))
    ledger['reconstruction_stage_scorecards'] = reconstruction
    ledger.setdefault('reconstruction_leader_history', []).append(dict(
        preserved_before=METHOD + ' measured', records=copy.deepcopy(old['reconstruction_stage_leaders'])))
    reconstruction_leaders = []
    for split in ('train', 'validation'):
        for metric in ('standardized_fixed_query_MAE', 'Gaussian_NLL_F64'):
            saved = [dict(method=r['instance'], score=finite(r['queries'][split][metric]),
                          checkpoint_stage='final', table_source=r['table_source'])
                     for r in reconstruction if r['checkpoint_stage'] == 'final']
            winner = min(saved, key=lambda row: row['score'])
            reconstruction_leaders.append(dict(dataset_id='MKT-HIST-2', scope='unchanged saved masked fixed query',
                split=split, metric=metric, method=winner['method'], score=winner['score'], all_final_scores=saved,
                direction='lower_is_better', numeric_advancement_floor=not(split == 'train' and metric == 'Gaussian_NLL_F64'),
                descriptive_only=True, no_new_floor_added=True))
    ledger['reconstruction_stage_leaders'] = reconstruction_leaders
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    candidate_stage = ('complete_measured_stronger_goal22_passed_not_promoted' if advancement
                       else 'complete_stopped_failed_stronger_goal22')
    candidate = dict(instance=METHOD, stage=candidate_stage, pretrained_ancestor='none', source_updates=0,
        new_causal_updates=2048, gradient_policy=GRADIENT, **OBJECTIVE_FIELDS, **TEMPORAL_FIELDS,
        shared_global_clip=True, optimizer_independence_claimed=False,
        card=by_id['experiment_card'], report='doc/CAUSAL_FRESH_V13_RESULTS.md',
        review='doc/CAUSAL_FRESH_V13_REVIEW.md', detail='doc/CAUSAL_FRESH_V13_DETAIL.md',
        instance_record=FAMILY + 'instances/CPB-v13.market-01.json', comparison=by_id['comparison'],
        table_rows=by_id['v13_table'], baseline=BASELINE, baseline_changed=False, full_coverage=True,
        runtime_trust_admission_passed=True, trust_admission_passed=True, measured_candidate_established=True,
        overall_establishment_passed=advancement,
        postmeasurement_establishment=by_id['establishment'], initial_native20_40_and_statistics_pairings=by_id['pairings'],
        initial_pairing_native_coordinate_slice=[20,40], whole_native40_initial_equality_claimed=False,
        temporal_pooling_proof=by_id['temporal_proof'], temporal_pooling_learning=by_id['temporal_learning'],
        temporal_moment_saved_learning=copy.deepcopy(moment_evidence),
        all_eight_temporal_moment_subgroups_learned=True,
        scientific_recipe_parent='CPB-v11.market-01', scientific_recipe_parent_is_source_only=True,
        scoped_reference='CPB-v11.market-01',
        mandatory_identifiable_tensor_value_capacity=294751,
        mandatory_identifiable_count_semantics=training['mandatory_identifiable_count_semantics'],
        per_scalar_nonzero_gradient_proof_claimed=False,
        all_three_between_bin_position_directions_learned=True,
        final_between_bin_position_projection_changed=True,
        individual_final_coefficient_nonzero_gate=False,
        normal_prospective_source_admission=copy.deepcopy(admission),
        historical_V9_reporting_recovery=copy.deepcopy(historical_recovery),
        prospective_freeze=by_id['prospective_freeze'], protocol_preflight=by_id['protocol_preflight'],
        numeric_passed=numeric_passed, numeric_total=22, advancement_passed=advancement,
        failed_conditions=[r['id'] for r in conditions if not r['passed']],
        gate_scope='V13 prospective strongest-final-V3-through-V11 retention22; historical counts retain their own cards',
        quality='Preserve each saved scoped gain and failure; initial stages descriptive; no automatic application selection',
        overall_winner=False, application_changed=False,
        human_report_assets={name: by_id[name] for name in ('results', 'review', 'detail')})
    ledger['candidates'].append(candidate)
    ledger.setdefault('direction_history', []).append(copy.deepcopy(old['next_direction']))
    ledger['next_direction'] = dict(status=card['next_direction']['status'], action=card['next_direction']['action'],
        goal_active=not advancement, goal_criteria_satisfied=advancement,
        validation_in_diagnosis=False, encoder_calls=0, head_fits=0, head_tuning=False,
        later_training_authorized_by_this_finalization=False, exact_V13_recipe_complete=True,
        exact_V13_recipe_stopped=not advancement)
    ledger.setdefault('stage_leader_update_history', []).append(copy.deepcopy(old['stage_leader_update']))
    ledger['stage_leader_update'] = dict(candidate=METHOD, current_leader_scopes=16,
        strictly_improved_scopes=len(replacements), earlier_scopes_retained=16-len(replacements),
        all_prior_records_preserved=True, stage_scorecards=len(ledger['stage_scorecards']), policy=POLICY, prior_live_ledger=by_id['ledger'])
    ledger['updated_utc'] = now
    for key in ('candidates', 'stage_scorecards', 'leader_history', 'displaced_leaders', 'candidate_status_history',
                'direction_history', 'stage_leader_update_history', 'reconstruction_stage_scorecards',
                'reconstruction_leader_history'):
        require(ledger[key][:len(old[key])] == old[key], 'Old complete history preserved: ' + key)
    current = copy.deepcopy(catalog)
    current.setdefault('fresh_research_candidate_history', []).append(copy.deepcopy(catalog['fresh_research_candidate']))
    current['fresh_research_candidate'] = copy.deepcopy(candidate)
    current['fresh_research_candidate'].update(design_file='CPB-v13.json', physical_version_root='v13',
                                              measured_instance_file='instances/CPB-v13.market-01.json')
    current['date'] = card['client_date']
    require(current['active_application'] == BASELINE
            and current['clean_causal_training_baseline'] == catalog['clean_causal_training_baseline'],
            'V3 baseline/application exact')
    instance = dict(kind='causal_encoder_measured_instance_v1', date=card['client_date'], version='CPB-v13',
        instance=METHOD, initial_quality_label=METHOD+'.init', protocol='causal-fresh-v13',
        procedure_status=candidate_stage, clean_causal_candidate_measured=True,
        measured_candidate_established=True, overall_establishment_passed=advancement,
        clean_baseline=BASELINE, clean_baseline_unchanged=True, application_selection=BASELINE,
        application_selection_changed=False, historical_reference_is_quality_gate=False,
        overall_winner=False, advancement_passed=advancement, new_research_published=False,
        version_root=FAMILY+'v13', prospective_descriptor=by_id['descriptor'], prospective_descriptor_rewritten=False,
        physical_owned_family_files=card['physical_owned_files']['family'],
        physical_owned_research_files=card['physical_owned_files']['research'], frozen_source_capsule_files=source_count,
        source_manifest=by_id['source_manifest'], source_capsule=inputs['mirror']['preserved_source_packages'],
        build_admission=by_id['build'], binaries=inputs['build']['binaries'], card=by_id['experiment_card'],
        gradient_policy=GRADIENT, **OBJECTIVE_FIELDS, **TEMPORAL_FIELDS, encoder_gradient_kind=training['encoder_gradient_kind'],
        single_AdamW=True, shared_global_clip=True, optimizer_independence_claimed=False,
        temporal_pooling_learning=by_id['temporal_learning'], temporal_pooling_proof=by_id['temporal_proof'],
        temporal_moment_saved_learning=copy.deepcopy(moment_evidence),
        all_eight_temporal_moment_subgroups_learned=True,
        scientific_recipe_parent='CPB-v11.market-01', scientific_recipe_parent_is_source_only=True,
        scoped_reference='CPB-v11.market-01',
        mandatory_identifiable_tensor_value_capacity=294751,
        mandatory_identifiable_count_semantics=training['mandatory_identifiable_count_semantics'],
        per_scalar_nonzero_gradient_proof_claimed=False,
        weight_decay_counted_as_data_learning=False,
        training_objective='R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v+log(v))], unit coefficients',
        reporting_metrics='Unchanged saved Gaussian NLL/standardized and physical MAE; reporting NLL is never backpropagated',
        native_width=40, native_layout=card['native_layout'], initialization_kind=training['initialization_kind'],
        initialization_seed=943001, pretrained_ancestor='none', source_updates=0, inherited_learned_values=0,
        fresh_learned_values=294752, fresh_learned_tensors=76, analytic_prior_values=3888, total_parameters=298640,
        fresh_initialization=by_id['initialization'], fresh_calibration=by_id['calibration'],
        input_scaler_identity=training['fresh_scaler_identity'], bank_identity=training['fresh_bank_identity'],
        initial_native20_40_and_statistics_pairings=by_id['pairings'],
        initial_pairing_native_coordinate_slice=[20,40], whole_native40_initial_equality_claimed=False,
        new_causal_updates=2048, attempts=training['attempted_batches'], successful_updates=2048,
        skipped_ineligible_attempts=audited_training['skipped_ineligible_attempts'], batch_size=8,
        training_runs=1, training_replays=0, training_summary=by_id['training'], typed_resume=by_id['resume'],
        all_75_mandatory_identifiable_tensors_with_nonzero_CUDA_gradients_and_movement=True,
        data_learning_is_tensor_event_evidence_not_every_scalar=True,
        all_four_temporal_export_row_groups_learned=True, all_four_pool_position_contrasts_learned=True,
        all_three_between_bin_position_directions_learned=True, final_between_bin_position_projection_changed=True,
        individual_final_coefficient_nonzero_gate=False,
        fixed_prior_scaler_normalizer_unchanged=True, GPU_compute_seconds=training['GPU_step_seconds_synchronized'],
        training_trajectory_wall_seconds=training['training_trajectory_wall_seconds'], calibration_seconds=training['calibration_seconds'],
        timing_scope=training['GPU_step_timing_scope'], main_checkpoints=checkpoint_pins, checkpoints_local_git_ignored=True,
        initial_checkpoint=next(p for p in checkpoint_pins if p['path'].endswith('/inference-0.pt')),
        usable_final_checkpoint=next(p for p in checkpoint_pins if p['path'].endswith('/inference-2048.pt')),
        stable_clean_baseline_checkpoint=copy.deepcopy(old['baseline_checkpoint']), engineering_proof=by_id['engineering'],
        variance_gradient_proof=by_id['gradient'], initial_causal_proof=by_id['initial_proof'], final_causal_proof=by_id['final_proof'],
        causal_proof_scope=dict(cutoffs=128, suffix_cases_per_stage=2350, internal_temporal_cases_per_stage=30, state_unchanged=True),
        independent_training_audit=by_id['training_audit'], independent_training_audit_checks=audit_counts['training_audit'],
        independent_training_audit_core_checks=audit_counts['training_audit'],
        independent_training_audit_module_checks=training_audit_modules,
        independent_training_audit_core_plus_modules_checks=training_audit_total,
        independent_quality_audit=by_id['quality_audit'], independent_quality_audit_checks=audit_counts['quality_audit'],
        postmeasurement_establishment=by_id['establishment'], producer_pending_trust_preserved=True,
        final_quality_report=candidate['report'], review=candidate['review'], detail=candidate['detail'],
        human_report_assets=candidate['human_report_assets'], table_rows=by_id['v13_table'], comparison=by_id['comparison'],
        quality_execution_receipt=by_id['quality_receipt'], quality=dict(numeric_passed=numeric_passed, numeric_total=22,
            runtime_trust_admission_passed=True, trust_admission_passed=True, full_coverage=True,
            advancement_passed=advancement, failed_conditions=candidate['failed_conditions'], gate_scope=candidate['gate_scope'],
            strict_new_scoped_quality_leaders=len(replacements), initial_and_final_descriptive_scorecards_retained=True),
        next_direction=ledger['next_direction'], local_run_root='output/runs/causal-fresh-v13/run-001',
        local_quality_root='output/runs/causal-fresh-v13/run-001/quality-001',
        local_engineering_root='output/engineering/causal-fresh-v13/attempt-001', container_root=card['container_attempt'],
        normal_prospective_source_admission=copy.deepcopy(admission),
        historical_V9_reporting_recovery=copy.deepcopy(historical_recovery),
        prospective_freeze=by_id['prospective_freeze'], protocol_preflight=by_id['protocol_preflight'],
        preservation=dict(complete_mirror=by_id['mirror'], baseline_changed=False, V4_reference_intact=True,
                          V5_reference_intact=True, V6_reference_intact=True, V7_reference_intact=True, V8_reference_intact=True, V9_reference_intact=True, V10_reference_intact=True, V11_reference_intact=True, V12_reference_intact=True,
                          V6_separate_establishment=by_id['v6_establishment'],
                          V7_separate_establishment=by_id['v7_establishment'], V8_separate_establishment=by_id['v8_establishment'], V9_separate_establishment=by_id['v9_establishment'], V10_separate_establishment=by_id['v10_establishment'], V11_separate_establishment=by_id['v11_establishment'], V12_separate_establishment=by_id['v12_establishment'], prior_live_snapshot=by_id['prior_snapshot']))
    outputs = {'doc/clean_causal_encoder_research_ledger.json': ledger, FAMILY+'current.json': current,
               FAMILY+'instances/CPB-v13.market-01.json': instance}
    require(not args.output.exists(), 'Exclusive proposal output required')
    args.output.mkdir(parents=True)
    output_pins = []
    for relative, value in outputs.items():
        path = bounded(args.output, relative)
        write(path, value)
        output_pins.append(pin(path, relative))
    receipt = dict(kind='cpb_v13_metadata_only_finalization_proposal_v1', status='passed_proposal_not_applied',
        created_utc=now, generator=own, card=pin(args.card), inputs=bindings, outputs=output_pins,
        old_candidates_preserved=len(old['candidates']), old_stage_scorecards_preserved=len(old['stage_scorecards']),
        old_quality_leaders_preserved_in_full_history=16, strict_new_quality_leaders=replacements,
        new_stage_scorecards=2, total_stage_scorecards=len(ledger['stage_scorecards']), reconstruction_stage_scorecards=len(reconstruction),
        reconstruction_scopes=4, baseline_application_unchanged=True, V4_reference_preserved=True,
        V5_reference_preserved=True, V6_reference_preserved=True, V7_reference_preserved=True, V8_reference_preserved=True, V9_reference_preserved=True, V10_reference_preserved=True, V11_reference_preserved=True, V12_reference_preserved=True,
        prospective_descriptor_unchanged=by_id['descriptor'], producer_pending_trust_preserved=by_id['comparison'],
        separate_establishment_receipt=by_id['establishment'], numeric_passed=numeric_passed, numeric_total=22,
        training_audit_core_checks=audit_counts['training_audit'], training_audit_module_checks=training_audit_modules,
        training_audit_core_plus_modules_checks=training_audit_total,
        measured_candidate_established=True, overall_establishment_passed=advancement,
        advancement_conditions_passed=advancement, overall_winner=False, winner_selection_made=False, new_numeric_floors=0,
        initial_pairing_native_coordinate_slice=[20,40], whole_native40_initial_equality_claimed=False,
        temporal_pooling_learning=by_id['temporal_learning'], temporal_pooling_proof=by_id['temporal_proof'],
        temporal_moment_saved_learning=copy.deepcopy(moment_evidence),
        normal_prospective_source_admission=copy.deepcopy(admission),
        historical_V9_reporting_recovery=copy.deepcopy(historical_recovery),
        model_calls=0, head_fits=0, training_updates=0, live_files_written=0, publication=False,
        apply_required='Root independent review and explicit seven-file metadata/front-note application remain separate')
    write(args.output/'finalization-proposal.json', receipt)
    print(json.dumps(dict(status=receipt['status'], output=str(args.output), metadata_files=3,
                          numeric_passed=numeric_passed, numeric_total=22, strict_new_quality_leaders=len(replacements))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.resolve().is_relative_to(FINALIZATION_ROOT)
            and args.output.resolve() != FINALIZATION_ROOT, 'Bounded exclusive output before failure handling')
    require(not args.output.exists(), 'Preserve the existing output leaf; no replay or append')
    try:
        run(args)
    except Exception as error:
        # Preserve any partially written proposal; never reuse its leaf.
        if not args.output.exists():
            args.output.mkdir(parents=True)
        failure = args.output/'failure.json'
        if not failure.exists():
            write(failure, dict(kind='cpb_v13_metadata_only_finalization_failure_v1', status='failed_not_applied',
                error=type(error).__name__+': '+str(error), generator=pin(__file__), card=pin(args.card),
                model_calls=0, head_fits=0, live_files_written=0))
        raise


if __name__ == '__main__':
    main()

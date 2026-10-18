# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Separately review three saved V12 metadata proposals; never apply them.

This owned derivative imports no proposer, model or numerical array library.
Actual outcome and check counts come from the admitted saved evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('/opt/cuwacunu/research/causal-fresh-v12-finalization')
METHOD = 'CPB-v12.market-01'
BASELINE = 'CPB-v3.market-01'
FAMILY = 'code/encoders/causal_patch_bottleneck_mae/versions/'
VERSIONS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
HISTORY_KEYS = ('candidates', 'stage_scorecards', 'reconstruction_stage_scorecards',
    'leader_history', 'displaced_leaders', 'candidate_status_history', 'direction_history',
    'stage_leader_update_history', 'reconstruction_leader_history')
TEMPORAL = dict(temporal_pooling_policy='original-position-four-bin-attention-mean4-uniform-centered-variance4-v1',
    temporal_pooling_partition=[4,16,32,4,8],temporal_pooling_mix=[0.5,0.5],
    temporal_moment_policy='componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1',
    temporal_export_layout='per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1',
    temporal_moment_partition=[4,4],
    empty_local_bin_policy='zero-whole-export-slice-after-bias-v1',registered_learned_tensors=76,
    mandatory_identifiable_learned_tensors=75,
    data_gradient_gauge_exemption='core.encoder.level.base.backbone.pool_score.bias')
OBJECTIVE = dict(objective_policy='huber1-mean-plus-detached-error-gaussian-variance-v1',
    mean_huber_delta=1, loss_coefficients=[1, 1], variance_error_detached=True,
    reporting_nll_backpropagated=False,
    direct_gradient_routes='mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1')
GRADIENT = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=Path, required=True)
    parser.add_argument('--proposals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux') or not Path('/.dockerenv').is_file():
        raise RuntimeError('Existing Linux research container only')
    if (not args.output.is_absolute() or not args.output.resolve().is_relative_to(ROOT)
            or args.output.resolve() == ROOT or args.output.exists()):
        raise RuntimeError('New exclusive bounded metadata review output required')
    checks = 0

    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            raise RuntimeError(message)

    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def bound(path, binding):
        check(path.is_file() and not path.is_symlink(), 'Existing unlinked metadata file')
        data = path.read_bytes()
        check(len(data) == binding['bytes'] and hashlib.sha256(data).hexdigest() == binding['sha256'],
              'Exact metadata binding: ' + str(path))

    def pin(path):
        data = path.read_bytes()
        return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def at(record, selectors):
        check(isinstance(selectors, list) and bool(selectors), 'Explicit actual receipt selector')
        for key in selectors:
            record = record[key]
        return record

    def safe(root, relative):
        path = Path(relative)
        check(not path.is_absolute() and '..' not in path.parts and bool(path.parts), 'Relative bounded metadata path')
        result = root / path
        check(result.resolve().is_relative_to(root.resolve()), 'Metadata path remains inside leaf')
        return result

    card = read(args.card)
    check(card['kind'] == 'cpb_v12_metadata_only_finalization_card_v1', 'Own admitted metadata card')
    inputs, by_id = {}, {}
    for row in card['inputs']:
        check(row['id'] not in by_id, 'Unique input roles')
        path = safe(args.card.parent, row['staged_path'])
        bound(path, row)
        by_id[row['id']] = row
        if row['format'] == 'json':
            inputs[row['id']] = read(path)
    receipt = read(args.proposals / 'finalization-proposal.json')
    check(receipt['kind'] == 'cpb_v12_metadata_only_finalization_proposal_v1'
          and receipt['status'] == 'passed_proposal_not_applied', 'Unapplied successful proposal')
    check(all(receipt['card'][k] == pin(args.card)[k] for k in ('bytes', 'sha256')), 'Actual generator card binding')
    expected = {'doc/clean_causal_encoder_research_ledger.json', FAMILY+'current.json',
                FAMILY+'instances/CPB-v12.market-01.json'}
    check(len(receipt['outputs']) == 3 and {r['path'] for r in receipt['outputs']} == expected,
          'Exactly three declared metadata outputs')
    for row in receipt['outputs']:
        bound(safe(args.proposals, row['path']), row)
    ledger = read(args.proposals/'doc/clean_causal_encoder_research_ledger.json')
    current = read(args.proposals/(FAMILY+'current.json'))
    instance = read(args.proposals/(FAMILY+'instances/CPB-v12.market-01.json'))
    old = inputs['ledger']
    check(ledger['baseline'] == old['baseline'] == current['active_application'] == BASELINE,
          'Clean baseline/application retained')
    check(ledger['baseline_checkpoint'] == old['baseline_checkpoint'], 'Complete baseline checkpoint record retained')
    check(ledger['overall_winner'] == old['overall_winner'] is False, 'No automatic overall-winner declaration')
    for key in HISTORY_KEYS:
        check(ledger[key][:len(old[key])] == old[key], 'Complete historical prefix retained: '+key)
    check(len(ledger['candidates']) == len(old['candidates'])+1, 'Exactly one measured candidate appended')
    check([r['instance'] for r in ledger['candidates']].count(METHOD) == 1, 'One V12 instance')
    for row in old['stage_leaders']:
        check(any(entry == row or entry.get('record') == row for entry in ledger['leader_history']),
              'Complete prior quality leader retained in history')
    check(any(row['records'] == old['reconstruction_stage_leaders']
              for row in ledger['reconstruction_leader_history'][len(old['reconstruction_leader_history']):]),
          'All four complete prior reconstruction leaders retained')
    previous = inputs['catalog']
    check(current['fresh_research_candidate_history'][:-1] == previous['fresh_research_candidate_history']
          and current['fresh_research_candidate_history'][-1] == previous['fresh_research_candidate'],
          'Whole V11 current entry appended once; entire catalog history prefix retained')
    for key in previous:
        if key not in ('fresh_research_candidate', 'fresh_research_candidate_history', 'date'):
            check(current[key] == previous[key], 'Other catalog field unchanged: '+key)
    check(len(old['stage_scorecards']) == len(old['reconstruction_stage_scorecards']) == 2*(len(VERSIONS)-1),
          'All protected prior stagecards')
    check(len(ledger['stage_scorecards']) == len(ledger['reconstruction_stage_scorecards']) == 2*len(VERSIONS),
          'All versions initial/final stagecards')
    for role, control in [('v%d_table'%v, inputs['experiment_card']['comparison_control']['v%d'%v])
                          for v in VERSIONS[:-1]] + [
                              ('v6_establishment', inputs['experiment_card']['comparison_control']['v6_establishment']),
                              ('v7_establishment', inputs['experiment_card']['comparison_control']['v7_establishment']),
                              ('v8_establishment', inputs['experiment_card']['comparison_control']['v8_establishment']),
                              ('v9_establishment', inputs['experiment_card']['comparison_control']['v9_establishment']),
                              ('v10_establishment', inputs['experiment_card']['comparison_control']['v10_establishment']),
                                  ('v11_establishment', inputs['experiment_card']['comparison_control']['v11_establishment'])]:
        check(all(by_id[role][k] == control[k] for k in ('bytes', 'sha256')),
              'Exact prospective protected control bytes: '+role)
    for version in (6,7,8,9,10,11):
        check(inputs['v%d_table'%version]['trust_protocol_admission_passed'] is False
              and inputs['v%d_establishment'%version]['trust_admission_passed'] is True
              and all(inputs['v%d_establishment'%version]['evidence']['table_rows'][k] == by_id['v%d_table'%version][k]
                      for k in ('bytes', 'sha256')), 'V6/V7/V8/V9/V10/V11 pending producer and separate bound full-trust receipt preserved')
        historical = inputs['v%d_establishment'%version]
        check(historical['initial_pairing_count'] == 13, 'Historical13 declared pairings unchanged')
        if version in (8,9,11):
            check(historical['initial_partial_native20_40_pairings_passed'] is True
                  and historical['initial_pairing_native_coordinate_slice'] == [20,40]
                  and historical['whole_native40_initial_equality_claimed'] is False,
                  'Historical V8/V9/V11 partial20:40 semantics retained')
        else:
            check(historical['initial_full40_pairings_passed'] is True, 'Historical V6/V7/V10 whole40 semantics retained')
    for index, stage in enumerate(('initial', 'final')):
        label = METHOD + ('.init' if stage == 'initial' else '')
        saved = inputs['v12_table']
        quality_card = ledger['stage_scorecards'][-2+index]
        query_card = ledger['reconstruction_stage_scorecards'][-2+index]
        check(quality_card['instance'] == query_card['instance'] == METHOD
              and quality_card['checkpoint_stage'] == query_card['checkpoint_stage'] == stage,
              'Exactly ordered V12 initial/final appended cards')
        check(quality_card['source_updates'] == query_card['source_updates'] == 0 and
              quality_card['causal_updates'] == query_card['causal_updates'] == (0 if stage == 'initial' else 2048),
              'Saved-stage counters retain source0 and actual0/2048')
        check(quality_card['market_rows'] == [r for r in saved['market_rows'] if r['method'] == label],
              'New stagecard copies four actual saved market rows')
        for panel, selected in zip(saved['synthetic']['panels'], quality_card['synthetic_panels']):
            rows = [r for r in panel['rows'] if r['method'] == label]
            check(len(rows) == 1 and selected['row'] == rows[0] and
                  all(selected[key] == value for key, value in panel.items() if key != 'rows'),
                  'New stagecard copies exact saved timing row and panel metadata')
        check(len(quality_card['synthetic_panels']) == len(saved['synthetic']['panels']) == 4,
              'Exactly four actual timing panels')
        for split in ('train', 'validation'):
            required_query_keys = {'standardized_fixed_query_MAE', 'Gaussian_NLL_F64', 'valid_rows', 'target_cells'}
            required_query_keys.update(key for key in ('mean_Huber_F64','variance_NLL_F64','joint_Gaussian_NLL_F64','total_objective_F64')
                                       if key in saved['fixed_queries'][stage][split])
            check(set(query_card['queries'][split]) == required_query_keys, 'Exact declared fixed-query scalar schema')
            check(query_card['queries'][split] == {key:saved['fixed_queries'][stage][split][key]
                  for key in query_card['queries'][split]}, 'New query card exact saved scalar fields')

    def scope(row):
        return tuple(row.get(k) for k in ('dataset_id', 'task', 'input_policy', 'view', 'metric'))

    def quality(leader, version, stage):
        table = inputs['v%d_table' % version]
        label = 'CPB-v%d.market-01' % version + ('.init' if stage == 'initial' else '')
        if leader['dataset_id'] == 'MKT-HIST-2':
            rows = [r for r in table['market_rows'] if r['method'] == label and r['task'] == leader['task']]
        else:
            check(leader['dataset_id'] == 'TEMPO-5', 'Declared timing dataset')
            panels = [p for p in table['synthetic']['panels']
                      if p['input_policy'] == leader['input_policy'] and p['view'] == leader['view']]
            check(len(panels) == 1, 'Unique timing panel')
            rows = [r for r in panels[0]['rows'] if r['method'] == label]
        check(len(rows) == 1 and rows[0]['checkpoint_stage'] == stage, 'One exact saved stage row')
        score = rows[0][leader['metric']]
        check(type(score) in (int, float) and math.isfinite(score), 'Finite saved scope score')
        return dict(method=label, score=score, checkpoint_stage=stage)

    old_scopes = {scope(r): r for r in old['stage_leaders']}
    check(len(old_scopes) == len(ledger['stage_leaders']) == 16 and
          {scope(r) for r in ledger['stage_leaders']} == set(old_scopes), 'Same sixteen quality scopes')
    replacements = []
    for leader in ledger['stage_leaders']:
        original = old_scopes[scope(leader)]
        scores = [quality(leader, v, 'final') for v in VERSIONS]
        best_old = max(scores[:-1], key=lambda r:r['score'])
        best = max(scores, key=lambda r:r['score'])
        check((original['score'], original['method']) == (best_old['score'], best_old['method']),
              'Old strongest final V3-through-V11 leader exact')
        check((leader['score'], leader['method']) == (best['score'], best['method']),
              'Strongest saved final score; stable chronology retains earlier ties')
        additions = [quality(leader, 12, s) for s in ('initial', 'final')]
        check(leader['stage_scores'] == original['stage_scores'] + additions,
              'Whole prior score history plus exactly the two saved V12 stages')
        expected_finals = [dict(row, table_source={k:by_id['v%d_table'%v][k]
                            for k in ('path','bytes','sha256')}) for v, row in zip(VERSIONS, scores)]
        check(leader['all_final_scores'] == expected_finals, 'All chronological actual final scores and exact table pins')
        if best['score'] > original['score']:
            check(best['method'] == METHOD, 'Only new final V12 result can strictly displace a protected leader')
            replacements.append(scope(leader))
    declared = receipt['strict_new_quality_leaders']
    check(len(declared) == len(replacements) and {scope(r) for r in declared} == set(replacements),
          'Actual dynamic changed-scope count, with no invented eight/nine result')
    for change in declared:
        original = old_scopes[scope(change)]
        check(change['former_complete_record'] == original and change['former_method'] == original['method']
              and change['former_score'] == original['score'] and change['new_method'] == METHOD
              and change['new_score'] > original['score'] and change['strictly_better'] is True,
              'Complete displaced record and strict inequality retained')
    check(ledger['stage_leader_update']['strictly_improved_scopes'] == len(replacements)
          and ledger['stage_leader_update']['earlier_scopes_retained'] == 16-len(replacements),
          'Dynamic leader summary agrees')
    check(len(ledger['reconstruction_stage_leaders']) == 4, 'Exactly four reconstruction scopes')
    for leader in ledger['reconstruction_stage_leaders']:
        check(leader['metric'] in ('standardized_fixed_query_MAE', 'Gaussian_NLL_F64'), 'Exact existing query metric')
        scores = [(inputs['v%d_table'%v]['fixed_queries']['final'][leader['split']][leader['metric']],
                   'CPB-v%d.market-01'%v) for v in VERSIONS]
        check((leader['score'], leader['method']) == min(scores, key=lambda r:r[0]),
              'Lowest saved final reconstruction score; ties retain earlier')
        check(leader['numeric_advancement_floor'] is not
              (leader['split'] == 'train' and leader['metric'] == 'Gaussian_NLL_F64'),
              'TRAIN Gaussian NLL remains descriptive, with no extra gate')
    contract = card['evidence_contract']
    for name, assertion in contract['assertions'].items():
        actual = at(inputs[assertion['input']], assertion['path'])
        check(actual == assertion['equals'] and type(actual) is type(assertion['equals']),
              'Actual mandatory postmeasurement trust: '+name)
    for role in ('training', 'comparison', 'engineering', 'gradient', 'initial_proof', 'final_proof', 'initialization'):
        record = inputs[role]
        check(all(record[key] == value for key, value in OBJECTIVE.items()),
              'Exact unchanged V11 objective JSON contract: '+role)
        check(('gradient_policy' not in record) if role == 'initialization' else
              record['gradient_policy'] == GRADIENT,
              'Only fresh JSON has the documented absent gradient selector: '+role)
    for role in ('training','comparison','engineering','initial_proof','final_proof','initialization'):
        check(all(inputs[role][key] == value for key, value in TEMPORAL.items()),
              'Exact new temporal policy,partition and75/76 gauge contract: '+role)
    check(inputs['training']['shared_global_clip'] is True
          and inputs['training']['optimizer_independence_claimed'] is False
          and inputs['training']['all_mandatory_identifiable_tensors_learned'] is True
          and inputs['training']['mandatory_identifiable_tensor_value_capacity'] == 294751,
          'V6 global clip retained;75 identifiable tensor events, not a per-scalar learning claim')
    check(inputs['gradient']['shared_clip_coupling_witness'] is True,
          'Admitted actual CUDA detached-error composite-objective fixture')
    for key,value in inputs['experiment_card']['mechanism_proof_contract'].items():
        if type(value) is bool:
            check(inputs['gradient'][key] is value, 'Actual unchanged V11 objective mechanism predicate: '+key)
    for key, value in inputs['experiment_card']['temporal_pooling_proof_contract']['required_proof_fields'].items():
        check(inputs['temporal_proof'][key] == value, 'Exact temporal fixture predicate: '+key)
    check('full_attention_dependent_moment_adjoints' not in inputs['temporal_proof']
          and inputs['temporal_proof']['direct_pool_kernel_calls'] == 47
          and inputs['temporal_proof']['F64_finite_difference_coordinates'] == inputs['temporal_proof']['F32_finite_difference_coordinates'] == 18
          and all(inputs['temporal_proof'][key] is True for key in (
              'attention_mean_and_uniform_dispersion_full_expression_adjoints',
              'uniform_dispersion_own_center', 'uniform_dispersion_score_independent',
              'uniform_dispersion_hidden_gradient_live', 'attention_mean_score_gradient_live',
              'dual_support_actual_trace')), 'Independent47/18/18 six-flag uniform-dispersion proof scope')
    learning = inputs['temporal_learning']
    check(learning['status'] == 'passed' and learning['updates'] == 2048
          and learning['group_rows'] == 8192 and learning['extra_encoder_calls'] == 0
          and learning['weight_decay_counted_as_data_gradient'] is False
          and learning['gauge']['name'] == TEMPORAL['data_gradient_gauge_exemption'],
          '8192 finer temporal rows and separately observed softmax gauge')
    check(all(inputs['training']['temporal_pooling_learning'][key] == by_id['temporal_learning'][key]
              for key in ('bytes','sha256')), 'Actual trajectory binds exact temporal learning receipt')
    between = learning['between_bin_positions']
    check(between['basis'] == 'hadamard-full16-signs-div4-v1'
          and between['basis_signs'] == [[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]
          and between['full_position_divisor'] == between['position_bin_width'] == 4
          and between['rows'] == 2048 and len(between['gradient_steps']) == len(between['movement_steps']) == 3
          and all(type(n) is int and 0 < n <= 2048 for n in between['gradient_steps']+between['movement_steps'])
          and len(between['final_contrast_changes']) == 3 and all(math.isfinite(n) for n in between['final_contrast_changes'])
          and math.isfinite(between['final_projection_change_l2']) and between['final_projection_change_l2'] > 0,
          'All three Hadamard gradient/movement EVENTS and overall final projection; no per-coefficient final gate')
    conditions = inputs['comparison']['numeric_conditions']
    check(len(conditions) == 22 and len({r['id'] for r in conditions}) == 22
          and all(type(r['passed']) is bool for r in conditions), 'Exactly twenty-two actual Boolean decisions')
    declared_conditions = inputs['experiment_card']['advancement']['criteria']
    check(len(declared_conditions) == 22 and
          [{key:row[key] for key in declared} for row, declared in zip(conditions, declared_conditions)] == declared_conditions,
          'All exact prospective condition identities, floors, sources and rules preserved')
    goal = inputs['experiment_card']['advancement']['goal_protocol']
    check(goal['sha256'] == '866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'
          and all(inputs['comparison']['goal_protocol'][key] == goal[key] for key in ('bytes','sha256')),
          'Unchanged current866dec prospective22 protocol and exact comparison binding')
    passed = sum(row['passed'] for row in conditions)
    coverage = inputs['comparison']['full_coverage'] is True
    advancement = passed == 22 and coverage
    check(inputs['comparison']['numeric_passed'] == passed and coverage, 'Actual count and full coverage')
    check(inputs['comparison']['trust_admission_passed'] is False
          and inputs['establishment']['trust_admission_passed'] is True,
          'Frozen pending producer plus separate completed independent establishment')
    check(set(inputs['comparison']['mandatory_postmeasurement_trust']) ==
          {'initial_native20_40_and_statistics_pairings','independent_training_audit','independent_quality_audit'}
          and all(value == 'pending' for value in inputs['comparison']['mandatory_postmeasurement_trust'].values())
          and inputs['comparison']['overall_establishment_passed'] is False
          and inputs['comparison']['advancement_conditions_passed'] is False,
          'Frozen producer pending fields remain pending despite separate full-trust join')
    check(inputs['establishment']['numeric_passed'] == passed
          and inputs['establishment']['overall_establishment_passed'] is advancement
          and inputs['establishment']['overall_winner'] is False
          and inputs['establishment']['winner_selection_made'] is False,
          'Separate establishment agrees with criterion achievement without winner selection')
    established = inputs['establishment']
    check(all(established[key] is True for key in (
              'attention_mean_and_uniform_dispersion_full_expression_adjoints',
              'uniform_dispersion_own_center', 'uniform_dispersion_score_independent',
              'uniform_dispersion_hidden_gradient_live', 'attention_mean_score_gradient_live',
              'dual_support_actual_trace')), 'Separate establishment preserves six actual uniform-dispersion proof flags')
    check(established['temporal_pooling_direct_kernel_calls'] == 47
          and established['temporal_pooling_F64_finite_difference_coordinates'] == 18
          and established['temporal_pooling_F32_finite_difference_coordinates'] == 18,
          'Separate establishment preserves47/18/18 actual fixture scope')
    check(established['initial_partial_native20_40_pairings_passed'] is True
          and established['initial_pairing_count'] == 13
          and established['initial_pairing_feature_archives'] == 8
          and established['initial_pairing_statistic_archives'] == 5
          and established['initial_pairing_native_coordinate_slice'] == [20,40]
          and established['whole_native40_initial_equality_claimed'] is False,
          'Independent partial20:40 DATA/whole-statistic pairing admission against saved V11')
    check(all(established[key] == value for key, value in TEMPORAL.items())
          and established['all_four_export_row_groups_and_position_contrasts_verified'] is True
          and established['all_three_between_bin_position_directions_verified'] is True
          and established['between_bin_overall_final_projection_verified'] is True
          and established['individual_final_coefficient_nonzero_gate'] is False,
          'Completed establishment retains75/gauge, four local/export and three between-bin event admission')
    moment = inputs['training_audit']['temporal_pooling_saved_learning']['temporal_moment_saved_learning']
    check(moment['status'] == 'passed' and not moment['mismatches']
          and type(moment['checks']) is int and moment['checks'] > 0
          and moment['moment_subgroup_rows_checked'] == learning['moment_subgroup_rows'] == 16384
          and moment['whole_group_two_moment_partition_rows_checked'] == 8192
          and moment['whole_group_partition_squared_norms_and_counts_verified'] is True
          and moment['manual_F32_absolute_tolerance'] == moment['manual_F32_relative_tolerance'] == 3e-5
          and moment['per_scalar_gradient_proof_claimed'] is False
          and moment['other2046_raw_gradients_reconstructed'] is False
          and moment['pool_centered_variance_is_native_uncertainty'] is False,
          'Nested saved moment arithmetic and whole8/two4 norm/element-count scope')
    check(inputs['training']['all_eight_temporal_moment_subgroups_learned'] is True
          and len(moment['moment_subgroup_event_counts']) == len(learning['moment_subgroups']) == 8,
          'Exactly eight actual mean4/dispersion4 subgroups')
    for slot, group in enumerate(learning['moment_subgroups']):
        b, half = divmod(slot,2)
        check(group['bin'] == b and group['moment'] == ('mean' if half == 0 else 'centered_variance')
              and group['export_row_begin'] == b*8+half*4 and group['export_row_end'] == b*8+(half+1)*4
              and group['final_weight_changed'] is True and group['final_bias_changed'] is True,
              'Exact moment subgroup rows and final-change evidence')
        for key in ('weight_nonzero_data_gradient_steps','bias_nonzero_data_gradient_steps',
                    'weight_movement_steps','bias_movement_steps'):
            check(type(group[key]) is int and 0 < group[key] <= 2048
                  and group[key] == moment['moment_subgroup_event_counts'][slot][key],
                  'Subgroup event counts exact; whole8 events use OR rather than adding booleans')
    local_parameters = {'temporal_export_weight','temporal_export_bias','temporal_pool_positions'}
    witnesses = moment['actual_same_forward_moment_witnesses']
    check(len(witnesses) == 2 and [row['update'] for row in witnesses] == [1,2048],
          'Two actual same-forward witnesses, without a full-gradient replay')
    for index, row in enumerate(witnesses):
        check(row['same_forward_only'] is True and row['total_component_archive_fields'] == 60
              and row['new_temporal_trace_fields'] == 25 and row['parameter_array_fields'] == 12
              and row['new_uniform_dispersion_trace_fields'] == 4
              and row['uniform_dispersion_direct_score_adjoint'] == 0
              and row['local_hidden_adjoint_array_available'] is False
              and row['score_to_hidden_route_reconstructed'] is False
              and row['hidden_reference_is_observed_gradient_parity'] is False
              and row['upstream_stage'] == 'PRECLIP-post-empty-export'
              and len(row['upstream_gradient_defined_by_route']) == 2
              and all(type(flag) is bool for flag in row['upstream_gradient_defined_by_route'])
              and type(row['recorded_clip_factor_applied_once']) in (int,float)
              and math.isfinite(row['recorded_clip_factor_applied_once'])
              and 0 < row['recorded_clip_factor_applied_once'] <= 1
              and row['local_export_and_positions_adjoint_verified'] is True
              and row['local_adjoint_absolute_tolerance'] == 1e-9
              and row['local_adjoint_relative_tolerance'] == 3e-4
              and row['postclip_export_nonvacuity_threshold'] == 1e-6
              and math.isfinite(row['postclip_export_expected_maximum'])
              and row['postclip_export_expected_maximum'] > 1e-6
              and row['full_hidden_encoder_or_global_norm_reconstruction'] is False
              and row['matched_pre_step_decoder_adjoint_replay'] is False,
              'Declared60/25/12, PRECLIP upstream, single clip and tight nonvacuous local adjoints')
        check(all(row['artifact'][key] == inputs['training_audit']['actual_pre_step_composite_objective']['witnesses'][index]['witness'][key]
                  == inputs['training_audit']['temporal_pooling_saved_learning']['actual_raw_array_witnesses'][index]['artifact'][key]
                  for key in ('bytes','sha256')), 'Actual archive alias cross-module binding')
        errors = row['local_parameter_gradient_error_summaries']
        check(set(errors) == local_parameters, 'Only three actual local parameter parity gates')
        for name, values in errors.items():
            check(set(values) == {'maximum_expected_absolute','maximum_actual_absolute','maximum_absolute_error','maximum_scaled_error'}
                  and all(type(value) in (int,float) and math.isfinite(value) and value >= 0 for value in values.values())
                  and values['maximum_scaled_error'] <= 1, 'Saved independent local error summary: '+name)
        gauge = row['common_score_bias_gauge_descriptive']
        check(gauge['parity_gate'] is False and gauge['nonzero_gradient_required'] is False
              and all(type(gauge[key]) in (int,float) and math.isfinite(gauge[key])
                      for key in ('actual','reference','absolute_error')), 'Gauge finite descriptive evidence only')
    for key,value in dict(all_eight_temporal_moment_subgroups_verified=True,
            whole8_two4_squared_norms_and_element_counts_verified=True,
            actual_first_last_local_moment_adjoints_verified=True,
            actual_local_moment_adjoint_parameter_tensors=sorted(local_parameters),
            actual_local_moment_adjoint_absolute_tolerance=1e-9, actual_local_moment_adjoint_relative_tolerance=3e-4,
            each_actual_local_export_nonvacuity_threshold=1e-6, common_score_bias_gauge_tight_parity_required=False,
            actual_component_archive_fields=60,same_forward_temporal_fields=25,
            new_uniform_dispersion_trace_fields=4,uniform_dispersion_direct_score_adjoint=0,
            local_hidden_adjoint_array_available=False,
            score_to_hidden_route_reconstructed=False,hidden_reference_is_observed_gradient_parity=False,
            full_encoder_gradient_or_global_norm_reconstructed=False).items():
        check(established[key] == value, 'Separate moment establishment summary: '+key)
    check(inputs['experiment_card']['scientific_recipe_parent'] ==
          'CPB-v11 source only;fresh weights/optimizer/statistics,not trained ancestry'
          and inputs['experiment_card']['scoped_reference'] == 'CPB-v11.market-01',
          'V11 source-only recipe; latest measured V11; V3 remains selected')
    for record in (current['fresh_research_candidate'],instance,receipt):
        check(record['temporal_moment_saved_learning'] == moment, 'Full bounded moment audit record retained')
    for record in (current['fresh_research_candidate'],instance):
        check(record['all_eight_temporal_moment_subgroups_learned'] is True
              and record['scientific_recipe_parent'] == 'CPB-v11.market-01'
              and record['scientific_recipe_parent_is_source_only'] is True
              and record['scoped_reference'] == 'CPB-v11.market-01', 'Truthful source recipe and moment learning labels')
    establishment_bindings = at(inputs['establishment'], contract['establishment_bindings_path'])
    for role, key in contract['establishment_binding_keys'].items():
        check(all(establishment_bindings[key][field] == by_id[role][field] for field in ('bytes','sha256')),
              'Separate establishment binds actual new completed evidence: '+role)
    check(ledger['next_direction']['goal_active'] is not advancement and
          ledger['next_direction']['goal_criteria_satisfied'] is advancement and
          ledger['next_direction']['later_training_authorized_by_this_finalization'] is False,
          'Goal criterion outcome recorded without authorizing a later training run')
    core = at(inputs['training_audit'], contract['training_audit_checks_path'])
    module_counts = {name:inputs['training_audit'][name]['checks'] for name in
                     ('numeric_statistics', 'saved_tensor_state', 'manual_saved_Gaussian',
                      'actual_pre_step_composite_objective', 'temporal_pooling_saved_learning')}
    quality_count = at(inputs['quality_audit'], contract['quality_audit_checks_path'])
    check(established['training_audit_core_checks'] == core
          and established['training_audit_five_module_checks'] == module_counts
          and established['training_audit_total_checks'] == core+sum(module_counts.values())
          and established['quality_audit_checks'] == quality_count,
          'Independent establishment core+five module counts agree with exact raw auditor fields')
    check(instance['independent_training_audit_core_checks'] == core and
          instance['independent_training_audit_module_checks'] == module_counts and
          instance['independent_training_audit_core_plus_modules_checks'] == core+sum(module_counts.values()) and
          instance['independent_quality_audit_checks'] == quality_count, 'Actual core plus five-module audit counts')
    check(current['fresh_research_candidate']['instance'] == instance['instance'] == METHOD,
          'Measured instance/catalog identity')
    check(instance['source_updates'] == 0 and instance['new_causal_updates'] == 2048 and instance['batch_size'] == 8,
          'Fresh single trajectory')
    audit_training = inputs['training_audit']
    check(type(instance['attempts']) is int and 2048 <= instance['attempts'] <= 4096
          and instance['attempts'] == inputs['training']['attempted_batches'] == audit_training['attempted_batches']
          and audit_training['CSV_rows'] == instance['attempts']
          and instance['successful_updates'] == audit_training['successful_CSV_rows'] == 2048
          and instance['skipped_ineligible_attempts'] == audit_training['skipped_ineligible_attempts'] == instance['attempts']-2048
          and audit_training['checkpoint_attempt_counters']['2048'] == instance['attempts'],
          'Actual successful/skipped/attempted counters retained without a zero-skip assumption')
    counts = inputs['experiment_card']['source_counts']
    check(instance['physical_owned_family_files'] == counts['family_inside']+counts['family_external_header']
          and instance['physical_owned_research_files'] == counts['research']
          and instance['frozen_source_capsule_files'] == counts['capsule_sources'], 'Actual declared physical/source counts')
    for record in (current['fresh_research_candidate'], instance):
        check(all(record[key] == value for key, value in TEMPORAL.items()), 'Exact new temporal mechanism contract')
        check(record['shared_global_clip'] is True and record['optimizer_independence_claimed'] is False,
              'V6 single global clipping contract retained')
        check(record['initial_pairing_native_coordinate_slice'] == [20,40]
              and record['whole_native40_initial_equality_claimed'] is False
              and all(record['initial_native20_40_and_statistics_pairings'][k] == by_id['pairings'][k]
                      for k in ('bytes','sha256')), 'Partial20:40 initial DATA pairings with exact whole statistics')
    check(instance['all_75_mandatory_identifiable_tensors_with_nonzero_CUDA_gradients_and_movement'] is True
          and instance['mandatory_identifiable_tensor_value_capacity'] == 294751
          and instance['data_learning_is_tensor_event_evidence_not_every_scalar'] is True
          and instance['weight_decay_counted_as_data_learning'] is False,
          'Only one declared gauge exemption; tensor event/capacity wording retained')
    check(instance['temporal_pooling_learning'] == {key:by_id['temporal_learning'][key] for key in ('path','bytes','sha256')}
          and instance['temporal_pooling_proof'] == {key:by_id['temporal_proof'][key] for key in ('path','bytes','sha256')},
          'Exact finer-learning/fixture bindings retained in measured instance')
    admission = established['normal_prospective_source_admission']
    check(admission == dict(source_files=counts['capsule_sources'],
          owner_source_files=counts['family_inside']+counts['family_external_header']+counts['research'],
          freeze_and_preflight_bound=True, normal_quality_all_stages_passed=True,
          scientific_model_head_or_training_replay=False), 'Actual normal completed-quality source admission')
    for record in (current['fresh_research_candidate'], instance, receipt):
        check(record['normal_prospective_source_admission'] == admission, 'Complete normal source admission record retained')
        check(record['historical_V9_reporting_recovery'] == inputs['experiment_card']['historical_V9_reporting_recovery']
              == established['historical_V9_reporting_recovery_preserved'], 'Historical V9 recovery copied opaquely')
    source = inputs['source_manifest']
    check(source['kind'] == 'causal_fresh_v12_source_capsule_v1'
          and source['card_sha256'] == by_id['experiment_card']['sha256']
          and len(source['files']) == len({row['path'] for row in source['files']}) == counts['capsule_sources'],
          'Unique exact actual main-source closure')
    for row in source['files']:
        bound(safe(Path(card['container_attempt'])/'source', row['path']), row)
    check(len(inputs['build']['binaries']) == counts['binaries']
          and len(inputs['build']['dependency_closures']) == counts['translation_units']
          and all(inputs['build']['source_manifest'][key] == by_id['source_manifest'][key] for key in ('bytes','sha256')),
          'Actual saved binary/dependency closure and manifest')
    for row in inputs['build']['binaries']:
        bound(Path(row['path']),row)
    check(inputs['mirror']['scope'] == 'Complete' and inputs['mirror']['source_files_verified'] == counts['capsule_sources']
          and inputs['mirror']['all_mirrored_pt_ignored'] is True, 'Exact complete normal mirror and ignored PT')
    check(inputs['training']['raw_load_calls'] == 512+instance['attempts']+36,
          'Calibration plus actual attempted loads plus36 fixed-query loads')
    freeze, preflight = inputs['prospective_freeze'], inputs['protocol_preflight']
    check(freeze['kind'] == 'causal_fresh_v12_prospective_freeze_v1' and freeze['status'] == 'passed'
          and all(freeze['card'][key] == by_id['experiment_card'][key] for key in ('bytes','sha256'))
          and freeze['owner_source_files'] == admission['owner_source_files'], 'Actual prospective freeze/card closure')
    check(preflight['kind'] == 'causal_fresh_v12_source_protocol_preflight_v1' and preflight['status'] == 'passed'
          and preflight['card_sha256'] == by_id['experiment_card']['sha256']
          and len(preflight['sources'])+1 == counts['capsule_sources'], 'Actual prospective preflight/card closure')
    for record in (current['fresh_research_candidate'], instance):
        for role in ('prospective_freeze','protocol_preflight'):
            check(all(record[role][key] == by_id[role][key] for key in ('path','bytes','sha256')),
                  'Exact normal prospective freeze/preflight evidence binding retained: '+role)
    for record in (current['fresh_research_candidate'], instance['quality']):
        check(record['numeric_passed'] == passed and record['numeric_total'] == 22
              and record['advancement_passed'] is advancement, 'Saved numeric result retained without outcome assumption')
        check(record['trust_admission_passed'] and record['full_coverage'], 'Separate full trust and coverage')
    check(len(instance['main_checkpoints']) == 5 and {Path(p['path']).name for p in instance['main_checkpoints']} ==
          {'inference-0.pt','inference-2048.pt','training-512.pt','training-1024.pt','training-2048.pt'},
          'All five checkpoint roles retained')
    check(instance['prospective_descriptor']['sha256'] == by_id['descriptor']['sha256'] and
          instance['prospective_descriptor_rewritten'] is False, 'Prospective descriptor immutable')
    check(receipt['model_calls'] == receipt['head_fits'] == receipt['training_updates'] == receipt['live_files_written'] == 0,
          'No scientific replay or live write')
    result = dict(kind='causal_fresh_v12_independent_metadata_proposal_review_v1', status='passed',
        checks=checks, metadata_outputs=receipt['outputs'], strict_new_quality_leaders=len(replacements),
        numeric_passed=passed, numeric_total=22, overall_establishment_passed=advancement,
        training_audit_core_plus_modules_checks=core+sum(module_counts.values()), quality_audit_checks=quality_count,
        baseline_changed=False, descriptor_changed=False, producer_changed=False,
        model_calls=0, head_fits=0, training_updates=0, live_files_written=0,
        card=pin(args.card), generator_receipt=pin(args.proposals/'finalization-proposal.json'), source=pin(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='passed', checks=checks, quality_leaders=len(replacements), numeric_passed=passed)))


if __name__ == '__main__':
    main()

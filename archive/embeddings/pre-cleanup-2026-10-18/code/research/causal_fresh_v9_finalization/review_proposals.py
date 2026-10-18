# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Separately review three saved V9 metadata proposals; never apply them.

This owned derivative imports no proposer, model or numerical array library.
Actual outcome and check counts come from the admitted saved evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('/opt/cuwacunu/research/causal-fresh-v9-finalization')
METHOD = 'CPB-v9.market-01'
BASELINE = 'CPB-v3.market-01'
FAMILY = 'code/encoders/causal_patch_bottleneck_mae/versions/'
VERSIONS = (3, 4, 5, 6, 7, 8, 9)
HISTORY_KEYS = ('candidates', 'stage_scorecards', 'reconstruction_stage_scorecards',
    'leader_history', 'displaced_leaders', 'candidate_status_history', 'direction_history',
    'stage_leader_update_history', 'reconstruction_leader_history')
TEMPORAL = dict(temporal_pooling_policy='original-position-four-bin-shared-global-half-mixture-v1',
    temporal_pooling_partition=[4,16,32,4,8],temporal_pooling_mix=[0.5,0.5],
    empty_local_bin_policy='zero-whole-export-slice-after-bias-v1',registered_learned_tensors=76,
    mandatory_identifiable_learned_tensors=75,
    data_gradient_gauge_exemption='core.encoder.level.base.backbone.pool_score.bias')
OBJECTIVE = dict(objective_policy='huber1-mean-plus-detached-error-gaussian-variance-v1',
    mean_huber_delta=1, loss_coefficients=[1, 1], variance_error_detached=True,
    reporting_nll_backpropagated=False)
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
    check(card['kind'] == 'cpb_v9_metadata_only_finalization_card_v1', 'Own admitted metadata card')
    inputs, by_id = {}, {}
    for row in card['inputs']:
        check(row['id'] not in by_id, 'Unique input roles')
        path = safe(args.card.parent, row['staged_path'])
        bound(path, row)
        by_id[row['id']] = row
        if row['format'] == 'json':
            inputs[row['id']] = read(path)
    receipt = read(args.proposals / 'finalization-proposal.json')
    check(receipt['kind'] == 'cpb_v9_metadata_only_finalization_proposal_v1'
          and receipt['status'] == 'passed_proposal_not_applied', 'Unapplied successful proposal')
    check(all(receipt['card'][k] == pin(args.card)[k] for k in ('bytes', 'sha256')), 'Actual generator card binding')
    expected = {'doc/clean_causal_encoder_research_ledger.json', FAMILY+'current.json',
                FAMILY+'instances/CPB-v9.market-01.json'}
    check(len(receipt['outputs']) == 3 and {r['path'] for r in receipt['outputs']} == expected,
          'Exactly three declared metadata outputs')
    for row in receipt['outputs']:
        bound(safe(args.proposals, row['path']), row)
    ledger = read(args.proposals/'doc/clean_causal_encoder_research_ledger.json')
    current = read(args.proposals/(FAMILY+'current.json'))
    instance = read(args.proposals/(FAMILY+'instances/CPB-v9.market-01.json'))
    old = inputs['ledger']
    check(ledger['baseline'] == old['baseline'] == current['active_application'] == BASELINE,
          'Clean baseline/application retained')
    check(ledger['baseline_checkpoint'] == old['baseline_checkpoint'], 'Complete baseline checkpoint record retained')
    check(ledger['overall_winner'] == old['overall_winner'] is False, 'No automatic overall-winner declaration')
    for key in HISTORY_KEYS:
        check(ledger[key][:len(old[key])] == old[key], 'Complete historical prefix retained: '+key)
    check(len(ledger['candidates']) == len(old['candidates'])+1, 'Exactly one measured candidate appended')
    check([r['instance'] for r in ledger['candidates']].count(METHOD) == 1, 'One V9 instance')
    for row in old['stage_leaders']:
        check(any(entry == row or entry.get('record') == row for entry in ledger['leader_history']),
              'Complete prior quality leader retained in history')
    check(any(row['records'] == old['reconstruction_stage_leaders']
              for row in ledger['reconstruction_leader_history'][len(old['reconstruction_leader_history']):]),
          'All four complete prior reconstruction leaders retained')
    previous = inputs['catalog']
    check(current['fresh_research_candidate_history'][:-1] == previous['fresh_research_candidate_history']
          and current['fresh_research_candidate_history'][-1] == previous['fresh_research_candidate'],
          'Whole V8 current entry appended once; entire catalog history prefix retained')
    for key in previous:
        if key not in ('fresh_research_candidate', 'fresh_research_candidate_history', 'date'):
            check(current[key] == previous[key], 'Other catalog field unchanged: '+key)
    check(len(old['stage_scorecards']) == len(old['reconstruction_stage_scorecards']) == 12,
          'Twelve protected prior stagecards')
    check(len(ledger['stage_scorecards']) == len(ledger['reconstruction_stage_scorecards']) == 14,
          'Seven versions initial/final stagecards')
    for role, control in [('v%d_table'%v, inputs['experiment_card']['comparison_control']['v%d'%v])
                          for v in VERSIONS[:-1]] + [
                              ('v6_establishment', inputs['experiment_card']['comparison_control']['v6_establishment']),
                              ('v7_establishment', inputs['experiment_card']['comparison_control']['v7_establishment']),
                              ('v8_establishment', inputs['experiment_card']['comparison_control']['v8_establishment'])]:
        check(all(by_id[role][k] == control[k] for k in ('bytes', 'sha256')),
              'Exact prospective protected control bytes: '+role)
    for version in (6,7,8):
        check(inputs['v%d_table'%version]['trust_protocol_admission_passed'] is False
              and inputs['v%d_establishment'%version]['trust_admission_passed'] is True
              and all(inputs['v%d_establishment'%version]['evidence']['table_rows'][k] == by_id['v%d_table'%version][k]
                      for k in ('bytes', 'sha256')), 'V6/V7/V8 pending producer and separate bound full-trust receipt preserved')
        historical = inputs['v%d_establishment'%version]
        check(historical['initial_pairing_count'] == 13, 'Historical13 declared pairings unchanged')
        if version == 8:
            check(historical['initial_partial_native20_40_pairings_passed'] is True
                  and historical['initial_pairing_native_coordinate_slice'] == [20,40]
                  and historical['whole_native40_initial_equality_claimed'] is False,
                  'Historical V8 partial20:40 semantics retained')
        else:
            check(historical['initial_full40_pairings_passed'] is True, 'Historical V6/V7 whole40 semantics retained')
    for index, stage in enumerate(('initial', 'final')):
        label = METHOD + ('.init' if stage == 'initial' else '')
        saved = inputs['v9_table']
        quality_card = ledger['stage_scorecards'][-2+index]
        query_card = ledger['reconstruction_stage_scorecards'][-2+index]
        check(quality_card['instance'] == query_card['instance'] == METHOD
              and quality_card['checkpoint_stage'] == query_card['checkpoint_stage'] == stage,
              'Exactly ordered V9 initial/final appended cards')
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
            required_query_keys.update(key for key in ('mean_Huber_F64','variance_NLL_F64','total_objective_F64')
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
              'Old strongest final V3/V4/V5/V6/V7/V8 leader exact')
        check((leader['score'], leader['method']) == (best['score'], best['method']),
              'Strongest saved final score; stable chronology retains earlier ties')
        additions = [quality(leader, 9, s) for s in ('initial', 'final')]
        check(leader['stage_scores'] == original['stage_scores'] + additions,
              'Whole prior score history plus exactly the two saved V9 stages')
        expected_finals = [dict(row, table_source={k:by_id['v%d_table'%v][k]
                            for k in ('path','bytes','sha256')}) for v, row in zip(VERSIONS, scores)]
        check(leader['all_final_scores'] == expected_finals, 'Seven actual final scores and exact table pins')
        if best['score'] > original['score']:
            check(best['method'] == METHOD, 'Only new final V9 result can strictly displace a protected leader')
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
              'Exact unchanged V6 composite objective JSON contract: '+role)
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
          'Admitted actual CUDA unchanged V6 composite-objective fixture')
    for key, value in inputs['experiment_card']['temporal_pooling_proof_contract']['required_proof_fields'].items():
        check(inputs['temporal_proof'][key] == value, 'Exact temporal fixture predicate: '+key)
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
    passed = sum(row['passed'] for row in conditions)
    coverage = inputs['comparison']['full_coverage'] is True
    advancement = passed == 22 and coverage
    check(inputs['comparison']['numeric_passed'] == passed and coverage, 'Actual count and full coverage')
    check(inputs['comparison']['trust_admission_passed'] is False
          and inputs['establishment']['trust_admission_passed'] is True,
          'Frozen pending producer plus separate completed independent establishment')
    check(inputs['establishment']['numeric_passed'] == passed
          and inputs['establishment']['overall_establishment_passed'] is advancement
          and inputs['establishment']['overall_winner'] is False
          and inputs['establishment']['winner_selection_made'] is False,
          'Separate establishment agrees with criterion achievement without winner selection')
    established = inputs['establishment']
    check(established['initial_partial_native20_40_pairings_passed'] is True
          and established['initial_pairing_count'] == 13
          and established['initial_pairing_feature_archives'] == 8
          and established['initial_pairing_statistic_archives'] == 5
          and established['initial_pairing_native_coordinate_slice'] == [20,40]
          and established['whole_native40_initial_equality_claimed'] is False,
          'Independent partial DATA/statistic pairing admission; no native0:20 gate')
    check(all(established[key] == value for key, value in TEMPORAL.items())
          and established['all_four_export_row_groups_and_position_contrasts_verified'] is True
          and established['all_three_between_bin_position_directions_verified'] is True
          and established['between_bin_overall_final_projection_verified'] is True
          and established['individual_final_coefficient_nonzero_gate'] is False,
          'Completed establishment retains75/gauge, four local/export and three between-bin event admission')
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
    check(instance['physical_owned_family_files'] == 51 and instance['physical_owned_research_files'] == 31
          and instance['frozen_source_capsule_files'] == 118, 'Actual declared physical/source counts')
    for record in (current['fresh_research_candidate'], instance):
        check(all(record[key] == value for key, value in TEMPORAL.items()), 'Exact new temporal mechanism contract')
        check(record['shared_global_clip'] is True and record['optimizer_independence_claimed'] is False,
              'V6 single global clipping contract retained')
        check(record['initial_pairing_native_coordinate_slice'] == [20,40]
              and record['whole_native40_initial_equality_claimed'] is False
              and all(record['initial_native20_40_and_statistics_pairings'][k] == by_id['pairings'][k]
                      for k in ('bytes','sha256')), 'Partial initial pairings without a whole40 equality claim')
    check(instance['all_75_mandatory_identifiable_tensors_with_nonzero_CUDA_gradients_and_movement'] is True
          and instance['mandatory_identifiable_tensor_value_capacity'] == 294751
          and instance['data_learning_is_tensor_event_evidence_not_every_scalar'] is True
          and instance['weight_decay_counted_as_data_learning'] is False,
          'Only one declared gauge exemption; tensor event/capacity wording retained')
    check(instance['temporal_pooling_learning'] == {key:by_id['temporal_learning'][key] for key in ('path','bytes','sha256')}
          and instance['temporal_pooling_proof'] == {key:by_id['temporal_proof'][key] for key in ('path','bytes','sha256')},
          'Exact finer-learning/fixture bindings retained in measured instance')
    admission = established['normal_prospective_source_admission']
    check(admission == dict(source_files=118, owner_source_files=82,
          freeze_and_preflight_bound=True, invented_V9_host_recovery=False,
          scientific_model_head_or_training_replay=False),
          'Normal prospective source admission, no inherited invented V9 host recovery')
    for record in (current['fresh_research_candidate'], instance, receipt):
        check(record['normal_prospective_source_admission'] == admission, 'Complete normal source admission record retained')
    recovered = established['report_only_recovery']
    recovery_card, recovery = inputs['reporting_recovery_card'], inputs['reporting_recovery_receipt']
    check(recovery_card['kind'] == 'cpb_v9_report_only_recovery_card_v1' and recovery_card['root_reviewed'] is True
          and recovery['kind'] == 'cpb_v9_report_only_recovery_receipt_v1' and recovery['status'] == 'passed'
          and all(recovery['card'][k] == by_id['reporting_recovery_card'][k] for k in ('bytes','sha256')),
          'Exact external report-only declaration and passed recovery receipt')
    check(recovery['original_quality_stage_status'] == recovered['original_quality_stage_status'] == 'failed'
          and recovery['completed_scientific_stages'] == recovered['completed_scientific_stages'] == 5
          and all(type(recovery[k]) is int and recovery[k] == 0 for k in
                  ('encoder_calls','head_fits','training_updates','science_stages_replayed'))
          and recovered['science_stages_replayed'] == 0 and recovered['status'] == 'passed',
          'Failed original disposition and five completed stages retained without scientific replay')
    identities = lambda rows:{(p['path'],p['bytes'],p['sha256']) for p in rows}
    check(recovery['scientific_inputs_before'] == recovery['scientific_inputs_after']
          and len(recovery['scientific_inputs_before']) == len(recovery_card['scientific_inputs']) == 531
          and identities(recovery['scientific_inputs_before']) == identities(recovery_card['scientific_inputs'])
          and recovered['scientific_inputs_before_after_exact'] is True
          and recovered['frozen_scientific_source_unchanged'] is True,
          'Whole531 scientific/failed/readout identities remain exact')
    check(set(recovery['scientific_stages']) == {'extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads'}
          and recovery['scientific_stages'] == recovery_card['scientific_stages']
          and recovered['failed_quality_stage'] == recovery['failed_quality_stage'] == recovery_card['failed_quality_stage']
          and recovered['failed_report_stage'] == recovery['failed_report_stage'] == recovery_card['failed_report_stage']
          and recovered['failure_mirror'] == recovery_card['failure_mirror'], 'Exact original failure/completed-stage provenance')
    for role,key in (('reporting_recovery_card','card'),('reporting_recovery_receipt','receipt')):
        check(all(recovered[key][k] == by_id[role][k] for k in ('bytes','sha256')),
              'Completed establishment binds reporting-only '+role)
    for role,key in (('v9_table','report'),('comparison','comparison')):
        check(any(all(p[k] == by_id[role][k] for k in ('bytes','sha256')) for p in recovery[key]['output_pins']),
              'Actual report-only aggregation binds proposed '+role)
    for record in (current['fresh_research_candidate'], instance, receipt):
        check(record['report_only_recovery'] == recovered, 'Transparent reporting recovery copied whole')
        for role in ('reporting_recovery_card','reporting_recovery_receipt'):
            check(all(record[role][k] == by_id[role][k] for k in ('path','bytes','sha256')),
                  'Exact reporting-only evidence retained: '+role)
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
    result = dict(kind='causal_fresh_v9_independent_metadata_proposal_review_v1', status='passed',
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

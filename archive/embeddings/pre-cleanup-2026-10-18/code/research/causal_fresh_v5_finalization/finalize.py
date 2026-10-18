# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Metadata-only V5 proposal generator. No model/array libraries or live writes."""
import argparse
import copy
import datetime
import hashlib
import json
import math
import pathlib

FAMILY = 'code/encoders/causal_patch_bottleneck_mae/versions/'
POLICY = 'Strictly higher saved final score updates only; exact ties retain earlier leader. Initial stages are descriptive, never checkpoint selection or overall promotion'
GRADIENT = 'separate-mean-and-uncertainty-bank-projections-live-gaussian-nll-v1'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def pin(path, label=None):
    path = pathlib.Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked file: ' + str(path))
    data = path.read_bytes()
    return {'path': label or str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def finite(value):
    require(isinstance(value, (float, int)) and math.isfinite(value), 'Finite scalar required')
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--card', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    card = json.loads(args.card.read_text())
    require(card['kind'] == 'cpb_v5_metadata_only_finalization_card_v1', 'Own finalization card required')
    source = args.card.parent
    own = pin(pathlib.Path(__file__))
    require(own['sha256'] == card['generator']['sha256'], 'Frozen generator identity')
    inputs, bindings = {}, []
    for row in card['inputs']:
        relative = pathlib.PurePosixPath(row['staged_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Bounded metadata input')
        path = source / relative
        actual = pin(path, row['path'])
        require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'Input hash mismatch: ' + row['id'])
        inputs[row['id']] = json.loads(path.read_text()) if row.get('format', 'json') == 'json' else path.read_text()
        bindings.append(actual)
    require(len(inputs) == len(card['inputs']), 'Unique input IDs')
    old, catalog = inputs['ledger'], inputs['catalog']
    tables = {3: inputs['v3_table'], 4: inputs['v4_table'], 5: inputs['v5_table']}
    comparison, training = inputs['comparison'], inputs['training']
    require(comparison['numeric_passed'] == 7 and comparison['numeric_total'] == 22, 'Original stronger goal7/22 retained')
    require(comparison['trust_admission_passed'] and comparison['full_coverage'] and not comparison['advancement_conditions_passed'], 'Measured failed candidate with trust/coverage')
    require(len(comparison['numeric_conditions']) == 22, 'Exact original22 conditions, no added floor')
    require(training['new_causal_updates'] == training['attempted_batches'] == 2048 and training['source_updates'] == 0, 'One fresh2048 trajectory')
    require(training['parameters'] == 298640 and training['parameters_with_recorded_updates'] == 294752 and training['fixed_prior_parameters'] == 3888, 'Actual learned/fixed partition')
    require(training['gradient_policy'] == GRADIENT and training['typed_resume_512_exact'], 'Typed actual trajectory and full disjoint gradient')
    for name in ('training_audit', 'quality_audit'):
        require(inputs[name]['status'] == 'passed', 'Independent audit passed')
    require(inputs['training_audit']['total_checks'] == 192589 and inputs['quality_audit']['checks'] == 11300, 'Actual audit check counts')
    require('31 CUDA checks' in inputs['gradient_log'], 'Observed31 routing checks')
    for name in ('initial_proof', 'final_proof'):
        proof = inputs[name]
        require(proof['status'] == 'passed' and proof['all_cutoffs'] == 128 and proof['suffix_cases'] == 2350 and
                proof['internal_temporal_cases'] == 30 and proof['state_unchanged'], 'Complete frozen saved causal proof')
    stage = inputs['source_manifest']
    require(len(stage['files']) == 97 and len(inputs['build']['binaries']) == 6, 'Exact97 source/six binary closure')
    capsule = pathlib.Path(card['container_attempt']) / 'source'
    for row in stage['files']:
        actual = pin(capsule / row['path'])
        require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'Immutable source binding')
    for row in inputs['build']['binaries']:
        actual = pin(row['path'])
        require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'Binary binding; binary never executed')
    actual_checkpoints = []
    for row in inputs['mirror']['files']:
        if '/run-001/checkpoints/' in row['container_path']:
            actual = pin(row['container_path'], row['path'])
            require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'Five saved checkpoint hashes')
            actual_checkpoints.append(actual)
    require(len(actual_checkpoints) == 5 and inputs['mirror']['all_mirrored_pt_ignored'], 'Five available/ignored canonical checkpoints')
    require(not args.output.exists(), 'Exclusive proposal output')
    args.output.mkdir(parents=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    by_id = {row['id']: {'path': row['path'], 'bytes': row['bytes'], 'sha256': row['sha256']} for row in card['inputs']}
    v5, method = tables[5], 'CPB-v5.market-01'
    require(v5['new_market_rows'] == 8, 'Eight newly measured market rows')
    market = [row for row in v5['market_rows'] if row['method'] in (method, method + '.init')]
    panels = v5['synthetic']['panels']
    require(len(market) == 8 and len(panels) == 4, 'Complete V5 stage score panels')
    stagecards = []
    for checkpoint_stage, label in (('initial', method + '.init'), ('final', method)):
        rows = [copy.deepcopy(row) for row in market if row['method'] == label]
        synthetics = []
        for panel in panels:
            item = {k: copy.deepcopy(v) for k, v in panel.items() if k != 'rows'}
            item['row'] = copy.deepcopy(next(row for row in panel['rows'] if row['method'] == label))
            synthetics.append(item)
        require(len(rows) == len(synthetics) == 4, 'Four tasks/four synthetic panels per stage')
        stagecards.append({'method': label, 'instance': method, 'checkpoint_stage': checkpoint_stage,
                           'source_updates': 0, 'causal_updates': 0 if checkpoint_stage == 'initial' else 2048,
                           'measurement_reused_for_v5_comparison': False, 'table_source': by_id['v5_table'],
                           'market_rows': rows, 'synthetic_panels': synthetics, 'selection_policy': POLICY})
    ledger = copy.deepcopy(old)
    require(len(old['stage_leaders']) == 16, 'All16 old scoped quality records')
    replacements = []
    for leader in ledger['stage_leaders']:
        original = copy.deepcopy(leader)
        if leader['dataset_id'] == 'MKT-HIST-2':
            selected = [next(row for row in market if row['task'] == leader['task'] and row['method'] == label)
                        for label in (method + '.init', method)]
        else:
            panel = next(p for p in panels if p['view'] == leader['view'] and p['input_policy'] == leader['input_policy'])
            selected = [next(row for row in panel['rows'] if row['method'] == label) for label in (method + '.init', method)]
        additions = [{'method': row['method'], 'score': finite(row[leader['metric']]), 'checkpoint_stage': row['checkpoint_stage']} for row in selected]
        leader.setdefault('stage_scores', []).extend(additions)
        final = additions[1]
        better = final['score'] > original['score']
        ledger.setdefault('leader_history', []).append({'preserved_before': method + ' measured', 'record': original,
            'displaced': better, 'successor_method': method if better else original['method'],
            'successor_score': final['score'] if better else original['score']})
        if better:
            change = {'dataset_id': leader['dataset_id'], 'task': leader['task'], 'view': leader['view'],
                      'input_policy': leader.get('input_policy'), 'metric': leader['metric'],
                      'former_method': original['method'], 'former_score': original['score'],
                      'new_method': method, 'new_score': final['score'], 'strictly_better': True,
                      'former_complete_record': original}
            ledger.setdefault('displaced_leaders', []).append(change)
            replacements.append(change)
            leader.update(method=method, score=final['score'], initial_and_final=additions)
            if leader['dataset_id'] == 'MKT-HIST-2':
                leader['recipe_id'] = selected[1]['recipe_id']
        leader['selection_policy'] = POLICY
    expected = {('MKT-HIST-2', 'past_volatility', None, 'intact chronological VALIDATION', 'neural_mean_r2'),
                ('MKT-HIST-2', 'past_relation', None, 'intact chronological VALIDATION', 'neural_mean_r2'),
                ('TEMPO-5', 'lag_sign', 'market_scaler', 'validation_intact', 'neural_head_accuracy')}
    require({(r['dataset_id'], r['task'], r['input_policy'], r['view'], r['metric']) for r in replacements} == expected,
            'Exactly three strict final scoped quality leaders')
    ledger.setdefault('stage_scorecards', []).extend(stagecards)
    candidate = {'instance': method, 'stage': 'complete_stopped_failed_stronger_goal22', 'pretrained_ancestor': 'none',
                 'source_updates': 0, 'new_causal_updates': 2048, 'gradient_policy': GRADIENT,
                 'card': by_id['experiment_card'], 'report': 'doc/CAUSAL_FRESH_V5_RESULTS.md',
                 'review': 'doc/CAUSAL_FRESH_V5_REVIEW.md', 'detail': 'doc/CAUSAL_FRESH_V5_DETAIL.md',
                 'instance_record': FAMILY + 'instances/CPB-v5.market-01.json', 'comparison': by_id['comparison'],
                 'table_rows': by_id['v5_table'], 'baseline': 'CPB-v3.market-01', 'baseline_changed': False,
                 'full_coverage': True, 'trust_admission_passed': True, 'numeric_passed': 7, 'numeric_total': 22,
                 'advancement_passed': False, 'failed_conditions': [r['id'] for r in comparison['numeric_conditions'] if not r['passed']],
                 'gate_scope': 'V5 stronger best-of-V3/V4 prospective22; V4 old19/22 is a different card',
                 'quality': 'Retain three scoped quality leaders and TRAIN MAE gain; stop this exact recipe without promotion',
                 'overall_winner': False, 'application_changed': False}
    candidate['human_report_assets'] = {name: by_id[name] for name in ('results', 'review', 'detail')}
    require(not any(c['instance'] == method for c in old['candidates']), 'V5 not already finalized')
    ledger['candidates'].append(candidate)
    ledger.setdefault('direction_history', []).append(copy.deepcopy(old['next_direction']))
    ledger['next_direction'] = {'status': 'proposal_only_saved_TRAIN_diagnosis',
        'action': 'Saved TRAIN-only V4/V5 comparison of uncertainty routing, volatility/activity and reconstruction tradeoffs before any separately declared candidate',
        'goal_active': True, 'validation_in_diagnosis': False, 'encoder_calls': 0, 'head_fits': 0,
        'head_tuning': False, 'later_training_authorized_by_this_finalization': False,
        'exact_V5_recipe_stopped': True}
    ledger.setdefault('stage_leader_update_history', []).append(copy.deepcopy(old['stage_leader_update']))
    ledger['stage_leader_update'] = {'candidate': method, 'current_leader_scopes': 16, 'strictly_improved_scopes': 3,
        'earlier_scopes_retained': 13, 'all_prior_records_preserved': True, 'stage_scorecards': len(ledger['stage_scorecards']),
        'policy': POLICY, 'prior_live_ledger': by_id['ledger']}
    reconstruction = []
    for version, table in tables.items():
        for stage_name in ('initial', 'final'):
            row = {'instance': 'CPB-v%d.market-01' % version, 'checkpoint_stage': stage_name,
                   'causal_updates': 0 if stage_name == 'initial' else 2048, 'source_updates': 0,
                   'native_width': 40 if version == 5 else 36, 'table_source': by_id['v%d_table' % version],
                   'queries': {split: {key: table['fixed_queries'][stage_name][split][key]
                               for key in ('standardized_fixed_query_MAE', 'Gaussian_NLL_F64', 'valid_rows', 'target_cells')}
                               for split in ('train', 'validation')},
                   'selection_policy': 'All queries retained; descriptive final-only scope leader, no checkpoint selection'}
            reconstruction.append(row)
    ledger['reconstruction_stage_scorecards'] = reconstruction
    reconstruction_leaders = []
    for split in ('train', 'validation'):
        for metric in ('standardized_fixed_query_MAE', 'Gaussian_NLL_F64'):
            saved = [{'method': r['instance'], 'score': finite(r['queries'][split][metric]),
                      'checkpoint_stage': 'final', 'table_source': r['table_source']} for r in reconstruction if r['checkpoint_stage'] == 'final']
            winner = min(saved, key=lambda r: r['score'])
            reconstruction_leaders.append({'dataset_id': 'MKT-HIST-2', 'scope': 'unchanged saved masked fixed query',
                'split': split, 'metric': metric, 'method': winner['method'], 'score': winner['score'],
                'all_final_scores': saved, 'direction': 'lower_is_better',
                'numeric_advancement_floor': not (split == 'train' and metric == 'Gaussian_NLL_F64'),
                'descriptive_only': True, 'no_new_floor_added': True})
    require([r['method'] for r in reconstruction_leaders] == ['CPB-v5.market-01', 'CPB-v3.market-01', 'CPB-v4.market-01', 'CPB-v4.market-01'], 'Four retained reconstruction strengths')
    ledger['reconstruction_stage_leaders'] = reconstruction_leaders
    ledger['updated_utc'] = now
    for key in ('candidates', 'stage_scorecards', 'leader_history', 'displaced_leaders', 'candidate_status_history', 'direction_history'):
        require(ledger[key][:len(old[key])] == old[key], 'Old complete history preserved: ' + key)
    current = copy.deepcopy(catalog)
    current.setdefault('fresh_research_candidate_history', []).append(copy.deepcopy(catalog['fresh_research_candidate']))
    current['fresh_research_candidate'] = copy.deepcopy(candidate)
    current['fresh_research_candidate'].update(design_file='CPB-v5.json', physical_version_root='v5', measured_instance_file='instances/CPB-v5.market-01.json')
    current['date'] = '2026-10-16'
    require(current['active_application'] == 'CPB-v3.market-01' and current['clean_causal_training_baseline'] == catalog['clean_causal_training_baseline'], 'V3 selected baseline/application unchanged')
    descriptor = {'kind': 'causal_encoder_postmeasurement_version_descriptor_v1', 'family': 'causal_patch_bottleneck_mae',
        'version': 'CPB-v5', 'instance': method, 'created_utc': now, 'created_after_measurement': True,
        'declaration_provenance': 'This descriptor is late metadata; the prospective experiment was frozen in the independently pinned V5 card before build/training/quality',
        'card': by_id['experiment_card'], 'physical_version_root': 'versions/v5',
        'public_entry': 'versions/v5/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v5.h',
        'architecture_copy_parent': 'CPB-v4', 'learned_weight_ancestor': 'none', 'source_updates': 0,
        'native_width': 40, 'total_parameters': 298640, 'learned_values': 294752, 'learned_tensors': 76,
        'fixed_prior_values': 3888, 'gradient_policy': GRADIENT, 'stage': candidate['stage'],
        'mechanism': 'Independent live bank270-to-mean4 and bank270-to-uncertainty4; mean reads parent32+mean4, variance reads uncertainty4; unchanged full Gaussian NLL',
        'baseline': 'CPB-v3.market-01', 'baseline_changed': False, 'application_changed': False, 'metadata_only': True}
    instance = {'kind': 'causal_encoder_measured_instance_v1', 'date': '2026-10-16', 'version': 'CPB-v5', 'instance': method,
        'initial_quality_label': method + '.init', 'protocol': 'causal-fresh-v5', 'procedure_status': candidate['stage'],
        'clean_causal_candidate_measured': True, 'advancement_passed': False, 'overall_winner': False,
        'clean_baseline': 'CPB-v3.market-01', 'clean_baseline_unchanged': True, 'application_selection': 'CPB-v3.market-01',
        'application_selection_changed': False, 'historical_reference_is_quality_gate': False, 'new_research_published': False,
        'version_root': FAMILY + 'v5', 'physical_owned_family_files': card['physical_owned_files']['family'],
        'physical_owned_research_files': card['physical_owned_files']['research'],
        'frozen_source_capsule_files': 97, 'source_manifest': by_id['source_manifest'], 'source_capsule': inputs['mirror']['preserved_source_packages'],
        'build_admission': by_id['build'], 'binaries': inputs['build']['binaries'], 'card': by_id['experiment_card'],
        'gradient_policy': GRADIENT, 'encoder_gradient_kind': training['encoder_gradient_kind'],
        'numeric_objective': 'Unchanged complete Gaussian NLL with disjoint live uncertainty/mean paths',
        'native_width': 40, 'native_layout': descriptor['mechanism'], 'initialization_kind': training['initialization_kind'],
        'initialization_seed': 943001, 'pretrained_ancestor': 'none', 'source_updates': 0, 'inherited_learned_values': 0,
        'fresh_learned_values': 294752, 'fresh_learned_tensors': 76, 'analytic_prior_values': 3888, 'total_parameters': 298640,
        'fresh_initialization': by_id['initialization'], 'fresh_calibration': by_id['calibration'],
        'input_scaler_identity': training['fresh_scaler_identity'], 'bank_identity': training['fresh_bank_identity'],
        'new_causal_updates': 2048, 'attempts': 2048, 'batch_size': 8, 'training_runs': 1, 'training_replays': 0,
        'training_summary': by_id['training'], 'typed_resume': by_id['resume'],
        'all_learned_tensors_with_nonzero_CUDA_gradients_and_movement': True,
        'fixed_prior_scaler_normalizer_unchanged': training['fixed_prior_scaler_normalizer_unchanged'],
        'GPU_compute_seconds': training['GPU_step_seconds_synchronized'], 'training_trajectory_wall_seconds': training['training_trajectory_wall_seconds'],
        'calibration_seconds': training['calibration_seconds'], 'timing_scope': training['GPU_step_timing_scope'],
        'main_checkpoints': actual_checkpoints, 'checkpoints_local_git_ignored': True,
        'initial_checkpoint': next(p for p in actual_checkpoints if p['path'].endswith('/inference-0.pt')),
        'usable_final_checkpoint': next(p for p in actual_checkpoints if p['path'].endswith('/inference-2048.pt')),
        'stable_clean_baseline_checkpoint': copy.deepcopy(old['baseline_checkpoint']),
        'engineering_proof': by_id['engineering'], 'variance_gradient_proof': by_id['gradient'], 'gradient_checks': 31,
        'initial_causal_proof': by_id['initial_proof'], 'final_causal_proof': by_id['final_proof'],
        'causal_proof_scope': {'cutoffs': 128, 'suffix_cases_per_stage': 2350, 'internal_temporal_cases_per_stage': 30, 'state_unchanged': True},
        'independent_training_audit': by_id['training_audit'], 'independent_training_audit_checks': 192589,
        'independent_quality_audit': by_id['quality_audit'], 'independent_quality_audit_checks': 11300,
        'final_quality_report': candidate['report'], 'review': candidate['review'], 'detail': candidate['detail'],
        'human_report_assets': candidate['human_report_assets'],
        'table_rows': by_id['v5_table'], 'comparison': by_id['comparison'], 'quality_execution_receipt': by_id['quality_receipt'],
        'quality': {'numeric_passed': 7, 'numeric_total': 22, 'trust_admission_passed': True, 'full_coverage': True,
                    'advancement_passed': False, 'failed_conditions': candidate['failed_conditions'], 'gate_scope': candidate['gate_scope'],
                    'strict_new_scoped_quality_leaders': 3, 'initial_and_final_descriptive_scorecards_retained': True},
        'next_direction': ledger['next_direction'], 'local_run_root': 'output/runs/causal-fresh-v5/run-001',
        'local_quality_root': 'output/runs/causal-fresh-v5/run-001/quality-001',
        'local_engineering_root': 'output/engineering/causal-fresh-v5/attempt-003', 'container_root': card['container_attempt'],
        'preservation': {'complete_mirror': by_id['mirror'], 'baseline_changed': False, 'V4_reference_intact': True}}
    outputs = { 'doc/clean_causal_encoder_research_ledger.json': ledger,
                FAMILY + 'current.json': current, FAMILY + 'CPB-v5.json': descriptor,
                FAMILY + 'instances/CPB-v5.market-01.json': instance }
    output_pins = []
    for relative, value in outputs.items():
        path = args.output / relative
        write(path, value)
        output_pins.append(pin(path, relative))
    receipt = {'kind': 'cpb_v5_metadata_only_finalization_proposal_v1', 'status': 'passed_proposal_not_applied',
        'created_utc': now, 'generator': own, 'card': pin(args.card), 'inputs': bindings, 'outputs': output_pins,
        'old_candidates_preserved': len(old['candidates']), 'old_stage_scorecards_preserved': len(old['stage_scorecards']),
        'old_quality_leaders_preserved_in_full_history': 16, 'strict_new_quality_leaders': replacements,
        'new_stage_scorecards': 2, 'reconstruction_stage_scorecards': 6, 'reconstruction_scopes': 4,
        'baseline_application_unchanged': True, 'V4_reference_preserved': True, 'original_numeric_gate': '7/22 stronger V5 card',
        'new_numeric_floors': 0, 'model_calls': 0, 'head_fits': 0, 'training_updates': 0, 'live_files_written': 0,
        'publication': False, 'apply_required': 'Root independent review and explicit file application remain separate'}
    write(args.output / 'finalization-proposal.json', receipt)
    print(json.dumps({'status': receipt['status'], 'output': str(args.output), 'metadata_files': 4, 'strict_new_quality_leaders': 3}))


if __name__ == '__main__':
    main()

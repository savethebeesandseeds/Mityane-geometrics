# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Owned V6 metadata-only proposal generator, derived from the complete V5 tool.

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
METHOD = 'CPB-v6.market-01'
BASELINE = 'CPB-v3.market-01'
POLICY = 'Strictly higher saved final score updates only; exact ties retain earlier leader. Initial stages are descriptive, never checkpoint selection or overall promotion'
OBJECTIVE = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
GRADIENT = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
OBJECTIVE_FIELDS = dict(objective_policy=OBJECTIVE, mean_huber_delta=1,
                        loss_coefficients=[1, 1], variance_error_detached=True,
                        reporting_nll_backpropagated=False)
REQUIRED_INPUTS = frozenset(('ledger', 'catalog', 'prior_snapshot', 'descriptor',
    'experiment_card', 'source_manifest', 'build', 'training', 'resume',
    'initialization', 'calibration', 'engineering', 'gradient', 'initial_proof',
    'final_proof', 'proof_recovery', 'proof_recovery_stage', 'training_audit',
    'quality_audit', 'pairings', 'establishment', 'mirror', 'quality_receipt',
    'comparison', 'v3_table', 'v4_table', 'v5_table', 'v6_table',
    'results', 'review', 'detail'))
FINALIZATION_ROOT = pathlib.Path('/opt/cuwacunu/research/causal-fresh-v6-finalization')
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
            measurement_reused_for_v6_comparison=False, table_source=source,
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
        for key in ('mean_Huber_F64', 'variance_NLL_F64', 'total_objective_F64'):
            if key in query:
                queries[split][key] = finite(query[key])
    return dict(instance='CPB-v%d.market-01' % version, checkpoint_stage=stage,
        causal_updates=0 if stage == 'initial' else 2048, source_updates=0,
        native_width=40 if version >= 5 else 36, table_source=source, queries=queries,
        selection_policy='All queries retained; descriptive final-only scope leader, no checkpoint selection')


def run(args):
    require(sys.platform.startswith('linux') and pathlib.Path('/opt/cuwacunu/research').is_dir(),
            'Existing Linux research container only')
    require(args.output.is_absolute() and args.output.resolve().is_relative_to(FINALIZATION_ROOT)
            and args.output.resolve() != FINALIZATION_ROOT, 'New exclusive finalization leaf only')
    card = json.loads(args.card.read_text(encoding='utf-8-sig'))
    require(card['kind'] == 'cpb_v6_metadata_only_finalization_card_v1', 'Own finalization card required')
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
    for input_id, suffix in (('ledger', 'doc/clean_causal_encoder_research_ledger.json'),
                             ('catalog', FAMILY + 'current.json')):
        matches = [p for p in snapshot_files if p['path'].replace('\\', '/').endswith(suffix)]
        require(len(matches) == 1 and all(matches[0][k] == by_id[input_id][k] for k in ('bytes', 'sha256')),
                'Prior-live snapshot binds ' + input_id)
    old, catalog = inputs['ledger'], inputs['catalog']
    require(old['baseline'] == BASELINE and catalog['active_application'] == BASELINE,
            'Protected V3 baseline/application')
    require(catalog['fresh_research_candidate']['instance'] == 'CPB-v5.market-01', 'Whole V5 current entry available')
    require(not any(c['instance'] == METHOD for c in old['candidates']), 'V6 not already finalized')
    tables = {version: inputs['v%d_table' % version] for version in (3, 4, 5, 6)}
    comparison, training = inputs['comparison'], inputs['training']
    conditions = comparison['numeric_conditions']
    require(len(conditions) == comparison['numeric_total'] == 22 and len({r['id'] for r in conditions}) == 22,
            'Original unique22 numeric conditions, no added floor')
    require(all(type(row['passed']) is bool for row in conditions), 'Boolean numeric outcomes')
    numeric_passed = sum(row['passed'] for row in conditions)
    require(comparison['numeric_passed'] == numeric_passed, 'Actual numeric count derived from22 records')
    require(comparison['candidate'] == METHOD and comparison['runtime_trust_admission_passed'] is True
            and comparison['trust_admission_passed'] is False and comparison['overall_establishment_passed'] is False
            and comparison['advancement_conditions_passed'] is False, 'Producer runtime-only pending trust preserved')
    require(all(value == 'pending' for value in comparison['mandatory_postmeasurement_trust'].values())
            and set(comparison['mandatory_postmeasurement_trust']) == {'initial_full40_pairings', 'independent_training_audit', 'independent_quality_audit'},
            'Producer mandatory postmeasurement trust remains pending')
    objective(comparison, 'Producer comparison')
    objective(training, 'Training summary')
    require(training['new_causal_updates'] == training['attempted_batches'] == 2048
            and training['source_updates'] == 0 and training['batch_size'] == 8, 'One fresh2048/B8 trajectory')
    require(training['parameters'] == 298640 and training['parameters_with_recorded_updates'] == 294752
            and training['fixed_prior_parameters'] == 3888 and training['native_width'] == 40, 'Actual native40 learned/fixed partition')
    require(training['typed_resume_512_exact'] and training['fixed_prior_scaler_normalizer_unchanged']
            and training['backward_tensor'] == 'objective.total_loss', 'Actual typed resume/frozen state/composite backward')
    objective(inputs['initialization'], 'Fresh initialization', require_gradient_policy=False)
    for name in ('engineering', 'initial_proof', 'final_proof'):
        objective(inputs[name], name)
    require(inputs['gradient']['status'] == 'passed' and inputs['engineering']['status'] == 'passed',
            'Completed actualCUDA mechanism and full-family engineering evidence')
    objective(inputs['gradient'], 'Mechanism proof')
    for name in ('initial_proof', 'final_proof'):
        proof = inputs[name]
        require(proof['status'] == 'passed' and proof['all_cutoffs'] == 128 and proof['suffix_cases'] == 2350
                and proof['internal_temporal_cases'] == 30 and proof['state_unchanged'], 'Complete saved-stage causal proof')
    require(inputs['proof_recovery']['model_source_or_weights_changed'] is False
            and inputs['proof_recovery_stage']['status'] == 'passed'
            and inputs['proof_recovery']['corrected_stage'] == 'final-saved-proof-attempt002'
            and inputs['proof_recovery_stage']['argv'] == inputs['proof_recovery']['corrected_argv']
            and inputs['proof_recovery_stage']['binary']['sha256'] == inputs['proof_recovery']['unchanged_binary_sha256'],
            'Additive final saved-proof invocation recovery retained')
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
    establishment_ids = {'experiment_card', 'comparison', 'v6_table', 'pairings', 'training_audit', 'quality_audit'}
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
                 'actual_pre_step_composite_objective'):
        module = inputs['training_audit'][name]
        count = module['checks']
        require(module['status'] == 'passed' and type(count) is int and count > 0,
                'Actual named independent training-audit module checks')
        training_audit_modules[name] = count
    training_audit_total = audit_counts['training_audit'] + sum(training_audit_modules.values())
    full_coverage = comparison['full_coverage'] is True
    require(full_coverage, 'Complete finite saved quality coverage required')
    require(inputs['quality_receipt']['status'] == 'passed', 'Completed saved quality execution')
    numeric_success = numeric_passed == 22
    advancement = numeric_success and full_coverage  # All independent trust assertions passed above.
    establishment = inputs['establishment']
    require(establishment['measured_candidate_established'] is True
            and establishment['initial_full40_pairings_passed'] is True
            and establishment['independent_training_audit_passed'] is True
            and establishment['independent_quality_audit_passed'] is True,
            'Measured trusted candidate established separately from numeric advancement')
    require(establishment['numeric_passed'] == numeric_passed and establishment['numeric_total'] == 22
            and establishment['numeric_conditions_passed'] is numeric_success
            and establishment['overall_establishment_passed'] is advancement
            and establishment['advancement_conditions_passed'] is advancement
            and establishment['overall_winner'] is False,
            'Separate final receipt agrees with actual numeric conditions and declares no overall winner')
    stage = inputs['source_manifest']
    require(len(stage['files']) == card['expected_source_capsule_files'] == 102
            and len(inputs['build']['binaries']) == card['expected_binaries'] == 6, 'Immutable102 source/six binary closure')
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
    require(descriptor['version'] == 'CPB-v6' and descriptor['instance'] == METHOD
            and descriptor['created_after_measurement'] is False, 'Existing prospective descriptor, never regenerated')
    ledger = copy.deepcopy(old)
    new_stagecards = stagecards(tables[6], 6, by_id['v6_table'])
    require(len(old['stage_scorecards']) == 6 and len(old['stage_leaders']) == 16, 'All six prior stages/16quality scopes')
    require({row['method'] for row in old['stage_scorecards']} == {
        'CPB-v%d.market-01' % v + suffix for v in (3, 4, 5) for suffix in ('', '.init')}, 'All prior stage labels')
    replacements = []
    for leader in ledger['stage_leaders']:
        original = copy.deepcopy(leader)
        finals = []
        for version in (3, 4, 5, 6):
            row = quality_row(tables[version], version, leader, 'final')
            finals.append(dict(method=row['method'], score=row[leader['metric']], checkpoint_stage='final',
                               table_source=by_id['v%d_table' % version]))
        old_best = max(finals[:3], key=lambda row: row['score'])
        require((old_best['method'], old_best['score']) == (original['method'], original['score']),
                'Prior complete leader matches all three saved final score tables')
        best = max(finals, key=lambda row: row['score'])  # Stable chronological tie keeps earlier.
        additions = [dict(method=row['method'], score=row[leader['metric']], checkpoint_stage=stage_name)
                     for stage_name in ('initial', 'final')
                     for row in (quality_row(tables[6], 6, leader, stage_name),)]
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
            selected = quality_row(tables[6], 6, leader, 'final')
            if 'recipe_id' in selected:
                leader['recipe_id'] = selected['recipe_id']
        leader['all_final_scores'] = finals
        leader['selection_policy'] = POLICY
    ledger['stage_scorecards'].extend(new_stagecards)
    reconstruction = copy.deepcopy(old['reconstruction_stage_scorecards'])
    require(len(reconstruction) == 6, 'All six reconstruction stagecards retained')
    require([(r['instance'], r['checkpoint_stage']) for r in reconstruction] == [
        ('CPB-v%d.market-01' % v, s) for v in (3, 4, 5) for s in ('initial', 'final')],
        'Chronological reconstruction order preserves earlier exact ties')
    for version in (3, 4, 5):
        for stage_name in ('initial', 'final'):
            expected = reconstruction_card(tables[version], version, stage_name, by_id['v%d_table' % version])
            actual = [r for r in reconstruction if r['instance'] == expected['instance'] and r['checkpoint_stage'] == stage_name]
            require(len(actual) == 1 and actual[0]['queries'] == expected['queries'], 'Prior reconstruction data exact')
    reconstruction.extend(reconstruction_card(tables[6], 6, s, by_id['v6_table']) for s in ('initial', 'final'))
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
        new_causal_updates=2048, gradient_policy=GRADIENT, **OBJECTIVE_FIELDS,
        card=by_id['experiment_card'], report='doc/CAUSAL_FRESH_V6_RESULTS.md',
        review='doc/CAUSAL_FRESH_V6_REVIEW.md', detail='doc/CAUSAL_FRESH_V6_DETAIL.md',
        instance_record=FAMILY + 'instances/CPB-v6.market-01.json', comparison=by_id['comparison'],
        table_rows=by_id['v6_table'], baseline=BASELINE, baseline_changed=False, full_coverage=True,
        runtime_trust_admission_passed=True, trust_admission_passed=True, measured_candidate_established=True,
        overall_establishment_passed=advancement,
        postmeasurement_establishment=by_id['establishment'], initial_full40_pairings=by_id['pairings'],
        numeric_passed=numeric_passed, numeric_total=22, advancement_passed=advancement,
        failed_conditions=[r['id'] for r in conditions if not r['passed']],
        gate_scope='V6 prospective strongest-final-V3/V4/V5 retention22; historical counts retain their own cards',
        quality='Preserve each saved scoped gain and failure; initial stages descriptive; no automatic application selection',
        overall_winner=False, application_changed=False,
        human_report_assets={name: by_id[name] for name in ('results', 'review', 'detail')})
    ledger['candidates'].append(candidate)
    ledger.setdefault('direction_history', []).append(copy.deepcopy(old['next_direction']))
    ledger['next_direction'] = dict(status=card['next_direction']['status'], action=card['next_direction']['action'],
        goal_active=True, validation_in_diagnosis=False, encoder_calls=0, head_fits=0, head_tuning=False,
        later_training_authorized_by_this_finalization=False, exact_V6_recipe_complete=True,
        exact_V6_recipe_stopped=not advancement)
    ledger.setdefault('stage_leader_update_history', []).append(copy.deepcopy(old['stage_leader_update']))
    ledger['stage_leader_update'] = dict(candidate=METHOD, current_leader_scopes=16,
        strictly_improved_scopes=len(replacements), earlier_scopes_retained=16-len(replacements),
        all_prior_records_preserved=True, stage_scorecards=8, policy=POLICY, prior_live_ledger=by_id['ledger'])
    ledger['updated_utc'] = now
    for key in ('candidates', 'stage_scorecards', 'leader_history', 'displaced_leaders', 'candidate_status_history',
                'direction_history', 'stage_leader_update_history', 'reconstruction_stage_scorecards'):
        require(ledger[key][:len(old[key])] == old[key], 'Old complete history preserved: ' + key)
    current = copy.deepcopy(catalog)
    current.setdefault('fresh_research_candidate_history', []).append(copy.deepcopy(catalog['fresh_research_candidate']))
    current['fresh_research_candidate'] = copy.deepcopy(candidate)
    current['fresh_research_candidate'].update(design_file='CPB-v6.json', physical_version_root='v6',
                                              measured_instance_file='instances/CPB-v6.market-01.json')
    current['date'] = card['client_date']
    require(current['active_application'] == BASELINE
            and current['clean_causal_training_baseline'] == catalog['clean_causal_training_baseline'],
            'V3 baseline/application exact')
    instance = dict(kind='causal_encoder_measured_instance_v1', date=card['client_date'], version='CPB-v6',
        instance=METHOD, initial_quality_label=METHOD+'.init', protocol='causal-fresh-v6',
        procedure_status=candidate_stage, clean_causal_candidate_measured=True,
        measured_candidate_established=True, overall_establishment_passed=advancement,
        clean_baseline=BASELINE, clean_baseline_unchanged=True, application_selection=BASELINE,
        application_selection_changed=False, historical_reference_is_quality_gate=False,
        overall_winner=False, advancement_passed=advancement, new_research_published=False,
        version_root=FAMILY+'v6', prospective_descriptor=by_id['descriptor'], prospective_descriptor_rewritten=False,
        physical_owned_family_files=card['physical_owned_files']['family'],
        physical_owned_research_files=card['physical_owned_files']['research'], frozen_source_capsule_files=102,
        source_manifest=by_id['source_manifest'], source_capsule=inputs['mirror']['preserved_source_packages'],
        build_admission=by_id['build'], binaries=inputs['build']['binaries'], card=by_id['experiment_card'],
        gradient_policy=GRADIENT, **OBJECTIVE_FIELDS, encoder_gradient_kind=training['encoder_gradient_kind'],
        shared_global_clip=True, optimizer_independence_claimed=False,
        training_objective='R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v+log(v))], unit coefficients',
        reporting_metrics='Unchanged saved Gaussian NLL/standardized and physical MAE; reporting NLL is never backpropagated',
        native_width=40, native_layout=card['native_layout'], initialization_kind=training['initialization_kind'],
        initialization_seed=943001, pretrained_ancestor='none', source_updates=0, inherited_learned_values=0,
        fresh_learned_values=294752, fresh_learned_tensors=76, analytic_prior_values=3888, total_parameters=298640,
        fresh_initialization=by_id['initialization'], fresh_calibration=by_id['calibration'],
        input_scaler_identity=training['fresh_scaler_identity'], bank_identity=training['fresh_bank_identity'],
        initial_full40_pairings=by_id['pairings'], new_causal_updates=2048, attempts=2048, batch_size=8,
        training_runs=1, training_replays=0, training_summary=by_id['training'], typed_resume=by_id['resume'],
        all_learned_tensors_with_nonzero_CUDA_gradients_and_movement=True,
        fixed_prior_scaler_normalizer_unchanged=True, GPU_compute_seconds=training['GPU_step_seconds_synchronized'],
        training_trajectory_wall_seconds=training['training_trajectory_wall_seconds'], calibration_seconds=training['calibration_seconds'],
        timing_scope=training['GPU_step_timing_scope'], main_checkpoints=checkpoint_pins, checkpoints_local_git_ignored=True,
        initial_checkpoint=next(p for p in checkpoint_pins if p['path'].endswith('/inference-0.pt')),
        usable_final_checkpoint=next(p for p in checkpoint_pins if p['path'].endswith('/inference-2048.pt')),
        stable_clean_baseline_checkpoint=copy.deepcopy(old['baseline_checkpoint']), engineering_proof=by_id['engineering'],
        variance_gradient_proof=by_id['gradient'], initial_causal_proof=by_id['initial_proof'], final_causal_proof=by_id['final_proof'],
        final_saved_proof_invocation_recovery=by_id['proof_recovery'], final_saved_proof_recovery_stage=by_id['proof_recovery_stage'],
        causal_proof_scope=dict(cutoffs=128, suffix_cases_per_stage=2350, internal_temporal_cases_per_stage=30, state_unchanged=True),
        independent_training_audit=by_id['training_audit'], independent_training_audit_checks=audit_counts['training_audit'],
        independent_training_audit_core_checks=audit_counts['training_audit'],
        independent_training_audit_module_checks=training_audit_modules,
        independent_training_audit_core_plus_modules_checks=training_audit_total,
        independent_quality_audit=by_id['quality_audit'], independent_quality_audit_checks=audit_counts['quality_audit'],
        postmeasurement_establishment=by_id['establishment'], producer_pending_trust_preserved=True,
        final_quality_report=candidate['report'], review=candidate['review'], detail=candidate['detail'],
        human_report_assets=candidate['human_report_assets'], table_rows=by_id['v6_table'], comparison=by_id['comparison'],
        quality_execution_receipt=by_id['quality_receipt'], quality=dict(numeric_passed=numeric_passed, numeric_total=22,
            runtime_trust_admission_passed=True, trust_admission_passed=True, full_coverage=True,
            advancement_passed=advancement, failed_conditions=candidate['failed_conditions'], gate_scope=candidate['gate_scope'],
            strict_new_scoped_quality_leaders=len(replacements), initial_and_final_descriptive_scorecards_retained=True),
        next_direction=ledger['next_direction'], local_run_root='output/runs/causal-fresh-v6/run-001',
        local_quality_root='output/runs/causal-fresh-v6/run-001/quality-001',
        local_engineering_root='output/engineering/causal-fresh-v6/attempt-001', container_root=card['container_attempt'],
        preservation=dict(complete_mirror=by_id['mirror'], baseline_changed=False, V4_reference_intact=True, V5_reference_intact=True))
    outputs = {'doc/clean_causal_encoder_research_ledger.json': ledger, FAMILY+'current.json': current,
               FAMILY+'instances/CPB-v6.market-01.json': instance}
    require(not args.output.exists(), 'Exclusive proposal output required')
    args.output.mkdir(parents=True)
    output_pins = []
    for relative, value in outputs.items():
        path = bounded(args.output, relative)
        write(path, value)
        output_pins.append(pin(path, relative))
    receipt = dict(kind='cpb_v6_metadata_only_finalization_proposal_v1', status='passed_proposal_not_applied',
        created_utc=now, generator=own, card=pin(args.card), inputs=bindings, outputs=output_pins,
        old_candidates_preserved=len(old['candidates']), old_stage_scorecards_preserved=6,
        old_quality_leaders_preserved_in_full_history=16, strict_new_quality_leaders=replacements,
        new_stage_scorecards=2, total_stage_scorecards=8, reconstruction_stage_scorecards=8,
        reconstruction_scopes=4, baseline_application_unchanged=True, V4_reference_preserved=True, V5_reference_preserved=True,
        prospective_descriptor_unchanged=by_id['descriptor'], producer_pending_trust_preserved=by_id['comparison'],
        separate_establishment_receipt=by_id['establishment'], numeric_passed=numeric_passed, numeric_total=22,
        training_audit_core_checks=audit_counts['training_audit'], training_audit_module_checks=training_audit_modules,
        training_audit_core_plus_modules_checks=training_audit_total,
        measured_candidate_established=True, overall_establishment_passed=advancement,
        advancement_conditions_passed=advancement, overall_winner=False, new_numeric_floors=0,
        model_calls=0, head_fits=0, training_updates=0, live_files_written=0, publication=False,
        apply_required='Root independent review and explicit three-file application remain separate')
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
            write(failure, dict(kind='cpb_v6_metadata_only_finalization_failure_v1', status='failed_not_applied',
                error=type(error).__name__+': '+str(error), generator=pin(__file__), card=pin(args.card),
                model_calls=0, head_fits=0, live_files_written=0))
        raise


if __name__ == '__main__':
    main()

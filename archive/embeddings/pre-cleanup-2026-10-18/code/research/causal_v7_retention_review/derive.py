# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Derive prospective retention floors from five saved final aggregate tables.

Owned standard-library metadata join. No model, head, tensor, fit, training,
quality scoring, old encoder replay, live application or historical rescoring.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('/opt/cuwacunu/research/causal-v7-retention-review')
INPUT_IDS = {'parent', 'v3', 'v4', 'v5', 'v6', 'v7', 'ledger', 'establishment', 'preservation'}
PARENT_SHA = '0965842331056b61faee827a5ea782703205b23bd326298929ed3ecfca4604fb'
EXPECTED_CHANGES = {'past_volatility:neural_mean_r2', 'train:standardized_MAE'}
STABLE = ('establishment', 'scope', 'defaults', 'version_policy', 'preservation',
          'search_policy', 'future_dimensions', 'application_promotion',
          'publication', 'container', 'independent_confirmation', 'baseline',
          'application', 'additional_train_NLL_gate')


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def pin(path, label=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked metadata')
    data = path.read_bytes()
    return dict(path=label or str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def scalar(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Finite saved aggregate')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(sys.platform.startswith('linux') and Path('/.dockerenv').is_file(), 'Existing cuwacunu container only')
    require(args.output.is_absolute() and args.output.resolve().is_relative_to(ROOT)
            and args.output.resolve() != ROOT and not args.output.exists(), 'New exclusive retention output leaf')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    require(manifest['kind'] == 'clean_causal_v7_retention_derivation_manifest_v1', 'Own prospective manifest')
    require(manifest['authorized_for_derivation'] is True, 'Root derivation admission required')
    require(all(pin(__file__)[key] == manifest['source'][key] for key in ('bytes', 'sha256')),
            'Frozen owned source')
    data, bindings, paths = {}, {}, {}
    for row in manifest['inputs']:
        require(row['id'] not in data, 'Unique nine input IDs')
        relative = Path(row['staged_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Bounded staged input path')
        path = args.manifest.parent/relative
        actual = pin(path, row['path'])
        require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Exact saved input: '+row['id'])
        data[row['id']] = json.loads(path.read_text(encoding='utf-8-sig'))
        bindings[row['id']], paths[row['id']] = actual, path
    require(set(data) == INPUT_IDS and bindings['parent']['sha256'] == PARENT_SHA,
            'Exact V6-retention parent and five saved controls')
    parent = data['parent']
    require(parent['criterion_count'] == len(parent['criteria']) == 22
            and len({r['id'] for r in parent['criteria']}) == 22, 'Unique original22 IDs')
    require(parent['additional_train_NLL_gate'] is False, 'No TRAIN NLL gate')
    require(data['establishment']['kind'] == 'causal_fresh_v7_postmeasurement_establishment_v1'
            and data['establishment']['candidate'] == 'CPB-v7.market-01'
            and data['establishment']['status'] == 'passed'
            and data['establishment']['trust_admission_passed'] is True
            and data['establishment']['measured_candidate_established'] is True
            and data['establishment']['full_coverage'] is True
            and data['establishment']['overall_winner'] is False
            and data['establishment']['winner_selection_made'] is False
            and data['preservation']['status'] == 'passed', 'Complete independent V7 trust and preservation')
    require(data['ledger']['baseline'] == parent['baseline'] == parent['application'] == 'CPB-v3.market-01',
            'Protected baseline/application')
    require(all(data['establishment']['evidence']['table_rows'][key] == bindings['v7'][key]
                for key in ('bytes', 'sha256')), 'Final trust receipt binds V7 table')
    tables = [(v, 'CPB-v%d.market-01' % v, data['v%d' % v]) for v in (3, 4, 5, 6, 7)]
    for _, method, table in tables:
        require(table['model'] == method and table['status'] == 'complete'
                and table['source_updates'] == 0 and table['full_coverage'] is True,
                'Exact completed fresh final-reference table')
    choices, evidence = {}, []
    for task in ('past_volatility', 'past_trend', 'past_activity', 'past_relation'):
        for metric in ('linear_r2', 'neural_mean_r2'):
            saved = []
            for version, method, table in tables:
                rows = [r for r in table['market_rows'] if r['method'] == method and r['task'] == task]
                require(len(rows) == 1 and rows[0]['checkpoint_stage'] == 'final'
                        and rows[0]['coverage'] == 1 and rows[0]['valid'] == rows[0]['total'] == 32,
                        'Complete saved final market row')
                saved.append(dict(instance=method, value=scalar(rows[0][metric]), table=bindings['v%d' % version]))
            identity = task+':'+metric
            choices[identity] = max(saved, key=lambda r:r['value'])
            evidence.append(dict(id=identity, direction='max_final', values=saved))
    for policy in ('synthetic_train', 'market_scaler'):
        for view in ('validation_intact', 'validation_deleted'):
            for metric, selector in (('linear_head_accuracy', 'ridge'), ('neural_head_accuracy', 'tiny_secondary')):
                saved = []
                for version, method, table in tables:
                    score = table['synthetic']['scores']['final'][policy][view]
                    require(score['coverage']['mean'] == 1, 'Complete saved final timing coverage')
                    saved.append(dict(instance=method, value=scalar(score[selector]['mean']), table=bindings['v%d' % version]))
                identity = policy+':'+view+':'+metric
                choices[identity] = max(saved, key=lambda r:r['value'])
                evidence.append(dict(id=identity, direction='max_final', values=saved))
    for split, metric, identity in (
            ('train', 'standardized_fixed_query_MAE', 'train:standardized_MAE'),
            ('validation', 'standardized_fixed_query_MAE', 'validation:standardized_MAE'),
            ('validation', 'Gaussian_NLL_F64', 'known_validation:Gaussian_NLL_F64')):
        saved = [dict(instance=method, value=scalar(table['fixed_queries']['final'][split][metric]),
                      table=bindings['v%d' % version]) for version, method, table in tables]
        choices[identity] = min(saved, key=lambda r:r['value'])
        evidence.append(dict(id=identity, direction='min_final', values=saved))
    require(len(choices) == 19, 'Sixteen quality plus three reconstruction floors only')
    proposed = copy.deepcopy(parent)
    changes = []
    for row in proposed['criteria']:
        if row['id'] not in choices:
            require(row['category'] in ('neural_trend_robustness', 'relation_usefulness')
                    and row['floor'] == 0, 'Exactly three unchanged positivity criteria')
            continue
        best = choices[row['id']]
        if (row['floor'], row['floor_source']) != (best['value'], best['instance']):
            previous, previous_source = row['floor'], row['floor_source']
            if row['category'] in ('market_retention', 'tempo5_retention'):
                require(best['value'] > previous, 'Quality floor only strengthens')
            else:
                require(best['value'] < previous, 'Reconstruction ceiling only strengthens')
            row['floor'], row['floor_source'] = best['value'], best['instance']
            changes.append(dict(id=row['id'], previous_threshold=previous, previous_source=previous_source,
                retained_threshold=best['value'], retained_source=best['instance'], rule=row['rule']))
    require(len(changes) == 2 and {row['id'] for row in changes} == EXPECTED_CHANGES,
            'Exactly neural volatility and TRAIN MAE strengthen; other20 conditions unchanged')
    old_by_id = {r['id']:r for r in parent['criteria']}
    for row in proposed['criteria']:
        old = old_by_id[row['id']]
        require({k:v for k,v in row.items() if k not in ('floor','floor_source')} ==
                {k:v for k,v in old.items() if k not in ('floor','floor_source')}, 'IDs/rules/tolerance/contracts unchanged')
    for key in STABLE:
        require(proposed[key] == parent[key], 'Unchanged protocol contract: '+key)
    require(proposed['criteria'][-3:] == parent['criteria'][-3:], 'Exact original positivity definitions')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    proposed.update(kind='clean_causal_encoder_search_goal_v7_retention_protocol_v1',
        declared_utc=now, declared_client_date=manifest['client_date'], state='prospective_future_use_awaiting_independent_review',
        parent_protocol=bindings['parent'], reference_ledger=bindings['ledger'],
        reference_comparison=copy.deepcopy(data['establishment']['evidence']['comparison']),
        reference_tables=[dict(instance=method, table=bindings['v%d' % v]) for v,method,_ in tables],
        scoped_references=['CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01'],
        v7_final_establishment=bindings['establishment'], v7_final_preservation=bindings['preservation'],
        changed_thresholds=changes, changed_threshold_count=2, unchanged_criteria_count=20,
        candidate_declared=False, model_declared=False, next_version_declared=False,
        review_status='derived_frozen_proposal_awaiting_independent_review',
        retained_quality_policy='Protect16 maximum final V3/V4/V5/V6/V7 quality values,2 minimum final MAE values and strict minimum final VALIDATION NLL; stable earlier ties; three positivity rules unchanged',
        historical_pass_count_policy='Prospective future floors only. Historical cards/results/counts remain unchanged and are not recomputed or ranked as identical protocols.',
        derivation_policy='Existing-container standard-library max/min of pinned saved final aggregate JSON only; no model/head calls,refits,training,new quality scores or held-out tensor diagnosis',
        future_use='No next candidate,mechanism,geometry or dimension-dependent head settings are declared; any candidate requires its own prospective source/card before execution',
        diagnosis='Any future saved TRAIN-only diagnosis requires a separately frozen card. No encoder/decoder/neural forwards,head/statistics fits,held-out tensor analysis or old encoder replays follow from this retention derivation.',
        derivation_requires_new_TRAIN_diagnosis=False,
        inherited_default_geometry_note='The inherited default width is historical protocol metadata, not a new candidate selection. Future dimensions/head settings must be separately declared.',
        derivation_manifest=pin(args.manifest), derivation_source=pin(__file__))
    proposed.setdefault('retention_derivation_history', []).append(dict(parent=bindings['parent'],
        prior_parent_protocol=copy.deepcopy(parent['parent_protocol']), prior_changed_thresholds=copy.deepcopy(parent['changed_thresholds']),
        prior_retained_quality_policy=parent['retained_quality_policy'], prior_derivation_policy=parent['derivation_policy'],
        prior_reference_comparison=copy.deepcopy(parent['reference_comparison']), prior_diagnosis=parent['diagnosis']))
    # Cross-check against already reviewed ledger; no result is fitted or rescored.
    for leader in data['ledger']['stage_leaders']:
        identity = (leader['task']+':'+leader['metric'] if leader['dataset_id']=='MKT-HIST-2'
                    else leader['input_policy']+':'+leader['view']+':'+leader['metric'])
        require((leader['score'],leader['method']) == (choices[identity]['value'],choices[identity]['instance']), 'Current saved quality leader agrees')
    for leader in data['ledger']['reconstruction_stage_leaders']:
        if leader['split']=='train' and leader['metric']=='Gaussian_NLL_F64':
            continue  # Descriptive strength retained in ledger; never a numeric gate.
        identity = ('known_validation:Gaussian_NLL_F64' if leader['metric']=='Gaussian_NLL_F64'
                    else leader['split']+':standardized_MAE')
        require((leader['score'],leader['method']) == (choices[identity]['value'],choices[identity]['instance']), 'Current saved reconstruction leader agrees')
    for identity,path in paths.items():
        require(pin(path,bindings[identity]['path']) == bindings[identity], 'Inputs unchanged after metadata reduction')
    args.output.mkdir(parents=True)
    target = args.output/'clean_causal_encoder_search_goal_v7_retention.json'
    write(target, proposed)
    lines = ['# Prospective retention of all five clean causal references', '',
        'This protocol is for future declared candidates. CPB-v3 remains baseline/application; V4/V5/V6/V7 remain preserved scoped references. No next candidate or head settings are declared.', '',
        'The same22 IDs/rules/tolerance/trust/coverage remain. Neural volatility and TRAIN MAE strengthen; the other20 criteria stay exact. Initial stages never select a floor. Historical pass counts are not recomputed. No TRAIN NLL gate is added.', '',
        'Dataset legend: MKT-HIST-2 measures past-window retention (designed complexity4/5).', '',
        'TEMPO-5 recovers a slow lag beneath fast9 features (designed complexity5/5). These ordinal design levels are not performance-derived.', '',
        '| Criterion | Previous floor/source | Future floor/source | Rule |', '|---|---|---|---|']
    lines.extend('| '+r['id']+' | '+repr(r['previous_threshold'])+' / '+r['previous_source']+' | '+repr(r['retained_threshold'])+' / '+r['retained_source']+' | '+r['rule']+' |' for r in changes)
    lines.extend(['', 'All reference pins and the complete22 criteria are in the companion JSON; the derivation receipt retains every saved final aggregate choice. Max/min uses chronological V3,V4,V5,V6,V7 order, so exact ties retain the earlier source. Strict best VALIDATION NLL remains strict. Freshness, causal/CUDA/resume proof, full coverage, preservation and separate candidate declarations remain required.', '', 'Only saved aggregate JSON was read inside the existing cuwacunu container. No models, heads, fits, encoder updates, new quality scores, historical rescoring or live application occurred. Independent review and root application remain separate.', ''])
    markdown = args.output/'clean_causal_encoder_search_goal_v7_retention.md'
    with markdown.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    receipt = dict(kind='clean_causal_v7_retention_derivation_v1', status='passed_proposal_not_applied',
        source=pin(__file__), manifest=pin(args.manifest), inputs=list(bindings.values()),
        criteria=22, selected_final_floor_scopes=19, unchanged_positivity_rules=3,
        changed_thresholds=changes, unchanged_criteria=20, saved_final_choices=evidence,
        stable_contract_fields=list(STABLE), candidate_declared=False, new_train_NLL_gate=False,
        historical_results_or_counts_recomputed=False, baseline_changed=False, model_calls=0,
        head_fits=0, training_updates=0, new_quality_scores=0, live_files_written=0,
        outputs=[pin(target),pin(markdown)])
    write(args.output/'derivation.json', receipt)
    print(json.dumps(dict(status=receipt['status'], criteria=22, strengthened_floors=2, outputs=receipt['outputs'])))


if __name__ == '__main__':
    main()

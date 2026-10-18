#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Verify prospective retention floors from saved final aggregate JSON only."""
import hashlib
import json
from pathlib import Path
import sys


def pin(path):
    data = path.read_bytes()
    return {'path': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    assert sys.platform.startswith('linux') and Path('/.dockerenv').is_file()
    root = Path(sys.argv[1]).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    for entry in manifest['files']:
        actual = pin(root/entry['path'])
        assert actual == entry, entry['path']
    load = lambda name: json.loads((root/name).read_text(encoding='utf-8-sig'))
    original, proposed = load('original-goal.json'), load('retention-goal.json')
    assert proposed['parent_protocol']['sha256'] == pin(root/'original-goal.json')['sha256']
    assert proposed['criterion_count'] == original['criterion_count'] == 22
    expected_changes = {
        'past_volatility:neural_mean_r2',
        'past_relation:neural_mean_r2',
        'market_scaler:validation_intact:neural_head_accuracy',
        'train:standardized_MAE',
    }
    old = {r['id']: r for r in original['criteria']}
    new = {r['id']: r for r in proposed['criteria']}
    assert len(old) == len(new) == 22 and old.keys() == new.keys()
    stable_contracts = ('establishment','scope','defaults','version_policy',
                        'preservation','search_policy','future_dimensions',
                        'application_promotion','publication','container','independent_confirmation')
    assert all(proposed[key] == original[key] for key in stable_contracts)
    changed = set()
    for identity in old:
        delta = {key for key in old[identity] | new[identity]
                 if old[identity].get(key) != new[identity].get(key)}
        if delta:
            assert delta == {'floor', 'floor_source'}, (identity, delta)
            changed.add(identity)
    assert changed == expected_changes
    tables = [(f'CPB-v{version}.market-01', load(f'v{version}-table.json')) for version in (3,4,5)]
    for (instance, _), binding in zip(tables, proposed['reference_tables']):
        version = instance.split('-v')[1].split('.')[0]
        actual = pin(root/f'v{version}-table.json')
        assert binding['instance'] == instance
        assert (binding['table']['bytes'], binding['table']['sha256']) == (actual['bytes'],actual['sha256'])
    expected, comparisons = {}, []
    tasks = ('past_volatility','past_trend','past_activity','past_relation')
    for task in tasks:
        for metric in ('linear_r2','neural_mean_r2'):
            values = []
            for instance, table in tables:
                rows = [r for r in table['market_rows'] if r['method'] == instance and r['task'] == task]
                assert len(rows) == 1 and rows[0]['coverage'] == 1 and rows[0]['valid'] == rows[0]['total'] == 32
                values.append((instance, rows[0][metric]))
            expected[f'{task}:{metric}'] = max(values, key=lambda x:x[1])
            comparisons.append({'id':f'{task}:{metric}', 'saved_final_values':values})
    for policy in ('synthetic_train','market_scaler'):
        for view in ('validation_intact','validation_deleted'):
            for metric, selector in (('linear_head_accuracy','ridge'),('neural_head_accuracy','tiny_secondary')):
                values = []
                for instance, table in tables:
                    score = table['synthetic']['scores']['final'][policy][view]
                    assert score['coverage']['mean'] == 1
                    values.append((instance, score[selector]['mean']))
                identity = f'{policy}:{view}:{metric}'
                expected[identity] = max(values, key=lambda x:x[1])
                comparisons.append({'id':identity, 'saved_final_values':values})
    for split in ('train','validation'):
        values = [(instance, table['fixed_queries']['final'][split]['standardized_fixed_query_MAE'])
                  for instance, table in tables]
        identity = f'{split}:standardized_MAE'
        expected[identity] = min(values, key=lambda x:x[1])
        comparisons.append({'id':identity, 'saved_final_values':values})
    values = [(instance, table['fixed_queries']['final']['validation']['Gaussian_NLL_F64'])
              for instance, table in tables]
    identity = 'known_validation:Gaussian_NLL_F64'
    expected[identity] = min(values, key=lambda x:x[1])
    comparisons.append({'id':identity, 'saved_final_values':values})
    assert len(expected) == 19
    for identity, (source, value) in expected.items():
        assert new[identity]['floor'] == value and new[identity]['floor_source'] == source, identity
    for identity in set(new) - set(expected):
        assert new[identity] == old[identity] and new[identity]['floor'] == 0
    assert proposed['baseline'] == proposed['application'] == 'CPB-v3.market-01'
    assert proposed['additional_train_NLL_gate'] is False
    assert not proposed['candidate_declared'] and not proposed['model_declared'] and not proposed['next_version_declared']
    assert proposed['reference_ledger']['sha256'] == pin(root/'ledger.json')['sha256']
    assert proposed['v5_final_preservation']['sha256'] == pin(root/'final-preservation.json')['sha256']
    for entry in manifest['files']:
        assert pin(root/entry['path']) == entry
    result = {
        'kind':'clean_causal_v5_retention_independent_review_v1', 'status':'passed',
        'original_goal':pin(root/'original-goal.json'), 'reviewed_protocol':pin(root/'retention-goal.json'),
        'source':pin(Path(__file__)), 'manifest':pin(root/'manifest.json'),
        'criteria_checked':22, 'retained_best_aggregate_scopes_checked':19,
        'positivity_rules_unchanged':3, 'changed_floor_ids':sorted(changed),
        'strict_validation_NLL_rule_unchanged':True,
        'unchanged_contract_fields':list(stable_contracts),
        'authorized_diagnosis_metadata_update':{'original':original['diagnosis'],'proposed':proposed['diagnosis'],
                                               'review':'Reference stage advances from saved V3/V4 to separately frozen saved V4/V5; prohibitions remain explicit; no numeric criterion change'},
        'prior_review_failure':{'attempt':1,'reason':'Reviewer incorrectly required the current diagnosis stage description to remain an identical historical V3/V4 string; all original/proposed bytes and failure evidence preserved'},
        'all_other_18_criteria_exact':True, 'ties_keep_earlier_source':True,
        'comparisons':comparisons, 'verified_files':manifest['files'],
        'aggregate_field_scope':'Saved final task/head scores, timing means/coverage, TRAIN/VAL MAE and VAL NLL only; no rowwise held-out diagnosis',
        'geometry_note':'The inherited default36 is not a candidate declaration; any native40 candidate needs an explicit prospective geometry/head override under unchanged future_dimensions policy',
        'original_criteria_or_historical_counts_rewritten':False,
        'training_updates':0, 'model_calls':0, 'head_fits':0, 'new_quality_scores':0,
    }
    with (root/'review.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'status':'passed','criteria':22,'changed_floors':4,'output':str(root/'review.json')}))


if __name__ == '__main__':
    main()

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Independent check of future floors using the six immutable aggregate tables.

Complete owned standard-library source; no derivation-module imports, model
libraries, tensor reads, fits, historical rescoring or live protocol writes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('/opt/cuwacunu/research/causal-v8-retention-review')
INPUTS = {'parent', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'ledger', 'establishment', 'complete_mirror', 'preservation'}
PARENT_SHA = 'c0a46527388032ea4f43dd9780ceedef486e5db7b4502ce37165d6e8616368b7'
EXPECTED_CHANGES = {'past_volatility:neural_mean_r2', 'market_scaler:validation_intact:neural_head_accuracy'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not (sys.platform.startswith('linux') and Path('/.dockerenv').is_file()
            and a.output.is_absolute() and a.output.resolve().is_relative_to(ROOT)
            and a.output.resolve() != ROOT and not a.output.exists()):
        raise RuntimeError('Existing container and new bounded metadata output only')
    checks = 0

    def check(value, reason):
        nonlocal checks
        checks += 1
        if not value:
            raise RuntimeError(reason)

    def pin(path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('Regular nonlinked metadata input')
        data = path.read_bytes()
        return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    manifest_bytes, source_bytes = a.manifest.read_bytes(), Path(__file__).read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    check(manifest['kind'] == 'clean_causal_v8_retention_derivation_manifest_v1'
          and manifest['authorized_for_derivation'] is True, 'Explicit prospective root admission')
    check(all(pin(Path(__file__))[key] == manifest['independent_reviewer'][key]
              for key in ('bytes', 'sha256')), 'Exact independently frozen review source')
    data, bindings, paths = {}, {}, {}
    for row in manifest['inputs']:
        relative = Path(row['staged_path'])
        check(not relative.is_absolute() and '..' not in relative.parts and row['id'] not in data
              and '\\' not in row['staged_path'] and relative.suffix == '.json',
              'Unique bounded metadata input')
        path = a.manifest.parent/relative
        check(path.resolve(strict=True).is_relative_to(a.manifest.parent.resolve(strict=True)),
              'Inputs stay inside staged source')
        actual = pin(path)
        check(all(actual[k] == row[k] for k in ('bytes','sha256')), 'Immutable input '+row['id'])
        data[row['id']] = json.loads(path.read_text(encoding='utf-8-sig'))
        bindings[row['id']], paths[row['id']] = row, path
    check(set(data) == INPUTS, 'Exact eleven input roles')
    check(next(row['sha256'] for row in manifest['inputs'] if row['id'] == 'parent') == PARENT_SHA,
          'Exact prior V7-retention protocol')
    parent = data['parent']
    trust, mirror, preserved = (data[name] for name in ('establishment', 'complete_mirror', 'preservation'))
    same = lambda left, right: all(left[key] == right[key] for key in ('bytes', 'sha256'))
    check(trust['kind'] == 'causal_fresh_v8_postmeasurement_establishment_v1'
          and trust['candidate'] == 'CPB-v8.market-01' and trust['status'] == 'passed'
          and all(trust[key] is True for key in ('measured_candidate_established', 'trust_admission_passed',
                  'full_coverage', 'independent_training_audit_passed', 'independent_quality_audit_passed'))
          and trust['overall_winner'] is False and trust['winner_selection_made'] is False,
          'Measured trustworthy V8 candidate without winner selection')
    check(same(trust['evidence']['table_rows'], bindings['v8'])
          and same(trust['evidence']['complete_mirror'], bindings['complete_mirror']),
          'Separate trust binds exact V8 table and Complete mirror')
    check(mirror['kind'] == 'causal_fresh_v8_container_to_local_evidence_mirror_v1'
          and mirror['status'] == 'passed' and mirror['scope'] == 'Complete' and mirror['attempt'] == 'attempt-001'
          and mirror['source_files_verified'] == 113 and mirror['all_mirrored_pt_ignored'] is True
          and mirror['source_stage']['status'] == 'passed'
          and mirror['source_stage']['card_sha256'] ==
              '6acfaadfe758c0d252c8138fc2830807d40444f9a9adfa61f85ec13dde7c63a6'
          and mirror['source_stage']['manifest_sha256'] ==
              '8daba935ee45332b892225c6994a00b59ad95b2e4da4fe7348f6c6518266cf2e'
          and mirror['source_stage']['archive_sha256'] ==
              'bc883e6e14365f68e02e7f5b0e5c3df9eeca204cbdb7140d34de775768d14591'
          and any(same(item, bindings['v8']) for item in mirror['files']),
          'Complete mirror of admitted113-source V8 table')
    check(preserved['kind'] == 'causal_fresh_v8_final_preservation_v1' and preserved['status'] == 'passed'
          and preserved['baseline'] == 'CPB-v3.market-01' and preserved['candidate'] == 'CPB-v8.market-01'
          and preserved['baseline_changed'] is False and preserved['application_changed'] is False
          and preserved['critical_checkpoint_total'] == len(preserved['critical_checkpoint_pins']) == 94
          and len({(item['repository'], item['path']) for item in preserved['critical_checkpoint_pins']}) == 94
          and len(preserved['all_pinned_PT_ignored']) == 2
          and {item['repository'] for item in preserved['all_pinned_PT_ignored']} ==
              {'cuwacunu_embedding', 'cuwacunu_dev'}
          and all(item['all_ignored'] is True for item in preserved['all_pinned_PT_ignored'])
          and preserved['selected_V3_source_files_each_repo'] == 47
          and preserved['selected_V3_checkpoint_sha256'] ==
              '50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'
          and preserved['prior_whole_metadata_chain_preserved'] is True
          and same(preserved['current_ledger'], bindings['ledger'])
          and same(preserved['final_establishment'], bindings['establishment'])
          and same(preserved['complete_V8_mirror'], bindings['complete_mirror']),
          'Final94-checkpoint preservation joins selected V3/current ledger/trust/mirror')
    future = json.loads(a.protocol.read_text(encoding='utf-8-sig'))
    check(future['kind'] == 'clean_causal_encoder_search_goal_v8_retention_protocol_v1',
          'Separately named future protocol')
    stable = ('establishment','scope','defaults','version_policy','preservation','search_policy',
              'future_dimensions','application_promotion','publication','container','independent_confirmation',
              'baseline','application','additional_train_NLL_gate')
    for key in stable:
        check(future[key] == parent[key], 'Unchanged scientific contract '+key)
    check(len(future['criteria']) == len(parent['criteria']) == future['criterion_count'] == 22, 'Same22 conditions')
    old = {r['id']:r for r in parent['criteria']}
    check({r['id'] for r in future['criteria']} == set(old), 'Same condition IDs')
    changed = []
    reviewed = []
    for criterion in future['criteria']:
        identifier = criterion['id']
        previous = old[identifier]
        if previous['floor_source'] == 'strict positivity':
            check(criterion == previous, 'Unchanged positivity condition')
            continue
        pairs = []
        for version in (3,4,5,6,7,8):
            table = data['v%d'%version]
            method = 'CPB-v%d.market-01'%version
            check(table['model'] == method and table['status'] == 'complete'
                  and table['source_updates'] == 0 and table['full_coverage'] is True,
                  'Exact completed fresh final-reference table')
            if identifier.startswith('past_'):
                task, metric = identifier.split(':')
                rows = [r for r in table['market_rows'] if r['method'] == method and r['task'] == task]
                check(len(rows) == 1, 'Unique saved final market row')
                row = rows[0]
                check(row['checkpoint_stage'] == 'final' and row['coverage'] == 1
                      and row['valid'] == row['total'] == 32, 'Complete final market support')
                score = row[metric]
                lower = False
            elif identifier.startswith(('train:', 'validation:', 'known_validation:')):
                split = 'train' if identifier.startswith('train:') else 'validation'
                metric = 'Gaussian_NLL_F64' if identifier.startswith('known_validation:') else 'standardized_fixed_query_MAE'
                score = table['fixed_queries']['final'][split][metric]
                lower = True
            else:
                policy, view, metric = identifier.split(':')
                panel = next(r for r in table['synthetic']['panels'] if r['input_policy'] == policy and r['view'] == view)
                row = next(r for r in panel['rows'] if r['method'] == method)
                check(row['checkpoint_stage'] == 'final' and row['coverage'] == 1,
                      'Complete saved final timing row')
                score = row[metric]
                lower = False
            check(type(score) in (int,float) and math.isfinite(score), 'Finite final aggregate')
            pairs.append((score,method))
        choose = min if lower else max
        check((previous['floor'],previous['floor_source']) == choose(pairs[:5],key=lambda x:x[0]),
              'Preserved parent strongest V3/V4/V5/V6/V7 floor')
        best = choose(pairs,key=lambda x:x[0])
        check((criterion['floor'],criterion['floor_source']) == best, 'Future strongest final floor with stable earlier ties')
        expected = dict(previous, floor=best[0], floor_source=best[1])
        check(criterion == expected, 'Only floor/source can change')
        if criterion != previous:
            changed.append(identifier)
        reviewed.append(dict(id=identifier, floor=best[0], source=best[1]))
    check(len(reviewed) == 19 and len(changed) == 2 and set(changed) == EXPECTED_CHANGES,
          '19 reference floors and exactly neural volatility plus intact market-scaler neural accuracy strengthened')
    check(future['changed_threshold_count'] == 2 and future['unchanged_criteria_count'] == 20,
          'Exactly2 stronger and20 unchanged conditions')
    check(not future['additional_train_NLL_gate'] and not future['candidate_declared'] and
          not future['model_declared'] and not future['next_version_declared'], 'No extra gate or new model declaration')
    check(same(future['v8_final_establishment'], bindings['establishment'])
          and same(future['v8_complete_mirror'], bindings['complete_mirror'])
          and same(future['v8_final_preservation'], bindings['preservation'])
          and same(future['reference_ledger'], bindings['ledger']), 'Protocol preserves complete V8 trust/preservation pins')
    for identifier, path in paths.items():
        check(all(pin(path)[key] == bindings[identifier][key] for key in ('bytes', 'sha256')),
              'Input unchanged during independent review '+identifier)
    check(a.manifest.read_bytes() == manifest_bytes and Path(__file__).read_bytes() == source_bytes,
          'Review source/manifest unchanged')
    result = dict(kind='clean_causal_v8_retention_independent_review_v1',status='passed',checks=checks,
        protocol=pin(a.protocol), manifest=pin(a.manifest), source=pin(Path(__file__)),
        reviewed_reference_floors=reviewed, strengthened_conditions=changed,
        model_calls=0,head_fits=0,training_updates=0,new_quality_scores=0,historical_counts_recomputed=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='passed',checks=checks,strengthened_conditions=len(changed))))


if __name__ == '__main__':
    main()

# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Independent check of future floors using the nine immutable aggregate tables.

Complete owned standard-library source; no derivation-module imports, model
libraries, tensor reads, fits, historical rescoring or live protocol writes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('/opt/cuwacunu/research/causal-v11-retention-review')
INPUTS = {'parent', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'ledger', 'establishment', 'complete_mirror', 'preservation'}
PARENT_SHA = '221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144'


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
    check(manifest['kind'] == 'clean_causal_v11_retention_derivation_manifest_v1'
          and manifest['authorized_for_derivation'] is True, 'Explicit prospective root admission')
    check(all(pin(Path(__file__))[key] == manifest['independent_reviewer'][key]
              for key in ('bytes', 'sha256')), 'Exact independently frozen review source')
    check(all(pin(Path(__file__).with_name('derive.py'))[key] == manifest['source'][key]
              for key in ('bytes', 'sha256')), 'Exact separate owned derivation source, never imported')
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
    check(set(data) == INPUTS, 'Exact fourteen input roles')
    check(next(row['sha256'] for row in manifest['inputs'] if row['id'] == 'parent') == PARENT_SHA,
          'Exact prior V9-retention protocol')
    parent = data['parent']
    trust, mirror, preserved = (data[name] for name in ('establishment', 'complete_mirror', 'preservation'))
    same = lambda left, right: all(left[key] == right[key] for key in ('bytes', 'sha256'))
    check(trust['kind'] == 'causal_fresh_v11_postmeasurement_establishment_v1'
          and trust['candidate'] == 'CPB-v11.market-01' and trust['status'] == 'passed'
          and all(trust[key] is True for key in ('measured_candidate_established', 'trust_admission_passed',
                  'full_coverage', 'independent_training_audit_passed', 'independent_quality_audit_passed'))
          and trust['overall_winner'] is False and trust['winner_selection_made'] is False,
          'Measured trustworthy V11 candidate without winner selection')
    check(same(trust['evidence']['table_rows'], bindings['v11'])
          and same(trust['evidence']['complete_mirror'], bindings['complete_mirror']),
          'Separate trust binds exact V11 table and Complete mirror')
    check(mirror['kind'] == 'causal_fresh_v11_container_to_local_evidence_mirror_v1'
          and mirror['status'] == 'passed' and mirror['scope'] == 'Complete' and mirror['attempt'] == 'attempt-001'
          and mirror['source_files_verified'] == 132 and mirror['all_mirrored_pt_ignored'] is True
          and mirror['source_stage']['status'] == 'passed'
          and mirror['source_stage']['card_sha256'] ==
              'aa0c8cf06e10b4da7ce1ede0d2cc914bc05b244043ad4f0d3039305499a10aaa'
          and mirror['source_stage']['manifest_sha256'] ==
              'b9a701bbfa3e98e4719853dfb1b2fad8d7610b3a86b475e43398ab43fea15778'
          and mirror['source_stage']['archive_sha256'] ==
              '7c76afbbce09eb84e17ce865d942ab340988d4127d5a06839ccb1634babf6a26'
          and any(same(item, bindings['v11']) for item in mirror['files']),
          'Complete mirror of admitted132-source V11 table')
    check(preserved['kind'] == 'causal_fresh_v11_final_preservation_v1' and preserved['status'] == 'passed'
          and preserved['baseline'] == 'CPB-v3.market-01' and preserved['candidate'] == 'CPB-v11.market-01'
          and preserved['baseline_changed'] is False and preserved['application_changed'] is False
          and preserved['critical_checkpoint_total'] == len(preserved['critical_checkpoint_pins']) == 109
          and len({(item['repository'], item['path']) for item in preserved['critical_checkpoint_pins']}) == 109
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
          and same(preserved['complete_V11_mirror'], bindings['complete_mirror']),
          'Final109-checkpoint preservation joins selected V3/current ledger/trust/mirror')
    check(trust['normal_prospective_source_admission']['source_files'] == 132
          and trust['normal_prospective_source_admission']['normal_quality_all_stages_passed'] is True
          and trust['normal_prospective_source_admission']['scientific_model_head_or_training_replay'] is False
          and 'report_only_recovery' not in trust and 'report_only_recovery' not in mirror
          and preserved['current_reporting_recovery_required'] is False
          and preserved['science_stages_replayed'] == 0
          and preserved['historical_V9_reporting_recovery_preserved'] is True,
          'Normal V11 quality is separate from opaque historical V9 recovery')
    check(trust['initial_partial_native20_40_pairings_passed'] is True
          and trust['initial_pairing_count'] == 13
          and trust['initial_pairing_native_coordinate_slice'] == [20, 40]
          and trust['whole_native40_initial_equality_claimed'] is False
          and preserved['initial_pairing_native_coordinate_slice'] == [20, 40]
          and preserved['whole_native40_initial_equality_claimed'] is False,
          'Current partial thirteen-reference pairing scope remains exact')
    historical = trust['historical_V9_reporting_recovery_preserved']
    check(historical['card']['sha256'] ==
              '9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d'
          and historical['receipt']['sha256'] ==
              '5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72'
          and historical['failure_mirror']['sha256'] ==
              'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532',
          'Historical V9 card/receipt/failure references remain untouched')
    future = json.loads(a.protocol.read_text(encoding='utf-8-sig'))
    check(future['kind'] == 'clean_causal_encoder_search_goal_v11_retention_protocol_v1',
          'Separately named future protocol')
    stable = ('establishment','scope','defaults','version_policy','preservation','search_policy',
              'future_dimensions','application_promotion','publication','container','independent_confirmation',
              'baseline','application','additional_train_NLL_gate')
    for key in stable:
        check(future[key] == parent[key], 'Unchanged scientific contract '+key)
    check(len(future['criteria']) == len(parent['criteria']) == future['criterion_count'] == 22, 'Same22 conditions')
    old = {r['id']:r for r in parent['criteria']}
    check({r['id'] for r in future['criteria']} == set(old), 'Same condition IDs')
    changed, expected_changes = [], []
    reviewed = []
    for criterion in future['criteria']:
        identifier = criterion['id']
        previous = old[identifier]
        if previous['floor_source'] == 'strict positivity':
            check(criterion == previous, 'Unchanged positivity condition')
            continue
        pairs = []
        for version in (3,4,5,6,7,8,9,10,11):
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
        check((previous['floor'],previous['floor_source']) == choose(pairs[:7],key=lambda x:x[0]),
              'Preserved parent strongest V3/V4/V5/V6/V7/V8/V9 floor')
        best = choose(pairs,key=lambda x:x[0])
        check((criterion['floor'],criterion['floor_source']) == best, 'Future strongest final floor with stable earlier ties')
        expected = dict(previous, floor=best[0], floor_source=best[1])
        check(criterion == expected, 'Only floor/source can change')
        if criterion != previous:
            changed.append(identifier)
            check(best[1] in ('CPB-v10.market-01', 'CPB-v11.market-01'), 'Only new V10/V11 strict improvements may raise a future floor')
            check(best[0] < previous['floor'] if lower else best[0] > previous['floor'],
                  'Future floor strengthens strictly; ties retain earlier source')
            expected_changes.append(dict(id=identifier, previous_threshold=previous['floor'],
                previous_source=previous['floor_source'], retained_threshold=best[0],
                retained_source=best[1], rule=previous['rule']))
        reviewed.append(dict(id=identifier, floor=best[0], source=best[1]))
    check(len(reviewed) == 19 and len(changed) <= 19 and len(set(changed)) == len(changed),
          '19 reference floors; only actual saved strict improvements, no expected outcome')
    check(future['changed_threshold_count'] == len(changed) and future['unchanged_criteria_count'] == 22-len(changed),
          'Dynamic stronger/unchanged counts match saved comparisons')
    check(future['changed_thresholds'] == expected_changes, 'Every old/new floor object retained exactly')
    check([r['id'] for r in future['criteria']] == [r['id'] for r in parent['criteria']],
          'Original22 ordered criterion identities retained')
    check(future['retention_derivation_history'][:-1] == parent['retention_derivation_history']
          and future['retention_derivation_history'][-1]['prior_criteria'] == parent['criteria']
          and future['retention_derivation_history'][-1]['prior_reference_tables'] == parent['reference_tables'],
          'Whole earlier history prefix and every original floor/reference object retained')
    check(same(future['parent_protocol'], bindings['parent']) and
          [row['instance'] for row in future['reference_tables']] == ['CPB-v%d.market-01'%v for v in (3,4,5,6,7,8,9,10,11)] and
          all(same(row['table'], bindings['v%d'%v]) for row,v in zip(future['reference_tables'],(3,4,5,6,7,8,9,10,11))),
          'Exact parent and all nine chronological immutable table references')
    check(not future['additional_train_NLL_gate'] and not future['candidate_declared'] and
          not future['model_declared'] and not future['next_version_declared'], 'No extra gate or new model declaration')
    check(same(future['v11_final_establishment'], bindings['establishment'])
          and same(future['v11_complete_mirror'], bindings['complete_mirror'])
          and same(future['v11_final_preservation'], bindings['preservation'])
          and same(future['reference_ledger'], bindings['ledger']), 'Protocol preserves complete V11 trust/preservation pins')
    check(future['historical_V9_reporting_recovery_preserved'] == historical,
          'Whole historical V9 recovery record preserved opaquely')
    for identifier, path in paths.items():
        check(all(pin(path)[key] == bindings[identifier][key] for key in ('bytes', 'sha256')),
              'Input unchanged during independent review '+identifier)
    check(a.manifest.read_bytes() == manifest_bytes and Path(__file__).read_bytes() == source_bytes,
          'Review source/manifest unchanged')
    result = dict(kind='clean_causal_v11_retention_independent_review_v1',status='passed',checks=checks,
        protocol=pin(a.protocol), manifest=pin(a.manifest), source=pin(Path(__file__)),
        reviewed_reference_floors=reviewed, strengthened_conditions=changed,
        model_calls=0,head_fits=0,training_updates=0,new_quality_scores=0,historical_counts_recomputed=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='passed',checks=checks,strengthened_conditions=len(changed))))


if __name__ == '__main__':
    main()

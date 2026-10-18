# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Independent saved-metadata review before applying the three V6 proposals."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--card', type=Path, required=True)
    p.add_argument('--proposals', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert Path('/.dockerenv').is_file() and not a.output.exists()
    checks = 0

    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            raise RuntimeError(message)

    def read(path):
        return json.loads(path.read_text())

    def bound(path, pin):
        data = path.read_bytes()
        check(len(data) == pin['bytes'] and hashlib.sha256(data).hexdigest() == pin['sha256'], 'Exact metadata binding '+str(path))

    card = read(a.card)
    inputs = {}
    for item in card['inputs']:
        path = a.card.parent/item['staged_path']
        bound(path, item)
        if item['format'] == 'json':
            inputs[item['id']] = read(path)
    receipt = read(a.proposals/'finalization-proposal.json')
    expected = {'doc/clean_causal_encoder_research_ledger.json',
                'code/encoders/causal_patch_bottleneck_mae/versions/current.json',
                'code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v6.market-01.json'}
    check({r['path'] for r in receipt['outputs']} == expected, 'Exactly three declared metadata outputs')
    for item in receipt['outputs']:
        bound(a.proposals/item['path'], item)
    ledger = read(a.proposals/'doc/clean_causal_encoder_research_ledger.json')
    current = read(a.proposals/'code/encoders/causal_patch_bottleneck_mae/versions/current.json')
    instance = read(a.proposals/'code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v6.market-01.json')
    old = inputs['ledger']
    check(ledger['baseline'] == old['baseline'] == current['active_application'] == 'CPB-v3.market-01', 'Clean baseline/application retained')
    check(ledger['baseline_checkpoint'] == old['baseline_checkpoint'], 'Complete baseline checkpoint record')
    check(not ledger['overall_winner'], 'No overall winner')
    for row in old['candidates']:
        check(row in ledger['candidates'], 'Complete old candidate retained')
    for key in ('stage_scorecards', 'reconstruction_stage_scorecards', 'displaced_leaders', 'candidate_status_history', 'direction_history', 'stage_leader_update_history'):
        check(ledger[key][:len(old[key])] == old[key], 'Complete historical prefix retained: '+key)
    for row in old['stage_leaders']:
        check(any(entry == row or entry.get('record') == row for entry in ledger['leader_history']), 'Complete prior leader retained in history')
    for row in old['leader_history']:
        check(row in ledger['leader_history'], 'Complete older leader history retained')
    check(inputs['catalog']['fresh_research_candidate'] in current['fresh_research_candidate_history'], 'Whole V5 current entry retained')
    check(len(ledger['stage_scorecards']) == len(ledger['reconstruction_stage_scorecards']) == 8, 'All four versions initial/final stagecards')

    def scope(row):
        return tuple(row.get(k) for k in ('dataset_id','task','input_policy','view','metric'))

    check({scope(r) for r in ledger['stage_leaders']} == {scope(r) for r in old['stage_leaders']}, 'Same16 quality scopes')
    replacements = 0
    for leader in ledger['stage_leaders']:
        scores = []
        for version in (3,4,5,6):
            table = inputs['v%d_table'%version]
            method = 'CPB-v%d.market-01'%version
            if leader['dataset_id'] == 'MKT-HIST-2':
                row = next(r for r in table['market_rows'] if r['method'] == method and r['task'] == leader['task'])
            else:
                panel = next(r for r in table['synthetic']['panels'] if r['input_policy'] == leader['input_policy'] and r['view'] == leader['view'])
                row = next(r for r in panel['rows'] if r['method'] == method)
            scores.append((row[leader['metric']], method))
        best = max(scores, key=lambda r:r[0])
        check((leader['score'],leader['method']) == best, 'Strongest final leader with earlier ties')
        check(len(leader['stage_scores']) == 8, 'All initial/final values on every leader')
        replacements += best[1] == 'CPB-v6.market-01'
    check(replacements == 8, 'Eight new scoped quality leaders')
    check(len(ledger['reconstruction_stage_leaders']) == 4, 'Four reconstruction scopes')
    for leader in ledger['reconstruction_stage_leaders']:
        scores = []
        for version in (3,4,5,6):
            query = inputs['v%d_table'%version]['fixed_queries']['final'][leader['split']]
            check(leader['metric'] in ('standardized_fixed_query_MAE','Gaussian_NLL_F64'), 'Exact reconstruction metric names')
            key = leader['metric']
            scores.append((query[key], 'CPB-v%d.market-01'%version))
        check((leader['score'],leader['method']) == min(scores,key=lambda r:r[0]), 'Minimum final reconstruction metric')
    check(current['fresh_research_candidate']['instance'] == instance['instance'] == 'CPB-v6.market-01', 'Measured instance/catalog identity')
    check(instance['source_updates'] == 0 and instance['new_causal_updates'] == 2048, 'Fresh single trajectory')
    check(instance['independent_training_audit_core_plus_modules_checks'] == 270721 and instance['independent_quality_audit_checks'] == 12218, 'Independent audit counts')
    for value in (current['fresh_research_candidate'], instance['quality']):
        check(value['numeric_passed'] == 12 and value['numeric_total'] == 22 and not value['advancement_passed'], 'Failed numeric gate preserved')
        check(value['trust_admission_passed'] and value['full_coverage'], 'Full independent trust/coverage')
    check(len(instance['main_checkpoints']) == 5, 'All five main checkpoints retained')
    check(inputs['comparison']['trust_admission_passed'] is False and inputs['establishment']['trust_admission_passed'] is True, 'Original pending producer and separate final trust')
    result = {'kind':'causal_fresh_v6_independent_metadata_proposal_review_v1','status':'passed',
              'checks':checks,'metadata_outputs':receipt['outputs'],'strict_new_quality_leaders':8,
              'baseline_changed':False,'descriptor_changed':False,'producer_changed':False,
              'model_calls':0,'head_fits':0,'training_updates':0,'live_files_written':0}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'passed','checks':checks,'quality_leaders':8}))


if __name__ == '__main__':
    main()

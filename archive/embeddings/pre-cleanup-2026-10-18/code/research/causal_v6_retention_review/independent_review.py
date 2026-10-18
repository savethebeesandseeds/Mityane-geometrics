# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Independent check of future floors using the four immutable aggregate tables."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert Path('/.dockerenv').is_file() and not a.output.exists()
    checks = 0

    def check(value, reason):
        nonlocal checks
        checks += 1
        if not value:
            raise RuntimeError(reason)

    def pin(path):
        data = path.read_bytes()
        return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    manifest = json.loads(a.manifest.read_text())
    data = {}
    for row in manifest['inputs']:
        path = a.manifest.parent/row['staged_path']
        actual = pin(path)
        check(all(actual[k] == row[k] for k in ('bytes','sha256')), 'Immutable input '+row['id'])
        data[row['id']] = json.loads(path.read_text())
    parent = data['parent']
    future = json.loads(a.protocol.read_text())
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
        for version in (3,4,5,6):
            table = data['v%d'%version]
            method = 'CPB-v%d.market-01'%version
            if identifier.startswith('past_'):
                task, metric = identifier.split(':')
                row = next(r for r in table['market_rows'] if r['method'] == method and r['task'] == task)
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
                score = row[metric]
                lower = False
            check(type(score) in (int,float) and math.isfinite(score), 'Finite final aggregate')
            pairs.append((score,method))
        choose = min if lower else max
        check((previous['floor'],previous['floor_source']) == choose(pairs[:3],key=lambda x:x[0]), 'Preserved parent strongest V3/V4/V5 floor')
        best = choose(pairs,key=lambda x:x[0])
        check((criterion['floor'],criterion['floor_source']) == best, 'Future strongest final floor with stable earlier ties')
        expected = dict(previous, floor=best[0], floor_source=best[1])
        check(criterion == expected, 'Only floor/source can change')
        if criterion != previous:
            changed.append(identifier)
        reviewed.append(dict(id=identifier, floor=best[0], source=best[1]))
    check(len(reviewed) == 19 and len(changed) == 9, '19 reference floors and9 strengthened conditions')
    check(not future['additional_train_NLL_gate'] and not future['candidate_declared'] and
          not future['model_declared'] and not future['next_version_declared'], 'No extra gate or new model declaration')
    result = dict(kind='clean_causal_v6_retention_independent_review_v1',status='passed',checks=checks,
        protocol=pin(a.protocol), manifest=pin(a.manifest), source=pin(Path(__file__)),
        reviewed_reference_floors=reviewed, strengthened_conditions=changed,
        model_calls=0,head_fits=0,training_updates=0,new_quality_scores=0,historical_counts_recomputed=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='passed',checks=checks,strengthened_conditions=len(changed))))


if __name__ == '__main__':
    main()

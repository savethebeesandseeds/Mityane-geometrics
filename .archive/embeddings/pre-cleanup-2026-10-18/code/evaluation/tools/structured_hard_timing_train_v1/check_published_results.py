#!/usr/bin/env python3
"""Independent output-only count/sign/aggregate check; no tensor archive reads."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path('/embedding/output/runs/rpb-structured-hard-timing-train')
PINS = {
    'head-fit-xrZMAS-v1/head-fit-diagnostic.json': '03624d5c349255d90a0851ead61914095b9ce179fb98a787fc647345b53ac59f',
    'query-timing-xrZMAS-v1/query-diagnostic.json': 'd9b545e18febe5fed9ab1a800db29a6f3b914f339af3790261cdc9098b7e228c',
}
checks = 0

def check(ok):
    global checks
    checks += 1
    assert ok, f'output check {checks}'

def mean(values):
    return float(sum(Fraction(v) for v in values) / len(values))

def main():
    check(Path('/.dockerenv').is_file())
    bodies = [(ROOT / name).read_bytes() for name in PINS]
    check([hashlib.sha256(b).hexdigest() for b in bodies] == list(PINS.values()))
    head, query = map(json.loads, bodies)
    check(head['status'] == query['status'] == 'passed')
    check(head['archive_decodes'] == 110 and query['archive_decodes'] == 20)
    for cohort in head['cohorts']:
        for method in cohort['methods']:
            for rep in method['repetitions']:
                for fitted in rep['heads'].values():
                    d = fitted['TRAIN']; r = d['rows']; n = len(r['labels'])
                    valid = [i for i, v in enumerate(r['valid']) if v]
                    correct = sum(r['classes'][i] == r['labels'][i] for i in valid)
                    check(n == d['total_rows'] == 256)
                    check(len(valid) == d['valid_rows'] and correct == d['correct_rows'])
                    check(d['accuracy'] == correct / len(valid) and d['coverage'] == len(valid) / n)
                    for i in valid:
                        check(r['classes'][i] == int(r['saved_logit_gap'][i] > 0))
                        check(r['true_class_margin'][i] == (2 * r['labels'][i] - 1) * r['saved_logit_gap'][i])
                    source_pairs = {}
                    for i, source in enumerate(r['sources']):
                        source_pairs.setdefault(source, {})[r['labels'][i]] = i
                    check(len(source_pairs) == 128)
                    check([p['source'] for p in d['pairs']] == sorted(source_pairs))
                    for pair in d['pairs']:
                        i, j = (source_pairs[pair['source']][label] for label in (0, 1))
                        supported = r['valid'][i] and r['valid'][j]
                        check(pair['supported'] == supported)
                        if supported:
                            a, b = r['saved_logit_gap'][i], r['saved_logit_gap'][j]
                            check(pair['label1_minus_label0_gap'] == b - a)
                            check(pair['strict_correct_order'] == (b > a))
                            check(pair['strict_opposite_signs'] == ((a < 0 < b) or (b < 0 < a)))
                            check(pair['both_correct'] == (r['classes'][i] == 0 and r['classes'][j] == 1))
                    check(d['source_pair_metrics']['both_correct']['count'] == sum(p['both_correct'] is True for p in d['pairs']))
    for summary in head['summary']:
        rows = []
        for cohort in head['cohorts']:
            reps = next(m for m in cohort['methods'] if m['method'] == summary['method'])['repetitions']
            rows.append(mean([r['heads'][summary['head']]['TRAIN']['accuracy'] for r in reps]))
        check(summary['TRAIN_accuracy_mean'] == mean(rows))
    for cohort in query['per_master']:
        labels = cohort['labels_scoring_only']
        for d in cohort['methods'].values():
            rows = d['rows']; selected = []
            for i, r in enumerate(rows):
                check(r['supported_centres'] == sum(r['bank_centres']))
                check(r['valid'] == (r['supported_centres'] >= 4 and r['margin'] is not None and r['margin'] != 0))
                check(r['prediction'] == (int(r['margin'] > 0) if r['valid'] else 0))
                if r['valid']:
                    selected.append(i)
            correct = sum(rows[i]['prediction'] == labels[i] for i in selected)
            check(d['score']['total'] == 256 and d['score']['valid'] == len(selected) and d['score']['correct'] == correct)
            check(d['score']['accuracy'] == correct / len(selected))
            check(d['score']['coverage'] == len(selected) / 256)
    for method, summary in query['aggregates'].items():
        for metric in ('accuracy', 'coverage', 'full_population_correctness'):
            values = [c['methods'][method]['score'][metric] for c in query['per_master']]
            check(summary[metric]['per_master'] == values)
            check(abs(summary[metric]['mean'] - mean(values)) < 1e-15)
    check([hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PINS] == list(PINS.values()))
    result = {'status': 'passed', 'checks': checks, 'input_sha256': PINS,
              'tensor_archive_reads': 0, 'model_forward': False, 'fits': 0}
    with (ROOT / 'published-output-check-v1.json').open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()

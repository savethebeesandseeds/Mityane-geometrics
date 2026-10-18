#!/usr/bin/env python3
"""Publish the completed saved-TRAIN diagnosis from its two result JSON files."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/embedding')
RUN = ROOT / 'output/runs/rpb-structured-hard-timing-train'
HEAD = RUN / 'head-fit-xrZMAS-v1/head-fit-diagnostic.json'
QUERY = RUN / 'query-timing-xrZMAS-v1/query-diagnostic.json'
CHECK = RUN / 'published-output-check-v1.json'
CARD_SHA = '3b7e8ea947d91f82ee3fb001dfdc06314945ea8888a763d37459ccc7410673b4'
MASTERS = [75272, 76373, 77474, 78575, 79676]
LABELS = {'raw': 'Raw data — no encoder', 'native_late': 'RPB-v7.alt-05', 'native_early': 'RPB-v10.alt-05'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in ('rows', 'pairs', 'labels_scoring_only', 'source_ids')}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return copy.deepcopy(value)

def pct(value):
    return 'undefined' if value is None else f'{100 * value:.2f}'

def main():
    assert Path('/.dockerenv').is_file(), 'managed container only'
    head, query = (json.loads(p.read_text()) for p in (HEAD, QUERY))
    assert head['status'] == query['status'] == 'passed'
    assert head['card_sha256'] == query['card_sha256'] == CARD_SHA
    assert [c['master'] for c in head['cohorts']] == [c['master'] for c in query['per_master']] == MASTERS
    assert head['archive_decodes'] == 110 and query['archive_decodes'] == 20
    output_check = json.loads(CHECK.read_text())
    assert output_check['status'] == 'passed' and output_check['checks'] == 115814
    assert all(s['TRAIN_coverage_mean'] == 1 for s in head['summary'])
    evidence = {'protocol': 'structured-hard-timing-train-diagnostic-v1', 'status': 'passed',
        'dataset': 'TEMPO-3', 'designed_complexity_level': 4, 'complexity_scale_max': 5,
        'card_sha256': CARD_SHA, 'head': compact(head), 'query': compact(query),
        'result_files': [{'path': str(p), 'sha256': digest(p)} for p in (HEAD, QUERY, CHECK)],
        'independent_output_check': output_check,
        'notes': {'unique_TRAIN_input_roles': 125, 'archive_decodes': 130,
                  'shared_controlled_TRAIN_roles': 5, 'encoder_updates': 0, 'head_refits': 0,
                  'new_validation_tensor_reads': 0, 'TEST_or_stress_reads': 0, 'promotion': False}}
    text = ['# TEMPO-3 saved training diagnosis', '',
        'This is a new diagnosis of saved TRAIN tensors. There was no model training, model inference, head fitting or PCA fitting. Validation percentages below reuse the audited TEMPO-3 comparison metadata. All five cohorts and all three fixed head repetitions are retained.', '',
        'RPB-v7.alt-05 uses late channel mixing. RPB-v10.alt-05 mixes channels before temporal encoding. Both were trained from scratch on TEMPO-3 for 512 CUDA updates, batch 8; their native embedding has 32 values. These are separate from the successful TEMPO-1 trained copies.', '']
    for view, title in [('TRAIN', 'TRAIN, 256 rows / 128 source pairs per cohort'),
                        ('validation_intact', 'VALIDATION, original observations, 128 rows / 64 pairs per cohort'),
                        ('validation_deleted', 'VALIDATION, additional 30% coordinate deletion, 128 rows / 64 pairs per cohort')]:
        text += ['Dataset: **TEMPO-3**, variable period and short lead/lag, independent gains and offsets, unrelated channel, noise and missing observations; **designed complexity 4/5**. View: ' + title + '. Fixed heads: Ridge penalty 1; tanh16, Adam .01, 100 updates; repetitions 2701/2802/2903.', '',
                 '| Method | Size | Linear head % | Neural head % | Coverage % |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for method in LABELS:
            linear = next(s for s in head['summary'] if s['method'] == method and s['head'] == 'ridge')
            neural = next(s for s in head['summary'] if s['method'] == method and s['head'] == 'tiny_secondary')
            text.append(f'| {LABELS[method]} | {576 if method == "raw" else 32} | {pct(linear[view + "_accuracy_mean"])} | {pct(neural[view + "_accuracy_mean"])} | 100.00 |')
        text.append('')
    text += ['The raw neural head fits almost every training example but performs near chance on unseen source pairs. It can memorize this small training set. The trained native embeddings fit TRAIN less well and also transfer weakly. These scores do not establish that the encoder has destroyed timing information.', '',
        'Dataset: **TEMPO-3**, same structured timing recipe, **designed complexity 4/5**. The following is a separate TRAIN timing rule applied within each saved eight-tick query patch. It uses four different masked contexts; it is not a classifier head attached to one full-context embedding.', '',
        '| Saved surface | Timing accuracy % | Coverage % | Correct / all rows % |',
        '| --- | ---: | ---: | ---: |']
    names = {'saved_target': 'Observed query targets', 'late_prediction': 'RPB-v7.alt-05 reconstruction', 'early_prediction': 'RPB-v10.alt-05 reconstruction'}
    for method, name in names.items():
        s = query['aggregates'][method]
        text.append(f'| {name} | {pct(s["accuracy"]["mean"])} | {pct(s["coverage"]["mean"])} | {pct(s["full_population_correctness"]["mean"])} |')
    text += ['', 'The target row checks that the same supported short patches contain the answer. Weak timing in the reconstructed waveforms locates a problem somewhere in the learned reconstruction path; it does not separate encoder and decoder causes or prove what information the full-context native32 contains.', '',
        'The next bounded candidate will add visible adjacent-time differences to the raw patch input. This supplies a local shape prior that cancels constant offsets in that added branch. Raw values remain available; the original reconstruction target, native32 bottleneck, context masks and fixed heads remain unchanged. The branch adds capacity and changes the optimization parameterization, so an improvement would not prove information recovery or general affine invariance.', '',
        'Evidence: 125 unique TRAIN payload roles, decoded 130 times across the two tools because five controlled TRAIN roles are shared. Both tools checked frozen input/source bytes before and after arithmetic. No VAL archive bodies, TEST or stress data were read. No promotion.', '',
        f'Head result: `{HEAD}`; SHA `{digest(HEAD)}`.',
        f'Query result: `{QUERY}`; SHA `{digest(QUERY)}`.', '']
    target = ROOT / 'doc/results/structured_hard_timing_train_diagnostic_v1.json'
    report = ROOT / 'code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_TRAIN_DIAGNOSTIC.md'
    for path, body in [(target, json.dumps(evidence, indent=2, allow_nan=False) + '\n'), (report, '\n'.join(text))]:
        with path.open('x', encoding='utf-8', newline='\n') as out:
            out.write(body)
    print(json.dumps({'status': 'published', 'report_sha256': digest(report), 'summary_sha256': digest(target)}))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Verify selected-fit CPU witnesses and score arithmetic; no search/model/refit."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/embedding')
MODULE_ROOT = ROOT / 'code/evaluation/protocols/visible_difference_v1/reader-v2'
PINS = {
    'archive_codec.py': '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d',
    'saved_cpu_math.py': '1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9'}
CHECKS = 0
DECODES = 0


def check(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, filename):
    check(sha(MODULE_ROOT / filename) == PINS[filename], 'pinned pure saved-arithmetic module')
    spec = importlib.util.spec_from_file_location(name, MODULE_ROOT / filename)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


R = module('two_component_cpu_codec', 'archive_codec.py')
M = module('two_component_cpu_arithmetic', 'saved_cpu_math.py')


def load(path):
    global DECODES
    DECODES += 1
    return R.load(path)


def tensor(a, key, dtype, shape):
    t = a[key]
    check(t['dtype'] == dtype and t['shape'] == shape, 'saved tensor geometry ' + key)
    return t['values']


def data(path, rows, master, task_bit):
    a = load(path)
    check(set(a) == {'observations', 'feature_mask', 'labels_scoring_only', 'source_ids'}, 'legal-only schema')
    x = tensor(a, 'observations', 'DoubleStorage', [rows, 3, 32, 3])
    mask = tensor(a, 'feature_mask', 'BoolStorage', [rows, 3, 32, 3])
    y = tensor(a, 'labels_scoring_only', 'LongStorage', [rows])
    ids = tensor(a, 'source_ids', 'LongStorage', [rows])
    check(all(math.isfinite(v) and (m or v == 0) for v, m in zip(x, mask)), 'finite legal values and zero hidden storage')
    check(set(mask) <= {0, 1} and set(y) == {0, 1}, 'binary masks and balanced labels')
    previous = -1
    for row in range(0, rows, 2):
        source = ids[row]
        check(ids[row + 1] == source and source > previous and source >> 31 == master and
              source & 1 == task_bit and ((source & ((1 << 31) - 1)) >> 1) < 192, 'injective packed pair source')
        check({y[row], y[row + 1]} == {0, 1}, 'opposite randomized pair labels')
        check(mask[row*288:(row+1)*288] == mask[(row+1)*288:(row+2)*288], 'pair-shared mask')
        check(x[row*288+192:(row+1)*288] == x[(row+1)*288+192:(row+2)*288], 'label-independent channel2')
        if task_bit == 0:
            check(x[row*288:row*288+96] == x[(row+1)*288:(row+1)*288+96], 'timing pair channel0 unchanged')
        previous = source
    return x, mask, y, ids


def selected_fit(x, mask, row, fs, ff, coefficients, eligible, task_bit, supported):
    residuals, ratios = [], []
    direction = [[[], []], [[], []]]
    for channel in range(2):
        union = set()
        features = 0
        for feature in range(3):
            fit_index = channel*3 + feature
            c = coefficients[(row*6+fit_index)*5:(row*6+fit_index+1)*5]
            ticks = [t for t in range(32) if mask[row*288+channel*96+t*3+feature]]
            values = [x[row*288+channel*96+t*3+feature] for t in ticks]
            mean = math.fsum(values)/len(values) if values else 0
            energy = math.fsum((v-mean)**2 for v in values)
            active = len(ticks) >= 10 and energy > 1e-12
            check(bool(eligible[row*6+fit_index]) == active, 'eligible selected-fit feature')
            if not active:
                check(all(v == 0 for v in c), 'unused coefficient witness remains zero')
                continue
            union.update(ticks)
            features += 1
            errors, basis = [], []
            for t, value in zip(ticks, values):
                b = [1, math.sin(math.tau*fs*t), math.cos(math.tau*fs*t),
                     math.sin(math.tau*ff*t), math.cos(math.tau*ff*t)]
                basis.append(b)
                errors.append(value - math.fsum(v*w for v, w in zip(b, c)))
            for j in range(5):
                stationarity = math.fsum(error*b[j] for error, b in zip(errors, basis))
                check(abs(stationarity) <= 2e-8*max(1, math.sqrt(energy)), 'selected LS normal-equation stationarity')
            residuals.append(math.fsum(e*e for e in errors)/energy)
            amplitudes = [math.hypot(c[1], c[2]), math.hypot(c[3], c[4])]
            total = sum(a*a for a in amplitudes)
            check(total > 1e-12 and all(a*a/total >= .05-1e-12 for a in amplitudes), 'two-component fit energy support')
            ratios.append(math.log(amplitudes[0]/amplitudes[1]))
            for component in range(2):
                j = 1 + component*2
                direction[channel][component].append((c[j]/amplitudes[component], c[j+1]/amplitudes[component]))
        check(features >= 2 and len(union) >= 16 and max(union)-min(union) >= 24, 'channel time/feature support')
    slow_directions = []
    for channel in range(2):
        for component in range(2):
            pairs = direction[channel][component]
            cosine, sine = math.fsum(p[0] for p in pairs), math.fsum(p[1] for p in pairs)
            length = math.hypot(cosine, sine)
            check(length/len(pairs) >= .95-1e-12, 'within-channel component phase coherence')
            if component == 0:
                slow_directions.append((cosine/length, sine/length))
    mean_ratio = math.fsum(ratios)/len(ratios)
    if task_bit:
        if supported:
            check(all(abs(r-mean_ratio) <= .1+1e-12 for r in ratios), 'supported component-balance ratio consistency')
        margin = mean_ratio
    else:
        a, b = slow_directions
        margin = math.atan2(a[0]*b[1]-a[1]*b[0], a[0]*b[0]+a[1]*b[1])
    return math.fsum(residuals)/len(residuals), margin


def decisions(path, values, cohort, view):
    x, mask, labels, _ = values
    rows = len(labels)
    a = load(path)
    expected = {'predictions', 'supported', 'slow_frequency', 'fast_frequency',
                'normalized_residual', 'margin', 'coefficients', 'eligible', 'frequency_pairs_evaluated'}
    check(set(a) == expected, 'exact selected-fit decision schema')
    pred = tensor(a, 'predictions', 'LongStorage', [rows])
    valid = tensor(a, 'supported', 'BoolStorage', [rows])
    fs = tensor(a, 'slow_frequency', 'DoubleStorage', [rows])
    ff = tensor(a, 'fast_frequency', 'DoubleStorage', [rows])
    residual = tensor(a, 'normalized_residual', 'DoubleStorage', [rows])
    margin = tensor(a, 'margin', 'DoubleStorage', [rows])
    coefficients = tensor(a, 'coefficients', 'DoubleStorage', [rows, 6, 5])
    eligible = tensor(a, 'eligible', 'BoolStorage', [rows, 6])
    searches = tensor(a, 'frequency_pairs_evaluated', 'LongStorage', [rows])
    check(set(pred) <= {0, 1} and set(valid) <= {0, 1} and set(eligible) <= {0, 1}, 'binary saved decision fields')
    check(all(math.isfinite(v) for field in (fs, ff, residual, margin, coefficients) for v in field), 'finite fit diagnostics')
    task_bit = int(cohort['task'] == 'component_balance')
    reasons = view['row_reasons']
    check(len(reasons) == rows and set(searches) <= {0, 425, 515}, 'bounded search and all row reasons')
    for row in range(rows):
        check(bool(valid[row]) == (reasons[row] == 'supported'), 'supported decision reason')
        if fs[row] == 0:
            check(not valid[row] and ff[row] == 0 and searches[row] in (0, 425), 'early abstention is retained')
            continue
        check(1/20 <= fs[row] <= 1/12 and 1/7 <= ff[row] <= 1/4.5 and searches[row] == 515, 'selected frequencies within declared bands')
        # These are selected-fit arithmetic witnesses, not another frequency search.
        computed_residual, computed_margin = selected_fit(x, mask, row, fs[row], ff[row], coefficients, eligible, task_bit, bool(valid[row]))
        check(abs(computed_residual-residual[row]) <= 2e-9, 'saved normalized residual arithmetic')
        if valid[row]:
            check(abs(computed_margin-margin[row]) <= 2e-9, 'saved phase/ratio margin arithmetic')
            lo, hi = (.35, 1.0) if task_bit else (.08, .8)
            check(residual[row] <= .0025 and lo <= abs(margin[row]) <= hi and pred[row] == int(margin[row] > 0), 'supported fixed decision rule')
    total, supported = rows, sum(valid)
    correct = sum(bool(v) and p == y for p, y, v in zip(pred, labels, valid))
    computed = {'total': total, 'valid': supported, 'correct': correct,
                'abstained': total-supported, 'accuracy': correct/supported if supported else None,
                'coverage': supported/total, 'full_population_correctness': correct/total}
    for key, value in computed.items():
        check(view[key] == value if value is None else abs(view[key]-value) <= 1e-14, 'saved score ' + key)
    deleted = view['view'] == 'validation-deleted'
    passed = supported > 0 and computed['accuracy'] >= (.98 if deleted else .99) and computed['coverage'] >= (.95 if deleted else .99)
    check(view['gate_passed'] == passed, 'prospective view gate')
    check(view['reason_counts'] == {reason: reasons.count(reason) for reason in set(reasons)}, 'all reasons counted')
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    capsule = args.admission
    freeze = json.loads((capsule/'freeze/freeze.json').read_text())
    run = capsule/'information'
    report = json.loads((run/'report.json').read_text())
    check(json.loads((run/'complete.json').read_text())['report_sha256'] == sha(run/'report.json'), 'completed original report')
    check(report['source_fingerprint'] == freeze['source_fingerprint'] and report['card_sha256'] == freeze['card_sha256'] and
          report['sdk_proof_sha256'] == freeze['sdk_proof_sha256'], 'source/card/SDK binding')
    for line in (capsule/'freeze/sources.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        check(sha(ROOT/relative) == digest == sha(capsule/'freeze/SOURCE'/relative), 'current and captured source preserved')
    check(len(report['files']) == 24, 'exact twelve data and twelve decision archives')
    for entry in report['files']:
        path = run/entry['path']
        check(path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'complete archive SHA inventory')
    check([(c['master'], c['task']) for c in report['cohorts']] ==
          [(m,t) for m in (910901,910902) for t in ('slow_lag_sign','component_balance')], 'all declared engineering cohorts')
    all_sources, all_passed = set(), True
    for cohort in report['cohorts']:
        master = cohort['master']; bit = int(cohort['task'] == 'component_balance')
        check(cohort['dataset_id'] == ('AMP-2' if bit else 'TEMPO-4') and cohort['designed_complexity_level'] == 5 and
              cohort['complexity_scale_max'] == 5, 'prospective dataset identity and complexity')
        directory = run/cohort['directory']
        check([v['view'] for v in cohort['views']] == ['training','validation-intact','validation-deleted'], 'all three views')
        values = [data(directory/v['data_archive'], 256 if i == 0 else 128, master, bit) for i,v in enumerate(cohort['views'])]
        for x, mask, y, ids in values[:2]:
            sources = set(ids)
            check(not sources & all_sources, 'disjoint tasks, masters and TRAIN/VALIDATION sources')
            all_sources.update(sources)
        train_ids, val_ids = set(values[0][3]), set(values[1][3])
        check(len(train_ids) == 128 and len(val_ids) == 64 and
              {((v & ((1 << 31)-1)) >> 1) for v in train_ids|val_ids} == set(range(192)), 'complete randomly split source universe')
        x, mask, y, ids = values[1]; dx, dm, dy, dids = values[2]
        check(y == dy and ids == dids, 'deleted rows/labels/source IDs unchanged')
        for row in range(0, 128, 2):
            engine = M.MT19937_64(M.stream_seed(M.stream_seed(master,0x74632d64656c7631),ids[row]))
            erasure = [(engine.draw() >> 11)/2**53 < .30 for _ in range(288)]
            for paired_row in (row,row+1):
                for cell, erased in enumerate(erasure):
                    i = paired_row*288+cell
                    expected = bool(mask[i]) and not erased
                    check(bool(dm[i]) == expected and dx[i] == (x[i] if expected else 0), 'exact source-keyed .30 deletion/value subset')
        cohort_passed = True
        for value, view in zip(values, cohort['views']):
            cohort_passed = decisions(directory/view['decisions_archive'], value, cohort, view) and cohort_passed
        check(cohort['gate_passed'] == cohort_passed, 'all prospective cohort gates')
        all_passed = all_passed and cohort_passed
    check(len(all_sources) == 768 and report['status'] == ('passed' if all_passed else 'failed'), 'all source populations and joint information gate')
    for key in ('encoder_calls','optimizer_updates','head_fits','PCA_fits'):
        check(report[key] == 0, 'data-only execution '+key)
    check(report['generator_calls'] == 4 and report['information_rows'] == 2048 and report['data_archives'] == 12 and
          report['decision_archives'] == 12 and not report['TEST_generated'] and not report['quality_masters_generated'], 'closed execution ledger')
    result = {'protocol': 'two-component-information-saved-check-v1', 'status': 'passed',
              'report_sha256': sha(run/'report.json'), 'reader_sha256': sha(Path(__file__)),
              'module_sha256': PINS, 'checks': CHECKS + R.CHECKS + M.CHECKS,
              'CPU_archives_decoded': DECODES, 'frequency_searches': 0, 'selected_fit_rows': 2048,
              'encoder_calls': 0, 'head_fits': 0, 'generator_calls': 0,
              'information_gate_passed': all_passed}
    output = capsule/'saved-check.json'
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

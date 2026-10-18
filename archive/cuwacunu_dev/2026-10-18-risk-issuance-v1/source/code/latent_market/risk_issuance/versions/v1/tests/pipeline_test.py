#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent artificial staged-head publication; no market inputs or claims."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def check_publication(leaf):
    publication = leaf / 'published'
    manifest = json.loads((publication / 'head.json').read_text())
    receipt = json.loads((leaf / 'run.json').read_text())
    assert all(isinstance(value, str) for value in manifest.values())
    assert manifest['schema'] == 'risk-issuance-publication-v1'
    assert manifest['policy_sha256'] == 'a7f52e1bc77f613041f56c8ce1ce7bfded42046dd4a9d2c31dfe3c7c2277220f'
    assert receipt['head_manifest_sha256'] == digest(publication / 'head.json')
    assert receipt['head_model_ref'] == manifest['head_model_ref']
    for name in ['readouts.pt', 'predictions.pt', 'origins.csv', 'training-selection.csv', 'prediction-ledger.csv']:
        field = {'readouts.pt': 'readouts_sha256', 'predictions.pt': 'predictions_sha256'}.get(name, name + '_sha256')
        assert manifest[field] == digest(publication / name)
    assert receipt['scores_sha256'] == digest(leaf / 'scores.csv')
    assert receipt['target_audit_sha256'] == digest(leaf / 'target-audit.csv')
    assert receipt['encoder_calls'] == receipt['upstream_updates_during_head_stage'] == '0'
    assert receipt['market_fitting'] == 'false'
    origins = table(publication / 'origins.csv')
    selection = table(publication / 'training-selection.csv')
    predictions = table(publication / 'prediction-ledger.csv')
    targets = table(leaf / 'target-audit.csv')
    scores = table(leaf / 'scores.csv')
    n, arms, labels = map(int, [manifest['origin_count'], manifest['arm_count'], manifest['observable_count']])
    assert n == len(origins) and arms == 7 and labels >= 2
    indexed = {int(row['row_id']): row for row in origins}
    assert len(indexed) == n and len({row['pair_id'] for row in origins}) == n
    assert len(selection) == len(targets) == n * labels
    assert len(predictions) == n * arms * labels
    train_begin, train_end, fit = map(int, [manifest['train_begin_ms'], manifest['train_end_ms'], manifest['head_fit_ms']])
    eval_begin, eval_end = map(int, [manifest['evaluation_begin_ms'], manifest['evaluation_end_ms']])
    upstream_fits = [int(manifest['upstream_' + name]) for name in ['encoder_fit_ms', 'online_fit_ms', 'target_fit_ms', 'dynamics_fit_ms']]
    upstream_fits.append(int(manifest['native_fit_ms']))
    assert max(upstream_fits) < train_begin <= train_end <= fit < eval_begin <= eval_end
    target_index = {(int(row['row_id']), int(row['observable_index'])): row for row in targets}
    assert len(target_index) == n * labels
    selected_counts = [0] * labels
    exclusions = set()
    for row in selection:
        row_id, observable = int(row['row_id']), int(row['observable_index'])
        origin = indexed[row_id]
        audit = target_index[(row_id, observable)]
        assert row['selected'] == audit['training_selected']
        assert row['reasons'] == audit['training_exclusion_reasons']
        assert int(row['selected']) == (int(row['reasons']) == 0)
        if row['selected'] == '1':
            assert int(origin['head_role']) == 1 and origin['common_eligible'] == '1'
            endpoint = int(origin['origin_ms'])
            assert max(upstream_fits) < endpoint and train_begin <= endpoint <= train_end
            assert train_begin <= int(origin['input_begin_ms']) <= endpoint
            assert int(origin['input_available_ms']) <= int(origin['decision_ms']) <= fit
            assert audit['target_valid'] == '1' and int(audit['available_ms']) <= fit
            assert train_begin <= int(audit['support_begin_ms']) <= int(audit['support_end_ms']) <= train_end
            for key in ['available_ms', 'support_begin_ms', 'support_end_ms']:
                assert row['label_' + key] == audit[key]
            selected_counts[observable] += 1
        else:
            exclusions.add(int(row['reasons']))
            assert all(row['label_' + key] == '-1' for key in ['available_ms', 'support_begin_ms', 'support_end_ms'])
    assert all(count > 1 for count in selected_counts) and exclusions
    assert list(map(int, receipt['common_training_rows'].split(';'))) == selected_counts
    seen = set()
    issued = unavailable = 0
    for row in predictions:
        row_id, observable = int(row['row_id']), int(row['observable_index'])
        origin = indexed[row_id]
        key = (row_id, row['arm'], observable)
        assert key not in seen
        seen.add(key)
        assert row['pair_id'] == origin['pair_id'] and row['origin_ms'] == origin['origin_ms']
        assert row['decision_ms'] == origin['decision_ms'] and int(row['head_fit_ms']) == fit
        value = float(row['prediction'])
        assert math.isfinite(value)
        if row['valid'] == '1':
            assert int(origin['head_role']) == 2 and origin['common_eligible'] == '1'
            assert fit < int(row['origin_ms']) and eval_begin <= int(row['origin_ms']) <= eval_end
            assert int(row['reasons']) == 0
            issued += 1
        else:
            assert value == 0 and int(row['reasons']) != 0
            unavailable += 1
    assert issued and unavailable
    for observable in {row['observable'] for row in scores}:
        rows = [row for row in scores if row['observable'] == observable]
        assert len(rows) == arms and len({row['common_scored'] for row in rows}) == 1
        assert all(int(row['common_scored']) > 0 for row in rows)
        assert all(int(row['total']) == sum(int(origin['head_role']) == 2 for origin in origins) for row in rows)
    return manifest, {'origins': n, 'arms': arms, 'observables': labels,
                      'selectedTrainCounts': selected_counts, 'issued': issued, 'unavailable': unavailable}


def main():
    assert Path('/.dockerenv').exists(), 'Use the managed development container'
    binary = Path(sys.argv[1]).resolve(strict=True)
    parent = Path(tempfile.mkdtemp(prefix='risk-stage-pipeline-', dir=os.environ['TMPDIR']))
    log = parent / 'pipeline.log'

    def run(arguments, succeeds=True):
        result = subprocess.run([str(binary), *map(str, arguments)], text=True, capture_output=True)
        with log.open('a') as stream:
            stream.write(f'COMMAND {arguments!r}\nEXIT {result.returncode}\nSTDOUT\n{result.stdout}\nSTDERR\n{result.stderr}\n')
        if (result.returncode == 0) != succeeds:
            raise RuntimeError(f'Unexpected exit {result.returncode}: {arguments}\n{result.stdout}\n{result.stderr}')
        return result

    leaf = parent / 'complete'
    run(['synthetic-engineering', leaf])
    manifest, counts = check_publication(leaf)
    publication = leaf / 'published'
    pin = digest(publication / 'head.json')
    run(['verify-synthetic', leaf / 'inputs', publication, pin])
    run(['synthetic-engineering', leaf], succeeds=False)
    rejected = parent / 'market-must-not-exist'
    run(['market', rejected], succeeds=False)
    assert not rejected.exists()
    run(['verify-synthetic', leaf / 'inputs', publication, '0' * 64], succeeds=False)
    original = {path.relative_to(publication): digest(path) for path in publication.iterdir() if path.is_file()}
    for name in ['readouts.pt', 'predictions.pt', 'origins.csv', 'training-selection.csv', 'prediction-ledger.csv', 'head.json']:
        bad = parent / ('corrupt-' + name)
        shutil.copytree(publication, bad)
        with (bad / name).open('ab') as stream:
            stream.write(b'!')
        run(['verify-synthetic', leaf / 'inputs', bad, pin], succeeds=False)
    for field, value in [('head_fit_ms', str(int(manifest['head_fit_ms']) + 1)),
                         ('native_model_ref', '0' * 64),
                         ('upstream_target_coordinate_id', manifest['upstream_target_coordinate_id'] + ':other-fold')]:
        bad = parent / ('relabel-' + field)
        shutil.copytree(publication, bad)
        altered = dict(manifest)
        altered[field] = value
        (bad / 'head.json').write_text(json.dumps(altered, separators=(',', ':')) + '\n')
        run(['verify-synthetic', leaf / 'inputs', bad, digest(bad / 'head.json')], succeeds=False)
    assert original == {path.relative_to(publication): digest(path) for path in publication.iterdir() if path.is_file()}
    (parent / 'summary.json').write_text(json.dumps(counts, indent=2) + '\n')
    print('PASS staged risk-head pipeline: seven checked feature arms, distinct legal TRAIN/evaluation stages, mature common labels, complete invalid rows, immutable reload and corruption/parent/coordinate/stage rejection; synthetic engineering only')
    print('Evidence preserved at ' + str(parent))


if __name__ == '__main__':
    main()

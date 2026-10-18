#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Add a verified transfer/selection event; never rewrite a market candidate."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/cuwacunu')
EVIDENCE = ROOT / 'doc/results/market_location_scale_synthetic_transfer_v1'
LEDGER = ROOT / 'doc/encoder_search_ledger.json'
MARKDOWN = ROOT / 'doc/ENCODER_SEARCH_LEDGER.md'
MODEL = 'RPB-v30.market-01'
PRIOR = 'RPB-v24.market-01.alt-01'
CARD = 'c3cfe0a04b174734599b2b01363c90c3f9b81fd5964fb30ffd4ce8a21976260e'
CHECKPOINT = 'c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb'
FLOORS = {
    ('synthetic_train', 'validation_intact', 'ridge'): .9895833333333334,
    ('synthetic_train', 'validation_intact', 'tiny_secondary'): .9774305555555556,
    ('synthetic_train', 'validation_deleted', 'ridge'): .93359375,
    ('synthetic_train', 'validation_deleted', 'tiny_secondary'): .9253472222222222,
    ('market_scaler', 'validation_intact', 'ridge'): 1.,
    ('market_scaler', 'validation_intact', 'tiny_secondary'): .9952256944444444,
    ('market_scaler', 'validation_deleted', 'ridge'): .9609375,
    ('market_scaler', 'validation_deleted', 'tiny_secondary'): .9505208333333334,
}


def require(value, why):
    if not value:
        raise RuntimeError(why)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def exclusive(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def asset(path):
    require(path.resolve().is_relative_to(ROOT), 'Asset must be in integration')
    return {'repository': 'integration', 'path': str(path.relative_to(ROOT)),
            'sha256': sha(path), 'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--closure-receipt', required=True, type=Path)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT, 'Use existing cuwacunu container')
    require(sha(LEDGER) == 'b8d3ab33dae4937a0202ac64302f58a5b6c803c024b7277604f26855c76a4012' and
            sha(MARKDOWN) == '31ca6e15a0c657e81d79befcd9fe45aa006e1c8f8dc8d516713a9fa01c7b7192',
            'Preserved11-entry market ledger must precede this separate selection event')
    run = args.run.resolve(strict=True)
    require(run.is_relative_to(ROOT / 'output/runs/market-location-scale-synthetic-transfer'), 'Declared transfer run')
    audit = load(run / 'independent-audit.json')
    execution = load(run / 'verification.json')
    completion_path = EVIDENCE / 'completion.json'
    completion = load(completion_path)
    closure = load(args.closure_receipt)
    review_path = EVIDENCE / 'report-verification.json'
    review = load(review_path)
    require(audit['status'] == execution['status'] == completion['status'] == closure['status'] == 'passed',
            'Verified execution, independent audit, report and closure required')
    require(review['status'] == 'passed' and review['completion_sha256'] == sha(completion_path),
            'Independent verification of this exact report completion required')
    for field, path in {
        'report_sha256': ROOT / 'doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_RESULTS.md',
        'table_rows_sha256': EVIDENCE / 'table_rows.json',
        'asset_hashes_sha256': EVIDENCE / 'asset_hashes.json',
    }.items():
        require(review[field] == completion[field] == sha(path), 'Verified report/table/asset binding: ' + field)
    require(completion['independent_audit_sha256'] == sha(run / 'independent-audit.json') and
            completion['wrapper_verification_sha256'] == sha(run / 'verification.json'),
            'Report must bind exact audit and execution')
    for receipt in (audit, execution, completion):
        require(receipt['card_sha256'] == CARD and receipt['checkpoint_sha256'] == CHECKPOINT,
                'Exact prospective card and frozen instance across every verification')
    require(audit['protocol'] == 'market-location-scale-synthetic-transfer-v1' and
            audit['encoder_updates'] == audit['decoder_calls'] == audit['auditor_encoder_calls'] ==
            audit['auditor_head_fits'] == audit['new_calibration_fits'] == 0 and
            audit['full_cuda_encode_calls'] == 768 and audit['same_checkpoint_parent_correctness_calls'] == 6 and
            audit['exact_parent_prefix_cells'] == 1536, 'Frozen-transfer invariant contract')
    require(closure['original_baseline_files'] == closure['verified_files'] == 77831 and
            closure['failed_files'] == 0 and closure['original_receipts_and_scripts_preserved'],
            'Complete original two-repository preservation required')
    rows = audit['retention_conditions']
    require(len(rows) == 8 and {(r['mode'], r['view'], r['head']) for r in rows} == set(FLOORS),
            'All eight prospective comparisons required')
    for row in rows:
        require(row['floor'] == FLOORS[(row['mode'], row['view'], row['head'])] and
                row['arithmetic_tolerance'] == 1e-12 and
                row['passed'] == (row['mean'] + 1e-12 >= row['floor']), 'Unchanged gate arithmetic')
    passed = all(row['passed'] for row in rows) and audit['full_coverage']
    require(passed == audit['balanced_transfer_gate_passed'] and
            passed == completion['balanced_transfer_gate_passed'], 'Independent audit/report decision must agree')
    before_bytes, before_md = LEDGER.read_bytes(), MARKDOWN.read_bytes()
    ledger = json.loads(before_bytes)
    entries = copy.deepcopy(ledger['candidate_entries'])
    views = copy.deepcopy(ledger['leader_views'])
    live = copy.deepcopy(ledger['live'])
    require(len(entries) == 11 and live['primary_architecture'] == 'RPB-v22' and
            live['primary_serving_market_checkpoint_selected'] is False and
            live['leading_balanced_research_reference'] == PRIOR, 'Keep primary and prior balanced selection')
    winner = next(entry for entry in entries if entry['model'] == MODEL)
    criteria = winner['own_original_criteria']
    require(criteria['passed'] == 9 and criteria['strict_passed'] == 8 and
            criteria['all_three_trend_neural_positive'] and criteria['full_coverage'] and
            criteria['full_market_gate_passed'], 'Verified complete market pass required')
    checkpoint_count = 0
    for entry in entries:
        for pin in entry['checkpoints']:
            path = ROOT / pin['path']
            require(sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'], 'Preserved checkpoint availability/hash')
            checkpoint_count += 1
    require(checkpoint_count == 55, 'All55 checkpoints preserved')
    event = {'event_id': '2026-10-15-frozen-transfer/' + MODEL,
             'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'model': MODEL, 'market_original_passed': 9, 'market_strict_passed': 8,
             'transfer_conditions_passed': sum(row['passed'] for row in rows),
             'transfer_conditions_total': 8, 'full_coverage': audit['full_coverage'],
             'balanced_transfer_gate_passed': passed, 'conditions': rows,
             'prior_balanced_reference': PRIOR, 'selected_balanced_reference': MODEL if passed else PRIOR,
             'balanced_pointer_changed': passed, 'primary_changed': False,
             'completion': asset(completion_path), 'independent_audit': asset(run / 'independent-audit.json'),
             'report_verification': asset(review_path),
             'execution_verification': asset(run / 'verification.json'), 'closure_preservation': asset(args.closure_receipt),
             'report': asset(ROOT / 'doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_RESULTS.md'),
             'previous_live': live, 'candidate_records_unchanged': 11, 'checkpoints_preserved': 55,
             'new_encoder_updates': 0, 'selection_encoder_calls': 0, 'selection_head_fits': 0,
             'new_git_commits_or_pushes': 0,
             'scope': 'Known MKT-HIST-2 and compatible TEMPO-5 development; fresh chronological confirmation remains unmeasured'}
    ledger.setdefault('frozen_transfer_events', []).append(event)
    ledger.setdefault('leader_history', []).append(copy.deepcopy(event))
    ledger['live'].update(
        active_work=('Verified9/9 market and8/8 frozen TEMPO-5 retention; fresh chronology next' if passed else
                     'Verified9/9 market; frozen TEMPO-5 retention failed; preserve balanced native36'),
        balanced_promotion_pending_transfer=None,
        verified_new_nine_of_nine_winner=MODEL if passed else None,
        latest_verified_frozen_transfer=asset(completion_path),
        leading_balanced_research_reference=MODEL if passed else PRIOR,
        balanced_reference_original_card_conditions=('9/9 market;8/8 strict floors;8/8 frozen TEMPO-5 retention' if passed else
                                                     live['balanced_reference_original_card_conditions']),
        preserved_previous_balanced_research_reference=PRIOR)
    ledger['live']['active_candidate_card']['status'] = ('verified_market_and_frozen_transfer_pass' if passed else
                                                        'verified_market_pass_frozen_transfer_failed')
    ledger['last_index_update'] = {'model': MODEL, 'mode': 'separate frozen-transfer selection event',
                                   'new_encoder_calls': 0, 'new_encoder_updates': 0, 'new_head_fits': 0,
                                   'new_scaler_fits': 0, 'tensor_archives_deserialized': 0,
                                   'Git_commits_or_pushes': 0, 'mirror_required': True}
    require(ledger['candidate_entries'] == entries and ledger['leader_views'] == views,
            'Market records and every task/head leader remain unchanged')
    ledger['evidence_stage'] = 'Verified known-source market and separate frozen synthetic development; no fresh chronology'
    heading = ('# Encoder search ledger: V30 verified balanced research reference\n\n' if passed else
               '# Encoder search ledger: V30 market pass, native36 balanced reference\n\n')
    intro = ('RPB-v30.market-01 passes9/9 market conditions,8/8 strict native36 floors, all three\n'
             'positive trend fits and full coverage. Its separately verified frozen TEMPO-5\n'
             f'transfer passes{event["transfer_conditions_passed"]}/8 retention comparisons; coverage is\n'
             f'{"complete" if audit["full_coverage"] else "incomplete"}. Balanced research reference: **{MODEL if passed else PRIOR}**.\n'
             'RPB-v22 remains primary. All11 immutable candidate entries,55 checkpoint pins,\n'
             'native36 and every stronger individual head reference remain preserved.\n'
             'See [the frozen transfer report](MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_RESULTS.md)\n'
             'and the exact JSON ledger for the additive selection event and unchanged leader views.\n'
             'Fresh chronological confirmation is the next separate research step.\n\n'
             'The following text is the complete preserved market-stage ledger at the time of\n'
             'this selection; its earlier transfer-pending descriptions are historical.\n\n')
    after_md = (heading + intro).encode() + before_md
    backups = [EVIDENCE / 'selection-before.json', EVIDENCE / 'selection-before.md']
    generated = [EVIDENCE / 'selection-generated.json', EVIDENCE / 'selection-generated.md']
    receipt = EVIDENCE / 'selection-receipt.json'
    require(not any(path.exists() for path in backups + generated + [receipt]), 'Preserve prior selection attempts')
    for path, content in zip(backups + generated, [before_bytes, before_md, encode(ledger), after_md]):
        exclusive(path, content)
    require(LEDGER.read_bytes() == before_bytes and MARKDOWN.read_bytes() == before_md, 'Concurrent ledger mutation')
    for path, content in ((LEDGER, encode(ledger)), (MARKDOWN, after_md)):
        stage = path.with_name(path.name + '.frozen-transfer')
        exclusive(stage, content)
        os.replace(stage, path)
    require(load(LEDGER)['candidate_entries'] == entries and load(LEDGER)['leader_views'] == views,
            'Final historical-record preservation')
    exclusive(receipt, encode({'kind': 'encoder_search_frozen_transfer_selection_v1', 'status': 'passed',
                               'balanced_transfer_gate_passed': passed, 'balanced_pointer_changed': passed,
                               'primary_changed': False, 'candidate_records_unchanged': 11,
                               'checkpoints_hash_verified': 55, 'before': asset(backups[0]),
                               'after': asset(LEDGER), 'after_markdown': asset(MARKDOWN),
                               'event': event, 'script': asset(Path(__file__))}))
    print('FROZEN_TRANSFER_SELECTION_RECORDED balanced_changed=' + str(passed), flush=True)


if __name__ == '__main__':
    main()

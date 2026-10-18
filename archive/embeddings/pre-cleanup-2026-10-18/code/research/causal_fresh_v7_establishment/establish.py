# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Join completed independent evidence without changing frozen producer reports."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'Existing project container required')
    require(not args.output.exists(), 'Preserve every previous establishment attempt')
    manifest = json.loads(args.manifest.read_text())
    own = Path(__file__).read_bytes()
    require(hashlib.sha256(own).hexdigest() == manifest['source']['sha256'], 'Frozen join source')
    values = {}
    for item in manifest['inputs']:
        relative = Path(item['staged_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Bounded evidence path')
        data = (args.manifest.parent / relative).read_bytes()
        require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], 'Evidence bytes: '+item['id'])
        require(item['id'] not in values, 'Unique evidence ID')
        values[item['id']] = json.loads(data)
    comparison = values['comparison']
    table = values['table_rows']
    quality = values['quality_audit']
    training = values['training_audit']
    require(values['pairings'] == quality, 'Pairing evidence is an explicit alias of the exact independently audited quality receipt')
    require(quality['status'] == training['status'] == 'passed', 'Both independent audits must pass')
    require(not quality['mismatches'] and not training['mismatches'], 'Zero audit mismatches')
    require(quality['postmeasurement_quality_and_pairing_audit_passed'], 'Independent quality completion')
    pairs = quality['paired_initial_equality']
    require(pairs['reference_data_archives'] == len(pairs['pairs']) == 13, 'All13 initial DATA/statistic pairs')
    require(pairs['all_declared_numeric_tensor_values_exact'] and
            all(p['all_declared_numeric_tensor_values_exact'] for p in pairs['pairs']), 'Exact whole40 initial pairing')
    require(quality['full_coverage'] and comparison['full_coverage'] and table['full_coverage'], 'Full coverage')
    require(comparison['runtime_trust_admission_passed'] and table['runtime_trust_admission_passed'], 'Runtime trust')
    mirror = values['complete_mirror']
    require(mirror['kind'] == 'causal_fresh_v7_container_to_local_evidence_mirror_v1' and
            mirror['status'] == 'passed' and mirror['scope'] == 'Complete' and
            mirror['attempt'] == 'attempt-001' and mirror['source_files_verified'] == 106 and
            mirror['all_mirrored_pt_ignored'], 'Complete V7 local evidence mirror')
    require(mirror['source_stage']['card_sha256'] ==
            'f913044fd4bdbeff637991087f91d0b47bca451b31dfe664fb70b9555bbae593',
            'Mirror binds the frozen V7 card')
    require(comparison['candidate'] == quality['model'] == training['model'] == table['model'] == 'CPB-v7.market-01', 'Named candidate across both audits and producer')
    require(comparison['baseline'] == 'CPB-v3.market-01' and comparison['baseline_unchanged'], 'Protected clean baseline')
    criteria = comparison['numeric_conditions']
    count = sum(item['passed'] is True for item in criteria)
    require(len(criteria) == comparison['numeric_total'] == 22 and count == comparison['numeric_passed'], 'Frozen22 numeric decisions')
    require(quality['clean_baseline_comparison']['numeric_passed'] == count, 'Independent numeric count')
    pending = {'initial_full40_pairings': 'pending', 'independent_training_audit': 'pending', 'independent_quality_audit': 'pending'}
    require(comparison['mandatory_postmeasurement_trust'] == pending and comparison['trust_admission_passed'] is False,
            'Original producer trust snapshot remains pending and unchanged')
    require(table['mandatory_postmeasurement_trust'] == pending and table['trust_protocol_admission_passed'] is False,
            'Original table trust snapshot remains pending and unchanged')
    for identifier in ('comparison', 'table_rows', 'card', 'quality_receipt'):
        item = next(i for i in manifest['inputs'] if i['id'] == identifier)
        require(any(p['bytes'] == item['bytes'] and p['sha256'] == item['sha256']
                    for p in quality['verified_artifacts']), 'Quality audit binds these exact producer bytes: '+identifier)
    require(values['card']['instance'] == 'CPB-v7.market-01', 'Prospective card identity')
    require(values['quality_receipt']['card_sha256'] == next(i['sha256'] for i in manifest['inputs'] if i['id'] == 'card'), 'Card-to-execution binding')
    overall = count == 22
    result = {
        'kind': 'causal_fresh_v7_postmeasurement_establishment_v1', 'status': 'passed',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidate': 'CPB-v7.market-01', 'baseline': 'CPB-v3.market-01', 'baseline_changed': False,
        'measured_candidate_established': True, 'runtime_trust_admission_passed': True,
        'trust_admission_passed': True, 'initial_full40_pairings_passed': True, 'initial_pairing_count': 13,
        'independent_training_audit_passed': True, 'independent_quality_audit_passed': True,
        'full_coverage': True, 'numeric_passed': count, 'numeric_total': 22,
        'numeric_conditions_passed': overall, 'numeric_and_coverage_conditions_passed': overall,
        'overall_establishment_passed': overall, 'advancement_conditions_passed': overall,
        'overall_winner': False, 'winner_selection_made': False,
        'frozen_producer_pending_trust_preserved': True, 'application_changed': False, 'automatic_promotion': False,
        'evidence': {i['id']: {k:i[k] for k in ('path','bytes','sha256')} for i in manifest['inputs']},
        'source': manifest['source'], 'model_calls': 0, 'head_fits': 0, 'training_updates': 0,
        'scope': 'Postmeasurement evidence join only; passed status establishes trustworthy measurements, not an overall quality winner.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':'passed','numeric_passed':count,'numeric_total':22,
                      'advancement_conditions_passed':overall,'overall_winner':False}))


if __name__ == '__main__':
    main()

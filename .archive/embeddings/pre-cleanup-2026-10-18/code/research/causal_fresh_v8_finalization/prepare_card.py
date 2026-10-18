# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE.
"""Prospective metadata-only card generator; actual frozen inputs arrive later.

The root-reviewed plan supplies hashes, actual container paths and a field map
for the independent auditors' JSON schemas. This tool cannot choose a quality
floor, rewrite a measured result, run a model or apply a metadata proposal.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import sys

INPUTS = frozenset(('ledger', 'catalog', 'prior_snapshot', 'descriptor',
    'experiment_card', 'source_manifest', 'build', 'training', 'resume',
    'initialization', 'calibration', 'engineering', 'gradient', 'initial_proof',
    'final_proof', 'training_audit',
    'quality_audit', 'pairings', 'establishment', 'mirror', 'quality_receipt',
    'comparison', 'v3_table', 'v4_table', 'v5_table', 'v6_table', 'v7_table', 'v8_table',
    'v6_establishment', 'v7_establishment', 'temporal_proof', 'temporal_learning',
    'host_recovery_card', 'host_recovery_review',
    'results', 'review', 'detail'))
ASSERTIONS = dict(pairing_status='passed', pairing_count=13,
    training_audit_status='passed', quality_audit_status='passed',
    establishment_status='passed', establishment_runtime=True,
    establishment_trust=True, establishment_coverage=True,
    establishment_measured=True)
EXPERIMENT_SHA = '6acfaadfe758c0d252c8138fc2830807d40444f9a9adfa61f85ec13dde7c63a6'
DESCRIPTOR_SHA = 'df3d0d61e912d0e5665500f5b2255f3300252f769ba359fb570660e1a992c5c4'
PRIOR_SNAPSHOT_SHA = '607ad940c9def57dd0390d7cbfb763c08fbaf769dbe911efc3d8eaf69f47ce7a'
V6_ESTABLISHMENT_SHA = 'ff3a26ea261fb9a8aa9ce681ee37ff882ef703f0056dce421eefc3e1a641c6fd'
V7_ESTABLISHMENT_SHA = '7678e84a676453c0294f0654af6117e06246f1639f4e67f9ff78e9cf51049512'
SOURCE_MANIFEST_SHA = '8daba935ee45332b892225c6994a00b59ad95b2e4da4fe7348f6c6518266cf2e'
NATIVE_LAYOUT = 'parent32[0:32];mean4[32:36];uncertainty4[36:40];independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;variance-reads-live-uncertainty4-only'
FINALIZATION_ROOT = pathlib.Path('/opt/cuwacunu/research/causal-fresh-v8-finalization')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def pin(path, label=None):
    path = pathlib.Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked metadata input')
    data = path.read_bytes()
    return dict(path=label or str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    require(sys.platform.startswith('linux') and pathlib.Path('/opt/cuwacunu/research').is_dir(),
            'Existing Linux research container only')
    require(args.output.is_absolute() and args.output.resolve().is_relative_to(FINALIZATION_ROOT)
            and args.output.resolve() != FINALIZATION_ROOT, 'New exclusive finalization source leaf only')
    require(not args.output.exists(), 'Exclusive prospective source/card leaf required')
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    require(plan['kind'] == 'cpb_v8_metadata_finalization_input_plan_v1', 'Root-reviewed input plan required')
    require(plan['approved_for_metadata_card_generation'] is True, 'Explicit root plan admission')
    require(len(plan['inputs']) == len(INPUTS) and {r['id'] for r in plan['inputs']} == INPUTS,
            'Exact saved metadata input IDs; no arrays or unpublished partial results')
    require(plan['container_attempt'] == '/opt/cuwacunu/research/causal-fresh-v8/attempt-001',
            'Original V8 completed attempt only')
    contract = plan['evidence_contract']
    require(set(contract['assertions']) == set(ASSERTIONS), 'All independent establishment requirements declared')
    for name, expected in ASSERTIONS.items():
        row = contract['assertions'][name]
        expected_input = ('pairings' if name.startswith('pairing_') else
                          'training_audit' if name.startswith('training_audit_') else
                          'quality_audit' if name.startswith('quality_audit_') else 'establishment')
        require(row['input'] == expected_input and isinstance(row['path'], list) and row['path']
                and row['equals'] == expected and type(row['equals']) is type(expected), 'Exact evidence requirement')
    for key in ('establishment_bindings_path', 'training_audit_checks_path', 'quality_audit_checks_path'):
        require(isinstance(contract[key], list) and contract[key], 'Actual receipt field selector required')
    require(set(contract['establishment_binding_keys']) == {
        'experiment_card', 'comparison', 'v8_table', 'pairings', 'training_audit', 'quality_audit',
        'source_manifest', 'build', 'mirror', 'quality_receipt', 'temporal_proof', 'temporal_learning',
        'host_recovery_card', 'host_recovery_review'},
        'Actual establishment evidence-map aliases required')
    files, data_by_id = [], {}
    for row in plan['inputs']:
        path = pathlib.Path(row['source_path'])
        require(path.is_absolute(), 'Absolute existing-container input path')
        label = pathlib.PurePosixPath(row['path'])
        require(not label.is_absolute() and '..' not in label.parts, 'Research-relative durable input label')
        actual = pin(path, row['path'])
        require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Root input plan hash mismatch: ' + row['id'])
        fmt = row.get('format', 'json')
        require(fmt in ('json', 'text'), 'Metadata-only JSON/text format')
        require(fmt == ('text' if row['id'] in ('results', 'review', 'detail') else 'json'),
                'Exact metadata format for each admitted input')
        # Validate UTF-8/JSON before writing any card; no tensors are opened.
        contents = path.read_bytes()
        text = contents.decode('utf-8-sig')
        if fmt == 'json':
            json.loads(text)
        suffix = '.json' if fmt == 'json' else '.txt'
        files.append(dict(id=row['id'], **actual, staged_path='inputs/'+row['id']+suffix, format=fmt))
        data_by_id[row['id']] = contents
    by_id = {row['id']: row for row in files}
    for name, expected in (('experiment_card', EXPERIMENT_SHA), ('descriptor', DESCRIPTOR_SHA),
                           ('prior_snapshot', PRIOR_SNAPSHOT_SHA),
                           ('v6_establishment', V6_ESTABLISHMENT_SHA),
                           ('v7_establishment', V7_ESTABLISHMENT_SHA),
                           ('host_recovery_card', '526e51c1357d6f89b289cd96e202a7853ad54c17e991f01c8bcb293e3cbd28b9'),
                           ('host_recovery_review', '8557b4e386ebe43a7f8cfd1bf50d3222a0e3371f8fde66306114d99bc1fcd67f'),
                           ('source_manifest', SOURCE_MANIFEST_SHA)):
        require(by_id[name]['sha256'] == expected, 'Original prospective/prior snapshot binding')
    generator_path = pathlib.Path(__file__).with_name('finalize.py')
    generator = pin(generator_path, 'code/research/causal_fresh_v8_finalization/finalize.py')
    require(all(generator[k] == plan['generator'][k] for k in ('bytes', 'sha256')), 'Prospective finalizer bytes')
    next_direction = plan['next_direction']
    require(next_direction['status'].startswith('proposal_only') and isinstance(next_direction['action'], str),
            'Next direction remains a saved-data proposal; no training authorization')
    card = dict(kind='cpb_v8_metadata_only_finalization_card_v1', client_date=plan['client_date'],
        declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Saved metadata proposal only, no live application/model/fit/head tuning/checkpoint selection/publication',
        generator=generator, card_generator=pin(__file__, 'code/research/causal_fresh_v8_finalization/prepare_card.py'),
        input_plan=pin(args.plan), inputs=files, evidence_contract=contract,
        prospective_encoder_card_sha256=EXPERIMENT_SHA, prospective_descriptor_sha256=DESCRIPTOR_SHA,
        prior_snapshot_manifest_sha256=PRIOR_SNAPSHOT_SHA, container_attempt=plan['container_attempt'],
        expected_source_capsule_files=113, expected_binaries=7, expected_translation_units=11,
        physical_owned_files=dict(family=51, research=31), native_layout=NATIVE_LAYOUT,
        next_direction=next_direction, numeric_conditions=22, numeric_pass_count='Derived from saved original22 records',
        producer_trust='Original runtime-only pending fields preserved; separate independent establishment receipt required',
        leader_policy='16 highest final quality scopes and4 lowest final reconstruction scopes acrossV3/V4/V5/V6/V7/V8; ties retain earlier',
        baseline='CPB-v3.market-01', application_change_allowed=False, prospective_descriptor_change_allowed=False,
        model_calls=0, head_fits=0, live_files_written=0)
    args.output.mkdir(parents=True)
    (args.output/'inputs').mkdir()
    for row in files:
        with (args.output/row['staged_path']).open('xb') as stream:
            stream.write(data_by_id[row['id']])
    with (args.output/'finalize.py').open('xb') as stream:
        stream.write(generator_path.read_bytes())
    write(args.output/'finalization-card.json', card)
    write(args.output/'card-generation.json', dict(kind='cpb_v8_metadata_only_card_generation_v1',
        status='passed_card_prepared_not_executed', generator=pin(__file__), input_plan=pin(args.plan),
        card=pin(args.output/'finalization-card.json'), inputs=files,
        model_calls=0, head_fits=0, metadata_proposals_generated=0, live_files_written=0))
    print(json.dumps(dict(status='passed_card_prepared_not_executed', card=str(args.output/'finalization-card.json'))))


if __name__ == '__main__':
    main()

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
    'comparison', 'v3_table', 'v4_table', 'v5_table', 'v6_table', 'v7_table', 'v8_table', 'v9_table', 'v10_table', 'v11_table', 'v12_table',
    'v6_establishment', 'v7_establishment', 'v8_establishment', 'v9_establishment', 'v10_establishment', 'v11_establishment', 'temporal_proof', 'temporal_learning',
    'prospective_freeze', 'protocol_preflight',
    'results', 'review', 'detail'))
ASSERTIONS = dict(pairing_status='passed', pairing_count=13,
    training_audit_status='passed', quality_audit_status='passed',
    establishment_status='passed', establishment_runtime=True,
    establishment_trust=True, establishment_coverage=True,
    establishment_measured=True)
NATIVE_LAYOUT = 'parent32[0:32];mean4[32:36];uncertainty4[36:40];independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;variance-reads-live-uncertainty4-only'
FINALIZATION_ROOT = pathlib.Path('/opt/cuwacunu/research/causal-fresh-v12-finalization')


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
    require(plan['kind'] == 'cpb_v12_metadata_finalization_input_plan_v1', 'Root-reviewed input plan required')
    require(plan['approved_for_metadata_card_generation'] is True, 'Explicit root plan admission')
    prior_snapshot_sha = plan['prior_snapshot_manifest_sha256']
    require(isinstance(prior_snapshot_sha, str) and len(prior_snapshot_sha) == 64
            and set(prior_snapshot_sha) <= set('0123456789abcdef'),
            'Actual separately root-reviewed new prior-live snapshot hash, not an inherited snapshot')
    require(len(plan['inputs']) == len(INPUTS) and {r['id'] for r in plan['inputs']} == INPUTS,
            'Exact saved metadata input IDs; no arrays or unpublished partial results')
    require(plan['container_attempt'] == '/opt/cuwacunu/research/causal-fresh-v12/attempt-001',
            'Original V12 completed attempt only')
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
        'experiment_card', 'comparison', 'v12_table', 'pairings', 'training_audit', 'quality_audit',
        'source_manifest', 'build', 'mirror', 'quality_receipt', 'temporal_proof', 'temporal_learning',
        'prospective_freeze', 'protocol_preflight'},
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
    require(by_id['prior_snapshot']['sha256'] == prior_snapshot_sha, 'Actual reviewed prior-live snapshot')
    evidence = {name: json.loads(contents.decode('utf-8-sig')) for name, contents in data_by_id.items()
                if name not in ('results','review','detail')}
    experiment, descriptor, source = evidence['experiment_card'], evidence['descriptor'], evidence['source_manifest']
    require(experiment['kind'] == 'causal_fresh_v12_prospective_card_v1', 'Original frozen V12 prospective experiment')
    require(experiment['scientific_recipe_parent'] ==
            'CPB-v11 source only;fresh weights/optimizer/statistics,not trained ancestry'
            and experiment['scoped_reference'] == 'CPB-v11.market-01',
            'Fresh V11 source recipe; V11 latest measured/source-only control; V3 remains selected')
    require(all(descriptor['card'][key] == by_id['experiment_card'][key] for key in ('bytes','sha256'))
            and descriptor['instance'] == 'CPB-v12.market-01' and descriptor['immutable_prospective_semantics'] is True,
            'Immutable actual descriptor binds actual experiment; no inherited identities')
    source_counts = experiment['source_counts']
    require(source['card_sha256'] == by_id['experiment_card']['sha256']
            and len(source['files']) == len({row['path'] for row in source['files']}) == source_counts['capsule_sources'],
            'Actual unique frozen scientific source count')
    require(experiment['advancement']['goal_protocol']['sha256'] ==
            '866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b',
            'Unchanged current prospective22 protocol')
    witness = experiment['actual_step_temporal_witness_contract']
    require(witness['total_fields'] == 60 and witness['base_fields'] == 23
            and witness['parameter_arrays'] == 12 and witness['temporal_trace_fields'] == 25
            and set(witness['uniform_trace_fields']) == {
                'temporal_pool_dispersion_weights', 'temporal_pool_dispersion_mean',
                'temporal_contextual_pool_dispersion_weights', 'temporal_contextual_pool_dispersion_mean'},
            'Frozen60-field23+12+25 schema with both uniform weights and own centers')
    proof = experiment['temporal_pooling_proof_contract']['required_proof_fields']
    require('full_attention_dependent_moment_adjoints' not in proof
            and proof['direct_pool_kernel_calls'] == 47
            and proof['F64_finite_difference_coordinates'] == proof['F32_finite_difference_coordinates'] == 18
            and all(proof[key] is True for key in (
                'attention_mean_and_uniform_dispersion_full_expression_adjoints',
                'uniform_dispersion_own_center', 'uniform_dispersion_score_independent',
                'uniform_dispersion_hidden_gradient_live', 'attention_mean_score_gradient_live',
                'dual_support_actual_trace')), 'Frozen47/18/18 and six new uniform-dispersion proof requirements')
    for version in (3,4,5,6,7,8,9,10,11):
        require(all(by_id['v%d_table'%version][key] == experiment['comparison_control']['v%d'%version][key]
                    for key in ('bytes','sha256')), 'Exact historical table pin')
    for version in (6,7,8,9,10,11):
        require(all(by_id['v%d_establishment'%version][key] == experiment['comparison_control']['v%d_establishment'%version][key]
                    for key in ('bytes','sha256')), 'Exact historical separate trust receipt pin')
    require(evidence['prospective_freeze']['kind'] == 'causal_fresh_v12_prospective_freeze_v1'
            and evidence['prospective_freeze']['status'] == 'passed'
            and all(evidence['prospective_freeze']['card'][key] == by_id['experiment_card'][key] for key in ('bytes','sha256')),
            'Actual normal freeze binds exact experiment')
    require(evidence['protocol_preflight']['kind'] == 'causal_fresh_v12_source_protocol_preflight_v1'
            and evidence['protocol_preflight']['status'] == 'passed'
            and evidence['protocol_preflight']['card_sha256'] == by_id['experiment_card']['sha256'],
            'Actual normal preflight binds exact experiment')
    generator_path = pathlib.Path(__file__).with_name('finalize.py')
    generator = pin(generator_path, 'code/research/causal_fresh_v12_finalization/finalize.py')
    require(all(generator[k] == plan['generator'][k] for k in ('bytes', 'sha256')), 'Prospective finalizer bytes')
    next_direction = plan['next_direction']
    require(next_direction['status'].startswith('proposal_only') and isinstance(next_direction['action'], str),
            'Next direction remains a saved-data proposal; no training authorization')
    card = dict(kind='cpb_v12_metadata_only_finalization_card_v1', client_date=plan['client_date'],
        declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Saved metadata proposal only, no live application/model/fit/head tuning/checkpoint selection/publication',
        generator=generator, card_generator=pin(__file__, 'code/research/causal_fresh_v12_finalization/prepare_card.py'),
        input_plan=pin(args.plan), inputs=files, evidence_contract=contract,
        prospective_encoder_card_sha256=by_id['experiment_card']['sha256'], prospective_descriptor_sha256=by_id['descriptor']['sha256'],
        prior_snapshot_manifest_sha256=prior_snapshot_sha, container_attempt=plan['container_attempt'],
        expected_source_capsule_files=source_counts['capsule_sources'], expected_binaries=source_counts['binaries'], expected_translation_units=source_counts['translation_units'],
        physical_owned_files=dict(family=source_counts['family_inside']+source_counts['family_external_header'], research=source_counts['research']), native_layout=NATIVE_LAYOUT,
        next_direction=next_direction, numeric_conditions=22, numeric_pass_count='Derived from saved original22 records',
        producer_trust='Original runtime-only pending fields preserved; separate independent establishment receipt required',
        leader_policy='16 highest final quality scopes and4 lowest final reconstruction scopes acrossV3-through-V12; ties retain earlier',
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
    write(args.output/'card-generation.json', dict(kind='cpb_v12_metadata_only_card_generation_v1',
        status='passed_card_prepared_not_executed', generator=pin(__file__), input_plan=pin(args.plan),
        card=pin(args.output/'finalization-card.json'), inputs=files,
        model_calls=0, head_fits=0, metadata_proposals_generated=0, live_files_written=0))
    print(json.dumps(dict(status='passed_card_prepared_not_executed', card=str(args.output/'finalization-card.json'))))


if __name__ == '__main__':
    main()

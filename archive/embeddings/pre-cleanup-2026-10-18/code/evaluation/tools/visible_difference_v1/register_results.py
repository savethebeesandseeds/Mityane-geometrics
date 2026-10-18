#!/usr/bin/env python3
"""Register passed evidence without replacing historical model groups."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path('/embedding')
MASTERS = [75272, 76373, 77474, 78575, 79676]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def local(path):
    return str(path.relative_to(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', required=True)
    parser.add_argument('--validation', required=True)
    args = parser.parse_args()
    assert Path('/.dockerenv').is_file(), 'managed container only'
    capsule, validation = Path(args.capsule), Path(args.validation)
    audit = json.loads(validation.read_text())
    assert audit['status'] == 'passed' and audit['protocol'] == 'visible-difference-v1'
    assert [c['timing_master'] for c in audit['per_master']] == MASTERS
    assert audit['capsule'] == str(capsule)
    assert audit['inventory_sha256'] == digest(capsule / 'artifact-integrity.json')
    durable = ROOT / 'doc/results/visible_difference_v1.json'
    report = ROOT / 'code/encoders/raw_patch_bottleneck_mae/VISIBLE_DIFFERENCE_DIAGNOSTIC.md'
    train_durable = ROOT / 'doc/results/structured_hard_timing_train_diagnostic_v1.json'
    train_report = ROOT / 'code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_TRAIN_DIAGNOSTIC.md'
    training = json.loads(train_durable.read_text())
    assert training['status'] == 'passed'
    published = json.loads(durable.read_text())
    assert published['protocol'] == audit['protocol']
    assert published['audit_record'] == audit and published['complete'] == audit['complete'], 'same published passed audit and completed run'
    path = ROOT / 'doc/embedding_versions.json'
    registry = json.loads(path.read_text())
    old = copy.deepcopy(registry)
    assert not any(v['tag'] == 'RPB-v13' for v in registry['versions']), 'exclusive new version registration'
    assert not any(g['id'] == 'RPB-v13' for g in registry['instance_bundles'])
    binding = {
        'protocol': audit['protocol'], 'dataset_codename': 'TEMPO-3',
        'dataset_recipe_id': 'structured-hard-timing-v1',
        'designed_complexity_level': 4, 'complexity_scale_max': 5,
        'measured_record': local(report), 'measured_record_sha256': digest(report),
        'durable_summary': local(durable), 'durable_summary_sha256': digest(durable),
        'capsule': local(capsule), 'artifact_inventory_sha256': audit['inventory_sha256'],
        'audit_validation': local(validation), 'audit_validation_sha256': digest(validation),
        'audit_checks': audit['checks'], 'audit_archive_decodes': audit['archive_decodes'],
        'card': 'code/evaluation/cards/visible_difference_v1.md',
        'card_sha256': audit['card_sha256'], 'reader_source_sha256': audit['source_sha256'],
        'promotion': False, 'existing_groups_replaced': False,
    }
    version = {
        'tag': 'RPB-v13', 'encoder': 'raw_patch_bottleneck_mae',
        'description': 'Early mixer with visible adjacent differences',
        'status': 'measured_descriptive_TEMPO3_no_promotion',
        'architecture_id': 'aligned-mixer-before-temporal-visible-first-difference-v1',
        'temporal_difference_input': 1, 'channel_mixer_placement': 1,
        'native_global_width': 32, 'parameter_count_at_C3_H32_F3': 228877,
        'common_initial_parameter_values': 225805, 'copied_parameter_values': 0,
        'zero_initial_difference_parameter_values': 3072,
        'comparison_instance_bundle': 'RPB-v10.alt-05',
        'encoder_updates_each': 512, 'training_policy_id': 'rpb-training-context-deletion-015-v1',
        **binding,
    }
    group = {
        'id': 'RPB-v13', 'encoder_tag': 'RPB-v13',
        'status': version['status'], 'description': version['description'],
        'master_seeds': MASTERS, 'task': 'lag_sign', 'encoder_updates_each': 512,
        'retained_points_each': [0, 512], 'native_size': 32, 'registered_parameters': 228877,
        'batch_size': 8, 'context_deletion_rate': .15,
        'checkpoint_path_pattern': local(capsule) + '/results/seed-{master}-lag_sign/candidate/point-{point}/checkpoint.pt',
        **binding,
    }
    for view, prefix in [('validation_intact', 'intact'), ('validation_deleted', 'deleted')]:
        repetitions = [r['views'].get(view) for c in audit['per_master'] for r in c['readouts']]
        assert len(repetitions) == 15, 'retain all five cohorts and three fixed repetitions'
        for head, name in [('ridge', 'linear'), ('tiny_secondary', 'neural')]:
            values = [r[head]['accuracy'] if r else None for r in repetitions]
            group[prefix + '_' + name + '_accuracy_percent'] = 100 * statistics.mean(values) if all(v is not None for v in values) else None
        coverage = [r['ridge']['coverage'] if r else None for r in repetitions]
        group[prefix + '_coverage_percent'] = 100 * statistics.mean(coverage) if all(v is not None for v in coverage) else None
    registry['versions'].append(version)
    registry['instance_bundles'].append(group)
    registry['last_updated_date'] = '2026-10-10'
    registry['latest_completed_diagnostic'] = {**binding, 'instance_groups': ['RPB-v13'],
        'reused_comparison_groups': ['RPB-v7.alt-05', 'RPB-v10.alt-05']}
    registry['visible_difference_diagnostic'] = {**binding, 'candidate': 'RPB-v13',
        'common_initialization_exact': True, 'initial_parity_exports': 15,
        'candidate_only_head_pipelines': audit['complete']['head_pipelines'],
        'candidate_only_heads': audit['complete']['individual_heads'],
        'planned_candidate_head_pipelines': 15, 'planned_candidate_heads': 30,
        'control_training_or_model_forwards': 0, 'control_head_refits': 0}
    registry['structured_hard_timing_train_diagnostic'] = {
        'protocol': training['protocol'], 'status': 'passed', 'dataset_codename': 'TEMPO-3',
        'designed_complexity_level': 4, 'complexity_scale_max': 5,
        'card_sha256': training['card_sha256'],
        'measured_record': local(train_report), 'measured_record_sha256': digest(train_report),
        'durable_summary': local(train_durable), 'durable_summary_sha256': digest(train_durable),
        'encoder_updates': 0, 'encoder_forwards': 0, 'head_refits': 0,
        'new_validation_tensor_reads': 0, 'promotion': False,
    }
    registry['last_measured_candidate_tag'] = 'RPB-v13'
    registry['bounded_candidate_tag'] = 'RPB-v13'
    registry['bounded_candidate_status'] = version['status']
    assert registry['versions'][:-1] == old['versions']
    assert registry['instance_bundles'][:-1] == old['instance_bundles']
    for key in ('active_research_tag', 'working_encoder_tag', 'working_instance_bundle', 'structured_hard_timing_diagnostic'):
        assert registry[key] == old[key], 'preserve protected reference and completed parent evidence'
    changed_keys = {'versions', 'instance_bundles', 'last_updated_date', 'latest_completed_diagnostic',
                    'last_measured_candidate_tag', 'bounded_candidate_tag', 'bounded_candidate_status'}
    assert all(registry[key] == value for key, value in old.items() if key not in changed_keys), 'preserve every other historical registry object'
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'registered', 'tag': 'RPB-v13', 'registry_sha256': digest(path), 'historical_versions_preserved': len(old['versions']), 'historical_groups_preserved': len(old['instance_bundles']), 'promotion': False}))

if __name__ == '__main__':
    main()

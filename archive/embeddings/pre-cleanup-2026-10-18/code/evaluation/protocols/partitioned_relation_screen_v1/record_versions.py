#!/usr/bin/env python3
"""Add measured versions to the registry without altering historical entries."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/embedding')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(summary_name, report_name, card_name, designs):
    summary_path = ROOT / 'doc/results' / summary_name
    summary = json.loads(summary_path.read_text())
    report = 'code/encoders/raw_patch_bottleneck_mae/' + report_name
    card = 'code/evaluation/cards/' + card_name
    records = []
    for tag, description, architecture, count in designs:
        stage = 'all_known' if any(r['label'] == tag for r in summary['stages'].get('all_known', {}).get('training', [])) else 'screen'
        panel = summary['stages'][stage]
        trained = next(r for r in panel['training'] if r['label'] == tag)
        status = ('measured_known_TEMPO3_architecture_prior_not_promoted' if summary['advancement'][tag]['passed']
                  else 'measured_failed_two_cohort_screen_stopped')
        sources = [x for x in summary['inputs'] if any(c['model_tag'] == tag for c in
                   json.loads((Path(x['run']) / 'report.json').read_text())['cohorts'])]
        records.append({'tag': tag, 'encoder': 'raw_patch_bottleneck_mae', 'description': description,
            'status': status, 'architecture_id': architecture, 'native_global_width': 32,
            'parameter_count_at_C3_H32_F3': count, 'common_initial_parameter_values': 225805,
            'additional_parameter_values': count - 225805, 'encoder_updates_each': 512,
            'training_policy_id': 'rpb-training-context-deletion-015-v1',
            'protocol': summary['protocol'], 'dataset_codename': 'TEMPO-3',
            'dataset_recipe_id': 'structured-hard-timing-v1', 'designed_complexity_level': 4,
            'complexity_scale_max': 5, 'evidence_stage': stage, 'master_seeds': panel['masters'],
            'measured_record': report, 'measured_record_sha256': sha(ROOT / report),
            'durable_summary': str(summary_path.relative_to(ROOT)), 'durable_summary_sha256': sha(summary_path),
            'card': card, 'card_sha256': sha(ROOT / card), 'runs': sources,
            'trained_quality': {view: {key: value for key, value in next(r for r in rows if r['label'] == tag).items() if key != 'per_master'}
                                for view, rows in panel['quality'].items()},
            'untrained_quality': {view: {key: value for key, value in next(r for r in rows if r['label'] == tag).items() if key != 'per_master'}
                                  for view, rows in panel['untrained'].items()},
            'training': {key: value for key, value in trained.items() if key != 'per_master'}, 'screen_gate_passed': summary['advancement'][tag]['passed'],
            'CPU_encoder_execution': False, 'old_model_reruns': 0, 'baseline_head_refits': 0,
            'testing_accessed': False, 'stress_accessed': False, 'promotion': False,
            'existing_groups_replaced': False})
    return records


def main():
    path = ROOT / 'doc/embedding_versions.json'
    original = json.loads(path.read_text())
    updated = json.loads(json.dumps(original))
    records = record('architecture_screen_v1.json', 'TEMPORAL_ARCHITECTURE_SCREEN.md',
        'architecture_screen_v1.md', [
            ('RPB-v14', 'Early mixer with generic temporal relations',
             'early-raw-global-plus-common-support-temporal-relations-v1', 236173),
            ('RPB-v15', 'Early mixer with label-free dynamics objective',
             'early-native32-label-free-dynamics-objective-v1', 226381)])
    records += record('partitioned_relation_screen_v1.json', 'PARTITIONED_TEMPORAL_RELATION_SCREEN.md',
        'partitioned_relation_screen_v1.md', [
            ('RPB-v16', 'Dedicated shape and grouped odd timing coordinates',
             'early-native20-shape-plus12-grouped-time-odd-relations-v1', 226877)])
    records += record('fixed_prior_relation_screen_v1.json', 'FIXED_PRIOR_TEMPORAL_RELATION_SCREEN.md',
        'fixed_prior_relation_screen_v1.md', [
            ('RPB-v17', 'Learned shape with fixed generic odd timing relations',
             'early-native20-shape-plus12-fixed-grouped-time-odd-relations-v1', 226877)])
    records[-1]['trainable_parameter_values'] = 226445
    records[-1]['frozen_parameter_values'] = 432
    records[-1]['encoder_training_recipe_id'] = 'original-waveform-huber1;fixed-grouped-odd-prior432;shape-backbone-decoder-trainable-v1'
    tags = {r['tag'] for r in updated['versions']}
    if tags & {r['tag'] for r in records}:
        raise ValueError('new tags only; no historical overwrite')
    updated['versions'] += records
    updated['architecture_screen_diagnostic'] = {
        'measured_record': 'code/encoders/raw_patch_bottleneck_mae/TEMPORAL_ARCHITECTURE_SCREEN.md',
        'durable_summary': 'doc/results/architecture_screen_v1.json', 'candidates': ['RPB-v14', 'RPB-v15'],
        'quality_trajectories': 7, 'encoder_updates': 3584, 'native_exports': 42,
        'head_pipelines': 42, 'individual_heads': 84, 'query_writers': 14,
        'necessary_masked_forwards': 56, 'CPU_encoder_execution': False,
        'old_model_reruns': 0, 'baseline_head_refits': 0, 'promotion': False}
    updated['partitioned_relation_diagnostic'] = {
        'measured_record': 'code/encoders/raw_patch_bottleneck_mae/PARTITIONED_TEMPORAL_RELATION_SCREEN.md',
        'durable_summary': 'doc/results/partitioned_relation_screen_v1.json', 'candidate': 'RPB-v16',
        'quality_trajectories': 2,
        'encoder_updates': 1024, 'CPU_encoder_execution': False,
        'old_model_reruns': 0, 'baseline_head_refits': 0, 'promotion': False}
    updated['fixed_prior_relation_diagnostic'] = {
        'measured_record': 'code/encoders/raw_patch_bottleneck_mae/FIXED_PRIOR_TEMPORAL_RELATION_SCREEN.md',
        'durable_summary': 'doc/results/fixed_prior_relation_screen_v1.json', 'candidate': 'RPB-v17',
        'quality_trajectories': len(records[-1]['master_seeds']),
        'encoder_updates': 512 * len(records[-1]['master_seeds']), 'CPU_encoder_execution': False,
        'old_model_reruns': 0, 'baseline_head_refits': 0, 'promotion': False}
    for key, value in original.items():
        if key == 'versions':
            assert updated[key][:len(value)] == value
        else:
            assert updated[key] == value, key
    path.write_text(json.dumps(updated, indent=2) + '\n')
    print(json.dumps({'added_tags': [r['tag'] for r in records],
                      'prior_versions_unchanged': len(original['versions']),
                      'prior_instance_bundles_unchanged': len(original['instance_bundles']),
                      'registry_sha256': sha(path)}))


if __name__ == '__main__':
    main()

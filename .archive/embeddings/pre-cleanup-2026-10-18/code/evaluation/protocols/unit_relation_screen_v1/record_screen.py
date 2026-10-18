#!/usr/bin/env python3
"""Append measured v18 without changing historical versions or instance bundles."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/embedding')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    registry = ROOT / 'doc/embedding_versions.json'
    original = json.loads(registry.read_text())
    if 'RPB-v18' in {v['tag'] for v in original['versions']}:
        raise ValueError('v18 already registered; do not overwrite')
    summary_path = ROOT / 'doc/results/unit_relation_screen_v1.json'
    summary = json.loads(summary_path.read_text())
    panel = summary['stages'].get('all_known', summary['stages']['screen'])
    training, = panel['training']
    report = 'code/encoders/raw_patch_bottleneck_mae/UNIT_TEMPORAL_RELATION_SCREEN.md'
    card = 'code/evaluation/cards/unit_relation_screen_v1.md'
    updated = json.loads(json.dumps(original))
    updated['versions'].append({
        'tag': 'RPB-v18', 'encoder': 'raw_patch_bottleneck_mae',
        'description': 'Learned shape with fixed generic unit temporal relations',
        'status': 'measured_known_development_not_promoted' if summary['advancement']['RPB-v18']['passed']
                  else 'measured_failed_two_cohort_screen_stopped',
        'architecture_id': 'early-native20-shape-plus12-fixed-unit-grouped-time-odd-relations-v1',
        'native_global_width': 32, 'parameter_count_at_C3_H32_F3': 226877,
        'trainable_parameter_values': 226445, 'frozen_parameter_values': 432,
        'additional_parameter_values_versus_v17': 0, 'encoder_updates_each': 512,
        'protocol': summary['protocol'], 'dataset_codename': 'TEMPO-3',
        'dataset_recipe_id': 'structured-hard-timing-v1', 'designed_complexity_level': 4,
        'complexity_scale_max': 5, 'evidence_stage': 'all_known' if 'all_known' in summary['stages'] else 'screen',
        'master_seeds': panel['masters'], 'runs': summary['inputs'],
        'measured_record': report, 'measured_record_sha256': sha(ROOT / report),
        'durable_summary': str(summary_path.relative_to(ROOT)), 'durable_summary_sha256': sha(summary_path),
        'card': card, 'card_sha256': sha(ROOT / card),
        'trained_quality': {view: {k: v for k, v in next(r for r in rows if r['label'] == 'RPB-v18').items() if k != 'per_master'}
                            for view, rows in panel['quality'].items()},
        'untrained_quality': {view: {k: v for k, v in next(r for r in rows if r['label'] == 'RPB-v18').items() if k != 'per_master'}
                              for view, rows in panel['untrained'].items()},
        'training': {k: v for k, v in training.items() if k != 'per_master'},
        'screen_gate_passed': summary['advancement']['RPB-v18']['passed'],
        'CPU_encoder_execution': False, 'old_model_reruns': 0, 'baseline_head_refits': 0,
        'testing_accessed': False, 'stress_accessed': False, 'promotion': False,
        'existing_groups_replaced': False})
    updated['unit_relation_diagnostic'] = {
        'candidate': 'RPB-v18', 'measured_record': report,
        'durable_summary': str(summary_path.relative_to(ROOT)),
        'quality_trajectories': len(panel['masters']), 'encoder_updates': 512*len(panel['masters']),
        'CPU_encoder_execution': False, 'old_model_reruns': 0, 'baseline_head_refits': 0,
        'promotion': False}
    for key, value in original.items():
        assert (updated[key][:-1] == value if key == 'versions' else updated[key] == value), key
    registry.write_text(json.dumps(updated, indent=2) + '\n')
    print(json.dumps({'added_tag': 'RPB-v18', 'prior_versions_unchanged': len(original['versions']),
                      'prior_instance_bundles_unchanged': len(original['instance_bundles']),
                      'registry_sha256': sha(registry)}))


if __name__ == '__main__':
    main()

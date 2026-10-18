#!/usr/bin/env python3
"""Append one fresh instance bundle, preserving every historical registry entry."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/embedding')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / 'doc/embedding_versions.json'
    original = json.loads(path.read_text())
    updated = json.loads(json.dumps(original))
    tag = 'RPB-v17.alt-01'
    if tag in {b['id'] for b in original['instance_bundles']}:
        raise ValueError('fresh bundle already registered; do not overwrite')
    summary_path = ROOT / 'doc/results/fixed_prior_fresh_confirmation_v1.json'
    summary = json.loads(summary_path.read_text())
    report = 'code/encoders/raw_patch_bottleneck_mae/FIXED_PRIOR_FRESH_CONFIRMATION.md'
    card = 'code/evaluation/cards/fixed_prior_fresh_confirmation_v1.md'
    v17 = next(v for v in original['versions'] if v['tag'] == 'RPB-v17')
    updated['instance_bundles'].append({
        'id': tag, 'encoder_tag': 'RPB-v17',
        'description': 'Unchanged fixed-prior v17 on five fresh TEMPO-3 source cohorts',
        'status': 'fresh_source_confirmation_measured_not_promoted',
        'architecture_id': v17['architecture_id'], 'native_global_width': 32,
        'parameter_count': 226877, 'trainable_parameter_count': 226445, 'frozen_parameter_count': 432,
        'master_seeds': summary['masters'], 'encoder_updates_each': 512,
        'protocol': summary['protocol'], 'dataset_codename': 'TEMPO-3',
        'dataset_recipe_id': 'structured-hard-timing-v1', 'designed_complexity_level': 4,
        'complexity_scale_max': 5, 'capsule': summary['input']['run'].removeprefix('/embedding/'),
        'checkpoint_path_pattern': summary['input']['run'].removeprefix('/embedding/') + '/seed-{master}/point-512/checkpoint.pt',
        'measured_record': report, 'measured_record_sha256': sha(ROOT / report),
        'durable_summary': str(summary_path.relative_to(ROOT)), 'durable_summary_sha256': sha(summary_path),
        'card': card, 'card_sha256': sha(ROOT / card),
        'source_fingerprint': summary['input']['source_fingerprint'],
        'report_sha256': summary['input']['report_sha256'],
        'saved_check_sha256': summary['input']['saved_check_sha256'],
        'inputs_sha256': summary['input']['inputs_sha256'],
        'trained_quality': {view: {k: v for k, v in row.items() if k != 'per_master'}
                            for view, row in summary['quality'].items()},
        'untrained_quality': {view: {k: v for k, v in row.items() if k != 'per_master'}
                              for view, row in summary['untrained'].items()},
        'training': {k: v for k, v in summary['training'].items() if k != 'per_master'},
        'confirmation': {k: v['passed'] for k, v in summary['confirmation'].items() if isinstance(v, dict)},
        'CPU_encoder_execution': False, 'old_encoder_or_head_refits': 0, 'promotion': False,
        'existing_groups_replaced': False})
    updated['fixed_prior_fresh_confirmation'] = {
        'instance_bundle': tag, 'model_design': 'RPB-v17', 'measured_record': report,
        'durable_summary': str(summary_path.relative_to(ROOT)),
        'quality_trajectories': 5, 'encoder_updates': 2560,
        'CPU_encoder_execution': False, 'old_encoder_or_head_refits': 0, 'promotion': False}
    for key, value in original.items():
        assert (updated[key][:-1] == value if key == 'instance_bundles' else updated[key] == value), key
    path.write_text(json.dumps(updated, indent=2) + '\n')
    print(json.dumps({'added_instance_bundle': tag, 'prior_versions_unchanged': len(original['versions']),
                      'prior_instance_bundles_unchanged': len(original['instance_bundles']),
                      'registry_sha256': sha(path)}))


if __name__ == '__main__':
    main()

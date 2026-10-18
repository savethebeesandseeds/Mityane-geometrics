#!/usr/bin/env python3
"""Publish passed saved evidence; no archive decoding, search or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/embedding')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    capsule = args.admission
    freeze = json.loads((capsule/'freeze/freeze.json').read_text())
    producer = capsule/'information/report.json'
    report = json.loads(producer.read_text())
    saved = json.loads((capsule/'saved-check.json').read_text())
    if saved['status'] != 'passed' or saved['report_sha256'] != sha(producer):
        raise ValueError('passed saved evidence bound to original producer report required')
    if report['source_fingerprint'] != freeze['source_fingerprint']:
        raise ValueError('source identity changed')
    protocol = ROOT/'code/evaluation/protocols/two_component_information_v1'
    metadata = capsule/'METADATA_SOURCE'
    metadata.mkdir(exist_ok=False)
    metadata_files = [protocol/name for name in ('admit.sh','publish.py')]
    for path in metadata_files:
        destination = metadata/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    manifest = capsule/'metadata-sources.sha256'
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in metadata_files))
    inventory = capsule/'artifact-inventory.sha256'
    inventory.write_text(''.join(f'{sha(p)}  {p.relative_to(capsule).as_posix()}\n'
                                 for p in sorted(capsule.rglob('*')) if p.is_file() and p != inventory))
    cohorts = []
    for cohort in report['cohorts']:
        item = {k:v for k,v in cohort.items() if k != 'views'}
        item['views'] = [{k:v for k,v in view.items() if k != 'row_reasons'} for view in cohort['views']]
        cohorts.append(item)
    summary = {
        'protocol':report['protocol'],'data_recipe':report['data_recipe'],
        'evidence_stage':'data_only_engineering_information_admission',
        'engineering_masters':report['engineering_masters'],
        'training_rows_each':256,'training_source_groups_each':128,
        'validation_rows_each':128,'validation_source_groups_each':64,
        'classifier_accuracy_measured':False,'encoder_quality_measured':False,
        'source_fingerprint':freeze['source_fingerprint'],'card_sha256':freeze['card_sha256'],
        'sdk_proof_sha256':freeze['sdk_proof_sha256'], 'cohorts':cohorts,
        'information_gate_passed':report['status'] == 'passed',
        'saved_check':saved,'captured_evidence':{
            'admission':str(capsule),'producer_report_sha256':sha(producer),
            'saved_check_sha256':sha(capsule/'saved-check.json'),
            'metadata_sources_sha256':sha(manifest),'artifact_inventory_sha256':sha(inventory)},
        'encoder_calls':0,'optimizer_updates':0,'head_fits':0,'PCA_fits':0,
        'quality_generation':False,'testing_accessed':False,'old_artifacts_replaced':False}
    summary_path = ROOT/'doc/results/two_component_information_v1.json'
    with summary_path.open('x') as stream:
        stream.write(json.dumps(summary,indent=2)+'\n')
    lines = [
        '# Two-component information admission', '',
        'Data-only engineering, not encoder or classifier accuracy. Both new datasets',
        'use two coherent sinusoidal components, positive gains/offsets, low noise,',
        'natural missingness and three-tick channel gaps. The labels describe a',
        'physical component property; they are not derived from a summed margin.', '',
        'Two engineering masters,910901/910902, per task. Each has TRAIN256/128',
        'source pairs and VALIDATION128/64 pairs; an extra30% pair-shared deletion',
        'view preserves the validation rows. No quality seeds or TEST data generated.', '',
        'The fixed observed-only information rule estimates two frequencies and',
        'affine sinusoidal coefficients from legal cells. It receives no labels,',
        'source IDs or hidden frequencies. This fitting is not an encoder feature,',
        'head or training target. The supported accuracy and coverage below judge',
        'whether the declared challenge retains usable task information.', '']
    for dataset,description in [('TEMPO-4','timing · slow-component lead/lag in a two-rhythm mixture'),
                                ('AMP-2','component balance · relative slow/fast strength in a two-rhythm mixture')]:
        for view in ('validation-intact','validation-deleted'):
            views = [v for c in cohorts if c['dataset_id'] == dataset for v in c['views'] if v['view'] == view]
            valid,total,correct = (sum(v[k] for v in views) for k in ('valid','total','correct'))
            accuracy = 100*correct/valid if valid else 0
            coverage = 100*valid/total
            lines.extend([
                f'Dataset: **{dataset}** · {description} · **complexity5/5** · {view}.', '',
                '| Method | Information accuracy % | Coverage % | Supported / total |',
                '| --- | ---: | ---: | ---: |',
                f'| Observed-only information check | {accuracy:.2f} | {coverage:.2f} | {valid} / {total} |',''])
    lines.extend([
        'All12 task/cohort/view gates '+('passed.' if summary['information_gate_passed'] else 'failed jointly.'),
        'TRAIN/intact gates require>=99% supported accuracy and>=99% coverage;',
        'deleted gates require>=98% accuracy and>=95% coverage on every cohort/task.',
        'No rows or failures are removed from the total scoring population.', '',
        'Generator invariants passed116,963 checks and40 malformed-input cases.',
        'The information primitive passed54 timing and54 component-balance affine',
        'noiseless fixtures, legal-mask and abstention cases. Actual full noisy',
        'engineering uses four generator calls and2,048 information rows.',
        f'Saved CPU verification passed{saved["checks"]:,} checks/{saved["CPU_archives_decoded"]} archives;',
        'it replayed selected-fit normal equations/residuals/phase or ratio margins',
        'and exact deletion streams, without rerunning frequency search or generation.', '',
        'Evidence is in `'+str(capsule.relative_to(ROOT))+'`.',
        'The [durable summary](results/two_component_information_v1.json) retains',
        'the card/source/SDK pins, per-cohort counts, costs and artifact inventory.',
        'The [prospective card](../code/evaluation/cards/two_component_information_v1.md)',
        'and [protocol](../code/evaluation/protocols/two_component_information_v1/README.md)',
        'retain the exact pre-outcome rules. Compact source/results are versioned;',
        'the full CPU archives remain in the local ignored capsule.', '',
        'This supports a separate small CUDA architecture screen. It does not',
        'establish encoder quality on the new datasets, universal observability',
        'outside this family, or production promotion. Keep TEMPO-3 and RPB-v18',
        'milestones intact; compare new dataset instances under unchanged fixed heads.', ''])
    report_path = ROOT/'doc/TWO_COMPONENT_INFORMATION_V1.md'
    with report_path.open('x') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({'summary_sha256':sha(summary_path),'report_sha256':sha(report_path),
                      'inventory_sha256':sha(inventory),'information_gate_passed':summary['information_gate_passed']}))


if __name__ == '__main__':
    main()

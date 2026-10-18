#!/usr/bin/env python3
"""Immutable experiment receipts; no tensor runtime, models or fitted heads."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

PROTOCOL = 'configurable-content-relation-v1'
ROOT = Path('/embedding')
RUN_ROOT = ROOT / 'output/runs/rpb-v22-content-relation'
BUILD = Path('/opt/cuwacunu_embedding/build/rpb-v22-content-relation')
ORIGINAL = ROOT / 'output/runs/rpb-v20-capacity/admission-0gN4uD'
BRANCH = ROOT / 'output/runs/rpb-v20-branches/diagnostic-YA3Nws'
PREVIOUS = ROOT / 'output/runs/rpb-v21-learned-relation/admission-0PGNmn'
ORIGINAL_MANIFEST = '62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83'
BRANCH_MANIFEST = 'cefa413c560bdf2ab1eee5be2e3f34feca6e2f723a99627f9eeae683c4c11d93'
PREVIOUS_MANIFEST = '6b138475bbeac3bde35cc45f331f21f5e10cff64b5971fb883f60121c20562bf'
ORIGINAL_SOURCE = '8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1'
ORIGINAL_INPUTS = '6121773c0b97e5490f191a9f8ff013e90f27a9a557376ac8256bdbb2d6af6091'
BINARIES = ('embedding_configurable_content_relation', 'embedding_configurable_content_relation_readout',
            'code/tests/encoders/raw_patch_bottleneck_mae/content_ordered_relation/content_ordered_relation_test')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def direct(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
          'direct absolute path: ' + str(path))


def capsule_path(root):
    direct(root)
    check(root.parent == RUN_ROOT and root.name.startswith('admission-') and len(root.name) > len('admission-'),
          'new candidate capsule stays in its declared experiment namespace')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def document(path):
    direct(path)
    check(path.is_file() and path.stat().st_nlink == 1, 'single-link regular receipt')
    return json.loads(path.read_text())


def new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_historical():
    for root, expected, protocol in ((ORIGINAL, ORIGINAL_MANIFEST, 'configurable-capacity-v1'),
                                     (BRANCH, BRANCH_MANIFEST, 'configurable-branch-v1'),
                                     (PREVIOUS, PREVIOUS_MANIFEST, 'configurable-learned-relation-v1')):
        direct(root)
        check(sha(root / 'artifacts.sha256') == expected, 'pinned historical manifest')
        done = document(root / 'complete.json')
        check(done['status'] == 'complete' and done['protocol'] == protocol
              and done['artifacts_manifest_sha256'] == expected, 'historical completion receipt')
    check(document(ORIGINAL / 'freeze/receipt.json')['source_id'] == ORIGINAL_SOURCE
          and sha(ORIGINAL / 'freeze/inputs.sha256') == ORIGINAL_INPUTS,
          'original source and exact nine quality role inputs')


def freeze(root, build):
    capsule_path(root)
    direct(build)
    check(build == BUILD, 'declared isolated build session')
    verify_historical()
    gate_reference = ROOT / 'code/embeddings/evaluation/protocols/configurable_learned_relation_v1/gate.py'
    gate_source = ROOT / 'code/embeddings/evaluation/protocols/configurable_content_relation_v1/gate.py'
    check(sha(gate_reference) == sha(gate_source), 'v22 uses exactly the unchanged 54-condition v21 gate')
    out = root / 'freeze'
    out.mkdir()
    source_list = subprocess.check_output(
        ['make', '-s', 'print-configurable-content-relation-sources', f'BUILD_DIR={build}'], text=True)
    paths = source_list.splitlines()
    check(paths and paths == sorted(set(paths)), 'closed sorted source list')
    expected = {}
    for name in paths:
        path = Path(name)
        check(not path.is_absolute() and '..' not in path.parts and path.is_file()
              and path.resolve(strict=True) == ROOT / path and not path.is_symlink(), 'direct source file')
        expected[name] = sha(path)
    manifest = ''.join(f'{expected[name]}  {name}\n' for name in paths)
    new(out / 'source-list.txt', source_list)
    new(out / 'sources.sha256', manifest)
    source_id = sha(out / 'sources.sha256')
    for binary in BINARIES[:2]:
        check(subprocess.check_output([str(build / binary), '--source-id'], text=True).strip() == source_id,
              'compiled producer/reader matches captured source')
    for name in paths:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        check(sha(target) == expected[name] == sha(Path(name)), 'source changed during capture')
    for name, source in (('card.md', 'code/embeddings/evaluation/cards/configurable_content_relation_v1.md'),
                         ('config.conf', 'code/embeddings/encoders/raw_patch_bottleneck_mae/config/content_ordered_relation.conf')):
        shutil.copyfile(source, out / name)
        check(sha(out / name) == expected[source], 'frozen card/settings exact source bytes')
    new(out / 'binaries.sha256', ''.join(f'{sha(build / name)}  {build / name}\n' for name in BINARIES))
    shutil.copyfile(ORIGINAL / 'freeze/inputs.sha256', out / 'inputs.sha256')
    shutil.copyfile(ORIGINAL / 'artifacts.sha256', out / 'original-artifacts.sha256')
    shutil.copyfile(BRANCH / 'artifacts.sha256', out / 'branch-artifacts.sha256')
    shutil.copyfile(PREVIOUS / 'artifacts.sha256', out / 'v21-artifacts.sha256')
    receipt = dict(protocol=PROTOCOL, source_fingerprint=source_id,
                   card_sha256=sha(out / 'card.md'), config_sha256=sha(out / 'config.conf'),
                   input_manifest_sha256=ORIGINAL_INPUTS, original_capsule=str(ORIGINAL),
                   original_source_fingerprint=ORIGINAL_SOURCE,
                   original_artifacts_manifest_sha256=ORIGINAL_MANIFEST,
                   branch_capsule=str(BRANCH), branch_artifacts_manifest_sha256=BRANCH_MANIFEST,
                   preserved_v21_capsule=str(PREVIOUS), preserved_v21_artifacts_manifest_sha256=PREVIOUS_MANIFEST,
                   unchanged_gate_sha256=sha(gate_source),
                   source_files=len(paths), build_directory=str(build),
                   git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    new(out / 'receipt.json', receipt)
    print(json.dumps(receipt))


def verify_frozen(root):
    capsule_path(root)
    receipt = document(root / 'freeze/receipt.json')
    check(receipt['protocol'] == PROTOCOL
          and sha(root / 'freeze/sources.sha256') == receipt['source_fingerprint']
          and sha(root / 'freeze/card.md') == receipt['card_sha256']
          and sha(root / 'freeze/config.conf') == receipt['config_sha256']
          and sha(root / 'freeze/inputs.sha256') == receipt['input_manifest_sha256'] == ORIGINAL_INPUTS,
          'frozen source/card/settings/input binding')
    check(receipt['build_directory'] == str(BUILD), 'frozen named build session')
    check(sha(root / 'freeze/v21-artifacts.sha256') == receipt['preserved_v21_artifacts_manifest_sha256'] == PREVIOUS_MANIFEST,
          'preserved v21 artifact inventory remains exact')
    check(sha(ROOT / 'code/embeddings/evaluation/protocols/configurable_content_relation_v1/gate.py')
          == sha(ROOT / 'code/embeddings/evaluation/protocols/configurable_learned_relation_v1/gate.py')
          == receipt['unchanged_gate_sha256'], 'prospective gate remains exactly unchanged')
    sources = (root / 'freeze/sources.sha256').read_text().splitlines()
    names = [line.split('  ', 1)[1] for line in sources]
    check(names == sorted(set(names)) and len(names) == receipt['source_files']
          and names == (root / 'freeze/source-list.txt').read_text().splitlines(), 'closed captured source population')
    for line in sources:
        digest, name = line.split('  ', 1)
        path = Path(name)
        check(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name, 'canonical source role')
        for file in (ROOT / path, root / 'freeze/source' / path):
            direct(file)
            check(file.is_file() and file.stat().st_nlink == 1 and sha(file) == digest, 'actual live and captured source bytes')
    binaries = (root / 'freeze/binaries.sha256').read_text().splitlines()
    check([line.split('  ', 1)[1] for line in binaries] == [str(BUILD / name) for name in BINARIES], 'exact compiled binary roles')
    for line in binaries:
        digest, name = line.split('  ', 1)
        file = Path(name)
        direct(file)
        check(file.is_file() and file.stat().st_nlink == 1 and sha(file) == digest, 'actual compiled binary bytes')
    return receipt


def verify_engineering(root):
    receipt = verify_frozen(root)
    admission = document(root / 'admission.json')
    check(admission == dict(protocol=PROTOCOL, status='admitted', quality_encoder_runs=0,
                            source_fingerprint=receipt['source_fingerprint'],
                            card_sha256=receipt['card_sha256']), 'actual engineering admission')
    runner = document(root / 'runner-engineering/complete.json')
    check(runner == dict(status='passed', protocol=PROTOCOL, master=941000,
                         profiles=['A'], updates_each=2, CPU_encoder_calls=0), 'actual two-update CUDA admission')
    reader = document(root / 'readout-engineering/complete.json')
    check(reader == dict(protocol=PROTOCOL + '-engineering', status='passed', fitted_pairs=6, encoder_calls=0),
          'artificial saved-head engineering scope')
    progress = document(root / 'runner-engineering/A/progress.json')
    check(progress['protocol'] == PROTOCOL and progress['master'] == 941000
          and progress['source_fingerprint'] == receipt['source_fingerprint']
          and progress['card_sha'] == receipt['card_sha256']
          and progress['config_sha'] == receipt['config_sha256']
          and progress['attempted'] == progress['completed'] == 2 and progress['skipped_attempts'] == 0
          and progress['CPU_encoder_calls'] == 0 and progress['training_device'] == 'cuda:0'
          and progress['parameter_count'] == progress['cuda_parameter_count'] and progress['weights_changed']
          and progress['relation_gradient_updates'] > 0
          and math.isfinite(progress['relation_weight_max_absolute_change'])
          and progress['relation_weight_max_absolute_change'] > 0,
          'actual two-update CUDA trace trains the new projection under frozen source/card/settings')
    check(len(progress['losses']) == 2
          and sum(row['relation_gradient_norm'] > 0 for row in progress['losses']) == progress['relation_gradient_updates']
          and all(math.isfinite(row['relation_gradient_norm']) and row['relation_gradient_norm'] >= 0
                  and math.isfinite(row['total_loss']) and row['gradient_norm'] > 0 for row in progress['losses']),
          'actual finite artificial CUDA gradient and loss witnesses')
    return receipt


def admission_fields(root, receipt):
    return dict(protocol=PROTOCOL, status='admitted', source_fingerprint=receipt['source_fingerprint'],
                card_sha256=receipt['card_sha256'], config_sha256=receipt['config_sha256'],
                input_manifest_sha256=ORIGINAL_INPUTS,
                runner_complete_sha256=sha(root / 'runner-engineering/complete.json'),
                runner_progress_sha256=sha(root / 'runner-engineering/A/progress.json'),
                runner_checkpoint_sha256=sha(root / 'runner-engineering/A/checkpoint.pt'),
                readout_complete_sha256=sha(root / 'readout-engineering/complete.json'),
                binaries_manifest_sha256=sha(root / 'freeze/binaries.sha256'),
                producer_binary_sha256=sha(BUILD / BINARIES[0]))


def bind_quality(root):
    receipt = verify_engineering(root)
    new(root / 'quality-admission.json', admission_fields(root, receipt))
    print('CONTENT_RELATION_QUALITY_ADMITTED', root)


def verify_admission(root):
    receipt = verify_engineering(root)
    check(document(root / 'quality-admission.json') == admission_fields(root, receipt),
          'immutable quality admission binds actual CUDA progress/checkpoint and exact binary')
    return receipt


def verify_consumed(path):
    direct(path)
    lines = path.read_text().splitlines()
    check(bool(lines), 'complete consumed-artifact ledger')
    seen = set()
    for line in lines:
        digest, name = line.split('  ', 1)
        file = Path(name)
        direct(file)
        check(len(digest) == 64 and file not in seen and file.is_file()
              and file.stat().st_nlink == 1 and sha(file) == digest, 'consumed saved evidence remains exact')
        seen.add(file)


def verify_stage(root, stage):
    receipt = verify_admission(root)
    masters = [941101, 941102] if stage == 'screen' else [941103]
    readout = root / stage / 'readouts'
    done = document(readout / 'complete.json')
    report = document(readout / 'report.json')
    summary = document(root / stage / 'published/summary.json')
    gate = document(root / stage / 'published/gate.json')
    check(done['protocol'] == PROTOCOL and done['status'] == 'complete'
          and done['saved_arithmetic_fits_replayed'] == done['head_fitted_pairs'] == 6 * len(masters)
          and done['report_sha256'] == sha(readout / 'report.json')
          and done['arithmetic_audit_sha256'] == sha(readout / 'arithmetic-audit.json'), 'completed readout artifact/counter binding')
    check(report['masters'] == summary['masters'] == masters
          and report['source_fingerprint'] == summary['source_fingerprint'] == receipt['source_fingerprint']
          and report['card_sha'] == summary['card_sha256'] == receipt['card_sha256']
          and report['head_fitted_pairs'] == summary['new_head_fit_pairs'] == 6 * len(masters)
          and summary['new_classifiers'] == 12 * len(masters)
          and summary['encoder_trajectories'] == len(masters)
          and summary['encoder_updates'] == 512 * len(masters)
          and summary['CPU_encoder_calls'] == summary['publisher_encoder_calls'] == summary['publisher_head_fits'] == 0,
          'independently audited stage scope/source/card')
    check(gate == summary['gate'] and gate['masters'] == masters
          and gate['passed'] == all(c['passed'] for c in gate['conditions']), 'prospective gate is exactly the audited numerical decision')
    verify_consumed(root / stage / 'published/consumed-files.sha256')
    for master in masters:
        candidate = root / 'candidates' / f'master-{master}'
        complete = document(candidate / 'complete.json')
        check(complete['protocol'] == PROTOCOL and complete['status'] == 'complete'
              and complete['master'] == master and complete['updates'] == 512
              and complete['source_fingerprint'] == receipt['source_fingerprint']
              and complete['card_sha'] == receipt['card_sha256']
              and complete['config_sha'] == receipt['config_sha256']
              and complete['input_manifest_sha'] == ORIGINAL_INPUTS
              and complete['report_sha256'] == sha(candidate / 'report.json'), 'completed candidate trajectory receipt')
        for point in (0, 512):
            for name in ('checkpoint.pt', 'features.pt', 'optimizer-witness.pt', 'scaler-witness.pt',
                         'reconstruction-training.pt', 'reconstruction-validation.pt', 'query-report.json', 'progress.json'):
                file = candidate / f'point-{point}' / name
                direct(file)
                check(file.is_file() and file.stat().st_nlink == 1, 'single-link retained quality evidence')
    return receipt, summary


def admit_third(root):
    receipt, summary = verify_stage(root, 'screen')
    check(summary['gate']['passed'] is True and not (root / 'candidates/master-941103').exists()
          and not (root / 'third').exists(), 'joint screen pass before allocating third cohort')
    evidence = dict(protocol=PROTOCOL, status='admitted', master=941103,
                    source_fingerprint=receipt['source_fingerprint'], card_sha256=receipt['card_sha256'],
                    screen_summary_sha256=sha(root / 'screen/published/summary.json'),
                    screen_gate_sha256=sha(root / 'screen/published/gate.json'),
                    consumed_manifest_sha256=sha(root / 'screen/published/consumed-files.sha256'))
    new(root / 'third-admission.json', evidence)
    print('CONTENT_RELATION_THIRD_ADMITTED', root)


def finish(root):
    check(not (root / 'artifacts.sha256').exists() and not (root / 'complete.json').exists(), 'new completion receipt')
    receipt, screen = verify_stage(root, 'screen')
    masters = [941101, 941102]
    gate_passed = screen['gate']['passed']
    if gate_passed:
        third_admission = document(root / 'third-admission.json')
        check(third_admission['source_fingerprint'] == receipt['source_fingerprint']
              and third_admission['card_sha256'] == receipt['card_sha256']
              and third_admission['screen_summary_sha256'] == sha(root / 'screen/published/summary.json')
              and third_admission['screen_gate_sha256'] == sha(root / 'screen/published/gate.json')
              and third_admission['consumed_manifest_sha256'] == sha(root / 'screen/published/consumed-files.sha256'),
              'third allocation preserves exact audited screen')
        _, third = verify_stage(root, 'third')
        masters.append(941103)
        disposition = 'third-development-pass' if third['gate']['passed'] else 'third-development-fail'
    else:
        check(not (root / 'third-admission.json').exists() and not (root / 'third').exists()
              and not (root / 'candidates/master-941103').exists(), 'failed screen does not allocate third')
        disposition = 'screen-fail-stop'
    check(sorted(int(p.name.split('-')[1]) for p in (root / 'candidates').iterdir()) == masters,
          'exact allocated trajectory inventory')
    verify_historical()
    for name, count in (('original-after-quality.log', 815),
                        ('branch-after-quality.log', len((BRANCH / 'artifacts.sha256').read_text().splitlines())),
                        ('v21-after-quality.log', len((PREVIOUS / 'artifacts.sha256').read_text().splitlines())),
                        ('source-after-quality.log', receipt['source_files']), ('binary-after-quality.log', len(BINARIES))):
        lines = (root / name).read_text().splitlines()
        manifest = (ORIGINAL / 'artifacts.sha256' if name.startswith('original-') else
                    BRANCH / 'artifacts.sha256' if name.startswith('branch-') else
                    PREVIOUS / 'artifacts.sha256' if name.startswith('v21-') else
                    root / 'freeze/sources.sha256' if name.startswith('source-') else root / 'freeze/binaries.sha256')
        expected = [line.split('  ', 1)[1] + ': OK' for line in manifest.read_text().splitlines()]
        check(lines == expected and len(lines) == count, 'exact final checksum role population: ' + name)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    for file in files:
        direct(file)
        check(file.stat().st_nlink == 1, 'single-link retained capsule artifact')
    new(root / 'artifacts.sha256', ''.join(f'{sha(file)}  {file.relative_to(root)}\n' for file in files))
    new(root / 'complete.json', dict(protocol=PROTOCOL, status='complete', disposition=disposition,
        masters=masters, encoder_trajectories=len(masters), completed_updates_per_run=512,
        quality_head_fit_pairs=6 * len(masters), quality_classifiers=12 * len(masters),
        engineering_head_fit_pairs=6, CPU_encoder_calls=0, control_head_fits=0,
        source_fingerprint=receipt['source_fingerprint'], card_sha256=receipt['card_sha256'],
        original_artifacts_manifest_sha256=ORIGINAL_MANIFEST, branch_artifacts_manifest_sha256=BRANCH_MANIFEST,
        preserved_v21_artifacts_manifest_sha256=PREVIOUS_MANIFEST,
        artifacts_manifest_sha256=sha(root / 'artifacts.sha256')))
    print('CONTENT_RELATION_COMPLETE', root, disposition)


def main():
    check(Path('/.dockerenv').exists() and Path.cwd() == ROOT, 'managed embedding development container')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'bind-quality', 'verify-admission', 'admit-third', 'finish'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--build', type=Path)
    args = parser.parse_args()
    capsule_path(args.root)
    if args.mode == 'freeze':
        check(args.build is not None, 'freeze requires named build session')
        freeze(args.root, args.build)
    elif args.mode == 'bind-quality':
        bind_quality(args.root)
    elif args.mode == 'verify-admission':
        verify_admission(args.root)
    elif args.mode == 'admit-third':
        admit_third(args.root)
    else:
        finish(args.root)


if __name__ == '__main__':
    main()

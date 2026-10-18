#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Container-only, additive CPB-v1 build/proof and one frozen quality pass.

No encoder training, old encoder execution, deletion, or historical source copies.
Use build, prove, quality in that order, inspecting each admission between stages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/cuwacunu')
VERSION = Path('code/encoders/causal_patch_bottleneck_mae/versions/v1')
APP = Path('code/evaluation/causal_patch_bottleneck_mae_v1')
BUILD = Path('/opt/cuwacunu/build/causal-patch-v1')
CARD = 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_CARD.md'
ADDENDUM = 'doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_LAYOUT_ADDENDUM.md'
PREFLIGHT = 'doc/results/causal_patch_bottleneck_mae_v1/preflight.json'
PINS = {
    CARD: '3b059b84a60a7d193ffcb034480f54fa5aba532ed6cb6b541fa620c4ab584d5f',
    ADDENDUM: 'ae0a8adae317a7b2daee3f58a645c9000faab130cde17ba6662a66fae06047dd',
    PREFLIGHT: '858f0b1896f764feb9c67cdef6b49e9e1a757ef4ac0243a3e2d2ba76bbfcb7e0',
    'doc/results/causal_patch_bottleneck_mae_v1/portable-sync-before-build.json':
        '091d019b5287238350893f5804e65f0da424282620c0541a6b5c418ea29ccff0',
    'doc/results/causal_patch_bottleneck_mae_v1/portable-sync-before-build-02.json':
        '4f7045ab4cb8cd435160ddc5f70be8134374ecf17a3564e6bb2036b115594d1c',
    'output/runs/market-location-scale-decoder/run-MAmvTJ/checkpoints/inference-2048.pt':
        'c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb',
    '/opt/cuwacunu/build/market-study/market_fixed_regression':
        '5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14',
    'output/inputs/market-synthetic-transfer-v1/manifest.json':
        'fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb',
}
SOURCE_CHECKPOINT = Path('output/runs/market-location-scale-decoder/run-MAmvTJ/checkpoints/inference-2048.pt')


def require(test, message):
    if not test:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def write_new(path, data):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def fresh(parent, prefix):
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        leaf = parent / f'{prefix}-{n:03d}'
        try:
            leaf.mkdir()
            return leaf
        except FileExistsError:
            pass
    raise RuntimeError('No unused attempt name')


def run(command, log):
    start = time.monotonic()
    print('COMMAND', shlex.join(map(str, command)), flush=True)
    with Path(log).open('xb') as f:
        p = subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT)
    print('FINISHED', p.returncode, str(log), flush=True)
    require(p.returncode == 0, f'Command failed; preserved log {log}')
    return time.monotonic() - start


def pins():
    for path, expected in PINS.items():
        require(sha(path) == expected, f'Frozen pin mismatch: {path}')
    pre = json.loads(Path(PREFLIGHT).read_text())
    require(pre['status'] == 'passed' and pre['checkpoints_verified'] == 55, '55 checkpoint preflight')
    for item in pre['checkpoints']:
        require(sha(item['path']) == item['sha256'], f'Checkpoint changed: {item["path"]}')
    for item in pre['frozen_records']:
        require(sha(item['path']) == item['sha256'], f'Frozen evidence changed: {item["path"]}')
    for item in pre['portable_pairs']:
        require(sha(item['integration_path']) == item['sha256'], f'Historical reusable source changed: {item["integration_path"]}')
    sync = json.loads(Path('doc/results/causal_patch_bottleneck_mae_v1/portable-sync-before-build-02.json').read_text())
    require(sync['status'] == 'passed' and sync['pairs_verified'] == 82 and sync['all_byte_identical'], 'New copied portable closure')
    verify_bindings(sync['files'])
    manifest_root = Path('output/inputs/market-synthetic-transfer-v1')
    manifest = json.loads((manifest_root / 'manifest.json').read_text())
    require(len(manifest['members']) == 18, '18 original TEMPO5 members')
    for item in manifest['members']:
        path = manifest_root / item['path']
        require(path.resolve().is_relative_to(manifest_root.resolve()) and sha(path) == item['sha256'] and
                path.stat().st_size == item['bytes'], f'TEMPO5 original input changed: {path}')
    return [binding(path) for path in PINS]


def source_files():
    files = list(VERSION.rglob('*')) + list(APP.rglob('*'))
    files += list(Path('code/encoders/raw_patch_bottleneck_mae/versions').glob('*'))
    files += list(Path('code/encoders/causal_patch_bottleneck_mae/versions').glob('*'))
    files += [Path('code/scripts/preflight-causal-patch-bottleneck-mae.ps1')]
    files += [Path('code/scripts/verify-causal-encoder-portable-sync.ps1')]
    files += [Path('Makefile'), Path(CARD), Path(ADDENDUM), Path(__file__).relative_to(ROOT)]
    return sorted(set(p for p in files if p.is_file()))


def protected_files():
    # No portable historical tree rescan: its completed 20GB scan is pinned by preflight.
    files = []
    for base in ('output', 'doc/results', 'code/encoders/raw_patch_bottleneck_mae',
                 'data/mkt-hist-1',
                 'code/evaluation/market_synthetic_transfer_v1',
                 'code/evaluation/market_location_scale_synthetic_transfer_v1'):
        files += [p for p in Path(base).rglob('*') if p.is_file()]
    excluded = ('output/engineering/causal-patch-bottleneck-mae-v1/',
                'output/runs/causal-patch-bottleneck-mae-v1/')
    return sorted(set(p for p in files if not str(p).startswith(excluded)))


def verify_bindings(items):
    for item in items:
        path = Path(item['path'])
        require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                f'Bound file changed: {path}')


def dependency_sources(path):
    value = path.read_text().replace('\\\n', ' ')
    require(': ' in value, f'Unexpected dependency syntax: {path}')
    words = shlex.split(value.split(': ', 1)[1])
    result = set()
    for word in words:
        if word.endswith(':'):
            continue
        path = Path(word).resolve()
        if path.is_relative_to(ROOT):
            result.add(path.relative_to(ROOT))
    return sorted(result)


def build():
    leaf = fresh(Path('output/engineering/causal-patch-bottleneck-mae-v1'), 'attempt')
    declared = pins()
    files = [binding(p) for p in source_files()]
    write_new(leaf / 'source-manifest.json', files)
    # One compact physical closure of this attempt; no old research capsules copied.
    for item in files:
        target = leaf / 'source' / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(item['path']).read_bytes())
    write_new(leaf / 'preservation-before.json', [binding(p) for p in protected_files()])
    seconds = run(['bash', 'code/scripts/task.sh', 'causal-patch-v1', '-j2',
                   'build-cpb-v1-engineering', 'build-cpb-v1-extract', 'build-cpb-v1-read'], leaf / 'build.log')
    verify_bindings(files)
    dependencies = {}
    for stem, kind in [('causal_patch_bottleneck_mae_v1_test', 'core'),
                       ('causal_patch_bottleneck_mae_v1_extract', 'app'),
                       ('causal_patch_bottleneck_mae_v1_read', 'reader')]:
        d = BUILD / f'{stem}.d'
        items = dependency_sources(d)
        require(items, f'Missing first-party dependencies: {stem}')
        for p in items:
            allowed = str(p).startswith(str(VERSION) + '/')
            if kind == 'app':
                allowed |= str(p).startswith(str(APP) + '/') or str(p).startswith('code/data/include/')
            if kind == 'reader':
                allowed = str(p).startswith(str(APP) + '/') or str(p).startswith('code/evaluation/market_synthetic_transfer_v1/deps/')
            require(allowed, f'First-party dependency escapes {kind}: {p}')
        dependencies[kind] = [binding(p) for p in items]
    binaries = [binding(BUILD / name) for name in ('test_causal_patch_bottleneck_mae_v1',
                 'causal_patch_bottleneck_mae_v1_extract', 'causal_patch_bottleneck_mae_v1_read')]
    old_heads = [binding(Path('/opt/cuwacunu/build/market-location-scale-transfer') /
                 f'market_location_scale_synthetic_transfer_{stem}.o')
                 for stem in ('data', 'feature_harness', 'fixed_feature_readouts')]
    require(len(old_heads) == 3, 'Exactly three unchanged fixed-head objects')
    write_new(leaf / 'admission.json', dict(kind='causal_patch_bottleneck_mae_v1_build_admission_v1',
        status='built_pending_source_review_and_cuda_proof', source=files, pins=declared,
        dependencies=dependencies, binaries=binaries, reused_fixed_head_objects=old_heads,
        build_seconds=seconds, source_capsule=str(leaf / 'source'), encoder_updates=0))
    print('ENGINEERING=' + str(leaf), flush=True)


def load_admission(leaf):
    admission = json.loads((leaf / 'admission.json').read_text())
    verify_bindings(admission['source'])
    verify_bindings(admission['binaries'])
    verify_bindings(admission['reused_fixed_head_objects'])
    for deps in admission['dependencies'].values():
        verify_bindings(deps)
    pins()
    return admission


def prove(leaf):
    load_admission(leaf)
    review = json.loads((leaf / 'source-review.json').read_text())
    require(review['status'] == 'passed', 'Independent source/dependency review is required before GPU proof')
    seconds = run([BUILD / 'test_causal_patch_bottleneck_mae_v1', '--checkpoint', SOURCE_CHECKPOINT,
                   '--settings', VERSION / 'config/v1.conf', '--output', leaf / 'proof'], leaf / 'proof.log')
    proof = json.loads((leaf / 'proof/proof.json').read_text())
    require(proof['status'] == 'passed', 'Causality proof must pass')
    before = json.loads((leaf / 'preservation-before.json').read_text())
    verify_bindings(before)
    write_new(leaf / 'proof-stage.json', dict(status='passed', proof=binding(leaf / 'proof/proof.json'),
        causal_checkpoint=binding(leaf / 'proof/inference.pt'), seconds_including_io=seconds,
        protected_files=len(before), protected_bytes=sum(x['bytes'] for x in before)))
    print('PROOF=' + str(leaf / 'proof/proof.json'), flush=True)


def quality(leaf):
    admission = load_admission(leaf)
    stage = json.loads((leaf / 'proof-stage.json').read_text())
    require(stage['status'] == 'passed', 'Pre-quality proof required')
    verify_bindings([stage['proof'], stage['causal_checkpoint']])
    audit = json.loads((leaf / 'independent-audit.json').read_text())
    require(audit['status'] == 'passed', 'Independent causality/evidence audit required')
    # Never overwrite or silently repeat quality after a failed wrapper stage.
    leaf_claim = leaf / 'quality-claimed.json'
    runroot = fresh(Path('output/runs/causal-patch-bottleneck-mae-v1'), 'run')
    write_new(leaf_claim, dict(run_root=str(runroot), intent='one frozen full-API extraction and unchanged fixed heads'))
    seconds = {}
    seconds['extraction'] = run([BUILD / 'causal_patch_bottleneck_mae_v1_extract', '--checkpoint',
        stage['causal_checkpoint']['path'], '--settings', VERSION / 'config/v1.conf', '--output',
        runroot / 'features'], runroot / 'extraction.log')
    shared = Path('output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features')
    seconds['market_heads'] = run(['/opt/cuwacunu/build/market-study/market_fixed_regression',
        '--train-targets', shared / 'train-targets.pt', '--validation-targets', shared / 'validation-targets.pt',
        '--method', 'cpb_v1', runroot / 'features/market/cpb_v1-train.pt',
        runroot / 'features/market/cpb_v1-validation.pt', '--output-directory', runroot / 'market-readouts',
        '--card-sha256', PINS[CARD], '--input-manifest-sha256',
        '828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8', '--dataset-id', 'MKT-HIST-2'],
        runroot / 'market-heads.log')
    seconds['synthetic_heads'] = run([BUILD / 'causal_patch_bottleneck_mae_v1_read', '--features',
        runroot / 'features/synthetic', '--output', runroot / 'synthetic-readouts'], runroot / 'synthetic-heads.log')
    load_admission(leaf)
    before = json.loads((leaf / 'preservation-before.json').read_text())
    verify_bindings(before)
    receipt = dict(kind='causal_patch_bottleneck_mae_v1_verification_v1', status='passed',
        run_root=str(runroot), proof_path=stage['proof']['path'], proof_sha256=stage['proof']['sha256'],
        causal_checkpoint_path=stage['causal_checkpoint']['path'], causal_checkpoint_sha256=stage['causal_checkpoint']['sha256'],
        admission_path=str(leaf / 'admission.json'), admission_sha256=sha(leaf / 'admission.json'),
        independent_audit_path=str(leaf / 'independent-audit.json'), independent_audit_sha256=sha(leaf / 'independent-audit.json'),
        source_capsule=admission['source_capsule'], stage_seconds=seconds,
        preservation_after=dict(status='passed', verified_files=len(before), verified_bytes=sum(x['bytes'] for x in before),
            manifest_path=str(leaf / 'preservation-before.json'), manifest_sha256=sha(leaf / 'preservation-before.json')),
        source=binding(leaf / 'source-manifest.json'), encoder_updates=0, old_encoder_calls=0,
        new_scaler_fits=0, post_encoder_pca=False)
    write_new(runroot / 'verification.json', receipt)
    print('QUALITY_RUN=' + str(runroot), flush=True)


def main():
    require(Path('/.dockerenv').exists(), 'Only the existing managed GPU container is authorized')
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'prove', 'quality'])
    parser.add_argument('--engineering', type=Path)
    args = parser.parse_args()
    if args.stage == 'build':
        build()
    else:
        require(args.engineering is not None, '--engineering required')
        {'prove': prove, 'quality': quality}[args.stage](args.engineering)


if __name__ == '__main__':
    main()

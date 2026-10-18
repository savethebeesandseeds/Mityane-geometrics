#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Record this application's completed integration checks; never run a model."""
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path('/cuwacunu')
BUILD = Path('/opt/cuwacunu/build/cpb-v3-primary-integration')
EVIDENCE = Path('/opt/cuwacunu/research/cpb-v3-primary-integration/attempt-001')
OUTPUT = ROOT / 'doc/cpb_v3_primary_integration'
EXPECTED = '50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'

def pin(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}

assert sys.platform.startswith('linux') and Path('/.dockerenv').is_file()
checkpoint = ROOT / 'output/baselines/CPB-v3.market-01/inference.pt'
assert pin(checkpoint)['sha256'] == EXPECTED
assert not (ROOT / 'code/encoders/causal_patch_bottleneck_mae/versions/v1').exists()
assert not (ROOT / 'code/encoders/causal_patch_bottleneck_mae/versions/v2').exists()
source = {ROOT / 'Makefile', ROOT / 'code/scripts/task.sh',
          ROOT / 'config/encoder.conf', ROOT / 'config/encoder_runtime.conf',
          Path(__file__).resolve()}
dependency_files = sorted(BUILD.glob('*.d'))
assert dependency_files
for dependency in dependency_files:
    first = dependency.read_text().replace('\\\n', '').splitlines()[0]
    for raw in shlex.split(first.split(':', 1)[1]):
        path = Path(raw).resolve()
        if path.is_relative_to(ROOT):
            assert 'causal_patch_bottleneck_mae/versions/v1/' not in str(path)
            assert 'causal_patch_bottleneck_mae/versions/v2/' not in str(path)
            source.add(path)
source.update((ROOT / 'code/encoders/causal_patch_bottleneck_mae/versions/v3').rglob('*'))
source = sorted(path for path in source if path.is_file())
assert any('causal_patch_bottleneck_mae/versions/v3/' in str(path) for path in source)
cli_lines = (EVIDENCE / 'encode-history.json').read_text().splitlines()
cli = json.loads(next(line for line in reversed(cli_lines) if line.startswith('{')))
assert cli['instance'] == 'CPB-v3.market-01' and cli['native_width'] == 36
assert cli['valid'] == [True] and cli['cutoffs'] == [1706486399999]
assert len(cli['native']) == 1 and len(cli['native'][0]) == 36
assert all(math.isfinite(x) for x in cli['native'][0])
capture = (BUILD / 'capture-output/preparation.txt').read_text()
assert 'experiment SYNTHETIC-CPB-V3-CACHE88-CORRECTNESS\n' in capture
assert 'unique_cuda_captures 18\n' in capture
assert 'encoder_updates 0\n' in capture and 'scaler_fits 0\n' in capture
names = ['test_memory_mapped', 'test_data_safety', 'test_config',
         'test_causal_encoder_input', 'encode_history',
         'latent_market_v1_prepare', 'latent_market_v1_study', 'latent_market_v1_utility',
         'latent_market_v1_timeline_test', 'latent_dynamics_v1_baselines_test',
         'latent_dynamics_v1_risk_readouts_test', 'test_predictive_v1']
assert (EVIDENCE / 'build-test.exit-code').read_text().strip() == '0'
for original, name in [(EVIDENCE/'build-test.log', 'build-test.log'),
                       (EVIDENCE/'encode-history.json', 'encode-history.json'),
                       (EVIDENCE/'capture.log', 'capture.log'),
                       (BUILD/'capture-output/preparation.txt', 'capture-receipt.txt'),
                       (BUILD/'capture-fixture/fixture.json', 'capture-fixture.json'),
                       (BUILD/'capture-fixture/synthetic-capture-card.txt', 'capture-card.txt')]:
    shutil.copyfile(original, OUTPUT/name)
receipt = {
    'kind': 'cpb_v3_primary_completed_container_integration_v1',
    'instance': 'CPB-v3.market-01', 'checkpoint': pin(checkpoint),
    'source_updates': 0, 'causal_updates_in_selected_artifact': 2048,
    'new_encoder_updates': 0, 'new_scaler_fits': 0, 'new_market_head_fits': 0,
    'test_command': 'bash code/scripts/task.sh cpb-v3-primary-integration -j2 test build-latent-market-v1 test-latent-market-v1',
    'test_exit_code': 0, 'compiler_dependency_files': len(dependency_files),
    'managed_container': {
        'name': 'cuwacunu',
        'verified_id': '82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e',
        'host_project_toolchains': False
    },
    'source_closure': [pin(path) for path in source],
    'binaries': [pin(BUILD/name) for name in names],
    'cli_witness': cli, 'synthetic_downstream_cuda_captures': 18,
    'cli_raw_inputs': [pin(ROOT / f'data/mkt-hist-1/csv/{asset}USDT-1m-2024-01.cache.bin')
                       for asset in ['BTC', 'ETH', 'SOL']],
    'capture_products_local_only': [pin(BUILD/'capture-output'/name)
        for name in ['timeline.bin', 'pairs.bin', 'future-risk.bin']],
    'scope': 'Application integration and synthetic correctness; existing v3 research quality reused without rerunning heads.'
}
(OUTPUT/'engineering.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
print(f'Integration evidence recorded: {len(source)} source files, {len(names)} binaries.')

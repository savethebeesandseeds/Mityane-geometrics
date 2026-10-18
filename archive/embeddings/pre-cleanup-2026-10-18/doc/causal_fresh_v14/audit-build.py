# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
"""Owned complete V13 infrastructure SOURCE closure auditor: bind the prospective V14 compiler/binary closure."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

source, build, destination = (Path(value).resolve() for value in sys.argv[1:])
manifest = json.loads((source / 'CAPSULE_MANIFEST.json').read_text(encoding='utf-8-sig'))
expected = {item['path']: item for item in manifest['files']}
if (manifest['card_sha256'] != expected['doc/CAUSAL_FRESH_V14_CARD.json']['sha256']
        or manifest['gradient_policy'] != 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
        or manifest['temporal_pooling_policy'] != 'original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1'
        or manifest['temporal_pooling_geometry'] != [4,16,32,8]
        or manifest['temporal_pooling_trailing_lengths'] != [4,8,12,16]
        or 'temporal_pooling_partition' in manifest
        or manifest['temporal_pooling_mix'] != [0.5,0.5]
        or manifest['empty_local_bin_policy'] != 'zero-whole-export-slice-after-bias-v1'
        or manifest['temporal_moment_policy'] != 'componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1'
        or manifest['temporal_export_layout'] != 'per-trailing-window-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1'
        or manifest['temporal_moment_partition'] != [4,4]
        or manifest['temporal_dispersion_transform_policy'] != 'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1'
        or manifest['temporal_dispersion_regularizer'] != 0.0009765625):
    raise RuntimeError('Unexpected frozen V14 source declaration')


def pin(path):
    value = path.read_bytes()
    return {'path': str(path), 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


for relative, item in expected.items():
    actual = pin(source / relative)
    if actual['bytes'] != item['bytes'] or actual['sha256'] != item['sha256']:
        raise RuntimeError(f'Admitted source changed: {relative}')
closures = {}
for dependency_file in sorted(build.glob('*.d')):
    lines = []
    for line in dependency_file.read_text().splitlines():
        lines.append(line)
        if not line.endswith('\\'):
            break
    dependencies = shlex.split('\n'.join(lines).split(':', 1)[1].replace('\\\n', ' '))
    project = set()
    for dependency in dependencies:
        path = Path(dependency)
        resolved = path.resolve() if path.is_absolute() else (source / path).resolve()
        try:
            relative = resolved.relative_to(source).as_posix()
        except ValueError:
            if str(resolved).startswith(('/cuwacunu/', '/embedding/', '/opt/cuwacunu/research/')) or '/code/' in str(resolved):
                raise RuntimeError(f'External project dependency: {resolved}')
            continue
        if relative not in expected:
            raise RuntimeError(f'Unadmitted compiler dependency: {relative}')
        if relative.startswith('code/encoders/') and not relative.startswith(
                'code/encoders/causal_patch_bottleneck_mae/versions/v14/'):
            raise RuntimeError(f'Other encoder in build closure: {relative}')
        project.add(relative)
    if not project:
        raise RuntimeError(f'Empty first-party dependency closure: {dependency_file}')
    closures[dependency_file.name] = {'file': pin(dependency_file), 'owned_paths': sorted(project)}
card = json.loads((source / 'doc/CAUSAL_FRESH_V14_CARD.json').read_text(encoding='utf-8-sig'))
if (card['source_counts']['translation_units'] != 11
        or card['source_counts']['binaries'] != 7
        or len(closures) != card['source_counts']['translation_units']):
    raise RuntimeError('Actual NEW source-reviewed eleven-TU/seven-binary closure required')
payload = {
    'kind': 'causal_fresh_v14_build_admission_v1', 'status': 'passed',
    'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'card_sha256': manifest['card_sha256'], 'gradient_policy': manifest['gradient_policy'], 'source_files': len(expected),
    'source_manifest': pin(source / 'CAPSULE_MANIFEST.json'),
    'dependency_closures': closures,
    'binaries': [pin(build / name) for name in (
        'test_causal_patch_bottleneck_mae_v14', 'test_cpb_v14_variance_gradient', 'test_cpb_v14_temporal_pooling', 'causal_fresh_v14_train',
        'causal_fresh_v14_extract', 'causal_fresh_v14_market_read',
        'causal_fresh_v14_synthetic_read')],
    'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
    'source_stage': pin(source.parent / 'source-stage.json'),
}
with destination.open('x') as stream:
    json.dump(payload, stream, indent=2)
    stream.write('\n')
print(json.dumps({'status': 'passed', 'source_files': len(expected),
                  'translation_units': len(closures), 'binaries': len(payload['binaries'])}))



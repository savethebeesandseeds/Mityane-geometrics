# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
"""Verify a frozen research build closure and bind its five binaries."""
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
            if str(resolved).startswith(('/cuwacunu/', '/embedding/')):
                raise RuntimeError(f'External project dependency: {resolved}')
            continue
        if relative not in expected:
            raise RuntimeError(f'Unadmitted compiler dependency: {relative}')
        if relative.startswith('code/encoders/') and not relative.startswith(
                'code/encoders/causal_patch_bottleneck_mae/versions/v3/'):
            raise RuntimeError(f'Other encoder in build closure: {relative}')
        project.add(relative)
    if not project:
        raise RuntimeError(f'Empty first-party dependency closure: {dependency_file}')
    closures[dependency_file.name] = {'file': pin(dependency_file), 'owned_paths': sorted(project)}
if len(closures) != 9:
    raise RuntimeError(f'Expected nine translation units, found {len(closures)}')
payload = {
    'kind': 'causal_fresh_v3_build_admission_v1', 'status': 'passed',
    'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'card_sha256': manifest['card_sha256'], 'source_files': len(expected),
    'source_manifest': pin(source / 'CAPSULE_MANIFEST.json'),
    'dependency_closures': closures,
    'binaries': [pin(build / name) for name in (
        'test_causal_patch_bottleneck_mae_v3', 'causal_fresh_v3_train',
        'causal_fresh_v3_extract', 'causal_fresh_v3_market_read',
        'causal_fresh_v3_synthetic_read')],
    'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
    'source_stage': pin(source.parent / 'source-stage.json'),
}
with destination.open('x') as stream:
    json.dump(payload, stream, indent=2)
    stream.write('\n')
print(json.dumps({'status': 'passed', 'source_files': len(expected),
                  'translation_units': len(closures), 'binaries': len(payload['binaries'])}))

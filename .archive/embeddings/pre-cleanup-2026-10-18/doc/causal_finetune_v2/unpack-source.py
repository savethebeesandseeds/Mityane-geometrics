# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
"""Container-side exclusive extraction and exact SHA audit of a source capsule."""
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

archive, destination = map(Path, sys.argv[1:])
with zipfile.ZipFile(archive) as capsule:
    manifest = json.loads(capsule.read('CAPSULE_MANIFEST.json').decode('utf-8-sig'))
    expected = {entry['path']: entry for entry in manifest['files']}
    names = capsule.namelist()
    if len(names) != len(set(names)) or set(names) != set(expected) | {'CAPSULE_MANIFEST.json'}:
        raise RuntimeError('Unexpected or duplicate capsule members')
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise RuntimeError(f'Unsafe capsule path: {name}')
        content = capsule.read(name)
        if name in expected:
            entry = expected[name]
            if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise RuntimeError(f'Capsule SHA mismatch: {name}')
    destination.mkdir(parents=False, exist_ok=False)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(capsule.read(name))
    for name, entry in expected.items():
        content = (destination / name).read_bytes()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Extracted SHA mismatch: {name}')
receipt = {
    'kind': 'causal_finetune_v2_container_source_stage_v1', 'status': 'passed',
    'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_files': len(expected), 'source_root': str(destination),
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'manifest_sha256': hashlib.sha256((destination / 'CAPSULE_MANIFEST.json').read_bytes()).hexdigest(),
    'card_sha256': manifest['card_sha256'],
}
with (destination.parent / 'source-stage.json').open('x') as output:
    json.dump(receipt, output, indent=2)
    output.write('\n')
print(json.dumps(receipt))

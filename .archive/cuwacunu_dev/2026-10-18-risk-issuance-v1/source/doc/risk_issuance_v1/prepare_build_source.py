#!/usr/bin/env python3
"""Verify and stage byte-identical inputs inside the existing container."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

root = Path('/cuwacunu')
assert Path('/.dockerenv').exists() and Path.cwd() == root
name = sys.argv[1]
assert re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}', name)
destination = Path('/opt/cuwacunu/build') / name / 'source'
assert not destination.exists(), 'Preserve the existing immutable source snapshot.'
archive = root / 'doc/risk_issuance_v1/build-source.zip'
manifest_path = root / 'doc/risk_issuance_v1/build-source-inputs.json'
digest = lambda data: hashlib.sha256(data).hexdigest()
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
assert manifest['schema'] == 'byte-identical-container-build-source-v1'
with zipfile.ZipFile(archive) as source:
    assert source.namelist() == [entry['path'] for entry in manifest['files']] + ['build-source-inputs.json']
    assert source.read('build-source-inputs.json') == manifest_bytes
    for entry in manifest['files']:
        relative = PurePosixPath(entry['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        data = source.read(entry['path'])
        assert digest(data) == entry['sha256'] and len(data) == entry['bytes']
        assert (root / relative).read_bytes() == data, 'Live source changed: ' + str(relative)
    destination.mkdir(parents=True)
    for entry in manifest['files']:
        path = destination / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read(entry['path']))
(destination / 'build-source-inputs.json').write_bytes(manifest_bytes)
receipt = {'schema': 'verified-container-build-source-snapshot-v1', 'sourceRoot': str(root),
           'buildSourceRoot': str(destination), 'files': manifest['files'],
           'manifestSha256': digest(manifest_bytes), 'archiveSha256': digest(archive.read_bytes())}
(root / 'doc/risk_issuance_v1/build-source-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Verified immutable container source snapshot:', destination, '; files:', len(manifest['files']))

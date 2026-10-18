#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Pin the exact risk-stage implementation closure before compiling in Linux."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

REGISTRY = Path('code/latent_market/risk_issuance/versions/v1/source-files.json')


def identity(root, files):
    fields = []
    if len(files) != len(set(files)) or not files:
        raise ValueError('Unique nonempty implementation paths are required')
    for name in sorted(files):
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise ValueError('Invalid implementation path')
        actual = (root / path).resolve(strict=True)
        actual.relative_to(root)
        fields.extend((name, hashlib.sha256(actual.read_bytes()).hexdigest()))
    digest = hashlib.sha256()
    for field in ['cuwacunu_artifact_record_v1', 'source_closure_v1', str(len(fields)), *fields]:
        raw = field.encode('utf-8')
        digest.update(struct.pack('<Q', len(raw)))
        digest.update(raw)
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--files', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    files = json.loads((root / REGISTRY).read_text())
    print(' '.join(sorted(files)) if args.files else identity(root, files))

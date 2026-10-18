# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Read only allowlisted tensor metadata; never storage values or model code."""
import hashlib
import json
from pathlib import Path
import sys
from archive_reader import Archive, Tensor

if not Path('/.dockerenv').is_file():
    raise RuntimeError('Metadata inspection belongs in the existing managed container')
card = json.loads(Path(sys.argv[1]).read_text())
out = {}
for binding in card['allowed_inputs']:
    if binding['role'] != 'TRAIN_archive':
        continue
    path = Path(binding['container_path'])
    assert path.stat().st_size == binding['bytes']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == binding['sha256']
    archive = Archive(path)
    try:
        fields = {}
        for name in binding['allowed_tensor_fields']:
            value = archive.field(*name.split('.'))
            assert isinstance(value, Tensor)
            fields[name] = {'shape': value.shape, 'dtype': value.storage.dtype}
        out[binding['id']] = fields
    finally:
        archive.close()
print(json.dumps({'kind': 'cpb_v3_v4_TRAIN_metadata_schema_v1', 'numeric_storage_reads': 0,
                  'encoder_calls': 0, 'head_fits': 0, 'fields': out}, indent=2))

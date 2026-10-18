# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Read only tensor names/types/shapes from declared saved TRAIN archives."""
import hashlib
import json
from pathlib import Path
import sys
from archive_reader import Archive, Record, Tensor

card = json.loads(Path(sys.argv[1]).read_text())
selected = {'initial_features', 'final_features', 'train_targets', 'initial_query',
            'final_query', 'final_checkpoint', 'final_fit_2701', 'final_head_2701'}
out = {}
for binding in card['allowed_inputs']:
    if binding['id'] not in selected:
        continue
    path = Path(binding['container_path'])
    assert path.stat().st_size == binding['bytes']
    assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == binding['sha256']
    archive = Archive(path)
    fields = {}
    def walk(value, prefix=''):
        if isinstance(value, Record):
            for name, child in value.fields.items():
                walk(child, prefix + '.' + name if prefix else name)
        elif isinstance(value, Tensor):
            if binding['id'] != 'final_checkpoint' or any(
                    key in prefix for key in ('change_decoder', 'variance_projection', 'variance_bias')):
                fields[prefix] = {'shape': value.shape, 'dtype': value.storage.dtype}
    walk(archive.root)
    out[binding['id']] = fields
    archive.close()
print(json.dumps(out, indent=2))

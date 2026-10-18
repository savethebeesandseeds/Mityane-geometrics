#!/usr/bin/env python3
"""Capture metadata tools and completed capsule inventory; no tensor execution."""
import argparse
import hashlib
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', required=True, type=Path)
    args = parser.parse_args()
    root = Path('/embedding')
    protocol = root / 'code/evaluation/protocols/unit_relation_screen_v1'
    tools = [protocol / name for name in ('admit.sh', 'run_screen.sh', 'publish_screen.py',
                                         'record_screen.py', 'capture_metadata.py')]
    target = args.admission / 'METADATA_SOURCE'
    target.mkdir(exist_ok=False)
    for path in tools:
        destination = target / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    with (args.admission / 'metadata-sources.sha256').open('x') as stream:
        stream.write(''.join(f'{sha(p)}  {p.relative_to(root).as_posix()}\n' for p in tools))
    inventory = args.admission / 'artifact-inventory.sha256'
    paths = sorted(p for p in args.admission.rglob('*') if p.is_file())
    with inventory.open('x') as stream:
        stream.write(''.join(f'{sha(p)}  {p.relative_to(args.admission).as_posix()}\n' for p in paths))
    print(f'Captured {len(tools)} metadata tools and {len(paths)} artifacts; inventory SHA {sha(inventory)}')


if __name__ == '__main__':
    main()

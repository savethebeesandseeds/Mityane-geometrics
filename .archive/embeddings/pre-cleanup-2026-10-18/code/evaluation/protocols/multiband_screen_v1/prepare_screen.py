#!/usr/bin/env python3
"""Freeze source before engineering; bind once-generated legal inputs afterwards."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat

ROOT = Path('/embedding')
SDK = Path('/opt/cuwacunu_embedding/setup/sdk-20261008T234205Z-6a74defe83e3/proof.json')
SDK_SHA = 'e370b91a50e264739c284c0fc5b461436b60a50735b36a2777750a6603b3a0bb'
INFO_SHA = '04fda53a38d8ead378dec5b276c62eba322ec5ea0361daab789a02ee3c03569f'
TASKS = ['slow_lag_sign', 'component_balance']
MASTERS = [920903, 920904]
RUN_ROOT = ROOT/'output/runs/rpb-multiband-screen'


def direct(path, regular=False):
    if not path.is_absolute() or path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError('canonical direct path required: '+str(path))
    info = path.stat()
    if regular and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise ValueError('distinct direct regular file required: '+str(path))
    return (info.st_dev, info.st_ino)


def admission(directory):
    direct(directory)
    if directory.parent != RUN_ROOT or not directory.name.startswith('admission-'):
        raise ValueError('direct admission leaf required')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(directory):
    for line in (directory/'sources.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha(ROOT/name) != digest or sha(directory/'SOURCE'/name) != digest:
            raise ValueError('frozen source changed: '+name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-list', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bind-input-root', type=Path)
    a = p.parse_args()
    admission(a.output.parent)
    if a.output.name != 'freeze':
        raise ValueError('freeze leaf required')
    if a.bind_input_root:
        if a.source_list:
            raise ValueError('bind existing freeze only')
        direct(a.output)
        verify_sources(a.output)
        direct(a.bind_input_root)
        if a.bind_input_root != a.output.parent/'quality-data':
            raise ValueError('same-admission quality-data leaf required')
        for path in (a.output/'engineering-freeze.json',a.bind_input_root/'data-report.json',a.bind_input_root/'data-complete.json'):
            direct(path, True)
        record = json.loads((a.output/'engineering-freeze.json').read_text())
        data = json.loads((a.bind_input_root/'data-report.json').read_text())
        complete = json.loads((a.bind_input_root/'data-complete.json').read_text())
        if (complete['status'] != 'complete' or complete['protocol'] != data['protocol'] or data['protocol'] != 'multiband-screen-v1'
                or data['data_recipe'] != 'two-component-v1'
                or complete['source_fingerprint'] != data['source_fingerprint'] or complete['human_card_sha256'] != data['human_card_sha256']
                or complete['masters'] != data['masters'] or complete['archive_count'] != data['archive_count']
                or complete['data_report_sha256'] != sha(a.bind_input_root/'data-report.json')
                or data['source_fingerprint'] != record['source_fingerprint'] or data['human_card_sha256'] != record['card_sha256']
                or data['masters'] != MASTERS or data['tasks'] != TASKS or data['generator_calls'] != 4
                or data['archive_count'] != 12 or data['encoder_calls'] != 0 or data['information_fit_calls'] != 0):
            raise ValueError('complete once-generated data with same source/card required')
        names = [f'results/seed-{m}-{t}/controlled-{v}.pt' for m in MASTERS for t in TASKS
                 for v in ('training', 'validation', 'validation-deleted')]
        if [f['path'] for f in data['files']] != names:
            raise ValueError('exact ordered twelve-file role matrix required')
        identities = set()
        for item in data['files']:
            path = a.bind_input_root/item['path']
            identity = direct(path, True)
            if identity in identities:
                raise ValueError('aliased role files')
            identities.add(identity)
        for item in data['files']:
            path = a.bind_input_root/item['path']
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('data writer bytes changed')
        manifest = ''.join(f'{sha(a.bind_input_root/name)}  {name}\n' for name in names)
        with (a.output/'inputs.sha256').open('x') as f:
            f.write(manifest)
        record.update(input_root=str(a.bind_input_root), inputs_sha256=sha(a.output/'inputs.sha256'),
                      input_files=12, data_report_sha256=sha(a.bind_input_root/'data-report.json'))
        with (a.output/'freeze.json').open('x') as f:
            f.write(json.dumps(record, indent=2)+'\n')
    else:
        if not a.source_list:
            raise ValueError('closed source list required')
        names = sorted(set(a.source_list.read_text().splitlines()))
        if not names or any(not n or Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('closed relative source list required')
        if sha(SDK) != SDK_SHA or sha(ROOT/'doc/results/two_component_information_v1.json') != INFO_SHA:
            raise ValueError('verified SDK/information admission changed')
        card = ROOT/'code/evaluation/cards/multiband_screen_v1.md'
        identities = [direct(ROOT/n, True) for n in names]
        if len(set(identities)) != len(names):
            raise ValueError('aliased source files')
        manifest = ''.join(f'{sha(ROOT/n)}  {n}\n' for n in names)
        a.output.mkdir(exist_ok=False)
        (a.output/'sources.sha256').write_text(manifest)
        for n in names:
            target = a.output/'SOURCE'/n
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/n, target)
        shutil.copyfile(card, a.output/'card.md')
        shutil.copyfile(SDK, a.output/'sdk-proof.json')
        record = dict(protocol='multiband-screen-v1', masters=MASTERS, tasks=TASKS,
                      candidates=['v18', 'v19'], source_files=len(names),
                      source_fingerprint=hashlib.sha256(manifest.encode()).hexdigest(),
                      card_sha256=sha(card), sdk_proof_sha256=SDK_SHA,
                      information_summary_sha256=INFO_SHA, model_execution=False,
                      encoder_training=False, head_fitting=False, old_artifacts_replaced=False)
        (a.output/'engineering-freeze.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    main()

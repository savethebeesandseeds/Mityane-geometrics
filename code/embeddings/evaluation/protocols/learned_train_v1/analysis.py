#!/usr/bin/env python3
"""TRAIN-only saved arithmetic: stdlib lazy tensors, no model or classifier execution."""
import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path('/mityane')
HERE = Path('code/embeddings/evaluation/protocols/learned_train_v1')
CARD = Path('code/embeddings/evaluation/cards/learned_train_v1.md')
CODEC = Path('.archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py')
CODEC_SHA = '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d'
ORIGINAL = ROOT / 'code/embeddings/output/runs/rpb-v20-capacity/admission-0gN4uD'
BRANCH = ROOT / 'code/embeddings/output/runs/rpb-v20-branches/diagnostic-YA3Nws'
BRANCH_MANIFEST = 'cefa413c560bdf2ab1eee5be2e3f34feca6e2f723a99627f9eeae683c4c11d93'
ORIGINAL_MANIFEST = '62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83'
ORIGINAL_SOURCE = '8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1'
ORIGINAL_CARD = '562e685c6e1283a7bc22bad7ddcf9fb138c75221cb3dbbd8cd576f8edcfe144e'
ORIGINAL_INPUT = '6121773c0b97e5490f191a9f8ff013e90f27a9a557376ac8256bdbb2d6af6091'
ORIGINAL_CONFIG = 'c136f42f4cf89e50d00b17f6d9ad13bbbe39c2f59ba71dc180eb1c63ef077c8b'
MASTERS = (941101, 941102, 941103)
POINTS = (0, 512)
BRANCHES = (('learned20', 0, 20), ('fixed12', 20, 12))
CHECKS = 0
DECODED = {}
STORAGE_SELECTIONS = {}


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(message)


def direct(path):
    check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
          and path.is_file() and path.stat().st_nlink == 1, 'direct single-link file: ' + str(path))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def inventory(root, expected=None):
    path = root / 'artifacts.sha256'
    direct(path)
    if expected:
        check(sha(path) == expected, 'pinned artifact manifest')
    out = {}
    for line in path.read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = Path(name)
        check(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
              and not relative.is_absolute() and relative.as_posix() == name
              and '..' not in relative.parts and name not in out, 'canonical manifest entry')
        out[name] = digest
    return out


def verify_inventory(root, items):
    for name, digest in items.items():
        path = root / name
        direct(path)
        check(sha(path) == digest, 'unchanged original artifact: ' + name)
    return len(items)


def codec():
    path = ROOT / CODEC
    direct(path)
    check(sha(path) == CODEC_SHA, 'pinned restricted archive codec')
    spec = importlib.util.spec_from_file_location('train_saved_codec', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StorageRef:
    def __init__(self, path, member, kind, location):
        self.path, self.member, self.kind, self.location = path, member, kind, location


class TensorRef:
    def __init__(self, storage, offset, shape, stride, *unused):
        self.storage, self.offset = storage, offset
        self.shape, self.stride = list(shape), list(stride)


class Archive:
    def __init__(self, path, allow, checkpoint=False):
        direct(path)
        self.path, self.allow, self.checkpoint = path, set(allow), checkpoint
        module = codec()
        owner = self

        class LazyReader(module.Reader):
            def find_class(self, package, name):
                if package == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
                    return TensorRef
                return super().find_class(package, name)

            def persistent_load(self, pid):
                check(pid[0] == 'storage' and pid[1] in ('DoubleStorage', 'FloatStorage', 'LongStorage',
                      'BoolStorage', 'ByteStorage') and pid[3] in ('cpu', 'cuda:0'), 'saved storage schema')
                check(checkpoint or pid[3] == 'cpu', 'non-checkpoint storage must be CPU')
                return StorageRef(owner.path, self.prefix + 'data/' + pid[2], pid[1], pid[3])

        with zipfile.ZipFile(path) as archive:
            keys = [name for name in archive.namelist() if name.endswith('/data.pkl')]
            check(len(keys) == 1, 'unique saved tensor pickle')
            key = keys[0]
            self.root = LazyReader(io.BytesIO(archive.read(key)), archive, key[:-len('data.pkl')]).load()

    def reference(self, key):
        node = self.root
        for part in key.split('.'):
            check(hasattr(node, part), 'saved field exists: ' + key)
            node = getattr(node, part)
        check(isinstance(node, TensorRef), 'saved tensor descriptor: ' + key)
        return node

    def schema(self):
        return {name: dict(shape=node.shape, dtype=node.storage.kind, saved_device=node.storage.location)
                for name,node in self.references().items()}

    def references(self):
        output={}

        def visit(node, prefix):
            if isinstance(node, TensorRef):
                output[prefix]=node
            elif isinstance(node, dict):
                for key,child in node.items():
                    visit(child,prefix+'.'+str(key))
            elif isinstance(node,(tuple,list)):
                for index,child in enumerate(node):
                    visit(child,prefix+'.'+str(index))
            elif hasattr(node,'__dict__'):
                for key, child in vars(node).items():
                    visit(child, prefix + ('.' if prefix else '') + key)
        visit(self.root, '')
        return output

    def tensor(self, key, dtype=None, shape=None):
        check(key in self.allow, 'numeric read outside predeclared field allowlist: ' + key)
        ref = self.reference(key)
        check((dtype is None or ref.storage.kind == dtype) and (shape is None or ref.shape == shape),
              'numeric tensor schema: ' + key)
        for other_name,other in self.references().items():
            check(other_name in self.allow or other.storage.member != ref.storage.member,
                  'selected storage aliases an unselected field: '+key+' / '+other_name)
        with zipfile.ZipFile(self.path) as archive:
            raw = archive.read(ref.storage.member)
        fmt={'DoubleStorage':'d','FloatStorage':'f','LongStorage':'q','BoolStorage':'B','ByteStorage':'B'}[ref.storage.kind]
        element_size=struct.calcsize('<'+fmt)
        check(len(raw)%element_size==0 and len(ref.shape)==len(ref.stride)
              and ref.offset>=0 and all(isinstance(x,int) and x>=0 for x in ref.shape+ref.stride),
              'bounded selected tensor offset/stride')
        logical=list(itertools.product(*(range(length) for length in ref.shape)))
        indices=[ref.offset+sum(i*step for i,step in zip(index,ref.stride)) for index in logical]
        check(len(indices)==math.prod(ref.shape) and all(i<len(raw)//element_size for i in indices),
              'selected tensor indices remain inside saved storage')
        # Unpack only selected logical elements; surrounding storage is never
        # converted into numbers, including noncontiguous views and offsets.
        values=[struct.unpack_from('<'+fmt,raw,i*element_size)[0] for i in indices]
        if ref.storage.kind in ('DoubleStorage', 'FloatStorage'):
            check(all(math.isfinite(x) for x in values), 'finite selected tensor: ' + key)
        if ref.storage.kind == 'BoolStorage':
            check(set(values) <= {0, 1}, 'boolean selected tensor: ' + key)
        DECODED.setdefault(str(self.path.relative_to(ROOT)), set()).add(key)
        STORAGE_SELECTIONS.setdefault(str(self.path.relative_to(ROOT)),{})[key]=dict(
            storage_bytes=len(raw),selected_logical_elements=len(indices),element_size=element_size,
            storage_offset=ref.offset,shape=ref.shape,stride=ref.stride,
            minimum_selected_storage_index=min(indices) if indices else None,
            maximum_selected_storage_index=max(indices) if indices else None,
            aliases_unselected_field=False)
        return values

    def text(self, key):
        return bytes(self.tensor(key, 'ByteStorage')).decode('utf-8')


FEATURE_TEXT = ('protocol', 'artifact_kind', 'dataset_id', 'profile', 'model_tag', 'source_fingerprint',
                'card_sha', 'input_manifest_sha', 'config_sha', 'scaler_identity', 'resolved_settings_json')
FEATURE_ALLOW = set(FEATURE_TEXT) | {'master', 'point', 'native_width', 'training_values',
                                     'training_valid'}
CP_TEXT = ('artifact_kind', 'model_tag', 'architecture_id', 'native_layout', 'dtype', 'training_device', 'scaler_identity')
SCALER_FIELDS = ('mean', 'scale', 'count', 'floor_applied', 'channel_ids', 'scale_floor')
CP_ALLOW = set(CP_TEXT) | {'config_integer_v1', 'config_number_v1', 'channel_ids', 'training_updates',
                         'model.backbone.decoder_first.weight'} | {'scaler.' + key for key in SCALER_FIELDS}
FIT_ALLOW = {'feature_mean', 'feature_scale', 'outer_normalizer_applied', 'outer_fitted_rows', 'fitted_rows',
             'ridge_mean', 'ridge_scale', 'tiny_mean', 'tiny_scale', 'training_source_ids_json'}


def selected_inputs():
    selected = []
    for master in MASTERS:
        for point in POINTS:
            for name in ('features.pt', 'checkpoint.pt'):
                selected.append(ORIGINAL / f'profiles/A/master-{master}/point-{point}' / name)
        for method in ('fixed_12', 'learned_20_0', 'learned_20_512'):
            selected.append(BRANCH / f'readouts/master-{master}/readouts/{method}/rep-2701/fit.pt')
    return selected


def freeze(output):
    check(output.is_absolute() and output.resolve(strict=True) == output and output.is_dir(), 'direct new capsule')
    check(not (output / 'freeze').exists(), 'fresh freeze leaf')
    original = inventory(ORIGINAL, ORIGINAL_MANIFEST)
    check(len(original) == 815, '815 original artifact entries')
    original_done = json.loads((ORIGINAL / 'complete.json').read_text())
    check(original_done['status'] == 'complete' and original_done['protocol'] == 'configurable-capacity-v1'
          and original_done['artifacts_manifest_sha256'] == ORIGINAL_MANIFEST, 'original completion binding')
    original_receipt = json.loads((ORIGINAL / 'freeze/receipt.json').read_text())
    check(original_receipt['source_id'] == ORIGINAL_SOURCE, 'original frozen source')
    branch_done = json.loads((BRANCH / 'complete.json').read_text())
    check(branch_done['status'] == 'complete' and branch_done['protocol'] == 'configurable-branch-v1'
          and branch_done['encoder_calls'] == 0 and branch_done['new_head_fit_pairs'] == 27
          and branch_done['artifacts_manifest_sha256'] == BRANCH_MANIFEST
          and branch_done['original_artifacts_manifest_sha256'] == ORIGINAL_MANIFEST,
          'completed immutable branch diagnostic')
    branch = inventory(BRANCH, BRANCH_MANIFEST)
    source = sorted([p.relative_to(ROOT) for p in (ROOT / HERE).glob('*') if p.is_file()] + [CARD, CODEC])
    check(source and len(source) == len(set(source)), 'closed unique source list')
    hashes = {p.as_posix(): sha(ROOT / p) for p in source}
    source_text = ''.join(f'{digest}  {name}\n' for name, digest in hashes.items())
    source_id = hashlib.sha256(source_text.encode()).hexdigest()
    (output / 'freeze').mkdir()
    for name, digest in hashes.items():
        path = ROOT / name
        direct(path)
        destination = output / 'freeze/source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(path.read_bytes())
        check(sha(destination) == digest and sha(path) == digest, 'exact frozen source: ' + name)
    write_new(output / 'freeze/sources.sha256', source_text)
    write_new(output / 'freeze/card.md', (ROOT / CARD).read_text())
    check(sha(output / 'freeze/card.md') == sha(ROOT / CARD), 'byte-exact prospective card')
    inputs = {}
    for path in selected_inputs():
        direct(path)
        parent, items = (ORIGINAL, original) if path.is_relative_to(ORIGINAL) else (BRANCH, branch)
        name = path.relative_to(parent).as_posix()
        digest = sha(path)
        check(name in items and items[name] == digest, 'original selected role: ' + name)
        inputs[str(path)] = digest
    receipt = dict(protocol='learned-train-v1', source_fingerprint=source_id, card_sha256=hashes[CARD.as_posix()],
                   original_manifest_sha256=ORIGINAL_MANIFEST, original_source_fingerprint=ORIGINAL_SOURCE,
                   branch_manifest_sha256=branch_done['artifacts_manifest_sha256'],
                   selected_inputs=inputs, selected_input_count=len(inputs), source_files=len(hashes))
    write_new(output / 'freeze/receipt.json', receipt)
    write_new(output / 'freeze/original-artifacts.sha256', (ORIGINAL / 'artifacts.sha256').read_text())
    write_new(output / 'freeze/branch-artifacts.sha256', (BRANCH / 'artifacts.sha256').read_text())
    return receipt


def moments(rows):
    check(len(rows) >= 2 and rows and len(rows[0]) > 0, 'supported coordinate rows')
    width, n = len(rows[0]), len(rows)
    check(all(len(row) == width and all(math.isfinite(x) for x in row) for row in rows), 'finite rectangular rows')
    means = [math.fsum(row[j] for row in rows) / n for j in range(width)]
    centered = [[value - means[j] for j, value in enumerate(row)] for row in rows]
    covariance = [[math.fsum(row[j] * row[k] for row in centered) / n for k in range(width)]
                  for j in range(width)]
    std = [math.sqrt(max(0., covariance[j][j])) for j in range(width)]
    trace = math.fsum(covariance[j][j] for j in range(width))
    square_trace = math.fsum(value * value for row in covariance for value in row)
    coordinates = []
    for j in range(width):
        values = [row[j] for row in rows]
        coordinates.append(dict(local_index=j, mean=means[j], population_std=std[j], minimum=min(values),
                                maximum=max(values), rms=math.sqrt(math.fsum(x*x for x in values)/n),
                                exact_distinct_count=len(set(values)), exact_constant=min(values) == max(values),
                                std_below_normalizer_floor=std[j] < 1e-8))
    return dict(coordinates=coordinates, mean_std=math.fsum(std)/width, centered_rms=math.sqrt(trace),
                covariance_participation_rank=trace*trace/square_trace if square_trace else 0.,
                exact_constant_coordinates=sum(c['exact_constant'] for c in coordinates),
                floor_coordinates=sum(c['std_below_normalizer_floor'] for c in coordinates)), covariance


def check_pairs(ids, validity):
    check(len(ids) == len(validity) and len(ids) > 0 and len(ids) % 2 == 0, 'paired TRAIN rows')
    check(all(ids[i] == ids[i+1] and validity[i] == validity[i+1] for i in range(0, len(ids), 2))
          and len(set(ids)) == len(ids)//2, 'unique adjacent TRAIN source pairs with equal support')


def pair_metrics(rows, ids, validity):
    check_pairs(ids, validity)
    width = len(rows[0])
    pairs = []
    for i in range(0, len(rows), 2):
        if not validity[i]:
            continue
        signed = [right-left for left, right in zip(rows[i], rows[i+1])]
        norm = math.sqrt(math.fsum(x*x for x in signed))
        pairs.append(dict(source_id=ids[i], row0=i, row1=i+1, signed_difference=signed,
                          l2_norm=norm, rms=norm/math.sqrt(width)))
    check(bool(pairs), 'supported TRAIN source pairs')
    norms = [p['l2_norm'] for p in pairs]
    return dict(pairs=pairs, mean_pair_l2=statistics.fmean(norms), median_pair_l2=statistics.median(norms),
                mean_absolute_coordinate_difference=math.fsum(abs(x) for p in pairs for x in p['signed_difference'])/(len(pairs)*width),
                signed_coordinate_means=[math.fsum(p['signed_difference'][j] for p in pairs)/len(pairs) for j in range(width)],
                exact_zero_pairs=sum(norm == 0 for norm in norms))


def change_metrics(initial, trained, ids, validity):
    check(len(initial) == len(trained) and all(len(a) == len(b) for a, b in zip(initial, trained)), 'matched update geometry')
    delta = [[right-left for left, right in zip(a, b)] for a, b in zip(initial, trained)]
    selected = [row for row, valid in zip(delta, validity) if valid]
    check(bool(selected), 'supported change rows')
    width = len(selected[0])
    return dict(changed_cells=sum(x != 0 for row in selected for x in row),
                max_absolute_change=max(abs(x) for row in selected for x in row),
                signed_coordinate_mean_change=[math.fsum(row[j] for row in selected)/len(selected) for j in range(width)],
                coordinate_rms_change=[math.sqrt(math.fsum(row[j]*row[j] for row in selected)/len(selected)) for j in range(width)],
                per_row_l2_change=[dict(row=i, source_id=ids[i], l2_norm=math.sqrt(math.fsum(x*x for x in row)))
                                   for i, (row, valid) in enumerate(zip(delta, validity)) if valid],
                pair_difference_changes=pair_metrics(delta, ids, validity))


def decoder_metrics(weights, coordinate_stats, covariance, start, width):
    check(len(weights) == 128 and all(len(row) == 64 for row in weights), 'decoder first matrix128x64')
    norms = [math.sqrt(math.fsum(row[j]*row[j] for row in weights)) for j in range(start, start+width)]
    adjusted = [norm * coordinate_stats[j]['population_std'] for j, norm in enumerate(norms)]
    contribution2 = math.fsum(weights[h][start+j]*covariance[j][k]*weights[h][start+k]
                              for h in range(128) for j in range(width) for k in range(width))
    check(contribution2 >= -1e-10, 'nonnegative covariance linear contribution')
    return dict(column_weight_l2=norms, raw_block_frobenius=math.sqrt(math.fsum(x*x for x in norms)),
                std_adjusted_column_l2=adjusted, std_adjusted_block_rss=math.sqrt(math.fsum(x*x for x in adjusted)),
                covariance_centered_linear_contribution_rms=math.sqrt(max(0., contribution2)))


def close_values(actual, expected, message):
    check(len(actual) == len(expected), message + ' shape')
    for a, b in zip(actual, expected):
        check(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 2e-9*max(1.,abs(a),abs(b)), message)


def feature(path, master, point, ids):
    a = Archive(path, FEATURE_ALLOW)
    expected = dict(protocol='configurable-capacity-v1', artifact_kind='configurable_capacity_native_feature_views_v1',
                    dataset_id='TEMPO-5', profile='A', model_tag='RPB-v20.alt-01', source_fingerprint=ORIGINAL_SOURCE,
                    card_sha=ORIGINAL_CARD, input_manifest_sha=ORIGINAL_INPUT, config_sha=ORIGINAL_CONFIG)
    for key, value in expected.items():
        check(a.text(key) == value, 'original feature binding: ' + key)
    for key, value in (('master', master), ('point', point), ('native_width', 32)):
        check(a.tensor(key, 'LongStorage', []) == [value], 'feature scalar binding: ' + key)
    settings = json.loads(a.text('resolved_settings_json'))
    check(settings['channels'] == 3 and settings['history'] == 128 and settings['features'] == 9
          and settings['learned_coordinates'] == 20 and settings['fixed_coordinates'] == 12,
          'declared original native partition')
    values = a.tensor('training_values', 'DoubleStorage', [512,32])
    valid = a.tensor('training_valid', 'BoolStorage', [512])
    check_pairs(ids, valid)
    check(all(isinstance(x,str) and x.startswith('TEMPO-5/') and (int(x.split('/')[1]) >> 31) == master for x in ids)
          and sum(valid) == 512, 'declared master and all512 supported TRAIN rows')
    return dict(rows=[values[i:i+32] for i in range(0,len(values),32)], validity=valid, ids=ids,
                scaler_identity=a.text('scaler_identity'))


def checkpoint(path, point, scaler_identity):
    a = Archive(path, CP_ALLOW, checkpoint=True)
    check(a.text('artifact_kind') == 'rpb_configurable_inference_checkpoint_v1'
          and a.text('model_tag') == 'RPB-v20' and a.text('dtype') == 'float32'
          and a.text('training_device') == 'cuda:0' and a.text('scaler_identity') == scaler_identity,
          'original typed checkpoint identity')
    check(a.text('architecture_id') == 'early-configurable-shape-plus-fixed-two-band-complex-relations-v1'
          and a.text('native_layout') == 'learned-D-minus-four-per-semantic-pair;fixed-four-per-semantic-pair;'
            'canonical-all-channel-pairs;ReLow-ImLow-ReHigh-ImHigh;q1-3-low-q4-8-high;original-H;feature-pair-mean;pair-unit-L2-floor1e-12',
          'original checkpoint architecture/layout binding')
    integers = a.tensor('config_integer_v1', 'LongStorage', [15])
    check(integers == [3,128,9,8,64,32,3,4,1,1,2,0,0,256,128], 'original A checkpoint geometry')
    check(a.tensor('channel_ids', 'LongStorage', [3]) == [0,1,2]
          and a.tensor('training_updates', 'LongStorage', []) == [point], 'checkpoint channel/update binding')
    check(a.tensor('config_number_v1','DoubleStorage',[6])==[0.,1e-5,.25,1.,1e-6,1.], 'original A numerical settings')
    # Decode only the declared first-layer parameter; no module is constructed.
    values = a.tensor('model.backbone.decoder_first.weight', 'FloatStorage', [128,64])
    check(a.reference('model.backbone.decoder_first.weight').storage.location=='cuda:0',
          'saved CUDA parameter bytes read without device execution')
    scaler = {}
    for key in SCALER_FIELDS:
        dtype = 'BoolStorage' if key == 'floor_applied' else 'LongStorage' if key in ('count','channel_ids') else 'DoubleStorage'
        shape = [3] if key == 'channel_ids' else [] if key == 'scale_floor' else [3,9]
        scaler[key] = a.tensor('scaler.' + key, dtype, shape)
    check(all(x >= 1e-6 for x in scaler['scale']) and all(x > 0 for x in scaler['count'])
          and scaler['scale_floor'] == [1e-6] and scaler['channel_ids'] == [0,1,2], 'TRAIN scaler validity')
    weights = [values[i:i+64] for i in range(0,len(values),64)]
    metadata = {}
    for name, start in (('patch_position',32),('channel_identity',48)):
        norms = [math.sqrt(math.fsum(row[j]*row[j] for row in weights)) for j in range(start,start+16)]
        metadata[name] = dict(column_weight_l2=norms, raw_block_frobenius=math.sqrt(math.fsum(x*x for x in norms)))
    return weights, scaler, metadata


def normalizer(master, method, stats, ids):
    path = BRANCH / f'readouts/master-{master}/readouts/{method}/rep-2701/fit.pt'
    a = Archive(path, FIT_ALLOW)
    width = len(stats['coordinates'])
    result = {}
    for key in ('feature_mean','feature_scale','ridge_mean','ridge_scale','tiny_mean','tiny_scale'):
        result[key] = a.tensor(key, 'DoubleStorage', [width])
    for key in ('outer_normalizer_applied','outer_fitted_rows','fitted_rows'):
        result[key] = a.tensor(key, 'BoolStorage' if key == 'outer_normalizer_applied' else 'LongStorage', [])
    check(result['outer_normalizer_applied'] == [1] and result['outer_fitted_rows'] == [512]
          and result['fitted_rows'] == [512], 'saved TRAIN normalizer fitted-row binding')
    check(json.loads(a.text('training_source_ids_json')) == ids, 'normalizer TRAIN source-order binding')
    close_values(result['feature_mean'], [c['mean'] for c in stats['coordinates']], 'saved TRAIN outer means')
    close_values(result['feature_scale'], [max(c['population_std'],1e-8) for c in stats['coordinates']], 'saved TRAIN outer scales')
    check(all(x > 0 for key in ('feature_scale','ridge_scale','tiny_scale') for x in result[key]), 'positive saved TRAIN scales')
    return result


def run(output):
    receipt = json.loads((output / 'freeze/receipt.json').read_text())
    source_text = (output / 'freeze/sources.sha256').read_text()
    check(hashlib.sha256(source_text.encode()).hexdigest() == receipt['source_fingerprint'], 'frozen source fingerprint')
    for line in source_text.splitlines():
        digest, name = line.split('  ',1)
        check(sha(ROOT/name) == digest and sha(output/'freeze/source'/name) == digest, 'exact execution source: '+name)
    check(sha(output/'freeze/card.md') == receipt['card_sha256'], 'predeclared card')
    for path, digest in receipt['selected_inputs'].items():
        check(sha(Path(path)) == digest, 'frozen selected input')
    items = inventory(ORIGINAL, ORIGINAL_MANIFEST)
    before = verify_inventory(ORIGINAL, items)
    write_new(output/'original-before.json', dict(status='pass', verified_artifacts=before, manifest_sha256=ORIGINAL_MANIFEST))
    results = []
    for master in MASTERS:
        features, scaler_at_zero, per_master = {}, None, dict(master=master, points=[], changes={})
        for point in POINTS:
            directory = ORIGINAL/f'profiles/A/master-{master}/point-{point}'
            id_archive = Archive(BRANCH / f'readouts/master-{master}/readouts/learned_20_{point}/rep-2701/fit.pt', FIT_ALLOW)
            ids = json.loads(id_archive.text('training_source_ids_json'))
            f = feature(directory/'features.pt',master,point,ids)
            weights, scaler, metadata = checkpoint(directory/'checkpoint.pt',point,f['scaler_identity'])
            if point == 0:
                scaler_at_zero = scaler
            else:
                check(scaler == scaler_at_zero, 'exact unchanged original TRAIN scaler')
                check(f['ids'] == features[0]['ids'] and f['validity'] == features[0]['validity']
                      and f['scaler_identity'] == features[0]['scaler_identity'], 'same-row point-to-point identity/support')
            features[point] = f
            point_report = dict(point=point, valid_rows=sum(f['validity']), source_pairs=len(set(f['ids'])),
                                train_scaler=scaler, decoder_metadata=metadata, branches={})
            for name,start,width in BRANCHES:
                rows = [row[start:start+width] for row in f['rows']]
                selected = [row for row,valid in zip(rows,f['validity']) if valid]
                stats,covariance = moments(selected)
                for coordinate in stats['coordinates']:
                    coordinate['native_index'] = start + coordinate['local_index']
                fit_name = 'fixed_12' if name == 'fixed12' else 'learned_20_'+str(point)
                point_report['branches'][name] = dict(variation=stats, pair_differences=pair_metrics(rows,f['ids'],f['validity']),
                    decoder_first_potential=decoder_metrics(weights,stats['coordinates'],covariance,start,width),
                    saved_train_head_normalizers=normalizer(master,fit_name,stats,f['ids']))
            per_master['points'].append(point_report)
        for name,start,width in BRANCHES:
            a = [row[start:start+width] for row in features[0]['rows']]
            b = [row[start:start+width] for row in features[512]['rows']]
            if name == 'fixed12':
                check(a == b, 'fixed12 exact unchanged TRAIN values')
            per_master['changes'][name] = change_metrics(a,b,features[0]['ids'],features[0]['validity'])
        per_master['fixed12_exact_0_512'] = True
        results.append(per_master)
    for path,digest in receipt['selected_inputs'].items():
        check(sha(Path(path)) == digest, 'selected input unchanged after arithmetic')
    for line in source_text.splitlines():
        digest,name=line.split('  ',1)
        check(sha(ROOT/name)==digest and sha(output/'freeze/source'/name)==digest,
              'execution source unchanged after arithmetic: '+name)
    check(all(not any(key.startswith(('validation','deleted')) or key in ('labels_scoring_only','ridge_weights','tiny_w1','tiny_w2')
                  for key in keys) for keys in DECODED.values()), 'no held-out values, labels or fitted classifier weights decoded')
    after = verify_inventory(ORIGINAL, items)
    write_new(output/'original-after.json',dict(status='pass',verified_artifacts=after,manifest_sha256=ORIGINAL_MANIFEST))
    report = dict(protocol='learned-train-v1',model_tag='RPB-v20.alt-01',dataset_id='TEMPO-5',designed_complexity=5,
                  evidence_stage='known development TRAIN-only saved arithmetic', source_fingerprint=receipt['source_fingerprint'],
                  card_sha256=receipt['card_sha256'],encoder_calls=0,encoder_updates=0,head_fits=0,validation_values_decoded=0,
                  original_artifacts_before=before,original_artifacts_after=after,selected_input_count=receipt['selected_input_count'],
                  masters=results,checks=CHECKS,decoded_fields={p:sorted(keys) for p,keys in sorted(DECODED.items())},
                  selected_storage_inventory=STORAGE_SELECTIONS,
                  limits=['Full-context TRAIN exports differ from masked training contexts.',
                          'Pair orientation is stored row order, without labels.',
                          'Decoder column metrics indicate potential first-layer linear use, not causal reconstruction or timing use.',
                          'Covariance participation rank is variance concentration, not information or task accuracy.'])
    write_new(output/'report.json', report)
    compact = []
    for master in results:
        for point in master['points']:
            for name in ('learned20','fixed12'):
                record=point['branches'][name]
                compact.append(dict(master=master['master'],point=point['point'],branch=name,
                    mean_std=record['variation']['mean_std'],rank=record['variation']['covariance_participation_rank'],
                    mean_pair_l2=record['pair_differences']['mean_pair_l2'],
                    decoder_std_adjusted_rss=record['decoder_first_potential']['std_adjusted_block_rss'],
                    decoder_covariance_rms=record['decoder_first_potential']['covariance_centered_linear_contribution_rms']))
    write_new(output/'compact.json',dict(status='complete',capsule=str(output),compact=compact,checks=CHECKS))


def seal(output):
    receipt=json.loads((output/'freeze/receipt.json').read_text())
    report=json.loads((output/'report.json').read_text())
    check(report['source_fingerprint']==receipt['source_fingerprint'] and report['card_sha256']==receipt['card_sha256']
          and report['original_artifacts_after']==815 and report['validation_values_decoded']==0,
          'finished report/freeze binding')
    for line in (output/'freeze/sources.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        check(sha(ROOT/name)==digest and sha(output/'freeze/source'/name)==digest,'source unchanged before sealing')
    for path,digest in receipt['selected_inputs'].items():
        check(sha(Path(path))==digest,'selected input unchanged before sealing')
    write_new(output/'artifacts.sha256',''.join(f'{sha(path)}  {path.relative_to(output).as_posix()}\n'
        for path in sorted(output.rglob('*')) if path.is_file()))
    write_new(output/'complete.json',dict(protocol='learned-train-v1',status='complete',source_fingerprint=receipt['source_fingerprint'],
        card_sha256=receipt['card_sha256'],report_sha256=sha(output/'report.json'),artifacts_manifest_sha256=sha(output/'artifacts.sha256'),
        encoder_calls=0,encoder_updates=0,head_fits=0,validation_values_decoded=0,original_artifacts_preserved=815))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--schema-only',action='store_true')
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--seal',type=Path)
    args=parser.parse_args()
    check(sum(bool(x) for x in (args.schema_only,args.freeze,args.run,args.seal))==1,'select one execution mode')
    if args.schema_only:
        paths=[ORIGINAL/'profiles/A/master-941101/point-0/features.pt',
               ORIGINAL/'profiles/A/master-941101/point-0/checkpoint.pt',
               BRANCH/'readouts/master-941101/readouts/learned_20_0/rep-2701/fit.pt']
        for path in paths:
            print(json.dumps(dict(path=str(path),schema=Archive(path,set(),checkpoint=path.name=='checkpoint.pt').schema()),indent=2))
        check(not DECODED,'schema inspection decoded no numeric storage')
    elif args.freeze:
        print(json.dumps(freeze(args.freeze),indent=2))
    elif args.run:
        run(args.run)
    else:
        seal(args.seal)


if __name__=='__main__':
    main()

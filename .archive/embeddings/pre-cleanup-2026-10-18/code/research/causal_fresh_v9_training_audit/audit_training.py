# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent saved-data audit; no Torch import, model, head or training call."""
import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import pickle
import struct
import time
import zipfile

CARD_SHA = MD_SHA = INPUT_SHA = None  # Bound by the independently frozen audit card before input access.
MASK64 = (1 << 64) - 1
GRADIENT_POLICY = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
OBJECTIVE_POLICY = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
DIRECT_ROUTES = 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1'
ENCODER_GRADIENT_KIND = 'Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1'
NATIVE_LAYOUT = 'parent32[0:32];mean4[32:36];uncertainty4[36:40];independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;variance-reads-live-uncertainty4-only'
POOL_POLICY = 'original-position-four-bin-shared-global-half-mixture-v1'
POOL_PARTITION = [4, 16, 32, 4, 8]
POOL_MIX = [.5, .5]  # ordered anchored-local, shared-whole-history
EMPTY_LOCAL_POLICY = 'zero-whole-export-slice-after-bias-v1'
BETWEEN_BASIS = 'hadamard-full16-signs-div4-v1'
BETWEEN_SIGNS = [[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]
GAUGE = 'core.encoder.level.base.backbone.pool_score.bias'
EXPORT_WEIGHT = 'core.encoder.level.base.backbone.export_projection.weight'
EXPORT_BIAS = 'core.encoder.level.base.backbone.export_projection.bias'
POOL_POSITIONS = 'core.encoder.level.base.backbone.pool_positions.weight'
checks = 0
bad = []


def ck(value, reason):
    global checks
    checks += 1
    if not value:
        bad.append(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def json_objective(record, label):
    """Validate declarations, not infer gradient routing from scalar values."""
    expected = dict(objective_policy=OBJECTIVE_POLICY, mean_huber_delta=1,
                    loss_coefficients=[1, 1], variance_error_detached=True,
                    reporting_nll_backpropagated=False)
    for key, value in expected.items():
        ck(record[key] == value, label + ' exact objective ' + key)


def json_pooling(record, label):
    ck(record['temporal_pooling_policy'] == POOL_POLICY
       and record['temporal_pooling_partition'] == POOL_PARTITION
       and record['temporal_pooling_mix'] == POOL_MIX
       and record['empty_local_bin_policy'] == EMPTY_LOCAL_POLICY,
       label + ' exact original-position half-mixture and zero-empty declaration')


def typed_pooling(archive, prefix, eq, label):
    for field, expected in (('temporal_pooling_policy', POOL_POLICY),
                            ('empty_local_bin_policy', EMPTY_LOCAL_POLICY)):
        text = archive.field(*(prefix + (field,)))
        eq(text.storage.dtype == 'ByteStorage' and len(text.shape) == 1
           and text.text() == expected, label + ' exact typed UTF8 ' + field)
    partition = archive.field(*(prefix + ('temporal_pooling_partition_v1',)))
    eq(partition.storage.dtype == 'LongStorage' and partition.shape == (5,)
       and partition.values() == POOL_PARTITION, label + ' exact typed Long[5] temporal partition')
    mixture = archive.field(*(prefix + ('temporal_pooling_mix_f64',)))
    eq(mixture.storage.dtype == 'DoubleStorage' and mixture.shape == (2,)
       and mixture.values() == POOL_MIX, label + ' exact typed Double[2] half-mixture')


def typed_objective(archive, prefix, eq, label):
    expected_text = dict(objective_policy=OBJECTIVE_POLICY,
                         gradient_policy=GRADIENT_POLICY,
                         direct_gradient_routes=DIRECT_ROUTES)
    for field, value in expected_text.items():
        eq(archive.field(*(prefix + (field,))).text() == value,
           label + ' typed objective text ' + field)
    delta = archive.field(*(prefix + ('mean_huber_delta_f64',)))
    weights = archive.field(*(prefix + ('loss_coefficients_f64',)))
    eq(delta.storage.dtype == 'DoubleStorage' and delta.shape == ()
       and delta.scalar() == 1, label + ' typed F64 Huber delta1')
    eq(weights.storage.dtype == 'DoubleStorage' and weights.shape == (2,)
       and weights.values() == [1, 1], label + ' typed F64 unit component weights')
    for field, value in (('variance_error_detached', 1), ('reporting_nll_backpropagated', 0)):
        tensor = archive.field(*(prefix + (field,)))
        eq(tensor.storage.dtype == 'LongStorage' and tensor.shape == ()
           and tensor.scalar() == value, label + ' typed objective flag ' + field)


def counter(base, attempt):
    x = (base + 0x9e3779b97f4a7c15 * (attempt + 1)) & MASK64
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK64
    return (x ^ (x >> 31)) & ((1 << 63) - 1)


class MT19937_64:
    def __init__(self, seed):
        self.state = [seed]
        for i in range(1, 312):
            x = self.state[-1]
            self.state.append((6364136223846793005 * (x ^ (x >> 62)) + i) & MASK64)
        self.index = 312

    def next(self):
        if self.index == 312:
            for i in range(312):
                y = ((self.state[i] & 0xffffffff80000000) |
                     (self.state[(i + 1) % 312] & 0x7fffffff))
                self.state[i] = (self.state[(i + 156) % 312] ^ (y >> 1) ^
                                 (0xb5026f5aa96619e9 if y & 1 else 0))
            self.index = 0
        y = self.state[self.index]
        self.index += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71d67fffeda60000
        y ^= (y << 37) & 0xfff7eee000000000
        y ^= y >> 43
        return y & MASK64


def anchors():
    first = 1704067259999 + 255 * 60000
    n = (1705881599999 - first) // 60000 + 1
    return [first + (i * (n - 1) // 4095) * 60000 for i in range(4096)]


# A deliberately restricted DATA reader for C++ OutputArchive. These inert
# descriptors never import torch, instantiate a learned module, resolve arbitrary
# pickle globals, or execute encoder code. Format failure is a separate blocker.
class Record:
    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise ValueError('Unsupported inert archive record state')
        self.fields = state


DTYPES = {'ByteStorage': ('B', 1), 'BoolStorage': ('?', 1),
          'LongStorage': ('q', 8), 'IntStorage': ('i', 4),
          'FloatStorage': ('f', 4), 'DoubleStorage': ('d', 8)}


class Storage:
    def __init__(self, archive, dtype, key, location, length):
        self.archive, self.dtype, self.key = archive, dtype, key
        self.location, self.length = location, length

    def bytes(self):
        value = self.archive.zip.read(self.archive.prefix + 'data/' + self.key)
        if len(value) != self.length * DTYPES[self.dtype][1]:
            raise ValueError('Storage size differs from typed metadata')
        return value


class Tensor:
    def __init__(self, storage, offset, shape, strides, *unused):
        self.storage, self.offset = storage, offset
        self.shape, self.strides = tuple(shape), tuple(strides)

    def values(self):
        fmt, size = DTYPES[self.storage.dtype]
        data = self.storage.bytes()
        positions = itertools.product(*(range(n) for n in self.shape))
        out = []
        for coordinate in positions:
            index = self.offset + sum(i * stride for i, stride in zip(coordinate, self.strides))
            if index < 0 or index >= self.storage.length:
                raise ValueError('Tensor view exceeds its storage')
            out.append(struct.unpack_from('<' + fmt, data, index * size)[0])
        return out

    def scalar(self):
        values = self.values()
        if len(values) != 1:
            raise ValueError('Scalar required')
        return values[0]

    def text(self):
        if self.storage.dtype != 'ByteStorage':
            raise ValueError('Byte text required')
        return bytes(self.values()).decode('utf-8')


class DataUnpickler(pickle.Unpickler):
    def __init__(self, archive, value):
        super().__init__(io.BytesIO(value))
        self.archive = archive

    def find_class(self, module, name):
        if module == 'torch' and name in DTYPES:
            return name
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return Tensor
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module.startswith('__torch__') and name == 'Module':
            return Record
        raise ValueError('Rejected pickle global: ' + module + '.' + name)

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5 or value[0] != 'storage' or
                value[1] not in DTYPES or not isinstance(value[2], str) or
                not value[2].isdigit() or value[3] not in ('cpu', 'cuda:0') or
                not isinstance(value[4], int) or value[4] < 0):
            raise ValueError('Rejected storage descriptor')
        return Storage(self.archive, *value[1:])


class Archive:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        names = [n for n in self.zip.namelist() if n.endswith('/data.pkl')]
        if len(names) != 1:
            raise ValueError('Exactly one C++ archive data.pkl required')
        self.prefix = names[0][:-len('data.pkl')]
        value = self.zip.read(names[0])
        if len(value) > 16 * 1024 * 1024:
            raise ValueError('Metadata exceeds bounded reader limit')
        self.root = DataUnpickler(self, value).load()

    def field(self, *names):
        value = self.root
        for name in names:
            if not isinstance(value, Record):
                raise ValueError('Expected inert module record at ' + name)
            value = value.fields[name]
        return value

    def close(self):
        self.zip.close()


def independent_bank_identity(bank):
    """Exact little-endian BankStandardizer FNV, including fresh provenance."""
    value = 14695981039346656037
    content = hashlib.sha256()

    def update(data):
        nonlocal value
        for byte in data:
            value = ((value ^ byte) * 1099511628211) & MASK64

    for name in ('source_identity', 'input_scaler_identity', 'layout_identity', 'recipe_identity'):
        data = bank.fields[name].text().encode('utf-8')
        update(struct.pack('<Q', len(data)))
        update(data)
    for name in ('input_geometry', 'sampling_interval', 'floor', 'minimum_support'):
        update(tensor_bytes(bank.fields[name])[2])
    scalar_types = {'LongStorage': 4, 'DoubleStorage': 7, 'BoolStorage': 11}
    for name in ('counts', 'sum', 'sumsq', 'mean', 'scale', 'support'):
        tensor = bank.fields[name]
        dtype, shape, data = tensor_bytes(tensor)
        descriptor = struct.pack('<qq', scalar_types[dtype], math.prod(shape))
        update(descriptor)
        update(data)
        content.update(descriptor)
        content.update(data)
    return 'rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-' + format(value, '016x'), content.hexdigest()


def numeric_audit(run, data_root, selected, attempts_by_update):
    numeric_bad, numeric_checks = [], 0

    def eq(ok, why):
        nonlocal numeric_checks
        numeric_checks += 1
        if not ok:
            numeric_bad.append(why)

    def near(actual, expected, relative, absolute, why):
        eq(math.isfinite(actual) and math.isfinite(expected) and
           abs(actual - expected) <= absolute + relative * abs(expected), why)

    opened = []
    try:
        fitted = Archive(run / 'calibration/input-scaler.pt')
        bank_file = Archive(run / 'calibration/bank-standardizer.pt')
        witness = Archive(run / 'calibration/first-TRAIN-batch-witness.pt')
        opened.extend((fitted, bank_file, witness))
        fresh_source = ('CPB-v9.market-01;fresh-TRAIN-only-CACHE88:67745d6a/0aad9d11/364fbf2c;'
                        'selection10df626e;4096windows;recent128;IDs0,1,2;no-encoder-calibration')
        actual_bank = bank_file.field('bank', 'bank')
        eq(fitted.field('source_identity').text() == bank_file.field('source_identity').text() == fresh_source,
           'Actual fresh V9 TRAIN calibration source provenance')
        eq(actual_bank.fields['source_identity'].text() == 'RPB-v28;raw-symmetric-relative-f64-v1;' + fresh_source,
           'Actual version-aware relative-bank provenance')
        identity, statistic_content_sha256 = independent_bank_identity(actual_bank)
        eq(identity == bank_file.field('bank', 'relative_identity').text()
           == read(run / 'calibration/summary.json')['bank_identity'], 'Independent fresh bank FNV metadata/tensor identity')
        eq(identity.removeprefix('rpb-relative-bank-v1:') == actual_bank.fields['normalizer_identity'].text(),
           'Independent nested bank FNV identity')
        eq(fitted.field('fit_row_ids').values() == selected, 'Input scaler exact 4096 fit rows')
        eq(bank_file.field('fit_row_ids').values() == selected, 'Bank exact 4096 fit rows')
        first_ids = witness.field('row_ids').values()
        eq(first_ids == selected[:8], 'Independent first calibration witness rows')
        stored_raw = witness.field('raw_F64').values()
        stored_mask = witness.field('observed').values()
        stored_rounded = witness.field('parent_rounded_F32_then_F64').values()
        stored_bank = witness.field('relative_bank_F32').values()
        stored_bank_valid = witness.field('relative_bank_valid').values()
        raw_records, input_count, input_sum, input_sumsq = [], [], [], []
        bank_sums = [[] for _ in range(270)]
        bank_squares = [[] for _ in range(270)]
        first_bank_values = [[0.] * 270 for _ in range(8)]
        first_bank_flags = [[False] * 270 for _ in range(8)]
        record = struct.Struct('<q5dq4d')
        f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
        for channel, symbol in enumerate(('BTCUSDT', 'ETHUSDT', 'SOLUSDT')):
            path = data_root / 'data/mkt-hist-1/csv' / (symbol + '-1m-2024-01.cache.bin')
            rows = list(record.iter_unpack(path.read_bytes()))
            first = rows[0][6]
            eq(all(r[6] == first + i * 60000 for i, r in enumerate(rows)), 'Exact CACHE88 time grid ' + symbol)
            ends = [(anchor - first) // 60000 for anchor in selected]
            # Parse the whole preserved cache envelope, then inspect numeric
            # feature values only through the last selected TRAIN endpoint.
            rows = rows[:max(ends) + 1]
            eq(all(first + end * 60000 == anchor and end >= 127 for anchor, end in zip(selected, ends)), 'Exact selected cache indices ' + symbol)
            weights_delta = [0] * (len(rows) + 1)
            for end in ends:
                weights_delta[end - 127] += 1
                weights_delta[end + 1] -= 1
            weights = list(itertools.accumulate(weights_delta))[:-1]
            eq(sum(weights) == 4096 * 128, 'Window-weighted input exposure ' + symbol)
            for feature in range(9):
                field_index = feature + 1 if feature < 5 else feature + 2
                values = [r[field_index] for r in rows]
                observed = [r[0] != -(1 << 63) and math.isfinite(v) for r, v in zip(rows, values)]
                count = sum(w for w, valid in zip(weights, observed) if valid)
                safe = [f32(v) if valid else 0. for v, valid in zip(values, observed)]
                input_count.append(count)
                input_sum.append(math.fsum(w * v for w, v in zip(weights, safe)))
                input_sumsq.append(math.fsum(w * v * v for w, v in zip(weights, safe)))
                increments, adjacent = [], []
                for index in range(1, len(rows)):
                    valid = observed[index] and observed[index - 1]
                    adjacent.append(valid)
                    if not valid:
                        increments.append(0.)
                        continue
                    a, b = values[index], values[index - 1]
                    scale = max(abs(a), abs(b))
                    if scale == 0.:
                        increments.append(0.)
                        continue
                    aa, bb = a / scale, b / scale
                    denominator = abs(aa) + abs(bb)
                    increments.append(f32(2. * (aa - bb) / denominator) if denominator else 0.)
                prefix_n = [0] + list(itertools.accumulate(adjacent))
                for row_index, end in enumerate(ends):
                    begin = end - 127
                    block = increments[begin:end]
                    total = prefix_n[end] - prefix_n[begin]
                    active = total >= 2
                    average = math.fsum(block) / max(total, 1)
                    energy = math.fsum(v * v for v in block) / max(total, 1)
                    rms = math.sqrt(max(energy, 1e-12))
                    moments = [average if active else 0., rms if active and energy > 0 else 0.]
                    flags = [active, active]
                    for quarter in range(4):
                        q0, q1 = quarter * 127 // 4, (quarter + 1) * 127 // 4
                        n = prefix_n[begin + q1] - prefix_n[begin + q0]
                        eligible = active and n >= 2
                        part = block[q0:q1]
                        mean = math.fsum(part) / max(n, 1)
                        square = math.fsum(v * v for v in part) / max(n, 1)
                        part_rms = math.sqrt(max(square, 1e-12)) if square > 0 else 0.
                        moments.extend((mean / rms if eligible else 0., part_rms / rms if eligible else 0.))
                        flags.extend((eligible, eligible))
                    for moment, (value, flag) in enumerate(zip(moments, flags)):
                        column = (channel * 9 + feature) * 10 + moment
                        rounded = f32(value)
                        if flag:
                            bank_sums[column].append(rounded)
                            bank_squares[column].append(rounded * rounded)
                        if row_index < 8:
                            first_bank_values[row_index][column] = rounded
                            first_bank_flags[row_index][column] = flag
                    if row_index < 8:
                        for offset in range(128):
                            flat = ((row_index * 3 + channel) * 128 + offset) * 9 + feature
                            index = end - 127 + offset
                            expected_raw = values[index] if observed[index] else 0.
                            eq(stored_raw[flat] == expected_raw and stored_mask[flat] == observed[index], 'Direct-record first witness raw/mask ' + str(flat))
                            eq(stored_rounded[flat] == safe[index], 'Direct F32 parent rounding witness ' + str(flat))
            raw_records.append(len(rows))
        saved_count = fitted.field('scaler', 'count').values()
        saved_sum, saved_sumsq = fitted.field('sum').values(), fitted.field('sumsq').values()
        saved_mean, saved_scale = fitted.field('scaler', 'mean').values(), fitted.field('scaler', 'scale').values()
        saved_floor = fitted.field('scaler', 'floor_applied').values()
        for column in range(27):
            eq(saved_count[column] == input_count[column] and input_count[column] >= 2, 'Exact input count ' + str(column))
            near(saved_sum[column], input_sum[column], 1e-12, 1e-12, 'Independent input sum ' + str(column))
            near(saved_sumsq[column], input_sumsq[column], 1e-12, 1e-12, 'Independent input sumsq ' + str(column))
            mean = input_sum[column] / input_count[column]
            deviation = math.sqrt(max(input_sumsq[column] / input_count[column] - mean * mean, 0.))
            near(saved_mean[column], mean, 1e-12, 1e-12, 'Independent input mean ' + str(column))
            near(saved_scale[column], max(deviation, 1e-6), 2e-8, 1e-9, 'Independent input scale ' + str(column))
            eq(saved_floor[column] == (deviation < 1e-6), 'Input floor flag ' + str(column))
        bank_counts = bank_file.field('bank', 'bank', 'counts').values()
        bank_sum = bank_file.field('bank', 'bank', 'sum').values()
        bank_sumsq = bank_file.field('bank', 'bank', 'sumsq').values()
        bank_mean = bank_file.field('bank', 'bank', 'mean').values()
        bank_scale = bank_file.field('bank', 'bank', 'scale').values()
        bank_support = bank_file.field('bank', 'bank', 'support').values()
        max_bank_sum_absolute_error = 0.
        for column in range(270):
            n = len(bank_sums[column])
            total, squared = math.fsum(bank_sums[column]), math.fsum(bank_squares[column])
            mean = total / max(n, 1) if n >= 2 else 0.
            scale = max(math.sqrt(max(squared / max(n, 1) - mean * mean, 0.)), 1e-6) if n >= 2 else 1.
            eq(bank_counts[column] == n and bank_support[column] == (n >= 2), 'Independent bank support/count ' + str(column))
            near(bank_sum[column], total, 2e-6, 1e-9, 'Independent bank sum ' + str(column))
            near(bank_sumsq[column], squared, 3e-6, 1e-11, 'Independent bank sumsq ' + str(column))
            near(bank_mean[column], mean, 2e-6, 1e-10, 'Independent bank mean ' + str(column))
            near(bank_scale[column], scale, 3e-6, 1e-10, 'Independent bank scale ' + str(column))
            max_bank_sum_absolute_error = max(max_bank_sum_absolute_error, abs(bank_sum[column] - total))
            for row in range(8):
                index = row * 270 + column
                eq(stored_bank_valid[index] == first_bank_flags[row][column], 'Bank witness validity ' + str(index))
                near(stored_bank[index], first_bank_values[row][column], 2e-6, 2e-7, 'Bank witness arithmetic ' + str(index))
        typed = []
        for update in (512, 1024, 2048):
            archive = Archive(run / 'checkpoints' / ('training-' + str(update) + '.pt'))
            opened.append(archive)
            progress = archive.field('progress_v1').values()
            eq(progress == [update, attempts_by_update[update], 943002, 943003, 943004],
               'Actual CSV-bound typed progress/all three seeds ' + str(update))
            eq(archive.field('encoder', 'source_updates').scalar() == 0 and
               archive.field('encoder', 'causal_updates').scalar() == update and
               archive.field('encoder', 'pretrained_ancestor').text() == 'none', 'Actual typed fresh ancestry ' + str(update))
            steps = archive.field('optimizer_steps_v1').values()
            eq(len(steps) == 76 and all(step == update for step in steps), 'Actual76 optimizer step counters ' + str(update))
            typed.append(dict(update=update, progress=progress, active_optimizer_states=len(steps)))
        return dict(status='passed' if not numeric_bad else 'failed', checks=numeric_checks,
                    mismatches=numeric_bad, raw_cache_records=raw_records,
                    input_coordinates=27, bank_coordinates=270, independent_TRAIN_windows=4096,
                    first_witness_raw_coordinates=8 * 3 * 128 * 9,
                    max_bank_sum_absolute_error=max_bank_sum_absolute_error,
                    independent_bank_identity=identity,
                    bank_source_identity=actual_bank.fields['source_identity'].text(),
                    bank_statistic_content_sha256=statistic_content_sha256,
                    actual_typed_states=typed,
                    method='Whitelisted inert C++ archive descriptors plus raw storage/struct; independent CACHE88 arithmetic, no torch import',
                    tolerance_policy='Input sums/means1e-12 relative, scale2e-8 relative; F32 bank cross-backend reduction2e-6/3e-6 relative with stated absolute floors; not quality-gate tolerance')
    except Exception as error:
        return dict(status='blocked_format', checks=numeric_checks, mismatches=numeric_bad,
                    blocker=type(error).__name__ + ': ' + str(error),
                    limitation='Core saved arithmetic/hash audit remains independent; numeric data reader could not finish. No fallback model execution.')
    finally:
        for archive in opened:
            archive.close()


def flat_tensors(value, prefix=''):
    result = {}
    items = value.fields.items() if isinstance(value, Record) else value.items() if isinstance(value, dict) else ()
    for name, child in items:
        key = prefix + str(name)
        if isinstance(child, Tensor):
            result[key] = child
        elif isinstance(child, (Record, dict)):
            result.update(flat_tensors(child, key + '.'))
    return result


def tensor_bytes(value):
    size = DTYPES[value.storage.dtype][1]
    storage = value.storage.bytes()
    result = bytearray()
    for coordinate in itertools.product(*(range(n) for n in value.shape)):
        position = value.offset + sum(i * stride for i, stride in zip(coordinate, value.strides))
        if position < 0 or position >= value.storage.length:
            raise ValueError('Tensor position exceeds storage')
        result.extend(storage[position * size:(position + 1) * size])
    return value.storage.dtype, value.shape, bytes(result)


def saved_tensor_audit(run, parameters, attempts_by_update):
    failures, count, opened = [], 0, []

    def eq(ok, reason):
        nonlocal count
        count += 1
        if not ok:
            failures.append(reason)

    try:
        stages = {}
        for label, update in (('initial', 0), ('final', 2048)):
            archive = Archive(run / 'checkpoints' / ('inference-' + str(update) + '.pt'))
            opened.append(archive)
            for names in (('gradient_policy',), ('payload', 'gradient_policy')):
                eq(archive.field(*names).text() == GRADIENT_POLICY, 'Actual inference gradient policy ' + label)
            for prefix in ((), ('payload',)):
                typed_objective(archive, prefix, eq, label + str(prefix))
                typed_pooling(archive, prefix, eq, label + str(prefix))
                eq(archive.field(*(prefix + ('native_layout',))).text() == NATIVE_LAYOUT,
                   'Actual explicit disjoint native layout ' + label)
                for field, expected in (('mean_width', 4), ('uncertainty_width', 4), ('native_width', 40)):
                    eq(archive.field(*(prefix + (field,))).scalar() == expected,
                       'Actual typed disjoint width ' + label + field)
            eq(archive.field('artifact_kind').text() == 'cpb_v9_inference_checkpoint_v1'
               and archive.field('model_tag').text() == 'CPB-v9'
               and archive.field('source_updates').scalar() == 0
               and archive.field('causal_updates').scalar() == update
               and archive.field('initialization_seed').scalar() == 943001
               and archive.field('pretrained_ancestor').text() == 'none',
               'Actual fresh typed inference identity ' + label)
            stages[label] = {
                'model': flat_tensors(archive.field('payload', 'model')),
                'scaler': flat_tensors(archive.field('payload', 'scaler')),
                'bank': flat_tensors(archive.field('payload', 'bank_standardizer')),
            }
        initial, final = stages['initial'], stages['final']
        for family, filename, member in (('scaler', 'input-scaler.pt', 'scaler'),
                                         ('bank', 'bank-standardizer.pt', 'bank')):
            archive = Archive(run / 'calibration' / filename)
            opened.append(archive)
            calibrated = flat_tensors(archive.field(member))
            eq(set(calibrated) == set(initial[family]), 'Actual fresh calibration carried state keys ' + family)
            for name in calibrated:
                eq(tensor_bytes(calibrated[name]) == tensor_bytes(initial[family][name]),
                   'Actual calibration-to-initial statistic equality ' + family + '.' + name)
        parameter_rows = []
        expected_names = {p['name'].removeprefix('core.') for p in parameters}
        eq(expected_names <= set(initial['model']) and set(initial['model']) == set(final['model']),
           'Actual serialized model parameter/buffer inventory')
        for parameter in parameters:
            name = parameter['name'].removeprefix('core.')
            before, after = initial['model'][name], final['model'][name]
            eq(before.shape == after.shape and math.prod(before.shape) == parameter['values']
               and before.storage.dtype == after.storage.dtype == 'FloatStorage',
               'Actual parameter shape/dtype ' + name)
            eq(before.storage.location == after.storage.location == 'cuda:0',
               'Serialized parameter actual CUDA placement ' + name)
            eq(all(math.isfinite(value) for value in before.values() + after.values()),
               'Actual finite parameter state ' + name)
            changed = tensor_bytes(before) != tensor_bytes(after)
            eq(changed == parameter['changed_from_initial'], 'Independent final movement receipt ' + name)
            if not parameter['requires_grad']:
                eq(not changed, 'Literal prior exactness ' + name)
            parameter_rows.append(dict(name=parameter['name'], values=parameter['values'],
                                       learned=parameter['requires_grad'], changed=changed))
        for name in set(initial['model']) - expected_names:
            eq(tensor_bytes(initial['model'][name]) == tensor_bytes(final['model'][name]),
               'Frozen registered buffer exact ' + name)
        for family in ('scaler', 'bank'):
            eq(set(initial[family]) == set(final[family]), 'Frozen state keys ' + family)
            for name in initial[family]:
                eq(tensor_bytes(initial[family][name]) == tensor_bytes(final[family][name]),
                   'Frozen saved statistic exact ' + family + '.' + name)
        typed = []
        for update in (512, 1024, 2048):
            archive = Archive(run / 'checkpoints' / ('training-' + str(update) + '.pt'))
            opened.append(archive)
            eq(archive.field('artifact_kind').text() == 'cpb_v9_causal_training_state_v1'
               and archive.field('model_tag').text() == 'CPB-v9',
               'Actual training artifact type ' + str(update))
            for prefix in ((), ('encoder',), ('encoder', 'payload')):
                typed_objective(archive, prefix, eq, 'training' + str(update) + str(prefix))
                typed_pooling(archive, prefix, eq, 'training' + str(update) + str(prefix))
            eq(archive.field('progress_v1').values() == [update, attempts_by_update[update], 943002, 943003, 943004],
               'Actual CSV-bound three typed streams/counters ' + str(update))
            eq(archive.field('encoder', 'gradient_policy').text() == GRADIENT_POLICY
               and archive.field('encoder', 'payload', 'gradient_policy').text() == GRADIENT_POLICY,
               'Actual typed composite objective gradient policy ' + str(update))
            eq(archive.field('gradient_policy').text() == GRADIENT_POLICY
               and archive.field('native_layout').text() == NATIVE_LAYOUT,
               'Actual training outer explicit policy/layout ' + str(update))
            recipe = archive.field('recipe_identity').text()
            eq(OBJECTIVE_POLICY in recipe and 'delta1;unit-component-weights' in recipe
               and 'reporting-NLL-never-backward' in recipe
               and 'shared-global-clip1-couples-branch-scaling' in recipe,
               'Actual continuation recipe binds composite objective/shared clipping ' + str(update))
            model = flat_tensors(archive.field('encoder', 'payload', 'model'))
            for parameter in parameters:
                name = parameter['name'].removeprefix('core.')
                if not parameter['requires_grad']:
                    eq(tensor_bytes(model[name]) == tensor_bytes(initial['model'][name]),
                       'Literal prior exact at typed update ' + str(update))
                if update == 2048:
                    eq(tensor_bytes(model[name]) == tensor_bytes(final['model'][name]),
                       'Final training/inference parameter equality ' + name)
            for family, member in (('scaler', 'scaler'), ('bank', 'bank_standardizer')):
                actual = flat_tensors(archive.field('encoder', 'payload', member))
                eq(set(actual) == set(initial[family]), 'Typed statistic inventory ' + family + str(update))
                for name in actual:
                    eq(tensor_bytes(actual[name]) == tensor_bytes(initial[family][name]),
                       'Typed frozen statistic ' + family + name + str(update))
            moments = {name: value for name, value in flat_tensors(archive.field('optimizer')).items()
                       if name.endswith(('.exp_avg', '.exp_avg_sq'))}
            eq(len(moments) == 152, 'Actual76 pairs of saved AdamW moments ' + str(update))
            for name, value in moments.items():
                eq(value.storage.location == 'cuda:0'
                   and all(math.isfinite(x) for x in value.values()),
                   'Actual finite CUDA moment ' + name + str(update))
            eq(archive.field('optimizer_state_identity').text().startswith('cpb-v9-optimizer-fnv1a64-'),
               'Typed optimizer content binding ' + str(update))
            typed.append(dict(update=update, moment_tensors=len(moments),
                              checkpoint=pin(run / 'checkpoints' / ('training-' + str(update) + '.pt'))))
        return dict(status='passed' if not failures else 'failed', checks=count, mismatches=failures,
                    independent_parameter_movement=parameter_rows, typed_states=typed,
                    fixed_prior_and_statistics_bitexact=True if not failures else None,
                    method='Inert saved tensor dtype/shape/storage bytes, no module construction')
    except Exception as error:
        return dict(status='blocked_format', checks=count, mismatches=failures,
                    blocker=type(error).__name__ + ': ' + str(error))
    finally:
        for archive in opened:
            archive.close()


def saved_gaussian_audit(run):
    failures, count, rows = [], 0, []

    def eq(ok, reason):
        nonlocal count
        count += 1
        if not ok:
            failures.append(reason)

    def near(a, b, reason):
        eq(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 2e-12 + 2e-12 * abs(b), reason)

    try:
        for stage in ('initial', 'final'):
            queries = read(run / ('query-' + stage + '.json'))
            for role in ('train', 'validation'):
                path = run / 'witnesses' / (stage + '-' + role + '-query.pt')
                archive = Archive(path)
                try:
                    ids = archive.field('row_ids').values()
                    tensors = {key: archive.field(key) for key in
                               ('raw', 'observed', 'hidden', 'visible', 'target_mask',
                                'targets', 'mean', 'variance', 'native', 'row_mae', 'row_nll',
                                'row_mean_huber', 'row_variance_nll', 'row_total_objective')}
                    eq(tensors['raw'].shape == (8, 3, 128, 9)
                       and tensors['raw'].storage.dtype == 'DoubleStorage'
                       and tensors['native'].shape == (8, 40), 'Actual B8 F64/native40 witness')
                    data = {key: value.values() for key, value in tensors.items()}
                    expected_ids = query_anchors(role)[:8]
                    eq(ids == expected_ids, 'Independent firstB8 query rows ' + stage + role)
                    maes, nlls, hubers, totals, cell_counts = [], [], [], [], []
                    for row in range(8):
                        channel_mae, channel_nll, channel_huber, total = [], [], [], 0
                        for channel in range(3):
                            errors, likelihoods, huber_cells = [], [], []
                            start = (row * 3 + channel) * 128 * 9
                            for offset in range(128 * 9):
                                index = start + offset
                                observed, hidden = data['observed'][index], data['hidden'][index]
                                visible, supported = data['visible'][index], data['target_mask'][index]
                                eq(supported == (observed and hidden)
                                   and visible == (observed and not hidden)
                                   and not (visible and supported), 'Exact fixed query/visibility support')
                                if not supported:
                                    continue
                                mean, target, variance = data['mean'][index], data['targets'][index], data['variance'][index]
                                eq(all(math.isfinite(x) for x in (mean, target, variance)) and variance > 0,
                                   'Finite positive Gaussian support')
                                # Python doubles promote stored F32 before subtraction/square/log/division.
                                error = float(mean) - float(target)
                                absolute = abs(error)
                                errors.append(absolute)
                                huber_cells.append(.5 * error * error if absolute < 1 else absolute - .5)
                                likelihoods.append(.5 * (error * error / float(variance) + math.log(float(variance))))
                            if errors:
                                channel_mae.append(math.fsum(errors) / len(errors))
                                channel_nll.append(math.fsum(likelihoods) / len(likelihoods))
                                channel_huber.append(math.fsum(huber_cells) / len(huber_cells))
                                total += len(errors)
                        mae = math.fsum(channel_mae) / len(channel_mae)
                        nll = math.fsum(channel_nll) / len(channel_nll)
                        huber = math.fsum(channel_huber) / len(channel_huber)
                        near(mae, data['row_mae'][row], 'Saved per-row manual MAE')
                        near(nll, data['row_nll'][row], 'Saved per-row manual F64 NLL')
                        near(mae, queries[role]['row_mae'][row], 'FirstB8 summary MAE association')
                        near(nll, queries[role]['row_nll'][row], 'FirstB8 summary NLL association')
                        for name, expected in (('row_mean_huber', huber), ('row_variance_nll', nll),
                                               ('row_total_objective', huber + nll)):
                            near(data[name][row], expected, 'Manual query component ' + stage + role + name)
                            near(queries[role][name][row], expected, 'Query component summary association ' + name)
                        eq(total == 216 == queries[role]['row_target_cells'][row], 'Exact firstB8 target count')
                        maes.append(mae); nlls.append(nll); hubers.append(huber)
                        totals.append(huber + nll); cell_counts.append(total)
                    rows.append(dict(stage=stage, role=role, rows=8, target_cells=sum(cell_counts),
                                     row_mae=maes, row_nll=nlls, row_mean_huber=hubers,
                                     row_variance_nll=nlls, row_total_objective=totals, witness=pin(path)))
                finally:
                    archive.close()
        return dict(status='passed' if not failures else 'failed', checks=count, mismatches=failures,
                    witnesses=rows, scope='First8 rows only per initial/final TRAIN/own fixed-query VALIDATION',
                    reduction='supported cells/channel, eligible channels/example; full-row summaries audited separately',
                    objective=OBJECTIVE_POLICY,
                    limitation='Detached-error and ordinary Gaussian NLL are numerically equal at the same saved forward; routing is source-bound CUDA evidence, not inferable from scalar equality')
    except Exception as error:
        return dict(status='blocked_format', checks=count, mismatches=failures,
                    blocker=type(error).__name__ + ': ' + str(error))


def saved_optimization_component_audit(run, summary, csv_rows):
    """Independent scalar reduction of the actual first/last pre-step forwards.

    No saved weight is evaluated. The error detachment changes derivatives, not
    the forward scalar; CUDA routing is separately bound to its nonzero fixture.
    """
    failures, count, witnesses = [], 0, []

    def eq(ok, reason):
        nonlocal count
        count += 1
        if not ok:
            failures.append(reason)

    def near(actual, expected, reason):
        eq(math.isfinite(actual) and math.isfinite(expected)
           and abs(actual - expected) <= 2e-12 + 2e-12 * abs(expected), reason)

    try:
        bindings = read(run / 'optimization-component-witnesses.json')
        eq(bindings == summary['optimization_component_witnesses'] and len(bindings) == 2,
           'Exact first/last pre-step component witness list')
        for binding, update in zip(bindings, (1, 2048)):
            expected_path = run / 'witnesses' / ('training-update-' + str(update) + '-components.pt')
            eq(binding['update'] == update and binding['pre_step_forward'] is True
               and binding['extra_encoder_calls'] == 0,
               'Declared actual pre-step witness/no extra forward ' + str(update))
            actual_pin = pin(expected_path)
            eq({key: binding[key] for key in ('path', 'bytes', 'sha256')} == actual_pin,
               'Exact actual optimization witness pin ' + str(update))
            archive = Archive(expected_path)
            try:
                row = csv_rows[update - 1]
                eq(archive.field('update').scalar() == update
                   and archive.field('attempt').scalar() == int(row['attempt']),
                   'Actual pre-step witness counters ' + str(update))
                ids = archive.field('row_ids').values()
                eq(ids == list(map(int, row['anchors'].split(';'))),
                   'Actual pre-step rows equal independent CSV schedule ' + str(update))
                times = archive.field('timestamps')
                eq(times.shape == (8, 128), 'Actual timestamp geometry ' + str(update))
                eq(times.values() == [anchor - (127 - tick) * 60000
                                      for anchor in ids for tick in range(128)],
                   'Actual pre-step TRAIN close-time grid ' + str(update))
                keys = ('observed', 'hidden', 'context_deleted', 'visible', 'target_mask',
                        'targets', 'mean', 'variance', 'native', 'eligible_channels',
                        'row_mean_huber', 'row_variance_nll', 'row_total_objective')
                tensors = {key: archive.field(key) for key in keys}
                for name in ('observed', 'hidden', 'context_deleted', 'visible', 'target_mask'):
                    eq(tensors[name].shape == (8, 3, 128, 9)
                       and tensors[name].storage.dtype == 'BoolStorage',
                       'Actual pre-step boolean mask ' + name + str(update))
                for name in ('targets', 'mean', 'variance'):
                    eq(tensors[name].shape == (8, 3, 128, 9)
                       and tensors[name].storage.dtype == 'FloatStorage',
                       'Actual pre-step F32 distribution ' + name + str(update))
                eq(tensors['native'].shape == (8, 40)
                   and tensors['native'].storage.dtype == 'FloatStorage',
                   'Actual pre-step native40 geometry ' + str(update))
                eq(tensors['eligible_channels'].shape == (8, 3)
                   and tensors['eligible_channels'].storage.dtype == 'BoolStorage',
                   'Actual pre-step eligible-channel geometry ' + str(update))
                for name in ('row_mean_huber', 'row_variance_nll', 'row_total_objective'):
                    eq(tensors[name].shape == (8,)
                       and tensors[name].storage.dtype == 'DoubleStorage',
                       'Actual pre-step F64 row component ' + name + str(update))
                data = {key: value.values() for key, value in tensors.items()}
                row_huber, row_variance, supported_cells = [], [], 0
                for example in range(8):
                    channel_huber, channel_variance = [], []
                    for channel in range(3):
                        huber_cells, nll_cells = [], []
                        start = (example * 3 + channel) * 128 * 9
                        for offset in range(128 * 9):
                            index = start + offset
                            observed, hidden = data['observed'][index], data['hidden'][index]
                            deleted, visible = data['context_deleted'][index], data['visible'][index]
                            supported = data['target_mask'][index]
                            eq(supported == (observed and hidden)
                               and visible == (observed and not hidden and not deleted)
                               and (not deleted or (observed and not hidden))
                               and not (visible and supported),
                               'Actual target/context separation ' + str(update) + ':' + str(index))
                            if not supported:
                                continue  # Unsupported values are sanitized away before arithmetic.
                            mean, target = data['mean'][index], data['targets'][index]
                            variance = data['variance'][index]
                            eq(all(math.isfinite(value) for value in (mean, target, variance))
                               and variance > 0, 'Finite supported pre-step distribution')
                            error = float(mean) - float(target)  # F64 operands before subtraction.
                            absolute = abs(error)
                            huber_cells.append(.5 * error * error if absolute < 1 else absolute - .5)
                            nll_cells.append(.5 * (error * error / float(variance) + math.log(float(variance))))
                        eligible = data['eligible_channels'][example * 3 + channel]
                        eq(bool(huber_cells) == eligible, 'Exact eligible-channel support ' + str(update))
                        if eligible:
                            supported_cells += len(huber_cells)
                            channel_huber.append(math.fsum(huber_cells) / len(huber_cells))
                            channel_variance.append(math.fsum(nll_cells) / len(nll_cells))
                    eq(bool(channel_huber), 'Every actual pre-step example eligible ' + str(update))
                    huber = math.fsum(channel_huber) / len(channel_huber)
                    variance_nll = math.fsum(channel_variance) / len(channel_variance)
                    near(data['row_mean_huber'][example], huber, 'Manual pre-step row Huber1')
                    near(data['row_variance_nll'][example], variance_nll, 'Manual pre-step row variance NLL')
                    near(data['row_total_objective'][example], huber + variance_nll,
                         'Manual pre-step row composite sum')
                    row_huber.append(huber)
                    row_variance.append(variance_nll)
                means = dict(mean_Huber_F64=math.fsum(row_huber) / 8,
                             variance_NLL_F64=math.fsum(row_variance) / 8)
                means['total_objective_F64'] = means['mean_Huber_F64'] + means['variance_NLL_F64']
                means['Gaussian_NLL_F64'] = means['variance_NLL_F64']
                for name, expected in means.items():
                    tensor = archive.field(name)
                    eq(tensor.storage.dtype == 'DoubleStorage' and tensor.shape == (),
                       'Actual F64 pre-step batch scalar ' + name)
                    near(tensor.scalar(), expected, 'Manual pre-step batch scalar ' + name + str(update))
                    near(float(row[name]), expected, 'Actual CSV/pre-step scalar ' + name + str(update))
                eq(supported_cells == int(row['target_cells']) == 6912,
                   'Actual pre-step equal-example/channel/cell support ' + str(update))
                for name in ('pre_clip_global_norm', 'clip_factor'):
                    near(archive.field(name).scalar(), float(row[name]),
                         'Exact recorded witness/CSV clipping value ' + name + str(update))
                witnesses.append(dict(update=update, batch_size=8, target_cells=supported_cells,
                                      manual_F64_components=means, witness=actual_pin))
            finally:
                archive.close()
        return dict(status='passed' if not failures else 'failed', checks=count, mismatches=failures,
                    witnesses=witnesses, objective_policy=OBJECTIVE_POLICY,
                    model_calls=0, extra_training_updates=0,
                    reduction='supported cells/channel; eligible channels/example; eligible examples/batch',
                    limitations=['Actual pre-step provenance and CUDA gradient routing are source-bound producer evidence.',
                                 'No full parameter-gradient arrays are saved: pre-clip global norm is observed evidence, not independently reconstructed.',
                                 'Shared clipping and AdamW couple optimization; direct separated derivatives do not establish optimizer independence.'])
    except Exception as error:
        return dict(status='blocked_format', checks=count, mismatches=failures,
                    blocker=type(error).__name__ + ': ' + str(error))


def temporal_learning_audit(run, summary, parameters):
    """Reduce observed CSV evidence and bounded saved arrays; never evaluate weights.

    All8192 subgroup,2048 gauge and2048 between-bin rows are checked. Only
    updates1/2048 carry raw clipped gradient arrays, so other per-step norms are
    observed evidence, not a reconstruction of unavailable upstream gradients.
    """
    failures, count, opened = [], 0, []

    def eq(ok, why):
        nonlocal count
        count += 1
        if not ok:
            failures.append(why)

    def near(actual, expected, why):
        eq(math.isfinite(actual) and math.isfinite(expected)
           and abs(actual - expected) <= 2e-12 + 2e-12 * abs(expected), why)

    def norm(values):
        return math.sqrt(math.fsum(value * value for value in values))

    def center(values):
        mean = math.fsum(values) / len(values)
        return [value - mean for value in values]

    def parseval(coefficients, projection_norm, why):
        squared = projection_norm * projection_norm
        independent = math.fsum(value * value for value in coefficients)
        eq(math.isfinite(squared) and math.isfinite(independent)
           and abs(independent - squared) <= 1e-12 + 1e-10 * abs(squared), why)

    def between(values, why):
        # Unit orthonormal R16 basis: four repeated signs per original bin /4.
        # Dot products are independently contracted over all16 F64 operands,
        # not read from a producer projection or unsaved intermediate tensor.
        values = [float(value) for value in values]
        eq(len(values) == 16 and all(math.isfinite(v) for v in values), why + ' finite16 values')
        coefficients = [math.fsum(signs[i // 4] * value / 4.
                                  for i, value in enumerate(values)) for signs in BETWEEN_SIGNS]
        means = [math.fsum(values[4*j:4*(j+1)]) / 4. for j in range(4)]
        projection_norm = math.sqrt(4. * math.fsum(value * value for value in center(means)))
        parseval(coefficients, projection_norm, why + ' full-position Parseval')
        return coefficients, projection_norm

    def tensor(archive, key, shape):
        t = archive.field(key)
        eq(t.shape == shape and t.storage.dtype == 'FloatStorage'
           and t.storage.location == 'cpu', 'Actual saved CPU F32 temporal array ' + key)
        values = t.values()
        eq(all(math.isfinite(value) for value in values), 'Finite temporal array ' + key)
        return t, values

    columns = ['update', 'bin', 'export_row_begin', 'export_row_end', 'position_begin', 'position_end',
               'weight_gradient_l2', 'bias_gradient_l2', 'weight_movement_l2', 'bias_movement_l2',
               'nonzero_export_gradient_values', 'moved_export_values', 'pool_position_centered_gradient_l2',
               'pool_position_centered_movement_l2', 'nonzero_centered_position_gradient_values',
               'moved_centered_position_values']
    gauge_columns = ['update', 'gradient_defined', 'clipped_data_gradient', 'step_movement', 'weight_before', 'weight_after']
    between_columns = ['update', 'between_bin_gradient_l2', 'between_bin_movement_l2',
                       'contrast_0_gradient', 'contrast_1_gradient', 'contrast_2_gradient',
                       'contrast_0_movement', 'contrast_1_movement', 'contrast_2_movement']
    count_fields = ['weight_nonzero_data_gradient_steps', 'bias_nonzero_data_gradient_steps',
                    'weight_movement_steps', 'bias_movement_steps',
                    'position_centered_data_gradient_steps', 'position_centered_movement_steps']
    norm_fields = ['weight_gradient_l2', 'bias_gradient_l2', 'weight_movement_l2', 'bias_movement_l2',
                   'pool_position_centered_gradient_l2', 'pool_position_centered_movement_l2']
    try:
        receipt_path = run / 'temporal-projection-learning.json'
        receipt = read(receipt_path)
        eq(summary['temporal_pooling_learning'] == pin(receipt_path), 'Training summary binds exact temporal receipt')
        eq(receipt['kind'] == 'cpb_v9_actual_temporal_pooling_learning_v1'
           and receipt['status'] == 'passed' and receipt['updates'] == 2048 and receipt['group_rows'] == 8192
           and receipt['temporal_pooling_policy'] == POOL_POLICY
           and receipt['temporal_pooling_partition'] == POOL_PARTITION
           and receipt['temporal_pooling_mix'] == POOL_MIX
           and receipt['empty_local_bin_policy'] == EMPTY_LOCAL_POLICY
           and receipt['movement_includes_AdamW_weight_decay'] is True
           and receipt['weight_decay_counted_as_data_gradient'] is False
           and receipt['extra_encoder_calls'] == 0, 'Actual policy/event receipt and honest decay scope')
        eq(receipt['artifacts'] == [pin(run / name) for name in
                                   ('temporal-projection-learning.csv', 'temporal-pool-score-bias-learning.csv',
                                    'temporal-between-bin-position-learning.csv')],
           'All three temporal CSV files pinned by receipt')
        with (run / 'temporal-projection-learning.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            eq(reader.fieldnames == columns, 'Exact temporal group CSV schema')
            rows = list(reader)
        with (run / 'temporal-pool-score-bias-learning.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            eq(reader.fieldnames == gauge_columns, 'Exact observed gauge CSV schema')
            gauge_rows = list(reader)
        with (run / 'temporal-between-bin-position-learning.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            eq(reader.fieldnames == between_columns, 'Exact nine-column between-bin observation CSV')
            between_rows = list(reader)
        eq(len(rows) == 8192 and len(gauge_rows) == len(between_rows) == 2048,
           'All2048x4 groups,2048 gauge and2048 between-bin rows')
        between_gradient_steps, between_movement_steps = [0, 0, 0], [0, 0, 0]
        for update, row in enumerate(between_rows, 1):
            eq(int(row['update']) == update, 'Exact successful between-bin update order ' + str(update))
            values = {key: float(row[key]) for key in between_columns[1:]}
            eq(all(math.isfinite(value) for value in values.values())
               and values['between_bin_gradient_l2'] >= 0 and values['between_bin_movement_l2'] >= 0,
               'Finite signed coefficients/nonnegative observed projection norms ' + str(update))
            for suffix, field, events in (('gradient', 'between_bin_gradient_l2', between_gradient_steps),
                                          ('movement', 'between_bin_movement_l2', between_movement_steps)):
                coefficients = [values['contrast_' + str(j) + '_' + suffix] for j in range(3)]
                parseval(coefficients, values[field], 'Recorded Hadamard coefficient/norm identity ' + str(update) + suffix)
                eq((values[field] > 0) == any(value != 0 for value in coefficients),
                   'Recorded projection zero-event consistency ' + str(update) + suffix)
                for j, value in enumerate(coefficients):
                    events[j] += value != 0
        between_receipt = receipt['between_bin_positions']
        eq(set(between_receipt) == {'basis', 'basis_signs', 'full_position_divisor', 'position_bin_width',
               'rows', 'gradient_steps', 'movement_steps', 'final_contrast_changes', 'final_projection_change_l2',
               'gradient_definition', 'movement_includes_AdamW_weight_decay',
               'weight_decay_counted_as_data_learning', 'extra_encoder_calls'},
           'Exact declared between-bin receipt keys')
        eq(between_receipt['basis'] == BETWEEN_BASIS and between_receipt['basis_signs'] == BETWEEN_SIGNS
           and between_receipt['full_position_divisor'] == between_receipt['position_bin_width'] == 4
           and between_receipt['rows'] == 2048
           and between_receipt['gradient_definition'] == 'observed-post-globalclip-data-gradient'
           and between_receipt['movement_includes_AdamW_weight_decay'] is True
           and between_receipt['weight_decay_counted_as_data_learning'] is False
           and between_receipt['extra_encoder_calls'] == 0,
           'Exact full-position Hadamard basis and honest observation/decay contract')
        eq(between_receipt['gradient_steps'] == between_gradient_steps
           and between_receipt['movement_steps'] == between_movement_steps
           and all(0 < count <= 2048 for count in between_gradient_steps + between_movement_steps),
           'Each restored direction has independently reduced gradient/movement EVENTS')
        totals = [collections.Counter() for _ in range(4)]
        for index, row in enumerate(rows):
            update, group = index // 4 + 1, index % 4
            eq([int(row[key]) for key in columns[:6]] == [update, group, group * 8, (group + 1) * 8, group * 4, (group + 1) * 4],
               'Exact temporal update/bin/original-row membership ' + str(index))
            numbers = [float(row[key]) for key in norm_fields]
            eq(all(math.isfinite(value) and value >= 0 for value in numbers), 'Finite nonnegative observed subgroup norms ' + str(index))
            for field, value in zip(count_fields, numbers):
                totals[group][field] += value > 0
            counts = [int(row[key]) for key in ('nonzero_export_gradient_values', 'moved_export_values',
                                               'nonzero_centered_position_gradient_values', 'moved_centered_position_values')]
            eq(all(0 <= value <= maximum for value, maximum in zip(counts, (520, 520, 4, 4))),
               'Observed nonzero/moved value counts in actual slice capacities')
            eq((counts[0] > 0) == (numbers[0] > 0 or numbers[1] > 0)
               and (counts[1] > 0) == (numbers[2] > 0 or numbers[3] > 0)
               and (counts[2] > 0) == (numbers[4] > 0) and (counts[3] > 0) == (numbers[5] > 0),
               'Norm/count zero-event arithmetic ' + str(index))
        initial = Archive(run / 'checkpoints/inference-0.pt')
        final = Archive(run / 'checkpoints/inference-2048.pt')
        opened.extend((initial, final))
        state0 = flat_tensors(initial.field('payload', 'model'))
        state1 = flat_tensors(final.field('payload', 'model'))
        initial_positions = state0[POOL_POSITIONS.removeprefix('core.')]
        final_positions = state1[POOL_POSITIONS.removeprefix('core.')]
        eq(initial_positions.shape == final_positions.shape == (16, 1)
           and initial_positions.storage.dtype == final_positions.storage.dtype == 'FloatStorage',
           'Pinned initial/final sixteen original position parameters')
        final_delta = [float(end) - float(start) for start, end in
                       zip(initial_positions.values(), final_positions.values())]
        final_coefficients, final_projection = between(final_delta, 'Final checkpoint between-bin displacement')
        eq(len(between_receipt['final_contrast_changes']) == 3,
           'Three final signed coefficients are recorded descriptively')
        for got, expected in zip(between_receipt['final_contrast_changes'], final_coefficients):
            near(got, expected, 'Independent signed final Hadamard displacement')
        for j, coefficient in enumerate(final_coefficients):
            near(math.fsum(float(row['contrast_' + str(j) + '_movement']) for row in between_rows),
                 coefficient, 'Observed2048 signed contrast movements telescope to saved final change')
        near(between_receipt['final_projection_change_l2'], final_projection,
             'Independent initial/final overall between-bin projection change')
        eq(final_projection > 0, 'Positive OVERALL final projection change, not each final coefficient')
        parameter_fields = [('temporal_export_weight', EXPORT_WEIGHT, (32, 64)),
                            ('temporal_export_bias', EXPORT_BIAS, (32,)),
                            ('temporal_pool_positions', POOL_POSITIONS, (16, 1)),
                            ('temporal_pool_score_bias', GAUGE, (1,))]
        by_name = {p['name']: p for p in parameters}
        for group, actual in enumerate(receipt['groups']):
            eq([actual[key] for key in ('bin', 'export_row_begin', 'export_row_end', 'position_begin', 'position_end')]
               == [group, group * 8, (group + 1) * 8, group * 4, (group + 1) * 4], 'Receipt exact four original-time groups')
            for field in count_fields:
                eq(actual[field] == totals[group][field] and 0 < totals[group][field] <= 2048,
                   'Independent observed event reduction ' + str(group) + field)
            for name, width, flag in ((EXPORT_WEIGHT, 64, 'final_weight_changed'), (EXPORT_BIAS, 1, 'final_bias_changed')):
                a, b = state0[name.removeprefix('core.')].values(), state1[name.removeprefix('core.')].values()
                changed = a[group * 8 * width:(group + 1) * 8 * width] != b[group * 8 * width:(group + 1) * 8 * width]
                eq(changed and actual[flag] is True, 'Pinned initial/final eight-row export movement ' + str(group) + name)
            a = state0[POOL_POSITIONS.removeprefix('core.')].values()[group * 4:(group + 1) * 4]
            b = state1[POOL_POSITIONS.removeprefix('core.')].values()[group * 4:(group + 1) * 4]
            displacement = norm([y - x for x, y in zip(center(a), center(b))])
            near(actual['final_position_centered_change_l2'], displacement, 'Independent final centered position contrast movement')
            eq(displacement > 0, 'Each bin has final identifiable position contrast movement')
        eq(len(receipt['groups']) == 4, 'Exactly four ordered temporal receipt groups')
        gauge = receipt['gauge']
        eq(gauge['name'] == GAUGE and gauge['data_gradient_requirement_exempt'] is True
           and gauge['registered_in_same_AdamW'] is True and gauge['movement_is_not_data_learning_proof'] is True,
           'Only declared scalar gauge exemption, optimizer membership retained')
        gauge_counts = collections.Counter()
        previous = state0[GAUGE.removeprefix('core.')].scalar()
        gradients, signed_movements = [], []
        for update, row in enumerate(gauge_rows, 1):
            defined = int(row['gradient_defined'])
            g, move, before, after = (float(row[name]) for name in gauge_columns[2:])
            eq(int(row['update']) == update and defined in (0, 1)
               and all(math.isfinite(v) for v in (g, move, before, after)), 'Finite monotone observed gauge row ' + str(update))
            eq(before == previous and move == after - before and (defined == 1 or g == 0),
               'Gauge step continuity and signed displacement ' + str(update))
            eq(all(struct.unpack('<f', struct.pack('<f', v))[0] == v for v in (g, before, after)),
               'Observed gauge values are actual promoted F32 scalars')
            previous = after; gradients.append(g); signed_movements.append(move)
            gauge_counts['gradient_defined_steps'] += defined
            gauge_counts['nonzero_recorded_gradient_steps'] += g != 0
            gauge_counts['movement_steps'] += move != 0
        eq(previous == state1[GAUGE.removeprefix('core.')].scalar(), 'Gauge last step equals pinned final checkpoint')
        for field, value in gauge_counts.items():
            eq(gauge[field] == value, 'Independent gauge event reduction ' + field)
        near(gauge['maximum_absolute_recorded_gradient'], max(map(abs, gradients)), 'Gauge maximum observed absolute data gradient')
        near(gauge['final_movement'], math.fsum(signed_movements), 'Gauge telescoping signed movement')
        eq(by_name[GAUGE]['values'] == 1 and by_name[GAUGE]['requires_grad'] is True
           and by_name[GAUGE]['data_gradient_requirement_exempt'] is True
           and by_name[GAUGE]['nonzero_CUDA_gradient_steps'] == gauge_counts['nonzero_recorded_gradient_steps']
           and by_name[GAUGE]['weight_movement_steps'] == gauge_counts['movement_steps'],
           'Parameter receipt matches observed gauge; no positive learning bound')
        witnessed = []
        for update in (1, 2048):
            path = run / 'witnesses' / ('training-update-' + str(update) + '-components.pt')
            archive = Archive(path);opened.append(archive)
            eq(archive.field('update').scalar() == update, 'Actual temporal witness update association')
            arrays = {}
            for key, canonical, shape in parameter_fields:
                arrays[key] = {stage: tensor(archive, key + '_' + stage, shape)
                               for stage in ('before', 'after', 'clipped_gradient')}
                if update == 1:
                    eq(tensor_bytes(arrays[key]['before'][0]) == tensor_bytes(state0[canonical.removeprefix('core.')]),
                       'Actual first pre-step pooling parameter equals typed initialization ' + key)
                else:
                    eq(tensor_bytes(arrays[key]['after'][0]) == tensor_bytes(state1[canonical.removeprefix('core.')]),
                       'Actual last post-step pooling parameter equals typed final checkpoint ' + key)
            for group in range(4):
                row = rows[(update - 1) * 4 + group]
                wg = arrays['temporal_export_weight']['clipped_gradient'][1][group * 512:(group + 1) * 512]
                bg = arrays['temporal_export_bias']['clipped_gradient'][1][group * 8:(group + 1) * 8]
                # Production promotes each saved F32 operand to F64 BEFORE
                # subtracting; Python float implements that exact operand route.
                wm = [b - a for a, b in zip(arrays['temporal_export_weight']['before'][1][group * 512:(group + 1) * 512], arrays['temporal_export_weight']['after'][1][group * 512:(group + 1) * 512])]
                bm = [b - a for a, b in zip(arrays['temporal_export_bias']['before'][1][group * 8:(group + 1) * 8], arrays['temporal_export_bias']['after'][1][group * 8:(group + 1) * 8])]
                pg = center(arrays['temporal_pool_positions']['clipped_gradient'][1][group * 4:(group + 1) * 4])
                pm = center([b - a for a, b in zip(arrays['temporal_pool_positions']['before'][1][group * 4:(group + 1) * 4], arrays['temporal_pool_positions']['after'][1][group * 4:(group + 1) * 4])])
                for key, values in zip(norm_fields, (wg, bg, wm, bm, pg, pm)):
                    near(float(row[key]), norm(values), 'Independent actual clipped-gradient/movement norm ' + str(update) + ':' + str(group) + key)
                eq([int(row[key]) for key in ('nonzero_export_gradient_values', 'moved_export_values', 'nonzero_centered_position_gradient_values', 'moved_centered_position_values')]
                   == [sum(v != 0 for v in wg + bg), sum(v != 0 for v in wm + bm), sum(v != 0 for v in pg), sum(v != 0 for v in pm)],
                   'Independent raw-array nonzero/movement counts')
            row = gauge_rows[update - 1]
            for stage, column in (('before', 'weight_before'), ('after', 'weight_after'), ('clipped_gradient', 'clipped_data_gradient')):
                eq(arrays['temporal_pool_score_bias'][stage][1] == [float(row[column])], 'Actual gauge array equals CSV ' + str(update) + stage)
            between_row = between_rows[update - 1]
            position_gradient = [float(value) for value in arrays['temporal_pool_positions']['clipped_gradient'][1]]
            position_movement = [float(end) - float(start) for start, end in
                                 zip(arrays['temporal_pool_positions']['before'][1],
                                     arrays['temporal_pool_positions']['after'][1])]
            actual_between = {}
            for suffix, vector, field in (('gradient', position_gradient, 'between_bin_gradient_l2'),
                                          ('movement', position_movement, 'between_bin_movement_l2')):
                coefficients, projection_norm = between(vector, 'Actual two-witness ' + str(update) + suffix)
                near(float(between_row[field]), projection_norm,
                     'Independent raw-array between projection norm ' + str(update) + suffix)
                for j, coefficient in enumerate(coefficients):
                    near(float(between_row['contrast_' + str(j) + '_' + suffix]), coefficient,
                         'Independent raw-array signed contrast ' + str(update) + ':' + str(j) + suffix)
                actual_between[suffix] = dict(coefficients=coefficients, full_position_projection_l2=projection_norm)
            witnessed.append(dict(update=update, artifact=pin(path), raw_parameter_array_count=12,
                                  between_bin_actual_arrays=actual_between))
        return dict(status='passed' if not failures else 'failed', checks=count, mismatches=failures,
                    temporal_pooling_policy=POOL_POLICY, temporal_pooling_partition=POOL_PARTITION,
                    temporal_pooling_mix=POOL_MIX, empty_local_bin_policy=EMPTY_LOCAL_POLICY,
                    group_rows_checked=len(rows), gauge_rows_checked=len(gauge_rows), group_event_counts=[dict(v) for v in totals],
                    actual_raw_array_witnesses=witnessed, final_projection_rows_and_position_contrasts_verified=True if not failures else None,
                    between_bin_positions=dict(rows_checked=len(between_rows), basis=BETWEEN_BASIS,
                        basis_signs=BETWEEN_SIGNS, full_position_divisor=4,
                        gradient_event_counts=between_gradient_steps, movement_event_counts=between_movement_steps,
                        final_signed_contrast_changes=final_coefficients,
                        final_projection_change_l2=final_projection,
                        individual_final_coefficient_nonzero_gate=False,
                        actual_raw_array_updates=[1, 2048], other2046_raw_gradients_reconstructed=False),
                    gauge_observed=dict(gauge), receipt=pin(receipt_path), model_calls=0, head_fits=0,
                    limits=['All2048 per-group norms are observed post-globalclip producer evidence; independent raw-array reconstruction is bounded to actual updates1/2048.',
                            'Movement includes AdamW decay; only separately observed nonzero data gradients establish data learning. The scalar gauge has no nonzero-gradient requirement.',
                            'Hadamard Parseval across2048 CSV rows checks recorded scalars, not unsaved2046 raw gradients; only actual1/2048 arrays reconstruct those projections.',
                            'Each direction needs gradient/movement events; only overall final projection must be positive. Signed final coefficients may be zero.',
                            'No scalar-parameter identifiability, unavailable upstream-gradient reconstruction, causal replay or optimizer-independence claim.'])
    except Exception as error:
        return dict(status='blocked_format', checks=count, mismatches=failures,
                    blocker=type(error).__name__ + ': ' + str(error))
    finally:
        for archive in opened:
            archive.close()


def query_anchors(role):
    begin, end, n = ((1704067259999, 1705881599999, 112) if role == 'train'
                     else (1705881659999, 1706486399999, 32))
    first, stride = begin + 255 * 60000, 256 * 60000
    available = (end - first) // stride + 1
    return [first + (i * (available - 1) // (n - 1)) * stride for i in range(n)]

def main():
    global CARD_SHA, MD_SHA, INPUT_SHA
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('run', 'attempt', 'source', 'data-root', 'audit-card',
                   'engineering-proof', 'gradient-proof', 'temporal-proof', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('Execute saved-data audit only in the existing managed container')
    if args.output.exists() or not args.output.parent.is_dir():
        raise RuntimeError('Existing parent and exclusive new audit output required; preserve prior attempts')
    run, attempt, source, data = (p.resolve(strict=True) for p in (args.run, args.attempt, args.source, args.data_root))
    audit_card = read(args.audit_card)
    CARD_SHA = audit_card['frozen_training_card_sha256']
    MD_SHA = audit_card['frozen_training_markdown_sha256']
    INPUT_SHA = audit_card['frozen_input_manifest_sha256']
    main_source_files = audit_card['main_source_files']
    ck(audit_card['kind'] == 'cpb_v9_saved_training_independent_audit_card_v1'
       and audit_card['stage'] == 'declared_before_independent_saved_data_execution'
       and all(isinstance(v, str) and len(v) == 64 and set(v) <= set('0123456789abcdef')
               for v in (CARD_SHA, MD_SHA, INPUT_SHA))
       and isinstance(main_source_files, int) and main_source_files > 0,
       'Frozen independent training-audit declaration and explicit source pins')
    own = audit_card['auditor_source']
    ck(Path(__file__).stat().st_size == own['bytes'] and sha(__file__) == own['sha256'],
       'Exact independently declared auditor source')
    for binding in audit_card['auditor_support']:
        path = Path(__file__).parent / Path(binding['path']).name
        ck(path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'],
           'Exact independent auditor support ' + binding['path'])
    ck(str(run) == audit_card['expected_main_paths']['run']
       and str(attempt) == audit_card['expected_main_paths']['attempt']
       and str(source) == audit_card['expected_main_paths']['source'],
       'Prospectively selected actual main attempt/run/source')
    ck(sha(source / 'CAPSULE_MANIFEST.json') == audit_card['main_source_manifest_sha256']
       and sha(attempt / 'source.zip') == audit_card['main_source_zip_sha256'],
       'Exact predeclared immutable V9 source capsule')
    ck(sha(source / 'doc/CAUSAL_FRESH_V9_CARD.json') == CARD_SHA
       and sha(source / 'doc/CAUSAL_FRESH_V9_CARD.md') == MD_SHA
       and sha(source / 'doc/causal_fresh_v9/input-manifest.json') == INPUT_SHA,
       'Actual prospective card/Markdown/data declaration before any saved numeric access')
    frozen_main_card = read(source / 'doc/CAUSAL_FRESH_V9_CARD.json')
    ck(frozen_main_card['model'] == 'CPB-v9' and frozen_main_card['instance'] == 'CPB-v9.market-01'
       and 'DRAFT' not in frozen_main_card['stage'] and bool(frozen_main_card['frozen_utc'])
       and frozen_main_card['mechanism']['temporal_pooling_policy'] == POOL_POLICY
       and frozen_main_card['mechanism']['temporal_pooling_partition'] == POOL_PARTITION
       and frozen_main_card['mechanism']['temporal_pooling_mix'] == POOL_MIX
       and frozen_main_card['mechanism']['empty_local_bin_policy'] == EMPTY_LOCAL_POLICY,
       'Immutable prospectively frozen V9 mechanism before saved numeric access')
    admitted_manifest = read(source / 'CAPSULE_MANIFEST.json')
    ck(len(admitted_manifest['files']) == main_source_files
       and admitted_manifest['card_sha256'] == CARD_SHA,
       'Exact independently pinned main closure before saved numeric access')
    for binding in admitted_manifest['files']:
        path = source / binding['path']
        ck(path.is_file() and not path.is_symlink() and path.stat().st_size == binding['bytes']
           and sha(path) == binding['sha256'], 'Pre-arithmetic admitted source exact ' + binding['path'])
    prior_failures = []
    for parent in audit_card['preserved_parent_sources']:
        ck(sha(Path(parent['container_path'])) == parent['sha256']
           and Path(parent['container_path']).stat().st_size == parent['bytes'],
           'Preserved historical parent auditor source ' + parent['path'])
    if bad:
        raise RuntimeError('Pre-arithmetic source/card admission refused: ' + '; '.join(bad))
    summary = read(run / 'training-summary.json')
    parameters = read(run / 'parameter-learning.json')
    initial = read(run / 'fresh-initialization.json')
    resume = read(run / 'typed-resume-512.json')
    calibration = read(run / 'calibration/summary.json')
    expected = dict(status='passed', model='CPB-v9.market-01', source_updates=0,
                    gradient_policy=GRADIENT_POLICY, encoder_gradient_kind='Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1',
                    conditional_variance_head_trainable=True,
                    new_causal_updates=2048, initialization_seed=943001,
                    batch_size=8, TRAIN_windows=4096, probe_TRAIN_windows=112,
                    probe_VALIDATION_windows=32, native_width=40, device='cuda:0',
                    parameters=298640,
                    fixed_prior_parameters=3888, typed_resume_512_exact=True,
                    fixed_prior_scaler_normalizer_unchanged=True, new_scaler_fits=1,
                    new_normalizer_fits=1, calibration_encoder_calls=0,
                    calibration_validation_rows=0, old_model_artifacts_opened=0,
                    old_normalization_artifacts_opened=0, saved_quality_features_reused=False,
                    CPU_encoder_calls=0, old_encoder_calls=0, head_fits=0, training_replays=0)
    for key, value in expected.items():
        ck(summary[key] == value, 'Training summary ' + key)
    json_objective(summary, 'Training summary')
    json_objective(initial, 'Fresh initialization')
    json_pooling(summary, 'Training summary')
    json_pooling(initial, 'Fresh initialization')
    json_pooling(resume, 'Actual typed512 continuation')
    ck('gradient_policy' not in initial,
       'Fresh initialization legitimately has no gradient-policy selector; typed artifacts remain strict')
    ck(294751 <= summary['learned_tensor_value_capacity_with_recorded_movement_events'] <= 294752
       and summary['all_mandatory_identifiable_tensors_learned'] is True
       and summary['mandatory_identifiable_tensor_value_capacity'] == 294751
       and summary['mandatory_identifiable_count_semantics'] == 'tensor numel capacity, not every-scalar gradient/movement proof'
       and summary['per_scalar_nonzero_gradient_proof_claimed'] is False
       and summary['recorded_movement_capacity_semantics'] == 'sum of numel of learned tensors with at least one recorded movement event; not a count of individually moved scalar values'
       and 'parameters_with_recorded_updates' not in summary
       and 'mandatory_identifiable_values_with_data_gradients_and_movement' not in summary,
       '75 mandatory tensors with294751 numel capacity; no per-scalar gradient proof')
    ck(summary['all_four_temporal_export_row_groups_learned'] is True
       and summary['all_four_pool_position_contrasts_learned'] is True
       and summary['all_three_between_bin_position_directions_learned'] is True
       and summary['final_between_bin_position_projection_changed'] is True,
       'Declared actual temporal learning flags, independently checked below')
    ck(summary['backward_tensor'] == 'objective.total_loss'
       and summary['shared_global_clip'] is True
       and summary['optimizer_independence_claimed'] is False,
       'Composite backward tensor and disclosed shared optimizer coupling')
    learned = [p for p in parameters if p['requires_grad']]
    fixed = [p for p in parameters if not p['requires_grad']]
    ck(len(parameters) == len({p['name'] for p in parameters}) == 77 and len(learned) == 76 and len(fixed) == 1, '77 tensors:76 learned,one fixed')
    ck(sum(p['values'] for p in learned) == 294752 and sum(p['values'] for p in fixed) == 3888, 'Exact learned/fixed partition')
    ck(sum(p['values'] for p in learned if p['weight_movement_steps'] > 0) == summary['learned_tensor_value_capacity_with_recorded_movement_events'],
       'Numel capacity of tensors with recorded movement, not individually moved scalar count')
    groups = collections.defaultdict(lambda: dict(tensors=0, values=0, nonzero_gradient_tensors=0,
                                                 moved_tensors=0, changed_final_tensors=0))
    for p in parameters:
        ck(p['data_gradient_requirement_exempt'] == (p['name'] == GAUGE),
           'Only declared one-value pooling score-bias gauge exempt ' + p['name'])
        if p['requires_grad']:
            if p['name'] == GAUGE:
                ck(p['values'] == 1 and 0 <= p['nonzero_CUDA_gradient_steps'] <= 2048
                   and 0 <= p['weight_movement_steps'] <= 2048,
                   'Observed gauge values; no nonzero or data-learning lower bound')
            else:
                ck(0 < p['nonzero_CUDA_gradient_steps'] <= 2048 and
                   0 < p['weight_movement_steps'] <= 2048, 'Actual mandatory learned evidence ' + p['name'])
            g = groups[p['group']]
            g['tensors'] += 1
            g['values'] += p['values']
            g['nonzero_gradient_tensors'] += p['nonzero_CUDA_gradient_steps'] > 0
            g['moved_tensors'] += p['weight_movement_steps'] > 0
            g['changed_final_tensors'] += p['changed_from_initial']
        else:
            ck(p['nonzero_CUDA_gradient_steps'] == p['weight_movement_steps'] == 0 and not p['changed_from_initial'], 'Literal prior protected')
    ck(len(groups) == 9, 'Nine learned categories')
    uncertainty = [p for p in learned if p['name'] == 'core.uncertainty_projection']
    ck(len(uncertainty) == 1 and uncertainty[0]['values'] == 1080,
       'New independent bias-free4x270 uncertainty projection')
    for group in summary['parameter_groups']:
        ck({k: v for k, v in group.items() if k != 'group'} == groups[group['group']], 'Recomputed learning group ' + group['group'])
    selected = anchors()
    selected_set = set(selected)
    csv_fields = ['attempt', 'update', 'Gaussian_NLL_F64', 'mean_Huber_F64',
                  'variance_NLL_F64', 'total_objective_F64', 'pre_clip_global_norm',
                  'clip_factor', 'eligible_examples', 'target_cells',
                  'nonzero_gradient_tensors', 'changed_tensors', 'row_seed',
                  'mask_seed', 'anchors']
    with (run / 'training.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        ck(reader.fieldnames == csv_fields, 'Exact declared successful/skipped CSV schema')
        all_rows = list(reader)
    attempted = summary['attempted_batches']
    ck(type(attempted) is int and 2048 <= attempted <= 4096
       and len(all_rows) == attempted, 'Actual attempt count within declared4096 limit')
    rows, skipped_rows, attempts_by_update = [], [], {}
    completed = 0
    for attempt_index, row in enumerate(all_rows):
        ck(set(row) == set(csv_fields) and int(row['attempt']) == attempt_index,
           'Dense unique zero-based attempted-batch counter ' + str(attempt_index))
        if row['Gaussian_NLL_F64'] == '':
            ck(int(row['update']) == completed
               and all(row[name] == '' for name in csv_fields[2:]),
               'Skipped ineligible attempt has no loss/update/learning evidence ' + str(attempt_index))
            skipped_rows.append(attempt_index)
            continue
        completed += 1
        ck(int(row['update']) == completed, 'Exactly one completed update per supported row ' + str(attempt_index))
        rows.append(row)
        attempts_by_update[completed] = attempt_index + 1
    ck(completed == len(rows) == 2048 and len(skipped_rows) == attempted - 2048
       and attempts_by_update[2048] == attempted,
       'Exactly2048 successful updates with explicit skipped-attempt accounting and no replay')
    gradient_events = movement_events = cells = 0
    context_stream_hash = hashlib.sha256()
    for update_index, row in enumerate(rows):
        attempt_index = int(row['attempt'])
        ck(int(row['update']) == update_index + 1, 'Monotone successful counters ' + str(attempt_index))
        row_stream, mask_stream = counter(943002, attempt_index), counter(943003, attempt_index)
        ck(int(row['row_seed']) == row_stream and int(row['mask_seed']) == mask_stream, 'Actual two CSV streams ' + str(attempt_index))
        context_stream_hash.update(struct.pack('<q', counter(943004, attempt_index)))
        rng = MT19937_64(row_stream)
        expected_rows = [selected[(rng.next() * 4096) >> 64] for _ in range(8)]
        actual_rows = list(map(int, row['anchors'].split(';')))
        ck(actual_rows == expected_rows, 'Independent MT19937_64 choices ' + str(attempt_index))
        ck(len(actual_rows) == 8 and all(x in selected_set and x - 127 * 60000 >= 1704067259999 and x <= 1705881599999 for x in actual_rows), 'Legal TRAIN only ' + str(attempt_index))
        ck(math.isfinite(float(row['Gaussian_NLL_F64'])) and int(row['eligible_examples']) == 8 and int(row['target_cells']) == 6912, 'Finite supported batch ' + str(attempt_index))
        huber, variance_nll, total = (float(row[name]) for name in
                                      ('mean_Huber_F64', 'variance_NLL_F64', 'total_objective_F64'))
        ck(all(math.isfinite(value) for value in (huber, variance_nll, total)) and huber >= 0,
           'Finite Huber/variance/total with nonnegative Huber ' + str(attempt_index))
        ck(abs(total - (huber + variance_nll)) <= 2e-12 + 2e-12 * abs(total)
           and abs(variance_nll - float(row['Gaussian_NLL_F64'])) <= 2e-12 + 2e-12 * abs(variance_nll),
           'Unit component sum and unchanged numeric Gaussian evaluation ' + str(attempt_index))
        norm, factor = float(row['pre_clip_global_norm']), float(row['clip_factor'])
        ck(math.isfinite(norm) and norm > 0 and math.isfinite(factor) and 0 < factor <= 1,
           'Finite positive observed global norm/clipping factor ' + str(attempt_index))
        ck(abs(factor - min(1., 1. / (norm + 1e-6))) <= 2e-15,
           'Declared shared global clip1 factor from observed norm ' + str(attempt_index))
        ck(0 < int(row['nonzero_gradient_tensors']) <= 76 and 0 < int(row['changed_tensors']) <= 76, 'Actual learning events ' + str(attempt_index))
        gradient_events += int(row['nonzero_gradient_tensors'])
        movement_events += int(row['changed_tensors'])
        cells += int(row['target_cells'])
    ck(cells == summary['training_target_cells'] == 14155776, 'Exact reconstruction target total')
    ck(gradient_events == sum(p['nonzero_CUDA_gradient_steps'] for p in learned) and
       movement_events == sum(p['weight_movement_steps'] for p in learned), 'CSV/parameter event totals')
    ck(float(rows[0]['Gaussian_NLL_F64']) == summary['first_training_NLL'] and
       float(rows[-1]['Gaussian_NLL_F64']) == summary['last_training_NLL'], 'First/final NLL trace association')
    for row, prefix in ((rows[0], 'first'), (rows[-1], 'last')):
        for name in ('mean_Huber_F64', 'variance_NLL_F64', 'total_objective_F64'):
            ck(float(row[name]) == summary[prefix + '_' + name],
               'Exact first/last component summary association ' + prefix + name)
    query_rows = []
    for stage in ('initial', 'final'):
        queries = read(run / ('query-' + stage + '.json'))
        ck(queries == summary['fixed_query_' + stage], 'Exact query summary ' + stage)
        for role, count in (('train', 112), ('validation', 32)):
            q = queries[role]
            ck(q['valid_rows'] == len(q['row_mae']) == len(q['row_nll']) == len(q['row_target_cells']) == count, 'Query row count ' + stage + role)
            ck(all(math.isfinite(x) and x >= 0 for x in q['row_mae']) and all(math.isfinite(x) for x in q['row_nll']), 'Finite separate MAE/NLL ' + stage + role)
            for row_key, aggregate in (('row_mean_huber', 'mean_Huber_F64'),
                                       ('row_variance_nll', 'variance_NLL_F64'),
                                       ('row_total_objective', 'total_objective_F64')):
                ck(len(q[row_key]) == count and all(math.isfinite(value) for value in q[row_key]),
                   'Full-population query component rows ' + stage + role + row_key)
                ck(abs(math.fsum(q[row_key]) / count - q[aggregate]) <= 2e-12 + 2e-12 * abs(q[aggregate]),
                   'Full-population component reduction ' + stage + role + aggregate)
            ck(all(value >= 0 for value in q['row_mean_huber']), 'Nonnegative fixed query Huber1')
            for index in range(count):
                ck(abs(q['row_variance_nll'][index] - q['row_nll'][index]) <= 2e-12 + 2e-12 * abs(q['row_nll'][index])
                   and abs(q['row_total_objective'][index] - (q['row_mean_huber'][index] + q['row_variance_nll'][index]))
                   <= 2e-12 + 2e-12 * abs(q['row_total_objective'][index]),
                   'Query unchanged Gaussian/unit component row ' + stage + role + str(index))
            ck(abs(math.fsum(q['row_mae']) / count - q['standardized_fixed_query_MAE']) <= 2e-15 and
               abs(math.fsum(q['row_nll']) / count - q['Gaussian_NLL_F64']) <= 2e-15, 'Independent fixed-query means ' + stage + role)
            ck(sum(q['row_target_cells']) == q['target_cells'] == count * 216 and
               all(x == 216 for x in q['row_target_cells']) and q['CUDA_encoder_decoder_calls'] == count // 8, 'Exact query masks/support ' + stage + role)
            query_rows.append(dict(stage=stage, role=role, MAE=q['standardized_fixed_query_MAE'],
                                   NLL=q['Gaussian_NLL_F64'], rows=count,
                                   mean_Huber_F64=q['mean_Huber_F64'], variance_NLL_F64=q['variance_NLL_F64'],
                                   total_objective_F64=q['total_objective_F64']))
    for key in ('same_parameters_buffers_moments_steps_order', 'same_attempt_counter_and_three_seeds', 'continued_actual_trajectory_from_saved_state'):
        ck(resume[key] is True, 'Actual resume receipt ' + key)
    ck(resume['status'] == 'passed' and resume['completed_updates'] == 512 and resume['repeated_training_updates'] == 0, 'Actual512 typed continuation without replay')
    for key, value in dict(status='passed', initialization_kind='fresh-seeded-constructor-no-import-v1',
                           initialization_seed=943001, pretrained_ancestor='none', inherited_learned_parameters=0,
                           fresh_learned_parameter_values=294752, analytic_fixed_prior_values=3888,
                           source_updates=0, causal_updates=0, old_model_artifacts_opened=0,
                           old_normalization_artifacts_opened=0, all_initial_parameters_equal_direct_seeded_constructor=True,
                           parameter_tensors_checked=77, parameter_values_checked=298640).items():
        ck(initial[key] == value, 'Fresh constructor receipt ' + key)
    ck(initial['direct_constructor_state_identity'] == initial['initial_state_identity'], 'Actual initial direct state identity')
    ck(initial['fresh_input_scaler_identity'] == summary['fresh_scaler_identity'] == calibration['input_scaler_identity'] and
       initial['fresh_bank_identity'] == summary['fresh_bank_identity'] == calibration['bank_identity'], 'Own fresh statistics carried through training')
    for key, value in dict(status='passed', TRAIN_anchors=4096, history=128, input_scaler_fits=1,
                           bank_normalizer_fits=1, validation_rows_used=0, test_rows_used=0,
                           encoder_calls=0, old_normalization_artifacts_opened=0, loader_calls=512,
                           raw_record_cells=512 * 8 * 3 * 128 * 9).items():
        ck(calibration[key] == value, 'Fresh TRAIN calibration ' + key)
    for binding in calibration['artifacts']:
        path = run / 'calibration' / binding['file']
        ck(path.is_file() and path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'], 'Calibration artifact ' + binding['file'])
    ck(summary['raw_load_calls'] == 512 + 2048 + 36 == 2596 and
       summary['direct_F64_record_cells_checked'] == 2596 * 8 * 3 * 128 * 9, 'All calibration/training/query loader presentations')
    ck(0 < summary['genuine_F64_differs_from_F32_cells'] <= summary['direct_F64_record_cells_checked'], 'Genuine raw double provenance')
    ck(0 < summary['GPU_step_seconds_synchronized'] < summary['training_trajectory_wall_seconds'], 'Training compute/wall scope')
    for binding in read(run / 'input-bindings.json'):
        ck(sha(Path(binding['path'])) == binding['sha256'], 'Immutable training input ' + binding['path'])
    card_path = source / 'doc/CAUSAL_FRESH_V9_CARD.json'
    ck(sha(card_path) == CARD_SHA and sha(source / 'doc/CAUSAL_FRESH_V9_CARD.md') == MD_SHA, 'Frozen prospective cards')
    ck(sha(source / 'doc/causal_fresh_v9/input-manifest.json') == INPUT_SHA, 'Frozen data-only input manifest')
    card = read(card_path)
    ck(card['mechanism']['source_updates'] == 0 and not card['mechanism']['pretrained_model_inputs'] and
       not card['mechanism']['teacher_model_inputs'] and not card['quality']['historical_quality_floors'], 'No pretrained ancestry/teacher/oracle gate')
    for binding in card['input_pins']:
        if not (binding['path'].endswith('.cache.bin') or Path(binding['path']).name in
                ('selections.pt', 'query-plan-train.pt', 'query-plan-validation.pt')):
            continue  # Training inputs only; never inspect synthetic/head quality data.
        ck(binding['repository'] == 'cuwacunu_dev' and sha(data / binding['path']) == binding['sha256'] and
           (data / binding['path']).stat().st_size == binding['bytes'], 'Frozen declared data pin ' + binding['path'])
    manifest = read(source / 'CAPSULE_MANIFEST.json')
    ck(len(manifest['files']) == main_source_files and manifest['card_sha256'] == CARD_SHA,
       'Exact prospectively pinned actual V9 source closure')
    for binding in manifest['files']:
        path = source / binding['path']
        ck(path.is_file() and path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'], 'Admitted source unchanged ' + binding['path'])
    build_evidence = read(attempt / 'build-admission.json')
    ck(build_evidence['kind'] == 'causal_fresh_v9_build_admission_v1'
       and build_evidence['status'] == 'passed' and build_evidence['card_sha256'] == CARD_SHA
       and build_evidence['gradient_policy'] == GRADIENT_POLICY
       and build_evidence['source_files'] == main_source_files, 'Actual V9 build admission')
    ck(pin(source / 'CAPSULE_MANIFEST.json') == build_evidence['source_manifest']
       and pin(attempt / 'source-stage.json') == build_evidence['source_stage'], 'Admitted exact source-stage/manifest bytes')
    source_stage = read(attempt / 'source-stage.json')
    ck(source_stage['kind'] == 'causal_fresh_v9_container_source_stage_v1'
       and source_stage['status'] == 'passed' and source_stage['source_files'] == main_source_files
       and source_stage['card_sha256'] == CARD_SHA and source_stage['gradient_policy'] == GRADIENT_POLICY
       and Path(source_stage['source_root']).resolve(strict=True) == source
       and source_stage['manifest_sha256'] == sha(source / 'CAPSULE_MANIFEST.json')
       and source_stage['archive_sha256'] == sha(attempt / 'source.zip'), 'Actual immutable source capsule admission')
    ck(len(build_evidence['binaries']) == 7 and len(build_evidence['dependency_closures']) == 11,
       'Seven owned binaries and eleven admitted translation units')
    ck({Path(item['path']).name for item in build_evidence['binaries']} == {
        'test_causal_patch_bottleneck_mae_v9', 'test_cpb_v9_variance_gradient', 'test_cpb_v9_temporal_pooling',
        'causal_fresh_v9_train', 'causal_fresh_v9_extract', 'causal_fresh_v9_market_read', 'causal_fresh_v9_synthetic_read'},
       'Exact seven executable identities, not just count')
    for binding in build_evidence['binaries']:
        ck(pin(Path(binding['path'])) == binding, 'Admitted executable unchanged ' + binding['path'])
    for binding in build_evidence['dependency_closures'].values():
        ck(pin(Path(binding['file']['path'])) == binding['file'], 'Admitted dependency file unchanged')
    expected_training_argv = [next(binding['path'] for binding in build_evidence['binaries']
                                   if Path(binding['path']).name == 'causal_fresh_v9_train'),
                              '--settings', str(source / 'code/encoders/causal_patch_bottleneck_mae/versions/v9/config/v9.conf'),
                              '--shared', str(data / 'output/runs/market-capacity-study/shared-MKT-HIST-2-v1')]
    for symbol in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT'):
        expected_training_argv.extend(('--cache', str(data / 'data/mkt-hist-1/csv' / (symbol + '-1m-2024-01.cache.bin'))))
    expected_training_argv.extend(('--output', str(run)))
    execution_stages = {}
    for name in ('training', 'engineering', 'variance-gradient', 'temporal-pooling'):
        stage = read(attempt / (name + '-stage.json'))
        ck(stage['kind'] == 'causal_fresh_v9_exclusive_stage_execution_v1'
           and stage['status'] == 'passed' and stage['exit_code'] == 0
           and stage['binary'] in build_evidence['binaries']
           and pin(Path(stage['binary']['path'])) == stage['binary']
           and pin(Path(stage['log']['path'])) == stage['log'], 'Actual source-bound execution stage ' + name)
        execution_stages[name] = stage
    ck(execution_stages['training']['argv'] == expected_training_argv, 'Exact actual fresh-training CLI with no imported checkpoint')
    engineering = read(args.engineering_proof)
    json_objective(engineering, 'Actual CUDA engineering')
    json_pooling(engineering, 'Actual CUDA engineering')
    for key, value in dict(kind='cpb_v9_fresh_cuda_engineering_v1', status='passed',
                           gradient_policy=GRADIENT_POLICY, source_updates=0,
                           initialization_seed=943001, pretrained_ancestor='none',
                           old_checkpoints_opened=0, all_initial_tensors_equal_seeded_constructor=True,
                           first_typed_artifact_exact=True, scaler_and_bank_fitted_fixture_TRAIN_only=True,
                           all_cutoffs_per_stage=128, suffix_cases_per_stage=2350,
                           internal_temporal_cases_per_stage=30, causal_updates_primary_fixture=2,
                           resume_witness_updates=1, quality_updates=0,
                           typed_resume_next_loss_model_moments_bitwise_exact=True,
                           CPU_encoder_calls=0, head_fits=0).items():
        ck(engineering[key] == value, 'Source-bound actual CUDA engineering ' + key)
    ck(engineering['stages'] == ['initialized', 'trained'] and engineering['typed_negative_cases'] >= 30
       and engineering['CUDA_encoder_calls'] > 0 and engineering['checks'] > 0
       and bool(engineering['actual_cuda_learning'])
       and all(group['nonzero_gradient_parameter_events'] > 0
               and group['weight_movement_parameter_events'] > 0
               for group in engineering['actual_cuda_learning'].values()), 'Actual engineering causal/learning coverage')
    gradient = read(args.gradient_proof)
    json_objective(gradient, 'Selected mechanism CUDA proof')
    for key, value in dict(status='passed', model='CPB-v9', gradient_policy=GRADIENT_POLICY,
                           actual_CUDA=True, mean_native_variance_gradient_blocked=True,
                           uncertainty_native_variance_gradient_live=True,
                           uncertainty_projection_learning=True,
                           mean_path_live=True, variance_head_learning=True,
                           forward_variance_input_dependence=True,
                           manual_composite_gradient_parity=True,
                           direct_NLL_mean_gradient_zero=True, huber_mean_gradient_live=True,
                           direct_Huber_uncertainty_gradient_zero=True, shared_clip_coupling_witness=True,
                           nonvacuous_nonzero_fixture=True).items():
        ck(gradient[key] == value, 'Source-bound selected mechanism proof ' + key)
    admitted_gradient_binary = next((binding for binding in build_evidence['binaries']
                                    if Path(binding['path']).name == 'test_cpb_v9_variance_gradient'), None)
    ck(admitted_gradient_binary is not None and gradient['binary'] == admitted_gradient_binary
       and pin(Path(gradient['binary']['path'])) == gradient['binary'], 'Actual mechanism admitted binary exact')
    ck(pin(Path(gradient['log']['path'])) == gradient['log']
       and all(token in Path(gradient['log']['path']).read_text() for token in (
           'PASS CPB-v9 huber1-mean-plus-detached-error-gaussian-variance-v1:',
           'manual composite-objective native40 gradient parity', 'synthetic optimizer steps=1')),
       'Actual mechanism observed PASS log exact')
    ck(gradient['binary'] == execution_stages['variance-gradient']['binary']
       and gradient['log'] == execution_stages['variance-gradient']['log'], 'Actual mechanism stage binds observed wrapper')
    temporal_proof = read(args.temporal_proof)
    json_pooling(temporal_proof, 'Actual temporal pooling fixture')
    for key, value in dict(kind='cpb_v9_temporal_pooling_mechanism_proof_v1', status='passed', model='CPB-v9', actual_CUDA=True,
                           original_position_membership=True, both_temporal_paths=True, empty_bins_finite_zero=True,
                           true_per_bin_trace_weights=True, manual_pooling_and_adjoint_parity=True,
                           four_export_slices_gradient_and_movement=True,
                           centered_position_contrasts_gradient_and_movement=True,
                           common_softmax_gauge_invariance=True, shared_global_half_mixture=True,
                           relative_bin_global_sensitivity=True,
                           three_between_bin_contrasts_gradient_and_movement=True,
                           backbone_encode_calls=1, direct_pool_kernel_calls=5,
                           public_encoder_calls=0, head_fits=0, synthetic_optimizer_steps=1,
                           manual_F32_absolute_tolerance=2e-5, manual_F32_relative_tolerance=2e-5,
                           softmax_gauge_tolerance=2e-6, bin_mass_tolerance=2e-6, flat_trace_tolerance=8e-6,
                           empty_and_membership_zero='exact').items():
        ck(temporal_proof[key] == value, 'Source-bound temporal pooling fixture ' + key)
    temporal_binary = next(binding for binding in build_evidence['binaries']
                           if Path(binding['path']).name == 'test_cpb_v9_temporal_pooling')
    ck(temporal_proof['binary'] == temporal_binary == execution_stages['temporal-pooling']['binary']
       and pin(Path(temporal_proof['binary']['path'])) == temporal_proof['binary']
       and temporal_proof['log'] == execution_stages['temporal-pooling']['log']
       and pin(Path(temporal_proof['log']['path'])) == temporal_proof['log']
       and temporal_proof['execution'] == pin(attempt / 'temporal-pooling-stage.json'),
       'Exact admitted temporal fixture binary/log/execution bindings')
    ck(all(token in Path(temporal_proof['log']['path']).read_text() for token in (
        'PASS CPB-v9 original-position-four-bin-shared-global-half-mixture-v1:',
        'manual original-position half-mixture pooling parity', 'four 8-row projection gradient and movement',
        'three between-bin Hadamard contrast gradient and movement', 'relative-bin global sensitivity',
        'empty-bin finite zero', 'common softmax gauge invariance', 'synthetic optimizer steps=1',
        'backbone_encode_calls=1', 'public_encoder_calls=0', 'direct_pool_kernel_calls=5', 'head_fits=0')),
       'Actual independently reviewed temporal fixture PASS output and call scope')
    ck(sha(run / 'checkpoints/inference-0.pt') == initial['initial_checkpoint']['sha256'] and
       (run / 'checkpoints/inference-0.pt').stat().st_size == initial['initial_checkpoint']['bytes'], 'First typed checkpoint unchanged')
    checkpoints = [pin(p) for p in sorted((run / 'checkpoints').iterdir()) if p.is_file()]
    ck([Path(p['path']).name for p in checkpoints] == ['inference-0.pt', 'inference-2048.pt', 'training-1024.pt', 'training-2048.pt', 'training-512.pt'], 'Five preserved main checkpoints')
    ck(initial['fresh_input_scaler_identity'] == 'rpb-scaler-v1-fnv1a64-9d33c2019209d321',
       'Paired TRAIN input scaler identity; bank provenance independently recomputed from fresh V9 metadata')
    main_source = (source / 'code/research/causal_fresh_v9/train_main.cpp').read_text()
    api_source = (source / 'code/encoders/causal_patch_bottleneck_mae/versions/v9/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v9.h').read_text()
    factory = api_source.split('inline Loaded fresh(', 1)[1].split('namespace checkpoint_detail', 1)[0]
    ck('cp::fresh(config,calibration.scaler,calibration.bank,943001)' in main_source and
       'torch::manual_seed(seed)' in factory and 'LocationScaleDecoderModel(' in factory and
       'load_checkpoint' not in factory and 'import_v1' not in factory and 'import_v2' not in factory, 'Actual admitted fresh constructor source without checkpoint import')
    ck('model=resumed.model;scaler=resumed.scaler;optimizer=std::move(resumed.optimizer);progress=resumed.progress;' in main_source
       and '++progress.completed_updates;++progress.attempted_batches;model->set_causal_updates(progress.completed_updates);' in main_source
       and 'context_seed,attempt,impl::ContextDeletionRecipe::coordinate15_v1' in main_source
       and 'updates=2048,max_attempts=4096' in main_source
       and 'if(!masks.eligible_channels.any().item<bool>()){++progress.attempted_batches;' in main_source,
       'Actual admitted typed continuation/live counter/context schedule source')
    ck('trace.objective.total_loss.backward();' in main_source and 'trace.nll.loss.backward()' not in main_source
       and 'clip_grad_norm_(model->parameters(),1.)' in main_source
       and 'if(progress.completed_updates==1||progress.completed_updates==2048)' in main_source
       and 'reporting-NLL-never-backward' in main_source,
       'Admitted composite backward/shared clip/actual-forward witness source contract')
    core_passed = not bad
    numeric = numeric_audit(run, data, selected, attempts_by_update)
    tensors = saved_tensor_audit(run, parameters, attempts_by_update)
    gaussian = saved_gaussian_audit(run)
    components = saved_optimization_component_audit(run, summary, rows)
    temporal = temporal_learning_audit(run, summary, parameters)
    result = dict(kind='causal_fresh_v9_training_independent_audit_v1', status='passed' if core_passed and numeric['status'] == 'passed' and tensors['status'] == 'passed' and gaussian['status'] == 'passed' and components['status'] == 'passed' and temporal['status'] == 'passed' else 'failed',
                  core_saved_arithmetic_status='passed' if core_passed else 'failed', checks=checks,
                  mismatches=bad, numeric_statistics=numeric, saved_tensor_state=tensors,
                  manual_saved_Gaussian=gaussian, actual_pre_step_composite_objective=components,
                  temporal_pooling_saved_learning=temporal,
                  audit_card=pin(args.audit_card), gradient_policy=GRADIENT_POLICY,
                  objective_policy=OBJECTIVE_POLICY, model='CPB-v9.market-01',
                  temporal_pooling_policy=POOL_POLICY, temporal_pooling_partition=POOL_PARTITION,
                  temporal_pooling_mix=POOL_MIX, empty_local_bin_policy=EMPTY_LOCAL_POLICY,
                  mandatory_nonzero_data_learning_tensors=75, mandatory_identifiable_tensor_value_capacity=294751,
                  per_scalar_nonzero_gradient_proof_claimed=False,
                  gauge_exemption=GAUGE, gauge_decay_not_data_learning=True,
                  shared_clipping=dict(rows_checked=2048, global_max_norm=1., norm_epsilon=1e-6,
                                       clipping_applied_rows=sum(float(row['clip_factor']) < 1 for row in rows),
                                       observed_norm_independently_recomputed=False,
                                       optimizer_independence_claimed=False),
                  source_updates=0, pretrained_ancestor='none', parameter_tensors=77,
                  learned_tensors=76, learned_values=294752, fixed_values=3888,
                  groups=dict(groups), CSV_rows=len(all_rows), successful_CSV_rows=len(rows),
                  attempted_batches=attempted, skipped_ineligible_attempts=len(skipped_rows),
                  skipped_attempt_indices=skipped_rows,
                  checkpoint_attempt_counters={str(update): attempts_by_update[update]
                                               for update in (512, 1024, 2048)},
                  independent_actual_CSV_stream_pairs=2048,
                  independent_MT19937_64_row_choices=2048 * 8,
                  context_stream=dict(base_seed=943004, expected_derived_streams=2048,
                                      expected_stream_sha256=context_stream_hash.hexdigest(),
                                      actual_attempts=[int(row['attempt']) for row in rows],
                                      evidence='Base and actual attempt counters verified in3typed states and admitted make_plan source; successful-row streams use actual CSV attempts; per-batch context values are not in CSV'),
                  query_arithmetic=query_rows, checkpoints=checkpoints,
                  calibration=calibration, fresh_initialization=initial,
                  source_bound_actual_engineering=pin(args.engineering_proof),
                  source_bound_variance_gradient_proof=pin(args.gradient_proof),
                  source_bound_temporal_pooling_proof=pin(args.temporal_proof),
                  preserved_audit_failures=prior_failures,
                  evidence_pins=[pin(run / name) for name in ('training-summary.json', 'training.csv', 'parameter-learning.json',
                                 'fresh-initialization.json', 'typed-resume-512.json', 'query-initial.json', 'query-final.json',
                                 'optimization-component-witnesses.json', 'input-bindings.json',
                                 'temporal-projection-learning.json', 'temporal-projection-learning.csv',
                                 'temporal-pool-score-bias-learning.csv', 'temporal-between-bin-position-learning.csv')] +
                                [pin(attempt / name) for name in ('build-admission.json', 'source-stage.json',
                                 'training-stage.json', 'engineering-stage.json', 'variance-gradient-stage.json', 'temporal-pooling-stage.json')],
                  auditor=pin(Path(__file__)), elapsed_seconds=time.monotonic() - started,
                  encoder_calls=0, decoder_calls=0, head_fits=0, encoder_updates=0,
                  method='Existing-container pure standard-library saved arithmetic and data-only archive storage inspection; no torch import or module execution',
                  limitations=['Gradient values and exact live resume continuation are source-bound producer evidence; this audit does not repeat training.',
                               'The scalar objective cannot prove stop-gradient routing; admitted nonzero CUDA fixture and typed policy evidence provide that check.',
                               'Shared clipping norm is observed evidence; factor arithmetic is independently checked without full gradient arrays.',
                               'Git ignore and host copy availability are verified separately by bounded host diagnostics.'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], core_checks=checks, numeric_status=numeric['status'],
                          numeric_checks=numeric['checks'], tensor_status=tensors['status'],
                          Gaussian_status=gaussian['status'], composite_status=components['status'],
                          temporal_status=temporal['status'], temporal_checks=temporal['checks'],
                          mismatches=len(bad), output=str(args.output))))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

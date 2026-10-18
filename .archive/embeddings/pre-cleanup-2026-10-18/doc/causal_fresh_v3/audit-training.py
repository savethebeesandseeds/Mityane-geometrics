# Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE.
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

CARD_SHA = 'f88fa9751ff9da0e808805da9975fb0bf3afd4433688f3cfb420686392e6396f'
MD_SHA = 'd03c4f60f2eb270bc7d6ba77d575f998703c2c204a90b3df9d5ec0d938039446'
INPUT_SHA = 'ce691b6ec2563cbf7a6b2a615c3912e3a9ec586f5198e1655450a22c9f9c9ffd'
MASK64 = (1 << 64) - 1
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


def numeric_audit(run, data_root, selected):
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
            eq(progress == [update, update, 943002, 943003, 943004], 'Actual typed progress/all three seeds ' + str(update))
            eq(archive.field('encoder', 'source_updates').scalar() == 0 and
               archive.field('encoder', 'causal_updates').scalar() == update and
               archive.field('encoder', 'pretrained_ancestor').text() == 'none', 'Actual typed fresh ancestry ' + str(update))
            steps = archive.field('optimizer_steps_v1').values()
            eq(len(steps) == 75 and all(step == update for step in steps), 'Actual75 optimizer step counters ' + str(update))
            typed.append(dict(update=update, progress=progress, active_optimizer_states=len(steps)))
        return dict(status='passed' if not numeric_bad else 'failed', checks=numeric_checks,
                    mismatches=numeric_bad, raw_cache_records=raw_records,
                    input_coordinates=27, bank_coordinates=270, independent_TRAIN_windows=4096,
                    first_witness_raw_coordinates=8 * 3 * 128 * 9,
                    max_bank_sum_absolute_error=max_bank_sum_absolute_error,
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


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('run', 'attempt', 'source', 'data-root', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('Execute saved-data audit only in the existing managed container')
    run, attempt, source, data = (p.resolve(strict=True) for p in (args.run, args.attempt, args.source, args.data_root))
    summary = read(run / 'training-summary.json')
    parameters = read(run / 'parameter-learning.json')
    initial = read(run / 'fresh-initialization.json')
    resume = read(run / 'typed-resume-512.json')
    calibration = read(run / 'calibration/summary.json')
    expected = dict(status='passed', model='CPB-v3.market-01', source_updates=0,
                    new_causal_updates=2048, attempted_batches=2048, initialization_seed=943001,
                    batch_size=8, TRAIN_windows=4096, probe_TRAIN_windows=112,
                    probe_VALIDATION_windows=32, native_width=36, device='cuda:0',
                    parameters=297560, parameters_with_recorded_updates=293672,
                    fixed_prior_parameters=3888, typed_resume_512_exact=True,
                    fixed_prior_scaler_normalizer_unchanged=True, new_scaler_fits=1,
                    new_normalizer_fits=1, calibration_encoder_calls=0,
                    calibration_validation_rows=0, old_model_artifacts_opened=0,
                    old_normalization_artifacts_opened=0, saved_quality_features_reused=False,
                    CPU_encoder_calls=0, old_encoder_calls=0, head_fits=0, training_replays=0)
    for key, value in expected.items():
        ck(summary[key] == value, 'Training summary ' + key)
    learned = [p for p in parameters if p['requires_grad']]
    fixed = [p for p in parameters if not p['requires_grad']]
    ck(len(parameters) == len({p['name'] for p in parameters}) == 76 and len(learned) == 75 and len(fixed) == 1, '76 tensors:75 learned,one fixed')
    ck(sum(p['values'] for p in learned) == 293672 and sum(p['values'] for p in fixed) == 3888, 'Exact learned/fixed partition')
    groups = collections.defaultdict(lambda: dict(tensors=0, values=0, nonzero_gradient_tensors=0,
                                                 moved_tensors=0, changed_final_tensors=0))
    for p in parameters:
        if p['requires_grad']:
            ck(0 < p['nonzero_CUDA_gradient_steps'] <= 2048 and
               0 < p['weight_movement_steps'] <= 2048 and p['changed_from_initial'], 'Actual learned evidence ' + p['name'])
            g = groups[p['group']]
            g['tensors'] += 1
            g['values'] += p['values']
            g['nonzero_gradient_tensors'] += p['nonzero_CUDA_gradient_steps'] > 0
            g['moved_tensors'] += p['weight_movement_steps'] > 0
            g['changed_final_tensors'] += p['changed_from_initial']
        else:
            ck(p['nonzero_CUDA_gradient_steps'] == p['weight_movement_steps'] == 0 and not p['changed_from_initial'], 'Literal prior protected')
    ck(len(groups) == 8, 'Eight learned categories')
    for group in summary['parameter_groups']:
        ck({k: v for k, v in group.items() if k != 'group'} == groups[group['group']], 'Recomputed learning group ' + group['group'])
    selected = anchors()
    selected_set = set(selected)
    rows = list(csv.DictReader((run / 'training.csv').open(newline='')))
    ck(len(rows) == 2048, 'Exactly2048 successful CSV rows')
    gradient_events = movement_events = cells = 0
    context_stream_hash = hashlib.sha256()
    for attempt_index, row in enumerate(rows):
        ck(int(row['attempt']) == attempt_index and int(row['update']) == attempt_index + 1, 'Monotone counters ' + str(attempt_index))
        row_stream, mask_stream = counter(943002, attempt_index), counter(943003, attempt_index)
        ck(int(row['row_seed']) == row_stream and int(row['mask_seed']) == mask_stream, 'Actual two CSV streams ' + str(attempt_index))
        context_stream_hash.update(struct.pack('<q', counter(943004, attempt_index)))
        rng = MT19937_64(row_stream)
        expected_rows = [selected[(rng.next() * 4096) >> 64] for _ in range(8)]
        actual_rows = list(map(int, row['anchors'].split(';')))
        ck(actual_rows == expected_rows, 'Independent MT19937_64 choices ' + str(attempt_index))
        ck(len(actual_rows) == 8 and all(x in selected_set and x - 127 * 60000 >= 1704067259999 and x <= 1705881599999 for x in actual_rows), 'Legal TRAIN only ' + str(attempt_index))
        ck(math.isfinite(float(row['Gaussian_NLL_F64'])) and int(row['eligible_examples']) == 8 and int(row['target_cells']) == 6912, 'Finite supported batch ' + str(attempt_index))
        ck(0 < int(row['nonzero_gradient_tensors']) <= 75 and 0 < int(row['changed_tensors']) <= 75, 'Actual learning events ' + str(attempt_index))
        gradient_events += int(row['nonzero_gradient_tensors'])
        movement_events += int(row['changed_tensors'])
        cells += int(row['target_cells'])
    ck(cells == summary['training_target_cells'] == 14155776, 'Exact reconstruction target total')
    ck(gradient_events == sum(p['nonzero_CUDA_gradient_steps'] for p in learned) and
       movement_events == sum(p['weight_movement_steps'] for p in learned), 'CSV/parameter event totals')
    ck(float(rows[0]['Gaussian_NLL_F64']) == summary['first_training_NLL'] and
       float(rows[-1]['Gaussian_NLL_F64']) == summary['last_training_NLL'], 'First/final NLL trace association')
    query_rows = []
    for stage in ('initial', 'final'):
        queries = read(run / ('query-' + stage + '.json'))
        ck(queries == summary['fixed_query_' + stage], 'Exact query summary ' + stage)
        for role, count in (('train', 112), ('validation', 32)):
            q = queries[role]
            ck(q['valid_rows'] == len(q['row_mae']) == len(q['row_nll']) == len(q['row_target_cells']) == count, 'Query row count ' + stage + role)
            ck(all(math.isfinite(x) and x >= 0 for x in q['row_mae']) and all(math.isfinite(x) for x in q['row_nll']), 'Finite separate MAE/NLL ' + stage + role)
            ck(abs(math.fsum(q['row_mae']) / count - q['standardized_fixed_query_MAE']) <= 2e-15 and
               abs(math.fsum(q['row_nll']) / count - q['Gaussian_NLL_F64']) <= 2e-15, 'Independent fixed-query means ' + stage + role)
            ck(sum(q['row_target_cells']) == q['target_cells'] == count * 216 and
               all(x == 216 for x in q['row_target_cells']) and q['CUDA_encoder_decoder_calls'] == count // 8, 'Exact query masks/support ' + stage + role)
            query_rows.append(dict(stage=stage, role=role, MAE=q['standardized_fixed_query_MAE'], NLL=q['Gaussian_NLL_F64'], rows=count))
    for key in ('same_parameters_buffers_moments_steps_order', 'same_attempt_counter_and_three_seeds', 'continued_actual_trajectory_from_saved_state'):
        ck(resume[key] is True, 'Actual resume receipt ' + key)
    ck(resume['status'] == 'passed' and resume['completed_updates'] == 512 and resume['repeated_training_updates'] == 0, 'Actual512 typed continuation without replay')
    for key, value in dict(status='passed', initialization_kind='fresh-seeded-constructor-no-import-v1',
                           initialization_seed=943001, pretrained_ancestor='none', inherited_learned_parameters=0,
                           fresh_learned_parameter_values=293672, analytic_fixed_prior_values=3888,
                           source_updates=0, causal_updates=0, old_model_artifacts_opened=0,
                           old_normalization_artifacts_opened=0, all_initial_parameters_equal_direct_seeded_constructor=True,
                           parameter_tensors_checked=76, parameter_values_checked=297560).items():
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
    card_path = source / 'doc/CAUSAL_FRESH_V3_CARD.json'
    ck(sha(card_path) == CARD_SHA and sha(source / 'doc/CAUSAL_FRESH_V3_CARD.md') == MD_SHA, 'Frozen prospective cards')
    ck(sha(source / 'doc/causal_fresh_v3/input-manifest.json') == INPUT_SHA, 'Frozen data-only input manifest')
    card = read(card_path)
    ck(card['mechanism']['source_updates'] == 0 and not card['mechanism']['pretrained_model_inputs'] and
       not card['mechanism']['teacher_model_inputs'] and not card['quality']['historical_quality_floors'], 'No pretrained ancestry/teacher/oracle gate')
    for binding in card['input_pins']:
        ck(binding['repository'] == 'cuwacunu_dev' and sha(data / binding['path']) == binding['sha256'] and
           (data / binding['path']).stat().st_size == binding['bytes'], 'Frozen declared data pin ' + binding['path'])
    manifest = read(source / 'CAPSULE_MANIFEST.json')
    ck(len(manifest['files']) == 82 and manifest['card_sha256'] == CARD_SHA, 'Admitted82 source closure')
    for binding in manifest['files']:
        path = source / binding['path']
        ck(path.is_file() and path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'], 'Admitted source unchanged ' + binding['path'])
    admission = read(attempt / 'training-admission.json')
    ck(admission['status'] == 'passed' and admission['card_sha256'] == CARD_SHA, 'Actual pretraining admission')
    if 'build_evidence_sha256' in admission:
        ck(sha(attempt / 'build-evidence.json') == admission['build_evidence_sha256'], 'Admitted exact binaries/source evidence')
    elif 'build_admission_sha256' in admission:
        ck(sha(attempt / 'build-evidence.json') == admission['build_admission_sha256'], 'Admitted exact binaries/source evidence')
    else:
        ck(False, 'Training admission lacks its exact build binding')
    if 'engineering_proof_sha256' in admission:
        ck(sha(attempt / 'engineering-proof/proof.json') == admission['engineering_proof_sha256'], 'Actual prior engineering proof binding')
    ck(sha(run / 'checkpoints/inference-0.pt') == initial['initial_checkpoint']['sha256'] and
       (run / 'checkpoints/inference-0.pt').stat().st_size == initial['initial_checkpoint']['bytes'], 'First typed checkpoint unchanged')
    checkpoints = [pin(p) for p in sorted((run / 'checkpoints').iterdir()) if p.is_file()]
    ck([Path(p['path']).name for p in checkpoints] == ['inference-0.pt', 'inference-2048.pt', 'training-1024.pt', 'training-2048.pt', 'training-512.pt'], 'Five preserved main checkpoints')
    core_passed = not bad
    numeric = numeric_audit(run, data, selected)
    result = dict(kind='causal_fresh_v3_training_independent_audit_v1', status='passed' if core_passed and numeric['status'] != 'failed' else 'failed',
                  core_saved_arithmetic_status='passed' if core_passed else 'failed', checks=checks,
                  mismatches=bad, numeric_statistics=numeric, model='CPB-v3.market-01',
                  source_updates=0, pretrained_ancestor='none', parameter_tensors=76,
                  learned_tensors=75, learned_values=293672, fixed_values=3888,
                  groups=dict(groups), CSV_rows=len(rows), independent_actual_CSV_stream_pairs=2048,
                  independent_MT19937_64_row_choices=2048 * 8,
                  context_stream=dict(base_seed=943004, expected_derived_streams=2048,
                                      expected_stream_sha256=context_stream_hash.hexdigest(),
                                      evidence='Base verified in3typed states and admitted make_plan source; per-batch context values are not in CSV'),
                  query_arithmetic=query_rows, checkpoints=checkpoints,
                  calibration=calibration, fresh_initialization=initial,
                  evidence_pins=[pin(run / name) for name in ('training-summary.json', 'training.csv', 'parameter-learning.json',
                                 'fresh-initialization.json', 'typed-resume-512.json', 'query-initial.json', 'query-final.json', 'input-bindings.json')] +
                                [pin(attempt / name) for name in ('training-admission.json', 'build-evidence.json', 'source-stage.json')],
                  auditor=pin(Path(__file__)), elapsed_seconds=time.monotonic() - started,
                  encoder_calls=0, decoder_calls=0, head_fits=0, encoder_updates=0,
                  method='Existing-container pure standard-library saved arithmetic and data-only archive storage inspection; no torch import or module execution',
                  limitations=['Gradient values and exact live resume continuation are source-bound producer evidence; this audit does not repeat training.',
                               'Git ignore and host copy availability are verified separately by bounded host diagnostics.'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], core_checks=checks, numeric_status=numeric['status'],
                          numeric_checks=numeric['checks'], mismatches=len(bad), output=str(args.output))))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

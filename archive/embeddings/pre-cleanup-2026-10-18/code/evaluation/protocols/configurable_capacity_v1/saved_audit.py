"""Saved-only capacity checks. No PyTorch import, model construction or fitting."""
import io
import math
import struct
import zipfile


def checkpoint_scaler(codec, path):
    """Decode checkpoint statistics into ordinary arrays, never execute a model.

    The frozen codec's allowlist handles LibTorch's serialized Module objects.
    Checkpoint model storage can be CUDA-tagged; storage bytes are decoded as
    arrays solely to access the CPU scaler metadata. This is not model replay.
    The original CPU-only archive codec remains unchanged.
    """
    class MetadataReader(codec.Reader):
        def persistent_load(self, pid):
            if len(pid) < 5 or pid[0] != 'storage' or pid[3] not in ('cpu', 'cuda:0'):
                raise ValueError('closed checkpoint storage metadata')
            return pid[1], self.archive.read(self.prefix + 'data/' + pid[2])

    with zipfile.ZipFile(path) as archive:
        keys = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(keys) != 1:
            raise ValueError('one checkpoint metadata pickle')
        key = keys[0]
        metadata = vars(MetadataReader(io.BytesIO(archive.read(key)), archive,
                                       key[:-len('data.pkl')]).load())
    return metadata, vars(metadata['scaler'])


def float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def scaler_identity(scaler):
    """Reproduce FrozenScaler::identity's little-endian scalar-type bytes."""
    scalar_types = dict(DoubleStorage=7, FloatStorage=6, LongStorage=4,
                        BoolStorage=11, ByteStorage=0)
    digest = 14695981039346656037
    def add(raw):
        nonlocal digest
        for byte in raw:
            digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
    for key in ('mean', 'scale', 'count', 'channel_ids', 'floor_applied'):
        value = scaler[key]
        add(struct.pack('<q', scalar_types[value['dtype']]))
        for length in value['shape']:
            add(struct.pack('<q', length))
        add(value['raw'])
    add(struct.pack('<d', scaler['scale_floor']['values'][0]))
    return f'rpb-scaler-v1-fnv1a64-{digest:016x}'


def audit_train_scaler(codec, scaler, training, check, close, tensor):
    raw = tensor(training, 'observed', 'DoubleStorage', [512, 3, 128, 9])
    mask = tensor(training, 'feature_mask', 'BoolStorage', [512, 3, 128, 9])
    means = tensor(scaler, 'mean', 'DoubleStorage', [3, 9])
    scales = tensor(scaler, 'scale', 'DoubleStorage', [3, 9])
    counts = tensor(scaler, 'count', 'LongStorage', [3, 9])
    floors = tensor(scaler, 'floor_applied', 'BoolStorage', [3, 9])
    check(tensor(scaler, 'channel_ids', 'LongStorage', [3]) == [0, 1, 2]
          and tensor(scaler, 'scale_floor', 'DoubleStorage', []) == [1e-6],
          'declared TRAIN-only encoder scaler geometry and floor')
    for c in range(3):
        for f in range(9):
            positions = ((b * 3 + c) * 1152 + h * 9 + f
                         for b in range(512) for h in range(128))
            values = [raw[i] for i in positions if mask[i]]
            coordinate = c * 9 + f
            check(values and len(values) == counts[coordinate], 'TRAIN-only encoder scaler fitting count')
            mean = math.fsum(values) / len(values)
            deviation = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
            close(means[coordinate], mean, 'TRAIN-only encoder scaler fitted mean')
            close(scales[coordinate], max(deviation, 1e-6), 'TRAIN-only encoder scaler fitted scale')
            check(bool(floors[coordinate]) == (deviation < 1e-6), 'TRAIN-only encoder scaler floor witness')
    return means, scales

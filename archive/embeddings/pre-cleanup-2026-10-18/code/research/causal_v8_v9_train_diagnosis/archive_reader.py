# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Inert C++ tensor archive DATA reader; no Torch import or model execution."""
import collections, io, itertools, pickle, struct, zipfile

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



# Restricted saved CPU tensor reader, candidate v1

This is an optional migration candidate. On 18 October 2026 it compiled with
GCC 12.2 and passed its crafted-archive engineering checks in the managed
`mityane-geometrics` container. It is not admitted as a replacement for a
historical reader. Original Python readers and sealed source fixtures remain
unchanged. These checks do not authorize a historical research run.

`Archive(path, policy)` indexes an explicitly admitted TRAIN, VALIDATION or
engineering fixture artifact. `policy.allowed_root` and a nonempty exact set of
`selected_fields` are mandatory. TEST role is refused before opening the file.
The caller must independently establish the artifact hash, source identity and
protocol/role ledger; a role label does not establish research permission.

`fields()` returns the flattened tensor schema, `info(field)` returns signed64
shape, strides, storage offset/count and dtype, and `read(field)` materializes
only a selected field. Nested module/dictionary fields use dot-separated names;
literal dots in component names are rejected to avoid ambiguity. Metadata and
the pickle member are parsed at open; storage payloads are read on demand.
ZIP directory and local-header reads, and a fallback EOCD tail scan for ZIP
comments, may physically overlap payload bytes. This is not a syscall-traced
claim that no byte belonging to an unselected member was read.

The supported closed profile is:

- Single-root, single-disk, unencrypted `ZIP_STORED`, classic ZIP or ZIP64 end
  records/central sizes, with optional signed or unsigned data descriptors.
  Duplicate/unsafe paths, overlaps, inconsistent record bounds and corrupt
  materialized-member CRC32 are rejected. No extraction or write API exists.
- Pickle protocols 2 through 5, restricted to the explicit integer/string,
  tuple/list/dictionary, memo, frame and construction opcodes in `reader.cpp`.
  Protocol declaration, stop and exact stream consumption are required.
- The only globals are `collections.OrderedDict`, the five CPU storage classes,
  `torch._utils._rebuild_tensor[_v2]` and `__torch__.*.Module` inert state.
  Globals are symbols interpreted by this parser; no callable is executed.
  Requires-grad tensors, nonempty backward hooks/metadata, unknown globals,
  general objects, unsupported opcodes and cyclic/deep state are rejected.
- `DoubleStorage`, `FloatStorage`, `LongStorage`, `BoolStorage`, `ByteStorage`,
  literal `cpu` location, exact five-field persistent storage identities, and
  decimal storage keys. Declared little-endian storage is required. Older
  archives without `byteorder` need the caller's explicit
  `permit_legacy_missing_byteorder` historical profile.
- Scalars, empty tensors and noncontiguous signed64 views, including negative
  strides when all addresses are valid. Arithmetic overflow, out-of-bounds or
  repeated element addresses are rejected. Reading a storage referenced by
  multiple fields is rejected, including aliases from excluded fields.
- Raw logical bytes preserve dtype, signed zeros and NaN payloads. Numeric
  access does not silently coerce Long values to double. `utf8()` requires a
  rank-one Byte tensor and strict UTF-8. Bool values must be canonical 0 or 1.

Compressed ZIP/DEFLATE, legacy non-ZIP Torch files, shared storage, sparse or
quantized tensors, extra tensor dtypes, storage views in six-field PIDs,
requires-grad state and arbitrary tensor metadata are unsupported. They fail
closed; there is no fallback through Torch, Python, `pickle`, external ZIP
commands or a dynamically installed package. Which historical artifacts need
these extensions remains to be established against admitted fixtures.

The header exposes configurable byte/member/opcode/node/depth/element limits.
An `Archive` owns a mutable input stream and is not safe for concurrent reads.
Callers must hold admitted input files immutable during a read; source hash
checks before/after an audit remain the protocol adapter's responsibility.

`tests/reader_test.cpp` constructs its own tiny ZIP/pickle fixtures, rather than
deserializing models. It covers strided/scalar/empty views, negative strides,
ZIP64/descriptors, signed-zero bytes, unsupported devices/opcodes, aliases,
corruption, role/root restrictions, canonical Bool/UTF-8 and resource limits.
The engineering test passed without compiler warnings after correcting four
misleading statement layouts. Source-identity sealing, further negative-fixture
expansion and parity against historical artifacts remain pending. The root target is
`check-saved-tensor-v1`, invoked through
`.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'migration-reader', 'check-saved-tensor-v1')`.
It runs inside the existing managed container and writes only
to a named build directory under `/opt/mityane-geometrics`. The root exposes
candidate migration goals explicitly; normal maintenance/model targets do not
run these tests. Build/test recipes enforce the managed container mount, check
canonical output paths, set a named `TMPDIR`, take a directory lock and publish
compiled binaries by atomic rename. The validation log is
`/opt/mityane-geometrics/embeddings/build/migration-native-validation/validation.log`.

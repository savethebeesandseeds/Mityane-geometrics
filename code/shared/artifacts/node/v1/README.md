# Node artifact files, version 1

Shared source for the language migration. Maintained metadata/preparation callers
are wired to this implementation. Artificial engineering fixtures have passed
in the existing managed Linux container; see the
[validation record](../../../../../doc/PYTHON_MIGRATION_VERIFICATION_2026-10-18.md).

`files.mjs` admits complete explicit file matrices before payload reads, checks
canonical paths and inode aliases, binds file signatures through each read/hash,
and publishes new records with exclusive creation and file/directory fsync.
No model, dataset discovery, training, registry or container lifecycle is owned
here. Callers supply their role/split/admission policy.

Metadata signatures are not a content lock: immediate same-size writes can
share timestamps, even when stat exposes nanosecond fields. Callers must compare
the returned content hashes to their expected inventory, writer report or source
manifest and keep admitted inputs stable throughout their protocol. Signature
checks reject observed identity/metadata changes; they do not certify every
metadata-invisible concurrent write. The timestamp-change fixture advances mtime
explicitly so it does not depend on filesystem clock resolution.

`parseJSON` makes duplicate-key and large-integer behavior explicit. Safe integer
tokens use `Number`; unsafe integer tokens use `BigInt`, or reject in the
number-only mode. The default duplicate policy preserves the last key, as in
plain Python `json.loads`; a caller may explicitly select rejection. Nonfinite
JSON extensions are rejected under this new versioned parser contract.

`jsonBytes` declares `mityane-artifact-json-v1`, retaining exact large integers,
Unicode ordering/escaping, explicit key order and a final newline. It does not
claim Python floating-point text equivalence. A receipt whose exact historical
JSON bytes enter a source identity needs a separately admitted encoding/parity
contract. New hashes must not replace historical pins.

Run the fixtures through the root `test-language-migration` task in the managed
Linux container. Synthetic fixtures do not certify historical receipt byte parity.

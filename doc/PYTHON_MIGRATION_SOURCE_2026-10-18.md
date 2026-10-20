# Python migration: authored replacement source

This dated first source record is superseded for current scope and integration
by the [complete implementation record](PYTHON_MIGRATION_IMPLEMENTATION_2026-10-18.md).
The owner subsequently authorized retaining well-suited scientific Python and
requested the entire remaining migration as one job.

This record describes file edits made after the owner requested that programming
continue while Docker is stopped. None of the new sources below has been run,
syntax-checked, compiled or tested. No Docker command, model execution, fit,
training, saved research payload read or experimental reproduction was performed
for this source batch. The earlier tested maintenance tranche has its own
[verification record](PYTHON_MIGRATION_2026-10-18.md).

The dated [inventory](PYTHON_MIGRATION_INVENTORY_2026-10-18.json) remains the
before-port baseline. Original Python, archived sources, historical outputs,
source hashes and protocol pins remain intact. The current milestone states,
dependencies and acceptance criteria live in
[`14-foundations.json`](../web/dashboard/heroes/14-foundations.json).

## Sources written

| Area | Authored source | Scope |
| --- | --- | --- |
| Shared file and record handling | [`code/shared/artifacts/node/v1/`](../code/shared/artifacts/node/v1/) | Explicit file-matrix admission, stable read/hash signatures, alias/path checks, exact integer parsing, versioned JSON encoding and exclusive durable record writes |
| Ten preparation scripts | [`code/embeddings/scripts/`](../code/embeddings/scripts/) `.mjs` counterparts listed below | Fixed source/card/parent roles, input inventories and capsule bookkeeping; explicit immutable adapter configuration |
| Configurable protocol capsules | `capsule.mjs` in branch, capacity, learned-relation and content-relation protocol directories | Separate new source identities, captured source closures, explicit historical mappings and guarded independent-reader binding |
| Learned/content gates | `gate.mjs` in the learned/content relation protocol directories | Closed one/two-cohort stages, original thresholds/slack, exact condition and verdict rules, no automatic promotion |
| Shared capsule mechanics | [`code/embeddings/evaluation/protocols/node/v1/`](../code/embeddings/evaluation/protocols/node/v1/) | Source capture, consumed-artifact ledgers, completion records and shared relation capsule mechanics |
| Independent tensor reader | [`code/shared/artifacts/saved_tensor/versions/v1/`](../code/shared/artifacts/saved_tensor/versions/v1/) | C++ restricted ZIP/pickle CPU reader with selected fields, dtype/stride checks, ZIP64/CRC checks and refusal of unsupported formats |
| Independent saved arithmetic | [`code/embeddings/evaluation/independent_cpu/versions/v1/`](../code/embeddings/evaluation/independent_cpu/versions/v1/) | C++ seeds, FNV, MT19937-64, unbiased index mapping, compensated sums, float32 conversion, scoring, bootstrap and query reductions |

The preparation counterparts are:

- `prepare-fresh-decoder-replication.mjs`
- `prepare-saved-native-reliability.mjs`
- `prepare-frozen-amplitude-transfer.mjs`
- `prepare-early-mixer-reliability.mjs`
- `prepare-early-mixer-learning-curve.mjs`
- `prepare-early-mixer-confirmation.mjs`
- `prepare-matched-target-gain.mjs`
- `prepare-pooled-context.mjs`
- `prepare-training-objective-inputs.mjs`
- `prepare-v7-decoder-inputs.mjs`

The first eight share the factory in `prepare-fresh-decoder-replication.mjs`
and explicit recipe fragments in `preparation-policies.mjs`, instead of changing
an imported Python module's globals. Their `--historical-parent-capsule` option
names an exact saved capsule. The two standalone binders use
`--historical-parent-root` for the explicit root containing their original
`output/runs/...` paths. No implicit relocation guesses are made.

The standalone binders retain the closed 195 TRAIN-file/15-instance declaration
and the 85 TRAIN/VALIDATION-role declaration respectively. Every payload path,
alias and expected size is checked across the entire matrix before the first
allowed payload hash. This complete size preflight strengthens the old
interleaved size/hash loops; it is an identified admission change, not a claim
of byte-for-byte old behavior. Neither binder decodes a tensor.

## Identity and execution boundaries

The shared encoder is named `mityane-artifact-json-v1`. Its duplicate-key policy
is explicit; large integer tokens can remain exact `BigInt`, while number-only
callers reject unsafe integer tokens. New receipt formats use new identities.
Python floating text and historical hashed JSON bytes are not claimed to be
equivalent. Existing source pins are never replaced with a new hash merely to
make a check pass.

New preparation outputs use `.temp/embeddings/runs`. Captured historical Python
readers remain source evidence. The new independent reader is not accepted by a
historical seal. Reader-dependent preparation and capacity admission explicitly
refuse an unaccepted replacement identity; the adapter requires a separately
bound reader interface and equivalence receipt before reading permitted archive
payloads. There is no implicit Python fallback.

The C++ reader currently implements a closed CPU profile. Compressed ZIP,
legacy non-ZIP serialization, shared storage, additional dtypes, CUDA tags,
requires-grad tensors and nonempty hook/metadata state are unsupported. These
limitations must be reconciled with each actual artifact role before cutover.
Archive directory/header scanning may physically overlap member bytes; this
source is not evidence of syscall-level zero reads of all unselected storage.
The arithmetic module does not yet replace every nonlinear-head/PCA/scaler
identity validator. Its README states the numerical and encoding limits.

Existing research launchers and their Python calls remain connected to their
old implementations. This source batch adds opt-in candidate build/test goals;
it does not establish a Python-free research workflow or reduce the baseline
caller count. Remaining family validators, numerical publishers, inline shell
Python and runner integration still require their own ports and validation.
The six retired publication commands remain retired.

## Unexecuted verification source

Written fixtures cover file admission and changes, JSON identities/integers,
fixed preparation recipes, both closed role matrices, capsule source/reader
binding, gate thresholds, malformed archives, forbidden globals/roles/devices,
storage aliases, strided/empty tensors, exact PRNG operations and sensitive
arithmetic. These are test programs, not test results. Static cross-review
corrected admission continuity, BOM handling, directory redirects, pickle graph
cleanup and frame bounds; static review cannot establish runtime correctness.

[`code/scripts/python-migration.mk`](../code/scripts/python-migration.mk) exposes
`test-python-migration-node` and `test-python-migration-candidates` through the
root Makefile for later managed-container execution. The C++ component targets
are included only through the explicit maintenance/candidate route. Normal
maintenance and research targets do not depend on these candidate tests.
Build products and fixtures stay in named `/opt/mityane-geometrics` directories;
no extra package, runtime or component environment configuration was introduced.

All compilation, runtime checks, saved-witness parity, accepted-reader sealing,
Python-blocked integration and actual launcher cutover remain unperformed. The
owner's file-edit-only instruction also defers dashboard validation/refresh:
Hero JSON records the authored work, while generated `public/index.html` is
intentionally unchanged until execution is permitted.

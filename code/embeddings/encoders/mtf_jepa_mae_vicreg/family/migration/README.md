# Export the MTF network and training method

Deliver one explicit source bundle containing the configurable network,
portable training implementation, complete method, examples/configuration,
tests, licenses and scoped evidence. A checkpoint is a separately identified
instance. The [inventory](package.json) defines this package; the
[dependency guide](DEPENDENCIES.md) identifies every numerical source tier and
external requirement. No sibling Mityane checkout is needed after import.

The new portable recipe is CPU/F32 with one intra-op Torch thread and a bound
inherited inter-op setting (12 in the primary proof), with explicit identity
preparation and absolute-update seeding. It differs from the original CLI,
shared evaluation adapter and historical evaluator. Read
[TRAINING_METHOD.md](TRAINING_METHOD.md) and
[COMPATIBILITY.md](COMPATIBILITY.md) before training or accepting raw state.
The [shared contract](../../../COMPONENT_DELIVERY.md) separates component
delivery from application data, substantial training and authenticated artifacts.

## Produce and verify

Use the managed Linux container, with a fresh output leaf under ignored
`.temp/embeddings/exports/`. Create its parent first if needed. From the
repository root:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh',
  'code/embeddings/scripts/export-mtf-family.mjs',
  '--output', '/mityane/.temp/embeddings/exports/mtf-family-2026-10-22-003'
)
```

Independently retain the printed manifest SHA256. Git HEAD describes ancestry;
the manifest pins selected **working-tree bytes**, original paths, sizes,
roles, include roots, quoted-include closure and source hashes. Every origin is
mapped to `sources/<original-relative-path>`. Export refuses existing outputs,
unsafe paths, symlinks, excluded data/binary/environment paths and missing or
ambiguous quoted includes. Admission completes before the new output is made;
failed outputs remain available for inspection.

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh',
  'code/embeddings/scripts/export-mtf-family.mjs',
  '--verify', '/mityane/.temp/embeddings/exports/mtf-family-2026-10-22-003',
  '--expected-manifest-sha256', 'REPLACE_WITH_INDEPENDENTLY_RETAINED_SHA256'
)
```

After relocation a recipient can use an available Node runtime:

```bash
node sources/code/embeddings/scripts/export-mtf-family.mjs \
  --verify /absolute/path/to/bundle --expected-manifest-sha256 RETAINED_PIN
```

Verification reads the bundle only and refuses changed/deleted/extra members,
manifest changes, redirected paths, symlinks and broken include closure. The
independent pin is required. It verifies bytes and declared source closure;
it is not application authentication, a new numerical proof or source truth
beyond the trusted pin. Exporter tests exercise relocated verification with
the origin unavailable as well as tampering, traversal and closure failures.

## Import and acceptance

Retain the verified bundle and record manifest pin, origin/destination
path/hash mapping, any adaptations and local runtime. Keep included source at
the original relative paths or explicitly map every relocation. Integrate
through the consumer's own managed root build using the source list/include
roots in [DEPENDENCIES.md](DEPENDENCIES.md); no full Mityane research environment
or unrelated root Makefile is needed. Fresh use must compile/link the local
import, not read a sibling checkout at runtime.

The consumer performs bounded construction/update/inference/resume/refusal
checks for its imported paths and artifact adapter. It separately owns real
features/units and availability, population/TRAIN membership, concrete recipe,
budgets/scaling/recovery, trusted publication and selection, ModelRef/capture,
exact learned parents and application utility. Import acceptance selects no
checkpoint and admits no new scientific campaign. Equal width does not make
separately trained coordinates compatible.

The [engineering protocol](../verification/PROTOCOL_2026-10-22.md),
[actual results](../verification/RESULTS_2026-10-22.md) and
[engineering receipt](../verification/engineering-proof-2026-10-22.json)
retain the passing primary scope and completed staged-source isolated build/run.
Both runs passed 3,627 native assertions and 21 exporter tests, each using 24
native plus six CLI physical updates (60 combined). Seven compiler dependency
files recorded 25 project dependencies, all within the staged source root.
The local [delivery receipt](DELIVERY_2026-10-22.json) records the
final export outside its own inventory to avoid a circular manifest identity.
Do not infer passing execution from these instructions alone.

The actual tested stage is
`/mityane/.temp/embeddings/exports/mtf-family-2026-10-22-001/sources`, manifest
SHA256 `a3cc64dc41b944db2e495a7bd27eb8540f37b4359cba575dd9cb12e410a64975`.
The final source bundle is the documentation/receipt-only
`mtf-family-2026-10-22-002` successor; its final pin is in the delivery receipt.
All 33 admitted numerical/configuration/fixture/tool pins remain identical.
Preserved primary and isolated receipts retain their actual paths and separate
binary/runtime identities; no rerun at the final successor path is claimed.
The `003` command above is an example fresh leaf for a future export, not a
request to replace the retained `001` or `002` bundles.

Datasets, weights/optimizer archives, generated runs, credentials, SDKs and
images remain separate and excluded. Historical MTF-v1 source/protocols,
original evidence and unknown checkpoint/preprocessing associations retain
their original meanings. The complete CPB family/migration case informed
packaging discipline only; its architecture, scaler, masks, causality and
selected V16 reference are outside this delivery.

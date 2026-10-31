# Export the encoder and its training method

A trainable encoder delivery includes its network **and the method that produces
its weights**: preprocessing, masking and targets, objectives, gradient
treatment, optimization, sampling and continuation. A checkpoint records one
instance. Deliver the reusable training component with its architecture.

Copy this component as one curated source bundle. Copying only `family/` misses
the preserved V16 numerical dependency. The explicit [inventory](package.json)
defines the delivery without mixing in unrelated versions, runs or data.

The [shared ownership contract](../../../COMPONENT_DELIVERY.md) separates
reusable learning semantics from application recipes and artifact trust.
The [Cuwacunu consumer review](CONSUMER_REVIEW_2026-10-22.md) found this source
delivery sufficient for its current pipeline and records the remaining boundary.

| Included | Purpose |
| --- | --- |
| Family headers and complete V16 include tree | Fresh construction, forward, fitting, masks, objectives, optimization, training state and inference codecs |
| Configurations and native engineering fixtures | Alternative input/capacity/output examples, exact resume and compatibility checks |
| Native training CLI, observation archive and synthetic writer | Concrete fresh/fit/train/save/resume/export workflow independent of market data |
| [Training method](TRAINING_METHOD.md), API and example guides | Numerical rules, insights, failure behavior and differences from historical research |
| Frozen V16 protocol/results and research reference/ranking | Scientific rationale, successes and protection failures |
| Licenses, dependencies, provenance and engineering receipts | Origin identities, requirements and limits of existing verification |
| Exporter/verifier, inventory and SHA256 manifest | Repeatable transfer and offline byte verification |

Datasets, weights, optimizer archives, generated runs, credentials, SDKs and
images are separately identified artifacts. Fresh training needs no reference
checkpoint. See [DEPENDENCIES.md](DEPENDENCIES.md) for runtime/build requirements.

## Produce and verify

Run from Mityane's repository root, choosing a new leaf each time. Create its
parent directory first if needed; outputs belong under ignored `.temp/`.

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh',
  'code/embeddings/scripts/export-cpb-family.mjs',
  '--output', '/mityane/.temp/embeddings/exports/cpb-family-2026-10-22'
)
```

Retain the printed manifest SHA256 independently with the delivery. The manifest
pins the selected **working-tree bytes**, original paths, roles, include roots
and source closure. Its Git HEAD describes ancestry; it does not claim local
changes were committed. `sources/` preserves repository-relative paths. Export
refuses existing output, missing quoted-include dependencies, unsafe paths and
symlinks. Failed output is preserved for inspection.

After transfer, verify with the independently retained pin:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh',
  'code/embeddings/scripts/export-cpb-family.mjs',
  '--verify', '/mityane/.temp/embeddings/exports/cpb-family-2026-10-22',
  '--expected-manifest-sha256', 'REPLACE_WITH_RETAINED_SHA256'
)
```

An isolated recipient can use an available Node runtime to run
`node sources/code/embeddings/scripts/export-cpb-family.mjs --verify BUNDLE
--expected-manifest-sha256 PIN`. Verification reads only the bundle. Node does
copying and metadata; numerical training/inference are C++. Verified bytes and
include closure do not constitute a new compile or training/quality result at
relocated consumer paths.

## First consumer: Cuwacunu

A read-only comparison on 22 October found all 49 frozen V16 files and the
three maintained family numerical headers already byte-identical in
`cuwacunu_dev`. Its local `code/encoder_training/versions/v1/` owns application
data/calibration, training/resume, publication, selection and capture. The
generic example complements that pipeline.

| Origin under bundle `sources/` | Cuwacunu ownership |
| --- | --- |
| `code/embeddings/encoders/causal_patch_bottleneck_mae/family/include/` | `code/encoders/causal_patch_bottleneck_mae/family/include/` |
| `code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v16/include/` | Existing byte-preserved `code/encoders/causal_patch_bottleneck_mae/versions/v16/include/` |
| Family configurations and applicable tests | Family examples/tests through the existing consumer root build |
| Generic CLI, observation format and writer | Optional reference/example source, separate from application pipeline |
| Method, dependencies, research and migration notes | Retained bundle documents, linked from consumer documentation |
| Raw reader, splits, sampling and application envelopes | `code/encoder_training/versions/v1/` |

Do not overwrite the consumer's build integration with Mityane's `family.mk` or
replace its application pipeline with the example. Cuwacunu uses counter-seeded
**SplitMix64 rejection-uniform** row sampling; the example uses **MT19937_64 with
modulo selection**. These are different complete recipes over shared training
primitives. Checked sidecars, ModelRef and authenticated selection records also
differ from raw family archives. Equal dimensions do not establish artifact or
resume interchangeability.

The example permits increased total budgets on resume. Cuwacunu instead binds
the original declared budget in its application recipe. Both continuation
contracts remain explicit and are tested within their own scopes.

Each import records the manifest pin, origin/destination path/hash mapping,
adaptations, runtime and its own tests. Relocation creates a new path-bound
source identity. Original receipts preserve their original paths/binaries;
copying a receipt does not certify an import. Review refreshes against current
consumer source/provenance before replacement. This export does not change the
live application pipeline or start training.

The [29 October handoff](HANDOFF_2026-10-29.md) is the shared-clock note for
Cuwacunu. The 22 October bundle predates the timeframe channel contract.
The shared-clock instance is CPB-family width 128, seed 949102, with its
settings and local inference archive named in that note. The packed-channel
instance to import is the [31 October handoff](HANDOFF_2026-10-31.md), seed
949501. Further Mityane training on that packed layout is closed. The research reference remains V16 seed 946103.

## Historical boundary

Frozen study protocol/results travel as context. Its market runner, data,
admitted calibration population and complete execution closure remain in
Mityane. This bundle supports training the maintained family; it does not claim
to reproduce the selected V16 checkpoint. Historical fitting arithmetic, row
sampling and context deletion differ from the portable example.

Original engineering receipts travel unchanged. Their source/artifact pins
retain their original meanings; named artifacts are not automatically included.
The research reference remains V16 seed946103/final2048/native36 with its mixed
verdict. Packaging creates no encoder-quality finding.

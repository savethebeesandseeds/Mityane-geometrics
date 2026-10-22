# Deliver CPB-v17 with its learning method

The [explicit inventory](package.json) extends the corrected configurable CPB
delivery with the V17 recipe, typed archives, research configuration, native
engineering checks and runnable fresh/resume CLI. It includes the full family
and preserved V16 numerical headers, generic observation reader/writer,
method/dependency/license notes, historical research context and the completed
study's results/compact record and operation contract. The inventory selects
**93 files**, including both saved closeout documents and the independent review. Every source
file keeps its repository-relative path under the bundle's `sources/` tree.

The actual learning rule is Gaussian NLL plus a fixed coefficient times the
mean of hidden temporal Huber losses at lags 1/4/16. **Candidate 0.03 and control 0.1
use the same corrected component path**. Preprocessing, masks, support,
reductions, detached variance-head input, optimizer/clipping and explicit
counter streams remain shared. Changing the coefficient does not change the
native layout, architecture or genuine supported geometry/capacity controls.
The [version guide](../README.md) specifies the API and artifact identities;
the [common training method](../../../family/migration/TRAINING_METHOD.md)
describes the retained numerical mechanisms and historical distinctions.

V17 inference and training archives have distinct outer typed kinds, an exact
F64 temporal coefficient and complete version/recipe bindings. Resuming requires
the expected coefficient and application recipe. Inference cannot resume
training. Nested family/V16 codec fields identify structural serialization;
the outer V17 recipe records the actual training. It is not an inherited trained
ancestor. Application trust envelopes, ModelRef/selection, real data/splits,
budgets/scaling/recovery and downstream compatibility stay application-owned.

The managed-container CUDA engineering fixture passed **859 assertions** and
**126 named gradient comparisons**, using **16 physical synthetic updates**,
**2 preprocessing fits** and **23 expected refusals** across native36/native15.
This verifies the numerical recipe and its construction/update/export/resume
interfaces. It does not establish scientific quality or promote a checkpoint.
The paired 256-update research study completed all six trajectories in
**1536 scientific updates/1536 attempts**, using 12 planned fresh128/resume256
processes, with six full-support independently audited readouts. The declared
continuation criterion **failed**: median candidate-minus-control R2 was
-0.003029098 for trend (1/3 nonnegative pairs) and -0.008995412 for activity
(0/3); volatility -0.002418553 passed its -0.01 median floor. Relation
+0.003185947 (2/3) is descriptive. **No new best encoder or reference change
followed; selected V16 seed 946103/final 2048 remains the reference.**

The local TRAIN diagnostic did not support raw temporal-gradient dominance;
backbone cosines were negative in four of eight batches. Raw gradient norms
and cosines do not quantify AdamW-preconditioned update contributions. Actual
outcomes, full-precision scores and scientific limitations are preserved in the
immutable
[study results](../../../../../evaluation/protocols/cpb_v17_balance_v1/RESULTS.md)
and [compact record](../../../../../evaluation/records/CPB-v17.balance-001.json).
The [independent review](../../../../../evaluation/protocols/cpb_v17_balance_v1/REVIEW.md)
checks all1536 saved update events and clarifies that the single primary-task
gain was trend seed947102; activity worsened in every pair.
The final package carries these completed records alongside the learning method.

## Build and fresh/resume example

In Mityane, build through the existing root task and managed Linux container:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/task.sh', 'cpb-v17-example', '-j1',
  'build-cpb-v17-trainer', 'build-observation-window-fixture'
)
```

The native trainer accepts the generic `observation_window_dataset_v1` archive
with declared role TRAIN. A source/split declaration needs actual application
admission; the format cannot independently prove calendar membership. Supply
an independently retained dataset and family settings, or use the included
synthetic writer as described in the [generic example](../../../family/examples/TRAINING.md).
Choose a new output leaf; these small commands illustrate behavior and are not
recommended research budgets. Inside the managed Linux container:

```bash
/opt/mityane-geometrics/embeddings/build/cpb-v17-example/cpb_v17_train \
  --temporal-weight 0.03 --settings /absolute/settings.conf \
  --dataset /absolute/admitted-TRAIN.pt --output /absolute/new-prefix \
  --seed 947101 --steps 2 --max-attempts 8 --batch-size 2 \
  --row-seed 948101 --mask-seed 949101 --context-seed 950101 --save-every 1

/opt/mityane-geometrics/embeddings/build/cpb-v17-example/cpb_v17_train \
  --temporal-weight 0.03 --settings /absolute/settings.conf \
  --dataset /absolute/admitted-TRAIN.pt --output /absolute/new-resume \
  --resume /absolute/new-prefix/training-final.pt \
  --steps 3 --max-attempts 8 --batch-size 2 \
  --row-seed 948101 --mask-seed 949101 --context-seed 950101 --save-every 1
```

Use the same executable, dataset/settings bytes, complete batching/sampling
policy, coefficient, optimizer options and stream seeds on resume; do not
refit preprocessing. Step/attempt values are total ceilings. The portable CLI
allows their increase and has no context deletion, scheduler, accumulation,
distributed/sharded loader or production recovery service. It loads the dataset
in memory. A control uses `--temporal-weight 0.1` in a separate fresh instance.
Keep experiment population, physical work, source/configuration, fitted state,
training recipe and selected inference artifact explicitly identified.

## Source export and recipient verification

Choose the final selected source/documentation snapshot before export. Preserve
all earlier 89-file preliminary bundles and their original pins; they remain
snapshots of their own inventories. Prepare an existing output
parent under ignored `.temp/embeddings/exports/` and run from the repository
root, using a new leaf:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh', 'code/embeddings/scripts/export-cpb-v17.mjs',
  '--output', '/mityane/.temp/embeddings/exports/cpb-v17-NEW'
)
```

Retain the printed manifest SHA256 independently. The wrapper uses the common
source exporter and its strict byte/dependency/path checks with the V17 inventory.
It preserves original paths and records the selected working-tree snapshot;
Git HEAD is descriptive ancestry, not a substitute for the actual member pins.
After transfer, an isolated recipient can verify with an available Node runtime:

```bash
node BUNDLE/sources/code/embeddings/scripts/export-cpb-v17.mjs \
  --verify BUNDLE --expected-manifest-sha256 INDEPENDENT_RETAINED_PIN
```

Verification needs no original checkout. A consumer imports the source through
its own authoritative managed Linux build and records destination provenance
and applicable integration tests. Numerical code uses C++20/LibTorch/CUDA with
the [declared dependencies](../../../family/migration/DEPENDENCIES.md); copying,
metadata and tests use Node standard libraries. No Python or npm package is
required by the portable numerical training or source export path.

Original engineering receipts remain scoped to their pinned source/runtime;
packaging does not renew them. The V17 [prospective study protocol](../../../../../evaluation/protocols/cpb_v17_balance_v1/PROTOCOL.md)
travels as study context. Its admitted market datasets and complete diagnostic/
readout execution closure are separate. The source package trains the corrected
component, without claiming to reproduce historical V16 fitting or a saved
checkpoint's score. The linked results and compact record carry the actual
256-update study outcomes; engineering passes do not substitute for them.

Datasets, checkpoints, optimizer archives, generated runs, credentials, SDKs and
images remain separately identified artifacts and are not added to public Git.
Fresh construction needs no pretrained checkpoint or sibling research checkout.

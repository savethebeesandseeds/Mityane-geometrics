# Portable MTF family and learning method

This maintained C++ layer delivers MTF-v1 construction and its learning method
as a reusable component. A configuration selects geometry and capacity; a recipe
binds preparation, sampling, optimization and stochastic rules; trained weights
are a separate instance. No reference checkpoint is required for fresh use.

The retained MTF headers and historical runners remain unchanged. The portable
layer has its own source, recipe and raw-format identities. Read the
[training method](migration/TRAINING_METHOD.md),
[compatibility contract](migration/COMPATIBILITY.md) and
[source requirements](migration/DEPENDENCIES.md) before importing it.

## Surface and ownership

The API lives in `embedding::encoders::mtf_jepa_mae_vicreg::family` in
`include/embedding/encoders/mtf_jepa_mae_vicreg/family.h`, implemented by
`src/family.cpp`. `Recipe` carries the full supported recipe. `Session` supplies
fresh construction, one actual learning update, progress, typed training-state
save/resume and online inference export. Inference loading uses the separately
identified model-only format. Shapes and trainable/frozen parameter inventories
come from actual configuration/construction.

Initial portable execution is **CPU, F32, one intra-op Torch thread**.
`Settings.threads=1` controls intra-op execution. The inherited inter-op setting
is recorded and bound (12 in the primary proof), rather than forced to one.
Preparation is
`identity-cpu-f32-observed-v1`: floating CPU BCHF observations and explicit bool
support, cast to F32 with missing positions zeroed and observed nonfinite or
overflow inputs refused. Each TRAIN row needs some observed support. There is
no fitted scaler and no hidden TRAIN fit. Applications may perform their own
explicitly identified preprocessing before this boundary. This does not turn
the historical evaluator's fitted normalization into an identity recipe.

`replacement-step-seeded-v1` samples rows with replacement. The distinct
`full-partition-step-seeded-v1` uses the entire supplied partition per update.
Both are new portable recipes with absolute-update seed rules; neither means
the original CLI trajectory or the shared evaluation adapter trajectory.
Initialization uses `manual_seed(seed)`; each update uses
`seed + completed_updates + 1` before sampling and forward computation
(`portable-absolute-step-seed-plus-one-v1`).
`settings.steps` is the declared total budget. Resume must retain that budget
and all bound configuration/data/recipe identities. Raw state does not confer
application authentication, selection, ModelRef or population admission.

Mityane owns the mechanism, source delivery and scoped engineering evidence.
Applications own features/units, actual TRAIN membership, concrete supported
choices, substantial training/scaling/recovery, trusted publication/selection,
representation capture, fitted descendants and downstream utility. The shared
[ownership contract](../../COMPONENT_DELIVERY.md) governs that separation.

## Build and bounded example

Use the root Makefile and `code/scripts/task.sh` through the existing managed
Linux container. Build products stay under `/opt/mityane-geometrics`; selected
artifacts go under ignored `.temp/`. No component environment or SDK install is
provided. A consumer supplies its own managed build using the exact include
roots and sources in [DEPENDENCIES.md](migration/DEPENDENCIES.md).

The C++ example is `code/embeddings/scripts/mtf_family_train.cpp`.
`family/config/small.conf` and `alternative.conf` are artificial-data examples,
not universal architecture restrictions. The example covers synthetic archive
creation, bound fresh training/resume and inference export/serving:

| Command | Required workflow |
| --- | --- |
| `synthetic` | `--config FILE --output NEW.pt [--rows 5]`; writes arbitrary-F artificial F64 observations and explicit support |
| `prepare` | `--config FILE --input RAW.pt --output NEW.pt`; writes canonical identity-prepared F32 values/support |
| `train` | `--config FILE --input RAW.pt --checkpoint NEW.pt --source-sha256 SHA --input-source ID --split ID --recipe ID --sampling POLICY --budget TOTAL [--steps COUNT] [--resume OLD.pt] [--inference NEW.pt]`; optional explicit `--beta1`, `--beta2`, `--epsilon`, `--amsgrad` |
| `encode` | `--config FILE --input RAW.pt --model INFERENCE.pt --source-sha256 SHA --state-sha256 SHA --output NEW.pt`; checks expected pins, serves online global/channel/time/frequency outputs and validity |

`--budget` is fixed across training invocations; `--steps` selects only this
invocation's remaining work. Outputs must be new paths. Its `--help` lists all
options. Source/split/recipe identities are caller
declarations; they do not independently prove the real population's admission.

The supplied bounded orchestration example creates F5 F64 input, prepares it,
compares three whole updates with one plus resumed two, exports/reloads inference
and writes a JSON result. After building `build-mtf-family-trainer`, run inside
the managed container with the actual source inventory SHA and a new output:

```bash
bash code/scripts/node.sh \
  code/embeddings/encoders/mtf_jepa_mae_vicreg/family/tests/cli_example.mjs \
  --trainer /opt/mityane-geometrics/embeddings/build/YOUR_SESSION/mtf_family_train \
  --config code/embeddings/encoders/mtf_jepa_mae_vicreg/family/config/alternative.conf \
  --source-sha256 REPLACE_WITH_ACTUAL_SOURCE_INVENTORY_SHA256 \
  --output /mityane/.temp/embeddings/runs/YOUR_SESSION/cli-001
```

This is six physical optimizer updates, not substantial training. Fresh,
training-state and inference use the typed APIs documented in the header.
Sessions touch global Torch RNG/thread settings and must be serialized.
Exact training resume binds the same executable/runtime and update boundary;
failed updates have no transactional rollback.

## Meaning and evidence

The encoder consumes observation windows. Its frequency descriptors use full
configured windows, and positive-depth token encoding mixes global context.
Per-channel outputs remain contextual. No prefix-causality, independent-channel,
forecasting, intervention or application-utility guarantee is added here.
Missing support is distinct from observed zero.

The [source audit](../delivery/COMPONENT_TRAINING_AUDIT_2026-10-22.md)
records retained capabilities, gaps and historical unknown associations. The
[prospective verification protocol](verification/PROTOCOL_2026-10-22.md)
defines the bounded engineering scope. The
[actual results](verification/RESULTS_2026-10-22.md) and
[engineering receipt](verification/engineering-proof-2026-10-22.json)
record the primary passing scope: 3,627 native assertions, 24 native plus six
CLI physical updates, and 21 exporter tests. The separately built staged source
bundle also passed 3,627 native assertions, 24 native plus six CLI updates and
21 exporter tests. The combined physical optimizer work is 60 updates. Seven
compiler dependency files listed 25 project dependencies and none outside the
staged source root. No model-quality finding follows.

Isolation used staged bundle `mtf-family-2026-10-22-001` with manifest SHA256
`a3cc64dc41b944db2e495a7bd27eb8540f37b4359cba575dd9cb12e410a64975`.
The admitted 33-file proof source SHA256 was identical to the primary run;
the isolated executable/runtime identities and paths are separately retained.
The final `mtf-family-2026-10-22-002` is a documentation/receipt successor
retaining those admitted bytes. It reuses the staged proof without asserting
a rerun at the final path. The [delivery receipt](migration/DELIVERY_2026-10-22.json)
binds both manifests, paths, source mapping and actual isolated evidence.

Use the [migration guide](migration/README.md) to produce an explicit, byte-bound
source bundle. Datasets, checkpoints, generated runs, credentials, SDKs and
container images travel separately and are excluded from that source package.

# Deliver CPB-v18 with its training-mask method

The [explicit inventory](package.json) selects **106 files**: the configurable
network and full preserved V16 numerical dependency closure, TRAIN fitter,
V18 mask/training/artifact API, runnable fresh/resume trainer, configurations,
native and source-export tests, dependency/license notes and bounded research
context. The training method travels with the network. Family, configuration,
recipe and trained instance remain separate objects under the
[component delivery contract](../../../../COMPONENT_DELIVERY.md).

The inventory also carries the comparison's V17 interface/configuration/trainer,
fixed-regression dependency, original scalar-audit source and V18 study sources.
These explain how the method was evaluated. The checkout-bound study runners
require separately preserved datasets, target artifacts, original executables,
control prefixes and admission/receipt trees. The bundle does not recreate that
execution environment or authorize a replay. Export the final snapshot only
after the linked results, compact record and independent review are finalized;
the strict exporter refuses missing inventory members.

Every delivered file retains its repository-relative path under `sources/`.
The source package contains no data, fitted statistics, trained weights,
optimizer archives, generated run outputs, credentials or third-party SDKs.
Fresh V18 construction needs no pretrained checkpoint or sibling checkout.

## What changes and what remains reusable

V18 changes the training query geometry. `Recipe{MaskMode::contiguous}` hides
one legal nonwrapping span on the original physical patch grid independently
per example/channel. The quota remains
`clamp(floor(mask_ratio*M + 0.5), 1, M-2)` for M observed patches. Crop at the
cutoff first; a patch is observed if any real cell is observed. A legal span
contains q observed patches with no missing patch inside it. M<3 or no legal
span produces explicit channel abstention, with no fallback. Private final
padding is cropped back to logical history; at least two observed patches
remain visible. Missing patches are never joined by compressing their ranks.

For each absolute attempt, a local `mt19937_64` uses the family's counter seed.
Rows traverse physical B*C order. For n legal starts, reject draws below
`(uint64_t(0)-n)%n`, then select `draw%n`. Singleton sets consume one draw;
unsupported channels consume none. Uniform legal starts give unequal per-patch
exposure near boundaries. Scattered mode calls the unchanged complete family
mask generator, including its original shuffle and RNG behavior. These are
different training recipes even when their hidden-patch quotas are equal.

The architecture, native layout, supported geometry/capacity, fresh constructor,
TRAIN fitting and numerical update remain the configurable family. The loss is
unchanged Gaussian NLL plus **0.1 times the fixed mean of temporal Huber losses
at lags 1/4/16**. Retain original hidden-observed Q at both endpoints, sqrt2
scaling, Huber delta1, reductions and graph-connected zeros for empty lags.
Targets remain detached; variance-head input gradients remain stopped. Use one
global clip and one AdamW update. Contiguity intentionally changes lag-pair
support; it does not expand Q or renormalize the objective over active lags.
On H128/P8 full support, both modes hide four of16 patches; sparse or partial
grids can have different hidden-cell counts. A native15 H33/P6 two-patch span
can have no lag16 pairs and leaves that lag unsupported.

The [version guide](../README.md) specifies exact API/mask/artifact identities.
The [common training method](../../../family/migration/TRAINING_METHOD.md)
describes the retained fitter, network, gradients and update machinery.
`MODEL_TAG=CPB-family` identifies configurable construction; the outer V18
recipe identifies the actual training-mask law. This is continued architecture
and training-method research, with a portable component as its deliverable.

## Construction, fitting and artifact contract

Include `embedding/encoders/causal_patch_bottleneck_mae/versions/v18.h` and use
namespace `embedding::encoders::causal_patch_bottleneck_mae::v18`. Genuine
constraints and derived shapes/counts remain in the configurable family;
[research.conf](../config/research.conf) is one geometry example. The same
family surface exposes configuration/settings, fresh construction, raw
histories, fitted scaler/bank, TRAIN preprocessing fitting and optimizer
construction. Applications bind real feature semantics and TRAIN membership.

The V18 surface adds `Recipe`, `make_mask`, `forward_at`, `train_step` and typed
save/load entry points. `forward_at` receives an explicit mask and preserves
the unchanged numerical forward/objective. `train_step` selects the recipe's
mask before the retained backward/clip/update. Its result includes actual Q
pair counts in lag1/4/16 order, CPU B,C selected-start/legal-span/quota tensors,
selected patch count and insufficient-support/no-span abstentions. Ineligible
batches consume an attempt without an optimizer update. Scattered starts are
-1 and legal-span counts0 because those fields do not describe that mask law.

Inference kind is `cpb_v18_mask_geometry_inference_v1`; resumable training kind
is `cpb_v18_mask_geometry_training_v1`. Both bind the mask enum, full recipe and
fixed F64 coefficient0.1. Typed loaders require the expected version/mode and
configuration. Training state additionally binds caller source/split/recipe,
optimizer options/content/order/steps, progress, frozen preprocessing and
counter streams. Loading returns the numerical payload plus recipe; inference
cannot resume training. V17/family kinds cannot impersonate V18 kinds.

Nested family/V16 fields describe structural serialization only. The outer
V18 recipe is authoritative for the method; it is not a trained-weight ancestor
or permission to strip recipe identity. Equal embedding width and successful
loading do not establish interchangeable coordinates. Applications own trusted
publication, ModelRef/selection, data admission, substantial training budgets,
recovery and final docking, including compatible fitted descendants.

## Build and fresh/resume example

In Mityane, use the authoritative root task and existing managed Linux container:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/task.sh', 'cpb-v18-example', '-j1',
  'build-cpb-v18-trainer', 'build-observation-window-fixture'
)
```

The [trainer source](../../../../../scripts/cpb_v18_train.cpp) accepts generic
`observation_window_dataset_v1` with caller-declared role TRAIN. The declaration
does not prove actual split membership. Supply an independently admitted TRAIN
archive and retained settings, or use the included synthetic writer as in the
[generic training example](../../../family/examples/TRAINING.md). New output
leaves are required. These tiny commands show the API, not research budgets:

```bash
/opt/mityane-geometrics/embeddings/build/cpb-v18-example/cpb_v18_train \
  --mask-mode contiguous --settings /absolute/settings.conf \
  --dataset /absolute/admitted-TRAIN.pt --output /absolute/new-prefix \
  --seed 947101 --steps 2 --max-attempts 8 --batch-size 2 \
  --row-seed 948101 --mask-seed 949101 --context-seed 950101 --save-every 1

/opt/mityane-geometrics/embeddings/build/cpb-v18-example/cpb_v18_train \
  --mask-mode contiguous --settings /absolute/settings.conf \
  --dataset /absolute/admitted-TRAIN.pt --output /absolute/new-resume \
  --resume /absolute/new-prefix/training-final.pt \
  --steps 3 --max-attempts 8 --batch-size 2 \
  --row-seed 948101 --mask-seed 949101 --context-seed 950101 --save-every 1
```

The coefficient is fixed0.1; there is no temporal-weight CLI option. A separate
fresh scattered instance uses `--mask-mode scattered`. The portable V18 typed
scattered archive is not the old V17 study control's typed state. The admitted
research comparison preserves and continues the original V17 executable/state;
it does not convert old state into V18 to obtain a convenient control.

Fresh fitting visits all declared TRAIN rows once in source-order batch chunks
and freezes the input scaler/bank; resume never refits. Sampling uses
`mt19937_64(counter_seed(row_seed,attempt)) % row_count`, with replacement.
Preserve that row algorithm as well as seeds; its modulo rule is distinct from
the candidate's unbiased bounded start selection. Context deletion is none.
The CLI logs exact decimal-string mask counter seeds, selected starts/quotas/
legal-span counts, Q-pair counts, observed hidden targets and update progress.

Resume requires the same executable, dataset/settings bytes, mask mode,
batching/sampling policy, optimizer options and stream seeds. Step/attempt
limits are total ceilings and may increase; they are not additional work.
The CLI loads data in memory and has no scheduler, gradient accumulation,
distributed/sharded loader or production recovery service. These limits do not
transfer ownership of application orchestration into the component.

The numerical training closure uses V18, family, V16 and generic observation
include roots. The optional fixed study readout additionally uses the V17
interface and fixed-regression root. The supplied build fragments describe
Mityane's root integration; they are not a standalone build/environment.
Recipients use their own authoritative Linux build with C++20, the declared
LibTorch/CUDA/ABI and licenses from the
[dependency guide](../../../family/migration/DEPENDENCIES.md). The header-only
model requires no V17 interface for ordinary fresh V18 training. Source copying,
verification and optional metadata tests use Node standard libraries, with no
Python or npm dependency.

## Verification and research evidence

The native [engineering fixture](../tests/engineering_test.cpp) checks exact
scattered-control mask/loss/named-gradient/update/optimizer parity, physical
span legality/quota, holes and explicit abstention, partial patches, cutoff-first
future poisoning, exact three-step continuation and typed version/mode/options
refusals at native36/native15. Actual source-bound receipts determine the
passed counts and runtime scope; packaging does not renew an old receipt.

The [prospective protocol](../../../../../evaluation/protocols/cpb_v18_span_v1/PROTOCOL.md)
and [operation contract](../../../../../evaluation/protocols/cpb_v18_span_v1/CONTRACT.md)
bind the study. Its [results](../../../../../evaluation/protocols/cpb_v18_span_v1/RESULTS.md),
[compact record](../../../../../evaluation/records/CPB-v18.span-001.json) and
[independent review](../../../../../evaluation/protocols/cpb_v18_span_v1/REVIEW.md)
carry the actual outcome, physical/reused work accounting and limitations.
Engineering success establishes implementation behavior; research quality and
application utility need their own evidence. A source package does not promote
a candidate or replace the separately maintained research reference.

The study uses three seeds and a fixed1024-update endpoint, preserving completed
old control prefixes rather than repeating them. Its settings bind the existing
V17 research.conf; the supplied V18 geometry example does not replace those
historical bytes. The unchanged linear readout uses already-consumed development
data; it does not establish fresh holdout, full-budget forecasting or global-best
quality. Missing detailed mask witnesses from the preserved control logs remain
unavailable and are not presented as measured observations.

## Source-only export and recipient smoke check

Before a release, run the existing generic integrity tests and the new thin
wrapper fixture inside the managed container, with TMPDIR in a named `/opt/`
build directory. They create only synthetic source files and invoke no model:

```powershell
.\container.ps1 -Action exec -Command @(
  'env', 'TMPDIR=/opt/mityane-geometrics/embeddings/build/cpb-v18-span-001/tmp',
  'bash', 'code/scripts/node.sh', '--test',
  'code/embeddings/scripts/tests/export-cpb-family.test.mjs',
  'code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v18/migration/exporter.test.mjs'
)
```

Prepare an existing ignored `.temp/embeddings/exports/` parent and choose a new
leaf after final closeout. Run the strict source wrapper from the repository:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh', 'code/embeddings/scripts/export-cpb-v18.mjs',
  '--output', '/mityane/.temp/embeddings/exports/cpb-v18-NEW'
)
```

The common exporter validates every inventory member and quoted C++ include
before writing, preserves exact bytes/paths, and prints a manifest SHA256.
Retain that pin independently. Git HEAD is descriptive ancestry; the member
hashes and source-closure/manifest pins identify the selected working-tree
snapshot. Do not rewrite earlier source packages or their receipts.

The intended delivery smoke check runs the delivered wrapper from a relocated
bundle, with the original checkout unavailable and the independently retained
pin supplied. The focused wrapper test exercises that isolation, checks its
own inventory routing and rejects a wrong pin or invalid/mixed options. A
recipient can perform the actual source-only verification without Mityane:

```bash
node BUNDLE/sources/code/embeddings/scripts/export-cpb-v18.mjs \
  --verify BUNDLE --expected-manifest-sha256 INDEPENDENT_RETAINED_PIN
```

The verifier rejects edited, missing or added members. Preserve the resulting
delivery receipt outside the bundle to avoid recursive identity. A recipient
then records destination provenance and its own numerical/integration acceptance
under the component contract. This smoke check verifies source delivery; it
does not train, replay research, renew native numerical evidence or establish
downstream utility. Dataset/model access and upstream binary license obligations
remain with their separately identified artifacts and runtimes.

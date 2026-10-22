# Deliver V19 with the complete teacher-learning method

The [explicit inventory](package.json) selects 106 source files: the configurable
student network and preserved numerical dependencies, TRAIN fitter, V19 teacher
and predictor implementation, combined optimizer and serialization, runnable
trainer, tests, configurations, research context and dependency/license notes.
The source package carries the training method with the encoder. Keep family,
configuration, recipe and trained instance separate under the
[component contract](../../../../COMPONENT_DELIVERY.md).

The teacher and predictor are training machinery. Inference uses the original
student with its configured native shape. A student inference checkpoint alone
cannot resume this method: the typed training artifact also needs teacher
weights, predictor weights, combined AdamW moments, initialization identities,
progress and fitted preprocessing.

The inventory contains comparison context from V17 and the original scalar
audit. Checkout-bound study runners additionally require preserved V17/V18
datasets, control states, binaries and evidence trees. Those artifacts are
separate from a portable fresh-training component. The package does not contain
data, fitted statistics, checkpoints, optimizer archives, credentials or SDKs.
Export only after the linked results, compact record and review are finalized;
the strict exporter refuses missing members.

## Learning semantics

The student retains the family architecture, scattered masks and objective
`GaussianNLL + 0.1 * mean(temporal Huber at lags 1,4,16)`. The V19 addition is
`0.1 * A`, directly supervising the learned prefix of width D, derived from
`dimensions(config).shape_width`. Literal spectral and change coordinates are
excluded from the auxiliary predictor and target. Inference still returns the
student's complete configured native representation.

The masked student supplies a learned-D vector to a training-only predictor:
`Linear(D,2D) -> GELU -> Linear(2D,D)`, both linear layers with biases and no
batch normalization or dropout. Its parameter count is `4*D*D + 3*D`. A separate
EMA teacher encodes the full observed history at the **same cutoff**. Crop before
reading values or masks; the teacher sees no suffix beyond that boundary.

Promote predictor and detached teacher outputs from F32 to F64. Normalize each
vector by `sqrt(clamp_min(sum(v*v), (1e-6)^2))`; clamping before sqrt preserves
finite derivatives at zero. The row loss is `2 - 2*dot(normalized prediction,
normalized target)`. Average over the intersection of base NLL eligibility,
masked-student validity and full-teacher validity. Empty auxiliary support is
a graph-connected zero and does not independently skip a supported NLL update.

Backpropagate the combined loss once. Globally clip the student's learned
parameters and predictor together, then apply one AdamW update. The teacher is
excluded from AdamW and receives no gradients. Only after a successful update,
under `NoGradGuard`, apply `teacher.mul_(0.99).add_(student, 0.01)` to learned
teacher parameters. Keep fixed parameters, buffers and fitted preprocessing
exact. A skipped attempt advances its attempt counter without an EMA update.

The retained core validates original `requires_grad` flags. Keep teacher flags
unchanged and use NoGrad forward/EMA; disabling flags after construction would
violate that structural contract. Teacher gradients must remain undefined.

Diagnostics report actual eligible-row counts, auxiliary loss, teacher forward
and update counts, raw-vector near-zero norms and population coordinate
dispersion for student, teacher and predictor. They expose possible collapse;
the method does not promise collapse prevention or change the fixed research
readout criteria. No variance/covariance safeguard is silently added.

## Construction and complete continuation

Include `embedding/encoders/causal_patch_bottleneck_mae/versions/v19.h` and use
namespace `embedding::encoders::causal_patch_bottleneck_mae::v19`:

- `PreprocessingFitter` accepts independently admitted TRAIN histories/cutoffs
  and freezes the family input scaler and bank.
- `fresh(config, scaler, bank, encoder_seed, predictor_seed)` creates the
  student, separate equal initial teacher, and configured predictor. The base
  student construction occurs before predictor initialization. The portable
  API accepts an explicit nonnegative predictor seed; this study chooses
  `encoder_seed + 4000`.
- `TrainingBundle` retains `model`, `scaler`, `teacher`, `predictor`, exact
  configuration, predictor seed, teacher update count and initial identities.
- `make_optimizer`, `train_step`, `save_training_state` and
  `load_training_state` own the complete joint method. Optimizer parameter
  order is student learned parameters followed by predictor parameters.
- `save_checkpoint` and `load_checkpoint` export/load the served student only.
  Their new typed kind and outer recipe identify V19; nested family fields
  identify the preserved structural codec and base objective.

Fresh fitting visits all caller-declared TRAIN rows once in source-order batch
chunks; resume never refits. Actual TRAIN membership remains caller-owned and
must be independently established. Rows use a per-attempt `mt19937_64` counter
stream with modulo sampling and replacement. Masking uses the unchanged family
counter stream. Dropout is zero, the predictor and teacher have no stochastic
training layers, and those explicit streams suffice under this method's scope.

Training artifacts bind recipe, configuration, student/teacher/predictor,
fitted state, initialization, source/split, parameter order, optimizer moments
and step counters. Incompatible artifacts are refused. The portable CLI also
binds exact executable/settings/dataset bytes, batch/fit/sampling policy,
optimizer options and initialization/stream seeds. Total stop budgets may
increase; they are not additional-step counts or a reason to refit.

## Build and example

Inside Mityane, use the existing managed container and authoritative root build:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/task.sh', 'cpb-v19-example', '-j1',
  'build-cpb-v19-trainer', 'build-observation-window-fixture'
)
```

The [trainer](../../../../../scripts/cpb_v19_train.cpp) consumes generic
`observation_window_dataset_v1` archives. Use independently admitted TRAIN data
or the included synthetic writer described by the
[family example](../../../family/examples/TRAINING.md). Example invocations:

```bash
cpb_v19_train --settings /absolute/settings.conf \
  --dataset /absolute/admitted-TRAIN.pt --output /absolute/new-prefix \
  --seed 947101 --predictor-seed 951101 --steps 2 --max-attempts 8 \
  --batch-size 2 --row-seed 948101 --mask-seed 949101 \
  --context-seed 950101 --save-every 1

cpb_v19_train --settings /absolute/settings.conf \
  --dataset /absolute/admitted-TRAIN.pt --output /absolute/new-resume \
  --resume /absolute/new-prefix/training-final.pt --predictor-seed 951101 \
  --steps 3 --max-attempts 8 --batch-size 2 --row-seed 948101 \
  --mask-seed 949101 --context-seed 950101 --save-every 1
```

The CLI is a single-device in-memory example, with no scheduler, accumulation,
distributed loader or application publication service. Consumers own their
data binding and orchestration. The supplied Make fragment describes Mityane
root integration; recipients use their own authoritative Linux build and the
declared C++20/LibTorch/CUDA/ABI from the
[dependency guide](../../../family/migration/DEPENDENCIES.md). Numerical source
uses the V19, family, V16 and generic observation include roots; comparison
sources additionally retain explicit historical roots. No Python or npm is
needed for source copying/verification.

## Evidence and source delivery

The [native fixture](../tests/engineering_test.cpp) covers auxiliary arithmetic
and gradient routing, shared base initialization/masks, teacher EMA and fixed
state, cutoff poisoning, alternative geometry, typed export/refusals and exact
full-state continuation. Actual execution receipts establish passed coverage.
Source presence alone does not establish numerical behavior or encoder quality.

The [protocol](../../../../../evaluation/protocols/cpb_v19_teacher_v1/PROTOCOL.md),
[contract](../../../../../evaluation/protocols/cpb_v19_teacher_v1/CONTRACT.md),
[results](../../../../../evaluation/protocols/cpb_v19_teacher_v1/RESULTS.md),
[record](../../../../../evaluation/records/CPB-v19.teacher-001.json) and
[independent review](../../../../../evaluation/protocols/cpb_v19_teacher_v1/REVIEW.md)
retain the bounded comparison. Reused controls/heads remain prior evidence;
new candidates and their teacher work are separately counted. This is related
to earlier native-view consistency research, not a novelty claim or automatic
replacement of the selected V16 reference.

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh', 'code/embeddings/scripts/export-cpb-v19.mjs',
  '--output', '.temp/embeddings/exports/NEW-V19-LEAF'
)
```

Retain the returned manifest SHA256 independently. Every selected source keeps
its repository-relative path under `sources/`. A recipient verifies the delivered
copy without the original checkout:

```bash
node BUNDLE/sources/code/embeddings/scripts/export-cpb-v19.mjs \
  --verify BUNDLE --expected-manifest-sha256 INDEPENDENT_RETAINED_PIN
```

The verifier checks member bytes, inventory, profile and quoted-include closure,
and refuses added/missing files. Keep the delivery receipt outside the bundle
to avoid recursive identity. Source integrity does not renew old numerical
evidence or establish application utility; consumers retain their own build,
numerical and integration acceptance.

# Configurable RPB family and portable training

This maintained C++ surface constructs and trains **fresh RPB-v22** over the
existing v20 backbone. The network, TRAIN fitter, original patch-mask/Huber
learning method, fixed AdamW update, checkpoint continuation and inference
export travel together. No trained parent, reference checkpoint, market reader
or sibling checkout is required. A configured model and its recipe are separate
from each resulting trained instance.

The [audit](audit/COMPONENT_TRAINING_AUDIT_2026-10-22.md) records the source basis,
later protected-parent designs and historical limits. The
[training method](migration/TRAINING_METHOD.md),
[runnable example](examples/TRAINING.md),
[compatibility notes](migration/COMPATIBILITY.md) and
[source package](migration/README.md) define the complete delivery. The
[verification protocol](evidence/PROTOCOL_2026-10-22.md) binds bounded artificial
engineering checks; completed evidence is recorded separately from that plan.

The [22 October producer receipt](evidence/portability-2026-10-22.json) records
actual CUDA verification at both configurations below. The API fixture passed
127 assertions and 59 refusal checks with six optimizer actions. The native
workflow passed 189 assertions and 26 refusal checks with 16 optimizer actions,
four skipped attempts, six successful fresh preprocessing fits and no refit on
resume. Exact continuation and inference/embedding reload passed in the named
runtime. All 26 source-package tests passed. These are bounded engineering
results; application training, admission and utility remain separate.

| Verified configuration | Parameters | Trainable / frozen |
| --- | --- | --- |
| C1/H65/F3/P8/D8/W16, one block | 7,170 | 7,170 / 0 |
| C3/H129/F9/P8/D32/W32, one block | 75,978 | 72,090 / 3,888 |

## Public C++ API

Use include roots for family/public RPB, settings, observation data and artifact
utilities as recorded in `family.mk` and the package profile. Numerical execution
requires the declared managed Linux LibTorch/CUDA runtime. Namespace:
`embedding::encoders::raw_patch_bottleneck_mae::family`.

| Surface | Contract |
| --- | --- |
| `family.h`, `family/family.h` | Aliases `Config`, `Input`, `FrozenScaler`, `Model`; `read_settings(path)`, `validate_config(c)`, `dimensions(c)` and `parameter_inventory(model)` |
| `fresh(c, scaler, initialization_seed)` | Validate config/scaler, seed LibTorch and construct fresh v22. Literal fixed priors and zero-start relation projection retain their existing arithmetic. Caller retains the complete initialization/recipe identity. |
| `PreprocessingFitter(c).add(raw_train)`, `.freeze()` | Bounded shared streaming TRAIN statistics, one-way freeze to existing `FrozenScaler`. Positive support required for each semantic channel/feature. Caller owns actual TRAIN membership and duplicate/window weighting. |
| `encode(model, scaler, raw)` | No-grad inference; standardize admitted logical endpoint window in CPU float64, execute configured CUDA float32, return existing native/support fields, restore caller's train/eval mode. |
| `make_optimizer(model)` | Complete registered-order single-group AdamW with existing fixed options. |
| `train_step(model, scaler, optimizer, progress, raw)` | Counter-derived original patch masks, no extra context deletion, eligible masked waveform Huber/backward, finite-gradient checks, clip1 and actual AdamW step. Ineligible batches consume attempts without weight decay or updates. Return loss/support/hidden-mask/pre-clip-gradient witness and counters. |
| `save_training_state(...)`, `load_training_state(path,c,identity,expected_scaler,expected_seeds)` | Existing distinct typed v22 training codec with explicit source/split/recipe/scaler/config bindings; wrapper independently compares all restored stream seeds. Save between completed attempts, with no pending accumulation. |
| `save_inference(...)`, `load_inference(...)` | Existing immutable typed v22 model/config/scaler export, separate from resumable optimizer state. Wrong kinds/configuration and malformed state are refused. |
| `config_identity`, `model_identity`, `tensor_identity`, `gradient_identity`, `optimizer_identity` | SHA256 numerical witnesses over names/dtypes/shapes/bytes, independent of archive pointer IDs. Hash utilities reuse the existing artifact header; this does not create application authentication or admission. |

`StepResult` records whether an update occurred, completed/attempted counters,
eligible examples/target cells, loss, pre-clip gradient norm, hidden mask and
pre-clip gradient hash. Failure after numerical mutation does not promise a
transactional rollback; preserve state and diagnose it. Supported counter-limit
refusals occur before action. Partial gradients, arbitrary ambient RNG, loader
iterators, schedulers and distributed state are outside the continuation format.

Raw `Input` is `[B,C,H,F]` values plus matching boolean observation mask, semantic
IDs `[C]` or `[B,C]`, interval and optional endpoints. Values may be CPU float32
or float64; preprocessing preserves observed raw float64 precision before final
CUDA float32 conversion. Naturally missing storage is ignored. Observed NaN/Inf
or overflow is refused. Prepared model input for direct `Model::forward` is
normalized float32; use the existing configurable transform and mask helpers.

The library accepts an already admitted retained window. The native example
additionally crops a typed observation-window archive at each declared endpoint
using existing `prepare_endpoint_history` with logical-H validation. Timestamps
must be ordered, uniformly sampled and on-grid; retained H ticks are validated
before fitting/model use. No source availability or real split membership can
be inferred from numerical tensors.

## Supported configuration and constraints

`Config` and strict v22 settings files configure C/H/F/P/D, semantic IDs, backbone
width/layers/heads, feedforward/decoder widths, mask ratio, Huber delta, scale
floor, layer-norm epsilon, interval and indexed CUDA device. Batch size can vary.
Configuration remains fixed per artifact; old weights cannot follow geometry
changes merely because D is unchanged.

Let Q=C(C-1)/2. The fixed spectral tail has 4Q coordinates, learned width is
D-4Q>0, and the v22 moment projection has `(D-4Q)*8*Q*F*F` parameters. C1 has no
cross-channel relation route. Require H>=17, positive dimensions, W divisible by
heads, unique IDs, finite positive numeric floors/interval/delta/epsilon and
0<mask_ratio<1, with the implementation's integer-overflow checks. Logical H
can be nondivisible by P; private backbone padding is absent/untargeted and
harmonics retain original H. P/available observations must still permit at
least three observed patches per training-eligible channel.

V20/v22 require actual CUDA float32 model execution, one early channel mixer,
global mode2/source0, raw patches and dropout0. Relations use CUDA float64
internally. CPU execution, arbitrary optimizer options/groups, mixed precision,
accumulation, multi-device/distributed training and general runtime migration
are not advertised. No maximum VRAM capacity or universal quality is claimed.
The older v20 model remains a separate explicitly tagged API/reference.
V23–v30 trained-parent designs are outside this initial fresh training surface.

`single.conf` and `multi.conf` are small artificial engineering configurations,
not application recipes or selected checkpoints. The evidence receipt records
which configuration/runtime checks actually ran; constructibility alone does
not establish training quality across the supported configuration space.

## Training, temporal meaning and application ownership

The portable reference binds the existing streaming raw fitter, original whole
patch masker, hierarchical waveform Huber through the served global vector,
detached targets and fixed AdamW/clip1. Row selection is counter-seeded
MT19937_64/modulo sampling with replacement. Context deletion is explicitly none;
the context seed is retained and checked but unused. Absolute attempts advance
on skips. The CLI binds exact executable/settings/dataset bytes, source/split,
initializer/stream seeds and batch/algorithm choices. Resume never refits; only
total stopping budgets may increase. See the method/example for every rule.

This differs from historical research recipes and from CPB. RPB's bidirectional
attention and moment banks can use the entire retained endpoint window;
reconstruction may use observations later than a hidden target within that
window. No prefix-causal or clean causal-training ancestry claim follows.
Native36 remains the historical balanced leader and V30 retains its market
success, pending independent transfer and later eligibility limits. No new
universal checkpoint or scientific quality result is produced here.

Applications own real observation/feature semantics, admitted TRAIN/calibration
members, substantial training, budgets/scaling/recovery and final docking.
Raw model/training archives remain separate from authenticated publication,
ModelRef/selection/capture and exact fitted downstream parents. Changed learned
coordinates require regenerated representations and compatible descendants.

## Managed build and checks

From the repository root, use a new named output leaf for every run:

```powershell
.\container.ps1 -Action exec -Command @('bash','code/scripts/task.sh','hx16-rpb-portable-example','-j1','build-rpb-family','build-rpb-family-trainer','build-rpb-family-fixtures')
.\container.ps1 -Action exec -Command @('bash','code/scripts/task.sh','hx16-rpb-portable-example','test-rpb-family','test-rpb-family-training-example','test-rpb-family-exporter')
```

The family Make fragment declares opt-in targets only; root environment/build
configuration remains authoritative. Builds/tmp stay inside
`/opt/mityane-geometrics`; generated states/runs stay under ignored `.temp/`.
`RPB_FAMILY_OUTPUT` and `RPB_FAMILY_EXAMPLE_OUTPUT` select new test leaves.
`RPB_FAMILY_SOURCE_ROOT` supports a verified imported source tree under the
recipient's managed build. The source bundle excludes SDKs, weights, datasets,
images and generated output, and its receipts preserve original source identities.

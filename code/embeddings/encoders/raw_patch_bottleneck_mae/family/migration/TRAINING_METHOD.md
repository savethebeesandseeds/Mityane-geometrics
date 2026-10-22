# RPB network and reusable learning method

This delivery exposes fresh configurable **RPB-v22 over the v20 backbone**.
The maintained `family` wrappers compose the existing streaming fitter,
whole-patch masking, waveform objective, AdamW and typed archive formats.
The numerical implementations under `shared/` and `versions/` retain their
source and format identities. The portable example is a separately identified
recipe, not a replay of a historical research instance.

## Four objects and two artifact layers

| Object | Required identity |
| --- | --- |
| Family | Architecture and mechanism/API revision, source inventory, genuine constraints and known failures |
| Configuration | Exact C/H/F/P/D, semantic channel IDs, interval, capacity, masks/loss numeric settings, dtype and device |
| Recipe | Fitter arithmetic/order, actual TRAIN/split/source bindings, initialization, row/mask/context algorithms and seeds, optimizer/clipping, attempts/updates and stopping rules |
| Trained instance | Weights and fitted state, configuration/recipe/source identities, completed and attempted counts, archive SHA256 and exposure/limitations |

The raw model archive is numerical inference state. A separate typed archive
adds optimizer state and continuation identities. Neither authenticates the
application's data admission, ModelRef, selected checkpoint, capture, publication
or fitted descendants. Consumers own those envelopes and independent checks.
New weights or fitted preprocessing define a new learned coordinate system;
equal dimensions do not make old indexes, readouts or dynamics compatible.

## Geometry, native output and temporal meaning

Input values and observation masks are `[B,C,H,F]`, semantic IDs `[C]` or
`[B,C]`, plus the configured uniform sampling interval and optional endpoints.
Raw reusable input may be CPU float32/64; the example archive uses CPU float64.
Normalized model execution is CUDA float32. Native global output is `[B,D]`.
Support masks and visible observation/patch counts accompany output. Missing
channels or examples produce finite zero vectors with false support; peers do
not establish observation support for a missing channel.

Let `Q=C(C-1)/2`, fixed spectral width `S=4Q` and learned width `L=D-S`.
Require `H>=17`, `D>S`, positive geometry/capacity, unique configured semantic
IDs, head-divisible W, positive interval/floors/Huber delta/layer-norm epsilon
and `0<mask_ratio<1`, plus the implementation's overflow guards. C1 is valid:
its relation bank/fixed tail are empty. B can vary; saved C/H/F/P/D/IDs/capacity
cannot silently vary. Resource availability further limits constructibility.

The v20/v22 configuration fixes raw patch inputs, one early channel mixer,
global bottleneck mode2/source0 and dropout0. Logical histories need not divide
P. The backbone privately pads to `ceil(H/P)*P`; padding is absent, the spectral
basis uses original H, and decoding is cropped. Preprocessing uses the logical
input configuration, not the padded backbone configuration.

This is an **end-of-window representation**. Early channel mixing,
bidirectional attention, pooling, spectral relations and content moments may
use the entire retained window. A timestamp cutoff/crop can enforce supplied
input availability but does not make an internal earlier prefix output causal.
Hidden-target exclusion is a masking guarantee, not a prefix-causality proof.
Fitted-state ancestry and historical information eligibility require their own
admission. No CPB architecture or causal guarantee is imported.

## TRAIN-only preprocessing

`family::PreprocessingFitter` wraps `StreamingScaler` and freezes one
`FrozenScaler`. Add only caller-admitted TRAIN batches. Numerical code cannot
independently discover the real split membership or deduplicate overlapping
windows. Repeated/overlapping windows affect fit weighting and must be declared.

Statistics are canonical per semantic-channel/feature CPU float64 counts,
means and centered second moments. Supported-value anchors and batch-centered
moments are combined with Chan's rule; only `[C,F]` state is retained. Masked
storage is removed before arithmetic. Every channel/feature must have positive
TRAIN support. Population SD is floored and the applied-floor mask is retained.
Rejected add/merge operations do not partially update the accumulator. Counts,
nonfinite moments and overflow are validated. Accumulator persistence and
distributed fitting orchestration are not provided.

The example fits source rows in declared source order/chunks and freezes that
result. Different batch/merge orders may round differently; fitter equivalence
uses explicit float64 tolerances, not universal byte equality. Record algorithm,
partition/order, source/split membership and exact resulting scaler identity.
The old whole-input fitter is a distinct supported arithmetic path. Its result
must not be substituted merely because it is numerically close.

Transformation centers/scales raw supported values in CPU float64 before
CUDA-float32 conversion. Float32 source values have already lost their original
precision. Observed NaN/Inf or scaled overflow is refused; unobserved storage
is sanitized. Freeze preprocessing for validation, test and serving. The built-in
`rpb-scaler-v1-fnv1a64-*` identity is a content fingerprint, not cryptographic
archive authentication. Retain separate SHA256 pins and source/split records.

## Initialization, architecture and gradient routes

Fresh construction records and sets the LibTorch initialization seed before
allocating the model. It needs no trained parent, teacher or reference
checkpoint. Allocation/registration order remains part of deterministic seed
parity. Value/visibility patch projections retain original positions; early
channel attention, bidirectional temporal blocks and learned global pooling
produce the learned representation used by the waveform decoder.

V20's fixed spectral tail uses low q1–3 and high q4–8 complex relations across
all semantic channel pairs and F² feature pairs. Its literal mean/prior weights
are validated and frozen; learned shape projection begins as a truncated
identity. Prior support/energy floors and per-pair normalization are part of
the mechanism. A frozen prior has no gradient or AdamW moments.

V22 adds normalized content odd/even moments at original tick lags 1/2/4/8 to
the learned coordinates. For `N=Q*F²*4`, its bank has `2N` columns; the bias-free
`L*2N` projection starts at literal zero without consuming initialization RNG.
Four-endpoint comparisons require common observed support >=2 and positive
energies above 1e-12. Bank arithmetic is CUDA float64, exported float32. Counts
and eligibility are metadata, not additive learned features. C1 adds zeros.
Changed masks can alter moments/support; arbitrary deletion invariance is not
promised. The default C3/F9/D32 projection has 38,880 parameters, a configured
example rather than a universal family size.

There is no EMA teacher, contrastive loss, task readout, forecast objective,
CPB variance rule or application decision policy in this training surface.

## Masks, targets, loss and actual update

The shared original-patch masker uses CPU MT19937_64 and `std::shuffle` over
supported original patch positions. For M observed patches in an eligible
channel it hides `clamp(floor(mask_ratio*M+0.5),1,M-2)`. Training requires at
least three observed patches: two retained and one target. Inference may have
less support and reports it. The partial final patch keeps its logical choice;
private padding never becomes a target.

Visible input is `observed & !hidden`; waveform target support is
`observed & hidden`. The model encodes visible support and reconstructs through
the exact served native global vector. Standardized targets are detached.
Hierarchical Huber averages supported target cells within each eligible
channel, channels within each eligible example, then examples. This is not a
global cell-weighted loss. Nonfinite supported targets/predictions are refused.

`family::train_step` normalizes the caller's batch, derives its patch-mask seed
from the absolute attempted-batch counter, clears gradients and sets training
mode. An eligible attempt performs forward/backward, verifies finite CUDA-F32
gradients, clips the global L2 norm to 1, then performs AdamW. The model's
learned backbone/content projection/decoder receive the objective; literal
spectral weights remain protected. An ineligible batch performs no optimizer
action or weight decay and advances attempts only. The pre-clip gradient hash
is an engineering witness, not a training objective.

The existing continuation codec admits one complete registered parameter group
in exact order and **AdamW lr0.001, weight decay0.0001, betas0.9/0.999,
epsilon1e-8, AMSGrad off**. Inactive diagnostic parameters may lack optimizer
slots; frozen weights must lack them. Per-parameter steps/mapping, moment
shape/device/dtype, finite moments/nonnegative second moments and progress
consistency are validated. There is no scheduler, accumulation, AMP,
distributed optimizer or historical step clamp. A numerical failure after
backward does not promise transactional model/optimizer rollback.

## Portable recipe and continuation

The portable native example names its full recipe separately from the model
configuration. Rows use **MT19937_64 with modulo selection with replacement**,
seeded per absolute attempt by `splitmix64-counter-v1`. Modulo selection has
its own exact meaning, including modulo bias; it is not uniform-integer or
rejection sampling. The original whole-patch mask is the only masking policy;
**additional context deletion is none**. The context base seed remains bound
for typed-state compatibility even when no context draws are consumed.

Record initialization seed, row/mask/context base seeds, batch size, fitter
chunk/order, clip/optimizer, source/split/recipe identities and attempt limits.
The native example binds exact dataset-archive, settings-file and executable
SHA256 in its recipe, in addition to resolved configuration and semantic source/
split identities. A rebuilt binary must therefore retain exactly matching bytes
to resume that example's state; source/API compatibility alone is insufficient.
`TrainingProgress` separates physical attempted batches from completed optimizer
updates. Each completed ineligible attempt consumes its counter and therefore
the next counter streams. The native example may resume toward an increased
total stopping budget; total budget is an orchestration choice, while the
mechanism/recipe identities and previous progress remain fixed.

The raw training kind is
`rpb_content_ordered_relation_training_state_v1`, distinct from inference.
Save only between completed attempts with no partial accumulation. Weights,
frozen scaler, AdamW and counters are restored; pending gradients, arbitrary
process CPU/CUDA RNG, loader iterators and partially consumed attempts are not.
The maintained loader independently checks restored seeds against the expected
recipe. Typed identity strings must match, but they do not authenticate external
files themselves. The example's content hashes and a consumer's trusted records
supply independent bindings.

Exact continuation requires identical fitted bytes, full recipe/algorithms,
draw order, configuration and the same supported runtime/device. Dropout0 and
counter-driven draws make this bounded claim possible. Library shuffle/MT
behavior, compiler/LibTorch/CUDA/backend settings and device ordinal matter.
No cross-device, cross-compiler, cross-SDK, hardware-independent or distributed
exactness is promised. An inference loader's device migration does not imply
optimizer-resume device migration.

## Historical recipe differences and research limits

| Historical scope | Difference and retained interpretation |
| --- | --- |
| v20 capacity and v22 content studies | C3/H128/F9/P8/W64/D32 reference, 512 completed updates/B8, 25% patch masking and 15% extra context erasure under frozen study RNG/data/head rules. The portable no-extra-context recipe does not reproduce these trajectories. Wider/larger capacity showed no consistent final classification benefit at that budget. |
| Typed v22 continuation fixture | MT19937_64/uniform-integer rows and independent 15% uniform-real cell deletion. Its exact-next-update witness is for that fixture, not the portable modulo/no-extra-context recipe. |
| v24 native36 balanced reference | Protected trained v22 parent32 plus learned4; retained historical 8/9 market conditions and all four primary TEMPO-5 transfer comparisons. Its linear-volatility failure remains. Fresh v22 is a different instance and recipe. |
| v30 location-scale market result | Protected v22 parent through saved v24 initialization, frozen V28 relative-bank normalizer, separate zero-start variance branch and hierarchical Gaussian NLL. It passed 9/9 market and 8/8 strict native36 floors; independent transfer remained pending. This package's v22 waveform objective does not reproduce v30. |

The audit records v23–v30's exact protected-parent/frozen-fit distinctions.
In particular, constructor availability does not make warm-started historical
recipes fresh-from-scratch. Later noncausal-training ancestry diagnosis limits
clean-causal eligibility without erasing those measured historical results.
Raw endpoint-window use and fitted-state eligibility remain separate questions.

Frozen records ship as rationale and known-failure context. Their original
paths, hashes, protocols, budgets and receipts retain their meaning. They do
not prove a relocated import, a changed recipe, an arbitrary configuration or
downstream application utility. The [bounded protocol](../evidence/PROTOCOL_2026-10-22.md)
and final portability receipt record the measured engineering scope; delivery
verification additionally checks exported bytes and isolated acceptance.

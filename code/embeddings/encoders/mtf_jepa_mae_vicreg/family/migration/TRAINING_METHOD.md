# MTF network and complete portable learning method

MTF-v1 combines multiscale time/frequency descriptors, JEPA prediction, masked
reconstruction, alignment and VICReg. The reusable component includes these
learning rules with the network. It does not select a trained market model.
The preserved [source audit](../../delivery/COMPONENT_TRAINING_AUDIT_2026-10-22.md)
and its SHA256 inventory supply detailed source locations and historical gaps.
The new portable implementation and reference recipe have separate identities.

## Construction and numerical input

Geometry is B/C/H/F (batch, channel, history, feature), with rank-three B/H/F
available in the retained core only for one channel. Positive dimensions and
scales/strides are required; encoder depth may be zero, predictor and decoder
depths are positive, and latent width must divide evenly by attention heads.
Empty stride configuration resolves to `max(1, scale/2)`. A scale clips to H and
the window enumerator includes the endpoint. Shapes and parameter counts follow
configuration, not a historical benchmark.

`Config{}`, historical `default_settings()` and explicit example files have
different defaults. A recipe stores its actual complete settings. Fresh
construction uses LibTorch module initialization; target tokenizer and encoder
constructors consume RNG before being overwritten with online state. The fresh
copy uses EMA tau=0 and freezes the teacher. No frozen-parent checkpoint is
required. Apparently inactive frequency/decoder modules still construct and
may affect initialization draw order. Do not remove them under an equivalence
claim without new source/recipe identity.

The portable policy `identity-cpu-f32-observed-v1` takes floating CPU BCHF
observations with shape-matched CPU bool support. It refuses observed nonfinite
values before conversion and F32 overflow after conversion, zeros missing
values and clones/detaches canonical F32 storage. Zero is a valid observed
measurement when support is true. There are no fitted statistics. The dataset
SHA256 binds prepared tensor values, shape/dtype and explicit support in row
order, rather than incidental archive-container bytes. Declared source/split
identities travel with that binding; real population/TRAIN membership remains
application-owned. An application-specific fitter belongs to a separately
identified preprocessing recipe, not an invented scaler in this component.

This portable boundary is stricter than the retained core's permissive casting
and inferred nonfinite masking. The shared evaluation adapter accepts floating
CPU inputs with explicit F64-to-F32 identity preparation and overflow refusal.
The historical evaluator privately fits observed TRAIN per-feature mean and
population standard deviation across B/C/H (floor 1e-6) and saves a sidecar.
These are three distinct runner/preparation policies around shared primitives.
Unknown historical checkpoint/preprocessing associations remain unknown.

## Descriptors and temporal/channel meaning

Time descriptors are observed mean and population standard deviation per
configured window/feature, accumulated in F64; variance receives 1e-6 before
square root and descriptors return in model dtype. Frequency descriptors use
the complete configured window: observed centering, Hann weights, explicit
DFT bases, magnitude normalized by observed Hann support, DC replaced by the
absolute observed mean, optional `log1p`, and padding to configured bins.
Padded frequency bins may be supported zero-valued reconstruction targets.

The token carries scale/channel/domain/position information. Positive-depth
encoder layers add a transformed masked global mean and residual token MLP
with dropout. Every eligible token can therefore contribute to global context;
channel pools remain contextual across channels/windows/scales/domains.
Predictor/decoder metadata queries attend over supplied context without a
temporal triangular mask. Full-history windows are permitted. These are
observation-window representations available after their required support;
token timestamps, strict support masks or per-channel pooling do not establish
prefix causality or independent channels. A zero-depth choice changes mixing
but does not confer a family-wide causal guarantee.

## Masks, targets and gradients

The legacy default proposes a contiguous time-token block in flattened order,
a random frequency subset and Bernoulli whole-channel targets. Time ratio zero
can still propose at least one token. Same-window cross-domain coupling and
optional whole-channel blocking are hard exclusions. The temporal-overlap
threshold is a soft exclusion across channels and may relax to preserve context.

Strict support is an opt-in method choice. It first consumes legacy proposal
RNG, then excludes same-channel temporal overlap across scales/domains without
relaxation, reduces targets to preserve context and can draw an additional CPU
permutation. Cross-channel context is retained. No-target outcomes are valid.
The conservative support diagnostic uses metadata windows, not exact sparse
feature support or causal ordering. Keeping its name does not establish absence
of leakage in an application.

The model computes full online latents, masked context, teacher/target,
predictor and decoder paths. Flags gate loss contributions, not every draw or
computation. The weighted objective is:

```
lambda_jepa * JEPA + lambda_mae * MAE
  + lambda_tf_align * time_frequency_alignment + lambda_vicreg * VICReg
```

Use the actual serialized configuration names/values; this expression states
composition rather than introducing additional controls.

| Term | Rule |
| --- | --- |
| JEPA | Selected-target latent MSE. EMA tokenizer and encoder run eval/no-grad and remain frozen even if stop-gradient is disabled. Without EMA, target latents use the full online path; stop-gradient determines whether that target branch receives gradients |
| MAE | Default targets are detached raw time mean/std and frequency descriptors. Loss normalizes separately over selected supported time/frequency descriptor elements, then sums. The alternate projected targets are detached current tokenizer bases |
| Alignment | Supported time/frequency pairs sharing channel/scale/start/width contribute weighted `1-cosine` from full online latents. It uses full observations, not just JEPA context |
| VICReg | Two weakly augmented online views and projector; valid global rows and optional flattened valid sample-channel rows. Invariance MSE, biased variance-floor penalty and off-diagonal covariance squared sum divided by feature dimension; covariance uses rows-1 |

VICReg zero-row inputs return zero; one row has variance penalty and no
covariance. The model does not surface the helper's `statistics_supported` flag.
All-missing, no-target or zero-weight configurations do not prove useful
learning. Strict finite mode throws on nonfinite loss; permissive mode sanitizes
loss values, including diagnostics. The portable trainer requires strict finite
loss, at least one effective objective and some support in every TRAIN row.
It refuses updates without selected JEPA/MAE target support when those terms
are active, a differentiable finite loss or a nonzero finite online gradient.
These refusals are not evidence of encoder quality.

Weak views use feature deletion capped at 0.20, shared-over-feature time deletion
capped at 0.10 and observed-position Gaussian jitter. Draws can still occur when
jitter or effective deletion is zero. Online dropout also consumes RNG. CPU
mask/augmentation/sample draw order and module initialization are part of the
recipe. Seed equality alone is insufficient if algorithms/draw order differ.

## Portable optimization, stochastic state and progress

Both portable sampling policies have their own explicit recipe names:

- `replacement-step-seeded-v1`: row sampling with replacement from the supplied
  archive, then forward/mask/augmentation/dropout for that update.
- `full-partition-step-seeded-v1`: the entire supplied archive per update,
  with its separately defined update seed and unchanged row order.

The complete `Recipe` binds Settings, source and dataset SHA256, input-source,
split and recipe identities, sampling policy and explicit AdamW beta1/beta2/
epsilon/AMSGrad choices. Learning rate, weight decay, clipping, configuration,
initialization seed and total update budget are bound too. Fresh construction
uses `manual_seed(seed)`. An update at zero-based `completed_updates` uses
`manual_seed(seed + completed_updates + 1)` before any row sampling or forward
draws, named `portable-absolute-step-seed-plus-one-v1`. This is
step-seeded continuation: raw generator snapshots are not substituted for its
algorithm. CPU/F32/runtime, one intra-op thread and the recorded inter-op
setting, and source draw order must remain fixed
for exact update-boundary comparisons.

Each successful update follows: validate bound state/data; set train mode and
its absolute-step seed; choose rows under the selected sampler; zero gradients;
model forward; target-support/differentiable-finite-loss check; backward;
finite/nonzero online-gradient check and L2 norm clipping; AdamW step;
then update **both** EMA tokenizer and EMA encoder under no-grad; increment
completed progress. Clipping zero disables clipping via an infinite threshold;
nonfinite gradient norm still fails. The EMA rule is
`teacher = tau*teacher + (1-tau)*online.detach()` with configured fixed tau.
There is no implicit teacher schedule. Omitting/reordering teacher update is a
different learning method. The implementation defines the raw serialization
fields as well as these named mechanisms.

`settings.steps` binds the original total budget. Save at a completed-update
boundary, retain all bindings and resume only remaining updates. A changed
budget, dataset/order, optimizer, preparation, sampling algorithm, geometry,
source/runtime or recipe is refused for exact continuation. Runtime binds the
same executable SHA256, Torch/compiler/C++/ABI/endian, intra-op/inter-op and
deterministic/MKLDNN flags. Same CPU/backend/hardware and pinned SDK remain
required assumptions. Global Torch RNG/thread settings are touched; callers
must serialize concurrent session updates. An error after the optimizer step
can leave mutated state: there is no transactional rollback or durable journal.
Stop the failed session and recover from independently retained valid saved
state under application orchestration. Supported learning
parameter choices create new recipes/instances; substantial data/training and
recovery orchestration remain application responsibilities.

The historical CLI instead reseeds `seed+absolute_step` before CPU randint or
synthetic generation and uses its v1 settings/AdamW checkpoint. The shared
evaluation adapter forces CPU/F32/dropout=0, initializes with
`stream_seed(run_seed,500000)` and trains the whole permitted partition using
`stream_seed(run_seed,600000+update_index)`; its saved control has no adapter
resume API. The historical evaluator has its own synthetic generation/fitted
normalization/training loop. None is relabelled as portable exact continuation.

## Raw state, inference and trust

`mtf-family-training-v1` binds configuration, complete portable recipe,
identity preparation, model/teacher state, AdamW state/options and completed
progress. The state/hash checks support the declared raw contract, not a
trusted signature or independent source/population admission. Preserve the
original legacy checkpoint under its original format and provenance.

`mtf-family-inference-v1` stores model-only registered state, including inactive
teacher/heads, plus its bound configuration/preparation. Serving uses the online
model in eval mode; no optimizer or training continuation is advertised by that
format. Keep inference and training states separate even when they share the
same weights. Refuse incompatible/malformed raw state rather than assuming
equal dimensions establish compatibility. Applications own authenticated
publication/selection/capture and exact downstream parent identities.

## Rationale, failures and evidence limits

JEPA encourages latent prediction from context; reconstruction retains raw
descriptor information; alignment couples paired time/frequency views; VICReg
penalizes collapse with explicit support limitations. Their combination and
weights are mechanism/recipe choices, not proof of operation suitability.
Teacher continuation, support masks and stochastic ordering travel with the
network because omitting them changes the learned trajectory.

Known constraints include full-window frequency support; contextual channel
outputs; permissive original core input conversion; strict masks legitimately
yielding no targets; no guarantee of useful gradients at zero weights or weak
support; and unknown original checkpoint/preprocessing associations. No
historical score, prefix-causality, independent channel, forecasting, identified
intervention or downstream utility claim is created by this delivery.

The [prospective bounded protocol](../verification/PROTOCOL_2026-10-22.md)
selects two artificial configurations, both sampling recipes, real updates,
teacher/optimizer continuation, inference reload and refusal cases. The
[actual results](../verification/RESULTS_2026-10-22.md) and
[engineering receipt](../verification/engineering-proof-2026-10-22.json)
record both passing primary and separately compiled staged-source proofs.
Each used 24 native and six CLI physical updates, for 60 combined; both native
runs passed 3,627 assertions and both exporter suites passed 21 tests.
`Settings.threads=1` enforces one intra-op Torch thread; the
inherited inter-op setting (12 in the primary proof) is recorded/bound, not
forced to one. Existing audit and
historical engineering receipts are retained as dated evidence; copying them
does not certify relocated source. No market experiment, historical score
replay, substantial GPU/application training or checkpoint promotion follows
from this method document.

Staged-source isolation ran from the preserved `mtf-family-2026-10-22-001`
bundle (manifest SHA256
`a3cc64dc41b944db2e495a7bd27eb8540f37b4359cba575dd9cb12e410a64975`),
with distinct executable/runtime identities and 25 project dependencies in seven
compiler dependency files all within that staged root. The final
`mtf-family-2026-10-22-002` changes documentation/receipts while preserving all
33 admitted source/configuration/fixture/tool pins. Its
[delivery receipt](DELIVERY_2026-10-22.json) retains the actual tested paths and
both manifest identities. No historical score replay or execution at the final
successor path is claimed.

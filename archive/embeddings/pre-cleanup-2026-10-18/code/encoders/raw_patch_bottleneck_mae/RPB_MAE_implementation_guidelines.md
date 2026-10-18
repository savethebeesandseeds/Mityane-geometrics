# Raw Patch Bottleneck MAE (RPB-MAE) implementation guidelines

Date: 2026-10-05

Scope: preserved initial engineering guidance. Later RPB designs have separate
contracts and frozen evidence. Start with [current status](../../../doc/CURRENT_STATUS.md)
and [RPB-v20 API](CONFIGURABLE_SPECTRAL.md); an extension called planned below
does not override a later implemented, versioned module.

Status: Engineering plan with an implemented first independent-channel core and
encoder adapters for shared feature evaluation and held-out reconstruction. See
[current implementation](README.md) for commands and
[validation evidence](IMPLEMENTATION_STATUS.md) for original protocol-v1 results
and completed feature-evaluator refactor validation. The separate
[decoder reliance record](DECODER_RELIANCE.md) reports validated held-out
reconstruction measurements, and the [frozen-probe record](FROZEN_PROBE_ADVANCE.md)
reports the stronger feature comparison. Extensions below remain plans
unless listed there.

The [embedding evaluation policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md)
governs shared harness ownership, experimental validity, reporting, and
acceptance. This guide defines RPB-MAE engineering choices and applicable model
contracts; its evaluation sections are interpreted under the shared policy.
The [shared integration guide](../../evaluation/README.md) and
[implementation review](../../../doc/EVALUATION_IMPLEMENTATION_REVIEW.md) define
current evaluator ownership and limits.

Read this alongside the
[architecture and evaluation specification](RPB_MAE_architecture_and_evaluation_spec.md).
That specification states the research direction. This guide resolves initial
engineering choices, identifies reusable code and defines the evidence required
at each implementation stage. Recommendations below are project choices, not
claims that the cited papers guarantee a better representation.

## 1. Architecture assessment

The proposed bottleneck masked autoencoder is a coherent first experiment:
ordered raw patches preserve information that summary descriptors can discard,
temporal attention exchanges information across patches, and reconstruction
trains through the exact compact vector that consumers receive. PatchTST provides
precedent for patching and shared channel-independent Transformer weights;
it does not establish the utility of this project's compact readout
([paper](https://arxiv.org/abs/2211.14730)).

The main risk is capacity: one small vector must retain enough information to
reconstruct several patches and support different downstream tasks. A small
decoder can also predict average patterns from position and channel identity
without using the observations. Reconstruction loss alone cannot select the
encoder. Held-out probes, metadata-only controls and embedding-ablation tests
must establish whether the exported vector carries useful information.

The first independent-channel candidate cannot condition a channel vector on
another channel. It can nevertheless retain phase/event timing that a joint
downstream function uses to recover lag. For example, local coordinates
`(cos(phi_c), sin(phi_c))` preserve phase differences modulo a period; this is a
mathematical possibility, not evidence that our learned bottleneck achieves it.
Post-pooling interaction can work if the readout retained the needed information.
Prefer an aligned mixer before pooling as an experiment, not as an impossibility
claim about other layouts. Evaluate local, concatenated and global surfaces
separately, and retain required cross-channel tasks even when a candidate fails.

| Issue found in the checkout | Implementation decision |
|---|---|
| Shared inputs are `[B,C,H,F]`; the proposal's shorthand is `[B,C,T]`. | Preserve C as semantic channels and F as features within each channel. Supply explicitly named adapters. |
| H=32 and patch length 16 produce only two patches. Hiding one leaves one visible patch and realizes 50%, not 25%, masking. | Start with H=32, P=8: four patches, normally one target and three visible patches. P=16/H=64 is a separate history-length experiment. |
| Existing data archives load float32 values and carry no semantic IDs or timestamps. | Add an RPB-MAE preparation/metadata format; preserve the baseline archive reader. |
| Historical scaler/probe implementations are private to the baseline evaluator. | Use the shared frozen-feature engine and encoder adapters; preserve the historical protocol and its results. |
| Baseline per-channel outputs already depend on other channels. | Name and version local, contextual and global surfaces separately. |
| The baseline default export width is 12, not the proposal's discussed 32. | Report native widths and separate dimension-matched comparisons. |
| Build sessions isolate artifacts but do not select an encoder. | Add an explicit RPB-MAE target, binary and checkpoint loader. |

## 2. Initial candidate and configuration

Use this configuration for the first correctness and optimization smoke tests.
It is deliberately small; it is neither a tuned configuration nor an acceptance
threshold for representation quality.

| Setting | Initial choice |
|---|---|
| Input | Smoke: `[B,3,32,3]`; general C and F come from an explicit schema |
| Patching | P=stride=8; non-overlapping, time-major values within each patch |
| Patch projection | Linear projection of values plus visibility indicators to width 64 |
| Temporal encoder | Three shared pre-normalized blocks, four attention heads |
| Feedforward block | Width 256, GELU; dropout 0 for initial reproducibility |
| Layer normalization | Epsilon 1e-5 |
| Readout | Learned position-aware pooling, then width-32 export projection |
| Global output | Deterministic mean over valid local channel vectors |
| Decoder | Two-layer MLP with one 128-wide GELU hidden layer; one P×F prediction per query |
| Objective | Unreduced Huber, delta=1 in the frozen scaled coordinates |
| Training mask | Requested patch rate 0.25; at least two visible patches and one target patch |
| Deferred components | Channel mixer, multiscale branch, forecasting head, JEPA/EMA, VICReg, codebooks and specialist banks |

Reject unsupported H/P remainders initially rather than dropping the tail.
Require positive dimensions, width divisible by head count, valid ID schemas,
and finite numerical configuration values through the public C++ API.
Learning rate, weight decay, batch size, update budget and any gradient clipping
must be explicit in the run configuration and recorded in the report. Select
quality-experiment settings on validation data within a declared budget.

The controlled-pairs-v2 card defaults to C=3, H=32, F=3 and uniform synthetic
sampling at one declared time unit per step. Each instantiated card owns C/H/F,
semantic IDs, units, endpoint and sampling interval; explicit model configuration
must agree with it. At H=32, P=8 produces K=4 before masking. Longer histories,
including the
feedback's illustrative H=512/P=16, are separately named experiments with
history-matched controls. Record H×F raw coordinates, K×W token coordinates and
D served coordinates; these counts are capacity descriptors, not a lossless
compression guarantee.

## 3. Input, precision and support contracts

### Shapes and metadata

Define B as examples, C as channels/entities, H as history steps, F as features
inside a channel, K=H/P as patches, W as encoder width and D as export width.
Do not reshape F into C implicitly.

| Object | Shape and meaning |
|---|---|
| X | `[B,C,H,F]` raw floating-point observations |
| O | Same shape, bool; originally observed |
| A | Same shape, bool; deliberately hidden by the training mask |
| V | `O & ~A`; encoder-visible support |
| Q | `O & A`; reconstruction-target support |
| Ordered patches | `[B,C,K,P,F]` with original patch positions |
| Padded visible tokens | `[B*C,K_visible_max,W]` plus token validity |
| z_local | `[B,C,D]`; information from that channel's visible observations |
| z_global | `[B,D]`; equal-valid-channel mean |

The scalar adapter maps `[B,C,T]` to `[B,C,T,1]`. The existing shared rank-three
path instead means `[B,H,F]` with C=1; never infer which convention a caller meant.
For the first schema, require a fixed feature order, unique known semantic
channel IDs, declared feature units and uniform sampling with a declared
interval. Channel storage order is not identity. Unknown/duplicate IDs fail
validation. Missing channels remain present with O=false.

Relative positions refer to the original history, including gaps left by removed
patches. An index-only model supports uniform sampling, not irregular elapsed
time. Add irregular-time inputs only with explicit elapsed-time validation and
tests; do not advertise that capability before then.

The preparation adapter accepts an explicit endpoint for each example and
supplies an already cropped history. Record its endpoint and raw support
interval; verify that every encoder observation and time-dependent metadata
item is available by that endpoint. Bidirectional attention inside this observed
history is allowed. The future-mutation test exercises preparation plus encoding,
not merely a model that never received the future tail. Per-timestep causal
outputs require a separate prefix/causal-attention contract.

### Frozen preprocessing

Fit centered float64 statistics `mu[channel_id,feature_id]` and
`scale[channel_id,feature_id]` from permitted pretraining training observations.
Store observation counts, units, the scale-floor policy and which coordinates
needed that floor. Reject unsupported coordinates with no fitted observations.
Freeze this asset and bind its identity to the checkpoint.

Preserve source precision until after centering/scaling; only then cast model
inputs to float32. An already quantized float32 archive cannot recover tiny
variations around a large offset. The RPB-MAE preparation format must support the
source precision needed by the application, rather than claiming that changing
the variance formula repairs lost values.

Observed NaN/Inf and post-scaling overflow are errors. Naturally absent cells
may contain arbitrary stored values; replace them with safe zeros using
`where` before arithmetic. Do the same for deliberately hidden encoder cells:
NaN multiplied by zero is not safe masking. Constant zero observations remain
valid and must differ from absent observations.

A frozen training-fit scaler is permitted to depend on the training corpus,
including observations later selected as reconstruction targets. Runtime
normalization must not depend on an example's hidden targets or future tail.
Leakage tests hold the fitted scaler and model parameters fixed while changing
forbidden values.

Frozen affine scaling without clipping does not inherently saturate: a nonzero
scale makes it invertible. Monitor scaled-input ranges/quantiles, scale floors,
overflow, activation/output norms and task performance under drift. Low exported
rank alone is not a reason to switch normalization. A visible-context window
normalizer is a later, declared experiment when task evidence justifies it.
If level/amplitude are required, preserve their context-derived statistics
through the served vector or separately declared served features; giving those
statistics only to a decoder does not establish that z retained them. Recheck
hidden-target and endpoint isolation after any preprocessing change.

### Discrete mask policy

For each channel, let M be the number of patches containing at least one
originally observed cell. If M<3, the channel is ineligible for the initial
training objective. Otherwise hide
`k = clamp(floor(0.25*M + 0.5), 1, M-2)` patches uniformly without replacement.
A=true for the chosen patch cells; Q still excludes naturally missing cells.
Remove hidden patches from the encoder, preserving the positions of the others.

This policy always keeps two visible observed patches and one target patch.
Log requested rate, realized hidden-patch rate, visible/target cell counts and
eligibility. Unequal natural missingness changes the realized cell rate.
Mask draws are independent of labels and signal family. Any later span sampler
with support constraints uses a recorded retry budget and then skips; it must
not silently relax those constraints.

Inference uses all observed patches, with no artificial mask. One observed patch
is sufficient for a supported inference output, although the output reports that
limited support. Training eligibility and inference validity are different flags.

## 4. Encoder, readout and decoder

The initial computation is:

~~~text
visible scaled [B,C,H,F] values + V + permitted metadata
  -> ordered [P,F] patches, zero at invisible cells
  -> shared linear projection + original-position/channel metadata
  -> shared temporal attention independently for each channel
  -> position-aware valid-token pooling -> z_local [B,C,32]
  -> valid-channel mean -> z_global [B,32]

training only:
z_local + target patch position/time/channel/feature metadata
  -> small decoder -> target patch values -> masked Huber
~~~

### Temporal attention

Flatten B and C only to apply the same temporal encoder; do not exchange values
across channels in the core candidate. Combine a linear projection of the
time-major P×F values and their P×F visibility indicators with position and
semantic metadata. Use original patch coordinates after packing visible tokens.

Retain an explicit `original_patch_index:[B*C,L_max]` beside packed states and
`packed_valid:[B*C,L_max]`, where L_max is the maximum visible length in that
batch. Valid indices are unique per row and in `[0,K)`; padding uses index -1,
false validity and zero state. Neither position embeddings nor decoder queries
use the packed slot as time.

The pinned LibTorch C++ 2.6 headers expose neither `norm_first` nor `batch_first`
in `TransformerEncoderLayerOptions`. Implement the intended pre-normalized block
explicitly from LayerNorm, MultiheadAttention, residual connections and a GELU
feedforward module. Set the feedforward width explicitly; its stock default is
2048. Keep a clear sequence-first adapter `[K,B*C,W]` for the chosen C++ attention
primitive. Verify against the
[bundled options header](../../../.external/libtorch/include/torch/csrc/api/include/torch/nn/options/transformerlayer.h)
instead of copying Python constructor arguments.

V=true means visible, whereas a boolean MultiheadAttention key-padding mask uses
true for a blocked key
([versioned API documentation](https://docs.pytorch.org/docs/2.6/generated/torch.nn.MultiheadAttention.html)).
Convert masks explicitly and test the selected primitive. Exclude entirely
invalid channel rows before attention/softmax and scatter zeros afterward.
Mask padded query states after residual blocks as well as masking keys.
Do not introduce a sentinel token whose learned bias manufactures observed support.

### Aligned mixer experiment

This extension is implemented as an optional mode and remains off by default.
The [mixer record](CHANNEL_MIXER_ADVANCE.md) contains validation and the bounded
development comparison. Its tensor contract is:

~~~text
visible temporal states                  [B*C,L_max,W]
  + original_patch_index, packed_valid
  -> scatter by original patch index
temporal_grid, grid_visible              [B,C,K,W], [B,C,K]
  -> reshape groups at original k
channel attention input                 [B*K,C,W]
  -> residual attention across C at each k
  -> reshape [B,C,K,W], valid temporal readout
z_contextual                            [B,C,D]
~~~

The grid must share physical start times and sampling intervals across channels,
not just equal integer indices. Reject misalignment until a tested alignment
adapter exists. Never rebuild this grid from unmasked raw observations.
Channels keeping original patches {0,2} and {1,2} join at original patch 2;
their first packed tokens are not simultaneous.

For this first mixer, both query and key eligibility equal grid_visible.
Skip groups with no valid keys, invert visibility for the attention API and zero
invalid queries after residual/feedforward biases. Pool only valid queries.
Introduce no inferred queries or observed validity for absent channels.
An inferred-channel mode requires a separate future contract.

Keep z_local computed before mixing, and declare whether it still has a direct
training objective or is only a diagnostic of weights trained for contextual
outputs. Reconstruction uses the declared active served vector; the decoder
does not concatenate local and contextual vectors as a hidden extra capacity.
Declare which surface the global pool consumes.

Channel-attention pair interactions cost O(B*K*C^2), excluding projections and
feature width; this is not the total encoder cost. Temporal states already see
the whole permitted history, so mixing at patch k does not create a causal
output at that patch. The endpoint contract still applies. A mean across C at
each k is a new aligned-mean control, not the baseline's existing global-token
mean mechanism.

### Export semantics

Use learned attention scores over valid temporal states and their positions,
then project the pooled W-wide state to D=32. Train through exactly this served
vector; do not apply an additional untrained projector or hidden inference-only
normalization. Keep exports without L2 normalization initially so amplitude
information is not deliberately discarded.

Export `z_local`, its validity, visible observation/patch counts, semantic IDs,
`z_global` and global validity. Set absent channels and entirely invalid samples
to deterministic finite zeros after projections, including their biases.
The global mean averages valid channels equally and becomes invalid when none
exist. It receives training signal indirectly through the trained local vectors.
Evaluate the global surface separately.

`z_contextual` is unavailable in the first candidate. A later mixer supplies a
new declared semantic mode, without silently renaming `z_local`. There is no
inferred embedding for an absent channel in the core. Version the encoder ID,
preprocessing ID, input schema and output semantics in every exported artifact.
The baseline's existing `pooled_by_channel` contract remains unchanged.

Complete channel-record permutations move values, masks, IDs and scaler lookups
together: local outputs permute accordingly and the global output stays the
same within numerical tolerance. With parameters/preprocessing fixed, changing
another channel must not change a local output.

### Decoder and loss

For a target patch, the decoder receives only z_local and declared target
metadata; it predicts P×F coordinates. No token-state skip, cross-attention to
encoder states, raw context bypass or additional unnamed embedding is allowed.
Position/channel information can predict averages, so include a metadata-only
decoder control trained with the same target protocol and held-out z interventions.
Freeze the stated decoder architecture for the initial comparison and report its
parameter count alongside the encoder. Small size alone prevents neither
memorization nor metadata shortcuts; do not tune it against the confirmation
probe set.

Shuffle z across independent trajectories within channel-ID/support-compatible
strata, keeping target metadata fixed and recording the permutation. Report
zeroed-z separately because it can be an out-of-distribution intervention.
Neither intervention replaces held-out task probes or the metadata-only control.

Compute Huber error at the fixed Q coordinates. Require finite targets and
predictions there; a numerical failure must fail the step, not shrink Q:
`0.5*e^2` for `|e| <= delta`, otherwise `delta*(|e|-0.5*delta)`
([PyTorch 2.6 definition](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/loss.py)).
Use delta=1 initially. A generic globally reduced MSE is not this objective.

Huber weights residuals, not signal variance directly. Its per-cell gradient
magnitude is bounded by delta, while large residuals still contribute linearly.
Record loss contributions by channel/feature, signal family and event/quiet
regions when defined; inspect which errors dominate before proposing weights
or another objective.

Fix the reduction policy explicitly: average target cells within each eligible
channel; average eligible channels within each example; then average eligible
examples. Record all denominators. This gives each eligible example equal
weight, even if it has fewer observed channels. It refines the specification's
count formula rather than assuming the two reduction policies are identical.

If the entire batch is ineligible, do not call backward or AdamW.step and do not
apply weight decay. Track attempted batches and completed optimizer updates
separately. Mask/data RNG continuation must advance on attempts, including skips;
save its state or use a tested counter-based policy. Check that resumed training
reproduces uninterrupted CPU continuation under the same execution conditions.
Set an explicit attempt limit so an entirely ineligible dataset cannot produce
an endless run while waiting for completed updates. Report exhaustion and skips
by signal family and missingness, rather than weakening mask eligibility.

## 5. Reuse and source ownership

Keep model-specific code under this encoder directory; promote a utility into
shared code only when its contract is independent of either encoder.

| Existing code | Reuse boundary |
|---|---|
| [shared types](../../shared/include/embedding/shared/types.h) | Reuse dimension terminology and basic value/mask types; add explicit RPB-MAE metadata rather than overloading rank conventions. |
| [shared tensor operations](../../shared/include/embedding/shared/tensor_ops.h) | Reuse audited masked reductions. The canonicalizer casts before finite handling; it is not the precision-preserving RPB-MAE ingress. Token channel pooling assumes baseline token-ID geometry. |
| [data APIs](../../shared/include/embedding/shared/data.h) and [implementation](../../shared/src/data.cpp) | Preserve baseline float32 archives and validation. Add a versioned RPB-MAE preparation adapter for precision/metadata, with round-trip and identity checks. |
| [evaluation statistics](../../shared/include/embedding/shared/evaluation.h) | Reuse independent synthetic streams and representation statistics. |
| [shared provider engine](../../shared/include/embedding/shared/feature_evaluation.h) and [feature fitting](../../shared/include/embedding/shared/feature_harness.h) | Own cards, legal fit inputs, support-aware surfaces, PCA/probes, paired populations and reports; encoder adapters supply model-specific fitting/extraction and provenance. |
| [shared reconstruction interface](../../shared/include/embedding/shared/reconstruction_evaluation.h) | Own reconstruction cards, target masks, latent interventions, generic metric scaling, exchange-block uncertainty and reports. The RPB adapter supplies exact exports, decoding and independently fitted metadata-control assets. |
| [historical baseline evaluation](../mtf_jepa_mae_vicreg/src/evaluation.cpp) | Preserve its four-class protocol and private scaler/probe behavior for regression. The shared driver's baseline adapter supplies separately identified train-only controls. |
| [shared objectives](../../shared/include/embedding/shared/objectives.h) | VICReg exists, but is outside the first candidate. Add the required Huber reduction under RPB-MAE first. |

Implemented first-candidate layout:

~~~text
code/encoders/raw_patch_bottleneck_mae/
  include/embedding/encoders/raw_patch_bottleneck_mae/
    config.h       types.h        preprocessing.h
    tokenization.h masking.h      encoder.h
    objectives.h   model.h        workflow.h     training_utils.h
    evaluation_adapter.h reconstruction_adapter.h
  src/
    workflow.cpp main.cpp evaluation_adapter.cpp reconstruction_adapter.cpp
  config/
    smoke.conf evaluation.conf
  tests/
    preprocessing_test.cpp numerics_test.cpp masking_test.cpp model_test.cpp
    workflow_test.cpp reconstruction_adapter_test.cpp cli_smoke.sh periodic_checkpoint.sh
~~~

The compact core is header-only; readout and bottleneck-only decoding are in
`model.h`. Generic evaluation protocols, fitting and scoring are in
`code/shared/feature_harness` headers/source/tests. The encoder-independent
provider contract and orchestration live in `code/shared/feature_evaluation`,
while `code/evaluation/` owns adapter registration and comparison CLI dispatch.
The RPB adapter fits only permitted training observations and schema metadata;
the provider fit interface excludes labels, hidden clean signals and held-out
observations. Ordinary RPB training/inference does not link either evaluator or
baseline. See the [shared integration guide](../../evaluation/README.md).
The parallel reconstruction adapter implements the shared prediction callbacks;
it does not perform target shuffling, metric fitting or held-out scoring.

Use `embedding::encoders::raw_patch_bottleneck_mae`; the original encoder uses
`embedding::encoders::mtf_jepa_mae_vicreg`. Each encoder owns its API and callers
use its canonical include path. Shared primitives stay in `code/shared/`.
Flat forwarding headers and root-level encoder aliases have been removed.
Preserve the original encoder's CLI behavior and archive formats when changing
shared components.
Expose RPB-MAE configuration validation, shape description, inference `encode` and
training `forward` with distinct typed outputs. No teacher-update hook is needed.
The generic evaluator consumes frozen feature surfaces plus validity and
provenance; it must not require either model class.

## 6. Build, checkpoint and concurrent workflow

Reuse the existing managed development container and pinned dependencies.
Build and test inside that container, using the documented
[environment workflow](../../../doc/ENVIRONMENT.md). Do not add a container or
install another framework for the core candidate.

The explicit model targets are `rpb-mae` and `test-rpb-mae`, with a binary named
`embedding_raw_patch_bottleneck_mae` and encoder-specific object/test paths.
Build `evaluation` for the separate `embedding_evaluate` executable;
`evaluate-rpb-mae` runs that evaluator. `feature-harness`, `test-feature-harness`
and `evaluate-minimum` cover the minimum shared driver. The ordinary RPB binary
has no evaluation command. Current baseline targets, default configuration and
historical CLI behavior remain available.
The same separate `embedding_evaluate` exposes `reconstruct`. Its targets are
`test-reconstruction-evaluation` for the encoder-free engine and
`test-rpb-reconstruction` for the RPB adapter. `reconstruct-rpb-mae` invokes
[the reconstruction wrapper](../../scripts/reconstruct-rpb-mae.sh), which
allocates a unique run directory. The minimum harness has no RPB registration
for this command. Commands and defaults are in [the README](README.md).

The current [task runner](../../scripts/task.sh) forces
`BIN=<session>/embedding`; a different session name alone still builds the
baseline selected by the [Makefile](../../../Makefile).
It also fixes `RPB_BIN`, `EVALUATION_BIN` and `HARNESS_BIN` within that session.
Explicit Make
targets select the binary while preserving same-session locking. Use distinct
baseline/RPB-MAE build sessions and run directories. Locks do not protect shared
source edits or GPU capacity; coordinate shared utility changes and resource
budgets before concurrent runs.

Use a separate RPB-MAE checkpoint envelope and loader. Validate encoder ID, format
version, architecture, input schema, output semantics and preprocessing identity
before loading weights. Store model/optimizer state, frozen scaler or verified
sidecar, attempted/completed counters, data/mask RNG state, configuration and
dataset/protocol identity. Baseline v1 checkpoints have no encoder ID; reject
cross-loading rather than guessing from tensor shapes.

Use unique artifact paths and atomic checkpoint replacement. Record repository
HEAD, dirty status and hashes of actual source/configuration inputs: a commit
alone does not identify an uncommitted checkout. Preserve existing reports,
checkpoints, data and legacy build evidence.
When the reconstruction adapter produces an ordinary training checkpoint, its
existing `source_fingerprint` field identifies the core checkpoint writer.
The separate `training-provenance.pt` and `provider-audit.json` identify the
adapter training producer using `EVALUATION_SOURCE_ID`, exact resolved recipe
and fit source manifests. Preserve both identities; the writer fingerprint
alone does not identify the adapter training procedure. Metadata/untrained
control files carry distinct artifact kinds and are not ordinary checkpoints.

## 7. Evaluation before the learner

The historical baseline's independent four-regime fixture is useful for
regression, but lacks
the full task set, validation protocol and compression controls required here.
Its `trained_channels` surface concatenates channel vectors into one probe;
it is not an average of per-channel probe scores. Previously inspected test
seeds are development evidence, not a fresh confirmation set. Historical FSPA-4
and AULC values require explicit artifact and metric provenance before reuse.

The shared training `synthetic_batch` fixture requires F=3 and includes
family-dependent missingness; use it for optimization smoke only. Controlled
quality tasks use separate RNG streams for signal generation, missingness and
task assignment. Masks are statistically independent of class/signal in this
track; task labels still follow the declared signal-generating definition.

The separate shared engine now supplies frozen-feature evaluation with explicit
registered adapters. Its [controlled-pairs-v2 recipe](../../evaluation/cards/controlled_pairs_v2.md)
is development evidence: an instantiated `evaluation-card.json` is saved before
generation or fitting and fixes geometry, matched widths, budgets, seeds and
named compared-method pairs. It does not establish consumer acceptance.
Archived protocol-v1 scores require their recorded source and scripts to
reproduce; current v2 commands do not recreate that protocol. The Stage A plan
below describes required controls and the later acceptance scope.

### Implemented decoder reliance track

The [controlled-reconstruction-v1 card](../../evaluation/cards/controlled_reconstruction_v1.md)
is separate from frozen-feature classification. See
[DECODER_RELIANCE.md](DECODER_RELIANCE.md) for its complete development recipe.
It uses fresh training only, fully observed histories, the current compact core
and unchanged checkpoint/export semantics. It adds no mixer or training objective.
The instantiated card freezes geometry, source budgets, tasks/seeds, bootstrap
budget and resolved adapter recipe before generation or fitting.

The RPB registration API is `make_reconstruction_provider(options, card)`,
returning a shared factory and recipe string. `ReconstructionOptions` resolves
configuration plus main/control updates and batch size; defaults are 128/128
completed updates and eight observations per sampled batch. CLI
`--metadata-steps` defaults to the chosen `--rpb-steps` budget. Unequal explicit
budgets are recorded as a deviation.

The shared provider exposes `encode_visible`, `decode`, `metadata_predict`,
optional `untrained_predict` and audit/asset callbacks. Fitting receives only
permitted training observations and declared metadata. Held-out encoding sees
only visible values, with target storage removed. Decoding receives the exact
compact local vectors and captured fixed metadata. Raw CPU float64 predictions
come from float32 neural decoding followed by float64 inverse scaling; shared
evaluation independently fits its training-only metric standardization.

The mask/metadata control starts at identical encoder/decoder weights. Freeze
the original encoder, feed normalized zeros with actual visible masks and fixed
IDs/positions, run it in evaluation mode without gradients/dropout, and fit only
`decoder_*` parameters. Main and control repeat the same sampled batches,
counter-seeded artificial masks, loss support, optimizer settings and default
completed-update budget. This permits mask metadata through the compact
control representation without signal values. A trained decoder evaluated
at zero z remains a separate intervention.

For each held-out trial, channel c targets original patch `(trial+c)%K`; all K
trials enumerate every patch for each channel. The engine owns intact, shuffled
and zeroed local vectors, metadata-control and untrained predictions, and a
training-mean reference. Shuffle complete different-source groups in disjoint
swaps while preserving receiver IDs, visible/query support and target values.
Bootstrap the resulting two-source exchange blocks, keeping paired variants
and repeated masks together. Use hierarchical standardized masked MAE as the
primary measurement, with standardized Huber secondary. Broader missingness
requires a separate declared track.

Adapter assets include the resumable ordinary main checkpoint and its versioned
raw training/scaler archives, independent typed metadata/untrained weights and
companion producer provenance. Container checks and the frozen 12-run
development experiment passed; see [measured reliance](DECODER_RELIANCE.md).
The held-out advantage over the matched control plus compatible shuffle
degradation supplies synthetic decoder reliance evidence; zero sensitivity or falling
training loss alone does not establish it. The track does not issue consumer
acceptance or replace frozen-feature task evidence.

### Minimum Stage A

Bound A to a runner and fixtures that existing components can exercise:

- Controls: frozen baseline with verified preprocessing, a seeded-untrained
  baseline of the same architecture, raw histories, complete baseline
  descriptors, mask/metadata-only features and shuffled labels. Give controls
  the same permitted metadata. Matched-untrained RPB-MAE is added after C/D
  and required at E; it cannot block A before its architecture exists.
- Probe: one fixed ridge family for the initial binary contrasts, with a
  declared regularization/label budget and train-fit PCA where rank permits.
  Add a fixed tiny nonlinear joint probe as a separately reported secondary
  track for interactions; do not search probe families on final test results.
- Fixtures: reversal/order, level/amplitude shifts and lag sign; correctness
  checks for observed zero versus missing and endpoint cropping. Required
  frequency/multiscale/future suites remain later acceptance work.
- Exit: reproducibility, split/fit isolation, supported compression, invalid-row
  handling, known task solvability and unchanged legacy regression. One
  deterministic correctness run precedes the proposed three development seeds.
  Neither a candidate win nor five confirmation seeds are needed to start B.

Validate tasks with an analytic/raw-observation oracle before judging exports.
A random common phase can make lag sign depend on cross-channel interactions
that a linear probe misses even on raw histories. Such a result is probe
insufficiency, not proof of encoder information loss. Keep the oracle as a
solvability control, not as an embedding score.

### Controlled pairs

| Contrast | Fix or match within each pair | Evidence to collect |
|---|---|---|
| Waveform/reversal | Same global mean, variance and Fourier magnitudes; matched masks/IDs/units and a declared, identifiable orientation label | Raw oracle, complete legacy-tokenizer outputs and served-vector probes; local windows may distinguish the pair. |
| Opposite lag/dependence | Matched marginals and nuisance features, fixed channel identities, known signed-lag range | Raw lag/correlation oracle; local and joint surfaces; linear and fixed nonlinear results reported separately. |
| Level/amplitude shift | Same shape/order, declared units, finite fixed scaling, nuisance metadata uninformative | Recovery from served z, not a decoder-only normalization sidecar. |
| Zero/dropout | Same declared schema; observed zero versus O=false | Correct validity, full-population coverage and separately reported task behavior. |

Keep each source trajectory, its transformed pair and all derived windows in
one split. Reversal does not automatically define an identifiable classification
task for an arbitrary symmetric generator; the oracle must establish that the
declared labels are recoverable. A global descriptor collision is not assumed
to be a collision of the full baseline multiscale tokenizer.

### Full protocol

1. Split source trajectories/generative draws before windows. For real series,
   use chronological splits with purging based on both context and future-label
   support. Keep the main claim support-disjoint; document walk-forward reuse
   separately.
2. Fit pretraining/scaling/PCA/probe assets only on their permitted training
   partitions. Use validation for declared selection; open a fresh confirmation
   test only after freezing decisions. Do not use unlabeled test examples for
   inductive pretraining.
3. Persist split manifests, support intervals, data/generator versions,
   separate signal/missingness/task, initialization, training-mask and sampling
   seed namespaces, and immutable artifact hashes. Pair manifests/corruptions
   across methods and keep confirmation seeds fresh after selection.
4. Evaluate temporal order, phase/event timing, frequency and amplitude/level;
   then cross-channel lag/dependence and declared future targets. Include
   missingness shifts and absent channels. Test the complete baseline tokenizer
   for descriptor collisions, rather than assuming a single descriptor is its
   whole information content.
5. Give every task the same declared label budget, probe family and selection
   budget across encoders. Fit probes on frozen exports. Corruption tests use
   the same fitted probe unless explicitly reporting a separate retrained track.

### Controls and dimensions

Required controls are raw histories, baseline descriptors, frozen baseline,
matched untrained RPB-MAE, and mask/metadata-only features. Include shuffled-label
controls with statistically interpreted results. Compress raw/descriptors using
training-fit PCA; report seeded random projection separately. Persist all
compression assets and numerical rank.

For a frozen baseline checkpoint, verify its actual training preprocessing and
associated assets; the historical evaluator reported an unverified scaler/checkpoint
association. The shared driver supplies fresh train-only baseline controls and
has no verified historical checkpoint/preprocessing association. Do not apply
the new per-ID scaler to old weights and present that
as an unchanged control. If provenance cannot be established, produce a separately
identified baseline control under the new permitted training protocol. Preserve
the historical artifacts and scores under their original protocol.

For C=3, current baseline widths are global 12 and channel concatenation 36;
the initial RPB-MAE widths are global 32 and concatenation 96. Report these native
dimensions and card-declared matched-dimension tiers, initially global width 12
and concatenated width 36. Protocol v2 applies training-fit PCA to every matched
surface, including when its native width equals the target. Per-channel width-12
compression is a separately named comparison because it differs from joint
PCA of the concatenation. A separately trained D=12 RPB-MAE is a capacity experiment,
not interchangeable with compressing the D=32 candidate.

For centered PCA, require `k <= min(input_dimensions, valid_fitted_training_rows-1)` and
sufficient numerical rank. The shared engine records insufficient valid fitting
support or rank as an unsupported fit. Do not advertise zero padding as meaningful
extra
components. Count any validity indicators appended to probe features in their
input dimension and provide the corresponding mask-only control.

Exclude invalid zero-placeholder exports from feature normalization, PCA and
probe fitting. In the primary track, a local surface requires that channel to
be valid; a global surface requires at least one valid channel; concatenation
requires every declared constituent channel to be valid. An incomplete-channel
concatenation with explicit indicators is a separate robustness track. Declare
equivalent support policies for raw/descriptor/external controls.

Declare compared methods and their tiers in the saved card before generation.
Each pair uses its own common-valid population; adding an unrelated provider
cannot change that population. Typed surface kinds, semantic order and support
rules determine eligibility, rather than naming suffixes. Current RPB
`*_channel_concatenation` surfaces concatenate independent local summaries and
require all declared channels; global surfaces require any observed-valid channel.
Report paired task scores on the common-valid evaluation rows, each method's
coverage and its invalid/abstained outcomes on the full population. Never present
a common-valid score alone as robustness. Before acceptance, define the task's
all-population treatment of abstention/failure; do not silently drop difficult
examples or fabricate valid predictions from zero placeholders.

MOMENT belongs in a separate external-pretraining track
([paper](https://proceedings.mlr.press/v235/goswami24a.html)).
Pin its checkpoint, code revision and extraction policy. Its
[embedding implementation](https://github.com/moment-timeseries-foundation-model/moment/blob/main/momentfm/models/moment.py)
uses scalar-channel input and an input mask shared across channels in a sample;
native normalization/reduction also affect comparisons. The adapter must define
F-to-stream mapping, per-stream missingness, padding, pooling back to channel
surfaces and actual support. Do not pretend arbitrary BCHF/per-feature masks
are a native interface, or silently impute unavailable values. No external
dependency is required to pass core structural checks.

The default embedding path applies window RevIN but does not export its saved
mean/scale; reconstruction denormalizes predicted samples. Centering therefore
removes absolute offset from the encoder input; scale invariance is approximate
because of epsilon. Verify this at the pinned revision using the
[official normalization source](https://github.com/moment-timeseries-foundation-model/moment/blob/main/momentfm/models/layers/revin.py).
If statistics are appended to embeddings, identify that as a separate feature
surface and account for its dimensions and support.

### Reports and promotion

Report task scores, selection choices, native/matched dimensions, coverage,
support/skip counts, reconstruction interventions and parameter/training/
extraction costs. Define latency measurement conditions, warmup and hardware.
Diagnostics use exported vectors before any training-only projection and retain
per-semantic-channel statistics across trajectories; flattened B×C variance can
be inflated by channel identity.

Report within/between-channel contributions, effective/numerical rank and
valid independent-trajectory counts. A covariance from n rows has rank at most
n-1; low sample support is not proof of collapse. Flag unsupported n<2 statistics
rather than interpreting returned zeros as measurements. Treat channels/windows
from one trajectory as clustered observations.

Use at least three development seeds and five final paired seeds as specified
by the proposal. Use trajectory-level or temporally blocked uncertainty, not
independent resampling of overlapping windows. Resolve historical AULC's actual
axis, orientation, integration rule, normalization and task weights before
computing or comparing it; its axis may be optimization steps rather than label
budget. Give a label-budget learning curve a separate name when its definition
differs from that historical metric.

Promotion requires all correctness gates, improvement on the preregistered
primary task versus the strongest reproducible control, required-task
noninferiority, acceptable coverage and declared resource limits. Specify margins
and uncertainty rules before final evaluation. Until those choices are frozen,
quality results are exploratory; lower training loss is an optimization result.

## 8. Mandatory implementation checks

These are behavioral tests, not copies of the implementation.

| Area | Required evidence |
|---|---|
| Precision and validation | Constants, observed zeros, float64 large-offset/tiny-variation ingress, scale floors, no-fit coordinates, masked NaN/Inf, observed nonfinite rejection and nonfinite public config values. |
| Mask isolation | With scaler/parameters fixed, perturb hidden observed values with finite alternatives: visible tensors and embeddings stay unchanged. Arbitrary NaN/Inf storage is tested only where O=false. Gradients from embeddings to forbidden raw cells are zero where differentiable. Natural missingness never creates a target. |
| Attention/padding | Vary padding contents without changing valid outputs. Singleton and all-invalid rows remain finite; absent outputs are zero/invalid even with projection biases. |
| Temporal/causal support | Preserve positions after patch removal. Future observations cannot change endpoint embeddings. Per-timestep causal claims require separate prefix/causal tests. |
| Channel semantics | Complete-record permutation equivariance/global invariance; changing another channel cannot change z_local. Changing a semantic ID is validated and uses the corresponding scaler lookup. |
| Bottleneck | C correctness: reconstruction gradients reach encoder/readout/export parameters and decoder has no token bypass. E evidence before claiming useful reconstruction: held-out decoding beats the metadata-only control and worsens under support-compatible shuffled z; zeroed z is a separate intervention. |
| Loss and skipped updates | Hand-check unequal target/channel counts; changing absent cells does not change loss. Ineligible batches leave weights/optimizer state unchanged while attempt/RNG counters advance. |
| Data/protocol isolation | Validation/test mutations cannot change fitted training assets. Reject split/support overlaps, input/output path aliases and incompatible normalization/compression identities. |
| Workflow | Inference checkpoint round-trip, CPU uninterrupted/resumed parity, periodic recovery, explicit encoder mismatch rejection and immutable artifact provenance. |
| Baseline preservation | Existing baseline tests and recorded extraction/old-checkpoint compatibility checks still pass after shared/build changes; compare under the unchanged legacy protocol. |

A new mixer must add observed/inferred support semantics and cross-channel mask
tests, unequal packed-layout alignment, common physical-grid validation and
absent-channel behavior with observed neighbors. Permutation and pre-mixer local
independence checks still apply. A frequency/CNN branch must prove that its full
receptive-field support excludes hidden observations. Those checks accompany
the extension.

## 9. Implementation milestones

| Stage | Deliverable | Gate to continue |
|---|---|---|
| A: minimum evaluation foundation | Bounded controls/fixtures above, split/support manifests, frozen-feature runner, compression and task reports | Fit isolation, reproducibility, raw task solvability, invalid-row policy and legacy regression pass; no RPB-MAE or external-model prerequisite. |
| B: ingress and tokenization | Explicit schema, precision-preserving scaler, masks and ordered patches | Precision, support isolation, metadata/permutation and missingness checks pass. |
| C: compact core | Shared temporal encoder, trained local readout, deterministic global pool and bottleneck decoder | Finite optimization smoke, correct gradients/loss/skip behavior and no decoder bypass. No utility claim yet. |
| D: workflow | Independent targets/binary/config, checkpoint/export schema and recovery | Round-trip/resume checks and baseline compatibility pass inside the existing container. |
| E: core experiment | Native and dimension-matched comparisons with paired controls | Validation-frozen confirmation report; promotion only under declared quality/cost criteria. |
| F: one extension | First aligned temporal channel mixer; later multiscale or future-observation prediction | Correct, frozen, reported parent with demonstrated local utility; predeclared deficit/hypothesis and affected checks. Full promotion still requires all required tasks. |

Prefer the aligned pre-pooling mixer described above for lag/phase experiments.
Do not require an independent parent to pass every lag task before testing the
mixer intended to address a documented interaction deficit. Require correctness
and useful local features first, retain every required task in reports, and
withhold full consumer promotion until its acceptance gates pass.

The evaluator in A can be built before the encoder. Establish its first bounded
task/control set, then extend it with the required cross-channel/future tasks
before any full consumer acceptance claim. Do not indefinitely delay basic
engineering because the application-quality threshold remains undecided.

If the bottleneck fails while a conventional token decoder succeeds, run that
diagnostic and separate export-width controls before discarding the tokenizer
or objective. Check optimization, decoder adequacy and bottleneck capacity
separately; do not automatically widen z first. Keep diagnostic decoder-capacity
and export-width changes distinct from the original matched-width comparison.
JEPA/EMA recipes, regularization and specialist/prediction-head banks remain
later research decisions. Keep task-head/routing state outside the encoder
export; a shared encoder with multiple heads is the first later regime hypothesis,
with specialist encoders only after that simpler explanation is tested.

## 10. Decisions to freeze before final experiments

The guide supplies initial engineering defaults. Record any deviations and
resolve these application choices before interpreting a final result:

- Semantic channel/feature identities, units, source precision and sampling
  interval; whether irregular time or inferred absent-channel output is required.
- Required local/contextual/global consumer surfaces, task/target definitions
  and forecast horizons. Keep required cross-channel tasks even if the core fails.
- Primary metric, required-task margins, uncertainty rule, acceptable
  improvement, coverage/skip limits and training/extraction resource budgets.
- Pretraining/label budgets, validation search budget, fresh confirmation seeds
  and the exact AULC/FSPA definitions and historical artifacts, if used.
- External checkpoint/adapter policy and disclosure of external pretraining.

These decisions govern quality acceptance. They do not authorize implementing
all deferred branches at once or changing the existing encoder's semantics.

## 11. Shared evaluation refactor status

The feature-provider boundary, separate evaluation executable and protocol-v2
cards are implemented. Container validation passed shared engine isolation,
both feature adapters, repeated-card reports and model/workflow regression
checks. Original protocol-v1
engineering checks and scores remain archived in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), with reproduction requiring
their recorded source and scripts. The reconstruction adapter and shared
decoder-reliance track passed container validation and their frozen 12-run
development experiment. [Measured results](DECODER_RELIANCE.md) establish
decoder reliance on these synthetic histories. The stronger
[frozen-probe comparison](FROZEN_PROBE_ADVANCE.md) finds lag information accessible
through native concatenation and the fixed nonlinear probe; compact linear
training gains remain inconclusive. The optional aligned
[channel-mixer ablation](CHANNEL_MIXER_ADVANCE.md) improves the declared compact
lag-sign ridge comparison in all three development seeds, with positive
within-run paired intervals. Its local/contextual surfaces remain distinct,
and mixer-off exports and existing checkpoints retain compatibility. This
does not establish universal nonlinear improvement or cost equivalence.
The subsequent [fresh-seed/stress track](FRESH_SEEDS_AND_STRESS.md) completes
five development seeds and a reusable fixed-readout synthetic outage sweep.
Four primary compact lag intervals are positive and the fifth crosses zero;
pooled-global evidence is weaker and heavy losses erode the benefit. Every
stress case reuses ordinary fitted assets, with explicit observed support and
conditional/full-population correctness. Consumer confirmation contracts,
domain-specific outages, costs and promotion evidence remain pending.
See the [shared review](../../../doc/EVALUATION_IMPLEMENTATION_REVIEW.md) for the
authoritative implementation limits and subsequent validation results.

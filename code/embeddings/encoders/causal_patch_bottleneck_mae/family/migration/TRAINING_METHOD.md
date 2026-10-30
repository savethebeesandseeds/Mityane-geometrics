# Training method travels with the encoder

For this locally trainable family, the deliverable includes the network,
training implementation and the knowledge needed to use that implementation.
An inference checkpoint alone cannot establish how to fit, train or resume a
fresh instance. Preserve the source/configuration/recipe/fitted-state/instance
identities separately when importing this package into Cuwacunu or another
application. Paths in the source map retain their original repository meaning;
relocation creates a separately recorded consumer source identity.

## Source map

| Responsibility | Maintained implementation |
| --- | --- |
| Configuration, causal cutoff, fresh construction, optimizer and typed artifacts | [family.h](../include/embedding/encoders/causal_patch_bottleneck_mae/family.h) and [shape.h](../include/embedding/encoders/causal_patch_bottleneck_mae/family/shape.h) |
| TRAIN fitting, mask generation and complete update step | [training.h](../include/embedding/encoders/causal_patch_bottleneck_mae/family/training.h) |
| Preserved numerical implementation, including objective and masks | [V16 include closure](../../versions/v16/include/) |
| Runnable fresh/resume example, row policy, budgets and run records | [cpb_family_train.cpp](../../../../scripts/cpb_family_train.cpp) and [commands/output contract](../examples/TRAINING.md) |
| Generic typed data boundary | [OBSERVATION_WINDOWS.md](../../../../../data/observations/OBSERVATION_WINDOWS.md) |
| Actual engineering coverage and unresolved coverage | [portability receipt](../evidence/portability-2026-10-22.json) and [compatibility receipt](../evidence/compatibility-2026-10-22.json) |
| Historical study recipe, evidence and reference limitations | [frozen V16 protocol](../../../../evaluation/protocols/cpb_v16_temporal_v1/PROTOCOL.md) and [research-reference.json](../../research-reference.json) |

`family/` alone is incomplete: its headers depend on the preserved
`versions/v16/include/` closure. The example runner also needs the generic
observation-window header. Keep inherited notices and the root license with
the source. LibTorch is an explicit external dependency, not bundled source.

## Causal input and preprocessing

Supply genuine CPU F64 observations, Boolean observation masks, semantic channel
IDs, a declared uniform timestamp grid and an explicit aligned cutoff per
example. The boundary retains the most recent configured H ticks ending at the
cutoff and masks any required left padding. It crops before reading observation
values/masks for validation, casting, statistics or encoding. Unobserved cells
are sanitized before numerical operations. Replacing missing observations with
observed zeros changes the data contract.

Fresh fitting accepts only the caller's admitted TRAIN calibration windows.
The helper cannot infer TRAIN membership from the data. Record source and split
manifests, the actual calibration population and feature/semantic ordering. The
portable example visits every declared TRAIN row once in source order, in
batch-size chunks; it weights windows rather than deduplicating overlapping
source observations.

The parent scaler canonicalizes channels by semantic ID, rounds observed raw
values F64→F32→F64, and accumulates population count/mean/centered-M2 using Chan
batch combination. It floors each standard deviation at the configured scale
floor, default `1e-6`, and refuses missing support for any input coordinate.
Centered moments protect rare F32-scale variation around a large offset from
sum-of-squares cancellation. This is a new fitting recipe; it is not claimed
bit-identical to historical study calibration.

The relative-change bank retains its separate raw-F64 feature arithmetic and
documented F32 rounding, validity and bounded-feature statistics. Its frozen
identity binds source, split, preprocessing recipe and parent-scaler identity.
Neither fitter calls the encoder or consumes supervised readout labels. Freeze
once before fresh construction; save both fitted states with the instance and
never refit them on resume, validation, TEST or inference.

## Mask and target support

The mask ratio defaults to `0.25` and is part of configuration. For each
example/channel, shuffle observed patch positions and hide
`clamp(floor(mask_ratio*M + 0.5), 1, M-2)` complete patches, where M is the number
of patches containing observations. The shared history width must be evenly
divisible by the patch length, and that quotient must be at least 3. Each
channel's own candle count must also be evenly divisible by the patch length,
and that quotient must be at least 3. Exactly 3 is legal. The shared row is the
longest of those counts, so a shorter channel's left pad is whole empty groups.
The counts do not share a span of time. One sample has one anchor. Each
channel ends at its own latest bar at or before that anchor. A daily count is
that many daily bars, not that many days of a finer channel.
A group is either empty or filled with that channel's candles. The loss
includes every such channel. The mask still needs at least three observed
patches: it hides one and leaves two visible. A larger filled-group count still
masks.

Let O be original observation support and A the artificial patch mask. Query
targets are **Q = O & A**; original visible context is **O & ~A**. Optional
additional context deletion E must be a subset of that visible context and
retain valid model support. It must never add targets. A batch without eligible
channels consumes an attempted-batch counter but performs no optimizer update.
Keep attempt and completed-update counters distinct.

## Objective and gradient treatment

The implemented training objective is:

`GaussianNLL + 0.1 * (HuberLag1 + HuberLag4 + HuberLag16) / 3`.

The Gaussian term uses `0.5 * ((mean-target)^2 / variance + log(variance))`,
omitting the constant `0.5 * log(2π)` term. It sanitizes unsupported cells and promotes
mean, target and variance to CUDA F64 before arithmetic. Reduction averages
supported cells within each channel, eligible channels within each example,
then eligible examples. A finite negative NLL is valid.

Each temporal term compares reconstructed and target differences at the same
example/channel/feature, with nonwrapping lag 1, 4 or 16 on the original grid.
Both endpoints must be in Q. The residual is divided by fixed √2 and passed to
Huber with delta1. Reduction uses each lag's own supported pairs, eligible
channels and examples. An empty lag contributes graph-connected zero and keeps
its fixed one-third weight; report it as unsupported. Do not renormalize over
active lags, expand targets or substitute compressed observed-time indices.

Targets are detached. Mean reconstruction receives Gaussian and temporal
gradients; the variance head learns from Gaussian NLL with its latent input
detached. This prevents that variance route from reshaping the encoder while
retaining variance-head learning. The shared global clip still couples update
scaling. Temporal differencing does not supervise constant offsets; the retained
value objective supplies that information. A composite training loss is separate
from reported Gaussian reconstruction NLL and downstream readout quality.

## Updates, randomness and continuation

`train_step()` owns zeroing, masking, forward, one backward pass, finite-gradient
validation, one global learned-parameter norm clip and one AdamW update. Default
learning rate is `0.001`, weight decay `0.0001`, clip norm `1`, betas `(0.9,0.999)`,
epsilon `1e-8`, AMSGrad false. Learning rate, decay and clip can be configured and
are checked on load. Frozen preprocessing and fixed priors are verified after
updates. No scheduler, gradient accumulation or distributed update is supplied.

The example sampler draws rows with replacement using
`mt19937_64(training_counter_seed(row_seed, attempt))` modulo row count. Masking
uses its separate saved seed and absolute attempt. The example performs **no
additional context deletion**, but binds its context seed. An application that
adds a deletion/augmentation policy must bind its algorithm and stochastic state
as part of its complete recipe.

Typed training states retain configuration, fresh initialization, weights,
frozen scaler/bank, source/split/recipe identities, learned-parameter order,
AdamW moments/steps/content identity, optimizer options, completed/attempted
counts and row/mask/context seeds. Exact continuation is established for the
declared dropout0 path and explicit counter streams. Implicit Torch CPU/CUDA
RNG state is not saved; additional stochastic mechanisms need their own state
contract before claiming exact continuation.

The example recipe also binds exact dataset, settings and executable SHA256
bytes, batch size and sampling policy. Resume performs no fitting and requires
those bindings to match. Total update/attempt ceilings and save cadence may
change. Inference archives cannot resume training. Export in eval mode with
cleared gradients, retain the saved configuration/fitted state and independently
check the published archive and its provenance at the consumer boundary.

## Historical recipe and scientific insight

The frozen V16 study trained fresh weights against matched fresh V4 controls.
Its controlled change was the hidden temporal auxiliary above; it imported no
trained ancestor or teacher. Historical C3/H128/F9/P8/native36, batch8,
2048 updates, 4096 attempts, January TRAIN/4096 calibration anchors, 25% masking,
15% context deletion and its row/context streams belong to that named protocol.
They are preserved research bindings, not general application recommendations.
The original training source is
`code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/train_main.cpp`, with
calibration/context/sampling helpers under that protocol's `include/` directory.
Those are origin references outside the portable bundle; the bundle carries
historical protocol/results context, not the complete market-study execution
closure. Exact historical replay also needs its frozen admission, data,
dependency closure and original receipts. The configurable example changes the
parent fitting arithmetic, row policy and context-deletion policy; equal
network/objective code alone does not reproduce historical training.

V16 supports the value of retaining temporal structure in reconstruction
supervision for the measured trend/activity tasks. Its study verdict remains
mixed: volatility protection regressed and no candidate retained all eight
strongest timing conditions. The default seed946103 reference is selected from
observed evidence, not a universal best encoder. January/March selection evidence
is consumed; forecasting, decisions and composed application utility remain
unestablished. Preserve V3, V4 and specialist results with their original scopes.

Engineering checks establish configurable construction, real synthetic updates,
typed export and declared resume behavior, including exact comparison with the
preserved implementation. They do not establish quality at every supported
shape or reproduce a historical calibration recipe. See the receipts for actual
source/binary pins and physical work. The example's attempt-budget-exhaustion
path is implemented but was not exercised end to end in that receipt.

Cuwacunu's application pipeline is a separate recipe and artifact envelope. Its
current sampler uses SplitMix64 with rejection-uniform replacement, rather than
the portable example's MT19937_64/modulo rule. Its authenticated publication and
selection records are application contracts; raw family archives alone are not
substitutes for them. Record this docking explicitly rather than silently
renaming the example or a historical V16 instance.

Cuwacunu owns real-data adaptation, verified TRAIN/calibration membership,
application feature and availability semantics, budgets, orchestration, durable
recovery and independent utility evaluation. The example loads its dataset in
memory on one CUDA device. Scaling, sharding and deployment require consumer
engineering; changing those policies creates a separately identified recipe.

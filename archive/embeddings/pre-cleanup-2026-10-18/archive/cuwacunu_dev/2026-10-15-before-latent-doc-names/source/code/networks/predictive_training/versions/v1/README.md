<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE. -->
# Frozen-encoder predictive training v1

This physical version owns
`cuwacunu::networks::predictive_training::v1`. It combines the separately
versioned affine State Representation Network and deterministic direct-horizon
Dynamics Model. It imports no encoder, market loader or application target.
Include the three component `versions/v1/include` roots and
`cuwacunu/networks/predictive_training/v1/training.h`.

`Trainer(config, identity, embedding_mean[D], embedding_scale[D],
reference_scale[d])` owns online G, detached target G_bar, A, AdamW, counters
and a resumable mt19937_64 sampler. The default dimensions/settings are API
defaults, not an admitted experiment. The runner must supply a prospective
card and exact immutable identities before quality training. There is no
retention auxiliary (`lambda_ret=0`); mandatory retention probes remain an
external advancement guard.

`step(PredictiveBatch)` consumes finite saved native embeddings `[B,K,D]`,
future embeddings `[B,Q,D]`, a boolean `[B,Q]` target-side eligibility mask,
and a separately declared regularizer population `[N,D]` plus distinct int64
origin IDs `[N]`. Eligibility must already account for input/target support,
roles and availability in the pair ledger. The trainer detaches every
embedding input at its boundary, including future and regularizer tensors.
No future value, validity mask or realized error enters A.

The matched arms are `Arm::frozen_state`, `Arm::blocked_prediction` and
`Arm::full`. The blocked arm detaches all K online states on the prediction
path but retains regularizer gradients into G. Its predictive loss still
trains A. Frozen G is excluded from the optimizer. G_bar is always excluded,
has no gradients and uses the same frozen embedding transform as G.

The predictive objective is mean squared reference-normalized error, averaged
equally over eligible origin/horizon pairs and all d coordinates. Variance
and covariance use the entire served state divided by fixed TRAIN reference
scales. One centering operation spans all N distinct-origin state rows, and
sample covariance divides by N-1. The variance penalty is
`mean(relu(floor - sqrt(diag(C)+epsilon)))`; covariance is the sum of squared
off-diagonal entries divided by d. N must be at least two. The graph remains
live for all rows; no detached memory bank, per-asset pseudo-population,
duplicate origin or average of separate microbatch covariance losses is used.
`StepReceipt` records N and `min(d,N-1)`; low rank when N<=d is legal.

AdamW and EMA run only for eligible prediction attempts. An all-invalid mask
consumes an attempt and clears gradients but updates neither component nor
teacher, even if regularizers are configured. An explicit runtime error
consumes no completed attempt. Nonfinite loss/gradients are rejected before
the optimizer action. Optional clipping uses separate norms for G and A so
G's gradients cannot change A's clipping factor across attribution arms.
EMA executes after each successful optimizer update using
`target += (online-target)*(1-decay)`, which also preserves an unchanged
teacher exactly. `online_updates` counts optimizer actions with a defined G
gradient; it is distinct from completed A updates and attempted batches.

Save checkpoints between attempts, after any sampler draws for that boundary.
`save_training(path)` preserves all three networks, frozen transforms/reference
scales, exact config/arm/loss/EMA/clipping/optimizer settings, AdamW parameter
mapping and moments, every counter, mt19937_64 engine and CPU/all-device CUDA
RNG states. `sample_indices(population,count)` samples with replacement and
records draws; the application must separately declare regularizer sampling
without repeated origins. `load_training(path,expected_config,expected_identity)`
requires exact independently supplied identities and rejects inference files,
configuration drift, orphan/remapped optimizer entries and inconsistent step
counters. Its successful return restores process-global Torch RNG streams.

The typed `save_inference`/`load_inference` pair excludes optimizer/RNG state
and freezes G, G_bar and A. It preserves their distinct roles: predictions use
online G, target-space baselines and realized targets use the archived G_bar.
Every forecast bank must additionally bind the actual archive SHA-256 and
coordinate/reference identity; a version tag is not a trained-instance ID.
Immutable publication reserves a private sibling staging name and atomically
links a new destination, refusing replacement. The parent directory must exist.
The caller is responsible for external file integrity/hash validation.

`serving.h` folds state-history, forecasts and arbitrary unchanged linear
readouts into an affine function of native embedding history. It includes the
frozen input transform, singular/rectangular G, forecast bias and optional
last-state residual. This is an algebraic check without refitting, and it
does not establish nonlinear expressive power or market usefulness.

Build and run only through the managed Linux development container with a
named session and the Make fragment `versions/v1/v1.mk`. Targets are
`build-predictive-v1-engineering` and `test-predictive-v1`. The artificial
fixture checks actual CUDA gradient isolation and updates, full-graph pooled
covariance, affine/readout folding, teacher/skip behavior and bitwise next
update resume in every arm. Quality runs, encoder changes, joint adaptation,
probabilistic transitions, portfolio decisions and publication are separate.

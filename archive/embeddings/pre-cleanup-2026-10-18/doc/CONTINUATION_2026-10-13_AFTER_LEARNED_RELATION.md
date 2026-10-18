# Continuation after the learned relation screen

The working configuration is **RPB-v20.alt-01**. Preserve its original capacity
cohorts and the byte-matched v20 API/settings in `../cuwacunu`.
RPB-v21.alt-01 is a separate measured candidate, not a replacement.

The [completed screen](CONFIGURABLE_LEARNED_RELATION_V1.md) found useful learned
timing: learned20 rises from approximately 50% to 62.50%/61.00% intact
Linear/Neural accuracy. Complete32 rises to 97.85%/96.48% intact, but drops to
65.43%/78.06% with extra deletion. Reconstruction protection passes. Nine of
54 gate conditions fail, all involving deleted readouts. Both 512-update
trajectories are complete and audited; no third master was allocated.

## Completed exploratory TRAIN diagnosis

The [scalar support audit](results/configurable_learned_relation_v1/train-support-audit/README.md)
was selected after the quality outcome. It used saved TRAIN masks and new
checkpoint weights, with no validation payload, encoder/classifier execution
or updates. It is exploratory mechanism evidence, not a prospective quality
test or causal proof.

The trainable support part of the bank feeds `n / (H - lag)` through free
additive weights alongside odd/even relations. All 19,440 support weights are
nonzero in each checkpoint. Independent scalar estimates of this contribution
have uncentered RMS 1.784/2.772 and centered RMS 0.074/0.085 for the two cohorts.
It acts chiefly as an offset. Paired TRAIN examples have identical masks, so
this term is exactly identical within each opposite-timing pair.

The frozen generator independently erases coordinates with probability 0.30;
it does not select an exact deletion count. Four distinct endpoints therefore
survive with expected probability `0.7^4 = 0.2401` under the ideal Bernoulli
policy. This gives a direct route for a density-dependent shift even when the
timing has not changed. It is an expectation from TRAIN, not an observed
deleted-feature effect or proof of the classifier failure's cause. Scalar
arithmetic is not a bit-exact CUDA matmul replay.

## Next architecture

Declare a **separate RPB-v22 content-only ordered relation design**. Retain
support for validity, common-support normalization and numerical guards,
while preventing it from becoming an unconstrained additive learned input.
Project only the normalized odd/even moments. At C3/F9, the bank becomes
1,944 inputs instead of 2,916; the learned projection is 20 by 1,944.

Keep all four lags, all channel/feature pairs, base v20, native32, the exact
fixed spectral tail, zero initialization, waveform decoder/objective, masks,
optimizer, fixed classifiers and 512-update budget. Give the new model a
separate API/config/checkpoint/layout/card/output contract and train it from
initialization. Never relabel or resume the stopped v21 instances.

This is prospective, not implemented or admitted. Before quality, verify
generic shapes/masks, original initial output/RNG parity, exact fixed-tail
preservation, actual CUDA updates and immutable typed checkpoint reload.
An artificial witness should change redundant support counts while retaining
the same odd/even moments and confirm the relation contribution is unchanged.
Do not claim arbitrary deletion invariance: finite estimates still change
with masks, and missing values can remove information.

Freeze the quality gate before training. Require learned-coordinate usefulness
and complete-embedding robustness in both views, plus the same original-query
protection. Reuse saved controls and fixed heads without refitting them. Keep
every allocated cohort, stop a failed frozen recipe and avoid rate/head/seed
or threshold rescue.

The learned-only decoder idea addresses whether reconstruction bypasses learned
coordinates. It is not the chosen next step after this deletion failure, because
v21 already demonstrates learned timing and passes reconstruction protection.

## Repository and environment boundaries

Keep failed v21 research code and evidence in cuwacunu_embedding. cuwacunu
continues using the selected v20 API; all 15 backport pairs remain pinned and
equal. A successful new portable model may later be mirrored as an isolated
API with a manifest and focused CUDA check. Synthetic gates do not choose a
market checkpoint or change serving defaults automatically.

Use the existing managed containers and named build sessions. Encoders run on
CUDA; fixed classifiers and saved arithmetic may use CPU. Preserve the latest
capsule `output/runs/rpb-v21-learned-relation/admission-0PGNmn`, its complete
source/card/settings/artifact receipts, and both artificial engineering attempts.
Do not rerun old encoders or controls to read their scores.

Market training, new TEST allocation, bounded streaming TRAIN scalers and H256
quality remain distinct work in the unified market preparation notes.

<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# CPB-v2.market-01 review — 15 October 2026

**Keep CPB-v1.market-01 as the baseline and sole dev application encoder.**
CPB-v2 is a completed, preserved research candidate whose frozen advancement
gate failed. The [per-task report](CAUSAL_FINETUNE_V2_RESULTS.md) contains all
market and TEMPO-5 panels, individual head fits, physical-unit errors, coverage,
reconstruction, costs and exact numerical conditions. The
[instance record](../code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v2.market-01.json)
and [finalization receipt](causal_finetune_v2/finalization.json) locate local
artifacts and audits. Neither v1 nor any earlier research record is replaced.

## What changed

V1 corrected the temporal computation while importing saved V30 weights with
zero new causal updates. V2 tested the concrete concern that those weights
needed adaptation: one native36 end-to-end causal fine-tune from v1, 2048 actual
CUDA updates/B8, with the same Gaussian waveform objective and fixed heads.
All 75 learned parameter tensors moved; the literal spectral prior, input
scaler and bank normalizer stayed exact. V2 owns a complete independent source
copy. Initial144-row/native5184-coordinate bit parity allowed reuse of saved
v1 initial quality with zero initial head refits. Only final2048 was scored.

## What the result supports

The original market gate passes8/9 and strict native36 protection7/8; neural
relation fails both. Against v1 itself, only2/8 market scores improve: linear
trend0.7970341369141087 and neural activity0.6944814572937575. Volatility and
relation deteriorate. Neural trend remains unstable:2/3 fits positive, mean
0.23900308395985534 versus v1's0.24368774940638713. Three head fits on this
single market population do not estimate independent market-cohort uncertainty.

On TEMPO-5 complexity5/5, three of eight stronger frozen retention floors pass.
The three scoped improvements over v1 are synthetic-scaler deleted Linear
93.359375%/Neural92.88194444444444% and market-scaler deleted Linear96.22395833333334%.
The other five timing measures decline. All coverage remains100%.

Fixed-query standardized TRAIN MAE improves0.27671297318712434→0.2715756327864138,
while known VALIDATION MAE worsens0.3887982074420022→0.41617756793800925.
Gaussian NLL moves TRAIN−0.3794596571636123→−0.548151951739501 and
VALIDATION0.8059467304502377→1.107129785299853. Encoder compute126.551047711s
excludes loader, checkpoint, proof, export and readout work. These observations
show a retention/generalization tradeoff in this exact recipe. They do not
isolate which parameter block caused it, rule out all causal training, quantify
a historical leak or establish forecasting performance.

Engineering passes before quality: masking/cutoff invariance, actual CUDA
learning, exact typed resume at512 on the same trajectory and final saved-model
causal proof. Independent saved-arithmetic audits pass. No old encoder training,
post-encoder PCA, head tuning, alternative checkpoint selection or rescue run.

## Next comparison

Propose a separately declared version initialized from the preserved v1
checkpoint, freezing the complete parent32 and its decoder while updating only
the four-coordinate change projection, residual decoder and variance blocks.
This tests whether branch adaptation can help without moving the representation
that already carries v1's volatility/relation strengths. Exact parent-prefix
invariance must be proved during learning and typed resume, alongside the full
future-value/mask interventions on initialized and final saved artifacts.

Keep the same data, input/bank scales, Gaussian objective, native36, 2048/B8
budget and fixed heads for this proposed comparison. Declare trainability,
random streams, numeric v1/market/TEMPO retention guards and final-only scoring
before invocation. Do not fit to synthetic labels or alter heads in response to
these results. This is a proposal, not a declared or running experiment; the
completed v2 recipe remains stopped.

## Saved work and publication boundary

The prior application integration is committed and pushed in both repos:
dev486886c361fe124b97659ba16c22857aae303d6d and research
e6bdd56ea16ddbb82444283e984158948e69032e. New v2 research is local and uncommitted.
Five main checkpoints, initial/final embeddings, fitted heads, query witnesses,
training trace, source capsules, engineering fixtures and failed preflight
attempts are preserved. Weights stay ignored by Git. Research-local v1 is an
exact independent copy of the original selected checkpoint, SHA256
54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895.

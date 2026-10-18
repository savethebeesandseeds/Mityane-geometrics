<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE. -->

# Causal patch bottleneck MAE

**CPB-v3.market-01** is the sole active application encoder in cuwacunu_dev and
our first clean causal training baseline. Its complete independent implementation
lives in [`versions/v3/`](versions/v3/), in namespace
`embedding::encoders::causal_patch_bottleneck_mae::v3`. The public version entry is
[`versions/v3.h`](versions/v3/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v3.h).
The directory owns all 47 source/configuration/test/build/provenance files and
matches the research copy exactly. No mutable encoder implementation is shared
with another version or forwarded into a historical family.

Use the application facade `cuwacunu::encoder::primary` rather than creating a
second selection path. [The current integration record](../../../doc/CAUSAL_PRIMARY_INTEGRATION.md)
records the checkpoint, input contract and application verification status.
`config/encoder.conf` fixes C3/H128/F9/P8/native36 on CUDA:0;
`config/encoder_runtime.conf` separately selects the artifact path and hash.
The [live version catalog](versions/index.json) contains one active version.

## Selected clean artifact

All 293672 learned values start from the fresh seed 943001 constructor. The
3888-value literal analytic prior is rebuilt directly. There is no pretrained
ancestor, imported learned state, old optimizer or old normalization state.
The single batch size 8 CUDA trajectory completed 2048 updates: source_updates=0,
causal_updates=2048. Both input and bank statistics were fitted anew from the
declared TRAIN-only raw windows and then frozen.

The selected local ignored checkpoint is
`output/baselines/CPB-v3.market-01/inference.pt`, 1290625 bytes, SHA-256
`50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e`.
The facade requires its typed v3 provenance, exact counters/seed and new TRAIN
statistic identities. Missing or incompatible artifacts fail without fallback.
It performs frozen inference without fitting statistics, training or head fits.

## Cutoff contract

The inference entry takes genuine raw CPU Float64 history, Boolean observation
masks, semantic channel IDs, actual timestamps and explicit per-row cutoffs.
It removes later values and masks before finiteness checks, rounding, scaling,
support counts or feature construction. Each output uses at most the last
128 ticks ending at its own cutoff, with hidden left padding for unavailable
past. Temporal attention blocks later original-patch endpoints; channel mixing
stays at the same patch time.

Return values include native36 coordinates, validity and cutoffs. Honor invalid
rows and call earlier times with their own cutoff. There is no full-history
inference bypass or post-encoder PCA. The saved initial and final research
artifacts passed all 128 cutoffs, 2350 future-value/mask interventions and
30 internal temporal cases, plus actual CUDA learning and exact typed resume.
New dev integration verification is recorded separately from those reused proofs.

## Measured quality and history

The [research report](../../../../../Mityane-geometrics/code/embeddings/doc/CAUSAL_FRESH_V3_RESULTS.md)
and [review](../../../../../Mityane-geometrics/code/embeddings/doc/CAUSAL_FRESH_V3_REVIEW.md) contain
new initial/final embeddings evaluated with the same fixed heads. V3 learns
strong volatility and TEMPO-5 timing, but neural trend and relation remain weak;
known-validation Gaussian NLL worsens despite better mean reconstruction.
Coverage is 100% in the measured scope. These mixed results establish a clean
baseline, not an overall winner or forecasting claim.

CPB-v1 is a preserved pretrained historical reference; v2 also inherits learned
weights. Neither supplies a quality target for clean causal candidates. Their
research implementations, checkpoints and results remain preserved. The old dev
v1 source, integration, settings and catalog were archived before replacement in
the [67-file snapshot](../../../../../Mityane-geometrics/code/embeddings/archive/cuwacunu_dev/2026-10-15-before-cpb-v3-primary/manifest.json).
The earlier RPB archive and all historical artifacts remain protected. Frozen
old records retain their original meaning and are not live application selectors.

Future encoder research belongs in the embeddings component of mityane_geometrics when requested. Each
new version must own a complete independent implementation. Preserve v3 as the
clean baseline, all historical scoped leaders and local ignored checkpoints.
This integration starts no new training or quality comparison.

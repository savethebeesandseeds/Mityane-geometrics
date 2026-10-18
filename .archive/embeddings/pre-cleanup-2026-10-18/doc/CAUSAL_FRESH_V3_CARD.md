<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# CPB-v3.market-01: fresh causal training baseline

Declared 15 October 2026 before calibration, builds, encoder training or quality
testing. The matching JSON card freezes the protocol. Read the separate
[provenance correction](CAUSAL_TRAINING_PROVENANCE_CORRECTION_2026-10-15.md).
CPB-v1 remains a preserved pretrained historical reference and the selected dev
artifact. CPB-v1/v2/RPB scores are not acceptance targets or retention floors.

## Question and model

Can this architecture learn useful native embeddings when every learned block
starts fresh and every forward is causal from update zero? This establishes a
clean training baseline; it does not ask a scratch model to retain a pretrained
model's scores. Trust evidence and task quality are reported separately.

CPB-v3 owns a full copy under
`code/encoders/causal_patch_bottleneck_mae/versions/v3/`. Keep C3/H128/F9/P8,
native36=parent32+change4, width64, three layers, four attention heads,
feedforward256 and decoder-hidden128. Keep backward-only relation banks,
cutoff-before-preprocessing, causal temporal attention and the same Gaussian
waveform location/scale reconstruction objective. No post-encoder PCA.

Initialize every learned parent, pooling, projection, mean decoder, change,
residual decoder and variance tensor directly under seed943001, including
intentional architectural zero initializers. Reconstruct the3888-value literal
analytic spectral prior directly. No checkpoint, teacher, pretrained parent,
distilled feature, decoder state or optimizer moment may initialize this run.
`source_updates=0` and `causal_updates=0` in the first saved typed artifact.
No old initializer/import API is part of the new public version.

## Data and fresh calibration

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute candles, nine raw
features and historical-window targets · **complexity4/5**. Reuse the pinned
4096 H256-legal TRAIN anchors from January1–21, consuming only their recent128
observations. Use112 TRAIN/32 known chronological VALIDATION probes from the
saved selections/targets, with VALIDATION January22–28. January29–31 is unused.
These are retrospective representation tasks, not forecasting measurements.

Freshly fit the input scaler from those4096 TRAIN recent128 windows only:
genuine recordF64→F32→CPUF64, per-channel/feature observed count/sum/sumsq,
window-weighted population statistics, minimum support2 and scale floor1e-6.
Freshly fit the270-coordinate relative-bank normalizer on the same windows:
genuine rawF64 causal bank→F32→CPUF64 observed count/sum/sumsq, minimum2 and
floor1e-6. No encoder forwards or validation statistics enter calibration.
Save sufficient-statistic witnesses, counts, IDs and hashes; freeze both
calibrations before construction/training and verify they stay unchanged.
The old scaler/normalizer artifacts are not loaded as fitted input state.

## Single bounded trajectory

One encoder run, exactly2048 successful CUDA updates, batch8, at most4096
attempts. Fresh AdamW at0.001, weight decay0.0001, global gradient clip1;
dropout0,25% hidden-query and15% extra-context deletion. Seeds943001
initialization,943002 row sampling,943003 masks and943004 context, with exact
absolute-counter stream/counter evidence. No new loss or task-label supervision.

Use the pinned original hidden reconstruction query plans directly (1/16 query
fraction); no new query seed is consumed. Report standardized fixed-query MAE
and Gaussian NLL separately on112 TRAIN/32 known VALIDATION windows. A random
minibatch loss is not the reconstruction query score.

Save typed inference states at0/2048 and full TRAIN states at512/1024/2048.
Resume the actual trajectory at512 and prove exact model/Adam state identity,
without replaying updates. Only initial0 and final2048 receive quality scoring;
no intermediate checkpoint selection, seed/rate/budget rescue or head tuning.
Separate synchronized training compute, calibration, loading, checkpoint I/O,
engineering proof, export, head fitting and report time.

## Trust admission before quality

Require all of the following, with failed attempts preserved:

1. Fresh constructor/first-artifact tensor equality, no inherited learned data,
   fresh optimizer and zero source updates; TRAIN-only fresh fitted statistics.
2. Actual CUDA inputs, parameters, loss, gradients and movement across every
   learned block; literal prior and fitted statistics stay exact.
3. Values or masks after each of128 cutoffs cannot affect the embedding at that
   cutoff, before and after learning. Include2350 suffix interventions,
   nonfinite future poison,30 internal temporal cases and past-positive controls.
4. Strict typed inference/TRAIN distinction, parameter/moment/counter integrity,
   refusal cases and exact typed continuation. Repeat frozen causal proof on
   both initial and final saved artifacts used for scoring.
5. Hash-verified owned source/compiler closure, configuration, inputs, binaries
   and scored checkpoints. No old encoder execution or initializer dependency.

## Fixed-head evaluation and interpretation

Extract new saved native embeddings for BOTH initial0 and final2048. Market:
112 TRAIN/32 VALIDATION,144 rows per checkpoint. TEMPO-5: three saved masters
941101/941102/941103, each512 TRAIN/256 VALIDATION examples from256/128 source
groups, with intact and extra30%-deleted validation. Dataset **TEMPO-5** is
slow/fast nine-feature lag-sign timing at designed **complexity5/5**. There are
zero synthetic encoder updates. Evaluate both declared input policies:
fresh cohort-TRAIN scaler (fit once per cohort and shared across checkpoints)
and the newly fitted v3 market scaler. No old scaler witness is imported.

Keep ridge penalty1 and the neural16-tanh/Adam0.01/100-update recipes unchanged.
Three repetitions2701/2802/2903 use the established dimension36 seeds
13088411117524848790,10018485049408248344,9083751155097190455.
Four-output market heads fit148/660 parameters (linear/neural); binary timing
heads fit74/626. Head normalization and fitting use their permitted TRAIN only.

Primary tables show raw recent128/full256 encoder-free market controls and
CPB-v3.market-01.init/CPB-v3.market-01, one task/panel per table. Reuse only
byte-pinned saved raw-control results. All fresh encoder initial/final heads are
new fits; no v1 quality reuse or pass/fail comparisons. Timing tables show the
two v3 stages. Report head variation and cohort spread without treating head
fits as independent encoder runs.

No historical quality floor or9/9 target applies. Establishing this first clean
baseline requires the trust admission and complete, finite fixed-head results
with100% declared coverage (32/32 market VALIDATION;256/256 in each timing panel).
Otherwise record it as incomplete/failed with all evidence retained. Quality
itself is descriptive: report initial→final changes, raw-control comparisons and
all regressions without an overall accuracy or automatic winner. Passing trust
admission does not establish useful task performance or application readiness.

## Execution and preservation

Only the existing GPU-enabled `cuwacunu` container may execute project code.
Use exclusive hash-verified source/build/run directories inside it. Preserve
every old checkpoint/result/card and all v1/v2 implementation bytes. The new
reusable version belongs in research; dev keeps its sole v1 selection while
this experiment is evaluated. No commit/push or automatic dev replacement.

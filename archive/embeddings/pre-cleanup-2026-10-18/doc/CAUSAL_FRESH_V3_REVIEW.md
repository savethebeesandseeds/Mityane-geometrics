<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# CPB-v3.market-01: first clean causal training baseline

CPB-v3 is the first completed from-scratch causal training run in this family.
CPB-v1 remains a pretrained historical reference and the current dev selection.
Its scores impose no targets or retention floors on clean research. V2 is a
preserved inherited-weight experiment. The earlier frozen-v1-parent proposal
is withdrawn. Read the [provenance correction](CAUSAL_TRAINING_PROVENANCE_CORRECTION_2026-10-15.md).

## What establishes the clean baseline

All293672 learned values start from the seed943001 constructor, including the
architecture's intentional zero initializers. The3888-value analytic prior is
rebuilt directly. No pretrained parent, decoder, teacher, optimizer or old
normalization state initializes the run. Main initial tensor state matches an
independently constructed fresh model before any update; source_updates=0.

Input and270-coordinate bank statistics are fitted freshly from4096 permitted
TRAIN recent128 raw windows, without encoder calls or validation statistics.
An independent saved-data audit recomputes those statistics from CACHE88 records.
Legacy `rpb-` strings in the copied scaler serialization identify its format;
the actual statistics are newly fitted and have new recorded identities.
All75 learned tensors receive gradients and move during the single2048/B8 CUDA
trajectory. The literal prior and fitted statistics stay exact. The actual
trajectory resumes from512 with exact model/optimizer/counter identity and no
repeated updates.

Engineering and the two actual saved checkpoints pass128-cutoff proofs,
2350 future-value/mask interventions and30 internal temporal cases per stage.
The saved initial0/final2048 artifacts each bind their proof by exact hash.
Typed refusal tests, source/compiler closure and independent arithmetic audits
are retained. This is bounded correctness evidence for the declared contract,
not a proof of forecasting ability or usefulness on every task.

## What the measured quality says

The [full per-task report](CAUSAL_FRESH_V3_RESULTS.md) compares new initial and
final native36 embeddings with the unchanged fixed heads. Only encoder-free raw
control scores are reused. No pretrained encoder supplies a quality target.

On MKT-HIST-2 complexity4/5, final volatility Linear/Neural R² is
0.964673433811061/0.4630519227215611. Activity is
0.8628403110608782/0.541426055582653. Trend improves from initialization but the
neural mean remains negative at−0.2539790011410479, with all three fits negative;
linear trend is0.2631329634696965. Relation is weak in both heads:
−0.00700444683432222/−0.08556567147059875. Activity neural and relation linear
decline from initialization. These weaknesses remain part of the baseline.

All eight TEMPO-5 complexity5/5 timing measures improve from initialization.
With a fresh synthetic-TRAIN scaler, final intact Linear/Neural accuracy is
96.484375%/95.703125%; deleted is91.14583333333334%/91.75347222222222%.
With the fresh market scaler, intact is99.34895833333334%/98.17708333333334%;
deleted is94.921875%/94.27083333333334%. All coverage is100%. Initial timing is
already strong, so retain the architecture/prior contribution separately from
the measured training gains. Three head fits are not three encoder runs.

Fixed-query standardized MAE improves TRAIN0.664167856625107→0.259045990434276
and known VALIDATION0.958064934323618→0.376414230576159. Gaussian NLL improves
TRAIN0.423815353462879→−0.380178079833758 but worsens VALIDATION
0.795376511159209→1.52179657791809. Better mean reconstruction does not imply
better variance calibration. Synchronized training compute is144.341116249s,
with fresh calibration2.991649873s reported separately; proof/export/head/I/O
work is excluded from training compute.

This establishes a clean measured reference with mixed task quality. It does
not declare an overall winner or automatically replace the dev application.
The market probes are112 TRAIN/32 known chronological VALIDATION windows;
TEMPO-5 uses three saved512/256-example cohorts and no synthetic encoder updates.
January29–31, new market periods and forecasting remain unmeasured.

## Next direction

Keep CPB-v3 as the clean training baseline. Diagnose the saved TRAIN artifacts
for weak cross-channel relation retention and the Gaussian variance objective,
using the existing embeddings/query witnesses and declared block layout. Keep
validation out of that diagnosis and fit no new heads to rescue this run.
Use the result to declare one controlled causal objective/representation change
with a prospective initial/final comparison. Do not freeze or import v1's parent.
No later experiment is running or scheduled.

## Where the usable files are

The complete owned implementation is
`code/encoders/causal_patch_bottleneck_mae/versions/v3/`; the single public entry
is its `include/embedding/encoders/causal_patch_bottleneck_mae/versions/v3.h`.
`versions/CPB-v3.json` preserves the declaration and the additive measured
instance record supplies the completed result. The source capsule, exact
binaries and proof fixtures are in
`output/engineering/causal-fresh-v3/attempt-001/`.

`output/runs/causal-fresh-v3/run-001/` contains the five main checkpoints,
fresh calibration, trace, query witnesses and saved embeddings/heads. The stable
clean-baseline copy is `output/baselines/CPB-v3.market-01/inference.pt`,1290625
bytes, SHA256
`50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e`.
Both original and copy remain local/ignored. All older artifacts remain intact.
The [finalization receipt](causal_fresh_v3/finalization.json) maps original
container paths to the local mirrors and pins the evidence. New work remains
local and uncommitted; dev's v1 source/checkpoint/runtime selection are unchanged.

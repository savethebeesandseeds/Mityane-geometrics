# Configurable RPB-v20 capacity screen v1

Prospective development VALIDATION experiment. Freeze this card and complete
source/binary/config/environment receipts after engineering admission and before
generating quality data or constructing quality models. Retain every outcome.
No TEST, market data, head tuning, checkpoint selection or automatic promotion.
Historical recipes, source captures, checkpoints and results stay immutable.

## Challenge and information admission

**TEMPO-5**, task `slow_lag_sign`, designed complexity **5/5**. Determine whether
channel1's slow component leads or lags channel0. Three channels, history128,
nine features, in original candle units. Complexity is an ordinal design label,
not an accuracy-derived estimate. This is a structured capacity challenge, not
a market simulation or proof of market usefulness.

Each independent source produces two opposite-label variants. Slow period is
uniform over integers32..48; fast period over integers8..14. Phases are uniform
[0,2pi), component amplitudes independently uniform[.8,1.2). Slow lag magnitude
is uniform[1.5,2.5) candles. Fast lag magnitude is uniform[.3,.7), with an
independent fair sign. Channel2 has independently drawn periods/phases and the
same selected component amplitudes. Only channel1's slow lag sign changes
between paired variants; nuisance, random noise and masks are pair-shared.

Features0..2 contain slow+.25fast; features3..5 contain fast+.25slow.
Features6..8 are channel-independent smooth distractors
`a*(sin(2pi*t/Pn+phi)+.6*sin(2pi*t/Qn+psi))`, with a uniform[.6,1),
Pn uniform[18,26), Qn uniform[70,90), and independent phases.
Each feature has a positive gain uniform[.5,1.5)*(1+.05*(f%3)) and
offset uniform[-.75,.75). Gaussian observation noise has standard deviation.005.
Natural masks delete10% of coordinates independently, plus a shared five-candle
gap per channel, beginning at floor((H-4)*U). Missing storage is exactly zero.
The extra-deleted validation view deletes30% of remaining coordinates using a
separate pair-shared stream, preserving rows, retained values, labels and IDs.

Generate exactly TRAIN512/256 source pairs and VALIDATION256/128 source pairs
for each of masters **941101,941102,941103**, once. Assign sources to roles
before constructing variants. Emit pairs in ascending original-source order and
randomize row order within each pair. Opposite variants remain adjacent to make
paired-source auditing explicit.
No latent waveform, period, delay, nuisance parameter or information fit enters
model inputs or targets. Legal archives contain only observed values, feature
masks, scoring-only labels and source identities. All three profiles reuse the
same nine role archives. Engineering master **941000** is excluded from quality.

Before any quality model, an observed-only analytic witness must correctly
classify **every** row of all nine role archives, with100% support. It jointly
fits all119 slow/fast period pairs using a constant and sine/cosine coefficients
for observed features0..2 of channels0/1. It does not read labels or latent
parameters while inferring. Labels are used only to audit its decisions.
Its fixed support rules:24 observed ticks per feature, two supported features
per channel,40 union ticks and48 tick span per channel; scaled pivot>=1e-8,
energy>=1e-12, each component fraction>=.02, phase coherence>=.95,
normalized residual<=.0025 and absolute slow phase in[.12,.60]. Retain selected
coefficients, residuals, margins, predictions and all unsupported rows.
An admission failure preserves artifacts and stops this recipe; no redraw,
seed extension or threshold relaxation. Engineering checks also cover history256
using the same absolute-period law. Challenge channel/feature roles are fixed;
the encoder API remains dimension-configurable independently.

## Closed model and training matrix

| Profile | Instance tag | Channels | History | Features | Patch | Internal width | Embedding size |
|---|---|---:|---:|---:|---:|---:|---:|
| A | RPB-v20.alt-01 |3|128|9|8|64|32|
| B | RPB-v20.alt-02 |3|128|9|8|128|32|
| C | RPB-v20.alt-03 |3|128|9|8|64|64|

A is the configurable baseline. B increases internal encoder width while
holding served embedding size fixed. C increases served embedding size while
holding internal width fixed. Tags identify new instances of the v20 design;
no historical encoder is renamed or overwritten.

Read the existing model-only `config/configurable_spectral.conf` through the
core settings reader; apply only the declared W/D profile overrides and tag.
Archive the fully resolved settings. Three layers, four heads, FF256 for every
profile, decoder128, early mixer1, global mode2/source0, dropout0,
no difference input, sampling interval1. Fixed spectral coordinates12 at C3;
learned coordinates D-12. Preserve the native global bottleneck as the sole
decoder input. The fixed bank uses original-time visible intersections and
the unchanged v20 frequency/normalization rules.

Exactly **nine** predetermined trajectories: profiles A,B,C, each masters
941101..103. Each reaches512 completed AdamW updates, batch8, learning rate.001,
weight decay.0001, gradient clipping1, masked hierarchical Huber delta1.
Patch masking ratio.25 plus the existing coordinate15 context deletion recipe:
erase15% of original visible context, restoring earliest coordinates only as
needed to retain two context patches per eligible channel. Preserve original
targets/eligibility. Do not use the legacy D32-only training wrapper.
Train-only observation scaler, scale floor1e-6, layer norm epsilon1e-5.
Retain counter-seeded row/mask/deletion streams, attempted/completed updates,
skips, target/erasure counts, losses and gradient norms. A nonfinite state or
unsupported training batch aborts and preserves the failed trajectory.
512*8/512=8 equivalent row presentations; this is not eight full shuffled epochs.

All encoder inputs, parameters, buffers, forwards, losses, gradients, optimizer
updates, checkpoint restoration and reconstruction queries run on CUDA.
Actual two-update CUDA admission must pass for all profiles before quality.
CPU may generate data, fit preprocessing/fixed heads, and audit saved arrays.
There is no CPU encoder replay. Save immutable typed checkpoints and native
exports at0 and512. Verify exact fixed12 equality across those points on every
view, learned-coordinate movement and bit-exact typed checkpoint reload.
Only initial and final checkpoints are evaluated; no best checkpoint search.

## Independent fixed evaluation and costs

Quality columns are exactly `Method | Size | Linear head % | Neural head % |
Coverage %`. Use only the native encoder export, without post-encoder PCA.
Baselines are **Raw data — no encoder** (3456 standardized observed values
plus3456 mask coordinates =6912), and **PCA only — no encoder** at32 and64,
each fitted on TRAIN only. Fit raw preprocessing/PCA once per master and share
across all profiles and scored views. Baselines are not encoder training runs.

Reuse shared fixed-feature readouts: ridge penalty1; neural16 tanh hidden units,
Adam.01 for100 updates; repetitions **2701,2802,2903**, paired same-width seed
policy. Fit each method/point/master/repetition once on TRAIN and reuse exactly
that fitted head for intact and deleted validation. No validation tuning.
Binary heads have D32:66/562 parameters, D64:130/1074; the raw heads are larger
(13826/110642), which must be disclosed. Accuracy is on supported scored rows;
coverage is supported rows/all scored rows. Preserve individual head fits,
normalizers, predictions, populations, cohort spread, paired comparisons and
source-group uncertainty. Do not treat head repetitions as independent sources.
Initial-point scores distinguish the fixed architectural prior from learning.

Training columns are exactly `Encoder | Updates | Train error ↓ | Validation
error ↓ | GPU training seconds`. Error is standardized hidden-query MAE on
fixed leave-one-patch-out TRAIN/intact VALIDATION queries, averaging supported
features/times within a channel, eligible channels within a row, then rows.
Keep all16 patch contexts fixed across points/profiles. Save compact arrays:
each held-out coordinate's prediction/target once, original/target masks,
per-trial channel eligibility and per-row/channel statistics. Context is
recoverable from original masks and held-out patch index; compact storage does
not alter the query. Retain optimization Huber trace separately from query MAE.
Report synchronized training-loop seconds separately from native extraction,
queries, checkpoint I/O and CPU head costs; retain CUDA allocator peak reserved
and allocated bytes and parameter counts. No quality threshold chooses a winner.
Finish all nine trajectories even if classification is weak, unless an integrity
or runtime failure requires stopping. Report failures plainly.

## Reproducibility and publication

Freeze source paths/hashes, prospective card bytes, model-settings bytes,
compiler/LibTorch/CUDA/cuDNN/device proof, binary hashes and engineering receipts.
Bind all nine input hashes before quality model construction and check sources,
inputs and binaries before/after execution. Use an absent uniquely named output
capsule; refuse to overwrite artifacts. Mark completion only after all runs,
fixed heads and saved-only arithmetic/provenance audits pass. Audit saved CPU
arrays without fitting new heads or forwarding an encoder.

Publish TEMPO-5 intact and extra-deleted views separately, with adjacent legends
(short recipe, complexity5/5, scored view). Give model descriptions outside the
quality table and counts/budgets/device/independent runs before results. Retain
per-cohort initial/final scores and native-size/cost comparisons in a durable
report. Add results and notes without rewriting historical cards or evidence.

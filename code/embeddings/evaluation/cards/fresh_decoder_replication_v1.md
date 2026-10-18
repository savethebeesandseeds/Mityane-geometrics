# Fresh decoder replication v1

Prospective development comparison; frozen before any quality data generation.
Protocol: `fresh-decoder-replication-v1`. No new embedding design/tag. RPB-v4
remains the active reference; RPB-v7 is the working candidate.

## Question and complete recipe

Does v7's lighter context deletion preserve its classification benefit on new
training sources, and does decoder adaptation resolve reconstruction at the
same decoder budget given to v4? Report recovery against v4 before calibration
separately from the fair comparison against v4 after calibration.

Use exactly five new masters: **9109, 10210, 11311, 12412, 13513**. Generate each
`lag_sign` TRAIN/VALIDATION cohort once with the existing development-only
generator: 128 TRAIN source pairs (256 examples), 64 VALIDATION source pairs
(128 examples), C3/H32/F3, 10% natural missingness, CPU float64 observations.
Prefix every source ID with `fresh-decoder-replication-v1/` before fitting,
corruption or saving. Fit namespace is `fresh-decoder-replication-v1/lag_sign`.
Save only legal observed values, masks, scoring labels and source IDs. Zero
hidden storage and replace the generator's clean placeholder with legal
observations. Never call a TEST generator or open historical quality payloads.

For each master, create one v4 and one v7 trainer, with exact complete paired
initialization and TRAIN scaler, original row/patch/Torch streams. Architecture:
patch8, encoder64, three blocks, four heads, feedforward256, mixer1, global
mode2, native32, decoder128, dropout0; 225,805 parameters. Train each once on
CUDA for 512 successful/unskipped updates, batch8, AdamW learning rate .001,
weight decay .0001, gradient clip1, one Torch CPU thread. V4 uses the ordinary
view. V7 uses its existing coordinate-deletion .15 policy and repair semantics.
Keep original reconstruction targets, eligibility, Huber loss and counter
streams. Do not search seeds, rates, loss coefficients, capacity or budgets.

Save ordinary point0 and point512 checkpoints and their audit/scaler/legal
TRAIN assets. The explicit fresh v4/v7 decoder admission loads point512. Freeze
all 214,277 non-decoder parameters, buffers and TRAIN scaler. Train only the
existing six decoder tensors (11,528 parameters) for 128 successful updates on
CUDA, using a fresh AdamW with the same rate/decay/clip/batch and original
absolute counters 512..639. Context deletion is absent during decoder training;
original O/A/Q reconstruction remains unchanged. Save typed
`rpb_fresh_frozen_decoder_calibration_v1` artifacts at decoder0 and decoder128.
Never serve these as ordinary resumable checkpoints.

## Shared readouts and views

Fit controls once per cohort and fixed classifier weights once per method and
repetition on TRAIN only. Methods: raw576 (288 TRAIN-scaled legal values plus
288 visibility flags), **PCA only — no encoder**32 (raw TRAIN normalizer then
TRAIN PCA), mask metadata288, paired point0 untrained native32, v4@512 native32,
v7@512 native32. PCA never follows an encoder. Raw scaler uses unchanged
ObservationScaler semantics. Native tensors retain their float32 CUDA export.

For each of six methods use repetitions 2701/2802/2903; actual neural seed is
`stream_seed(repetition, feature_width)`. Linear: ridge penalty1. Neural:
tanh16, Adam .01, 100 updates. Both use unchanged TRAIN FeatureNormalizer.
Raw/native/mask/untrained preserve the existing outer TRAIN FeatureNormalizer
followed by each probe's own TRAIN FeatureNormalizer. PCA-only is already
prepared by raw outer-normalizer then PCA; its helper outer stage is identity,
followed by each probe's normalizer. There are no held-out normalization fits.
This is 90 paired pipelines /180 individual head fits, not 90 encoder runs.
Save fitted assets and TRAIN/VALIDATION predictions. Repetitions are not
independent encoder retrainings. Point0 control is shared because full paired
initialization/scaler and same-CUDA native features are exact.

Score intact VALIDATION and one additional .30 coordinate-deletion VALIDATION
view using the same TRAIN-fitted heads. Namespace:
`fresh-decoder-replication-v1/validation-coordinate-dropout`; seed:
`stream_seed(master, 0x6672642d76616c30ULL)`. The existing grouped generator
shares erasure within source pairs, preserves retained values, zeroes hidden
storage and applies no repair. All methods use that one saved view. No new
fits on deletion or other held-out data.

Extract needed encoder/query inference on CUDA only. CPU fixed heads, raw PCA
and saved-tensor arithmetic are allowed. Decoder128 classification reuses
decoder0 scores/heads only after exact same-CUDA native/support invariants on
TRAIN and both VALIDATION views; do not fit or score a second copy merely
because decoder weights changed. Save this reuse binding. No CPU encoder
training or forward, TEST/stress, checkpoint selection or automatic promotion.

## Predeclared interpretation

Mandatory reliability: source/card/admission binding before generation; actual
CUDA parameters/inputs/loss/gradients/weight change; full100% native coverage;
all five paired runs; complete equal budgets; exact initialized named state,
scaler, frozen non-decoder state and same-CUDA native exports; independent
saved arithmetic and source-role audit. Failures remain additive evidence.

Classification guards are positive equal-master mean v7-minus-v4 linear
accuracy in both views and no lower worst-master linear accuracy in either
view. Neural heads and controls are secondary and cannot rescue these guards.
Recovery guards: mean post-calibration v7 TRAIN and VALIDATION original-query
standardized MAE no greater than v4 PRE-calibration512, separately. Equal-budget
guards: mean post-calibration v7 TRAIN and VALIDATION MAE no greater than v4
POST-calibration128, separately. The joint development guard comprises the
four classification and two equal-budget reconstruction guards plus mandatory
reliability. Recovery success alone does not establish equal-cost superiority.
Always report every master and both reconstruction stages regardless of outcome.

For head/view/master conditional source-group intervals use the existing
whole-pair 95% percentile bootstrap, 1,000 draws; pair native v7/v4 effects.
Across-master effects use 10,000 bootstrap draws of the five paired effects,
fixed seed `0x6672642d626f6f74ULL`, reported as descriptive uncertainty. An
interval crossing zero makes the direction imprecise even if a point guard
passes. These synthetic development results are not consumer confirmation.

Original-query summary and saved double arithmetic tolerance: 2e-11; saved
float32 per-update Huber: absolute2e-6 + relative2e-5. Exact state/feature
invariants use equal bytes within the same CUDA backend; no CPU/CUDA encoder
parity requirement. Preserve finite losses and optimizer groups/traces.

## Evidence and cost

Capture the explicit sorted source set, card, metadata-only launch plan and
actual CUDA admission before quality generation. Use exclusive new output
capsules and an exact inventory; no prior quality input roles. Reports bind
the ten parent trajectories, twenty decoder stages, five control sets and all
90 retained readout pipelines. Keep encoder training, decoder training,
feature/query inference, CPU head fitting and independent audit times separate.
GPU training counters mean updates, not guaranteed epochs. Each encoder's
4,096 sampled row presentations equals 16 TRAIN-size passes; decoder calibration
adds1,024 presentations/four passes while the encoder remains frozen.

Use the standard Method/Size/Linear head/Neural head/Coverage table and separate
Encoder/Updates/TRAIN error/VALIDATION error/GPU seconds table. Put short tag
descriptions beside tables. Clearly label fresh versus reused classification,
intact versus additional-deletion view, pre versus post decoder errors, and
original encoder versus extra decoder costs. Preserve historical cards/reports.

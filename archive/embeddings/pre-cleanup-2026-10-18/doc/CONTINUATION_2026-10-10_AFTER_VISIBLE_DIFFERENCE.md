# Continuation after visible adjacent differences

Date: 10 October 2026. Completed protocol: `visible-difference-v1`.

The [RPB-v13 report](../code/encoders/raw_patch_bottleneck_mae/VISIBLE_DIFFERENCE_DIAGNOSTIC.md)
and [durable JSON](results/visible_difference_v1.json) preserve five newly trained
CUDA instances on the exact saved TEMPO-3 cohorts. RPB-v13 adds a zero-initialized
projection of currently visible adjacent differences to early mixing. It retains
native32 and the original decoder/objective, and adds 3,072 parameters for 228,877
total. The original 225,805 common initial values, buffers, scaler and all 15 initial
native exports match the saved early control. Missing or erased coordinates never
enter differences, and gaps are not bridged.

Each master has 256 TRAIN rows/128 independent source pairs and 128 VALIDATION
rows/64 separate pairs. Five new trajectories completed 512 CUDA updates each at
batch 8 and .15 training context deletion, without skips: 20,480 sampled rows total.
The fixed heads are Ridge penalty 1 and tanh 16/Adam .01/100 updates, with three
repetitions 2701/2802/2903. Each candidate head fits TRAIN once and scores the two
validation views. All seven parent baseline/control rows and fitted heads were
reused. No old encoder training/forwards, control head refits or new PCA fits.

Dataset: **TEMPO-3** · timing · **designed complexity 4/5** · variable delays,
positive gains, offsets and 3-tick channel gaps · intact VALIDATION.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.25 | 50.73 | 100.00 |
| PCA only — no encoder | 32 | 48.44 | 51.15 | 100.00 |
| RPB-v7.alt-05 | 32 | 55.47 | 54.64 | 100.00 |
| RPB-v10.alt-05 | 32 | 52.81 | 54.90 | 100.00 |
| RPB-v13 | 32 | 53.28 | 52.03 | 100.00 |

RPB-v7.alt-05 is the saved late mixer; RPB-v10.alt-05 is the saved early mixer.
RPB-v13 is the new early mixer with visible adjacent differences. Means include
every cohort and every head repetition. Intact linear accuracy improves in two
cohorts, falls in two and ties in one. The slight mean increase is not a reliable
improvement; neural accuracy falls. All cohort scores and conditional intervals
are in the full report. These intervals hold models/heads fixed and do not measure
variation across new training runs.

Dataset: **TEMPO-3** · timing · **designed complexity 4/5** · same recipe ·
VALIDATION with extra 30% coordinate deletion.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.53 | 50.78 | 100.00 |
| PCA only — no encoder | 32 | 48.75 | 50.68 | 100.00 |
| RPB-v7.alt-05 | 32 | 54.22 | 54.01 | 100.00 |
| RPB-v10.alt-05 | 32 | 53.75 | 54.32 | 100.00 |
| RPB-v13 | 32 | 52.81 | 50.52 | 100.00 |

Dataset: **TEMPO-3** · timing · **designed complexity 4/5** · same recipe ·
original fixed TRAIN/VALIDATION reconstruction queries.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-05 | 512 | 0.649343 | 0.689378 | 15.88 |
| RPB-v10.alt-05 | 512 | 0.644282 | 0.682958 | 16.79 |
| RPB-v13 | 512 | 0.622320 | 0.664399 | 19.53 |

Errors are original-query standardized MAE; lower is better. Timers are means
of synchronized training-loop wall time, including scalar/counter traces.
Checkpoint writing, live CPU state capture, extraction and classifier fitting
have separate timers. Times are descriptive, not a repeated hardware benchmark.
Predicting hidden values improves on average, but timing classification does not
improve consistently. Input parameterization and capacity change together; this
does not isolate either effect or prove absence of timing information.

## Next experiment: independent training support

The [saved-TRAIN diagnosis](../code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_TRAIN_DIAGNOSTIC.md)
found early-v10 neural accuracy 82.06% on TRAIN versus 54.90% on intact VALIDATION.
Raw neural accuracy was 99.69%/50.73%. New RPB-v13 is 78.28%/52.03%. The substantial
fit/generalization gap motivates testing more independent sources. It does not
identify whether the encoder, fixed head access or optimization is the unique
cause. The legal-observation analytic rule and saved query targets still provide
strong task-information checks.

Freeze one bounded 2×2 comparison before implementation or measurement:

1. Keep TEMPO-3's exact signal law, noise, natural missingness, gaps and complexity
   label. Declare five fresh masters and construct 512 TRAIN source pairs plus 64
   source-disjoint VALIDATION pairs per master. Freeze source identities and both
   validation views before fitting. No TEST or stress evaluation.
2. Choose a nested 128-pair TRAIN subset using a declared source-only hash order
   independent of labels and measurements. Keep complete pairs. Use the exact same
   fresh validation rows for the small and large TRAIN conditions. The generator
   shuffles assignments over the total source count; merely reusing old seeds with
   a larger count would not preserve old validation.
3. Compare the early architecture and visible-difference architecture under each
   TRAIN size. Use a common scaler fitted only on the nested small TRAIN subset,
   so original-query reconstruction units remain comparable. Bind that explicit
   preprocessing contract and separate instance identities; apply the versioning
   rules if implementing it changes a training recipe. Never use VAL to fit it.
4. Keep native32, .15 context deletion, batch 8, 512 completed encoder updates,
   AdamW/Huber1, original query recipe and fixed classifier recipes. Paired common
   initial state and parameter counts must be checked before AdamW. Retain every
   arm/cohort/repetition; no selected checkpoint, head seed or subgroup.
5. Fit fresh raw/PCA-only baselines and heads on each permitted TRAIN condition,
   reusing them across architecture arms with identical data support. Fresh data
   require fresh features, scalers/bindings and matched controls. Old scores are
   historical context, not paired measurements on the new cohorts. Add a fresh
   late-v7 arm only if a late-versus-early ranking is a declared question.

At 512×8, each encoder gets 4,096 sampled rows: 16 equivalent presentations with 256
TRAIN rows, versus 4 with 1,024 rows. These are replacement draws, not full epochs.
The larger set tests diversity at an unchanged encoder-draw budget. Full-batch
heads still make 100 neural updates but process four times as many rows; raw PCA,
normalizers, extraction and head work also change cost. Report these separately
and do not claim equal total compute. A failure at 512 would not settle the effect
of more data under a larger optimization budget.

This is prospective: no new card, source implementation or measured group exists
yet. Freeze the card, exact data/scaler/identity contracts and independent reader;
test synthetic schemas and actual CUDA behavior before quality. Do not tune the
heads, add post-encoder PCA or reopen stopped gain/pooled/view-agreement recipes.
Gain/offset normalization is a distinct future hypothesis, not part of this test.

## Preserved evidence and execution

The completed local capsule is
`output/runs/rpb-visible-difference/visible-difference-wnUUCa`, with723 files,
210,928,579 bytes and inventory SHA
`384ed378bd0cd563e2f0e38f65c35a20d4e8db746dac4830c86cb8f29129f2fb`.
Its producer source identity is
`14c99891d6a4a9b1831116431a31e3eef9b4b352565db63f1df6889828332bdb`.
Actual CUDA admission is preserved in `admission/admission-P7tDKh`.

The passed saved-arithmetic audit is
`output/runs/rpb-visible-difference/audit-tools/run-wnUUCa-v2/validation.json`,
SHA`b4849d9aca2530523b12093706f8f52229bc512cf9947447c68870d2d31ee070`:
17,152,339 checks/280 CPU witness archives/170.61 seconds, with zero model
execution, fitting or CUDA checkpoint-body decoding. Failed v1 remains preserved.
Its sole mismatch was a basename-versus-full-path query report check. The
[additive repaired reader](../code/evaluation/protocols/visible_difference_v1/reader-v2/validate_visible_difference.py)
SHA`74fde8c189e22bc9e590a0f5c7059fadface9540500f8861e035b18d11c164c8`
proves exact reversal to the original sealed reader apart from that call and
explicit repair provenance. Numerical functions, tolerances, card, weights and
all capsule bytes are unchanged. Its synthetic tests passed 76,292 checks/69
negatives with zero real archive reads.

Original saved RPB-v7, formal RPB-v4, TEMPO-1 RPB-v10.alt-03, all earlier groups,
cards and reports remain unchanged. No automatic promotion or destructive cleanup.
Develop/build/test/run through the existing managed `cuwacunu_embedding` container
and named task sessions; admit actual CUDA parameters, inputs, loss, gradients and
weight changes before new training. CPU work is limited to fixed heads/PCA,
metadata and saved arithmetic. Never repeat an encoder on CPU for validation.

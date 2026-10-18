# Continuation after the TEMPO-3 comparison

Date: 9 October 2026. Completed protocol: `structured-hard-timing-comparison-v2`.

The structured harder timing comparison completed once on five fresh paired
masters 75272/76373/77474/78575/79676. Ten CUDA trajectories each made512 updates
at batch 8, without skips: 40,960 sampled rows, 16 equivalent TRAIN presentations.
Each master has 256 TRAIN rows from 128 independent sources and 128 VALIDATION
rows from 64 separate sources. Native32 and all 225,805 parameters remain unchanged.
The two architectures share exact initial named state/scalers before training.

The [full report](../code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_DIAGNOSTIC.md)
and [durable JSON](results/structured_hard_timing_comparison_v2.json) preserve
all seven methods, five cohorts, three head repetitions, full training traces,
conditional intervals and paired records. Late RPB-v7.alt-05 mixes after the
temporal pass; early RPB-v10.alt-05 mixes before it and keeps an independent
unmixed local pass. Equal parameter counts do not imply equal compute.

Dataset: **TEMPO-3** · timing · **designed complexity 4/5** · variable delay,
positive gains, offsets and 3-tick channel gaps · intact VALIDATION.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder | 576 | 51.25 | 50.73 | 100.00 |
| Mask metadata — no encoder | 288 | 50.00 | 50.00 | 100.00 |
| PCA only — no encoder | 32 | 48.44 | 51.15 | 100.00 |
| Untrained late mixer | 32 | 50.78 | 51.56 | 100.00 |
| Untrained early mixer | 32 | 50.47 | 51.88 | 100.00 |
| RPB-v7.alt-05 | 32 | 55.47 | 54.64 | 100.00 |
| RPB-v10.alt-05 | 32 | 52.81 | 54.90 | 100.00 |

Dataset: **TEMPO-3** · timing · **designed complexity 4/5** · variable delay,
positive gains, offsets and 3-tick channel gaps · extra 30% deletion VALIDATION.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder | 576 | 49.53 | 50.78 | 100.00 |
| Mask metadata — no encoder | 288 | 50.00 | 50.00 | 100.00 |
| PCA only — no encoder | 32 | 48.75 | 50.68 | 100.00 |
| Untrained late mixer | 32 | 48.91 | 50.05 | 100.00 |
| Untrained early mixer | 32 | 49.53 | 50.57 | 100.00 |
| RPB-v7.alt-05 | 32 | 54.22 | 54.01 | 100.00 |
| RPB-v10.alt-05 | 32 | 53.75 | 54.32 | 100.00 |

Ridge penalty 1 and tanh16/Adam .01/100 updates remain fixed, with repetitions
2701/2802/2903. Each head fits TRAIN once and scores both declared views.
Means average all three repetitions, then equally weight all five masters.
Native outputs receive no PCA. Raw/PCA share one TRAIN outer map per cohort.
The report includes every cohort range and conditional source-group interval;
those intervals hold saved models/heads fixed and are not retraining uncertainty.

Dataset: **TEMPO-3** · timing · **designed complexity 4/5** · same recipe ·
all five paired cohorts, linear head means across the three repetitions.

| Master | Late intact % | Early intact % | Late deleted % | Early deleted % |
| --- | --- | --- | --- | --- |
| 75272 | 49.22 | 51.56 | 49.22 | 53.91 |
| 76373 | 51.56 | 48.44 | 50.78 | 54.69 |
| 77474 | 59.38 | 50.00 | 55.47 | 53.91 |
| 78575 | 55.47 | 54.69 | 54.69 | 52.34 |
| 79676 | 61.72 | 59.38 | 60.94 | 53.91 |

Early mixing improves one intact and two deleted linear pairs; the other pairs
worsen. Late/early worst linear scores are 49.22%/48.44% intact and49.22%/52.34%
deleted. Neural results differ: early means are slightly higher in both views.
Classification remains weak for both trained architectures. Neither group is
promoted or selected, and original saved RPB-v7 plus the formal RPB-v4 reference
remain unchanged. Gainv11 and wider pooledv12 remain stopped.

Dataset: **TEMPO-3** · timing · **designed complexity 4/5** · same recipe ·
original fixed reconstruction queries, equal means across all five retrainings.

| Encoder | Updates | Train error | Validation error | GPU training seconds |
| --- | --- | --- | --- | --- |
| RPB-v7.alt-05 | 512 | 0.649343 | 0.689378 | 15.88 |
| RPB-v10.alt-05 | 512 | 0.644282 | 0.682958 | 16.79 |

Error is standardized original-query MAE. GPU loop seconds are synchronized wall
time including CPU trace capture, excluding checkpoint writing. Ordinary joint
decoder training occurred during these 512 updates; extra decoder calibration
updates were zero. Query/native extraction and checkpoint/binding I/O are
separate mixed transfer/verification/I/O scopes, detailed in the report.

## What this resolves and leaves open

TEMPO-3's4/5 level is an ordinal designed challenge assigned before results,
not a mathematical or accuracy-derived measurement. The generator remains
`structured-hard-timing-v1`; the v2 comparison changes the observed-only
analytic support rule, not the signal recipe or information gates. The first
same-feature rule failed deleted coverage on separate engineering seeds 80383/
81484, with every supported row correct; no encoders or heads were fitted.
The cross-feature rule passed on new 82585/83686 seeds, at 100% conditional
accuracy and 99.8046875% mean deleted coverage. Both attempts remain in the
[information diagnostic](STRUCTURED_HARD_TIMING_INFORMATION_DIAGNOSTIC.md).

In the actual five quality cohorts, that legal-observation analytic rule solves
all TRAIN/intact/deleted rows with 100% coverage and no fitted heads. Thus the
legal data has a demonstrated accessible timing cue. Raw/PCA fixed heads also
score near chance, however: classification alone does not identify whether the
encoder, served feature geometry, fixed head or optimization limits access to
that cue. These are fresh harder-data models, not damaged versions of the
original saved network or paired estimates of a cross-dataset decline.

## Proposed next action, not executed

Before another architecture or training change, freeze a prospective saved-TRAIN
diagnostic card and its closed input matrix. Keep TEMPO-3 and existing heads
fixed. Across all five late/early pairs, compare saved TRAIN fit accuracy,
margins and source-pair separation with already audited held-out metadata.
Use saved features, fitted weights and predictions without any head/PCA refit.

Then test relative timing in the saved original-query reconstructions. Their
shape is [4,B,3,32,3]: four patch banks have different masked-context embeddings.
Restrict each analytic triplet to the same bank's 8-tick patch and average only
valid centres across the four banks. Apply the identical restriction to saved
targets as the information baseline; invert the frozen TRAIN scaler only on
legal cells. Preserve invalid rows, source-pair denominators and every cohort.
These outputs are not one full-original-context reconstruction.

Poor reconstructed timing against a strong saved-target baseline would identify
a reconstruction-path limitation, not establish encoder information loss.
Strong reconstructed timing alongside weak fixed-head scores would show
multi-context encoder/decoder accessibility, not sufficiency of the served 32
numbers. This proposed analysis performs zero encoder updates, head fits or PCA.
Its card and saved-TRAIN input matrix still need to be frozen before execution;
keep rates, heads, budgets, gain and width unchanged.

## Evidence and preservation

Capsule: `output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS`.
Its inventory SHA is
`9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa`;
the enclosing 180-file SOURCE SHA is
`9cd4af69f71b1cdc0368987185aa836040d349ec7936b2ef6b11e1838c2b6eb8`.
The independent saved-arithmetic audit passed 52,934,423 checks/715 archive
decodes in 307.101958641s. Validation SHA:
`cf79abdca511f22778622d2b08e62b74b592c10618d1090ea2bd53eefd84539c`.
Actual reader SHA:
`5a77eee23619d7efbc71dcb086680508b9f54f68fb761a39bfeb5aeabd2b9819`.

The inventory ordering repair changes only metadata path-component ordering;
its full reversal restores the captured prequality reader, and numerical
functions/modules remain unchanged. The failed release attempt read no
archives and started no audit. A separate compile-only failure renamed an
ambiguous archive-writing helper before a fresh successful CUDA admission;
no quality was produced by that failed attempt. Both records remain preserved.

Durable report SHA:
`d5715fa734e5c5685d0b6eb1252e9987f9730a715817f761d9507f297a554d86`.
Durable JSON SHA:
`1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4`.
Independent metadata QA passed 1,416 checks before one byte-exact durable copy.
Reporting/copying updated no models and ran no tensors, heads, PCA, bootstrap
or audit again. No TEST/stress or historical model payloads were used.

All13 previous design objects,13 previous instance groups and 24 historical
top-level registry objects remain unchanged. Only the fresh alt-05 groups/new
diagnostic/latest pointer were added. The tag registry keeps concise training
means; full per-master traces remain in the exact linked durable JSON. The
machine dataset recipe JSON remains unchanged. Use the existing managed
container and internal SDK; portability/report tools lie outside the frozen
producer SOURCE closure.

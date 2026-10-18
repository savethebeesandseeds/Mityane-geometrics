# Frozen amplitude transfer diagnostic

Measured 9 October 2026 (Asia/Dubai) under the frozen
[amplitude-transfer card](../../evaluation/cards/frozen_amplitude_transfer_v1.md).
This is a new amplitude task evaluation of reused timing-trained encoders,
with zero encoder or decoder updates and newly fitted fixed amplitude heads.
The independent saved-arithmetic audit passed 38,784,404 checks.

Each of five new data cohorts has 256 TRAIN rows from 128 paired sources and
128 VALIDATION rows from 64 disjoint sources. There are 640 TRAIN and 320
VALIDATION source groups in total; no TEST data or stress track is generated.
Natural missingness is 10%, with C3/H32/F3 legal float64 observations. The task
pairs `.5 * base + nuisance` against `2 * base + nuisance`; nuisance itself is
shared rather than multiplied. The new data masters are
14614/15715/16816/17917/19018, paired respectively with retained encoder masters
9109/10210/11311/12412/13513. New data seeds are not newly trained encoders.

The ten trained instances retain their original 512 CUDA updates at batch 8.
Their 225,805-parameter mode-2 networks with one channel mixer export the exact native global vector of size 32.
Required frozen model inference runs on the NVIDIA RTX A2000; CPU work fits raw
PCA and the fixed heads and checks saved arithmetic inside the managed container.
Three head repetitions per checkpoint do not create independent encoder runs.

## Native amplitude quality

RPB-v4.alt-01 is the ordinary learned global bottleneck timing-trained group.
RPB-v7.alt-01 has the same inference architecture with lighter .15 training
context deletion. Original RPB-v7 remains untouched. The untrained encoder is
paired point0 with the original timing TRAIN-fitted scaler; it still receives
new amplitude TRAIN-fitted heads. Raw uses 288 amplitude-TRAIN-scaled observed
values plus 288 visibility flags; mask metadata uses only the 288 flags.
PCA is fitted to raw amplitude TRAIN and never follows an encoder. Native
encoders retain their original timing scalers without refitting them.

Intact amplitude VALIDATION. Equal-master means average all three fixed head
repetitions within a cohort before averaging the five cohorts.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 64.21875 | 62.91667 | 100 |
| PCA only — no encoder | 32 | 55 | 66.35417 | 100 |
| Mask metadata | 288 | 50 | 50 | 100 |
| Untrained encoder | 32 | 99.375 | 98.64583 | 100 |
| RPB-v4.alt-01 | 32 | 99.21875 | 98.64583 | 100 |
| RPB-v7.alt-01 | 32 | 98.59375 | 97.55208 | 100 |

One additional 30% coordinate-deletion view of those same VALIDATION sources.
Every method uses the same saved source-pair-shared erasure mask, without repair
or held-out fitting.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 66.71875 | 47.23958 | 100 |
| PCA only — no encoder | 32 | 57.1875 | 55.05208 | 100 |
| Mask metadata | 288 | 50 | 50 | 100 |
| Untrained encoder | 32 | 89.6875 | 84.21875 | 100 |
| RPB-v4.alt-01 | 32 | 96.875 | 92.60417 | 100 |
| RPB-v7.alt-01 | 32 | 93.4375 | 91.875 | 100 |

Linear heads use Ridge penalty 1. Neural heads use 16 tanh hidden units, Adam at
0.01 and 100 full-batch updates over the 256 amplitude TRAIN rows. All seeds
2701/2802/2903 are retained, using `stream_seed(repetition, feature_width)`.
At size 32 the heads have 66/562 parameters; raw inputs of size 576 have 1,154/9,266 and mask inputs of size 288
heads have 578/4,658. Raw, mask and native methods use outer TRAIN normalization
followed by each head's TRAIN normalizer. PCA-only uses raw TRAIN normalization,
PCA32 and each head's TRAIN normalizer. These are newly fitted classifiers,
not hand-written raw-oracle rules or reused timing classifiers.

All 90 declared pipelines and 180 heads are supported. There are 25 shared
outer TRAIN normalizer fits, five raw observation scalers and five raw PCA
preparations. Each head fits its own TRAIN normalizer. Heads fit once and score
both VALIDATION views; there are no VALIDATION fits. Every method has 128/128
supported rows per cohort and view, so conditional and full-population accuracy
coincide here.

## Training and cost

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4.alt-01 | 0 new | — | — | 0 |
| RPB-v7.alt-01 | 0 new | — | — | 0 |

The retained budget is 512 per trained instance; the untrained controls retain
zero updates. Amplitude reconstruction MAE is **not measured**: this protocol
admits no reconstruction query or decoder-calibration state. Old timing MAE is
not substituted for an amplitude error. No encoder, decoder or optimizer is
updated in this check.

| Measurement stage | Wall seconds |
| --- | ---: |
| Factory CUDA loading and parent bindings | 12.10106 |
| Original TRAIN CUDA inference, transfer and parent verification | 17.02967 |
| New amplitude CUDA inference, transfer and parent verification | 40.75400 |
| Parent-role SHA256 verification | 6.57818 |
| Original TRAIN parity evidence I/O | 6.94935 |
| New observation generation and I/O | 0.29005 |
| CPU raw scaler/PCA preparation and I/O | 0.44812 |
| CPU head fitting, prediction, bootstrap and asset I/O | 25.34413 |
| Independent CPU saved-arithmetic audit | 159.34612 |

The synchronized CUDA callback stages include CPU transfers and immutable-parent
hash checks/I/O; factory time includes archive loading and binding. They are
mixed wall times, not pure GPU compute times. Pure GPU compute time is
unmeasured. There are 60 CUDA feature forwards: 15 original full-TRAIN parity
witnesses before generation and 45 new TRAIN/intact/deleted extraction calls.
No CPU encoder forward is used. These are descriptive costs, not a repeated
performance benchmark.

## All five cohorts

All cells below average the three retained head repetitions within that cohort.
The complete per-repetition scores, seeds, support counts and conditional
intervals remain in the [durable summary](../../../doc/results/frozen_amplitude_transfer_v1.json).

Intact amplitude VALIDATION:

| Old encoder master | New data master | Method | Linear head % | Neural head % | Coverage % |
| ---: | ---: | --- | ---: | ---: | ---: |
| 9109 | 14614 | Raw data — no encoder | 61.71875 | 61.19792 | 100 |
| 9109 | 14614 | PCA only — no encoder | 52.34375 | 63.28125 | 100 |
| 9109 | 14614 | Mask metadata | 50 | 50 | 100 |
| 9109 | 14614 | Untrained encoder | 100 | 100 | 100 |
| 9109 | 14614 | RPB-v4.alt-01 | 96.09375 | 96.875 | 100 |
| 9109 | 14614 | RPB-v7.alt-01 | 94.53125 | 91.66667 | 100 |
| 10210 | 15715 | Raw data — no encoder | 65.625 | 58.59375 | 100 |
| 10210 | 15715 | PCA only — no encoder | 53.125 | 68.48958 | 100 |
| 10210 | 15715 | Mask metadata | 50 | 50 | 100 |
| 10210 | 15715 | Untrained encoder | 98.4375 | 98.69792 | 100 |
| 10210 | 15715 | RPB-v4.alt-01 | 100 | 97.65625 | 100 |
| 10210 | 15715 | RPB-v7.alt-01 | 98.4375 | 96.09375 | 100 |
| 11311 | 16816 | Raw data — no encoder | 67.96875 | 62.5 | 100 |
| 11311 | 16816 | PCA only — no encoder | 54.6875 | 67.44792 | 100 |
| 11311 | 16816 | Mask metadata | 50 | 50 | 100 |
| 11311 | 16816 | Untrained encoder | 98.4375 | 96.35417 | 100 |
| 11311 | 16816 | RPB-v4.alt-01 | 100 | 100 | 100 |
| 11311 | 16816 | RPB-v7.alt-01 | 100 | 100 | 100 |
| 12412 | 17917 | Raw data — no encoder | 62.5 | 64.84375 | 100 |
| 12412 | 17917 | PCA only — no encoder | 61.71875 | 67.44792 | 100 |
| 12412 | 17917 | Mask metadata | 50 | 50 | 100 |
| 12412 | 17917 | Untrained encoder | 100 | 98.17708 | 100 |
| 12412 | 17917 | RPB-v4.alt-01 | 100 | 100 | 100 |
| 12412 | 17917 | RPB-v7.alt-01 | 100 | 100 | 100 |
| 13513 | 19018 | Raw data — no encoder | 63.28125 | 67.44792 | 100 |
| 13513 | 19018 | PCA only — no encoder | 53.125 | 65.10417 | 100 |
| 13513 | 19018 | Mask metadata | 50 | 50 | 100 |
| 13513 | 19018 | Untrained encoder | 100 | 100 | 100 |
| 13513 | 19018 | RPB-v4.alt-01 | 100 | 98.69792 | 100 |
| 13513 | 19018 | RPB-v7.alt-01 | 100 | 100 | 100 |

Additional 30% coordinate deletion:

| Old encoder master | New data master | Method | Linear head % | Neural head % | Coverage % |
| ---: | ---: | --- | ---: | ---: | ---: |
| 9109 | 14614 | Raw data — no encoder | 65.625 | 46.35417 | 100 |
| 9109 | 14614 | PCA only — no encoder | 60.15625 | 52.86458 | 100 |
| 9109 | 14614 | Mask metadata | 50 | 50 | 100 |
| 9109 | 14614 | Untrained encoder | 86.71875 | 94.53125 | 100 |
| 9109 | 14614 | RPB-v4.alt-01 | 88.28125 | 85.67708 | 100 |
| 9109 | 14614 | RPB-v7.alt-01 | 87.5 | 84.63542 | 100 |
| 10210 | 15715 | Raw data — no encoder | 68.75 | 48.17708 | 100 |
| 10210 | 15715 | PCA only — no encoder | 61.71875 | 54.16667 | 100 |
| 10210 | 15715 | Mask metadata | 50 | 50 | 100 |
| 10210 | 15715 | Untrained encoder | 94.53125 | 88.54167 | 100 |
| 10210 | 15715 | RPB-v4.alt-01 | 98.4375 | 93.48958 | 100 |
| 10210 | 15715 | RPB-v7.alt-01 | 90.625 | 93.48958 | 100 |
| 11311 | 16816 | Raw data — no encoder | 67.1875 | 48.17708 | 100 |
| 11311 | 16816 | PCA only — no encoder | 57.8125 | 54.6875 | 100 |
| 11311 | 16816 | Mask metadata | 50 | 50 | 100 |
| 11311 | 16816 | Untrained encoder | 88.28125 | 76.5625 | 100 |
| 11311 | 16816 | RPB-v4.alt-01 | 99.21875 | 96.875 | 100 |
| 11311 | 16816 | RPB-v7.alt-01 | 99.21875 | 98.69792 | 100 |
| 12412 | 17917 | Raw data — no encoder | 66.40625 | 47.39583 | 100 |
| 12412 | 17917 | PCA only — no encoder | 58.59375 | 57.55208 | 100 |
| 12412 | 17917 | Mask metadata | 50 | 50 | 100 |
| 12412 | 17917 | Untrained encoder | 85.15625 | 75.26042 | 100 |
| 12412 | 17917 | RPB-v4.alt-01 | 100 | 99.21875 | 100 |
| 12412 | 17917 | RPB-v7.alt-01 | 97.65625 | 91.66667 | 100 |
| 13513 | 19018 | Raw data — no encoder | 65.625 | 46.09375 | 100 |
| 13513 | 19018 | PCA only — no encoder | 47.65625 | 55.98958 | 100 |
| 13513 | 19018 | Mask metadata | 50 | 50 | 100 |
| 13513 | 19018 | Untrained encoder | 93.75 | 86.19792 | 100 |
| 13513 | 19018 | RPB-v4.alt-01 | 98.4375 | 87.76042 | 100 |
| 13513 | 19018 | RPB-v7.alt-01 | 92.1875 | 90.88542 | 100 |

Across-cohort intact ranges:

| Method | Linear head min–max % | Neural head min–max % |
| --- | ---: | ---: |
| Raw data — no encoder | 61.71875–67.96875 | 58.59375–67.44792 |
| PCA only — no encoder | 52.34375–61.71875 | 63.28125–68.48958 |
| Mask metadata | 50–50 | 50–50 |
| Untrained encoder | 98.4375–100 | 96.35417–100 |
| RPB-v4.alt-01 | 96.09375–100 | 96.875–100 |
| RPB-v7.alt-01 | 94.53125–100 | 91.66667–100 |

Across-cohort deletion ranges:

| Method | Linear head min–max % | Neural head min–max % |
| --- | ---: | ---: |
| Raw data — no encoder | 65.625–68.75 | 46.09375–48.17708 |
| PCA only — no encoder | 47.65625–61.71875 | 52.86458–57.55208 |
| Mask metadata | 50–50 | 50–50 |
| Untrained encoder | 85.15625–94.53125 | 75.26042–94.53125 |
| RPB-v4.alt-01 | 88.28125–100 | 85.67708–99.21875 |
| RPB-v7.alt-01 | 87.5–99.21875 | 84.63542–98.69792 |

All methods have 100% coverage across every cohort and repetition. Source-group
percentile bootstrap intervals use 1,000 draws and 95% confidence within each
master, conditional on its fitted head. V7-minus-v4 intervals use the common
128-row/64-source population. They do not estimate uncertainty across five
encoders, and their bounds are not averaged. Neural repetitions are retained
without choosing a favorable seed.

## Interpretation and next action

Both native encoders make this amplitude task accessible on the new sources.
The intact untrained control is already at 99.375% linear and 98.64583% neural,
so trained encoders do not earn an intact training-gain claim. V4 matches the
untrained neural mean and falls slightly below its linear mean; v7 falls below
both untrained means.

Under extra deletion, trained mean linear accuracy exceeds the untrained mean
by 7.1875 percentage points for v4 and 3.75 for v7; neural mean gains are
8.38542 and 7.65625 points. These are mean effects, not uniform improvements:
new master 14614 has lower trained neural deletion scores than its untrained
control, and v7's linear deletion score at 19018 is also lower. V4 exceeds v7
means here by 0.625 intact / 3.4375 deletion linear points and
1.09375 / 0.72917 neural points. This describes accessibility through the fixed
heads, without identifying an internal causal mechanism or broad task quality.

No amplitude encoder-training branch is needed for this bounded question.
Return to the unresolved timing information-path question after preserving this
milestone. The next planned comparison is a separately frozen earlier-channel
mixer candidate, RPB-v10, against fresh RPB-v7.alt-02 at fixed budgets and heads.
That is a prospective timing experiment, not a causal consequence of amplitude
scores, a measured gain, or promotion. Do not tune the existing deletion rate,
heads, decoder budget or rejected v9 loss. Original RPB-v7 and all retained
instances remain unchanged; RPB-v4 remains the active reference.

## Evidence and audit limits

Before any amplitude generation, the entire closed 80-role parent matrix was
admitted and all 15 original CUDA TRAIN exports passed exact float32
value-byte/bool-support parity with saved witnesses, with source/label row
association checked separately. Parent files, captured sources and reader
bytes remain unchanged. No old heads, old VALIDATION/TEST/stress, decoder-state
assets or mixed quality reports are analysis inputs.

Independent reader v2 passed 38,784,404 checks over 570 CPU archive decodes in
159.34612 seconds. It checks closed parent/source/card/admission bindings,
original TRAIN export bytes, saved raw/PCA maps, head logits/classes/support,
new source associations and within-master paired bootstrap arithmetic.
Float64 replay uses the predeclared absolute `2e-9` plus relative `2e-9`
tolerance; saved logits' own argmax classes and original CUDA exports are exact.
The reader does not run CUDA forwards, models, optimizers, head training or PCA
SVD. Ordinary checkpoint CUDA bodies are whole-byte SHA256/FNV-bound; CPU
companions and new provider audits are checked. Actual CUDA execution, model
immutability, generator behavior and fitting backends remain tied to the
captured source and engineering admission. No selection or promotion is declared.

- [Frozen card](../../evaluation/cards/frozen_amplitude_transfer_v1.md): `901247488a9588e4b44dd0ae29169bb6c8d59c8af9e3f0b0a84d2568007ed52e`.
- [Measured report](../../../output/runs/rpb-frozen-amplitude-transfer/frozen-amplitude-transfer-JFPbYJ/results/report.json): `fa7e5a15dd805b366396d07404f4ab1fd6854bcac2d12e62256ecf779fa1ab7c`.
- [Captured source](../../../output/runs/rpb-frozen-amplitude-transfer/frozen-amplitude-transfer-JFPbYJ/source-manifest.json): source fingerprint `171e6793957da291ad2913164caf508d31b5383e385bc200a5968827b80c679a`; 82 source entries.
- [Actual CUDA admission](../../../output/runs/rpb-frozen-amplitude-transfer/frozen-amplitude-transfer-JFPbYJ/admission/passed.json): `13db4209cbe4ff9d39e736eea8fa152ced4b02b503109b0d2daa07422816dd27`; log `62054cfaa028b68e5c4ca1bdb39049f40eae937ef36110e18644c55301ef2cc5`.
- Compiled binary: `110221b5b67ae96e07fc17d06117abe70f81534b4bf9f19b99fc746bdf943629`; 1,994,744 bytes.
- [Original TRAIN parity](../../../output/runs/rpb-frozen-amplitude-transfer/frozen-amplitude-transfer-JFPbYJ/results/parity-before-generation.json): `5c7d920b0b320649d4fca04ed345afe2f7fb8779fab650d72e334e8ac978999c`.
- [New capsule inventory](../../../output/runs/rpb-frozen-amplitude-transfer/frozen-amplitude-transfer-JFPbYJ/artifact-integrity.json): `16e9a7bbeefe659eeb7a9ac0feb13152f385f0104f45ea99ce30669309c05464`; 757 files / 113,292,680 bytes, excluding itself.
- [Passed independent audit](../../../output/runs/rpb-frozen-amplitude-transfer/audit-tools/run-JFPbYJ-v2/validation.json): `d531531935cdf8d0106d05b8f839318b0726f97d202ba4696eb3347e57aa923d`.
- [Sealed independent reader](../../../output/runs/rpb-frozen-amplitude-transfer/audit-tools/independent-amplitude-transfer-20261009-v2/validate_frozen_amplitude_transfer.py): `186301e6c95703c54dc1fbe7c14de9bac1a248fcb0f08fb27fca54fc7dacc6e9`.
- [Durable summary](../../../doc/results/frozen_amplitude_transfer_v1.json): `b0c129160aa2066c616e5c5ebc1f8c025cab5e22588beea6c5f0d92fb59c7933`.
- [Prior saved-TRAIN diagnosis](SAVED_NATIVE_RELIABILITY_DIAGNOSTIC.md): inconclusive timing geometry/margin evidence, preserved separately.

# Configurable RPB-v20 saved-coordinate diagnostic

Completed 12 October 2026. The complete **RPB-v20.alt-01** encoder remains strong
and unchanged. Its fixed spectral coordinates alone achieve roughly the same
timing accuracy. The learned coordinates alone stay near chance with the two
fixed classifier recipes, before and after training. That locates an improvement
opportunity in the learned route; it does not establish that its coordinates
contain no information or invalidate the complete embedding.

This is a new subset-head diagnostic on the three known TEMPO-5 cohorts.
Complete32, raw baselines and encoder training records are reused exactly;
only the declared subset classifiers were newly fitted. Every initial/final
point, cohort, repetition and validation view is retained below and in the
[machine record](results/configurable_branch_v1.json).
No encoder forward, encoder update, TEST or market evaluation occurred.

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

New saved-feature diagnostic; zero encoder execution or updates. RPB-v20.alt-01 is unchanged. Size32 means its complete export; size20 means the learned coordinates; size12 means the fixed spectral coordinates. The fixed12 arrays are exactly identical at0/512, so their nine TRAIN-fitted head pairs are shared across both points. Learned20 uses18 new pairs. Complete32 and raw results reuse original fitted decisions without new fits.

Linear head = ridge1; Neural head = tanh16,100 Adam steps at0.01. Accuracy is conditional on valid rows; coverage is the fraction scored. Means retain all three cohorts and all three head repetitions. Changing branch width changes head parameter counts and initialization seeds; this tests branch sufficiency, rather than providing a perfectly controlled attribution of each coordinate.

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

Initial, zero updates; intact VALIDATION. The TRAIN-fitted head is reused unchanged across views.

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 94.01 | 93.23 | 100.00 |
| RPB-v20.alt-01 | 20 | 50.13 | 49.78 | 100.00 |
| RPB-v20.alt-01 | 12 | 95.18 | 94.66 | 100.00 |
| Raw data — no encoder | 6912 | 49.61 | 50.69 | 100.00 |

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

Initial, zero updates; VALIDATION with an additional30% coordinate deletion. The TRAIN-fitted head is reused unchanged across views.

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 89.32 | 87.89 | 100.00 |
| RPB-v20.alt-01 | 20 | 50.91 | 50.69 | 100.00 |
| RPB-v20.alt-01 | 12 | 89.97 | 89.54 | 100.00 |
| Raw data — no encoder | 6912 | 49.22 | 50.35 | 100.00 |

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

Trained,512 updates; intact VALIDATION. The TRAIN-fitted head is reused unchanged across views.

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 94.66 | 94.31 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.74 | 50.17 | 100.00 |
| RPB-v20.alt-01 | 12 | 95.18 | 94.66 | 100.00 |
| Raw data — no encoder | 6912 | 49.61 | 50.69 | 100.00 |

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

Trained,512 updates; VALIDATION with an additional30% coordinate deletion. The TRAIN-fitted head is reused unchanged across views.

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 90.10 | 88.72 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.61 | 49.91 | 100.00 |
| RPB-v20.alt-01 | 12 | 89.97 | 89.54 | 100.00 |
| Raw data — no encoder | 6912 | 49.22 | 50.35 | 100.00 |

Dataset: **TEMPO-5** · slow timing beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5**.

Reused original training record; no training occurred in this diagnostic.

| Encoder | Updates | Train error | Validation error | GPU training seconds |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 |512|0.7825|0.7986|20.8|

Training errors are original fixed-query standardized MAE; the source training usedCUDA float32,batch8. These known VALIDATION results do not establish unseen TEST or market quality.

## Interpretation and continuation

Sizes32/20/12 are views of one unchanged encoder. They are not replacement
versions or proposed serving widths. Head parameter counts and neural seeds
change with input width, so subset comparisons measure usefulness under the
fixed recipes, rather than a perfectly controlled causal contribution.

Learned20 intact Linear/Neural means move from50.13%/49.78% at initialization
to49.74%/50.17% after512 updates. Deleted means move from50.91%/50.69%
to49.61%/49.91%. The learned-only TRAIN scores after training are also near50%;
these observations do not show ordinary high-TRAIN/low-VALIDATION overfitting.
All18 saved learned-training comparisons are retained; none has a positive
lower source-group interval bound. Those intervals are conditional on fitted
heads and do not measure uncertainty across independently trained encoders.

The learned path mixes aligned channels, applies temporal attention, pools the
patch states into channel vectors and compresses them into a global vector.
Position encodings are present. It has no explicit phase/lag comparison like
the fixed spectral path. The decoder uses complete32 plus channel/position
metadata, and the objective reconstructs all feature values with Huber loss.
It does not explicitly require learned20 alone to preserve slow timing.
These are plausible mechanisms to investigate, not a proven cause or a
measurement of collapse.

Next, predeclare a small saved-TRAIN-only investigation of learned-coordinate
variation, its TRAIN normalization and decoder-column weights. Read saved arrays
and parameters; do not refit heads, execute encoders or tune on validation.
Use that evidence to select one learned-relation or pooling change with its own
design tag, retaining the working fixed prior. Larger capacity sweeps and harder
challenge expansion are deferred. The matched H128/H256 quality check and a
feature-specific relation challenge remain later steps before market use.

## Verification and preserved evidence

The publisher independently passed1,051,888 checks:81 new and81 reused
prediction archives,27 new quality head pairs/54 classifiers, and18 learned
training comparisons. All36,864 fixed-coordinate values and their support
match exactly between initialization and512. The26 auditor regression tests
pass normally and with Python optimization; the artificial engineering
self-test passes. All815 original artifact checksums pass before and after;
new source/binary hashes and the completed capsule manifest pass too.

The initial attempt failed before any quality fit because the new reader
incorrectly assumed contiguous split source IDs. The original generator uses
shuffled role membership and sorted order inside each role. Exact membership
validation and actual noncontiguous-ID fixtures correct this. The failed
attempt, its original source capture and its zero-quality-fit receipt remain
in the evidence bundle. Each attempt fitted9 artificial engineering pairs,
separately from the27 successful quality pairs.

CPU wall times in the completed run: head/readout section9.645s; complete
reader46.221s including input/output and its automatic audit; separate saved
arithmetic replay15.929s; independent publisher47.709s. These stages execute
saved-array/classifier work. New GPU encoder training time is zero. The20.8s
training table value above is the original reused CUDA measurement.

All15 shared encoder/config/test file pairs in `cuwacunu_embedding` and
`cuwacunu` match byte-for-byte and match the pinned backport hashes. No reusable
encoder change was made in this diagnostic. Evaluation-specific additions stay
in this repository; the market loader stays in `cuwacunu`.

[Prospective card](../code/evaluation/cards/configurable_branch_v1.md) ·
[Protocol and reproduction](../code/evaluation/protocols/configurable_branch_v1/README.md) ·
[Captured summary](results/configurable_branch_v1/completed/published/summary.json) ·
[Completion receipt](results/configurable_branch_v1/completed/complete.json) ·
[Synchronization receipt](results/configurable_branch_v1/completed/encoder-sync.json) ·
[Continuation](CONTINUATION_2026-10-12_AFTER_CONFIGURABLE_BRANCH.md)

Complete local capsule: `output/runs/rpb-v20-branches/diagnostic-YA3Nws`.
Tensor archives, classifier parameters and their complete checksum manifest
remain local. The committed evidence retains exact source/card captures,
reports, arithmetic audits, logs, timing and both attempt receipts.

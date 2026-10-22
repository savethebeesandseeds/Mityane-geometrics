# CPB V18 independent closeout review

Reviewed 22 October 2026 against the saved [protocol](PROTOCOL.md),
[Operation Contract](CONTRACT.md), [results](RESULTS.md) and
[compact record](../../records/CPB-v18.span-001.json).

**The bounded comparison is complete and does not support advancing the
contiguous-mask candidate. All five predeclared conditions fail. No material
correctness blocker was found in the reviewed source and saved evidence.**
V16 seed 946103 at update 2048 remains the selected research reference.

## Review scope and evidence identity

This independent review inspected source, JSON/JSONL reports, commands and
receipts; recomputed file hashes; and performed scalar arithmetic over the
recorded metrics and counters. It did not decode tensor archives, execute an
encoder, repeat training, refit a ridge head or rerun the native audits.
Numerical audit findings below refer to the six retained native scalar-audit
executions and their verified reports.

The reviewed closeout identities are:

| Artifact | SHA-256 |
| --- | --- |
| `CPB-v18.span-001.json` | `b522f1bf16f57c19cab28ee7fdb8fd3ab848523005320063b1c839a0fbdf2719` |
| `RESULTS.md` | `0ca69746d5a4c2503dddbfa22668d91366dbfdffda19ef92ed46b11949799c5d` |
| Ignored `closeout-input-manifest.json` | `ed825deee26147489d27beb1eb2afaaf8e82a343766f05fb1a84feb4a2f52425` |

The final verification rehashed 465 distinct local files referenced by the
record, manifest, reports and native receipts. The 107-file current source
closure and executable-set digests recomputed exactly. The five existing
trainer, engineering, readout and scalar-audit binaries were separately hashed
inside the managed container and matched their admissions. This review is a
later documentation addition and does not amend the admitted protocol or the
closeout identities above.

## Training and comparison integrity

All three original control chains were checked from fresh 128 through resumed
256 updates, including their reports, native receipts, artifact hashes and
resume-input links. Each new control continued its own preserved state through
640 and 1024 using the original V17 executable and recipe. Each candidate
started fresh, stopped at 512 and resumed its own typed state to 1024. All 12
new training commands and receipts passed, followed the declared counterbalanced
order and completed before the seal.

Per-seed initial model, scaler and bank numerical identities match the original
controls; resume reports preserve recipe identity and show no preprocessing
refit. The FNV tensor identities are engineering witnesses, not cryptographic
proofs of numerical equality. All six complete 1024-event histories have the
expected ordered attempts and update counters. The same-seed arms have equal
realized source-row IDs, attempts and update flags across those histories.
All 5,376 new recorded event losses agree with Gaussian NLL plus 0.1 times the
temporal loss within the review's scalar rounding tolerance.

The actual work counts reconcile independently:

| Work | Count |
| --- | ---: |
| New control continuation updates | 2,304 |
| New candidate updates | 3,072 |
| Total new scientific optimizer updates and attempts | 5,376 |
| Previously executed control-prefix updates and attempts | 768 |
| Logical updates and attempts across six histories | 6,144 |
| New native training processes | 12 |
| New TRAIN preprocessing fits / reused control fits | 3 / 3 |
| New scientific head fits / native scalar audits | 24 / 6 |
| Separate scalar target refits within those audits | 24 |

The engineering report separately records 1,045 assertions, 126 compared
gradient tensors, 16 synthetic updates, two preprocessing fits and 27 refusals.
Those synthetic updates and fits are not included in the scientific counts.
There was no control-prefix replay, new dataset preparation or new execution of
the old TRAIN gradient diagnostic in the declared command set.

Every candidate event records 24 channel draws per batch, each with 13 legal
original-grid start positions and quota 4. Recorded starts lie in 0 through 12.
Each update selects 96 patches and 6,912 target cells, with no unsupported
channels and lag-1/4/16 pair counts 6,696/6,048/3,456. The independently summed
counters and start histograms match the seal for each seed. These are actual
returned candidate witnesses. The original control executable did not log
detailed mask or lag-support witnesses; no directly measured control counts are
claimed. Contiguity intentionally changes eligible temporal-pair coverage while
the objective and its reductions remain fixed.

All 108 preserved original source copies match their original manifest entries.
The 49 frozen V16 files and three family headers match their admitted hashes.
The engineering source pins and trainer phase source pins also match. The V16
reference declaration, checkpoint and fitted statistics remain unchanged.

## Evaluation integrity and arithmetic result

All six final 1024 states were sealed before the first readout. Each readout
binds the correct final checkpoint and actual encoder kind and mask policy.
The V17-named ridge witness format is explicitly a generic tensor codec; it is
not used as the candidate's encoder identity.

The original and adapted feature/target functions and
TRAIN-normalization/ridge/prediction/scoring source regions compare exactly
after only the declared line-ending and diagnostic-message normalization.
Typed loading, update admission and metadata changes sit outside those regions.
The source therefore retains the same 112 TRAIN / 32 known-validation anchors,
legacy target arithmetic, TRAIN standardization and ridge-1 mechanics.

All 24 head metric sets have complete 32/32 support and match the compact
record. All six retained scalar long-double Cholesky audits passed, report
original target-bit and row/group-association parity, and used the original
audit binary. Their saved tolerance is absolute 1e-9 plus relative 1e-8. The
largest reported fraction of an audit tolerance is approximately 2.44234e-5.
The review's comparison of recorded readout and independent-audit R2, MAE,
RMSE and coverage values found a maximum absolute difference of
1.0658141036401503e-14. These checks validate consistency of the saved audit
results; they do not constitute a fresh numerical refit by this reviewer.

All 12 candidate-minus-control R2 differences and their medians, minima,
maxima and pair counts were independently recomputed from the six per-run
metric sets. They match the screen summary and compact record.

| Task | Median paired R2 change | Minimum | Maximum | Positive pairs | Nonnegative pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Past volatility | -0.05785742068700628 | -0.081093927557088 | +0.010075213743628186 | 1/3 | 1/3 |
| Past trend | -0.22184588859150822 | -0.26636723531776185 | -0.09292290608782316 | 0/3 | 0/3 |
| Past activity | -0.09061982373884936 | -0.1039316800427863 | +0.014212271778777263 | 1/3 | 1/3 |
| Past relation | +0.015733336683406707 | -0.0018954940880400173 | +0.1672056860346819 | 2/3 | 2/3 |

The trend median and two-pair condition both fail. The activity median and
two-pair condition both fail. The volatility median is below its -0.01 floor.
There are no ties. Activity improves only for seed 947103; relation improves
for seeds 947101 and 947103 and remains descriptive. These successes do not
replace the declared primary conditions.

## Interpretation and limits

This result rejects advancement of this declared contiguous-mask recipe on
the known January development readouts at 1024 updates. It does not establish
that all contiguous masking is inferior, that the mechanism would fail at
another budget or that any encoder is universally best. No 2048-update
extension, fresh-period confirmation, neural head, fixed-query reconstruction,
TEMPO timing, forecasting or application utility was measured. Previously
consumed January/February/March evidence remains consumed.

The candidate's reusable construction, TRAIN fitting, typed training/resume
and inference-export surfaces remain component work. Dense benchmark witnesses
do not prove every supported sparse or partial-input configuration, and a
research checkpoint does not define the family's entire interface. No reference
replacement, application selection or historical ranking change follows from
this study.

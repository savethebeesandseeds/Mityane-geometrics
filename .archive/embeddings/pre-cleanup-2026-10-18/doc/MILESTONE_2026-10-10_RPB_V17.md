# RPB-v17 milestone, 10 October 2026

At commit c9eb14d, RPB-v17 was the strongest measured TEMPO-3 timing candidate. Preserve
this result and its exact encoder, checkpoints, scalers and fitted heads.
The saved implementation and evidence are in commit
[`c9eb14d`](https://github.com/savethebeesandseeds/cuwacunu_embedding/commit/c9eb14d2df42b8960d18e333f2476dac232c1037).

RPB-v17 exports 32 numbers: 20 learned waveform-shape coordinates and 12
generic signed temporal-relation coordinates. Only the 432 weights that
aggregate those relations are frozen; the other 226,445 parameters learn.
The relations cover every channel pair, feature pair and spacing 1–4.

Dataset: **TEMPO-3**, designed complexity **4/5**. Variable periods and small
signed delays, gains, offsets, natural missingness and three-tick channel gaps.
These are the five known development cohorts, TRAIN256/VALIDATION128 each.
The deleted view adds 30% coordinate deletion. CUDA training uses B8/512
updates. Fixed heads: Ridge penalty1; tanh16/Adam .01/100 updates, three
declared repetitions. The same TRAIN-fitted heads score both validation views.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 | 32 | 100.00 | 100.00 | 100 |

TEMPO-3, complexity4/5, deleted validation (extra30% coordinate deletion):

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 | 32 | 85.94 | 99.84 | 100 |

The zero-update encoder already reaches 100%/100% intact and 86.09%/99.74%
deleted, after its classifier heads fit TRAIN. Credit the fixed architectural
prior for timing success. This does not show that waveform training learned
the timing relation; it learns the shape block and reconstruction route.

TEMPO-3, complexity4/5, original fixed hidden-target waveform queries;
standardized mean absolute error, five-cohort means:

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 | 512 | 0.689181 | 0.723849 | 20.23 |

The time is synchronized training-loop wall time, including batch preparation
and transfers, excluding inference, heads and artifact writing. Reconstruction
remains a separate weakness; excellent timing accuracy is not universal quality.

The [full result](../code/encoders/raw_patch_bottleneck_mae/FIXED_PRIOR_TEMPORAL_RELATION_SCREEN.md)
and [durable summary](results/fixed_prior_relation_screen_v1.json) retain all
five cohorts and the prospective two-cohort screen gate. Saved checks passed
5,828,142 checks across 180 CPU witness archives and verified 30,720 exact
relation-coordinate values between updates0/512. The ordinary training path
runs once on CUDA; CPU work only fits fixed heads or verifies saved arithmetic.

The lesson from v14/v16 is concrete: waveform reconstruction can damage a
useful temporal relation. Giving that generic relation dedicated coordinates
and freezing its aggregation preserves it. Avoid resuming stopped recipes or
changing classifier capacity to improve this score.

Next: preregister **RPB-v17.alt-01** on five fresh TEMPO-3 source cohorts with
the same architecture, update budget and heads. Retain every initial/trained
result. Fresh confirmation is separate from the development screen, and neither
changes the protected v7/v4 references or defaults. If confirmed, introduce a
new coherent multi-component challenge to test richer timing and learned shape.

## Fresh-source follow-up

The [separate fresh confirmation](../code/encoders/raw_patch_bottleneck_mae/FIXED_PRIOR_FRESH_CONFIRMATION.md)
is now measured as **RPB-v17.alt-01** with the unchanged design and budget.
Intact Linear/Neural remains100%/100%; extra-deleted scores are82.19%/99.90%,
coverage100%. All five new cohorts were retained. The neural timing capability
replicates strongly; the rule requiring both heads to reach75% on every cohort
fails because master84090's deleted Linear is70.31%. Its Neural is100%.
That is a limitation of this newly measured group, not a change to saved v17.

The follow-up saved check passed5,833,011 checks/180 CPU archives and verified
30,720 exact fixed coordinates. No historical encoder or head was rerun.
Keep the useful milestone and the failed condition together. A separate
RPB-v18 development screen tests generic per-pair unit normalization inside
native32 to reduce relation-strength variation; it cannot rewrite this result.

That separate [v18 screen](../code/encoders/raw_patch_bottleneck_mae/UNIT_TEMPORAL_RELATION_SCREEN.md)
has now passed its two-cohort gate and completed the remaining three cohorts.
On those five known cohorts, intact means are100%/100% and extra-deleted
means100%/99.95%, coverage100%. Its
[own milestone note](MILESTONE_2026-10-10_RPB_V18.md) preserves initial/trained
results and fixed-prior credit. This adds a new architecture result while
retaining the exact v17 milestone and fresh follow-up.

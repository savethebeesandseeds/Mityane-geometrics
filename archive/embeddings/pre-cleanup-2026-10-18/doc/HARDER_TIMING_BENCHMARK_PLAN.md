# Harder timing benchmark plan

Date: 9 October 2026. Status: the generator and analytic information check are
implemented. Data-only fixtures passed in the managed container. Harder-data
encoder and classifier results have not been measured.

The user asked for a harder challenge because RPB-v10.alt-03 approaches 100%
on timing. The separately frozen fresh late-v7/early-v10 confirmation at the
original difficulty is complete and independently audited. Use a new named
benchmark to measure whether either encoder preserves a subtler timing relation.
Keep the original benchmark and its results available as an anchor.

The fresh RPB-v7.alt-04/RPB-v10.alt-04 comparison improves mean linear timing
from 94.21875% to 95.00000% intact and from 88.28125% to 92.81250% with extra
30% deletion. Mean reconstruction also improves. Both worst-cohort timing
scores fall, however, and two of five paired cohorts worsen in both views.
The earlier 98.28125% RPB-v10.alt-03 mean is a retained result on different
sources and seeds. The harder challenge provides additional headroom; it does
not resolve the observed variation or establish a promoted winner. Preserve
both encoder designs in the next matched comparison.

## What makes the existing challenge simple

The current shared generator in
[feature_harness.cpp](../code/shared/src/feature_harness.cpp) uses period 16,
absolute delay 2 and Gaussian noise with standard deviation 0.005. Channels 0
and 1 share a random phase; the label changes whether channel 1 leads or lags.
Channel 2 has an unrelated phase. The three features are copies with multipliers
1, 1.1 and 1.2. Natural missingness is 0.10. Source-paired variants share their
nuisance noise and mask, and their order is randomized.

This gives a clear physical relation with considerable regularity. Nearly
perfect performance on it establishes useful access to that relation; it does
not establish reliability across variable frequencies or small delays. The
strong initial amplitude controls also make the existing amplitude task a weak
indicator of learning progress near its ceiling.

## First harder recipe

Use a new, evaluation-owned generator and protocol,
`variable-delay-timing-v1`. Leave every historical generator entry point, source
capture, card and artifact unchanged. The harder generator must be reusable by
either encoder and must not import an encoder implementation.

For each independent source, draw a period P uniformly from [12,20) and a
positive delay L uniformly from [0.5,1.5). Keep the source-shared phase uniformly
distributed over a full cycle. Channel 0 uses sin(2*pi*t/P + phase). The paired
channel-1 variants use sin(2*pi*(t-L)/P + phase) for label 0 and
sin(2*pi*(t+L)/P + phase) for label 1. Keep channel 2's unrelated phase, the three
feature multipliers, noise standard deviation 0.005 and natural missingness
0.10. Both variants of a source share P, L, nuisance noise and masks.

This changes both period regularity and delay magnitude; it is one benchmark
recipe, not an experiment isolating which data change causes a score difference.
The known phase separation 2*pi*L/P ranges from about 0.157 to 0.785 radians.
It stays positive and below pi, so the intended sign avoids zero-delay and
period-alias ambiguity. Period, delay and clean signals must never be available
to encoder fitting, raw preprocessing, PCA or classifier heads.

Keep shape C3/H32/F3, semantic channel IDs 0/1/2 and the 32-number encoder
export. Draw source groups and assign splits before forming the paired variants.
Keep TRAIN and VALIDATION source IDs disjoint; randomize paired row order and
zero all hidden storage. Mask statistics and row order must not encode labels.
New protocol/source namespaces and independent random streams are mandatory.
Do not convert harder data into larger heads or increase the update budget.

## Check that difficulty still contains usable information

Add a separately labelled analytic solvability check using only legal observed
values. It must not receive labels, P, L or hidden clean values. For each legal
adjacent-time tuple in channels 0 and 1, compute

    x1(t)*x0(t+1) - x0(t)*x1(t+1)

Without nuisance noise this equals coefficient^2 * sin(2*pi/P) *
sin(2*pi*L/P) for label 1, and changes sign for label 0. It does not require
knowing the period or delay. A feature tuple is legal only when all four values
in that expression are observed. Aggregate available feature tuples within a
time position, then average across observed time positions. Freeze the support rule
before measurement; validity requires at least four distinct supported
adjacent-time positions. A zero margin must abstain rather than acquire a label
from hidden data. Report valid/total counts, accuracy, coverage and margins.

This analytic check is a task-information diagnostic, not another fitted
classifier and not a replacement for raw/PCA baseline rows. Its score is
unmeasured here. Added noise and missingness can reduce its accuracy or coverage;
those outcomes must remain visible. Do not drop difficult rows or tune the
validity rule after seeing scores.

## Fixed comparison and reporting

Freeze a prospective card, complete source closure, seed schedule and independent
saved-evidence checks before generating quality cohorts. Use five new paired
cohorts, 128 TRAIN and 64 VALIDATION source pairs per cohort (256/128 examples).
Compare fresh late-v7 and compact early-v10 instances with matched original
TRAIN observations, initial parameters, scaler and row/query/context streams.
Preserve original v7 and every v10 group. New run groups receive separate
`.alt-NN` identities; changing only the benchmark does not create a new encoder
design tag.

Retain the original .15 training context deletion, native32, batch8, 512 CUDA
updates, optimizer and reconstruction query. Use fixed Ridge penalty1 and the
existing tanh16 head, Adam0.01/100 updates, all three declared head repetitions.
No PCA follows an encoder. Fit raw PCA32 and all normalizers on legal TRAIN
observations only. Keep raw576, mask288, PCA only — no encoder, both untrained
controls and both trained native32 methods.

Measure intact VALIDATION and one fixed additional .30 deletion view, with the
same TRAIN-fitted heads reused across views. Preserve every cohort, unsupported
fit, conditional source interval, mean and worst-cohort score. Report this task
separately from amplitude and from the easier benchmark. Keep loss/error and
training cost separate from classifier accuracy, following the
[reporting standard](RESULTS_REPORTING_STANDARD.md).

The first version is a development benchmark. It opens no historical TEST or
stress payloads and selects no model or budget. A future held-out acceptance
procedure requires its own prospective card and untouched test sources. Do not
change the harder generator after scores in order to obtain a preferred amount
of headroom. If it is still near ceiling or is poorly observable, retain that
result and define a separately versioned benchmark afterward.

## Implementation boundary and next action

The unchanged-difficulty confirmation is complete; see the
[confirmation card](../code/evaluation/cards/early_mixer_confirmation_v1.md) and
[audited result](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_CONFIRMATION_DIAGNOSTIC.md).
Its sole saved-arithmetic audit passed 96,153,106 checks over 1,255 decoded
archives. All ten new encoder trajectories trained once on CUDA; the reader
performed no encoder or classifier fitting.
The [generator and analytic check](../code/evaluation/benchmarks/variable_delay_timing/README.md)
are implemented separately from the encoders. Their source and noiseless formula
were independently reviewed. The managed-container data-only target passed
determinism, split/source isolation, paired masks, hidden-value independence,
analytic signs and the fixed support rule. The saved log is
`output/runs/variable-delay-timing/data-fixtures-F3PUUS/build-and-tests.log`.
This is artificial correctness evidence, not a measured harder-data accuracy.
The [fixture record](results/variable_delay_timing_data_fixtures_v1.json) records
the tested source and log hashes.

Before the harder encoder comparison, freeze its prospective card, five new
cohort seeds, source closure and independent saved-evidence reader. Keep that run
separate from the original-difficulty confirmation and retain both results.
Actual CUDA admission precedes any new encoder training. CPU work fits the fixed
heads/raw PCA and checks saved arithmetic; it must not repeat encoder execution.

Later difficulty axes can include noise, contiguous gaps, irrelevant channels or
different feature content. Keep those separate from this first recipe. Reaching
100% should lead to a named new challenge while retaining the easier scores,
rather than changing the old data or continually increasing classifier capacity.

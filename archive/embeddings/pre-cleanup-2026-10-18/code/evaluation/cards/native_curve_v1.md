# Native curve v1

Protocol: `native-curve-v1`. Policy: 1.2. Evidence stage: synthetic development.

This new protocol evaluates **RPB-v4**, the learned global bottleneck, through
its exact native 32-number export. No PCA or random projection follows the
encoder, including during checkpoint selection. Standalone raw576 and raw-PCA32
controls use the same legal cohorts. Older encoder designs are not trained.
Historical drivers, cards, checkpoints and reports retain their original rules.

The following recipe is fixed before generation. The runner saves the resolved
card and experiment plan in a new capsule under `output/runs/rpb-native-curve/`.
Source/configuration identities and actual seed streams accompany that plan.
There are no measurements in this card and no consumer acceptance decision.

## Data and task scope

Use fresh development masters **3101/3202/3303**, CPU float64 legal observations
with shape C3/H32/F3, semantic channel IDs 0/1/2, unitless features and sampling
interval 1. Each master/task has 128 TRAIN source pairs (256 examples), 64
VALIDATION pairs (128 examples) and 64 fresh TEST pairs (128 examples). Natural
missingness is 10%; both variants of a source share their observation mask.
Source pairs remain together and TRAIN/VALIDATION/TEST source groups are disjoint.
Clean hidden values are never supplied to encoder fitting or readout preparation.

The legal raw-data oracle must achieve at least 95% accuracy with nonzero support
on each TRAIN, VALIDATION and eventual fresh TEST cohort as a declared task
solvability gate. Failure aborts the run and preserves its artifacts; it does
not authorize replacement seeds, regeneration or adjustment of this recipe.
The oracle is a separately labelled diagnostic and does not select checkpoints.

The primary task is timing (`lag_sign`). Its completed-update milestones are
0/128/512/2048. Sanity tasks are direction (`reversal`), level and amplitude;
their milestones are 0/512, with 512 fixed independently of sanity scores.
There are twelve persistent master/task trainers and thirty measured checkpoint
points. Tasks train separate models; this does not test one universal encoder.

Development generation creates TRAIN and VALIDATION only. Every TEST draw,
including sanity and initialization comparisons, occurs after supported timing
selection has been durably saved. The fresh TEST namespace is
`native-curve-v1/fresh-testing`, with stream constant
`0x6e63763174657374` (ASCII `ncv1test`, decimal 7954331321644708724). For each master,
the actual fresh seed is `stream_seed(master, 0x6e63763174657374ULL)`; task identity
is retained by the controlled generator. The draw does not depend on the winning
budget or any score. No prior TEST artifact supplies selection evidence.

## Encoder training and immutable points

Register only RPB-v4. Use the existing
[learned-global configuration](../../encoders/raw_patch_bottleneck_mae/config/learned_global.conf),
which specifies channel mixer layers 1 and global bottleneck mode 2. The CLI
overrides device to CUDA and the maximum update budget to 2048; batch size 8,
patch length 8 and export width 32 stay fixed. Record the complete resolved
configuration, per-cohort training seeds and runtime overrides, not just a tag.

Keep encoder width 64, three layers, four heads, feedforward width 256, decoder
hidden width 128, dropout 0, layer-norm epsilon 1e-5, mask ratio 0.25 and Huber
delta 1. Use AdamW learning rate 0.001, weight decay 0.0001, gradient clipping 1
and the existing RPB scaler floor 1e-6. The scaler fits legal TRAIN observations
once and remains frozen. Rows are sampled with replacement and masks are fresh.
Each trainer uses one continuous optimizer/RNG stream and absolute completed
budgets; record attempts, skips, sampled rows and cumulative training time.

Point zero is the exact initialization of that persistent trainer, with its own
TRAIN-fitted heads. It is the matched untrained control, not another trainer.
Save ordinary resumable checkpoints and frozen readouts at every declared point.
Measurement must preserve the ambient CPU/CUDA RNG state, including snapshot
construction and head fitting, so it cannot alter subsequent encoder updates.
Earlier checkpoint snapshots and fitted tensors must remain immutable.

Before expensive training, verify actual CUDA parameters/inputs/loss, finite
gradients, changed weights, and checkpoint/export/reconstruction parity. The
decoder must consume solely the exact served 32-number global bottleneck.
Encoder labels and held-out observations are absent from the fitting API.

## Fixed representations and heads

Fit each raw control **once per master/task cohort**, and retain all its fitted
objects across milestones, final TEST and stress:

1. **Raw data — no encoder:** the shared observed-TRAIN `ObservationScaler`,
   float64 per channel/feature across rows/history, population standard deviation
   with floor 1e-8. Flatten normalized values with missing values zero, followed
   by original visibility flags: 288 values plus 288 flags, size 576.
2. **PCA only — no encoder:** TRAIN-valid-row feature normalization and centered
   TRAIN PCA of that raw576 representation, width 32. Insufficient rank remains
   explicitly unsupported; neither width nor representation changes afterward.
3. **RPB-v4:** exact native global32 at each frozen point, with TRAIN-only feature
   normalization and no projection.

Each head additionally owns its valid-TRAIN-row normalizer. Ridge uses penalty 1.
The fixed neural head has sixteen tanh hidden units, Adam learning rate 0.01 and
100 updates. Head seeds are 2701/2802/2903; actual initialization uses
`stream_seed(head_seed, input_width)`, paired across equal-width methods and
milestones. Fit weights separately, retain all three repetitions, and never
select a head seed. At width32 the heads have 66/562 fitted parameters; raw576
has 1154/9266. Head repetitions are not independent encoder trainings.

Raw/PCA support is any observed coordinate. Native support follows its declared
global export contract and must be identical across all timing milestones of a
master. An all-missing signal abstains. Report each method's valid/total rows,
class/source support and coverage. An unsupported raw/PCA control cannot disable
a legal native fit. Mask-only heads and legal raw analytic oracles are separate
sanity diagnostics and cannot change selection.

## Selection and fresh TEST

For each positive timing budget, compute mean native32 ridge VALIDATION accuracy
over all three declared masters, equally weighted. Retain all declared head
repetitions; ridge fits are deterministic under the fixed recipe. Every primary
fit and scoring population must be supported. One unsupported primary entry
excludes the entire budget; do not average only successful entries. Mandatory
native support invariance prevents checkpoint-dependent population selection.

Choose the supported positive budget with highest mean accuracy; exact ties
choose the smaller budget. Point zero cannot win. Neural, raw/PCA, reconstruction,
stress and sanity-task scores do not influence this decision. If no positive
budget is supported, stop with an explicit failure and do not generate TEST.

Durably persist `selection.json` before generating TEST. Restore and verify the
exact chosen checkpoint and its already fitted normalizer/heads; there is no
retraining or readout refitting. Timing TEST compares chosen RPB-v4, exact
point-zero RPB-v4, raw576 and PCA-only32 on the same fresh cohort. Sanity TEST
uses its fixed512 point, exact point zero and the same fixed controls. A chosen
snapshot must still match its saved validation-stage exports and reconstruction
witness after the trainer has reached later milestones.

For every measured method/head report conditional accuracy, coverage and
full-population correct/total with abstentions counted as failure. Pair effects
use each declared pair's common valid population. Report candidate-minus-
comparator intervals from 1000 whole-source-group percentile bootstrap draws,
keeping both variants together. Intervals are conditional on a master checkpoint
and fitted heads; do not create an across-retraining interval from three masters
or from the three head fits. Summaries average heads within a master and then
masters equally; keep task tables separate.

## Reconstruction, cost and stress

At each point score fixed TRAIN/VALIDATION reconstruction queries using legal
observed target cells and the frozen RPB training scaler. Enumerate each original
whole patch as a hidden query, shared across channels; retain the natural mask
and require sufficient visible-patch support. Reduce standardized MAE and Huber
equally through target cells, channels and examples. Save target masks,
eligibility, predictions and per-example losses. Masked optimization Huber and
fixed-query reconstruction MAE are distinct measurements.

Record CUDA-synchronized cumulative encoder training seconds, attempts/completed
updates, parameter counts and loss/gradient traces. Timing excludes extraction,
head fitting, scoring, stress and writes. Costs are descriptive; this is not a
speed acceptance benchmark.

Apply the unchanged
[fixed-readout-stress-v1 recipe](fixed_readout_stress_v1.md) to every measured
method and declared head repetition, including point zero, only on fresh TEST:
intact; additional coordinate deletion at 10/30/60/90%; contiguous blackout at
25/50/75% of H; one case for each semantic channel 0/1/2 absent; and all absent.
These are twelve cases. Both source variants and all methods receive identical
corruptions; no encoder, scaler, PCA, normalizer or head fits under stress.

Intact predictions/validity must equal cached ordinary TEST outputs. Save
corruptions, requested/actual masks, semantic-channel support counts, source
mapping, readout predictions and paired intervals. Distinguish stressed common-
valid conditional effects from full-population correctness effects. All-absent
signal coverage is zero; metadata controls may retain structural validity and
remain labelled as controls. Stress never changes the selected budget.

## Required evidence and limitations

Use a new output root and preserve all prior capsules. Save the instantiated
card/plan before generation; legal source manifests; exact source/configuration,
dataset/scaler/checkpoint identities; immutable checkpoints; fitting assets;
row-level predictions; selection; reconstruction witnesses; stress sidecars;
coverage/population records; and final per-master reports. Focused generic tests
must demonstrate label-free model fitting, selection-before-TEST, unsupported/
tie behavior, no post-encoder PCA, fit-once controls, snapshot/RNG/clone isolation,
exact final restore, and stress without refitting.

The protocol provides fresh synthetic development evidence. It does not define
consumer chronology, acceptance thresholds, external tasks or an across-seed
confidence interval. A failed primary cannot be rescued by the neural head or
by changing the reported split. Follow the
[reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md),
[evaluation policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md) and
[advance record](../../encoders/raw_patch_bottleneck_mae/NATIVE_CURVE_ADVANCE.md).

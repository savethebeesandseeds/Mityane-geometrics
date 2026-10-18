# TEMPO-3: structured harder timing comparison

Prospective development card, 9 October 2026. Protocol:
`structured-hard-timing-comparison-v2`. Dataset: **TEMPO-3**,
`structured-hard-timing-v1`, **designed complexity 4/5**. This ordinal level
describes the declared challenge recipe, not observed accuracy, mathematical
entropy, sample count or an encoder quality score. Retain TEMPO-1 and TEMPO-2.

## Data and information admission

The preserved v1 information admission passed both intact populations but its
same-feature rule covered only950/1024 and900/1024 deleted examples, below95%.
No encoder or classifier was fitted under v1. This v2 card keeps the exact
TEMPO-3 generator and all numeric gates. It revises only the observed-data
information rule, before any encoder result, and uses fresh engineering seeds.

Single task: `lag_sign`. Legal observations remain C3/H32/F3, semantic channel
IDs 0/1/2 and native encoder width32. Per independent source draw period
P uniformly from [10,24), absolute delay L from [.25,1), and a shared full-cycle
phase. Channel0 is sin(2*pi*t/P+phase); channel1 is sin(2*pi*(t-L)/P+phase)
for label0 and sin(2*pi*(t+L)/P+phase) for label1. Channel2 has independent
period [6,12) and phase. Each channel/feature has a time-constant positive
gain: channels0/1 use [.5,1.5), channel2 uses [1,2), multiplied by 1+.1*f.
Each channel/feature offset is uniformly [-.75,.75). Gaussian noise SD stays
.005. Natural coordinate missingness stays .10, plus one randomly located
three-tick gap per channel across every feature, with starts0..29.

Paired variants share periods, absolute delay, phases, gains, offsets, noise
and masks; randomize their row order. Split source groups before forming
variants. TRAIN and VALIDATION sources are disjoint. Local signal/mask/gap/
order/split random streams are independent of Torch runtime state. Hidden
storage is zero. The clean field clones only legal observations. No true
period, delay, hidden waveform or TEST dataset is exported to fitting.

The observed-only analytic information rule accepts values and bool masks
only, never labels, source IDs, true parameters or fitted assets. At centre c
and spacing k in1..4, collect each channel's features whose three observations
at c-k,c,c+k are all legal. Channel0 and channel1 may use different features.
For each channel independently average its eligible left and right differences:
d0_left=mean_f[x0_f(c)-x0_f(c-k)], d0_right=mean_f[x0_f(c+k)-x0_f(c)],
and likewise d1. Require at least one eligible feature in each channel. Compute

    d1_left*d0_right - d0_left*d1_right

This equals averaging determinants over every legal channel0-feature by
channel1-feature pair. Average legal spacings within a centre,
then supported centres. Validity requires at least four distinct centres and
nonzero finite margin; otherwise abstain with prediction0. A valid positive
margin predicts label1, a negative margin label0. In the noiseless case the
feature-pair tuple equals 4*g0*g1*sin(pi*k/P)^2*sin(2*pi*k/P)*sin(2*pi*L/P)
for label1; g0 and g1 may belong to different features, both are positive;
constant offsets cancel. P>=10,k<=4 prevents spacing aliases. Added noise
and missingness may reduce accuracy or coverage; retain every row and margin.

Before encoder cohorts, run the separately declared engineering information
admission `structured-hard-timing-information-v2`, rule
`observed-cross-feature-determinant-v2`, on fresh seeds82585/83686, each512
TRAIN and512 VALIDATION source pairs.
Evaluate all1,024 VALIDATION rows per seed intact and with fixed extra.30
coordinate deletion. Require each seed's intact conditional accuracy>=99%
and coverage>=99%, and deleted accuracy>=98% and coverage>=95%. Report full-
population correctness too. Capture actual source/binary/compiler log and the
counts/hash before/after. These engineering data never fit an encoder or head.
If admission fails, preserve this recipe/result and stop encoder generation;
do not tune it using model scores or silently revise TEMPO-3.

## Paired encoder comparison

Use five new paired masters75272/76373/77474/78575/79676. Each supplies128
TRAIN and64 VALIDATION source pairs, 256/128 rows. Fresh RPB-v7.alt-05 and
RPB-v10.alt-05 preserve the exact late/early designs and coordinate15 policy:
mode2, one mixer, dropout0, pooling source0, patch8, temporal width64,
projection/native width32, 225,805 parameters. Mix placement0/1 is the only
architecture difference. Equal parameters do not mean equal compute; early
mixing retains an independent local temporal pass. All same-name initial
parameters/buffers and the TRAIN scaler are paired exactly before training.

Both complete512 unskipped CUDA updates at batch8, cap1,024 attempts, AdamW
lr.001/wd.0001/clip1/Huber1, original.15 training context deletion, thread1,
trace interval1. Retain every0/512 state, optimizer, scaler, all512 trace
entries and both initial controls. Ten trajectories give5,120 joint updates
and40,960 sampled rows,16 equivalent presentations each. No extra decoder
calibration, continuation, rescue tuning or best point/seed selection.

The new external namespace remains distinct from unchanged implementation
`early-mixer-reliability-v1/lag_sign` (and its engineering identity). The four
old companions keep truthful legacy metadata. A new fifth `.structured-timing.pt`
binds the external data/order/scaler/schema/counters/source and four parent
bytes; a supplemental snapshot witness binds both identities. No old quality
payload is an input, and no historical model/card/report is overwritten.

## Readouts and reconstruction

Seven fixed-head methods: raw576, mask288, raw PCA32, untrained late32,
untrained early32, native late32 and native early32. No PCA follows an encoder.
Raw PCA and normalizers fit legal TRAIN only. Ridge penalty1 and tanh16,
Adam.01/100 steps are unchanged. Repetitions2701/2802/2903 use the existing
paired width-dependent seed law. Fit each head once on ordinary TRAIN and
reuse it intact and on one fixed additional.30 VALIDATION deletion view.
Use stream_seed(master,0x746d70332d64656c) and purpose
`structured-hard-timing-comparison-v2/lag_sign/validation-coordinate-dropout`.

Maximum105 pipelines/210 heads,5 shared driver raw/PCA outer maps and25
helper outer maps. Four native methods times three surfaces times five
cohorts give60 full CUDA exports. Ten trained states each score TRAIN and
VALIDATION original whole-patch queries:20 writers/four banks each,80
necessary Q-masked CUDA forwards. Extract each needed surface/query once.
Report original-query TRAIN/VALIDATION standardized MAE separately from the
random-mask optimization Huber trace and from classifier accuracy.

Save the analytic rule's TRAIN/intact/deleted predictions, validity, margins
and supported-centre counts separately. It has zero fitted heads; its accuracy
must not appear under Linear head or Neural head columns. Report conditional
accuracy, valid/total, coverage and full-population correctness, including nulls
for zero support. It is a declared solvability check, not Bayes optimality.

Retain every method/cohort and all three head repetitions, fixed1000/95%
conditional source-group intervals, paired effects on common support, means,
ranges and worst-cohort scores. Do not average interval bounds into an across-
encoder confidence interval or different tasks into one encoder accuracy.

## Evidence and disposition

Before quality, complete real CUDA artificial admission for the new binding,
old model/adapter contracts, direct versus split optimizer trajectory, actual
CUDA input/loss/gradient/weight updates, snapshot/RNG isolation and typed role
rejections. Force current compiler records for every producer object inside the
checker's captured task log; do not rely on a separate cached preliminary build.
Capture the complete source closure, card, binary, information admission and
reviewed FALSE independent reader bundle before generating cohorts.

The sole post-completion independent reader replays saved CPU arithmetic,
source/parent/scaler/data/support/masks/query/logits/own-argmax/intervals and
analytic margins. F64 abs2e-9+rel2e-9 and F32 abs2e-6+rel2e-5 tolerances stay
fixed; promised exact witnesses/classes/support/order are exact. It imports
no encoder, autograd or optimizer and fits no head/PCA/SVD. Ordinary CUDA
checkpoint bodies are hash-bound; source and actual CUDA admission establish
trajectory/inference association, not independent neural reenactment.

Assess intact/deleted timing linear means and worst cohorts with per-cohort
coverage, reconstruction, neural scores and analytic information coverage.
There is no automatic numeric promotion rule. Lower accuracy on harder data
is not evidence that a saved easier-data encoder was damaged. Preserve all
original v7, every v10 group, formal v4 reference and stopped mechanisms.

Use exclusive leaves under `output/runs/rpb-structured-hard-timing` and named
container build `rpb-structured-hard-timing`. Use the existing managed container
and internal SDK; no host builds, new dependency installs or container deletion.
Separate synchronized training-loop, feature/query/transfer/verification/I/O,
CPU-head/bootstrap and independent-audit costs. GPU loop timers include CPU
trace capture and are not pure kernel benchmarks. Every table has a short
TEMPO-3 legend with complexity4/5 and scoring view, following the reporting
standard. Keep both standard head tables and the two-row training table.
No TEST/stress, model/head/budget selection, architecture promotion or old
artifact replacement is part of this development comparison.

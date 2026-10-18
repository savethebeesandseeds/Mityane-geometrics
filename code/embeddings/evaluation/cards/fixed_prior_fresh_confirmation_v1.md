# Fixed-prior fresh confirmation, version 1

Protocol: `fixed-prior-fresh-confirmation-v1`. Instance bundle **RPB-v17.alt-01**,
unchanged design **RPB-v17**. This prospectively confirms the known-development
milestone on five fresh TRAIN/VALIDATION cohorts. No other encoder is retrained.

Dataset: **TEMPO-3**, recipe `structured-hard-timing-v1`, designed complexity
**4/5**. Use the authoritative existing generator with its original variable
period/small signed lag, positive gains, offsets, natural missingness .10 and
three-tick channel gaps. Each master has192 independent opposite-label pairs:
TRAIN128 pairs/256 examples and VALIDATION64 pairs/128 examples. Preserve the
original source-disjoint random split and randomized within-pair row order.
The extra validation view uses original pair-shared .30 coordinate deletion,
`stream_seed(master,0x746d70332d64656c)`, and preserves source IDs and labels.

The complete declared master list is **80787,81888,82989,84090,85191**. Generate
these fresh cohorts once in a new exclusive output directory. Source IDs start
`fixed-prior-fresh-confirmation-v1/lag_sign/structured-hard-timing-v1/seed-M/lag_sign/source-N`.
Capture exact generator/source/card/SDK/data hashes before quality training.
The generator is evaluation infrastructure; encoder inputs contain only legal
observations, masks and fixed semantic channel IDs. Class labels only fit/score
heads. No clean waveform, selected task pair, phase, period or delay enters the
encoder or its targets. Do not inspect fresh quality scores before the freeze.

## Unchanged recipe

Native32:20 learned shape+12 fixed generic odd relations, all channel pairs,
all ordered feature pairs and spacings1–4. Original v17 header and parent
headers are unchanged. Total226,877 parameters; trainable226,445; frozen432.
The sole decoder input is that same native32 plus original fixed IDs.
Original waveform Huber1, no supervised auxiliary or post-encoder PCA.

Train each of the five cohorts once from0 through512 completed CUDA updates.
B8, context deletion .15, patch8, hidden64, dropout0, AdamW .001/weight decay
.0001, clip1. Preserve original initialization, row/mask/Torch counter streams.
Retain checkpoints at0/512, frozen TRAIN scalers, AdamW state, RNG, counters,
loss traces and synchronized training-loop timing. No best seed/checkpoint
selection, added training, rescue, head search or automatic promotion.

Shared heads remain Ridge penalty1 and tanh16/Adam .01/100 updates. Use paired
repetitions2701/2802/2903 and `stream_seed(repetition,width)`. Fit TRAIN once per
point/cohort/repetition and score intact/deleted validation with those same fits.
Export all three native feature views at0/512 and original fixed four-bank
TRAIN/VALIDATION hidden-target queries at512. No raw/PCA or older-encoder refits
are required for this focused replication; historical baselines are not matched
controls on these new cohorts.

## Prospective confirmation rule

Run and retain **all five cohorts**, irrespective of intermediate scores. The
primary continuity rule preserves the development screen's threshold: BOTH
heads must reach75% in BOTH views for EACH trained512 cohort, with100% coverage.

Separately, the strong timing capability target is met only if EACH trained512
cohort has:

- intact validation Linear and Neural accuracy at least95%;
- deleted validation Neural accuracy at least95%, Linear accuracy at least75%;
- coverage100% for both heads in both views.

The stronger target was chosen from the known development result and is now
prospective; it has no independent statistical basis. These are practical
thresholds, not a statistical guarantee or default-promotion rule. Report both
rules, means and worst cohorts,
including any failure. Keep the initial0 controls and original query MAE so
timing-prior benefit and waveform learning remain distinguishable. Credit the
fixed prior when initial controls already solve the task. No learned-benefit
claim follows simply from high trained accuracy.

## Admission and execution

Use the existing managed GPU container and a separate named session. Reuse
unchanged generic evaluation objects and pinned saved-arithmetic modules.
The new data writer must pass deterministic generator, split, labels, hidden
zero, pair-mask and exact deletion-subset checks. The new loader must reject
wrong namespaces/roles/types/keys. Actual CUDA admission verifies parameters,
inputs, loss, nonzero trainable gradients/changed weights, frozen432/no-gradient
and typed immutable save/reload before quality. Reuse the already admitted v17
primitive model implementation; run the new protocol's CUDA engineering gate.

Root schedules GPU programs sequentially. Saved checks use CPU witness tensors
only: head predictions/scores/coverage/source association, waveform MAE and
exact12 fixed coordinates between0/512. They perform no encoder execution,
optimizer updates, classifier refits or PCA. Initial budget:5 trajectories,
2,560 CUDA updates,30 native exports,30 head pipelines/60 individual heads,
10 query writers/40 masked forwards. Admission fixtures are separately counted.

Preserve all known development evidence, original v7, formal v4, TEMPO-1
v10.alt-03 and the other family. A different architecture, objective, source
recipe or budget requires a separate prospective card and outputs.

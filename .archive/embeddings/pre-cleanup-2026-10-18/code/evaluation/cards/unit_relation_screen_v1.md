# Unit temporal-relation screen, version 1

Protocol: `unit-relation-screen-v1`; new design **RPB-v18**. This is a targeted
development architecture screen using retained fresh-confirmation data, not
another unseen-data confirmation. Preserve RPB-v17 and RPB-v17.alt-01 intact.

Dataset: **TEMPO-3**, recipe `structured-hard-timing-v1`, designed complexity
**4/5**: variable period/small signed delay, gains/offsets, .10 natural
missingness, three-tick channel gaps and the saved extra .30 deletion view.
Reuse exact15 controlled archives from
`/embedding/output/runs/rpb-fixed-prior-fresh-confirmation/admission-3CGpxM/fresh-data`.
Each cohort has TRAIN256/128 independent pairs and VALIDATION128/64 pairs.
Keep the original source-disjoint splits and fresh-confirmation source namespace.
Only legal values/masks/fixed semantic IDs reach the encoder; scoring labels,
source IDs, true phase/period/delay and clean waveforms do not.

## One architecture change

Fresh RPB-v17.alt-01 retains100% intact with both heads and99.90% deleted Neural,
but deleted Linear averages82.19%, with master84090 at70.31%. Initial controls
already show that weakness, so this is not destruction of the saved v17 model.
The fixed generic prior works well; its strength varies with period, delay and
available observations. Shared TRAIN feature scaling does not remove variation
between examples. This is a hypothesis for the weak linear readout, not a
proven causal diagnosis.

RPB-v18 keeps the learned20 shape coordinates and the same fixed12 relation
coordinates, grouped as three semantic channel pairs × four original-time
spacings. Within the encoder, divide each four-vector by its own Euclidean norm
clamped below at1e-12. Allzero vectors remainzero. Apply the same rule to ALL
pairs; preserve signs and relative spacing geometry. No selected pair, binary
sign shortcut, new learned coefficient, task label or post-encoder PCA.

The decoder receives only this modified native32 plus fixed IDs. The parent
backbone, shape projection, decoder and literal frozen432 relation weights,
parameter names/order and initialization stay unchanged. Registered226,877,
trainable226,445, frozen432. Group normalization discards relation magnitude;
retain waveform MAE and future richer-task limitations. Do not claim universal
quality or waveform training learning timing from this fixed prior.

## Fixed budget and continuation rule

Run masters **80787 and84090** once from0 through512 completed CUDA updates.
They are explicitly chosen development cohorts, including the known weak
linear case. B8/.15, original waveform Huber1, AdamW .001/wd .0001/clip1,
all initialization/row/mask/Torch counter streams and native32 remain unchanged.
Keep point0/512, scaler, optimizer/RNG/counters, full losses and synchronized
training time. No rate, normalization-floor, head, deletion or budget search.

Use unchanged shared Ridge1 and tanh16/Adam .01/100 heads, repetitions2701/
2802/2903 under `stream_seed(repetition,width)`. Fit TRAIN once per point/cohort/
repetition and score intact/deleted with the same fits. Export three native views
at0/512 and original four-bank TRAIN/VALIDATION waveform queries at512.

Reuse the saved matched RPB-v17.alt-01 scores from
`doc/results/fixed_prior_fresh_confirmation_v1.json`, SHA
`6255530c52f710dbb5311e385d7529f878ba3637ce5f5c9101b93864a43bab11`.
No old encoder execution, baseline head refits or PCA. Advance to retained
masters81888/82989/85191 only if:

- BOTH trained heads reach75% in BOTH validation views for EACH starting cohort;
- coverage is100% in both views;
- Neural is no worse than that cohort's saved v17 score in each view;
- mean deleted Linear across the two improves at least5 percentage points.

Use the mean of all three repetitions, retain every result and initial control,
and report exact failed conditions. Otherwise stop after two. This is a cost
gate for a focused hypothesis, not promotion or evidence of learned timing.
No stopped-recipe rescue or automatic replacement of protected references.

## Admission and saved evidence

Build/test/run in the existing managed GPU container in a new named session.
Reuse nine unchanged generic evaluation objects and pinned saved-arithmetic
modules. Admit same common initialization/RNG/first20 coordinates, unit prior
geometry/finitezero behavior, semantic/mask invariance, joint time reversal,
sole-native32 decoding, real CUDA gradients/updates and frozen432/coordinate
parity. Typed immutable save/reload must preserve this new architecture identity.

Capture card, closed sources, exact retained input hashes and verified SDK proof
before quality. Root schedules GPU jobs sequentially. Saved CPU witness checks
verify source/label/split/heads/scores/coverage, legal TRAIN scaler, original query
MAE and exact12 prior coordinates at0/512. They fit nothing and run no model.
Initial budget:2 trajectories/1,024 CUDA updates,12 native exports,12 head
pipelines/24 heads,4 query writers/16 masked forwards. Count admission separately.

# Architecture direction, 10 October 2026

The user asked to prioritize architecture and approach changes rather than
spend more time comparing weak TEMPO-3 variants. The prospective data-support
2×2 plan remains preserved in the earlier continuation; it is deferred.

Three subagents independently inspected the input/mixing/pooling route,
nuisance handling and objective against saved TEMPO-3 evidence. They converged
on explicit cross-channel temporal relations as the strongest first change.
The early backbone mixes aligned channels, then pools their temporal states;
v13 adds within-channel increments but no explicit oriented cross-channel
products. Lower waveform error has not produced useful fixed-head timing scores.
Raw fixed-head validation is also weak, so accuracy alone does not prove
information was lost by the encoder.

The frozen [architecture card](../code/evaluation/cards/architecture_screen_v1.md)
defines two independent changes and a small starting budget:

- RPB-v14: generic visible-only signed/symmetric relation bank, all semantic
  channel pairs/features/spacings, learned into the same served native32 and
  original waveform decoder.
- RPB-v15: unchanged early encoder input path plus a training-only native32
  decoder for detached legal-observation temporal relations, raw+relation Huber
  fixed1:1. Original observed training data supplies self-supervised targets.

Both first ran once on75272/76373 from0 to512 on CUDA. Keep point0 and512,
original masked-query MAE, all fixed-head repetitions, checkpoints/optimizer,
scaler, RNG/counters, source/data hashes and costs. A design advances to the
remaining three known masters only if each starting cohort reaches75% in BOTH
heads and BOTH validation views with100% coverage. Record all failures. Credit
a strong untrained relation design to its architectural prior; do not claim
that unsupervised training learned the same gain.

The [measured screen](../code/encoders/raw_patch_bottleneck_mae/TEMPORAL_ARCHITECTURE_SCREEN.md)
now shows RPB-v14 passing that gate and completing all five known cohorts.
Intact fixed linear/neural accuracy is97.03%/95.10%; extra deletion is
72.97%/75.21%, all at100% coverage. Its untrained results are100%/100% intact
and80.31%/82.71% deleted. Credit the generic relation prior, and retain the fact
that waveform training weakens it. Mean query MAE is0.685666/0.725161 TRAIN/VAL.
RPB-v15 stays near52% on its two-cohort screen and stops without expanded runs.

The separate [RPB-v16 card](../code/evaluation/cards/partitioned_relation_screen_v1.md)
tests dedicated native32 allocation:20 learned shape+12 grouped odd relations.
The relation coordinates cannot mix in raw, symmetric or support coordinates,
and remain time-odd under arbitrary trained weights. Its projections may still
collapse useful cues, while reducing shape capacity may hurt reconstruction.
Measure its own untrained control and fixed512 screen before broader evaluation.
It reuses the existing shared evaluation objects and fixed heads, with a separate
model header, test, producer, checkpoint kind, source freeze and output session.

A later architecture variant must get a separate tag/card/output. A possible
distinct direction is visible-only instance normalization with all moments
carried inside native32:14 learned shape coordinates plus9 means/9 log-scales,
with decoder inversion reading those32 only. This is a deferred proposal,
not an implementation or measured result. Do not add a statistics bypass to the
decoder or change head budgets to make either candidate look better.

Use the existing managed container and named task runner. Root schedules GPU
jobs sequentially after actual CUDA/mask/gradient/save-load engineering tests.
Saved CPU features fit the shared fixed heads; the saved-evidence checker only
verifies arithmetic and lineage and executes no encoder or training. Keep this
exploratory screen's provenance compact. Preserve original v7, active v4,
TEMPO-1 v10.alt-03 and every previous measured group.

## Completed fixed-prior result and next work

[RPB-v16's measured screen](../code/encoders/raw_patch_bottleneck_mae/PARTITIONED_TEMPORAL_RELATION_SCREEN.md)
reaches100%/100% intact and83.20%/80.34% deleted on two cohorts, but
master76373's69.53% deleted neural score fails its fixed gate. Stop it there.
The [new v17 card](../code/evaluation/cards/fixed_prior_relation_screen_v1.md)
freezes only432 generic odd weights while the other226,445 values remain
trainable. Its [completed record](../code/encoders/raw_patch_bottleneck_mae/FIXED_PRIOR_TEMPORAL_RELATION_SCREEN.md)
retains all five known CUDA trajectories at512, initial controls, fixed heads,
original queries, scalers, optimizer/RNG/counters and saved checks.

V17 intact accuracy is100% with both heads; with extra deletion it is
85.94% Linear/99.84% Neural, coverage100%. Its untrained deleted result is
86.09%/99.74%; trained heads still fit TRAIN in that control. Credit the fixed
relation prior. Waveform query MAE is0.689181/0.723849 TRAIN/VAL, worse than the
reused early-v10 reference; timing success is not universal embedding quality.
Saved checks confirm exact0/512 parity for30,720 odd-coordinate values without
model reruns. The five-cohort expansion follows the original two-cohort gate,
so it is screen-selected known-development evidence, not unseen confirmation.

Next session should preregister a fresh-source TRAIN/VALIDATION confirmation
for **RPB-v17**, keeping native32, .15/B8/512, frozen432 and the existing heads,
with no old encoder training, head/rate/deletion search or chosen best seed.
Verify actual CUDA before running, retain every initial/trained result and reuse
saved feature files for scoring. If that holds, design a separate coherent
multi-component challenge (new codename/complexity before quality measurement)
to test richer timing and the contribution of the learned shape coordinates.
Do not reopen v15/v16, tune heads or promote a default from known-cohort results.

All four new designs are separate from original saved v7, formal v4, original
TEMPO-1 v10.alt-03 and MTF-v1. Their sources remain in the RPB family; evaluation
uses the shared fixed feature/query/readout helpers. Fourteen quality trajectories
ran once on CUDA this advance,7,168 updates total; no CPU encoder duplicate.
The prospective TRAIN-size2×2 and instance-normalization proposal remain deferred.

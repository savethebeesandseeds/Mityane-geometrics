<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v7.market-01 review: preserve a scoped gain, stop this clipping recipe

**Provisional draft: training/quality audits passed; final independent evidence
join is pending.** V7 is a measured clean causal reference with4/22 numeric
conditions and100% coverage. CPB-v3.market-01 remains the baseline and sole dev
encoder. Preserve every V3/V4/V5/V6 checkpoint, result and displaced leader;
V7 does not replace them or establish a balanced overall winner.

V7 improves neural market volatility to R²0.6733049756106732, versus the retained
V5 reference0.6481906853790493. It also improves fixed-query TRAIN MAE to
0.24180533789857825 and TRAIN Gaussian NLL to-0.40882727825899934. The NLL record
is descriptive; it adds no TRAIN NLL advancement gate. Preserve these three
scopes even though18 of22 conditions fail. Strong known-development scores in
one scope do not compensate for a failed protected scope elsewhere.

The passed conditions are neural volatility, TRAIN MAE, and positive linear/
neural relation. Linear volatility remains below V3; both trend and activity
heads fail their strongest floors; relation stays below V6 linear and V5 neural
leaders. Neural trend fits0.239513425,-0.194818323,0.284144619 fail all-three
positivity. All eight TEMPO-5 retention conditions fail, despite high absolute
accuracy in several panels. VALIDATION MAE0.3704205114995439 and NLL
0.868789873717446 fail the protected V4 values0.30831692401624544 and
-0.04538148809081624. V7 also regresses both validation reconstruction metrics
relative to V6. None of this is forecasting or a fresh-period confirmation.

This is one controlled fresh clipping experiment relative to V6. Native40,
forward geometry, initial constructor draw order, unit-weight Huber1 mean plus
detached-error Gaussian variance objective, fresh TRAIN statistics, seeds,
2048/B8 update budget and D40 fixed heads stay unchanged. All294752 learned
values start fresh; every one of76 learned tensors receives nonzero CUDA
gradients and changes. All13 initial whole40 DATA/statistic pairings versus V6
are exact. That pairing strengthens the initialization comparison without
importing V6 weights/statistics or equating file/metadata bytes.

Separate norm2 max1 clips for mean73 tensors/293537 values and uncertainty3/1215
remove direct cross-branch clipping attenuation at the same current gradients.
They also permit combined gradient norm up tosqrt2 instead of1. This is a real
change in the admissible joint update geometry, not proof of one isolated cause.
A single AdamW and changing future mean residuals continue to couple trajectories.
No claim of complete optimizer independence or uncertainty-gradient dominance
follows from group sizes, norms, loss magnitudes or the observed quality shifts.

In the actual V7 trajectory, the recorded mean group never clips, while the
uncertainty group clips on1313/2048 steps. The independent audit checks both
recorded factor formulas and group coverage; full gradient norms are observed
producer evidence, not independently reconstructed arrays. These are V7's own
states. They cannot be substituted for counterfactual V6 gradients at the same
weights, nor prove why V6's shared clip engaged on1326 steps. A different
trajectory already changes residuals, variance calibration and AdamW history.

Trust checks are strong but answer a different question from quality. Engineering
passes213824 checks/7372 CUDA calls/110 typed refusals, and the separate nonzero
mechanism fixture passes359 CUDA checks, manual composite native40 gradient
parity, live mean/uncertainty learning and branch-clipping isolation. Initialized
and trained saved artifacts each pass all128 cutoffs,2350 future-value/mask
interventions and30 internal temporal cases with unchanged state and zero new
updates. Actual typed resume occurs at512. Training audit passes283252 checks;
quality audit passes12493 with all13 initial pairings and no mismatches. These
checks establish the declared causal/training/evaluation procedure; they do not
turn a4/22 candidate into a quality winner.

The strongest V3–V6 floor protocol096584… is prospective for V7. Preserve original
historical pass counts under their own cards: V4's19/22, V5's7/22 and V6's12/22
were not scored against this identical set of floors. A future retention
protocol should protect V7's neural volatility/TRAIN MAE improvements in addition
to every earlier scope before any new candidate quality; TRAIN NLL remains a
record, not a new gate. No historical report or threshold is rewritten here.

Stop this exact separate-clipping recipe. Do not rescue it with cap, coefficient,
rate, seed, budget, checkpoint or head tuning. The next bounded step is a
separately declared saved TRAIN-only V6/V7 diagnosis: inspect the same112 TRAIN
features/readouts and query summaries, matched first8 fixed-query cell witnesses,
and the actual component/branch-norm records with their true state boundaries.
Avoid held-out tensor diagnosis, refits, model replays, cross-basis gradient
cosines and unsupported full-Jacobian attribution. Use that evidence before
choosing a broader causal temporal-summary architecture; no next encoder is
declared or running. The clean encoder goal remains active.

See [per-task results](CAUSAL_FRESH_V7_RESULTS.md),
[physical errors, repetitions and evidence](CAUSAL_FRESH_V7_DETAIL.md),
[the card](CAUSAL_FRESH_V7_CARD.md),
[training audit](causal_fresh_v7/independent-training-audit.json),
[quality audit](causal_fresh_v7/quality-independent-audit.json) and
[the preserved ledger](clean_causal_encoder_research_ledger.json).

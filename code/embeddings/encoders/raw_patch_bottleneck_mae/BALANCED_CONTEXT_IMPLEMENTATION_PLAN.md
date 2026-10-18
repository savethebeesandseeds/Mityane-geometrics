# Balanced context implementation plan

Date: 2026-10-07. Status: proposed, unimplemented and unmeasured.
Scope: one TRAIN/known-VALIDATION development hypothesis; no TEST, stress,
rate/budget search, selection or promotion. RPB-v4 remains active.
RPB-v8 is reserved in the version registry for this proposed training policy;
it is not an implemented model or a measured improvement.

## Evidence and hypothesis

The completed [RPB-v7 diagnostic](CONTEXT_LIGHTER_VALIDATION_DIAGNOSTIC.md)
uses the same five known masters and 512 updates as retained RPB-v4/RPB-v6.
Native linear means improve from RPB-v6's 95.3125% / 93.90625% to
RPB-v7's 97.96875% / 95.15625% (intact / additional-30% VALIDATION deletion),
at equal 100% coverage. Mean TRAIN/VALIDATION MAE falls from
0.091178964 / 0.092983954 to 0.081178695 / 0.083629869, but stays above
RPB-v4's mean 0.067593144 / 0.071309066. Separately, each master's RPB-v7
TRAIN and VALIDATION error exceeds its own paired RPB-v4 error. Both mean
reconstruction guards fail. Independent archive audit v3 passed 67,097,380 checks
before this plan was finalized.

The improvement is uneven: master 8808 loses 5.46875 linear points under
deletion versus RPB-v6. Neural means also fall, from 99.739583% / 97.864583%
to 98.958333% / 96.25%. A higher mean linear score does not establish universal
quality improvement or resolve reconstruction. The earlier unchanged-policy
[longer-training diagnostic](CONTEXT_OPTIMIZATION_DIAGNOSTIC.md) reduced
reconstruction error while weakening linear access; it does not support
simply adding updates as the joint remedy.

A balanced mixture is justified as one falsifiable input-view hypothesis:
ordinary-context updates may anchor reconstruction, while occasional stronger
deletion preserves the incentive to encode timing under incomplete context.
It is not a demonstrated repair. Half the updates use the existing 0.30
deletion law; half use no extra deletion. The average nominal request rate is
0.15 before repair, but the view distribution differs from deleting at 0.15
on every update. Actual coordinate counts depend on support and repair and
must not be claimed equal to RPB-v7's counts.

Risks are reduced corrupted-view exposure, different gradient weighting and
retaining the same capacity tradeoff despite the mixture. This intervention
tests the encoder's training distribution through fixed heads; it changes
neither classifier capacity nor classifier optimization. Do not add an
auxiliary loss, decoder branch, teacher, projection or second forward pass.

## Frozen schedule and unchanged route

Use the unchanged RPB-v4 mode-2/mixer-1 architecture and exact served global32:
225,805 parameters, C3/H32/F3, patch 8, width 64, three temporal blocks,
four heads, feed-forward 256 and decoder 128. Preserve batch 8, AdamW,
clipping, dropout, scaler and the fixed 512 completed-update budget.

Define `a` as the zero-based absolute attempted-update counter read **before**
incrementing it, matching the current trainer. Freeze this sole schedule:

- `a % 2 == 0`: ordinary visible context, extra erasure E=0.
- `a % 2 == 1`: existing `coordinate30_v1` deletion and repair.

Thus attempt indices 0..511 contain exactly 256 ordinary and 256 deletion
updates. The first update is ordinary; do not reverse, shuffle or tune the
schedule. Row sampling already uses an independent counter stream; require
the exact same sampled rows/artificial masks as references at every attempt.
Do not choose the branch from labels, signal values, losses, completed steps,
local loop indices or support. A skipped attempt aborts this paired experiment;
do not shift the schedule or add replacement updates to restore balance.

Use the original initialization, row, patch-mask and Torch counter streams.
On deletion attempts, use the original context stream
`0x6374782d64726f70` with the absolute `a`, unchanged semantic-canonical BCHF
ordinal and top-53-bit law. Its requested E matches the existing RPB-v6 law
on that attempt's original support. Do not renumber active attempts as 0..255.
Ordinary attempts need no context draws; the context generator is stateless
and consumes no Torch RNG. Keep per-attempt Torch seeding on **both** branches.

Construct original O, A, Q=O&A and eligibility before choosing the view.
Ordinary V=O&~A; deletion V=(O&~A)&~E. Preserve original target values,
queries, eligibility, Huber 1 and cell/channel/example denominators exactly.
Existing repair clears only E, never natural missingness or Q, until originally
eligible channels retain two original visible patch groups. Zero nonvisible
storage before encode. Decode solely the exact served32 vector with the same
metadata. Frozen inference applies neither schedule, deletion nor repair;
observed-support rules and invalid/all-missing export behavior stay unchanged.

## Smallest implementation and checkpoint contract

Add one explicitly identified training recipe to the encoder-owned
context helper/trainer. Preserve existing disabled, 0.30 and 0.15 branches and
their numerical/state contracts. Suggested new policy ID:
`rpb-training-context-balanced-030-v1`; reserve RPB-v8 for this recipe.
Record the zero-based alternating rule, active rate 0.30, context stream and
ordinary/deletion attempt counts separately; do not serialize the policy as
uniform rate 0.15. Count requested/actual/restored coordinates on deletion
updates only, with zero added counts on ordinary updates.

Keep the current fresh continuous trainer rather than introduce a general
augmented-resume API. `train_to` uses absolute cumulative budgets; snapshots,
extraction, reconstruction and readout fitting restore ambient CPU/all-CUDA RNG
and thread state before training continues. Checkpoint tags must identify the
balanced policy and ordinary workflow resume must reject them. Save model,
buffers, AdamW steps/moments, TRAIN scaler/identity, attempted/completed/sample
counters, branch counts and context counts. A deterministic fresh replay must
reproduce these before continuation is permitted; saving a checkpoint does
not by itself promise arbitrary tagged-resume support. Any later restoration
API must validate policy/schedule identity before restoring training state.

No shared evaluator change is needed. Reuse `NativeCurveRun` development-only
TRAIN/VAL writers and `ArchiveReadoutRun`'s additional validation view. Keep
encoder training/admission in its own adapter and generic evaluation free of
encoder includes. No historical driver, card or result is altered.

## Admission before five-master measurement

First run focused support/counter tests and one small actual-CUDA admission:

1. Show schedule counts and branch choices at 0/1/2/3/511, including a live
   split at odd point 1; exactly 256/256 at 512. Reject unknown policy/recipe.
2. Verify common initialized parameters/buffers/scaler match RPB-v4 exactly.
   The first ordinary CUDA update must reproduce ordinary v4 at the same seed,
   rows and A. On the next deletion attempt, compare requested/repaired E
   directly with the existing 0.30 primitive at absolute index 1.
3. Check untouched O/A/Q/eligibility/targets, semantic reorder invariance,
   E-only repair, invalid support, hidden-value isolation and exact32 decode.
4. Compare direct 0→4 with 0→1→2→4 interrupted by checkpoint snapshots and
   evaluation. Require exact named weights/buffers, AdamW steps/moments,
   scaler, counters and context counts, plus unchanged earlier witnesses.
   Verify actual CUDA parameters/inputs/loss/gradients and changed weights.
5. Prove tagged ordinary resume rejects; old disabled/0.30/0.15 regressions
   remain exact. Verify evaluator inspection does not advance trainer RNG.

These checks establish policy mechanics and device use, not model quality.

## One prospective known-cohort diagnostic and stopping rule

After saving the passed v7 audit and measured milestone, freeze one card, exact
source/settings/input hashes and admission record before fitting. Proposed
outer protocol: `context-balanced-validation-v1`. Use only timing and the
five known masters 4404/5505/6606/7707/8808, their original 128 TRAIN pairs
(256 rows) and 64 VALIDATION pairs (128 rows), and fixed points 0/512.
Train five new candidate trajectories; reuse retained v4/v6/v7, with no older
encoder retraining. Bind the original legal fitting namespace and compare
cohort/order/scaler/common initialization before any update.

Reuse the exact completed v7 additional-30% VALIDATION mask for each master,
not another favorable draw. Fit candidate heads only on ordinary TRAIN once
per checkpoint; use those same fits for intact and deleted VALIDATION.
Freeze ridge 1 and tanh16/Adam0.01/100 updates, all repetitions
2701/2802/2903 and width-paired seeds. No PCA follows native32. If ordinary
reference/control refits are needed for schema reuse, disclose them and require
exact retained-fit/prediction or identical-input raw/PCA parity; no corrupted
VALIDATION fitting. Retain original fixed-query MAE/Huber, full support,
per-master/head effects, conditional source-group intervals and GPU loop costs.

Predeclare the joint question at 512: mean native ridge in both views no lower
than RPB-v7 at equal coverage, worst-master linear scores no lower, and mean
original TRAIN and VALIDATION MAE each no worse than paired RPB-v4. Report
neural changes and every master even though ridge is primary. Crossing-zero
within-master intervals limit interpretation; no head or master is selected.
The candidate can fail this demanding joint question despite partial gains.

If it fails, preserve the result and stop this local schedule/rate/budget
tuning sequence; report the timing–reconstruction tradeoff rather than soften
the guard or launch another grid. Even a pass would be known-development
evidence requiring a separate frozen confirmation card and CUDA admission,
with no automatic TEST draw or promotion.

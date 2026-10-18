# Configurable content-only ordered relation development screen

Protocol: `configurable-content-relation-v1`.
Stage: prospective RPB-v22 known-TRAIN/VALIDATION development evidence, not
TEST or fresh confirmation. This architecture was selected after the known
RPB-v21 outcome. Freeze this card and the complete source closure before any
RPB-v22 candidate quality training or fitting.

## Question and single architectural factor

Can removing unconstrained additive observation-density inputs retain the
learned timing gains of the ordered relation route while protecting the
complete embedding under the same saved additional-deletion view?

The separately tagged design is **RPB-v22**; this experiment's run group is
**RPB-v22.alt-01**. It wraps the preserved configurable RPB-v20 module. Relative
to the stopped RPB-v21 architecture, the sole architectural factor is removing
common-support fractions from the inputs to its bias-free, zero-initialized
learned relation projection. Support counts remain necessary for validity,
joint centering, normalization and numerical guards. They are not separately
projected as free additive learned inputs.

Keep original tick lags1/2/4/8, canonical semantic channel pairs, all ordered
feature pairs, the same four-endpoint common-support mask, centered odd/even
normalized covariance, minimum common count2 and energy floor1e-12. Keep the
same CUDA float64 moment accumulation and float32 bank/projection arithmetic.
At C3/F9, the projection receives1,944 moment inputs instead of2,916 inputs
and has38,880 weights for20 learned coordinates. The final12 fixed spectral
coordinates, base backbone and decoder architecture remain unchanged. Exact
arithmetic and content-only layout are bound by captured source and a distinct
typed checkpoint architecture/layout identity.

The route receives only visible observations. Timing labels and hidden target
values never enter the bank or optimizer. The zero projection consumes no
initialization RNG. With the declared seed, initialized native32 values and
validity must exactly equal the saved v20 control on all three views before
initialized readout reuse is permitted. Train from initialization; never resume
or relabel the stopped v21 checkpoints.

The preceding TRAIN support audit found an active chiefly offset-like support
contribution that is identical within each opposite-timing source pair. Its
additional-deletion expectation derives from the frozen Bernoulli erasure
policy, not observed deleted-feature execution. It is an exploratory mechanism
hypothesis selected after the v21 failure, not evidence that support caused
that failure. Removing support inputs does not guarantee arbitrary deletion
invariance: moment estimates and the base encoder may still change with masks.
The engineered relation bank remains an architectural choice, not proof of
universally learned structure or causal decoder use.

## Fixed data and training

Dataset: **TEMPO-5**, slow lead/lag beneath independent fast rhythms, nine
grouped features and structured distractors; designed complexity **5/5**.
Use the exact saved original capacity-v1 role archives and TRAIN scaler rule.
No quality generator call, new source, changed task or harder data is allocated.
Natural gaps and masks remain unchanged. The deleted view uses the original
saved additional30% coordinate-erasure policy/output; it is not redrawn and
does not imply an exact deletion count.

Original capsule: `output/runs/rpb-v20-capacity/admission-0gN4uD`.
Original source ID:
`8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1`.
Original input manifest SHA256:
`6121773c0b97e5490f191a9f8ff013e90f27a9a557376ac8256bdbb2d6af6091`.
Original complete artifact manifest SHA256:
`62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83`.
Saved branch controls: `output/runs/rpb-v20-branches/diagnostic-YA3Nws`.
Branch completion manifest SHA256:
`cefa413c560bdf2ab1eee5be2e3f34feca6e2f723a99627f9eeae683c4c11d93`.
Preserve the stopped v21 capsule
`output/runs/rpb-v21-learned-relation/admission-0PGNmn`, with artifact manifest
`6b138475bbeac3bde35cc45f331f21f5e10cff64b5971fb883f60121c20562bf`.
The numerical gate uses the saved v20 controls, not a replacement v21 baseline.

Screen masters: **941101 and941102**, in that order. A third already-known
master **941103** runs only after the complete joint screen passes. It is an
additional development check, not unseen confirmation. Retain every completed
trajectory and all scores; do not select a best checkpoint or seed.

Each cohort has512 TRAIN examples from256 paired source groups and256
VALIDATION examples from128 disjoint paired source groups. Both variants of a
source share their role and observation mask. Preserve shuffled role membership
and original ascending source order.

One independent candidate trajectory per allocated cohort; C3/H128/F9/P8,
W64/D32, three temporal layers/four heads, feedforward256, decoder128.
CUDA:0 float32 parameters only; batch8; retained checkpoints0 and512;
exactly512 completed AdamW updates, learning rate0.001, weight decay0.0001,
clipping1. Original initialization seed, row sampling, mask streams, patch
masking25% and training context deletion15% remain unchanged. The sole objective
is the original hierarchical waveform Huber loss. No supervised auxiliary, new
loss weight, learning-rate/budget rescue, decoder enlargement or head tuning.

Record actual CUDA parameter/input/native/prediction/loss/gradient devices and
new projection gradients and updates. CPU encoder calls must remain zero.
Expected quality parameter counts with the unchanged base are282,521 total,
278,633 trainable and3,888 frozen. Reconcile actual counts rather than inferring
execution from settings. Admission uses only artificial master941000 and two
actual CUDA updates; it proves execution and integrity, not quality.

Before quality, admission must verify generic geometry/masks, retained odd/even
moment arithmetic, original initial output/RNG parity, exact fixed12, actual
CUDA updates, and immutable typed checkpoint reload/malformed-state rejection.
Include a meaningful artificial witness whose valid support counts differ at
every declared lag but whose odd/even moments agree. With deterministic nonzero
projection weights, its relation contribution must agree. A zero projection
alone is insufficient, and equality of the full backbone under arbitrary mask
changes is not required.

## Evaluation and reuse

Evaluation remains separate from the model. Use native complete32 and its
exact first20 coordinate subset, without PCA. TRAIN-only normalization,
ridge penalty1 and neural16-tanh/Adam0.01/100 updates remain fixed. Three
head repetitions2701/2802/2903 use the existing width-derived seed policy.
Head repetitions are not independently trained encoders.

Fit only trained complete32 and trained learned20: six new head pairs/twelve
classifiers per candidate cohort, two methods times three repetitions. Capture
TRAIN, intact VALIDATION and deleted VALIDATION predictions for each fit.
Reuse initialized complete32 and learned20 predictions only after exact
candidate0 export/value/validity/source equality with their original saved
inputs. Reuse saved trained v20 complete32/learned20 and fixed12 controls
without fitting them again: five saved methods and15 reused fit pairs per
cohort. Raw/PCA baseline fitting and scoring remain outside this screen; their
historical measurements stay preserved. Independently verify saved fitted
weights, normalizers, input values, logits, decisions and metrics when publishing.

The fixed12 values and validity must match the original v20 control exactly at
both points and all three views. This is a preservation check, not new fixed
embedding training. Distinguish candidate and saved-control prediction identities
even when their method strings coincide. The four paired comparisons remain
trained candidate complete32 versus initialized/trained v20 complete32 and
trained candidate learned20 versus initialized/trained v20 learned20, each for
all three repetitions and both VALIDATION views:24 comparisons per cohort.
Intervals use common saved source groups and are conditional on each fixed
cohort and classifier fit; they do not measure encoder retraining uncertainty.

Reconstruction is the unchanged leave-one-complete-patch-out query, evaluated
on original TRAIN and intact VALIDATION observations under TRAIN-fitted scales.
Report hierarchical standardized MAE, coverage and original target counts;
keep its timing separate from GPU training, feature extraction and CPU heads.

## Prospective joint gate

The two-cohort numerical gate is identical to configurable-learned-relation-v1:
the same54 conditions, thresholds, units and aggregation. Integrity is a
prerequisite, never a substitute numeric score. Evaluate the first two masters
only after their trajectories and saved readout audits complete. Take an equal
mean across the two cohorts and three declared head repetitions. Apply each
condition separately to each head and view, never average task types.

1. Engineering, source/config/card/input binding, immutable typed checkpoints,
   exact initial control equality, exact fixed12 preservation and saved
   arithmetic audits pass. Every declared readout and query has100% coverage.
2. At512, learned20 mean accuracy is at least60% for each head on intact and
   deleted VALIDATION. Each of these four means exceeds its initialized20
   control and saved trained-v20 learned20 control by at least10 percentage
   points.
3. At512, complete32 mean accuracy loses at most2 percentage points against
   saved trained v20 complete32 for each head/view. No per-cohort
   three-repetition mean loses more than5 points for either head/view.
4. Candidate mean original-query MAE is at most1.02 times the corresponding
   saved trained-v20 mean, separately for TRAIN and VALIDATION. No individual
   cohort exceeds1.05 times its own corresponding reference MAE. This declared
   small protection margin is not a claim of exact reconstruction equivalence.

The screen's54 numeric conditions are20 method/cohort/view coverage checks,
12 learned usefulness checks,12 complete32 protection checks and10 query
coverage/error checks. Use accuracy fractions in machine-readable evidence;
percentage-point changes above correspond to0.10,0.02 and0.05.

Only a joint pass allocates master941103 at the same unchanged recipe. Report
its entire evidence separately and the equal three-cohort means. A third-cohort
pass uses the same learned20 thresholds and complete32/MAE protection margins
against its own saved controls; it does not retroactively redefine the screen.
If a gate fails, preserve the complete failed stage and stop this frozen recipe.
No seed, rate, lag subset, coefficient, width, loss, head or update-budget rescue.
There is no promotion, historical replacement or serving default change.

## Evidence and interpretation

Use a distinct new API/config/checkpoint/layout/protocol/output identity; v20
and the stopped v21 code, checkpoints, evidence and cards remain preserved.
Capture source/card/settings/input manifests, binary hashes, role/source IDs,
TRAIN scaler witnesses, typed checkpoints, optimizer/RNG witnesses, complete
update traces, native features, query arrays and persisted readout artifacts.
Freeze the complete transitive model/protocol/shared-helper/source closure
before quality; verify the executed source stays byte-identical to that capture.
Use distinct new screen/third leaves and never overwrite a saved run. Verify
original and saved-control artifact hashes before and after. Completion requires
independent saved-arithmetic publication and complete counter/summary
reconciliation.

A learned20 gain establishes usefulness under these fixed heads on this known
synthetic challenge. It does not establish general market quality, intrinsic
information recovery, arbitrary history quality, arbitrary deletion invariance
or causal decoder attribution. A failed gate is useful evidence about this
single architecture/objective pair. Keep working RPB-v20.alt-01, native32,
historical checkpoints/cards and the stopped RPB-v21.alt-01 evidence.

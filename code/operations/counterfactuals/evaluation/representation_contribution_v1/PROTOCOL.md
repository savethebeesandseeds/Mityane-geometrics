# Prospective representation contribution comparison

**ID:** `OR06-REP-CONTRIB-001`, version `1.0.0`, 20 October 2026.
**Operation:** [OR06-PCR 0.1.0](../../CONTRACT.md), controlled model response.
**Instance:** [requirements and evidence](RER.md). Exact [configuration](configuration.json).

## Question and finite scope

Does the old mixed/raw-polynomial tie reflect equivalent features, or do
different fixed representation families change response recovery at matched
head capacity? This separately identified follow-up preserves the original
[study](../synthetic_response_v1/RESULTS.md) and its source capsule unchanged.
The old task and score are exposed. Only new observations are protected; these
are not novel task formulas or independent evidence about learned encoders.

This is a readout contribution study: the native evaluator fits and saves
responses, then subtracts alternative minus baseline. An independent auditor
reconstructs every value. It does not revalidate the core operation's supplied
attestations or establish new observational/causal identification. No matching,
intervention effect, external data, inherited encoder, GPU or training experiment
from another Hero is admitted.

## Representations, targets and matched budgets

Raw observations are static pairs `(a,b)`, without a time horizon. Each feature
map is fixed before seeing data. Native shape is four real coordinates; geometry
is the declared coordinate map with TRAIN-only population standardization.
No distance, alignment, clustering or metric superiority is evaluated.

| Family | Fixed coordinates | Information |
| --- | --- | --- |
| raw_polynomial | a, b, ab, a² | retains a,b; old simple descriptor control |
| raw_additive | a, b, a², b² | retains a,b; different additive span |
| fourier | a, b, sin(πa), cos(πa) | retains a,b; periodic span |
| hinge | a, b, abs(a), abs(b) | retains a,b; piecewise linear span |
| lossy | b, b², sin(πb), cos(πb) | erases changed a; adverse information control |

The first four access exactly the same information and have four features plus
one intercept per head. Each receives exactly one OLS fit per response axis,
with identical training rows, noise, standardization, fitting and selection
budgets. No family is a mere invertible linear reparameterization of another's
four coordinates. Matching downstream parameter counts does **not** equate total
function classes or approximation bias. Lossy is excluded from equal-information
superiority claims. Zero response is an unfitted baseline, not a capacity match.

Three response axes are reported separately in their native numerical units:

- polynomial: `1 + 2a - b + 1.5ab + 0.5a²` (unchanged original response);
- periodic: `1 + 2a - b + sin(πa) + 0.5cos(πa)`;
- hinge: `1 + 2a - b + 1.5abs(a) + 0.5abs(b)`.

The axes deliberately include known feature compatibility and misspecification.
The two new axes are a prospective synthetic sensitivity test, not evidence
that an unknown real task resembles them. No scalar cross-axis winner is used.
Each training label is its axis response plus the same row's `noise_sign/32`.
All 45 heads are fitted only on TRAIN. No fitted encoder is present.

## Generator, protection and exposure

Use unsigned LCG `s=(1664525*s+1013904223) mod 2^32`. Draw
`a_num=next%6145-3072`, `b_num=next%8193-4096`; retry the whole pair if b is
reserved, then draw the noise sign from the next state's high bit. Divide
coordinates by 4096. IDs are `training-i`, `development-i`, `assessment-i`.

Independently regenerate the original recipe's three seeds and all original
splits solely to reserve their b numerators; no old models or evaluation execute.
Save this exposure/exclusion record. New b reservations are global across all
seeds and splits, so every old/new and new/new source or a-only alternative is
disjoint. Record the pre-assessment generator states. The auditor independently
regenerates all rows and verifies exclusions and disjointness.

Before scientific admission, an engineering overlap test inadvertently generated
the candidate populations for seeds 12011, 23003, 34019 (512/128/384 rows).
It fitted no models and observed no response scores. Treat all those coordinates
as exposed: regenerate their staged sequence after the original exclusions and
reserve every additional b. The configuration and exposure receipt retain this
preflight history; these candidates never enter scientific results. The overlap
test now uses only engineering seed 40031. Scientific exclusions also reserve
every b from the failed initial 40031 preflight (original exclusions only), the
final engineering recipe (original plus candidate exclusions), and native
selftest rows. Engineering preflights always use their fixed original-plus-
candidate boundary; scientific exclusions additionally reserve these engineering
rows, preventing repeated preflights from shifting the boundary recursively.
The exact recipes are recorded in configuration and independently audited.
The final counts below were
chosen before scientific admission to leave sufficient coordinate capacity
(the runner checks actual exclusion count + 2,208 new b values <=8,193).

Use fresh seeds 12037, 23039, 34057, each with 384 TRAIN, 96 development and 256
assessment sources. Generate TRAIN/development and fit in seed order. Seal all
45 heads and their fit inputs **before generating any assessment rows**. Report
development without changing anything. Open assessment once, then generate its
three populations in seed order using saved per-seed states and global b set.
Separate engineering seed 40031 (64/16/24) is excluded from scientific metrics.

## Metrics, breakdowns and uncertainty

Every source requests changes to a of -1/4, -1/8, 0, +1/8, +1/4, +2, holding b
fixed. The first five are supported; +2 must abstain because a exceeds [-1,1].
All requested/succeeded/abstained/failed/unresolved counts enter coverage for
each axis/family/seed and split. Unsupported rows have no numerical contrast.
Reference contrasts use noiseless `F(a+h,b)-F(a,b)`, not noisy training labels.

Primary MAE averages the five absolute contrast errors within each source,
then across sources. Report each seed and pooled source-weighted results for
every family and zero baseline, separately for each axis. Predeclared groups
are all sources; a<0 / a>=0; b<0 / b>=0; change magnitude zero / 1/8 / 1/4;
and direction negative / zero / positive. A separate magnitude_stress group
reports refusals with null MAE/intervals; positive-direction coverage includes
the stress requests. The +2 case never enters ordinary MAE. For change groups,
first average the applicable alternatives
within source. Keep all alternatives together under resampling.

Use 1,000 source-block bootstrap draws within each seed, common across all
families/axes/groups, LCG seed 90037, index `floor(next/2^32*N)` (not modulo).
Each draw samples N sources with replacement per seed. Subgroup pooled means
weight sampled subgroup sources equally; report empty draws if any and refuse
an interval when fewer than 975 finite resamples remain. Sort finite draws and
use indices floor(.025*M), floor(.975*M). Report MAE and paired
raw-polynomial-minus-family and zero-minus-family intervals, including per-seed
and pooled groups. Positive differences favor the family. These are conditional
descriptive sensitivity intervals for the frozen fitted models, not training-
process confidence intervals or multiplicity-controlled superiority tests.
Report seed variability separately; retain every unfavorable axis/group.

Prospective synthetic screen: each task-aligned family (raw polynomial on
polynomial, fourier on periodic, hinge on hinge) has MAE <=0.02 in every seed;
all no-change contrasts are exactly zero, all supported requests succeed and all
stress requests abstain, lossy contrasts are zero, and numerical audit error is
<=1e-9. A valid negative or inconclusive screen still resolves this bounded goal.
Interpret apparent gains as nonlinear feature/readout compatibility. Retained
information alone does not establish linear accessibility or encoder value.

## Admission, resources, failures and readout decision

The user's 20 October request explicitly authorizes finishing this assigned goal
through evaluation and decision. Before fitting, the runner registers an exclusive
attempt and snapshots this protocol/RER/configuration, complete execution/audit
source closure and binary. It records actual environment, permissions, exposure
and all state transitions. A permanent single scientific admission and ledger
prevent calling a replay fresh confirmation. Existing/exposed attempts are never
overwritten. Source and model hashes are rechecked across every phase.

Run sequentially through `container.ps1` and the root task entry in the existing
managed container. Native stage timeout 60 seconds, independent audit 300 seconds,
one CPU process at a time, expected peak below 1 GiB. No dependency installations.
Build/binary capsules remain under `/opt/mityane-geometrics`; saved inputs,
predictions, models, source capsule and receipts remain in the exclusive ignored
`.temp/operations/counterfactuals/representation-contribution-v1/<attempt>`.
SIGINT/SIGTERM terminates the current child and records interruption. Recovery
only finalizes an absent original process; it never replays or invents a verdict.
Retain failed attempts and exposure. No score-based retries, post-hoc family
selection or edits to the frozen source are allowed.

After a valid audit, publish all breakdowns and a keep/revise/replace decision
for the OLS binding. A scoped reference method may be kept while revising the
binding policy to require representation/target/population validation and to
withhold universal defaults. Any new revision gets a new identity and TRAIN-only
fit/selection rules. No new profile may depend on the old or revised binding
without linking the decision and its limits. The separately authored NSW profiles
currently declare no OLS dependency. A valid negative or inconclusive result
completes the comparison; an invalid audit leaves the goal incomplete.

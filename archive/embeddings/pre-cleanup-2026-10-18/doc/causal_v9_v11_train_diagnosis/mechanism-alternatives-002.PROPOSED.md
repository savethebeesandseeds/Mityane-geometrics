<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Proposed mechanisms after saved V9/V11 TRAIN diagnosis

Status: exploratory design memorandum only. No candidate, version number, source
change, experiment card, execution, quality gate or application selection is
declared. CPB-v3.market-01 remains the clean baseline and sole DEV encoder.
Preserve every V3–V11 scoped leader, source, checkpoint, result and failure.

This memo reads the completed saved TRAIN diagnosis and its actual passed
independent audit: 68264 checks, zero mismatches. I authored the auditor source;
this note supplies no additional independent validation or result recomputation.
No raw tensor, old model/head, numerical reduction, schema or diagnostic body was
opened or run to prepare it. Numbers below are copied from existing JSON
descriptors/reports.

## What the saved evidence supports

The diagnosis uses one fresh V11 encoder trajectory, its saved initial/final
TRAIN112 exports/readouts, the fixed first8 query capture, actual steps1/2048 and
the recorded moment CSV. V9 comparisons are four cached previously audited TRAIN
subtrees. Three readout repetitions are not three encoder runs. Known
chronological VALIDATION and TEMPO-5 results are reused existing context.

**Volatility.** MKT-HIST-2 · complexity 4/5 (ordinal recipe level) · chronological TRAIN112 · saved final TRAIN R², mean across saved readout repetitions. Linear R²: cached V9 0.9909796360333586, V11 0.9915892992218828. Neural mean R²: cached V9 0.9611169144523425, V11 0.9535589501168299.

**Trend.** MKT-HIST-2 · complexity 4/5 (ordinal recipe level) · chronological TRAIN112 · saved final TRAIN R², mean across saved readout repetitions. Linear R²: cached V9 0.9075641664780169, V11 0.9110286797260279. Neural mean R²: cached V9 0.9524614608176227, V11 0.9382186199146011.

**Activity.** MKT-HIST-2 · complexity 4/5 (ordinal recipe level) · chronological TRAIN112 · saved final TRAIN R², mean across saved readout repetitions. Linear R²: cached V9 0.9753968808764729, V11 0.967674942228831. Neural mean R²: cached V9 0.9731731581851234, V11 0.9699659772782342.

**Relation.** MKT-HIST-2 · complexity 4/5 (ordinal recipe level) · chronological TRAIN112 · saved final TRAIN R², mean across saved readout repetitions. Linear R²: cached V9 0.717685902495712, V11 0.734564930707089. Neural mean R²: cached V9 0.8734249471927815, V11 0.8704845323114275.

JSON selectors:
comparison.saved_TRAIN_readout_scalar_differences, stage=final/metric=r2.
V11 has no broad TRAIN readout improvement over V9: several saved TRAIN scores
are slightly lower, despite the independently established five scoped gains in
the existing quality results. High in-sample volatility and trend scores weaken
the claim that simply adding more readout or encoder capacity is the obvious
missing ingredient. They do not establish a generalization mechanism.

Final all112 TRAIN query MAE is 0.2517975124166082 versus cached V9
0.2445403465007532. Reporting Gaussian NLL is -0.3696593810373693 versus
-0.39245162593123745; mean Huber is 0.0846670254518659 versus
0.08305971606592429. V11's known-VALIDATION MAE/NLL in the existing results report
are 0.320134118 and 0.312348846. Better reconstruction on the saved TRAIN panel is
therefore not a sufficient explanation or selection rule for the established
quality gains. TRAIN NLL remains descriptive and creates no new gate.

All eight mean/centered-variance projection groups have positive recorded mean
weight-gradient and weight-movement norms. For example, bin0's recorded means are
0.008813839776737876/0.002559086183165191 for mean gradient/movement and
0.0012984404674345116/0.0027097263051329696 for centered variance. Smaller raw
gradient units are not smaller importance; recorded AdamW movement is not a
per-coordinate influence estimate. This weakens a "variance never learned" story.

Actual step2048 contextual bin0–3 variance medians are
0.08297605812549591, 0.07635536789894104, 0.08160998672246933,
0.07743926718831062. Their mixture effective-count medians are
5.282577019121241, 5.400323247149342, 6.418424565005893,
6.466856099747874. These are positive, finite, non-one-hot descriptors for one
captured batch, not all112 windows or the whole trajectory. The ordinary route's
upstream is explicitly undefined in this witness; it is not evidence of an
untrained or missing ordinary inference route.

The saved final participation ratios are 3.2767671947152373 for learned20,
1.0488750143207968 for uncertainty4 and 3.941626973135483 for complete40.
These covariance descriptors are neither fitted PCA nor exact ranks. The V9
comparison also has low-dimensional structure, so this is not a V11-specific
diagnosed collapse. Native uncertainty4 is distinct from the intermediate
centered-variance exports.

The fixed saved final volatility ridge decomposition has contribution
correlations to the target of -0.6743042167686012 for learned20,
0.9013002511890265 for mean4 and 0.98185301868604 for uncertainty4. Its learned20/
uncertainty4 contribution correlation is -0.7048588998690646 and mean4/
uncertainty4 is 0.8481828505935791. Those components belong to one joint fitted
ridge prediction; their cancellation is descriptive. They are not independent
block scores, marginal utility estimates or permission to drop a block.

For final all112 volatility, saved reconstruction-MAE versus absolute neural
readout-residual correlations are 0.128158049052141, 0.14800903468802382 and
0.23057175050342602 across the three readouts. NLL correlations are
0.258236924575287, 0.28523315480544836 and 0.2748709117645296.
They are associations between in-sample residuals with shared windows, not
evidence that a particular reconstruction component caused held-out failures.
No new significance test or aggregation was performed here.

## A. Separate the support used by dispersion from learned salience

**First design-review priority because it preserves the complete V11 moment
allocation and changes one kernel-level mechanism. It is not selected.**

Keep V11's exact learned half-local/half-global mean and its four mean-export
rows per bin. For the four centered-variance rows only, use a deterministic
half-local/half-global uniform distribution over the currently valid support:

- local uniform mass is supported on valid original positions within that bin;
  global uniform mass is supported on all valid positions at the current cutoff;
- use the same fixed one-half/one-half mixture and its own corresponding actual
  two-sum mean: one-half local uniform mean plus one-half global uniform mean;
  center variance around that mean, never around the attention-weighted mean,
  and keep the same four existing projection rows/bias;
- use the componentwise sum of uniform-mixture-weighted squared centered hidden
  states: no Bessel correction, epsilon, square root, new parameter or coefficient
  search;
- retain whole-bin zero after bias whenever the local bin is absent, and sanitize
  unsupported hidden values before all arithmetic;
- keep hidden-state derivatives live. The dispersion statistic would no longer
  differentiate through learned score weights, while the learned mean route
  would still train the existing score/position parameters.

This tests whether selection optimized for reconstruction salience is an
unhelpful definition of window dispersion. The saved witness shows finite,
moderately concentrated weighting, not a proven selection failure. The rationale
is a prospective inductive-bias test, weakened by positive variance/movement
evidence and the already high TRAIN volatility scores. It is not a treatment of
an observed dead branch.

Intended shapes, parameter registration, constructor draw order, mean4/
uncertainty4 bank projections and native40 stay unchanged. There is no appended
coordinate, additional learned pool or new fitted statistic. The small extra
uniform-mean arithmetic and changed gradient routes must still be disclosed.
This is distinct from repeating V8 local-only mean pooling, V9's fixed learned
half blend or a search over its mixture coefficient. Do not try several blend
coefficients or support policies against known quality and choose the best.

Uniform weight per valid patch is not uniform weight per observed raw tick or
feature cell, and these learned hidden-state moments are not a direct raw-return
volatility estimate. Deletion/missingness changes support. The shared global
half can still make bin summaries correlated. The two actual witnesses do not
show which events attention suppressed across all112 windows.

The removed direct derivative is from the intermediate variance statistic to
the temporal pool scores/position offsets. Hidden derivatives remain live,
including through the causal Transformer attention inside the hidden encoder.
This statistic is part of the content/mean reconstruction route; it is distinct
from native uncertainty4 and the detached-error conditional-variance NLL branch.
Shared global clipping and AdamW still couple later updates. With F32 actual
two-sum centering, do not silently assume the weighted centered residual is
exactly zero when deriving gradients. Uniform weights themselves depend only on
fixed support/geometry, while their center and hidden operands must remain live.

Risk: attention may be usefully rejecting distractors. Uniform support could
lose the five V11 gains, especially deleted-view timing, activity and relation.
There is no promise of preserving trained V11 coordinates or its gains: a fresh
trajectory changes subsequent shared states. A full frozen all-leader comparison,
rather than a volatility-only claim, is required.

## B. Give a few summary rows a signed temporal contrast

**Different mechanism: represent within-bin direction, rather than change the
sampling weights used by dispersion. Secondary design alternative.**

Keep the V11 hidden encoder, causal learned mixtures and four mean rows. A
single prospectively declared alternative partition could use two centered-
variance rows and two signed original-position/hidden covariance rows per bin,
instead of four centered-variance rows. The contrast uses actual original patch
positions, centered under that bin's current normalized mixture and normalized
by a fixed declared history geometry. It must not use packed rank, future values
or future masks. The sign distinguishes rising and falling hidden-state
structure that an even centered second moment does not express directly.

The existing original-bin means, positional embeddings and causal contextual
states already carry order, so the present representation is not proven
order-blind. High TRAIN trend R² and V11's still weaker known-VALIDATION trend
weaken any claim that "trend information is absent." This hypothesis adds a
direct signed summary bias; the diagnosis does not demonstrate that this is the
cause of the remaining trend/timing failures.

The export tensor can keep its existing shape and native40/head dimensions,
with no extra learned values or constructor draws. However, replacing two
variance rows reduces the degrees of freedom devoted to dispersion. That is a
real allocation confound even at equal parameter count and could sacrifice
activity/volatility gains. A later declaration must freeze one partition and
normalization before quality, document exact gradients/support/empty behavior,
and avoid ratio/lag/contrast searches on known quality. Adding extra coordinates
instead would introduce a separate capacity/head-width confound.

This differs from A: A keeps every V11 moment row and changes support; B keeps
learned support and changes which signed statistic two rows read.

## C. Add a linear residual component at the existing global compression activation

**Different mechanism: change the nonlinear compression route after temporal
summaries; keep the V11 attention and moments intact. Lower priority.**

The current learned global pooling source computes a first affine transform,
GELU, then the existing second affine transform. A parameter-neutral prospective
variant could replace that one GELU activation with GELU(h)+h. This adds a direct
linear component through the same two learned matrices, without another branch,
new parameters, new fitted transform or post-encoder PCA. It would preserve
native40 and the frozen fixed-head shapes.

Low saved learned20 participation ratio and jointly fitted readout cancellation
make compression/conditioning a plausible alternative place to investigate.
They do not prove GELU saturation, dead units, insufficient rank or that a
linear residual will help. No all112 intermediate activation or moment captures
exist to diagnose that explanation. The residual does not bypass the fixed
output-width bottleneck or preserve any old trained coordinate exactly.

This operator also changes forward scale, gradient scale and interaction with
global clipping/AdamW. Those are confounds, not a pure rank intervention. A fixed
unit residual would need one separate declaration with no coefficient sweep.
It could amplify dominant directions rather than preserve weaker ones; it is
therefore less specifically supported than a narrowly declared pooling test.
A learned extra bypass or wider global MLP would add capacity and must be treated
as a different experiment.

## Boundaries shared by any later proposal

No option is admitted by this memo. The actual independent TRAIN audit has
passed; root review/reporting and any next design/source admission remain
separate. Do not describe my reading of that receipt or of the auditor source I
authored as an additional independent confirmation.

Any later chosen mechanism needs a complete owned physical source copy,
separate typed version/policy identity, prospective criteria and actual source/
mask/gradient/learning/resume proofs. Future values or masks after a cutoff must
remain unable to affect that cutoff's output, including absent-bin and poisoned
unsupported cases. New moment/support/contrast adjoints need their own bounded
proofs; historical local-adjoint/learning audits remain preserved and reused.

Retain fresh source_updates=0 initialization, no oracle/pretrained/teacher or
old fitted statistics, identical declared data/targets/TRAIN fitting recipe,
seed/masks/update streams, 2048 successful updates at batch8, global clip1 and
one AdamW unless a separately approved mechanism explicitly concerns those
operators. Keep initial/final saved embedding evaluation and the unchanged
native40 fixed-head recipes. No post-encoder PCA, head tuning, old training/head
replay, checkpoint rescue or repeated seed/budget/rate search is proposed.

The applied strongest V3–V11 protocol protects all established gains prospectively.
Do not rescore old candidates or rewrite their historical 22-condition counts.
Preserve V3 volatility, V4 trend/reconstruction, V6 relation/timing, V7 TRAIN
reconstruction, V8 neural-volatility/timing, V9 timing, V11's five scoped gains
and every other earlier scope. V11 remains its historical 8/22 measured
candidate, not an overall winner or DEV replacement.

Do not repeat V7 split clipping, V10 joint live NLL or the closed V9/V11 recipes
under a renamed seed/rate/coefficient. V4's native36 and later native40 differ
in encoder capacity, head sizes and dimension-derived head seeds; their quality
differences do not isolate architecture or objective alone. These options aim
to retain native40/parameter structure, but equal parameter count is not equal
forward compute, gradient geometry or effective representational allocation.

TRAIN associations cannot choose a guaranteed held-out winner. The historical
VALIDATION windows are already known, and the designed TEMPO-5 charts are not
fresh future-period forecasts. Any proposed option may lose existing gains.

## Exact evidence bindings

- `doc/causal_v9_v11_train_diagnosis/diagnosis.json`: 4218354 B; SHA256 `62f80b93ef4ad7b2a2a994832ab3f2f6921e09f37e96aca441ec283064f3b742`.
- `doc/causal_v9_v11_train_diagnosis/independent-audit.json`: 48403 B; SHA256 `70697acc72fd010b48431516cb15dc1537b615cde5835ed90536cd4de625bc8c`.
- `doc/CAUSAL_FRESH_V11_RESULTS.md`: 29946 B; SHA256 `a24c17710d115ce140f13b0616edc0995f5974fda8e6f748f09874086589778d`.
- `doc/CAUSAL_FRESH_V7_RESULTS.md`: 12345 B; SHA256 `03688509bee4b141c1a303b631273657cea8b856a7ead0c74a6e1aeccd5fecb7`.
- `doc/CAUSAL_FRESH_V10_RESULTS.md`: 28666 B; SHA256 `2cd73cd233c946f07ff0911a704856fd4b1260a5ffd3918b3f777c6aa3fc7c46`.
- `doc/clean_causal_encoder_search_goal_v11_retention.json`: 48667 B; SHA256 `866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b`.
- `code/encoders/causal_patch_bottleneck_mae/versions/v11/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v11/temporal_pooling.h`: 7749 B; SHA256 `6c2eb9081c72d0d7a42f5b6af1b11028af857fcced67a8eaf715155f7c8c2841`.
- `code/encoders/causal_patch_bottleneck_mae/versions/v11/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v11/model.h`: 22683 B; SHA256 `414fa0d486fd8b04fc2d5ff05233f28b3e7d1894154499274e00dc82a86b289a`.
- `code/encoders/causal_patch_bottleneck_mae/versions/v4/README.md`: 5057 B; SHA256 `5a2a0c54c98dc8bde435a483d941d8cfd4f46ba20eddf1a12d028bc45fe8363f`.
- `code/encoders/causal_patch_bottleneck_mae/versions/v5/README.md`: 4837 B; SHA256 `6dc176b0403869d131873dd7a09f2e82f7fb7e703e18c193d42b0e232e655e4c`.
- `code/encoders/causal_patch_bottleneck_mae/versions/v6/README.md`: 4355 B; SHA256 `a2b41bbd2cd3f334d64b072eb4a0e1293c20b389f997b531c84092315abb07a4`.

This additive memorandum changes no source, frozen card, metric, checkpoint,
ledger, catalog, result or DEV file.

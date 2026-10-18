<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Planning handoff: saved TRAIN-only V5/V6 diagnosis

Planning status, 16 October 2026: **No new experiment. This is not a frozen
scientific card, source admission, candidate declaration or executed diagnosis.**
It records the next bounded step for the active clean causal encoder goal. No
new diagnosis code, model, head, fit, training run or quality measurement is
declared here. Preserve all existing scientific cards, reports, sources,
checkpoints, saved embeddings/heads, failed attempts and audit evidence.

Start with the [V6 review](CAUSAL_FRESH_V6_REVIEW.md),
[results](CAUSAL_FRESH_V6_RESULTS.md), [detail](CAUSAL_FRESH_V6_DETAIL.md),
[current clean research ledger](clean_causal_encoder_research_ledger.json) and
[V6 final establishment](causal_fresh_v6/final-establishment.json). The last
receipt distinguishes a trusted measured candidate from overall advancement:
V6 has full trust and coverage but fails the complete numeric retention goal.
The goal remains active. CPB-v3.market-01 remains the clean baseline and sole dev
encoder; this plan changes no application selection.

## Question and comparison boundary

Inspect how the saved TRAIN representations, fixed readout contributions and
reconstruction allocation changed between V5 and V6. In particular, investigate
the volatility, trend, activity and neural-relation losses alongside the retained
timing, linear-relation and TRAIN reconstruction gains. Use TRAIN evidence to
bound a later mechanism proposal, rather than choosing a rescue from held-out
scores.

Both trajectories are already complete: one fresh 2048-update, batch8 CUDA run
per version. Both export native40: learned content20 `[0:20]`, literal
spectral12 `[20:32]`, mean4 `[32:36]` and uncertainty4 `[36:40]`. V6 preserves
V5's forward capacity, fresh constructor draw order, TRAIN calibration policy
and D40 fixed-head recipe/seeds. Its 13 saved whole-native40 DATA/statistic
pairings against V5 are verified in the
[independent quality audit](causal_fresh_v6/quality-independent-audit.json).
This pairing is a starting-state witness, not evidence of equal trained states.

V5 trains the mean and uncertainty paths with Gaussian NLL; its local mean
derivative is inverse-variance weighted. V6 uses fixed unweighted
Huber(delta=1) mean supervision plus detached-error Gaussian variance
calibration, coefficient1 each. The forward Gaussian reporting MAE/NLL stay
unchanged. Shared global clip1/AdamW still couples optimization, and changing
mean residuals changes subsequent variance targets. Direct gradient separation
does not establish independent optimization or explain every observed tradeoff.

## Saved evidence to admit prospectively

A later scientific card must enumerate exact paths, sizes, hashes, tensor
selectors and row identities before arithmetic. The completed
[V4/V5 TRAIN diagnosis](CAUSAL_V4_V5_TRAIN_DIAGNOSIS.md) and its
[owned scalar implementation](../code/research/causal_v4_v5_train_diagnosis/diagnose.py)
are a design reference only. A new implementation would need complete owned
copies, including its inert archive reader, rather than imports from that
historical experiment. No such new source or card is created by this plan.

The intended evidence roles are:

- V5 and V6 initial/final saved TRAIN features, each with 112 native40 rows;
  their shared four-task TRAIN targets, validity masks, row IDs and source groups.
- The same stages' saved ridge and all three neural-head predictions and fit
  archives. Read saved predictions for both head types; use only saved ridge
  coefficients and TRAIN normalization for contribution reconstruction. Do not
  execute the neural heads or refit either head.
- Initial/final fixed-query scalar summaries for all 112 TRAIN rows, and the
  existing first8 TRAIN query archives containing cellwise targets, masks,
  means, variances and native coordinates. These are different scopes.
- Initial/final saved decoder parameters needed for bounded linear residual and
  variance-path calculations. No encoder loading or forward execution.
- V6's actual pre-step batch witnesses for updates1 and2048, their binding JSON
  and corresponding recorded CSV/summary fields. These witnesses are V6-only.
- Frozen cards, source manifests, audits, final establishment and completed
  preservation receipts as metadata/hash bindings. They do not authorize
  numerical access to held-out tensors.

This follows the prior 41-asset common-role template, plus V6's extra component
witness evidence; the final allowlist cardinality and selectors must follow
actual inspected assets. Do not silently extend the scope to raw-cache
calibration, arbitrary training batches or additional checkpoint stages.

Every numeric row must belong to TRAIN, with cutoff `1705881599999`. Require
exact agreement of the 112 row IDs, groups, target validity and declared support
across relevant saved assets. Aggregate query JSON includes validation; numeric
access must be restricted to its `train` subtree. Whole-file hashes and
preservation metadata may cover broader files without opening their held-out
numeric tensors. No TEMPO-5 or market held-out numeric diagnosis is planned.

## Actual schema limits that the card must preserve

The frozen [V5 trainer](../code/research/causal_fresh_v5/train_main.cpp),
`QueryCapture`, `capture` and `query_json` around lines86–112, stores full-population
`row_mae`, `row_nll` and `row_target_cells`. It writes the cellwise witness only
when `offset==0`, so that archive covers the first batch8. The V5 full112 query
summary has no Huber, variance-component or total-objective rows. Those missing
rows cannot be recovered from aggregate MAE/NLL scalars. A hypothetical fixed
Huber(delta1) calculation is possible only on the saved first8 cellwise
distributions and must be labelled hypothetical, never V5's applied objective.

The frozen [V6 trainer](../code/research/causal_fresh_v6/train_main.cpp), around
lines87–118, adds `mean_Huber_F64`, `variance_NLL_F64`, `total_objective_F64` and
their row arrays to the fixed-query summary. Its cellwise query archive still
covers only the first8. Full112 component comparisons therefore cannot be
presented as symmetric V5/V6 evidence.

V6 additionally saves `witnesses/training-update-1-components.pt` and
`witnesses/training-update-2048-components.pt`; the update condition and fields
are around lines202–215, with pins in `optimization-component-witnesses.json`
around lines244–247. They retain tensors from those actual pre-step forwards,
without replay: rows, observed/hidden/context-deleted/visible masks, targets,
eligibility, mean, variance, native40, row/component losses, observed global
gradient norm and clip factor. V5 has no corresponding saved main-batch tensor
witnesses. Do not invent paired V5 batches or replay V5 to obtain them.

The two V6 witnesses use different randomly drawn training batches. They are
not a matched fixed-query curve or evidence for all2048 batch Jacobians. Their
recorded global norm is an observed scalar; checking clip-factor arithmetic
does not independently reconstruct a full parameter-gradient norm. The
[independent training audit](causal_fresh_v6/independent-training-audit.json)
already states the saved-evidence limits and verifies the actual component
reductions.

## Bounded calculations for a later declaration

Use existing diagnostic quantities and fixed recipes only. For the natural-mask
112-row TRAIN feature/head evidence, reproduce per-task saved prediction R²,
physical MAE and RMSE for the ridge and three neural repetitions. Inspect the
chronological early56/late56 subsets without refitting on either subset. Keep
these TRAIN readouts distinct from held-out quality tables and from the
masked-query reconstruction population.

Inspect each of the four native blocks' coordinate scales, covariance,
participation ratio and Pearson associations with TRAIN targets. These are
descriptive saved-feature calculations, not a new readout, post-encoder PCA,
export transformation or selection rule. Reconstruct the existing ridge
prediction using its saved feature mean/scale, target mean/scale,
coefficients and intercept. Decompose that same prediction into the four
blocks and inspect their contributions and covariances. Do not claim that a
contribution decomposition measures performance after deleting/refitting a
block, or that correlation establishes causation.

For reconstruction, keep full112 row-scalar summaries separate from first8
cellwise evidence. Inspect supported errors and inverse-variance pressure per
channel/feature, then the already used local mean/variance adjoints. On a
single saved mean4 basis, compare Gaussian mean-NLL, fixed Huber(delta1) and
MAE derivatives. Treat uncertainty4 as its separate live variance branch.
Independent mean4 and uncertainty4 coordinates have unrelated bases: no
cross-basis gradient cosine, orthogonality or conflict claim. Their derivative
norms are scale-dependent and do not establish full encoder-gradient pressure.

The saved linear `change_decoder[3456,4]` permits bounded residual calculation
from mean4. Subtracting that residual from saved total mean yields an
approximate parent32 remainder, subject to the original F32 addition/rounding.
It does not split content20 and spectral12 inside the nonlinear parent decoder
or recover their parameter Jacobians. Only the fixed ridge contribution
decomposition separates those two blocks directly.

Use pure standard-library scalar arithmetic with an inert, whitelisted archive
reader. Sanitize/promote each saved operand to F64 before subtraction and retain
the frozen supported-cell/channel/example reduction hierarchy. Any scalar
adjoint reconstruction must disclose F32 forward/rounding limitations; it is
not a new CUDA bit-parity proof. No Torch/NumPy, encoder or neural-head forward,
fit, calibration, model Jacobian or additional training is needed or authorized
by this plan.

## Strengths to preserve while searching

Preserve current final leaders and every displaced whole historical record,
with its original data/view/head/metric scope and evidence. The ledger and
frozen per-version tables, rather than an aggregate encoder rank, define these
scopes:

- V3: clean baseline/application selection, linear market volatility and linear
  activity strengths.
- V4: both market trend heads, neural activity, market-scaler deleted neural
  timing and best validation query MAE/NLL; keep its displaced timing and other
  results as historical evidence too.
- V5: neural market volatility and neural relation, with its earlier timing and
  TRAIN MAE records preserved even where V6 supersedes them.
- V6: linear market relation and seven TEMPO-5 quality scopes, plus the lower
  TRAIN query MAE and descriptive TRAIN Gaussian NLL records. TRAIN NLL is
  descriptive, **not an additional retention gate**.

These scope protections do not promote V6 or discard older initial/final
checkpoints, saved heads or source attempts. Historical V4/V5/V6 pass counts
retain their original prospective floor sets and cannot be compared as if the
thresholds were identical. V1/V2/RPB historical encoders supply neither
pretrained initialization nor oracle floors for clean research.

## Later retention declaration and execution boundary

Before any later candidate quality measurement, derive a separately named
prospective strongest-final protocol from all four frozen V3/V4/V5/V6 control
tables. Preserve the [original goal](clean_causal_encoder_search_goal.json) and
[V5-retention goal](clean_causal_encoder_search_goal_v5_retention.json) byte
exact. Retain the same22 criterion IDs, directions, positivity, trust and
coverage rules: protect the maximum final score in each of16 quality scopes,
the minimum TRAIN/validation MAE ceilings and the strict best known validation
Gaussian NLL ceiling. Keep the existing three positivity conditions. Do not add
a TRAIN NLL gate.

Any such reduction must run only in the existing managed container, bind all
four final tables, the ledger, final trust and completed preservation evidence,
and receive independent review before candidate quality. A stronger future
protocol cannot rewrite an old card or retrospectively relabel its pass count.
This plan performs no reduction and creates no replacement goal file.

The next continuation should first inspect exact saved schemas/availability,
write new owned diagnosis source and a prospective allowlisted card, freeze
them and obtain independent static preflight. Root then stages an exclusive
leaf in the existing `cuwacunu` container and authorizes one saved-data numeric
diagnosis, followed by an independent audit. Only that bounded TRAIN evidence
can support a reviewed next mechanism proposal and a separately declared
candidate. No new version, loss routing, coefficient/seed/rate/head/budget
rescue, training run, dev replacement, commit or push is declared here.

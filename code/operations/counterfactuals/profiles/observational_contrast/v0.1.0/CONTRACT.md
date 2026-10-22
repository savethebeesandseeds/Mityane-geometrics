# Matched observational-contrast Operation Contract

| Field | Definition |
| --- | --- |
| Family / owner | OR-06; Hx06, counterfactuals |
| Operation | `matched-observational-contrast` |
| Contract ID / revision | `OR06-MOC`, `0.1.0` |
| Claim level | `observational_contrast`, explicitly requested |
| Primary question | How do observed outcomes differ between a declared focal population and observed comparison cases matched on permitted information? |
| Scope | Association within the declared matched population; no intervention or individual counterfactual claim |
| Status | Authored reusable profile; no implementation or experimental admission established |
| Requirements and evidence | [OR06-MOC-RER 0.1.0](EVIDENCE.md) |
| Governing specification | [ORR-DESIGN-001 0.1.0](../../../../../../doc/operations_over_representations_design_spec_v0.1.0.md), sections 4, OR-06, 6 and 7 |

## 1. Comparison semantics

The input contains actual observations with an observed binary grouping variable
`A`, a focal group `A=1`, a comparison group `A=0`, permitted matching information
`C` and independently defined measured outcome axes `Y`. These group labels
describe observations; the operation neither assigns a treatment nor generates
an alternative. Both groups share the same outcome definition, horizon and units.
An instance declares its observational contrast target and population before
assessment, including whether it describes all eligible focal units or the
supported subset retained by the matching rule.

For supported focal unit `i`, the matching procedure returns comparison IDs
`j` and nonnegative finite weights `w[i,j]` summing to one. For axis `k`,
`delta[i,k] = Y[i,k] - sum_j w[i,j] * Y[j,k]`.
Focal aggregation weights `v[i]`, if used, are frozen independently of assessment
outcomes and sum to one over the explicitly reported aggregate population.
`contrast[k] = sum_i v[i] * delta[i,k]` is an observed matched contrast, not an
average treatment effect. The output records that sign convention and any change
from the initially intended population. No ordinary aggregate is emitted if its
predeclared support, coverage or numerical requirements fail.

No missing unit is a zero contrast. Subset aggregation is permissible only when
the prospective protocol explicitly defines that descriptive target, its weights,
minimum coverage and full-population accounting. Such a result cannot represent
the excluded population. Multiple axes retain separate definitions and units;
categorical labels need a declared numerical outcome mapping, never subtraction
of category IDs. Time-varying exposure regimes, longitudinal estimators and
effects of interventions are outside this revision.

## 2. Logical input contract

Every reference resolves to a versioned record or immutable artifact identity.

| Role | Required binding |
| --- | --- |
| Request | Request/OperationRun IDs, exact contract/RER/instance revisions, explicit claim token, purpose, permissions, code/configuration and evidence destination |
| Population | Source cohort and complete roster; focal/comparison designation; sampling, inclusion, exclusions and duplicate policy; unit/entity/group IDs; outcome observation window and target population |
| Observations | ObservationRefs, source bytes or verifiable retained references, measured `A`, separately versioned outcome definitions and `Y`, quality/missingness flags and lineage |
| Matching inputs | Named covariates and EmbeddingRecords/SpaceRefs; source-variable and information-availability audit; permitted/prohibited variable table; missing-covariate handling |
| Matching procedure | Exact algorithm/revision, metric/geometry, hard strata, calipers, `k` or cardinality, replacement/reuse policy, candidate filtering, weights, tie-breaking, seeds and support thresholds |
| Dependencies | Provider/checkpoint/preprocessing/fit/selection identities, alignment if any, optional propensity/support models, training exposures and uncertainty about upstream exposure |
| Diagnostics | Prespecified balance/overlap/weight/reuse/coverage diagnostics and gates, measured-confounder and omitted-variable checks, estimator and uncertainty procedure |
| Time and mode | Index/exposure/outcome intervals, event time, available-at/vintage for every source and derived dependency, retrospective or as-of mode, cutoff and fit/selection eligibility |

Predicted readout values are not substitutes for measured outcomes in this
profile. If OR-01 supplies candidates, the input records the retrieval version,
candidate restrictions and recall limitations; all exclusions remain visible.
Any OR-10 alignment, OR-05 nuisance score or learned support model is a distinct
upstream dependency with its own fit and evidence restrictions. Composition does
not turn a generated or predicted state into an observation.

## 3. Admissible matching and representation structure

The instance freezes the following procedure before protected outcome assessment:

An archival public benchmark records prior outcome exposure and freezes its new
computation protocol without claiming fresh protected assessment. Such a binding
is explicitly retrospective/reference or exploratory; public exposure cannot be
erased by writing a new protocol.

1. Define the source roster, focal/comparison eligibility and target population
   independently of realized outcome values. Retain exclusions and reasons.
2. Audit every matching input and dependency against an explicit allowlist.
   Use source information measured before the declared index/exposure boundary.
   Prohibit `Y`, outcome-derived variables, future/post-index measurements,
   descendants of the focal grouping/exposure, mediators, design-relevant
   colliders, target leakage, and the group label itself or identifiers coding
   it. A declared propensity score fitted from permitted pre-index covariates is
   allowed; it does not become an identification argument. An exception
   needs a separately revised target/profile; it cannot be silently admitted.
3. Apply the frozen missingness, exact-stratum and candidate rules, then the
   declared distance/score matcher. Record every candidate restriction and actual
   match/weight. No post hoc outcome-guided match selection or threshold tuning.
4. Compute the frozen support and balance diagnostics without outcome-guided
   rematching. Evaluate declared gates; retain a diagnostic even when it fails.
5. Only after matching is sealed, attach measured outcomes, compute supported
   per-unit contrasts and the permitted aggregate, and report uncertainty.

Covariate timing alone is insufficient: source semantics must exclude the
prohibited information, including information encoded indirectly in vectors or
used to fit a representation. A representation with unverified input provenance
cannot be presumed clean. Known/unknown pretraining exposure, group isolation
and all fitting/selection populations remain explicit; the instance states how
each exposure affects admission. Unknown source availability refuses an as-of
claim; a retrospective replacement requires an explicit new scoped request.

Representations must have stable source links and an appropriate, versioned
matching geometry. Bind provider revision, native shape/layout, pooling,
normalization, masks, preprocessing and fitted artifacts. Equal dimensions or
provider aliases do not establish compatibility. Cross-space matching requires
a declared validated mapping. Geometry must have operation-specific evidence
against measured matching needs; proximity does not establish equivalence of
cases, absence of omitted confounding, or intervention identification.

Fit encoders/adapters, scalers, imputation, propensity/support models and all
learned thresholds using only their declared fit partitions. Separate design,
selection/calibration and protected assessment populations with group/time
isolation as required. A fitted-readout dependency requires the separately
recorded [readout-binding decision](../../../../../../web/dashboard/heroes/06-comparisons.json)
before a new instance relies on the existing TRAIN-only OLS binding. This profile
does not itself require that readout or select an encoder.

## 4. Required diagnostics and confounding checks

The instance specifies formulas, denominators, thresholds, estimation units and
actions for at least:

- Measured source-covariate balance before/after matching, using standardized
  mean differences and distributional/categorical diagnostics suitable to each
  variable, including important prespecified interactions. Check these source
  variables separately from distances in embedding space.
- Overlap/support by group and stratum: candidate counts, nearest distances or
  declared propensity-score overlap, caliper exclusions, weight concentration,
  effective sample size, comparison reuse and unmatched focal fraction.
- Missingness, duplicate/shared-source contamination and group/time separation;
  subgroup-specific support and balance, with every denominator retained.
- A documented measured-confounder inventory from source knowledge, a check of
  which factors the representation retains or omits, comparison with raw
  covariate matching, and sensitivity to declared omitted/unmeasured factors.
  Non-applicable negative controls require a reason; plausible controls and
  alternative matching specifications are frozen before their outcomes are seen.

Balance, overlap and sensitivity remain diagnostics rather than
causal-identification certificates. Good measured balance cannot test absence
of unmeasured confounding. Failure of a predeclared mandatory balance/support
gate yields typed abstention; an exploratory diagnostic table cannot be promoted
to a supported association estimate by renaming that gate after inspection.

## 5. Output contract and typed refusal

Every admitted focal unit retains an envelope with request/run/profile IDs,
claim token, source/space/artifact references, group, candidate/match IDs,
weights, outcome axes/units, matching diagnostics, availability boundary and
evidence pointers. Successful envelopes include measured outcomes and the
signed contrast. Derived availability is the maximum required availability
across the matches, outcomes and fitted dependencies unless a tighter
per-output dependency proof is retained.

| State | Meaning / reason codes |
| --- | --- |
| `success` | Valid, supported observed comparison and finite aligned contrast; association label mandatory |
| `abstained` | Otherwise valid unit/aggregate lacks defensible descriptive support: `NO_ELIGIBLE_MATCH`, `OUTSIDE_DECLARED_SUPPORT`, `BALANCE_GATE_FAILED`, `INSUFFICIENT_COVERAGE`, `OUTCOME_UNAVAILABLE`, `UNCERTAINTY_UNSUPPORTED` |
| `failed` | Invalid or unsupported request/dependency: `UNSUPPORTED_CLAIM`, `INVALID_RECORD`, `INVALID_ROSTER`, `INCOMPATIBLE_ROLE_OR_SPACE`, `PROHIBITED_MATCHING_INFORMATION`, `UNVERIFIED_MATCHING_INFORMATION`, `TEMPORAL_VIOLATION`, `FIT_OR_SELECTION_LEAKAGE`, `ARTIFACT_INTEGRITY_FAILURE`, `INVALID_MATCH_OR_WEIGHT`, `UPSTREAM_FAILURE`, `NONFINITE_VALUE` |

Each reason identifies the affected record, violated rule, observed diagnostic
and any permissible correction. Invalid input is never concealed by an
abstention; failures take precedence when both conditions occur. A causal or
controlled-model-response request fails with `UNSUPPORTED_CLAIM`; there is no
implicit fallback or claim downgrade. Refused envelopes omit ordinary numeric
contrasts. Retained diagnostic values are marked diagnostic, not success values.

Admission rejection is recorded before execution without inventing executed
units. Every execution reconciles the full admitted roster into successful,
abstained, failed and unresolved units, and separately retains pre-admission
exclusions. An interrupted run retains unresolved IDs. Aggregate envelopes have
their own state and population/weights; they cannot hide failed units, inflate
coverage by changing the denominator, or treat dependent reused matches as
independent observations. All attempts, including rejected and failed ones,
remain discoverable under the instance's ledger and retention permissions.

## 6. Scope and required downstream restrictions

Use dependence-aware uncertainty for the declared sampling unit, including
comparison reuse, clustering, repeated entities and temporal overlap. Freeze
whether matching is re-fitted/repeated within resampling and justify the
method; a naive independent-pair bootstrap is not a default for reused matches.
Report intervals as sampling/design uncertainty under their assumptions, never
as an uncertainty bound for unmeasured confounding. If an interval is required
but not supportable, the aggregate abstains rather than publishing a nominal one.

Downstream consumers inherit the observed-association label, source restrictions,
population, horizon, support, temporal limits and missingness. The matched cases
need not differ only in `A`. This profile cannot identify an individual's missing
potential outcome, a real-system intervention effect, or the utility of a policy.
The separate [controlled-response contract](../../../CONTRACT.md) remains
unchanged and cannot execute this profile. Implementation, non-synthetic binding,
prospective admission and scientific evaluation are distinct work states.

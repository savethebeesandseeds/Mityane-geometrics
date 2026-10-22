# OR-06 identified intervention-effect Operation Contract

| Field | Definition |
| --- | --- |
| Family / owner | OR-06; Hx06, counterfactuals |
| Operation | `identified-intervention-effect` |
| Contract ID / revision | `OR06-IIE`, `0.1.0` |
| Claim level | `identified_intervention_effect`; explicit and mandatory |
| Primary question | What effect would the specified intervention have on the specified outcome for the declared population and horizon, under the bound identification argument? |
| Status | Reusable profile definition; no estimator implementation, execution admission or scientific result is implied |
| Requirements and evidence | [OR06-IIE-RER, revision 0.1.0](EVIDENCE.md) |
| Governing specification | [ORR-DESIGN-001, v0.1.0](../../../../../../doc/operations_over_representations_design_spec_v0.1.0.md), sections 4, OR-06, 6 and 7 |

## 1. Estimand and claim boundary

An instance MUST define the causal estimand before inspecting its protected
assessment results: target population, analysis and assignment units, treatment
or policy alternatives and their versions, outcome definition and units,
treatment-outcome horizon, aggregation/weighting and any conditional subgroup.
For example, a population-average difference between two explicitly defined
potential outcomes is a different estimand from an effect among treated units,
an assignment effect, a per-protocol effect or an individual effect. An estimate
for one MUST NOT be labeled as another. No individual counterfactual outcome or
individual treatment effect is identified by this profile by default.

The instance MUST supply an explicit identification argument connecting the
observed-data design to that estimand. This includes the causal structure or
design rationale, the observable functional or design-based estimator being
identified, and every assumption on which the connection depends. List which
assumptions can be diagnosed, which are not testable with the available data,
and which remain uncertain. A checked declaration is not evidence that an
untestable assumption is true.

A documented randomized assignment design is permitted. Bind the actual
assignment mechanism, assignment unit, allocation probabilities/strata,
adherence, interference, attrition and outcome observation. An intention-to-treat
effect concerns assignment as randomized; substituting actual received treatment,
conditioning on adherence, or changing the analysis population needs its own
identification argument. An observational or other quasi-experimental design is
permitted only when its own argument identifies its exact requested estimand;
the profile supplies no default exchangeability, instrument, discontinuity or
parallel-trends assumption.

Embedding proximity, measured covariate balance, predictive accuracy, overlap,
model sensitivity and successful controlled-response comparisons cannot supply
the identification argument. Their diagnostics may reveal problems but cannot
certify causal identification. The separately versioned
[OR06-PCR contract](../../../CONTRACT.md) remains a model-response contract; its
implementation and synthetic results do not implement or validate this profile.
An observed outcome may be a declared model score, but then the causal question
is about an intervention on that score-generating system; a score difference
alone still does not identify a real-system outcome effect.

## 2. Logical input contract

Every identity below resolves to a versioned logical record or an immutable
artifact reference. These roles prescribe semantics rather than a file format.

| Role | Required content |
| --- | --- |
| Request and execution | Request/OperationRun IDs; exact contract, RER and instance revisions; purpose, permissions, code/configuration, resource budget and evidence destination |
| Population and roster | Target and sampled populations, assignment/analysis/cluster IDs, complete source roster, sampling/eligibility rules, exclusions, treatment arms and dependence structure |
| Estimand | Exact contrast/functional, intervention versions and comparator, outcome and units, horizon, weights, conditional scope and population to which it applies |
| Design and identification | Actual design/source documentation, assignment or exposure mechanism, identification argument, assumptions and their evidence/limitations, interference/consistency conditions and admissible estimator |
| Observations | ObservationRefs, immutable source snapshot, measurement definitions, actual treatment/assignment and outcomes, missingness/censoring/adherence and record linkage |
| Covariate policy | Named permitted adjustment/matching/stratification variables, their causal roles and availability, prohibited variables and representation-input audit |
| Representation | Separate EmbeddingRecords/SpaceRefs, source dependencies, representation kind, provider/checkpoint or retained response identities, preprocessing/fit scope, native shape, masks, pooling, normalization and geometry where used |
| Estimation and uncertainty | Estimator/revision, nuisance models, matching/weighting rules if used, fit/selection partitions, cross-fitting if required, support/positivity criteria, target-preserving missing-data policy and dependence-aware uncertainty |
| Timing and information | Baseline/eligibility/assignment/treatment/outcome windows, source event and availability/vintage times, decision cutoff if applicable, fitted-artifact exposure and execution availability |
| Provenance and stewardship | Dataset revision, access/license/retention, source and artifact digests, seeds, environment, lineage, prospective protocol, admission and retained attempt records |

Outcome and exposure measurement error, losses to follow-up and selection into
the observed cohort need an explicit identification/estimation treatment. A
missing label is not a zero outcome. Known or possible interference must be
addressed at the assignment/analysis unit with a declared exposure mapping or
otherwise refuse an incompatible estimand. Longitudinal treatment and
time-varying confounding require a corresponding longitudinal design; a single
baseline match cannot silently substitute for it.

## 3. Representation, adjustment and information restrictions

- State the role of the representation: adjustment/matching, precision-only
  covariates, nuisance prediction, outcome measurement or another explicit role.
  Required structure follows that role. For adjustment, document why relevant
  information is preserved and how information loss affects the identification
  argument. Neighborhood quality alone does not establish sufficiency.
- Permitted matching/adjustment variables MUST follow the design-specific causal
  argument and actual information policy. Explicitly exclude outcome-derived or
  future variables, post-treatment descendants/mediators and colliders where
  conditioning on them invalidates that argument. Record exceptions only as part
  of a different explicitly justified estimand/design, never as a convenient
  predictive feature choice. A variable's timestamp alone does not determine
  its causal role.
- Audit all encoder inputs, source windows, pooling/context, learned transforms,
  fit/selection exposure and missingness encodings for prohibited information.
  A pre-treatment row can contain a future revision or an embedding can hide a
  treatment descendant, outcome proxy or collider. Such information does not
  become admissible by compression. Unknown or uninspectable dependencies that
  prevent the required audit cannot be certified as permissible adjustment.
- Distinguish real observed outcomes from predictions and generated potential
  outcomes. A generated or edited observation remains generated; prediction of
  an unobserved outcome is not observation of it. Any outcome readout needs its
  own measurement/validation argument and inherited evidence restrictions.
- Bind actual SpaceRefs, preprocessing and fit artifacts; equal dimensions,
  metric names or provider aliases do not establish compatibility. Combining
  spaces needs a justified mapping or role-specific compatibility argument.
  Observed embeddings, task states and predicted states remain distinct kinds.
- Fit representations, readouts, nuisance models, support thresholds and
  hyperparameters only on their permitted fit/selection populations. Preserve
  known/unknown pretraining exposure. Record whether supervision uses treatment
  or outcome labels and why this is permitted by the design and split policy.
  Hold protected assessment outcomes out of selection. A representation intended
  only for precision in a randomized design still needs that design's valid
  fitting and uncertainty procedure.
- Treatment time, outcome time and information availability are distinct.
  Retrospective causal evaluation may observe outcomes after treatment as
  outcomes; it cannot use those outcomes as pre-treatment matching information.
  Any as-of or policy-deployment claim additionally requires transitive source
  and fitted-artifact eligibility at its cutoff. Unknown availability refuses
  that claim rather than silently changing to retrospective scope.

If an instance depends on the existing TRAIN-only OLS readout binding from the
controlled-response study, it MUST first link the keep/revise/replace decision
required by the Hero's representation-contribution goal. This requirement does
not block drafting a profile or an instance that does not use that binding.

## 4. Support, estimation and complete accounting

An instance MUST define support/positivity at the level required by its
estimand, treatment policy, adjustment set and design. Bind allocation or
propensity support, overlap, effective sample size/weights, extrapolation bounds
and finite-sample precision criteria where relevant. Select thresholds without
using protected assessment outcomes. State which are design assumptions and
which are observable diagnostics. Passing diagnostics does not prove positivity
or eliminate unmeasured confounding.

Matching is optional; it must not be imposed on an otherwise valid randomized
design merely to use representations. If matching, weighting, trimming or
subgroup restriction is used, bind its algorithm, eligibility, calipers,
replacement, weights and target estimand. Dropping unsupported units changes
the population unless the original estimand/protocol already accounts for that
rule. Do not report the mean of surviving units as the original population
effect. If the declared estimand is no longer supported, refuse it; any revised
population/estimand needs an explicit versioned binding and its own evidence.

Retain the complete sampled/admitted unit roster, treatment/assignment and
cluster IDs, predeclared eligibility/exclusion reasons, missingness and
censoring, match/weight/selection membership, support diagnostics and all
execution states. Reconcile the source population, eligible population, admitted
roster, analyzed units and requested estimand. Report denominators and coverage
at every stage and arm. A unit absent from analysis is never silently replaced
by a zero effect or removed from reported refusal/failure rates.

Bind an uncertainty method consistent with randomization/sampling, dependent
units, fitted nuisance models, matching/weights and the estimand. Record its
coverage interpretation and conditions. Distinguish interval uncertainty from
identification uncertainty and untestable assumptions. Use sensitivity analysis
where applicable; an interval conditional on assumptions does not establish
that those assumptions hold. No default numerical estimator or interval method
is authorized by this reusable profile.

## 5. Typed result contract and refusal conditions

Each request yields a result envelope identifying the requested claim and
estimand, contract/instance/run, population and full roster, design and
identification record, representations/geometry, estimator and fit artifacts,
timing, diagnostics, assumptions/limitations and evidence pointers. Unit-level
accounting records do not imply unit-level causal effect estimates. Each
requested aggregate or conditional estimate has one explicit state:

| State | Meaning and permitted payload |
| --- | --- |
| `success` | Admitted design and estimand meet their declared identification, support and integrity conditions; finite estimate, units, uncertainty and assumption-qualified interpretation are present |
| `abstained` | Well-formed request cannot support its requested causal claim under the bound identification/support/measurement conditions; diagnostics and typed reasons are retained without an ordinary causal estimate |
| `failed` | Malformed/incompatible records, integrity or temporal violation, upstream execution failure or non-finite/invalid numerical output prevent valid execution; no ordinary causal estimate |

`success` records satisfaction of the declared checks and the resulting
assumption-qualified estimate, not proof of untestable assumptions or a positive
scientific verdict. Refusals MUST name the affected estimand/units and the
missing or violated requirement. At minimum retain these typed codes:

| Code | Disposition | Trigger |
| --- | --- | --- |
| `UNSUPPORTED_CLAIM` | `failed` | Request is not explicitly for this profile's claim level |
| `INVALID_RECORD_OR_ROSTER` | `failed` | Missing/duplicate/ambiguous identities, inconsistent target/units or absent mandatory fields |
| `INCOMPATIBLE_ROLE_OR_SPACE` | `failed` | Representation role or space is unsupported or lacks required compatibility |
| `ARTIFACT_INTEGRITY_VIOLATION` | `failed` | Changed/unverifiable bound artifacts or inconsistent provenance |
| `PROHIBITED_MATCHING_INFORMATION` | `failed` | Adjustment/matching consumes a variable prohibited by the bound design, including hidden representation inputs |
| `TEMPORAL_OR_FIT_VIOLATION` | `failed` | Leakage or ineligible source/fit/selection information violates the bound mode |
| `UPSTREAM_FAILURE` | `failed` | Required source, encoder, measurement or estimator execution fails |
| `NONFINITE_OR_INVALID_ESTIMATE` | `failed` | Estimate/required uncertainty or numerical diagnostics are invalid |
| `IDENTIFICATION_NOT_ESTABLISHED` | `abstained` | Missing/insufficient argument or unmet identifying assumption prevents the requested causal interpretation |
| `ASSUMPTION_OR_INFORMATION_UNRESOLVED` | `abstained` | Required causal-role, source-availability or dependency evidence cannot be established |
| `INSUFFICIENT_SUPPORT_OR_PRECISION` | `abstained` | Declared positivity/support/effective-sample-size or precision criteria fail |
| `UNSUPPORTED_OUTCOME_OR_INTERFERENCE` | `abstained` | Measurement, attrition, treatment versions or interference prevents identification for this estimand |
| `ESTIMAND_POPULATION_CHANGED` | `abstained` | Dropping/weighting/subsetting units no longer targets the bound estimand |

When multiple conditions occur, retain every reason. A failure takes precedence
over abstention for the same requested result. Admission rejection is recorded
before execution without inventing executed units; interruption keeps unresolved
units/results explicit. Run summaries reconcile all admitted units and requested
results, including failed, abstained and unresolved ones. Partial outputs do not
authorize a complete-population causal claim. There is no automatic fallback to
an association or model response; that needs a distinct request/profile binding.

## 6. Composition, admission and evidence boundary

Upstream retrieval, representations, readouts and matching retain their own
contracts, identities and restrictions. Every pipeline edge states whether it
passes an observed covariate/outcome, generated value, predicted state, weight
or diagnostic. A retrieval score is not a propensity, assignment probability or
causal effect without its separately justified role. Changes to any dependency
require new identities and a compatibility assessment. The final composed causal
claim needs end-to-end evaluation; component tests cannot establish it.

Reusable authoring, non-synthetic instance binding, actual execution admission,
execution disposition and scientific verdict remain separate. A bound instance
must resolve all required fields in this contract and its RER, including actual
data permissions and prospective protocol, before its dependent execution.
Implementation must enforce the typed contract before it is advertised as a
callable intervention-effect tool. This document registers no callable operation,
runs no experiment and promotes no causal claim. Original OR06-PCR evidence is
preserved under its original identity.

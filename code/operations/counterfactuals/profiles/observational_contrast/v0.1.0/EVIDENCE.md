# Matched observational-contrast Requirements-and-Evidence Record

**Record:** `OR06-MOC-RER`, revision `0.1.0`, 20 October 2026.  
**Operation:** [OR06-MOC 0.1.0 / matched-observational-contrast](CONTRACT.md).  
**Purpose:** reusable requirements and evidence boundary. Every actual study
requires a separately versioned instance binding these requirements, its actual
data/artifacts, prospective protocol and execution permissions.

## 1. Hypothesis and allowed claim

The claim token is `observational_contrast`. An instance supplies a
falsifiable hypothesis about whether a specified representation and matching
procedure improve prespecified descriptive comparability, observed contrast
stability or coverage against meaningful controls on a defined population.
It defines the observational contrast, focal/comparison groups, outcome axes,
direction and units, index and horizon, and intended versus retained population.
The hypothesis must distinguish representation contribution from nuisance-model,
readout, information, candidate-pool and fitting-budget differences.

No hypothesis, population, provider or numeric success threshold is selected
by this reusable record. A nonzero observed contrast is not proof of causal
effect or representation quality. A negative or inconclusive valid study can
complete its scoped evaluation without a positive scientific claim.

## 2. Required data and provenance bindings

Bind the actual non-synthetic source snapshot and source revision/license,
access/retention/consent where applicable, raw-data artifact identities or lawful
verification pointers, complete source roster, unit/group/duplicate relationships,
exclusions and focal/comparison designations. Define measured `A` and `Y`
independently of embeddings, and retain missingness, measurement error and
outcome-capture/selection mechanisms. Measurement timing and information
availability include all source and derived dependencies, not event time alone.

The instance includes an allowlist/prohibition table for every matching covariate
and representation input, with source meaning, exposure/index/outcome timing,
possible descendant/collider/mediator status and access mode. Audit hidden
outcome or grouping leakage through text, windows, metadata, embeddings,
normalization, provider training and candidate filtering. Record unknown
provider/pretraining exposure and its admission implication rather than assert
that a frozen or external encoder is uncontaminated.

Bind provider/checkpoint, preprocessing, native representation layout, masks,
SpaceRef and geometry, fit/selection exposures, optional score/support model,
alignment, candidate retrieval and all artifact hashes. Observation embeddings,
task states, predicted states and generated data retain distinct kinds and
parents; generated observations cannot count as non-synthetic source cases.
Any shared fitted OLS dependency needs the evidence-backed keep/revise/replace
decision before use; the profile does not require or endorse that dependency.

## 3. Controls and baselines

Select controls before assessment and justify omitted or inapplicable ones:

- The unmatched observed difference, clearly reporting its different population
  or information basis rather than treating it as the truth.
- Raw measured-covariate exact/stratified or conventional distance matching,
  and a conventional propensity-based design when meaningful and admissible.
- Simple descriptors and random/alternative representation families with
  matched accessible information, fitting/selection and candidate/compute budgets.
- A sham/no-change or duplicate-safe engineering control of contrast arithmetic,
  matching/weighting and output units; invalid-input and no-support cases.
- Plausible negative-control exposures/outcomes or falsification checks with a
  source-specific rationale. Passing a negative control cannot verify that
  all relevant confounders were observed.

Assess source-covariate balance and support under each design, not just embedding
distances. Make changes in target population and successful coverage explicit.
A representation that looks favorable only after dropping a difficult population
cannot be declared better over the original roster without an appropriate test.

## 4. Prospective evaluation and admission

Freeze the actual source/population, observational question, group/index/outcome
definitions, permitted covariates, matched target, algorithm and all thresholds,
seeds, exclusions/missingness, candidate pool, matching replacement/reuse and
weighting, uncertainty procedure, primary metrics, controls and decision rules
before protected assessment. Match/candidate selection cannot inspect assessment
outcome values. Seal matches before outcome attachment. Retain all exposure and
selection history; multiple looks and exploratory post hoc analyses are labeled.

Separate fit, development/calibration and assessment by the actual dependence
unit and chronology. State whether a design is retrospective or as-of, and check
the full information/fit dependency graph. Fit transformations and nuisance
models on permitted data only. Assessment outcomes remain evaluator-only.
Predeclare what information a design-stage balance inspection may use and how
it affects the final protected evaluation; balance-guided redesign is a logged
new selection step, not an invisible change to a frozen assessment.

The bound instance supplies actual lawful permissions, resource budget,
environment/code/configuration identity, admission checks and a pre-execution
attempt record. Missing required inputs, an unverified prohibited-information
audit or temporal leakage block admission. An operation-local ledger can provide
discovery pending common infrastructure. Preserve rejected, failed, interrupted
and abstained attempts with distinct dispositions and permitted evidence.
Reusable authoring and instance binding grant no execution admission themselves.

## 5. Measures and acceptance decisions

Engineering conformance requires correct source/axis pairing, match weights and
contrast sign/units; fit/temporal/integrity checks; typed refusal; preserved
lineage; and complete requested/eligible/admitted/result denominator accounting.
No implementation or conformance result exists for this profile yet.

Before execution the instance must select quantitative measures and thresholds
for its hypothesis. Required reporting includes the intended and actual matched
population; source-variable balance before/after matching; overlap, caliper and
coverage diagnostics; effective sample size and comparison reuse; per-axis
contrasts with valid uncertainty; all failure/abstention rates; and matched-budget
baseline differences. A diagnostic-only study can omit an ordinary contrast only
when its prespecified question and output explicitly say so.

Matching success, a small distance, statistically nonzero contrast, high coverage
or improved balance alone cannot certify operation utility, absent confounding
or causal identification. A scientific verdict is `supported_in_scope`,
`inconclusive` or `refuted_in_scope` under the frozen question; it remains
separate from execution disposition and authoring/engineering completion.

## 6. Robustness, confounding and uncertainty

Prespecify checks across relevant source populations, outcome axes, strata,
seeds, geometry/representation versions, missingness and shift. Compare declared
matching specifications and caliper boundaries without outcome-driven cherry
picking. Report measured confounder balance and representation retention, plausible
omitted/unmeasured-factor sensitivity and negative controls with their limits.
Retain negative, inconclusive and support-failure findings.

Define the independent sampling unit and dependencies created by repeated units,
shared sources, overlapping windows, clustered entities and reused controls.
Use a justified dependence-aware interval or resampling method, including whether
the matching design is recomputed, with fixed repeats and seed policy. Separate
sampling uncertainty from design sensitivity, provider stochasticity and
unmeasured confounding. An ordinary confidence interval cannot bound all four
by default. Mandatory unsupported uncertainty causes aggregate abstention.

Diagnostics of balance, overlap and sensitivity remain diagnostics rather than
causal-identification certificates. The matched observed cases may differ in
unmeasured causes or measurement processes; the strongest permissible conclusion
is an association for the recorded population and matching design.

## 7. Current results and limitations

**Current evidence:** reusable contract and RER authored. The
[NSW archival instance](../../instances/nsw_reference_v1/OBSERVATIONAL.md) binds
an actual non-synthetic source, raw descriptor, protocol and permissions. There
is no OR06-MOC implementation, registered CLI adapter, matching execution,
scientific result or intervention-effect evidence in these records. Published
outcome exposure is retained; the binding is not fresh protected assessment.

The existing [OR06-PCR evidence](../../../EVIDENCE.md), controlled-response
reference tests and frozen synthetic response study address another claim level.
They cannot establish observational comparability or validate this profile.
The synthetic mixed/raw-polynomial MAE tie motivates separately planned
representation-contribution work; it provides no observed matched-contrast
finding and does not select a representation or readout here.

Authoring can be accepted independently of instance binding. Before any scientific
claim, at least one actual non-synthetic instance must bind this contract/RER,
its actual source permissions, hypothesis, matching design and artifacts; its
admitted execution and evidence must then support that claim. Merely binding
an instance or passing engineering checks is insufficient.

## 8. Reproduction pointer and evidence preservation

For authoring, retain this RER and [contract](CONTRACT.md) revision. For a study,
retain the exact bound versions, source/code/configuration digest manifest,
dataset and SpaceRefs, source roster/exclusions/matches/weights, fitted artifacts,
protocol/admission and attempt IDs, environment, seeds, raw diagnostic/output
artifacts, disposition, scientific verdict and reproduction commands.

Use the authoritative [managed environment](../../../../../../doc/ENVIRONMENT.md).
Keep operation-owned code and selected evidence with this component and ignored
runtime artifacts in the prescribed container/repository locations. Preserve
original bytes/identities; corrections create new superseding records rather
than rewrite prior evidence. Propagate access, retention, information horizon,
claim level, geometry, population, support and missingness to downstream users.

## 9. Decision

**Reusable profile authored; scientific suitability unestablished.** No execution
is authorized or reported by this document. Its contract/RER define the remaining
instance, engineering, admission and evidence requirements without resuming or
relabeling the completed controlled-response study.

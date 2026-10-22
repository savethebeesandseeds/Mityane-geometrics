# OR-06 identified intervention-effect Requirements-and-Evidence Record

**Record:** `OR06-IIE-RER`, revision `0.1.0`, 20 October 2026.  
**Operation:** [OR06-IIE / identified-intervention-effect, revision 0.1.0](CONTRACT.md).  
**Purpose:** reusable requirements for an assumption-qualified causal claim.
An instance supplies actual bindings and a prospective protocol; this profile
record is neither an admitted experiment nor evidence of a real intervention
effect. Governing requirements are
[ORR-DESIGN-001 v0.1.0](../../../../../../doc/operations_over_representations_design_spec_v0.1.0.md),
sections 4, OR-06, 6 and 7.

## 1. Hypothesis and interpretation

An instance MUST state a falsifiable question for its declared intervention,
outcome, population, horizon and estimand. A representation-contribution claim
also states what the representation is expected to improve over equal-information
controls: precision, estimation error, measurement validity or another defined
role. Better prediction, visually plausible matches or a nonzero contrast alone
cannot establish an identified effect or that representations are useful.

Bind a complete identification argument independently of any observed effect
estimate. Record consistency/treatment versions, interference, assignment or
exchangeability conditions, support/positivity, selection/missingness and every
other design-specific assumption. Label untestable assumptions as untestable;
diagnostic checks and sensitivity analyses are not proofs of them. Randomized
assignment may identify its specified assignment effect when the actual design
and outcome observation support it. Other designs and received-treatment or
individual-effect claims require their own adequate argument.

## 2. Data, design and provenance

The instance binds the actual dataset/source revision, license/access/retention
and execution permissions, source manifest, immutable ObservationRefs,
assignment/analysis/group identities, complete roster and exclusions, treatment
definitions/assignment records, outcome measurement, eligibility, missingness,
adherence, attrition and interference. Preserve the sampled population and the
target estimand separately; reporting only a successful matched subset is
insufficient.

Bind the actual encoder/provider revision or retained-response identity,
SpaceRefs, source inputs/windows, representation kind, geometry, preprocessing,
native layout, fit/selection populations and artifact hashes. Record known or
unknown pretraining exposure, hosted-revision uncertainty and any independent
measurement/readout validation. Treatment-outcome timing and information
availability/vintage are separate fields. An explicit retrospective mode does
not permit outcome leakage into matching. An as-of claim inherits every source
and fitted-dependency cutoff.

For every variable/representation, record its proposed adjustment, precision,
measurement or other role; the causal reason that role is permitted; and a
dependency audit for prohibited outcome-derived, post-treatment, future and
collider information. Match/adjustment eligibility cannot be asserted from
vector dimensions, an event timestamp or an encoder's reputation. Unknown
information that prevents the required audit blocks the dependent claim.

## 3. Controls and baselines

Before protected assessment, select controls with the same estimand, admissible
information and valid design. Justify non-applicability rather than imposing a
control that changes the estimand:

- A design-appropriate conventional estimator, including an unadjusted
  design-based assignment contrast where that actual randomized design permits
  it, and a prespecified raw/simple-covariate estimator.
- Capacity/fit-budget/information-matched alternative or random representations
  where meaningful, with their identities and permitted source exposures.
- Covariate balance, support/overlap, weight/effective-sample-size and attrition
  diagnostics, including arm/group breakdowns and predeclared refusal thresholds.
- Design-appropriate negative controls, placebo/falsification checks and
  sensitivity analyses; state their assumptions and what each can detect.
  Passing one cannot establish absence of unmeasured confounding or prove the
  entire causal argument.
- Engineering malformed-record, forbidden-variable, support, changed-estimand,
  uncertainty and complete-roster refusal cases. Artificial fixtures establish
  mechanics only; simulations with known effects are labeled synthetic.

A treatment-label permutation or placebo is not automatically a valid causal
test for every design. Bind when it is justified, its null and assignment/dependence
structure. Ground-truth causal effects are generally not observed in a
non-synthetic instance; compare appropriately identified estimators and uncertainty
without declaring a preferred answer to be truth.

## 4. Evaluation and admission protocol

Before any dependent execution, bind actual inputs, permissions, the causal
argument, source/availability audit, implementation and estimator revision,
source/environment/configuration hashes, resource budget and retained attempt
record. Resolve admission blockers explicitly. Reusable authoring and a dataset
binding alone confer no execution admission.

Freeze treatment/outcome definitions, eligibility and target population,
estimand, permitted/prohibited covariates, matching/weights/support rules,
fit/calibration/assessment partitions, tuning policy, uncertainty, hypothesis
tests if any, subgroup analyses and primary success criteria before protected
assessment. Fit only on permitted populations; respect assignment, repeated
entities, clusters, temporal boundaries and source overlap. Nuisance estimation
uses cross-fitting or another explicitly justified design when needed, without
assuming any one estimator is mandatory. Selection must not exploit protected
outcomes. Log prior exposure/repeated looks; changed protocols need fresh
protected confirmation or an explicitly exploratory scope.

Predeclare how missing outcomes, censoring, nonadherence, interference and
unsupported units affect identification and the estimand. Trimming or exclusion
cannot silently change the requested target. Retain all excluded, failed,
abstained and unresolved units with reasons and denominators. Report effects
only when the bound estimand remains supported; otherwise retain the typed
refusal. A new target population needs a versioned rebind and claim.

Finalize every admission rejection, execution failure/interruption, abstention
and completed evaluation with its actual disposition and artifact limitations.
Do not reinterpret failed execution or rejected admission as a scientific null.

## 5. Measures of success

Engineering conformance requires explicit claim/estimand binding, compatible
record roles and identities, permissible inputs/timing, verified artifacts,
correct estimator/units, valid uncertainty, typed refusals and complete unit/run
accounting. A document review is authoring evidence and supplies no numerical
test result.

Scientific criteria MUST be instance-specific and predeclared. Report the
identified estimate with uncertainty and the assumptions/identification scope,
coverage, exclusions, support and effective sample size. A valid null or
inconclusive causal estimate may complete a study; a positive effect is not
required. Testing representation contribution additionally needs predefined
comparison criteria against the controls, with equal information and matched
capacity/budgets as applicable. Report failure to improve and wider intervals.
No threshold of balance, predictive accuracy or proximity is a
causal-identification certificate. A statistically significant effect does not validate
an invalid design or establish downstream decision utility.

## 6. Robustness and uncertainty

Bind design-appropriate uncertainty at the assignment/sampling/dependence unit,
including fitted models, weights/matching and multiple comparisons. State the
interval's target, coverage conditions and limits; distinguish finite-sample
uncertainty, provider/representation variation and identification uncertainty.
Retain sensitivity to unmeasured confounding, alternative causal assumptions,
support/trimming, missingness, treatment versions, interference, geometry and
representation family where applicable. If a check is not applicable, explain
why. Sensitivity results limit the claim rather than verifying an untestable
assumption.

Predeclare population/arm/group and outcome-axis breakdowns, weak-support
regimes and shift tests. Exploratory analyses are labeled and require fresh
confirmation for stronger claims. Report negative and inconclusive cases and
how much of the original requested population each result covers. A new dataset,
encoder, treatment/horizon or domain does not inherit the old result by name.

## 7. Results and limitations

This revision supplies the reusable contract and evidence requirements. It
does not supply an estimator implementation, conformance execution, scientific
verdict or claim of an identified real-system effect. Actual non-synthetic
instances and any later results must be distinct linked records with their own
artifact identities, protocol/admission status and limitations.

The [NSW archival instance](../../instances/nsw_reference_v1/INTERVENTION.md)
now supplies an actual-source binding, raw baseline descriptor and explicit
conditional identification argument. Missing design/uncertainty information is
retained and causal execution remains unadmitted; there is no estimated effect
or fresh protected assessment claim.

The original [controlled-response evidence](../../../EVIDENCE.md), its C++
fixtures and frozen synthetic results support their own model-response scope.
They supply neither identification nor empirical validation for OR06-IIE.
Neither a general-purpose provider boundary nor a balanced neighborhood waives
the causal input/design requirements. The present profile cannot certify its
own untestable assumptions and does not claim all requested effects are
identifiable.

## 8. Reproduction pointer

For authoring, retain this record and the linked contract at revision 0.1.0.
An instance/run must retain the exact profile and instance revisions, dataset
snapshot/access pointer, roster and exclusions, assignment/design documentation,
identification argument, covariate audit, SpaceRefs and artifacts, source and
configuration hashes, fit/selection and assessment manifests, environment,
seeds, estimator/uncertainty settings, logs, typed results and all attempt
dispositions. Public evidence may reference access-controlled data; it must not
publish source records without permission. Reproduction and build execution
follow the project's managed Linux container and existing root entry points.

## 9. Decision and change policy

**Reusable profile authored; empirical support remains instance-specific.**
Authoring review, binding acceptance, execution disposition and scientific
verdict must be recorded separately. Promotion requires an actually admitted
execution and sufficient evidence for its stated causal/representation claim;
negative or inconclusive evidence is retained. Contract changes, corrections
and supersession preserve earlier versions and evidence identities. No new
instance may silently reinterpret the existing controlled-response study.

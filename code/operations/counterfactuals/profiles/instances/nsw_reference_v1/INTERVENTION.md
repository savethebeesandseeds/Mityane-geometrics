# NSW assignment-effect instance

**Instance:** `OR06-IIE-NSW-REF-001`, revision `1.0.0`.  
**Instance RER:** `OR06-IIE-NSW-REF-RER`, revision `1.0.0`.  
**Profile:** [OR06-IIE / OR06-IIE-RER 0.1.0](../../intervention_effect/v0.1.0/CONTRACT.md).  
**Disposition:** actual-source binding complete; causal execution not admitted; no estimated effect.

## Estimand, source design and identification argument

The proposed estimand is the average effect of **assignment/offer** of the NSW
job-training intervention on real 1978 earnings, in thousands of dollars, for
the finite retained 445-person Dehejia-Wahba reference cohort. It is an ITT
assignment effect, not an effect of actual training received, an individual
effect, an effect among all original NSW participants or an effect in a new
labor-market population. Treatment arms use the actual `train` assignment field.

The maintainer documents randomized assignment; the original authors document
subsetting using pre-intervention information. The identification argument is:
if assignment remains exchangeable with potential outcomes within this retained
cohort, each arm has positive assignment probability, outcome measurement/retention
does not create differential selection, and consistency/no-interference hold,
then the two arm-specific observed outcome means identify the corresponding
assignment potential-outcome means. Their difference identifies this cohort's
average assignment effect under those assumptions. Randomization and the
retained-cohort selection argument supply this connection; **proximity,
matching balance and model sensitivity do not supply it**.

Explicit, not empirically certified assumptions are:

1. The recorded assignment indicator and outcomes faithfully represent the
   documented study; treatment versions/assignment-to-offer semantics are stable.
2. Retention based on pre-intervention variables preserved the required assignment
   comparability; no arm/potential-outcome-dependent selection invalidated it.
3. Assignment has positive support for both arms in the target; any actual
   stratification or unequal probabilities is handled by the eventual estimator.
4. Consistency and no relevant interference at the person-level analysis unit;
   no unrecorded clustering or spillover invalidates the uncertainty argument.
5. Observed 1978 earnings and source retention/missingness support the estimand;
   unavailable records are not zero and any attrition bias is addressed.

The export does **not** supply original individual allocation probabilities,
stratum/site/cluster labels, assignment dates, compliance histories or the full
original recruitment/attrition roster. These fields are explicitly unavailable.
The observed ratio 185/445 is not a known individual assignment probability.
This binding supplies an assumption-qualified argument and its open requirements;
it does not certify them or admit an estimator needing missing design details.

## Actual source, representation and timing

[bindings.json](bindings.json) pins the actual dataset/license/design documentation,
source roster, raw `[re74]` descriptor and SpaceRef. Requested analysis roster
is all 445 retained rows: 185 assigned, 260 controls, no outcome-based exclusion.
Source row IDs are archive locators, not recovered subject or cluster IDs.

The raw 1974 earnings descriptor is a baseline diagnostic/possible precision
input only. The proposed primary mean contrast does not adjust or match on it;
it supplies no identification and asserts no encoder contribution. No fitted
readout, OLS, scaler or nuisance model is used. Any future adjusted estimator
requires a separately frozen role and valid design argument.

Prohibit assignment/group labels as matching features; `re78`, `unem78`,
`lre78`, `mostrn`, adherence/received-treatment measures, outcome-derived,
post-assignment/future variables, colliders, descendants and hidden embedding
versions of them. Do not condition on actual training duration to rename ITT
as a received-treatment effect. Only source-justified pre-intervention
information may enter a separately declared precision/adjustment model.

1974 earnings precede intervention according to the source documentation;
1978 earnings are the downstream outcome. Exact individual assignment, support
and information-availability times are unknown. Mode is retrospective archival
analysis with no live/as-of decision claim. Published outcome exposure is logged;
this is not fresh protected confirmation or a new trial.

## Frozen reference protocol and refusal gates

The proposed point functional is the unadjusted observed assigned-minus-control
mean of `re78`, retaining the full roster. This is the conventional design
baseline; it is not ground truth for an unobserved individual effect. The
predeclared reference question is the two-sided null of zero cohort-average
assignment effect, conditional on the identification/design assumptions above.
No positivity of the estimated effect is required for a valid eventual study.

Before any causal execution, an admission review must justify an estimator and
uncertainty method against the actual allocation/dependence/retention evidence.
A conservative Neyman mean-difference variance and approximate 95% interval are
candidate methods only under a justified compatible randomization design; they
are **not admitted** by the available arm counts. No exact randomization test,
known-probability weighting, cluster interval or arbitrary independent-person CI
is authorized from missing design metadata. If the needed design cannot be
established, return `IDENTIFICATION_NOT_ESTABLISHED` or
`ASSUMPTION_OR_INFORMATION_UNRESOLVED` without an ordinary causal estimate.

Admission also checks both arms, finite aligned outcomes, complete roster and
source/representation integrity, recorded missingness and outcome retention,
prohibited-information audit and requested estimand preservation. Report baseline
balance/support and all source limitations as diagnostics, not identification
certificates. Any required positivity/precision failure abstains; dropping
unsupported rows cannot silently redefine the 445-person estimand. A revised
population needs a new binding. Negative controls/assignment permutations need
actual design justification and are not currently authorized by this record.

No fitting/calibration partition is used; no learned component is bound.
All source rows belong to an exposed archival reference cohort. A later learned
or representation-comparison study needs new fit/selection/evaluation bindings;
this instance cannot inherit synthetic OLS adequacy or claim fresh confirmation.

## Permissions, evidence and decision

Present permission covers source acquisition, schema/artifact checking,
baseline-descriptor materialization and instance authoring checks only. No new
matching, effect estimation, training, provider/API calls or scientific execution
is admitted. An actual estimator, code/configuration/environment identity,
design/uncertainty resolution, resource budget, evidence destination and
pre-execution attempt record are required before any dependent run.

All [profile refusal types](../../intervention_effect/v0.1.0/CONTRACT.md) apply.
Preserve every requested unit and aggregate state, exclusions and limitations;
failures take precedence over abstention and interruption retains unresolved
records. A refusal is neither a zero effect nor evidence of no effect. Scientific
verdict and execution disposition remain separate, with every attempt retained.

**Current decision:** bound non-synthetic archival instance with explicit
identification/design limitations; no causal result. The actual data receipt,
profile/RER versions, source/representation pins and this protocol are the
reproduction pointer. Binding completion does not resolve missing design evidence
or supply scientific admission.

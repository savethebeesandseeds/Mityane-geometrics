# OR-06 Operation Contract

| Field | Definition |
| --- | --- |
| Family / owner | OR-06; Hx06, counterfactuals |
| Operation | `paired-controlled-response` |
| Contract ID / revision | `OR06-PCR`, `0.1.0` |
| Claim level | `controlled_model_response` only; mandatory, with no implicit default |
| Primary question | How does a fixed numerical model response change between a baseline and a specified alternative? |
| Status | Reusable interface definition; implementation and study admission remain separate |
| Requirements and evidence | [OR06-PCR-RER, revision 0.1.0](EVIDENCE.md) |
| Governing specification | [ORR-DESIGN-001, v0.1.0](../../../doc/operations_over_representations_design_spec_v0.1.0.md), sections 4, OR-06, 6 and 7 |

## 1. Boundary and comparison semantics

A request supplies recorded baseline/alternative responses from the same frozen
response pipeline, together with their source, representation and execution
references. This operation validates those relationships and computes paired
contrasts. Producing the alternatives, encoding inputs and executing the response
model are upstream operations, separately bound in the run record.

For pair `i` and numerical response axis `k`, the contrast is
`delta[i,k] = alternative[i,k] - baseline[i,k]`, in the declared response units.
Both sides use the same response definition, units, output selector and fitted
pipeline. A categorical response needs a declared numerical score definition;
subtracting category IDs is unsupported. No relative effect, derivative,
population average or significance claim is implicit in this contrast.

The supported claim concerns model behavior for the specified alternatives and
population. Observational differences and identified real-system intervention
effects have different meanings. Requests for either claim level are rejected
by this revision. A model score change does not identify an individual person's
or system's otherwise-unobserved outcome.

## 2. Logical input roles

These are logical fields, independent of implementation language or file format.
Every identity resolves to a versioned record or immutable artifact reference.

| Role | Required content |
| --- | --- |
| Request and run | Unique request/OperationRun IDs; exact contract and RER revisions; purpose, permissions, code/configuration, evidence destination and resource budget |
| Claim and population | Explicit `controlled_model_response`; population definition, complete input pair roster, unit/entity/group IDs, sampling rule, response horizon and permitted interpretation |
| Baseline | ObservationRef and source identity; baseline input; representation record/kind, SpaceRef, masks and output positions; response record and response execution identity |
| Alternative | Parent baseline ID; alternative input and representation/response records; construction kind (`edited_input`, `simulated_input` or `latent_candidate`), procedure/revision and validity evidence |
| Change definition | Attributes/coordinates changed, baseline and alternative values, units, allowed bounds, intended held-fixed attributes, actual change mask/diff and any unavoidable coupled changes |
| Response pipeline | Provider/checkpoint or retained API-response identity, preprocessing and fitted transforms, response model/readout, output selector/axes/units, stochastic policy, artifact digests and fit/selection populations |
| Comparison parameters | Exact one-to-one pair keys, axis alignment, change-validity tests, declared support region and thresholds, numerical precision/tolerance, handling of stochastic repetitions and failure policy |
| Timing and provenance | Source support/availability and vintage; response and representation parents; retrospective or as-of mode and cutoff; fitted-artifact exposure/availability; justified non-applicability where relevant |

An input may refer to a vector, structured representation or distribution when
the selected response function supports it. Observed embeddings, task states,
predicted states, transformed candidates and generated content retain distinct
kinds and parent lineage. An alternative derived from an observation retains
that parent link and its derived kind; it does not become another observation.

## 3. Pairing, representation and dependency requirements

- Pairing is explicit source-parent pairing, not nearest-neighbor matching.
  Declare the source population and eligibility/exclusion rules before response
  inspection. Missing, duplicate or ambiguous pair keys fail validation.
- The change definition must be valid at the claimed input level. Retain the
  actual changed and held-fixed values. An unconstrained latent edit supports
  only the declared coordinate/model-response interpretation unless independent
  evidence grounds its source-level meaning. There is no default semantic axis.
- Both sides must be supported by the declared response input role and versioned
  geometry. Bind provider revision, preprocessing, fitted transforms, native
  output layout, pooling, normalization and metric where used. Equal dimensions
  or family names are insufficient. A directional mapping, if needed, is an
  explicit fitted upstream dependency with source/target roles and lineage;
  it cannot be assumed from this contract.
- Keep the response pipeline fixed across a pair. Record provider uncertainty,
  stochastic settings, repetitions and any uncontrolled differences. Unknown
  hosted revisions or changed readout/checkpoint artifacts cannot support a
  controlled single-change assertion merely because a model alias is unchanged.
- Predeclare perturbation bounds and validity/support tests. A valid input
  outside the supported region yields abstention; observed output plausibility
  alone cannot establish that the alternative is meaningful.
- Fit preprocessing, encoders/adapters, readouts, support models and thresholds
  only on permitted fit populations. Record pretraining/selection exposure as
  known, unknown or excluded by design. Hold assessment outcomes out of input
  construction, response fitting and alternative selection.
- For as-of claims, source dependencies, transformations, reference artifacts
  and allowed fit/selection information must respect the cutoff. Alternative
  target time is distinct from its information origin. Future outcomes enter
  scoring only after responses are fixed. Unknown availability requires refusal
  of the as-of request; any retrospective replacement needs its own explicit
  request and recorded scope. Non-temporal use needs a recorded justification.

## 4. Outputs and result states

Each requested pair produces a result envelope, including refused pairs. It
contains contract/run/request/pair IDs, requested claim level, source and parent
references, SpaceRefs and representation kinds, response pipeline identity,
change definition, response axes/units, diagnostics, availability boundary and
evidence pointers. Successful envelopes include both responses and `delta`.
The output availability boundary is the maximum required availability across
both branches and their fitted dependencies, unless a tighter per-output
dependency boundary is proven and retained.

| State | Meaning and payload |
| --- | --- |
| `success` | Supported pair, finite aligned responses and finite contrast; validation diagnostics retained |
| `abstained` | Structurally valid pair whose declared validity/support/interpretability criteria are not met; no ordinary numeric contrast |
| `failed` | Malformed or unsupported request, incompatible dependencies, integrity/temporal violation, upstream execution failure or non-finite numerical output; no ordinary numeric contrast |

Reason codes identify the affected record and diagnostic: `UNSUPPORTED_CLAIM`,
`INVALID_RECORD`, `INVALID_PAIRING`, `INCOMPATIBLE_ROLE_OR_SPACE`,
`STALE_OR_CHANGED_PIPELINE`, `UNVERIFIED_CONTROL`, `TEMPORAL_VIOLATION`,
`UPSTREAM_FAILURE`, `NONFINITE_RESPONSE_OR_CONTRAST`, `INVALID_ALTERNATIVE`,
`OUTSIDE_DECLARED_SUPPORT` and `UNSUPPORTED_INTERPRETATION`. The last three are
abstentions for otherwise valid requests; other listed codes are failures.

Admission rejection is recorded before execution and does not invent an executed
pair. Interrupted execution retains the original roster and records unresolved
pairs explicitly. Run summaries reconcile requested, successful, abstained,
failed and unresolved counts; rates use the full admitted denominator. No pair
silently disappears or becomes a zero response. This revision allows partial
results only as individually labeled envelopes; it has no fallback or automatic
claim downgrade. Summary estimators and uncertainty require a prebound protocol
and cannot silently exclude unsuccessful pairs.

## 5. Inherited rules and future claim profiles

This contract inherits the following requirements from the
[Hx14 record](../../../web/dashboard/heroes/14-foundations.json). Their
implementation status remains independent of this authoring milestone.

| Shared requirement | Application here |
| --- | --- |
| `foundations-record-validation` | Distinct record kinds and identities, operation-specific input-role compatibility, lineage and typed results |
| `foundations-temporal-integrity` | Transitive dependency availability, fit/selection exposure and explicit temporal mode |
| `foundations-data-protocol` | Versioned population/splits, permitted fitting, matched information/budget controls and selection history |
| `foundations-lifecycle-admission` | Separate reusable authoring, instance binding, authorized execution and scientific verdict |
| `foundations-evidence-ledger` | Discoverable records for all admission and execution dispositions; explicit corrections and supersession |

Each future run must document how these requirements are enforced by actual
artifacts/checks, or justify a rule's non-applicability. Deferring Hx14 does not
block this reusable definition or waive requirements of a dependent execution.

A future observational profile must define matching/weighting, comparison
population, balance/overlap, uncertainty and permitted association claims.
A future intervention profile must additionally bind an estimand (including
population/conditional versus individual scope), treatment and outcome timing,
assignment or identification argument, consistency/interference, exchangeability
and positivity assumptions, and any untestable assumptions. It must identify
prohibited outcome-derived, post-treatment, mediator or collider matching inputs
as required by that design, including such information hidden in embeddings.
Overlap diagnostics, balance and embedding proximity do not supply this
identification argument. Unsupported identification or support must refuse the
causal claim. Those profiles require separately reviewed contracts and evidence.

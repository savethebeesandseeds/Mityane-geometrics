# Hx06 profile-expansion acceptance

**Date:** 20 October 2026.  
**Decision:** complete for **Expand operation profiles**, covering reusable
authoring and actual non-synthetic instance binding, not estimator delivery or
scientific evaluation.

## Two separately versioned profiles

The [OR06-MOC contract](../profiles/observational_contrast/v0.1.0/CONTRACT.md)
and linked [OR06-MOC-RER](../profiles/observational_contrast/v0.1.0/EVIDENCE.md)
define observed association, matching/weighting, matching-information audit,
balance/overlap, measured and omitted-confounding checks, dependence and typed
full-roster failure/abstention. Balance/overlap/sensitivity remain diagnostics
rather than causal-identification certificates.

The [OR06-IIE contract](../profiles/intervention_effect/v0.1.0/CONTRACT.md)
and linked [OR06-IIE-RER](../profiles/intervention_effect/v0.1.0/EVIDENCE.md)
require an explicit estimand and identification argument, untestable assumptions,
assignment/treatment/outcome and availability timing, support and prohibited
matching/adjustment variables, including information hidden in representations.
Proximity is explicitly insufficient. Unsupported causal identification receives
a typed refusal; it is not relabeled as an observed or model-response result.

Independent peer reviews checked both profiles against charter sections 4,
OR-06, 6 and 7 and found no authoring blocker. The original OR06-PCR contract,
implementation and frozen synthetic evidence retain their existing claim scope.
Neither new profile is registered as a callable operation.

## Actual-source instances

The owner authorized selection of a documented open randomized dataset. The
[NSW reference binding set](../profiles/instances/nsw_reference_v1/README.md)
links two distinct instance/RER IDs and protocols:

- `OR06-MOC-NSW-REF-001` / `OR06-MOC-NSW-REF-RER`, revision 1.0.0:
  [observed contrast protocol](../profiles/instances/nsw_reference_v1/OBSERVATIONAL.md).
- `OR06-IIE-NSW-REF-001` / `OR06-IIE-NSW-REF-RER`, revision 1.0.0:
  [assignment-effect protocol](../profiles/instances/nsw_reference_v1/INTERVENTION.md).

[bindings.json](../profiles/instances/nsw_reference_v1/bindings.json) pins actual
dataset/source/license documents, profile and protocol bytes, permission record,
provider, SpaceRef, roster and representations. The data-owned importer selected
`wooldridge::jtrain2` 1.4-7, source revision
`427d117f3888cd3a630ee22a1aedc8f036d128ff`, publisher declaration GPL-3. The exact
served CSV is 37,008 bytes, SHA256
`3dab3d903e07549343ae3aa51ce6530afc490c5861f6ebec76360e48fed9f9df`.
All 445 source rows are retained: 185 assigned, 260 controls. Raw data,
documentation downloads and derived row artifacts remain ignored locally.

The actual raw descriptor `[re74]` uses one finite 1974-earnings coordinate,
absolute distance and no fitted preprocessing, encoder or readout. Assignment
and 1978 outcomes are not coordinates. Source IDs retain snapshot/ordinal
lineage rather than assert recovered person identities. The synthetic OLS
binding is explicitly not applicable; no new profile depends on its adequacy.

The [import receipt](../../../../.temp/data/hx06-nsw-reference-v1/import-001/import-receipt.json)
retains seven exact publisher/documentation downloads and their identities.
The [materialization receipt](../../../../.temp/data/hx06-nsw-reference-v1/binding-preparation-001/materialization-receipt.json)
pins actual provider, SpaceRef, roster and representation bytes. Both activities
were covered by a preceding data-only admission record. The CSV serialization
is a separate artifact from the retained RData binary; no independent numeric
equivalence was asserted.

The MOC instance freezes a one-nearest-control matching rule, replacement,
fixed caliper/ties, source balance/coverage/reuse/ESS gates, exact SMD/ESS formulas
and complete refusal accounting. Its target is a finite descriptive cohort,
with explicit no-sampling-CI/no-external-generalization limits. Requests requiring
unsupported uncertainty refuse.

The IIE instance targets assignment/offer ITT within the retained cohort and
states the conditional randomization/subset-selection, consistency, interference
and observation assumptions. Allocation probabilities, strata/clusters and exact
assignment dates are unavailable; the observed arm ratio is not a known
assignment probability. Its estimator/uncertainty remain unadmitted pending
the needed design justification. Binding an actual instance is complete while
those scientific-execution requirements remain unresolved. Neither binding is
fresh protected assessment or a new estimated effect.

Independent final instance reviews accepted these distinctions and found no
blocking defect in the authoring/binding scope.

## Executed checks and preservation

The existing managed container executed:

```text
bash code/scripts/task.sh hx06-profiles-final-v1 test-counterfactual-profiles
```

All **nine tests** passed: six artificial data/import/baseline-isolation checks
and three binding guards against prohibited outcome/group inputs, unreviewed
OLS, invented randomization details, false admission/freshness and artifact
corruption/path escape. No test was skipped or cancelled. The following actual
binding integrity check passed and verified **27 files**, two instances and
445 rows. Its [retained receipt](../../../../.temp/operations/counterfactuals/profile-expansion-v1/review-001/binding-check.json)
reports no scientific execution, registration or effect estimate.

An earlier engineering invocation used the full research-source scan because
the new target was absent from maintenance routing. It was superseded and its
exact make process stopped with exit 143; the root routing entry was corrected
and the focused invocation above passed. No research target or experiment was
executed by that scan.

No matching, contrast/effect estimation, model/API execution, fitting or training
was performed. Scientific runs require actual implementations, conformance,
resolved applicable design/evidence requirements and separate admission/attempt
records. Import, binding acceptance, execution disposition and scientific
verdict remain distinct. Frozen synthetic records, prior model studies and the
other Hx06 goals were preserved.

## Dashboard closeout

The mandatory managed-container refresh passed all 34 dashboard tests and
regenerated the standalone HTML. The maintained dataset validated 18 Heroes,
71 Goals and 113 Milestones, including 26 done and one in progress; all evidence
targets existed. All three profile-expansion steps are done, making this whole
goal complete. Schema remains version 2 and the other five Hx06 goals are
unchanged. The final link check verified 1,200 maintained Markdown links and
71 literal shell roots; historical artifact trees remain ignored.

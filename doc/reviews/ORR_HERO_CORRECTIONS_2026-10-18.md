# Operations over Representations — Hero corrections

**Review ID:** ORR-HERO-REVIEW-001  
**Date:** 2026-10-18  
**Status:** Proposed planning amendment; not adopted repository policy or execution evidence  
**Scope:** The eighteen supplied schemaVersion 2 Hero JSON files  
**Preserved:** OR-01 through OR-13; the Foundations and four encoder Heroes; all existing Hero, goal and milestone IDs; all existing statuses, evidence references and completed milestone objects.

## 1. Decision

Keep the segmentation and the Contract → Implementation → Evaluation structure. Correct the transition from descriptions to executable, auditable work rather than reorganizing the capability families again.

The main amendment is:

> An admitted operation must know what its inputs represent, which information and fitted artifacts it may use, why each input role is compatible, how failure propagates, and where every result—including a negative or invalid one—will remain discoverable.

Make those shared obligations part of Foundations. Let individual Heroes specify their own behavior and acceptance tests rather than copy a second charter into every contract. The proposed JSON copies retain the original descriptions and organization where possible, and add explicit completion criteria. Five proposed milestones fill missing shared ownership and the diagnostic-probe pattern.

**This is not an implementation receipt.** The supplied Heroes contain references to code, provider records, handoffs and historical verification artifacts; those referenced materials were not supplied as the basis of this review and were not independently inspected. Existing `done` entries have been preserved as the project's recorded assertions, not newly verified. No encoder, service, dataset or experiment was executed, selected or authorized by this amendment.

## 2. Source observations and proposed changes

Source anchors below refer to the original filenames and milestone IDs. Original-file hashes are recorded in `review/hero_changes.json`; the revised files will naturally have different line positions.

| Source observation | Correction proposed here |
|---|---|
| `14-foundations.json`, `foundations-record-validation`, already owns shared records, compatibility and observed-versus-predicted distinctions. | Give it an operation-specific compatibility decision and explicit refusal/failure semantics. Do not invent a second owner. |
| `11-or-11-dynamics.json`, its implementation milestone, explicitly mentions temporal cutoffs; `17-encoder-cpb.json`, its provider milestone, also mentions them. | Generalize temporal integrity to every temporally derived record and composition, including reference populations, fitted transforms and other encoders. |
| `14-foundations.json`, `foundations-pipeline-protocol`, already requires payload meaning, compatible versions, failure propagation and end-to-end evidence. | Turn the existing requirement into enumerated edge declarations and executable negative-test criteria. This is a tightening, not a newly discovered missing concept. |
| Every OR Hero repeats the same charter link in its proposed milestones; their detailed Operation Contracts and Requirements-and-Evidence Records are not present in the supplied files. | Treat these as design references, not completion evidence. Add concrete acceptance criteria and explicit binding gates without pretending the contracts already exist. |
| `18-encoder-google-multimodal.json`, the adapter milestone, already mentions response identity; its retrieval fixture already requires replayable response artifacts. | Specify canonical request fingerprints, response/artifact digests, per-attempt identity and replay limits. Do not describe response provenance as wholly absent. |
| `17-encoder-cpb.json` explicitly preserves a pause and pending registry work. | Preserve that boundary through generic probes, integration work and scheduling; resolve execution authority before any affected CPB action. |
| `16-encoder-rpb.json` explicitly requires instance selection and rejects a universal default; Foundations says no initial dataset is selected. | Recommend a first path without guessing a checkpoint, dataset or experimental admission. |

## 3. Main correction: temporal integrity is about dependencies, not one timestamp

### 3.1 Required distinctions

For temporally meaningful data, require enough metadata or resolvable lineage to distinguish:

- **Event/support time:** when the represented observation occurred and which source interval or positions can influence the output.
- **Information availability:** when the required source values, revisions, labels or reference entries were available under the declared setting.
- **Run cutoff and claim mode:** the information boundary the operation is allowed to use, and whether the result is retrospective analysis, a walk-forward simulation, or replay of an actually deployed historical system.
- **Fitted-artifact eligibility:** the permitted fit and selection information behind the encoder, preprocessing, pooling, alignment, index/reference model, thresholds and downstream models.

These are semantic requirements, not a demand to repeat a long timestamp schema in every vector. A record may resolve them through immutable manifests and parent references. A validity mask can participate, but only when its relationship to time and dependency support is defined.

An audit `createdAt` timestamp is separate again. A historical simulation can compute a model today using only the data and selection rules permitted before its simulated cutoff. That is a reconstructed simulation, not proof that the model actually existed or was deployed then. Conversely, loading a model that was frozen today does not establish that its training or selection was independent of a historical assessment period.

### 3.2 Output-level causality

A complete-window encoder may validly use all observations through the window end. It does not need a causal attention architecture to produce an end-of-window feature under that boundary. A token labeled as an earlier prefix state has a different obligation: its dependency support must not include the later portion of the window.

The conformance test must match the output's actual claim. Alter data beyond a prefix cutoff and check that the protected prefix output is unaffected under the declared numerical tolerance. For an end-of-window output, test the boundary after the complete available window instead. Padding or missingness masks are not automatically causal masks.

Do not infer any of these properties solely from the names MTF, RPB or CPB. Their provider contracts must expose the relevant native output semantics. The proposed CPB criteria describe future, authorized tests; they do not authorize running them now.

### 3.3 Propagation

A derived record inherits the dependencies of its inputs unless a documented per-output dependency analysis justifies a tighter bound. In particular:

- An aggregate cannot omit the availability of a late member merely by assigning an earlier window label.
- Retrieval cannot use a future candidate or future outcome annotation as though it were already in the admissible corpus.
- Alignment cannot remove the temporal restriction of its source or its fitted mapping.
- A forecast for a future target time retains its original information cutoff. Its target timestamp is not a future observation becoming available.

Future labels may be used later by the evaluator after the corresponding predictions are fixed. They may not enter the inference inputs, fitting, threshold selection or adaptive acquisition outside the admitted protocol.

**Failure policy:** reject a violated or unresolved as-of claim, or explicitly admit a different retrospective mode with a narrower claim. Never silently downgrade the meaning of the same run.

**Acceptance fixtures:** future-window contamination, future corpus entry, whole-dataset fitted preprocessing, delayed or revised label availability, contaminated fitted mapping, and an unavailable dependency passed through a composition.

## 4. Main correction: compatibility is typed, directional and scoped

Do not implement a universal rule of “same dimension means compatible.” Also do not implement a universal rule of “different SpaceRefs must always be refused.” OR-10 intentionally connects distinct spaces, and retrieval may intentionally use distinct query/document task roles.

The proposed shared validator asks:

> Does this operation contract permit this producer's output kind and space in this particular consumer role, under the declared artifacts and evidence scope?

Keep four identities separate:

| Identity | Purpose |
|---|---|
| Source/record identity | Identifies the observation or derived record and its lineage. |
| Space/configuration identity | Identifies the representation recipe, instance/revision policy, output semantics and geometry. |
| Fitted-artifact identity | Identifies a checkpoint, fitted transform, mapping, reference model, threshold or index snapshot. |
| Execution/response identity | Identifies a particular local run or external request/response attempt. |

A response ID or content hash should not create an unrelated coordinate space for every sample. Conversely, a broad encoder family name cannot identify the trained instance and fitted preprocessing used in an evaluation.

An OR-10 mapping returns a derived record with its original source and mapping provenance. It must not simply relabel a source vector as a target-native observation. Consumer acceptance is limited to the mapping's declared direction, target-side role and evidence scope.

For hosted models whose internal revision is not exposed, state that uncertainty. A frozen collection of retained outputs can support evaluation and replay of that collection without claiming that the provider's undisclosed checkpoint has been pinned. Do not silently extend that evidence to later live responses or assert immutable coordinate identity from a stable alias.

### Result semantics

Specify successful output, abstention and execution failure separately. Give incompatibility, temporal violations, stale artifacts, malformed/non-finite inputs and unsupported roles explicit reasons. A valid empty retrieval result is not automatically an execution failure.

The consumer must not receive a normal-looking zero vector, normal score or action in place of an error. Partial output and fallbacks can be legitimate, but only under a declared policy, with preserved path identity, coverage and end-to-end evaluation.

**Acceptance fixtures:** equal-dimensional vectors from different instances; changed preprocessing under the same family name; stale index/query combinations; a valid role-paired retrieval path; a valid directional mapping; and attempted predicted-to-observed relabeling.

## 5. Main correction: work status, execution admission and scientific verdict are different

The colleague's concern about uniformly `proposed` work is useful. Adding `contract-ready` and `blocked-on-dataset` directly to the existing `status` enum is not the best immediate repair: the consumer schema is not supplied, and those labels mix readiness with lifecycle and blockers.

Keep the current enum in this amendment. Separate the following concepts in the shared workflow, and introduce any future structured fields only through a verified schema migration:

| Concept | Question |
|---|---|
| Work lifecycle | Has this milestone been started or completed under its acceptance criteria? |
| Readiness and blockers | Which concrete dependency or authorization is missing, and what clears it? |
| Execution admission | Is this specific run bound to an adequate contract, data, artifacts, protocol and permissions? |
| Run disposition | Did it execute and complete, fail, abort, or get rejected before execution? |
| Claim verdict | Does valid evidence support the scoped claim, fail to support it, remain inconclusive, or leave it unassessed? |

Reusable interfaces can be drafted while a dataset or trained instance is unselected. Execution cannot be admitted by filling those gaps with assumed defaults. A valid negative or inconclusive evaluation can complete an evaluation milestone; an invalid or unexecuted attempt alone does not satisfy an operation-quality evaluation requirement.

Keep charter links and planning notes distinguishable from actual completion artifacts. The existing `evidence` entries remain untouched for compatibility and historical preservation, but new acceptance criteria require real accepted contracts, implementations or test receipts when work is completed.

### Keep the gate light enough to use

The correction is not “finish all Foundations before trying anything.” For the first path, implement the smallest shared subset that its cards and negative tests require. A compact Operation Contract and linked Requirements-and-Evidence Record can reference the shared rules and bind the local choices without copying the entire design specification.

A dependency blocker should block its dependent execution, not all other authoring or independent engineering work. An explicit authorization pause is different: it cannot be removed merely by relabeling the same action as a probe.

## 6. Main correction: provenance permits replay only when the needed artifacts are retained

The Google adapter already mentions response identity, and the fixture already calls for replayable responses. Strengthen those requirements rather than replace them.

For each external attempt, specify:

| Item | Required meaning |
|---|---|
| Semantic request manifest and digest | Versioned canonicalization over permitted content digests or immutable source versions, input ordering/grouping, role/task instructions, effective parameters, model and service route; no credentials. |
| Local attempt identity | A durable local identity even when the provider exposes no request/response ID. |
| Provider identity when exposed | Preserve returned IDs and model/revision metadata exactly as received; explicitly record absence rather than inventing it. |
| Response/artifact digest | Identifies the returned artifact, including the vectors needed by the downstream replay. |
| Retained permitted artifact | Stores the necessary output or an immutable accessible reference under the declared retention policy. |
| Attempt status and time | Separates retries, errors, partial responses and successful executions; a retry does not overwrite the earlier attempt. |

Hashing a request and retaining a remote response ID supports traceability. It cannot recover an unretained vector, identify an undisclosed checkpoint, or force a future service call to return the same response.

Distinguish **offline replay of saved outputs** from **fresh service re-execution** in the evidence record. Pin the former to a frozen response-artifact snapshot. Later calls belong to a new acquisition/run and must not silently alter a frozen corpus. Reference-anchor checks can help detect changes; they are not proof of immutable provider identity.

The adapter must check source/output grouping, ordering, finite values, shape, partial batch failures, cache provenance and effective parameters. Do not compare encoder providers as though they necessarily saw identical information: input modality, preprocessing, instruction roles and unknown pretrained exposure can change the comparison's scope.

This review does not verify current Google model names, dimensions, modality support or availability. It preserves the existing documented provider record and proposes stronger integration criteria. No live call is authorized by these edits.

## 7. Main correction: lightweight probes must not become an untracked experiment lane

Adopt the probe pattern under OR-05, but reject the “non-admitted” loophole.

A diagnostic probe should use a compact contract/RER pair, development-only data, a small declared target set, a fixed small readout family, a bounded fit/selection budget, matched-information controls and the same basic provenance, temporal and permission rules as other experiments.

Its legitimate conclusion is narrow: a specified target was recoverable to a measured degree through a specified probe from a specified representation and population. It is not a universal embedding-quality score, a proof of sufficient state, a latent-dynamics result or a decision-utility claim.

Log all targets, configurations and selection choices. When the probe helps select an encoder, target or operation, that same information cannot later be presented as untouched confirmatory evidence. Reserve a separate confirmation path or retain an explicitly exploratory interpretation.

The pattern may be reusable across providers; execution is conditional on comparable input/target information and actual authorization. **It does not restart CPB, apply its pending proposals, replay its evidence or permit an otherwise unauthorized external API call.**

## 8. Main correction: failed and negative results need a durable home

Agree on a canonical, discoverable namespace. The proposed default below is a design decision for adoption, not a claim that these directories already exist in the repository:

```text
research/orr/
  contracts/<contract-id>/<revision>.json
  requirements/<rer-id>/<revision>.json
  runs/<run-id>/
    operation-run.json
    ... permitted artifacts or immutable artifact references ...
  evidence/<evidence-id>.json
  index.jsonl
```

Register the attempt before experimental execution. Record an admission rejection without pretending execution occurred. Finalize the run disposition even after failure or interruption; recovery should identify unfinished attempts.

The index should resolve owner Hero, purpose, contract/RER revisions, dataset/source identities, SpaceRefs, fitted artifacts, protocol, disposition, claim verdict and evidence/artifact locations. Link preserved model-owned history without rewriting its original protocols or comparisons.

A completed experiment with no useful effect, an inconclusive experiment, an invalid experiment and a crashed process are four different outcomes. Keep all visible and do not count the last two as negative scientific evidence for the hypothesis.

Finalized records are immutable or explicitly versioned with correction/supersession links. Check for missing artifacts, digest mismatches, dangling references and lost unfinished runs. When raw content cannot legitimately be retained, record permitted hashes/references, the retention limitation and an auditable permitted tombstone where necessary. Do not trade source permissions or confidentiality for a false promise of replayability.

The same registry records development exposure and selection history. This prevents repeated evaluation, dropped failures or a newly chosen configuration from disappearing into chat logs and local directories.

## 9. Main correction: composition must preserve semantics and carry its own evidence

Elevate the existing Foundations pipeline milestone. For each edge, require:

| Edge declaration | What must be explicit |
|---|---|
| Producer and consumer | Contract versions and roles. |
| Payload | Schema, record kind and source/parent lineage. |
| Space relationship | Source/output SpaceRefs, accepted role pairing or mapping identity, and the concrete compatibility check. |
| Temporal relationship | Information cutoff, dependency support and propagation rule where applicable. |
| Fitted dependencies | Reference/index/transform/model identities and their fit scope. |
| Failure policy | Refusal, abstention, partial-result, retry or fallback behavior and coverage accounting. |

Do not implement full infrastructure for every conceivable graph before the first experiment. Implement and test these declarations for one small path, then expand through real needs.

When a checkpoint, provider configuration, preprocessing artifact, mapping, fit population or schema changes, review the affected dependent artifacts. Some will need rebuilding or reevaluation. Preserve old artifacts and their evidence rather than silently authorizing an existing index or downstream model for the new representation.

Component evidence remains valuable but does not add up automatically to end-to-end utility. The pipeline needs its own population, denominator, baselines, costs, exclusions and failure/abstention reporting. This remains true when every individual component passes its tests.

## 10. First bounded path: selected RPB instance → OR-01 exact retrieval

Prefer retrieval for the first integration candidate because it can exercise provider records, source provenance, compatibility, temporal corpus eligibility and end-to-end evidence without first adding a learned dynamics model, planner or anomaly-threshold policy.

This is a prioritization decision, **not an assertion that an eligible checkpoint or dataset is already selected**. The supplied RPB Hero explicitly leaves instance selection open.

### Bind these choices before admission

| Choice | Minimum decision |
|---|---|
| Provider | Exact RPB version and trained checkpoint; fitted preprocessing; native output selector, layout, pooling and normalization. |
| Data | A fixed permitted synthetic or public series fixture, immutable source/generator identity, split and grouping rules. |
| Temporal mode | End-of-window support and declared retrospective/as-of setting; independently controlled source and artifact eligibility. |
| Retrieval question | Query/corpus populations, fixed snapshot, metric, `k`, tie handling and relevance defined independently of the encoder's own distances. |
| Exclusions | Self, duplicates, shared-source/window overlap where required, and entries not admissible at the query's cutoff. |
| Controls | A simple matched-input raw-descriptor retrieval baseline and an appropriate random/chance ranking reference. |
| Evaluation | Metrics, independent sampling unit, uncertainty approach or limitation, budget and failure criteria. |
| Evidence | Bound Operation Contract and RER, run identity, output artifacts, registry entry and scoped conclusion. |

A synthetic generator's known factors may define relevance in a bounded fixture, but that rule must be frozen independently of what the embedding happens to retrieve. Keep nuisance factors, seeds and source overlap visible; avoid defining relevance as the same geometric distance used by the method.

### Close two different loops

**Engineering loop:** emit records, perform exact retrieval, deliberately inject an incompatible space, future corpus entry, non-finite vector and upstream failure, and verify both propagation and evidence indexing.

**Bounded research loop:** apply the frozen relevance protocol and baselines to its assessment population, report uncertainty and limitations, and retain the conclusion even when the encoder does not help.

A successful synthetic fixture does not establish general industrial usefulness. A negative bounded retrieval result does not make the project fail: the machinery has learned to distinguish a valid result from a broken experiment, which is the necessary basis for choosing the next question.

Do not require all thirteen families or every provider to be implemented before closing this loop. CPB remains outside it; other operation contracts can continue to be authored independently.

## 11. Operation-specific acceptance additions

The shared corrections are insufficient unless each Hero states the relevant local distinction. The proposed copies add the following focused criteria:

| Hero | Main local correction |
|---|---|
| OR-01 Retrieval | Freeze candidate eligibility and independent relevance; distinguish exact-reference fidelity, approximate-index fidelity and actual task relevance. |
| OR-02 Structure | Bind cluster/prototype/density identities to the fitted population and scale; assess stability without turning compactness into semantic truth. |
| OR-03 Novelty | Separate reference fit, threshold calibration and assessment; define alert unit and false-alert denominator. Normal-only evidence does not establish anomaly-class sensitivity. |
| OR-04 Shift | Evaluate a repeated alert policy, reference updates and provider changes; do not issue a population-drift verdict for an incompatible representation change. |
| OR-05 Readouts | Specify target timing and accessible information; add the bounded probe pattern and keep its evidence separate from latent prediction and policy value. |
| OR-06 Comparisons | Bind claim level and identification argument; balancing/overlap diagnostics are not automatic causal certification. |
| OR-07 Transformations | Keep candidate outputs and full dependency provenance; assess meaning independently of the arithmetic or metric that generated them. |
| OR-08 Compression | Define the new code/aggregation contract and source membership; count side artifacts and actual resource costs, not dimensions alone. |
| OR-09 Generation | Preserve generated/reconstructed kind; distinguish self-consistency, source grounding and independent output validity. |
| OR-10 Alignment | Use directional, role-scoped compatibility and held-out anchors; derived mapped records retain mapping and original-source lineage. |
| OR-11 Dynamics | Separate origin time from future target time and observed from predicted state; distinguish teacher-forced and free-running assessment. |
| OR-12 Decisions | State action authority, fallback policy and realized/simulated/observational evaluation limits; model fit does not establish decision value. |
| OR-13 Acquisition | Protect assessment feedback; record full acquisition history and costs; compare learning value rather than rewarding the selection proxy itself. |

For MTF and RPB, bind the actual checkpoint/output/preprocessing contract and test its actual output-time semantics. For CPB, resolve the authoritative execution boundary before affected work. For Google, separate documented provider scope, adapter engineering, live response verification, retained-artifact replay and measured operation utility.

## 12. Patch inventory and adoption boundary

The package proposes revisions to all eighteen Hero JSON files, without renumbering or changing their segmentation.

Five milestones are added, all `proposed`:

| Owner | New milestone ID |
|---|---|
| Foundations / shared records | `foundations-lifecycle-admission` |
| Foundations / data | `foundations-temporal-integrity` |
| Foundations / data | `foundations-evidence-ledger` |
| Foundations / reproducibility | `foundations-starter-composition` |
| OR-05 / contract | `or-05-readouts-probe-protocol` |

The original 56 milestones become 61 proposed-package milestones. All eight existing `done` milestone objects are preserved exactly, including their evidence and completion dates. All original statuses, IDs, evidence entries and root `schemaVersion` values are retained. Existing proposed milestones receive explicit acceptance criteria; no new enum values or unfamiliar JSON keys are introduced beyond the observed structure.

The bundle includes proposed Heroes, this review, a field-level change inventory with original/proposed hashes, a unified text diff, and a structural validation report.

**Validation actually performed:** JSON parsing; local structure consistency against the observed input shape; stable identities/statuses; preservation of completed entries and existing evidence; unique goal/milestone IDs; acceptance criteria presence; and no new completion claims.

**Validation not performed:** the repository's real consumer schema, parser, renderer or CI; historical receipts and pause handoff contents; model or provider behavior; current external-service specifications; dataset fitness; temporal conformance; or experimental performance. Preserving the observed structure reduces change risk but is not proof of compatibility with an unseen consumer.

Before adoption, reconcile the review with repository conventions, inspect the real consumer schema, and choose whether these proposed milestones and registry paths belong in the main charter or a versioned companion protocol. Apply the supplied diff relative to the actual Hero directory, not an assumed repository root. None of these planning changes constitutes CPB resumption or live external-service authorization.

## 13. Governing principle

Keep the entry cost small and the conclusion precise. A contract should prevent an ambiguous or misleading experiment, not require a finished research organization before any experiment can begin.

Build one modest path that can reject invalid inputs, explain its time boundary, reproduce its retained evidence, and preserve an unwelcome result. Then expand the project by adding operations whose requirements have become concrete through use.

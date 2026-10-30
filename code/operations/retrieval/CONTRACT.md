# OR-01 Operation Contract

| Field | Definition |
| --- | --- |
| Family / owner | OR-01; Hx01, retrieval |
| Operation | `ranked-source-retrieval` |
| Contract ID / revision | `OR01-RET`, `0.1.0` |
| Claim levels | `exploratory_description` or `task_association`. Both are explicit. There is no implicit default. |
| Primary question | Which existing source items account for a query coordinate under a declared geometry, and which of those items are relevant under a separate rule? Exact k-retrieval returns the nearest sources. Positive sparse combination, specified in section 11, returns non-negative weights. The registered tool implements exact k-retrieval. |
| Status | Reusable interface definition. Profile binding, the operation envelope, and study admission remain separate. Section 11 is specified and is not part of the registered tool. |
| Requirements and evidence | [OR01-RET-RER, revision 0.1.0](EVIDENCE.md) |
| Governing specification | [ORR-DESIGN-001, v0.1.0](../../../doc/operations_over_representations_design_spec_v0.1.0.md), sections 4, OR-01, 6 and 7 |
| Engineering reference | [Exact search on saved vectors](IMPLEMENTATION.md) |

## 1. Boundary

A request supplies a coordinate and a frozen corpus of saved vectors that already share one SpaceRef. This operation checks that binding and returns ranked source references, pairwise similarities, or neighborhood diagnostics. A source-space input, such as text, is encoded by an upstream producer before it reaches this contract. The producer may be a text encoder, a kernel sample, or another construction. Calling a provider and deciding that a neighbor is relevant stay outside this operation.

An admitted run may claim exploratory description of the neighborhood, or task association when an independent relevance rule is bound. It does not claim a controlled model response, an identified intervention effect, or a property of any encoder family. Retrieved neighbors may be inputs to Hx06. The retrieval record does not state what a contrast means.

Hx18 owns the Gemini Embedding 2 provider, adapter, credentials, request budget, and response retention. Hx15–Hx17 own production of local encoder vectors. Hx14 owns the shared record, temporal, data, and admission rules. Those shared validators are still proposed, so an admitted run applies the checks in this contract directly. Hx19 owns source loading. Hx02 owns structure and density models. Hubness and mean neighbor distance here are inspectable retrieval diagnostics.

## 2. Unbound profiles

This revision defines two profiles and selects neither a corpus nor a query set.

| Profile | Representation source | What must be bound before a run |
| --- | --- | --- |
| `text-gemini-embedding-2` | Text inputs to `gemini-embedding-2`, provider record `google-gemini-embedding-2` | Frozen text corpus and queries, task or role pair, requested dimensions, normalization, metric, saved response artifacts, and an independent relevance rule with a lexical baseline on the same eligible texts |
| `local-saved-vectors` | Saved vectors from one named MTF, RPB, or other local encoder instance, or from a declared synthetic generator | Frozen vector snapshot and the inputs those vectors came from, SpaceRef, metric, normalization, and an independent relevance rule: hand-adjudicated regimes or a synthetic motif rule fixed before ranking |

Image, video, audio, and PDF inputs stay outside the text profile. The text profile uses the Hx18 provider record. A live call also needs the Hx18 adapter, a request budget, and permissions. Replay of a saved response artifact is a different execution from a fresh call. The local profile reads vectors that already exist. It does not train an encoder and it does not wait on the Hx18 adapter.

[OR01-RET-ABS 0.1.0](bindings/oewn-2025/BINDING.md) begins one local-profile instance for an English abstract text population. The two profile rows above stay unbound. That instance does not revise this contract.

A hosted model ID is not an immutable checkpoint. The text profile retains the request-manifest digest, any returned revision or its absence, the response-artifact digest, and the execution time. Credentials stay outside the operation record.

## 3. Logical input roles

Every identity resolves to a versioned record or an immutable artifact reference.

| Role | Required content |
| --- | --- |
| Request and run | Request and OperationRun IDs; exact contract and RER revisions; claim level; code and configuration identity; evidence destination |
| Query | Query ID; source ID when the query is itself a corpus member; vector values; representation kind; parent observation or saved-response identity |
| Corpus | Snapshot ID; member source IDs; vector values; parent lineage for each member; eligibility rule |
| SpaceRef | Provider or encoder identity; checkpoint, generator, or saved-response revision; dimensions; layout; normalization; metric; task or role pair; preprocessing and fitted-transform identities when any were applied |
| Geometry parameters | Metric `cosine` or `l2`; normalization `none` or `l2`; neighborhood size `k`; tie policy |
| Exclusions | Self-match policy; duplicate policy; explicit excluded source IDs; source-window overlap rule when the corpus is temporal |
| Timing | `non_temporal`, `retrospective`, or `as_of`; query eligibility; availability cutoff; justified non-applicability |
| Relevance | The independent rule, its labels or generator, and the baseline. Required for `task_association`. Omitted for a purely exploratory neighborhood description, which then cannot be reported as relevance quality |

Equal vector length is not a SpaceRef match. The query and every corpus member must carry the same provider, revision policy, dimensions, normalization, metric, and task or role pair.

## 4. Output

A successful ranking returns one row per emitted neighbor:

| Field | Meaning |
| --- | --- |
| Source reference | Corpus source ID and parent lineage |
| Rank | 1-based position after the declared metric and tie policy |
| Distance | Euclidean distance, or one minus cosine, on the vectors after the declared normalization |
| Similarity | Cosine, or `1 / (1 + distance)` for `l2`. This is the ranking score, not a relevance label |
| Binding | Space ID, snapshot ID, metric, normalization, dimensions |
| Disposition | `ranked`, together with the eligible count |

Skipped corpus members are returned with a reason: `self_match`, `excluded_source`, or `undefined_geometry`. A skip is not a silent drop. Pairwise similarity uses the same geometry and returns one distance and one similarity. Neighborhood diagnostics return, per source, the leave-one-out k nearest neighbors' mean distance and the count of other points that list it among their k neighbors.

The registered operation envelope wraps these rows with OperationRun identity, input hashes, and the common result schema. The engineering reference computes the rows. The envelope applies as-of availability and source-window overlap before search and reports those skips. Registration admits no profile.

## 5. Ranking, ties, and diagnostics

Cosine ranks by descending cosine. L2 ranks by ascending Euclidean distance. Ties break by ascending source ID under exact binary64 equality. `k` is at least 1. When fewer than `k` members are eligible, the result contains every eligible member.

Pairwise similarity, kNN, and the diagnostics share that metric, normalization, and tie policy. The diagnostics exclude the point itself and do not apply a caller's other exclusion list; a caller who needs those exclusions filters the snapshot first. Hubness and mean neighbor distance do not close Hx02.

Approximate search is outside this revision. A later index is judged by recall of this exact ranking and by the distribution of approximation loss on the same eligible corpus. Those figures are not relevance metrics.

## 6. Exclusions, emptiness, abstention, and failure

| Outcome | When it is used |
| --- | --- |
| `ranked` | At least one eligible neighbor |
| `empty` | The query and corpus are valid and every member was ineligible or absent |
| `abstained` | Cosine or L2 normalization is undefined for the query because its norm is zero |
| `failed` | The request is malformed, a source ID is duplicated or empty, a value is non-finite, a length disagrees with the SpaceRef, or the query binding disagrees with the corpus snapshot |

A zero-norm corpus member under cosine, or under L2 normalization, is skipped as `undefined_geometry`. A zero vector under L2 with normalization `none` is a defined point. An unused exclusion ID does not fail the search. Source-window overlap is applied by the caller as excluded source IDs or by removing members before the snapshot is hashed. This reference does not interpret timestamps.

## 7. Relevance and baselines

Relevance is a labeled judgment or a rule fixed before any ranking is examined. It is not the distance or the similarity. A `task_association` run compares the ranking with a baseline that sees the same eligible corpus, queries, and information horizon: a lexical baseline for the text profile, and either that lexical baseline or the declared raw descriptors or synthetic motif rule for the local profile. Exploratory neighborhood output stays labeled exploratory.

Negative and inconclusive relevance results remain indexed. An engineering execution of the exact-search reference establishes arithmetic, ordering, exclusion, and refusal behavior only.

## 8. Revision limit

OR01-RET 0.1.0 admits no relevance run. It freezes the fields an admitted profile must bind. It does not select a dataset, a checkpoint, a Gemini request budget, or an approximate index. An engineering tool may expose this revision on the common operation boundary. That registration does not admit a profile or establish relevance.

## 9. Deprecated geometric constructors

Difference, shared residual, and cone constructors are internal utilities. They are not the retrieval strategy. Exact k-retrieval takes a coordinate that is already in hand. A constructor may still build a vector or a cone and then call that same ranker. It is a pure function of saved vectors, the SpaceRef normalization, and its parameters. Cosine and L2, both normalizations, and the source-ID tie rule stay in force. No approximate index is introduced.

The difference of two points is the prepared target minus the prepared origin. The shared direction is the component of the first point that remains after removing the span of the other points' differences from it. A cone is a unit axis, an angular aperture, and optional radial bounds on the prepared length. Members outside the cone are reported skips. A full aperture is pi. A zero aperture admits only an exact cosine of 1.

Non-finite values, a mismatched length, a zero difference, a zero shared residual, and an aperture outside `[0, pi]` refuse or abstain with an explicit reason. Empty, abstained, and failed stay distinct. Leave-one-out hubness, mean neighbor distance inside the cone, and cone purity are diagnostics. Cone purity counts a caller-supplied partition. It is not a relevance rule. These constructors do not close Hx02 or Hx06, do not evaluate Nonverba, and do not admit the abstract corpus. The registered tool does not expose them.

## 10. English abstract binding

[OR01-RET-ABS 0.1.0](bindings/oewn-2025/BINDING.md) freezes one English abstract text population on the local saved-vector profile: Open English WordNet 2025 hypernyms of depth at most 3 from ten pinned noun synsets, with a shared-principle label taken from the intersection of those closures. The concrete seeds stay outside that corpus. Its SpaceRef is local EmbeddingGemma 2 text, 768 dimensions, L2 normalization, cosine, and the saved-vector hashes are in that record. The scored comparison is [OR01-RET-ABS-EVAL 0.1.0](bindings/oewn-2025/PROTOCOL.md). That score uses the deprecated geometric queries. It stays indexed, and it does not revise OR01-RET 0.1.0.

## 11. Positive sparse combination

`OR01-COMB` is the specified second retrieval method. The registered OR01-RET 0.1.0 tool does not run it. An engineering check is `exploratory_description`. The weights are fit coefficients. The residual and the weights are not a relevance rule. Output order is descending weight, then ascending source ID. That order is the combination order. Exact k-retrieval remains the fidelity baseline and the early-stop primitive.

The input is a target coordinate `t` and a frozen corpus that share one SpaceRef. Source-space text is encoded upstream. Both are passed through the same `prepare()` as exact search. The matrix `X` has one prepared corpus vector per column, and `t` is prepared the same way. A prepared zero vector is `undefined_geometry`. A zero target abstains.

The solver is non-negative least squares: minimize the L2 norm of `Xw - t` over `w >= 0`. The solution `Xw` is the Euclidean projection of `t` onto the cone of non-negative combinations, so `||Xw|| <= ||t||` and `||Xw|| / ||t||` equals the cosine between the reconstruction and the target. Among optimal solutions, return a basic solution: the active columns are linearly independent. If several basic optima exist, choose the fewest positive weights, then the lexicographically smallest ascending source-ID list.

Sparsity is one declared rule. A minimum weight `tau >= 0` is fixed before the solve. `tau = 0` keeps the plain solution. When `tau > 0`, drop every column whose weight is below `tau` and refit non-negative least squares on the surviving columns. Repeat until every surviving weight is at least `tau`, or no column remains. The gates below apply to that refit.

Weights are non-negative. The solve does not subtract a corpus vector and does not negate a coordinate. A negation is available only when the corpus already contains that text.

Two gates are fixed before the solve.

The fit floor is a cosine `gamma` in `[0, 1]`. A solution whose reconstruction cosine `||Xw|| / ||t||` is below `gamma` is `empty`. This ratio is a floor. Raising it demands a closer fit. It is at most 1 on the projection above.

The directness ceiling is a factor `rho >= 1`. The path length is the sum of `w_i` times the prepared norm of column `i`. After L2 preparation that path length is the L1 norm of `w`. The net length is `||Xw||`. A positive net length whose path length divided by the net length exceeds `rho` is `empty`, with reason `directness`. The ratio is 1 when every used vector points the same way. An exact sum of vectors that point apart can have a zero residual and still fail this ceiling. A declaration of `gamma` outside `[0, 1]`, `rho` below 1, or `tau` below 0 fails the request.

A cosine threshold `kappa` in `[0, 1]` can end the method before the solve. If the rank-1 exact-search hit has cosine at least `kappa` with `t`, the result is that hit. Its weight is the one-vector solution `max(0, dot) / ||x||^2` on the prepared vectors. After L2 preparation that weight equals the cosine. Ties follow the exact-search source-ID rule. This stop is a policy. A close neighbor can still leave residual that other vectors would reduce.

All-zero weights are `empty`, with reason `outside_cone`. A fit below `gamma` is `empty`, with reason `fit`. A zero target abstains. A request with a duplicate source ID, a non-finite value, a dimension mismatch, or a different SpaceRef fails. Empty, abstained, and failed stay distinct.

The output lists the selected source identities and their positive weights. A zero weight is omitted.

Exact k-retrieval remains the registered tool. This method adds no live call, no approximate index, and no relevance claim. It applies to any frozen vector corpus under a declared SpaceRef. The English abstract corpus is one population it may later use.

The engineering fixtures ran on 30 October 2026. The record is [positive-sparse-combination-v0.1.0.md](reviews/positive-sparse-combination-v0.1.0.md). They keep the early stop from hiding the ceiling: a one-direction sum that passes both gates, a rank-1 neighbor returned with its one-vector weight, a short combination that passes both gates, a path ratio above `rho` with outcome `empty` and reason `directness`, and a target outside the cone with outcome `empty` and reason `outside_cone`.

# Insight discovery Operation Contract

| Field | Definition |
| --- | --- |
| Owner | Hx22, insight discovery |
| Operation | `insight-discovery` |
| Contract ID / revision | `INSIGHT-DISC`, `0.1.0` |
| Family identifier | None. OR-01 through OR-13 remain the charter families. A run records `INSIGHT-DISC` 0.1.0. |
| Claim level | `exploratory_description` |
| Primary question | Given the same rows and the same evaluation budget, which procedure surfaces more valid and diverse propositions: a deterministic search, or a proposer that reads an Hx21 state? |
| Status | Operative interface. The same-day cite-only draft bound no population and is replaced by this text. A same-day refinement separates evaluation of one candidate from discovery under a budget. No evaluator, search, generator, checkpoint, or run is bound. |
| Requirements and evidence | [INSIGHT-DISC-RER, revision 0.1.0](EVIDENCE.md) |
| Design record | [Architecture, training, Lean, and application notes](DESIGN.md) |
| Governing specification | [ORR-DESIGN-001, v0.1.0](../../../doc/operations_over_representations_design_spec_v0.1.0.md), section 4 |

## 1. Candidate and validity

A **candidate** names a possible relationship: the class, the fields, the filters, the aggregator, and the relationship it proposes. A candidate carries no computed statistic.

**Validity** is a deterministic evaluation of that candidate against the caller's original rows. The evaluator writes the statistics, the uncertainty, the provenance, and the validation status.

A learned model may propose candidates. It does not invent a median, a coefficient, a slope, or a dominance relation. Those quantities are computed from the rows. A compressed state is not a substitute for those rows.

`evaluate` answers a closed question: does this one candidate hold on these rows under this evidence rule? A caller who already chose the candidate is using evaluation.

`search` answers the open question: which candidates in a declared space hold, within a declared budget? The discovery baseline is `search`. A learned proposer is compared with `search` on the same rows, the same evidence rule, and the same evaluation budget. Handing the model a relationship and asking it to recognize that relationship is an evaluation of the model. It is not the discovery comparison.

Charts and descriptions are projections of an evaluated proposition. They are not a second source of numbers. A verified proposition is a statement about the evaluated rows. It is not a policy recommendation.

## 2. Inputs

| Role | Required content |
| --- | --- |
| Table | The original records. Each record maps declared fields to a real number, a category, an ordered level, or missing. The caller retains this table for evaluation. |
| Population | Filters recorded on the proposition. A filter is a predicate on one record. The population is the records that satisfy every filter. |
| Candidate | Class, fields, aggregator, direction or relationship, and any declared split. Produced by a caller or, later, by a head on an Hx21 query. |
| Evidence rule | Minimum pairwise-complete count `n_min`, the uncertainty method and its parameters, and the margin rule. These are part of the proposition. They are fixed before the statistic is read. |
| Provenance seed | Dataset or snapshot identifier, source of the rows, and the code identity of the evaluator |

Vector length is not a population identity. Two tables with the same row count are different inputs when their snapshot identifiers differ.

## 3. Proposition

An evaluated insight is one structured value. The field names below are the conceptual schema. A later C++ struct may use the repository's naming style. It must carry the same roles.

| Field | Role |
| --- | --- |
| `id` | Identifier of this proposition |
| `insight_type` | `ranking`, `comparison`, `correlation`, `trend`, or `tradeoff` |
| `relationship` | The proposed relationship, as data: leading group and direction, signed difference, coefficient, slope sign, dominance set, or pairwise opposition |
| `variables` | Fields the relationship uses |
| `population` | Snapshot identifier and filters |
| `aggregation` | Aggregator and its parameters |
| `supporting_observations` | Count of records admitted, and the count excluded as missing or filtered |
| `computed_statistics` | Values written by the evaluator from the original rows |
| `uncertainty` | Interval or other estimate from the declared method. Empty when the method abstains |
| `model_confidence` | Optional score from a candidate model. Absent on the deterministic path |
| `provenance` | Section 7 |
| `validation_status` | Section 6 |
| `claim_level` | `exploratory_description` in this revision |
| `margin_rule` | The rule from section 4 that this proposition used |
| `dominance_basis` | On a trade-off, `point_estimate` in this revision |
| `chart_specification` | Present only after evaluation, and only for `verified` or `insufficient_evidence` |
| `semantic_description` | Present only with the chart. A template filled from this proposition |

`model_confidence` is not copied into `uncertainty`.

## 4. Shared arithmetic

Pairwise complete means every field named by the proposition is present on the record.

Aggregators are `min`, `max`, `mean`, `median`, and `quantile`. A quantile names its level `q` in `(0, 1)`. The sample median is the quantile at `q = 0.5` with a declared even-count rule, low or high, fixed on the proposition.

Ties in an order break by ascending group identifier. Group identifiers are caller data, compared as text in byte order.

The margin rules are:

| Rule | Holds when |
| --- | --- |
| `interval_excludes_zero` | The declared interval for a difference, slope, or correlation excludes 0 |
| `interval_excludes_one` | The declared interval for a ratio excludes 1 |
| `interval_excludes_next` | The leader's interval excludes the next group's point estimate. This is the initial ranking rule. It is not an interval on the gap. |
| `interval_on_gap` | A declared interval for the difference between two groups excludes 0. Specified for a later ranking check. The first evaluator does not implement it. |
| `sign_stable` | The interval lies entirely above 0 or entirely below 0, matching the stated direction |

The uncertainty method is named on the proposition. This revision does not install a hidden default. The baseline implementation must freeze one method, its parameters, and the absolute tolerance for "recomputed matches stated" before any fixture is scored.

## 5. Classes

These five classes are the initial scope. `insight_type` can gain a later value only by a new contract revision. The definitions below are the meanings an implementation has to meet.

### 5.1 Ranking

**Relationship.** Under a grouping key, or under the record identifier when no key is named, one group leads the others by an aggregator of one numeric field. The direction is `highest` or `lowest`.

**Computed.** The aggregator on each group that has at least one observed value; the total order; the leading group; the gap from the leader to the next distinct value; the count per group.

**Verified.** The recomputed leader and order match the relationship, and `interval_excludes_next` holds for every group that the relationship treats as strictly behind the leader.

**Limit of this rule.** The leader's interval is compared with the next group's point estimate. A small noisy gap can change who leads, and the check is weaker than an interval on the difference between the two groups. Every ranking proposition from the first evaluator records `margin_rule: interval_excludes_next`. The later rule `interval_on_gap` is the planned strengthening. The first evaluator may ship before that rule exists. It must not present the initial rule as a robust ranking.

This class orders groups by a statistic of a field. Hx01 owns nearest-source retrieval under a SpaceRef.

### 5.2 Comparison

**Relationship.** Two filters name two subpopulations of one numeric field. The stated quantity is a difference of aggregators, or a ratio when the denominator aggregator is nonzero. The null for a ratio is 1.

**Computed.** Both aggregators, both counts, and the difference or ratio.

**Verified.** The recomputed quantity matches, and `interval_excludes_zero` or `interval_excludes_one` holds, according to the quantity.

The claim is an observed contrast on the filtered rows. Hx06 owns controlled model responses and identified intervention effects. A comparison proposition does not carry those claim levels.

### 5.3 Correlation

**Relationship.** Two numeric fields on the same records move together or apart. The method is `pearson` or `spearman`, named on the proposition. The stated sign is positive or negative.

**Computed.** The coefficient on pairwise-complete records, and the count.

**Verified.** The recomputed coefficient has the stated sign, and `interval_excludes_zero` holds.

The relationship is statistical association. This revision's `claim_level` stays `exploratory_description`. A causal reading is a different operation.

### 5.4 Trend

**Relationship.** A numeric field changes along an ordered covariate. The covariate may be time or any other total order the caller declares. The method is one of:

| Method | What is computed |
| --- | --- |
| `spearman_of_bins` | A declared partition of the covariate. Each nonempty bin contributes the aggregator of the field. Spearman is taken between the bin order and that series. |
| `slope_on_index` | Ordinary least squares of the field on the rank of the covariate among pairwise-complete rows. The slope is the statistic. |
| `declared_split` | A split index `k` named in the candidate. The statistic is the difference of aggregators before and after `k`. |

**Verified.** The recomputed direction matches, and `sign_stable` holds.

`declared_split` does not search for `k`. A search policy requires a later revision that records the penalty and the selection rule before any index is chosen.

This class describes the observed order. Hx11 owns latent dynamics. Hx20 owns quantile forecasts. A trend proposition does not predict a future covariate value.

### 5.5 Trade-off

**Relationship.** Two or more numeric objectives, each with a direction `higher` or `lower`. Groups come from a grouping key. One of these forms is named:

| Form | What is stated |
| --- | --- |
| `pareto` | The set of groups that are non-dominated on the point estimates. Group A dominates B when A is at least as good on every objective and strictly better on one. |
| `opposition` | Two named groups and two named objectives. One group is better on the first objective and worse on the second. |

**Computed.** The aggregator of each objective in each named group, the counts, and the dominance set or the two gaps.

**Verified.** For `pareto`, the recomputed non-dominated set equals the stated set. For `opposition`, both gaps match and each gap satisfies `interval_excludes_zero` after the directions are aligned so that "better" is positive.

**Limit of point-estimate dominance.** The initial `pareto` form uses point estimates and records `dominance_basis: point_estimate`. A difference smaller than the noise can change which groups appear non-dominated. A later form, `robust`, treats A as dominating B when the declared intervals support "at least as good" on every objective and "strictly better" on one. The first evaluator does not implement `robust` and must not label a point-estimate set as robust.

The proposition does not choose a preferred group. Hx12 owns action choice. Preference among verified propositions is a later relevance stage.

## 6. Outcomes

| Status | When it is used |
| --- | --- |
| `verified` | Recomputation matches the relationship within the declared tolerance, and the evidence rule holds |
| `insufficient_evidence` | Recomputation matches, and the evidence rule fails. The statistic remains on the proposition. This is a completed evaluation |
| `rejected` | The rows contradict the stated relationship or the stated numbers |
| `abstained` | The named uncertainty method cannot be formed, including when the count is below that method's own minimum. Also when the table has no records or every record is a hole. The reason is `empty_set` in that case. See section 8 |
| `failed` | The proposition is malformed, a field is unknown, or a class is unknown |

`insufficient_evidence` is the outcome for a well-formed candidate that the rows do not support strongly enough. The evaluator must be able to return it. It must not invent a weaker claim to avoid that status.

A chart specification and a semantic description are attached only to `verified` and `insufficient_evidence`. The description of an `insufficient_evidence` proposition states that status. `rejected`, `abstained`, and `failed` produce neither a claim chart nor a claim sentence.

## 7. Provenance and uncertainty

Provenance on every evaluated proposition:

- snapshot or dataset identifier, and the origin of the rows when the caller has it
- filters and the aggregation
- records admitted, records excluded as missing, records excluded by filters
- analytical operation name and the evaluator version
- uncertainty method and its parameters, including a bootstrap seed when the method uses one
- candidate-model version when a model proposed the candidate, and otherwise an explicit absence
- when a search emitted the proposition: the search identifier, the candidate space, the budget, the number evaluated, the number pruned, the pruning rule, the rank score, and the multiplicity record from section 12

Uncertainty reports the interval from the named method. That interval is marginal: it is the interval for this proposition read on its own. Measurement limits, missing-data counts, and a caller-supplied selection note are stored when the caller supplies them. This revision does not estimate a selection bias that the caller did not declare. A search that looked at many candidates records that look in section 12. The marginal interval is not silently rewritten to look as if only one candidate had been examined.

`model_confidence` and the statistical interval answer different questions. A presentation may show both. It may not label the model score as the interval.

## 8. State fusion composition

Hx21 owns `code/fusion/state_fusion_transformer`. The callable surface used here is `encode` and `query` on `StateFusionModel`.

`encode` maps one unordered `SetBatch` to a state of shape `[batch, seeds, d_model]`. `query(state, query_batch)` answers against that state and does not encode the set again. One encoded state may serve several candidate queries.

The insight head, when it exists, reads `query`. It writes a candidate. It does not write `computed_statistics`. `query_scalars` reconstructs scalar features for the Hx21 capacity protocol. It is not the insight head.

A finished candidate is not the source of the first query, and a query is not the source of the candidate it was built from. The two learned paths break that circle in different places. Both use the same query dictionary below. Neither adds a function to Hx21.

**Approach A, skeleton-guided.** This is a baseline. `search`, or another declared skeleton policy, emits a skeleton: class, fields, aggregator, relationship form, and filters, with no statistic. `encode_query` turns that skeleton into one query row. `query(state, row)` returns an answer. A head may keep the skeleton or edit it. `evaluate` checks the candidate that comes out. This path asks whether the cached state helps choose among skeletons the policy already listed. It is not autonomous discovery.

**Approach B, intention-guided.** This is the research objective. A learned intention module reads only the cached state. It emits an intention in the `INSIGHT-QUERY 0.1.0` vocabulary. `encode_query` builds the row from that intention. `query(state, row)` runs, and a readout of the answer emits the candidate skeleton with empty statistics. `evaluate` checks it. The first query comes from the intention module. `search` does not supply the skeleton.

A study that runs only Approach A does not answer whether Hx21 can discover. The discovery comparison reports `search`, Approach A, and Approach B separately. Approach B is the learned-discovery claim.

### Query dictionary `INSIGHT-QUERY 0.1.0`

The study freezes a column map from table field names to scalar indexes and categorical slots. `encode_query` writes one query row.

The semantic vector carries the analytical intention. It is observed. Its width must be at least 16, which is the Hx21 default. A narrower width fails the query rather than dropping codes. Layout, indexes from 0:

| Indexes | Content |
| --- | --- |
| 0–4 | One-hot class: ranking, comparison, correlation, trend, trade-off |
| 5–9 | One-hot aggregator: min, max, mean, median, quantile |
| 10 | Quantile level `q`, or 0 when the aggregator is not quantile |
| 11 | Relationship-form code, as a float integer. Ranking: 0 highest, 1 lowest. Comparison: 0 difference, 1 ratio. Correlation: 0 pearson positive, 1 pearson negative, 2 spearman positive, 3 spearman negative. Trend: 0 slope increasing, 1 slope decreasing, 2 bin-Spearman increasing, 3 bin-Spearman decreasing, 4 declared split with a positive difference, 5 declared split with a negative difference. Trade-off: 0 pareto, 1 opposition |
| 12 | Primary scalar code `(index + 1)`, or 0 when the question has no primary scalar. This is the measured field, the first correlated field, or the first objective |
| 13 | Secondary code. A scalar index `i` is `(i + 1)`. A categorical slot `j` is `(scalar_count + 1 + j)`. 0 means none. This is the grouping field, the second correlated field, or the second objective |
| 14 | Number of filter slots, as a float |
| 15 | One predicate code for every filter in the row: 0 none, 1 `eq`, 2 `lt`, 3 `gt`. Mixed predicates are two intentions |

Scalar and categorical slots carry filters only. A filter slot is observed and holds the predicate's value. Every non-filter slot is a hole. Marking a measured field "observed" is not how the question names that field. Indexes 12 and 13 do that. A data record's semantic vector, when the record has one, is an embedding. A query row's semantic vector is this dictionary. The two uses share a tensor and do not share a meaning.

Two intentions that differ in class, aggregator, relationship form, primary field, secondary field, or filter signature must differ in the semantic vector or in the observation mask. The fixture that checks this compares query rows. It does not require a trained network, and it does not show that the network uses the difference.

One worked row is the question "which category has the highest median compensation?". Class ranking sets index 0. Aggregator median sets index 8. Index 10 is 0. Index 11 is 0, the highest form. Index 12 is the compensation field's scalar code. Index 13 is the category field's categorical code. Indexes 14 and 15 are 0. The compensation slot and the category slot are holes. The question is not encoded by marking compensation observed. A second intention that asks for the lowest mean of the same field changes the aggregator one-hot and sets index 11 to 1. Those two rows differ. A filter adds an observed slot that holds the predicate value, increments index 14, and sets index 15 to that one predicate. Mixed predicates in one row fail rather than sharing index 15.

`query(state, query_batch)` attends that row over the cached state. The set is not encoded again.

Hx21 rejects a set in which every record is unobserved. The proposer does not call `encode` on an empty table or on a table whose every record is a hole. It returns `abstained` with reason `empty_set`. That return counts as one unit of the evaluation budget. `evaluate` on the same table returns `abstained` with the same reason, so `observed_evidence` matches. The agreement is scored as agreement. It is not a miss against `generator_truth`.

The evaluator reads the original table. A hole policy in `SetBatch` does not replace the table.

This contract adds no block to the Hx21 package. An implementation that needs a new accessor records the gap before editing that package.

The deterministic evaluator and the deterministic search do not link LibTorch and do not require CUDA.

`Config` in Hx21 accepts one or two PMA seeds. The default is one. With one seed the pooled state is one vector of width `d_model`, and `query` attends to that single key. With two seeds the state is two vectors. Two seeds are the widest state the current package constructs.

The one-seed and two-seed comparison is a hypothesis, not a predicted winner. Hypothesis: a ranking on one field is recoverable from one seed, and a trade-off on two objectives has more of its structure available when the state has two seeds. The study reports those two classes separately under one seed, two seeds, and the table-feature proposer. Either outcome, including a tie, is a result about how much relational structure survives the pooled state. Neither outcome is assumed, and neither outcome by itself authorizes an edit to Hx21. A request for more than two seeds remains an edit to Hx21 and is not part of this contract.

The table-feature proposer is the non-Hx21 control. It emits candidates under the same evaluation budget, and each candidate is passed to `evaluate`. It does not return a feature score as its result. Its interface matches the Hx21 proposer: propose a candidate, then evaluate it.

## 9. Presentation

`chart_specification` names the chart form, the field bindings, and the series already stored in `computed_statistics`. The chart layer does not calculate a second statistic. If it repeats a calculation, it must match the proposition or refuse.

`semantic_description` is a class template filled from the proposition, including the validation status, the counts, the interval, and, for correlation, the word association. A free-form generator is outside this revision.

An embedding of that description, when a later caller requests one, uses an encoder the caller names. Hx15 through Hx18 own those encoders. Hx22 does not add a language-model subsystem. The first discovery comparison does not embed descriptions and does not retrieve charts. Nearest-chart matching can be added after that comparison. It is not on the path that asks whether Hx21 beats `search`.

## 10. Claim limits

This revision admits `exploratory_description` only. It refuses `controlled_model_response` and `identified_intervention_effect`.

Association, order, contrast, trend, and dominance are claims about the evaluated rows. They become claims about a wider population only under a later protocol that freezes the sample design.

Relevance may later reorder `verified` propositions. It does not change `validation_status`, and it does not promote `rejected` or `insufficient_evidence` to `verified`. The first discovery comparison does not call it. That comparison does not report usefulness. Usefulness enters only when a utility has been written down.

The comparison does report, for each arm:

- validity, as precision of `verified` propositions against sample-supported relationships
- diversity, as defined in section 12
- verified discoveries per evaluation-budget unit
- recall of the relationships the sample supports
- abstention rate, counting every `abstained` return, including `empty_set`, divided by budget units
- wall-clock time
- GPU time when the arm calls Hx21, and an explicit zero when it does not

A matched budget means the same number of `evaluate` calls, with each `empty_set` abstention charged as one call. It does not mean the same wall-clock or the same GPU time. An arm that agrees with `observed_evidence` by abstaining is visible in the abstention rate and in discoveries per budget unit. A report that omits any of these figures is an incomplete comparison.

## 11. Revision limit

INSIGHT-DISC 0.1.0 selects no population, no Non-Verba table, no checkpoint, and no Lean development. It registers no tool. The class definitions are the contract an implementation must meet. Meeting them on a fixture shows that one candidate was checked. It does not show that a procedure can discover a relationship, and it is not a demonstration on an application.

## 12. Discovery search

`search(table, space, budget, evidence_rule) -> ranked propositions and a search record`

| Input | Meaning |
| --- | --- |
| `space` | The declared family of candidates: which classes, fields, filters, aggregators, and directions may be proposed |
| `budget` | The maximum number of calls to `evaluate` |
| `evidence_rule` | The same rule `evaluate` would use on one candidate |
| Pruning | A named heuristic that may drop candidates before evaluation. Dropped candidates count toward the record and do not count as evaluations |

The initial policy is enumeration of `space` with that pruning. Each emitted proposition has been through `evaluate`. The rank score is named in the policy and is a function of the evaluated proposition. `model_confidence` is not that score.

Diversity of a ranked list is the number of distinct verified propositions under this tuple: `insight_type`, the variables in declared order, the aggregator, the relationship form, and the canonical filter signature. The filter signature is the ordered list of field, predicate, and value. The semantic description is not part of the tuple, so two wordings of the same ranking count once. A highest claim and a lowest claim differ in relationship form and stay distinct. An `empty_set` abstention is not a member of this count. It still consumes one budget unit.

The marginal interval on a proposition stays the interval from its own evidence rule. Searching many candidates can make the most extreme verified results accidents. A search result therefore carries one of these multiplicity records, chosen and frozen before the winners are read:

| Record | What it means |
| --- | --- |
| A named adjustment | The margin a searched proposition must pass is the declared adjustment for the number examined. The stored marginal interval is left as the unadjusted interval. |
| `confirmation` | A row split was declared before search. Search uses one part. The emitted propositions are evaluated again on the unused part. |
| `unadjusted` | No adjustment and no confirmation. Every emitted proposition says so, and the count examined is on the search record. |

A ranked list that omits this record is `failed`. The first search implementation uses `unadjusted` plus the count, or a named adjustment if that adjustment is frozen in the implementation note before any fixture is scored. It does not invent the adjustment after seeing the winners.

## 13. Synthetic labels

When a generator draws the rows, the example stores two different objects.

| Object | Meaning |
| --- | --- |
| `generator_truth` | The relationship entailed by the mechanism, including the parameters that entail it. It is not a validation status and it is not a sentence. |
| `observed_evidence` | The proposition or propositions that `evaluate` returns on the realized sample. |

A mechanism can entail a positive association while a small noisy sample shows a weak or reversed coefficient. The sample's status is then `insufficient_evidence` or `rejected`. That status agrees with the rows. A proposer is trained and scored against `observed_evidence`. It is not penalized for refusing a relationship the sample does not establish.

`generator_truth` records what was planted. It measures how often a planted relationship is visible in the sample, and it marks rare relationships for a separate slice. It is not a target that requires the proposer to assert an invisible relationship.

One table may carry several truths and several sample-supported propositions. The label is that set. A single caption is not the label.

Each class is planted by one short rule. `generator_truth` records the rule and its parameters. `observed_evidence` is whatever `evaluate` returns on the draw, including `abstained` when the draw is empty or every record is a hole.

| Class | Planting rule |
| --- | --- |
| Ranking | Groups differ by a declared shift on one field. The truth is the order of a named aggregator. |
| Comparison | Two filters and a declared difference on one field. |
| Correlation | Two fields and a declared association, plus noise. |
| Trend | One field along a declared ordered covariate, plus noise. |
| Trade-off | Two objectives whose group order is opposed. |

A draw may plant more than one rule. The truth is the set of rules that were planted. The sample may support only some of them.

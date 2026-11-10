# Insight discovery design

Hx22, 1 November 2026. This note records what Geometrics already provides, the smallest additions, and the plans for training, Lean 4, presentation, and Non-Verba. The normative class definitions are [INSIGHT-DISC 0.1.0](CONTRACT.md). Nothing in this note has been implemented or measured, except the Hx21 facts cited from that family's own records.

Guiding rule: the mathematics checks a candidate and, under a declared budget, searches for candidates. A learned model proposes candidates. It is trained against what the realized sample supports. The mechanism that drew the sample is a separate record.

The question Hx22 is built to answer: given the same rows and the same evaluation budget, does a proposer that reads an Hx21 state surface more valid and diverse propositions than the deterministic search? Usefulness joins that comparison only when a utility is declared. A verified proposition remains a statement about the evaluated rows, not a policy recommendation.

## 1. What is already in the repository

| Piece | Where | What it already does | What Insight Discovery does with it |
| --- | --- | --- | --- |
| State fusion | `code/fusion/state_fusion_transformer/` | Set Transformer: two ISAB blocks, PMA, `encode` to `[batch, seeds, d_model]`, `query` against that cached state. CUDA. Holes are flags, and order does not change `encode`. `query_scalars` is a linear probe used by the capacity protocol. | Call `encode` once and `query` for each candidate question. Do not add a second transformer. Do not use `query_scalars` as the insight head. |
| Hx21 evidence | The family README and the Hx21 hero | One synthetic panel dropped masked error from 1.25887 to 1.85048e-06. The capacity protocol, 1600 Adam steps, held-out filter feature 0 at 0.0925034 against a set-mean baseline of 0.348477. | Those figures show that a pooled state can answer a synthetic query. They are not insight-discovery results. |
| Hx01 retrieval | `code/operations/retrieval/` | Exact nearest sources under a SpaceRef, and a separate non-negative combination fit. | Ranking insights order groups by a field statistic. They do not call OR01-RET. |
| Hx06 comparisons | `code/operations/counterfactuals/` | OR06-PCR compares recorded responses of a frozen model. Observational and intervention profiles exist as contracts. | A comparison insight is an observed contrast of aggregators on two filters. It does not call OR06-PCR and does not take those claim levels. |
| Hx20 forecasts | `code/fusion/temporal_fusion_transformer/` | Quantile forecasts along a time axis, with its own uncertainty checks. | A trend insight describes an observed order. It does not forecast. |
| Encoders | `code/embeddings/encoders/` | Families that embed observations or text. | Used only if a caller embeds the description string. No new generator of sentences. |
| Tables | `code/data/`, `code/dataloaders/` | Loaders and observation windows. `SetBatch` is the fusion batch, with five categorical slots whose names are configuration. | The evaluator keeps the caller's table. Slot names are not the insight schema. |

No source file in this repository defines a Lean 4 project. Lean is not a dependency of the managed image as recorded in the environment configuration. No insight evaluator, synthetic insight generator, or insight checkpoint exists.

## 2. Smallest additions

The Hx21 package stays as it is. Insight Discovery adds files only under `code/operations/insight_discovery/`.

1. A deterministic evaluator, C++ and the standard library, with no LibTorch. It accepts a table and one candidate and returns an evaluated proposition. This checks arithmetic, intervals, and rejection. It does not discover which candidate to try. This is the first implementation milestone.
2. A deterministic search over a declared candidate space, with an evaluation budget, a named pruning rule, a rank score, and a multiplicity record. This is the discovery baseline. It calls the evaluator. It does not add a model.
3. Generators in the same directory. Each example stores `generator_truth` and `observed_evidence` as different objects. The generator does not live inside Hx21.
4. A candidate head that links `StateFusionModel`, calls `encode` and `query`, and emits candidates. It links the existing library. It does not copy ISAB or PMA. The study compares that head with the search at the same budget, and it compares one PMA seed, two PMA seeds, and a vector of ordinary table features.
5. An edit to Hx21 is justified only if `encode` or `query` cannot express the candidate questions, or if a recorded study shows that two seeds drop relationships the table-feature proposer still finds and a wider state is the proposed remedy. The current signatures can serve the first study. No edit is proposed now.
6. Lean, chart rendering, a stronger ranking interval, robust dominance, and a relevance model are later. They are not required to build the evaluator or the search.

`SetBatch` remains the way a caller presents rows to Hx21. The mapping from table columns to scalar features, categorical slots, and the semantic vector is caller configuration. The evaluator ignores that mapping and reads the table.

## 3. Path

```text
Unordered table
      │
      ├──────────────► deterministic search ──► candidates under a budget
      │                                              │
      ├──────────────► caller-supplied candidate     │
      │                                              │
      ▼                                              ▼
StateFusionModel.encode                    deterministic evaluator
      │                                              │
      ▼                                              ▼
cached state, one or two seeds            evaluated proposition
      │                                    (marginal interval,
      ▼                                     multiplicity record
StateFusionModel.query                      when a search emitted it)
      │                                              │
      ▼                                              │
candidate head ──────────────────────────────────────┘
      │
      ▼
chart_specification and semantic_description
```

The evaluator is the shared checker. The search is the discovery baseline and does not use Hx21. The learned path replaces the search's choice of candidates and still ends at the same evaluator, on the original table. With one PMA seed, `query` has one key. That bottleneck is measured by comparing one seed, two seeds, and table features. It is not patched in advance.

## 4. Interfaces to implement

These signatures are the plan. They are not a header in the tree yet.

`evaluate(table, candidate, evidence_rule) -> proposition`

Pure with respect to the table and the rule. The only randomness is a seed stored in the evidence rule when the uncertainty method needs one. The same inputs return the same proposition, including `insufficient_evidence` and `failed`. This function does not choose the candidate.

`search(table, space, budget, evidence_rule) -> ranked propositions, search record`

The discovery baseline. Enumerates the declared space, applies a named pruning rule, and calls `evaluate` until the budget is spent. The search record stores the space, the budget, the counts evaluated and pruned, the rank score, and either a named multiplicity adjustment, a confirmation split, or `unadjusted` together with the count examined. The marginal interval stored on each proposition stays the single-candidate interval.

`encode_query(intention, column_map) -> query row`

Builds one `SetBatch` row from `INSIGHT-QUERY 0.1.0`. The intention is a class, an aggregator, a relationship form, field indexes, and filters. The function does not read the cached state and does not compute a statistic.

`propose_skeleton(state, skeletons, budget) -> candidates`

Approach A. The skeletons come from `search` or from another declared policy. Each skeleton is encoded, `query` is called, and a head may keep or edit the skeleton. This is a baseline.

`propose_intentions(state, budget) -> candidates`

Approach B, the research objective. The intention module reads only `state`. It emits vocabulary codes. `encode_query` builds each row, `query` runs, and the readout emits a skeleton with empty statistics. No skeleton is passed in.

Both paths then call `evaluate`. The number of `evaluate` calls is limited by the same budget `search` receives. A comparison that gives either learned path a larger budget is a different study and is labeled as one. The report also gives wall-clock for every arm and GPU time for every arm that calls Hx21.

`propose_features(table, budget) -> candidates`

The non-Hx21 control. It emits candidates under that same budget. Each candidate then goes through `evaluate`. It does not return a feature score in place of a candidate.

On an empty table, or a table whose every record is a hole, Approach A, Approach B, and the table-feature proposer return `abstained` with reason `empty_set` and do not call `encode`. That return consumes one budget unit.

`project_chart(proposition) -> chart_specification`

`project_description(proposition) -> semantic_description`

Both refuse `rejected`, `abstained`, and `failed`.

A relevance function, later, may reorder a list of propositions whose status is already `verified`. It returns an order. It does not rewrite status or statistics.

## 5. Training plan

No generator has been written. The plan for the synthetic milestone is:

Each example is a finite table drawn from a declared generator. The example stores `generator_truth`, the relationships the mechanism entails, and `observed_evidence`, the propositions `evaluate` returns on that table. A table may carry more than one of each. Candidates the sample does not support are part of `observed_evidence`, with status `insufficient_evidence` or `rejected`.

| Class | Planting rule | Counterexample in the same family |
| --- | --- | --- |
| Ranking | Groups differ by a declared shift on one field. The truth is the order of a named aggregator. | Overlapping groups whose intervals fail `interval_excludes_next` |
| Comparison | Two filters and a declared difference on one field. | A difference whose interval contains 0 |
| Correlation | Two fields and a declared association, plus noise. | A coupling of 0, or a sign flip |
| Trend | One field along a declared ordered covariate, plus noise. | A flat series, or a split index that was not the entailed split |
| Trade-off | Two objectives whose group order is opposed. | A pair in which one group dominates, so the opposition claim is false |

`generator_truth` records the rule and its parameters. `observed_evidence` records what `evaluate` returns, including `abstained` with reason `empty_set` when the draw has no observed record. An empty draw is part of the fixture set.

Every family also draws missing entries, heavy-tailed noise, a small-sample regime, and a distractor field that looks related under a naive view and is independent under the generator. In the small-sample regime a planted positive association is allowed to come out weak or reversed. `generator_truth` still records the planted sign. `observed_evidence` records the sample. The training label for whether to assert the relationship is `observed_evidence`.

The candidate head is trained to propose relationships the sample supports, and to withhold relationships the sample does not support. The evaluator fills the numbers. The loss does not ask the network to emit the coefficient. The loss does not punish a refusal when `observed_evidence` is `insufficient_evidence` or `rejected`, even if `generator_truth` planted the relationship.

Negative examples are first-class. A training set that contains only supported relationships does not teach `insufficient_evidence`. A training set that treats generator truth as the assertion target teaches the model to claim relationships its rows do not show.

### Generalization

A report separates these tests:

| Test | What changes |
| --- | --- |
| Resample | New draws from a generator family used in training |
| Recombination | Familiar relationships composed in a new combination |
| New distribution | A family whose parameter law was held out |
| New geometry of the table | Set size and missingness outside the training ranges |
| New relationship | A class or a method absent from training |

The generalization claim uses the last three rows. A score on new seeds of a training generator is a resample score. The generator-family identifier and the parameter ranges are part of the split. They are fixed before training.

## 6. Lean 4

Lean 4 is a possible home for the reusable rules, not a dependency of Geometrics.

Worth specifying, if a later milestone admits the work:

- The total order of ranking, including the tie break.
- Dominance for the Pareto form.
- The aggregator identities used by the fixtures, so the C++ results match the specification on those fixtures.

A proof about those functions establishes that the algorithm matches the specification. It does not establish that a sample is unbiased, that an association is causal, or that a model score is a confidence interval.

Individual propositions are checked by the evaluator. They do not each carry a Lean proof.

The managed environment has no Lean project. Installing Lean is a separate environment change and is not part of the baseline.

## 7. Non-Verba

Non-Verba is the first intended application. Requesters and operators would read propositions about activity, opportunities, compensation, risk, and operational relationships. Those words are application fields. They are not types in `evaluate`.

A future binding can map a Non-Verba snapshot onto the five classes. That binding names the snapshot, the field dictionary, and the evidence rule. It lives with the application or in a later binding record. It is not compiled into the evaluator.

Operators and requesters may later prefer different verified propositions. That preference is the relevance stage. It is not required to build or test the evaluator.

## 8. Roadmap and milestones

| Phase | Hero milestone | State on 1 November 2026 |
| --- | --- | --- |
| 1. Discipline, including evaluation versus discovery | `insight-discovery-record-contract`, `insight-discovery-record-discovery-question`, `insight-discovery-record-handoff`, `insight-discovery-record-query-source` | Done. Documents only. |
| 2. Deterministic evaluation of one candidate | `insight-discovery-deterministic-baseline` | Ready to implement. Not started. Does not wait on the query dictionary. |
| 3. Deterministic search under a budget | `insight-discovery-deterministic-search` | Proposed. This is the discovery baseline. |
| 4. Stronger rank and dominance checks | `insight-discovery-uncertainty-contrasts` | Proposed. Does not block phase 2. |
| 5. Synthetic tables with both labels | `insight-discovery-synthetic-labels` | Proposed. |
| 5b. Query-row dictionary | `insight-discovery-query-encoding` | Ready. Distinct intentions must produce distinct rows. Does not call Hx21 and does not block phase 2. |
| 6. Learned discovery on a cached Hx21 state | `insight-discovery-state-fusion-candidates` | Proposed. Reports `search`, Approach A, and Approach B separately, at a matched count of `evaluate` calls. |
| 7. Lean specifications of reusable rules | `insight-discovery-lean-rules` | Proposed. |
| 8. Chart and description projections | `insight-discovery-presentation` | Proposed. |
| 9. Relevance | `insight-discovery-relevance` | Proposed. This is where a declared usefulness enters. |

Phase 2 checks arithmetic. Phase 3 is the first procedure that can discover a candidate. Phase 5b checks that the dictionary rows are distinguishable. Phase 6 asks whether Approach B improves on phase 3. Approach A is reported beside it and is not that claim. A claim about Approach B requires the same rows, the same evidence rule, and the same number of `evaluate` calls, plus wall-clock and GPU time. The one-seed versus two-seed split stays a hypothesis.

## 9. How a later study will be scored

The measures below are the ones a bound study has to report. None of them has a number yet.

- Fixture agreement for `evaluate`: each class, a supported case, an unsupported case, and one abstention.
- For search: recovery of sample-supported relationships inside the budget, the share of emitted propositions that are `verified`, and a search record that states the multiplicity choice.
- Agreement of a proposer's assertions with `observed_evidence`, including refusals. A refusal that matches the sample is a success even when `generator_truth` planted the relationship.
- Visibility: how often `generator_truth` is supported by the sample. This is a property of the draw, not a score against the proposer.
- Precision and recall of `verified` discoveries against sample-supported relationships, split by resample, recombination, and held-out family.
- The same comparison for the learned proposer and for `search`, at the same budget.
- Ranking and trade-off reported separately under one PMA seed, two PMA seeds, and the table-feature proposer, for `search`, Approach A, and Approach B. The hypothesis is that ranking survives one seed and trade-off has more structure at two seeds. The study does not treat either side as the default finding.
- The table-feature condition emits candidates and calls `evaluate` under the same budget. It is not a score computed from features alone.
- Empty and all-hole tables: the proposer abstains, the abstention consumes one budget unit, and agreement with `observed_evidence` counts. It is not a miss.
- Diversity: the number of distinct verified relationships under the tuple (`insight_type`, variables in declared order, aggregator, relationship form, canonical filter signature). The description text is not part of the tuple.
- Verified discoveries per budget unit, recall of sample-supported relationships, and abstention rate, each on its own.
- Wall-clock for every arm, and GPU time for every arm that calls Hx21. Matched `evaluate` counts are not reported as matched cost.
- The first study does not report usefulness and does not embed descriptions.
- Coverage of the declared uncertainty method on the fixtures where the method's assumptions hold.
- A statement of what was proved in Lean, and what remains an assumption about the rows.

## 10. Limitations of this record

- No evaluator, search, generator, head, chart projector, or relevance model is in the tree.
- No synthetic table has been drawn, and no insight metric has been computed.
- No Lean file has been checked.
- Hx21's published errors are pipeline and capacity figures for that family. They do not measure whether one pooled seed preserves a rare relationship.
- The five classes do not cover causation, retrieval, forecasting, or action choice.
- The initial ranking rule compares a leader interval with the next point estimate. The initial Pareto rule uses point estimates. Both limits are recorded. Neither stronger rule is implemented.
- A verified proposition is a statement about the evaluated rows under the recorded rule. It is not a policy recommendation.
- `evaluate` is not a discovery baseline. No search has been run, so the budget-matched question has no answer.

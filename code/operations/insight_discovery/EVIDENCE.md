# Insight discovery Requirements-and-Evidence Record

**Record:** `INSIGHT-DISC-RER`, revision `0.1.0`, 1 November 2026.
**Operation:** [INSIGHT-DISC / insight-discovery, revision 0.1.0](CONTRACT.md).
**Design:** [DESIGN.md](DESIGN.md).
**Purpose:** requirements for a later instance, and the evidence boundary of the documents written on 1 November 2026. This record is not an executable study protocol.

## 1. Hypothesis and claim

No population hypothesis is bound.

The working scientific question, for a future instance, is whether Approach B surfaces more valid and diverse propositions than a deterministic search, on the same rows, the same evidence rule, and the same evaluation budget, including on generator families excluded from training. Approach B reads the cached Hx21 state and emits an `INSIGHT-QUERY 0.1.0` intention. Approach A only refines a skeleton the search already listed. A run of Approach A alone does not answer the question. The comparison also reports verified discoveries per budget unit, recall of sample-supported relationships, abstention rate, wall-clock, and GPU time. That question is not tested here. Checking a candidate that was already chosen is a different and smaller question.

This revision admits `exploratory_description` only. Correlation states association. Comparison states an observed contrast. Neither states a cause. Trade-off does not select an action.

## 2. Data and provenance

An instance binds the table snapshot, field dictionary, filters, generator family and parameter ranges when the rows are synthetic, the evidence rule, the search space and budget, the multiplicity record, the evaluator version, and any candidate-model checkpoint. A synthetic instance stores `generator_truth` and `observed_evidence` as separate objects.

**Current bindings:** none. Non-Verba is named in the design note as the first intended application. No Non-Verba table is bound.

## 3. Controls and baselines

`evaluate` is the checker for one candidate. It is not the discovery baseline.

`search` is the discovery baseline. Approach B is the learned-discovery arm. Approach A is the skeleton-guided baseline. Both are compared with `search` on the same rows and the same number of `evaluate` calls. A comparison that gives one arm a larger evaluation budget is reported as a different study. Equal `evaluate` counts are not equal wall-clock or GPU time, and the report says so.

`observed_evidence` is the control for what the sample supports. `generator_truth` is the control for what the mechanism planted. A caption written after seeing a table is neither. A refusal that matches `observed_evidence` is not scored as a miss against `generator_truth`.

Hx21 masked reconstruction and the Hx21 capacity protocol are controls for that family. They are not baselines for insight status.

## 4. Evaluation protocol

**Current protocol:** none. These documents grant no execution admission.

A later protocol freezes, before scoring:

- generator families and parameter ranges in the training split and in the held-out split
- `n_min`, the uncertainty method, its parameters, and the margin rule
- the candidate space, the evaluation budget, the pruning rule, the rank score, and the multiplicity record
- the tolerance for a recomputed statistic
- the metrics in section 5
- the rule that a resample of a training family is reported separately from a held-out family
- the representation conditions: one PMA seed, two PMA seeds, and a table-feature proposer

Every attempt stays indexed, including `insufficient_evidence`, `rejected`, `abstained`, and `failed`.

## 5. Measures of success

For the evaluator, success is fixture agreement: each class, a supported case, an unsupported case, and one abstention, with status and recomputed values matching the hand-computed fixture. That success shows that one candidate was checked. It does not transfer to discovery or to an application.

For the search, success on a fixture is a ranked list whose members match `evaluate`, plus a search record that states the space, the budget, the counts, and `unadjusted` or a named adjustment frozen beforehand.

For a later learned study, each arm reports precision of `verified` propositions, recall of sample-supported relationships, verified discoveries per budget unit, abstention rate, and diversity. Abstentions, including `empty_set`, count in the abstention rate and are not folded into precision. The arms are `search`, Approach A, and Approach B, plus the table-feature proposer. Scores are split by resample and by held-out family. Ranking and trade-off are reported separately under one seed and under two seeds, as a hypothesis rather than as a predicted winner. Wall-clock is reported for every arm. GPU time is reported for every arm that calls Hx21, and an explicit zero otherwise. The first study does not report usefulness and does not embed descriptions. No target number is set in this revision. Setting one is part of binding the instance, before the run.

## 6. Robustness

A later instance records the sample unit and the uncertainty method's assumptions. A result on one generator family stays with that family. Missingness, set size, and distractor fields are part of the held-out design in [the training plan](DESIGN.md).

## 7. Results and limitations

No insight run has been executed. There is no numeric insight result.

Documented on this date, and not a result of this operation:

- The Hx21 package exposes `encode` and `query`. Its synthetic capacity measurement is recorded by Hx21. It is not an insight metric.
- The repository contains no Lean 4 development and no insight evaluator.

The class definitions, the separation of candidate from validity, the separation of `evaluate` from `search`, the two synthetic labels, the split between Approach A and Approach B, the `INSIGHT-QUERY 0.1.0` dictionary, and the decision to keep both deterministic paths free of LibTorch are specified. They are not demonstrated in code. The distinguishability fixture is not implemented. The initial ranking and Pareto rules have recorded limits. The stronger rules are not implemented. The one-seed versus two-seed comparison is a hypothesis. This record does not pick a winner.

## 8. Reproduction pointer

| Item | Identity |
| --- | --- |
| Contract | `code/operations/insight_discovery/CONTRACT.md`, INSIGHT-DISC 0.1.0 |
| This record | `code/operations/insight_discovery/EVIDENCE.md`, INSIGHT-DISC-RER 0.1.0 |
| Design | `code/operations/insight_discovery/DESIGN.md` |
| Entry | `code/operations/insight_discovery/README.md` |
| Date | 1 November 2026 |
| Hx21 surface consulted | `code/fusion/state_fusion_transformer/include/state_fusion/model.h` |

No dataset, checkpoint, seed, or artifact belongs to this revision.

## 9. Decision

`exploratory` interface. The five classes, the evaluator, and the discovery search are specified. No study is promoted. The next implementation step is the evaluator. The search follows it. Neither has been started.

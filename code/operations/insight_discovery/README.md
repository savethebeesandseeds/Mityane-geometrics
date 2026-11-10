# Insight discovery

Hx22 owns this directory. The owner specified the discipline on 1 November 2026.

The operation discovers relationships in an unordered set of observations and records each relationship as a structured proposition. `evaluate` checks one candidate against the original rows. `search` is the discovery baseline: it enumerates a declared space under an evaluation budget and records how many candidates were examined. A sentence and a chart are projections of an evaluated proposition.

The question for a later study is whether a proposer that reads an Hx21 state finds more valid and diverse propositions than `search`, on the same rows and the same budget. The first query comes from a learned intention vocabulary, not from a finished candidate. A skeleton from `search` is a separate baseline. `INSIGHT-QUERY 0.1.0` writes the class, aggregator, relationship form, and field roles into the semantic vector. The first study reports validity, diversity, discoveries per budget unit, recall, abstention rate, wall-clock, and GPU time. It does not embed descriptions and does not score usefulness.

[INSIGHT-DISC 0.1.0](CONTRACT.md) defines the five classes: ranking, comparison, correlation, trend, and trade-off. It also records the initial ranking and Pareto rules and their limits, the split between generator truth and sample evidence, and the multiplicity record on a search. The linked [Requirements-and-Evidence Record](EVIDENCE.md) is INSIGHT-DISC-RER 0.1.0. The [design note](DESIGN.md) records the existing pieces, the smallest additions, the training plan, the Lean 4 boundary, and Non-Verba as the first intended application.

No evaluator, search, generator, or checkpoint is in this directory. The evaluator is specified and is the next implementation. Hx21 remains the State Fusion Transformer. A learned candidate may call `encode` and `query`. This operation does not add a transformer.

OR-01 through OR-13 stay the charter families. This operation uses the identifier INSIGHT-DISC.

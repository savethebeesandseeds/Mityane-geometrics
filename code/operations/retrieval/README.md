# Hx01: retrieval and similarity matching

This component owns Hx01's work in **OR-01 — Retrieval and similarity matching**. The reusable operation is **OR01-RET 0.1.0**, ranked source retrieval. The query is one point in an embedding space. Each corpus member is another point in that same space. Exact search returns the source identities of the nearest points under a declared metric and normalization. Relevance stays a separate rule.

Retrieval takes a coordinate in a declared SpaceRef. Encoding text or other raw content into that coordinate is upstream and stays outside this operation. Two methods are named. **Exact k-retrieval** returns the nearest sources. **Positive sparse combination** approximates the coordinate as a sparse non-negative combination of corpus vectors. OR01-COMB is implemented beside the ranker and is not the registered tool.

Two general profiles are defined and remain unbound as milestones. `text-gemini-embedding-2` is text input to the Hx18 Gemini Embedding 2 provider. `local-saved-vectors` is a frozen snapshot from a local encoder instance or a synthetic generator. One English instance of the local profile is bound. OR01-RET-ABS 0.1.0 freezes those texts and one EmbeddingGemma 2 text SpaceRef. OR01-RET-ABS-EVAL 0.1.0 scores the deprecated geometric queries on the seven eligible sets: the shared residual is inconclusive, and the difference is negative. That score stays indexed. It is not the current retrieval strategy.

## Records

| Record | Scope |
| --- | --- |
| [OR01-RET contract](CONTRACT.md) and [RER](EVIDENCE.md) | Input and output roles, SpaceRef, exclusions, claim limits, and the two general profiles that remain unbound |
| [OR01-RET-ABS 0.1.0](bindings/oewn-2025/BINDING.md) | English abstract text population, shared-principle labels, and one local text SpaceRef. Vector hashes are recorded there |
| [OR01-RET-ABS-EVAL 0.1.0](bindings/oewn-2025/PROTOCOL.md) | Historical score of the deprecated geometric queries. The result is [oewn-abstract-evaluation-v0.1.0.md](reviews/oewn-abstract-evaluation-v0.1.0.md) |
| [C++ exact-search reference](IMPLEMENTATION.md) | Engineering-only cosine and L2 search, plus OR01-COMB. The registered tool links only exact search |
| [Deprecated constructors](include/mityane/operations/retrieval/geometry.h) | Internal difference, shared residual, and cone utilities. They are not the retrieval strategy |

**Exact k-retrieval** is the registered tool **OR01-RET 0.1.0** in [TOOLS.md](../TOOLS.md). It accepts a coordinate, returns the nearest sources, and refuses live calls and unbound relevance claims. **Positive sparse combination** is specified in the contract as OR01-COMB. It prepares vectors with the same `prepare()` as exact search, solves non-negative least squares, refits to a declared minimum weight, and keeps those weights in the result. A fit floor requires the reconstruction cosine to meet `gamma`. A path-length ceiling empties a wandering combination. The solve does not construct a negation. Exact search remains the fidelity baseline. Difference, shared residual, and cone functions remain in [geometry.h](include/mityane/operations/retrieval/geometry.h) as internal utilities. The tool does not expose them. The English abstract texts, labels, and one local text SpaceRef are bound by OR01-RET-ABS 0.1.0. The geometric-query score is [oewn-abstract-evaluation-v0.1.0.md](reviews/oewn-abstract-evaluation-v0.1.0.md): the shared residual is inconclusive and the difference is negative. An exploratory neighborhood listing is [oewn-abstract-neighborhoods-v0.1.0.md](reviews/oewn-abstract-neighborhoods-v0.1.0.md). Encoding stays with the producer that wrote the coordinate.

The reference does not call a provider or read a dataset.

```powershell
.\container.ps1 exec make test-retrieval-tool
.\code\operations\tools.ps1 describe OR01-RET
```

Current priorities and acceptance criteria belong to the [Hero record](../../../web/dashboard/heroes/01-retrieval.json).

## Check

From the repository root, through the managed container:

```powershell
.\container.ps1 exec make test-retrieval
```

The target builds and runs the standard-library test. It does not download data or contact a service.

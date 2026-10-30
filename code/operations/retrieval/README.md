# Hx01: retrieval and similarity matching

This component owns Hx01's work in **OR-01 — Retrieval and similarity matching**. The reusable operation is **OR01-RET 0.1.0**, ranked source retrieval. The query is one point in an embedding space. Each corpus member is another point in that same space. Exact search returns the source identities of the nearest points under a declared metric and normalization. Relevance stays a separate rule.

Two general profiles are defined and remain unbound as milestones. `text-gemini-embedding-2` is text input to the Hx18 Gemini Embedding 2 provider. `local-saved-vectors` is a frozen snapshot from a local encoder instance or a synthetic generator. One English instance of the local profile is bound and scored. OR01-RET-ABS 0.1.0 freezes those texts and one EmbeddingGemma 2 text SpaceRef. OR01-RET-ABS-EVAL 0.1.0 scores the seven eligible queries: the shared residual is inconclusive, and the difference is negative. That score is this population and this SpaceRef only.

## Records

| Record | Scope |
| --- | --- |
| [OR01-RET contract](CONTRACT.md) and [RER](EVIDENCE.md) | Input and output roles, SpaceRef, exclusions, claim limits, and the two general profiles that remain unbound |
| [OR01-RET-ABS 0.1.0](bindings/oewn-2025/BINDING.md) | English abstract text population, shared-principle labels, and one local text SpaceRef. Vector hashes are recorded there |
| [OR01-RET-ABS-EVAL 0.1.0](bindings/oewn-2025/PROTOCOL.md) | Frozen scoring rule for the seven eligible queries. The result is [oewn-abstract-evaluation-v0.1.0.md](reviews/oewn-abstract-evaluation-v0.1.0.md) |
| [C++ exact-search reference](IMPLEMENTATION.md) | Engineering-only cosine and L2 search, geometric constructors, and the saved-vector inspector |

Two uses share this operation and this ranker. **Exact k-retrieval** takes a query vector that is already in hand and returns the nearest sources. That is the registered tool **OR01-RET 0.1.0** in [TOOLS.md](../TOOLS.md). It accepts exploratory saved-vector requests and refuses live calls and unbound relevance claims. **Geometric abstract retrieval** builds a difference, a shared residual, or a cone in [geometry.h](include/mityane/operations/retrieval/geometry.h), then calls the same exact search. The tool does not expose those constructors. The English abstract texts, labels, and one local text SpaceRef are bound by OR01-RET-ABS 0.1.0. The scored comparison is [oewn-abstract-evaluation-v0.1.0.md](reviews/oewn-abstract-evaluation-v0.1.0.md): the shared residual is inconclusive and the difference is negative. An exploratory neighborhood listing is [oewn-abstract-neighborhoods-v0.1.0.md](reviews/oewn-abstract-neighborhoods-v0.1.0.md). Encoding stays with the encoder or provider that produced the vectors.

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

# Hx01: retrieval and similarity matching

This component owns Hx01's work in **OR-01 — Retrieval and similarity matching**. The reusable operation is **OR01-RET 0.1.0**, ranked source retrieval. The query is one point in an embedding space. Each corpus member is another point in that same space. Exact search returns the source identities of the nearest points under a declared metric and normalization. Relevance stays a separate rule.

Two profiles are defined and unbound. `text-gemini-embedding-2` is text input to the Hx18 Gemini Embedding 2 provider. `local-saved-vectors` is a frozen snapshot from a local encoder instance or a synthetic generator. Neither profile admits a run.

## Records

| Record | Scope |
| --- | --- |
| [OR01-RET contract](CONTRACT.md) and [RER](EVIDENCE.md) | Input and output roles, SpaceRef, exclusions, claim limits, and the two unbound profiles |
| [C++ exact-search reference](IMPLEMENTATION.md) | Engineering-only cosine and L2 search, pairwise similarity, and neighborhood diagnostics on caller-supplied arrays |

The reference does not call a provider or read a dataset. The registered tool **OR01-RET 0.1.0** exposes that reference through the common envelopes in [TOOLS.md](../TOOLS.md). It accepts exploratory saved-vector requests and refuses live calls and unbound relevance claims. Encoding stays with the encoder or provider that produced the vectors.

Geometric constructors in the same library build a difference, a shared residual, or a cone, and exact search ranks that neighborhood. They are engineering functions over caller-supplied arrays. The registered tool does not expose them. No abstract corpus is bound.

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

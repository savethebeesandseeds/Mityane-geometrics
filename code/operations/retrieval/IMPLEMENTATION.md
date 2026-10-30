# OR01-RET exact-search reference

Standard-library C++ for the numerical slice of [OR01-RET 0.1.0](CONTRACT.md). The query is one point in an embedding space. Each corpus member is another point in that same space. The reference returns the source identities of the nearest points under cosine or L2. It does not load a dataset, call Gemini, read an encoder checkpoint, mint an OperationRun, or judge relevance.

| Piece | Path |
| --- | --- |
| Interface | [exact_search.h](include/mityane/operations/retrieval/exact_search.h) |
| Implementation | [exact_search.cpp](src/exact_search.cpp) |
| Checks | [exact_search_test.cpp](tests/exact_search_test.cpp) |

Build and run inside the managed container:

```text
make test-retrieval
```

## Geometry

| Choice | Rule |
| --- | --- |
| `cosine` | Rank by descending cosine. Distance is `1 - cosine`. |
| `l2` | Rank by ascending Euclidean distance. Similarity is `1 / (1 + distance)`. |
| `normalization: l2` | Divide by the L2 norm before the metric. A zero vector is undefined. |
| `normalization: none` | Use the supplied values. Cosine is still undefined at norm zero. L2 of a zero vector is defined. |
| Ties | Exact binary64 equality, then ascending source ID. |
| `k` | At least 1. A smaller eligible set returns every eligible member. |

Pairwise similarity uses the same preparation and the same two numbers. Neighborhood diagnostics are leave-one-out kNN: hub count is how often a source appears in other members' neighborhoods, and mean neighbor distance is the mean of that point's own neighbor distances. The diagnostics do not apply a caller's exclusion list.

## Outcomes

`ranked`, `empty`, `abstained`, and `failed` stay distinct. Skipped members are listed as `self_match`, `excluded_source`, or `undefined_geometry`. A query binding that disagrees in space ID, snapshot ID, metric, normalization, or dimensions fails as an incompatible space. Duplicate source IDs, empty IDs, non-finite values, and the wrong length fail the snapshot. Source-window overlap is the caller's exclusion list or a snapshot that already removed those members.

## Caller responsibilities

The caller keeps the OperationRun ID, contract revision, claim level, artifact hashes, and evidence destination. The text profile's request digest and response artifact stay with Hx18. The local profile's checkpoint or generator identity stays with the encoder record. This library's result is engineering evidence of the arithmetic only.

## Operation envelope

`OR01-RET` 0.1.0 is registered on the common tool boundary. `check` and `run` send saved vectors to this reference. The call retains the OperationRun id, claim level, SpaceRef, code and configuration hashes, and fitted-artifact id. Live calls, unbound relevance, and causal claim levels are refused before ranking. As-of members that are not yet available, and source windows excluded for overlap, are reported skips. A mismatched saved-vector artifact fails before the reference runs.

```text
make test-retrieval-tool
```

The recorded run is [envelope-v0.1.0.md](reviews/envelope-v0.1.0.md). Pairwise similarity and neighborhood diagnostics remain direct calls to this library.

## Check recorded with this revision

`make test-retrieval` runs seven checks: cosine tie order, L2 order and pairwise agreement, L2 normalization, reported exclusions, empty versus failure, refusal cases, and hubness on a three-point line. The recorded run is [exact-search-v0.1.0.md](reviews/exact-search-v0.1.0.md). A later look the same day repeated those checks, passed the nine envelope checks, and confirmed the visible order: L2 distance 1 before distance 4, and cosine 1 before cosine 0. That look is [nearest-source-v0.1.0.md](reviews/nearest-source-v0.1.0.md).

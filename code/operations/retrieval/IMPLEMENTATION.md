# OR01-RET exact-search reference

Standard-library C++ for the numerical slice of [OR01-RET 0.1.0](CONTRACT.md). The query is one point in an embedding space. Each corpus member is another point in that same space. The reference returns the source identities of the nearest points under cosine or L2. It does not load a dataset, call Gemini, read an encoder checkpoint, mint an OperationRun, or judge relevance.

| Piece | Path |
| --- | --- |
| Exact k-retrieval | [exact_search.h](include/mityane/operations/retrieval/exact_search.h), [exact_search.cpp](src/exact_search.cpp) |
| Geometric abstract retrieval | [geometry.h](include/mityane/operations/retrieval/geometry.h), [geometry.cpp](src/geometry.cpp) |
| Saved-vector inspector | [saved_vector_inspect.cpp](src/saved_vector_inspect.cpp) |
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

## Geometric abstract retrieval

`geometry.cpp` builds a query from saved vectors and then calls `search()`. Exact k-retrieval stays in `exact_search.cpp`. The registered tool links only that ranker. `difference` returns prepared target minus prepared origin. `shared_direction` removes the span of differences among two to eight points and returns the residual of the first point. `make_cone` returns a unit axis, an aperture in radians, and optional radial bounds. `search_cone` ranks eligible members against that axis and reports everyone else as `outside_cone`. A zero prepared vector has no angle and is `undefined_geometry`.

A zero difference or a zero residual is `collinear_inputs`. A zero-norm input under cosine or L2 normalization abstains. `cone_diagnostics` reports leave-one-out hubness and mean neighbor distance on the members inside the cone. Purity is the fraction of those members named by the caller. An empty name list leaves purity unset. None of these figures is a relevance result, and none closes Hx02.

The 30 October 2026 run printed `geometric constructors: 2 checks passed` after the seven exact-search checks. The fixtures use `[1, 0]`, `[4, 0]`, `[1, 1]`, and a three-axis residual `[0, 0, 1]`. The record is [geometric-operators-v0.1.0.md](reviews/geometric-operators-v0.1.0.md). The registered tool does not expose these functions.

## Saved-vector inspection

`saved_vector_inspect` reads two `mityane-saved-vectors-v1` files and a query list. The commands are `point`, `difference`, `shared`, and `cone`. The binary links `exact_search.o` and `geometry.o`. The registered tool links only the ranker. Build the inspector with `make build-retrieval-inspect`.

The English texts, labels, and local EmbeddingGemma 2 SpaceRef are [OR01-RET-ABS 0.1.0](bindings/oewn-2025/BINDING.md). The frozen scoring rule is [OR01-RET-ABS-EVAL 0.1.0](bindings/oewn-2025/PROTOCOL.md). The score is [oewn-abstract-evaluation-v0.1.0.md](reviews/oewn-abstract-evaluation-v0.1.0.md). On the seven eligible queries, mean full-list binary nDCG is 0.688317949329 for the point baseline, 0.421848381196 for the lexical baseline, and 0.662943315673 for the shared residual. The protocol calls the shared residual inconclusive. On the six pairs, the difference mean is 0.390420899151, below both pair baselines, so the protocol calls it negative. Cone membership and purity stay out of that claim. The ranks were listed before the score, and the protocol has no unseen split. This library does not load the WordNet snapshot or the encoder weights. The vector bytes stay under ignored `.temp/`.

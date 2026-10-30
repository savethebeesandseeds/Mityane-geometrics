# Geometric constructors check, 30 October 2026

**Scope:** engineering constructors for OR01-RET 0.1.0. A difference, a shared residual, or a cone is built from caller-supplied vectors. Ranking inside the cone is exact search. No abstract corpus, encoder, Gemini call, approximate index, or relevance claim.

**Command:** `make test-retrieval` in the managed container `mityane-geometrics`, compiler `g++ -std=c++20 -Wall -Wextra -Wpedantic`.

**Result:** exit 0. The compiler printed no warnings. The test printed `exact-search reference: 7 checks passed` and `geometric constructors: 2 checks passed`. `make build-retrieval-tool` then relinked the registered tool with no warnings. The tool does not expose the constructors.

The constructor checks cover:

- `[4, 0] - [1, 0] = [3, 0]` under L2 with no normalization.
- The shared residual of `[1, 1]` and `[2, 1]` is `[0, 1]`.
- The shared residual of `[1, 0, 1]`, `[2, 0, 1]`, and `[1, 1, 1]` is `[0, 0, 1]`.
- Collinear inputs, a zero difference, a non-finite value, and positive scale copies under L2 normalization are refused.
- A zero aperture around `[1, 0]` keeps `[2, 0]` at distance 1 and reports the other points as `outside_cone`.
- A full aperture ranks `[2, 0]`, then `[0, 1]`, then `[-1, 0]` by exact L2 distance to the unit axis.
- A radial maximum of 1.5 removes `[2, 0]` and keeps exact-search order for the remaining points.
- A zero vector is an `undefined_geometry` skip. A negative aperture and a mismatched axis length are refused. An empty cone stays empty.
- Cone purity for one labeled member among three is one third. A repeated label counts once. An empty label list leaves purity unset. Hubness remains leave-one-out inside the cone.

This run does not admit either retrieval profile and does not bind an abstract corpus.

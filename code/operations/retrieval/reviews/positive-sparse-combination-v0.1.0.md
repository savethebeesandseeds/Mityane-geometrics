# Positive sparse combination check, 30 October 2026

**Scope:** OR01-COMB engineering fixtures. Non-negative least squares on prepared vectors, a refit to a declared minimum weight, a fit floor, a path-length ceiling, and the rank-1 early stop. No corpus, encoder, Gemini call, approximate index, or relevance claim.

**Command:** `make test-retrieval` in the managed container `mityane-geometrics`, compiler `g++ -std=c++20 -Wall -Wextra -Wpedantic`.

**Result:** exit 0. The compiler printed no warnings. The test printed `exact-search reference: 7 checks passed`, `geometric constructors: 2 checks passed`, and `positive sparse combination: 6 checks passed`.

The combination checks cover:

- `(1, 0)` and `(1, 0.25)` recover weights 1 and 1 for target `(2, 0.25)`. The reconstruction cosine is above 0.999 and the path ratio stays under 1.05.
- Target `(4, 0)` stops on column `(2, 0)` with one-vector weight 2.
- Target `(1, 1)` on the two axes keeps weights 1 and 1. The path ratio is sqrt(2).
- The same pair with ceiling 1.2 is empty, reason `directness`.
- Target `(-1, 0)` against the positive axes is empty, reason `outside_cone`. A lone axis against `(0.2, 1)` is empty, reason `fit`.
- Identical columns `(1, 0)` keep source `a`. A coefficient 0.2 is refit away when the minimum weight is 0.5. Columns `(1, 0)` and `(-0.2, 1)` against target `(0.45, 0.3)` refit below that same minimum and finish empty, reason `outside_cone`. A negative tau, a fit floor above 1, a ceiling below 1, a zero target, and a duplicate source id are refused or abstained.

The registered tool does not link this solver.

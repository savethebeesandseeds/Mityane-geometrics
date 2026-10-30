# Exact-search reference check, 30 October 2026

**Scope:** engineering arithmetic for OR01-RET 0.1.0. No corpus, encoder, Gemini call, or relevance claim.

**Command:** `make test-retrieval` in the managed container `mityane-geometrics`, compiler `g++ -std=c++20 -Wall -Wextra -Wpedantic`.

**Result:** exit 0. The compiler printed no warnings. The test printed `exact-search reference: 7 checks passed`.

The seven checks cover cosine tie order, L2 order and pairwise agreement, L2 normalization, reported self-match and named exclusions, an empty eligible set, refusal of a zero cosine query, duplicate source IDs, non-finite values, a snapshot mismatch, `k = 0`, a skipped zero-norm member, and hubness on the points 0, 1, and 2.

This run does not admit either retrieval profile.

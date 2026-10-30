# OR01-RET nearest-source confirmation, 30 October 2026

**Scope:** engineering arithmetic for OR01-RET 0.1.0. The query is one point in an embedding space. Each corpus member is another point in that same space. Exact search returns the source identities of the nearest points. No corpus, encoder, Gemini call, or relevance claim.

**Commands:** in the managed container `mityane-geometrics`, `make test-retrieval`, then the toy-a and toy-b engineering examples through `code/operations/tools.sh`, then `make test-retrieval-tool`. Compiler `g++ -std=c++20 -Wall -Wextra -Wpedantic`.

**Result:** exit 0. The library printed `exact-search reference: 7 checks passed`. The envelope printed 9 passed and 0 failed. Both example runs completed with one requested query and one success. The claim was `exploratory_description`.

| Fixture | Geometry | Rank 1 | Rank 2 |
| --- | --- | --- | --- |
| toy-a | L2, normalization none, query `[0, 0]` | `near` `[1, 0]`, distance 1, similarity 0.5 | `far` `[4, 0]`, distance 4, similarity 0.2 |
| toy-b | cosine, L2 normalization, query `[1, 0, 0, 0]` | `same` `[2, 0, 0, 0]`, distance 0, similarity 1 | `other` `[0, 3, 0, 0]`, distance 1, similarity 0 |

The saved bundles are `.temp/operations/retrieval/see-073927-toy-a-result` and `see-073927-toy-b-result`. Those directories are local run output and are not part of this revision.

The envelope checks also held: a self-only corpus completes empty, a zero cosine query abstains, a live call and an unbound relevance claim are rejected with no hits, a future as-of member is skipped, a future fitted artifact is rejected, an overlapping window is reported and not ranked, and a vector that disagrees with its saved artifact fails before ranking.

This run does not admit either retrieval profile.

# k geometric reconstructions v0.1.0

Date: 11 November 2026.
Entry: `reconstruct_k` in `combination.h` / `combination.cpp`.
Claim: exploratory_description. Weights are fit coefficients.

One prepared pass over the corpus scores every column against the target. Those scores rank the anchors. Retrieval `i` is a peel chain that starts at the `i`-th nearest source and keeps adding the next unused positive column until leftover / `||t||` meets `leftover_floor` (default 0.2) or `peel_cap` (default 16). K is how many of those chains to keep. Depth is not a requested count. Exact k-retrieval and `combine()` stay aside. `tau`, `gamma`, and `rho` apply per retrieval.

`make test-retrieval` printed `k geometric reconstructions: 2 checks passed` after the existing exact-search, constructor, and combination lines. The fixtures are in `tests/exact_search_test.cpp`: two axes yield a chain from the closest source and a chain from the second-closest; a high leftover floor stops the first chain after the start; a collinear target leaves the first chain at depth one; tau on a chain is a cosine with leftover so a shrinking leftover still peels. The registered tool does not link this entry.

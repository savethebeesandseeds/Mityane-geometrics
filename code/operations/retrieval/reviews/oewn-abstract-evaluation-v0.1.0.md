# Abstract retrieval score, OR01-RET-ABS-EVAL 0.1.0

**Status:** historical score of the deprecated geometric-query interpretation. Exact k-retrieval remains the registered method. Positive sparse combination is specified separately and was not scored here.

**Claim level:** `task_association` for this population and this SpaceRef. The claim rule was fixed in [PROTOCOL.md](../bindings/oewn-2025/PROTOCOL.md) before these numbers were calculated.

**Verdict:** `shared` is inconclusive. `difference` is negative. Neither system beats both baselines on full-list binary nDCG.

**Date:** 30 October 2026.

**Population:** the seven eligible queries in [BINDING.md](../bindings/oewn-2025/BINDING.md). Corpus of 29 abstract entries. SpaceRef `google/embeddinggemma-2` revision `f9972ac22e696a105c482202646979e33e4b5a1e`, text only, 768 dimensions, L2, cosine. Text snapshot sha256 `ce66b539ec47ab3380e076230c3fc55db4520b1c54f3f86d69e2c6223f16338c`. Abstract vectors sha256 `0096a2d4338d03ba94973d2679ad0604a0003958985208171acc13f2da535a81`. Concrete vectors sha256 `6c8e7bccf3b2e16cdb01abe1a80489d266b167954dfe035271ee30e8fe90e885`.

**Command:** `saved_vector_inspect` on `eval-queries.txt` sha256 `3924b4bdea3e7a3658c33b7db3e71935156a919890351cd0fbe6b9e62e2577da`, listing sha256 `2220254874e62721937ac0750a532a1cf5437057a557d21f0bdbbd022c13b6a3`. `score.mjs` checked those vector hashes, built the lexical order, and applied the protocol. The compiler line was `g++ -std=c++20 -O1 -Wall -Wextra -Wpedantic`. It printed no warnings. Every point, lexical, shared, and difference query was `ranked` with 29 eligible entries. None were empty, abstained, or failed.

## Full-list nDCG

The number is binary nDCG over all 29 entries. Precision and recall at 1, 3, and 5 are in the scorer listing and do not choose the verdict.

| Query | Point | Lexical | Shared | Difference |
| --- | ---: | ---: | ---: | ---: |
| dog+wolf | 0.644784158469 | 0.505128489511 | 0.871078544000 | 0.690398487919 |
| dog+cat | 0.455605149587 | 0.422789834407 | 0.455605149587 | 0.343977771453 |
| wolf+cat | 0.831554629584 | 0.422789834407 | 0.524981033201 | 0.266825262889 |
| bicycle+car | 1.00000000000 | 0.231378213160 | 1.00000000000 | 0.224243824218 |
| chisel+knife | 1.00000000000 | 0.726997733023 | 1.00000000000 | 0.606770130567 |
| bread+apple | 0.430676558073 | 0.221064729458 | 0.333333333333 | 0.210309917857 |
| dog+wolf+cat | 0.455605149587 | 0.422789834407 | 0.455605149587 |  |

Means, unrounded comparison values:

| System | Queries | Mean nDCG |
| --- | ---: | ---: |
| point | 7 | 0.688317949329 |
| lexical | 7 | 0.421848381196 |
| shared | 7 | 0.662943315673 |
| point, pairs only | 6 | 0.727103415952 |
| lexical, pairs only | 6 | 0.421691472328 |
| difference | 6 | 0.390420899151 |

`shared` is above `lexical` and below `point`, so the protocol calls it inconclusive. `difference` is below both pair baselines, so the protocol calls it negative. Dog and wolf are the query where the shared residual is clearly above the point baseline (0.871 against 0.645). Wolf and cat are the query where the point baseline is clearly above the shared residual (0.832 against 0.525), because the wolf vector already puts carnivore first. Dog and cat, and the three-animal set, put the two relevant entries at ranks 5 and 6 in both orders, so their nDCG matches.

## Cones

The axis is the shared residual. Aperture `0` is `empty` for all seven queries: no abstract entry has cosine exactly 1 with that axis. Purity is unset. Apertures `pi/3`, `pi/2`, and `pi` each keep all 29 entries, so purity is the label count divided by 29. Aperture `pi/6` is the selective row:

| Query | Inside | Relevant inside | Purity |
| --- | ---: | ---: | ---: |
| dog+wolf | 3 | 2 | 0.66666667 |
| dog+cat | 4 | 0 | 0 |
| wolf+cat | 4 | 1 | 0.25 |
| bicycle+car | 5 | 1 | 0.2 |
| chisel+knife | 5 | 3 | 0.6 |
| bread+apple | 6 | 0 | 0 |
| dog+wolf+cat | 5 | 1 | 0.2 |

Cone purity is a diagnostic. It is not the relevance rule and it is not the claim.

## Limit

Seven dependent queries are the whole eligible population. There is no unseen split. The same geometric ranks were inspected before this score, and the metric and the claim rule were fixed afterward in the protocol. This result does not transfer to Gemini Embedding 2, to another encoder, to Hx18, to a Nonverba evaluation, or to a contrast in Hx06.

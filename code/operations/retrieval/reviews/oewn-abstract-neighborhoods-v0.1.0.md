# Exploratory neighborhoods, OR01-RET-ABS 0.1.0

**Claim level:** `exploratory_description`. This listing describes neighborhoods under the frozen SpaceRef. It does not score the shared-principle rule, does not report nDCG or precision, and does not complete the abstract evaluation protocol.

**Date:** 30 October 2026.

**SpaceRef:** `google/embeddinggemma-2` revision `f9972ac22e696a105c482202646979e33e4b5a1e`, text only, 768 dimensions, L2 normalization, cosine. Prompt `SentenceSimilarity`: `task: sentence similarity | query: `. Surface `{member}: {definition}` from the first member and the first definition. The text snapshot sha256 is `ce66b539ec47ab3380e076230c3fc55db4520b1c54f3f86d69e2c6223f16338c`. Saved vectors are the local artifact under `.temp/operations/retrieval/or01-ret-abs-0.1.0-local-text/`. Concrete sha256 `6c8e7bccf3b2e16cdb01abe1a80489d266b167954dfe035271ee30e8fe90e885`. Abstract sha256 `0096a2d4338d03ba94973d2679ad0604a0003958985208171acc13f2da535a81`.

**Ranker:** `saved_vector_inspect` linked from `exact_search.o`, `geometry.o`, and `saved_vector_inspect.o`. Corpus file `abstract.vec`, point file `concrete.vec`, query file `queries.txt` sha256 `a5493917ae7a14d6322b550c231419046a4d7359cc5e5b65e0565aecdcc8d6a7`. Full listing `inspect.txt` sha256 `7f3ace713406053c96a9c42ad918f7538c977a70331e5a65c7344d788e7081fe`. Every query returned `ranked`, reason `none`, eligible count 29. `k` is 29, so a label rank is its place in the whole abstract corpus.

The seven eligible sets are the ones named in [BINDING.md](../bindings/oewn-2025/BINDING.md). A point query is the plain nearest neighbor of the first concrete source id in ascending order. A difference is prepared(target) minus prepared(origin), with the lower synset id as the origin. A shared query is the residual of the first point after removing the span of the later points. The ranker L2-normalizes that constructed vector before cosine.

## Label ranks and the nearest abstract entry

| Query | Constructor | Shared-principle ranks | Nearest abstract entry |
| --- | --- | --- | --- |
| dog, for dog+wolf | point | placental 5, carnivore 6, canine 2 | domestic animal, cosine 0.87754886 |
| dog to wolf | difference | placental 17, carnivore 1, canine 19 | carnivore, cosine 0.10950409 |
| dog and wolf | shared | placental 6, carnivore 3, canine 1 | canine, cosine 0.90155849 |
| dog, for dog+cat | point | placental 5, carnivore 6 | domestic animal, cosine 0.87754886 |
| dog to cat | difference | placental 20, carnivore 7 | solid, cosine 0.085539643 |
| dog and cat | shared | placental 6, carnivore 5 | canine, cosine 0.91110519 |
| wolf, for wolf+cat | point | placental 6, carnivore 1 | carnivore, cosine 0.86616084 |
| wolf to cat | difference | placental 19, carnivore 29 | solid, cosine 0.073716053 |
| wolf and cat | shared | placental 6, carnivore 3 | canine, cosine 0.90865771 |
| bicycle | point | wheeled vehicle 1 | wheeled vehicle, cosine 0.87940941 |
| bicycle to car | difference | wheeled vehicle 21 | carnivore, cosine 0.079092121 |
| bicycle and car | shared | wheeled vehicle 1 | wheeled vehicle, cosine 0.91821769 |
| chisel | point | cutter 2, cutting implement 3, edge tool 1 | edge tool, cosine 0.94757873 |
| chisel to knife | difference | cutter 4, cutting implement 2, edge tool 19 | foodstuff, cosine 0.077228469 |
| chisel and knife | shared | cutter 2, cutting implement 3, edge tool 1 | edge tool, cosine 0.97029687 |
| bread | point | solid food 4 | baked goods, cosine 0.90149579 |
| bread to apple | difference | solid food 26 | pome, cosine 0.10607564 |
| bread and apple | shared | solid food 7 | edible fruit, cosine 0.89493672 |
| dog, for dog+wolf+cat | point | placental 5, carnivore 6 | domestic animal, cosine 0.87754886 |
| dog, wolf, and cat | shared | placental 6, carnivore 5 | canine, cosine 0.92320058 |

## What the neighborhoods show

Point neighborhoods follow the WordNet wording. Dog is nearest domestic animal, then canine, feline, animal, and placental. Wolf is nearest carnivore, then canine. Bicycle is nearest wheeled vehicle, then vehicle. Chisel is nearest edge tool, then cutter and cutting implement. Bread is nearest baked goods, then foodstuff, starches, and solid food. Those cosines are about 0.83 to 0.95.

Shared residuals stay in that same tight band, about 0.89 to 0.97 against the nearest entry. Dog and wolf leave canine first, carnivore third, and placental sixth. Bicycle and car leave wheeled vehicle first. Chisel and knife leave edge tool, cutter, and cutting implement as the first three entries, which are exactly the three shared-principle labels. Dog and cat, and wolf and cat, leave canine and feline ahead of carnivore. Bread and apple leave edible fruit, pome, and baked goods ahead of solid food, which is rank 7. Dog, wolf, and cat leave canine first and carnivore fifth.

Difference directions are a weak alignment with this corpus. After the ranker normalizes the difference, the best cosine is about 0.07 to 0.11. Dog to wolf is nearest carnivore at 0.10950409. Dog to cat and wolf to cat are nearest the synset solid. Bicycle to car is nearest carnivore. Chisel to knife is nearest foodstuff. Bread to apple is nearest pome, then edible fruit. A rank of 1 in that list is the least-far abstract entry along a direction that is close to none of them.

The lexical token-overlap baseline named in the binding was not run. No cone was constructed. This inspection does not transfer to Gemini Embedding 2, does not close Hx18, and does not close Hx02, Hx06, or a Nonverba evaluation.

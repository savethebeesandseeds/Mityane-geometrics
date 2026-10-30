# OR01-RET-ABS-EVAL 0.1.0

Evaluation protocol for the English abstract population in [BINDING.md](BINDING.md). This file fixes the comparison before the scores are calculated. It does not revise OR01-RET 0.1.0 or the relevance rule `oewn-shared-hypernym-depth-1-3`.

The same seven rankings were listed exploratorily in [oewn-abstract-neighborhoods-v0.1.0.md](../../reviews/oewn-abstract-neighborhoods-v0.1.0.md) before this file assigned a number. That listing did not choose the metric, the cutoffs, or the claim rule below.

## 1. Population and partition

The evaluation unit is one eligible query: a set of two or three concrete seeds whose relevant abstract entries are the intersection of their depth-1-through-3 noun hypernym closures. All seven eligible queries are the assessment population. The other 158 sets stay in the snapshot and are not scored, because the rule gives them no relevant entry. There is no fit split and no unseen split. Queries share seeds, so the mean is not a standard error. The uncertainty display is the per-query table.

The corpus is the 29 abstract entries. The 10 concrete seeds stay outside it. The SpaceRef, vector hashes, and text snapshot hash are the ones in the binding. A scorer that sees a different abstract or concrete hash stops.

## 2. Systems

Each system produces one total order of the 29 abstract entries.

| System | Query subset | Order |
| --- | --- | --- |
| `point` | all 7 | Exact cosine search from the saved vector of the first concrete source id in ascending order. This is the plain nearest-neighbor baseline. |
| `lexical` | all 7 | Token overlap. Seed tokens are the ASCII letter tokens of the pinned lemmas. Abstract tokens are the ASCII letter tokens of every member and every definition. A token is a maximal run of `a` through `z` after lowercase. The score is the size of the set intersection. Higher score ranks first. Equal scores break by ascending source id. |
| `shared` | all 7 | Exact cosine search from the shared residual of the concrete vectors, in ascending source-id order. |
| `difference` | the 6 pairs | Exact cosine search from prepared(higher id) minus prepared(lower id). The triple has no difference. |

`difference` is compared with `point` and `lexical` on those same 6 pairs. `shared` is compared with `point` and `lexical` on all 7. A 6-query mean is not compared with a 7-query mean.

Cone queries use the shared residual as the axis, no radial bounds, and the apertures `0`, `pi/6`, `pi/3`, `pi/2`, and `pi`. The aperture is `std::numbers::pi_v<double>` divided as named. The whole grid is reported. No aperture is selected afterward. Cone membership and cone purity are not the relevance rule and do not enter the claim.

## 3. Metrics

Relevance is binary and is the frozen rule. Rank `i` starts at 1.

Full-list nDCG uses the order of all eligible corpus entries:

- gain is 1 on a relevant entry and 0 otherwise
- discounted gain at rank `i` is `1 / log2(i + 1)`
- nDCG divides that sum by the ideal sum over the relevant set

Precision@k and recall@k are reported at k = 1, 3, and 5. They do not choose the claim. The claim uses the unrounded mean of full-list nDCG.

## 4. Failure criteria and the claim rule

Empty, abstained, and failed stay separate. A constructor that does not return `constructed`, or a search that is not `ranked` over all 29 eligible entries, is recorded as that outcome. It is not stored as nDCG 0.

| Condition on a system | Verdict |
| --- | --- |
| Any query in its subset is empty, abstained, failed, or not constructed | `blocked`. Partial scores stay in the table. This is not a negative result. |
| Every query is ranked, and the mean nDCG is strictly greater than both baseline means on the same queries | `positive` |
| Every query is ranked, and the mean nDCG is strictly less than both baseline means on the same queries | `negative` |
| Every query is ranked, and neither of those two comparisons holds | `inconclusive` |

The comparison is binary64 strict inequality of the means. A tie with either baseline is `inconclusive`. A `positive` verdict is a `task_association` claim for this population, this SpaceRef, and this system only. A `negative` or `inconclusive` verdict is the result of that same comparison. Neither transfers to Gemini Embedding 2, to another encoder, to Hx18, to a Nonverba evaluation, or to a contrast in Hx06.

## 5. Reproduction

`score.mjs` writes the query file, checks the vector hashes, and applies this rule to an inspector listing. The inspector is `saved_vector_inspect`. It ranks with the same exact search and constructors as the library. The registered tool is not changed.

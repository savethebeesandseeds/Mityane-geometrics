# OR-01 Requirements-and-Evidence Record

**Record:** `OR01-RET-RER`, revision `0.1.0`, 30 October 2026.  
**Operation:** [OR01-RET / ranked-source-retrieval, revision 0.1.0](CONTRACT.md).  
**Purpose:** reusable requirements and the current evidence boundary. Each admitted profile supplies a versioned instance record that references this RER and binds the fields below. This record is not an executable study protocol.

## 1. Hypothesis and claim

No hypothesis is bound. A future `task_association` instance states a falsifiable relevance claim for one frozen query set, one corpus snapshot, one SpaceRef, and one baseline. A future `exploratory_description` instance describes neighborhoods under a declared geometry and does not report that description as relevance quality.

This revision refuses `controlled_model_response` and `identified_intervention_effect`. Neighborhoods passed to Hx06 stay retrieval output.

## 2. Data and provenance

The instance record must bind the corpus snapshot, query set, license, access, and retention; source IDs and parent lineage; partitions; duplicate and source-window overlap handling; the SpaceRef, including encoder or provider identity, revision policy, dimensions, normalization, metric, and task or role pair; preprocessing and fitted artifacts when used; code and configuration identity; environment; and temporal claim mode.

The text profile additionally binds the Hx18 request-manifest digest, any returned model revision or its absence, the response-artifact digest, and whether the vectors are replayed artifacts or a fresh call. The local profile binds the saved-vector artifact and the encoder checkpoint or synthetic generator those vectors came from. Neither profile is bound in this revision.

**Current bindings:** none. The [exact-search reference](IMPLEMENTATION.md) and the registered engineering tool rank caller-supplied arrays. They select no encoder checkpoint, no Gemini response, and no dataset. A tool completion is not a relevance result.

## 3. Controls and baselines

Before a `task_association` execution, bind:

- An independent relevance rule recorded before ranking.
- A baseline on the same eligible queries, corpus, and information horizon. Text uses a lexical baseline. Local uses a lexical baseline, raw descriptors, or the synthetic motif rule, as declared for that instance.
- The exact ranking as the reference for any later approximate index. Approximation loss is reported separately from relevance.

Engineering checks of metric arithmetic, tie order, exclusion reporting, and refusal establish mechanics only.

## 4. Evaluation protocol

The instance freezes partitions, query eligibility, the as-of or non-temporal mode, `k`, metric, normalization, exclusion rules, baselines, metrics, and failure criteria before held-out assessment. Diagnostic exposure and repeated looks are indexed. A revised protocol does not reuse an exposed confirmation set silently.

Every initiated attempt is retained, including rejected admission, failure, interruption, abstention, and an empty eligible set. Execution disposition is separate from the scientific verdict. A valid negative or inconclusive relevance result can complete its scoped evaluation.

**Current protocol:** none. Reusable documents grant no execution admission.

## 5. Measures of success

Contract conformance for the engineering reference is exact ordering under the declared metric, deterministic source-ID ties, reported skips, and distinct empty, abstained, and failed outcomes. Relevance success is recall, precision, nDCG, or MRR against the independent rule, as frozen by the instance, compared with the declared baseline. Approximate-neighbor success is recall of the exact reference and the approximation-loss distribution. Documentation review supplies no numerical relevance result.

## 6. Robustness

The instance records the evaluation unit, including dependent windows or repeated sources, and the uncertainty method or a justified limitation. Subgroup and temporal checks are required when the corpus has those groups. A result on one profile does not transfer to another encoder, modality, dimension, or time range.

## 7. Results and limitations

No profile run has been admitted. No relevance number is claimed. The engineering reference described in [IMPLEMENTATION.md](IMPLEMENTATION.md) checks nearest-source arithmetic only. On 30 October 2026, `make test-retrieval` passed seven checks and `make test-retrieval-tool` passed nine. A visible engineering run ranked source `near` (`[1, 0]`, L2 distance 1) ahead of `far` (`[4, 0]`, distance 4), and source `same` (cosine 1) ahead of `other` (cosine 0). The same library later constructed a difference, a shared residual, and a cone on synthetic vectors; that check is [geometric-operators-v0.1.0.md](reviews/geometric-operators-v0.1.0.md). It binds no abstract corpus. The records are [exact-search-v0.1.0.md](reviews/exact-search-v0.1.0.md), [envelope-v0.1.0.md](reviews/envelope-v0.1.0.md), and [nearest-source-v0.1.0.md](reviews/nearest-source-v0.1.0.md).

OR01-RET 0.1.0 does not establish retrieval quality, encoder quality, or Gemini service behavior.

## 8. Reproduction pointer

| Item | Identity |
| --- | --- |
| Contract | `code/operations/retrieval/CONTRACT.md`, OR01-RET 0.1.0 |
| This record | `code/operations/retrieval/EVIDENCE.md`, OR01-RET-RER 0.1.0 |
| Engineering reference | `code/operations/retrieval/include/mityane/operations/retrieval/exact_search.h` |
| Library check | `make test-retrieval` in the managed container |
| Envelope check | `make test-retrieval-tool` in the managed container |
| Visible order | `code/operations/retrieval/reviews/nearest-source-v0.1.0.md` |

No query set, corpus hash, seed, or provider response is part of this revision.

## 9. Decision

`exploratory` interface only. Both profiles remain unbound. No study is promoted, and none is admitted.

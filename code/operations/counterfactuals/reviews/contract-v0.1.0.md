# Hx06 contract-authoring review

**Date:** 19 October 2026.  
**Scope:** `or-06-comparisons-define-contract`, reusable interface authoring.  
**Reviewed artifacts:** [README](../README.md), [OR06-PCR 0.1.0](../CONTRACT.md)
and [OR06-PCR-RER 0.1.0](../EVIDENCE.md).  
**Review basis:** [Hero acceptance criteria](../../../../web/dashboard/heroes/06-comparisons.json)
and [ORR-DESIGN-001 v0.1.0](../../../../doc/operations_over_representations_design_spec_v0.1.0.md),
sections 4, OR-06 and 7.

## Acceptance review

| Requirement | Observed authored outcome |
| --- | --- |
| Versioned Operation Contract and linked RER | Both records have IDs, revision 0.1.0 and reciprocal links; the README identifies component ownership |
| Explicit claim level and primary question | Paired controlled model response is selected; observational and identified intervention requests are unsupported in this revision |
| Input/output roles and parameters | Source-parent pairs, representation kinds/SpaceRefs, pipeline identities, alternatives, held-fixed attributes, contrast sign/units, precision and support policies are defined |
| Representation structure and dependencies | Role compatibility, provider/fit/geometry identity, mapped inputs, stochastic controls and alternative validity are explicit |
| Typed failure/abstention conditions | Success, abstention and failure have distinct envelopes/reasons; rejection and unresolved execution preserve disposition and denominator accounting |
| Required shared rules | Contract section 5 maps record validation, temporal integrity, data protocol and lifecycle admission; evidence discovery is also inherited |
| Claim and identification limits | Contract sections 1 and 5 distinguish model response, observed associations and intervention effects; future intervention profiles require estimand, timing, assumptions, support and prohibited matching inputs |
| Complete RER fields | Hypothesis, data/provenance, controls, protocol, success criteria, robustness, results/limitations, reproduction and decision are all present; actual study bindings are explicitly unbound |
| Separation from experimental execution | No dataset, trained instance, experiment or scientific finding is selected or claimed; implementation and evaluation stay proposed |

Two independent read-only draft reviews found no blocking defects. The suggested
clarification of transitive output availability was incorporated: retain the
maximum required availability across both branches and fitted dependencies,
or a documented tighter proof.

## Engineering verification

Checks ran through `container.ps1` in the existing managed container
`mityane-geometrics` (ID
`381c6b78f4fdb301cd132050ffd8095587c0794829a7a73a547fdbc0484630e1`).
They verify documentation integration and dashboard consistency, not numerical
operation behavior or scientific quality.

- `bash code/scripts/node.sh code/scripts/check-layout.mjs --links-only`:
  the repository-wide check reported six unrelated broken links, all targeting
  the absent `code/embeddings/evaluation/records/CPB-v15.objective-001.json`.
  The reported sources belong to the concurrent Hx17 work and were preserved.
  No broken link was reported in the Hx06 documents or operations README.
- A scoped invocation of the existing `checkMarkdownLinks` export passed all
  24 local link targets across the four new component documents and the edited
  operations README; the six unrelated failures were reported separately.
- `web/dashboard/scripts/refresh.ps1`: passed all 34 focused tests, validated
  18 Heroes / 51 Goals / 69 Milestones (17 done, 1 in progress), resolved all
  dashboard evidence file targets and regenerated the standalone HTML.
- Bounded `git diff --check` for the edited operations README and Hx06 JSON:
  passed. The contract milestone is done; implementation and evaluation remain
  proposed.

## Decision and limits

The first milestone's contract-definition acceptance is satisfied by these
authored artifacts and this review. The chosen profile makes no assertion about
identified intervention effects or individual counterfactual outcomes.
No counterfactual implementation, numerical test, study execution or empirical
utility has been established. The next implementation milestone retains its own
conformance and evidence requirements.

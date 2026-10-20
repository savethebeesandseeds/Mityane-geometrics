# Counterfactuals and comparisons

This component belongs to **Hx06 / OR-06 — Matched comparisons, sensitivity,
and counterfactual analysis**. It owns operation-specific contracts,
implementations, configurations, tests and evidence under this directory.
Encoder fitting and provider adapters stay with their encoder; reusable data
handling and cross-component utilities retain their existing owners.

The first defined operation is **paired controlled model response**, revision
`0.1.0`: compare a frozen model's numerical responses to a baseline and a
declared alternative. Its result describes the model under that change.

| Record | Purpose |
| --- | --- |
| [Operation Contract](CONTRACT.md) | Input/output roles, comparison semantics, validity checks and refusal conditions |
| [Requirements-and-Evidence Record](EVIDENCE.md) | Required study bindings, controls, evaluation and current evidence limits |
| [Contract review](reviews/contract-v0.1.0.md) | Acceptance review for the contract-authoring milestone |
| [C++ reference guide](IMPLEMENTATION.md) | Public interface, supported slice, caller responsibilities and build command |
| [Usable tool guide](TOOL.md) | Common CLI, reusable Node API, provider-neutral request bundles and saved results |
| [Usable tool guide](TOOL.md) | Common CLI, reusable Node API, provider-neutral request bundles and saved results |
| [Reference verification](reviews/reference-v0.1.0.md) | Source identities and 47 executed engineering conformance cases |
| [Synthetic evaluation results](evaluation/synthetic_response_v1/RESULTS.md) | Frozen three-seed end-to-end study, independent audit and scoped findings |

The reusable contract, deterministic C++ reference and initial synthetic
evaluation are complete for their stated scopes. The benchmark uses fixed
project-authored representations and TRAIN-fitted readouts. Other datasets,
providers and claim profiles need their own bindings. Current priorities and
work states belong to the [Hero record](../../../web/dashboard/heroes/06-comparisons.json).

OR-06 also includes matched observational contrasts and identified intervention
effects. They require separately versioned operation profiles; the initial
profile accepts only `controlled_model_response`. Editing latent candidates
can be supplied by OR-07, and decoding by OR-09, with their own validity evidence.

Development and checks use the root managed-container launcher and task entries
described in the [environment guide](../../../doc/ENVIRONMENT.md). This
component introduces no separate environment. Its operation-owned Make rules
are included by the authoritative root Makefile.

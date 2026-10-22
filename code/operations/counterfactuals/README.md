# Hx06: counterfactuals and comparisons

This component owns Hx06's work in **OR-06 — Matched comparisons, sensitivity,
and counterfactual analysis**. The callable operation is **OR06-PCR 0.1.0**,
paired controlled model response: compare recorded numerical responses to a
baseline and a declared alternative. Its contrast describes the frozen model
under that change. The sole admitted claim is `controlled_model_response`.

## Use the tools

Start with the usage guide. Windows PowerShell launches the existing managed
Linux container; the reusable Node API and C++ comparison run inside it.

| Guide | Use it for |
| --- | --- |
| [Tool usage](TOOL.md) | CLI and Node quickstart, prepare/check/run/read, sweep/diff/study verification, schemas and ownership |
| [C++ reference](IMPLEMENTATION.md) | Numerical interface, supported slice, caller responsibilities and focused build/check entry |
| [Common operation tools](../TOOLS.md) | Shared envelopes, publication, exit codes and adapter conventions |

Upstream owners supply actual inputs, representations, responses and evidence.
These tools perform no encoding or fitting. Operation code, configurations,
tests and protocols stay here; model/provider work, reusable data handling and
shared mechanisms retain their existing component owners. Development uses the
root launcher and task entries in the [environment guide](../../../doc/ENVIRONMENT.md).


## Contracts and scientific evidence

| Record | Scope |
| --- | --- |
| [OR06-PCR contract](CONTRACT.md) and [RER](EVIDENCE.md) | Recorded paired responses, required bindings, controls and typed refusals |
| [Original synthetic results](evaluation/synthetic_response_v1/RESULTS.md) | Frozen three-seed response-fidelity and refusal study in its declared toy population |
| [Representation contribution results](evaluation/representation_contribution_v1/RESULTS.md) | Capacity-matched feature-family comparison with response-axis and population breakdowns |
| [Readout binding decision](evaluation/representation_contribution_v1/READOUT_DECISION.md) | TRAIN-only OLS retained as a scoped reference; validation required for each representation, target and population |
| [OR06-MOC contract](profiles/observational_contrast/v0.1.0/CONTRACT.md) and [RER](profiles/observational_contrast/v0.1.0/EVIDENCE.md) | Matched observational contrast, confounding diagnostics and typed refusals |
| [OR06-IIE contract](profiles/intervention_effect/v0.1.0/CONTRACT.md) and [RER](profiles/intervention_effect/v0.1.0/EVIDENCE.md) | Identified intervention effect, explicit identification assumptions, timing/support and causal refusals |
| [Non-synthetic instance bindings](profiles/instances/nsw_reference_v1/README.md) | Separate NSW archival bindings and protocols for MOC and IIE; no scientific execution |

MOC and IIE have authored profiles and actual-source bindings, but no callable
estimators or scientific result evidence. Authoring a profile or binding an
instance does not admit execution. Synthetic results support their recorded
settings and supply no universal encoder or causal claim. Current priorities,
acceptance criteria and work states belong to the
[Hero record](../../../web/dashboard/heroes/06-comparisons.json).

## Design background: two camps

Non-normative orientation for Hx06, checked 20 October 2026. The
[project charter](../../../doc/operations_over_representations_design_spec_v0.1.0.md),
[OR06-PCR contract](CONTRACT.md) and [RER](EVIDENCE.md) govern the operation.
This note adds no claim profile, implementation, goal or execution permission.

### Where is the alternative constructed?

The two camps are useful shorthand for construction choices. They are not an
exhaustive division of the literature or a ranking of causal validity. A method
can combine both, and a frozen versus fitted representation is a separate choice.

| Camp | Construction or comparison | Evidence needed for interpretation |
| --- | --- | --- |
| **Camp 1: latent or representation-space edits** | Change coordinates, directions or structural-model residuals; optionally decode the candidate. | Bind the coordinate system, constructor, validity/support tests and any decoder. A semantic or causal interpretation needs evidence beyond the arithmetic. |
| **Camp 2: source-defined alternatives or observational comparisons** | Change source attributes or treatment conditions, or compare distinct observed records, using representations in the analysis. | Distinguish a constructed alternative from a matched observation. Bind the source meaning, matching/design assumptions, timing and support appropriate to the claim. |

Neither construction location supplies causal identification. A source-level edit
alone does not establish an intervention effect; it may support a model-behavior
question. A latent candidate can support a limited coordinate-response comparison.
The charter's three OR-06 claim levels remain separate:

| Claim | Permitted interpretation | Current tool status |
| --- | --- | --- |
| Matched observational contrast | Association between observed cases under declared matching conditions. | [OR06-MOC profile](profiles/observational_contrast/v0.1.0/CONTRACT.md) authored and bound; no callable estimator. |
| Controlled model response | Response of the specified frozen model to declared alternatives. | **OR06-PCR 0.1.0**; sole admitted claim `controlled_model_response`. |
| Identified intervention effect | Effect on the studied system within an explicit identification argument and assumptions. | [OR06-IIE profile](profiles/intervention_effect/v0.1.0/CONTRACT.md) authored and bound; no callable estimator. |

### Where OR06-PCR sits

The comparator consumes **recorded** baseline and alternative responses. It
validates their relationships and computes `alternative - baseline`; it invokes
no encoder, decoder, response model or fitting routine. A request binds outputs
and evidence from the declared frozen upstream pipeline.

The contract permits `edited_input`, `simulated_input` and `latent_candidate`
alternatives. The current reference requires a latent candidate to retain the
explicit `latent_coordinate_response` interpretation. It cannot silently become
a source intervention. Independent evidence is needed to ground source-level
meaning; plausible decoded output alone does not provide it. See the contract's
[input roles](CONTRACT.md#2-logical-input-roles) and
[dependency requirements](CONTRACT.md#3-pairing-representation-and-dependency-requirements),
and the [reference checks](src/paired_response.cpp).

Ownership is composable: OR-07 constructs/transforms candidate representations;
OR-09 owns decoding/generation where used; the upstream model/task owner records
responses; OR-06 compares admitted records. Each step retains its contract,
lineage, restrictions and evidence. Comparing responses to an OR-07 candidate
does not transfer construction ownership to OR-06.

Support and validity attestations remain supplied evidence. Byte binding and
typed refusals check declared relationships and preserve unsuccessful outcomes;
they do not prove semantic truth, provider authenticity or identification.
The [tool guide](TOOL.md) describes the executable boundary. The same restrictions
apply to sweeps and result differences.

### Literature anchors and their limits

These primary sources illustrate different questions and mechanisms. Their
classification below is an orientation for this codebase, not the authors'
shared taxonomy or evidence of a gap that OR06-PCR uniquely fills.

- **Crupi et al., CEILS:** [Counterfactual Explanations as Interventions in Latent Space](https://arxiv.org/abs/2106.07754v2)
  (2021 preprint; [journal record](https://link.springer.com/article/10.1007/s10618-022-00889-2)
  online 2022, volume 38 in 2024). Constructs changes in exogenous/residual
  coordinates and propagates them through estimated structural equations.
  Its causal constraints are relative to the supplied graph and structural model,
  not guaranteed by arbitrary embedding geometry.
- **Bender et al., DiDAE:** [Towards Fast and Disentangled Counterfactuals for Visual Foundation Models](https://arxiv.org/abs/2610.00895v1)
  (2026). Combines a frozen encoder, dictionary-direction edits and diffusion
  decoding. It includes empirical component bounds and decoded/re-encoded
  verification; its trust region is not a manifold guarantee. Decoder/dictionary
  fitting is separate from gradient-free counterfactual generation. The earlier
  [Bender and Morik paper](https://arxiv.org/abs/2601.21851v2) is a preliminary
  version of this research, not an independent example.
- **Johansson, Shalit and Sontag:** [Learning Representations for Counterfactual Inference](https://proceedings.mlr.press/v48/johansson16.html)
  (ICML 2016). Learns representations that balance treated and control
  populations for counterfactual inference. This is representation learning for
  outcome estimation, distinct from comparing recorded frozen-model responses.
- **Muandet et al.:** [Counterfactual Mean Embeddings](https://jmlr.org/papers/v22/20-185.html)
  (JMLR 22(162), 2021). Represents counterfactual outcome distributions in an RKHS
  and estimates distributional treatment effects under unconfoundedness and the
  paper's support assumptions. Here an embedding represents a distribution,
  rather than an edited hidden vector from a pretrained encoder.
- **Zellinger, Stephan and Roth:** [Counterfactual Reasoning with Knowledge Graph Embeddings](https://aclanthology.org/2024.eacl-long.168/)
  (EACL 2024). COULDD adapts graph embeddings under hypothetical premises. Its
  reported difficulty with changes outside learned inference patterns concerns
  reasoning generalization; it is not a validation of OR06-PCR's refusal policy
  or an identified real-system effect.

These examples do not justify claims that latent methods lack support controls,
that source-defined methods require retraining, or that a frozen encoder plus
refusals is unique. In particular, DiDAE already freezes its encoder and applies
empirical edit constraints.

### Vector arithmetic as a related probe

Analogy arithmetic such as `king - man + woman` is a relational probe; see
[Mikolov, Yih and Zweig (NAACL 2013)](https://aclanthology.org/N13-1090/).
Its accuracy depends on the representation, relation, metric and evaluation
protocol; there is no universal vector law. Such a probe ordinarily belongs to
OR-07. A resulting candidate may feed an independently admitted PCR request
with recorded numerical responses and the required evidence. The arithmetic
alone establishes neither a source-level controlled response nor a causal effect.

## Delivery history

These records describe their dated scopes. Contracts, frozen protocols, evidence
and earlier toolset snapshots retain their recorded identities.

| Record | Coverage |
| --- | --- |
| [Contract review](reviews/contract-v0.1.0.md) | Contract-authoring acceptance |
| [Reference verification](reviews/reference-v0.1.0.md) | Source identities and 47 C++ conformance cases |
| [Toolset surface review](reviews/toolset-surface-v0.1.0.md) | Original exposure review and then-selected delivery scope |
| [Toolset delivery verification](reviews/toolset-coverage-v1.md) | Preparation, workflow, structured reuse and schemas; 28 tool checks |
| [Ergonomic tool verification](reviews/tool-ergonomics-v1.md) | Sweep/diff/study verification, CLI/Node parity and 49 tool tests |
| [Profile review](reviews/profiles-v0.1.0.md) | Separately versioned profiles and non-synthetic bindings |
| [Representation contribution review](reviews/representation-contribution-v1.md) | Follow-up evaluation, reporting and readout decision |
| Toolset snapshots [0.1.0](archive/TOOLSET-v0.1.0.md), [0.2.0](archive/TOOLSET-v0.2.0.md), [0.3.0](archive/TOOLSET-v0.3.0.md) | Earlier plans and delivered surfaces; use the [current tool guide](TOOL.md#public-calling-surface) for today's interface |

The [archive relocation record](archive/relocations.json) preserves original paths and SHA256 identities. Snapshot contents retain their original relative-link base; current navigation uses the guides above.

# Hx06 toolset surface

**Map:** `HX06-TOOLSET`, revision `0.1.0`, 20 October 2026.  
**Family:** OR-06. **Currently registered operation:** OR06-PCR `0.1.0`.  
**Contract and evidence:** [Operation Contract](CONTRACT.md),
[Requirements-and-Evidence Record](EVIDENCE.md).

This versioned map identifies the public calling surface, implementation gaps
and ownership boundaries for Hx06. It completes the surface-definition step;
planned entries below are not callable tools. Milestone states and priorities
remain in the [Hero record](../../../web/dashboard/heroes/06-conterfactuals.json).
The [tool guide](TOOL.md) remains the executable usage reference.

## Claim profiles

| Profile | Current support | Contract, evidence and limits |
| --- | --- | --- |
| Recorded paired controlled model response | Implemented and registered as OR06-PCR `0.1.0` | [Contract](CONTRACT.md), [RER](EVIDENCE.md), [C++ verification](reviews/reference-v0.1.0.md), [tool verification](reviews/tool-v1.md). Computes retained alternative-minus-baseline responses under declared conditions; upstream validity remains supplied evidence. |
| Matched observational contrast | No profile, estimator or registered tool yet | Separate versioned contract/RER, matching diagnostics, confounding checks and typed refusals remain under **Expand operation profiles**. No request schema is defined. OR06-PCR rejects this claim. |
| Identified intervention effect | No profile, estimator or registered tool yet | Separate versioned contract/RER, explicit identification, untestable assumptions, timing, support and prohibited matching variables remain under **Expand operation profiles**. No request schema is defined. Proximity supplies no identification argument. OR06-PCR rejects this claim. |

The [charter](../../../doc/operations_over_representations_design_spec_v0.1.0.md)
sections 4, OR-06, 6 and 7 govern these distinctions. A new profile also needs
its own instance bindings and evaluation evidence before scientific claims.
The completed synthetic study does not supply either missing profile.

## Existing public interfaces

The common CLI is `code/operations/tools.ps1` from Windows and
`bash code/operations/tools.sh` inside the managed Linux container. The reusable
Node entry is `operations` from [tools.mjs](../tools.mjs). All computation runs
inside that container; PowerShell forwards arguments only.

| Building block | CLI | Node API | Inputs, outputs and implementation |
| --- | --- | --- | --- |
| Discovery | `list` | `operations.list()` | Lists registered operations with IDs, revisions, family, title and claim levels. [Registry](../tools.mjs) and [shared runtime](../../shared/representation/tools/v1/runtime.mjs). |
| Requirements and schemas | `describe OR06-PCR` | `operations.describe('OR06-PCR')` | Returns the exact descriptor, specialized request JSON Schema, response-artifact schema, limits and claim boundary. [Operation descriptor](tool/adapter.mjs) and [input schema](tool/request.mjs). |
| Engineering example | `example OR06-PCR --provider toy-a\|toy-b --output NEW_DIRECTORY` | No `operations.example` method | [Example writer](tool/example.mjs) creates an explicit artificial request and byte inventory. It does not infer a real provider configuration. |
| Byte checks, compatibility and refusal diagnostics | `check --request FILE` | `await operations.check(FILE)` | Loads the full request and artifact bytes, then calculates the pure recorded comparison in memory. Returns a check record with native outcomes, counts and verified artifacts; publishes no result bundle. [Runtime](../../shared/representation/tools/v1/runtime.mjs), [adapter](tool/adapter.mjs), [C++ implementation](src/paired_response.cpp). |
| Paired contrast or recorded alternative roster | `run --request FILE --output NEW_DIRECTORY` | `await operations.run(FILE, NEW_DIRECTORY)` | Same validated inputs; returns saved bundle paths and execution status. Each pair retains its own alternative, controls, support checks, state and optional contrast. Multiple alternatives can share a baseline without becoming independent samples. Same implementations as `check`. |
| Verified readable result | `inspect --result FILE` | `operations.inspect(FILE)` | Verifies the saved file manifest and returns the human-readable summary string. It does not return a structured result object or execute the operation. [Shared runtime](../../shared/representation/tools/v1/runtime.mjs). |
| In-memory typed comparison | No separate CLI | C++ API, independent of Node | `mityane::operations::counterfactuals::v1::compare(const Request&, optional<size_t> stop_after = nullopt)` returns `RunResult`. [Public types](include/mityane/operations/counterfactuals/paired_response.h), [reference guide](IMPLEMENTATION.md). The C++ layer checks supplied records; the file adapter adds actual byte and response-payload binding. |

The C++ `stop_after` argument can leave part of an admitted roster explicitly
unresolved. This optional argument has no JSON/CLI counterpart; exposing bounded
execution or resumption is deferred, outside the next delivery increment. The
engineering example remains a CLI convenience rather than a general library
constructor; explicit request preparation below addresses application use.

`check.valid` is true for a completed comparison, including explicit abstentions.
Inspect every pair's state and diagnostics before using a contrast. A rejected
admission retains the requested count without executed rows; admitted runs retain
success, abstention, failure and unresolved counts against the complete roster.
CLI exit meanings are 0 completed, 2 invalid usage/bytes or execution error,
3 rejected admission and 4 partial results. Execution completion supplies no
scientific verdict.

## Schema and record ownership

| Record | Current definition | Exposure or remaining gap |
| --- | --- | --- |
| Request envelope | `mityane.operation.request/v1`; exported `requestEnvelopeSchema` in the [shared runtime](../../shared/representation/tools/v1/runtime.mjs) | Available through `describe` specialized to the selected operation/revision. Requires `operation`, `input` and the exact artifact inventory. |
| OR06-PCR input | `inputSchema`, `$id` `urn:mityane:operations:paired-response:input:1`, in the [request codec](tool/request.mjs); maps the public C++ `Request` | Available through `describe`. All fields are explicit, including required nullable fields. Codec helpers are adapter internals, not a separate stable application API. |
| Recorded response artifact | `response_artifact_schema` in the [descriptor](tool/adapter.mjs) | Published object with exactly `axes` and `values`; the adapter verifies equality with the branch's inline response values and axes. |
| Check result | `mityane.operation.check/v1`, emitted by the shared runtime | Procedurally produced; a complete published JSON Schema is a delivery gap. |
| Saved result and native outcomes | `mityane.operation.result/v1`, emitted by the runtime; native outcome payload derived from OR06-PCR `RunResult` | Native JSON contains `admitted`, `admission_diagnostics`, `pairs` and `summary`; the bound input remains separately in retained `request.json`. Procedural validation and typed C++ semantics exist. Full published result/native-payload schemas are a delivery gap. |
| Saved file inventory | `mityane.operation.files/v1`, emitted and checked by the runtime | Procedural validation exists; a published JSON Schema is a delivery gap. The manifest detects byte inconsistency, not source authenticity. |

The reusable file boundary owns envelopes, file inventories, publication and
inspection. OR06-PCR owns its input fields, response artifacts, native payload,
pair semantics and typed refusals. Publishing schemas must preserve this split;
the common runtime must not acquire counterfactual-specific fields.

## Coverage and next delivery scope

The following engineering exposure gaps define the next toolset-coverage
increment. Names for new commands/APIs and any new schema identities are proposals
until implemented and documented; existing commands retain their meanings.

| Selected building block | What exists | Gap to deliver | Contract, schema and claim boundary |
| --- | --- | --- | --- |
| Explicit request preparation | Full strict request schema, editable examples and private encoding/reference helpers | A documented CLI and reusable API to assemble a request bundle from caller-supplied typed input and explicit artifact files. Verify bindings and refuse missing/conflicting metadata; never guess support, permissions, axes, provider, fit populations or claim level. | Reuses OR06-PCR input and common request envelope. Any preparation-spec schema must be published. Preparation does not admit a scientific run or generate alternatives. |
| Recorded sensitivity workflow | One `Request.pairs` roster already supports multiple named alternatives and retained per-pair deltas | A provider-neutral CLI/library workflow example using the preparation/check/run/inspect boundaries, explicit change magnitudes/directions, a shared baseline and refused pairs. Retain the full roster and source dependencies. | Reuses OR06-PCR `0.1.0`. These are paired response contrasts. Derivatives, normalized sensitivities, stochastic averaging and uncertainty estimation require separately specified semantics; they are not supplied by this increment. |
| Structured verified result reuse | Raw `result.json` plus manifest, readable `inspect`, and typed C++ results | A public CLI/library path returning the verified structured result and retained request/artifact references, without requiring consumers to bypass bundle verification or parse summary text. | Common result envelope plus operation-owned native outcome schema. Return all states and claim limits; do not turn refusal into zero or silently select successes. |
| Complete public output schemas | Request and response-artifact schemas are discoverable | Publish complete check/result/file-manifest schemas and the OR06-PCR native outcome schema. Discover them through the common describe convention with exact versions. | Keep common schemas with the shared boundary and paired-response payload schemas under counterfactuals. Existing execution and claim semantics are unchanged. |

Delivery acceptance is an end-to-end engineering workflow for distinct compatible
provider identities using the same interfaces, with manually checkable contrasts,
support refusal, incompatible-space failure, full denominators, saved-byte
verification and output collision protection. Changes to public calling behavior
need meaningful boundary checks. This map alone does not complete delivery.

## Caller workflow and composition

1. **Prepare upstream evidence.** The source owner supplies lawful source/input
   artifacts, source/parent relationships, representation records and exact
   SpaceRefs, frozen response-pipeline dependencies, recorded responses,
   constructor/validity/support evidence, temporal mode and permissions. Select
   the actual operation contract/RER. A research invocation additionally binds
   its actual instance and prospective protocol.
2. **Discover and bind.** Use `list` and `describe`; build the full request and
   exact artifact inventory according to the current [tool guide](TOOL.md).
   Today callers author that request explicitly or edit an engineering example;
   the planned preparation helper will package supplied records, not invent them.
3. **Check.** Use `check` and inspect admission and every pair outcome. It hashes
   files and calculates recorded arithmetic. It neither executes the upstream
   model nor verifies the truth of caller-supplied semantic attestations.
4. **Run and retain.** Use a new output directory. Keep the exact request,
   copied artifacts, structured result, summary and manifest together. Preserve
   failed, abstained and unresolved rows. A prior check is not a reservation:
   `run` independently reads and verifies the inputs again.
5. **Inspect and compose.** Use `inspect` before reading a saved bundle; retain
   verified file identities when consuming its structured records. Currently
   `inspect` returns text, so a caller must explicitly join `result.json` to its
   manifest and retained request. A reusable structured loader is a stated gap.
   The in-memory C++ route keeps `RunResult.input` with all pair results.

For each downstream edge, declare what is passed: response axes/units and signed
contrasts, pair/source/parent IDs, claim level, SpaceRef/pipeline revisions,
population/horizon, artifact identities, and every refusal/state. Repeated
baselines retain their shared-source dependence; no automatic independent-sample
or population-effect interpretation follows. Preserve upstream access and timing
restrictions. The receiving operation must check its own contract and record
its transformation and output identity. Engineering composition checks do not
establish a scientific pipeline claim; that needs bound end-to-end evaluation.

## Upstream and deferred capabilities

| Capability | Owner or dependency | Current boundary |
| --- | --- | --- |
| Source reading, dataset validation and preprocessing | `code/data/` and the actual dataset owner | Supply retained source/input evidence; this map adds no data importer. |
| Encoder execution, embedding APIs, provider adapters and model fitting | `code/embeddings/`; task readouts keep their owning component | Supply explicit frozen identities, fit/selection exposure, representation and response bytes. No model/provider is selected or invoked here. |
| Alternative construction and semantic/support validation | The actual constructor/validator owner; OR-07 or OR-09 when applicable | Supply changed/held-fixed controls, baseline/alternative lineage and recorded checks. General input editing, latent generation and decoding are not implemented by OR06-PCR. |
| Cross-space mapping | OR-10 or an explicitly bound upstream adapter | No direct cross-space comparison. Canonical input identity and all mapping dependencies must already satisfy the selected contract. Width alone is insufficient. |
| Matching, confounding diagnostics and causal estimation | Future separately versioned Hx06 profiles | Contract/RER, typed refusals and evidence gaps remain. They must not appear in `list` as implemented capabilities. |
| Aggregate uncertainty, derivatives and population-level sensitivity | Future method-specific contract/profile or evidence protocol | No generic reusable implementation exists here; supplied support checks are not computed balance/confounding diagnostics. |
| Synthetic generator, TRAIN-only OLS fitting, aggregate reporting and audit | [Frozen synthetic study](evaluation/synthetic_response_v1/INTERFACE.md) and its [protocol](evaluation/synthetic_response_v1/PROTOCOL.md) | Study-specific C++ runner and Node runner/audit exist, but are not registered general tools. The completed study is preserved; this map neither reruns it nor exposes it as a general fitting API. |

Provider neutrality means the same contract can consume records from different
compatible providers. The current profile still requires an exact known,
deterministic revision and compatible space/pipeline within each request. Unknown
hosted revisions, stochastic repetitions and arbitrary cross-space inputs are
unsupported by OR06-PCR `0.1.0`. Suitability for a new provider/population remains
an instance-specific evidence question.

The broader representation-contribution goal owns the new family-controlled
comparison, per-axis/population breakdowns and keep/revise/replace OLS decision.
It is independent of these engineering exposure gaps unless a new consumer
actually relies on that readout. Frozen evidence and artifacts retain their
identities throughout future toolset revisions.

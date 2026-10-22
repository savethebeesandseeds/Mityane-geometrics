# Hx06 toolset surface

**Map:** `HX06-TOOLSET`, revision `0.2.0`, 20 October 2026.  
**Registered operation:** OR06-PCR `0.1.0`, controlled model response only.  
**Contract and evidence:** [Operation Contract](CONTRACT.md), [RER](EVIDENCE.md).

The four selected exposure blocks are implemented: explicit request packaging,
recorded-alternative workflows, verified structured result loading and complete
output schemas. The [original surface plan](TOOLSET-v0.1.0.md) is retained.
Current work states belong to the
[Hero record](../../../web/dashboard/heroes/06-conterfactuals.json).

## Public calling surface

Use `code/operations/tools.ps1` on Windows, forwarding to the managed Linux
container. Inside it use `bash code/operations/tools.sh`. The Node entry is
`operations` from [tools.mjs](../tools.mjs); numerical computation remains in
the [C++20 comparison API](IMPLEMENTATION.md).

| Capability | CLI | Reusable Node API | Owner and result |
| --- | --- | --- | --- |
| Discovery | `list` | `operations.list()` | Common registry; registered IDs, revisions, family, title and claims. |
| Requirements and schemas | `describe OR06-PCR` | `operations.describe('OR06-PCR')` | Common envelopes and operation-owned payloads, exact revision, commands and limits. |
| Explicit request packaging | `prepare --spec FILE --output NEW_DIRECTORY` | `await operations.prepare(FILE, NEW_DIRECTORY)` | Common byte-bound packaging; returns new request path and artifact count. No comparison/model hooks. |
| Byte, compatibility and outcome checks | `check --request FILE` | `await operations.check(FILE)` | File admission plus Hx06 adapter/C++ engine; pure recorded arithmetic in memory, no result publication. |
| Retained comparison | `run --request FILE --output NEW_DIRECTORY` | `await operations.run(FILE, NEW_DIRECTORY)` | Same validated inputs; new bundle with full roster, states and contrasts. |
| Verified readable result | `inspect --result FILE` | `operations.inspect(FILE)` | Common bundle and operation-owned outcome checks; returns text. |
| Verified structured reuse | `read --result FILE` | `operations.read(FILE)` | Returns record, retained request, manifest and verified artifact paths using retained bytes. |
| Single-pair engineering example | `example OR06-PCR --provider toy-a\|toy-b --output NEW_DIRECTORY` | Operation-owned writer; no `operations.example` method | Artificial recorded fixture, no fitted/external provider. |
| Recorded alternative composition | `workflow OR06-PCR --provider toy-a\|toy-b --output NEW_DIRECTORY` | `await operations.workflow('OR06-PCR', {provider, output})` | [Hx06 workflow](tool/workflow.mjs) calls prepare/check/run/inspect/read; retains complete outcomes and expectations. |

The public C++ namespace is `mityane::operations::counterfactuals::v1`.
`compare(const Request&, optional<size_t> stop_after = nullopt)` returns
`RunResult`, including the supplied input. Bounded execution remains C++-only,
outside this file/CLI interface. C++ checks supplied records; the file adapter
adds actual byte binding.

## Published schemas

`describe` exposes complete schemas locally. The dependency-free validator
supports the exact vocabulary emitted here; it is not a general JSON Schema
implementation. Schema validity is separate from admission and scientific evidence.

| Descriptor field | Record identity | Definition |
| --- | --- | --- |
| `request_schema`, `input_schema` | `mityane.operation.request/v1`; `urn:mityane:operations:paired-response:input:1` | [Common schemas](../../shared/representation/tools/v1/schemas.mjs), [Hx06 input codec](tool/request.mjs) |
| `preparation_schema`, `prepared_schema` | `mityane.operation.prepare/v1`, `mityane.operation.prepared/v1` | Common schemas, selected operation/input specialized |
| `check_schema` | `mityane.operation.check/v1` | Common envelope with native payload |
| `result_schema`, `native_result_schema` | `mityane.operation.result/v1`; `urn:mityane:operations:paired-response:native-result:1` | Common envelope, [Hx06 output schemas](tool/schemas.mjs) |
| `response_artifact_schema` | `urn:mityane:operations:paired-response:response-artifact:1` | Hx06 axes and numerical responses |
| `files_schema`, `loaded_schema` | `mityane.operation.files/v1`, `mityane.operation.loaded/v1` | Common inventory/load wrapper with specialized operation payloads |
| `workflow_schema`, `workflow_source_schema`, `workflow_expectations_schema` | `mityane.operation.workflow/v1`, `mityane.operation.workflow-source/v1`, versioned expected roster | Operation-owned engineering workflow |

Preparation verifies the exact inventory, byte hashes and response bindings.
All metadata is caller-supplied: no guessed axes, providers, support, fit
populations, permissions or claims. It copies artifacts into a new request
bundle and retains the original specification. Invalid inputs fail before
publication; a write failure preserves its partial directory.

Structured loading checks file hashes/sizes and exact manifest coverage,
request/result operation and revision, claim/run IDs, original inventory joins,
retained response payloads, native pair identity/order, counts and contrasts.
It invokes no native executable, model or original input file. Returned paths
identify the verified retained files; later readers must account for subsequent
mutation. Hashes detect inconsistency, not authenticity or truth of supplied
attestations. The manifest is not a signature.

## Workflow and composition

The [usage guide](TOOL.md) gives Windows and library commands. Five alternatives
share one baseline. Signed changes +1, +2 and -1 produce manual expected deltas
`[1.5, -1]`, `[3, -2]` and `[-1.5, 1]`. An outside-support row abstains and a
same-width incompatible-space row fails. All rows remain: five requested,
three succeeded, one abstained, one failed, none unresolved. Toy providers use
widths four and seven in independent compatible requests, not cross-space comparisons.

`check.valid` means completed execution, possibly with abstentions; inspect
every pair before using a contrast. CLI codes are 0 completed, 2 invalid
usage/bytes or execution error, 3 rejected admission and 4 partial results.
`inspect` retains its readable-inspection exit-0 behavior. `read` and `workflow`
return 4 for this intentionally partial example; the workflow reports
`verified: true` when all expected outcomes match. Completion is no scientific verdict.

For reuse, call `read` and bind the record/request/manifest and artifact
identities to the receiving operation. Preserve axes/units, contrast sign,
source/parent/pair IDs, SpaceRef/pipeline revisions, population/horizon and every
refusal/state. Shared baselines retain their dependence; alternatives are not
independent samples. Each edge declares semantics and inherits access and timing
restrictions; the receiver checks its own contract and records transformations.
Scientific pipeline claims still require bound end-to-end evaluation.

## Deferred profiles and upstream ownership

| Capability | Explicit boundary |
| --- | --- |
| Matched observational contrast | No separate contract/profile, estimator or registered tool yet. The profile-expansion goal owns matching/confounding diagnostics, typed refusals and non-synthetic bindings. OR06-PCR rejects this claim. |
| Identified intervention effect | Separate contract/RER and evidence remain required: identification argument, untestable assumptions, timing, support and prohibited matching information. Proximity is not identification. OR06-PCR rejects this claim. |
| Encoding, providers, fitting and response execution | Remain with their model/task owners. Hx06 consumes explicitly frozen representation/response records; it selects or calls no model. |
| Source handling/preprocessing | Remain under `code/data/` and the dataset owner; lawful source evidence and permissions are supplied. |
| Alternative construction, decoding and semantic validity | Retain the constructor/validator owner and, where applicable, OR-07/OR-09 identities/evidence. The workflow packages constants rather than generating scientific alternatives. |
| Cross-space mapping | Requires a bound upstream mapping with transitive dependencies. Width alone gives no compatibility. |
| Derivatives, normalized sensitivities, stochastic averaging, aggregate uncertainty and population effects | No general implementation claimed; this increment composes recorded paired contrasts. |
| Frozen synthetic fitting/evaluation | [Study-specific interfaces](evaluation/synthetic_response_v1/INTERFACE.md) and evidence remain preserved; they are not registered general fitting tools and are not replayed. |
| Representation contribution/OLS adequacy | The separate contribution goal owns family controls, breakdowns and the keep/revise/replace decision. Tooling establishes no encoder signal. |

Provider neutrality still requires a known deterministic revision and compatible
space/pipeline within each request. Unknown hosted revisions, stochastic policies
and unsupported claims are refused. Reusable tooling supplies no causal effect
or arbitrary-provider suitability guarantee.

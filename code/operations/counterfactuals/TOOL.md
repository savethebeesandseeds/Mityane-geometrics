# Paired response tool

**Surface:** `HX06-TOOLSET 0.4.0`. This is the single CLI, Node and schema guide.

`OR06-PCR` is a callable tool for comparing **recorded numerical responses** to a
baseline and a declared alternative. It uses the common operation commands and
request/result envelopes described in the [toolset guide](../TOOLS.md). The
operation implementation and its provider-independent adapter remain in this
component.

The tool accepts any frozen representation provider that can supply the
[operation contract](CONTRACT.md). An upstream pipeline supplies the inputs,
representations, response values and evidence artifacts. The tool checks their
recorded relationships and computes `alternative - baseline` in the declared
response units. It does not invoke an encoder, call an embedding API, fit a
readout or materialize new representations/responses. Scenario sweeps enumerate
scalar recipes or package already recorded alternatives; upstream execution
keeps its own owner and evidence.

## Choose a workflow

| Task | Commands and guide |
| --- | --- |
| Try a recorded pair | `list`, `describe`, `example`, `check`, `run`, `inspect`: [quickstart](#quickstart) |
| Supply your own records | Request/artifact requirements [below](#supply-another-providers-records); `prepare` and `workflow` [examples](#prepare-and-compose-recorded-alternatives) |
| Load a saved result | `read` and `inspect`: [results and refusals](#results-and-refusals) |
| Plan or package a grid | `sweep`: [scenario space and catalog guide](#sweep-a-scenario-space) |
| Compare pipelines on the same experiment | `diff`: [roster identity, tolerances and output](#compare-two-retained-results) |
| Verify a completed study's retained records | `verify-study`: [supported studies and coverage limits](#verify-retained-synthetic-study-records) |
| Call the library | [Public Node API](#public-node-api), plus [advanced methods](#sweep-diff-and-verification-in-node) |

The [interface reference](#public-calling-surface) lists exact interfaces, published schemas,
composition requirements and upstream ownership. The [component index](README.md)
links scientific evidence and historical delivery records.

## Quickstart

Run these commands from the repository root in PowerShell. The wrapper forwards
to the existing managed Linux container. Commands that use native checks
(`check`, `run`, `workflow`, `sweep`, `verify-study`) build the adapter through
the root task entry. Inside the container, use `bash code/operations/tools.sh`
with repository-relative or `/mityane/` paths.

```powershell
.\code\operations\tools.ps1 list
.\code\operations\tools.ps1 describe OR06-PCR
.\code\operations\tools.ps1 example OR06-PCR --provider toy-a --output .temp\operations\counterfactuals\tool-demo-a
.\code\operations\tools.ps1 check --request .temp\operations\counterfactuals\tool-demo-a\request.json
.\code\operations\tools.ps1 run --request .temp\operations\counterfactuals\tool-demo-a\request.json --output .temp\operations\counterfactuals\tool-demo-a-result
.\code\operations\tools.ps1 inspect --result .temp\operations\counterfactuals\tool-demo-a-result\result.json
```

Every output directory must be new and under the ignored
`.temp/operations/counterfactuals/` directory. Choose another name when repeating
a command; existing evidence is preserved. Native build products
stay under `/opt/mityane-geometrics/` inside the container.

The engineering example has one pair, baseline `[2, -1]` and alternative
`[5, -3]`; its successful contrast is **`[3, -2]`**. `toy-a` declares an example
provider with a four-coordinate representation. Repeat the workflow with
`--provider toy-b` and fresh directory names to use another provider identity
and seven coordinates through the same interface. These are transparent,
project-authored engineering fixtures, with no fitted model or scientific
performance claim.

## Supply another provider's records

`describe OR06-PCR` returns the exact operation revision and input schema. The
generated `request.json` is a complete editable example of the common envelope:

| Field | Required meaning |
| --- | --- |
| `schema_version` | `mityane.operation.request/v1` |
| `operation` | Exact `{ "id": "OR06-PCR", "revision": "0.1.0" }` |
| `input` | The full typed paired-response request, including its complete pair roster |
| `artifacts` | The complete inventory of `{id, sha256, path}` records referenced by `input` |

Artifact paths resolve relative to the request file. Each recorded identity must
resolve to actual bytes with the declared SHA256. A response artifact is JSON
with exactly the branch's declared `axes` and `values`; a matching digest alone
does not excuse a response payload that disagrees with the request. Keep provider,
preprocessing, fitted-transform, representation-space and response-pipeline
identities explicit. A different vector width is supported in a separately bound
request; compatible coordinates within each pair remain required. Equal widths
do not establish compatibility between unrelated spaces.

The input also binds the claim, population, source/parent links, axis definitions
and units, temporal mode, changed/held-fixed attributes, support/validity checks,
permissions, resource budget and evidence destination. Missing, malformed and
unknown fields are rejected. The adapter keeps the [C++ reference semantics](IMPLEMENTATION.md);
it does not supply scientific defaults for omitted metadata.

The tool verifies bytes and consistency of supplied records. Support, validity,
provider revision and model-execution attestations remain supplied evidence:
successful validation does not independently authenticate a provider or prove
that its outputs are useful. Operation-specific suitability still needs the
[requirements and evidence record](EVIDENCE.md).

## Results and refusals

`check` verifies artifacts and runs the recorded comparison in memory. It writes
no result directory. `run` performs those checks and publishes a new bundle:

| File | Contents |
| --- | --- |
| `request.json` | Exact original request bytes |
| `artifacts/` | Copies of the verified input artifact bytes |
| `result.json` | Common result envelope, native pair outcomes, full denominator and execution provenance |
| `summary.txt` | Human-readable outcome and pair contrasts or refusal reasons |
| `manifest.json` | Hash-bound file inventory used by `inspect` |

`inspect` verifies the retained bundle against its manifest before showing the
summary. Preserve the original request and artifact identities when integrating
this result into another operation; the retained request and artifacts make the
input binding recoverable.

`read --result FILE` returns verified structured records instead of summary
text. It verifies saved bytes, exact inventory joins, operation/revision,
claim/run identity, full pair order/counts and recorded numerical bindings.
It uses the retained request/artifacts and does not need the original input
directory. Keep all states and lineage when passing this result downstream.

## Prepare and compose recorded alternatives

Use `prepare --spec FILE --output NEW_DIRECTORY` with a complete explicit
specification: `schema_version` is `mityane.operation.prepare/v1`; `operation`,
`input` and `artifacts` follow the full request schema returned by `describe`.
Artifact paths are relative to the specification file. All IDs, hashes,
metadata, responses and checks are supplied by the caller. Preparation checks
structure, actual bytes and response bindings; it calls no comparison engine
or model and infers no scientific configuration. It writes a new self-contained
request and retains the original specification.

The complete engineering workflow demonstrates preparation and result reuse:

```powershell
.\code\operations\tools.ps1 workflow OR06-PCR --provider toy-a --output .temp\operations\counterfactuals\workflow-a-001
.\code\operations\tools.ps1 read --result .temp\operations\counterfactuals\workflow-a-001\run\result.json
.\code\operations\tools.ps1 inspect --result .temp\operations\counterfactuals\workflow-a-001\run\result.json
```

Use fresh names; repeat with `toy-b` for a distinct seven-coordinate provider.
The workflow retains `source/prepare.json`, source artifacts and expected
outcomes, `prepared/request.json`, the complete `run/` bundle and `workflow.json`.
Both providers use five recorded alternatives sharing one baseline. Signed
changes +1, +2 and -1 yield `[1.5, -1]`, `[3, -2]` and `[-1.5, 1]`; one
outside-support alternative abstains and a same-width incompatible-space
alternative fails. Counts are **5 requested, 3 succeeded, 1 abstained, 1 failed,
0 unresolved**. This intentionally partial example makes `workflow` and `read`
return exit **4**. `verified: true` means the engineering expectations matched,
not that every pair succeeded. `inspect` returns exit 0 for a valid bundle.

## Public Node API

Inside the managed container, import the public entry point below. The same
preparation, validation, execution and verified loading are reusable without
the PowerShell wrapper:

```javascript
import {operations} from './code/operations/tools.mjs';
const prepared = await operations.prepare('/mityane/path/to/prepare.json',
  '/mityane/.temp/operations/counterfactuals/my-prepared-001');
const checked = await operations.check(prepared.request_file);
// Handle admission and every pair's state explicitly before deciding to run.
const saved = await operations.run(prepared.request_file,
  '/mityane/.temp/operations/counterfactuals/my-run-001');
const loaded = operations.read(saved.result_file);
console.log(loaded.record.summary, loaded.record.result.pairs);
// Or run the complete artificial example through the same public methods:
const workflow = await operations.workflow('OR06-PCR', {
  provider: 'toy-b', output: '/mityane/.temp/operations/counterfactuals/my-workflow-001'});
```

`loaded` includes the structured record, bound retained request, verified
manifest and retained artifact paths. Shared baselines retain their dependence;
do not treat alternatives as independent samples or drop refused rows.
The example composes recorded contrasts, not derivatives, normalized
sensitivities, confidence intervals or causal effects. Encoding, fitting,
response execution and scientific alternative construction remain upstream.

### Sweep, diff and verification in Node

Inside the managed container:

```javascript
import {operations} from './code/operations/tools.mjs';
const grid = await operations.sweep('OR06-PCR', {
  request: '/mityane/path/to/request.json', space: '/mityane/path/to/space.json',
  catalog: '/mityane/path/to/catalog.json', // omit for a plan only
  output: '/mityane/.temp/operations/counterfactuals/my-grid-001'});
const comparison = await operations.diff('OR06-PCR', {
  left: '/mityane/path/to/left/result.json', right: '/mityane/path/to/right/result.json',
  absoluteTolerance: 0, relativeTolerance: 0,
  output: '/mityane/.temp/operations/counterfactuals/my-diff-001'});
const verification = await operations.verifyStudy('OR06-PCR', {
  attempt: '/mityane/.temp/operations/counterfactuals/representation-contribution-v1/attempt-001',
  output: '/mityane/.temp/operations/counterfactuals/my-verification-001'});
```

These are public methods, alongside `prepare`, `check`, `run`, `read` and
`workflow`. `describe OR06-PCR` publishes their schemas and a separate
non-callable profile note for matched observational contrasts and identified
intervention effects. Its admitted `claim_levels` remains unchanged.

`describe` publishes the input/output schemas listed in [published schemas](#published-schemas).

## Exit codes and claim limits

| Exit code | Meaning |
| --- | --- |
| `0` | Comparison completed with successful and/or explicitly abstained pairs |
| `2` | Usage, schema, artifact-integrity or result-bundle validation error |
| `3` | Operation admission rejected, including an unsupported claim profile |
| `4` | Admitted roster includes failed or unresolved pairs |

`read` preserves the saved execution's status. `inspect` returns 0 for a valid
saved bundle, including partial results. A plan-only sweep's exit 0 means
successful planning; no comparison has executed.
Node results expose `status`; thrown admission errors carry `exitCode: 3`.

A valid pair outside its declared support abstains without a numeric contrast.
An incompatible space or upstream failure produces a failed pair. Every admitted
pair stays in the result and denominator; refusal is never converted to zero.
The sole supported claim is `controlled_model_response`. Requests for
observational contrasts or identified intervention effects are rejected by this
revision. The separately authored profiles, their archival bindings and the
scoped synthetic results are indexed in the [component guide](README.md#contracts-and-scientific-evidence).
These commands do not rerun fitting or generate scientific data. Sweep planning
labels and verification's retained-only abstentions stay separate from executed
pair outcomes; the [advanced-command sections](#sweep-a-scenario-space) explain their reports.

## Sweep a scenario space

```powershell
.\code\operations\tools.ps1 sweep OR06-PCR --request PATH\request.json --space PATH\space.json --output .temp\operations\counterfactuals\plan-001
```

The [space schema](tool/sweep.mjs) requires an existing baseline pair ID, new
request/run IDs, an explicit sampling rule, varied axes and an explicit hold
list. Every declared scalar attribute must be varied or held. Only attributes
already declared `changed` can vary; held-fixed/coupled rules are preserved.
The base request, frozen pipeline, claim, support policy and original baseline
are retained. Each axis uses `add`, `set` or `scale` relative to the baseline,
and either `values: [...]` or an inclusive `start`, `end`, `step` range.

For example, the supplied [engineering space](tool/examples/sweep-space.json)
adds -1, +1, +2 and +10 to setting=1 while holding load=2. Multiple axes form the
complete Cartesian product in declaration order. Decimal ranges are constructed
without repeated floating-point addition and must reach their end exactly.
Zero steps, reversed ranges, duplicate/collapsed scenarios, unknown attributes,
incomplete hold lists, nonfinite arithmetic and grids over 10,000 cells refuse.
Expanded manifests and source/artifact byte budgets are also bounded.

Without a catalog, the output is a **plan**. It includes every cell, target
attributes and explicit `ready_for_materialization` or `blocked_support` planning
status. Planning checks the declared scalar bounds; opaque support/validity
policies still require upstream evidence. These are planning labels, not native
pair outcomes. An unsupported cell stays visible. The output has no runnable `request_file`: actual upstream input,
representation, response and evidence records are still required.

To package existing records, add `--catalog PATH\catalog.json`. A catalog has
schema `mityane.operation.sweep-catalog/v1` and one entry for every coordinate:

```json
{
  "schema_version": "mityane.operation.sweep-catalog/v1",
  "entries": [
    {"coordinate": [-1], "request": "source/request.json", "pair_id": "recorded-pair-minus-one"}
  ]
}
```

Coordinates contain axis **operands**, in the space's axis order. Request paths
are canonical child paths relative to the catalog; place the catalog above its
source requests. Every selected pair must already have complete real records
and match the baseline, target attributes, source scope, pipeline and support
declarations. Missing entries, changed bindings or altered bytes fail clearly.
No response or representation is copied from another scenario as a substitute.

An out-of-support recorded cell is accepted only when the unchanged comparator
actually abstains for its declared support. A hold/mask/integrity violation
retains failure/rejection behavior. Existing failed upstream records remain
failed; the command never turns them into support abstentions or numerical zero.

The recorded mode calls the existing `prepare` and `check` paths, returning a
self-contained `prepared/request.json` directly usable with `check` and `run`.
The sweep output retains `plan.json`, `sweep.json`, exact base/space/catalog bytes,
`source-pins.json`, source artifacts and the preparation specification. The exact
`base-request.json` audit copy is not the runnable prepared request.

### Complete engineering example

Use the existing artificial workflow as the catalog. Its exit 4 is expected:
the original five-row workflow contains one deliberate failed pair. The grid
selects three supported records and one genuine support abstention.

```powershell
.\code\operations\tools.ps1 workflow OR06-PCR --provider toy-a --output .temp\operations\counterfactuals\workflow-a-001
Copy-Item -LiteralPath code\operations\counterfactuals\tool\examples\sweep-catalog.json -Destination .temp\operations\counterfactuals\sweep-catalog.json
.\code\operations\tools.ps1 sweep OR06-PCR --request .temp\operations\counterfactuals\workflow-a-001\prepared\request.json --space code\operations\counterfactuals\tool\examples\sweep-space.json --catalog .temp\operations\counterfactuals\sweep-catalog.json --output .temp\operations\counterfactuals\grid-001
.\code\operations\tools.ps1 run --request .temp\operations\counterfactuals\grid-001\prepared\request.json --output .temp\operations\counterfactuals\grid-run-a-001
.\code\operations\tools.ps1 run --request .temp\operations\counterfactuals\grid-001\prepared\request.json --output .temp\operations\counterfactuals\grid-run-b-001
```

Choose fresh directory names and adjust the catalog child paths when repeating
the example. These are supplied artificial records, not scientific model calls.

## Compare two retained results

```powershell
.\code\operations\tools.ps1 diff OR06-PCR --left .temp\operations\counterfactuals\grid-run-a-001\result.json --right .temp\operations\counterfactuals\grid-run-b-001\result.json --output .temp\operations\counterfactuals\grid-diff-001 --abs-tolerance 0 --rel-tolerance 0
```

Both source bundles are independently verified through `read` first. Their
experimental-roster fingerprints must match: ordered pair IDs, source/raw-input
content and lineage, changed/held-fixed declarations, constructor/control
evidence, mask contents, support policy, population, timing/horizon and response
axes/units. Different model, provider, representation space, readout and pipeline
identities remain separately recorded. Full request hashes are not required to
match because those pipeline records intentionally differ.

For encoder/readout comparisons, both upstream pipelines must preserve the same
source experiment and response definitions. The tool supplies no mapping between
different input IDs, masks, units or populations. Mismatches fail before output.

Only success/success pairs receive a **right contrast minus left contrast** and
absolute difference. Comparison values are null whenever either contrast is
missing. All `success`, `abstained`, `failed` and `unresolved` states and reason
codes remain explicit. Finite nonnegative absolute/relative tolerances use
`abs(right-left) <= absolute + relative*max(abs(left),abs(right))`; both default
to zero. Overflow is rejected. Tolerance agreement is numerical agreement,
not evidence that either pipeline is accurate or scientifically useful.

Output contains `comparison.json`, `roster.json`, `summary.txt`, a hash manifest,
and complete verified `left/` and `right/` source bundles. The outer comparison
is not a normal `result.json` bundle; existing `read`/`inspect` continue to accept
the retained `left/result.json` and `right/result.json` normally.

## Verify retained synthetic study records

```powershell
.\code\operations\tools.ps1 verify-study OR06-PCR --attempt .temp\operations\counterfactuals\synthetic-response-v1\attempt-001 --output .temp\operations\counterfactuals\verify-original-001
.\code\operations\tools.ps1 verify-study OR06-PCR --attempt .temp\operations\counterfactuals\representation-contribution-v1\attempt-001 --output .temp\operations\counterfactuals\verify-contribution-001
```

This finite verifier supports the two reviewed completed study identities:
`OR06-SYNTH-RESP-001 1.0.0` and `OR06-REP-CONTRIB-001 1.0.0`. The retained files
are local ignored evidence, not a dataset included in a Git checkout. Missing,
altered, extra or redirected files fail; nothing is downloaded or regenerated.
Known final-record hashes anchor the complete manifests, source capsules,
configuration, permissions, model seals, assessment-open records and audits.
Exact relocated copies can be verified without rewriting historical path pins.

Every retained development and assessment row is accounted for. Recorded branch
values are adapted with explicit new verification provenance and passed through
the **current** `check` loop. No old study runner, auditor, generator, encoder,
model, fit, selection or bootstrap is executed. Original records remain exposed;
this is engineering verification, not fresh scientific confirmation or a new
representation-contribution experiment.

The contribution study's support-stress rows have no saved upstream predictions
or vectors. They retain their archived abstention and null contrast in
`retained-only.json`. They are explicitly marked not executable by the current
checker; the verifier cannot invent complete branches to make them run. Separate
source counts and current-check counts make that distinction visible. The first
study retains complete stress records, so those can pass through current refusal
checks as well.

MAE is recomputed only where per-pair reference truth was actually retained
(the contribution study). The original study's MAE is reported as a verified
**archived audit statistic**, not recomputed truth or a new measurement. Its
saved uncertainty stays archived. `verification.json` partitions both splits,
states, checked coverage, retained-only rows, numerical checks and metric origins.

The [delivery review](reviews/tool-ergonomics-v1.md) records actual verification
of both studies and the CLI/Node regression checks. Original development plus
assessment contains 27,648 executable records. Contribution development plus
assessment contains 95,040 source records: 79,200 executable successes and 15,840
retained-only abstentions. Counts for each split remain available in the report.

Output also retains a manifest and compressed checked-record archives containing
exact request envelopes, artifact payloads and current check outputs. These are
verification records, not `run` bundles. Scratch files stay in a named
`/opt/mityane-geometrics/operations/counterfactuals/retained-verification/` directory.

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
| Scenario sweep | `sweep OR06-PCR --request FILE --space FILE [--catalog FILE] --output NEW_DIRECTORY` | `await operations.sweep('OR06-PCR', options)` | [Sweep](tool/sweep.mjs) enumerates a complete scalar grid; real supplied catalog records pass through existing prepare/check. A plan alone is not runnable. |
| Retained-result difference | `diff OR06-PCR --left RESULT_JSON --right RESULT_JSON --output NEW_DIRECTORY` | `await operations.diff('OR06-PCR', options)` | [Difference](tool/diff.mjs) independently reads both bundles, verifies experimental identity, retains states/reasons and calculates success-only right-minus-left contrasts. |
| Retained-study verification | `verify-study OR06-PCR --attempt RETAINED_DIRECTORY --output NEW_DIRECTORY` | `await operations.verifyStudy('OR06-PCR', options)` | [Verifier](tool/verify-study.mjs) binds reviewed final hashes and retained artifacts, checks recorded responses and labels non-executable retained abstentions. No fitting or data generation. |

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
| `sweep_options_schema`, `sweep_space_schema`, `sweep_catalog_schema`, `sweep_result_schema` | Versioned Cartesian-plan/catalog and packaging schemas | Operation-owned sweep; full scalar hold declarations and bounded grids |
| `diff_options_schema`, `diff_output_schema`, `diff_result_schema`, `diff_files_schema` | Public `diff-output/v1` wrapper, `mityane.operation.diff/v1` comparison, experimental roster fingerprint and retained-source manifest | Operation-owned comparison with explicit numerical tolerances |
| `verify_study_options_schema`, `verify_study_result_schema` | `mityane.hx06.retained-study-verification/v1` | Reviewed exposed study records; source/current-check coverage kept distinct |

The [common calling contract](../TOOLS.md#the-common-calling-contract-version-1)
defines byte-bound preparation, exclusive publication, execution states and
exit codes. [Structured loading](../TOOLS.md#reusable-apis-and-ownership) verifies
the retained bundle independently of the original source paths. Schema validity,
byte consistency, provider authenticity and scientific evidence remain distinct.

## Workflow and composition

Scenario probing, feature ablation/necessity probes, distribution-shift rehearsals
and comparative encoder evaluation can use the same planning/catalog/comparison
surface. Each still needs concrete upstream records and an appropriate scientific
protocol. A declared shift or ablation recipe does not establish OR-04 detection,
causal necessity, encoder superiority or an identified intervention effect.

The [recorded-alternative example](#prepare-and-compose-recorded-alternatives)
demonstrates preparation, a complete roster with deliberate abstention/failure,
saved output and reuse. `check.valid` means completed execution, possibly with
abstentions; inspect every pair before using a contrast. Completion and matched
engineering expectations supply no scientific verdict.

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
| Matched observational contrast | [OR06-MOC 0.1.0](profiles/observational_contrast/v0.1.0/CONTRACT.md) and its RER are authored, with an actual [NSW archival instance](profiles/instances/nsw_reference_v1/OBSERVATIONAL.md). No estimator/registered tool or matched-result evidence exists. OR06-PCR still rejects this claim. |
| Identified intervention effect | [OR06-IIE 0.1.0](profiles/intervention_effect/v0.1.0/CONTRACT.md) and its RER are authored, with a separately bound [NSW assignment-effect instance](profiles/instances/nsw_reference_v1/INTERVENTION.md). Identification assumptions and missing design/uncertainty details are explicit; no causal execution is admitted. Proximity is not identification. OR06-PCR still rejects this claim. |
| Encoding, providers, fitting and response execution | Remain with their model/task owners. Hx06 consumes explicitly frozen representation/response records; it selects or calls no model. |
| Source handling/preprocessing | Remain under `code/data/` and the dataset owner; lawful source evidence and permissions are supplied. |
| Alternative construction, decoding and semantic validity | Retain the constructor/validator owner and, where applicable, OR-07/OR-09 identities/evidence. The workflow packages constants rather than generating scientific alternatives. |
| Cross-space mapping | Requires a bound upstream mapping with transitive dependencies. Width alone gives no compatibility. |
| Derivatives, normalized sensitivities, stochastic averaging, aggregate uncertainty and population effects | No general implementation claimed; this increment composes recorded paired contrasts. |
| Frozen synthetic fitting/evaluation | [Study-specific interfaces](evaluation/synthetic_response_v1/INTERFACE.md) and evidence remain preserved. `verify-study` checks retained exposed records without rerunning fitting, models, generators or scientific assessment. It is not a general fitting tool. |
| Representation contribution/OLS adequacy | The [completed contribution evidence](evaluation/representation_contribution_v1/RESULTS.md) and [binding decision](evaluation/representation_contribution_v1/READOUT_DECISION.md) keep their scope. The ergonomic tools establish no new encoder signal. |

Provider neutrality still requires a known deterministic revision and compatible
space/pipeline within each request. Unknown hosted revisions, stochastic policies
and unsupported claims are refused. Reusable tooling supplies no causal effect
or arbitrary-provider suitability guarantee.

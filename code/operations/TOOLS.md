# Operation tools

Operations are reusable tools over explicitly described representations and
records. Model-specific encoding stays with the provider. A tool declares its
requirements and checks whether supplied records meet them; changing an embedding
model does not require rewriting the operation.

The first callable tool is **OR06-PCR 0.1.0**, recorded paired model-response
comparison. Its [usage guide](counterfactuals/TOOL.md) provides examples and the
exact boundary. Other OR families retain their existing interfaces until they
adopt this boundary; they are not implicitly registered or migrated.

## Start using the toolset

From the repository root in PowerShell:

```powershell
.\code\operations\tools.ps1 list
.\code\operations\tools.ps1 describe OR06-PCR
```

Continue with the [OR06-PCR quickstart](counterfactuals/TOOL.md#quickstart) for a
complete engineering example, then its [advanced commands](counterfactuals/TOOL.md#sweep-a-scenario-space)
for sweeps, pipeline differences and retained-study verification. Use fresh
output directories for every invocation; earlier evidence is preserved.

The host wrapper forwards arguments through the existing managed container.
`check`, `run`, `workflow`, `sweep` and `verify-study` first build the registered
tools using the root task runner;
there are no new dependencies, services or environments. Relative repository
paths and absolute Windows paths inside this checkout are supported by the
PowerShell wrapper. Inside the container use
`bash code/operations/tools.sh` with repository-relative or `/mityane/` paths.

## The common calling contract, version 1

| Command | Common behavior |
| --- | --- |
| `list` | List only implemented, registered tools and their revisions |
| `describe ID` | Report requirements, claim scope, limits and a complete request JSON schema |
| `example ID` | Write an operation-owned example request bundle to a new directory |
| `prepare --spec FILE --output NEW_DIRECTORY` | Package explicitly supplied typed input and byte-bound artifacts without executing the operation |
| `check --request FILE` | Validate records and bytes through an explicitly read-only adapter hook; publish nothing |
| `run --request FILE --output NEW_DIRECTORY` | Execute the selected operation and retain its request, input artifacts, result and readable summary |
| `inspect --result FILE` | Verify the saved bundle's file hashes and display its retained result; execute no operation |
| `read --result FILE` | Verify the saved bundle and operation-specific relationships, then return structured records and retained artifact paths |
| `workflow ID --provider toy-a\|toy-b --output NEW_DIRECTORY` | Run an operation-owned engineering composition through the same public library; currently the Hx06 recorded-alternative example |
| `sweep ID --request FILE --space FILE [--catalog FILE] --output NEW_DIRECTORY` | Operation-owned Cartesian scenario plan or packaging of supplied recorded alternatives |
| `diff ID --left RESULT_JSON --right RESULT_JSON --output NEW_DIRECTORY` | Operation-owned comparison of independently verified results on an identical source experiment |
| `verify-study ID --attempt RETAINED_DIRECTORY --output NEW_DIRECTORY` | Operation-owned retained-record verification for reviewed study identities; no research rerun |

Requests use `mityane.operation.request/v1`:

```json
{
  "schema_version": "mityane.operation.request/v1",
  "operation": {"id": "OR06-PCR", "revision": "0.1.0"},
  "input": {},
  "artifacts": [{"id": "an-immutable-id", "sha256": "64-lowercase-hex-characters", "path": "artifacts/record.json"}]
}
```

This abbreviated illustration is not an executable request: `input` follows the
selected operation's schema and every referenced artifact needs a matching file.
Use `describe` and the generated example for the full request. Duplicate keys,
unknown fields, invalid numbers and incomplete inventories are rejected. File
paths are explicit and relative to the request bundle; there is no dataset or
model discovery, network access, or provider selection during execution.

The common result format is `mityane.operation.result/v1`. It records the
operation/revision, input request identity, claim level, execution disposition,
requested/succeeded/abstained/failed/unresolved counts, operation-owned results,
diagnostics, implementation/source identities and retained artifact references.
An admission rejection retains the requested count with no invented executed
rows or state counts. Once admitted, all requested units enter the denominator.

Preparation specifications use `mityane.operation.prepare/v1` with the same
`operation`, complete typed `input` and exact `artifacts` inventory as a request.
Paths resolve relative to that specification. IDs and hashes are explicit;
preparation verifies bytes and operation-owned artifact semantics, never guesses
scientific fields, and never invokes check/execute or a model. It retains the
original specification as `preparation.json`, copies input bytes under
`artifacts/`, and writes a self-contained `request.json` with rebased paths.
Existing output directories are refused; invalid inputs publish no output and
write failures preserve any partial directory.

`completed` means the invocation finished with successes and/or explicit
abstentions. `rejected` means admission failed; `partial` retains failed or
unresolved admitted units. These execution states do not grant an evidence
verdict. The CLI returns 0 for completion, 2 for invalid usage/bytes or execution
errors, 3 for rejected admission and 4 for partial results. Errors before native
execution do not publish a result directory. A publication failure preserves its
partial directory and writes a failure record when possible.

Saved bundles contain `request.json`, `result.json`, `summary.txt`, `manifest.json`
and all admitted input artifact bytes. They use exclusive creation and never
overwrite an existing output. Hash verification detects inconsistency against
the retained manifest; it is not a signature or proof of source authenticity.

## Reusable APIs and ownership

Inside the managed container, applications can use the same boundary directly:

```javascript
import {operations} from './code/operations/tools.mjs';
const description = operations.describe('OR06-PCR');
const checked = await operations.check('/mityane/path/to/request.json');
// Inspect checked.valid and its diagnostics before running.
const saved = await operations.run('/mityane/path/to/request.json',
  '/mityane/.temp/operations/counterfactuals/tool-runs/application-001');
console.log(operations.inspect(saved.result_file));
const loaded = operations.read(saved.result_file);
// Keep loaded.record, loaded.request, loaded.manifest and loaded.artifacts together.
```

The existing [C++ comparison API](counterfactuals/IMPLEMENTATION.md) also remains
available. Neither API installs dependencies or starts a model. An app or agent
can wrap this same versioned boundary later.

The [shared file runtime](../shared/representation/tools/v1/runtime.mjs) owns
discovery, strict envelope/inventory handling, publication and inspection.
Operation-owned adapters supply input encoding, artifact semantics, a read-only
check hook, execution, outcome accounting and readable details. JavaScript-only
adapters can record `native_binary: null`; the common runtime requires no tensor
library, model architecture or C++ payload. A registry engineering fixture checks
a different, non-pair payload without claiming a second implemented capability.

Keep operation logic in its `code/operations/` component, model adapters in
`code/embeddings/`, data handling in `code/data/`, and shared mechanisms in the
existing `code/shared/` ownership boundaries. Each added tool still declares its
own kinds, roles, geometry and dependencies. Width alone never establishes
compatibility or suitability, and an input supported by one operation need not
support every other operation.

`operations.prepare(specFile, newOutputDirectory)` exposes explicit packaging;
`operations.workflow(id, {provider, output})` dispatches operation-owned
composition through the same library. The common runtime does not select
scientific alternatives or execute providers. CLI `read` returns the saved
execution's exit status; readable `inspect` returns 0 on a valid bundle even
when it retains abstained, failed or unresolved outcomes.

`read` returns `mityane.operation.loaded/v1`: the verified `record`, retained
`request`, `manifest`, `result_file`, and `artifacts` entries with IDs, hashes,
byte counts and absolute retained paths. It reads saved bytes independently of
the original source paths. It verifies operation/revision, claim/run identity,
complete inventory, native outcome accounting and operation-owned payload
bindings without calling check/execute. Subsequent file mutation still requires
fresh verification. Hash-bound records provide consistency, not authenticity.

`describe` publishes versioned preparation, prepared, check, result,
file-manifest and structured-load schemas alongside the operation's input,
response-artifact and native-result schemas. Hx06 also publishes its workflow
report/source/expectation schemas. Common schemas remain in
[the shared boundary](../shared/representation/tools/v1/schemas.mjs); native
payloads stay with their operation. Runtime validation supports the schema
vocabulary actually emitted here without dependencies or network fetching.

Ordinary tool invocations and research evaluations share provenance conventions
but have different purposes. A new scientific claim still needs its bound
contract, evidence record and evaluation protocol. The completed Hx06 synthetic
studies remain preserved. Hx06's [interface reference](counterfactuals/TOOL.md#public-calling-surface)
lists operation-owned `sweep`, `diff` and `verify-study` methods and schemas.
Study verification uses retained exposed records and the current comparison
check, without fitting, generating data or reopening a protected scientific
assessment. Supported study identities remain explicit in `describe`.

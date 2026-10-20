# Paired response tool

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
readout or generate alternatives. Those steps keep their own owners and evidence.

## Try the complete workflow

Run these commands from the repository root in PowerShell. Each command uses
the existing managed container; `check` and `run` build the small native adapter
through the root task entry when needed.

```powershell
.\code\operations\tools.ps1 list
.\code\operations\tools.ps1 describe OR06-PCR
.\code\operations\tools.ps1 example OR06-PCR --provider toy-a --output .temp\operations\counterfactuals\tool-demo-a
.\code\operations\tools.ps1 check --request .temp\operations\counterfactuals\tool-demo-a\request.json
.\code\operations\tools.ps1 run --request .temp\operations\counterfactuals\tool-demo-a\request.json --output .temp\operations\counterfactuals\tool-demo-a-result
.\code\operations\tools.ps1 inspect --result .temp\operations\counterfactuals\tool-demo-a-result\result.json
```

Example and result directories must be new. Choose another name when repeating
the workflow; existing evidence is preserved. Both kinds of output live under
the ignored `.temp/operations/counterfactuals/` directory. Native build products
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

| Exit code | Meaning |
| --- | --- |
| `0` | Comparison completed with successful and/or explicitly abstained pairs |
| `2` | Usage, schema, artifact-integrity or result-bundle validation error |
| `3` | Operation admission rejected, including an unsupported claim profile |
| `4` | Admitted roster includes failed or unresolved pairs |

A valid pair outside its declared support abstains without a numeric contrast.
An incompatible space or upstream failure produces a failed pair. Every admitted
pair stays in the result and denominator; refusal is never converted to zero.
The sole supported claim is `controlled_model_response`. Requests for
observational contrasts or identified intervention effects are rejected by this
revision. The [initial synthetic study](evaluation/synthetic_response_v1/RESULTS.md)
provides evidence for its recorded setting; these commands do not rerun it.

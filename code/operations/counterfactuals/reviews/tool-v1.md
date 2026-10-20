# Hx06 callable tool verification

**Date:** 19 October 2026.  
**Decision:** complete for the recorded-response tool boundary, version 1.  
**Operation:** OR06-PCR 0.1.0; controlled model response only.

## Delivered boundary

The [common toolset](../../TOOLS.md) exposes list, describe, example, check, run
and inspect through one container-backed command entry. Its reusable Node API
uses the same registry, request envelopes, byte inventories and saved results.
The existing C++ numerical interface remains unchanged. The operation-owned
[adapter](../tool/adapter.mjs), [strict request codec](../tool/request.mjs) and
[native bridge](../tool/native.cpp) are kept under counterfactuals.

The shared runtime imports neither an encoder nor counterfactuals. Each adapter
retains its own schema, artifact semantics, read-only check hook, execution and
outcome interpretation. The current production registry contains only OR06-PCR;
other capability families have not been silently migrated or claimed complete.

## Executed checks

The existing managed container executed:

```text
bash code/scripts/task.sh operation-tools-v1 test-operation-tools test-counterfactuals
```

All **17 tool-boundary tests** passed, followed by all **47 existing C++
conformance cases**, with zero failures. The native code compiled with C++20,
`-Wall -Wextra -Wpedantic`, without warnings. The Node test suite's total recorded
duration was 217.733 seconds, including filesystem publication and inspection;
this is not a performance benchmark of comparison arithmetic.

The [boundary tests](../tool/tool.test.mjs) check:

- Two independent recorded provider identities with native widths four and seven
  use the same callable operation; manual expected response difference `[3, -2]`
  matches the actual native result.
- A distinct JavaScript-only engineering adapter accepts `items`, returns doubled
  values, publishes and inspects through the common runtime without `pairs`,
  `admitted`, encoder types or a native executable. Checking uses its separate
  read-only hook. This is a registry fixture, not a second implemented OR family.
- Full input bytes, exact original request and artifact hashes survive publication;
  check publishes nothing and inspect executes no operation.
- Unicode survives framing; unsafe/fractional integer metadata, missing explicit
  optionals, malformed Unicode, unknown fields/enums and duplicate JSON keys reject.
- Missing inventory, changed bytes, even correctly rehashed mismatched response
  values, and escaped artifact paths reject before result publication.
- Supported and abstained pairs retain their denominator; incompatible spaces
  retain failed pairs; unsupported causal claims reject admission without invented
  executed rows. Native reason codes and rejected-request counts are preserved.
- Existing outputs survive collision and retained-byte tampering makes inspection
  refuse to return a summary.

The final engineering fixture directory is
[boundary-t4lW3M](../../../../.temp/operations/counterfactuals/toolset-tests/boundary-t4lW3M).
It intentionally includes invalid/tampered cases. The earlier 14-case development
attempt, with three failures corrected against the existing native API, remains
preserved as
[boundary-kxvmrd](../../../../.temp/operations/counterfactuals/toolset-tests/boundary-kxvmrd).
Neither directory supplies scientific provider-quality evidence.

## User workflow

The Windows [launcher](../../tools.ps1) was exercised for list, example, check,
run and inspect. Both repository-relative paths and absolute Windows paths were
forwarded successfully into the managed Linux container. The native build stayed
under `/opt/mityane-geometrics/embeddings/build/operation-tools-v1/`.

The retained [example request](../../../../.temp/operations/counterfactuals/toolset-demo-001/provider-a/request.json)
passed check. Its [saved result](../../../../.temp/operations/counterfactuals/toolset-demo-001/provider-a-result/result.json)
reports one requested/succeeded pair, no abstentions/failures/unresolved rows,
baseline `[2, -1]`, alternative `[5, -3]`, and delta `[3, -2]`.
Inspection independently verified the file manifest before displaying the
[readable summary](../../../../.temp/operations/counterfactuals/toolset-demo-001/provider-a-result/summary.txt).
Actual native/source hashes and execution records are retained in that bundle.

The final integration checks verified 1,064 maintained Markdown links and 71
literal shell roots. The mandatory dashboard refresh passed all 34 focused tests
and regenerated the standalone HTML with the new Hx06 tool milestone recorded.

## Scope and limits

The tool reads recorded numerical responses. It does not construct alternatives,
execute an embedding model or response head, train, access an API or require a
selected external provider. Bytes and declared relationships are checked;
upstream execution, support and semantic attestations remain caller-supplied.
Compatible widths are never a substitute for declared representation identity.

These are engineering checks. No new scientific experiment was run and the
frozen synthetic protocol, runners and source capsule were preserved. The common
interface does not establish suitability across arbitrary providers, causal
effects or downstream application quality. Broader shared-record validation and
other Heroes retain their own scope and status.

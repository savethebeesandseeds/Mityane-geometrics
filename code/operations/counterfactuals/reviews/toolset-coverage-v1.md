# Hx06 toolset delivery verification

**Date:** 20 October 2026.  
**Decision:** complete for the current Hx06 reusable-toolset goal.  
**Surface:** HX06-TOOLSET `0.2.0`; numerical operation OR06-PCR `0.1.0`.

## Delivered coverage

The [delivered surface map](../archive/TOOLSET-v0.2.0.md) and [usage guide](../TOOL.md) document
all four exposure blocks selected by the
[original map](../archive/TOOLSET-v0.1.0.md). Independent source review found no remaining
delivery blocker; the executed checks below establish their engineering acceptance.

| Selected block | Delivered CLI and library behavior |
| --- | --- |
| Explicit input packaging | `prepare --spec FILE --output NEW_DIRECTORY` / `operations.prepare(spec, output)` verify the complete caller-supplied typed input, exact artifact inventory, hashes and response bindings. They copy retained bytes and write a self-contained request without invoking comparison/model hooks or inventing metadata. |
| Recorded-alternative workflow | `workflow OR06-PCR --provider toy-a\|toy-b --output NEW_DIRECTORY` / `operations.workflow(id, options)` compose prepare/check/run/inspect/read over five explicit alternatives sharing one source baseline. |
| Structured verified reuse | `read --result FILE` / `operations.read(file)` return the verified result, retained request, manifest and retained artifact paths. They operate independently of original source files and retain every state. |
| Public output schemas | `describe` publishes complete preparation/prepared/check/result/file-manifest/load schemas, plus Hx06 response/native/workflow payload schemas. Common records stay shared; Hx06 fields stay operation-owned. |

Numerical semantics and the C++ interface are unchanged. The adapter additionally
checks returned vector widths, response bindings and signed arithmetic before
publication or reuse. Its source inventory includes the new executable schema
and workflow modules. Existing `inspect` still returns readable text with exit 0;
structured read and workflow preserve execution exit status, including 4 for
the intentionally partial example.

## Executed engineering checks

The existing managed Linux container built and ran:

```text
bash code/scripts/task.sh hx06-toolset-coverage-v1 test-operation-tools test-counterfactuals
```

All **28 tool-boundary tests** passed: the original 17 plus 11 new coverage
checks. All **47 C++ conformance cases** passed. There were zero failures,
skips or cancellations. The build used C++20 with `-Wall -Wextra -Wpedantic`
without compiler warnings. The boundary suite took 518.099 seconds, including
retained-file publication and verification on the Windows bind mount; this is
not a numerical performance benchmark.

The retained [full log](../../../../.temp/operations/counterfactuals/toolset-coverage-v1/verification-2026-10-20-001/tests.log)
and [18-file source inventory](../../../../.temp/operations/counterfactuals/toolset-coverage-v1/verification-2026-10-20-001/source-inventory.json)
bind this check. Build products stayed under
`/opt/mityane-geometrics/embeddings/build/hx06-toolset-coverage-v1/`.
The [coverage fixtures](../../../../.temp/operations/counterfactuals/toolset-tests/coverage-4mf3i7)
include intentionally invalid, redirected and coherently corrupted cases; they
are engineering artifacts, not scientific evidence.

Checks establish:

- Explicit metadata and artifact bytes survive preparation. Malformed/duplicate
  JSON, missing or conflicting records, changed bytes, redirects, path escapes
  and existing destinations are refused without replacing prior output.
- Structured reads work after original source/prepared directories are detached.
  Byte tampering and coherently rehashed claim, revision, run, roster,
  inventory-path and arithmetic corruption are refused.
- A separate JavaScript adapter with `items` and no pairs/native binary uses
  preparation, checking, execution and structured reuse. Preparation and saved
  reads invoke no execution hook. This fixture registers no new research capability.
- Discovery schemas accept actual outputs and reject malformed counts, unknown
  states, invalid diagnostics and incompatible shapes. Unsupported causal
  metadata is preserved during packaging and then receives typed admission
  refusal, without invented executed pairs.

## Both provider workflows

The [API provider-A report](../../../../.temp/operations/counterfactuals/toolset-tests/coverage-4mf3i7/api-workflow-48/workflow.json)
and [CLI provider-B report](../../../../.temp/operations/counterfactuals/toolset-tests/coverage-4mf3i7/cli-workflow-49/workflow.json)
use widths four and seven through the same interfaces. Each retains 57 bound
artifact identities and all five alternatives. Expected deltas are
`[1.5, -1]`, `[3, -2]` and `[-1.5, 1]`; the remaining rows have null contrasts
and typed support abstention/incompatible-space failure. Counts reconcile:
**5 requested, 3 succeeded, 1 abstained, 1 failed, 0 unresolved**. Shared-baseline
dependencies and signed change magnitudes/directions remain explicit.

Windows launcher checks separately used a repository-relative `--spec` and an
absolute Windows output path for preparation. An absolute Windows `--result`
loaded provider B's retained result, preserving 57 artifacts, all five rows and
the expected exit 4. The [preparation report](../../../../.temp/operations/counterfactuals/toolset-coverage-v1/verification-2026-10-20-001/windows-prepared-report.json)
and [structured-read report](../../../../.temp/operations/counterfactuals/toolset-coverage-v1/verification-2026-10-20-001/windows-read-report.json)
retain those outcomes. PowerShell only forwarded commands and inspected results;
all project code ran in the managed container.

## Claim and preservation boundary

This completes the agreed current toolset coverage, not the entire Hero.
Observational/intervention profile authoring and representation contribution
remain separate goals. Provider execution, fitting, response generation,
scientific alternative construction and semantic attestations retain their
owners. Matching widths, successful invocation or hash consistency provide no
causal identification or arbitrary-provider suitability guarantee.

No encoder, model API, training, GPU work or scientific evaluation ran for this
delivery. The numerical source/header, operation contract/RER and frozen
synthetic protocol, runner, configuration, results and source capsule were
unchanged. No historical result or artifact was overwritten. Earlier milestone
dates and their original verification records remain intact.

## Dashboard closeout

The final managed-container refresh passed all 34 dashboard tests and regenerated
the standalone HTML with the entire current toolset goal complete. The maintained
dataset validated 18 Heroes, 71 Goals and 113 Milestones, including 23 done and
one in progress. All evidence targets existed. The final link check verified
1,137 maintained Markdown links and 71 literal shell roots. All 18 tested source
pins still matched after closeout; the other five Hx06 goals and original callable
tool milestone were preserved. Schema version remains 2.

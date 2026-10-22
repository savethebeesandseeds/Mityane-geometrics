# Hx06 toolset surface review

**Date:** 20 October 2026.  
**Record:** `HX06-TOOLSET` revision `0.1.0`.  
**Decision:** complete for the toolset-surface definition milestone.

## Acceptance and source review

The [versioned surface map](../archive/TOOLSET-v0.1.0.md) was reviewed against the
[Hero milestone](../../../../web/dashboard/heroes/06-comparisons.json), the
[operation contract](../CONTRACT.md), linked [RER](../EVIDENCE.md), and the
[charter](../../../../doc/operations_over_representations_design_spec_v0.1.0.md)
sections 4, OR-06, 6 and 7. Independent read-only reviews covered implementation
accuracy and scientific scope. Three precision points were incorporated: exact
discovery fields, the C++-only `stop_after` argument, and the distinction between
native JSON outcomes and the separately retained bound request. No completion
blocker remains.

| Milestone requirement | Authored evidence |
| --- | --- |
| Versioned map links intended public capabilities to implementation or explicit gaps, contracts, schemas and claim levels | The map inventories public commands, Node/C++ APIs, schema definitions and output-schema gaps, the four selected delivery blocks, and separately unsupported observational/intervention profiles. |
| Preparation, checks, execution, composition and result reuse follow consistent conventions | The caller workflow binds explicit input records and byte inventories, performs check/run/inspect, retains complete pair states and source dependencies, and declares semantics and restrictions on downstream edges. |
| Provider/model responsibilities retain their owners and unsupported profiles are explicit | The ownership table separates data, encoding/fitting, construction, mappings and future Hx06 profiles. Only recorded controlled response is registered; the frozen synthetic study remains a study-specific interface. |

Source review checked the [registry](../../tools.mjs),
[runtime](../../../shared/representation/tools/v1/runtime.mjs),
[adapter](../tool/adapter.mjs), [request schema/codec](../tool/request.mjs),
[native bridge](../tool/native.cpp), [C++ header](../include/mityane/operations/counterfactuals/paired_response.h)
and existing [tool verification](tool-v1.md). The map distinguishes internal
adapter hooks from public application APIs and does not invent an
`operations.example()` method or a structured return from `inspect`.

## Executed interface inspection

The existing managed Linux container was used through the Windows launcher:

```powershell
.\code\operations\tools.ps1 list
.\code\operations\tools.ps1 describe OR06-PCR
```

Discovery returned only OR06-PCR `0.1.0`, family OR-06, with claim level
`controlled_model_response`. Description confirmed `model_execution: false`,
the existing describe/example/check/run/inspect commands, input-schema identity
`urn:mityane:operations:paired-response:input:1`, result version
`mityane.operation.result/v1`, and required envelope fields `schema_version`,
`operation`, `input`, `artifacts`. These read-only calls ran no comparison,
encoding, fitting or evaluation and published no study artifacts.

## Remaining delivery and limits

The next coverage increment is explicit request packaging, a recorded-alternative
workflow example, verified structured result reuse and published output schemas.
These are defined engineering gaps, not delivered features. Their CLI/API names
and any new schema identities remain proposals until implemented and verified.
The C++ bounded-execution argument, derivatives, normalized or aggregate
sensitivities, uncertainty estimators and new claim profiles are outside this
increment.

Surface completion establishes an interface plan. It supplies no new scientific
verdict, provider suitability result, causal identification or authority to
replay the frozen synthetic study. Existing engineering verification and the
completed evaluation retain their original scope and dates. Runtime sources,
operation contracts and frozen study files were not changed for this milestone.

## Integration verification

The mandatory dashboard refresh through the managed container passed all 34
focused tests and regenerated the standalone HTML: 18 Heroes, 71 Goals and 113
Milestones, with 22 done and one in progress. All evidence file targets existed.
The layout link check verified 1,113 maintained Markdown links and 71 literal
shell roots; historical artifact trees remain ignored. The working diff showed
no changes to operation runtime sources, contracts or frozen evaluation files.
No numerical tests or scientific studies were rerun for this documentation-only
surface definition.

# Hx06 reference implementation verification

**Date:** 19 October 2026.  
**Scope:** `or-06-comparisons-reference-implementation`, `OR06-PCR-CPP 0.1.0`.  
**Contract:** [OR06-PCR 0.1.0](../CONTRACT.md).  
**Implementation and limits:** [reference guide](../IMPLEMENTATION.md).

## Actual verification

The existing managed `mityane-geometrics` container was reused, immutable ID
`381c6b78f4fdb301cd132050ffd8095587c0794829a7a73a547fdbc0484630e1`.
Through `container.ps1`, the authoritative task entry ran:

```text
bash code/scripts/task.sh hx06-reference-001 test-counterfactuals
```

GNU C++ `12.2.0`, C++20, `-O1 -g0 -Wall -Wextra -Wpedantic` compiled the
standard-library implementation and test harness. The finalized run passed
**47 named cases, zero failures, with no compiler warnings**. The throwing-check
harness remains active independently of `NDEBUG`. No LibTorch, encoder, dataset,
provider, model fitting or GPU operation was used.

The source manifest was captured immediately before and after the run and
matched exactly. Its selected local records are retained under ignored
[reference-001](../../../../.temp/operations/counterfactuals/reference-001):
`conformance.log`, `compiler-version.txt`, `source-before.sha256` and
`source-after.sha256`. Public source review can use the digest table below;
the local logs need not exist in another checkout.

| Tested source | SHA256 |
| --- | --- |
| `include/mityane/operations/counterfactuals/paired_response.h` | `29b381bde5d4ac6f386457c69e5ade88f28b85575bd78bdeb840269f8d9a6e8a` |
| `src/paired_response.cpp` | `25b58ace6cce50a78455894a34faf6c16e206836321de9c14fafe34c64a2d1a0` |
| `tests/paired_response_test.cpp` | `3cac82588d2352047ce2113fcaa2cd64736af8993c8090ef1cf38c66f6ec49c4` |
| `counterfactuals.mk` | `bedb699f4a7f1b8c5949832c9004d2911e53fcfee8fa7dc5038c4e09877ae7b1` |
| Root `Makefile` at verification | `6360f0bdb3d5a53ce2bdd864280ff0c763571e8591facdc344dc7df17974db32` |
| Root `code/scripts/task.sh` | `c43a8bbd94dfd26ee3bf525b97776b2edc796d2ab9485fdc40b08bea8ac8e5b1` |

## Acceptance and independent review

- Conformance covers exact role/space/pipeline/dependency identity, malformed
  records, axes/units, scalar controls/bounds, validity/support refusals,
  upstream failures, finite contrasts including overflow, cutoff violations and
  complete denominator accounting.
- Results retain the original request with source/parent lineage,
  representation kinds, observed/edited/simulated/latent origins, model and
  fitted dependency pins, code/configuration, OperationRun and evidence IDs.
  No refusal becomes an ordinary zero vector or omitted row.
- Admission rejection is distinct from executed pair failure. Interrupted runs
  retain unresolved roster entries. Reused baselines with different alternatives
  remain possible; contradictory immutable metadata and duplicate tuples fail.
- Independent read-only review found and prompted fixes for conflicting
  immutable pins and repeated-record metadata contradictions. The final review
  reported no remaining correctness blockers for this bounded slice. Tests were
  authored independently from the implementation and then extended for those
  identified edge cases.

## Decision and remaining work

Integration verification also passed through the managed container:

- `bash code/scripts/node.sh code/scripts/check-layout.mjs --links-only`:
  1,008 maintained Markdown links and 70 literal shell roots verified;
  historical artifact trees remain ignored.
- `web/dashboard/scripts/refresh.ps1`: all 34 tests passed, all evidence file
  targets resolved, and standalone HTML regenerated from the current Hero
  records. Hx06 now has two done milestones and one proposed evaluation.

The reference implementation milestone is complete for the documented
deterministic recorded-response slice. Supplied provenance/check attestations
remain caller responsibilities. These fixtures establish engineering behavior;
they establish no real-data usefulness, counterfactual semantic validity,
population uncertainty, identified intervention effect or scientific quality.
The operation-specific evaluation milestone remains proposed and requires its
actual bindings, frozen protocol and execution admission.

# Shared code

This directory owns utilities used by multiple components. Follow
the logical records and compatibility rules in the
[project design specification](../../doc/operations_over_representations_design_spec_v0.1.0.md).

| Path | Responsibility |
| --- | --- |
| `settings/` | Owning settings/document parser and its syntax/provenance |
| `representation/` | Shared tensor types/operations, objectives, evaluation engines and reusable diagnostics |
| `artifacts/` | Versioned immutable sidecars and model/component/parent identity validation |

Observation archive code lives in [`../data/observations/`](../data/observations/),
and model-specific adapters in [`../embeddings/`](../embeddings/README.md).
Existing `embedding/shared/`, `cuwacunu/config/` and `cuwacunu/artifacts/`
include names/namespaces retain their API identities. Root Makefile include
roots express physical ownership without duplicating these implementations.

Keep model-specific behavior with its representation provider and
operation-specific behavior with its operation. Sharing dimensions does not
establish compatible spaces; preserve explicit encoder, preprocessing,
normalization and metric identities across component boundaries.

Artifact integrity hashes identify selected bytes and parent relationships;
they do not establish scientific validity or source authenticity. Preserve
historical source closures and receipts in
[the archive](../../.archive/embeddings/README.md). Run shared utility checks
through the managed container, root Makefile and `code/scripts/task.sh`.

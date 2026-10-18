# Data code

This directory owns reusable readers, observation/schema validation,
preprocessing and split definitions. Follow the [project design specification](../../doc/operations_over_representations_design_spec_v0.1.0.md)
for observation and representation provenance. Retain source identities, group
boundaries, event times and information-availability times when relevant.

| Path | Responsibility |
| --- | --- |
| `observations/` | Tensor observation archives, validation, smoke-data construction and related data tests |
| `market/versions/v1/` | Market timelines, chronological associations/arrays, risk-target definitions and ledger I/O |

The existing `embedding/shared/data.h` and `cuwacunu/latent_market/` include
names are retained as source compatibility boundaries. Physical ownership is
here; namespaces do not put these modules back under an encoder. Forecast
issuance belongs in [`../operations/forecasting/issuance/`](../operations/forecasting/issuance/).
Market data is one application lane, with no change to the domain-agnostic
project scope.

Store local dataset files under ignored [`.temp/`](../../doc/ENVIRONMENT.md#local-artifacts)
directories only when needed. Fit learned preprocessing using the declared
training split and retain its version and
training scope. Keep generated observations distinguishable from real source
cohorts. Imported historical artifacts retain their preserved locations under
`code/embeddings/output/`, `.build/` and `.external/`; source and evidence
snapshots live in [the archive](../../.archive/embeddings/README.md).

Build and test from the repository root through the managed container's root
Makefile and `code/scripts/task.sh`. Keep data admission, representation
fitting, forecasts and target scoring independently testable.

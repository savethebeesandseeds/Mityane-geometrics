# Representation providers and model evaluation

This directory owns the imported MTF, RPB and causal patch bottleneck encoders,
their model-specific training/evaluation, provider integration definitions, and
the separately versioned State Representation Network. The predecessor is
`cuwacunu_embedding`, with recorded upstream base revision
`b15702b29753188482b11ca85e997006b503d0f0`; migration file inventories identify
the actual imported bytes. Model family/version tags, native input/output
contracts and existing C++ include names remain unchanged. Maintained source
has new path-bound source identities after integration.

| Path | Responsibility |
| --- | --- |
| `encoders/` | Encoder families, provider definitions, adapters and model-owned workflows |
| `state_representation/` | Mapping observation embeddings to task-dependent latent states |
| `evaluation/` | Encoder/model protocols, cards, benchmarks and evaluation adapters |
| `scripts/` | Maintained model and evaluation runners |

Shared representation/evaluation utilities are in
[`../shared/representation/`](../shared/representation/), common settings in
[`../shared/settings/`](../shared/settings/), and immutable model metadata in
[`../shared/artifacts/`](../shared/artifacts/). Observation archives and market
timeline/array/target definitions live in [`../data/`](../data/README.md).
Latent dynamics, predictive training, direct-embedding forecast controls,
forecast issuance and their evaluation live in
[`../operations/forecasting/`](../operations/forecasting/).

The [project charter](../../doc/operations_over_representations_design_spec_v0.1.0.md)
distinguishes observation embeddings, learned latent states and forecast states.
Source placement does not establish prediction skill or downstream utility.
External/open models and API embeddings remain first-class representation sources.

## Development

From the repository root in PowerShell:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', '-j2', 'all', 'test')
.\container.ps1 -Action exec -Command @('/opt/mityane-geometrics/embeddings/build/baseline/embedding', '--help')
```

Inside the existing managed container, use `/mityane` as the source root:

```bash
cd /mityane
bash code/scripts/task.sh baseline -j2 all test
```

The root Makefile, task runner, dependency lock and setup are authoritative.
Named builds and locks use `/opt/mityane-geometrics/embeddings/build/<session>`;
new task results use `/mityane/.temp/embeddings/runs/<session>`. See the
[environment guide](../../doc/ENVIRONMENT.md) for the existing container and SDK.

## Research and preservation

Research remains paused at the saved 18 October 2026 handoff. Start with the
[current research guide](../../doc/EMBEDDINGS.md); archived next-step notes do
not authorize experiments or unfinished metadata application.

The complete pre-cleanup source/documentation component is retained byte-for-byte under
[`../../.archive/embeddings/pre-cleanup-2026-10-18/`](../../.archive/embeddings/pre-cleanup-2026-10-18/).
Historical ignored `output/`, `.build/` and `.external/` remain in this directory.
Frozen paths, source capsules, cards, checkpoints and receipts retain their
identities. Read the [archive guide](../../.archive/embeddings/README.md) before
following a historical source or reproduction pointer.

The repository [MIT license](../../LICENSE) covers the imported project code;
[third-party notices](../../THIRD_PARTY_NOTICES.md) retain upstream obligations.

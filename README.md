# Mityane geometrics

“Mityane” literally means “many things” in the Bora language.

Research and engineering on operations over learned and externally supplied
representations. The project asks which operations a representation supports,
under what conditions, and with what evidence. Its core research unit is an
operation, its input/output contract, required representation properties,
additional dependencies and an explicit test.

The [Operations over Representations design specification](doc/operations_over_representations_design_spec_v0.1.0.md)
is the project charter: `ORR-DESIGN-001`, version `0.1.0`. It defines thirteen
capability families with stable IDs, common contracts and evidence standards.
The families are composable areas of research; the catalogue does not establish
that each capability has been implemented or demonstrated.

## Research scope

| Area | Families | Questions and outputs |
| --- | --- | --- |
| Describe and organize observations | OR-01–OR-04 | Retrieval, structure and density, novelty, and distribution shift |
| Estimate outcomes and compare alternatives | OR-05–OR-06 | Target readouts, matched comparisons, sensitivity and explicitly scoped intervention claims |
| Transform representations and connect systems | OR-07–OR-10 | Latent transformations, compression, decoding and generation, and cross-space alignment |
| Model evolution, choose actions, and acquire evidence | OR-11–OR-13 | Latent dynamics, planning and control, and information acquisition |

Custom and frozen encoders, open/pretrained models and API-supplied embeddings
are representation sources. Vector, structured and probabilistic representations
fit the conceptual scope. Time series, text, images, audio and multimodal data
are application lanes; existing market research retains its original scope.

Representation suitability is specific to an operation, population, information
horizon and metric. Experiments need declared splits, meaningful baselines,
versioned geometry, failure conditions and reproducible evidence. Target
prediction, latent-state forecasting and decision quality need separate tests.
Observational comparisons, controlled model responses and identified causal
effects support different claims. Composed pipelines also need an end-to-end
evaluation.

## Structure

| Path | Purpose |
| --- | --- |
| `code/embeddings/` | Encoders, state representation models and model-owned evaluation |
| `code/operations/` | Operation implementations, contracts and evidence, organized by the capability map |
| `code/data/` | Data readers, validation and preprocessing code |
| `code/shared/` | Utilities shared by multiple components |
| `code/scripts/` | Environment checks and project task entry points |
| `config/` | Versioned configurations without credentials |
| `tests/` | Component and integration checks |
| `doc/` | Environment guide, design and research notes |
| `archive/embeddings/` | Unmodified predecessor source, research evidence and dated instructions |
| `web/dashboard/` | Standalone project dashboard; Hero records and branding are owner-defined |
| `data/` | Local datasets; contents excluded from Git |
| `output/` | Generated runs, checkpoints, embeddings and reports; excluded from Git |

## Current implementation

The imported `cuwacunu_embedding` implementation is organized by responsibility:
encoders and state representation models in [embeddings](code/embeddings/README.md),
latent forecasting and its evaluation in [operations](code/operations/README.md),
observation and market data handling in [data](code/data/README.md), and common
settings, tensor/evaluation utilities and artifact identities in
[shared](code/shared/README.md). Existing C++ include names, namespaces and model
version tags retain their identities.

Inherited implementations and engineering fixtures do not establish every
catalogue capability or scientific utility. Encoder research remains paused;
[the current research guide](doc/EMBEDDINGS.md) points to the preserved handoff
and evidence. The dashboard records 18 workstreams: thirteen
operation families, shared foundations and four encoder families. Proposed work
and documented outcomes have separate milestone states; the capability catalogue
does not establish implementation status.

## Project dashboard

Open [the dashboard](web/dashboard/public/index.html) directly in a browser.
Its [work map](web/dashboard/WORKSTREAMS.md) separates operation contracts,
implementations and evaluation from shared data/evidence and encoder work.
Each Hero has its own editable JSON file. The external Google multimodal
provider is documented; its API adapter and local evaluation remain proposed.
Hero records hold current priorities, dependencies and acceptance criteria;
documentation describes shared principles and component boundaries. Example
sequences in the charter do not add prerequisites to the current work plan.
See [the dashboard guide](web/dashboard/README.md) for authoring, branding and
container-only refresh commands.

## Development

Use Docker Desktop with Linux containers and NVIDIA GPU support. From PowerShell
in this directory:

```powershell
.\container.ps1 up
.\container.ps1 -Action exec -Command @('bash', 'setup.sh', '--no-download')
.\container.ps1 status
.\container.ps1 shell
```

`setup.sh` verifies exact dependency versions and the complete inherited LibTorch
SDK. A complete image needs no APT install or download. The managed container
reuses the pinned local research-environment snapshot; no Dockerfile or duplicate
SDK installation is needed.

The managed container is `mityane-geometrics`. It mounts only this project at
`/mityane`, has all NVIDIA GPUs available and publishes
`127.0.0.1:4874` to container `4874/tcp`. This port is reserved for future
application development; no application service starts during setup.

Builds, tests and services run inside the container. Store build products and
dependency caches under `/opt/mityane-geometrics`, and selected persistent
artifacts under `output/`. The inherited Linux SDK remains at
`/opt/cuwacunu_embedding/libtorch` (LibTorch `2.6.0+cu124`, C++11 ABI1).

See [the environment guide](doc/ENVIRONMENT.md) for the authoritative container
definition, image provenance and normal lifecycle commands. See
[source provenance](doc/PROVENANCE.md) before importing research code or artifacts.
The [initial verification receipt](doc/ENVIRONMENT_VERIFICATION_2026-10-18.json)
records the container identity, offline dependency checks and visible GPU.

## Build and test

From this repository root:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', '-j2', 'all', 'test')
.\container.ps1 -Action exec -Command @('/opt/mityane-geometrics/embeddings/build/baseline/embedding', '--help')
```

The repository-root Makefile and `code/scripts/task.sh` own current build/test
entry points. Each named session has a locked build directory under
`/opt/mityane-geometrics/embeddings/build/<session>` and writes new task results
to `output/embeddings/runs/<session>`. The retained `embeddings` artifact
namespace avoids disturbing existing task paths while source ownership is split.

The [archive guide](archive/embeddings/README.md) explains the byte-preserved
pre-cleanup snapshot and historical path identities. Historical ignored data,
checkpoints and transfer inputs remain at `code/embeddings/output/`, `.build/`
and `.external/`. The [initial migration record](doc/EMBEDDINGS_MIGRATION_2026-10-18.md)
describes the earlier import; its original layout is historical. Current
instructions are this guide, [the environment guide](doc/ENVIRONMENT.md) and
[the research guide](doc/EMBEDDINGS.md).

The [cleanup receipt](doc/EMBEDDINGS_CLEANUP_2026-10-18.json) links the original
file hashes, maintained path map and verification results.

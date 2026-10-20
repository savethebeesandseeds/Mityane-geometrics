# Mityane geometrics

“Mityane” literally means “many things” in the Bora language.

Building a composable foundation for machine intelligence in representation
space: discovery, prediction, generation, planning and action across models
and domains.

Mityane geometrics develops representations and the operations that make them
useful: discovering structure, comparing alternatives, transforming and
connecting latent spaces, predicting how systems evolve, generating
possibilities and choosing what to do or learn next. The ambition is to compose
these capabilities into intelligent systems that work across data modalities
and representation sources, from our own encoders to open models and external
embedding services.

The central research question is how the information and geometry in a
representation enable these capabilities, where they fail, and whether their
composition improves real tasks. Each operation has an explicit input/output
contract and tests of its required representation properties and dependencies.

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
| `doc/` | Environment guide, design and research notes |
| `.archive/embeddings/` | Unmodified predecessor source, research evidence and dated instructions |
| `web/dashboard/` | Standalone project dashboard; Hero records and branding are owner-defined |
| `.temp/` | Local datasets, generated runs, checkpoints, embeddings and recovery records; excluded from Git |

`code/data/` owns readers, validation and preprocessing source. Store local
dataset files under ignored `.temp/` directories only when needed.
Tests and configuration files live with their owning component.

## Current implementation

The imported `cuwacunu_embedding` implementation is organized by responsibility:
encoders and state representation models in [embeddings](code/embeddings/README.md),
latent forecasting and its evaluation in [operations](code/operations/README.md),
observation and market data handling in [data](code/data/README.md), and common
settings, tensor/evaluation utilities and artifact identities in
[shared](code/shared/README.md). Existing C++ include names, namespaces and model
version tags retain their identities.

The family-owned [live CPB ranking](code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md)
owns current checkpoint standings, including the new audited January scoped
records. The [current encoder research guide](doc/EMBEDDINGS.md) preserves
CPB-v4's historical broad clean-causal breakthrough, RPB-v24 native36's
historical balanced-leader role, V3's baseline role and later task-specific gains.
Independent V14 metadata review and compatible application are complete; the
guide links the actual records. V14 is the latest catalogued historical instance;
the separate V15 study supplies newer measurements. The owner approved the
[V15 objective comparison](code/embeddings/evaluation/protocols/cpb_v15_objective_v1/PROTOCOL.md)
on 19 October: three fresh V4/V15 pairs at native36, with V15 reusing V4's source
design and inheriting no trained weights. V5–V14 use native40. All six runs and
twelve independent arithmetic audits are complete and valid. The
[validated results](code/embeddings/evaluation/protocols/cpb_v15_objective_v1/RESULTS.md)
report a **mixed** prospective screen: February linear activity R² and descriptive
query MAE/NLL improve in all three pairs, while the volatility, trend-tolerance
and all-candidate timing requirements are unmet. Fresh V4-control records remain
distinct from V15 gains. This study does not support a broad encoder replacement
or Mityane selection. The
[compact study record](code/embeddings/evaluation/records/CPB-v15.objective-001.json)
binds the closeout evidence.

Inherited implementations and engineering fixtures do not establish every
catalogue capability. The dashboard records 18 workstreams: thirteen
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
SDK, and verifies/reuses the existing cached Node runtime for maintenance tools.
A complete image needs no APT install or download. The managed container
reuses the pinned local research-environment snapshot; no Dockerfile or duplicate
SDK installation is needed.

The managed container is `mityane-geometrics`. It mounts only this project at
`/mityane`, has all NVIDIA GPUs available and publishes
`127.0.0.1:4874` to container `4874/tcp`. This port is reserved for future
application development; no application service starts during setup.

Builds, tests and services run inside the container. Store build products and
dependency caches under `/opt/mityane-geometrics`, and selected persistent
artifacts under `.temp/`. The inherited Linux SDK remains at
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
to `.temp/embeddings/runs/<session>`. The retained `embeddings` artifact
namespace avoids disturbing existing task paths while source ownership is split.

The [archive guide](.archive/embeddings/README.md) explains the byte-preserved
pre-cleanup snapshot and historical path identities. Historical ignored data,
checkpoints and transfer inputs remain at `code/embeddings/output/`, `.build/`
and `.external/`. The [initial migration record](doc/EMBEDDINGS_MIGRATION_2026-10-18.md)
describes the earlier import; its original layout is historical. Current
instructions are this guide, [the environment guide](doc/ENVIRONMENT.md) and
[the research guide](doc/EMBEDDINGS.md).

Active maintenance uses pinned Node and a small C++ artwork helper. Run the
focused checks through the same container and task entry:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'maintenance', 'test-maintenance', 'test-dashboard-branding', 'check-layout')
```

The [language migration implementation](doc/PYTHON_MIGRATION_IMPLEMENTATION_2026-10-18.md)
records the complete source migration and the explicit scientific Python boundary.
Maintenance, preparation and record keeping use Node/Bash; independent scientific
checks retain Python where its established numerical behavior is valuable.
The [working-copy removal record](doc/PYTHON_MIGRATION_REMOVAL_2026-10-18.md)
records deletion of all 40 replaced Python files and the checks after cleanup;
40 scientific Python files remain in maintained source.
The [managed-container validation record](doc/PYTHON_MIGRATION_VERIFICATION_2026-10-18.md)
records the executed checks and their limits. The root `test-language-migration`
target checks the complete Node/source boundary, physical-path mapping and
retained artificial scientific fixtures. The separate
`test-migration-producer-compile` target compiles affected C++ translation units
without executing a research run. Earlier maintenance results remain in their
[dated record](doc/PYTHON_MIGRATION_2026-10-18.md).

The [cleanup receipt](doc/EMBEDDINGS_CLEANUP_2026-10-18.json) links the original
file hashes, maintained path map and verification results.

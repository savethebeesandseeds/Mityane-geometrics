# Mityane geometrics

Research on embeddings and their applications, continuing the work in
`cuwacunu/cuwacunu_embedding`. This repository starts with an isolated development
environment and room for existing and future embedding methods and consumers.
Models and application implementations will be imported or added explicitly.

## Structure

| Path | Purpose |
| --- | --- |
| `code/embeddings/` | Embedding adapters, methods and their input/output contracts |
| `code/applications/` | Applications that consume embeddings |
| `code/evaluation/` | Shared protocols, metrics and comparison tools |
| `code/data/` | Data readers, validation and preprocessing code |
| `code/shared/` | Utilities shared by multiple components |
| `code/scripts/` | Environment checks and project task entry points |
| `config/` | Versioned configurations without credentials |
| `tests/` | Component and integration checks |
| `doc/` | Environment guide, design and research notes |
| `data/` | Local datasets; contents excluded from Git |
| `output/` | Generated runs, checkpoints, embeddings and reports; excluded from Git |

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
SDK. A complete image needs no APT install or download. The new container reuses
the installed `cuwacunu_embedding` snapshot under its own local image tag; no
Dockerfile or duplicate SDK installation is needed.

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

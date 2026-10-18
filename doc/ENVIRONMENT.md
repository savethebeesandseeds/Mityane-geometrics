# Development environment

This project reuses the local `cuwacunu-dev:2026-10-12` image, which was made by
`docker commit cuwacunu_embedding cuwacunu-dev:2026-10-12`. Its source container
and immutable image ID are recorded in `environment-image.json`. A project tag,
`mityane-geometrics:embedding-2026-10-12`, names that same image. Docker shares
all existing layers: no dependency download, additional SDK copy or new base
image is required. The original container name is retained here as provenance;
the current project uses only the managed `mityane-geometrics` container.

The image contains installed tools and historical internal caches, but excludes
the source container's host bind-mounted research files. Neither sibling source
repository is mounted in this project's container.

## Authoritative container definition

| Setting | Value |
| --- | --- |
| Name | `mityane-geometrics` |
| Image | Local `mityane-geometrics:embedding-2026-10-12`, pinned ID in `environment-image.json` |
| Command | `/bin/bash`, interactive stdin and terminal |
| Working directory | `/mityane` |
| Bind mount | `C:\Work\Mityane-geometrics` to `/mityane`, read/write |
| Named volumes | None |
| Published port | `127.0.0.1:4874` to `4874/tcp` |
| GPU access | All NVIDIA GPUs through `--gpus all`; no other device mappings |
| Shared memory | 1 GiB |
| Restart policy | `no` |
| Labels | `io.waajacu.managed-by=mityane-geometrics`, project root, configuration `1` |

Port 4874 was checked against host listeners, Docker mappings (including stopped
containers), and Windows excluded TCP ranges before creation. The launcher checks
availability before creation or starting a stopped container. Only the published
mapping exists initially; there is no application server yet. Future services
must listen on `0.0.0.0:4874` inside the container to use the loopback host mapping.

`container.ps1` verifies the image ID, labels, command, mount, port, restart
policy and GPU configuration before acting. It reuses a matching stopped or
running container. Conflicts are preserved and reported. It never pulls an image,
deletes a container or modifies a volume.

## Normal operations

```powershell
.\container.ps1 status
.\container.ps1 up
.\container.ps1 -Action exec -Command @('bash', 'setup.sh', '--no-download')
.\container.ps1 -Action exec -Command @('bash', '-lc', 'nvcc --version')
.\container.ps1 shell
.\container.ps1 stop
```

If local script policy blocks the launcher, use a process-scoped invocation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\container.ps1 status
```

`setup.sh` is adapted from the existing offline snapshot setup. It verifies every
entry of `dependencies.lock` exactly. Missing packages can only be installed from
the existing cache using `--no-download --no-remove --no-install-recommends`.
Version drift or missing cached packages stops setup for inspection. A complete
snapshot skips APT entirely. The optional `--no-download` flag documents the
default behavior.

The SDK verifier checks the original immutable proof, every installed SDK member,
version/ABI and runtime profile, without reading a sibling workspace or changing
the SDK. Its historical `/embedding` source path is an evidence identity only.
Setup checks library resolution, the CUDA toolkit and actual GPU visibility and
writes a package inventory under `/opt/mityane-geometrics/setup`.

The source image retains the SDK at `/opt/cuwacunu_embedding/libtorch` and CUDA
12.4 under `/usr/local/cuda-12.4`. Login and interactive shells load the inherited
`/etc/profile.d/embedding.sh`; direct commands can use `bash -lc` or absolute
paths. Keep build artifacts, temporary files, toolchains and caches on the
container filesystem under `/opt/mityane-geometrics`. Persist selected research
artifacts under the ignored `.temp/` host directory. Create dataset
subdirectories there only when needed; `code/data/` contains source code.

## Local artifacts

The repository's local artifact directory is `.temp/`, mounted at
`/mityane/.temp`. Its complete contents are ignored by Git. It holds named run
results, selected datasets, checkpoints, branding exports, maintenance files and
migration recovery history. Preserve existing evidence and recovery files.

This directory was renamed from root `output/` without changing the retained
file contents. Dated receipts keep their original `output/...` path identities;
those root-relative local files now live at `.temp/...`. Imported artifacts under
`code/embeddings/output/`, `.build/` and `.external/` stay in their original
locations. Build caches and build temporary files remain under
`/opt/mityane-geometrics` inside the container.

## Embeddings component

The former standalone embedding project is integrated across `code/embeddings/`,
`code/operations/forecasting/`, `code/data/` and `code/shared/`. The same
container, pinned image, single bind mount, GPU access and port mapping are
reused. The root `container.ps1`, `setup.sh`, `dependencies.lock` and Makefile
own the environment and build. Component container/setup/lock duplicates have
been removed. Run from `/mityane` through the root task entry:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', '-j2', 'all', 'test')
```

New component builds, temporary files and task locks live under
`/opt/mityane-geometrics/embeddings/build/<session>`. Named task outputs live
under `/mityane/.temp/embeddings/runs/<session>`. Imported historical artifacts
remain in the component's ignored `output/`, `.build/` and `.external/`
directories. Prefer named tasks; direct research runners require their own
declared output path and frozen procedure before research. The complete
pre-cleanup source/documentation component is preserved in
`.archive/embeddings/pre-cleanup-2026-10-18/`. Historical source capsules and
receipts retain their original bytes, paths and hashes. Those identities are
evidence, not paths to rewrite or a request to replay old experiments. Read
[the archive guide](../.archive/embeddings/README.md) for source ownership and
historical path mapping, and [the research guide](EMBEDDINGS.md) for pause status.

The root archive directory is now `.archive/`. Dated receipts retain their
original `archive/...` paths and hashes; map that leading root segment to
`.archive/...` when locating the same files. Nested `archive/` directories
inside a preserved snapshot keep their original names. The layout verifier
resolves this root mapping without rewriting the receipts.

The imported archived headers exceed Windows' legacy path length under the
component directory. This checkout uses repository-local `core.longpaths=true`;
set the same local Git option on another Windows checkout before adding these
paths. No global Git setting or hosting identity is changed.

## Optional dashboard runtime

The local dashboard opens directly as `web/dashboard/public/index.html`, without
a service. Refreshes run through `container.ps1` inside this same container.
The dashboard's dependency-only `web/dashboard/scripts/setup.sh` verifies and
installs pinned Linux Node 22.23.3 from an existing exact archive under
`/opt/mityane-geometrics/web/downloads/`. It performs no downloads or APT changes.
The runtime, upstream license, refresh lock and temporary files stay under
`/opt/mityane-geometrics/web/`. The base image and root dependency lock are
unchanged. See [the dashboard guide](../web/dashboard/README.md) for the checksum,
cache restoration details and refresh command.

## Restoring the local image

The launcher requires the exact recorded local image and never downloads a
replacement. If the project tag is absent but the original snapshot still has
the recorded ID, restore only the alias:

```powershell
docker image inspect cuwacunu-dev:2026-10-12 --format '{{.Id}}'
# Compare with environment-image.json before tagging.
docker tag cuwacunu-dev:2026-10-12 mityane-geometrics:embedding-2026-10-12
```

For transfer to another machine, save/load the recorded image with Docker's
image export/import tools, keeping the third-party binary licenses. Do not
publish the environment image to a registry without separate authorization.

# Project scope

Mityane geometrics extends the research in `cuwacunu/cuwacunu_embedding` to
embedding methods and their applications. Keep reusable embedding contracts,
application code and evaluation protocols distinct. Import any model, dataset
or research implementation explicitly, with its source revision, license,
configuration and artifact identity. Preserve original research evidence.

# Development environment

Read `README.md` and `doc/ENVIRONMENT.md` before changing the environment.
Develop, build, test and run project code exclusively in the managed Linux
container `mityane-geometrics`, through `container.ps1`. Do not install or run
Windows toolchains, Java, Python environments or project services on the host.
Host PowerShell is limited to orchestration, file inspection and OS integration.

Use `setup.sh` only for dependency verification, cached exact dependency
installation and environment configuration. It must not manage containers,
build, test, train or start services. Keep lifecycle operations in
`container.ps1`. No Dockerfile is used. APT installation must retain
`--no-install-recommends`, `--no-download` and `--no-remove`.

The environment image is pinned in `environment-image.json` and reuses the
existing embedding snapshot. Never silently pull or substitute an image.
Inspect and reuse a matching existing container, including a stopped one.
Preserve unmanaged or mismatched containers, all volumes and existing data.
Container replacement or data deletion needs explicit authorization.

Build products, temporary files and dependency caches go under
`/opt/mityane-geometrics` inside the container. The inherited LibTorch SDK stays
at `/opt/cuwacunu_embedding/libtorch`; its original verification records bind
that path. Do not relocate, duplicate or modify the SDK. Persist selected data,
checkpoints and reports under this repository's ignored `data/` and `output/`.
Use named directories for concurrent work and account for shared GPU capacity.

# Research and publication

Keep fitted preprocessing tied to its training split. Record model/checkpoint
identity, native embedding shape, masks, data splits, seeds, environment and
evaluation protocol with every result. Preserve failures and earlier references.
Do not claim inherited dependencies establish model or application quality.
Verify actual CUDA execution before substantial GPU work.

The GitHub repository is private. Do not make it public, publish container
images or include credentials, raw datasets, checkpoints or generated output
in Git without explicit authorization.

# Project scope

Mityane geometrics studies operations over learned and externally supplied
representations. Read `doc/operations_over_representations_design_spec_v0.1.0.md`
(`ORR-DESIGN-001`, v0.1.0) as the project charter before changing scope, contracts
or evaluation policy. The research unit is an operation, its input/output
contract, required representation properties, dependencies and explicit test.

Refer to Heroes consistently as Hx01 through Hx18 (Hx plus the zero-padded
Hero number) in the dashboard and maintained prose. Keep Hero titles descriptive;
derive the Hx reference from `number`. Preserve existing JSON IDs, file paths,
dependencies and historical evidence identities. The mapping in
`web/dashboard/WORKSTREAMS.md` distinguishes Heroes from capability families.

Keep the OR-01 through OR-13 capability IDs stable. The families are composable
research categories, not a claim of implementation, a sequential roadmap or
authorization to start an experiment. Custom/frozen encoders and external/open
models or API embeddings are first-class representation sources. Keep the
project domain-agnostic; market/time-series research is one application lane.

Keep representation providers and model-owned research in `code/embeddings/`,
operation implementations and their protocols/evidence in `code/operations/`,
data code in `code/data/`, and cross-component utilities in `code/shared/`.
Use the existing flat ownership boundaries: embeddings `encoders/`,
`state_representation/` and `evaluation/`; operation forecasting modules and
their `evaluation/`; data `observations/` and `market/`; shared `settings/`,
`representation/` and `artifacts/`. Do not recreate the imported nested
`code/embeddings/code/` tree or component-level environment/build configuration.
Keep model-specific runners in `code/embeddings/scripts/`.
Keep tests and versioned non-secret configurations with their owning component.
Credentials belong in ignored environment files or external secret storage.
All encoder Heroes Hx15–Hx18 develop reusable encoder/provider components and
tools. Follow `code/embeddings/encoders/AGENTS.md`: separate the family/provider,
supported configuration, training/request recipe and each trained instance or
provider result. Research checkpoints and benchmark shapes are scoped evidence,
not the family's complete supported interface. Locally trainable families need
portable construction, TRAIN fitting, training and checkpoint/resume surfaces
with explicit coverage/gaps; external providers retain their actual upstream
capabilities. Application projects such as Cuwacunu own substantial application
training and final docking. Mityane retains bounded architecture research and
tool verification; this direction does not start an experimental execution.
Each serious operation needs an Operation Contract and a linked
Requirements-and-Evidence Record as defined by the charter. Record which
operation, population, horizon, representation version and geometry a result
supports; include baselines, failure/abstention conditions and limitations.

Keep current priorities, dependencies, acceptance criteria and work states in
the Hero JSONs. Documentation defines shared principles and component boundaries;
illustrative plans are not additional execution prerequisites. Reusable contracts
and independent engineering work may proceed before an experimental instance is
selected. Each experimental execution still needs its actual bindings, protocol
and permissions; a proposed milestone or diagnostic label grants no authority.

Distinguish observation embeddings, task-dependent latent states and predicted
future states. Keep target readouts, latent dynamics and decision policies
independently testable. Distinguish observed comparisons, controlled model
responses and identified intervention effects. Test composed pipeline claims
end to end; component tests alone do not establish downstream utility.

Import any model, dataset or research implementation explicitly, with its source
revision, license, configuration and artifact identity. Preserve original
research evidence, model IDs, frozen protocols and hash-bound records.

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
checkpoints and reports under this repository's ignored `.temp/`.
Keep data-handling source in `code/data/`; create local dataset directories
under `.temp/` only when needed. This directory also preserves migration recovery
history and evidence; its name does not authorize deleting its contents.
The imported component's ignored `output/`, `.build/` and `.external/` retain
historical files in place; do not delete or rewrite them as cleanup. New named
embedding tasks write to `/opt/mityane-geometrics/embeddings/build` and
`/mityane/.temp/embeddings/runs`. The root Makefile and `code/scripts/task.sh`
are the build/test entries; `container.ps1`, `setup.sh` and `dependencies.lock`
at the root are the authoritative shared environment configuration. Never recreate the
retired standalone container.
Use named directories for concurrent work and account for shared GPU capacity.

Read `doc/EMBEDDINGS.md` for inherited research status and
`.archive/embeddings/README.md` for the preserved evidence boundary. The complete
pre-cleanup source/documentation component is frozen at
`.archive/embeddings/pre-cleanup-2026-10-18/`, including source capsules, dated
instructions, original configurations and research documents. Do not rewrite
its bytes or treat archived instructions as current lifecycle commands.
Original paths and hashes remain evidence identities. A historical reproduction
needs an explicit path mapping and the relevant frozen admission/protocol;
maintenance does not resume paused experiments, metadata application or training.
The imported 18 October pause is a historical boundary. Subsequent authorized
V14 metadata application and V15/V16 studies are complete. Read `doc/EMBEDDINGS.md`
and the family-owned `research-reference.json` for the current V16 research
reference chosen on 22 October; preserve its mixed study verdict and limitations.
Further experimental execution still needs its actual scope, bindings, protocol
and permissions. Relocated or edited source has a new path-bound source identity; never
substitute it for the source recorded by a historical receipt.

# Project dashboard

The shared dashboard backbone is in `web/dashboard/`. The owner defines its
Heroes and branding; do not seed guessed Heroes, Goals or Milestones. Read
`web/dashboard/SCHEMA.md` before progress edits. Keep one JSON per Hero in
`heroes/`, listed in `manifest.json`, with the fixed Hero -> Goal -> Milestone
hierarchy. Record only confirmed scope and evidence; only Milestones store states.

After changing Hero JSON, the manifest, branding or interface, run
`./web/dashboard/scripts/refresh.ps1` from the repository root. It delegates
through `container.ps1` to the existing managed container, runs focused tests and
regenerates `public/index.html`. Do not hand-edit that generated file or run a
host toolchain. The finished HTML opens directly without a server. Use `-Check`
to check freshness without rewriting it. Preserve the imported AGPL notices.

# Research and publication

Keep fitted preprocessing tied to its training split. Record model/checkpoint
identity, native embedding shape, masks, data splits, seeds, environment and
evaluation protocol with every result. Preserve failures and earlier references.
For inherited encoder evaluations, keep preprocessing TRAIN-only and retain
the frozen heads and protection floors. Do not replay completed research
experiments solely to reproduce an existing score.
Do not claim inherited dependencies establish model or application quality.
Verify actual CUDA execution before substantial GPU work.

The GitHub repository is public. Do not change its visibility or publish
container images without explicit authorization. Public source availability
does not authorize including credentials, raw datasets, checkpoints or generated
output in Git. Preserve dataset and model access, retention and license limits.

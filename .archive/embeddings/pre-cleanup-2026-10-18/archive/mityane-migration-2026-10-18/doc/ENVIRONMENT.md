# Shared embedding development environment

Historical causal experiments used verified source capsules in an existing
GPU-enabled environment. Their frozen cards and execution receipts retain the
exact environment, input and output bindings. Preserve those records and follow
the experiment-specific constraints when reproducing evidence. The general
standalone launcher below does not replace a frozen experiment's procedure.

This repository uses the existing managed `cuwacunu_embedding` Linux container
for compilation, tests and training. Named task sessions keep their build
products separate while sharing the source tree and installed dependencies.
Encoder implementations are isolated under `code/encoders/`, while reusable code
lives under `code/shared/`. Documentation is in `doc/`, and persisted generated
artifacts are in `output/`. Both the baseline and RPB-MAE have explicit build,
test and workflow targets in the same managed container. RPB-v20 additionally
has explicit configurable-API engineering targets; it does not replace either
historical CLI. Read [current status](CURRENT_STATUS.md) before choosing an
experiment or interpreting a dated command below.

## Start and run tasks

From PowerShell in the repository root, use the authoritative launcher:

```powershell
.\container.ps1 status
.\container.ps1 up
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', '-j2', 'all', 'test', 'baseline')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', 'smoke-cuda')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'docs-v20', '-j2', 'test-rpb-v20')
```

`up` reuses the matching container, including when it is stopped. The task
runner does not create, stop or replace containers and does not install packages.
Run `setup.sh` only when preparing dependencies according to the [root README](../README.md).
Development toolchains run exclusively inside the Linux container.

Inside the container, the same commands are:

```bash
cd /embedding
bash code/scripts/task.sh baseline -j2 all test baseline
bash code/scripts/task.sh baseline smoke-cuda
bash code/scripts/task.sh baseline evaluate
bash code/scripts/task.sh rpb-mae -j2 rpb-mae test-rpb-mae
bash code/scripts/task.sh rpb-mae smoke-rpb-mae-cuda
bash code/scripts/task.sh docs-v20 -j2 test-rpb-v20
bash code/scripts/task.sh comparison -j2 evaluation feature-harness test-feature-harness
bash code/scripts/task.sh comparison evaluate-minimum
bash code/scripts/task.sh comparison evaluate-rpb-mae
bash code/scripts/task.sh rpb-native-curve -j2 native-curve test-native-curve test-rpb-native-gate
bash code/scripts/task.sh rpb-native-curve evaluate-native-curve
```

The first argument is a session name: 1 to 64 lowercase letters, digits,
underscores or hyphens, beginning with a letter or digit. Use a distinct name
for another concurrent session. Every task in one session holds a nonblocking
file lock; an overlapping invocation reports that the session is busy. Use this
runner consistently for concurrent work. Calling `make` directly bypasses the
lock, although its default build directory is still inside the container.

The session name selects isolation, not an architecture. `all`/`test` select the
baseline; `rpb-mae`/`test-rpb-mae` select the new encoder, and `feature-harness`
builds a minimum comparison driver that registers only the baseline adapter.
`evaluation` builds the separate full driver with baseline and RPB adapters.
`rpb-v20` builds four configurable config/model/checkpoint/integrity check
executables; `test-rpb-v20` runs them. They are engineering targets, not a general
v20 training/serving CLI. Experiment-specific producers and saved-feature readers
live under `code/evaluation/protocols/`; see the
[capacity protocol guide](../code/evaluation/protocols/configurable_capacity_v1/README.md)
and its [completed report](CONFIGURABLE_CAPACITY_V1.md). Reuse saved results for
reporting instead of rerunning completed trajectories.
The corresponding executables are `embedding`, `embedding_raw_patch_bottleneck_mae`,
`feature_harness` and `embedding_evaluate` under the selected build directory.
`test-feature-evaluation` uses dummy providers to test the engine independently;
`test-feature-harness` also includes those checks. Give concurrent manual CLI
jobs separate inputs/output destinations as well.

## Build and data locations

| Location | Purpose |
| --- | --- |
| `/opt/cuwacunu_embedding/build/<session>` | CLI executable and task lock inside the container |
| `/opt/cuwacunu_embedding/build/<session>/code` | Objects, dependency files and test binaries following the source layout |
| `/opt/cuwacunu_embedding/build/<session>/tmp` | Temporary files for task processes and C++ tests |
| `/opt/cuwacunu_embedding/setup` | Future installer downloads and package inventory |
| `/opt/cuwacunu_embedding/libtorch` | Verified pinned Linux SDK and runtime inside the container |
| `/embedding/output/runs/<session>` | Persisted smoke and evaluation results in the project bind mount |
| `/embedding/.external/libtorch` | Preserved Linux bundle transfer input; setup copies it into the container |
| `/embedding/.build/reference` | Existing reference header for the optional extraction audit |

Smoke and evaluation runners allocate a new directory for each invocation and
print its path. Repeated runs retain earlier datasets, checkpoints, exports and
reports. The periodic interruption test uses its own temporary directory and
removes only that test's files after terminating its own training child.

Existing `.build`, `.external` and `output` contents are retained. In particular,
`.build` contains dependency-installation evidence as well as old binaries, and
`output` contains existing checkpoints and datasets. Build products in the
container survive a normal stop/start; removing or replacing the container would
remove its writable-layer build state. Container removal is a separate action.

The Makefile accepts `BUILD_DIR`, `LIBTORCH`, `REFERENCE_DIR` and `RUN_ROOT` for
explicit environments. `LIBTORCH` defaults to `/opt/cuwacunu_embedding/libtorch`;
an override must also identify a container filesystem or named-volume SDK,
following the container-only development rule. `setup.sh` installs the pinned
Python dependency before invoking `code/scripts/install-libtorch.py`, which
verifies the staged bundle, copies it into an exclusive destination and records
every file hash under a fresh `setup/sdk-*` directory. An existing complete copy
is verified and reused; a partial or conflicting destination is preserved and
rejected. No container, volume or host bundle is removed or replaced.
The task runner fixes its own build and run paths so
they match the session lock. Binaries use the selected LibTorch directory as
their runtime library path. Shell scripts and the Makefile have LF checkout
rules in `.gitattributes`.

## Run the built CLI

For the `baseline` session, the executable is
`/opt/cuwacunu_embedding/build/baseline/embedding`. Choose explicit destination
paths for manual CLI commands; named-session locking covers Make tasks rather
than arbitrary CLI processes.

To evaluate with a custom configuration and retain a unique output directory:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/encoders/mtf_jepa_mae_vicreg/scripts/evaluate.sh', '--config', 'code/encoders/mtf_jepa_mae_vicreg/config/evaluation.conf', '--seeds', '101', '--steps', '10')
```

The evaluation runner defaults to the baseline executable and run root.
`EMBEDDING_BIN` and `EMBEDDING_RUN_ROOT` select another already-built session.
It also accepts checkpoint evaluation arguments. The runner assigns `--output`;
choose its parent through `EMBEDDING_RUN_ROOT` instead.

The `code/scripts/evaluate-minimum.sh` runner selects `feature_harness`, while
`evaluate-rpb-mae.sh` selects `embedding_evaluate`. Make tasks pass the current
session's binary; direct runner calls use `EMBEDDING_BIN` for the minimum driver
and `EVALUATION_BIN` for the full driver. Both use `EMBEDDING_RUN_ROOT`, allocate
a fresh parent and pass a new `results` child to the evaluator. They save the
instantiated card, `report.json`, per-task inputs and fitted/adapter assets.
`--baseline-steps N` adds a newly trained, then frozen baseline control;
`--rpb-steps N` adds a freshly trained RPB control to the full driver. Neither
option reuses an unverified historical checkpoint/scaler association.

The full CLI accepts `--encoders baseline,rpb` and explicit comparison pairs.
Cards own geometry, channel metadata, tasks, splits and matched widths; encoder
configurations must agree. The RPB prepare/train/embed executable does not link
the evaluator. Shared fitting/scoring and dummy-provider tests compile without
encoder include paths. See [evaluation integration](../code/evaluation/README.md)
for commands and the development-only contract, and the
[implementation review](EVALUATION_IMPLEMENTATION_REVIEW.md) for policy gaps.

## Coordinate source changes

Build sessions isolate objects and binaries, not source files. Encoder-owned
source, headers, configuration and C++ tests live under
`code/encoders/mtf_jepa_mae_vicreg/`; the RPB-MAE implementation has its own folder at
`code/encoders/raw_patch_bottleneck_mae/`.
Architecture-independent code and tests live under `code/shared/`. The evaluator
registry, CLI and protocol cards live under `code/evaluation/`; exact model state,
preprocessing and extraction stay in each encoder's evaluation adapter.

When multiple chats work concurrently, agree on ownership of encoder source
and measured results. Shared code, the Makefile, common evaluation, setup/runners
and repository documentation affect every encoder; coordinate those changes
between active chats. During a structural refactor, pause source edits and builds
until the validating session hands the paths back. Already-built jobs can
continue using their private inputs and output destinations. A dated claim that
another chat is stopped is not an ownership guarantee for later work. Check
current activity and preserve concurrent edits; do not take over shared files
based on a historical session note.

All sessions share the same container's CPU, memory and GPU capacity. Keep build
parallelism modest while another session compiles, and coordinate GPU runs when
their combined memory requirements are unknown. The existing container
definition, pinned packages and library bundle remain the authoritative
environment; no additional container is needed for named build sessions.

## Fresh decoder replication

Use the existing managed GPU container and the named `rpb-paired-pooling` build
session for the frozen fresh-source v4/v7 comparison:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/check-fresh-decoder-replication.sh')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-paired-pooling', 'evaluate-fresh-decoder-replication')
```

The admission script owns its task calls; do not nest it inside `task.sh`.
Quality generation requires matching source/card/binary hashes and a passed
actual-CUDA admission. Every invocation creates a new output capsule, with no
historical quality inputs or recipe overrides. Inspect and reuse completed
saved evidence for reporting instead of running this command to repeat scores.
Encoder/decoder optimization and needed model inference use CUDA. Fixed heads,
raw PCA and saved-arithmetic verification run separately on CPU in this same
container. See the [evaluation instructions](../code/evaluation/README.md).

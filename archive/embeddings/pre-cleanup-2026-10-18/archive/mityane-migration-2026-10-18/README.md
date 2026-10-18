Current causal research: **CPB-v3.market-01 remains the clean from-scratch baseline.** The new [CPB-v4.market-01 results](doc/CAUSAL_FRESH_V4_RESULTS.md) pass19/22 conditions: stronger trend/relation and hard timing, with linear volatility/activity and TRAIN MAE regressions. See [review and next direction](doc/CAUSAL_FRESH_V4_REVIEW.md), [live versions](code/encoders/causal_patch_bottleneck_mae/versions/current.json) and [current status](doc/CURRENT_STATUS.md). V1/v2 remain pretrained historical references without oracle floors. V3 publication was completed earlier; this new research remains local with weights ignored. Historical selections below are milestone records.

# cuwacunu_embedding

Standalone C++20/LibTorch representation models and independent evaluation
research. The retained MTF-JEPA-MAE-VICReg encoder and the RPB-MAE design family
share tensor/archive utilities, core settings and frozen-feature evaluation.
Research includes synthetic signals and historical market signals. Reusable
encoder contracts and measured experiments are documented within this repository.

Start with [current status](doc/CURRENT_STATUS.md) and the
[documentation index](doc/README.md). They distinguish implemented APIs,
completed measurements, preserved references and proposed work.
The [RPB-v22 encoder guide](doc/RPB_V22_ENCODER_GUIDE.md) explains the input,
output, architecture and training in plain words.

| Track | What is available | Guide |
| --- | --- | --- |
| RPB-v22.alt-01 | Content-only relation API; both completed development stages pass | [Encoder guide](doc/RPB_V22_ENCODER_GUIDE.md) and [measured report](doc/CONFIGURABLE_CONTENT_RELATION_V1.md) |
| RPB-v21.alt-01 | Stopped candidate: learned timing improves, but extra-deletion protection fails | [Preserved report](doc/CONFIGURABLE_LEARNED_RELATION_V1.md) |
| RPB-v20 | Preserved configurable spectral API and three-profile TEMPO-5 capacity record | [API](code/encoders/raw_patch_bottleneck_mae/CONFIGURABLE_SPECTRAL.md) and [capacity report](doc/CONFIGURABLE_CAPACITY_V1.md) |
| RPB-v19.alt-01 | Preserved confirmation of unchanged v19 on fresh synthetic timing and component-balance sources | [Confirmation](doc/SPECTRAL_CONFIRMATION_V1.md) |
| RPB-v4 / RPB-v7 | Formal evaluated reference / preserved original five-instance bundle | [Versions and instances](doc/EMBEDDING_VERSIONS.md) |
| MTF-v1 | Retained multiscale JEPA, reconstruction and VICReg encoder; historical CLI | [Baseline guide](code/encoders/mtf_jepa_mae_vicreg/README.md) |

RPB-v20 and the separate RPB-v22 API accept values/masks `[B,C,H,F]` and serve
native `[B,D]`. Both settings default to C3/H128/F9/P8/D32. Geometry and semantic IDs are fixed
per checkpoint; B can vary. H must be at least 17, may be nondivisible by P,
and D must exceed four times the semantic channel-pair count. All F features
reach the learned path and fixed spectral bank. Execution requires CUDA float32.
Use TRAIN-fitted scaling and the explicit model-only
[v20 settings](code/encoders/raw_patch_bottleneck_mae/config/configurable_spectral.conf)
or [v22 settings](code/encoders/raw_patch_bottleneck_mae/config/content_ordered_relation.conf)
for the corresponding API.

The `rpb-v20` target builds engineering check executables and `test-rpb-v20`
runs them. RPB-v20 is a C++ API; it is not selected by the historical
`embedding_raw_patch_bottleneck_mae` CLI or the MTF `embedding` CLI.
Its recorded config/model/checkpoint/integrity checks passed in the managed
container. The preserved capacity study retains compact RPB-v20.alt-01
(W64/D32); larger profiles show no consistent final classification benefit at
its fixed budget. Original v20 measurements, settings and checkpoints remain preserved.
Historical standalone CLI defaults stay fixed.

The latest completed RPB-v22 content-only relation experiment passes its two-cohort
screen and third development check, retaining learned timing gains and recovering
missing-value protection. Its API keeps the same configurable dimensions,
scaler, masks and decoder while excluding additive observation-density fractions
from the relation projection. See the [content module](code/encoders/raw_patch_bottleneck_mae/CONTENT_ORDERED_RELATION.md)
and [continuation](doc/CONTINUATION_2026-10-13_AFTER_CONTENT_RELATION.md).
The `rpb-v22` and `test-rpb-v22` targets build/run its separate artificial CUDA
engineering check, rather than selecting it through the historical CLI.
These results use known development data; unchanged-v22 fresh synthetic
confirmation is proposed, and market quality remains unmeasured in that study.
The stopped RPB-v21 recipe and all of its failed-stage evidence remain preserved.

RPB-v19's confirmed scores apply to its declared synthetic datasets and fixed
geometry. They do not measure the separate v20 profiles. Market quality remains
unmeasured. No reference promotion
or historical default change occurred. The harmonic-decoder continuation is a
preserved proposal; [current status](doc/CURRENT_STATUS.md) records the open work.

Preserve original model instances, scalers, fitted heads, frozen cards and reports.
Use native exports without PCA afterward; standalone raw PCA remains a baseline.
See the [evaluation guide](code/evaluation/README.md),
[evaluation policy](doc/EMBEDDING_EVALUATION_POLICY.md) and
[reporting standard](doc/RESULTS_REPORTING_STANDARD.md).

## Start and build

From PowerShell in this directory:

```powershell
.\container.ps1 up
.\container.ps1 exec bash setup.sh
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'docs-v20', '-j2', 'test-rpb-v20')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', '-j2', 'all', 'test')
.\container.ps1 -Action exec -Command @('/opt/cuwacunu_embedding/build/baseline/embedding', '--help')
```

The managed container is `cuwacunu_embedding`, using the pinned Debian 12 image
digest in `container.ps1`, command `/bin/bash`, and this directory mounted at
`/embedding`. It has all GPUs available, 1 GiB shared memory, no published ports,
no named volumes, and restart policy `no`. Docker Desktop must have Linux
containers and NVIDIA GPU support available. Existing matching containers are
reused, including stopped ones; a conflicting container is preserved and rejected.

`setup.sh` installs the pinned Debian and NVIDIA dependency versions in
`dependencies.lock` with `--no-install-recommends` and checks LibTorch dependencies.
Its installation steps provide C++ build/debug tools (GCC, GDB, Valgrind, ccache, mold, clang-format), Git/cURL/shell tools,
CUDA toolkit 12.4.1 and cuDNN 9 from NVIDIA's Debian 12 repository.
It does not build, train, or manage containers. No Dockerfile is used.
`container.ps1 status`, `stop`, and `shell` provide the other lifecycle operations.
After entering the shell, commands below run directly from `/embedding`.
For arguments that PowerShell treats as its own options, pass an explicit array:
`./container.ps1 -Action exec -Command @('make', '-j12')`.

Build products and task temporary files live inside the container under
`/opt/cuwacunu_embedding/build/<session>`. Objects and test binaries follow the
source layout under its `code/` subdirectory; the CLI remains at
`/opt/cuwacunu_embedding/build/<session>/embedding`. Use `bash code/scripts/task.sh baseline`
followed by make arguments for this model; another session name isolates its
objects and executable. A nonblocking same-session lock refuses overlapping
tasks. Source files and GPU capacity are shared. Generated data, reports,
checkpoints and exports persist under `output/runs/<session>`; smoke and
evaluation runners create a fresh subdirectory for each invocation. Existing
`.build`, `.external` and `output` contents are retained. `.external/libtorch` is
the staged transfer input; the installed SDK and runtime live inside the
container at `/opt/cuwacunu_embedding/libtorch`. `.build/reference` is the optional
read-only extraction-audit input. See [environment coordination](doc/ENVIRONMENT.md) for details.

Choose the model with an explicit Make target; session names only select build
and output directories. Inside the container, these can run in distinct chats:

```bash
bash code/scripts/task.sh baseline -j2 all test
bash code/scripts/task.sh rpb-mae -j2 rpb-mae test-rpb-mae
bash code/scripts/task.sh rpb-mae smoke-rpb-mae-cuda
bash code/scripts/task.sh comparison -j2 evaluation feature-harness test-feature-harness
```

The RPB binary is `embedding_raw_patch_bottleneck_mae`. The `evaluation` target
builds `embedding_evaluate`, which registers both encoder adapters; `feature-harness`
builds `feature_harness`, which registers only the baseline and minimum controls.
Executables live under the selected `/opt/cuwacunu_embedding/build/<session>`
directory. RPB prepare/train/embed has no evaluator dependency. Existing `all`,
`test`, `smoke` and `evaluate` targets continue to select the historical baseline.

The independent LibTorch `2.6.0+cu124` C++11 ABI bundle is staged in
`.external/libtorch`; it was copied from the existing local bundle. To transfer
this project, also stage that same Linux bundle at this path before setup.
After installing the pinned Python dependency, setup copies and verifies every
file into `/opt/cuwacunu_embedding/libtorch`. Builds and runtime loading use
that container filesystem copy. The installer reuses a verified complete copy
and preserves conflicting or partial destinations for inspection. Completion
evidence lives under `/opt/cuwacunu_embedding/setup/sdk-*`. For an already
configured container, run `python3 -B code/scripts/install-libtorch.py` there
to install or verify just this bundle. The staged input stays unchanged. It
includes the CUDA/cuDNN runtime libraries used by the model. The system CUDA
toolkit and cuDNN packages additionally provide `nvcc` and development tools.
Bundled LibTorch libraries take precedence over system
cuDNN to keep runtime versions consistent. Interactive container shells load the
CUDA path from `/etc/profile.d/embedding.sh`; for direct execution without a shell,
use `/usr/local/cuda-12.4/bin/nvcc`. `.external`, `.build`, and generated archives are excluded from Git.
See `THIRD_PARTY_NOTICES.md` for binary license requirements.

## MTF-v1 CLI training and embeddings

The following commands use the retained MTF-v1 baseline, not RPB-v20. Its small
default has C3/H32/F3 and 12-dimensional embeddings. The generated waves,
trends, noise and missing features are a smoke fixture, not downstream quality.
Run inside the managed container, after building the baseline. A new output
directory protects existing archives.

```bash
baseline_bin=/opt/cuwacunu_embedding/build/baseline/embedding
baseline_config=code/encoders/mtf_jepa_mae_vicreg/config/default.conf
mkdir -p output/runs/baseline
baseline_run="$(mktemp -d "$PWD/output/runs/baseline/manual-XXXXXX")"
"$baseline_bin" synthetic --config "$baseline_config" --output "$baseline_run/data.pt" --samples 32
"$baseline_bin" train --config "$baseline_config" --input "$baseline_run/data.pt" \
  --checkpoint "$baseline_run/model.pt" --device cuda
"$baseline_bin" embed --checkpoint "$baseline_run/model.pt" --input "$baseline_run/data.pt" \
  --output "$baseline_run/embeddings.pt" --device cuda
"$baseline_bin" train --resume "$baseline_run/model.pt" --input "$baseline_run/data.pt" \
  --checkpoint "$baseline_run/continued.pt" --steps 8 --device cuda
```

These are optional workflow examples; use saved evidence when reporting an
already completed experiment. New research model execution uses explicit CUDA.
`--steps` means additional updates. `--batch-size` overrides minibatch size.
Fresh training accepts `--seed`; resume restores the saved seed and rejects
`--config`/`--seed`. Pass the same `--input` when continuing a dataset; data is
not embedded in checkpoints. Omitting it selects synthetic training, which
requires F3; synthetic generation supports other channel counts/history lengths.

The historical CLI seeds Torch with `seed + absolute_step` before each update.
Its CPU continuation checks are compatibility history; no parity across devices
or LibTorch versions is promised. CPU remains the historical default for embed
and resume, so request `--device cuda`. New training otherwise uses the config
device. AdamW checks finite loss/gradients, clips gradients and updates the frozen
teacher by EMA.

`--checkpoint-every N` saves at that interval of absolute completed updates.
The historical MTF workflow atomically replaces its requested checkpoint path
during periodic saves. This differs from the RPB-v20 typed inference save API,
which requires a new filename and omits optimizer/RNG resume.

## MTF-v1 CLI configuration

Configuration is plain `key=value` text with `#` comments, strict key names,
`true`/`false` booleans, and comma-separated integer lists. Unknown or duplicate
keys and invalid values fail early. `code/encoders/mtf_jepa_mae_vicreg/config/default.conf` is the readable small
baseline; omitted keys use `default_settings()` in `code/encoders/mtf_jepa_mae_vicreg/src/workflow.cpp`.

The model's scalar fields in
`code/encoders/mtf_jepa_mae_vicreg/include/embedding/encoders/mtf_jepa_mae_vicreg/config.h`
are accepted using their C++ names, along with `time_scales`, `scale_strides`,
and `device=cpu|cuda`.
`dtype` is fixed to float32 in the CLI. An empty `scale_strides=` derives strides
as half each time scale, with a minimum of one. `latent_dim` must be divisible by
`num_heads`. Architecture dimensions C/H/F must match the dataset exactly.

Training settings are:

| Key | Default | Meaning |
| --- | --- | --- |
| `steps` | 8 | Additional optimizer updates |
| `batch_size` | 4 | Training minibatch or embedding chunk size |
| `seed` | 101 | Nonnegative seed; per-step randomness uses seed + step |
| `threads` | 1 | Torch CPU worker threads |
| `log_every` | 1 | Print scalar losses every this many completed steps |
| `checkpoint_every` | 0 | Save every this many absolute completed steps; 0 saves only at completion |
| `learning_rate` | 0.001 | AdamW learning rate |
| `weight_decay` | 0.0001 | AdamW weight decay |
| `gradient_clip_norm` | 1.0 | Maximum gradient L2 norm; 0 disables clipping |

## MTF-v1 tensor archives and C++ API

Archives use LibTorch `torch::serialize::OutputArchive` / `InputArchive`, rather
than Python pickle dictionaries, CSV, or a custom dataset format. All public data
and output tensor fields are buffers (`write`/`read` third argument `true`).

Input keys:

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `data` | float32 `[B,C,H,F]` | Batch, channel, history, feature |
| `feature_mask` | bool `[B,C,H,F]` | True for an observed feature |

Observed values must be finite. Masked values are replaced with zero before model
use; changing them cannot change the embeddings. Each training sample must contain
an observation. Embedding permits entirely missing samples/channels and reports
their validity separately. The loader keeps the dataset on CPU; the model moves
each minibatch to its configured device. Minibatches are sampled with replacement.
This deliberately small loader holds the dataset in memory.

For real sensors, center and scale each feature using statistics from the training
split only, and reuse those statistics for validation and serving. The ordinary
`train`/`embed` commands consume the supplied values without fitting preprocessing.
Descriptor reductions use float64 accumulation before returning the model dtype;
this avoids float32 offset cancellation and intermediate squared-value overflow.
Neural layers still use float32 and require sensible input ranges. `encode` fails
explicitly if neural computation produces nonfinite embeddings.

Example of writing your own data from C++:

```cpp
#include <torch/torch.h>

torch::serialize::OutputArchive archive;
archive.write("data", data.to(torch::kCPU).to(torch::kFloat32), true);
archive.write("feature_mask", observed.to(torch::kCPU).to(torch::kBool), true);
archive.save_to("input.pt");
```

Embedding output keys, all stored on CPU and in the original sample order:

| Key | Shape | Meaning |
| --- | --- | --- |
| `pooled_embedding` | float32 `[B,D]` | Global embedding |
| `pooled_by_channel` | float32 `[B,C,D]` | Per-channel embeddings |
| `sample_valid_mask` | bool `[B]` | At least one valid token |
| `channel_valid_mask` | bool `[B,C]` | At least one valid token for this channel |

Read them with `archive.load_from(path, torch::kCPU)` followed by
`archive.read("pooled_embedding", tensor, true)`, and likewise for other keys.

For direct integration, include
`embedding/encoders/mtf_jepa_mae_vicreg/model.h` and construct `Model(config)`
using `Config` from `embedding::encoders::mtf_jepa_mae_vicreg`.
`encode(data, mask)` returns
token embeddings/masks/metadata as well as the two pooled surfaces. `forward`
returns the training losses; follow optimizer updates with
`update_target_network()`. Use `model->eval()` and `torch::NoGradGuard` for
inference. The model owns a copy of the config; choose its device before
construction. `embedding/encoders/mtf_jepa_mae_vicreg/workflow.h` supplies this
encoder's settings and checkpoint helpers. Generic inputs and archives live in
`embedding/shared/types.h` and `embedding/shared/data.h`. Link the encoder's
`src/workflow.cpp` and `src/evaluation.cpp` with `code/shared/src/data.cpp` and
`code/shared/src/evaluation.cpp` when using its workflow API; its `src/main.cpp`
owns the standalone CLI.

Checkpoint format version 1 contains the canonical settings as UTF-8 bytes,
completed-step counter, nested module state (including EMA teacher), and nested
AdamW state. `load_checkpoint` reconstructs the saved architecture and maps it to
the requested device. `load_optimizer` restores optimizer state after constructing
AdamW over the same ordered trainable parameters. Existing parent-project
checkpoints are not a supported import format.

## Source layout and verification

The root separates implementation in `code/`, documentation in `doc/`, and
generated artifacts in `output/`. Source within `code/` is organized by encoder,
with reusable code kept separately:

```text
code/
  shared/
    include/cuwacunu/config/     # Reusable core file/settings module
    include/piaabo/parse/       # Preserved key/value block parser
    include/embedding/shared/
    src/
    tests/
  encoders/
    mtf_jepa_mae_vicreg/
      include/embedding/encoders/mtf_jepa_mae_vicreg/
      src/
      config/
      tests/
      scripts/                  # Original encoder's evaluation runner
    raw_patch_bottleneck_mae/
      include/embedding/encoders/raw_patch_bottleneck_mae/
      src/
      config/
      tests/
      README.md                 # Architecture, commands and validation links
  evaluation/
    src/main.cpp                # Separate comparison CLI and encoder registry
    cards/                      # Versioned controlled evaluation protocols
    benchmarks/                 # Declared synthetic challenge generators
    protocols/                  # Frozen experiment producers/readers and admission
  scripts/                      # Shared container task dispatch
doc/                            # Current status, versions, evidence, environment and paper
output/                         # Persisted generated artifacts
Makefile                        # Root build entry point
container.ps1                   # Managed container launcher
setup.sh                        # Dependency installation
dependencies.lock               # Pinned dependency versions
```

| Path | Responsibility |
| --- | --- |
| `code/shared/` | Reusable tensor operations, losses, archives, diagnostics, controlled protocols, provider/card contracts, fitting/scoring, comparison engine and tests |
| `code/encoders/mtf_jepa_mae_vicreg/` | Existing model, configuration, training/checkpoint/evaluation adapters and encoder regression tests |
| `code/encoders/raw_patch_bottleneck_mae/` | Independent RPB-MAE model, versioned raw/scaler/checkpoint/export workflow, evaluation adapter, configuration, tests and [implementation guidelines](code/encoders/raw_patch_bottleneck_mae/RPB_MAE_implementation_guidelines.md) |
| `code/evaluation/` | Comparison CLI, encoder-adapter registry, versioned cards, benchmarks and frozen protocol drivers |
| `code/scripts/` | Shared container task dispatch; each encoder owns its CLI and end-to-end checks |
| `doc/` | Current status, version/dataset registries, evidence, environment instructions and preserved paper sources |
| `output/` | Generated datasets, checkpoints, exports and reports |

Shared sources and tests compile without encoder include paths. Each encoder's
public includes begin with `embedding/encoders/<encoder>/`, and its C++ API lives
in the matching `embedding::encoders::<encoder>` namespace. Add
`code/shared/include` and the selected encoder's `include` folder to a consumer's
include directories. The flat forwarding headers and root-level encoder aliases
have been removed; C++ callers must use the explicit encoder API. Checkpoint
formats retain their existing versions and parameter registration names.
See [shared components](code/shared/README.md) and the
[encoder folder](code/encoders/mtf_jepa_mae_vicreg/README.md).

`make test` builds/runs the standalone model and workflow tests. `make baseline`
is an optional extraction audit against the original header copied into the local
`.build/reference` fixture; it is not needed for normal build, testing, or use.

`make smoke` exercises the complete CPU CLI workflow with the checked-in config:
generate data, train eight steps, resume two more, and export embeddings. Artifacts
are written to a fresh `output/runs/<session>/smoke-cpu-*` directory. `make smoke-cuda`
does the same on the GPU and also loads that checkpoint on CPU, writing to a fresh
`output/runs/<session>/smoke-cuda-*` directory.

Verified on 2026-09-13 with Debian 12, GCC 12.2.0, LibTorch 2.6.0+cu124,
and an NVIDIA RTX A2000 8 GB Laptop GPU:

- `make test`: shapes, missing inputs, masks, finite gradients, frozen teacher,
  EMA, checkpoint round-trip, strict config, optimizer continuation, and batched
  embedding export passed. The fixed-batch 12-step check reduced loss from
  1.83091 to 1.56688.
- `make baseline`: seeded CPU float32 parameters, embeddings, masks, losses,
  and gradients matched the original header on the deterministic test fixture.
  This checks the preserved baseline, not every possible model configuration.
- `make smoke smoke-cuda`: both devices completed eight training steps plus
  two resumed steps, then exported embeddings for nine inputs. The CUDA-trained
  checkpoint also loaded and generated embeddings on CPU.

The earlier encoder/shared split was verified on 2026-10-04 in container session
`structure`. The model and shared test suites, extraction audit, CPU/CUDA CLI
smoke checks, and interrupted periodic-checkpoint recovery passed. Exports from
preserved CPU/CUDA checkpoints, including CUDA checkpoints served on CPU,
matched all tensor buffers and archive metadata exactly. A CPU update resumed
from a preserved checkpoint also matched the pre-refactor executable's export
exactly. The evaluation runner passed with the relocated default configuration
and with a preserved checkpoint's original normalization archive. New artifacts
are retained under `output/runs/structure`.

The root `code/` and `doc/` layout was verified on 2026-10-04 in session
`root-layout`. All 43 C++ and configuration files matched their pre-move hashes.
Model/shared tests, extraction parity, CPU/CUDA smoke workflows and periodic
checkpoint recovery passed. The moved runners worked from another directory,
and the task runner refused overlapping work in the same session. Evaluation
passed with both the default config and a preserved checkpoint's normalization.
A preserved checkpoint's export matched its original schema, tensor metadata
and all four tensor buffers exactly. Artifacts are retained under
`output/runs/root-layout`; the move manifest is under `output/runs/reorganization`.

The subsequent source ownership cleanup on 2026-10-05 removed the forwarding
headers and migrated the original encoder's C++ namespace and callers. Final
encoder/shared regression suites and original-header extraction parity passed in
the managed container. The relocated baseline CPU/CUDA workflows, checkpoint
recovery and evaluation runner also passed. September CPU/CUDA checkpoints still
load: export schema and validity masks match exactly, while embedding values
agree within `rtol=1e-5, atol=1e-6` (maximum absolute difference `2.4e-7`).
See the [validation record](output/runs/source-ownership/validation.json) for
sessions, checks and retained artifacts. This is source-refactor evidence, not
representation-quality evidence.

## Representation evaluation

The [embedding evaluation policy](doc/EMBEDDING_EVALUATION_POLICY.md) defines
shared evidence, protocol, reporting, and acceptance requirements for every
encoder. The workflow and historical measurements below describe the existing
baseline's protocol; they do not by themselves satisfy a new acceptance claim.

The separate [evaluation integration](code/evaluation/README.md) uses a shared
engine with encoder-owned adapters. Its instantiated versioned card owns input
geometry, channel IDs/units, tasks, splits, matched dimensions and compared-method
pairs. Adapter fitting receives training observations and permitted metadata;
labels, hidden clean signals and held-out observations stay inside evaluation.
Typed surfaces declare global, channel-concatenation or control semantics and
validity. Baseline channel constituents are contextual; RPB local constituents
are independent.

Build the `comparison` session above, then run inside the managed container:

```bash
bash code/scripts/task.sh comparison evaluate-minimum
bash code/scripts/task.sh comparison evaluate-rpb-mae
/opt/cuwacunu_embedding/build/comparison/embedding_evaluate --output output/new-comparison \
  --encoders baseline,rpb --baseline-steps 8 --rpb-steps 20 \
  --compare rpb_untrained_global,untrained_baseline_global,matched_global
```

The direct CLI requires a new output directory; runners allocate unique ones.
Each run persists its actual `evaluation-card.json`, report, fitted readouts and
adapter assets. Protocol v2 reports pair-specific common-valid populations and
unsupported fits. It provides development evidence, with incomplete acceptance,
cost and robustness requirements documented in the
[implementation review](doc/EVALUATION_IMPLEMENTATION_REVIEW.md). Earlier v1
reports retain their original interpretation.

The baseline's original `embedding evaluate` workflow follows:

The independent quality workflow is separate from the small synthetic smoke
fixture. Its regime labels and missingness are generated independently, and its
training and held-out histories use separate seeded trajectories. It compares
standardized raw inputs, time/frequency descriptors, a mask-only baseline, an
untrained encoder, and the trained encoder with the same ridge-probe protocol.

```bash
bash code/scripts/task.sh baseline evaluate
/opt/cuwacunu_embedding/build/baseline/embedding evaluate --config code/encoders/mtf_jepa_mae_vicreg/config/evaluation.conf --output output/evaluation/report.json --seeds 101,202,303 --steps 100
```

The JSON report records seeds, split provenance, training-only preprocessing,
individual training losses, held-out probe results, embedding diversity, validity
coverage, missing-sensor robustness, and conservative JEPA support overlap. It
measures the exported global and per-channel embeddings as well as projector
outputs. Small or singleton valid batches do not establish representation
diversity: the VICReg result exposes `statistics_supported=false` below two valid
rows, and differences between channel identities do not prove variation between
histories.

Fresh evaluation runs save a resumable checkpoint and normalized training/test
archives per seed beside the report. Input-normalization statistics are saved
with each checkpoint and must be reused when serving it. Evaluation of an existing
checkpoint uses its saved normalization archive; it does not refit the encoder's
input transform on the new evaluation histories. Training-data independence for
an externally supplied checkpoint remains the caller's responsibility.

```bash
/opt/cuwacunu_embedding/build/baseline/embedding evaluate --checkpoint output/evaluation/report.json.seed-101.pt --output output/evaluation/checkpoint-review.json --seeds 101
```

The default normalization path is `CHECKPOINT.normalization.pt`; use
`--normalization PATH` to supply the saved archive explicitly. The normalized
training archive saved by a fresh evaluation run can also be passed to ordinary
`train --resume` to continue that exact dataset.

Strict support masks leave other channels available unless
`mask_same_channel_block=true`. Audits measure window intervals per sample and
channel, merging repeated scales and domains. Because metadata lacks per-feature
observation support, these are conservative overlap estimates for sparse inputs.
Reports include target retention and samples with no feasible targets; a zero
overlap score must be interpreted together with those counts.

Time means/stds discard within-window ordering and frequency magnitudes discard
phase. The evaluator includes an explicit single-window information-retention
check; this is a structural demonstration, not a robotics performance result.
Use held-out probes and time-only/objective/masking ablations to decide whether a
richer raw-patch token branch is justified.

For real data, split by trajectory or session before constructing histories so no
raw observation occurs in two splits. Chronological splits also require a gap
covering the history and prediction horizons. Synthetic probe scores establish
only behavior on the controlled generator; they do not validate a real world model
or control system.

In the September 13 setup verification, after adding the original setup's
development tools, CUDA toolkit 12.4.1 and
system cuDNN 9.26.0.51, every package pin was verified and `dpkg --audit` was
clean. A fresh three-step CUDA training run and nine-sample embedding export
passed under the configured shell environment. These artifacts are in
`output/setup-gpu-check`; LibTorch continues to load its own bundled cuDNN.

Before the code/doc layout move, verification on 2026-10-04 passed all encoder/shared tests, the original-header
extraction audit, CPU/CUDA smoke and resume/export, periodic interrupted
checkpoint recovery, and evaluation with the relocated configuration and saved
normalization. Shared code built without encoder headers. Pre-move CPU/CUDA
checkpoint exports (including CUDA checkpoints served on CPU) and a one-update
CPU continuation matched the pre-move executable's tensor buffers exactly.
Evidence is retained under `output/runs/structure/`.

## Measured synthetic baseline (2026-10-04)

The quality benchmark used three seeds (101, 202, 303), 128 training and 64
held-out trajectories per seed, four balanced regimes, 100 optimizer updates,
batch size 32, and CPU execution with one Torch thread. Each comparison used
the same split, training-only input scaler and initial parameter fingerprints.
All probes used train-fitted coordinate scaling and the same fixed ridge penalty
of one. These are exploratory synthetic results; the ablations have exposed the
test set, so configuration selection needs a validation split and confirmation
on fresh test seeds.

| Configuration | Effective override | Global accuracy | Channel accuracy |
| --- | --- | ---: | ---: |
| Full objective | Evaluation defaults | 68.75% | 68.23% |
| Strict support | `strict_jepa_support=true` | 69.79% | 67.71% |
| Time only | `use_frequency_tokens=false` | 64.58% | 67.71% |
| Without JEPA | `use_jepa_loss=false` | 66.67% | 70.83% |
| Without MAE | `use_mae_decoder=false` | 70.31% | 68.75% |
| Without alignment | `use_tf_align_loss=false` | 69.27% | 71.35% |
| Without VICReg | `use_vicreg_loss=false` | 72.92% | 74.48% |
| Untrained full encoder | Same initial weights, no updates | 72.92% | 75.52% |

Values are means across seeds. The full trained global score has population
standard deviation 5.56 percentage points, versus 1.47 for its untrained control.
Removing VICReg improves channel scores relative to the full objective at all
three seeds, but still trails the untrained channel encoder. It ties the
untrained global mean. No tested condition establishes a consistent learned
advantage. Time-only also removes frequency reconstruction and time/frequency
alignment, making it a composite domain/objective ablation.

Raw descriptors scored 99.48%, standardized raw inputs 26.04%, and masks alone
23.44% (chance is 25%). Descriptors have 3,168 reported coordinates, including
zero padding/domain slots, versus 12 global or 36 concatenated channel
coordinates. This reveals available regime information, but is not a
dimension-matched compression comparison. Random phase also makes the raw-input
linear probe a limited measurement of information in the original histories.

The full model's served global effective ranks decrease from 1.63–2.13 before
training to 1.20–1.75 afterward, while standard deviations increase. This is
greater concentration rather than constant-vector collapse. More than 99% of
its flattened served channel-row population variance is between channel means;
within-channel variation across trajectories is much smaller. Global projector
mean dimension standard deviations are only 0.048–0.063 against a configured
variance floor of one. These findings motivate reviewing regularization
statistics; they do not justify simply increasing the existing VICReg weight.

Across three seeds, eight mask trials and 64 clean held-out rows per trial,
legacy masking overlapped context support for 63.74% of target tokens. Strict
masking produced zero measured same-channel overlap, retaining 60.84% of target
tokens and 71.24% of target temporal support, with no valid row losing every
target in this fixture. This verifies direct support separation, not improved
downstream utility. Different channels can remain statistically dependent,
full-window support is conservative for sparse observations, and fixed update
budgets expose strict training to fewer targets. Strict selection is heuristic
and may return no targets on other geometries when separation and minimum
context conflict. The legacy default remains appropriate for compatibility.

Outage probes reuse the original training-fitted probe on the clean version of
the held-out histories or its corruptions:

| Input condition | Trained global | Untrained global | Raw descriptors |
| --- | ---: | ---: | ---: |
| Clean, fully observed | 68.23% | 74.48% | 100.00% |
| 60% random dropout | 45.83% | 39.06% | 86.98% |
| 25% contiguous outage | 31.25% | 40.10% | 80.21% |
| 50% contiguous outage | 29.17% | 38.54% | 59.90% |
| One of three channels missing | 24.48% | 19.79% | 100.00% |

All these cases retain 100% sample validity; the gap concerns utility, not
whether the model produces an output. All-missing inputs correctly have zero
coverage and null scores. Training helps at 60% random dropout but structured
outages remain weak. The generator supplies redundant regime signals across
channels; this controlled stress test does not establish recoverability of
unique information from a missing real sensor.

The next bounded experiments should:

1. Add train-fitted descriptor PCA/random-projection controls at 12 and 36
   dimensions, and a validation split for probe/training-duration selection.
2. Compare current VICReg with statistics across trajectories within each
   channel, retaining the architecture, objective coefficients and training
   budget. Confirm candidates on fresh seeds, and keep served-space diversity
   diagnostics.
3. Diagnose the change in pooled embeddings when channels disappear, and test
   outage-matched training before selecting a different pooling rule.

The single-window reversal collision proves an ordering/phase limitation in
that deliberately restricted tokenizer fixture. It does not prove that the
default overlapping multiscale model is reversal-invariant or explain the
current regime gap. A richer raw-patch branch should be evaluated on tasks
requiring that missing information; a larger alternative encoder remains
deferred.

Archived original benchmark invocation (CPU); retained for provenance, not a current research rerun instruction:

```bash
bash code/scripts/task.sh baseline -j2 all test baseline
bash code/encoders/mtf_jepa_mae_vicreg/scripts/evaluate.sh --config code/encoders/mtf_jepa_mae_vicreg/config/evaluation.conf \
  --seeds 101,202,303 --train-samples 128 --test-samples 64 --steps 100 --device cpu
```

For an ablation, copy the evaluation configuration and change the indicated
effective setting, preserving every other setting and the same CLI arguments.
The runner creates a fresh output directory. The original 21 runs completed
with finite losses and retain 84 checkpoint/data/scaler archives. Their local
[summary](output/evaluation/goal-01a107a5/summary.json), per-variant reports under
`output/evaluation/goal-01a107a5/`, and
[source hashes](output/evaluation/goal-01a107a5/source-sha256.txt) preserve the
evidence. Those runs preceded the folder refactor; the hash snapshot records
the source paths at that time. Generated reports and archives are excluded from
Git. The earlier encoder/shared folder move was separately validated against the original extraction
and checkpoints.

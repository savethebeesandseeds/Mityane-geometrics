# Source and runtime requirements

The numerical closure contains three maintained family headers, the complete
frozen V16 `include/` tree and, for the example CLI, the generic observation
archive header. All retain their original relative paths under `sources/`.
The inventory lists the three include roots. After import, these sources need
no sibling Mityane checkout.

The verified implementation uses **C++20, LibTorch 2.6.0+cu124, C++11 ABI1**,
an explicit CUDA device with F32 model execution, and CPU F64 raw observations.
Existing checks exercised CUDA:0. Other Torch/CUDA/compiler/ABI combinations
and devices require their own verification. The model is header-only; the CLI
and fixtures are separate entry points. A consumer build adds both Torch
include roots (`include/` and `include/torch/csrc/api/include/`), defines
`_GLIBCXX_USE_CXX11_ABI=1` and links Torch/CUDA and pthread. Mityane's root build
links `torch`, `torch_cpu`, `torch_cuda`, `c10_cuda` and `c10`, using the SDK's
library path.

In Mityane, use the root Makefile, `code/scripts/task.sh` and managed Linux
container. Its inherited SDK remains at `/opt/cuwacunu_embedding/libtorch`;
export never relocates or copies it. Cuwacunu uses its own root build and managed
container. This package introduces no component-owned environment/build
configuration, dependency installation or host toolchain.

Python and Node are not numerical runtime requirements. Export/verification
and optional CLI test orchestration use Node standard libraries. The bundle
includes no Node executable/cache and has no external npm dependencies.

Project code retains **MIT** in `sources/LICENSE` and inherited copyright
notices. Original V16 provenance and predecessor third-party notices are copied
with their original paths/bytes. Dated statuses and paths are historical
provenance, not current lifecycle instructions. LibTorch, CUDA and cuDNN retain
their upstream licenses; no third-party binaries are redistributed here.
Recipients distributing binaries retain their corresponding licenses/notices.
An upstream revision alone cannot identify maintained source: the bundle pins
each delivered file, including local changes.

The application supplies data, source/split manifests and trained artifacts
under its access, chronology and provenance rules. A caller-declared TRAIN role
does not independently prove real observation membership.

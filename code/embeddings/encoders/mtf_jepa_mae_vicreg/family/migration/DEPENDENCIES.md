# MTF source and runtime requirements

The [explicit inventory](package.json) defines one source delivery. Its core
and training runtime contain no sibling-checkout dependency. All repository
paths are retained under bundle `sources/`; they are origins and intended
destinations, not historical evidence-path replacements.

| Required source | Purpose |
| --- | --- |
| Original MTF `include/` tree except optional adapter header | Configurable network, tokenization, masks, objectives, teacher and historical workflow declarations |
| Original `src/workflow.cpp` and `src/evaluation.cpp` | Settings helpers and raw legacy loading; workflow command dispatch retains its evaluator linkage |
| New `family/include/` and `family/src/` | Typed portable recipe/session, update, identity preparation, new training and inference codecs |
| Observation `include/embedding/shared/data.h` and `src/data.cpp` | Batch/archive, synthetic data and atomic archive helper |
| Shared representation `types.h`, `tensor_ops.h`, `objectives.h`, `evaluation.h` and `src/evaluation.cpp` | Numerical helpers, masks/losses and retained historical evaluator linkage |
| Artifact v1 `sidecar.h`, `tensor_state.h` | SHA256 and deterministic tensor/module state hashing; no application authentication is inferred |
| Native example/tests/configurations | Runnable fresh/train/save/resume/export example and bounded proof fixtures |

The optional shared evaluation adapter is documented in the audit but excluded
from this source profile. Its feature-harness/evaluation-engine closure is not
required for fresh portable use. Original `src/main.cpp`, historical shell
runners and `baseline_test.cpp` are also excluded: they are not the new example.
The ignored `.build/reference/reference_model.h` remains a historical parity
fixture outside this package. No historical parity or score replay is claimed.

The pinned Mityane environment uses **C++20, LibTorch 2.6.0+cu124, C++11 ABI1**,
little-endian host tensor hashing and CPU F32 execution. `Settings.threads=1`
enforces one intra-op Torch thread; the inherited inter-op setting is recorded
and bound, not forced to one (actual primary proof: 12). Torch's CPU and
CUDA-capable libraries are linked even
for CPU execution. Other compiler/Torch/ABI/platform/device combinations need
their own verification. Add these source include roots:

```
code/embeddings/encoders/mtf_jepa_mae_vicreg/family/include
code/embeddings/encoders/mtf_jepa_mae_vicreg/include
code/shared/representation/include
code/data/observations/include
code/shared/artifacts/versions/v1/include
```

Add LibTorch's `include/` and `include/torch/csrc/api/include/`, define
`_GLIBCXX_USE_CXX11_ABI=1`, and link `torch`, `torch_cpu`, `torch_cuda`,
`c10_cuda`, `c10`, pthread with the SDK library path. A consumer's own root
build compiles these translation units for the CLI:

```
code/embeddings/scripts/mtf_family_train.cpp
code/embeddings/encoders/mtf_jepa_mae_vicreg/family/src/family.cpp
code/embeddings/encoders/mtf_jepa_mae_vicreg/src/workflow.cpp
code/embeddings/encoders/mtf_jepa_mae_vicreg/src/evaluation.cpp
code/data/observations/src/data.cpp
code/shared/representation/src/evaluation.cpp
```

Compile `family/tests/family_test.cpp` instead of the CLI translation unit for
the native fixture. Tests additionally use family and shared test include roots
listed in the package profile. No original Mityane root Makefile is supplied as
a standalone consumer build: that Makefile references unrelated components.
The included `family.mk` is a build-rule reference for root integration, not an
environment installer. Root isolated verification can bind all source paths to
the relocated bundle through its source-root option.

For example, in a recipient's managed Linux environment with an already
available compatible SDK, use its own build/output directory and these commands.
The variables are illustrative paths to existing dependencies; this installs
nothing and reads no original checkout:

```bash
mtf_source=/absolute/path/to/bundle/sources
mtf_build=/absolute/path/to/consumer-build
mtf_sdk=/absolute/path/to/existing/libtorch
mtf_core="$mtf_source/code/embeddings/encoders/mtf_jepa_mae_vicreg"
mkdir -p "$mtf_build"
g++ -std=c++20 -O1 -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$mtf_core/family/include" -I"$mtf_core/include" \
  -I"$mtf_source/code/shared/representation/include" \
  -I"$mtf_source/code/data/observations/include" \
  -I"$mtf_source/code/shared/artifacts/versions/v1/include" \
  -isystem "$mtf_sdk/include" -isystem "$mtf_sdk/include/torch/csrc/api/include" \
  "$mtf_source/code/embeddings/scripts/mtf_family_train.cpp" \
  "$mtf_core/family/src/family.cpp" "$mtf_core/src/workflow.cpp" \
  "$mtf_core/src/evaluation.cpp" \
  "$mtf_source/code/data/observations/src/data.cpp" \
  "$mtf_source/code/shared/representation/src/evaluation.cpp" \
  -L"$mtf_sdk/lib" -Wl,-rpath,"$mtf_sdk/lib" -Wl,-rpath-link,"$mtf_sdk/lib" \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 \
  -Wl,--as-needed -pthread -o "$mtf_build/mtf_family_train"
"$mtf_build/mtf_family_train" --help
```

Within Mityane the SDK stays at `/opt/cuwacunu_embedding/libtorch`; export does
not relocate, modify or copy it. `container.ps1`, the root task runner and root
Makefile govern development. The full environment lock is larger than this
minimal source closure and is not a requirement to copy a research environment.

Python and Node are not numerical runtime requirements. The exporter and its
independent verifier use Node standard libraries only; no npm dependency or
Node runtime is bundled. Shell orchestration stays in the managed environment.

Project source retains MIT `sources/LICENSE`, original copyright notices and
the imported cuwacunu_embedding/cuwacunu_torch attribution in
`sources/THIRD_PARTY_NOTICES.md` and `sources/doc/PROVENANCE.md`. The recorded
upstream base alone does not identify maintained bytes: every delivered file is
pinned. Dated audit/provenance records retain their original path/hash meaning.
LibTorch/CUDA/cuDNN and compiler/runtime binaries retain their upstream terms;
no binaries are redistributed here. Any separate binary redistribution must
carry the corresponding upstream licenses/notices. Dashboard AGPL code is
outside this package. Applications supply data and trained artifacts subject
to their own access, license, chronology and provenance rules.

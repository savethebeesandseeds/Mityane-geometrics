# RPB source, runtime and license dependencies

The [curated profile](package.json) declares entry files, local include roots
and allowed dependency roots. The exporter discovers every quoted local C++
include recursively and fails a missing, ambiguous, unsafe, symlinked or
out-of-scope dependency. Its manifest records the delivered complete closure,
per-file original and destination paths, byte counts and SHA256. Canonical
source plus compatibility wrappers retain their repository-relative layout
under `sources/`; copying only `family/include/` is incomplete.

The numerical closure reaches shared RPB code and older v13/v14/v16–v19
optional-mode headers through the v20 backbone. Existing compatibility fixtures
also reach v21. Settings parsing needs `cuwacunu/config/document.h` and
`piaabo/parse/simple_kv_block.h`. The portable runner additionally needs the
generic observation archive and the artifact header's SHA256 primitives.
That header's presence does not enroll the family in application publication
or artifact-selection policy. No dataset reader, training parent, evaluator,
market runner or application checkout is a numerical dependency.

The tested project environment is **Linux, C++20, LibTorch2.6.0+cu124,
C++11 ABI1 and actual CUDA float32 model execution**. Raw example archives and
TRAIN moment arithmetic are CPU float64; relation arithmetic includes CUDA
float64. Alternative SDK/compiler/backend/device combinations require their
own verification. The root build supplies both Torch include roots and
`_GLIBCXX_USE_CXX11_ABI=1`, O1, Torch library search/rpath, and links `torch`,
`torch_cpu`, `torch_cuda`, `c10_cuda`, `c10` and pthread. No CPU model,
mixed-precision, distributed or cross-environment exact-resume contract is
advertised. Memory capacity is a runtime constraint, not guaranteed by valid
configuration syntax.

Mityane uses the existing root Makefile, `code/scripts/task.sh` and managed
`mityane-geometrics` container through `container.ps1`. The inherited SDK remains
`/opt/cuwacunu_embedding/libtorch`; packaging does not copy, relocate or modify
it. `family.mk` is target integration with a configurable source root, not an
independent environment definition. A recipient supplies its own managed
compiler/SDK/root build and can point `RPB_FAMILY_SOURCE_ROOT` to exported
`sources/`. The native CLI has no runtime dependency on a sibling checkout.
Node used for metadata/export/example test orchestration is separate from
numerical execution; consumers may run the native binaries directly.

Checkpoint publication needs Linux/POSIX `mkstemp`, `fsync` and atomic
no-replace `renameat2`, with hardlink fallback on supported filesystems.
New filenames are required; existing artifacts are preserved. This contract
does not promise power-loss durability or Windows runtime portability.

Project source retains **MIT** in bundled `sources/LICENSE` and file notices.
Root third-party notices, source provenance and frozen predecessor notices
retain the origin boundary. The imported predecessor base is
`b15702b29753188482b11ca85e997006b503d0f0`; current per-file hashes, rather
than that ancestry revision alone, identify maintained bytes. Historical
receipts and copied prose retain their dated paths and scope.

LibTorch, CUDA and cuDNN retain their upstream licenses; Debian compiler/runtime
notices remain under `/usr/share/doc/<package>/copyright` in the environment.
The source package redistributes no SDK or third-party binaries. A separate
binary redistribution must carry its corresponding upstream notices. The
dashboard's AGPL source/artwork and Markdown verifier's Python-2.0 entity table
are outside this closure. Project MIT does not relicense either those assets
or external binary dependencies.

The exporter/verifier uses Node standard libraries only, with no npm dependency.
Python is not a numerical, packaging or example-runtime requirement. Dataset
archives, fitted scalers, weights, optimizer states, generated outputs, caches,
credentials, SDKs and container images remain separate artifacts. Applications
admit their actual data/splits, licenses/access, resources and downstream use.

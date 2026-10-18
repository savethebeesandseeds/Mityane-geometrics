# Shared embedding components

This folder contains reusable code that compiles without any encoder headers.
Both encoders use these components directly; they are not duplicated
model implementations.

The reusable core settings module also supports the separate
[RPB-v20 API](../encoders/raw_patch_bottleneck_mae/CONFIGURABLE_SPECTRAL.md).
See [current status](../../doc/CURRENT_STATUS.md) for model/evidence scope.
The table and examples below describe shared utilities; protocol-specific
information-admission and frozen research drivers additionally live under
`code/evaluation/benchmarks/` and `code/evaluation/protocols/`.

The [embedding evaluation policy](../../doc/EMBEDDING_EVALUATION_POLICY.md)
defines the shared harness, task/protocol, encoder-adapter, and report boundaries.
Encoder-specific training losses and structural contracts stay with their encoder;
common fitting, scoring, coverage, and acceptance rules follow that policy.

| File group | Responsibility |
| --- | --- |
| `include/cuwacunu/config/document.h`, `include/piaabo/parse/simple_kv_block.h` | Core owning file/settings parser, standard library only; [syntax/provenance](config/README.md) |
| `include/embedding/shared/types.h` | Explicit input dimensions/device/dtype, data/mask tensors and VICReg loss result types |
| `include/embedding/shared/tensor_ops.h` | Input canonicalization, masked pooling/MSE and channel validity/pooling |
| `include/embedding/shared/objectives.h` | Pure VICReg loss calculation |
| `include/embedding/shared/data.h`, `src/data.cpp` | Data archives, input validation, synthetic smoke data, atomic archive and path/text helpers |
| `include/embedding/shared/evaluation.h`, `src/evaluation.cpp` | Independent synthetic benchmark data and representation statistics |
| `include/embedding/shared/feature_harness.h`, `src/feature_harness.cpp` | Source-group paired protocols, observed-only scaling, valid-row normalization/PCA, frozen ridge/tiny probes, coverage and grouped uncertainty |
| `include/embedding/shared/feature_evaluation.h`, `src/feature_evaluation.cpp` | Typed provider/surface contract, versioned cards, comparison orchestration, pair-specific populations, fitted assets and reports |
| `include/embedding/shared/feature_stress.h`, `src/feature_stress.cpp` | Testing-only missingness cases, immutable fitted-readout callbacks, support populations and robustness sidecars |
| `include/embedding/shared/reconstruction_evaluation.h`, `src/reconstruction_evaluation.cpp` | Label-free reconstruction-provider contract, fixed held-out targets, latent interventions, train-fit metric scaling, matched populations and exchange-block uncertainty |
| `include/embedding/shared/learning_curve.h`, `src/learning_curve.cpp` | Continuous training callbacks, immutable snapshots, validation-only budget selection and fresh testing |
| `include/embedding/shared/projection_diagnostic.h`, `src/projection_diagnostic.cpp` | Archived training/validation features, matched PCA/random projection, fixed readout repetitions and paired comparisons |
| `include/embedding/shared/global_bottleneck_experiment.h`, `src/global_bottleneck_experiment.cpp` | Generic three-provider bottleneck curves, common selection, fixed-readout stress and matched source-diversity experiments |
| `tests` | Shared loss, data-generation, archive/mask, diagnostic, fitting and dummy-provider engine checks and common assertion helpers |

Paths in the table are relative to this `code/shared/` folder. Public includes
begin with `embedding/shared/`; add `code/shared/include` from the repository root
to a C++ consumer's include directories. Data and evaluation generators accept
`embedding::input_shape_t`, so they do not require model configuration. The
baseline retains thin configuration adapters inside its own folder.

Encoder folders import these headers directly using their existing C++ include
names and link `code/shared/src/data.cpp` and `code/shared/src/evaluation.cpp` when
using the archive and evaluation helpers. Shared components do not import an
encoder's configuration or headers.

Encoder-specific token metadata, masking policies, model settings, optimizer
checkpoint reconstruction, training orchestration and projector diagnostics
belong to their encoder. The Makefile uses only shared include paths when
compiling this folder's sources and tests, enforcing the dependency boundary.

`make test-feature-evaluation` checks cards, provider contracts and comparisons
with dummy providers; `make test-feature-harness` includes those checks and the
common fitting/protocol tests. Run these through the managed container task runner.
`make test-feature-stress` checks the missingness track; the combined harness
target includes it.

The [comparison CLI](../evaluation/README.md) registers encoder-owned adapters.
`ProviderFitInput` contains only permitted training observations and metadata;
labels, hidden clean signals and held-out observations stay in the engine.
`FeatureMap` exposes CPU `[B,D]` values and bool `[B]` validity, with typed
`SurfaceDescription` semantics/support/channel order. The card owns geometry,
tasks, seeds, matched widths and comparison pairs. Names do not determine a
surface's kind or validity population. Each pair uses its own declared common
population, so an unrelated provider does not change that comparison.

Generic harness and engine code import neither encoder. The `evaluation` target
builds `embedding_evaluate` with both registered adapters; `feature-harness`
builds `feature_harness` with baseline/raw/descriptor/mask controls and no RPB
model/workflow. Both use the same shared engine and protocol v2. They produce
development evidence; the [implementation review](../../doc/EVALUATION_IMPLEMENTATION_REVIEW.md)
records the remaining acceptance, robustness, cost and bottleneck-evidence gaps.

The separate reconstruction engine evaluates exact compact decoder inputs.
Providers own model fitting and an independently trained metadata-only control;
the engine supplies cloned visible observations or boolean support, and owns
hidden targets, source-group swaps, zero interventions, MAE/Huber and reports.
Its [versioned card](../evaluation/cards/controlled_reconstruction_v1.md) is saved
before data generation/fitting. `test-reconstruction-evaluation` uses dummy
providers with no encoder includes; `test-rpb-reconstruction` validates the RPB
adapter and its saved assets. Both run inside the managed container.

`learning_curve.h/.cpp` owns validation-only checkpoint curves and a subsequent
fresh test stage. The model callback receives training observations only and
supplies exact frozen checkpoint features/compact reconstruction. The engine
owns targets, training-only readout fits and common budget selection; testing
data are generated only after the selection artifact is saved. The
[versioned card](../evaluation/cards/learning_curve_v1.md) describes this boundary.

`projection_diagnostic.h/.cpp` accepts explicit archived training and validation
features. It fits normalization, PCA and readouts on training rows and compares
data-independent orthonormal projections at matched widths. It has no encoder
dependency or testing stage. `global_bottleneck_experiment.h/.cpp` uses the same
label-free trainer/snapshot callbacks for three declared variants. It saves
validation-only common-budget selection before generating fresh testing sources,
and supports a separately preregistered fixed budget with nested training-source
prefixes and an independent validation stream. Protocol tests use dummy providers.
The [RPB experiment record](../encoders/raw_patch_bottleneck_mae/GLOBAL_BOTTLENECK_ADVANCE.md)
preserves that historical integration recipe and its limits.

`saved_feature_reliability.h/.cpp` analyzes explicitly supplied saved TRAIN
features, normalizers, classifier weights/predictions and optimization traces.
It replays fixed arithmetic, margins and source-paired geometry with no encoder,
head fitting or PCA dependency. Geometry, dimensions, parameter/update counts
and comparison identities are caller contracts. The separate protocol CLI
owns its frozen85-role/card/admission bindings; a different encoder can reuse
the tensor analysis through its own declared inputs. Run
`test-saved-feature-reliability` through the managed container task runner.

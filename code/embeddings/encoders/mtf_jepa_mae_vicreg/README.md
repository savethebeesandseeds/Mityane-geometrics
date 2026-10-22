# MTF JEPA MAE VICReg encoder

This folder owns the existing MTF-JEPA-MAE-VICReg implementation. Its model
behavior, parameter registration names, configuration defaults and version 1
checkpoint format are preserved.

## Agent instructions: reusable encoder components

Follow the shared [encoder Hero instructions](../AGENTS.md). Hx15 develops
reusable MTF architecture and training tools that applications can configure,
train and compose. Keep the family definition, supported configuration,
training recipe and trained instance separate. A retained checkpoint or
benchmark configuration supplies scoped evidence; it does not define the only
supported input geometry, latent width or model capacity.

Derive dimensions and parameter counts from configuration and validate actual
architectural constraints. Document the implemented fresh-construction,
TRAIN preprocessing, training, export and checkpoint/resume paths, their
dependencies and missing coverage. Verify supported alternative configurations
with focused engineering fixtures before claiming portability. This guidance
does not assert that every retained MTF interface already meets that standard.
Cuwacunu and other applications own substantial application training and final
docking. Preserve historical source, checkpoint formats and research records;
these instructions do not start another experiment or training run.

The [22 October component and training-method audit](delivery/COMPONENT_TRAINING_AUDIT_2026-10-22.md)
preserves the source-pinned state before portable delivery: construction,
preparation, objective/EMA, stochastic continuation and dependencies. Its
runtime, proof and packaging gaps describe that earlier snapshot; the completed
delivery below records their resolution within its stated scope. The audit
itself adds no numerical or quality result.

The [portable family layer](family/README.md) now supplies a typed C++ training
session, explicit identity preparation and sampling recipes, fixed-budget
training-state continuation and a separate inference codec over the preserved
core. Its [engineering record](family/verification/RESULTS_2026-10-22.md) states
the actual verification scope; [source delivery](family/migration/README.md)
keeps the network and complete learning method together. Historical MTF-v1 and
application-owned artifact admission remain separate.

Its stable design tag is **MTF-v1 — Multiscale JEPA and reconstruction** in the
[version registry](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md). It remains a retained
reference with verified portable engineering coverage. As confirmed by the owner
on 22 October 2026, it is not currently used in downstream applications.

The [archived predecessor status](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CURRENT_STATUS.md)
describes RPB-v20's separate API and completed quality records at that stage.
Use the [current research guide](../../../../doc/EMBEDDINGS.md) for current
references and continuation scope. The retained `embedding` CLI and its MTF-v1
checkpoint format remain unchanged. The portable `mtf_family_train` CLI uses
separate typed training-state and inference formats; these do not establish
compatibility with historical checkpoint/scaler associations.

The [archived embedding evaluation policy](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_EVALUATION_POLICY.md)
records the evidence, protocol, reporting and acceptance rules for its historical
experiments. The retained baseline evaluator and historical reports keep their
original protocol and limitations. Current work follows the shared encoder
instructions and [project charter](../../../../doc/operations_over_representations_design_spec_v0.1.0.md).
The separate shared evaluator produces protocol v2 development evidence; its
[archived implementation review](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EVALUATION_IMPLEMENTATION_REVIEW.md)
records implementation limits at that stage.

| Folder | Contents |
| --- | --- |
| `include/embedding/encoders/mtf_jepa_mae_vicreg` | Model configuration, tokenization, masks, encoder, predictor, decoder, objective composition and public workflow declarations |
| `src` | Model settings, checkpoint/resume, training/embedding CLI, historical evaluator and shared-engine evaluation adapter |
| `config` | Retained CLI training and synthetic evaluation configurations |
| `tests` | Model, numerical, masking, checkpoint/workflow, evaluator and extraction regression checks |
| `scripts` | This encoder's historical evaluation runner |
| `family` | Portable training/inference API, example configurations, bounded engineering proof and source delivery |
| `delivery` | Preserved component and training-method audit of the pre-delivery source snapshot |

The implementation imports masked pooling/MSE, VICReg calculation, synthetic
generators and representation diagnostics from
[shared representation components](../../../shared/representation/README.md).
Observation batches and archives are owned by
[`code/data/observations/`](../../../data/README.md). The portable layer also
uses shared artifact hashing helpers; its
[dependency inventory](family/migration/DEPENDENCIES.md) lists the complete
source and runtime requirements. Model-specific metadata and augmentation
policy stay here.

From the repository root inside the managed container:

```bash
bash code/scripts/task.sh hx15-mtf-build -j2 build-mtf-family build-mtf-family-trainer
```

The [family guide](family/README.md) documents the portable example and completed
CPU/F32 verification. The retained baseline tests additionally depend on the
historical ignored reference fixture; CUDA smoke and historical evaluation are
separate workflows and are not evidence for the portable CPU surface.

For reusable evaluation, build `evaluation` to obtain `embedding_evaluate` with
baseline and RPB adapters, or `feature-harness` to obtain the baseline-only
`feature_harness`. Both use the shared engine; the original `embedding evaluate`
and this folder's evaluation runner retain their historical behavior. See the
[integration guide](../../evaluation/README.md) for commands and protocol cards.

`evaluation_adapter.h/.cpp` supplies a label-free `FeatureProviderFactory` with
global exports, complete time/frequency descriptors and a joint channel
concatenation. Those channel vectors are contextual through global-mean mixing;
the concatenation is not a set of independent per-channel probe scores.
The adapter declares support and fixed semantic channel order, uses explicit
identity float64-to-float32 preparation, and records initialization/training
streams and saved assets. Optional fresh training consumes only permitted
training observations. It does not associate these new controls with historical
checkpoint weights or scores. The card owns input dimensions and matched widths;
an explicit configuration must agree.

Canonical C++ includes begin with
`embedding/encoders/mtf_jepa_mae_vicreg/`. Its C++ API lives in
`embedding::encoders::mtf_jepa_mae_vicreg`.
Core consumers add `code/embeddings/encoders/mtf_jepa_mae_vicreg/include` and
`code/shared/representation/include`; workflow consumers also need
`code/data/observations/include`. Portable consumers use all include roots and
translation units in the [dependency inventory](family/migration/DEPENDENCIES.md).
The portable API adds the nested `family` namespace. Paths in the table are
relative to this encoder folder. Retained training/evaluation configurations
live in `config/`; portable example configurations live in `family/config/`.
`Model`, `Config`, `Settings` and the model output types belong to this encoder's
namespace. Generic batch/data types and archive primitives remain shared. The
flat forwarding headers and root-level encoder aliases have been removed; source
callers use the canonical includes and namespace. Retained CLI commands,
checkpoint keys and parameter registration names keep their existing behavior.

Use the [current environment guide](../../../../doc/ENVIRONMENT.md) for named
build sessions and persisted output locations. The
[archived environment notes](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/ENVIRONMENT.md)
retain predecessor context and commands.

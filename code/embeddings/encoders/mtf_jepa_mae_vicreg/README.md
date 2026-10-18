# MTF JEPA MAE VICReg encoder

This folder owns the existing MTF-JEPA-MAE-VICReg implementation. Its model
behavior, parameter registration names, configuration defaults and version 1
checkpoint format are preserved.

Its stable design tag is **MTF-v1 — Multiscale JEPA and reconstruction** in the
[version registry](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md). It remains a retained
reference; active RPB development does not routinely retrain this encoder.

See [current repository status](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CURRENT_STATUS.md) for RPB-v20's
separate API and the latest completed quality record. This folder's CLI and
checkpoint format remain MTF-v1.

The [embedding evaluation policy](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_EVALUATION_POLICY.md)
defines shared evidence, protocol, reporting, and acceptance rules. The retained
baseline evaluator and historical reports keep their original protocol and
limitations. The separate shared evaluator produces protocol v2 development
evidence; its [implementation review](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EVALUATION_IMPLEMENTATION_REVIEW.md)
records policy requirements that remain incomplete.

| Folder | Contents |
| --- | --- |
| `include/embedding/encoders/mtf_jepa_mae_vicreg` | Model configuration, tokenization, masks, encoder, predictor, decoder, objective composition and public workflow declarations |
| `src` | Model settings, checkpoint/resume, training/embedding CLI, historical evaluator and shared-engine evaluation adapter |
| `config` | Default training and synthetic evaluation configurations |
| `tests` | Model, numerical, masking, checkpoint/workflow, evaluator and extraction regression checks |
| `scripts` | This encoder's historical evaluation runner |

The implementation imports architecture-independent input tensors, archive
operations, masked pooling/MSE, VICReg calculation, synthetic generators and
representation diagnostics from [shared components](../../../shared/representation/README.md) in
`code/shared/representation/`. Model-specific metadata and
augmentation policy stay here.

From the repository root inside the managed container:

```bash
bash code/scripts/task.sh baseline -j2 all test baseline
bash code/scripts/task.sh baseline smoke-cuda
```

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
From the repository root, add `code/embeddings/encoders/mtf_jepa_mae_vicreg/include` and
`code/shared/representation/include` to the C++ include directories. Paths in the table are
relative to this encoder folder.
Training and evaluation configurations live at
`code/embeddings/encoders/mtf_jepa_mae_vicreg/config/`.
`Model`, `Config`, `Settings` and the model output types belong to this encoder's
namespace. Generic batch/data types and archive primitives remain shared. The
flat forwarding headers and root-level encoder aliases have been removed; source
callers use the canonical includes and namespace. CLI commands, checkpoint keys
and parameter registration names retain their existing behavior.

See [environment coordination](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/ENVIRONMENT.md) for named build
sessions and persisted output locations.

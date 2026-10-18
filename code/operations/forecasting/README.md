# Forecasting and common-observable readouts

This directory owns imported **OR-11 forecasting** implementations and their
separately testable **OR-05 observable readouts**. Read the
[Operation Contracts](CONTRACT.md) and linked
[Requirements-and-Evidence Record](EVIDENCE.md) before interpreting outputs.
The code is exploratory: synthetic engineering checks and prospective market
registrations do not establish forecasting skill or application utility.

| Directory | Responsibility |
| --- | --- |
| [latent_dynamics](latent_dynamics/README.md) | Direct-horizon affine predictions of future target states |
| [predictive_training](predictive_training/README.md) | Predictive training arms, objectives, EMA target state, exact resume and checked serving |
| [direct_embedding_control](direct_embedding_control/README.md) | Native-coordinate residual-MLP forecast comparator; “control” means experimental comparator |
| [issuance](issuance/versions/v1/README.md) | Checked input-only forecast issuance with fit, availability and provenance boundaries |
| [evaluation](evaluation/README.md) | Saved-array forecasting baselines/scoring and separate common-observable ridge readouts |

Representation providers remain in [embeddings](../../embeddings/README.md),
including the [State Representation Network](../../embeddings/state_representation/README.md).
Application timelines and observable labels belong in
[data](../../data/README.md); cross-component artifact identities belong in
[shared](../../shared/README.md). Imported C++ namespaces and model version
tags stay stable across this directory cleanup.

Build and test from the repository root through `container.ps1`, using the root
Makefile and `code/scripts/task.sh` inside the existing `mityane-geometrics`
container. For a named engineering session, the command inside `/mityane` is:

```sh
bash code/scripts/task.sh forecasting-check test-predictive-v1 \
  test-predictive-snapshot test-direct-control-v1 \
  test-direct-control-artifact test-forecast-issuance-v1 \
  test-latent-dynamics-v1-evaluation test-latent-risk-v1-readouts
```

These targets use artificial fixtures. Their current verification receipts must
identify the exact targets/source actually run. Training and model-selection
research require their separately admitted frozen protocol; this maintenance
entry point does not start a study.

Original research documentation, source closures, policies, results and failure
records are preserved under
[the complete pre-cleanup snapshot](../../../archive/embeddings/pre-cleanup-2026-10-18/README.md).
Historical paths and hashes keep their original meaning. New code has a current
source identity and must not silently inherit an earlier captured-source claim.

Copyright (c) 2026 Waajacu. The imported implementation remains under the root
[MIT license](../../../LICENSE) and its preserved notices.

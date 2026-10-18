<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE. -->
# Predictive state training

This component owns objectives, attribution arms, frozen EMA teacher, complete
typed resume and fixed-checkpoint affine serving checks. It consumes saved
embeddings admitted by an external temporal ledger. It has no dependency on
an Encoder implementation, application loader or future-risk label.

| Version | Physical implementation | Explicit dependencies | Namespace |
| --- | --- | --- | --- |
| v1 | [versions/v1](versions/v1/README.md) | `state_representation/versions/v1`, `latent_dynamics/versions/v1` | `cuwacunu::networks::predictive_training::v1` |

Each future training version owns its complete objective, checkpoint schema,
serving helpers and tests, and names its component versions explicitly. Do not
silently redirect a historical version to a new mapping or transition. Model
design tags and trained-instance/checkpoint hashes remain separate identities.

<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE. -->
# State Representation Network

The State Representation Network maps each native signal embedding to a distinct
predictive state. It owns neither market data admission nor the Encoder's
reconstruction objective. Hx21 owns the State fusion transformer in
`code/fusion/state_fusion_transformer`. This network is not that family, and this page
does not define it.

| Version | Physical implementation | Namespace | Contract |
| --- | --- | --- | --- |
| v1 | [versions/v1](versions/v1/README.md) | `cuwacunu::networks::state_representation::v1` | Frozen TRAIN input transform plus affine D-to-d state mapping |

Versions are physical source/config/API boundaries. A future version must copy
and own its complete implementation and relevant tests; no mutable shared
implementation or forwarding to another version is permitted. Experiment
checkpoints and seeds identify trained instances separately from design tags.

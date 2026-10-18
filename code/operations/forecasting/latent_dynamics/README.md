<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../LICENSE. -->
# Dynamics Model

The Dynamics Model maps a chronological state history into direct future-state
forecasts. It is separate from the State Representation Network and the reserved
Portfolio Policy, which is not implemented.

| Version | Physical implementation | Namespace | Contract |
| --- | --- | --- | --- |
| v1 | [versions/v1](versions/v1/README.md) | `cuwacunu::networks::latent_dynamics::v1` | Small independent affine transition per horizon, optional declared last-state residual |

Each future version owns a complete implementation and relevant tests in its
own physical folder. No version forwards into a mutable shared implementation.
This design version does not imply an admitted market checkpoint or forecast
skill. Probabilistic dynamics are a future separately versioned design.

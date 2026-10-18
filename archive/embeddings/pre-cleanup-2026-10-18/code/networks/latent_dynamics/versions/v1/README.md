<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE. -->
# Deterministic Dynamics Model v1

This physical version owns `cuwacunu::networks::latent_dynamics::v1`.
`AffineDynamics(Config)` accepts `[B,K,d]` in chronological order and emits
`[B,Q,d]` in the explicitly supplied horizon order. The learned weight has
shape `[Q*d,K*d]`, and the bias has shape `[Q*d]`. Each horizon has independent
coefficients. Positive, distinct horizons are mandatory; this is a collection
of direct forecasts rather than a recursive rollout.

With `residual=false`, initialization is LibTorch Linear's seeded initializer.
With `residual=true`, learned coefficients start at zero and the last input
state is added to every horizon. The latter changes parameter regularization
and optimization and must be selected prospectively. `absolute_weight()`
includes the last-state residual for exact fixed-checkpoint serving algebra.

This model neither selects a history spacing nor checks timestamp support:
those belong to the admitted timeline/pair ledger. It receives no future
observations or target validity in its forward method. It does not produce a
probability distribution or an uncertainty estimate. Future versions own
complete implementations in separate folders. The coupled engineering target
is `test-predictive-v1` in the managed integration container.

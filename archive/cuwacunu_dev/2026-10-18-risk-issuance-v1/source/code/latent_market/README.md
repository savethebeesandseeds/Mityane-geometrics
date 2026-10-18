<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE. -->
# Latent market integration

The deterministic backbone is explicitly `versions/v1/`. Read the
[implementation and usage guide](../../doc/LATENT_MARKET_DYNAMICS_IMPLEMENTATION_V1.md)
and [research roadmap](../../doc/LATENT_MARKET_DYNAMICS_ROADMAP.md).

This component owns market time/support/availability, immutable artifacts and
application runners. The reusable State Representation Network, Target State
Representation Network, Dynamics Model and predictive optimizer state belong to
the separately versioned `code/networks` components. Evaluation consumes
saved arrays independently. New versions receive independent source directories;
no package entry selects a mutable latest implementation.

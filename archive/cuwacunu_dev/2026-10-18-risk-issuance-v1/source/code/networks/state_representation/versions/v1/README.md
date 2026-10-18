<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE. -->
# State Representation Network v1

This physical version owns `cuwacunu::networks::state_representation::v1`.
Future versions must own complete implementations and tests in their own
folders; this version does not forward to a mutable implementation elsewhere.

## Mathematical mapping

`AffineState(Config, embedding_mean, embedding_scale)` maps native embeddings
with final axis D into learned states with final axis d:
`G(e) = W ((e - mean) / scale) + b`. Both transform buffers are detached,
copied, finite and frozen. The caller fits and identifies them on TRAIN before
quality training; this module never fits or modifies the Encoder or its output.
Inputs can be `[B,D]`, `[B,K,D]` or `[B,Q,D]`. Equal widths permit an explicit
identity initialization after the transform. Other widths require declared
nonidentity initialization. Float32/float64 and a device are configurable.

## Target State Representation Network and serving

The whole d-dimensional output is the state. Native Encoder blocks are not
implicitly retained, and equal widths do not imply invertibility. `frozen_copy`
creates the exact detached Target State Representation Network. `native_affine()`
returns the equivalent native-e coefficients including preprocessing, without fitting.

The coupled engineering fixture lives in
`predictive_training/versions/v1/tests/v1_test.cpp`; its Make target is
`test-predictive-v1`. No quality result is implied by this implementation.

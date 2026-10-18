// SPDX-License-Identifier: MIT
#pragma once
#include "structured_hard_timing.h"

namespace embedding::evaluation {
inline constexpr const char *kCrossFeatureSolvabilityRule =
    "observed-cross-feature-determinant-v2";

// Observed-only rule: each channel independently uses features with a legal
// three-time triplet. Mean left/right differences precede the determinant,
// which equals averaging all legal cross-channel feature-pair determinants.
// Spacing/centre reductions and the >=4-centre/nonzero-margin support remain
// explicit. Invalid rows retain prediction0. No labels or fitted inputs.
HardTimingSolvabilityResult cross_feature_solvability(const Batch &observations);
} // namespace embedding::evaluation

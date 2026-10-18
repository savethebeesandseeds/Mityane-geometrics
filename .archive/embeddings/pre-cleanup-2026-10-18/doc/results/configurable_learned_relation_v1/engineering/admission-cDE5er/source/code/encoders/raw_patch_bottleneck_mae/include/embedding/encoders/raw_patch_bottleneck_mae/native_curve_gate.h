// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <string>

namespace embedding::encoders::raw_patch_bottleneck_mae {

// A bounded, independent learned-global preflight. Requires explicit CUDA, one
// channel mixer layer, learned global mode 2 or 3 and native width 32. Uses only separately
// generated legal TRAIN/VALIDATION observations; no TEST data or classifier fit.
// The caller's settings are unchanged. Saves ordinary points 0/2/4 and serving
// witnesses in a NEW directory, restores RNG/thread state, and returns the JSON
// also written to gpu-check.json. This is engineering evidence, not quality.
std::string run_native_curve_cuda_gate(const Settings &settings,
                                       const std::string &output_directory);

} // namespace embedding::encoders::raw_patch_bottleneck_mae

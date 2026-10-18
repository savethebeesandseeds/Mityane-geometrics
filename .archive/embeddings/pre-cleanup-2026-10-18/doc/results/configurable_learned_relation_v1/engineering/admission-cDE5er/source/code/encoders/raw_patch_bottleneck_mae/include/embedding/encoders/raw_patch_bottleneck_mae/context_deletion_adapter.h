// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/paired_pooling.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

// RPB-v6 has the exact RPB-v4 inference architecture. Every initialized named
// parameter/buffer, including global pooling, must match retained mode2 point0.
// Original TRAIN/scaler/source order and row/patch/Torch counter streams match;
// the additional training-only deletion policy is verified separately against
// the checkpoint tag and its producer companion. No fit, training or TEST access.
std::map<std::string,std::string> audit_context_deletion_initialization(
    const std::string &candidate_point0,
    const embedding::evaluation::RetainedPoolingCohort &reference,
    const embedding::evaluation::ProviderFitInput &training_only);

} // namespace embedding::encoders::raw_patch_bottleneck_mae

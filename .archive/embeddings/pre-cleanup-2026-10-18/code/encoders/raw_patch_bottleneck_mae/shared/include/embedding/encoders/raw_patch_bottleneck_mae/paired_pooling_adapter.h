// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/paired_pooling.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

// Loads the retained model/scaler without fitting or training. Exact CPU serving
// and CUDA reconstruction copies are independent of the candidate trainer.
embedding::evaluation::CurveSnapshot make_retained_curve_snapshot(
    const std::string &checkpoint_path,
    const embedding::evaluation::ProviderFitInput &metadata);

// Requires identical common point-zero weights, TRAIN dataset/scaler and counter
// streams; excludes only the two architecture-specific global pooling modules.
std::map<std::string,std::string> audit_pooling_initialization(
    const std::string &candidate_point0,
    const embedding::evaluation::RetainedPoolingCohort &reference,
    const embedding::evaluation::ProviderFitInput &metadata);

} // namespace embedding::encoders::raw_patch_bottleneck_mae

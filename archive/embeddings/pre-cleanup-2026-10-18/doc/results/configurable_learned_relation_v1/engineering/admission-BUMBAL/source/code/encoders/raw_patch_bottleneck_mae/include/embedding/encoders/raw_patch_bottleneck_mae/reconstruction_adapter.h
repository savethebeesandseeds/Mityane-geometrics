// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/reconstruction_evaluation.h"
#include <cstdint>
#include <string>

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct ReconstructionOptions {
  std::string config_path;
  int64_t pretraining_updates{128};
  int64_t metadata_updates{128};
  int64_t batch_size{8};
};

struct ReconstructionRegistration {
  embedding::evaluation::ReconstructionProviderFactory factory;
  std::string recipe;
};

// Resolve configuration and freeze the recipe before the shared engine fits
// anything. Fresh training only; decoder interventions and scores stay shared.
ReconstructionRegistration make_reconstruction_provider(
    const ReconstructionOptions &options,
    const embedding::evaluation::ReconstructionCard &card);

} // namespace embedding::encoders::raw_patch_bottleneck_mae

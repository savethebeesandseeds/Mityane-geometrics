// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_evaluation.h"
#include <cstdint>
#include <string>

namespace embedding::encoders::mtf_jepa_mae_vicreg {

struct EvaluationOptions {
  std::string config_path;
  int64_t pretraining_updates{0};
};

// Evaluation-only adapter. Receives legal training observations, never labels,
// hidden clean signals or held-out observations; legacy evaluation stays separate.
embedding::evaluation::FeatureProviderFactory
make_evaluation_provider(const EvaluationOptions &options = {});

} // namespace embedding::encoders::mtf_jepa_mae_vicreg

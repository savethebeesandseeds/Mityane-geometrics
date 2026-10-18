// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/mtf_jepa_mae_vicreg/workflow.h"
#include "embedding/shared/evaluation.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {

// Encoder configuration adapter; shared generation accepts an explicit shape.
::embedding::SyntheticEvaluationData make_evaluation_data(
    const Config &config, int64_t samples, uint64_t signal_seed,
    uint64_t mask_seed, uint64_t label_seed);

int run_evaluation_cli(int argc, char **argv);

} // namespace embedding::encoders::mtf_jepa_mae_vicreg

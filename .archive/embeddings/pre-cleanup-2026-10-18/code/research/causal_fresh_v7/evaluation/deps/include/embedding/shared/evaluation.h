// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/data.h"
#include <cstdint>

namespace embedding {

// Label, signal and missingness have separate random streams. Each independent
// trajectory supplies exactly one history, so no split shares a source window.
struct SyntheticEvaluationData {
  Batch clean;
  Batch observed;
  torch::Tensor labels; // CPU int64 [B], four balanced regimes, shuffled.
};

// Evaluation fixtures are CPU float32; only the C/H/F dimensions are used.
SyntheticEvaluationData make_evaluation_data(const input_shape_t &shape, int64_t samples,
                                             uint64_t signal_seed, uint64_t mask_seed,
                                             uint64_t label_seed);

struct RepresentationDiagnostics {
  int64_t valid_rows{0};
  int64_t dimensions{0};
  double valid_fraction{0};
  double std_mean{0}, std_min{0}, std_max{0};
  double norm_mean{0}, norm_min{0}, norm_max{0};
  double covariance_effective_rank{0};
  torch::Tensor per_dimension_std;
};

// [B,D] or [B,C,D] with a matching validity mask. Invalid zero rows are excluded.
RepresentationDiagnostics representation_diagnostics(const torch::Tensor &values,
                                                       const torch::Tensor &valid);

} // namespace embedding

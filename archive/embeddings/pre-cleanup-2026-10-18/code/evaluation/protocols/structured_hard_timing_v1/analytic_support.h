// SPDX-License-Identifier: MIT
#pragma once
#include <torch/torch.h>

namespace embedding::evaluation::structured_timing {
// Invalid rows retain the solver's explicit zero prediction. A positive margin
// alone does not establish support when fewer than four centers are observed.
inline torch::Tensor supported_analytic_predictions(const torch::Tensor &valid,
                                                   const torch::Tensor &margins) {
  return torch::where(valid, margins.gt(0).to(torch::kInt64),
                      torch::zeros_like(margins, torch::kInt64));
}
}

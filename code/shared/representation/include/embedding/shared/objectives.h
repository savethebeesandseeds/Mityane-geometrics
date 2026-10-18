// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/tensor_ops.h"

#include <algorithm>

namespace embedding {

namespace vicreg_stability_detail {

inline torch::Tensor valid_rows(const torch::Tensor &z,
                                const torch::Tensor &mask) {
  TORCH_CHECK(z.defined(), "[embedding] stability z is undefined");
  TORCH_CHECK(mask.defined(),
              "[embedding] stability mask is undefined");
  TORCH_CHECK(z.dim() == 3,
              "[embedding] stability z must be [B,C,D]");
  TORCH_CHECK(mask.dim() == 2,
              "[embedding] stability mask must be [B,C]");
  TORCH_CHECK(z.size(0) == mask.size(0) && z.size(1) == mask.size(1),
              "[embedding] stability z/mask shape mismatch");
  const auto rows = mask.to(torch::kBool).reshape({-1}).nonzero().reshape({-1});
  return z.reshape({-1, z.size(2)}).index_select(/*dim=*/0, rows);
}

inline torch::Tensor variance_term(const torch::Tensor &z, double floor,
                                   double eps) {
  const auto std = torch::sqrt(z.var(/*dim=*/0, /*unbiased=*/false) + eps);
  return torch::relu(floor - std).mean();
}

inline torch::Tensor covariance_term(const torch::Tensor &z) {
  const auto rows = z.size(0);
  const auto dim = z.size(1);
  if (rows <= 1 || dim <= 1) {
    return torch::zeros({}, z.options());
  }
  const auto centered = z - z.mean(/*dim=*/0, /*keepdim=*/true);
  const auto cov =
      centered.transpose(0, 1).matmul(centered) / static_cast<double>(rows - 1);
  const auto off_diag = cov - torch::diag(torch::diag(cov));
  return off_diag.pow(2).sum() / static_cast<double>(dim);
}

} // namespace vicreg_stability_detail

[[nodiscard]] inline vicreg_stability_loss_result_t
compute_vicreg_stability_loss(
    const torch::Tensor &z1, const torch::Tensor &mask1,
    const torch::Tensor &z2, const torch::Tensor &mask2,
    const vicreg_stability_loss_options_t &options = {}) {
  TORCH_CHECK(z1.sizes() == z2.sizes(),
              "[embedding] stability z1/z2 shape mismatch");
  TORCH_CHECK(mask1.sizes() == mask2.sizes(),
              "[embedding] stability mask1/mask2 shape mismatch");
  const auto joint_mask =
      mask1.to(torch::kBool).logical_and(mask2.to(torch::kBool));
  auto rows1 = vicreg_stability_detail::valid_rows(z1, joint_mask);
  auto rows2 = vicreg_stability_detail::valid_rows(z2, joint_mask);

  vicreg_stability_loss_result_t out{};
  out.valid_rows = std::min(rows1.size(0), rows2.size(0));
  out.statistics_supported = out.valid_rows >= 2;
  if (out.valid_rows <= 0) {
    out.loss = torch::zeros({}, z1.options());
    out.invariance_loss = torch::zeros({}, z1.options());
    out.variance_loss = torch::zeros({}, z1.options());
    out.covariance_loss = torch::zeros({}, z1.options());
    return out;
  }
  if (rows1.size(0) != out.valid_rows) {
    rows1 = rows1.narrow(0, 0, out.valid_rows);
  }
  if (rows2.size(0) != out.valid_rows) {
    rows2 = rows2.narrow(0, 0, out.valid_rows);
  }

  out.invariance_loss = torch::mse_loss(rows1, rows2);
  out.variance_loss = 0.5 * (vicreg_stability_detail::variance_term(
                                 rows1, options.variance_floor, options.eps) +
                             vicreg_stability_detail::variance_term(
                                 rows2, options.variance_floor, options.eps));
  out.covariance_loss = 0.5 * (vicreg_stability_detail::covariance_term(rows1) +
                               vicreg_stability_detail::covariance_term(rows2));
  out.loss = options.invariance_weight * out.invariance_loss +
             options.variance_weight * out.variance_loss +
             options.covariance_weight * out.covariance_loss;
  return out;
}

} // namespace embedding

// SPDX-License-Identifier: MIT
#include "embedding/encoders/mtf_jepa_mae_vicreg/tokenization.h"
#include "test_support.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace mtf = embedding::encoders::mtf_jepa_mae_vicreg;

namespace {

template <typename Function>
void rejects(Function function, const std::string &label) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception &) {
    rejected = true;
  }
  test::check(rejected, label + " was accepted");
}

torch::Tensor descriptor(const std::vector<float> &values) {
  const auto patch = torch::tensor(values, torch::kFloat32)
                         .view({1, static_cast<int64_t>(values.size()), 1});
  return mtf::detail::masked_patch_descriptor(
      patch, torch::ones_like(patch, torch::kBool));
}

void test_time_descriptor_range() {
  const auto offset = descriptor({9999998.0f, 9999999.0f,
                                  10000001.0f, 10000002.0f});
  test::check(offset.scalar_type() == torch::kFloat32,
              "time descriptor changed public dtype");
  test::close(offset, torch::tensor({{10000000.0f,
                                     static_cast<float>(std::sqrt(2.5 + 1e-6))}}),
              "small variation around large offset", 1e-6, 1e-6);

  const float maximum = std::numeric_limits<float>::max();
  const auto extreme = descriptor({maximum, -maximum, maximum, -maximum});
  test::finite(extreme, "extreme finite time descriptor");
  test::close(extreme, torch::tensor({{0.0f, maximum}}),
              "symmetric float32 extremes", 1e-6, 1e-6);

  const auto constant = descriptor({maximum, maximum, maximum, maximum});
  test::finite(constant, "large constant time descriptor");
  test::close(constant, torch::tensor({{maximum, 0.001f}}),
              "constant window variance", 1e-6, 1e-6);

  const float center = 1e20f;
  const float lower = std::nextafter(center, -std::numeric_limits<float>::infinity());
  const float upper = std::nextafter(center, std::numeric_limits<float>::infinity());
  const auto adjacent = descriptor({lower, upper});
  const float expected_mean = static_cast<float>(
      (static_cast<double>(lower) + static_cast<double>(upper)) / 2.0);
  const float expected_std = static_cast<float>(
      (static_cast<double>(upper) - static_cast<double>(lower)) / 2.0);
  test::finite(adjacent, "representable variation at huge offset");
  test::close(adjacent, torch::tensor({{expected_mean, expected_std}}),
              "adjacent float32 values retain variation", 1e-6, 1e-6);
}

void test_time_descriptor_missing_features() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto patch = torch::tensor(
      std::vector<float>{1.0f, nan, nan, inf, 3.0f, -inf, inf, nan},
      torch::kFloat32).view({1, 4, 2});
  const auto mask = torch::tensor(
      {{true, false}, {false, false}, {true, false}, {false, false}},
      torch::kBool).unsqueeze(0);
  const auto partial = mtf::detail::masked_patch_descriptor(patch, mask);
  test::finite(partial, "partially observed time descriptor");
  test::close(partial,
              torch::tensor({{2.0f, 0.0f,
                              static_cast<float>(std::sqrt(1.0 + 1e-6)), 0.001f}}),
              "partial and empty feature descriptors", 1e-6, 1e-6);
  const auto empty = mtf::detail::masked_patch_descriptor(
      patch, torch::zeros_like(mask));
  test::close(empty, torch::tensor({{0.0f, 0.0f, 0.001f, 0.001f}}),
              "empty time descriptor", 1e-6, 1e-6);
}

void test_frequency_descriptor_range() {
  auto config = test::small_config<mtf::Config>();
  config.channel_count = 1;
  config.history_length = 8;
  config.input_width = 1;
  config.time_scales = {8};
  config.scale_strides = {8};
  config.frequency_num_bins = 5;
  config.frequency_log_magnitude = false;
  auto tokenizer = mtf::FrequencyTokenizer(config);
  torch::NoGradGuard no_grad;
  const auto signal = torch::tensor(
      std::vector<float>{1, 3, 1, -1, 1, 3, 1, -1}, torch::kFloat32)
                          .view({1, 1, 8, 1});
  const auto observed = torch::ones_like(signal, torch::kBool);
  const auto normal = tokenizer->forward(signal, observed);
  const auto large = tokenizer->forward(signal * 1e20f, observed);
  test::finite(large.frequency_reconstruction_targets,
               "large baseline/amplitude frequency descriptor");
  test::check(large.frequency_reconstruction_targets.scalar_type() == torch::kFloat32,
              "frequency descriptor changed public dtype");
  // Magnitudes scale linearly; epsilon matters only around zero magnitude.
  test::close(large.frequency_reconstruction_targets / 1e20f,
              normal.frequency_reconstruction_targets,
              "frequency amplitude scaling", 1e-5, 1e-4);

  auto mask = observed.clone();
  mask.index_put_({0, 0, 3, 0}, false);
  const auto poisoned = (signal * 1e20f).masked_fill(
      mask.logical_not(), std::numeric_limits<float>::quiet_NaN());
  test::finite(tokenizer->forward(poisoned, mask).frequency_reconstruction_targets,
               "partially observed large frequency descriptor");
  test::finite(tokenizer->forward(poisoned, torch::zeros_like(mask))
                   .frequency_reconstruction_targets,
               "empty frequency descriptor");

  config.frequency_log_magnitude = true;
  tokenizer = mtf::FrequencyTokenizer(config);
  test::finite(tokenizer->forward(signal * 1e20f, observed)
                   .frequency_reconstruction_targets,
               "large log-magnitude frequency descriptor");
}

struct ScalarField {
  const char *name;
  double mtf::Config::*member;
};

void test_public_config_scalars() {
  const ScalarField fields[] = {
      {"dropout", &mtf::Config::dropout},
      {"mask_ratio_time", &mtf::Config::mask_ratio_time},
      {"mask_ratio_frequency", &mtf::Config::mask_ratio_frequency},
      {"mask_ratio_channel", &mtf::Config::mask_ratio_channel},
      {"min_context_ratio", &mtf::Config::min_context_ratio},
      {"lambda_jepa", &mtf::Config::lambda_jepa},
      {"lambda_mae", &mtf::Config::lambda_mae},
      {"lambda_tf_align", &mtf::Config::lambda_tf_align},
      {"lambda_vicreg", &mtf::Config::lambda_vicreg},
      {"vicreg_sim_weight", &mtf::Config::vicreg_sim_weight},
      {"vicreg_var_weight", &mtf::Config::vicreg_var_weight},
      {"vicreg_cov_weight", &mtf::Config::vicreg_cov_weight},
      {"vicreg_variance_floor", &mtf::Config::vicreg_variance_floor},
      {"vicreg_variance_epsilon", &mtf::Config::vicreg_variance_epsilon},
      {"target_ema_tau", &mtf::Config::target_ema_tau},
      {"lambda_global_vicreg", &mtf::Config::lambda_global_vicreg},
      {"lambda_channel_vicreg", &mtf::Config::lambda_channel_vicreg},
      {"max_context_target_time_overlap", &mtf::Config::max_context_target_time_overlap},
      {"vicreg_view_gaussian_jitter_std", &mtf::Config::vicreg_view_gaussian_jitter_std},
      {"vicreg_view_time_dropout_scale", &mtf::Config::vicreg_view_time_dropout_scale}};
  const double invalid[] = {std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity()};
  for (const auto &field : fields) {
    for (const auto value : invalid) {
      auto config = mtf::Config{};
      config.*(field.member) = value;
      rejects([&] { mtf::validate_config(config); },
              std::string("non-finite ") + field.name);
    }
  }
  const ScalarField weights[] = {
      {"lambda_jepa", &mtf::Config::lambda_jepa},
      {"lambda_mae", &mtf::Config::lambda_mae},
      {"lambda_tf_align", &mtf::Config::lambda_tf_align},
      {"lambda_vicreg", &mtf::Config::lambda_vicreg},
      {"lambda_global_vicreg", &mtf::Config::lambda_global_vicreg},
      {"lambda_channel_vicreg", &mtf::Config::lambda_channel_vicreg},
      {"vicreg_sim_weight", &mtf::Config::vicreg_sim_weight},
      {"vicreg_var_weight", &mtf::Config::vicreg_var_weight},
      {"vicreg_cov_weight", &mtf::Config::vicreg_cov_weight}};
  for (const auto &field : weights) {
    auto config = mtf::Config{};
    config.*(field.member) = -1.0;
    rejects([&] { mtf::validate_config(config); },
            std::string("negative ") + field.name);
    config.*(field.member) = 0.0;
    mtf::validate_config(config);
  }
  for (const auto member : {&mtf::Config::vicreg_variance_floor,
                            &mtf::Config::vicreg_variance_epsilon}) {
    auto config = mtf::Config{};
    config.*member = 0.0;
    rejects([&] { mtf::validate_config(config); }, "zero variance option");
  }
  test::check(!mtf::Config{}.strict_jepa_support,
              "strict JEPA support must be opt-in");
}

} // namespace

int main() {
  try {
    torch::set_num_threads(1);
    test_time_descriptor_range();
    test_time_descriptor_missing_features();
    test_frequency_descriptor_range();
    test_public_config_scalars();
    std::cout << "PASS: stable descriptors and finite public configuration\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

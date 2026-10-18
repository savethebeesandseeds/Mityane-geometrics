// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_model.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/spectral_temporal_relation.h"
#include <torch/cuda.h>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
void check(bool value, const char *message) {
  TORCH_CHECK(value, "[rpb-v20 CUDA test] ", message);
}
template <typename Callable> void rejects(Callable action, const char *message) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  check(rejected, message);
}
rpb::Input fixture(const rpb::Config &c, int64_t batch = 2) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat64);
  auto t = torch::arange(c.history_length, options).reshape({1, 1, c.history_length, 1});
  auto channels = torch::arange(c.channel_count, options).reshape({1, c.channel_count, 1, 1});
  auto features = torch::arange(c.input_width, options).reshape({1, 1, 1, c.input_width});
  auto examples = torch::arange(batch, options).reshape({batch, 1, 1, 1});
  const auto angle = t * (6.283185307179586476925286766559 / c.history_length);
  auto data = (angle * 2. + channels * .42 + features * .07 + examples * .13).sin() *
                  (1. + features * .1) +
              (angle * 6. - channels * .25 + features * .12).cos() * .6 + features * .2;
  auto observed = torch::ones_like(data, torch::kBool);
  observed[0][c.channel_count - 1].narrow(0, 9, 3).fill_(false);
  return {data, observed, torch::tensor(rpb::resolved_channel_ids(c), torch::kInt64),
          torch::full({batch}, double(c.history_length - 1), options), c.sampling_interval};
}
torch::Tensor native(rpb::ConfigurableSpectralModel &model, const rpb::Input &input) {
  return rpb::compact_reconstruction_export(model->encode(input), model->config());
}

void original_shape_parity() {
  auto c = rpb::default_configurable_spectral_config();
  c.history_length = 32;
  c.input_width = 3;
  c.export_width = 32;
  torch::manual_seed(20001);
  rpb::SpectralTemporalRelationModel original(c);
  torch::manual_seed(20001);
  rpb::ConfigurableSpectralModel generalized(c);
  const auto before = original->named_parameters();
  const auto after = generalized->named_parameters();
  check(before.size() == after.size(), "old-shape parameter count and names match");
  for (const auto &parameter : before) {
    check(after.contains(parameter.key()) && torch::equal(parameter.value(), after[parameter.key()]),
          "old-shape initialization consumes the same RNG and has identical parameters");
  }
  const auto raw = fixture(c);
  const auto scaler = rpb::fit_configurable_spectral_scaler(raw, c);
  const auto input = rpb::transform_configurable_spectral_input(scaler, raw, c);
  original->eval(); generalized->eval();
  torch::NoGradGuard no_grad;
  const auto expected = rpb::compact_reconstruction_export(original->encode(input), c);
  check(torch::equal(expected, native(generalized, input)), "old-shape native embedding is bit-exact");
  check(torch::equal(original->decode(expected, input.channel_ids),
                     generalized->decode(expected, input.channel_ids)),
        "old-shape reconstruction is bit-exact");
  const auto old_bank = rpb::spectral_relation_bank(input, c);
  const auto new_bank = rpb::configurable_spectral_relation_bank(input, c);
  check(torch::equal(old_bank.values, new_bank.values) &&
            torch::equal(old_bank.common_counts, new_bank.common_counts) &&
            torch::equal(old_bank.eligible, new_bank.eligible), "old-shape spectral bank is bit-exact");
  auto changed = original->native_shape_projection->weight;
  changed[0][1].add_(.125f);
  torch::serialize::OutputArchive weights;
  original->save(weights);
  std::stringstream bytes;
  weights.save_to(bytes);
  torch::serialize::InputArchive restored;
  restored.load_from(bytes, c.device);
  generalized->load(restored);
  generalized->validate_fixed_prior();
  check(torch::equal(rpb::compact_reconstruction_export(original->encode(input), c),
                     native(generalized, input)), "old-shape raw weights load with bit-exact parity");
  const auto mask = rpb::make_training_mask(input.observed, c, 20002);
  const auto extended_mask = rpb::configurable_spectral_make_training_mask(input.observed, c, 20002);
  check(torch::equal(mask.hidden, extended_mask.hidden) &&
            torch::equal(mask.visible, extended_mask.visible) &&
            torch::equal(mask.target, extended_mask.target) &&
            torch::equal(mask.eligible_channels, extended_mask.eligible_channels),
        "old-shape training masks preserve patch semantics exactly");
  const auto old_forward = original->forward(input, mask.hidden);
  const auto new_forward = generalized->forward(input, mask.hidden);
  check(torch::equal(old_forward.reconstruction, new_forward.reconstruction) &&
            torch::equal(old_forward.loss, new_forward.loss), "old-shape masked objective is bit-exact");
}

void configurable_shapes_and_masks() {
  auto c = rpb::default_configurable_spectral_config();
  c.history_length = 129;
  c.export_width = 64;
  check(rpb::configurable_spectral_shape_width(c) == 52 &&
            rpb::configurable_spectral_width(c) == 12, "D64 retains shape52 plus spectral12");
  check(rpb::configurable_spectral_backbone_config(c).history_length == 136,
        "H129 pads privately to 136 with P8");
  auto raw = fixture(c);
  const auto scaler = rpb::fit_configurable_spectral_scaler(raw, c);
  const auto input = rpb::transform_configurable_spectral_input(scaler, raw, c);
  rpb::ConfigurableSpectralModel model(c);
  model->eval();
  torch::Tensor baseline;
  {
    torch::NoGradGuard no_grad;
    baseline = native(model, input);
    check(baseline.is_cuda() && baseline.sizes() == torch::IntArrayRef({2, 64}) &&
              torch::isfinite(baseline).all().item<bool>(), "nine-feature long history has finite CUDA native64");
    check(model->decode(baseline, input.channel_ids).sizes() == torch::IntArrayRef({2, 3, 129, 9}),
          "reconstruction crops private padding to the requested history");
    const auto encoded = model->encode(input);
    check(torch::equal(encoded.visible_observation_counts,
              input.observed.sum(std::vector<int64_t>{2, 3})), "padding does not add observed support");
    auto poisoned = input;
    poisoned.data = torch::where(input.observed, input.data,
        torch::full_like(input.data, std::numeric_limits<float>::quiet_NaN()));
    check(torch::equal(baseline, native(model, poisoned)), "masked NaNs never reach the embedding");
    poisoned.observed = torch::zeros_like(input.observed);
    const auto absent = model->encode(poisoned);
    check(!absent.sample_valid_mask.any().item<bool>() &&
              absent.z_contextual_global.eq(0).all().item<bool>(), "absent windows are invalid exact zero");
    auto permuted = input;
    auto order = torch::tensor({2, 0, 1}, input.channel_ids.options());
    permuted.data = input.data.index_select(1, order);
    permuted.observed = input.observed.index_select(1, order);
    permuted.channel_ids = input.channel_ids.index_select(0, order);
    check(torch::allclose(baseline, native(model, permuted), 1e-5, 1e-6),
          "semantic channel identity survives physical channel permutation");
    const auto bank = rpb::configurable_spectral_relation_bank(input, c);
    const auto reordered = rpb::configurable_spectral_relation_bank(permuted, c);
    check(torch::equal(bank.values, reordered.values), "canonical spectral pairs are exactly permutation invariant");
    for (int64_t f = 0; f < 9; ++f) {
      auto perturbed = input;
      perturbed.data = input.data.clone();
      auto changed = perturbed.data.select(3, f);
      auto t = torch::arange(c.history_length, changed.options()).reshape({1, 1, c.history_length});
      changed.add_((t * .231 + f * .071).sin() * .7);
      check(!torch::equal(baseline, native(model, perturbed)), "each of the nine features influences the embedding");
    }
    auto feature_reordered = input;
    const auto feature_order = torch::arange(8, -1, -1, input.channel_ids.options());
    feature_reordered.data = input.data.index_select(3, feature_order);
    feature_reordered.observed = input.observed.index_select(3, feature_order);
    const auto feature_bank = rpb::configurable_spectral_relation_bank(feature_reordered, c);
    check(torch::allclose(model->grouped_odd_relation_projection(bank.values),
                         model->grouped_odd_relation_projection(feature_bank.values), 1e-6, 1e-7),
          "fixed spectral mean includes all feature pairs symmetrically");
    const auto single = rpb::transform_configurable_spectral_input(scaler, fixture(c, 1), c);
    check(native(model, single).sizes() == torch::IntArrayRef({1, 64}), "batch size remains independently variable");
  }
  auto last_patch = torch::zeros_like(input.observed);
  last_patch.narrow(2, 128, 1).fill_(true);
  const auto partial = rpb::configurable_spectral_mask_from_hidden(input.observed, last_patch, c);
  check(partial.target.sum().item<int64_t>() == 2 * 3 * 9 &&
            partial.eligible_channels.all().item<bool>(), "partial final patch targets only real observed ticks");
  auto incomplete = last_patch.clone();
  incomplete[0][0][128][0] = false;
  rejects([&] { rpb::configurable_spectral_mask_from_hidden(input.observed, incomplete, c); },
          "partial feature masks cannot masquerade as complete artificial patches");
  torch::serialize::OutputArchive saved;
  model->save(saved);
  std::stringstream bytes;
  saved.save_to(bytes);
  rpb::ConfigurableSpectralModel replay(c);
  torch::serialize::InputArchive loaded;
  loaded.load_from(bytes, c.device);
  replay->load(loaded);
  replay->eval();
  {
    torch::NoGradGuard no_grad;
    check(torch::equal(baseline, native(replay, input)), "configurable raw weights roundtrip exactly");
  }
  const auto frozen = model->grouped_odd_relation_projection->weight.detach().clone();
  const auto shape = model->native_shape_projection->weight.detach().clone();
  model->train();
  torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(.001));
  const auto plan = rpb::configurable_spectral_make_training_mask(input.observed, c, 20004);
  optimizer.zero_grad();
  const auto result = model->forward(input, plan.hidden);
  check(result.loss.is_cuda() && torch::isfinite(result.loss).item<bool>() &&
            result.eligible_example_count == 2 && result.target_cell_count > 0,
        "masked reconstruction trains on real nine-feature CUDA targets");
  result.loss.backward();
  const auto gradient = model->native_shape_projection->weight.grad();
  check(gradient.defined() && gradient.is_cuda() && torch::isfinite(gradient).all().item<bool>() &&
            gradient.abs().sum().item<double>() > 0, "learned shape has finite nonzero CUDA gradient");
  check(!model->grouped_odd_relation_projection->weight.grad().defined(), "fixed spectral prior has no gradient");
  for (const auto &parameter : model->parameters()) {
    check(parameter.is_cuda(), "all model parameters stay on CUDA");
    if (parameter.grad().defined())
      check(torch::isfinite(parameter.grad()).all().item<bool>(), "all parameter gradients are finite");
  }
  torch::nn::utils::clip_grad_norm_(model->parameters(), 1., 2., true);
  optimizer.step();
  model->validate_fixed_prior();
  check(torch::equal(frozen, model->grouped_odd_relation_projection->weight) &&
            !torch::equal(shape, model->native_shape_projection->weight),
        "one disposable optimizer update preserves the prior and updates learned shape");
}

void additional_geometry() {
  for (const auto &dimensions : std::vector<std::vector<int64_t>>{
           {1, 17, 9, 8}, {3, 130, 9, 32}, {4, 65, 9, 48}, {2, 257, 9, 16}}) {
    auto c = rpb::default_configurable_spectral_config();
    c.channel_count = dimensions[0]; c.history_length = dimensions[1];
    c.input_width = dimensions[2]; c.export_width = dimensions[3];
    auto raw = fixture(c, 1);
    const auto scaler = rpb::fit_configurable_spectral_scaler(raw, c);
    const auto input = rpb::transform_configurable_spectral_input(scaler, raw, c);
    rpb::ConfigurableSpectralModel model(c);
    model->eval();
    torch::NoGradGuard no_grad;
    const auto encoded = native(model, input);
    check(encoded.sizes() == torch::IntArrayRef({1, c.export_width}) &&
              torch::isfinite(encoded).all().item<bool>(), "configured C/H/D have finite native output");
    check(model->decode(encoded, input.channel_ids).sizes() == raw.data.sizes(),
          "configured reconstruction matches the complete logical shape");
    model->validate_fixed_prior();
    if (c.channel_count == 1)
      check(rpb::configurable_spectral_width(c) == 0 &&
                model->grouped_odd_relation_projection->weight.numel() == 0,
            "single channel uses the full learned embedding without invented pair relations");
  }
  auto c = rpb::default_configurable_spectral_config();
  c.history_length = 16;
  rejects([&] { rpb::ConfigurableSpectralModel invalid(c); }, "degenerate eight-harmonic histories are rejected");
  c = rpb::default_configurable_spectral_config(); c.export_width = 12;
  rejects([&] { rpb::ConfigurableSpectralModel invalid(c); }, "native width must retain learned shape capacity");
  c = rpb::default_configurable_spectral_config(); c.device = torch::Device(torch::kCPU);
  rejects([&] { rpb::ConfigurableSpectralModel invalid(c); }, "CPU model execution is rejected");
}
} // namespace

int main() try {
  check(torch::cuda::is_available(), "actual CUDA is required");
  torch::set_num_threads(1);
  original_shape_parity();
  configurable_shapes_and_masks();
  additional_geometry();
  std::cout << "RPB-v20 CUDA checks passed: bit-exact v19 old-shape weights/arithmetic; "
               "all 9 features; H17/65/129/130/257; D8/16/32/48/64; "
               "C1/2/3/4; masks and partial patches; roundtrip; one disposable optimizer update\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

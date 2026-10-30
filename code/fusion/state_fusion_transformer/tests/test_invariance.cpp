#include "state_fusion/model.h"
#include "state_fusion/synthetic.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr double kClose = 1e-4;
constexpr double kGap = 1e-4;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

state_fusion::SetBatch clone_batch(const state_fusion::SetBatch& batch) {
  state_fusion::SetBatch out;
  out.scalars = batch.scalars.clone();
  out.scalar_observed = batch.scalar_observed.clone();
  out.categoricals = batch.categoricals.clone();
  out.categorical_observed = batch.categorical_observed.clone();
  out.semantic = batch.semantic.clone();
  out.semantic_observed = batch.semantic_observed.clone();
  out.record_observed = batch.record_observed.clone();
  return out;
}

state_fusion::SetBatch permute_records(const state_fusion::SetBatch& batch, const torch::Tensor& perm) {
  state_fusion::SetBatch out;
  out.scalars = batch.scalars.index_select(1, perm);
  out.scalar_observed = batch.scalar_observed.index_select(1, perm);
  out.categoricals = batch.categoricals.index_select(1, perm);
  out.categorical_observed = batch.categorical_observed.index_select(1, perm);
  out.semantic = batch.semantic.index_select(1, perm);
  out.semantic_observed = batch.semantic_observed.index_select(1, perm);
  out.record_observed = batch.record_observed.index_select(1, perm);
  return out;
}

double max_abs(const torch::Tensor& left, const torch::Tensor& right) {
  require(torch::isfinite(left).all().item<bool>() && torch::isfinite(right).all().item<bool>(),
          "comparison received a non-finite state");
  return (left - right).abs().max().item<double>();
}

state_fusion::SetBatch manual_batch(const state_fusion::Config& config, torch::Device device) {
  constexpr int64_t kBatch = 4;
  constexpr int64_t kRecords = 5;
  const auto floats = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto flags = torch::TensorOptions().dtype(torch::kBool).device(device);
  const auto ids = torch::TensorOptions().dtype(torch::kLong).device(device);
  const auto fields = static_cast<int64_t>(config.categorical_cardinalities.size());

  state_fusion::SetBatch batch;
  batch.scalars = torch::randn({kBatch, kRecords, config.scalar_features}, floats);
  batch.scalar_observed = torch::ones({kBatch, kRecords, config.scalar_features}, flags);
  batch.semantic = torch::randn({kBatch, kRecords, config.semantic_width}, floats);
  batch.semantic_observed = torch::ones({kBatch, kRecords}, flags);
  batch.record_observed = torch::ones({kBatch, kRecords}, flags);
  batch.categoricals = torch::zeros({kBatch, kRecords, fields}, ids);
  batch.categorical_observed = torch::ones({kBatch, kRecords, fields}, flags);
  for (int64_t field = 0; field < fields; ++field) {
    const auto card = config.categorical_cardinalities[static_cast<size_t>(field)];
    batch.categoricals.select(2, field).copy_(torch::randint(0, card, {kBatch, kRecords}, ids));
  }

  // Observed zero, one scalar hole, one category hole, one absent semantic,
  // and one fully unobserved record whose feature flags stay true.
  batch.scalars.select(0, 0).select(0, 0).select(0, 0).fill_(0);
  batch.scalar_observed.select(0, 0).select(0, 2).select(0, 1).fill_(false);
  batch.scalars.select(0, 0).select(0, 2).select(0, 1).fill_(0);
  batch.categorical_observed.select(0, 0).select(0, 3).select(0, 0).fill_(false);
  batch.categoricals.select(0, 0).select(0, 3).select(0, 0).fill_(0);
  batch.semantic_observed.select(0, 0).select(0, 4).fill_(false);
  batch.record_observed.select(1, 1).fill_(false);
  return batch;
}

}  // namespace

int main() {
  try {
    require(torch::cuda::is_available(), "CUDA is required for the state fusion test");
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    torch::manual_seed(2101);
    const auto device = torch::Device(torch::kCUDA, 0);
    const auto index = torch::TensorOptions().dtype(torch::kLong).device(device);

    state_fusion::Config config;
    state_fusion::StateFusionModel model(config);
    model->to(device);
    model->eval();

    bool found_semantic_absent = false;
    for (const auto& item : model->named_parameters()) {
      if (item.key().find("semantic_absent") != std::string::npos) {
        require(item.value().abs().sum().item<double>() > 0.0, "semantic absent token is the zero vector");
        found_semantic_absent = true;
      }
      if (item.key().find("categorical_absent") != std::string::npos) {
        require(item.value().abs().sum().item<double>() > 0.0, "categorical absent token is the zero vector");
      }
    }
    require(found_semantic_absent, "semantic absent token was not registered");

    const auto batch = manual_batch(config, device);
    torch::Tensor base;
    double permutation_gap = 0;
    double hole_gap = 0;
    double nan_gap = 0;
    double record_gap = 0;
    double padding_gap = 0;
    double two_gap = 0;
    double zero_gap = 0;
    double panel_mse = 0;
    state_fusion::Panel panel = state_fusion::make_panel(config, 4, 6, 7, device);
    {
      torch::NoGradGuard guard;
      base = model->encode(batch);
    require(base.is_cuda(), "encode left CUDA");
    require(base.size(0) == 4 && base.size(1) == config.pma_seeds && base.size(2) == config.d_model,
            "encode shape is not [batch, seeds, d_model]");
    require(torch::isfinite(base).all().item<bool>(), "encode is not finite");

    const auto perm = torch::tensor({2, 0, 4, 1, 3}, index);
    permutation_gap = max_abs(base, model->encode(permute_records(batch, perm)));
    require(permutation_gap <= kClose, "permutation changed the state by " + std::to_string(permutation_gap));

    auto hole_values = clone_batch(batch);
    hole_values.scalars.masked_fill_(batch.scalar_observed.logical_not(), 1000.f);
    auto hole_categories = clone_batch(hole_values);
    hole_categories.categoricals.masked_fill_(batch.categorical_observed.logical_not(), 1);
    hole_categories.semantic.masked_fill_(batch.semantic_observed.logical_not().unsqueeze(-1), 1000.f);
    hole_gap = max_abs(base, model->encode(hole_categories));
    require(hole_gap <= kClose, "editing a hole changed the state by " + std::to_string(hole_gap));

    auto nan_holes = clone_batch(batch);
    nan_holes.scalars.masked_fill_(
        batch.scalar_observed.logical_not(), std::numeric_limits<float>::quiet_NaN());
    nan_holes.semantic.masked_fill_(
        batch.semantic_observed.logical_not().unsqueeze(-1), std::numeric_limits<float>::quiet_NaN());
    nan_gap = max_abs(base, model->encode(nan_holes));
    require(nan_gap <= kClose, "a non-finite hole changed the state by " + std::to_string(nan_gap));

    auto scrambled_record = clone_batch(batch);
    scrambled_record.scalars.select(1, 1).fill_(std::numeric_limits<float>::quiet_NaN());
    scrambled_record.semantic.select(1, 1).fill_(std::numeric_limits<float>::quiet_NaN());
    scrambled_record.categoricals.select(1, 1).fill_(static_cast<int64_t>(888888));
    record_gap = max_abs(base, model->encode(scrambled_record));
    require(record_gap <= kClose, "an unobserved record changed the state by " + std::to_string(record_gap));

    const auto kept = torch::tensor({0, 2, 3, 4}, index);
    padding_gap = max_abs(base, model->encode(permute_records(batch, kept)));
    require(padding_gap <= kClose, "key padding kept an unobserved record; gap " + std::to_string(padding_gap));

    auto zero_as_hole = clone_batch(batch);
    zero_as_hole.scalar_observed.select(0, 0).select(0, 0).select(0, 0).fill_(false);
    zero_gap = max_abs(base, model->encode(zero_as_hole));
    require(zero_gap > kGap, "an observed zero matched a hole; gap " + std::to_string(zero_gap));

    auto on_cpu = clone_batch(batch);
    on_cpu.scalars = on_cpu.scalars.to(torch::kCPU);
    on_cpu.scalar_observed = on_cpu.scalar_observed.to(torch::kCPU);
    on_cpu.categoricals = on_cpu.categoricals.to(torch::kCPU);
    on_cpu.categorical_observed = on_cpu.categorical_observed.to(torch::kCPU);
    on_cpu.semantic = on_cpu.semantic.to(torch::kCPU);
    on_cpu.semantic_observed = on_cpu.semantic_observed.to(torch::kCPU);
    on_cpu.record_observed = on_cpu.record_observed.to(torch::kCPU);
    bool rejected_cpu = false;
    try {
      model->encode(on_cpu);
    } catch (const std::invalid_argument& error) {
      rejected_cpu = std::string(error.what()).find("CUDA") != std::string::npos;
    }
    require(rejected_cpu, "a CPU batch was accepted");

    auto none_observed = clone_batch(batch);
    none_observed.record_observed.fill_(false);
    bool rejected_empty = false;
    try {
      model->encode(none_observed);
    } catch (const std::invalid_argument& error) {
      rejected_empty = std::string(error.what()).find("observed record") != std::string::npos;
    }
    require(rejected_empty, "a fully unobserved set was accepted");

      const auto visible = panel.visible();
      require(!(panel.hidden & visible.scalar_observed).any().item<bool>(),
              "a withheld scalar stayed visible");
      require(!(panel.hidden & panel.full.scalar_observed.logical_not()).any().item<bool>(),
              "a withheld scalar was not an original observation");
      const auto revealed = visible.scalar_observed;
      require(torch::allclose(visible.scalars.masked_select(revealed),
                              panel.clean_scalars.masked_select(revealed)),
              "a visible scalar does not match the clean value");
      const auto loss = model->reconstruction_loss(visible, panel.clean_scalars, panel.hidden);
      require(loss.is_cuda() && torch::isfinite(loss).item<bool>(),
              "panel reconstruction is not a finite CUDA loss");
      panel_mse = loss.item<double>();

      state_fusion::Config two_seeds = config;
      two_seeds.pma_seeds = 2;
      state_fusion::StateFusionModel second(two_seeds);
      second->to(device);
      second->eval();
      const auto two_base = second->encode(batch);
      require(two_base.size(1) == 2, "two seeds did not return two state vectors");
      two_gap = max_abs(two_base, second->encode(permute_records(batch, perm)));
      require(two_gap <= kClose, "two-seed permutation changed the state by " + std::to_string(two_gap));
    }

    model->train();
    const auto graded = model->reconstruction_loss(panel.visible(), panel.clean_scalars, panel.hidden);
    require(graded.is_cuda(), "reconstruction loss left CUDA");
    graded.backward();
    bool found_inducing = false;
    for (const auto& item : model->named_parameters()) {
      if (item.key().find("inducing") == std::string::npos) {
        continue;
      }
      found_inducing = true;
      require(item.value().grad().defined() && item.value().grad().is_cuda(),
              "inducing gradient is not a CUDA tensor: " + item.key());
      require(torch::isfinite(item.value().grad()).all().item<bool>(),
              "inducing gradient is not finite: " + item.key());
      require(item.value().grad().abs().sum().item<double>() > 0.0,
              "inducing gradient is zero: " + item.key());
    }
    require(found_inducing, "inducing points were not registered");
    for (const auto& parameter : model->parameters()) {
      if (parameter.grad().defined()) {
        require(parameter.grad().is_cuda(), "a parameter gradient left CUDA");
        require(torch::isfinite(parameter.grad()).all().item<bool>(), "a parameter gradient is not finite");
      }
    }

    std::cout << "PASS state_fusion_invariance device=cuda:0"
              << " permutation_max_abs=" << permutation_gap
              << " hole_max_abs=" << hole_gap
              << " nan_hole_max_abs=" << nan_gap
              << " unobserved_record_max_abs=" << record_gap
              << " padding_max_abs=" << padding_gap
              << " two_seed_permutation_max_abs=" << two_gap
              << " observed_zero_gap=" << zero_gap
              << " panel_masked_mse=" << panel_mse << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << "\n";
    return 1;
  }
}

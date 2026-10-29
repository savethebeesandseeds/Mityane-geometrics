#include "state_fusion/blocks.h"
#include "state_fusion/model.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

constexpr double kClose = 1e-4;
constexpr double kDistinct = 1e-3;
constexpr int64_t kWidth = 64;
constexpr int64_t kHeads = 4;
constexpr int64_t kFeedforward = 128;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Module>
std::shared_ptr<Module> child_of(const torch::nn::Module& module, const std::string& name) {
  require(module.named_children().contains(name), "missing child " + name);
  auto pointer = std::dynamic_pointer_cast<Module>(module.named_children()[name]);
  require(pointer != nullptr, name + " has the wrong module type");
  return pointer;
}

torch::Tensor parameter_of(const torch::nn::Module& module, const std::string& name) {
  require(module.named_parameters().contains(name), "missing parameter " + name);
  return module.named_parameters()[name];
}

double max_abs(const torch::Tensor& left, const torch::Tensor& right) {
  require(left.sizes().equals(right.sizes()), "compared tensors have different shapes");
  require(torch::isfinite(left).all().item<bool>() && torch::isfinite(right).all().item<bool>(),
          "comparison received a non-finite tensor");
  return (left - right).abs().max().item<double>();
}

torch::Tensor attention_values(torch::nn::MultiheadAttentionImpl& attention,
                               const torch::Tensor& query, const torch::Tensor& key) {
  return std::get<0>(attention.forward(query.contiguous(), key.contiguous(), key.contiguous(), {}, false));
}

// Lee et al. apply the attention residual before the first LayerNorm and the
// feed-forward residual before the second. The swapped form is the control.
torch::Tensor paper_mab(const state_fusion::MABImpl& block, const torch::Tensor& query,
                        const torch::Tensor& key) {
  const auto attended = attention_values(*child_of<torch::nn::MultiheadAttentionImpl>(block, "attention"),
                                         query, key);
  const auto dropout = child_of<torch::nn::DropoutImpl>(block, "dropout");
  const auto mixed = child_of<torch::nn::LayerNormImpl>(block, "norm_attention")
                         ->forward(query.contiguous() + dropout->forward(attended));
  const auto transformed = child_of<torch::nn::LinearImpl>(block, "feedforward_out")
                               ->forward(dropout->forward(torch::relu(
                                   child_of<torch::nn::LinearImpl>(block, "feedforward_in")->forward(mixed))));
  return child_of<torch::nn::LayerNormImpl>(block, "norm_feedforward")->forward(mixed + transformed);
}

torch::Tensor swapped_mab(const state_fusion::MABImpl& block, const torch::Tensor& query,
                          const torch::Tensor& key) {
  const auto attended = attention_values(*child_of<torch::nn::MultiheadAttentionImpl>(block, "attention"),
                                         query, key);
  const auto dropout = child_of<torch::nn::DropoutImpl>(block, "dropout");
  const auto mixed = query.contiguous() +
                     dropout->forward(child_of<torch::nn::LayerNormImpl>(block, "norm_attention")->forward(attended));
  const auto transformed = child_of<torch::nn::LinearImpl>(block, "feedforward_out")
                               ->forward(dropout->forward(torch::relu(
                                   child_of<torch::nn::LinearImpl>(block, "feedforward_in")->forward(mixed))));
  return mixed + dropout->forward(child_of<torch::nn::LayerNormImpl>(block, "norm_feedforward")->forward(transformed));
}

double check_mab_order(torch::Device device) {
  state_fusion::MAB block(kWidth, kHeads, kFeedforward, 0.0);
  block->to(device);
  block->eval();
  const auto options = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto query = torch::randn({5, 2, kWidth}, options);
  const auto key = torch::randn({7, 2, kWidth}, options);
  const auto actual = block->forward(query, key);
  require(actual.sizes().equals(query.sizes()), "MAB changed the query length");
  const double paper_gap = max_abs(actual, paper_mab(*block, query, key));
  require(paper_gap <= kClose, "MAB does not apply residuals before LayerNorm; gap " + std::to_string(paper_gap));
  const double swapped_gap = max_abs(actual, swapped_mab(*block, query, key));
  require(swapped_gap > kDistinct, "MAB matched the swapped LayerNorm order; gap " + std::to_string(swapped_gap));
  return paper_gap;
}

double check_isab_role(torch::Device device) {
  constexpr int64_t kInducing = 16;
  constexpr int64_t kSet = 32;
  state_fusion::ISAB block(kWidth, kHeads, kFeedforward, kInducing, 0.0);
  block->to(device);
  block->eval();
  const auto inducing = parameter_of(*block, "inducing");
  require(inducing.size(0) == kInducing && inducing.size(1) == kWidth, "inducing-point shape is not [m, d_model]");

  const auto options = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto set = torch::randn({kSet, 2, kWidth}, options);
  const auto actual = block->forward(set, {});
  require(actual.size(0) == kSet && actual.size(2) == kWidth, "ISAB output length is not the set length");

  const auto queries = inducing.unsqueeze(1).repeat({1, set.size(1), 1});
  const auto induce = child_of<state_fusion::MABImpl>(*block, "induce");
  const auto process = child_of<state_fusion::MABImpl>(*block, "process");
  const auto hidden = induce->forward(queries, set);
  require(hidden.size(0) == kInducing, "the first ISAB attention did not keep the inducing length");
  const double role_gap = max_abs(actual, process->forward(set, hidden));
  require(role_gap <= kClose, "ISAB does not use inducing points as the first queries; gap " + std::to_string(role_gap));

  const auto swapped = process->forward(set, induce->forward(set, queries));
  const double swapped_gap = max_abs(actual, swapped);
  require(swapped_gap > kDistinct, "ISAB accepted the set as the first queries; gap " + std::to_string(swapped_gap));
  return role_gap;
}

double check_pma_role(torch::Device device) {
  constexpr int64_t kSeeds = 2;
  constexpr int64_t kSet = 32;
  state_fusion::PMA block(kWidth, kHeads, kFeedforward, kSeeds, 0.0);
  block->to(device);
  block->eval();
  const auto seeds = parameter_of(*block, "seeds");
  require(seeds.size(0) == kSeeds && seeds.size(1) == kWidth, "seed shape is not [k, d_model]");

  const auto options = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto set = torch::randn({kSet, 2, kWidth}, options);
  const auto padding = torch::zeros({2, kSet}, torch::TensorOptions().dtype(torch::kBool).device(device));
  const auto actual = block->forward(set, padding);
  require(actual.size(0) == kSeeds && actual.size(1) == 2 && actual.size(2) == kWidth,
          "PMA output shape is not [seeds, batch, d_model]");

  const auto prepared = child_of<torch::nn::LinearImpl>(*block, "feedforward_out")
                            ->forward(torch::relu(child_of<torch::nn::LinearImpl>(*block, "feedforward_in")->forward(set)));
  const auto queries = seeds.unsqueeze(1).repeat({1, set.size(1), 1});
  const double role_gap = max_abs(actual, child_of<state_fusion::MABImpl>(*block, "pool")->forward(queries, prepared, padding));
  require(role_gap <= kClose, "PMA does not use seeds as queries; gap " + std::to_string(role_gap));

  const auto swapped = child_of<state_fusion::MABImpl>(*block, "pool")->forward(set, prepared, padding);
  require(swapped.size(0) == kSet, "using the set as PMA queries did not keep the set length");
  require(swapped.size(0) != actual.size(0), "PMA seed output has the set length");
  return role_gap;
}

state_fusion::SetBatch observed_batch(const state_fusion::Config& config, int64_t rows, int64_t records,
                                      torch::Device device) {
  const auto floats = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto flags = torch::TensorOptions().dtype(torch::kBool).device(device);
  const auto ids = torch::TensorOptions().dtype(torch::kLong).device(device);
  const auto fields = static_cast<int64_t>(config.categorical_cardinalities.size());
  state_fusion::SetBatch batch;
  batch.scalars = torch::randn({rows, records, config.scalar_features}, floats);
  batch.scalar_observed = torch::ones({rows, records, config.scalar_features}, flags);
  batch.semantic = torch::randn({rows, records, config.semantic_width}, floats);
  batch.semantic_observed = torch::ones({rows, records}, flags);
  batch.record_observed = torch::ones({rows, records}, flags);
  batch.categoricals = torch::zeros({rows, records, fields}, ids);
  batch.categorical_observed = torch::ones({rows, records, fields}, flags);
  return batch;
}

struct GradientReport {
  double inducing = 0;
  double seeds = 0;
  double semantic_absent = 0;
  double categorical_absent = 0;
};

GradientReport check_gradients(torch::Device device) {
  state_fusion::Config config;
  state_fusion::StateFusionModel model(config);
  model->to(device);
  model->train();
  auto batch = observed_batch(config, 4, 6, device);
  batch.scalar_observed.select(2, 0).fill_(false);
  batch.categorical_observed.select(2, 0).fill_(false);
  batch.semantic_observed.select(1, 0).fill_(false);
  const auto clean = torch::randn_like(batch.scalars);
  const auto hidden = torch::zeros_like(batch.scalar_observed);
  hidden.select(2, 0).fill_(true);

  const auto loss = model->reconstruction_loss(batch, clean, hidden);
  require(loss.is_cuda() && torch::isfinite(loss).item<bool>(), "reconstruction loss is not a finite CUDA scalar");
  loss.backward();

  GradientReport report;
  auto take = [&](const std::string& name, double& slot) {
    bool found = false;
    for (const auto& item : model->named_parameters()) {
      if (item.key() != name && item.key().find(name) == std::string::npos) {
        continue;
      }
      found = true;
      require(item.value().grad().defined(), name + " received no gradient");
      require(torch::isfinite(item.value().grad()).all().item<bool>(), name + " gradient is not finite");
      const double norm = item.value().grad().abs().sum().item<double>();
      require(norm > 0.0, name + " gradient is zero");
      slot += norm;
    }
    require(found, name + " was not registered");
  };
  take("inducing", report.inducing);
  take("seeds", report.seeds);
  take("semantic_absent", report.semantic_absent);
  take("categorical_absent", report.categorical_absent);
  return report;
}

double milliseconds(int64_t records, state_fusion::StateFusionModel& model, const state_fusion::Config& config,
                    torch::Device device) {
  constexpr int kWarmup = 5;
  constexpr int kIters = 20;
  const auto batch = observed_batch(config, 32, records, device);
  torch::Tensor last;
  for (int step = 0; step < kWarmup; ++step) {
    last = model->encode(batch);
  }
  torch::cuda::synchronize();
  const auto started = std::chrono::steady_clock::now();
  for (int step = 0; step < kIters; ++step) {
    last = model->encode(batch);
  }
  torch::cuda::synchronize();
  require(last.defined() && last.is_cuda() && torch::isfinite(last).all().item<bool>(),
          "timed encode was not a finite CUDA state");
  const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return elapsed / kIters;
}

double check_linear_scaling(torch::Device device) {
  state_fusion::Config config;
  state_fusion::StateFusionModel model(config);
  model->to(device);
  model->eval();
  torch::NoGradGuard guard;
  const double time_16 = milliseconds(16, model, config, device);
  const double time_64 = milliseconds(64, model, config, device);
  const double time_256 = milliseconds(256, model, config, device);
  require(time_16 > 0.0 && time_64 > 0.0 && time_256 > 0.0, "encoder timing was zero");
  // Quadratic growth from 16 to 256 is a factor of 16 in time per record.
  // Linear growth keeps that ratio near 1. The bound rejects quadratic scaling
  // and still allows fixed CUDA launch cost.
  const double per_record_ratio = (time_256 / 256.0) / (time_16 / 16.0);
  require(per_record_ratio < 4.0, "encoder time grew faster than linear; per-record ratio " +
                                      std::to_string(per_record_ratio) + " times " + std::to_string(time_16) +
                                      " " + std::to_string(time_64) + " " + std::to_string(time_256));
  std::cout << "scale_16_ms=" << time_16 << " scale_64_ms=" << time_64 << " scale_256_ms=" << time_256
            << " per_record_ratio=" << per_record_ratio << "\n";
  return per_record_ratio;
}

void check_device_and_loss_contract(torch::Device device) {
  state_fusion::Config config;
  auto batch = observed_batch(config, 2, 4, device);
  state_fusion::StateFusionModel cpu_model(config);
  bool rejected_parameters = false;
  try {
    cpu_model->encode(batch);
  } catch (const std::invalid_argument& error) {
    rejected_parameters = std::string(error.what()).find("CUDA") != std::string::npos;
  }
  require(rejected_parameters, "CPU parameters were accepted");

  state_fusion::StateFusionModel model(config);
  model->to(device);
  model->eval();
  auto empty = observed_batch(config, 1, 0, device);
  bool rejected_empty = false;
  try {
    model->encode(empty);
  } catch (const std::invalid_argument& error) {
    rejected_empty = std::string(error.what()).find("record") != std::string::npos;
  }
  require(rejected_empty, "an empty set was accepted");

  const auto hidden = torch::zeros_like(batch.scalar_observed);
  hidden.select(0, 0).select(0, 0).select(0, 0).fill_(true);
  bool rejected_visible = false;
  try {
    model->reconstruction_loss(batch, torch::randn_like(batch.scalars), hidden);
  } catch (const std::invalid_argument& error) {
    rejected_visible = std::string(error.what()).find("holes") != std::string::npos;
  }
  require(rejected_visible, "a hidden cell that was still observed was accepted");
}

}  // namespace

int main() {
  try {
    require(torch::cuda::is_available(), "CUDA is required for the state fusion fidelity test");
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    torch::manual_seed(2101);
    const auto device = torch::Device(torch::kCUDA, 0);

    const double mab_gap = check_mab_order(device);
    const double isab_gap = check_isab_role(device);
    const double pma_gap = check_pma_role(device);
    const auto gradients = check_gradients(device);
    const double per_record_ratio = check_linear_scaling(device);
    check_device_and_loss_contract(device);

    std::cout << "PASS state_fusion_fidelity device=cuda:0"
              << " mab_order_gap=" << mab_gap
              << " isab_role_gap=" << isab_gap
              << " pma_role_gap=" << pma_gap
              << " inducing_grad=" << gradients.inducing
              << " seeds_grad=" << gradients.seeds
              << " semantic_absent_grad=" << gradients.semantic_absent
              << " categorical_absent_grad=" << gradients.categorical_absent
              << " per_record_ratio=" << per_record_ratio << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << "\n";
    return 1;
  }
}

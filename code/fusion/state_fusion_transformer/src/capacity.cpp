#include "state_fusion/model.h"
#include "state_fusion/synthetic.h"

#include <torch/cuda.h>
#include <torch/optim/adam.h>
#include <torch/version.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kBatch = 32;
constexpr int64_t kRecords = 8;
constexpr int kSteps = 1600;
constexpr int kLogEvery = 200;
constexpr int64_t kSeed = 2101;
constexpr int64_t kHeldout = 32;
constexpr int64_t kHeldoutSeed = 50000;
constexpr double kReloadTolerance = 1e-4;
constexpr const char* kCheckpoint =
    "/mityane/.temp/fusion/state-fusion-transformer/capacity-probe.pt";

struct Score {
  double reconstruction = 0;
  double empty_query = 0;
  double filter = 0;
  double filter_feature0 = 0;
  double zero_baseline = 0;
  double set_mean_baseline = 0;
  double filter_feature0_set_mean = 0;
};

struct Accumulator {
  double reconstruction_error = 0;
  double reconstruction_count = 0;
  double empty_error = 0;
  double empty_count = 0;
  double filter_error = 0;
  double filter_count = 0;
  double filter0_error = 0;
  double filter0_count = 0;
  double zero_error = 0;
  double set_mean_error = 0;
  double set_mean_count = 0;
  double filter0_baseline_error = 0;
  double filter0_baseline_count = 0;

  void add_square(double& error, double& count, const torch::Tensor& value) {
    error += value.sum().item<double>();
    count += static_cast<double>(value.numel());
  }

  Score mean() const {
    if (reconstruction_count < 1 || empty_count < 1 || filter_count < 1 || filter0_count < 1 ||
        set_mean_count < 1 || filter0_baseline_count < 1) {
      throw std::runtime_error("capacity score has an empty denominator");
    }
    Score score;
    score.reconstruction = reconstruction_error / reconstruction_count;
    score.empty_query = empty_error / empty_count;
    score.filter = filter_error / filter_count;
    score.filter_feature0 = filter0_error / filter0_count;
    score.zero_baseline = zero_error / reconstruction_count;
    score.set_mean_baseline = set_mean_error / set_mean_count;
    score.filter_feature0_set_mean = filter0_baseline_error / filter0_baseline_count;
    return score;
  }
};

double parameter_grad_abs(const state_fusion::StateFusionModel& model, const std::string& prefix) {
  double total = 0;
  bool found = false;
  for (const auto& item : model->named_parameters()) {
    if (item.key().rfind(prefix, 0) != 0) {
      continue;
    }
    found = true;
    if (!item.value().grad().defined()) {
      throw std::runtime_error(prefix + " received no gradient");
    }
    total += item.value().grad().abs().sum().item<double>();
  }
  if (!found || !std::isfinite(total) || total <= 0.0) {
    throw std::runtime_error(prefix + " gradient is not a positive finite value");
  }
  return total;
}

Score evaluate(state_fusion::StateFusionModel& model,
               const std::vector<state_fusion::CapacityPanel>& heldout) {
  torch::NoGradGuard guard;
  model->eval();
  Accumulator totals;
  for (const auto& sample : heldout) {
    const auto visible = sample.panel.visible();
    const auto clean = sample.panel.clean_scalars;
    const auto hidden = sample.panel.hidden;
    const auto prediction = model->reconstruct(visible);
    const auto recon = (prediction - clean).square();
    totals.reconstruction_error += torch::where(hidden, recon, torch::zeros_like(recon)).sum().item<double>();
    totals.reconstruction_count += static_cast<double>(hidden.sum().item<int64_t>());
    totals.zero_error += torch::where(hidden, clean.square(), torch::zeros_like(clean)).sum().item<double>();

    const auto shown = visible.scalar_observed.to(clean.dtype());
    const auto shown_count = shown.sum(/*dim=*/1);
    const auto set_mean = (clean * shown).sum(/*dim=*/1) / shown_count.clamp_min(1);
    const auto set_mean_error = (set_mean.unsqueeze(1) - clean).square();
    const auto comparable = hidden & shown_count.unsqueeze(1).gt(0);
    totals.set_mean_error +=
        torch::where(comparable, set_mean_error, torch::zeros_like(set_mean_error)).sum().item<double>();
    totals.set_mean_count += static_cast<double>(comparable.sum().item<int64_t>());

    const auto answered = model->query_scalars(model->encode(visible), sample.queries);
    const auto query_error = (answered - sample.query_targets).square();
    const auto empty_error = query_error.select(1, 0);
    const auto filter_error = query_error.select(1, 1);
    totals.add_square(totals.empty_error, totals.empty_count, empty_error);
    totals.add_square(totals.filter_error, totals.filter_count, filter_error);
    totals.add_square(totals.filter0_error, totals.filter0_count, filter_error.select(1, 0));

    const auto shown0 = shown.select(2, 0);
    const auto feature0_mean = (clean.select(2, 0) * shown0).sum(/*dim=*/1) / shown0.sum(/*dim=*/1).clamp_min(1);
    const auto feature0_target = sample.query_targets.select(1, 1).select(1, 0);
    totals.add_square(totals.filter0_baseline_error, totals.filter0_baseline_count,
                      (feature0_mean - feature0_target).square());
  }
  return totals.mean();
}

void print_score(const std::string& label, const Score& score) {
  std::cout << label
            << " reconstruction=" << score.reconstruction
            << " empty_query_mse=" << score.empty_query
            << " filter_mse=" << score.filter
            << " filter_feature0_mse=" << score.filter_feature0
            << "\n" << std::flush;
}

}  // namespace

int main() {
  try {
    if (!torch::cuda::is_available()) {
      std::cerr << "CUDA is required for state fusion capacity training.\n";
      return 1;
    }
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    torch::manual_seed(static_cast<uint64_t>(kSeed));
    torch::cuda::manual_seed_all(kSeed);

    const auto device = torch::Device(torch::kCUDA, 0);
    state_fusion::Config config;
    state_fusion::StateFusionModel model(config);
    model->to(device);

    std::vector<state_fusion::CapacityPanel> heldout;
    heldout.reserve(static_cast<size_t>(kHeldout));
    for (int64_t panel = 0; panel < kHeldout; ++panel) {
      heldout.push_back(state_fusion::make_capacity_panel(
          config, kBatch, kRecords, kHeldoutSeed + panel, device));
    }

    const Score untrained = evaluate(model, heldout);
    std::cout << "device=cuda:" << static_cast<int>(device.index())
              << " libtorch=" << TORCH_VERSION
              << " d_model=" << config.d_model
              << " inducing_points=" << config.inducing_points
              << " seeds=" << config.pma_seeds
              << " steps=" << kSteps
              << " batch=" << kBatch
              << " records=" << kRecords
              << " heldout_panels=" << kHeldout << "\n";
    std::cout << "note=synthetic_capacity\n";
    print_score("untrained", untrained);
    std::cout << "baselines zero_reconstruction=" << untrained.zero_baseline
              << " set_mean_reconstruction=" << untrained.set_mean_baseline
              << " set_mean_filter_feature0=" << untrained.filter_feature0_set_mean << "\n" << std::flush;
    if (!std::isfinite(untrained.reconstruction) || untrained.reconstruction <= 0.0 ||
        !std::isfinite(untrained.empty_query) || untrained.empty_query <= 0.0 ||
        !(untrained.filter_feature0_set_mean > 0.05)) {
      std::cerr << "untrained capacity scores do not leave a measurable filter gap\n";
      return 1;
    }

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();
    for (int step = 1; step <= kSteps; ++step) {
      const auto sample = state_fusion::make_capacity_panel(
          config, kBatch, kRecords, kSeed + step, device);
      const auto visible = sample.panel.visible();
      optimizer.zero_grad();
      const auto reconstruction = model->reconstruction_loss(
          visible, sample.panel.clean_scalars, sample.panel.hidden);
      const auto state = model->encode(visible);
      const auto predicted = model->query_scalars(state, sample.queries);
      const auto query_loss = (predicted - sample.query_targets).square().mean();
      const auto loss = reconstruction + query_loss;
      loss.backward();
      if (step == 1) {
        std::cout << "query_block_grad=" << parameter_grad_abs(model, "query.")
                  << " query_scalar_head_grad=" << parameter_grad_abs(model, "query_scalar_head")
                  << "\n" << std::flush;
      }
      optimizer.step();
      if (!torch::isfinite(loss).item<bool>()) {
        std::cerr << "capacity loss is not finite at step " << step << "\n";
        return 1;
      }
      if (step % kLogEvery == 0 || step == kSteps) {
        const Score held = evaluate(model, heldout);
        std::cout << "step=" << step;
        print_score("", held);
        model->train();
      }
    }

    const Score final_score = evaluate(model, heldout);
    print_score("final", final_score);

    const std::filesystem::path checkpoint(kCheckpoint);
    std::filesystem::create_directories(checkpoint.parent_path());
    state_fusion::save_checkpoint(model, checkpoint.string());
    auto restored = state_fusion::load_checkpoint(checkpoint.string(), device);
    restored->eval();
    model->eval();
    double reload_gap = 0;
    {
      torch::NoGradGuard guard;
      for (const auto& sample : heldout) {
        const auto visible = sample.panel.visible();
        const auto original = model->query_scalars(model->encode(visible), sample.queries);
        const auto loaded = restored->query_scalars(restored->encode(visible), sample.queries);
        reload_gap = std::max(reload_gap, (original - loaded).abs().max().item<double>());
      }
    }
    std::cout << "checkpoint=" << checkpoint.string()
              << " checkpoint_query_scalars_max_abs=" << reload_gap << "\n" << std::flush;

    const bool reconstruction_ok = std::isfinite(final_score.reconstruction) &&
                                   final_score.reconstruction < 0.5 * untrained.reconstruction;
    const bool empty_ok = std::isfinite(final_score.empty_query) &&
                          final_score.empty_query < 0.5 * untrained.empty_query;
    const bool filter_ok = std::isfinite(final_score.filter_feature0) &&
                           final_score.filter_feature0 < untrained.filter_feature0_set_mean;
    const bool reload_ok = std::isfinite(reload_gap) && reload_gap <= kReloadTolerance;
    if (!reconstruction_ok || !empty_ok || !filter_ok || !reload_ok) {
      std::cerr << "capacity gates failed"
                << " reconstruction=" << reconstruction_ok
                << " empty_query=" << empty_ok
                << " filter_feature0=" << filter_ok
                << " reload=" << reload_ok << "\n";
      return 1;
    }
    std::cout << "state_fusion_capacity: pass\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}

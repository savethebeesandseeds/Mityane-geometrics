#include "state_fusion/model.h"
#include "state_fusion/synthetic.h"

#include <torch/cuda.h>
#include <torch/optim/adam.h>
#include <torch/version.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int64_t kBatch = 32;
constexpr int64_t kRecords = 8;
constexpr int kSteps = 400;
constexpr int64_t kSeed = 2101;

double masked_mse(state_fusion::StateFusionModel& model, const state_fusion::Panel& panel) {
  torch::NoGradGuard guard;
  const auto loss = model->reconstruction_loss(panel.visible(), panel.clean_scalars, panel.hidden);
  if (!loss.is_cuda()) {
    throw std::runtime_error("masked reconstruction loss must be a CUDA tensor");
  }
  return loss.item<double>();
}

}  // namespace

int main() {
  try {
    if (!torch::cuda::is_available()) {
      std::cerr << "CUDA is required for state fusion training.\n";
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

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    const auto panel = state_fusion::make_panel(config, kBatch, kRecords, kSeed, device);
    model->eval();
    const double initial = masked_mse(model, panel);

    std::cout << "device=cuda:" << device.index()
              << " libtorch=" << TORCH_VERSION
              << " d_model=" << config.d_model
              << " inducing_points=" << config.inducing_points
              << " seeds=" << config.pma_seeds
              << " steps=" << kSteps << "\n";
    std::cout << "initial_masked_mse=" << initial << "\n" << std::flush;
    if (!std::isfinite(initial) || initial <= 0.0) {
      std::cerr << "initial masked mse is not a positive finite value\n";
      return 1;
    }

    model->train();
    for (int step = 1; step <= kSteps; ++step) {
      optimizer.zero_grad();
      const auto loss = model->reconstruction_loss(panel.visible(), panel.clean_scalars, panel.hidden);
      loss.backward();
      optimizer.step();
      if (step % 50 == 0 || step == kSteps) {
        model->eval();
        std::cout << "step=" << step << " masked_mse=" << masked_mse(model, panel) << "\n" << std::flush;
        model->train();
      }
    }

    model->eval();
    const double final_mse = masked_mse(model, panel);
    std::cout << "final_masked_mse=" << final_mse << "\n";
    std::cout << "note=synthetic_pipeline_validation\n" << std::flush;
    if (!std::isfinite(final_mse) || !(final_mse < initial * 0.5)) {
      std::cerr << "masked mse did not fall below half the initial value\n";
      return 1;
    }
    std::cout << "state_fusion_train: pass\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}

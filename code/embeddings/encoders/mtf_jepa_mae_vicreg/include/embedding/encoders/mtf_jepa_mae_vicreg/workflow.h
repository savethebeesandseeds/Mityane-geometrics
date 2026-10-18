// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/mtf_jepa_mae_vicreg/model.h"
#include "embedding/shared/data.h"

#include <cstdint>
#include <string>

namespace embedding::encoders::mtf_jepa_mae_vicreg {

// Architecture and training settings are saved together in each checkpoint.
struct Settings {
  Config model{};
  int64_t steps{8};
  int64_t batch_size{4};
  int64_t seed{101};
  int64_t threads{1};
  int64_t log_every{1};
  int64_t checkpoint_every{0}; // 0 saves only at the end; otherwise periodic updates.
  double learning_rate{1e-3};
  double weight_decay{1e-4};
  double gradient_clip_norm{1.0}; // 0 disables clipping.
};

struct Checkpoint {
  Settings settings;
  Model model{nullptr};
  int64_t completed_steps{0};
};

Settings default_settings();
Settings parse_settings(const std::string &text);
Settings read_settings(const std::string &path);
std::string settings_text(const Settings &settings);

// Synthetic fixture requires input_width=3 and supports arbitrary C/H.
// Call torch::manual_seed first when reproducible fixture noise is needed.
::embedding::Batch synthetic_batch(const Config &config, int64_t samples,
                                  int64_t step = 0);
::embedding::Batch load_batch(const std::string &path, const Config &config);

void save_checkpoint(const std::string &path, const Settings &settings,
                     Model &model, torch::optim::AdamW &optimizer,
                     int64_t completed_steps);
Checkpoint load_checkpoint(const std::string &path,
                           const torch::Device &device = torch::kCPU);
void load_optimizer(const std::string &path, torch::optim::AdamW &optimizer,
                    const torch::Device &device);

int run_cli(int argc, char **argv);

} // namespace embedding::encoders::mtf_jepa_mae_vicreg

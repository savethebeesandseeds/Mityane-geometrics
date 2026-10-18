// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/types.h"

#include <cstdint>
#include <string>
#include <torch/torch.h>

namespace embedding {

// Archive data are float32 [B,C,H,F] on CPU; feature_mask is bool and true
// means observed. Models and data helpers use the same tensor contract.
using Batch = input_t;

// The synthetic smoke fixture uses three features and arbitrary C/H dimensions.
// Call torch::manual_seed first when reproducible fixture noise is needed.
Batch synthetic_batch(const input_shape_t &shape, int64_t samples,
                      int64_t step = 0);
void save_batch(const std::string &path, const Batch &batch);
Batch load_batch(const std::string &path, const input_shape_t &shape);

namespace archive {

void validate_batch(const Batch &batch, const input_shape_t &shape);

// Write next to the destination, then atomically replace it on Debian.
// A failed write leaves an existing destination intact.
void save_archive(const std::string &path,
                  torch::serialize::OutputArchive &value);
void distinct_paths(const std::string &input, const std::string &output);
torch::Tensor text_tensor(const std::string &text);
std::string tensor_text(const torch::Tensor &tensor);

} // namespace archive
} // namespace embedding

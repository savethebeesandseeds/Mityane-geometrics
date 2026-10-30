#pragma once

#include "state_fusion/model.h"

#include <torch/torch.h>

namespace state_fusion {

// Synthetic panel for pipeline validation. clean_scalars holds the generated
// values, including real zeros. full stores garbage in holes. hidden marks
// observed scalars withheld from the encoder.
struct Panel {
  SetBatch full;
  torch::Tensor clean_scalars;
  torch::Tensor hidden;
  SetBatch visible() const;
};

Panel make_panel(const Config& config, int64_t batch, int64_t set_size, int64_t seed,
                 torch::Device device);

}  // namespace state_fusion

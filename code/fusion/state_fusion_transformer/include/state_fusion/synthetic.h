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

// Separate generator for the capacity protocol. make_panel is unchanged.
// An even set is split into pairs. Each pair shares a place in 0..3 and a
// kind in 0..1, and no pair in the set repeats that combination. Scalar
// feature 0 is the shared latent plus 0.5 * (place - 1.5) + 0.4 * (kind - 0.5)
// plus noise of scale 0.05. Other features are the shared latent plus that
// noise. There are no planted real zeros. Records 0 and 1 stay observed,
// place and kind stay observed on observed records, feature 0 of record 0 is
// hidden, and feature 0 of record 1 stays visible.
// queries has two rows. Row 0 is empty. Row 1 observes the place and kind of
// record 0. query_targets is [batch, 2, features]: the empty target is the
// mean clean scalar over observed records, and the filter target is the
// conditional expectation (latent feature 0 plus that pair's shift).
struct CapacityPanel {
  Panel panel;
  SetBatch queries;
  torch::Tensor query_targets;
};

CapacityPanel make_capacity_panel(const Config& config, int64_t batch, int64_t set_size,
                                  int64_t seed, torch::Device device);

}  // namespace state_fusion

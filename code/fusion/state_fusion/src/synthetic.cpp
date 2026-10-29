#include "state_fusion/synthetic.h"

#include <stdexcept>

namespace state_fusion {
namespace {

constexpr double kFieldHoleRate = 0.15;
constexpr double kRecordHoleRate = 0.10;
constexpr double kSemanticAbsentRate = 0.25;
constexpr double kRealZeroRate = 0.08;
constexpr double kWithholdRate = 0.30;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}  // namespace

SetBatch Panel::visible() const {
  SetBatch out = full;
  out.scalar_observed = full.scalar_observed & hidden.logical_not();
  return out;
}

Panel make_panel(const Config& config, int64_t batch, int64_t set_size, int64_t seed,
                 torch::Device device) {
  require(device.is_cuda(), "synthetic panels are built on CUDA");
  require(batch >= 1 && set_size >= 1, "a panel needs a positive batch and set size");
  require(seed >= 0, "panel seed must be non-negative");
  require(config.scalar_features >= 1 && config.semantic_width >= 1, "panel feature widths must be positive");
  require(!config.categorical_cardinalities.empty(), "a panel needs categorical fields");

  torch::manual_seed(static_cast<uint64_t>(seed));
  const auto floats = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto ids = torch::TensorOptions().dtype(torch::kLong).device(device);
  const auto features = config.scalar_features;
  const auto fields = static_cast<int64_t>(config.categorical_cardinalities.size());

  const auto latent = torch::randn({batch, features}, floats);
  auto clean = latent.unsqueeze(1) + 0.05 * torch::randn({batch, set_size, features}, floats);
  const auto real_zero = torch::rand({batch, set_size, features}, floats) < kRealZeroRate;
  clean = torch::where(real_zero, torch::zeros_like(clean), clean);

  auto record_observed = torch::rand({batch, set_size}, floats) >= kRecordHoleRate;
  record_observed.select(1, 0).fill_(true);

  auto scalar_observed = torch::rand({batch, set_size, features}, floats) >= kFieldHoleRate;
  scalar_observed.select(1, 0).select(1, 0).fill_(true);
  scalar_observed = scalar_observed & record_observed.unsqueeze(-1);

  auto hidden = scalar_observed & (torch::rand({batch, set_size, features}, floats) < kWithholdRate);
  hidden.select(1, 0).select(1, 0).fill_(true);

  const auto revealed = scalar_observed & hidden.logical_not();
  const auto garbage = 5.0 * torch::randn({batch, set_size, features}, floats);

  Panel panel;
  panel.clean_scalars = clean;
  panel.hidden = hidden;
  panel.full.scalars = torch::where(revealed, clean, garbage);
  panel.full.scalar_observed = scalar_observed;
  panel.full.record_observed = record_observed;

  panel.full.categoricals = torch::empty({batch, set_size, fields}, ids);
  for (int64_t field = 0; field < fields; ++field) {
    const auto card = config.categorical_cardinalities[static_cast<size_t>(field)];
    require(card >= 1, "categorical cardinality must be positive");
    panel.full.categoricals.select(2, field).copy_(torch::randint(0, card, {batch, set_size}, ids));
  }
  auto categorical_observed = torch::rand({batch, set_size, fields}, floats) >= kFieldHoleRate;
  panel.full.categorical_observed = categorical_observed & record_observed.unsqueeze(-1);

  panel.full.semantic = torch::randn({batch, set_size, config.semantic_width}, floats);
  const auto semantic_observed = torch::rand({batch, set_size}, floats) >= kSemanticAbsentRate;
  panel.full.semantic_observed = semantic_observed & record_observed;
  return panel;
}

}  // namespace state_fusion

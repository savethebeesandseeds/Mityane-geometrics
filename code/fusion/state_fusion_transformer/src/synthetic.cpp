#include "state_fusion/synthetic.h"

#include <stdexcept>
#include <tuple>

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

CapacityPanel make_capacity_panel(const Config& config, int64_t batch, int64_t set_size, int64_t seed,
                                  torch::Device device) {
  require(device.is_cuda(), "synthetic panels are built on CUDA");
  require(batch >= 1 && set_size >= 2 && set_size % 2 == 0, "a capacity panel needs an even set of at least two records");
  require(seed >= 0, "panel seed must be non-negative");
  require(config.scalar_features >= 1 && config.semantic_width >= 1, "panel feature widths must be positive");
  require(config.categorical_cardinalities.size() >= 2, "a capacity panel needs place and kind fields");
  require(config.categorical_cardinalities[0] >= 4, "place cardinality must cover 0..3");
  require(config.categorical_cardinalities[1] >= 2, "kind cardinality must cover 0..1");
  const auto pairs = set_size / 2;
  require(pairs <= 8, "a capacity panel uses at most eight place-kind pairs");

  torch::manual_seed(static_cast<uint64_t>(seed));
  const auto floats = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto ids = torch::TensorOptions().dtype(torch::kLong).device(device);
  const auto flags = torch::TensorOptions().dtype(torch::kBool).device(device);
  const auto features = config.scalar_features;
  const auto fields = static_cast<int64_t>(config.categorical_cardinalities.size());

  // Eight place-kind codes, then the first `pairs` of a random order. Repeating
  // each chosen code twice gives the two records of that pair.
  const auto ranked = torch::rand({batch, 8}, floats).sort(/*dim=*/1, /*descending=*/false);
  const auto chosen = std::get<1>(ranked).narrow(1, 0, pairs);
  const auto place = chosen.remainder(4).repeat_interleave(/*repeats=*/2, /*dim=*/1);
  const auto kind = chosen.div(4, /*rounding_mode=*/"floor").repeat_interleave(2, 1);
  const auto effect = 0.5 * (place.to(floats) - 1.5) + 0.4 * (kind.to(floats) - 0.5);

  const auto latent = torch::randn({batch, features}, floats);
  const auto noise = 0.05 * torch::randn({batch, set_size, features}, floats);
  auto clean = latent.unsqueeze(1) + noise;
  clean.select(2, 0).copy_(latent.select(1, 0).unsqueeze(1) + effect + noise.select(2, 0));

  auto record_observed = torch::rand({batch, set_size}, floats) >= kRecordHoleRate;
  record_observed.select(1, 0).fill_(true);
  record_observed.select(1, 1).fill_(true);

  auto scalar_observed = torch::rand({batch, set_size, features}, floats) >= kFieldHoleRate;
  scalar_observed.select(1, 0).select(1, 0).fill_(true);
  scalar_observed.select(1, 1).select(1, 0).fill_(true);
  scalar_observed = scalar_observed & record_observed.unsqueeze(-1);

  auto hidden = scalar_observed & (torch::rand({batch, set_size, features}, floats) < kWithholdRate);
  hidden.select(1, 0).select(1, 0).fill_(true);
  hidden.select(1, 1).select(1, 0).fill_(false);

  const auto revealed = scalar_observed & hidden.logical_not();
  const auto garbage = 5.0 * torch::randn({batch, set_size, features}, floats);

  CapacityPanel sample;
  sample.panel.clean_scalars = clean;
  sample.panel.hidden = hidden;
  sample.panel.full.scalars = torch::where(revealed, clean, garbage);
  sample.panel.full.scalar_observed = scalar_observed;
  sample.panel.full.record_observed = record_observed;

  sample.panel.full.categoricals = torch::empty({batch, set_size, fields}, ids);
  for (int64_t field = 0; field < fields; ++field) {
    const auto card = config.categorical_cardinalities[static_cast<size_t>(field)];
    require(card >= 1, "categorical cardinality must be positive");
    sample.panel.full.categoricals.select(2, field).copy_(torch::randint(0, card, {batch, set_size}, ids));
  }
  sample.panel.full.categoricals.select(2, 0).copy_(place);
  sample.panel.full.categoricals.select(2, 1).copy_(kind);
  auto categorical_observed = torch::rand({batch, set_size, fields}, floats) >= kFieldHoleRate;
  categorical_observed = categorical_observed & record_observed.unsqueeze(-1);
  categorical_observed.select(2, 0).copy_(record_observed);
  categorical_observed.select(2, 1).copy_(record_observed);
  sample.panel.full.categorical_observed = categorical_observed;

  sample.panel.full.semantic = torch::randn({batch, set_size, config.semantic_width}, floats);
  const auto semantic_observed = torch::rand({batch, set_size}, floats) >= kSemanticAbsentRate;
  sample.panel.full.semantic_observed = semantic_observed & record_observed;

  constexpr int64_t kQueries = 2;
  sample.queries.scalars = torch::full({batch, kQueries, features}, 1000.f, floats);
  sample.queries.scalar_observed = torch::zeros({batch, kQueries, features}, flags);
  sample.queries.categoricals = torch::zeros({batch, kQueries, fields}, ids);
  sample.queries.categorical_observed = torch::zeros({batch, kQueries, fields}, flags);
  sample.queries.semantic = torch::full({batch, kQueries, config.semantic_width}, 1000.f, floats);
  sample.queries.semantic_observed = torch::zeros({batch, kQueries}, flags);
  sample.queries.record_observed = torch::ones({batch, kQueries}, flags);
  sample.queries.categoricals.select(1, 1).select(1, 0).copy_(place.select(1, 0));
  sample.queries.categoricals.select(1, 1).select(1, 1).copy_(kind.select(1, 0));
  sample.queries.categorical_observed.select(1, 1).select(1, 0).fill_(true);
  sample.queries.categorical_observed.select(1, 1).select(1, 1).fill_(true);

  const auto weight = record_observed.to(floats.dtype());
  const auto denom = weight.sum(/*dim=*/1).clamp_min(1).unsqueeze(1);
  const auto empty = (clean * weight.unsqueeze(-1)).sum(/*dim=*/1) / denom;
  auto filter = latent.clone();
  filter.select(1, 0).copy_(latent.select(1, 0) + effect.select(1, 0));
  sample.query_targets = torch::stack({empty, filter}, /*dim=*/1);
  return sample;
}

}  // namespace state_fusion

// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// One checkpoint, one validation probe. Blank a channel by removing its bars
// and measure the native embedding. A removed channel has NaN times, a false
// mask and zero values, which is the leading-pad state retain already uses.
// The repeat encode is the noise floor. Blanking the five-minute channel is
// the scale. Hour or day moves the embedding when its median relative L2 is
// above that floor and at least one percent of the five-minute blank.
#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"
#include "embedding/data/observation_windows.h"
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cp = embedding::encoders::causal_patch_bottleneck_mae::family;
namespace ow = embedding::data::observation_windows;
namespace fs = std::filesystem;

namespace {
std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else out << c;
  }
  out << '"';
  return out.str();
}

void write_text(const fs::path &path, const std::string &body) {
  std::ofstream out(path, std::ios::binary);
  TORCH_CHECK(static_cast<bool>(out), "[channel blank] cannot write ", path.string());
  out << body;
  TORCH_CHECK(static_cast<bool>(out), "[channel blank] short write ", path.string());
}

cp::RawHistory raw_of(const ow::Dataset &data) {
  return {data.values, data.observed, data.channel_ids, data.timestamps, data.sampling_interval};
}

ow::Dataset without_channels(const ow::Dataset &data, const std::vector<int64_t> &channels) {
  ow::Dataset out = data;
  out.values = data.values.clone();
  out.observed = data.observed.clone();
  out.timestamps = data.timestamps.clone();
  for (const auto channel : channels) {
    TORCH_CHECK(channel >= 0 && channel < out.values.size(1), "[channel blank] channel index outside the probe");
    out.values.select(1, channel).zero_();
    out.observed.select(1, channel).fill_(false);
    out.timestamps.select(1, channel).fill_(std::numeric_limits<double>::quiet_NaN());
  }
  return out;
}

struct Encoded {
  torch::Tensor native;
  torch::Tensor valid;
};

Encoded encode_all(cp::Encoder model, const cp::FrozenScaler &scaler, const ow::Dataset &data) {
  std::vector<torch::Tensor> native, valid;
  torch::NoGradGuard guard;
  model->eval();
  for (int64_t first = 0; first < data.rows(); first += 8) {
    std::vector<int64_t> rows;
    for (int64_t i = first; i < std::min<int64_t>(first + 8, data.rows()); ++i) rows.push_back(i);
    const auto batch = ow::subset(data, rows);
    const auto encoded = model->encode_at(raw_of(batch), batch.cutoffs, scaler);
    native.push_back(encoded.native.detach().to(torch::kCPU).to(torch::kFloat64));
    valid.push_back(encoded.valid.detach().to(torch::kCPU).to(torch::kBool).flatten());
  }
  return {torch::cat(native), torch::cat(valid)};
}

std::vector<double> mean_finite_bars(const cp::Config &config, const ow::Dataset &data) {
  const auto window = cp::boundary::retain(raw_of(data), data.cutoffs, config);
  const auto counts = window.times.isfinite().to(torch::kFloat64).sum(2).sum(0) / static_cast<double>(data.rows());
  std::vector<double> out(static_cast<std::size_t>(counts.size(0)));
  for (int64_t c = 0; c < counts.size(0); ++c) {
    const torch::Tensor one = counts[c];
    out[static_cast<std::size_t>(c)] = one.item<double>();
  }
  return out;
}

double median_of(std::vector<double> values) {
  TORCH_CHECK(!values.empty(), "[channel blank] empty distance list");
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  if (n % 2 == 1) return values[n / 2];
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double mean_of(const std::vector<double> &values) {
  double sum = 0;
  for (const auto value : values) sum += value;
  return sum / static_cast<double>(values.size());
}

struct Distances {
  int64_t rows{0};
  int64_t base_valid{0};
  int64_t alt_valid{0};
  double max_abs{0};
  double l2_mean{0};
  double l2_median{0};
  double l2_max{0};
  double cosine_mean{0};
  double cosine_median{0};
  double cosine_min{0};
  double relative_l2_mean{0};
  double relative_l2_median{0};
  double relative_l2_max{0};
};

Distances compare(const Encoded &base, const Encoded &alt) {
  TORCH_CHECK(base.native.sizes() == alt.native.sizes() && base.valid.sizes() == alt.valid.sizes(),
      "[channel blank] encoded shapes disagree");
  TORCH_CHECK(torch::isfinite(base.native).all().item<bool>() && torch::isfinite(alt.native).all().item<bool>(),
      "[channel blank] native embedding must be finite");
  std::vector<double> l2, cosine, relative;
  double max_abs = 0;
  const auto rows = base.native.size(0);
  for (int64_t row = 0; row < rows; ++row) {
    const auto a = base.native[row];
    const auto b = alt.native[row];
    const auto delta = a - b;
    const auto abs = delta.abs().max().item<double>();
    const auto norm_a = a.norm().item<double>();
    const auto norm_b = b.norm().item<double>();
    const auto distance = delta.norm().item<double>();
    max_abs = std::max(max_abs, abs);
    l2.push_back(distance);
    relative.push_back(norm_a > 0 ? distance / norm_a : std::numeric_limits<double>::quiet_NaN());
    cosine.push_back(norm_a > 0 && norm_b > 0 ? a.dot(b).item<double>() / (norm_a * norm_b)
                                               : std::numeric_limits<double>::quiet_NaN());
  }
  for (const auto value : relative) TORCH_CHECK(std::isfinite(value), "[channel blank] relative L2 requires a nonzero base embedding");
  for (const auto value : cosine) TORCH_CHECK(std::isfinite(value), "[channel blank] cosine requires nonzero embeddings");
  Distances out;
  out.rows = rows;
  out.base_valid = base.valid.sum().item<int64_t>();
  out.alt_valid = alt.valid.sum().item<int64_t>();
  out.max_abs = max_abs;
  out.l2_mean = mean_of(l2);
  out.l2_median = median_of(l2);
  out.l2_max = *std::max_element(l2.begin(), l2.end());
  out.cosine_mean = mean_of(cosine);
  out.cosine_median = median_of(cosine);
  out.cosine_min = *std::min_element(cosine.begin(), cosine.end());
  out.relative_l2_mean = mean_of(relative);
  out.relative_l2_median = median_of(relative);
  out.relative_l2_max = *std::max_element(relative.begin(), relative.end());
  return out;
}

std::string bars_json(const std::vector<double> &bars) {
  std::ostringstream out;
  out << std::setprecision(17) << '[';
  for (std::size_t i = 0; i < bars.size(); ++i) {
    if (i) out << ',';
    out << bars[i];
  }
  out << ']';
  return out.str();
}

std::string distances_json(const Distances &d) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "{\"rows\":" << d.rows
      << ",\"base_valid\":" << d.base_valid
      << ",\"alt_valid\":" << d.alt_valid
      << ",\"max_abs\":" << d.max_abs
      << ",\"l2_mean\":" << d.l2_mean
      << ",\"l2_median\":" << d.l2_median
      << ",\"l2_max\":" << d.l2_max
      << ",\"cosine_mean\":" << d.cosine_mean
      << ",\"cosine_median\":" << d.cosine_median
      << ",\"cosine_min\":" << d.cosine_min
      << ",\"relative_l2_mean\":" << d.relative_l2_mean
      << ",\"relative_l2_median\":" << d.relative_l2_median
      << ",\"relative_l2_max\":" << d.relative_l2_max << '}';
  return out.str();
}

struct Condition {
  std::string name;
  std::vector<int64_t> channels;
  std::vector<double> finite_bars;
  Distances distances;
  bool moves{false};
};

void run(int argc, char **argv) {
  TORCH_CHECK(argc == 9, "[channel blank] --settings FILE --checkpoint FILE --validation FILE --output NEWLEAF");
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; i += 2) {
    TORCH_CHECK(args.emplace(argv[i], argv[i + 1]).second, "[channel blank] duplicate option");
  }
  for (const char *key : {"--settings", "--checkpoint", "--validation", "--output"}) {
    TORCH_CHECK(args.count(key), "[channel blank] missing ", key);
  }
  const fs::path settings = args.at("--settings");
  const fs::path checkpoint = args.at("--checkpoint");
  const fs::path validation = args.at("--validation");
  const fs::path output = args.at("--output");
  TORCH_CHECK(fs::is_regular_file(settings) && fs::is_regular_file(checkpoint) && fs::is_regular_file(validation),
      "[channel blank] settings, checkpoint and validation must exist");
  TORCH_CHECK(!fs::exists(output) && fs::is_directory(output.parent_path()),
      "[channel blank] new output leaf with existing parent required");
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  TORCH_CHECK(torch::cuda::is_available(), "[channel blank] CUDA required");
  const auto config = cp::read_settings(settings);
  auto live = cp::load_checkpoint(checkpoint.string(), config.level.device, &config);
  TORCH_CHECK(live.causal_updates == 2048 && live.initialization_seed == 949501,
      "[channel blank] this probe is bound to the packed development checkpoint, seed 949501, 2048 updates");
  const auto identity = live.model->state_identity();
  auto data = ow::read(validation);
  TORCH_CHECK(data.role == "VALIDATION", "[channel blank] validation role required");
  TORCH_CHECK(data.timestamps.dim() == 3, "[channel blank] per-channel timestamps required");
  const auto ids = data.channel_ids.dim() == 1 ? data.channel_ids : data.channel_ids.select(0, 0);
  TORCH_CHECK(ids.size(0) == 3 && ids[0].item<int64_t>() == 6 && ids[1].item<int64_t>() == 9 && ids[2].item<int64_t>() == 15,
      "[channel blank] channel order must be 5m, 1h, 1d");
  const auto baseline_bars = mean_finite_bars(live.model->config(), data);
  TORCH_CHECK(baseline_bars.size() == 3 && baseline_bars[0] == 32 && baseline_bars[1] == 24 && baseline_bars[2] == 24,
      "[channel blank] the probe must retain 32, 24 and 24 finite bars");
  const auto base = encode_all(live.model, live.scaler, data);
  TORCH_CHECK(base.valid.all().item<bool>(), "[channel blank] the untouched probe must be valid on every row");
  const auto repeated = encode_all(live.model, live.scaler, data);
  std::vector<Condition> conditions;
  conditions.push_back({"repeat", {}, baseline_bars, compare(base, repeated), false});
  const std::vector<std::pair<std::string, std::vector<int64_t>>> blanks = {
      {"blank_hour", {1}},
      {"blank_day", {2}},
      {"blank_hour_and_day", {1, 2}},
      {"blank_five_minute", {0}},
  };
  for (const auto &blank : blanks) {
    const auto altered = without_channels(data, blank.second);
    const auto bars = mean_finite_bars(live.model->config(), altered);
    for (const auto channel : blank.second) {
      TORCH_CHECK(bars[static_cast<std::size_t>(channel)] == 0, "[channel blank] a blanked channel must retain no finite bars");
    }
    const auto encoded = encode_all(live.model, live.scaler, altered);
    conditions.push_back({blank.first, blank.second, bars, compare(base, encoded), false});
  }
  TORCH_CHECK(live.model->state_identity() == identity, "[channel blank] encoding changed the checkpoint");
  const auto &repeat = conditions.front().distances;
  const auto &anchor = conditions.back().distances;
  const bool anchor_informative = anchor.relative_l2_median > repeat.relative_l2_median;
  const double scale = anchor.relative_l2_median;
  for (auto &condition : conditions) {
    if (condition.name == "repeat" || condition.name == "blank_five_minute") continue;
    condition.moves = anchor_informative &&
        condition.distances.relative_l2_median > repeat.relative_l2_median &&
        condition.distances.relative_l2_median >= 0.01 * scale;
  }
  std::ostringstream report;
  report << std::setprecision(17)
         << "{\"kind\":\"cpb_packed_channel_blank_v1\",\"status\":\"passed\""
         << ",\"checkpoint_seed\":949501,\"checkpoint_updates\":2048"
         << ",\"native_width\":" << base.native.size(1)
         << ",\"rows\":" << data.rows()
         << ",\"state_identity\":" << quote(identity)
         << ",\"state_unchanged\":true"
         << ",\"intervention\":\"NaN time, false mask, zero values on the named channels\""
         << ",\"rule\":\"hour or day moves when its median relative L2 exceeds the repeat encode and is at least 0.01 of the five-minute blank\""
         << ",\"anchor_blank_informative\":" << (anchor_informative ? "true" : "false")
         << ",\"baseline_mean_finite_bars\":" << bars_json(baseline_bars)
         << ",\"settings_sha256\":" << quote(cp::boundary::sha256_file(settings))
         << ",\"checkpoint_sha256\":" << quote(cp::boundary::sha256_file(checkpoint))
         << ",\"validation_sha256\":" << quote(cp::boundary::sha256_file(validation))
         << ",\"conditions\":[";
  for (std::size_t i = 0; i < conditions.size(); ++i) {
    const auto &condition = conditions[i];
    if (i) report << ',';
    report << "{\"name\":" << quote(condition.name)
           << ",\"moves\":" << (condition.moves ? "true" : "false")
           << ",\"mean_finite_bars\":" << bars_json(condition.finite_bars)
           << ",\"distances\":" << distances_json(condition.distances) << '}';
  }
  report << "]}";
  TORCH_CHECK(fs::create_directory(output), "[channel blank] exclusive output leaf required");
  write_text(output / "summary.json", report.str());
  std::cout << report.str() << '\n';
}
}  // namespace

int main(int argc, char **argv) {
  try {
    run(argc, argv);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

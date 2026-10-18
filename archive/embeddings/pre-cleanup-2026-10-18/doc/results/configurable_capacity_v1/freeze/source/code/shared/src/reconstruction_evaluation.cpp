// SPDX-License-Identifier: MIT
#include "embedding/shared/reconstruction_evaluation.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
namespace fs = std::filesystem;
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[reconstruction evaluation] " + message);
}
std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
template <typename T> std::string numbers(const std::vector<T> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  return out.str() + ']';
}
std::string strings(const std::vector<std::string> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << quote(values[i]);
  }
  return out.str() + ']';
}
std::string fields(const std::map<std::string, std::string> &values) {
  std::ostringstream out;
  out << '{';
  bool first = true;
  for (const auto &[name, value] : values) {
    if (!first) out << ',';
    first = false;
    out << quote(name) << ':' << quote(value);
  }
  return out.str() + '}';
}
void write_text(const fs::path &path, const std::string &value) {
  require(!fs::exists(path), "artifact already exists: " + path.string());
  std::ofstream output(path);
  require(bool(output), "cannot create artifact: " + path.string());
  output << value;
  output.close();
  require(bool(output), "cannot save artifact: " + path.string());
}
void save(const fs::path &path, torch::serialize::OutputArchive &output) {
  require(!fs::exists(path), "archive already exists: " + path.string());
  archive::save_archive(path.string(), output);
}
uint64_t named_stream(const std::string &name) {
  uint64_t value = 14695981039346656037ULL;
  for (const unsigned char c : name) { value ^= c; value *= 1099511628211ULL; }
  return value;
}
Batch clone_batch(const Batch &value) { return {value.data.detach().clone(), value.feature_mask.clone()}; }
void check_validity(const torch::Tensor &valid, int64_t rows, int64_t channels) {
  require(valid.defined() && valid.device().is_cpu() && valid.scalar_type() == torch::kBool &&
          valid.sizes() == torch::IntArrayRef({rows, channels}), "expected CPU bool channel validity [B,C]");
}
void check_latent(const ReconstructionLatent &value, int64_t rows, int64_t channels) {
  require(value.values.defined() && value.values.device().is_cpu() && value.values.is_floating_point() &&
          !value.values.requires_grad() && value.values.dim() == 3 && value.values.size(0) == rows &&
          value.values.size(1) == channels && value.values.size(2) > 0, "expected detached CPU latent [B,C,D]");
  check_validity(value.valid, rows, channels);
  require(torch::isfinite(value.values).all().item<bool>(), "latent vectors must be finite");
  const auto invalid = value.valid.logical_not().unsqueeze(-1).expand_as(value.values);
  require(value.values.masked_select(invalid).eq(0).all().item<bool>(), "invalid latent vectors must be exact zeros");
}
void check_prediction(const torch::Tensor &value, const Batch &batch) {
  require(value.defined() && value.device().is_cpu() && value.is_floating_point() &&
          value.sizes() == batch.data.sizes(), "decoder must return floating CPU raw-unit predictions [B,C,H,F]");
}
void save_observations(const fs::path &path, const ControlledDataset &split) {
  torch::serialize::OutputArchive output;
  output.write("observed", split.observed.data, true);
  output.write("observation_mask", split.observed.feature_mask, true);
  output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
  // Retained solely to identify the generator artifact; never used by this evaluator.
  output.write("generator_labels_unused", split.labels, true);
  save(path, output);
}
struct Exchanges {
  torch::Tensor donor_rows, block_ids;
  int64_t source_groups{0}, blocks{0};
  uint64_t seed{0};
};
Exchanges exchanges(const std::vector<std::string> &sources, uint64_t seed) {
  std::map<std::string, std::vector<int64_t>> rows;
  for (size_t row = 0; row < sources.size(); ++row) rows[sources[row]].push_back(row);
  require(rows.size() >= 2 && rows.size() % 2 == 0, "exchange requires an even count of at least two independent source groups");
  std::vector<std::string> groups;
  for (const auto &[id, records] : rows) {
    require(!id.empty() && records.size() == 2, "exchange requires exactly two records per source group");
    groups.push_back(id);
  }
  std::mt19937_64 random(seed);
  std::shuffle(groups.begin(), groups.end(), random);
  std::vector<int64_t> donor(sources.size(), -1), block(sources.size(), -1);
  for (size_t i = 0; i < groups.size(); i += 2) {
    const auto &a = rows.at(groups[i]), &b = rows.at(groups[i + 1]);
    for (size_t slot = 0; slot < 2; ++slot) {
      donor[a[slot]] = b[slot]; donor[b[slot]] = a[slot];
      block[a[slot]] = block[b[slot]] = i / 2;
    }
  }
  return {torch::tensor(donor, torch::kInt64), torch::tensor(block, torch::kInt64),
          static_cast<int64_t>(groups.size()), static_cast<int64_t>(groups.size() / 2), seed};
}
void save_exchanges(const fs::path &directory, const std::string &split_name,
                    const ControlledDataset &split, const Exchanges &exchange) {
  torch::serialize::OutputArchive output;
  output.write("donor_rows", exchange.donor_rows, true); output.write("exchange_block_ids", exchange.block_ids, true);
  output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
  output.write("seed", archive::text_tensor(std::to_string(exchange.seed)), true);
  save(directory / (split_name + "-donors.pt"), output);
  std::ostringstream manifest;
  manifest << "{\"format_version\":1,\"mapping\":\"disjoint source-group swaps; original row ordinal within each source, never labels\","
      << "\"reuse\":\"same mapping across every target trial and tensor\",\"source_groups\":" << exchange.source_groups
      << ",\"independent_exchange_blocks\":" << exchange.blocks << ",\"seed\":" << quote(std::to_string(exchange.seed))
      << ",\"rows\":[";
  for (int64_t row = 0; row < split.observed.data.size(0); ++row) {
    if (row) manifest << ',';
    const auto donor = exchange.donor_rows[row].item<int64_t>();
    manifest << "{\"row\":" << row << ",\"source_id\":" << quote(split.source_ids.at(row))
        << ",\"donor_row\":" << donor << ",\"donor_source_id\":" << quote(split.source_ids.at(donor))
        << ",\"exchange_block\":" << exchange.block_ids[row].item<int64_t>() << '}';
  }
  write_text(directory / (split_name + "-donors.json"), manifest.str() + "]}");
}
struct CaseTrials {
  std::vector<torch::Tensor> predictions, valid;
};
struct Metrics {
  torch::Tensor channel_mae, channel_huber, target_counts, eligible_channels;
  torch::Tensor example_mae, example_huber, valid;
  int64_t total_examples{0}, valid_examples{0}, valid_channels{0}, target_cells{0}, full_target_cells{0};
  double mae{0}, huber{0};
};
Metrics metrics(const torch::Tensor &prediction, const torch::Tensor &channel_valid,
                const torch::Tensor &query, const Batch &batch, const ObservationScaler &scaler, double delta) {
  // Shapes are [trial,B,C,H,F] and [trial,B,C]; every raw target cell is queried once.
  const auto support = query.logical_and(channel_valid.unsqueeze(-1).unsqueeze(-1));
  const auto target = batch.data.unsqueeze(0).expand_as(prediction);
  require(torch::isfinite(prediction.masked_select(support)).all().item<bool>() &&
          torch::isfinite(target.masked_select(support)).all().item<bool>(), "nonfinite prediction/target on fixed valid query support");
  const auto safe_prediction = torch::where(support, prediction.to(torch::kFloat64), torch::zeros_like(prediction, torch::kFloat64));
  const auto safe_target = torch::where(support, target, torch::zeros_like(target));
  const auto residual = (safe_prediction - safe_target) / scaler.scale.view({1, 1, batch.data.size(1), 1, batch.data.size(3)});
  require(torch::isfinite(residual.masked_select(support)).all().item<bool>(), "standardized target residual overflow");
  const auto absolute = residual.abs();
  const auto quadratic = residual.clamp(-delta, delta).square() * .5;
  const auto huber = torch::where(absolute.le(delta), quadratic, delta * (absolute - .5 * delta));
  const std::vector<int64_t> cell_axes{0, 3, 4};
  Metrics out;
  out.target_counts = support.sum(cell_axes);
  out.eligible_channels = out.target_counts.gt(0);
  out.valid = out.eligible_channels.any(1);
  out.channel_mae = absolute.sum(cell_axes) / out.target_counts.clamp_min(1).to(torch::kFloat64);
  out.channel_huber = huber.sum(cell_axes) / out.target_counts.clamp_min(1).to(torch::kFloat64);
  const auto channel_count = out.eligible_channels.sum(1).clamp_min(1).to(torch::kFloat64);
  out.example_mae = out.channel_mae.sum(1) / channel_count;
  out.example_huber = out.channel_huber.sum(1) / channel_count;
  require(torch::isfinite(out.example_mae).all().item<bool>() && torch::isfinite(out.example_huber).all().item<bool>(),
          "reconstruction reduction overflow");
  out.total_examples = batch.data.size(0); out.valid_examples = out.valid.sum().item<int64_t>();
  out.valid_channels = out.eligible_channels.sum().item<int64_t>(); out.target_cells = out.target_counts.sum().item<int64_t>();
  out.full_target_cells = query.sum().item<int64_t>();
  if (out.valid_examples) {
    out.mae = out.example_mae.masked_select(out.valid).mean().item<double>();
    out.huber = out.example_huber.masked_select(out.valid).mean().item<double>();
    require(std::isfinite(out.mae) && std::isfinite(out.huber), "reconstruction summary mean overflow");
  }
  return out;
}
void write_metrics(torch::serialize::OutputArchive &output, const std::string &prefix, const Metrics &value) {
  output.write(prefix + "channel_standardized_mae", value.channel_mae, true);
  output.write(prefix + "channel_standardized_huber", value.channel_huber, true);
  output.write(prefix + "target_counts", value.target_counts, true);
  output.write(prefix + "eligible_channels", value.eligible_channels, true);
  output.write(prefix + "example_standardized_mae", value.example_mae, true);
  output.write(prefix + "example_standardized_huber", value.example_huber, true);
  output.write(prefix + "example_valid", value.valid, true);
}
std::string metrics_json(const Metrics &value, int64_t channels) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"status\":" << quote(value.valid_examples ? "measured" : "unsupported_zero_support")
      << ",\"total_examples\":" << value.total_examples << ",\"valid_examples\":" << value.valid_examples
      << ",\"example_coverage\":" << double(value.valid_examples) / value.total_examples
      << ",\"total_channels\":" << value.total_examples * channels << ",\"valid_channels\":" << value.valid_channels
      << ",\"channel_coverage\":" << double(value.valid_channels) / (value.total_examples * channels)
      << ",\"full_target_cells\":" << value.full_target_cells << ",\"valid_target_cells\":" << value.target_cells
      << ",\"target_coverage\":" << double(value.target_cells) / value.full_target_cells << ",\"standardized_mae\":";
  if (value.valid_examples) out << value.mae; else out << "null";
  out << ",\"standardized_huber\":";
  if (value.valid_examples) out << value.huber; else out << "null";
  return out.str() + '}';
}
struct Effect {
  double estimate{0}, lower{0}, upper{0};
  int64_t examples{0}, groups{0}, blocks{0};
  bool supported{false};
};
Effect effect(const torch::Tensor &candidate, const torch::Tensor &intact, const torch::Tensor &valid,
              const Exchanges &exchange, const std::vector<std::string> &sources, uint64_t seed, int64_t replicates) {
  std::map<int64_t, std::pair<double, int64_t>> blocks;
  std::set<std::string> groups;
  Effect out;
  for (int64_t row = 0; row < valid.size(0); ++row) if (valid[row].item<bool>()) {
    const double value = candidate[row].item<double>() - intact[row].item<double>();
    require(std::isfinite(value), "nonfinite paired reconstruction effect");
    auto &block = blocks[exchange.block_ids[row].item<int64_t>()];
    block.first += value; ++block.second; ++out.examples;
    require(std::isfinite(block.first), "paired exchange-block sum overflow");
    groups.insert(sources.at(row));
  }
  out.groups = groups.size(); out.blocks = blocks.size();
  if (!out.examples) return out;
  std::vector<std::pair<double, int64_t>> observations;
  double sum = 0;
  for (const auto &[id, block] : blocks) {
    (void)id;
    observations.push_back(block);
    sum += block.first;
    require(std::isfinite(sum), "paired effect total overflow");
  }
  out.estimate = sum / out.examples;
  require(std::isfinite(out.estimate), "paired effect estimate overflow");
  if (out.blocks < 2) return out;
  std::mt19937_64 random(seed);
  std::uniform_int_distribution<size_t> draw(0, observations.size() - 1);
  std::vector<double> samples;
  samples.reserve(replicates);
  for (int64_t replicate = 0; replicate < replicates; ++replicate) {
    double numerator = 0; int64_t denominator = 0;
    for (size_t sampled = 0; sampled < observations.size(); ++sampled) {
      const auto &block = observations[draw(random)];
      numerator += block.first; denominator += block.second;
      require(std::isfinite(numerator), "paired bootstrap sum overflow");
    }
    const double sample = numerator / denominator;
    require(std::isfinite(sample), "paired bootstrap sample overflow");
    samples.push_back(sample);
  }
  std::sort(samples.begin(), samples.end());
  out.lower = samples[size_t(.025 * (samples.size() - 1))];
  out.upper = samples[size_t(.975 * (samples.size() - 1))]; out.supported = true;
  require(std::isfinite(out.lower) && std::isfinite(out.upper), "paired bootstrap interval overflow");
  return out;
}
std::string effect_json(const Effect &value, int64_t replicates) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"status\":"
      << quote(value.supported ? "measured" : value.examples ? "unsupported_insufficient_exchange_blocks" : "unsupported_zero_support")
      << ",\"method\":\"paired percentile bootstrap of disjoint two-source exchange blocks; within-run only\","
      << "\"replicates\":" << replicates << ",\"confidence\":0.95,\"valid_examples\":" << value.examples
      << ",\"valid_source_groups\":" << value.groups << ",\"valid_independent_exchange_blocks\":" << value.blocks << ",\"estimate\":";
  if (value.supported) out << value.estimate; else out << "null";
  out << ",\"lower\":";
  if (value.supported) out << value.lower; else out << "null";
  out << ",\"upper\":";
  if (value.supported) out << value.upper; else out << "null";
  return out.str() + '}';
}
std::string evaluate_split(const fs::path &directory, const std::string &split_name, const ControlledDataset &split,
                           const ReconstructionProvider &provider, const ObservationScaler &scaler,
                           const ReconstructionCard &card, uint64_t seed, Task task,
                           int64_t &latent_width, torch::Dtype &latent_dtype) {
  const auto B = split.observed.data.size(0), C = card.shape.channel_count, H = card.shape.history_length;
  const auto F = card.shape.input_width, K = H / card.patch_length;
  const auto exchange = exchanges(split.source_ids, stream_seed(seed, named_stream(task_name(task) + "/" + split_name + "/donors")));
  save_exchanges(directory, split_name, split, exchange);
  std::map<std::string, CaseTrials> cases;
  std::vector<torch::Tensor> queries;
  for (int64_t trial = 0; trial < K; ++trial) {
    auto query = torch::zeros_like(split.observed.feature_mask);
    for (int64_t channel = 0; channel < C; ++channel) {
      const auto begin = ((trial + channel) % K) * card.patch_length;
      query.select(1, channel).narrow(1, begin, card.patch_length).fill_(true);
    }
    const auto visible = split.observed.feature_mask.logical_and(query.logical_not());
    Batch input{torch::where(visible, split.observed.data, torch::zeros_like(split.observed.data)), visible};
    ReconstructionLatent latent;
    std::map<std::string, ReconstructionPrediction> predictions;
    {
      torch::NoGradGuard no_grad;
      latent = provider.encode_visible(clone_batch(input));
      check_latent(latent, B, C);
      latent.values = latent.values.detach().clone(); latent.valid = latent.valid.clone();
      if (latent_width < 0) { latent_width = latent.values.size(2); latent_dtype = latent.values.scalar_type(); }
      require(latent.values.size(2) == latent_width && latent.values.scalar_type() == latent_dtype,
              "provider changed latent width/dtype across target trials or splits");
      const auto donor_latent = latent.values.index_select(0, exchange.donor_rows);
      predictions.emplace("intact", ReconstructionPrediction{provider.decode(latent.values.clone()).detach().clone(), latent.valid.clone()});
      predictions.emplace("shuffled", ReconstructionPrediction{provider.decode(donor_latent.clone()).detach().clone(),
          latent.valid.index_select(0, exchange.donor_rows).clone()});
      predictions.emplace("zeroed", ReconstructionPrediction{provider.decode(torch::zeros_like(latent.values)).detach().clone(), latent.valid.clone()});
      predictions.emplace("metadata_only", ReconstructionPrediction{provider.metadata_predict(visible.clone()).detach().clone(),
          torch::ones({B, C}, torch::kBool)});
      predictions.emplace("training_mean", ReconstructionPrediction{scaler.mean.view({1, C, 1, F}).expand({B, C, H, F}).clone(),
          torch::ones({B, C}, torch::kBool)});
      if (provider.untrained_predict) {
        auto random = provider.untrained_predict(clone_batch(input));
        random.values = random.values.detach().clone(); random.valid = random.valid.clone();
        predictions.emplace("untrained", std::move(random));
      }
    }
    torch::serialize::OutputArchive artifact;
    artifact.write("trial", torch::tensor(trial), true);
    artifact.write("visible_values", input.data, true); artifact.write("visible_support", visible, true);
    artifact.write("target_mask", query, true);
    artifact.write("target_values", torch::where(query, split.observed.data, torch::zeros_like(split.observed.data)), true);
    artifact.write("latent", latent.values, true); artifact.write("latent_valid", latent.valid, true);
    artifact.write("donor_rows", exchange.donor_rows, true); artifact.write("exchange_block_ids", exchange.block_ids, true);
    for (const auto &[name, prediction] : predictions) {
      check_prediction(prediction.values, split.observed);
      check_validity(prediction.valid, B, C);
      const auto fixed_support = query.logical_and(prediction.valid.unsqueeze(-1).unsqueeze(-1));
      require(torch::isfinite(prediction.values.masked_select(fixed_support)).all().item<bool>(),
              "nonfinite " + name + " prediction on fixed valid targets");
      cases[name].predictions.push_back(prediction.values.to(torch::kFloat64));
      cases[name].valid.push_back(prediction.valid);
      artifact.write(name + "_prediction", prediction.values, true);
      artifact.write(name + "_valid", prediction.valid, true);
    }
    save(directory / (split_name + "-trial-" + std::to_string(trial) + ".pt"), artifact);
    queries.push_back(query);
  }
  const auto query = torch::stack(queries);
  require(query.sum(0).eq(1).all().item<bool>(), "target trials must query each declared cell exactly once");
  std::ostringstream out;
  out << "{\"split\":" << quote(split_name) << ",\"source_groups\":" << exchange.source_groups
      << ",\"independent_exchange_blocks\":" << exchange.blocks << ",\"query_trials\":" << K
      << ",\"latent_width_per_channel\":" << latent_width
      << ",\"target_cells_per_example\":" << C * H * F << ",\"donor_archive\":" << quote(split_name + "-donors.pt")
      << ",\"donor_manifest\":" << quote(split_name + "-donors.json") << ",\"cases\":[";
  bool first = true;
  for (const auto &[name, value] : cases) {
    const auto prediction = torch::stack(value.predictions), valid = torch::stack(value.valid);
    const auto measured = metrics(prediction, valid, query, split.observed, scaler, card.huber_delta);
    torch::serialize::OutputArchive output;
    write_metrics(output, "", measured);
    output.write("trial_channel_valid", valid, true);
    save(directory / (split_name + "-" + name + "-metrics.pt"), output);
    if (!first) out << ',';
    first = false;
    out << "{\"case\":" << quote(name) << ",\"coverage_semantics\":"
        << quote(name == "metadata_only" || name == "training_mean" ? "structural query support; control, not learned observed-support validity" : "observed-support latent validity")
        << ",\"metrics\":" << metrics_json(measured, C) << ",\"metrics_archive\":" << quote(split_name + "-" + name + "-metrics.pt") << '}';
  }
  if (!provider.untrained_predict)
    out << ",{\"case\":\"untrained\",\"status\":\"unavailable\",\"reason\":\"provider did not supply an untrained control\"}";
  out << "],\"comparisons\":[";
  first = true;
  const auto &intact = cases.at("intact");
  const auto intact_prediction = torch::stack(intact.predictions), intact_valid = torch::stack(intact.valid);
  for (const auto &[name, value] : cases) {
    if (name == "intact") continue;
    const auto prediction = torch::stack(value.predictions), valid = torch::stack(value.valid).logical_and(intact_valid);
    const auto candidate = metrics(prediction, valid, query, split.observed, scaler, card.huber_delta);
    const auto actual = metrics(intact_prediction, valid, query, split.observed, scaler, card.huber_delta);
    require(torch::equal(candidate.target_counts, actual.target_counts) && torch::equal(candidate.valid, actual.valid),
            "paired reconstruction populations differ");
    const auto comparison_seed = stream_seed(seed, named_stream(task_name(task) + "/" + split_name + "/" + name + "/bootstrap"));
    const auto mae = effect(candidate.example_mae, actual.example_mae, actual.valid, exchange, split.source_ids,
                            comparison_seed, card.bootstrap_replicates);
    const auto huber = effect(candidate.example_huber, actual.example_huber, actual.valid, exchange, split.source_ids,
                              comparison_seed, card.bootstrap_replicates);
    torch::serialize::OutputArchive output;
    write_metrics(output, "candidate_", candidate); write_metrics(output, "intact_", actual);
    output.write("common_trial_channel_valid", valid, true);
    output.write("exchange_block_ids", exchange.block_ids, true);
    output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
    save(directory / (split_name + "-" + name + "-minus-intact.pt"), output);
    if (!first) out << ',';
    first = false;
    out << "{\"comparison\":" << quote(name + "_minus_intact") << ",\"effect_direction\":\"positive error difference favors intact latents\","
        << "\"population_rule\":\"same candidate AND intact channel validity and fixed target coordinates\","
        << "\"intact_common_metrics\":" << metrics_json(actual, C) << ",\"candidate_common_metrics\":" << metrics_json(candidate, C)
        << ",\"primary_standardized_mae_effect\":" << effect_json(mae, card.bootstrap_replicates)
        << ",\"secondary_standardized_huber_effect\":" << effect_json(huber, card.bootstrap_replicates)
        << ",\"bootstrap_seed\":" << quote(std::to_string(comparison_seed))
        << ",\"metrics_archive\":" << quote(split_name + "-" + name + "-minus-intact.pt") << '}';
  }
  return out.str() + "]}";
}
} // namespace

void validate_reconstruction_card(const ReconstructionCard &card) {
  require(!card.id.empty() && card.version == 1 && card.policy_version == "1.1", "only reconstruction card version 1 and policy 1.1 are implemented");
  require(card.stage == "development", "reconstruction track supports development only");
  require(card.shape.channel_count > 0 && card.shape.history_length >= 8 && card.shape.input_width > 0 &&
          card.shape.dtype == torch::kFloat64 && card.shape.device.is_cpu(), "controlled reconstruction requires positive C/F, H>=8 and CPU float64");
  require(card.patch_length > 0 && card.shape.history_length % card.patch_length == 0 &&
          card.shape.history_length / card.patch_length >= 3, "complete target patches require H divisible by P and at least three patches");
  require(card.channel_ids.size() == size_t(card.shape.channel_count) &&
          std::set<int64_t>(card.channel_ids.begin(), card.channel_ids.end()).size() == card.channel_ids.size(), "unique channel IDs must match declared channels");
  std::istringstream input(card.feature_units);
  std::string unit;
  int64_t units = 0;
  while (std::getline(input, unit, ',')) { require(!unit.empty(), "empty feature unit"); ++units; }
  require(!card.feature_units.empty() && card.feature_units.back() != ',' && units == card.shape.input_width, "feature units must match declared features");
  require(std::isfinite(card.sampling_interval) && card.sampling_interval > 0 &&
          std::isfinite((card.shape.history_length - 1) * card.sampling_interval) &&
          std::isfinite(card.huber_delta) && card.huber_delta > 0, "invalid interval, endpoint or Huber delta");
  require(card.train_pairs >= 2 && card.validation_pairs >= 2 && card.test_pairs >= 2 &&
          card.validation_pairs % 2 == 0 && card.test_pairs % 2 == 0 && card.threads > 0 &&
          card.bootstrap_replicates >= 100, "invalid fit budget, exchange group count, thread count or bootstrap budget");
  require(card.train_pairs <= std::numeric_limits<int64_t>::max() / 6 &&
          card.validation_pairs <= std::numeric_limits<int64_t>::max() / 6 &&
          card.test_pairs <= std::numeric_limits<int64_t>::max() / 6, "pair count overflow");
  require(!card.seeds.empty() && std::set<uint64_t>(card.seeds.begin(), card.seeds.end()).size() == card.seeds.size(), "nonempty unique seeds required");
  require(!card.tasks.empty(), "at least one task required");
  std::set<Task> tasks;
  for (const auto task : card.tasks) {
    (void)task_name(task);
    require(tasks.insert(task).second, "duplicate task");
    require(task != Task::lag_sign || card.shape.channel_count >= 2, "lag-sign task needs at least two channels");
  }
  require(!card.provider_recipe.empty(), "resolved provider recipe must be frozen before generation or fitting");
}
std::string reconstruction_card_json(const ReconstructionCard &card) {
  validate_reconstruction_card(card);
  std::ostringstream out;
  out << std::setprecision(17) << "{\"id\":" << quote(card.id) << ",\"version\":" << card.version
      << ",\"policy_version\":" << quote(card.policy_version) << ",\"stage\":" << quote(card.stage)
      << ",\"shape\":[" << card.shape.channel_count << ',' << card.shape.history_length << ',' << card.shape.input_width
      << "],\"raw_dtype\":\"float64\",\"device\":\"cpu\",\"channel_ids\":" << numbers(card.channel_ids)
      << ",\"feature_units\":" << quote(card.feature_units) << ",\"sampling_interval\":" << card.sampling_interval
      << ",\"patch_length\":" << card.patch_length << ",\"train_pairs\":" << card.train_pairs
      << ",\"validation_pairs\":" << card.validation_pairs << ",\"test_pairs\":" << card.test_pairs
      << ",\"threads\":" << card.threads << ",\"bootstrap_replicates\":" << card.bootstrap_replicates << ",\"seeds\":[";
  for (size_t i = 0; i < card.seeds.size(); ++i) { if (i) out << ','; out << quote(std::to_string(card.seeds[i])); }
  out << "],\"tasks\":[";
  for (size_t i = 0; i < card.tasks.size(); ++i) { if (i) out << ','; out << quote(task_name(card.tasks[i])); }
  out << "],\"provider_recipe\":" << quote(card.provider_recipe)
      << ",\"population\":\"fully observed controlled source pairs; labels unused\","
      << "\"target_recipe\":\"all K trials; channel c targets patch (trial+c)%K; identical visible/query support for all examples\","
      << "\"interventions\":[\"intact\",\"source-group shuffled\",\"zeroed\",\"metadata-only fitted control\",\"untrained if supplied\",\"training-mean control\"],"
      << "\"scoring\":{\"prediction_units\":\"raw input units\",\"metric_scaling\":\"training-only per-channel/feature population std, floor 1e-8\","
      << "\"primary\":\"standardized MAE\",\"secondary\":\"standardized Huber\",\"huber_delta\":" << card.huber_delta
      << ",\"reduction\":\"target cells across all trials within channel, eligible channels within example, eligible examples equally\"},"
      << "\"uncertainty\":\"disjoint two-source exchange-block percentile bootstrap, 95%, within-run only\","
      << "\"limits\":[\"zeroed latents can be out of distribution\",\"no outage sweep\",\"no external or future-target evidence\",\"no acceptance or across-training-run uncertainty\"]}";
  return out.str();
}
void run_reconstruction_evaluation(const ReconstructionRun &run, const ReconstructionProviderFactory &factory) {
  validate_reconstruction_card(run.card);
  require(bool(factory), "empty reconstruction provider factory");
  const fs::path output(run.output_directory);
  require(!output.empty() && !fs::exists(output), "output must name a new artifact directory");
  fs::create_directories(output);
  write_text(output / "reconstruction-card.json", reconstruction_card_json(run.card));
  const auto &card = run.card;
  torch::set_num_threads(card.threads);
  std::ostringstream report;
  report << "{\"format_version\":1,\"protocol\":\"controlled-reconstruction-v1\",\"card\":\"reconstruction-card.json\","
      << "\"policy_version\":" << quote(card.policy_version) << ",\"stage\":" << quote(card.stage)
      << ",\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"interpretation\":\"held-out development decoder reliance, not consumer acceptance\","
      << "\"effect_direction\":\"positive control/intervention minus intact error favors intact latents\","
      << "\"uncertainty\":\"within-run independent exchange blocks; variants and target trials are dependent\","
      << "\"limits\":[\"metadata/untrained budgets disclosed by adapter\",\"zero intervention may be out of distribution\","
      << "\"no acceptance, cost or outage claim\",\"no across-training-run intervals\"],\"runs\":[";
  bool first = true;
  for (const auto seed : card.seeds) for (const auto task : card.tasks) {
    const auto directory = output / ("seed-" + std::to_string(seed) + "-" + task_name(task));
    fs::create_directory(directory);
    const auto protocol = make_controlled_protocol(task, card.shape, card.train_pairs, card.validation_pairs, card.test_pairs, seed, 0);
    validate_protocol(protocol);
    require(protocol.training.observed.feature_mask.all().item<bool>() &&
            protocol.validation.observed.feature_mask.all().item<bool>() &&
            protocol.testing.observed.feature_mask.all().item<bool>(), "this card requires fully observed inputs");
    save_observations(directory / "training-observations.pt", protocol.training);
    save_observations(directory / "validation-observations.pt", protocol.validation);
    save_observations(directory / "testing-observations.pt", protocol.testing);
    write_text(directory / "source-manifest.json", "{\"card\":\"../reconstruction-card.json\",\"training\":" + strings(protocol.training.source_ids) +
        ",\"validation\":" + strings(protocol.validation.source_ids) + ",\"testing\":" + strings(protocol.testing.source_ids) +
        ",\"labels\":\"unused generator artifact; no scoring or provider access\"}");
    const ObservationScaler scaler(protocol.training.observed);
    torch::serialize::OutputArchive metric_fit;
    metric_fit.write("mean", scaler.mean, true); metric_fit.write("scale", scaler.scale, true); metric_fit.write("counts", scaler.counts, true);
    metric_fit.write("channel_ids", torch::tensor(card.channel_ids, torch::kInt64), true);
    metric_fit.write("units", archive::text_tensor(card.feature_units), true);
    metric_fit.write("source_ids_json", archive::text_tensor(strings(protocol.training.source_ids)), true);
    save(directory / "metric-scaler.pt", metric_fit);
    const ProviderFitInput input{clone_batch(protocol.training.observed), card.shape, seed, protocol.training.source_ids, card.channel_ids,
        card.feature_units, card.id + "/v1/" + task_name(task), card.sampling_interval, (card.shape.history_length - 1) * card.sampling_interval};
    const auto provider = factory(input);
    require(!provider.name.empty() && !provider.export_semantics.empty() && !provider.provenance.empty() &&
            bool(provider.encode_visible) && bool(provider.decode) && bool(provider.metadata_predict),
            "provider needs identity, export semantics, provenance, visible encoder, decoder and metadata-only control");
    const auto assets = directory / "provider-assets";
    fs::create_directory(assets);
    if (provider.save_assets) provider.save_assets(assets.string());
    const auto audit = "{\"provider\":" + quote(provider.name) + ",\"export_semantics\":" + quote(provider.export_semantics) +
        ",\"provenance\":" + quote(provider.provenance) + ",\"fit_seed\":" + quote(std::to_string(seed)) +
        ",\"fit_training_rows\":" + std::to_string(protocol.training.observed.data.size(0)) + ",\"audit_fields\":" + fields(provider.audit_fields) + '}';
    write_text(assets / "provider-audit.json", audit);
    int64_t latent_width = -1;
    torch::Dtype latent_dtype = torch::kFloat64;
    const auto validation = evaluate_split(directory, "validation", protocol.validation, provider, scaler, card, seed, task, latent_width, latent_dtype);
    const auto testing = evaluate_split(directory, "testing", protocol.testing, provider, scaler, card, seed, task, latent_width, latent_dtype);
    if (!first) report << ',';
    first = false;
    report << "{\"seed\":" << quote(std::to_string(seed)) << ",\"task\":" << quote(task_name(task))
        << ",\"artifact_directory\":" << quote(directory.filename().string()) << ",\"provider_assets\":\"provider-assets\","
        << "\"provider_audit\":" << audit << ",\"metric_fit\":\"metric-scaler.pt\",\"validation\":" << validation << ",\"testing\":" << testing << '}';
  }
  write_text(output / "report.json", report.str() + "]}");
}
} // namespace embedding::evaluation

// SPDX-License-Identifier: MIT
#include "embedding/encoders/mtf_jepa_mae_vicreg/evaluation.h"
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_solve.h>
#include <torch/version.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace embedding::encoders::mtf_jepa_mae_vicreg {
namespace {
constexpr int64_t classes = 4;
constexpr double ridge_penalty = 1.0;

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[embedding evaluate] " + message);
}

uint64_t stream_seed(uint64_t seed, uint64_t stream) {
  uint64_t x = seed + 0x9e3779b97f4a7c15ULL * stream;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}

std::string tensor_array(const torch::Tensor &value) {
  const auto cpu = value.detach().to(torch::kCPU).to(torch::kFloat64).contiguous().reshape({-1});
  std::ostringstream out;
  out << std::setprecision(12) << '[';
  const auto *data = cpu.data_ptr<double>();
  for (int64_t i = 0; i < cpu.numel(); ++i) {
    if (i) out << ',';
    require(std::isfinite(data[i]), "report tensor contains non-finite values");
    out << data[i];
  }
  out << ']';
  return out.str();
}

// A reproducibility checksum, not a cryptographic integrity claim.
std::string fingerprint(const std::vector<torch::Tensor> &tensors) {
  uint64_t hash = 14695981039346656037ULL;
  for (const auto &tensor : tensors) {
    const auto cpu = tensor.detach().to(torch::kCPU).contiguous();
    for (const auto dimension : cpu.sizes())
      for (int i = 0; i < 8; ++i) {
        hash ^= (static_cast<uint64_t>(dimension) >> (i * 8)) & 255;
        hash *= 1099511628211ULL;
      }
    const auto *bytes = static_cast<const unsigned char *>(cpu.data_ptr());
    for (size_t i = 0; i < static_cast<size_t>(cpu.numel()) * cpu.element_size(); ++i) {
      hash ^= bytes[i];
      hash *= 1099511628211ULL;
    }
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

struct InputScaler {
  torch::Tensor mean, scale, counts;
  explicit InputScaler(const ::embedding::Batch &train) {
    const auto x = train.data.to(torch::kFloat64);
    const auto weights = train.feature_mask.to(torch::kFloat64);
    counts = weights.sum({0, 1, 2});
    require(counts.gt(0).all().item<bool>(), "training split has an entirely missing feature");
    mean = (x * weights).sum({0, 1, 2}) / counts;
    scale = (((x - mean).pow(2) * weights).sum({0, 1, 2}) / counts).sqrt().clamp_min(1e-6);
  }
  InputScaler(const std::string &path, int64_t input_width) {
    torch::serialize::InputArchive archive;
    archive.load_from(path, torch::kCPU);
    torch::Tensor version;
    archive.read("format_version", version, true);
    require(version.numel() == 1 && version.item<int64_t>() == 1, "unsupported normalization archive version");
    archive.read("mean", mean, true);
    archive.read("scale", scale, true);
    archive.read("observed_counts", counts, true);
    for (const auto &value : {mean, scale, counts})
      require(value.scalar_type() == torch::kFloat64 && value.dim() == 1 && value.numel() == input_width &&
                  torch::isfinite(value).all().item<bool>(),
              "normalization archive must contain finite float64 [F] mean/scale/observed_counts");
    require(scale.gt(0).all().item<bool>() && counts.gt(0).all().item<bool>(),
            "normalization scale and observed_counts must be positive");
  }
  void save(const std::string &path) const {
    const std::filesystem::path destination(path);
    if (!destination.parent_path().empty()) std::filesystem::create_directories(destination.parent_path());
    const auto temporary = path + ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    torch::serialize::OutputArchive archive;
    archive.write("format_version", torch::tensor(int64_t{1}), true);
    archive.write("mean", mean, true);
    archive.write("scale", scale, true);
    archive.write("observed_counts", counts, true);
    archive.save_to(temporary);
    std::filesystem::rename(temporary, destination);
  }
  ::embedding::Batch apply(const ::embedding::Batch &batch) const {
    auto standardized = ((batch.data.to(torch::kFloat64) - mean) / scale).to(torch::kFloat32);
    return {torch::where(batch.feature_mask, standardized, torch::zeros_like(standardized)),
            batch.feature_mask.clone()};
  }
};

struct Probe {
  torch::Tensor mean, scale, weights, intercept;
  Probe(const torch::Tensor &features, const torch::Tensor &labels, const torch::Tensor &valid) {
    const auto rows = valid.nonzero().reshape({-1});
    require(rows.size(0) >= classes, "too few valid training rows for a four-class probe");
    const auto x = features.to(torch::kFloat64).index_select(0, rows);
    mean = x.mean(0);
    scale = (x - mean).pow(2).mean(0).sqrt().clamp_min(1e-6);
    const auto centered = (x - mean) / scale;
    auto y = torch::zeros({rows.size(0), classes}, torch::kFloat64);
    y.scatter_(1, labels.index_select(0, rows).unsqueeze(1), 1.0);
    intercept = y.mean(0);
    const auto target = y - intercept;
    if (x.size(1) > x.size(0)) {
      auto gram = centered.matmul(centered.transpose(0, 1));
      gram += torch::eye(gram.size(0), gram.options()) * ridge_penalty;
      weights = centered.transpose(0, 1).matmul(at::linalg_solve(gram, target));
    } else {
      auto gram = centered.transpose(0, 1).matmul(centered);
      gram += torch::eye(gram.size(0), gram.options()) * ridge_penalty;
      weights = at::linalg_solve(gram, centered.transpose(0, 1).matmul(target));
    }
    require(torch::isfinite(weights).all().item<bool>(), "probe solve was non-finite");
  }
  torch::Tensor predict(const torch::Tensor &features) const {
    return (((features.to(torch::kFloat64) - mean) / scale).matmul(weights) + intercept).argmax(1);
  }
};

using FeatureMap = std::map<std::string, torch::Tensor>;

FeatureMap feature_surfaces(Model &random_model, Model &trained_model, const ::embedding::Batch &batch) {
  torch::NoGradGuard no_grad;
  random_model->eval();
  trained_model->eval();
  const auto random = random_model->encode(batch.data, batch.feature_mask);
  const auto trained = trained_model->encode(batch.data, batch.feature_mask);
  const auto tokenized = trained_model->tokenize(batch.data, batch.feature_mask);
  const auto time = torch::where(tokenized.time_reconstruction_mask,
                                tokenized.time_reconstruction_targets,
                                torch::zeros_like(tokenized.time_reconstruction_targets));
  const auto frequency = torch::where(tokenized.frequency_reconstruction_mask,
                                     tokenized.frequency_reconstruction_targets,
                                     torch::zeros_like(tokenized.frequency_reconstruction_targets));
  return {{"standardized_raw", batch.data.flatten(1)},
          {"raw_descriptors", torch::cat({time.flatten(1), frequency.flatten(1)}, 1).to(torch::kCPU)},
          {"mask_only", batch.feature_mask.to(torch::kFloat32).flatten(1)},
          {"untrained_global", random.pooled_embedding.to(torch::kCPU)},
          {"untrained_channels", random.pooled_by_channel.flatten(1).to(torch::kCPU)},
          {"trained_global", trained.pooled_embedding.to(torch::kCPU)},
          {"trained_channels", trained.pooled_by_channel.flatten(1).to(torch::kCPU)}};
}

torch::Tensor sample_valid(const ::embedding::Batch &batch) { return batch.feature_mask.flatten(1).any(1); }

std::string score_json(const Probe &probe, const torch::Tensor &features,
                       const torch::Tensor &labels, const torch::Tensor &valid) {
  const auto rows = valid.nonzero().reshape({-1});
  std::ostringstream out;
  out << std::setprecision(12) << "{\"valid_samples\":" << rows.size(0)
      << ",\"total_samples\":" << labels.size(0)
      << ",\"coverage\":" << valid.to(torch::kFloat64).mean().item<double>();
  auto confusion = torch::zeros({classes, classes}, torch::kInt64);
  if (!rows.numel()) {
    out << ",\"accuracy\":null,\"balanced_accuracy\":null,\"confusion\":"
        << tensor_array(confusion) << '}';
    return out.str();
  }
  const auto predicted = probe.predict(features).index_select(0, rows);
  const auto truth = labels.index_select(0, rows);
  auto matrix = confusion.accessor<int64_t, 2>();
  const auto p = predicted.accessor<int64_t, 1>();
  const auto t = truth.accessor<int64_t, 1>();
  for (int64_t i = 0; i < rows.numel(); ++i) ++matrix[t[i]][p[i]];
  double balanced = 0;
  int present = 0;
  for (int64_t label = 0; label < classes; ++label) {
    const auto count = confusion[label].sum().item<int64_t>();
    if (count) { balanced += static_cast<double>(matrix[label][label]) / count; ++present; }
  }
  out << ",\"accuracy\":" << predicted.eq(truth).to(torch::kFloat64).mean().item<double>()
      << ",\"balanced_accuracy\":" << balanced / std::max(1, present)
      << ",\"classes_with_valid_samples\":" << present
      << ",\"confusion\":" << tensor_array(confusion) << '}';
  return out.str();
}

std::string diagnostics_json(const ::embedding::RepresentationDiagnostics &d) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"valid_rows\":" << d.valid_rows << ",\"dimensions\":" << d.dimensions
      << ",\"coverage\":" << d.valid_fraction
      << ",\"statistics_supported\":" << (d.valid_rows >= 2 ? "true" : "false")
      << ",\"std_mean\":" << d.std_mean << ",\"std_min\":" << d.std_min << ",\"std_max\":" << d.std_max
      << ",\"per_dimension_std\":" << tensor_array(d.per_dimension_std)
      << ",\"norm_mean\":" << d.norm_mean << ",\"norm_min\":" << d.norm_min << ",\"norm_max\":" << d.norm_max
      << ",\"covariance_effective_rank\":" << d.covariance_effective_rank << '}';
  return out.str();
}

std::string model_diagnostics(Model &model, const ::embedding::Batch &batch) {
  torch::NoGradGuard no_grad;
  model->eval();
  const auto encoded = model->encode(batch.data, batch.feature_mask);
  const auto global = encoded.pooled_embedding.to(torch::kCPU);
  const auto channels = encoded.pooled_by_channel.to(torch::kCPU);
  const auto sample_mask = encoded.sample_valid_mask.to(torch::kCPU);
  const auto channel_mask = encoded.channel_valid_mask.to(torch::kCPU);
  const auto projected_global = model->project_vicreg(encoded.pooled_embedding).to(torch::kCPU);
  const auto projected_channels = model->project_vicreg(encoded.pooled_by_channel).to(torch::kCPU);
  std::ostringstream out;
  out << "{\"served_global\":" << diagnostics_json(::embedding::representation_diagnostics(global, sample_mask))
      << ",\"served_channels_flattened_rows\":" << diagnostics_json(::embedding::representation_diagnostics(channels, channel_mask))
      << ",\"projector_global\":" << diagnostics_json(::embedding::representation_diagnostics(projected_global, sample_mask))
      << ",\"projector_channels_flattened_rows\":" << diagnostics_json(::embedding::representation_diagnostics(projected_channels, channel_mask))
      << ",\"per_channel\":[";
  for (int64_t c = 0; c < channels.size(1); ++c) {
    if (c) out << ',';
    out << diagnostics_json(::embedding::representation_diagnostics(channels.select(1, c), channel_mask.select(1, c)));
  }
  out << "]}";
  return out.str();
}

::embedding::Batch corrupt(const ::embedding::Batch &clean, const std::string &mode, double severity, uint64_t seed) {
  using torch::indexing::Slice;
  auto mask = torch::ones_like(clean.feature_mask);
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  auto mask_a = mask.accessor<bool, 4>();
  const auto B = clean.data.size(0), C = clean.data.size(1), H = clean.data.size(2), F = clean.data.size(3);
  if (mode == "random") {
    for (int64_t b = 0; b < B; ++b)
      for (int64_t c = 0; c < C; ++c)
        for (int64_t h = 0; h < H; ++h)
          for (int64_t f = 0; f < F; ++f) mask_a[b][c][h][f] = uniform(generator) >= severity;
  } else if (mode == "contiguous") {
    const auto width = std::clamp<int64_t>(static_cast<int64_t>(std::llround(H * severity)), 0, H);
    std::uniform_int_distribution<int64_t> start(0, H - width);
    for (int64_t b = 0; b < B; ++b) {
      const auto first = start(generator);
      mask.index_put_({b, Slice(), Slice(first, first + width), Slice()}, false);
    }
  } else if (mode == "full_channel") {
    const auto count = std::clamp<int64_t>(static_cast<int64_t>(std::ceil(C * severity)), 0, C);
    std::vector<int64_t> order(static_cast<size_t>(C));
    std::iota(order.begin(), order.end(), 0);
    for (int64_t b = 0; b < B; ++b) {
      std::shuffle(order.begin(), order.end(), generator);
      for (int64_t c = 0; c < count; ++c) mask.index_put_({b, order[c]}, false);
    }
  } else require(mode == "clean", "unknown corruption mode");
  return {torch::where(mask, clean.data, torch::zeros_like(clean.data)), mask};
}

std::string observed_rates(const ::embedding::SyntheticEvaluationData &data) {
  std::ostringstream out;
  out << '[';
  for (int64_t label = 0; label < classes; ++label) {
    if (label) out << ',';
    const auto indices = data.labels.eq(label).nonzero().reshape({-1});
    out << data.observed.feature_mask.index_select(0, indices).to(torch::kFloat64).mean().item<double>();
  }
  out << ']';
  return out.str();
}

std::string collision_json(const Config &original, uint64_t seed) {
  auto config = original;
  config.history_length = 8; config.channel_count = 1;
  config.time_scales = {8}; config.scale_strides = {8}; config.dropout = 0;
  torch::manual_seed(seed);
  auto model = Model(config);
  model->eval();
  torch::NoGradGuard no_grad;
  const auto t = torch::arange(8, torch::kFloat32).view({1, 1, 8, 1}) / 7.0;
  const auto x = (t + 0.37 * t.pow(2)).expand({1, 1, 8, config.input_width}).contiguous();
  const auto reversed = x.flip({2});
  const auto a = model->tokenize(x), b = model->tokenize(reversed);
  const auto descriptor_error = std::max(
      (a.time_reconstruction_targets - b.time_reconstruction_targets).abs().max().item<double>(),
      (a.frequency_reconstruction_targets - b.frequency_reconstruction_targets).abs().max().item<double>());
  const auto token_error = (a.tokens - b.tokens).abs().max().item<double>();
  const auto embedding_error = (model->encode(x).pooled_embedding - model->encode(reversed).pooled_embedding).abs().max().item<double>();
  std::ostringstream out;
  out << std::setprecision(12)
      << "{\"kind\":\"constructive_single_window_tokenization_collision\",\"history_length\":8,"
         "\"channel_count\":1,\"time_scales\":[8],\"scale_strides\":[8],"
         "\"description\":\"Reversing one full-history window preserves means, standard deviations and symmetric-Hann Fourier magnitudes. This identifies information lost in this single-window configuration; it is not a downstream score or a claim that every multiscale layout is reversal invariant.\","
      << "\"raw_input_max_difference\":" << (x - reversed).abs().max().item<double>()
      << ",\"descriptor_max_difference\":" << descriptor_error << ",\"token_max_difference\":" << token_error
      << ",\"embedding_max_difference\":" << embedding_error
      << ",\"tolerance\":0.00001,\"collision_within_tolerance\":"
      << (descriptor_error <= 1e-5 && token_error <= 1e-5 ? "true" : "false") << '}';
  return out.str();
}

std::string mask_audits(Model &model, const ::embedding::Batch &batch, uint64_t seed) {
  torch::NoGradGuard no_grad;
  const auto tokens = model->tokenize(batch.data, batch.feature_mask);
  std::ostringstream out;
  int64_t legacy_targets = 0, strict_targets = 0;
  out << '{';
  for (int policy = 0; policy < 2; ++policy) {
    auto config = model->config();
    config.strict_jepa_support = policy != 0;
    JEPAContextTargetMasker masker(config);
    int64_t contexts = 0, targets = 0, overlaps = 0, relaxed = 0, reduced = 0, missing = 0;
    int64_t support = 0, shared = 0;
    for (int64_t trial = 0; trial < 8; ++trial) {
      torch::manual_seed(stream_seed(seed, 80 + trial));
      const auto audit = audit_jepa_masks(tokens, masker.create_masks(tokens));
      contexts += audit.context_tokens; targets += audit.target_tokens;
      overlaps += audit.overlapping_target_tokens;
      support += audit.target_support_points; shared += audit.shared_support_points;
      relaxed += audit.support_relaxed_samples; reduced += audit.target_reduced_samples;
      missing += audit.valid_no_target_samples;
    }
    if (policy) strict_targets = targets;
    else legacy_targets = targets;
    if (policy) out << ',';
    out << (policy ? "\"strict\":" : "\"legacy\":")
        << "{\"trials\":8,\"context_tokens\":" << contexts << ",\"target_tokens\":" << targets
        << ",\"overlapping_target_tokens\":" << overlaps
        << ",\"target_overlap_fraction\":" << static_cast<double>(overlaps) / std::max<int64_t>(1, targets)
        << ",\"target_support_points\":" << support << ",\"shared_support_points\":" << shared
        << ",\"shared_support_ratio\":" << static_cast<double>(shared) / std::max<int64_t>(1, support)
        << ",\"support_relaxed_samples\":" << relaxed << ",\"target_reduced_samples\":" << reduced
        << ",\"valid_no_target_samples\":" << missing << '}';
  }
  out << ",\"strict_to_legacy_target_count_ratio\":"
      << static_cast<double>(strict_targets) / std::max<int64_t>(1, legacy_targets)
      << ",\"geometry\":\"Same-channel raw-history interval support across domains/scales. This measures direct window overlap, not statistical dependence between channels.\"}";
  return out.str();
}

std::string training_json(Model &model, const Settings &settings, const ::embedding::Batch &train,
                          uint64_t seed, const std::string &checkpoint_path) {
  std::vector<torch::Tensor> parameters;
  for (const auto &parameter : model->parameters())
    if (parameter.requires_grad()) parameters.push_back(parameter);
  torch::optim::AdamW optimizer(parameters,
      torch::optim::AdamWOptions(settings.learning_rate).weight_decay(settings.weight_decay));
  std::ostringstream out;
  out << std::setprecision(12) << "{\"updates\":" << settings.steps << ",\"losses\":[";
  int64_t min_unique = settings.batch_size;
  int64_t min_global = settings.batch_size, min_channel = settings.batch_size * settings.model.channel_count;
  model->train();
  for (int64_t step = 0; step < settings.steps; ++step) {
    // Match the public workflow's continuation schedule.
    torch::manual_seed(seed + static_cast<uint64_t>(step));
    const auto indices = torch::randint(train.data.size(0), {settings.batch_size}, torch::kInt64);
    std::set<int64_t> unique;
    const auto index_a = indices.accessor<int64_t, 1>();
    for (int64_t i = 0; i < indices.numel(); ++i) unique.insert(index_a[i]);
    min_unique = std::min(min_unique, static_cast<int64_t>(unique.size()));
    const ::embedding::Batch batch{train.data.index_select(0, indices), train.feature_mask.index_select(0, indices)};
    optimizer.zero_grad();
    const auto result = model->forward(batch.data, batch.feature_mask);
    require(torch::isfinite(result.loss).all().item<bool>(), "non-finite training loss");
    result.loss.backward();
    const auto norm = torch::nn::utils::clip_grad_norm_(parameters,
        settings.gradient_clip_norm > 0 ? settings.gradient_clip_norm : std::numeric_limits<double>::infinity(),
        2.0, true);
    require(std::isfinite(norm), "non-finite gradient norm");
    optimizer.step();
    model->update_target_network();
    min_global = std::min(min_global, result.sample_valid_mask.sum().item<int64_t>());
    min_channel = std::min(min_channel, result.channel_valid_mask.sum().item<int64_t>());
    if (step) out << ',';
    out << "{\"step\":" << step + 1 << ",\"total\":" << result.loss.item<double>()
        << ",\"jepa\":" << result.loss_jepa.item<double>() << ",\"mae\":" << result.loss_mae.item<double>()
        << ",\"mae_time\":" << result.loss_mae_time.item<double>()
        << ",\"mae_frequency\":" << result.loss_mae_frequency.item<double>()
        << ",\"tf_align\":" << result.loss_tf_align.item<double>() << ",\"vicreg\":" << result.loss_vicreg.item<double>()
        << ",\"vicreg_global\":" << result.loss_vicreg_global.item<double>()
        << ",\"vicreg_channel\":" << result.loss_vicreg_channel.item<double>()
        << ",\"gradient_norm_before_clip\":" << norm << '}';
    if ((step + 1) % settings.log_every == 0 || step + 1 == settings.steps)
      std::cout << "evaluation seed=" << seed << " step=" << step + 1 << " loss=" << result.loss.item<double>() << '\n';
    if (settings.checkpoint_every > 0 && (step + 1) % settings.checkpoint_every == 0) {
      auto saved = settings; saved.seed = static_cast<int64_t>(seed);
      save_checkpoint(checkpoint_path, saved, model, optimizer, step + 1);
    }
  }
  auto saved = settings; saved.seed = static_cast<int64_t>(seed);
  save_checkpoint(checkpoint_path, saved, model, optimizer, settings.steps);
  out << "],\"checkpoint\":" << quote(checkpoint_path)
      << ",\"min_unique_minibatch_samples\":" << min_unique
      << ",\"min_base_valid_global_rows\":" << min_global << ",\"min_base_valid_channel_rows\":" << min_channel
      << ",\"global_covariance_rank_upper_bound\":" << std::max<int64_t>(0, std::min(settings.model.projector_dim, min_global - 1))
      << ",\"channel_covariance_rank_upper_bound\":" << std::max<int64_t>(0, std::min(settings.model.projector_dim, min_channel - 1))
      << ",\"vicreg_rows_note\":\"Base-input valid rows are upper bounds for weak-view joint validity. Channel rows share trajectories and are not independent samples. Variance/covariance evidence needs at least two valid rows.\"}";
  return out.str();
}

int64_t integer(const std::string &text) {
  int64_t value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  require(parsed.ec == std::errc() && parsed.ptr == text.data() + text.size(), "expected integer: " + text);
  return value;
}

std::vector<int64_t> seed_list(const std::string &text) {
  require(!text.empty() && text.back() != ',', "expected comma-separated nonnegative seeds");
  std::vector<int64_t> seeds;
  std::set<int64_t> seen;
  std::istringstream in(text);
  std::string item;
  while (std::getline(in, item, ',')) {
    const auto first = item.find_first_not_of(" \t"), last = item.find_last_not_of(" \t");
    require(first != std::string::npos, "empty seed");
    const auto value = integer(item.substr(first, last - first + 1));
    require(value >= 0 && seen.insert(value).second, "seeds must be nonnegative and unique");
    seeds.push_back(value);
  }
  return seeds;
}

void write_report(const std::string &path, const std::string &json) {
  const std::filesystem::path destination(path);
  if (!destination.parent_path().empty()) std::filesystem::create_directories(destination.parent_path());
  const auto temporary = path + ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  {
    std::ofstream out(temporary, std::ios::binary);
    require(out.good(), "could not write report: " + temporary);
    out << json << '\n';
    out.close();
    require(!out.fail(), "report write failed: " + temporary);
  }
  std::filesystem::rename(temporary, destination);
}
} // namespace

::embedding::SyntheticEvaluationData make_evaluation_data(
    const Config &config, int64_t samples, uint64_t signal_seed,
    uint64_t mask_seed, uint64_t label_seed) {
  validate_config(config);
  return ::embedding::make_evaluation_data(input_shape(config), samples,
                                          signal_seed, mask_seed, label_seed);
}

int run_evaluation_cli(int argc, char **argv) {
  try {
    if (argc == 2 || (argc == 3 && std::string(argv[2]) == "--help")) {
      std::cout << "Usage:\n"
          "  embedding evaluate --output report.json [--config settings.conf]\n"
          "      [--seeds 101,202,303] [--train-samples 128] [--test-samples 64]\n"
          "      [--steps 100] [--device cpu|cuda]\n"
          "  embedding evaluate --checkpoint model.pt --output report.json\n"
          "      [--normalization model.pt.normalization.pt] [--seeds 101,202,303]\n"
          "      [--train-samples 128] [--test-samples 64] [--device cpu|cuda]\n"
          "Fresh runs save a checkpoint, its exact normalization archive, and normalized\n"
          "train/test inputs per seed. Checkpoint evaluation requires its saved normalization;\n"
          "the default archive path is CHECKPOINT.normalization.pt. Checkpoint training-data\n"
          "independence is unverified. All scores describe a synthetic diagnostic benchmark.\n";
      return 0;
    }
    const std::set<std::string> allowed{"--config", "--output", "--seeds", "--train-samples", "--test-samples",
        "--steps", "--device", "--checkpoint", "--normalization"};
    std::map<std::string, std::string> args;
    for (int i = 2; i < argc; i += 2) {
      const std::string option(argv[i]);
      require(allowed.count(option), "unknown option: " + option);
      require(i + 1 < argc && args.emplace(option, argv[i + 1]).second, "missing value or duplicate option: " + option);
    }
    const auto get = [&](const std::string &key, const std::string &fallback) {
      const auto it = args.find(key);
      return it == args.end() ? fallback : it->second;
    };
    const auto output = get("--output", "");
    require(!output.empty(), "--output is required");
    const auto supplied_checkpoint = get("--checkpoint", "");
    require(supplied_checkpoint.empty() || (!args.count("--config") && !args.count("--steps")),
            "a supplied checkpoint restores its config; omit --config and --steps");
    const auto supplied_normalization = supplied_checkpoint.empty() ? ""
        : get("--normalization", supplied_checkpoint + ".normalization.pt");
    if (!supplied_checkpoint.empty()) {
      require(std::filesystem::weakly_canonical(output) != std::filesystem::weakly_canonical(supplied_checkpoint),
              "report and checkpoint paths must differ");
      require(std::filesystem::exists(supplied_normalization),
              "checkpoint evaluation requires its exact saved normalization archive; use --normalization or provide CHECKPOINT.normalization.pt");
    } else require(!args.count("--normalization"), "--normalization requires --checkpoint");
    const auto device_text = get("--device", "cpu");
    require(device_text == "cpu" || device_text == "cuda", "--device must be cpu or cuda");
    const torch::Device device(device_text);
    require(!device.is_cuda() || torch::cuda::is_available(), "CUDA is unavailable");
    Settings settings;
    int64_t supplied_completed_steps = 0;
    if (!supplied_checkpoint.empty()) {
      const auto loaded = load_checkpoint(supplied_checkpoint, device);
      settings = loaded.settings;
      supplied_completed_steps = loaded.completed_steps;
    } else settings = args.count("--config") ? read_settings(args.at("--config")) : default_settings();
    settings.model.device = device;
    if (args.count("--steps")) settings.steps = integer(args.at("--steps"));
    else if (!args.count("--config") && supplied_checkpoint.empty()) settings.steps = 100;
    require(settings.steps > 0, "--steps must be positive");
    const auto train_samples = integer(get("--train-samples", "128"));
    const auto test_samples = integer(get("--test-samples", "64"));
    require(train_samples >= 8 && test_samples >= 4, "train/test samples must be at least 8/4");
    const auto seeds = seed_list(get("--seeds", "101,202,303"));
    // Reject every known input/output alias before creating any report sidecar.
    // weakly_canonical resolves existing symlinks as well as relative paths.
    std::set<std::filesystem::path> input_paths, output_paths;
    for (const auto &input : {get("--config", ""), supplied_checkpoint, supplied_normalization})
      if (!input.empty()) input_paths.insert(std::filesystem::weakly_canonical(input));
    std::vector<std::string> destinations{output};
    if (supplied_checkpoint.empty())
      for (const auto seed : seeds) {
        const auto prefix = output + ".seed-" + std::to_string(seed);
        destinations.insert(destinations.end(), {prefix + ".pt", prefix + ".train.pt", prefix + ".test.pt",
                                                  prefix + ".pt.normalization.pt"});
      }
    for (const auto &destination : destinations) {
      const auto canonical = std::filesystem::weakly_canonical(destination);
      require(!input_paths.count(canonical), "an evaluation output would overwrite an input: " + destination);
      require(output_paths.insert(canonical).second, "evaluation output paths alias one another: " + destination);
    }
    torch::set_num_threads(static_cast<int>(settings.threads));
    std::map<std::string, std::vector<double>> accuracies;
    std::ostringstream report;
    report << std::setprecision(12)
           << "{\"format_version\":1,\"protocol\":\"independent_synthetic_trajectory_regime_v1\","
              "\"task\":\"four-class frozen ridge probe, one window per independently generated trajectory\","
              "\"scope\":\"Synthetic diagnostic measurements; no robotics, real-world generalization, future prediction or control claim.\","
              "\"split_method\":\"Separate splitmix-derived signal/mask/label streams for train and test. No trajectory is windowed into multiple splits.\","
              "\"label_names\":[\"slow_periodic\",\"fast_periodic\",\"chirp\",\"amplitude_change\"],"
              "\"mask_label_independence\":\"Missingness random draws depend only on the mask seed and shape, never on regime labels or signal values. Empirical mask-only accuracy and per-label rates remain reported.\","
              "\"probe\":{\"method\":\"centered least-squares one-hot ridge classification\",\"ridge_penalty\":1,"
              "\"feature_scaling\":\"per-coordinate mean/std fitted on valid train rows only\","
              "\"hyperparameter_selection\":\"none; fixed penalty for every surface\","
              "\"confusion_layout\":\"flattened 4x4, true rows and predicted columns\"},"
              "\"covariance_effective_rank_definition\":\"exp(entropy of normalized nonnegative covariance eigenvalues)); zero for zero covariance\","
              "\"invalid_rows\":\"Excluded from scores and embedding statistics; validity coverage reported separately.\","
              "\"environment\":{\"torch\":" << quote(TORCH_VERSION) << ",\"compiler\":" << quote(__VERSION__)
           << ",\"device\":" << quote(device_text)
           << ",\"reproducibility\":\"Same build/device/settings/seeds; cross-device or library-version bitwise equivalence is not promised. Fingerprints are FNV-1a reproducibility checksums, not cryptographic hashes.\"},"
              "\"settings_source\":" << quote(get("--config", supplied_checkpoint.empty() ? "default_settings()" : "checkpoint"))
           << ",\"settings\":" << quote(settings_text(settings))
           << ",\"supplied_checkpoint\":" << quote(supplied_checkpoint)
           << ",\"supplied_normalization\":" << quote(supplied_normalization)
           << ",\"supplied_checkpoint_completed_steps\":" << supplied_completed_steps
           << ",\"checkpoint_evaluation_note\":"
           << quote(supplied_checkpoint.empty()
               ? "Each evaluated trained encoder sees only its generated training split. Saved checkpoints and training archives use the report's input normalization."
               : "No retraining. Every evaluation split uses the exact saved checkpoint normalization. Checkpoint training-data independence and supplied archive/checkpoint association remain unverified.")
           << ",\"runs\":[";
    for (size_t run = 0; run < seeds.size(); ++run) {
      const auto seed = static_cast<uint64_t>(seeds[run]);
      const auto train_signal = stream_seed(seed, 1), train_mask = stream_seed(seed, 2), train_labels = stream_seed(seed, 3);
      const auto test_signal = stream_seed(seed, 4), test_mask = stream_seed(seed, 5), test_labels = stream_seed(seed, 6);
      const auto train = make_evaluation_data(settings.model, train_samples, train_signal, train_mask, train_labels);
      const auto test = make_evaluation_data(settings.model, test_samples, test_signal, test_mask, test_labels);
      const InputScaler scaler = supplied_checkpoint.empty() ? InputScaler(train.observed)
          : InputScaler(supplied_normalization, settings.model.input_width);
      const auto train_scaled = scaler.apply(train.observed), test_scaled = scaler.apply(test.observed);
      torch::manual_seed(seed);
      auto untrained = Model(settings.model);
      torch::manual_seed(seed);
      auto trained = supplied_checkpoint.empty() ? Model(settings.model) : load_checkpoint(supplied_checkpoint, device).model;
      const auto before = fingerprint(trained->parameters());
      const auto checkpoint_path = output + ".seed-" + std::to_string(seed) + ".pt";
      const auto train_archive_path = output + ".seed-" + std::to_string(seed) + ".train.pt";
      const auto test_archive_path = output + ".seed-" + std::to_string(seed) + ".test.pt";
      if (supplied_checkpoint.empty()) {
        scaler.save(checkpoint_path + ".normalization.pt");
        ::embedding::save_batch(train_archive_path, train_scaled);
        ::embedding::save_batch(test_archive_path, test_scaled);
      }
      const auto training = supplied_checkpoint.empty()
          ? training_json(trained, settings, train_scaled, seed, checkpoint_path) : "{\"updates\":0}";
      const auto after = fingerprint(trained->parameters());
      const auto train_features = feature_surfaces(untrained, trained, train_scaled);
      const auto test_features = feature_surfaces(untrained, trained, test_scaled);
      const auto train_valid = sample_valid(train_scaled), test_valid = sample_valid(test_scaled);
      std::map<std::string, Probe> probes;
      for (const auto &[name, features] : train_features)
        probes.emplace(name, Probe(features, train.labels, train_valid));
      if (run) report << ',';
      report << "{\"seed\":" << seed << ",\"train_samples\":" << train_samples << ",\"test_samples\":" << test_samples
             << ",\"streams\":{\"train_signal\":" << quote(std::to_string(train_signal))
             << ",\"train_mask\":" << quote(std::to_string(train_mask)) << ",\"train_labels\":" << quote(std::to_string(train_labels))
             << ",\"test_signal\":" << quote(std::to_string(test_signal)) << ",\"test_mask\":" << quote(std::to_string(test_mask))
             << ",\"test_labels\":" << quote(std::to_string(test_labels)) << '}'
             << ",\"train_fingerprint\":" << quote(fingerprint({train.clean.data, train.observed.feature_mask, train.labels}))
             << ",\"test_fingerprint\":" << quote(fingerprint({test.clean.data, test.observed.feature_mask, test.labels}))
             << ",\"train_labels\":" << tensor_array(train.labels)
             << ",\"test_labels\":" << tensor_array(test.labels)
             << ",\"parameters_before\":" << quote(before) << ",\"parameters_after\":" << quote(after)
             << ",\"normalized_train_archive\":" << quote(supplied_checkpoint.empty() ? train_archive_path : "")
             << ",\"normalized_test_archive\":" << quote(supplied_checkpoint.empty() ? test_archive_path : "")
             << ",\"normalization_archive\":" << quote(supplied_checkpoint.empty() ? checkpoint_path + ".normalization.pt" : supplied_normalization)
             << ",\"normalization_fingerprint\":" << quote(fingerprint({scaler.mean, scaler.scale, scaler.counts}))
             << ",\"input_normalization\":{\"source\":" << quote(supplied_checkpoint.empty() ? "train_observed_only" : "saved_checkpoint_archive")
             << ","
                "\"aggregation\":\"batch, channel and history per input feature\",\"mean\":" << tensor_array(scaler.mean)
             << ",\"scale\":" << tensor_array(scaler.scale) << ",\"observed_counts\":" << tensor_array(scaler.counts) << '}'
             << ",\"train_observed_rate_by_label\":" << observed_rates(train)
             << ",\"test_observed_rate_by_label\":" << observed_rates(test)
             << ",\"training\":" << training << ",\"heldout_scores\":{";
      bool first = true;
      for (const auto &[name, features] : test_features) {
        if (!first) report << ',';
        first = false;
        report << quote(name) << ":{\"dimensions\":" << features.size(1)
               << ",\"test\":" << score_json(probes.at(name), features, test.labels, test_valid)
               << ",\"train\":" << score_json(probes.at(name), train_features.at(name), train.labels, train_valid) << '}';
        const auto rows = test_valid.nonzero().reshape({-1});
        accuracies[name].push_back(probes.at(name).predict(features).index_select(0, rows)
            .eq(test.labels.index_select(0, rows)).to(torch::kFloat64).mean().item<double>());
      }
      report << "},\"heldout_diagnostics\":{\"untrained\":" << model_diagnostics(untrained, test_scaled)
             << ",\"trained\":" << model_diagnostics(trained, test_scaled)
             << "},\"mask_policy_audit_clean_test\":" << mask_audits(trained, scaler.apply(test.clean), seed)
             << ",\"information_retention\":" << collision_json(settings.model, stream_seed(seed, 70))
             << ",\"robustness\":[";
      const std::vector<std::pair<std::string, double>> corruptions{
          {"clean", 0}, {"random", 0.1}, {"random", 0.3}, {"random", 0.6}, {"random", 0.9},
          {"contiguous", 0.25}, {"contiguous", 0.5}, {"contiguous", 0.75},
          {"full_channel", 1.0 / settings.model.channel_count}, {"full_channel", 1.0}};
      for (size_t i = 0; i < corruptions.size(); ++i) {
        const auto &[mode, severity] = corruptions[i];
        const auto corrupted = corrupt(test.clean, mode, severity, stream_seed(seed, 200 + i));
        const auto normalized = scaler.apply(corrupted);
        const auto surfaces = feature_surfaces(untrained, trained, normalized);
        const auto valid = sample_valid(normalized);
        if (i) report << ',';
        report << "{\"mode\":" << quote(mode) << ",\"requested_severity\":" << severity
               << ",\"observed_fraction\":" << normalized.feature_mask.to(torch::kFloat64).mean().item<double>()
               << ",\"scores\":{";
        bool first_surface = true;
        for (const auto &[name, features] : surfaces) {
          if (!first_surface) report << ',';
          first_surface = false;
          report << quote(name) << ':' << score_json(probes.at(name), features, test.labels, valid);
        }
        report << "},\"trained_diagnostics\":" << model_diagnostics(trained, normalized) << '}';
      }
      report << "]}";
      std::cout << "evaluation seed=" << seed << " complete\n";
    }
    report << "],\"heldout_accuracy_summary\":{";
    bool first = true;
    for (const auto &[name, values] : accuracies) {
      const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
      double variance = 0;
      for (const auto value : values) variance += (value - mean) * (value - mean);
      if (!first) report << ',';
      first = false;
      report << quote(name) << ":{\"mean\":" << mean << ",\"std_population\":" << std::sqrt(variance / values.size())
             << ",\"seeds\":" << values.size() << '}';
    }
    report << "}}";
    write_report(output, report.str());
    std::cout << "saved evaluation report to " << output << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
} // namespace embedding::encoders::mtf_jepa_mae_vicreg

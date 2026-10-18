// SPDX-License-Identifier: MIT
// Dimension-aware RPB-v20 training/export only. Scoring remains a separate
// CPU reader of these saved features; labels never enter this optimizer.
#include "configurable_capacity.h"
#include "frozen_role_guard.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_settings.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#ifndef CONFIGURABLE_CAPACITY_SOURCE_ID
#define CONFIGURABLE_CAPACITY_SOURCE_ID "unrecorded"
#endif
#ifndef CONFIGURABLE_CAPACITY_CARD_SHA256
#define CONFIGURABLE_CAPACITY_CARD_SHA256 "unrecorded"
#endif

namespace {
namespace cap = embedding::evaluation::configurable_capacity;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace io = embedding::evaluation::frozen_inputs;
namespace fs = std::filesystem;
using io::require;
using io::quote;
using Clock = std::chrono::steady_clock;
constexpr const char *protocol = "configurable-capacity-v1";
constexpr int64_t batch_size = 8, quality_budget = 512;
constexpr uint64_t quality_masters[] = {941101, 941102, 941103};

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
torch::Tensor cpu(const torch::Tensor &value) {
  return value.detach().to(torch::kCPU).contiguous().clone();
}
bool exact(const torch::Tensor &a, const torch::Tensor &b) {
  const auto x = cpu(a), y = cpu(b);
  return x.scalar_type() == y.scalar_type() && x.sizes() == y.sizes() &&
      (!x.numel() || !std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.numel() * x.element_size()));
}
std::string tensor_digest(const torch::Tensor &value) {
  const auto t = cpu(value);
  return io::sha256(std::string(reinterpret_cast<const char *>(t.const_data_ptr()), t.numel() * t.element_size()));
}
void text(torch::serialize::OutputArchive &a, const std::string &key, const std::string &value) {
  a.write(key, rpb::configurable_checkpoint_detail::text_tensor(value), true);
}
void integer(torch::serialize::OutputArchive &a, const std::string &key, int64_t value) {
  a.write(key, torch::tensor(value, torch::kInt64), true);
}
void save_archive(const fs::path &path, torch::serialize::OutputArchive &a) {
  rpb::configurable_checkpoint_detail::publish_checkpoint(a, path.string());
}
void new_leaf(const fs::path &path) {
  require(path.is_absolute() && path.lexically_normal() == path && !path.filename().empty(), "absolute normalized output leaf");
  require(fs::is_directory(path.parent_path()) && fs::canonical(path.parent_path()) == path.parent_path(), "direct existing output parent");
  for (auto p = path.parent_path(); p != p.root_path(); p = p.parent_path())
    require(!fs::is_symlink(p), "direct output ancestors");
  require(!fs::exists(path) && !fs::is_symlink(path) && fs::create_directory(path), "output leaf must be new");
}
struct RuntimeIsolation {
  int threads = at::get_num_threads();
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &g : generators) states.push_back(g.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try {
      for (size_t i = 0; i < states.size(); ++i) generators[i].set_state(states[i]);
      at::set_num_threads(threads);
    } catch (...) { std::terminate(); }
  }
};
struct MemoryPeak { int64_t allocated = 0, reserved = 0; };
MemoryPeak memory_peak() {
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto aggregate = static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
  return {stats.allocated_bytes[aggregate].peak, stats.reserved_bytes[aggregate].peak};
}

std::string tag(const std::string &profile) {
  require(profile == "A" || profile == "B" || profile == "C", "closed capacity profile");
  return "RPB-v20.alt-0" + std::string(profile == "A" ? "1" : profile == "B" ? "2" : "3");
}
rpb::Config config(const fs::path &path, const std::string &profile) {
  auto c = rpb::read_configurable_spectral_settings(path);
  require(c.channel_count == 3 && c.history_length == 128 && c.input_width == 9 && c.patch_length == 8 &&
      c.encoder_width == 64 && c.export_width == 32 && c.num_layers == 3 && c.num_heads == 4 &&
      c.feedforward_width == 256 && c.decoder_hidden_width == 128 && c.mask_ratio == .25 &&
      c.huber_delta == 1. && c.scale_floor == 1e-6 && c.layer_norm_epsilon == 1e-5 &&
      c.sampling_interval == 1. && c.device == torch::Device(torch::kCUDA, 0) &&
      rpb::resolved_channel_ids(c) == std::vector<int64_t>({0, 1, 2}), "frozen base model settings");
  (void)tag(profile);
  if (profile == "B") c.encoder_width = 128;
  if (profile == "C") c.export_width = 64;
  rpb::validate_configurable_spectral_config(c);
  return c;
}
std::string config_json(const rpb::Config &c) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"channels\":" << c.channel_count << ",\"history\":" << c.history_length
      << ",\"features\":" << c.input_width << ",\"patch_length\":" << c.patch_length
      << ",\"encoder_width\":" << c.encoder_width << ",\"embedding_size\":" << c.export_width
      << ",\"learned_coordinates\":" << rpb::configurable_spectral_shape_width(c)
      << ",\"fixed_coordinates\":" << rpb::configurable_spectral_width(c)
      << ",\"layers\":" << c.num_layers << ",\"heads\":" << c.num_heads
      << ",\"feedforward_width\":" << c.feedforward_width << ",\"decoder_hidden_width\":" << c.decoder_hidden_width
      << ",\"channel_mixer_layers\":1,\"channel_mixer_placement\":1,\"global_bottleneck_mode\":2,\"global_pool_input_source\":0,\"temporal_difference_input\":0"
      << ",\"mask_ratio\":" << c.mask_ratio << ",\"context_deletion_ratio\":0.15,\"huber_delta\":" << c.huber_delta
      << ",\"dropout\":0,\"layer_norm_epsilon\":" << c.layer_norm_epsilon << ",\"scale_floor\":" << c.scale_floor
      << ",\"sampling_interval\":" << c.sampling_interval << ",\"channel_ids\":[0,1,2],\"device\":\"cuda:0\",\"dtype\":\"float32\""
      << ",\"batch_size\":8,\"optimizer\":\"AdamW\",\"learning_rate\":0.001,\"weight_decay\":0.0001,\"gradient_clip_norm\":1}";
  return out.str();
}
rpb::Input input(const cap::Split &split, const rpb::Config &c) {
  return {split.observed, split.mask, torch::tensor(rpb::resolved_channel_ids(c), torch::kInt64),
      torch::full({split.observed.size(0)}, double(c.history_length - 1), torch::kFloat64), c.sampling_interval};
}
rpb::Input select(const rpb::Input &value, int64_t begin, int64_t count) {
  return rpb::training_detail::selected(value, torch::arange(begin, begin + count, torch::kInt64));
}
struct Cohort { cap::Split training, validation, deleted; };
void check_cohort(const Cohort &data, uint64_t master, int64_t train_rows, int64_t validation_rows) {
  const auto shape = [&](const cap::Split &s, int64_t rows) {
    cap::validate(s);
    require(s.observed.sizes() == torch::IntArrayRef({rows, 3, 128, 9}), "declared dataset geometry and rows");
    const auto ids = s.source_ids.accessor<int64_t, 1>();
    const auto labels = s.scoring_labels.accessor<int64_t, 1>();
    std::set<int64_t> sources;
    for (int64_t row = 0; row < rows; row += 2) {
      require(ids[row] >= 0 && uint64_t(ids[row]) >> 31 == master && ids[row] == ids[row + 1] &&
          sources.insert(ids[row]).second && labels[row] != labels[row + 1] &&
          torch::equal(s.mask[row], s.mask[row + 1]), "paired source identities, opposite labels and shared masks");
    }
    return sources;
  };
  const auto training = shape(data.training, train_rows), validation = shape(data.validation, validation_rows);
  shape(data.deleted, validation_rows);
  for (const auto id : validation) require(!training.count(id), "source-disjoint TRAIN/VALIDATION");
  require(torch::equal(data.validation.source_ids, data.deleted.source_ids) &&
      torch::equal(data.validation.scoring_labels, data.deleted.scoring_labels) &&
      !data.deleted.mask.logical_and(~data.validation.mask).any().item<bool>() &&
      torch::equal(data.deleted.observed.masked_select(data.deleted.mask), data.validation.observed.masked_select(data.deleted.mask)),
      "deleted validation preserves identities/labels and only original observations");
}

struct Binding {
  std::string profile, model_tag, source_fingerprint, card_sha, input_manifest_sha, config_sha;
  uint64_t master = 0;
  std::string json() const {
    return "\"protocol\":" + quote(protocol) + ",\"dataset_id\":\"TEMPO-5\",\"profile\":" + quote(profile) +
        ",\"model_tag\":" + quote(model_tag) + ",\"master\":" + std::to_string(master) +
        ",\"source_fingerprint\":" + quote(source_fingerprint) + ",\"card_sha\":" + quote(card_sha) +
        ",\"input_manifest_sha\":" + quote(input_manifest_sha) + ",\"config_sha\":" + quote(config_sha);
  }
  void write(torch::serialize::OutputArchive &a, int64_t point, const rpb::Config &c, const std::string &scaler_id) const {
    text(a, "protocol", protocol); text(a, "dataset_id", "TEMPO-5"); text(a, "profile", profile); text(a, "model_tag", model_tag);
    text(a, "source_fingerprint", source_fingerprint); text(a, "card_sha", card_sha); text(a, "input_manifest_sha", input_manifest_sha);
    text(a, "config_sha", config_sha); text(a, "scaler_identity", scaler_id); text(a, "resolved_settings_json", config_json(c));
    integer(a, "master", int64_t(master)); integer(a, "point", point); integer(a, "native_width", c.export_width);
  }
};
struct Trace {
  int64_t attempted = 0, completed = 0, target_cells = 0, requested = 0, actual = 0, restored = 0;
  uint64_t mask_seed = 0, torch_seed = 0;
  double loss = 0, gradient = 0;
  std::vector<int64_t> rows;
  std::string hidden_sha, visible_sha;
};
struct Trainer {
  rpb::Config c;
  Binding binding;
  rpb::Input raw;
  rpb::FrozenScaler scaler;
  rpb::ConfigurableSpectralModel model{nullptr};
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::vector<torch::Tensor> initial;
  std::vector<Trace> trace;
  int64_t parameter_count = 0, trainable_count = 0, frozen_count = 0;
  double training_seconds = 0;
  MemoryPeak training_memory;

  Trainer(const rpb::Config &settings, const cap::Split &training, Binding identity)
      : c(settings), binding(std::move(identity)), raw(input(training, c)) {
    scaler = rpb::fit_configurable_spectral_scaler(raw, c); // TRAIN only; labels never passed.
    torch::manual_seed(rpb::training_detail::mixed(binding.master ^ 0x7270622d696e6974ULL));
    model = rpb::ConfigurableSpectralModel(c);
    validate_trainability();
    for (const auto &p : model->parameters()) initial.push_back(cpu(p));
    optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
  }
  void validate_trainability() {
    parameter_count = trainable_count = frozen_count = 0;
    bool prior_found = false;
    for (const auto &p : model->named_parameters()) {
      require(p.value().is_cuda() && p.value().scalar_type() == torch::kFloat32 &&
          torch::isfinite(p.value()).all().item<bool>(), "all parameters finite CUDA float32");
      const bool prior = p.key() == "grouped_odd_relation_projection.weight";
      require(p.value().requires_grad() != prior, "only literal spectral prior frozen");
      if (prior) {
        prior_found = true;
        frozen_count += p.value().numel();
        require(!p.value().grad().defined(), "frozen prior has no gradient");
      } else trainable_count += p.value().numel();
      parameter_count += p.value().numel();
    }
    require(prior_found && frozen_count == 3 * 4 * 9 * 9 * 4 && trainable_count > 0, "expected all-feature fixed prior count");
    model->validate_fixed_prior();
  }
  bool changed() const {
    const auto parameters = model->parameters();
    for (size_t i = 0; i < parameters.size(); ++i) if (!exact(parameters[i], initial[i])) return true;
    return false;
  }
  void verify_frozen() {
    validate_trainability();
    const auto p = model->parameters();
    for (size_t i = 0; i < p.size(); ++i) if (!p[i].requires_grad()) require(exact(p[i], initial[i]), "frozen prior unchanged from initialization");
  }
  void train_to(int64_t budget) {
    require(budget >= int64_t(trace.size()) && budget <= quality_budget, "monotonic closed update budget");
    model->train();
    torch::cuda::synchronize(0);
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    const auto start = Clock::now();
    while (int64_t(trace.size()) < budget) {
      const auto attempt = int64_t(trace.size());
      const auto rows = rpb::training_detail::sampled_indices(raw.data.size(0), batch_size, int64_t(binding.master), attempt);
      const auto normalized = rpb::transform_configurable_spectral_input(scaler, rpb::training_detail::selected(raw, rows), c);
      require(normalized.data.is_cuda() && normalized.observed.is_cuda(), "CUDA training data and observed masks");
      Trace record; record.attempted = attempt + 1; record.completed = attempt + 1;
      record.rows.assign(rows.const_data_ptr<int64_t>(), rows.const_data_ptr<int64_t>() + rows.numel());
      record.mask_seed = rpb::training_detail::counter_seed(int64_t(binding.master), attempt, 0x6d61736bULL);
      record.torch_seed = rpb::training_detail::counter_seed(int64_t(binding.master), attempt, 0x746f726368ULL);
      const auto mask = rpb::configurable_spectral_make_training_mask(normalized.observed, c, record.mask_seed);
      require(mask.eligible_channels.any().item<bool>(), "no skipped original-mask attempts; abort if unsupported");
      // H128/P8 is divisible. The support-only shared deletion planner has no
      // D32 restriction; we intentionally do not use its legacy forward helper.
      const auto context = rpb::context_deletion::make_plan(mask, normalized.channel_ids, c,
          int64_t(binding.master), attempt, rpb::ContextDeletionRecipe::coordinate15_v1);
      record.requested = context.requested_count; record.actual = context.actual_count; record.restored = context.restored_count;
      record.hidden_sha = tensor_digest(mask.hidden); record.visible_sha = tensor_digest(context.visible);
      torch::manual_seed(record.torch_seed);
      const rpb::Input visible{torch::where(context.visible, normalized.data, torch::zeros_like(normalized.data)),
          context.visible, normalized.channel_ids, normalized.endpoints, normalized.sampling_interval};
      optimizer->zero_grad();
      const auto encoded = model->encode(visible);
      const auto native = rpb::compact_reconstruction_export(encoded, c);
      require(native.is_cuda() && native.sizes() == torch::IntArrayRef({batch_size, c.export_width}), "sole CUDA native configured bottleneck");
      const auto prediction = model->decode(native, normalized.channel_ids);
      const auto loss = rpb::hierarchical_huber(prediction, normalized.data.detach(), mask.target, mask.eligible_channels, c.huber_delta);
      require(prediction.is_cuda() && loss.loss.is_cuda() && loss.eligible_example_count > 0 &&
          torch::isfinite(loss.loss).item<bool>(), "finite eligible CUDA waveform Huber loss");
      loss.loss.backward();
      for (const auto &p : model->parameters())
        require(!p.grad().defined() || (p.grad().is_cuda() && torch::isfinite(p.grad()).all().item<bool>()), "finite CUDA gradients");
      record.gradient = torch::nn::utils::clip_grad_norm_(model->parameters(), 1., 2., true);
      require(std::isfinite(record.gradient) && record.gradient > 0, "finite nonzero CUDA gradient norm");
      optimizer->step();
      record.target_cells = loss.target_cell_count; record.loss = loss.loss.item<double>();
      trace.push_back(std::move(record));
      if (trace.size() % 128 == 0) std::cout << binding.model_tag << " master=" << binding.master << " CUDA_updates=" << trace.size() << '/' << budget << '\n' << std::flush;
    }
    torch::cuda::synchronize(0);
    training_seconds += seconds(start); training_memory = memory_peak(); verify_frozen();
  }
  std::string progress() const {
    std::ostringstream out; out << std::setprecision(17) << '{' << binding.json() << ",\"resolved_settings\":" << config_json(c)
        << ",\"attempted\":" << trace.size() << ",\"completed\":" << trace.size() << ",\"skipped_attempts\":0,\"sampled_rows\":" << batch_size * trace.size()
        << ",\"parameter_count\":" << parameter_count << ",\"cuda_parameter_count\":" << parameter_count
        << ",\"trainable_parameter_count\":" << trainable_count << ",\"frozen_parameter_count\":" << frozen_count
        << ",\"frozen_parameter_name\":\"grouped_odd_relation_projection.weight\",\"training_device\":\"cuda:0\",\"CPU_encoder_calls\":0"
        << ",\"training_seconds\":" << training_seconds << ",\"training_peak_allocated_bytes\":" << training_memory.allocated
        << ",\"training_peak_reserved_bytes\":" << training_memory.reserved << ",\"weights_changed\":" << (changed() ? "true" : "false")
        << ",\"scaler_identity\":" << quote(scaler.identity()) << ",\"initialization_seed\":" << rpb::training_detail::mixed(binding.master ^ 0x7270622d696e6974ULL)
        << ",\"context_recipe\":\"rpb-training-context-deletion-015-v1\",\"losses\":[";
    for (size_t i = 0; i < trace.size(); ++i) {
      const auto &t = trace[i]; if (i) out << ',';
      out << "{\"attempted\":" << t.attempted << ",\"completed\":" << t.completed << ",\"target_cells\":" << t.target_cells
          << ",\"total_loss\":" << t.loss << ",\"waveform_loss\":" << t.loss << ",\"dynamics_loss\":0,\"gradient_norm\":" << t.gradient
          << ",\"context_requested\":" << t.requested << ",\"context_actual\":" << t.actual << ",\"context_restored\":" << t.restored
          << ",\"mask_seed\":" << t.mask_seed << ",\"torch_seed\":" << t.torch_seed
          << ",\"hidden_sha256\":" << quote(t.hidden_sha) << ",\"visible_sha256\":" << quote(t.visible_sha) << ",\"sampled_indices\":[";
      for (size_t j = 0; j < t.rows.size(); ++j) { if (j) out << ','; out << t.rows[j]; }
      out << "]}";
    }
    return out.str() + "]}";
  }
  void save(const fs::path &path) {
    verify_frozen();
    rpb::save_configurable_spectral_checkpoint(path.string(), model, scaler, int64_t(trace.size()));
  }
};

struct Features { torch::Tensor values, valid; };
Features extract(const cap::Split &split, rpb::LoadedConfigurableSpectralEncoder &loaded) {
  torch::NoGradGuard no_grad; loaded.model->eval();
  const auto raw = input(split, loaded.config);
  std::vector<torch::Tensor> values, validity;
  for (int64_t begin = 0; begin < raw.data.size(0); begin += batch_size) {
    const auto count = std::min(batch_size, raw.data.size(0) - begin);
    const auto normalized = rpb::transform_configurable_spectral_input(loaded.scaler, select(raw, begin, count), loaded.config);
    require(normalized.data.is_cuda(), "CUDA extraction input");
    const auto e = loaded.model->encode(normalized);
    const auto z = rpb::compact_reconstruction_export(e, loaded.config);
    require(z.is_cuda() && z.sizes() == torch::IntArrayRef({count, loaded.config.export_width}), "native CUDA export geometry");
    values.push_back(z.detach().to(torch::kCPU, torch::kFloat64).contiguous());
    validity.push_back(cpu(e.sample_valid_mask));
  }
  return {torch::cat(values), torch::cat(validity)};
}
struct FeatureViews { Features training, validation, deleted; };
FeatureViews feature_views(const Cohort &data, rpb::LoadedConfigurableSpectralEncoder &loaded) {
  return {extract(data.training, loaded), extract(data.validation, loaded), extract(data.deleted, loaded)};
}
void write_features(const fs::path &path, const FeatureViews &features, const Binding &binding,
    int64_t point, const rpb::Config &c, const std::string &scaler_id) {
  torch::serialize::OutputArchive a; binding.write(a, point, c, scaler_id);
  text(a, "artifact_kind", "configurable_capacity_native_feature_views_v1");
  for (const auto &entry : std::vector<std::pair<std::string, Features>>{{"training", features.training}, {"validation", features.validation}, {"deleted", features.deleted}}) {
    a.write(entry.first + "_values", entry.second.values, true);
    a.write(entry.first + "_valid", entry.second.valid, true);
  }
  text(a, "provenance", std::string(protocol) + "/" + binding.model_tag + "/master-" + std::to_string(binding.master) +
      "/point-" + std::to_string(point) + "; exact native CUDA export; no post-encoder PCA");
  save_archive(path, a);
}
struct NativeMovement { int64_t changed_cells = 0; double max_absolute_change = 0; };
NativeMovement unchanged_fixed_features(const FeatureViews &initial, const FeatureViews &trained, const rpb::Config &c) {
  const auto start = rpb::configurable_spectral_shape_width(c), width = rpb::configurable_spectral_width(c);
  for (const auto &pair : std::vector<std::pair<Features, Features>>{{initial.training, trained.training}, {initial.validation, trained.validation}, {initial.deleted, trained.deleted}})
    require(exact(pair.first.valid, pair.second.valid) && exact(pair.first.values.narrow(1, start, width), pair.second.values.narrow(1, start, width)),
        "exact unchanged native fixed spectral coordinates at 0/512 for all legal views");
  const auto delta = trained.training.values.narrow(1, 0, start) - initial.training.values.narrow(1, 0, start);
  NativeMovement movement{delta.ne(0).sum().item<int64_t>(), delta.abs().max().item<double>()};
  require(movement.changed_cells > 0 && std::isfinite(movement.max_absolute_change) && movement.max_absolute_change > 0,
      "learned TRAIN native coordinates must move after CUDA training");
  return movement;
}

// Compact original query bank: every time/feature coordinate appears in one
// leave-one-patch-out context. trial_index_for_time explicitly identifies that
// context, avoiding K copies of zeros and never implying one shared embedding.
std::string write_queries(const fs::path &path, const cap::Split &split,
    rpb::LoadedConfigurableSpectralEncoder &loaded, const Binding &binding) {
  torch::NoGradGuard no_grad; loaded.model->eval();
  const auto &c = loaded.config;
  const auto B = split.observed.size(0), K = c.history_length / c.patch_length;
  auto prediction = torch::zeros_like(split.observed), target = torch::zeros_like(split.observed);
  auto query = torch::zeros_like(split.mask);
  auto requested = torch::zeros_like(split.mask);
  auto trial_eligible = torch::zeros({K, B, c.channel_count}, torch::kBool);
  const auto raw = input(split, c);
  int64_t calls = 0;
  for (int64_t begin = 0; begin < B; begin += batch_size) {
    const auto count = std::min(batch_size, B - begin);
    const auto normalized = rpb::transform_configurable_spectral_input(loaded.scaler, select(raw, begin, count), c);
    for (int64_t trial = 0; trial < K; ++trial) {
      auto hidden = torch::zeros_like(normalized.observed);
      hidden.narrow(2, trial * c.patch_length, c.patch_length).fill_(true);
      const auto requested_query = normalized.observed.logical_and(hidden);
      const auto visible = normalized.observed.logical_and(~hidden);
      const auto eligible = visible.reshape({count, c.channel_count, K, -1}).any(-1).sum(-1).ge(2)
          .logical_and(requested_query.flatten(2).any(2));
      hidden = hidden.logical_and(eligible.unsqueeze(-1).unsqueeze(-1));
      const auto mask = rpb::configurable_spectral_mask_from_hidden(normalized.observed, hidden, c);
      require(torch::equal(mask.eligible_channels, eligible), "fixed original-query eligibility");
      const rpb::Input context{torch::where(mask.visible, normalized.data, torch::zeros_like(normalized.data)),
          mask.visible, normalized.channel_ids, normalized.endpoints, normalized.sampling_interval};
      const auto e = loaded.model->encode(context);
      const auto z = rpb::compact_reconstruction_export(e, c);
      const auto p = loaded.model->decode(z, normalized.channel_ids); ++calls;
      require(p.is_cuda() && torch::isfinite(p).all().item<bool>(), "finite CUDA original query reconstruction");
      const auto target_mask = mask.target.logical_and(eligible.unsqueeze(-1).unsqueeze(-1));
      const auto start = trial * c.patch_length;
      prediction.narrow(0, begin, count).narrow(2, start, c.patch_length).copy_(cpu(torch::where(target_mask, p, torch::zeros_like(p))).narrow(2, start, c.patch_length));
      target.narrow(0, begin, count).narrow(2, start, c.patch_length).copy_(cpu(torch::where(target_mask, normalized.data, torch::zeros_like(normalized.data))).to(torch::kFloat64).narrow(2, start, c.patch_length));
      query.narrow(0, begin, count).narrow(2, start, c.patch_length).copy_(cpu(target_mask).narrow(2, start, c.patch_length));
      requested.narrow(0, begin, count).narrow(2, start, c.patch_length).copy_(cpu(requested_query).narrow(2, start, c.patch_length));
      trial_eligible[trial].narrow(0, begin, count).copy_(cpu(eligible));
    }
  }
  torch::cuda::synchronize(0);
  const auto error = prediction - target, absolute = error.abs();
  const auto huber = torch::where(absolute.le(c.huber_delta), .5 * error.clamp(-c.huber_delta, c.huber_delta).square(),
      c.huber_delta * (absolute - .5 * c.huber_delta));
  const std::vector<int64_t> axes{2, 3};
  const auto counts = query.sum(axes), channel_valid = counts.gt(0);
  const auto channel_mae = absolute.sum(axes) / counts.clamp_min(1).to(torch::kFloat64);
  const auto channel_huber = huber.sum(axes) / counts.clamp_min(1).to(torch::kFloat64);
  const auto example_valid = channel_valid.any(1);
  const auto channel_count = channel_valid.sum(1).clamp_min(1).to(torch::kFloat64);
  const auto example_mae = channel_mae.sum(1) / channel_count, example_huber = channel_huber.sum(1) / channel_count;
  const auto valid_examples = example_valid.sum().item<int64_t>(), valid_cells = counts.sum().item<int64_t>();
  require(valid_examples > 0 && torch::isfinite(example_mae).all().item<bool>(), "finite supported original-query summaries");
  torch::serialize::OutputArchive a; binding.write(a, loaded.training_updates, c, loaded.scaler.identity());
  text(a, "artifact_kind", "configurable_capacity_compact_original_query_bank_v1");
  text(a, "query_semantics", "K leave-one-complete-patch-out contexts; each coordinate once; context=time/P; original observations only; no extra context deletion");
  a.write("standardized_prediction", prediction, true); a.write("standardized_target", target, true);
  a.write("target_mask", query, true); a.write("requested_observed_target_mask", requested, true);
  a.write("original_observed_mask", split.mask, true); a.write("trial_channel_eligible", trial_eligible, true);
  a.write("trial_index_for_time", torch::arange(c.history_length, torch::kInt64).floor_divide(c.patch_length), true);
  a.write("channel_target_counts", counts, true); a.write("channel_valid", channel_valid, true);
  a.write("channel_standardized_mae", channel_mae, true); a.write("channel_standardized_huber", channel_huber, true);
  a.write("example_valid", example_valid, true); a.write("example_standardized_mae", example_mae, true);
  a.write("example_standardized_huber", example_huber, true); a.write("source_ids", split.source_ids, true);
  text(a, "source_ids_json", cap::source_ids_json(split)); integer(a, "patch_length", c.patch_length); integer(a, "query_cuda_forward_calls", calls);
  save_archive(path, a);
  std::ostringstream out; out << std::setprecision(17) << "{\"status\":\"measured\",\"units\":\"frozen TRAIN-scaler standardized\",\"total_examples\":" << B
      << ",\"valid_examples\":" << valid_examples << ",\"coverage\":" << double(valid_examples) / B
      << ",\"valid_target_cells\":" << valid_cells << ",\"requested_observed_target_cells\":" << requested.sum().item<int64_t>()
      << ",\"standardized_mae\":" << example_mae.masked_select(example_valid).mean().item<double>()
      << ",\"standardized_huber\":" << example_huber.masked_select(example_valid).mean().item<double>()
      << ",\"query_contexts_per_example\":" << K << ",\"CUDA_forward_calls\":" << calls << ",\"CPU_encoder_calls\":0,\"artifact\":" << quote(path.filename().string()) << '}';
  return out.str();
}

void save_training_state(const fs::path &path, Trainer &trainer) {
  torch::serialize::OutputArchive a, optimizer;
  trainer.binding.write(a, int64_t(trainer.trace.size()), trainer.c, trainer.scaler.identity());
  text(a, "artifact_kind", "configurable_capacity_optimizer_witness_v1");
  text(a, "resume_contract", "state witness only; no resume CLI");
  trainer.optimizer->save(optimizer); a.write("optimizer", optimizer);
  a.write("rng_cpu", at::globalContext().defaultGenerator(at::Device(at::kCPU)).get_state().clone(), true);
  a.write("rng_cuda_0", at::globalContext().defaultGenerator(at::Device(at::kCUDA, 0)).get_state().clone(), true);
  save_archive(path, a);
}

void engineering(const fs::path &output, const fs::path &settings, bool &owned) {
  require(torch::cuda::is_available(), "CUDA admission mandatory"); new_leaf(output); owned = true;
  const auto development = cap::make_development(8, 4, 941000);
  Cohort fixture{development.training, development.validation,
      cap::delete_observations(development.validation, .30, rpb::training_detail::mixed(941000 ^ cap::kDeletionStream))};
  check_cohort(fixture, 941000, 16, 8);
  for (const std::string profile : {"A", "B", "C"}) {
    const auto dir = output / profile; require(fs::create_directory(dir), "new admission profile");
    const auto c = config(settings, profile);
    Binding binding{profile, tag(profile), CONFIGURABLE_CAPACITY_SOURCE_ID, CONFIGURABLE_CAPACITY_CARD_SHA256,
        "engineering-artificial", io::sha256(io::bytes(settings)), 941000};
    Trainer trainer(c, fixture.training, binding); trainer.train_to(2); require(trainer.changed(), "actual CUDA updates change trainable weights");
    trainer.save(dir / "checkpoint.pt");
    bool refused = false;
    try { trainer.save(dir / "checkpoint.pt"); } catch (const std::exception &) { refused = true; }
    require(refused, "typed checkpoint refuses replacing admission state");
    const auto normalized = rpb::transform_configurable_spectral_input(trainer.scaler, input(fixture.validation, c), c);
    torch::NoGradGuard no_grad; trainer.model->eval();
    const auto before = rpb::compact_reconstruction_export(trainer.model->encode(normalized), c);
    auto loaded = rpb::load_configurable_spectral_checkpoint((dir / "checkpoint.pt").string(), c.device, &c);
    require(loaded.training_updates == 2 && loaded.scaler.identity() == trainer.scaler.identity(), "typed immutable checkpoint metadata");
    const auto after = extract(fixture.validation, loaded);
    require(exact(before.to(torch::kFloat64), after.values), "bit-exact CUDA checkpoint native reload");
    const auto features = feature_views(fixture, loaded); write_features(dir / "features.pt", features, binding, 2, c, trainer.scaler.identity());
    const auto query = write_queries(dir / "reconstruction-validation.pt", fixture.validation, loaded, binding);
    io::write_new(dir / "progress.json", trainer.progress() + '\n');
    io::write_new(dir / "query.json", query + '\n');
    std::cout << binding.model_tag << " admission CUDA2 updates; parameters=" << trainer.parameter_count
        << " trainable=" << trainer.trainable_count << " frozen=" << trainer.frozen_count << " native=" << c.export_width << '\n' << std::flush;
  }
  io::write_new(output / "complete.json", "{\"status\":\"passed\",\"protocol\":" + quote(protocol) +
      ",\"master\":941000,\"profiles\":[\"A\",\"B\",\"C\"],\"updates_each\":2,\"CPU_encoder_calls\":0}\n");
}
} // namespace

int main(int argc, char **argv) {
  fs::path output; bool owned = false;
  try {
    if (argc == 2 && std::string(argv[1]) == "--source-id") { std::cout << CONFIGURABLE_CAPACITY_SOURCE_ID << '\n'; return 0; }
    if (argc == 2 && std::string(argv[1]) == "--card-sha256") { std::cout << CONFIGURABLE_CAPACITY_CARD_SHA256 << '\n'; return 0; }
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2)
      require(i + 1 < argc && std::string(argv[i]).rfind("--", 0) == 0 && args.emplace(argv[i], argv[i + 1]).second, "unique key/value CLI");
    RuntimeIsolation isolation; torch::set_num_threads(1);
    if (args.count("--engineering")) {
      require(args.size() == 2 && args.count("--config"), "engineering output and model config only");
      output = args.at("--engineering"); engineering(output, args.at("--config"), owned); return 0;
    }
    const std::set<std::string> required{"--data-root", "--inputs-sha256", "--output", "--profile", "--master", "--config", "--card", "--card-sha256", "--source-id"};
    require(args.size() == required.size(), "closed quality CLI");
    for (const auto &key : required) require(args.count(key), "required CLI: " + key);
    require(io::is_sha(CONFIGURABLE_CAPACITY_SOURCE_ID) && io::is_sha(CONFIGURABLE_CAPACITY_CARD_SHA256) &&
        args.at("--source-id") == CONFIGURABLE_CAPACITY_SOURCE_ID && args.at("--card-sha256") == CONFIGURABLE_CAPACITY_CARD_SHA256 &&
        io::sha256(io::bytes(args.at("--card"))) == args.at("--card-sha256"), "compiled source and exact prospective card binding");
    const auto master_text = args.at("--master");
    require(master_text == "941101" || master_text == "941102" || master_text == "941103", "one of three prospective quality masters");
    const auto master = std::stoull(master_text); const auto profile = args.at("--profile");
    const auto c = config(args.at("--config"), profile);
    const fs::path parent = args.at("--data-root");
    require(parent.is_absolute() && fs::canonical(parent) == parent && !fs::is_symlink(parent), "direct data root");
    std::map<std::string, fs::path> roles;
    for (const auto m : quality_masters) for (const std::string role : {"training", "validation", "validation-deleted"}) {
      const auto name = "master-" + std::to_string(m) + "/" + role + ".pt"; roles.emplace(name, parent / name);
    }
    io::Guard inputs(args.at("--inputs-sha256"), roles);
    std::map<uint64_t, Cohort> all;
    for (const auto m : quality_masters) {
      const auto prefix = "master-" + std::to_string(m) + "/";
      Cohort cohort{cap::load_split_archive(inputs.bind(prefix + "training.pt")), cap::load_split_archive(inputs.bind(prefix + "validation.pt")),
          cap::load_split_archive(inputs.bind(prefix + "validation-deleted.pt"))};
      check_cohort(cohort, m, 512, 256); all.emplace(m, std::move(cohort));
    }
    require(torch::cuda::is_available(), "quality CUDA required after complete data admission");
    output = args.at("--output"); new_leaf(output); owned = true;
    Binding binding{profile, tag(profile), CONFIGURABLE_CAPACITY_SOURCE_ID, args.at("--card-sha256"),
        io::sha256(io::bytes(args.at("--inputs-sha256"))), io::sha256(io::bytes(args.at("--config"))), master};
    const auto &data = all.at(master);
    auto start = Clock::now(); Trainer trainer(c, data.training, binding); double binding_seconds = seconds(start), feature_seconds = 0, query_seconds = 0;
    FeatureViews initial_features;
    NativeMovement learned_movement;
    std::string training_query, validation_query, initial_training_query, initial_validation_query;
    io::write_new(output / "input-manifest.json", inputs.json() + '\n');
    for (const int64_t point : {int64_t{0}, quality_budget}) {
      if (point) { trainer.train_to(point); require(trainer.changed() && trainer.trace.size() == quality_budget, "complete changed CUDA quality trajectory"); }
      const auto directory = output / ("point-" + std::to_string(point)); require(fs::create_directory(directory), "new retained point");
      start = Clock::now(); trainer.save(directory / "checkpoint.pt"); save_training_state(directory / "optimizer-witness.pt", trainer);
      auto loaded = rpb::load_configurable_spectral_checkpoint((directory / "checkpoint.pt").string(), c.device, &c);
      require(loaded.training_updates == point && loaded.scaler.identity() == trainer.scaler.identity(), "served checkpoint metadata/scaler"); binding_seconds += seconds(start);
      start = Clock::now(); const auto features = feature_views(data, loaded);
      write_features(directory / "features.pt", features, binding, point, c, loaded.scaler.identity()); torch::cuda::synchronize(0); feature_seconds += seconds(start);
      if (!point) initial_features = features;
      else learned_movement = unchanged_fixed_features(initial_features, features, c);
      start = Clock::now();
      training_query = write_queries(directory / "reconstruction-training.pt", data.training, loaded, binding);
      validation_query = write_queries(directory / "reconstruction-validation.pt", data.validation, loaded, binding);
      query_seconds += seconds(start);
      if (!point) { initial_training_query = training_query; initial_validation_query = validation_query; }
      io::write_new(directory / "query-report.json", "{" + binding.json() + ",\"point\":" + std::to_string(point) +
          ",\"training_reconstruction\":" + training_query + ",\"validation_reconstruction\":" + validation_query + "}\n");
      io::write_new(directory / "progress.json", trainer.progress() + '\n');
    }
    inputs.verify();
    require(io::sha256(io::bytes(args.at("--inputs-sha256"))) == binding.input_manifest_sha &&
        io::sha256(io::bytes(args.at("--config"))) == binding.config_sha && io::sha256(io::bytes(args.at("--card"))) == binding.card_sha,
        "metadata inputs unchanged");
    io::write_new(output / "progress.json", trainer.progress() + '\n');
    std::ostringstream report; report << std::setprecision(17) << '{' << binding.json() << ",\"progress\":" << trainer.progress()
        << ",\"training_reconstruction\":" << training_query << ",\"validation_reconstruction\":" << validation_query
        << ",\"initial_training_reconstruction\":" << initial_training_query << ",\"initial_validation_reconstruction\":" << initial_validation_query
        << ",\"learned_training_coordinate_movement\":{\"changed_cells\":" << learned_movement.changed_cells
        << ",\"max_absolute_change\":" << learned_movement.max_absolute_change << '}'
        << ",\"fixed_spectral_coordinates_exact_at_0_512\":true,\"costs\":{\"binding_checkpoint_snapshot_seconds\":" << binding_seconds
        << ",\"CUDA_feature_transfer_IO_seconds\":" << feature_seconds << ",\"CUDA_query_transfer_IO_seconds\":" << query_seconds
        << ",\"CPU_head_seconds\":0},\"counts\":{\"encoder_trajectories\":1,\"encoder_updates\":512,\"retained_points\":2,\"native_exports\":6,\"query_writer_calls\":4,\"query_cuda_forward_calls\":3072,\"generator_calls\":0,\"CPU_encoder_calls\":0,\"head_fits\":0},\"testing_accessed\":false,\"stress_accessed\":false,\"promotion\":false}";
    io::write_new(output / "report.json", report.str() + '\n');
    io::write_new(output / "complete.json", "{\"status\":\"complete\"," + binding.json() +
        ",\"updates\":512,\"report_sha256\":" + quote(io::sha256(report.str() + '\n')) + "}\n");
    std::cout << binding.model_tag << " master=" << master << " CUDA_updates=512 complete; native=" << c.export_width << " training_seconds=" << trainer.training_seconds << '\n';
    return 0;
  } catch (const std::exception &error) {
    if (owned && fs::is_directory(output) && !fs::exists(output / "failure.json"))
      try { io::write_new(output / "failure.json", "{\"status\":\"failed\",\"error\":" + quote(error.what()) + "}\n"); } catch (...) {}
    std::cerr << error.what() << '\n'; return 1;
  }
}

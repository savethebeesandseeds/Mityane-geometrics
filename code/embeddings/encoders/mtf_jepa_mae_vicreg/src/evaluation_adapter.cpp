// SPDX-License-Identifier: MIT
#include "embedding/encoders/mtf_jepa_mae_vicreg/evaluation_adapter.h"
#include "embedding/encoders/mtf_jepa_mae_vicreg/workflow.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
#ifndef EVALUATION_GIT_HEAD
#define EVALUATION_GIT_HEAD "unrecorded"
#endif
#ifndef EVALUATION_GIT_DIRTY
#define EVALUATION_GIT_DIRTY "unrecorded"
#endif

namespace embedding::encoders::mtf_jepa_mae_vicreg {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error("[baseline evaluation adapter] " + message);
}

std::string quote(const std::string &text) {
  std::ostringstream out; out << '"';
  for (const unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << '?';
    else out << c;
  }
  out << '"'; return out.str();
}

void write_text(const fs::path &path, const std::string &text) {
  require(!fs::exists(path), "artifact already exists: " + path.string());
  std::ofstream out(path); require(bool(out), "cannot create artifact: " + path.string());
  out << text; out.close(); require(bool(out), "cannot save artifact: " + path.string());
}

std::string model_checksum(Model &model) {
  std::vector<torch::Tensor> parameters;
  for (const auto &parameter : model->parameters())
    parameters.push_back(parameter.detach().to(torch::kCPU).flatten());
  auto values = torch::cat(parameters).contiguous();
  uint64_t hash = 14695981039346656037ULL;
  const auto *bytes = static_cast<const unsigned char *>(values.const_data_ptr());
  for (int64_t i = 0; i < values.numel() * values.element_size(); ++i) {
    hash ^= bytes[i]; hash *= 1099511628211ULL;
  }
  std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

embedding::Batch identity_input(const embedding::Batch &raw, const Config &config) {
  require(raw.data.defined() && raw.data.dim() == 4 && raw.data.size(0) > 0 &&
              raw.data.size(1) == config.channel_count && raw.data.size(2) == config.history_length &&
              raw.data.size(3) == config.input_width && raw.data.is_floating_point() &&
              raw.data.device().is_cpu(), "observations must be CPU BCHF with the declared dimensions");
  require(raw.feature_mask.defined() && raw.feature_mask.scalar_type() == torch::kBool &&
              raw.feature_mask.sizes() == raw.data.sizes() && raw.feature_mask.device().is_cpu(),
          "observed support must be CPU bool BCHF");
  // This is the explicitly declared baseline identity recipe, not the RPB scaler.
  const auto data = raw.data.to(torch::kFloat32);
  require(torch::isfinite(data.masked_select(raw.feature_mask)).all().item<bool>(),
          "observed input is nonfinite or overflows float32 identity preparation");
  return {torch::where(raw.feature_mask, data, torch::zeros_like(data)), raw.feature_mask.clone()};
}

std::string zero_missing_audit(Model &model, const Config &config) {
  torch::NoGradGuard guard; model->eval();
  auto zeros = torch::zeros({1, config.channel_count, config.history_length, config.input_width}, torch::kFloat32);
  const auto observed = model->encode(zeros, torch::ones_like(zeros, torch::kBool));
  const auto absent = model->encode(zeros, torch::zeros_like(zeros, torch::kBool));
  require(observed.sample_valid_mask.all().item<bool>() && observed.channel_valid_mask.all().item<bool>(),
          "observed zero was treated as missing");
  require(!absent.sample_valid_mask.any().item<bool>() && !absent.channel_valid_mask.any().item<bool>() &&
              torch::isfinite(absent.pooled_embedding).all().item<bool>() && absent.pooled_embedding.eq(0).all().item<bool>() &&
              torch::isfinite(absent.pooled_by_channel).all().item<bool>() && absent.pooled_by_channel.eq(0).all().item<bool>(),
          "absent input manufactures valid support or a nonzero export");
  return "observed_zero_valid=true;missing_invalid=true;missing_exports_finite_zero=true";
}

ev::FeatureMap extract(Model &model, const embedding::Batch &raw,
                       const Config &config, const std::string &prefix,
                       bool include_descriptors) {
  torch::NoGradGuard guard; model->eval();
  const auto batch = identity_input(raw, config);
  const auto encoded = model->encode(batch.data, batch.feature_mask);
  const std::string provenance = prefix +
      "; online encoder state; eval mode; identity raw float64-to-float32 preparation; "
      "fixed storage/channel order; channel-indexed vectors are contextual through global-mean mixing";
  ev::FeatureMap out{
      {prefix + "_global", {encoded.pooled_embedding.to(torch::kCPU), encoded.sample_valid_mask.to(torch::kCPU), provenance}},
      {prefix + "_channel_concatenation", {encoded.pooled_by_channel.flatten(1).to(torch::kCPU),
          encoded.channel_valid_mask.all(1).to(torch::kCPU), provenance + "; one joint concatenation, not per-channel probe accuracy"}}};
  if (include_descriptors) {
    const auto tokens = model->tokenize(batch.data, batch.feature_mask);
    const auto time = torch::where(tokens.time_reconstruction_mask, tokens.time_reconstruction_targets,
                                   torch::zeros_like(tokens.time_reconstruction_targets));
    const auto frequency = torch::where(tokens.frequency_reconstruction_mask, tokens.frequency_reconstruction_targets,
                                        torch::zeros_like(tokens.frequency_reconstruction_targets));
    out.emplace("descriptors", ev::FeatureSurface{
        torch::cat({time.flatten(1), frequency.flatten(1),
                    tokens.time_reconstruction_mask.flatten(1).to(time.scalar_type()),
                    tokens.frequency_reconstruction_mask.flatten(1).to(time.scalar_type()),
                    tokens.token_mask.flatten(1).to(time.scalar_type())}, 1).to(torch::kCPU),
        encoded.channel_valid_mask.all(1).to(torch::kCPU),
        "complete configured baseline time/frequency reconstruction descriptors plus descriptor/token support; "
        "fixed chronological/channel/domain/start/width slots; identity preparation; masks counted in feature dimensions"});
  }
  return out;
}

struct State {
  Settings settings;
  Model random{nullptr}, trained{nullptr};
  std::shared_ptr<torch::optim::AdamW> optimizer;
  embedding::Batch training;
  ev::ProviderFitInput fit;
  int64_t updates{0};
  uint64_t initialization_seed{0};
  std::string initial_checksum, trained_checksum, zero_missing;
};

void write_field(torch::serialize::OutputArchive &archive, const char *key, const std::string &text) {
  archive.write(key, embedding::archive::text_tensor(text), true);
}

void save_random(const fs::path &directory, const State &state) {
  torch::serialize::OutputArchive archive, weights;
  write_field(archive, "encoder_id", "mtf_jepa_mae_vicreg");
  write_field(archive, "artifact_kind", "baseline_frozen_evaluation_model_v1");
  archive.write("evaluation_format_version", torch::tensor(int64_t{1}), true);
  write_field(archive, "settings", settings_text(state.settings));
  write_field(archive, "protocol_id", state.fit.protocol_id);
  write_field(archive, "preprocessing", "identity observed float64-to-float32; no fitted input scaler");
  write_field(archive, "served_state", "online encoder; contextual channel concatenation; eval mode");
  write_field(archive, "initialization_seed", std::to_string(state.initialization_seed));
  write_field(archive, "weights_fnv1a64", state.initial_checksum);
  write_field(archive, "source_fingerprint", EVALUATION_SOURCE_ID);
  write_field(archive, "git_head", EVALUATION_GIT_HEAD);
  write_field(archive, "git_dirty", EVALUATION_GIT_DIRTY);
  write_field(archive, "feature_units", state.fit.feature_units);
  archive.write("channel_order", torch::tensor(state.fit.channel_ids, torch::kInt64), true);
  archive.write("sampling_interval", torch::tensor(state.fit.sampling_interval, torch::kFloat64), true);
  archive.write("endpoint", torch::tensor(state.fit.endpoint, torch::kFloat64), true);
  state.random->save(weights); archive.write("model", weights);
  embedding::archive::save_archive((directory / "untrained-baseline-model.pt").string(), archive);
}

std::string provenance_json(const State &state) {
  std::ostringstream out; out << std::setprecision(17)
      << "{\"encoder\":\"MTF-JEPA-MAE-VICReg\",\"protocol_id\":" << quote(state.fit.protocol_id)
      << ",\"preprocessing\":\"identity raw finite float64-to-float32; no fitted input scaling\","
      << "\"served_state\":\"online encoder; eval mode\",\"channel_semantics\":\"contextual global-mean mixing; fixed-order joint concatenation\","
      << "\"training_partition_only\":true,\"pretraining_updates\":" << state.updates
      << ",\"training_examples\":" << state.training.data.size(0)
      << ",\"training_example_exposures\":" << quote(std::to_string(state.training.data.size(0)) + " * " + std::to_string(state.updates))
      << ",\"sampling\":\"full training partition each update; no labels, clean targets or heldout observations\","
      << "\"optimizer\":\"AdamW\",\"learning_rate\":" << state.settings.learning_rate
      << ",\"weight_decay\":" << state.settings.weight_decay
      << ",\"gradient_clip_norm\":" << state.settings.gradient_clip_norm
      << ",\"teacher_update\":\"update_target_network after each optimizer update under saved configuration\","
      << "\"initialization_seed\":" << quote(std::to_string(state.initialization_seed))
      << ",\"mask_seed_policy\":\"stream_seed(run_seed,600000+update_index)\","
      << "\"run_seed\":" << quote(std::to_string(state.fit.seed))
      << ",\"initial_weights_fnv1a64\":" << quote(state.initial_checksum)
      << ",\"checksum_policy\":\"reproducibility fingerprints, not cryptographic proofs\","
      << "\"source_fingerprint\":" << quote(EVALUATION_SOURCE_ID)
      << ",\"git_head\":" << quote(EVALUATION_GIT_HEAD) << ",\"git_dirty\":" << quote(EVALUATION_GIT_DIRTY)
      << ",\"feature_units\":" << quote(state.fit.feature_units)
      << ",\"sampling_interval\":" << state.fit.sampling_interval << ",\"endpoint\":" << state.fit.endpoint;
  if (state.trained) out << ",\"frozen_weights_fnv1a64\":" << quote(state.trained_checksum)
      << ",\"configuration\":\"fresh-baseline-settings.conf\",\"training_archive\":\"fresh-baseline-training.pt\","
      << "\"checkpoint\":\"fresh-baseline.pt\",\"continuation_protocol\":\"frozen evaluation control; legacy CLI continuation uses a different sampling/RNG protocol\"";
  out << ",\"historical_checkpoint_association\":\"unavailable; this is a new identity-preprocessed control, not historical weights or scores\"}";
  return out.str();
}

} // namespace

ev::FeatureProviderFactory make_evaluation_provider(const EvaluationOptions &options) {
  require(options.pretraining_updates >= 0, "pretraining_updates must be nonnegative");
  return [options](const ev::ProviderFitInput &fit) {
    auto state = std::make_shared<State>();
    state->settings = options.config_path.empty() ? default_settings() : read_settings(options.config_path);
    auto &settings = state->settings;
    if (options.config_path.empty()) {
      settings.model.channel_count = fit.shape.channel_count;
      settings.model.history_length = fit.shape.history_length;
      settings.model.input_width = fit.shape.input_width;
    }
    require(settings.model.channel_count == fit.shape.channel_count &&
                settings.model.history_length == fit.shape.history_length && settings.model.input_width == fit.shape.input_width,
            "configured C/H/F differs from the evaluation card");
    require(static_cast<int64_t>(fit.channel_ids.size()) == fit.shape.channel_count &&
                std::set<int64_t>(fit.channel_ids.begin(), fit.channel_ids.end()).size() == fit.channel_ids.size(),
            "card channel order must contain unique IDs matching C");
    require(std::isfinite(fit.sampling_interval) && fit.sampling_interval > 0 && std::isfinite(fit.endpoint),
            "invalid uniform observation metadata");
    settings.model.device = torch::kCPU; settings.model.dtype = torch::kFloat32; settings.model.dropout = 0;
    settings.threads = torch::get_num_threads();
    validate_config(settings.model);
    state->fit = fit; state->updates = options.pretraining_updates;
    state->training = identity_input(fit.training_observations, settings.model);
    state->initialization_seed = ev::stream_seed(fit.seed, 500000);
    torch::manual_seed(state->initialization_seed); state->random = Model(settings.model);
    state->zero_missing = zero_missing_audit(state->random, settings.model);
    state->initial_checksum = model_checksum(state->random);
    if (state->updates > 0) {
      require(fit.seed <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
              "legacy checkpoint settings require a run seed representable as int64");
      torch::manual_seed(state->initialization_seed); state->trained = Model(settings.model);
      state->trained->train();
      state->optimizer = std::make_shared<torch::optim::AdamW>(state->trained->parameters(),
          torch::optim::AdamWOptions(settings.learning_rate).weight_decay(settings.weight_decay));
      for (int64_t step = 0; step < state->updates; ++step) {
        torch::manual_seed(ev::stream_seed(fit.seed, 600000ULL + static_cast<uint64_t>(step)));
        state->optimizer->zero_grad();
        const auto result = state->trained->forward(state->training.data, state->training.feature_mask);
        require(torch::isfinite(result.loss).item<bool>(), "fresh baseline training loss is nonfinite");
        result.loss.backward();
        if (settings.gradient_clip_norm > 0)
          torch::nn::utils::clip_grad_norm_(state->trained->parameters(), settings.gradient_clip_norm);
        for (const auto &parameter : state->trained->parameters())
          require(!parameter.grad().defined() || torch::isfinite(parameter.grad()).all().item<bool>(),
                  "fresh baseline training gradient is nonfinite");
        state->optimizer->step(); state->trained->update_target_network();
      }
      state->trained->eval(); state->trained_checksum = model_checksum(state->trained);
    }
    ev::FeatureProvider provider;
    provider.provenance = "MTF-JEPA-MAE-VICReg evaluation adapter; fixed channel order; online frozen inference; "
        "identity float64-to-float32 preparation; new control only, no historical checkpoint association";
    const auto add = [&](const std::string &prefix) {
      provider.surfaces.emplace(prefix + "_global", ev::SurfaceDescription{
          ev::SurfaceKind::global, "at least one baseline token with observed support", {}});
      provider.surfaces.emplace(prefix + "_channel_concatenation", ev::SurfaceDescription{
          ev::SurfaceKind::channel_concatenation, "all declared constituent channels observed-valid; contextual vectors", fit.channel_ids});
    };
    add("untrained_baseline");
    provider.surfaces.emplace("descriptors", ev::SurfaceDescription{
        ev::SurfaceKind::control, "all declared channels observed-valid; includes descriptor/token masks", {}});
    if (state->trained) add("fresh_frozen_baseline");
    provider.audit_fields = {
        {"zero_missing_control_audit", state->zero_missing},
        {"untrained_baseline_weights_fnv1a64", state->initial_checksum},
        {"initialization_seed", std::to_string(state->initialization_seed)},
        {"training_mask_seed_policy", "stream_seed(run_seed,600000+update_index)"},
        {"fresh_baseline_steps", std::to_string(state->updates)},
        {"training_rows_per_update", std::to_string(state->training.data.size(0))},
        {"served_state", "online encoder; channel vectors contextual; teacher state not served"},
        {"historical_frozen_baseline", "unavailable: original checkpoint/preprocessing association not proven"},
        {"source_fingerprint", EVALUATION_SOURCE_ID}};
    if (state->trained) provider.audit_fields.emplace("fresh_frozen_baseline_weights_fnv1a64", state->trained_checksum);
    provider.extract = [state](const embedding::Batch &observations) {
      auto out = extract(state->random, observations, state->settings.model, "untrained_baseline", true);
      if (state->trained) {
        auto trained = extract(state->trained, observations, state->settings.model, "fresh_frozen_baseline", false);
        out.insert(trained.begin(), trained.end());
      }
      return out;
    };
    provider.save_assets = [state](const std::string &directory) {
      const fs::path root(directory);
      write_text(root / "baseline-settings.conf", settings_text(state->settings));
      save_random(root, *state);
      if (state->trained) {
        auto saved = state->settings; saved.steps = state->updates;
        saved.seed = static_cast<int64_t>(state->fit.seed); saved.batch_size = state->training.data.size(0);
        save_checkpoint((root / "fresh-baseline.pt").string(), saved, state->trained, *state->optimizer, state->updates);
        embedding::save_batch((root / "fresh-baseline-training.pt").string(), state->training);
        write_text(root / "fresh-baseline-settings.conf", settings_text(saved));
      }
      write_text(root / "baseline-adapter-provenance.json", provenance_json(*state));
    };
    return provider;
  };
}

} // namespace embedding::encoders::mtf_jepa_mae_vicreg

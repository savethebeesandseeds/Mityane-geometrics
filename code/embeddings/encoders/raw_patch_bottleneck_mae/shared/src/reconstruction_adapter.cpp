// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/reconstruction_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
#ifndef EVALUATION_GIT_HEAD
#define EVALUATION_GIT_HEAD "unrecorded"
#endif
#ifndef EVALUATION_GIT_DIRTY
#define EVALUATION_GIT_DIRTY "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {

namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using training_detail::counter_seed;
using training_detail::mixed;
using training_detail::sampled_indices;
using training_detail::selected;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("[rpb reconstruction adapter] " + message);
}

bool same_real(double left, double right) {
  return std::isfinite(left) && std::isfinite(right) &&
      std::abs(left - right) <= 1e-12 * std::max({1.0, std::abs(left), std::abs(right)});
}

void write_text(torch::serialize::OutputArchive &archive, const char *key,
                const std::string &text) {
  archive.write(key, embedding::archive::text_tensor(text), true);
}

std::string source_manifest(const std::vector<std::string> &ids) {
  std::string out;
  for (const auto &id : ids) out += std::to_string(id.size()) + ":" + id;
  return out;
}

std::string manifest_identity(const std::string &text) {
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ULL; }
  std::ostringstream out;
  out << "rpb-reconstruction-fit-sources-fnv1a-v1-" << std::hex
      << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

std::string number(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

Input raw_input(const embedding::Batch &batch, const Config &config,
                const std::vector<int64_t> &ids, double endpoint) {
  require(batch.data.defined() && batch.data.dim() == 4 && batch.data.size(0) > 0 &&
      batch.data.device().is_cpu() &&
      batch.data.scalar_type() == torch::kFloat64,
      "observations must preserve CPU float64 source precision");
  Input input{batch.data, batch.feature_mask, torch::tensor(ids, torch::kInt64),
      torch::full({batch.data.size(0)}, endpoint, torch::kFloat64),
      config.sampling_interval};
  validate_input(input, config);
  require(input.observed.device().is_cpu(), "observation support must be on CPU");
  return input;
}

void validate_support(const torch::Tensor &support, const Config &config) {
  require(support.defined() && support.device().is_cpu() &&
      support.scalar_type() == torch::kBool && support.dim() == 4 &&
      support.size(0) > 0 && support.size(1) == config.channel_count &&
      support.size(2) == config.history_length && support.size(3) == config.input_width,
      "metadata prediction requires CPU bool BCHF visible support");
}

Input metadata_input(const torch::Tensor &support, const Config &config,
                     const std::vector<int64_t> &ids, double endpoint) {
  validate_support(support, config);
  // These are normalized zeros, not raw zeros transformed using signal values.
  Input input{torch::zeros(support.sizes(),
      torch::TensorOptions().dtype(torch::kFloat32).device(config.device)),
      support.to(config.device), torch::tensor(ids, torch::kInt64).to(config.device),
      torch::full({support.size(0)}, endpoint, torch::kFloat64).to(config.device),
      config.sampling_interval};
  validate_input(input, config, true);
  return input;
}

torch::Tensor raw_prediction(const torch::Tensor &prediction,
                            const FrozenScaler &scaler, const Config &config,
                            const std::vector<int64_t> &ids) {
  require(prediction.defined() && prediction.dim() == 4 && prediction.size(0) > 0 &&
      prediction.size(1) == config.channel_count && prediction.size(2) == config.history_length &&
      prediction.size(3) == config.input_width && prediction.scalar_type() == torch::kFloat32 &&
      torch::isfinite(prediction).all().item<bool>(), "decoder produced malformed/nonfinite predictions");
  scaler.validate(config);
  const auto indices = channel_indices(torch::tensor(ids, torch::kInt64),
      config, prediction.size(0), torch::kCPU).reshape({-1});
  const auto dimensions = std::vector<int64_t>{prediction.size(0), config.channel_count, 1, config.input_width};
  const auto mean = scaler.mean.index_select(0, indices).reshape(dimensions);
  const auto scale = scaler.scale.index_select(0, indices).reshape(dimensions);
  // Inverse scaling is float64. Neural predictions retain their float32 precision.
  auto raw = (prediction.detach().to(torch::kCPU, torch::kFloat64) * scale + mean).contiguous();
  require(raw.device().is_cpu() && raw.scalar_type() == torch::kFloat64 &&
      torch::isfinite(raw).all().item<bool>(), "inverse scaling produced nonfinite raw predictions");
  return raw;
}

void freeze_model(Model &model) {
  model->eval();
  for (auto &parameter : model->parameters()) parameter.set_requires_grad(false);
}

std::vector<torch::Tensor> metadata_parameters(Model &model) {
  model->eval(); // The original encoder remains fixed and dropout is disabled.
  std::vector<torch::Tensor> parameters;
  for (const auto &parameter : model->named_parameters()) {
    const bool decoder = parameter.key().rfind("decoder_", 0) == 0;
    parameter.value().set_requires_grad(decoder);
    if (decoder) parameters.push_back(parameter.value());
  }
  require(!parameters.empty(), "metadata control has no decoder parameters");
  return parameters;
}

void verify_same_initialization(const Model &reference, const Model &other) {
  const auto reference_parameters = reference->named_parameters();
  const auto other_parameters = other->named_parameters();
  require(reference_parameters.size() == other_parameters.size(),
      "matched initialization parameter counts differ");
  for (const auto &parameter : reference_parameters)
    require(other_parameters.contains(parameter.key()) &&
        torch::equal(parameter.value(), other_parameters[parameter.key()]),
        "matched initialization differs at " + parameter.key());
}

struct TrainingProgress {
  int64_t attempted{0}, completed{0};
  double first_loss{0.0}, last_loss{0.0};
};

struct ReconstructionState {
  Settings settings;
  Dataset training;
  FrozenScaler scaler;
  Model trained{nullptr}, metadata{nullptr}, untrained{nullptr};
  std::shared_ptr<torch::optim::AdamW> optimizer, metadata_optimizer;
  std::vector<torch::Tensor> metadata_parameters;
  TrainingProgress main_progress, metadata_progress;
  std::vector<int64_t> ids;
  double endpoint{0.0};
  uint64_t initialization_seed{0};
  std::string recipe, protocol_id, source_manifest;
};

TrainingProgress train_model(Model &model, torch::optim::AdamW &optimizer,
    const std::vector<torch::Tensor> &parameters, const ReconstructionState &state,
    int64_t updates, bool metadata_only) {
  if (metadata_only) model->eval(); else model->train();
  TrainingProgress progress;
  const auto &settings = state.settings;
  while (progress.completed < updates && progress.attempted < settings.attempt_limit) {
    const auto attempt = progress.attempted;
    const auto indices = sampled_indices(state.training.input.data.size(0),
        settings.batch_size, settings.seed, attempt);
    auto batch = state.scaler.transform(selected(state.training.input, indices), settings.model);
    const auto masks = make_training_mask(batch.observed, settings.model,
        counter_seed(settings.seed, attempt, 0x6d61736bULL));
    torch::manual_seed(counter_seed(settings.seed, attempt, 0x746f726368ULL));
    ++progress.attempted;
    if (!masks.eligible_channels.any().item<bool>()) continue;
    optimizer.zero_grad();
    torch::Tensor loss;
    if (metadata_only) {
      // The frozen backbone sees only normalized zeros and the actual visible mask.
      // Real targets reach the independently trained decoder loss, never its encoder.
      auto zero_visible = metadata_input(masks.visible.to(torch::kCPU),
          settings.model, state.ids, state.endpoint);
      torch::Tensor latent;
      { torch::NoGradGuard no_grad;
        const auto output = model->encode(zero_visible);
        latent = compact_reconstruction_export(output, settings.model).detach(); }
      const auto prediction = model->decode(latent, batch.channel_ids);
      auto result = hierarchical_huber(prediction, batch.data.detach(), masks.target,
          masks.eligible_channels, settings.model.huber_delta);
      require(result.eligible_example_count > 0, "metadata loss has no eligible examples");
      loss = result.loss;
    } else {
      auto result = model->forward(batch, masks.hidden);
      require(result.eligible_example_count > 0, "main loss has no eligible examples");
      loss = result.loss;
    }
    require(torch::isfinite(loss).all().item<bool>(), "training loss is nonfinite");
    loss.backward();
    torch::nn::utils::clip_grad_norm_(parameters, settings.gradient_clip_norm > 0 ?
        settings.gradient_clip_norm : std::numeric_limits<double>::infinity(), 2.0, true);
    optimizer.step();
    if (progress.completed == 0) progress.first_loss = loss.item<double>();
    progress.last_loss = loss.item<double>();
    ++progress.completed;
  }
  require(progress.completed == updates, std::string(metadata_only ? "metadata" : "main") +
      " training exhausted its attempt budget");
  model->eval();
  return progress;
}

void write_provenance(torch::serialize::OutputArchive &archive,
                      const ReconstructionState &state, const char *kind,
                      bool untrained = false) {
  write_text(archive, "encoder_id", kEncoderId);
  archive.write("format_version", torch::tensor(int64_t{1}), true);
  write_text(archive, "artifact_kind", kind);
  write_text(archive, "output_semantics", output_semantics(state.settings.model));
  write_text(archive, "settings", settings_text(state.settings));
  write_text(archive, "recipe", state.recipe);
  write_text(archive, "protocol_id", state.protocol_id);
  write_text(archive, "schema_id", state.training.schema_id);
  write_text(archive, "permitted_training_observation_dataset_id", state.training.dataset_id);
  write_text(archive, "pretraining_dataset_id", untrained ? "none" : state.training.dataset_id);
  write_text(archive, "scaler_fit_dataset_id", state.training.dataset_id);
  write_text(archive, "preprocessing_id", state.scaler.identity());
  write_text(archive, "fit_source_manifest", state.source_manifest);
  write_text(archive, "fit_source_manifest_id", manifest_identity(state.source_manifest));
  write_text(archive, "source_fingerprint_algorithm", "sha256-source-manifest-v1");
  write_text(archive, "checkpoint_writer_core_source_fingerprint", workflow_source_fingerprint());
  write_text(archive, "adapter_training_source_fingerprint", EVALUATION_SOURCE_ID);
  write_text(archive, "source_fingerprint_scope",
      "ordinary checkpoint source field identifies its core writer; adapter training producer is recorded separately here");
  write_text(archive, "git_head", EVALUATION_GIT_HEAD);
  write_text(archive, "git_dirty", EVALUATION_GIT_DIRTY);
  write_text(archive, "initialization_seed", std::to_string(state.initialization_seed));
  write_text(archive, "rng_policy", "splitmix64-counter-rows-masks-torch-attempt-v1");
}

} // namespace

ReconstructionRegistration make_reconstruction_provider(
    const ReconstructionOptions &options, const ev::ReconstructionCard &card) {
  require(options.pretraining_updates > 0 && options.metadata_updates > 0 &&
      options.batch_size > 0, "training/control updates and batch size must be positive");
  require(card.shape.channel_count > 0 && card.shape.history_length > 0 &&
      card.shape.input_width > 0 && card.shape.dtype == torch::kFloat64 &&
      card.shape.device.is_cpu(), "card must declare positive CPU float64 C/H/F dimensions");
  require(card.patch_length > 0 && card.shape.history_length % card.patch_length == 0 &&
      card.shape.history_length / card.patch_length >= 3, "card requires at least three complete patches");
  require(static_cast<int64_t>(card.channel_ids.size()) == card.shape.channel_count &&
      std::set<int64_t>(card.channel_ids.begin(), card.channel_ids.end()).size() == card.channel_ids.size(),
      "card semantic IDs must be unique and match C");
  require(!card.feature_units.empty() && std::isfinite(card.sampling_interval) &&
      card.sampling_interval > 0 && card.threads > 0, "card requires units, positive interval and threads");
  auto settings = options.config_path.empty() ? default_settings() : read_settings(options.config_path);
  require(settings.model.global_bottleneck_mode == 0,
      "this per-channel reconstruction card requires global_bottleneck_mode=0; use the global-bottleneck experiment's exact global reconstruction callback");
  if (options.config_path.empty()) {
    settings.model.channel_count = card.shape.channel_count;
    settings.model.history_length = card.shape.history_length;
    settings.model.input_width = card.shape.input_width;
    settings.model.patch_length = card.patch_length;
    settings.model.channel_ids = card.channel_ids;
    settings.model.sampling_interval = card.sampling_interval;
  }
  require(settings.model.channel_count == card.shape.channel_count &&
      settings.model.history_length == card.shape.history_length &&
      settings.model.input_width == card.shape.input_width && settings.model.patch_length == card.patch_length,
      "explicit configuration geometry differs from the reconstruction card");
  const auto configured_ids = resolved_channel_ids(settings.model);
  require(std::set<int64_t>(configured_ids.begin(), configured_ids.end()) ==
      std::set<int64_t>(card.channel_ids.begin(), card.channel_ids.end()) &&
      same_real(settings.model.sampling_interval, card.sampling_interval),
      "explicit configuration IDs/interval differ from the reconstruction card");
  settings.model.device = torch::kCPU;
  settings.batch_size = options.batch_size;
  settings.steps = options.pretraining_updates;
  settings.threads = card.threads;
  settings.checkpoint_every = 0;
  validate_settings(settings);
  const auto endpoint = (card.shape.history_length - 1) * card.sampling_interval;
  require(std::isfinite(endpoint), "card endpoint overflows");
  // Also validate units before generation/fitting, using only a schema-shaped sentinel.
  const auto sentinel = metadata_input(torch::ones(
      {1, card.shape.channel_count, card.shape.history_length, card.shape.input_width},
      torch::kBool), settings.model, card.channel_ids, endpoint);
  (void)describe_dataset(sentinel, settings.model, card.feature_units);
  std::ostringstream recipe;
  recipe << "rpb-reconstruction-recipe-v1\nfresh_training_only=true\n"
      << "config_origin=" << (options.config_path.empty() ? "built_in_defaults" : options.config_path) << '\n'
      << "main_completed_updates=" << options.pretraining_updates << '\n'
      << "metadata_completed_updates=" << options.metadata_updates << '\n'
      << "budgets_matched=" << (options.pretraining_updates == options.metadata_updates ? "true" : "false") << '\n'
      << "training_seed=fit_seed_and_0x7fffffffffffffff\n"
      << "initialization_seed=splitmix64(fit_seed_xor_0x7270622d696e6974)\n"
      << "sampler_mask_torch_streams=splitmix64-counter-rows-masks-torch-attempt-v1\n"
      << "metadata_control=same_initialized_frozen_encoder;normalized_zero_values;"
         "actual_visible_support_and_fixed_IDs_time;eval_mode;decoder_parameters_only\n"
      << "main_checkpoint_source_scope=core_writer_only;see_training_provenance_for_adapter_producer\n"
      << "checkpoint_writer_core_source_fingerprint=" << workflow_source_fingerprint() << '\n'
      << "adapter_training_source_fingerprint=" << EVALUATION_SOURCE_ID << '\n'
      << "[resolved_settings;seed_is_replaced_by_declared_fit_seed]\n" << settings_text(settings);
  const auto frozen_recipe = recipe.str();
  ev::ReconstructionProviderFactory factory =
      [settings, options, card, endpoint, frozen_recipe](const ev::ProviderFitInput &fit) {
    require(fit.shape.channel_count == card.shape.channel_count &&
        fit.shape.history_length == card.shape.history_length && fit.shape.input_width == card.shape.input_width &&
        fit.shape.dtype == card.shape.dtype && fit.shape.device.is_cpu() &&
        fit.channel_ids == card.channel_ids && fit.feature_units == card.feature_units &&
        same_real(fit.sampling_interval, card.sampling_interval) && same_real(fit.endpoint, endpoint),
        "fit metadata differs from the frozen reconstruction card");
    require(!fit.protocol_id.empty() &&
        std::find(card.seeds.begin(), card.seeds.end(), fit.seed) != card.seeds.end(),
        "fit must identify a declared protocol and seed");
    auto state = std::make_shared<ReconstructionState>();
    state->settings = settings;
    state->settings.seed = static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL);
    state->ids = fit.channel_ids; state->endpoint = fit.endpoint;
    state->recipe = frozen_recipe; state->protocol_id = fit.protocol_id;
    auto raw = raw_input(fit.training_observations, state->settings.model, state->ids, endpoint);
    // Retained checkpoint/raw assets must not share mutable caller observations.
    raw.data = raw.data.detach().clone();
    raw.observed = raw.observed.clone();
    require(fit.training_source_ids.size() == static_cast<size_t>(raw.data.size(0)),
        "training source IDs must accompany each fit observation");
    for (const auto &id : fit.training_source_ids) require(!id.empty(), "empty training source ID");
    state->source_manifest = source_manifest(fit.training_source_ids);
    state->training = describe_dataset(raw, state->settings.model, fit.feature_units);
    state->scaler = fit_scaler(raw, state->settings.model);
    state->initialization_seed = mixed(fit.seed ^ 0x7270622d696e6974ULL);
    torch::manual_seed(state->initialization_seed); state->trained = Model(state->settings.model);
    torch::manual_seed(state->initialization_seed); state->metadata = Model(state->settings.model);
    torch::manual_seed(state->initialization_seed); state->untrained = Model(state->settings.model);
    verify_same_initialization(state->trained, state->metadata);
    verify_same_initialization(state->trained, state->untrained);
    state->metadata_parameters = metadata_parameters(state->metadata);
    freeze_model(state->untrained);
    const auto optimizer_options = torch::optim::AdamWOptions(settings.learning_rate).weight_decay(settings.weight_decay);
    state->optimizer = std::make_shared<torch::optim::AdamW>(state->trained->parameters(), optimizer_options);
    state->metadata_optimizer = std::make_shared<torch::optim::AdamW>(state->metadata_parameters, optimizer_options);
    state->main_progress = train_model(state->trained, *state->optimizer, state->trained->parameters(),
        *state, options.pretraining_updates, false);
    state->metadata_progress = train_model(state->metadata, *state->metadata_optimizer,
        state->metadata_parameters, *state, options.metadata_updates, true);
    if (options.pretraining_updates == options.metadata_updates)
      require(state->main_progress.attempted == state->metadata_progress.attempted,
          "matched controls did not consume the same attempts");
    freeze_model(state->trained);
    freeze_model(state->metadata);
    // Ensure decoder-only fitting left every encoder parameter identical to initialization.
    const auto original = state->untrained->named_parameters();
    for (const auto &parameter : state->metadata->named_parameters())
      if (parameter.key().rfind("decoder_", 0) != 0)
        require(torch::equal(parameter.value(), original[parameter.key()]),
            "metadata training changed frozen encoder parameter " + parameter.key());

    ev::ReconstructionProvider provider;
    provider.name = kEncoderId;
    provider.export_semantics = reconstruction_output_semantics(state->settings.model);
    provider.provenance = "fresh train-only RPB bottleneck; frozen train-fit float64 scaler; "
        "raw predictions invert float32 neural output in float64; independently trained mask-metadata "
        "control has identical initial encoder/decoder, frozen normalized-zero encoder and decoder-only updates";
    provider.audit_fields = {
      {"provider_recipe", frozen_recipe}, {"protocol_id", fit.protocol_id},
      {"fit_policy", "permitted_training_observations_only;fresh_model_and_scaler"},
      {"fit_source_manifest_id", manifest_identity(state->source_manifest)},
      {"training_observation_dataset_id", state->training.dataset_id},
      {"pretraining_dataset_id", state->training.dataset_id},
      {"scaler_fit_dataset_id", state->training.dataset_id},
      {"preprocessing_id", state->scaler.identity()},
      {"schema_id", state->training.schema_id},
      {"training_observation_rows", std::to_string(fit.training_source_ids.size())},
      {"training_source_groups", std::to_string(
          std::set<std::string>(fit.training_source_ids.begin(), fit.training_source_ids.end()).size())},
      {"prediction_domain", "raw_units;CPU_float64;float32_neural_values_inverted_in_float64"},
      {"checkpoint_writer_core_source_fingerprint", workflow_source_fingerprint()},
      {"adapter_training_source_fingerprint", EVALUATION_SOURCE_ID},
      {"checkpoint_source_scope", "core_writer_only;adapter_producer_in_companion_provenance"},
      {"initialization_seed", std::to_string(state->initialization_seed)},
      {"training_seed", std::to_string(state->settings.seed)},
      {"main_attempted_steps", std::to_string(state->main_progress.attempted)},
      {"main_completed_steps", std::to_string(state->main_progress.completed)},
      {"metadata_attempted_steps", std::to_string(state->metadata_progress.attempted)},
      {"metadata_completed_steps", std::to_string(state->metadata_progress.completed)},
      {"main_first_training_loss", number(state->main_progress.first_loss)},
      {"main_last_training_loss", number(state->main_progress.last_loss)},
      {"metadata_first_training_loss", number(state->metadata_progress.first_loss)},
      {"metadata_last_training_loss", number(state->metadata_progress.last_loss)},
      {"metadata_frozen_encoder_verification", "all_non_decoder_parameters_exactly_equal_untrained_initialization"},
      {"metadata_policy", "only_decoder_*_parameters_fit;frozen_same_initial_encoder;"
          "normalized_zeros_with_actual_visible_support_and_ID_time;eval_dropout_disabled"},
      {"budgets_matched", options.pretraining_updates == options.metadata_updates ? "true" : "false"}
    };
    provider.encode_visible = [state](const embedding::Batch &batch) {
      torch::NoGradGuard no_grad;
      auto input = raw_input(batch, state->settings.model, state->ids, state->endpoint);
      auto output = state->trained->encode(state->scaler.transform(input, state->settings.model));
      return ev::ReconstructionLatent{compact_reconstruction_export(output, state->settings.model).detach().to(torch::kCPU).clone(),
          output.channel_valid_mask.detach().to(torch::kCPU).clone()};
    };
    provider.decode = [state](const torch::Tensor &local) {
      torch::NoGradGuard no_grad;
      require(local.defined() && local.device().is_cpu() && local.dim() == 3 &&
          local.size(0) > 0 && local.size(1) == state->settings.model.channel_count &&
          local.size(2) == state->settings.model.export_width && local.scalar_type() == torch::kFloat32 &&
          torch::isfinite(local).all().item<bool>(), "decoder requires the exact finite CPU float32 declared compact export");
      return raw_prediction(state->trained->decode(local,
          torch::tensor(state->ids, torch::kInt64)), state->scaler, state->settings.model, state->ids);
    };
    provider.metadata_predict = [state](const torch::Tensor &support) {
      torch::NoGradGuard no_grad;
      const auto input = metadata_input(support, state->settings.model, state->ids, state->endpoint);
      const auto output = state->metadata->encode(input);
      const auto local = compact_reconstruction_export(output, state->settings.model);
      return raw_prediction(state->metadata->decode(local, input.channel_ids),
          state->scaler, state->settings.model, state->ids);
    };
    provider.untrained_predict = [state](const embedding::Batch &batch) {
      torch::NoGradGuard no_grad;
      auto input = raw_input(batch, state->settings.model, state->ids, state->endpoint);
      auto output = state->untrained->encode(state->scaler.transform(input, state->settings.model));
      return ev::ReconstructionPrediction{
          raw_prediction(state->untrained->decode(compact_reconstruction_export(output, state->settings.model), input.channel_ids),
              state->scaler, state->settings.model, state->ids),
          output.channel_valid_mask.detach().to(torch::kCPU).clone()};
    };
    provider.save_assets = [state](const std::string &directory) {
      require(fs::is_directory(directory), "asset destination must be an existing unique directory");
      const std::vector<std::string> files{"training-raw.pt", "scaler.pt",
          "model.pt", "training-provenance.pt", "metadata-decoder.pt", "untrained-model.pt"};
      for (const auto &file : files)
        require(!fs::exists(fs::path(directory) / file), "refusing to replace existing reconstruction assets");
      auto path = [&](const char *name) { return (fs::path(directory) / name).string(); };
      save_dataset(path("training-raw.pt"), state->training);
      save_scaler(path("scaler.pt"), state->scaler, state->settings.model,
          state->training.schema_id, state->training.dataset_id);
      Checkpoint checkpoint;
      checkpoint.settings = state->settings; checkpoint.model = state->trained;
      checkpoint.scaler = state->scaler; checkpoint.schema_id = state->training.schema_id;
      checkpoint.dataset_id = state->training.dataset_id;
      checkpoint.scaler_fit_dataset_id = state->training.dataset_id;
      checkpoint.attempted_steps = state->main_progress.attempted;
      checkpoint.completed_steps = state->main_progress.completed;
      save_checkpoint(path("model.pt"), checkpoint, *state->optimizer);
      torch::serialize::OutputArchive provenance;
      write_provenance(provenance, *state, "rpb_reconstruction_training_provenance_v1");
      write_text(provenance, "training_checkpoint_file", "model.pt");
      write_text(provenance, "training_raw_file", "training-raw.pt");
      write_text(provenance, "scaler_file", "scaler.pt");
      provenance.write("attempted_steps", torch::tensor(state->main_progress.attempted), true);
      provenance.write("completed_steps", torch::tensor(state->main_progress.completed), true);
      embedding::archive::save_archive(path("training-provenance.pt"), provenance);
      torch::serialize::OutputArchive metadata, metadata_weights, metadata_optimizer;
      write_provenance(metadata, *state, "rpb_mask_metadata_decoder_control_v1");
      write_text(metadata, "control_policy",
          "same_initial_encoder_and_decoder;encoder_frozen;normalized_zero_values;"
          "actual_visible_support_and_fixed_ID_time;eval_mode;only_decoder_*_parameters_optimized");
      auto metadata_settings = state->settings;
      metadata_settings.steps = state->metadata_progress.completed;
      write_text(metadata, "decoder_training_settings", settings_text(metadata_settings));
      metadata.write("attempted_steps", torch::tensor(state->metadata_progress.attempted), true);
      metadata.write("completed_steps", torch::tensor(state->metadata_progress.completed), true);
      state->metadata->save(metadata_weights); state->metadata_optimizer->save(metadata_optimizer);
      metadata.write("model", metadata_weights); metadata.write("optimizer", metadata_optimizer);
      embedding::archive::save_archive(path("metadata-decoder.pt"), metadata);
      torch::serialize::OutputArchive untrained, untrained_weights;
      write_provenance(untrained, *state, "rpb_reconstruction_untrained_model_v1", true);
      write_text(untrained, "weight_training_policy", "random_initialization;zero_optimizer_updates;train_fit_scaler_only");
      state->untrained->save(untrained_weights); untrained.write("model", untrained_weights);
      untrained.write("attempted_steps", torch::tensor(int64_t{0}), true);
      untrained.write("completed_steps", torch::tensor(int64_t{0}), true);
      embedding::archive::save_archive(path("untrained-model.pt"), untrained);
    };
    return provider;
  };
  return {std::move(factory), frozen_recipe};
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae

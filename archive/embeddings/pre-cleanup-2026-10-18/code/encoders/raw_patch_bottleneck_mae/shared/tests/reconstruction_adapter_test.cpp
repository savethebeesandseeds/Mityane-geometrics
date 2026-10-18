// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/reconstruction_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/data.h"
#include "rpb_test_support.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;

rpb::Model load_weights(const fs::path &path, const rpb::Config &config) {
  torch::serialize::InputArchive archive, weights;
  archive.load_from(path.string(), torch::kCPU);
  torch::Tensor settings, semantics;
  archive.read("settings", settings, true); archive.read("output_semantics", semantics, true);
  check(rpb::parse_settings(embedding::archive::tensor_text(settings)).model.channel_mixer_layers ==
      config.channel_mixer_layers, "control asset mixer configuration");
  check(embedding::archive::tensor_text(semantics) == rpb::output_semantics(config),
        "control asset architecture semantics");
  archive.read("model", weights);
  rpb::Model model(config); model->load(weights); model->eval();
  return model;
}

void adapter_contracts(int64_t mixer_layers) {
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-reconstruction-adapter-mixer-" + std::to_string(mixer_layers) + "-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  auto c = config(); c.channel_mixer_layers = mixer_layers;
  auto settings = rpb::default_settings(); settings.model = c;
  const auto configuration = directory / "settings.conf";
  { std::ofstream out(configuration); out << rpb::settings_text(settings); }

  ev::ReconstructionCard card;
  card.shape = {c.channel_count, c.history_length, c.input_width, torch::kFloat64, torch::kCPU};
  // Physical order is deliberately different from the model's semantic schema.
  card.channel_ids = {202, 101};
  card.feature_units = "volts,amperes"; card.patch_length = c.patch_length;
  card.seeds = {44}; card.tasks = {ev::Task::level};
  rpb::ReconstructionOptions options{configuration.string(), 2, 2, 2};
  for (const int64_t global_mode : {1, 2, 3}) {
    auto global_settings = settings; global_settings.model.global_bottleneck_mode = global_mode;
    const auto global_configuration = directory / ("global-" + std::to_string(global_mode) + ".conf");
    { std::ofstream out(global_configuration); out << rpb::settings_text(global_settings); }
    auto global_options = options; global_options.config_path = global_configuration.string();
    rejects([&] { rpb::make_reconstruction_provider(global_options, card); },
            "global BD export must not enter per-channel BCD reconstruction card");
  }
  const auto registration = rpb::make_reconstruction_provider(options, card);
  check(!registration.recipe.empty(), "resolved recipe absent");
  check(registration.recipe.find("channel_mixer_layers=" + std::to_string(mixer_layers)) != std::string::npos,
        "recipe omits resolved mixer architecture");
  // Registration freezes actual settings, not a path reopened during fitting.
  { std::ofstream out(configuration); out << "unknown_after_registration=true\n"; }
  auto raw = input(c, 4);
  const auto order = torch::tensor({1, 0}, torch::kInt64);
  raw.data = raw.data.index_select(1, order);
  raw.observed = raw.observed.index_select(1, order);
  raw.channel_ids = raw.channel_ids.index_select(0, order);
  const ev::ProviderFitInput fit{{raw.data, raw.observed}, card.shape, 44,
      {"source-a", "source-a", "source-b", "source-b"}, card.channel_ids,
      card.feature_units, "adapter-contract-v1", card.sampling_interval,
      (c.history_length - 1) * card.sampling_interval};
  auto provider = registration.factory(fit);
  check(provider.audit_fields.at("main_completed_steps") == "2" &&
        provider.audit_fields.at("metadata_completed_steps") == "2" &&
        provider.audit_fields.at("budgets_matched") == "true" &&
        provider.audit_fields.at("training_seed") == "44", "training budget/actual seed audit");
  check(provider.export_semantics == rpb::reconstruction_output_semantics(c),
        "served bottleneck semantics");

  embedding::Batch visible{raw.data.clone(), raw.observed.clone()};
  visible.feature_mask.narrow(2, 0, c.patch_length).fill_(false);
  visible.data = torch::where(visible.feature_mask, visible.data, torch::zeros_like(visible.data));
  torch::NoGradGuard no_grad;
  const auto latent = provider.encode_visible(visible);
  const auto predicted = provider.decode(latent.values);
  check(latent.valid.all().item<bool>() && !latent.values.requires_grad(), "frozen local validity");
  check(predicted.scalar_type() == torch::kFloat64 && predicted.sizes() == visible.data.sizes(),
        "raw float64 prediction shape");
  auto poisoned = visible;
  poisoned.data = visible.data.masked_fill(visible.feature_mask.logical_not(),
      std::numeric_limits<double>::quiet_NaN());
  close(provider.encode_visible(poisoned).values, latent.values,
        "adapter must not read hidden storage", 0, 0);
  const auto metadata = provider.metadata_predict(visible.feature_mask);
  const auto untrained = provider.untrained_predict(visible);
  check(untrained.valid.all().item<bool>(), "untrained output validity");
  finite(metadata, "metadata-only prediction");

  const auto assets = directory / "assets"; fs::create_directory(assets);
  provider.save_assets(assets.string());
  auto checkpoint = rpb::load_checkpoint((assets / "model.pt").string());
  checkpoint.model->eval();
  check(checkpoint.settings.model.channel_mixer_layers == mixer_layers &&
        checkpoint.settings.seed == 44 && checkpoint.settings.steps == 2,
        "checkpoint lost actual architecture/seed/budget");
  check(checkpoint.scaler.identity() == provider.audit_fields.at("preprocessing_id"),
        "loaded preprocessing identity");
  for (const auto &parameter : checkpoint.model->parameters())
    check(parameter.requires_grad(), "ordinary checkpoint must restore trainable parameters");
  close(provider.encode_visible(visible).values, latent.values,
        "asset serialization must not mutate served vector", 0, 0);
  const auto saved = rpb::load_dataset((assets / "training-raw.pt").string(), checkpoint.settings.model);
  check(saved.dataset_id == checkpoint.dataset_id &&
        checkpoint.scaler_fit_dataset_id == saved.dataset_id &&
        saved.feature_units == card.feature_units &&
        checkpoint.attempted_steps == 2 && checkpoint.completed_steps == 2,
        "ordinary checkpoint and dataset association");
  close(saved.input.data, raw.data, "raw archive preserves input precision", 0, 0);
  auto served = raw; served.data = visible.data; served.observed = visible.feature_mask;
  const auto served_latent = checkpoint.model->encode(
      checkpoint.scaler.transform(served, checkpoint.settings.model));
  const auto compact = rpb::compact_reconstruction_export(served_latent, checkpoint.settings.model);
  check(compact.defined() && (mixer_layers > 0 ?
        torch::equal(compact, served_latent.z_contextual) : torch::equal(compact, served_latent.z_local)),
        "decoder compact selector differs from architecture export");
  std::cout << "saved export max_abs=" << (compact - latent.values).abs().max().item<double>()
            << "; asset_directory=" << assets << '\n';
  // Existing workflow parity uses 1e-6 for independently restored float32 inference.
  close(compact, latent.values, "saved checkpoint compact export", 1e-6, 1e-6);
  for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(false);
  const auto frozen_restored = checkpoint.model->encode(
      checkpoint.scaler.transform(served, checkpoint.settings.model));
  std::cout << "same-flags restored export max_abs="
            << (rpb::compact_reconstruction_export(frozen_restored, c) - latent.values).abs().max().item<double>() << '\n';
  const auto coordinate_indices = rpb::channel_indices(served.channel_ids, c, 4, torch::kCPU).flatten();
  const auto mean = checkpoint.scaler.mean.index_select(0, coordinate_indices).reshape({4, 2, 1, 2});
  const auto scale = checkpoint.scaler.scale.index_select(0, coordinate_indices).reshape({4, 2, 1, 2});
  const auto decoded = checkpoint.model->decode(compact, served.channel_ids).to(torch::kFloat64) * scale + mean;
  close(decoded, predicted, "saved checkpoint raw decoder prediction", 1e-6, 1e-6);
  auto metadata_model = load_weights(assets / "metadata-decoder.pt", c);
  auto untrained_model = load_weights(assets / "untrained-model.pt", c);
  for (auto &parameter : metadata_model->parameters()) parameter.set_requires_grad(false);
  bool decoder_changed = false;
  const auto original = untrained_model->named_parameters();
  for (const auto &parameter : metadata_model->named_parameters()) {
    if (parameter.key().rfind("decoder_", 0) == 0)
      decoder_changed = decoder_changed || !torch::equal(parameter.value(), original[parameter.key()]);
    else close(parameter.value(), original[parameter.key()], "metadata encoder must remain frozen", 0, 0);
  }
  check(decoder_changed, "metadata decoder did not train");
  auto mask_input = served;
  mask_input.data = torch::zeros(visible.data.sizes(), torch::kFloat32);
  const auto metadata_encoded = metadata_model->encode(mask_input);
  const auto meta_local = rpb::compact_reconstruction_export(metadata_encoded, c);
  close(metadata_model->decode(meta_local, served.channel_ids).to(torch::kFloat64) * scale + mean,
        metadata, "metadata-only callback matches saved mask-only decoder", 1e-6, 1e-6);
  const auto random_encoded = untrained_model->encode(checkpoint.scaler.transform(served, c));
  close(untrained_model->decode(rpb::compact_reconstruction_export(random_encoded, c), served.channel_ids)
            .to(torch::kFloat64) * scale + mean,
        untrained.values, "untrained callback uses its saved exact bottleneck", 1e-6, 1e-6);
  auto missing_channel = visible;
  missing_channel.feature_mask = visible.feature_mask.clone();
  missing_channel.feature_mask[0][1].fill_(false);
  missing_channel.data = visible.data.masked_fill(missing_channel.feature_mask.logical_not(),
      std::numeric_limits<double>::quiet_NaN());
  const auto sparse = provider.encode_visible(missing_channel);
  check(sparse.valid[0][0].item<bool>() && !sparse.valid[0][1].item<bool>(),
        "peer context cannot expand recipient observed validity");
  close(sparse.values[0][1], torch::zeros({c.export_width}), "missing recipient compact vector", 0, 0);
  auto empty = visible;
  empty.feature_mask = torch::zeros_like(visible.feature_mask);
  empty.data = torch::zeros_like(visible.data);
  const auto absent = provider.encode_visible(empty);
  check(!absent.valid.any().item<bool>(), "all-missing export validity");
  close(absent.values, torch::zeros_like(absent.values), "all-missing export placeholder", 0, 0);
  check(!provider.untrained_predict(empty).valid.any().item<bool>(), "untrained absent validity");
  // Saved bias terms must not synthesize valid/nonnull exports from absent storage.
  for (auto &parameter : checkpoint.model->named_parameters())
    if (parameter.key().find("bias") != std::string::npos) parameter.value().fill_(0.75);
  auto absent_input = served; absent_input.data = empty.data; absent_input.observed = empty.feature_mask;
  const auto biased_absent = checkpoint.model->encode(checkpoint.scaler.transform(absent_input, c));
  check(!biased_absent.channel_valid_mask.any().item<bool>(), "biased model expanded absent validity");
  const auto biased_compact = rpb::compact_reconstruction_export(biased_absent, c);
  close(biased_compact, torch::zeros_like(biased_compact), "biased absent compact export", 0, 0);
  rejects([&] { provider.save_assets(assets.string()); }, "asset overwrite");
  rejects([&] { rpb::load_checkpoint((assets / "metadata-decoder.pt").string()); },
          "metadata control masquerading as train checkpoint");
  std::cout << "RPB reconstruction adapter contracts passed; artifacts=" << directory << '\n';
}
} // namespace

int main() {
  try { torch::set_num_threads(1); for (const int64_t mixer_layers : {0, 1}) adapter_contracts(mixer_layers); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

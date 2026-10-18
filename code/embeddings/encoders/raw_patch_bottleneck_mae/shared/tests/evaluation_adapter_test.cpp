// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/data.h"
#include "rpb_test_support.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;

std::string read_text(torch::serialize::InputArchive &archive, const char *key) {
  torch::Tensor value; archive.read(key, value, true);
  return embedding::archive::tensor_text(value);
}

int64_t read_count(torch::serialize::InputArchive &archive, const char *key) {
  torch::Tensor value; archive.read(key, value, true);
  check(value.scalar_type() == torch::kInt64 && value.numel() == 1, std::string(key) + " type");
  return value.item<int64_t>();
}

struct FrozenAsset {
  rpb::Settings settings;
  rpb::Model model{nullptr};
  rpb::FrozenScaler scaler;
  std::string pretraining_id, permitted_fit_id, schema_id, producer, policy;
  int64_t attempted{0}, completed{0}, weight_budget{0};
};

FrozenAsset load_asset(const fs::path &directory, const std::string &prefix,
                      const ev::ProviderFitInput &fit) {
  torch::serialize::InputArchive archive, weights;
  archive.load_from((directory / (prefix + "-model.pt")).string(), torch::kCPU);
  check(read_text(archive, "encoder_id") == rpb::kEncoderId, "frozen asset encoder");
  check(read_text(archive, "artifact_kind") == "rpb_frozen_feature_model_v1", "frozen asset kind");
  check(read_count(archive, "format_version") == 1, "frozen asset version");
  FrozenAsset result;
  result.settings = rpb::parse_settings(read_text(archive, "settings"));
  check(read_text(archive, "output_semantics") == rpb::output_semantics(result.settings.model),
        "frozen asset architecture semantics");
  result.pretraining_id = read_text(archive, "pretraining_dataset_id");
  result.permitted_fit_id = read_text(archive, "permitted_fit_observation_dataset_id");
  result.schema_id = read_text(archive, "schema_id");
  result.producer = read_text(archive, "training_producer_source_fingerprint");
  result.policy = read_text(archive, "weight_training_policy");
  result.attempted = read_count(archive, "attempted_steps");
  result.completed = read_count(archive, "completed_steps");
  result.weight_budget = read_count(archive, "model_weight_update_budget");
  check(read_count(archive, "actual_training_seed") == result.settings.seed, "saved actual seed");
  check(read_text(archive, "feature_units") == fit.feature_units, "custom units preserved");
  check(read_text(archive, "protocol_id") == fit.protocol_id, "protocol identity preserved");
  check(read_text(archive, "rng_policy") == "splitmix64-counter-rows-masks-torch-attempt-v1", "counter policy");
  torch::Tensor ids, interval, endpoint;
  archive.read("channel_order", ids, true); archive.read("sampling_interval", interval, true);
  archive.read("endpoint", endpoint, true);
  close(ids, torch::tensor(fit.channel_ids, torch::kInt64), "physical semantic order", 0, 0);
  check(interval.item<double>() == fit.sampling_interval && endpoint.item<double>() == fit.endpoint,
        "uniform endpoint metadata");
  result.model = rpb::Model(result.settings.model);
  archive.read("model", weights); result.model->load(weights); result.model->eval();
  result.scaler = rpb::load_scaler((directory / (prefix + "-scaler.pt")).string(),
      result.settings.model, result.schema_id);
  check(result.scaler.identity() == read_text(archive, "preprocessing_id"), "frozen scaler identity");
  return result;
}

void same_parameters(const rpb::Model &actual, const rpb::Model &expected, const std::string &label) {
  const auto expected_parameters = expected->named_parameters();
  check(actual->named_parameters().size() == expected_parameters.size(), label + " parameter count");
  for (const auto &parameter : actual->named_parameters())
    close(parameter.value(), expected_parameters[parameter.key()], label + "/" + parameter.key(), 0, 0);
}

void adapter_provenance(int64_t mixer_layers) {
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-feature-adapter-mixer-" + std::to_string(mixer_layers) + "-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  auto c = config(); c.sampling_interval = 0.5; c.channel_mixer_layers = mixer_layers;
  auto configured = rpb::default_settings(); configured.model = c;
  configured.seed = 101; configured.steps = 8; configured.batch_size = 2;
  const auto configuration = directory / "settings.conf";
  { std::ofstream output(configuration); output << rpb::settings_text(configured); }
  const auto two_factory = rpb::make_evaluation_provider({configuration.string(), "", 2});
  const auto three_factory = rpb::make_evaluation_provider({configuration.string(), "", 3});
  // The existing API captures resolved settings; fitting must not reopen this path.
  { std::ofstream output(configuration); output << "invalid_after_factory_creation=true\n"; }
  auto raw = input(c, 4);
  const auto order = torch::tensor({1, 0}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order);
  raw.channel_ids = raw.channel_ids.index_select(0, order);
  // Retain natural missingness while every channel has enough observed patches.
  raw.observed.select(0, 0).select(0, 1).narrow(0, 0, 3).fill_(false);
  const ev::ProviderFitInput fit{{raw.data, raw.observed},
      {c.channel_count, c.history_length, c.input_width, torch::kFloat64, torch::kCPU}, 202,
      {"source-a", "source-a", "source-b", "source-b"}, {202, 101},
      "volts,amperes", "controlled-pairs-v2/v2/level", c.sampling_interval,
      (c.history_length - 1) * c.sampling_interval};
  const auto raw_before = raw.data.clone(), observed_before = raw.observed.clone();
  auto two = two_factory(fit), three = three_factory(fit);
  close(raw.data, raw_before, "fit must not mutate observations", 0, 0);
  close(raw.observed, observed_before, "fit must not mutate support", 0, 0);
  check(two.audit_fields.at("actual_training_seed") == "202" &&
      two.audit_fields.at("rpb_trained_completed_steps") == "2" &&
      three.audit_fields.at("rpb_trained_completed_steps") == "3" &&
      two.audit_fields.at("rpb_untrained_model_weight_update_budget") == "0" &&
      two.audit_fields.at("rpb_untrained_pretraining_dataset_id") == "none",
      "actual budgets/seeds and random provenance");
  const auto audit_settings = rpb::parse_settings(two.audit_fields.at("resolved_settings"));
  check(audit_settings.seed == 202 && audit_settings.steps == 2 &&
      audit_settings.batch_size == configured.batch_size &&
      audit_settings.model.channel_mixer_layers == mixer_layers, "canonical resolved audit settings");
  check(two.surfaces.at("rpb_trained_channel_concatenation").channel_order == fit.channel_ids,
      "declared concatenation order");
  check(two.surfaces.size() == (mixer_layers > 0 ? 8 : 4), "architecture surface inventory");
  if (mixer_layers > 0) {
    for (const std::string prefix : {"rpb_trained", "rpb_untrained"}) {
      check(two.surfaces.at(prefix + "_contextual_global").kind == ev::SurfaceKind::global,
            "contextual global routing");
      const auto &description = two.surfaces.at(prefix + "_contextual_channel_concatenation");
      check(description.kind == ev::SurfaceKind::channel_concatenation &&
          description.channel_order == fit.channel_ids && !description.support_rule.empty(),
          "contextual semantic concatenation routing");
    }
  }
  const auto two_assets = directory / "two", three_assets = directory / "three";
  fs::create_directory(two_assets); fs::create_directory(three_assets);
  two.save_assets(two_assets.string()); three.save_assets(three_assets.string());
  auto random_two = load_asset(two_assets, "rpb_untrained", fit);
  auto trained_two = load_asset(two_assets, "rpb_trained", fit);
  auto random_three = load_asset(three_assets, "rpb_untrained", fit);
  auto trained_three = load_asset(three_assets, "rpb_trained", fit);
  const auto described = rpb::describe_dataset(raw, c, fit.feature_units);
  const auto reference_scaler = rpb::fit_scaler(raw, c);
  check(trained_two.settings.seed == 202 && trained_two.settings.steps == 2 &&
      trained_three.settings.seed == 202 && trained_three.settings.steps == 3 &&
      trained_two.settings.model.channel_mixer_layers == mixer_layers &&
      random_two.settings.model.channel_mixer_layers == mixer_layers,
      "saved settings use actual seed and requested updates");
  check(trained_two.completed == 2 && trained_two.attempted == 2 && trained_two.weight_budget == 2 &&
      trained_three.completed == 3 && trained_three.attempted == 3 && trained_three.weight_budget == 3,
      "saved attempt/completed counters");
  check(random_two.pretraining_id == "none" && random_two.attempted == 0 &&
      random_two.completed == 0 && random_two.weight_budget == 0 && random_two.producer == "none",
      "random weights must not claim pretraining");
  check(trained_two.pretraining_id == described.dataset_id &&
      random_two.permitted_fit_id == described.dataset_id &&
      trained_two.permitted_fit_id == described.dataset_id &&
      trained_two.schema_id == described.schema_id && random_two.schema_id == described.schema_id,
      "permitted fit and custom-unit schema identity");
  check(random_two.scaler.identity() == reference_scaler.identity() &&
      trained_two.scaler.identity() == reference_scaler.identity() &&
      trained_three.scaler.identity() == reference_scaler.identity(),
      "same train-only scaler for random and trained controls");
  check(trained_two.producer == two.audit_fields.at("evaluation_source_fingerprint") &&
      trained_two.producer == two.audit_fields.at("rpb_trained_training_producer_source_fingerprint"),
      "actual adapter producer identity");
  same_parameters(random_two.model, random_three.model, "matched random controls across update budgets");
  torch::manual_seed(rpb::training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL));
  auto initialized = rpb::Model(c);
  same_parameters(random_two.model, initialized, "saved random weights match declared initialization");
  bool changed = false;
  const auto two_parameters = trained_two.model->named_parameters();
  for (const auto &parameter : trained_three.model->named_parameters())
    changed = changed || !torch::equal(parameter.value(), two_parameters[parameter.key()]);
  check(changed, "third update must change trained weights");
  {
    torch::NoGradGuard no_grad;
    const auto features = two.extract({raw.data, raw.observed});
    const auto normalized = trained_two.scaler.transform(raw, trained_two.settings.model);
    const auto served = trained_two.model->encode(normalized);
    close(features.at("rpb_trained_global").values, served.z_global, "loaded frozen global feature", 1e-6, 1e-6);
    close(features.at("rpb_trained_channel_concatenation").values, served.z_local.flatten(1),
        "loaded frozen semantic concatenation", 1e-6, 1e-6);
    if (mixer_layers > 0) {
      close(features.at("rpb_trained_contextual_global").values, served.z_contextual_global,
            "loaded frozen contextual global feature", 1e-6, 1e-6);
      close(features.at("rpb_trained_contextual_channel_concatenation").values,
            served.z_contextual.flatten(1), "loaded frozen contextual semantic concatenation", 1e-6, 1e-6);
    }
    auto poisoned = raw;
    poisoned.data = raw.data.masked_fill(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
    const auto absent_storage = two.extract({poisoned.data, poisoned.observed});
    close(absent_storage.at("rpb_trained_global").values, features.at("rpb_trained_global").values,
         "natural absent storage ignored", 0, 0);
    if (mixer_layers > 0)
      close(absent_storage.at("rpb_trained_contextual_global").values,
            features.at("rpb_trained_contextual_global").values,
            "contextual natural absent storage ignored", 0, 0);
    auto missing_channel = raw;
    missing_channel.observed = raw.observed.clone();
    missing_channel.observed.select(0, 0).select(0, 1).fill_(false);
    missing_channel.data = raw.data.masked_fill(missing_channel.observed.logical_not(),
        std::numeric_limits<double>::quiet_NaN());
    const auto sparse_features = two.extract({missing_channel.data, missing_channel.observed});
    for (const std::string prefix : {"rpb_trained", "rpb_untrained"}) {
      check(sparse_features.at(prefix + "_global").valid[0].item<bool>() &&
          !sparse_features.at(prefix + "_channel_concatenation").valid[0].item<bool>(),
          "partial observed support policy");
      if (mixer_layers > 0) {
        check(sparse_features.at(prefix + "_contextual_global").valid[0].item<bool>() &&
            !sparse_features.at(prefix + "_contextual_channel_concatenation").valid[0].item<bool>(),
            "context cannot expand observed support");
        close(sparse_features.at(prefix + "_contextual_channel_concatenation").values
                  .reshape({raw.data.size(0), c.channel_count, c.export_width})[0][1],
              torch::zeros({c.export_width}), "absent contextual channel remains zero", 0, 0);
      }
    }
    const embedding::Batch empty{torch::zeros_like(raw.data), torch::zeros_like(raw.observed)};
    for (const auto &[name, surface] : two.extract(empty)) {
      check(!surface.valid.any().item<bool>(), name + " invalid support");
      close(surface.values, torch::zeros_like(surface.values), name + " zero invalid placeholder", 0, 0);
    }
    // Held-out extraction must not change the frozen training weights or scaler.
    two.extract({raw.data + 100.0, raw.observed});
  }
  const auto after_assets = directory / "after-heldout"; fs::create_directory(after_assets);
  two.save_assets(after_assets.string());
  const auto after = load_asset(after_assets, "rpb_trained", fit);
  same_parameters(after.model, trained_two.model, "heldout inference cannot refit weights");
  check(after.scaler.identity() == trained_two.scaler.identity(), "heldout inference cannot refit scaler");
  rejects([&] { rpb::load_checkpoint((two_assets / "rpb_trained-model.pt").string()); },
      "feature asset masquerading as resumable checkpoint");

  // A supplied checkpoint retains its actual original seed/settings, independently
  // of the current fit seed and of the matched random control's initialization.
  rpb::Checkpoint checkpoint;
  checkpoint.settings = trained_three.settings; checkpoint.settings.seed = 777; checkpoint.settings.steps = 11;
  checkpoint.model = trained_three.model; checkpoint.scaler = trained_three.scaler;
  checkpoint.schema_id = described.schema_id; checkpoint.dataset_id = described.dataset_id;
  checkpoint.scaler_fit_dataset_id = described.dataset_id; checkpoint.attempted_steps = 3; checkpoint.completed_steps = 3;
  torch::optim::AdamW optimizer(checkpoint.model->parameters(),
      torch::optim::AdamWOptions(checkpoint.settings.learning_rate).weight_decay(checkpoint.settings.weight_decay));
  const auto checkpoint_file = directory / "supplied.pt";
  rpb::save_checkpoint(checkpoint_file.string(), checkpoint, optimizer);
  auto frozen = rpb::make_evaluation_provider({"", checkpoint_file.string(), 0})(fit);
  check(frozen.audit_fields.at("actual_training_seed") == "777" &&
      frozen.audit_fields.at("training_seed_policy") == "preserved_checkpoint_pretraining_seed",
      "frozen checkpoint seed policy");
  const auto frozen_assets = directory / "frozen"; fs::create_directory(frozen_assets);
  frozen.save_assets(frozen_assets.string());
  const auto reexported = load_asset(frozen_assets, "rpb_trained", fit);
  check(reexported.settings.seed == 777 && reexported.settings.steps == 11 &&
      reexported.completed == 3 && reexported.pretraining_id == described.dataset_id &&
      reexported.settings.model.channel_mixer_layers == mixer_layers,
      "frozen checkpoint pretraining settings retained");
  same_parameters(reexported.model, trained_three.model, "supplied weights unchanged");
  // Registration namespaces affect names/provenance, never fitting or initialization.
  { std::ofstream output(configuration); output << rpb::settings_text(configured); }
  for (const std::string invalid_prefix : {"", "../rpb", "rpb-mixer", "rpb mixer"})
    rejects([&] { rpb::make_evaluation_provider(
        {configuration.string(), "", 2, invalid_prefix, false}); }, "invalid registration namespace");
  const auto named_factory = rpb::make_evaluation_provider(
      {configuration.string(), "", 2, "rpb_mixer", true});
  if (mixer_layers == 0) {
    rejects([&] { named_factory(fit); }, "mixer-required registration with disabled configuration");
  } else {
    auto named = named_factory(fit);
    check(named.surfaces.size() == 8 &&
        named.surfaces.at("rpb_mixer_trained_contextual_global").kind == ev::SurfaceKind::global &&
        named.surfaces.at("rpb_mixer_trained_contextual_channel_concatenation").channel_order == fit.channel_ids,
        "custom contextual namespace routing");
    for (const auto &[name, description] : named.surfaces) {
      (void)description;
      check(name.rfind("rpb_mixer_", 0) == 0 && two.surfaces.count(name) == 0,
            "custom namespace collides with plain registration");
    }
    check(named.audit_fields.at("surface_prefix") == "rpb_mixer" &&
        named.audit_fields.at("rpb_mixer_untrained_pretraining_dataset_id") == "none" &&
        named.audit_fields.at("rpb_mixer_untrained_completed_steps") == "0" &&
        named.audit_fields.at("rpb_mixer_untrained_model_weight_update_budget") == "0" &&
        named.audit_fields.at("rpb_mixer_untrained_training_producer_source_fingerprint") == "none" &&
        named.audit_fields.at("rpb_mixer_trained_training_producer_source_fingerprint") ==
            named.audit_fields.at("evaluation_source_fingerprint"),
        "custom namespace random/trained provenance");
    const auto named_assets = directory / "named"; fs::create_directory(named_assets);
    named.save_assets(named_assets.string());
    const auto named_random = load_asset(named_assets, "rpb_mixer_untrained", fit);
    const auto named_trained = load_asset(named_assets, "rpb_mixer_trained", fit);
    check(named_random.pretraining_id == "none" && named_random.completed == 0 &&
        named_random.weight_budget == 0 && named_random.producer == "none" &&
        named_trained.producer == named.audit_fields.at("evaluation_source_fingerprint"),
        "custom namespace frozen asset provenance");
    same_parameters(named_random.model, random_two.model, "namespace preserves random initialization");
    same_parameters(named_trained.model, trained_two.model, "namespace preserves trained weights");
  }
  std::cout << "RPB feature adapter provenance tests passed; artifacts=" << directory << '\n';
}
} // namespace

int main() {
  try { torch::set_num_threads(1); for (const int64_t mixer_layers : {0, 1}) adapter_provenance(mixer_layers); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

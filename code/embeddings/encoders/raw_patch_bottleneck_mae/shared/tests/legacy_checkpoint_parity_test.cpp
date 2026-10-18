// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/data.h"
#include "rpb_test_support.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace fs = std::filesystem;
using namespace rpb_test;

std::pair<uint64_t, uint64_t> file_identity(const fs::path &path) {
  std::ifstream file(path, std::ios::binary);
  check(file.good(), "missing archived fixture: " + path.string());
  uint64_t hash = 14695981039346656037ULL, bytes = 0;
  std::array<char, 65536> buffer{};
  while (file) {
    file.read(buffer.data(), buffer.size());
    const auto count = file.gcount();
    for (std::streamsize i = 0; i < count; ++i) {
      hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
      hash *= 1099511628211ULL;
    }
    bytes += static_cast<uint64_t>(count);
  }
  check(!file.bad(), "failed reading archived fixture: " + path.string());
  return {bytes, hash};
}

embedding::Batch load_observations(const fs::path &path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), torch::kCPU);
  embedding::Batch result;
  // Scoring labels and clean hidden signals are deliberately never loaded.
  archive.read("observed", result.data, true);
  archive.read("feature_mask", result.feature_mask, true);
  check(result.data.scalar_type() == torch::kFloat64 &&
      result.feature_mask.scalar_type() == torch::kBool &&
      result.data.sizes() == result.feature_mask.sizes(),
      "archived observed precision/support contract");
  return result;
}

void check_archived_surface(const fs::path &path, const torch::Tensor &values,
                            const torch::Tensor &valid) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), torch::kCPU);
  torch::Tensor original_values, original_valid;
  archive.read("features", original_values, true);
  archive.read("valid", original_valid, true);
  check(values.scalar_type() == original_values.scalar_type() &&
      values.sizes() == original_values.sizes() && torch::equal(values, original_values),
      "exact archived CPU features changed: " + path.string());
  check(valid.scalar_type() == original_valid.scalar_type() &&
      valid.sizes() == original_valid.sizes() && torch::equal(valid, original_valid),
      "exact archived feature support changed: " + path.string());
}

void archived_checkpoint(const fs::path &root, int64_t master, bool mixer) {
  const auto seed_directory = root / ("seed-" + std::to_string(master));
  const auto directory = seed_directory / (mixer ? "mixer" : "independent") / "milestone-128";
  const auto checkpoint_path = directory / "checkpoint.pt";
  const auto raw_path = directory / "checkpoint.pt.training-raw.pt";
  const auto scaler_path = directory / "checkpoint.pt.scaler.pt";
  std::map<fs::path, std::pair<uint64_t, uint64_t>> preserved;
  for (const auto &path : {checkpoint_path, raw_path, scaler_path,
                           seed_directory / "controlled-training.pt",
                           seed_directory / "controlled-validation.pt"})
    preserved.emplace(path, file_identity(path));
  for (const std::string split : {"training", "validation"})
    for (const std::string surface : {"curve_global", "curve_channel_concatenation"}) {
      const auto path = directory / (surface + "-" + split + ".pt");
      preserved.emplace(path, file_identity(path));
    }

  torch::serialize::InputArchive saved;
  saved.load_from(checkpoint_path.string(), torch::kCPU);
  torch::Tensor original_settings, unexpected;
  saved.read("settings", original_settings, true);
  const auto text = embedding::archive::tensor_text(original_settings);
  check(text.find("global_bottleneck_mode=") == std::string::npos &&
      !saved.try_read("global_bottleneck_mode", unexpected, true) &&
      !saved.try_read("reconstruction_export_semantics", unexpected, true),
      "expected actual pre-global-bottleneck checkpoint without new optional tags");
  check(rpb::settings_text(rpb::parse_settings(text)) == text,
      "old canonical settings/configuration identity changed");
  auto checkpoint = rpb::load_checkpoint(checkpoint_path.string(), torch::kCPU);
  check(checkpoint.settings.model.global_bottleneck_mode == 0 &&
      checkpoint.settings.model.channel_mixer_layers == (mixer ? 1 : 0) &&
      checkpoint.completed_steps == 128 && checkpoint.attempted_steps >= 128 &&
      checkpoint.settings.seed == master,
      "old checkpoint mode/counters/seed changed");
  // Archived feature-provider extraction used eval/no-grad with trainable CPU
  // parameter flags. Match all three settings, retaining exact comparisons.
  checkpoint.model->eval();
  for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(true);
  auto restored_settings = checkpoint.settings;
  restored_settings.model.device = rpb::parse_settings(text).model.device;
  check(rpb::settings_text(restored_settings) == text, "loaded old settings lost canonical identity");
  const auto training = rpb::load_dataset(raw_path.string(), checkpoint.settings.model);
  const auto scaler = rpb::load_scaler(scaler_path.string(), checkpoint.settings.model, training.schema_id);
  check(training.dataset_id == checkpoint.dataset_id && training.schema_id == checkpoint.schema_id &&
      checkpoint.scaler_fit_dataset_id == training.dataset_id &&
      scaler.identity() == checkpoint.scaler.identity(), "old raw/scaler/checkpoint association changed");
  for (const auto &pair : {
      std::pair<torch::Tensor, torch::Tensor>{scaler.mean, checkpoint.scaler.mean},
      {scaler.scale, checkpoint.scaler.scale}, {scaler.count, checkpoint.scaler.count},
      {scaler.channel_ids, checkpoint.scaler.channel_ids},
      {scaler.floor_applied, checkpoint.scaler.floor_applied}})
    close(pair.first, pair.second, "archived checkpoint versus companion scaler state", 0, 0);
  const auto endpoint = training.input.endpoints[0].item<double>();
  check(training.input.endpoints.eq(endpoint).all().item<bool>(),
        "controlled archive must preserve its one declared endpoint");

  torch::NoGradGuard no_grad;
  for (const std::string split : {"training", "validation"}) {
    const auto observations = load_observations(seed_directory / ("controlled-" + split + ".pt"));
    if (split == "training") {
      close(observations.feature_mask, training.input.observed, "archived raw training support", 0, 0);
      close(observations.data.masked_select(observations.feature_mask),
            training.input.data.masked_select(training.input.observed),
            "archived observed raw training values/precision", 0, 0);
    }
    rpb::Input raw{observations.data, observations.feature_mask, training.input.channel_ids,
        torch::full({observations.data.size(0)}, endpoint, torch::kFloat64),
        training.input.sampling_interval};
    const auto output = checkpoint.model->encode(checkpoint.scaler.transform(raw, checkpoint.settings.model));
    const auto global = mixer ? output.z_contextual_global : output.z_global;
    const auto channels = mixer ? output.z_contextual : output.z_local;
    check_archived_surface(directory / ("curve_global-" + split + ".pt"),
                           global, output.sample_valid_mask);
    check_archived_surface(directory / ("curve_channel_concatenation-" + split + ".pt"),
                           channels.flatten(1), output.channel_valid_mask.all(1));
  }
  for (const auto &[path, identity] : preserved)
    check(file_identity(path) == identity, "read-only parity test mutated old archive: " + path.string());
  std::cout << "PASS: exact old CPU training/validation global+channel features; master="
      << master << " architecture=" << (mixer ? "mixer" : "independent") << '\n';
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::string root = "output/runs/rpb-implementation/learning-curve-49fd5f8227/results";
    if (argc != 1) {
      check(argc == 3 && std::string(argv[1]) == "--input-root", "usage: legacy_checkpoint_parity_test [--input-root OLD_CURVE_RESULTS]");
      root = argv[2];
    }
    torch::set_num_threads(1);
    for (const int64_t master : {901, 1002, 1103})
      for (const bool mixer : {false, true}) archived_checkpoint(root, master, mixer);
    std::cout << "PASS: six archived mode0 checkpoints and 24 exact CPU surfaces; all inputs byte-preserved\n";
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

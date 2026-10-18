// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "cuwacunu/artifacts/v1/sidecar.h"
#include "cuwacunu/artifacts/v1/tensor_state.h"
#include "cuwacunu/data/kline_feature_registry.h"
#include "cuwacunu/encoder/encoder.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>

// Application-owned adoption and checked loading. The evaluated Encoder's
// version-owned checkpoint reader and numerical implementation stay unchanged.
namespace cuwacunu::artifacts::selected_encoder {
namespace primary = cuwacunu::encoder::primary;
namespace version = primary::version;
namespace implementation = version::implementation;
namespace contract = cuwacunu::artifacts::v1;
namespace fs = std::filesystem;

inline std::vector<fs::path> source_paths(
    const fs::path &repo_root) {
  const auto root = fs::canonical(repo_root);
  const fs::path owned = "code/encoders/causal_patch_bottleneck_mae/versions/v3/include";
  TORCH_CHECK(fs::is_directory(root / owned),
      "[Encoder sidecar] missing owned v3 source tree");
  std::vector<fs::path> paths;
  // These are the same self-contained owned headers recorded by the existing
  // CPB-v3 integration source receipt. Include the whole tree conservatively,
  // so a newly introduced numerical include cannot escape the source digest.
  for (const auto &entry : fs::recursive_directory_iterator(root / owned)) {
    TORCH_CHECK(!entry.is_symlink(),
        "[Encoder sidecar] symbolic source entries are unsupported");
    if (entry.is_regular_file() && entry.path().extension() == ".h")
      paths.push_back(fs::relative(entry.path(), root));
  }
  for (const auto &relative : {
      "code/integration/include/cuwacunu/encoder/encoder.h",
      "code/integration/include/cuwacunu/integration/causal_encoder_input.h",
      "code/data/include/cuwacunu/data/kline_feature_registry.h",
      "code/data/include/cuwacunu/data/sample.h",
      "code/config/include/cuwacunu/config/document.h",
      "code/config/include/piaabo/parse/simple_kv_block.h"})
    paths.emplace_back(relative);
  std::sort(paths.begin(), paths.end());
  TORCH_CHECK(!paths.empty() && fs::is_regular_file(root / owned /
      "embedding/encoders/causal_patch_bottleneck_mae/versions/v3.h"),
      "[Encoder sidecar] missing Encoder v3 public entry");
  for (const auto &relative : paths) {
    const auto path = root / relative;
    TORCH_CHECK(fs::is_regular_file(path) && !fs::is_symlink(path),
        "[Encoder sidecar] missing/unsupported source: ", relative.generic_string());
  }
  return paths;
}

inline std::string source_digest(const fs::path &repo_root) {
  return contract::source_closure_sha256(repo_root, source_paths(repo_root));
}

inline torch::Tensor text_state(const std::string &value) {
  return torch::tensor(std::vector<uint8_t>(value.begin(), value.end()), torch::kUInt8);
}

inline std::vector<contract::NamedTensor> preprocessing_state(
    const primary::LoadedEncoder &loaded) {
  const auto &s = loaded.scaler;
  const auto &n = loaded.model->normalizer();
  // Neither FrozenScaler nor BankStandardizer is a registered Torch module.
  // Their actual values must be hashed, rather than only the legacy FNV IDs.
  return {
      {"input_scaler.mean", s.mean}, {"input_scaler.scale", s.scale},
      {"input_scaler.count", s.count}, {"input_scaler.channel_ids", s.channel_ids},
      {"input_scaler.floor_applied", s.floor_applied},
      {"input_scaler.scale_floor", torch::tensor(s.scale_floor, torch::kFloat64)},
      {"bank_normalizer.counts", n.counts}, {"bank_normalizer.sum", n.sum},
      {"bank_normalizer.sumsq", n.sumsq}, {"bank_normalizer.mean", n.mean},
      {"bank_normalizer.scale", n.scale}, {"bank_normalizer.support", n.support},
      {"bank_normalizer.input_geometry", torch::tensor(n.input_geometry, torch::kInt64)},
      {"bank_normalizer.sampling_interval", torch::tensor(n.sampling_interval, torch::kFloat64)},
      {"bank_normalizer.floor", torch::tensor(n.floor, torch::kFloat64)},
      {"bank_normalizer.minimum_support", torch::tensor(n.minimum_support, torch::kInt64)},
      {"bank_normalizer.frozen", torch::tensor(n.frozen, torch::kBool)},
      {"bank_normalizer.source_identity", text_state(n.source_identity)},
      {"bank_normalizer.input_scaler_identity", text_state(n.input_scaler_identity)},
      {"bank_normalizer.layout_identity", text_state(n.layout_identity)},
      {"bank_normalizer.recipe_identity", text_state(n.recipe_identity)},
      {"bank_normalizer.bank_family", text_state(n.bank_family)},
      {"bank_normalizer.raw_contract", text_state(n.raw_contract)},
      {"bank_normalizer.increment_recipe", text_state(n.increment_recipe)}};
}

inline std::string inference_config_digest(const primary::Config &config) {
  primary::validate_config(config);
  const auto &c = config.level;
  auto integers = implementation::configurable_checkpoint_detail::integers(c);
  integers.push_back(config.change_width);
  const std::vector<contract::NamedTensor> geometry{
      {"geometry", torch::tensor(integers, torch::kInt64)},
      {"channel_ids", torch::tensor(implementation::resolved_channel_ids(c), torch::kInt64)},
      {"mapping_numbers", torch::tensor(std::vector<double>{c.dropout,
          c.layer_norm_epsilon, c.scale_floor, c.sampling_interval}, torch::kFloat64)}};
  std::vector<std::string> fields{
      contract::tensor_state_sha256(geometry), "float32", c.device.str(),
      "semantic-assets:0=BTC;1=ETH;2=SOL", "raw-CPU-float64;observed-bool-B,C,N,F",
      "timestamps:CPU-float64-close-time-ms;uniform-grid;explicit-cutoff-per-row",
      "oldest-to-newest;cutoff-inclusive", version::kCausalPolicy,
      implementation::kLocationScaleDecoderArchitectureId,
      implementation::kLocationScaleDecoderLayout,
      implementation::kRelativeChangeIncrementRecipe,
      implementation::kRelativeChangeBankLayout,
      "native36=learned-level32,relative-change4;no-post-encoder-PCA"};
  for (const auto &feature : cuwacunu::data::kKlineFeatureRegistry) {
    fields.push_back(std::to_string(feature.coord));
    fields.emplace_back(feature.name);
  }
  // Mask ratio, reconstruction loss and training counters do not define the
  // inference mapping and are deliberately absent from this configuration ID.
  return contract::digest_fields("Encoder inference configuration v1", fields);
}

inline contract::Component descriptor(const primary::LoadedEncoder &loaded,
    const fs::path &repo_root) {
  primary::validate_selection(loaded);
  loaded.scaler.validate(version::input_config(loaded.config));
  return contract::make_component(
      {"Encoder", "v3", source_digest(repo_root)},
      contract::module_state_sha256(*loaded.model, preprocessing_state(loaded)),
      inference_config_digest(loaded.config), {});
}

inline primary::LoadedEncoder load_selected_bytes(const std::string &bytes,
    const primary::Config &config) {
  primary::validate_config(config);
  TORCH_CHECK(contract::sha256_bytes(bytes) == primary::checkpoint_sha256,
      "[Encoder sidecar] archive differs from selected CPB-v3.market-01 bytes");
  std::istringstream stream(bytes, std::ios::in | std::ios::binary);
  torch::serialize::InputArchive archive;
  archive.load_from(stream, config.level.device);
  auto loaded = version::checkpoint_detail::read(archive, config.level.device, &config);
  primary::validate_selection(loaded);
  // Preserve the evaluated version's parameter flags: its typed state checks
  // require the original trainability declarations even in frozen inference.
  // The checked object hides every handle and guards execution with NoGrad.
  loaded.model->eval();
  return loaded;
}

class CheckedEncoder {
public:
  const primary::Config &config() const { return loaded_.config; }
  const contract::Component &component() const { return component_; }
  const std::string &sidecar_sha256() const { return sidecar_sha256_; }
  std::string state_identity() const { return loaded_.model->state_identity(); }
  std::string scaler_identity() const { return loaded_.scaler.identity(); }
  void validate_frozen() const { primary::validate_selection(loaded_); }
  primary::Encoding encode_at(const primary::RawHistory &history,
      const torch::Tensor &cutoffs) {
    torch::NoGradGuard guard;
    return primary::encode_at(loaded_, history, cutoffs);
  }

private:
  // No accessor exposes registered parameters, buffers or external scaler
  // tensors. Admission always deserializes private storage; a caller cannot
  // retain a mutable training alias under the admitted ModelRef.
  primary::LoadedEncoder loaded_;
  contract::Component component_;
  std::string sidecar_sha256_;
  CheckedEncoder(primary::LoadedEncoder loaded, contract::Component component,
      std::string sidecar_sha256)
      : loaded_(std::move(loaded)), component_(std::move(component)),
        sidecar_sha256_(std::move(sidecar_sha256)) {}
  friend CheckedEncoder load_checked(const fs::path &, const std::string &,
      const primary::Config &, const fs::path &);
};

inline CheckedEncoder load_checked(const fs::path &checkpoint,
    const std::string &expected_sidecar_sha256, const primary::Config &config,
    const fs::path &repo_root) {
  const auto verified = contract::read_verified(checkpoint,
      expected_sidecar_sha256, "inference");
  TORCH_CHECK(verified.manifest.components.size() == 1 &&
      verified.manifest.components.front().module_ref.component == "Encoder" &&
      verified.manifest.archive_sha256 == primary::checkpoint_sha256,
      "[Encoder sidecar] selected Encoder-only inference archive required");
  auto loaded = load_selected_bytes(verified.bytes, config);
  auto actual = descriptor(loaded, repo_root);
  const auto &expected = verified.manifest.components.front();
  TORCH_CHECK(actual.model_ref == expected.model_ref,
      "[Encoder sidecar] Encoder implementation, weights or inference configuration mismatch; expected ",
      expected.model_ref, "; actual ", actual.model_ref);
  contract::validate_parents({actual});
  return {std::move(loaded), std::move(actual), verified.sidecar_sha256};
}

inline CheckedEncoder load_selected_checked(
    const fs::path &settings = "config/encoder.conf",
    const fs::path &runtime = "config/encoder_runtime.conf",
    const fs::path &repo_root = ".") {
  const auto selection = primary::read_runtime(runtime);
  const auto document = cuwacunu::config::Document::read(runtime);
  const auto &block = document.block("ENCODER_RUNTIME");
  auto result = load_checked(selection.checkpoint, block.string("SIDECAR_SHA256"),
      primary::read_settings(settings), repo_root);
  TORCH_CHECK(result.component().model_ref == block.string("MODEL_REF"),
      "[Encoder sidecar] runtime Encoder ModelRef differs from checked artifact");
  return result;
}

struct Adoption {
  std::string sidecar_sha256, model_ref, archive_sha256;
};

inline Adoption adopt_selected(const fs::path &settings, const fs::path &runtime,
    const fs::path &repo_root, const std::string &expected_source_sha256) {
  TORCH_CHECK(source_digest(repo_root) == expected_source_sha256,
      "[Encoder sidecar] actual Encoder source closure differs from the trusted adoption pin");
  const auto selection = primary::read_runtime(runtime);
  const auto config = primary::read_settings(settings);
  std::ifstream stream(selection.checkpoint, std::ios::binary);
  TORCH_CHECK(stream, "[Encoder sidecar] selected archive is missing");
  const std::string bytes((std::istreambuf_iterator<char>(stream)), {});
  TORCH_CHECK(!stream.bad(), "[Encoder sidecar] selected archive read failed");
  auto loaded = load_selected_bytes(bytes, config);
  auto component = descriptor(loaded, repo_root);
  TORCH_CHECK(component.module_ref.source_sha256 == expected_source_sha256,
      "[Encoder sidecar] Encoder source changed during adoption");
  contract::Manifest manifest;
  manifest.archive_kind = "inference";
  manifest.archive_sha256 = contract::sha256_bytes(bytes);
  manifest.components = {component};
  TORCH_CHECK(contract::sha256_file(selection.checkpoint) == manifest.archive_sha256,
      "[Encoder sidecar] selected archive changed during adoption");
  const auto sidecar_sha256 = contract::publish_sidecar(selection.checkpoint, manifest);
  return {sidecar_sha256, component.model_ref, manifest.archive_sha256};
}

} // namespace cuwacunu::artifacts::selected_encoder

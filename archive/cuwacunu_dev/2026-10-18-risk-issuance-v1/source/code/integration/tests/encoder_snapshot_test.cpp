// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Additive selected-Encoder adoption and identity checks; no fits or updates.
#include "cuwacunu/artifacts/selected_encoder.h"
#include <torch/cuda.h>
#include <concepts>
#include <iostream>
#include <type_traits>
#include <unistd.h>

namespace {
namespace adapter = cuwacunu::artifacts::selected_encoder;
namespace contract = cuwacunu::artifacts::v1;
namespace primary = cuwacunu::encoder::primary;
namespace fs = std::filesystem;
int64_t checks = 0;
template<class T> concept ExposesLoadedEncoder =
    requires(T &value) { value.loaded; } || requires(T &value) { value.loaded(); };
template<class T> concept ExposesModel =
    requires(T &value) { value.model; } || requires(T &value) { value.model(); };
template<class T> concept ExposesScaler =
    requires(T &value) { value.scaler; } || requires(T &value) { value.scaler(); };
static_assert(!ExposesLoadedEncoder<adapter::CheckedEncoder>);
static_assert(!ExposesModel<adapter::CheckedEncoder>);
static_assert(!ExposesScaler<adapter::CheckedEncoder>);
static_assert(std::same_as<decltype(std::declval<adapter::CheckedEncoder>().config()),
    const primary::Config &>);
static_assert(!std::is_constructible_v<adapter::CheckedEncoder,
    primary::LoadedEncoder, contract::Component, std::string>);
void check(bool ok, const char *message) {
  ++checks;
  TORCH_CHECK(ok, "[Encoder snapshot test] ", message);
}
template<class F> void rejects(F action, const char *message) {
  bool failed = false;
  try { action(); } catch (const std::exception &) { failed = true; }
  check(failed, message);
}
struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    auto pattern = (fs::temp_directory_path() / "cuwacunu-encoder-snapshot-XXXXXX").string();
    std::vector<char> bytes(pattern.begin(), pattern.end());
    bytes.push_back('\0');
    const auto created = ::mkdtemp(bytes.data());
    TORCH_CHECK(created, "[Encoder snapshot test] private fixture directory unavailable");
    path = created;
  }
  ~TemporaryDirectory() { std::error_code error; fs::remove_all(path, error); }
};
void write(const fs::path &path, const std::string &text) {
  std::ofstream output(path, std::ios::binary);
  output << text;
  TORCH_CHECK(output, "[Encoder snapshot test] cannot write private fixture");
}
} // namespace

int main() try {
  TORCH_CHECK(torch::cuda::is_available(), "[Encoder snapshot test] managed CUDA required");
  torch::set_num_threads(1);
  const auto root = fs::current_path();
  const auto settings = root / "config/encoder.conf";
  const auto original = primary::read_runtime(root / "config/encoder_runtime.conf").checkpoint;
  const auto original_hash = contract::sha256_file(original);
  const auto source_hash = adapter::source_digest(root);
  const auto config = primary::read_settings(settings);
  TemporaryDirectory temporary;
  const auto archive = temporary.path / "inference.pt";
  const auto runtime = temporary.path / "encoder_runtime.conf";
  fs::copy_file(original, archive);
  write(runtime, "ENCODER_RUNTIME { MODEL_TAG = CPB-v3; INSTANCE = CPB-v3.market-01; "
      "CHECKPOINT = inference.pt; CHECKPOINT_SHA256 = " + original_hash + "; }\n");

  rejects([&] { adapter::adopt_selected(settings, runtime, root, std::string(64, '0')); },
      "adoption requires the actual trusted source closure");
  rejects([&] { adapter::load_checked(archive, std::string(64, '0'), config, root); },
      "missing sidecar fails closed");
  const auto adopted = adapter::adopt_selected(settings, runtime, root, source_hash);
  check(adopted.archive_sha256 == original_hash &&
      contract::sha256_file(archive) == original_hash &&
      contract::sha256_file(original) == original_hash,
      "adoption preserves original and copied checkpoint bytes");
  rejects([&] { adapter::adopt_selected(settings, runtime, root, source_hash); },
      "published sidecar cannot be overwritten");
  auto loaded = adapter::load_checked(archive, adopted.sidecar_sha256, config, root);
  check(loaded.component().model_ref == adopted.model_ref &&
      loaded.component().module_ref.component == "Encoder" && loaded.component().parents.empty(),
      "checked Encoder identity and empty parent array");
  loaded.validate_frozen();
  rejects([&] { adapter::load_checked(archive, std::string(64, '0'), config, root); },
      "wrong sidecar pin fails closed");

  write(runtime, "ENCODER_RUNTIME { MODEL_TAG = CPB-v3; INSTANCE = CPB-v3.market-01; "
      "CHECKPOINT = inference.pt; CHECKPOINT_SHA256 = " + original_hash +
      "; SIDECAR_SHA256 = " + adopted.sidecar_sha256 + "; MODEL_REF = " +
      adopted.model_ref + "; }\n");
  const auto selected = adapter::load_selected_checked(settings, runtime, root);
  check(selected.component().model_ref == adopted.model_ref,
      "runtime pins both the Encoder sidecar and ModelRef");
  write(runtime, "ENCODER_RUNTIME { MODEL_TAG = CPB-v3; INSTANCE = CPB-v3.market-01; "
      "CHECKPOINT = inference.pt; CHECKPOINT_SHA256 = " + original_hash +
      "; SIDECAR_SHA256 = " + adopted.sidecar_sha256 + "; MODEL_REF = " +
      std::string(64, '5') + "; }\n");
  rejects([&] { adapter::load_selected_checked(settings, runtime, root); },
      "a mismatched runtime ModelRef cannot expose a usable Encoder");

  const auto verified = contract::read_verified(archive, adopted.sidecar_sha256, "inference");
  int64_t malformed_count = 0;
  const auto mismatched = [&](contract::Manifest changed, const char *message) {
    const auto path = temporary.path / ("mismatch-" + std::to_string(++malformed_count) + ".pt");
    fs::copy_file(original, path);
    const auto pin = contract::publish_sidecar(path, changed);
    rejects([&] { adapter::load_checked(path, pin, config, root); }, message);
  };
  auto changed = verified.manifest;
  changed.components[0].module_ref.source_sha256 = std::string(64, '1');
  changed.components[0].model_ref = contract::derive_model_ref(changed.components[0]);
  mismatched(changed, "self-consistent sidecar cannot substitute an unsupported implementation");
  changed = verified.manifest;
  changed.components[0].state_sha256 = std::string(64, '2');
  changed.components[0].model_ref = contract::derive_model_ref(changed.components[0]);
  mismatched(changed, "loaded inference state must equal the sidecar state digest");
  changed = verified.manifest;
  changed.components[0].inference_config_sha256 = std::string(64, '3');
  changed.components[0].model_ref = contract::derive_model_ref(changed.components[0]);
  mismatched(changed, "loaded resolved configuration must equal the sidecar digest");
  changed = verified.manifest;
  changed.archive_kind = "training";
  changed.resume_identity_sha256 = std::string(64, '4');
  mismatched(changed, "training sidecar cannot be selected for inference");

  auto legacy = primary::load_checkpoint(original, config);
  auto extras = adapter::preprocessing_state(legacy);
  const auto state = contract::module_state_sha256(*legacy.model, extras);
  check(state == loaded.component().state_sha256,
      "opaque checked Encoder has the complete legacy inference state");
  extras[0].second = extras[0].second.clone() + .125;
  check(contract::module_state_sha256(*legacy.model, extras) != state,
      "external frozen input scaler values participate in WeightsRef");
  extras = adapter::preprocessing_state(legacy);
  for (auto &[name, tensor] : extras)
    if (name == "bank_normalizer.mean") tensor = tensor.clone() + .125;
  check(contract::module_state_sha256(*legacy.model, extras) != state,
      "external frozen bank values participate in WeightsRef");

  const auto copy_root = temporary.path / "source";
  for (const auto &relative : adapter::source_paths(root)) {
    fs::create_directories((copy_root / relative).parent_path());
    fs::copy_file(root / relative, copy_root / relative);
  }
  check(adapter::source_digest(copy_root) == source_hash,
      "source closure identity is independent of checkout location");
  const auto transitive = copy_root /
      "code/encoders/causal_patch_bottleneck_mae/versions/v3/include/embedding/encoders/"
      "causal_patch_bottleneck_mae/detail/v3/tokenization.h";
  { std::ofstream append(transitive, std::ios::app); append << "\n// private source mutation fixture\n"; }
  rejects([&] { adapter::load_checked(archive, adopted.sidecar_sha256, config, copy_root); },
      "changed transitive numerical source invalidates ModuleRef");

  const int64_t count = config.level.history_length;
  auto raw = torch::arange(3 * count * 9, torch::kFloat64).reshape({1, 3, count, 9}) * .001 + 100.;
  auto times = torch::arange(count, torch::kFloat64) * 60000. + 1704067259999.;
  primary::RawHistory history{raw, torch::ones_like(raw, torch::kBool),
      torch::tensor({0, 1, 2}, torch::kInt64), times, 60000.};
  auto cutoffs = times[-1].reshape({1});
  const auto before = primary::encode_at(legacy, history, cutoffs);
  const auto after = loaded.encode_at(history, cutoffs);
  check(!after.native.requires_grad(),
      "opaque checked execution uses NoGrad while preserving evaluated parameter flags");
  check(torch::equal(before.native, after.native) && torch::equal(before.valid, after.valid) &&
      torch::equal(before.cutoffs, after.cutoffs), "checked loading preserves exact inference output");

  { std::ofstream append(archive, std::ios::binary | std::ios::app); append << 'x'; }
  rejects([&] { adapter::load_checked(archive, adopted.sidecar_sha256, config, root); },
      "archive corruption fails before exposing a usable Encoder");
  check(contract::sha256_file(original) == original_hash,
      "all refusal fixtures preserve the selected original checkpoint");
  std::cout << "Encoder snapshot sidecar: " << checks
      << " checks; original archive preserved; zero training/scaler updates\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

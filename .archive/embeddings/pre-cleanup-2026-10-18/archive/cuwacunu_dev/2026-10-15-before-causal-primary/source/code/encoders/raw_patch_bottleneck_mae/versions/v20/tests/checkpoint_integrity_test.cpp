// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <unistd.h>

namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace d = rpb::configurable_checkpoint_detail;
namespace fs = std::filesystem;
struct Fixture {
  fs::path directory;
  explicit Fixture(const fs::path &parent) {
    fs::create_directories(parent);
    const auto pattern = (parent / "rpb-v20-checkpoint-integrity-XXXXXX").string();
    std::vector<char> path(pattern.begin(), pattern.end()); path.push_back('\0');
    const auto result = ::mkdtemp(path.data());
    TORCH_CHECK(result, "cannot create checkpoint test directory"); directory = result;
  }
  ~Fixture() { std::error_code error; fs::remove_all(directory, error); }
};
std::vector<char> bytes(const fs::path &path) {
  std::ifstream f(path, std::ios::binary);
  TORCH_CHECK(f.good(), "cannot read saved checkpoint");
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
template<class F> bool rejects(F action) {
  try { action(); } catch (const std::exception &) { return true; }
  return false;
}
// Build deliberately inconsistent archives independently of the production
// saver, so validating save does not hide defects in validating load.
void write_bad_archive(const fs::path &path, const rpb::Config &metadata,
                       const rpb::ConfigurableSpectralModel &weights, const rpb::FrozenScaler &scaler) {
  torch::serialize::OutputArchive a, m, s;
  a.write("artifact_kind", d::text_tensor(rpb::kConfigurableSpectralCheckpointKind), true);
  a.write("model_tag", d::text_tensor(rpb::kConfigurableSpectralModelTag), true);
  a.write("architecture_id", d::text_tensor(rpb::kConfigurableSpectralArchitectureId), true);
  a.write("native_layout", d::text_tensor(rpb::kConfigurableSpectralLayout), true);
  a.write("dtype", d::text_tensor("float32"), true);
  a.write("training_device", d::text_tensor(metadata.device.str()), true);
  a.write("config_integer_v1", torch::tensor(d::integers(metadata), torch::kInt64), true);
  a.write("config_number_v1", torch::tensor(d::numbers(metadata), torch::kFloat64), true);
  a.write("channel_ids", torch::tensor(rpb::resolved_channel_ids(metadata), torch::kInt64), true);
  a.write("training_updates", torch::tensor(int64_t{0}, torch::kInt64), true);
  a.write("scaler_identity", d::text_tensor(scaler.identity()), true);
  weights->save(m); scaler.save(s); a.write("model", m); a.write("scaler", s);
  a.save_to(path.string());
}
int main(int argc, char **argv) try {
  TORCH_CHECK(torch::cuda::is_available(), "CUDA required");
  torch::set_num_threads(1); torch::NoGradGuard guard;
  Fixture fixture(argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path());
  auto c = rpb::default_configurable_spectral_config();
  c.history_length=33; c.num_layers=1; c.encoder_width=16;
  c.feedforward_width=32; c.decoder_hidden_width=16;
  auto data = torch::randn({2, c.channel_count, c.history_length, c.input_width}, torch::kFloat64);
  rpb::Input input{data, torch::ones_like(data, torch::kBool),
      torch::tensor(rpb::resolved_channel_ids(c), torch::kInt64), {}, c.sampling_interval};
  const auto scaler = rpb::fit_configurable_spectral_scaler(input, c);
  rpb::ConfigurableSpectralModel model(c);
  const auto valid = fixture.directory / "immutable.pt";
  rpb::save_configurable_spectral_checkpoint(valid.string(), model, scaler, 1);
  const auto original = bytes(valid);
  auto failures = 0;
  const auto check = [&](bool ok, const char *message) {
    if (!ok) { ++failures; std::cerr << "FAIL " << message << '\n'; }
  };
  check(rejects([&] { rpb::save_configurable_spectral_checkpoint(valid.string(), model, scaler, 2); }),
        "immutable checkpoint destination must not be overwritten");
  check(bytes(valid) == original, "rejected overwrite must preserve original bytes");
  auto different = c; different.export_width=64;
  rpb::ConfigurableSpectralModel wrong_width(different);
  const auto malformed = fixture.directory / "wrong-shape.pt";
  write_bad_archive(malformed, c, wrong_width, scaler);
  check(rejects([&] { rpb::load_configurable_spectral_checkpoint(malformed.string(), c.device, &c); }),
        "matching header with wrong learned tensor shapes must be rejected at load");
  auto &weight = model->native_shape_projection->weight;
  const auto backup = weight.clone();
  weight[0][0] = std::numeric_limits<float>::quiet_NaN();
  const auto nonfinite = fixture.directory / "nonfinite.pt";
  write_bad_archive(nonfinite, c, model, scaler);
  check(rejects([&] { rpb::load_configurable_spectral_checkpoint(nonfinite.string(), c.device, &c); }),
        "nonfinite learned weights must be rejected at load");
  const auto bad_save = fixture.directory / "must-not-exist.pt";
  check(rejects([&] { rpb::save_configurable_spectral_checkpoint(bad_save.string(), model, scaler, 2); }),
        "nonfinite learned weights must be rejected before saving");
  check(!fs::exists(bad_save), "failed save must not publish an invalid checkpoint");
  weight.copy_(backup);
  const auto good = rpb::load_configurable_spectral_checkpoint(valid.string(), c.device, &c);
  check(good.training_updates == 1, "original checkpoint remains readable with original update count");
  if (failures) return 1;
  std::cout << "PASS checkpoint integrity: immutable atomic publication, finite model state, exact tensor geometry; storage "
            << fixture.directory.parent_path() << '\n';
  return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }

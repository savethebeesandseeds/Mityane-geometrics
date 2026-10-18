// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_settings.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_checkpoint.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <unistd.h>

namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using cuwacunu::config::Document;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action, const char *message) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  require(rejected, message);
}

// Saves intentionally refuse existing destinations. Give each invocation its
// own directory so interrupted or concurrent checks cannot poison another run.
struct CheckpointFixture {
  std::filesystem::path directory;
  CheckpointFixture() {
    const auto pattern = (std::filesystem::temp_directory_path() /
        "rpb-v20-checkpoint-test-XXXXXX").string();
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    const auto created = ::mkdtemp(path.data());
    require(created != nullptr, "cannot create private checkpoint test directory");
    directory = created;
  }
  CheckpointFixture(const CheckpointFixture &) = delete;
  CheckpointFixture &operator=(const CheckpointFixture &) = delete;
  ~CheckpointFixture() {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
};

int main(int argc, char **argv) {
  try {
    require(torch::cuda::is_available(), "actual CUDA required");
    torch::set_num_threads(1);
    auto defaults = rpb::configurable_spectral_settings(Document::parse("ENCODER {}"));
    require(defaults.input_width == 9 && defaults.history_length == 128 && defaults.export_width == 32,
            "file defaults disagree with model defaults");
    if (argc > 1) {
      const auto file = rpb::read_configurable_spectral_settings(argv[1]);
      require(rpb::configurable_checkpoint_detail::same_config(defaults, file), "default file mismatch");
    }
    for (const auto *text : {
        "ENCODER { HISTORY_LENGTH = 16; }", "ENCODER { INPUT_WIDTH = 0; }",
        "ENCODER { CHANNEL_COUNT = 4; EMBEDDING_SIZE = 24; }",
        "ENCODER { CHANNEL_IDS = 1,1,2; }", "ENCODER { DEVICE = cpu; }",
        "ENCODER { EMBEDDING_SIZE = 32x; }", "ENCODER { SCALE_FLOOR = nan; }",
        "ENCODER { HISTROY_LENGTH = 128; }", "ENCODER { MODEL_TAG = RPB-v19; }",
        "ENCODER {} APP {}"}) {
      rejects([&] { rpb::configurable_spectral_settings(Document::parse(text)); },
              "malformed/unsupported model settings accepted");
    }
    auto c = rpb::configurable_spectral_settings(Document::parse(
        "ENCODER { HISTORY_LENGTH=129; EMBEDDING_SIZE=64; CHANNEL_IDS=11,21,31; "
        "SAMPLING_INTERVAL=60; ENCODER_WIDTH=32; NUM_HEADS=4; NUM_LAYERS=2; "
        "FEEDFORWARD_WIDTH=64; DECODER_HIDDEN_WIDTH=64; MASK_RATIO=.2; }"));
    const auto raw = torch::randn({2, c.channel_count, c.history_length, c.input_width}, torch::kFloat64);
    rpb::Input input{raw, torch::ones_like(raw, torch::kBool),
        torch::tensor(c.channel_ids, torch::kInt64), torch::tensor({123., 183.}, torch::kFloat64),
        c.sampling_interval};
    auto scaler = rpb::fit_configurable_spectral_scaler(input, c);
    auto scaled = rpb::transform_configurable_spectral_input(scaler, input, c);
    auto model = rpb::ConfigurableSpectralModel(c);
    model->eval();
    torch::NoGradGuard guard;
    const auto before = model->encode(scaled).z_contextual_global.clone();
    auto fixture = std::make_unique<CheckpointFixture>();
    const auto directory = fixture->directory;
    const auto file = directory / "checkpoint.pt";
    rpb::save_configurable_spectral_checkpoint(file.string(), model, scaler, 7);
    rejects([&] { rpb::save_configurable_spectral_checkpoint(file.string(), model, scaler, 99); },
            "existing typed checkpoint must not be overwritten");
    {
      CheckpointFixture concurrent;
      require(concurrent.directory != directory, "concurrent fixture directories collide");
      const auto concurrent_file = concurrent.directory / file.filename();
      rpb::save_configurable_spectral_checkpoint(concurrent_file.string(), model, scaler, 9);
      require(rpb::load_configurable_spectral_checkpoint(concurrent_file.string(), c.device, &c)
                  .training_updates == 9,
              "same-basename concurrent checkpoint lost its own update count");
    }
    auto loaded = rpb::load_configurable_spectral_checkpoint(file.string(), c.device, &c);
    require(loaded.training_updates == 7 && loaded.scaler.identity() == scaler.identity(),
            "checkpoint lost update/scaler metadata");
    require(rpb::configurable_checkpoint_detail::same_config(c, loaded.config), "config did not round trip");
    const auto restored = rpb::transform_configurable_spectral_input(loaded.scaler, input, loaded.config);
    require(torch::equal(before, loaded.model->encode(restored).z_contextual_global),
            "checkpoint changed native output");
    auto wrong = c;
    wrong.history_length = 130; // same private paddedH, still a different logical model.
    rejects([&] { rpb::load_configurable_spectral_checkpoint(file.string(), c.device, &wrong); },
            "same-paddedH but wrong logical history accepted");
    wrong = c; wrong.channel_ids = {11,31,21};
    rejects([&] { rpb::load_configurable_spectral_checkpoint(file.string(), c.device, &wrong); },
            "different semantic channel order accepted");
    rejects([&] { rpb::load_configurable_spectral_checkpoint(file.string(), torch::kCPU); },
            "CPU model execution accepted");
    wrong = c; wrong.sampling_interval = 30;
    rejects([&] { rpb::load_configurable_spectral_checkpoint(file.string(), c.device, &wrong); },
            "sampling interval mismatch accepted");
    // Preserve the schema marker rather than accepting an arbitrary raw archive.
    torch::serialize::OutputArchive legacy;
    model->save(legacy);
    const auto raw_file = directory / "raw-module.pt";
    legacy.save_to(raw_file.string());
    rejects([&] { rpb::load_configurable_spectral_checkpoint(raw_file.string()); },
            "raw untyped module accepted as an inference checkpoint");
    fixture.reset();
    require(!std::filesystem::exists(directory), "private checkpoint test directory was not cleaned up");
    std::cout << "PASS RPB-v20 typed file settings + full config/scaler/native CUDA checkpoint round trip; "
                 "immutable saves, isolated temporary directories and cleanup\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}

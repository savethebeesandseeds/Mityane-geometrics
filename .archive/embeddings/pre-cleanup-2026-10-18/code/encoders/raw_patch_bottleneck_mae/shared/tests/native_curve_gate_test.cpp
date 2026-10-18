// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;
namespace {
std::string read(const fs::path &path) {
  std::ifstream input(path, std::ios::binary); check(bool(input), "gate artifact absent");
  std::ostringstream out; out << input.rdbuf(); return out.str();
}
void native_gate(int64_t mode) {
  auto settings = rpb::default_settings();
  settings.model.device = torch::Device(torch::kCUDA, 0);
  settings.model.channel_count = 3; settings.model.channel_ids = {101, 202, 303};
  settings.model.encoder_width = 16; settings.model.num_heads = 2; settings.model.num_layers = 1;
  settings.model.feedforward_width = 32; settings.model.decoder_hidden_width = 24;
  settings.model.export_width = 32; settings.model.channel_mixer_layers = 1;
  settings.model.global_bottleneck_mode = mode; settings.model.dropout = 0.1;
  settings.steps = 128; settings.batch_size = 2; settings.log_every = 3; settings.threads = 1;
  const auto root = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-native-gate-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(root);
  const auto path = root / "gate";
  auto invalid = settings; invalid.model.device = torch::kCPU;
  rejects([&] { rpb::run_native_curve_cuda_gate(invalid, (root / "cpu").string()); }, "gate CPU fallback");
  invalid = settings; invalid.model.global_bottleneck_mode = 0;
  rejects([&] { rpb::run_native_curve_cuda_gate(invalid, (root / "legacy").string()); }, "gate old mode");
  invalid = settings; invalid.model.channel_mixer_layers = 0;
  rejects([&] { rpb::run_native_curve_cuda_gate(invalid, (root / "independent").string()); }, "gate missing mixer");
  invalid = settings; invalid.model.export_width = 16;
  rejects([&] { rpb::run_native_curve_cuda_gate(invalid, (root / "wrong-width").string()); }, "gate nonnative32");
  check(!fs::exists(root / "cpu") && !fs::exists(root / "legacy") && !fs::exists(root / "independent")
        && !fs::exists(root / "wrong-width"), "rejected gate created outputs");
  const auto frozen_settings = rpb::settings_text(settings);
  auto cpu_generator = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  auto cuda_generator = at::globalContext().defaultGenerator(at::Device(at::kCUDA, 0));
  const auto cpu_rng = cpu_generator.get_state().clone(), cuda_rng = cuda_generator.get_state().clone();
  const int threads = at::get_num_threads();
  const auto audit = rpb::run_native_curve_cuda_gate(settings, path.string());
  check(read(path / "gpu-check.json") == audit + '\n', "returned/saved gate audit differs");
  check(rpb::settings_text(settings) == frozen_settings, "gate modified caller settings");
  check(torch::equal(cpu_rng, cpu_generator.get_state()) && torch::equal(cuda_rng, cuda_generator.get_state())
        && threads == at::get_num_threads(), "gate did not restore RNG/thread state");
  for (const auto &token : {"\"status\":\"passed\"", "\"completed\":4",
       "\"test_generation\":false", "\"main_experiment_training\":false", "\"cpu_checkpoint_export_exact\":true",
       "\"gpu_checkpoint_reconstruction_exact\":true", "\"earlier_snapshot_immutable\":true",
       "\"restore_earlier_checkpoint_exact\":true", "\"hidden_target_isolation\":true",
       "\"decoder_first\":true", "\"decoder_second\":true"})
    check(audit.find(token) != std::string::npos, "gate audit missing evidence field");
  const std::string tag = mode == 3 ? "RPB-v5" : "RPB-v4";
  const std::string group = mode == 3 ? "global_patch_pool_" : "global_pool_";
  check(audit.find("\"embedding_tag\":\"" + tag + "\"") != std::string::npos &&
        audit.find("\"" + group + "first\":true") != std::string::npos &&
        audit.find("\"" + group + "second\":true") != std::string::npos,
        "gate did not identify the actual architecture and changed pooling groups");
  const auto zero = rpb::load_checkpoint((path / "point-0.pt").string());
  const auto two = rpb::load_checkpoint((path / "point-2.pt").string());
  const auto four = rpb::load_checkpoint((path / "point-4.pt").string());
  check(zero.completed_steps == 0 && two.completed_steps == 2 && four.completed_steps == 4
        && four.settings.steps == 4 && four.settings.seed == 424243 && four.settings.model.global_bottleneck_mode == mode
        && four.settings.model.channel_mixer_layers == 1 && four.settings.model.export_width == 32,
        "ordinary gate checkpoint metadata incorrect");
  check(zero.scaler.identity() == two.scaler.identity() && two.scaler.identity() == four.scaler.identity()
        && zero.dataset_id == two.dataset_id && two.dataset_id == four.dataset_id,
        "gate scaler/training identity changed across milestones");
  torch::serialize::InputArchive witness; witness.load_from((path / "reconstruction-witness.pt").string(), torch::kCPU);
  torch::Tensor served, prediction, decoded, eligible;
  witness.read("served_global", served, true); witness.read("prediction", prediction, true);
  witness.read("decoded_from_global", decoded, true); witness.read("eligible", eligible, true);
  check(served.sizes() == torch::IntArrayRef({4, 32}) && torch::equal(prediction, decoded)
        && eligible.any().item<bool>(), "saved exact native32 decoder witness invalid");
  finite(served, "gate served global"); finite(prediction, "gate reconstruction");
  const auto saved_audit = read(path / "gpu-check.json");
  rejects([&] { rpb::run_native_curve_cuda_gate(settings, path.string()); }, "gate output overwrite");
  check(read(path / "gpu-check.json") == saved_audit, "rejected overwrite changed audit");
  for (const auto &entry : fs::recursive_directory_iterator(path))
    check(entry.path().filename().string().find("testing") == std::string::npos, "gate retained testing artifacts");
  std::cout << tag << " native CUDA gate integration passed; artifacts=" << path << '\n';
}
} // namespace
int main() {
  try { check(torch::cuda::is_available(), "native curve gate test requires CUDA"); native_gate(2); native_gate(3); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  return 0;
}

// SPDX-License-Identifier: MIT
#include "embedding/encoders/mtf_jepa_mae_vicreg/evaluation.h"
#include "test_support.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace mtf = embedding::encoders::mtf_jepa_mae_vicreg;

namespace {
std::string read(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  std::ostringstream text; text << input.rdbuf(); return text.str();
}

void test_report() {
  const auto directory = std::filesystem::temp_directory_path() /
      ("embedding-evaluation-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
  } cleanup{directory};
  auto settings = mtf::default_settings();
  settings.model = test::small_config<mtf::Config>();
  settings.batch_size = 4; settings.threads = 1;
  const auto config = directory / "config.conf", output = directory / "report.json";
  { std::ofstream stream(config); stream << mtf::settings_text(settings); }
  std::vector<std::string> args{"embedding", "evaluate", "--config", config.string(), "--output", output.string(),
      "--seeds", "17", "--train-samples", "16", "--test-samples", "8", "--steps", "2"};
  std::vector<char *> argv;
  for (auto &arg : args) argv.push_back(arg.data());
  test::check(mtf::run_evaluation_cli(static_cast<int>(argv.size()), argv.data()) == 0, "evaluation CLI failed");
  const auto first = read(output);
  test::check(first.find("\"protocol\":\"independent_synthetic_trajectory_regime_v1\"") != std::string::npos,
              "evaluation report has no protocol provenance");
  for (const auto &name : {"standardized_raw", "raw_descriptors", "mask_only", "untrained_global", "trained_global",
       "served_global", "per_channel", "projector_global", "train_fingerprint", "test_fingerprint",
       "input_normalization", "robustness", "information_retention", "mask_policy_audit_clean_test"})
    test::check(first.find(name) != std::string::npos, std::string("report omitted ") + name);
  test::check(first.find("\"accuracy\":null") != std::string::npos, "all-missing inputs received an accuracy score");
  test::check(std::filesystem::exists(output.string() + ".seed-17.pt"), "evaluation did not save a training checkpoint");
  test::check(std::filesystem::exists(output.string() + ".seed-17.train.pt"), "normalized train input was not saved");
  test::check(std::filesystem::exists(output.string() + ".seed-17.pt.normalization.pt"),
              "checkpoint normalization was not saved");
  test::check(mtf::run_evaluation_cli(static_cast<int>(argv.size()), argv.data()) == 0, "repeated evaluation failed");
  test::check(read(output) == first, "same-build CPU evaluation report is not reproducible");
  const auto checkpoint = output.string() + ".seed-17.pt";
  const auto checkpoint_before = read(checkpoint);
  const auto invoke = [](std::vector<std::string> arguments) {
    std::vector<char *> pointers;
    for (auto &arg : arguments) pointers.push_back(arg.data());
    return mtf::run_evaluation_cli(static_cast<int>(pointers.size()), pointers.data());
  };
  test::check(invoke({"embedding", "evaluate", "--output", checkpoint, "--checkpoint", checkpoint,
       "--normalization", checkpoint + ".normalization.pt"}) == 1, "evaluation could overwrite its input checkpoint");
  test::check(read(checkpoint) == checkpoint_before, "rejected evaluation modified the input checkpoint");
  test::check(invoke({"embedding", "evaluate", "--output", (directory / "rejected.json").string(),
       "--checkpoint", checkpoint, "--normalization", (directory / "missing-normalization.pt").string()}) == 1,
       "external checkpoint normalization was silently refitted");
  test::check(!std::filesystem::exists(directory / "rejected.json"), "rejected normalization wrote a report");
  test::check(invoke({"embedding", "evaluate", "--output", (directory / "invalid.json").string(),
       "--seeds", "17,17", "--steps", "1"}) == 1, "duplicate evaluation seeds were accepted");
  test::check(invoke({"embedding", "evaluate", "--help"}) == 0, "evaluation help failed");
  const auto config_before = read(config);
  test::check(invoke({"embedding", "evaluate", "--config", config.string(), "--output", config.string(),
       "--seeds", "17", "--steps", "1"}) == 1, "evaluation report could overwrite its configuration");
  test::check(read(config) == config_before, "rejected report output modified its config input");
  const auto aliased_report = (directory / "sidecar-alias.json").string();
  const auto aliased_config = aliased_report + ".seed-17.train.pt";
  { std::ofstream stream(aliased_config); stream << config_before; }
  test::check(invoke({"embedding", "evaluate", "--config", aliased_config, "--output", aliased_report,
       "--seeds", "17", "--steps", "1"}) == 1, "evaluation archive could overwrite its configuration");
  test::check(read(aliased_config) == config_before, "rejected sidecar output modified its config input");
  test::check(!std::filesystem::exists(aliased_report + ".seed-17.pt.normalization.pt"),
              "preflight rejection created output sidecars");
  const auto supplied_report = directory / "supplied.json";
  test::check(invoke({"embedding", "evaluate", "--checkpoint", checkpoint, "--output", supplied_report.string(),
       "--seeds", "19", "--train-samples", "12", "--test-samples", "8"}) == 0,
       "saved checkpoint with its own normalization could not be evaluated");
  const auto supplied_text = read(supplied_report);
  const auto scalar_identity = [](const std::string &text) {
    const std::string key = "\"normalization_fingerprint\":";
    const auto start = text.find(key);
    test::check(start != std::string::npos, "report has no normalization fingerprint");
    const auto end = text.find(',', start);
    return text.substr(start, end - start);
  };
  test::check(scalar_identity(first) == scalar_identity(supplied_text),
              "checkpoint evaluation refitted input normalization on a different seed");
  test::check(supplied_text.find("\"source\":\"saved_checkpoint_archive\"") != std::string::npos,
              "supplied checkpoint normalization provenance was omitted");
  test::check(read(checkpoint) == checkpoint_before, "checkpoint evaluation modified the input checkpoint");
  const auto malformed_scaler = directory / "zero-scale.pt";
  {
    torch::serialize::OutputArchive archive;
    archive.write("format_version", torch::tensor(int64_t{1}), true);
    archive.write("mean", torch::zeros({settings.model.input_width}, torch::kFloat64), true);
    archive.write("scale", torch::zeros({settings.model.input_width}, torch::kFloat64), true);
    archive.write("observed_counts", torch::ones({settings.model.input_width}, torch::kFloat64), true);
    archive.save_to(malformed_scaler.string());
  }
  test::check(invoke({"embedding", "evaluate", "--checkpoint", checkpoint, "--normalization", malformed_scaler.string(),
       "--output", (directory / "bad-scaler.json").string(), "--seeds", "19",
       "--train-samples", "12", "--test-samples", "8"}) == 1, "zero normalization scales were accepted");
  test::check(!std::filesystem::exists(directory / "bad-scaler.json"), "invalid normalization produced a report");
}
} // namespace

int main() {
  try {
    torch::set_num_threads(1);
    test_report();
    std::cout << "PASS: independent evaluation streams, valid-row diagnostics, deterministic heldout report\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
  }
}

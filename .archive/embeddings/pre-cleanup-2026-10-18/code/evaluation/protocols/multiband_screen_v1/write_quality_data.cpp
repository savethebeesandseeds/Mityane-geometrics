// SPDX-License-Identifier: MIT
// CPU legal-data preparation only; no encoder, information fit or fixed head.
#include "retained_input.h"
#include "two_component.h"
#include <ATen/Context.h>
#include <chrono>
#include <cstring>
#include <iostream>

#ifndef MULTIBAND_SCREEN_SOURCE_ID
#define MULTIBAND_SCREEN_SOURCE_ID "unrecorded"
#endif
#ifndef MULTIBAND_SCREEN_CARD_SHA256
#define MULTIBAND_SCREEN_CARD_SHA256 "unrecorded"
#endif
namespace {
using namespace multiband_screen;
namespace tc = embedding::evaluation::two_component;
using Clock = std::chrono::steady_clock;
constexpr const char *protocol = "multiband-screen-v1";
tc::Task generator_task(DatasetTask task) {
  return task == DatasetTask::slow_lag_sign ? tc::Task::SlowLagSign : tc::Task::ComponentBalance;
}
bool exact(const torch::Tensor &a, const torch::Tensor &b) {
  return a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() && a.is_contiguous() && b.is_contiguous() &&
      (!a.numel() || std::memcmp(a.const_data_ptr(), b.const_data_ptr(), a.numel() * a.element_size()) == 0);
}
void save_data(const fs::path &path, const tc::Split &data) {
  torch::serialize::OutputArchive a;
  a.write("observations", data.observed, true); a.write("feature_mask", data.mask, true);
  a.write("labels_scoring_only", data.scoring_labels, true); a.write("source_ids", data.source_ids, true);
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  require(fd >= 0 && ::close(fd) == 0, "exclusive data archive"); embedding::archive::save_archive(path.string(), a);
}
void confirm_data(const tc::Split &data, const ev::ControlledDataset &loaded, DatasetTask task) {
  require(exact(data.observed, loaded.observed.data) && exact(data.mask, loaded.observed.feature_mask) &&
          exact(data.scoring_labels, loaded.labels), "exact four-key serialized legal observations");
  for (int64_t row = 0; row < data.source_ids.size(0); ++row)
    require(loaded.source_ids[row] == helper_source(data.source_ids[row].item<int64_t>(), task), "exact packed serialized IDs");
}
void write_cohort(const fs::path &dir, uint64_t master, DatasetTask task, bool engineering) {
  const auto data = tc::make_development(128, 64, master, generator_task(task));
  const auto deleted = tc::delete_observations(data.validation, .30, master);
  save_data(dir / "controlled-training.pt", data.training); save_data(dir / "controlled-validation.pt", data.validation);
  save_data(dir / "controlled-validation-deleted.pt", deleted);
  const auto tr = load_controlled(dir / "controlled-training.pt", 256, master, task, engineering);
  const auto va = load_controlled(dir / "controlled-validation.pt", 128, master, task, engineering);
  const auto de = load_controlled(dir / "controlled-validation-deleted.pt", 128, master, task, engineering);
  confirm_data(data.training, tr, task); confirm_data(data.validation, va, task); confirm_data(deleted, de, task);
  check_splits(tr, va, de, master, task);
}
void engineering(const fs::path &output, bool &owned) {
  new_leaf(output); owned = true; auto &generator = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  const auto before = generator.get_state().clone();
  for (const auto task : {DatasetTask::slow_lag_sign, DatasetTask::component_balance}) {
    const auto dir = output / task_name(task); require(fs::create_directory(dir), "new artificial task directory");
    write_cohort(dir, 910901, task, true);
  }
  require(exact(before, generator.get_state().contiguous()), "CPU generator and deletion do not consume Torch RNG");
  write_new(output / "engineering.json", "{\"status\":\"passed\",\"protocol\":\"multiband-quality-data-engineering-v1\","
      "\"engineering_master\":910901,\"tasks\":[\"slow_lag_sign\",\"component_balance\"],"
      "\"quality_masters_generated\":false,\"serialized_data_archives\":6,\"encoder_calls\":0,\"head_fits\":0}\n");
  std::cout << "Multiband quality data engineering passed\n";
}
std::vector<uint64_t> parse_masters(const std::string &value) {
  require(value == "920903,920904", "exact ordered prospective quality masters"); return {920903, 920904};
}
void generate(const fs::path &output, const fs::path &card, const std::string &card_sha, const std::string &master_text, bool &owned) {
  const auto masters = parse_masters(master_text); admit_files({card});
  require(is_sha(MULTIBAND_SCREEN_SOURCE_ID) && is_sha(MULTIBAND_SCREEN_CARD_SHA256) &&
          card_sha == MULTIBAND_SCREEN_CARD_SHA256 && sha256(bytes(card)) == card_sha, "compiled prospective card pin before generation");
  new_leaf(output); owned = true; require(fs::create_directory(output / "results"), "new data results");
  std::vector<fs::path> paths;
  for (const auto master : masters) for (const auto task : {DatasetTask::slow_lag_sign, DatasetTask::component_balance}) {
    const auto dir = output / role_prefix(master, task); require(fs::create_directory(dir), "new data cohort");
    for (const std::string view : {"training", "validation", "validation-deleted"}) {
      const auto path = dir / ("controlled-" + view + ".pt"); require(!fs::exists(path), "whole absent twelve-file matrix"); paths.push_back(path);
    }
  }
  require(paths.size() == 12, "exact two masters by two tasks by three archives before generation");
  const auto started = Clock::now();
  for (const auto master : masters) for (const auto task : {DatasetTask::slow_lag_sign, DatasetTask::component_balance})
    write_cohort(output / role_prefix(master, task), master, task, false);
  admit_files(paths); std::ostringstream files; files << '[';
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto body = bytes(paths[i]); if (i) files << ',';
    files << "{\"path\":" << quote(paths[i].lexically_relative(output).generic_string())
          << ",\"bytes\":" << body.size() << ",\"sha256\":" << quote(sha256(body)) << '}';
  }
  files << ']'; require(sha256(bytes(card)) == card_sha, "card preserved after generation");
  std::ostringstream report; report << std::setprecision(17)
      << "{\"protocol\":" << quote(protocol) << ",\"data_recipe\":" << quote(data_recipe)
      << ",\"source_fingerprint\":" << quote(MULTIBAND_SCREEN_SOURCE_ID) << ",\"human_card_sha256\":" << quote(card_sha)
      << ",\"masters\":[920903,920904],\"tasks\":[\"slow_lag_sign\",\"component_balance\"],"
      << "\"datasets\":{\"slow_lag_sign\":\"TEMPO-4\",\"component_balance\":\"AMP-2\"},"
      << "\"designed_complexity_level_each\":5,\"complexity_scale_max\":5,\"shape\":[3,32,3],"
      << "\"train_pairs_each\":128,\"validation_pairs_each\":64,\"test_pairs\":0,"
      << "\"generator_calls\":4,\"archive_count\":12,\"deletion_rate\":0.3,\"deletion_stream\":\"0x74632d64656c7631\","
      << "\"encoder_calls\":0,\"optimizer_updates\":0,\"head_fits\":0,\"information_fit_calls\":0,\"TEST_generated\":false,"
      << "\"CPU_generation_validation_archive_IO_seconds\":" << std::chrono::duration<double>(Clock::now() - started).count()
      << ",\"files\":" << files.str() << "}\n";
  write_new(output / "data-report.json", report.str());
  write_new(output / "data-complete.json", "{\"status\":\"complete\",\"protocol\":" + quote(protocol) +
      ",\"source_fingerprint\":" + quote(MULTIBAND_SCREEN_SOURCE_ID) + ",\"human_card_sha256\":" + quote(card_sha) +
      ",\"masters\":[920903,920904],\"archive_count\":12,\"data_report_sha256\":" + quote(sha256(report.str())) + "}\n");
  std::cout << "Multiband quality data generation completed: twelve legal CPU archives; no model/head execution\n";
}
} // namespace
int main(int argc, char **argv) {
  fs::path output; bool owned_output = false;
  try {
    if (argc == 2 && std::string(argv[1]) == "--source-id") { std::cout << MULTIBAND_SCREEN_SOURCE_ID << '\n'; return 0; }
    if (argc == 2 && std::string(argv[1]) == "--plan") {
      std::cout << "{\"protocol\":\"multiband-screen-v1\",\"data_recipe\":\"two-component-v1\",\"masters\":[920903,920904],"
          "\"tasks\":[\"slow_lag_sign\",\"component_balance\"],\"archives\":12,\"generator_calls\":4,\"encoder_calls\":0,\"head_fits\":0}\n"; return 0;
    }
    torch::set_num_threads(1);
    if (argc == 3 && std::string(argv[1]) == "--engineering") { output = argv[2]; engineering(output, owned_output); return 0; }
    require(argc == 9, "four exact writer arguments"); std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) require(args.emplace(argv[i], argv[i + 1]).second, "unique writer arguments");
    require(args.size() == 4 && args.count("--output") && args.count("--card") && args.count("--card-sha256") &&
            args.count("--masters"), "closed writer CLI"); output = args.at("--output");
    generate(output, args.at("--card"), args.at("--card-sha256"), args.at("--masters"), owned_output); return 0;
  } catch (const std::exception &e) {
    if (owned_output && fs::is_directory(output) && !fs::exists(output / "failure.json")) {
      try { write_new(output / "failure.json", "{\"status\":\"failed\",\"error\":" + quote(e.what()) + "}\n"); } catch (...) {}
    }
    std::cerr << e.what() << '\n'; return 1;
  }
}

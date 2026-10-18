// SPDX-License-Identifier: MIT
#include "embedding/shared/saved_feature_reliability.h"
#include "frozen_role_guard.h"
#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
namespace ev = embedding::evaluation;
namespace frozen = embedding::evaluation::frozen_inputs;
namespace fs = std::filesystem;
namespace {
constexpr std::array<uint64_t, 5> masters{9109, 10210, 11311, 12412, 13513};
constexpr std::array<uint64_t, 3> reps{2701, 2802, 2903};
const std::string protocol = "saved-native-reliability-v1";
const std::string card_sha = "f93898f6dd92869e319b251eabdb466fc9ac57da13e46c53685dcd25f7b38a91";
const fs::path parent_relative = "output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b";
const std::string parent_source = "6463926ae71ee5e5547aa660d654afb02832fa843c978c41609cbd814266d78c";
const std::string parent_inventory = "a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90";
void require(bool ok, const std::string &why) { frozen::require(ok, why); }
std::vector<std::string> split(const std::string &line) {
  std::vector<std::string> out; size_t start = 0;
  while (true) { const auto end = line.find('\t', start); out.push_back(line.substr(start, end - start));
    if (end == std::string::npos) return out;
    start = end + 1;
  }
}
bool inside(const fs::path &path, const fs::path &root) {
  auto p = path.begin();
  for (auto r = root.begin(); r != root.end(); ++r, ++p) if (p == path.end() || *p != *r) return false;
  return true;
}
fs::path absolute(const std::string &value) {
  const fs::path path(value);
  require(path.is_absolute() && path.lexically_normal() == path, "absolute normalized paths required"); return path;
}
void metadata_path(const fs::path &path) {
  require(fs::is_regular_file(path) && !fs::is_symlink(path) && fs::canonical(path) == path,
          "declared metadata must be a direct regular file");
  struct stat value {};
  require(::stat(path.c_str(), &value) == 0 && S_ISREG(value.st_mode) && value.st_nlink == 1,
          "metadata hardlink/alias rejected");
}
void whole_matrix(const std::map<std::string, fs::path> &roles, const fs::path &parent) {
  require(roles.size() == 85, "closed 85 TRAIN roles required");
  std::set<std::pair<dev_t, ino_t>> identities;
  for (const auto &[name, path] : roles) {
    require(name == path.string() && inside(path, parent) && fs::is_regular_file(path) && !fs::is_symlink(path) &&
            fs::canonical(path) == path, "missing/escaping/redirected TRAIN role: " + name);
    struct stat value {};
    require(::stat(path.c_str(), &value) == 0 && S_ISREG(value.st_mode) && value.st_size > 0 && value.st_nlink == 1 &&
            identities.emplace(value.st_dev, value.st_ino).second, "hardlink/inode alias in TRAIN matrix");
  }
}
std::string plan() {
  return "{\"protocol\":\"saved-native-reliability-v1\",\"master_seeds\":[9109,10210,11311,12412,13513],"
      "\"display_tags\":[\"RPB-v4.alt-01\",\"RPB-v7.alt-01\"],\"instances\":10,\"unique_train_roles\":85,"
      "\"encoder_updates\":0,\"decoder_updates\":0,\"head_refits\":0,\"heldout_roles\":0,"
      "\"testing_accessed\":false,\"stress_accessed\":false,\"source_fingerprint\":" + frozen::quote(EVALUATION_SOURCE_ID) +
      ",\"human_card_sha256\":" + frozen::quote(card_sha) + ",\"parent_source_fingerprint\":" + frozen::quote(parent_source) +
      ",\"parent_inventory_sha256\":" + frozen::quote(parent_inventory) +
      ",\"CPU_saved_tensor_arithmetic_only\":true,\"PCA_fits\":0,\"encoder_forwards\":0,\"selection\":false,\"promotion\":false}";
}
std::vector<ev::SavedFeatureReliabilityInput> expected_inputs(const fs::path &root) {
  std::vector<ev::SavedFeatureReliabilityInput> out;
  for (auto master : masters) for (const std::string method : {"v4", "v7"}) {
    const auto base = root / parent_relative / "results" / ("seed-" + std::to_string(master) + "-lag_sign");
    const auto feature = base / "readouts" / ("native_" + method);
    ev::SavedFeatureReliabilityInput value;
    value.id = method + '-' + std::to_string(master);
    value.display_tag = method == "v4" ? "RPB-v4.alt-01" : "RPB-v7.alt-01";
    value.master_seed = master; value.controlled_training_path = (base / "controlled-training.pt").string();
    value.native_training_path = (feature / "training-features.pt").string(); value.encoder_progress_path = (base / method / "encoder-progress.json").string();
    for (size_t r = 0; r < reps.size(); ++r) {
      const auto directory = feature / ("rep-" + std::to_string(reps[r]));
      value.fit_paths[r] = (directory / "fit.pt").string(); value.training_prediction_paths[r] = (directory / "training-predictions.pt").string();
    }
    out.push_back(value);
  }
  return out;
}
std::vector<std::string> fields(const ev::SavedFeatureReliabilityInput &value) {
  return {value.id, value.display_tag, std::to_string(value.master_seed), value.controlled_training_path,
      value.native_training_path, value.encoder_progress_path, value.fit_paths[0], value.training_prediction_paths[0],
      value.fit_paths[1], value.training_prediction_paths[1], value.fit_paths[2], value.training_prediction_paths[2]};
}
std::map<std::string, fs::path> admit_tsv(const fs::path &path, const std::vector<ev::SavedFeatureReliabilityInput> &expected) {
  std::istringstream in(frozen::bytes(path)); std::string line;
  require(bool(std::getline(in, line)), "instances TSV required");
  if (!line.empty() && line.back() == '\r') line.pop_back();
  require(line == "id\tdisplay_tag\tmaster_seed\tcontrolled_training\tnative_training\tencoder_progress\tfit_2701\tpred_2701\tfit_2802\tpred_2802\tfit_2903\tpred_2903", "exact 12-column TRAIN TSV header");
  std::map<std::string, std::vector<std::string>> matrix;
  for (const auto &value : expected) matrix.emplace(value.id, fields(value));
  std::set<std::string> seen;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto row = split(line);
    require(row.size() == 12 && matrix.count(row[0]) && seen.insert(row[0]).second && matrix.at(row[0]) == row,
            "entire TSV identity/path row must equal the closed TRAIN template");
  }
  require(in.eof() && seen.size() == 10, "all ten declared instances required");
  std::map<std::string, fs::path> roles;
  for (const auto &[id, row] : matrix) for (size_t column = 3; column < row.size(); ++column)
    roles.emplace(row[column], absolute(row[column]));
  require(roles.size() == 85, "only the controlled TRAIN role may be shared"); return roles;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc == 2 && std::string(argv[1]) == "--source-id") { std::cout << EVALUATION_SOURCE_ID << '\n'; return 0; }
  if (argc == 2 && std::string(argv[1]) == "--plan") { std::cout << plan() << '\n'; return 0; }
  std::map<std::string, std::string> options;
  const std::set<std::string> names{"--instances", "--checksums", "--input-root", "--output", "--admission-log", "--admission-sha256", "--card", "--card-sha256"};
  require(argc == 17, "exact eight named measurement options required");
  for (int i = 1; i < argc; i += 2) require(names.count(argv[i]) && options.emplace(argv[i], argv[i + 1]).second, "unknown/duplicate measurement option");
  require(options.size() == names.size() && frozen::is_sha(EVALUATION_SOURCE_ID) && options.at("--card-sha256") == card_sha,
          "compiled source and frozen human card identities required");
  const auto root = absolute(options.at("--input-root")), output = absolute(options.at("--output"));
  require(fs::is_directory(root) && fs::canonical(root) == root && !fs::is_symlink(root), "direct existing input root");
  const auto parent = root / parent_relative;
  require(inside(output, root / "output/runs/rpb-saved-native-reliability") && !inside(output, parent) && !fs::exists(output),
          "exclusive additive saved-TRAIN output root required");
  const auto tsv = absolute(options.at("--instances")), checksums = absolute(options.at("--checksums")), card = absolute(options.at("--card")), log = absolute(options.at("--admission-log"));
  for (const auto &path : {tsv, checksums, card, log}) metadata_path(path);
  const auto card_bytes = frozen::bytes(card), log_bytes = frozen::bytes(log), tsv_bytes = frozen::bytes(tsv), checksum_bytes = frozen::bytes(checksums);
  require(frozen::sha256(card_bytes) == card_sha && frozen::is_sha(options.at("--admission-sha256")) &&
          frozen::sha256(log_bytes) == options.at("--admission-sha256") &&
          log_bytes.find("Saved native reliability CPU fixtures passed") != std::string::npos && log_bytes.find(EVALUATION_SOURCE_ID) != std::string::npos,
          "actual CPU fixture admission log and card must bind this compiled source before TRAIN hashes");
  const auto inputs = expected_inputs(root);
  const auto roles = admit_tsv(tsv, inputs);
  // This full inode/path pass precedes Guard's first payload hash. The Guard
  // independently admits every checksum key before beginning its byte reads.
  whole_matrix(roles, parent);
  const frozen::Guard guard(checksums, roles);
  ev::SavedFeatureReliabilityRun run; run.output_directory = output.string(); run.source_fingerprint = EVALUATION_SOURCE_ID;
  run.card_sha256 = card_sha; run.inputs = inputs;
  for (auto master : masters) run.comparisons.push_back({"v7-minus-v4-" + std::to_string(master), "v7-" + std::to_string(master), "v4-" + std::to_string(master)});
  const auto begin = std::chrono::steady_clock::now();
  ev::run_saved_feature_reliability(run);
  whole_matrix(roles, parent); guard.verify();
  require(frozen::bytes(card) == card_bytes && frozen::bytes(log) == log_bytes && frozen::bytes(tsv) == tsv_bytes && frozen::bytes(checksums) == checksum_bytes,
          "declared metadata changed during saved arithmetic");
  frozen::write_new(output / "recipe-plan.json", plan() + "\n");
  frozen::write_new(output / "input-manifest.json", guard.json() + "\n");
  frozen::write_new(output / "input-integrity-after.json", "{\"passed\":true,\"roles\":85,\"all_bytes_exact\":true,\"whole_path_inode_matrix_checked_before_hashes\":true}\n");
  frozen::write_new(output / "admission-binding.json", "{\"source_fingerprint\":" + frozen::quote(EVALUATION_SOURCE_ID) +
      ",\"human_card_sha256\":" + frozen::quote(card_sha) + ",\"admission_log\":" + frozen::quote(log.string()) +
      ",\"admission_log_sha256\":" + frozen::quote(options.at("--admission-sha256")) + ",\"instances_sha256\":" + frozen::quote(frozen::sha256(tsv_bytes)) +
      ",\"checksums_sha256\":" + frozen::quote(frozen::sha256(checksum_bytes)) + "}\n");
  std::ostringstream complete; complete << "{\"protocol\":\"saved-native-reliability-v1\",\"status\":\"complete\",\"instances\":10,\"unique_train_roles\":85,"
      "\"replayed_head_pairs\":30,\"individual_heads\":60,\"paired_source_comparisons\":15,\"TRAIN_source_groups\":640,"
      "\"encoder_updates\":0,\"decoder_updates\":0,\"head_refits\":0,\"encoder_forwards\":0,\"PCA_fits\":0,\"heldout_roles\":0,"
      "\"testing_accessed\":false,\"stress_accessed\":false,\"selection\":false,\"promotion\":false,\"CPU_saved_arithmetic_seconds\":"
      << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() << "}\n";
  frozen::write_new(output / "complete.json", complete.str());
  std::cout << "Saved native reliability complete: ten instances, 85 TRAIN roles, zero updates/refits.\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }

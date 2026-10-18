// SPDX-License-Identifier: MIT
#include "embedding/shared/archive_readout.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
#ifndef EVALUATION_GIT_HEAD
#define EVALUATION_GIT_HEAD "unrecorded"
#endif
#ifndef EVALUATION_GIT_DIRTY
#define EVALUATION_GIT_DIRTY "unrecorded"
#endif

namespace {
namespace fs = std::filesystem;
namespace ev = embedding::evaluation;

uint64_t integer(const std::string &value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("expected a nonnegative integer: " + value);
  return std::stoull(value);
}
int64_t positive(const std::string &value) {
  const auto result = integer(value);
  if (!result || result > uint64_t(INT64_MAX))
    throw std::runtime_error("expected a positive int64: " + value);
  return int64_t(result);
}
std::vector<std::string> fields(const std::string &line, char separator) {
  std::vector<std::string> result;
  size_t begin = 0;
  while (true) {
    const auto end = line.find(separator, begin);
    result.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
    if (end == std::string::npos) return result;
    begin = end + 1;
  }
}
std::string input_path(const fs::path &parent, const std::string &value) {
  if (value.empty()) throw std::runtime_error("empty archive path");
  fs::path path(value);
  return fs::absolute(path.is_absolute() ? path : parent / path).lexically_normal().string();
}
std::vector<ev::ArchiveReadoutInput> manifest(const fs::path &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read input manifest: " + path.string());
  const std::string header = "id\ttag\ttask\tmaster_seed\tcheckpoint_steps\tproducer_source_fingerprint\tcohort_provenance\ttraining_observations\tvalidation_observations\ttraining_features\tvalidation_features";
  const std::string bound_header = header + "\ttraining_observations_sha256\tvalidation_observations_sha256\ttraining_features_sha256\tvalidation_features_sha256\texpected_feature_provenance";
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty input manifest");
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line.starts_with("\xef\xbb\xbf")) line.erase(0, 3);
  const bool bound = line == bound_header;
  if (line != header && !bound) throw std::runtime_error("input manifest must have the exact archive-readout-v1 TSV header (11 fields, or 16 with identity bindings)");
  std::vector<ev::ArchiveReadoutInput> result;
  size_t number = 1;
  while (std::getline(input, line)) {
    ++number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.front() == '#') continue;
    const auto row = fields(line, '\t');
    if (row.size() != (bound ? 16 : 11)) throw std::runtime_error("wrong TSV field count at line " + std::to_string(number));
    for (const auto &field : row)
      if (field.empty()) throw std::runtime_error("empty TSV field at line " + std::to_string(number));
    ev::ArchiveReadoutInput item;
    item.id = row[0]; item.tag = row[1]; item.task = row[2];
    item.master_seed = integer(row[3]);
    const auto steps = integer(row[4]);
    if (steps > uint64_t(INT64_MAX)) throw std::runtime_error("checkpoint budget exceeds int64");
    item.checkpoint_steps = int64_t(steps);
    item.producer_source_fingerprint = row[5]; item.cohort_provenance = row[6];
    item.training_observations = input_path(path.parent_path(), row[7]);
    item.validation_observations = input_path(path.parent_path(), row[8]);
    item.training_features = input_path(path.parent_path(), row[9]);
    item.validation_features = input_path(path.parent_path(), row[10]);
    if (bound) {
      item.training_observations_sha256 = row[11];
      item.validation_observations_sha256 = row[12];
      item.training_features_sha256 = row[13];
      item.validation_features_sha256 = row[14];
      item.expected_feature_provenance = row[15];
    }
    result.push_back(std::move(item));
  }
  if (input.bad() || result.empty()) throw std::runtime_error("input manifest is unreadable or declares no cohorts");
  return result;
}
}

int main(int argc, char **argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help") {
      std::cout << "embedding_archive_readout --manifest INPUTS.tsv --output NEW_DIRECTORY\n"
        "  [--channels 3] [--history 32] [--features 3] [--width 32]\n"
        "  [--probe-seeds 2701,2802,2903] [--threads 1] [--bootstrap-replicates 1000]\n"
        "TRAIN/VALIDATION archives only; raw, standalone raw PCA, native embeddings.\n"
        "Fixed CPU heads; no encoder dependency, training or TEST input option.\n";
      return argc < 2 ? 1 : 0;
    }
    const std::set<std::string> permitted{"--manifest", "--output", "--channels", "--history",
      "--features", "--width", "--probe-seeds", "--threads", "--bootstrap-replicates"};
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; i += 2)
      if (i + 1 >= argc || !permitted.count(argv[i]) || !options.emplace(argv[i], argv[i + 1]).second)
        throw std::runtime_error("unknown/duplicate option or missing value");
    const auto get = [&](const std::string &key, const std::string &fallback) {
      const auto found = options.find(key); return found == options.end() ? fallback : found->second;
    };
    ev::ArchiveReadoutRun run;
    run.output_directory = get("--output", "");
    const auto path = fs::absolute(get("--manifest", ""));
    if (!options.count("--manifest") || run.output_directory.empty())
      throw std::runtime_error("--manifest and --output are required");
    run.shape = {positive(get("--channels", "3")), positive(get("--history", "32")),
      positive(get("--features", "3")), torch::kFloat64, torch::kCPU};
    run.compact_width = positive(get("--width", "32"));
    run.threads = positive(get("--threads", "1"));
    run.bootstrap_replicates = positive(get("--bootstrap-replicates", "1000"));
    run.repetitions.clear();
    std::set<uint64_t> used;
    for (const auto &value : fields(get("--probe-seeds", "2701,2802,2903"), ',')) {
      const auto seed = integer(value);
      if (!used.insert(seed).second) throw std::runtime_error("duplicate probe seed");
      run.repetitions.push_back({"rep-" + std::to_string(run.repetitions.size() + 1), seed});
    }
    run.source_fingerprint = EVALUATION_SOURCE_ID;
    run.git_head = EVALUATION_GIT_HEAD; run.git_dirty = EVALUATION_GIT_DIRTY;
    run.inputs = manifest(path);
    ev::run_archive_readout(run);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}

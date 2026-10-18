// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// New application adapter; inherited fixed-head arithmetic remains in deps/.
#include "embedding/shared/fixed_feature_readouts.h"

#include <array>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using embedding::evaluation::FeatureSurface;
using embedding::evaluation::FixedFeatureMethod;
using embedding::evaluation::FixedFeatureReadoutRun;
constexpr std::array<uint64_t, 3> masters{941101, 941102, 941103};
constexpr std::array<const char *, 2> modes{"synthetic_train", "market_scaler"};
constexpr int64_t training_rows = 512, validation_rows = 256, native_width = 40;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error("[market synthetic transfer reader] " + message);
}

std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                         << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}

torch::Tensor tensor(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::Tensor result;
  archive.read(key, result, true);
  require(result.defined() && result.device().is_cpu() && !result.requires_grad(),
          key + " must be a detached CPU tensor");
  return result;
}

// The frozen generator uses ASCII source identifiers. Reject unsupported JSON
// instead of silently changing the source grouping used by the fixed heads.
std::vector<std::string> source_ids(const std::string &json) {
  size_t cursor = 0;
  const auto space = [&]() {
    while (cursor < json.size() && std::isspace(static_cast<unsigned char>(json[cursor]))) ++cursor;
  };
  const auto take = [&](char expected) {
    space();
    require(cursor < json.size() && json[cursor] == expected, "invalid source ID JSON");
    ++cursor;
  };
  take('[');
  space();
  std::vector<std::string> result;
  if (cursor < json.size() && json[cursor] != ']') {
    for (;;) {
      take('"');
      std::string value;
      bool closed = false;
      while (cursor < json.size()) {
        unsigned char c = json[cursor++];
        if (c == '"') { closed = true; break; }
        require(c >= 32 && c < 127, "source ID must contain ASCII printables");
        if (c == '\\') {
          require(cursor < json.size(), "truncated source ID JSON escape");
          c = json[cursor++];
          require(c == '"' || c == '\\' || c == '/', "unsupported source ID JSON escape");
        }
        value.push_back(static_cast<char>(c));
      }
      require(closed && !value.empty(), "empty or unterminated source ID");
      result.push_back(std::move(value));
      space();
      require(cursor < json.size(), "truncated source ID JSON array");
      if (json[cursor] == ']') break;
      take(',');
    }
  }
  take(']');
  space();
  require(cursor == json.size(), "trailing source ID JSON content");
  return result;
}

void validate_labels(const torch::Tensor &labels, int64_t rows, const std::string &name) {
  require(labels.scalar_type() == torch::kInt64 && labels.dim() == 1 && labels.size(0) == rows &&
          labels.ge(0).logical_and(labels.le(1)).all().item<bool>(), name + " must be binary int64[N]");
}

void validate_sources(const std::vector<std::string> &ids, const torch::Tensor &labels) {
  require(ids.size() == static_cast<size_t>(labels.size(0)), "source and label row association mismatch");
  std::map<std::string, std::vector<int64_t>> groups;
  for (int64_t row = 0; row < labels.size(0); ++row) groups[ids.at(row)].push_back(labels[row].item<int64_t>());
  for (const auto &[id, classes] : groups)
    require(!id.empty() && classes.size() == 2 && classes[0] != classes[1],
            "every source must have exactly two opposite-label variants");
}

struct Loaded {
  FixedFeatureMethod method;
  torch::Tensor training_labels, validation_labels;
  std::vector<std::string> training_source_ids, validation_source_ids;
  fs::path input;
};

Loaded load(const fs::path &path, const std::string &mode) {
  require(fs::is_regular_file(path), "missing saved feature archive: " + path.string());
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), torch::kCPU);
  Loaded result;
  result.input = fs::weakly_canonical(path);
  result.method.name = mode;
  const auto read_surface = [&](const std::string &role, int64_t rows) {
    FeatureSurface surface{tensor(archive, role + "_values"), tensor(archive, role + "_valid"),
        "CPB-v8.market-01; newly measured fresh initial/final checkpoint; TEMPO-5; input_mode=" + mode +
        "; role=" + role + "; saved_feature_archive=" + result.input.string()};
    embedding::evaluation::validate_features(surface);
    require(surface.values.scalar_type() == torch::kFloat64 && surface.values.size(0) == rows &&
            surface.values.size(1) == native_width, "expected float64 saved native40 feature rows for " + role);
    return surface;
  };
  result.method.training = read_surface("training", training_rows);
  result.method.validation_intact = read_surface("validation", validation_rows);
  result.method.validation_deleted = read_surface("deleted", validation_rows);
  result.training_labels = tensor(archive, "training_labels");
  result.validation_labels = tensor(archive, "validation_labels");
  const auto deleted_labels = tensor(archive, "deleted_labels");
  validate_labels(result.training_labels, training_rows, "training_labels");
  validate_labels(result.validation_labels, validation_rows, "validation_labels");
  validate_labels(deleted_labels, validation_rows, "deleted_labels");
  require(torch::equal(result.validation_labels, deleted_labels), "deleted and intact labels differ");
  result.training_source_ids = source_ids(embedding::archive::tensor_text(tensor(archive, "training_source_ids_json")));
  result.validation_source_ids = source_ids(embedding::archive::tensor_text(tensor(archive, "validation_source_ids_json")));
  const auto deleted_ids = source_ids(embedding::archive::tensor_text(tensor(archive, "deleted_source_ids_json")));
  require(result.validation_source_ids == deleted_ids, "deleted and intact sources differ");
  validate_sources(result.training_source_ids, result.training_labels);
  validate_sources(result.validation_source_ids, result.validation_labels);
  const std::set<std::string> training_set(result.training_source_ids.begin(), result.training_source_ids.end());
  for (const auto &id : result.validation_source_ids)
    require(!training_set.count(id), "TRAIN/VALIDATION source overlap");
  return result;
}

void write_new(const fs::path &path, const std::string &contents) {
  require(!fs::exists(path), "output file already exists: " + path.string());
  std::ofstream stream(path);
  require(bool(stream), "cannot create output: " + path.string());
  stream << contents << '\n';
  stream.close();
  require(bool(stream), "cannot save output: " + path.string());
}
} // namespace

int main(int argc, char **argv) {
  try {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    fs::path features, output;
    std::string stage;
    for (int argument = 1; argument < argc; ++argument) {
      const std::string option(argv[argument]);
      require(argument + 1 < argc, "usage: causal_fresh_v8_synthetic_read --features ROOT --output NEWLEAF --stage initial|final");
      if (option == "--features") { require(features.empty(), "duplicate --features"); features = argv[++argument]; }
      else if (option == "--output") { require(output.empty(), "duplicate --output"); output = argv[++argument]; }
      else if (option == "--stage") { require(stage.empty(), "duplicate --stage"); stage = argv[++argument]; }
      else throw std::runtime_error("unknown option: " + option);
    }
    require(!features.empty() && fs::is_directory(features) && !output.empty() && !fs::exists(output),
            "existing input directory and new exclusive output leaf are required");
    require(stage == "initial" || stage == "final", "declared checkpoint stage required");
    features = fs::weakly_canonical(features);
    output = fs::weakly_canonical(output);
    require(features != output, "input and output paths must differ");

    // Validate every saved archive before fitting any head. This executable
    // links no encoder implementation and never opens a model checkpoint.
    std::vector<std::array<Loaded, 2>> cohorts;
    for (const auto master : masters) {
      const auto name = "master-" + std::to_string(master);
      std::array<Loaded, 2> cohort{load(features / modes[0] / name / "features.pt", modes[0]),
                                 load(features / modes[1] / name / "features.pt", modes[1])};
      require(torch::equal(cohort[0].training_labels, cohort[1].training_labels) &&
              torch::equal(cohort[0].validation_labels, cohort[1].validation_labels) &&
              cohort[0].training_source_ids == cohort[1].training_source_ids &&
              cohort[0].validation_source_ids == cohort[1].validation_source_ids,
              "input preprocessing modes must retain identical labels and source associations");
      cohorts.push_back(std::move(cohort));
    }
    require(fs::create_directories(output), "cannot create exclusive output leaf");
    std::ostringstream summary;
    summary << std::setprecision(17)
        << "{\"protocol\":\"causal-fresh-v8\",\"stage\":\"frozen_saved_feature_readouts\","
        << "\"model\":\"CPB-v8.market-01\",\"checkpoint_stage\":" << quote(stage) << ",\"encoder_fitting_dataset_id\":\"MKT-HIST-2\","
        << "\"head_fitting_dataset_id\":\"TEMPO-5\",\"scoring_dataset_id\":\"TEMPO-5\","
        << "\"task\":\"lag_sign\",\"recipe_id\":\"frozen-cpb-v8-native40-on-tempo5-v1\","
        << "\"designed_complexity_level\":5,\"complexity_scale_max\":5,\"native_width\":40,"
        << "\"training_rows_per_cohort\":512,\"validation_rows_per_cohort\":256,"
        << "\"training_sources_per_cohort\":256,\"validation_sources_per_cohort\":128,"
        << "\"encoder_calls\":0,\"encoder_updates\":0,\"old_encoder_calls\":0,\"old_head_fits\":0,"
        << "\"post_encoder_pca_fits\":0,\"device\":\"cpu\",\"primary_input_mode\":\"synthetic_train\","
        << "\"secondary_input_mode\":\"market_scaler\",\"cohorts\":[";
    const auto all_started = std::chrono::steady_clock::now();
    for (size_t index = 0; index < masters.size(); ++index) {
      auto &cohort = cohorts[index];
      FixedFeatureReadoutRun run;
      run.master_seed = masters[index];
      const auto relative = "master-" + std::to_string(run.master_seed);
      run.output_directory = (output / relative).string();
      run.training_labels = cohort[0].training_labels;
      run.validation_labels = cohort[0].validation_labels;
      run.training_source_ids = cohort[0].training_source_ids;
      run.validation_source_ids = cohort[0].validation_source_ids;
      run.methods = {std::move(cohort[0].method), std::move(cohort[1].method)};
      const auto started = std::chrono::steady_clock::now();
      embedding::evaluation::run_fixed_feature_readouts(run);
      const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      if (index) summary << ',';
      summary << "{\"master_seed\":" << run.master_seed << ",\"report\":" << quote(relative + "/report.json")
              << ",\"head_fitting_and_scoring_seconds_including_io\":" << seconds << '}';
      std::cout << "saved fixed-head report " << relative << "/report.json\n";
    }
    summary << "],\"head_fitting_and_scoring_seconds_including_io\":"
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - all_started).count() << '}';
    write_new(output / "reader-summary.json", summary.str());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

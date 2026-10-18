// SPDX-License-Identifier: MIT
#include "reconstruction_cli.h"
#include <iostream>

#ifndef EMBEDDING_WITH_RPB
#define EMBEDDING_WITH_RPB 0
#endif

#if EMBEDDING_WITH_RPB
#include "embedding/encoders/raw_patch_bottleneck_mae/reconstruction_adapter.h"
#include <charconv>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
namespace ev = embedding::evaluation;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error("[embedding reconstruction] " + message);
}

std::vector<std::string> csv(const std::string &text, const std::string &option) {
  require(!text.empty() && text.back() != ',', option + " has an empty comma-separated field");
  std::vector<std::string> fields;
  size_t begin = 0;
  while (begin < text.size()) {
    const auto end = text.find(',', begin);
    auto field = text.substr(begin, end == std::string::npos ? end : end - begin);
    require(!field.empty(), option + " has an empty comma-separated field");
    fields.push_back(std::move(field));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return fields;
}

int64_t integer(const std::string &text, const std::string &option, bool signed_value = false) {
  const auto digits = signed_value && !text.empty() && text.front() == '-' ? size_t{1} : size_t{0};
  require(digits < text.size() && text.find_first_not_of("0123456789", digits) == std::string::npos,
          option + (signed_value ? " requires an int64" : " requires a nonnegative int64"));
  int64_t value{0};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(result.ec == std::errc{} && result.ptr == text.data() + text.size(),
          option + " is outside the int64 range");
  return value;
}

uint64_t seed(const std::string &text) {
  require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
          "--seeds requires nonnegative uint64 values");
  uint64_t value{0};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(result.ec == std::errc{} && result.ptr == text.data() + text.size(),
          "--seeds value is outside the uint64 range");
  return value;
}

void usage(std::ostream &out) {
  out <<
      "embedding_evaluate reconstruct --output NEW_DIRECTORY\n"
      "  [--seeds 101,202,303] [--tasks reversal,level,amplitude,lag_sign]\n"
      "  [--channels 3] [--history 32] [--features 3] [--channel-ids 0,1,2]\n"
      "  [--units unitless,unitless,unitless] [--patch-length 8]\n"
      "  [--train-pairs 32] [--validation-pairs 16] [--test-pairs 64]\n"
      "  [--threads 1] [--bootstrap-replicates 1000]\n"
      "  [--rpb-config FILE] [--rpb-steps 128] [--metadata-steps RPB_STEPS]\n"
      "  [--batch-size 8] [--card-id controlled-reconstruction-v1]\n"
      "Fresh-only training; card geometry is fixed before adapter configuration.\n"
      "Fully observed histories; enumerate one whole hidden patch per channel.\n"
      "Primary standardized MAE, secondary standardized Huber; disjoint source\n"
      "swaps and whole exchange-block bootstrap. Development evidence only.\n";
}
} // namespace
#endif

int run_reconstruction_cli(int argc, char **argv) {
#if !EMBEDDING_WITH_RPB
  (void)argc; (void)argv;
  std::cerr << "[embedding reconstruction] this minimum executable does not register RPB reconstruction; "
               "build the evaluation target and use embedding_evaluate reconstruct\n";
  return 1;
#else
  try {
    require(argc > 1 && std::string(argv[1]) == "reconstruct", "expected the reconstruct command");
    if (argc == 2) { usage(std::cerr); return 1; }
    if (std::string(argv[2]) == "--help") {
      require(argc == 3, "--help must be the only reconstruction option");
      usage(std::cout); return 0;
    }
    const std::set<std::string> permitted{
        "--output", "--seeds", "--tasks", "--channels", "--history", "--features",
        "--channel-ids", "--units", "--patch-length", "--train-pairs", "--validation-pairs",
        "--test-pairs", "--threads", "--bootstrap-replicates", "--rpb-config", "--rpb-steps",
        "--metadata-steps", "--batch-size", "--card-id"};
    std::map<std::string, std::string> options;
    for (int i = 2; i < argc; i += 2) {
      const std::string key = argv[i];
      require(permitted.count(key), "unknown option: " + key);
      require(i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0,
              "missing value for " + key);
      require(options.emplace(key, argv[i + 1]).second, "duplicate option: " + key);
    }
    const auto get = [&](const std::string &key, const std::string &fallback) {
      const auto found = options.find(key);
      return found == options.end() ? fallback : found->second;
    };
    const auto number = [&](const std::string &key, int64_t fallback) {
      return integer(get(key, std::to_string(fallback)), key);
    };

    ev::ReconstructionRun run;
    auto &card = run.card;
    run.output_directory = get("--output", "");
    require(!run.output_directory.empty(), "--output is required");
    run.source_fingerprint = EVALUATION_SOURCE_ID;
    run.git_head = EVALUATION_GIT_HEAD;
    run.git_dirty = EVALUATION_GIT_DIRTY;
    card.id = get("--card-id", card.id);
    require(!card.id.empty(), "--card-id must not be empty");
    card.shape.channel_count = number("--channels", card.shape.channel_count);
    card.shape.history_length = number("--history", card.shape.history_length);
    card.shape.input_width = number("--features", card.shape.input_width);
    require(card.shape.channel_count > 0 && card.shape.history_length > 0 && card.shape.input_width > 0,
            "--channels, --history and --features must be positive");
    card.patch_length = number("--patch-length", card.patch_length);
    card.train_pairs = number("--train-pairs", card.train_pairs);
    card.validation_pairs = number("--validation-pairs", card.validation_pairs);
    card.test_pairs = number("--test-pairs", card.test_pairs);
    card.threads = number("--threads", card.threads);
    card.bootstrap_replicates = number("--bootstrap-replicates", card.bootstrap_replicates);
    const auto maximum = std::numeric_limits<int64_t>::max();
    require(card.shape.channel_count <= maximum / card.shape.history_length &&
                card.shape.channel_count * card.shape.history_length <= maximum / card.shape.input_width,
            "C*H*F exceeds the tensor index range");
    const auto row_cells = card.shape.channel_count * card.shape.history_length * card.shape.input_width;
    for (const auto pairs : {card.train_pairs, card.validation_pairs, card.test_pairs})
      require(pairs > 0 && pairs <= maximum / 2 && pairs * 2 <= maximum / row_cells,
              "split pair count must be positive and fit the tensor index range");
    require(card.train_pairs <= maximum - card.validation_pairs &&
                card.train_pairs + card.validation_pairs <= maximum - card.test_pairs,
            "total source count exceeds the index range");
    if (options.count("--channel-ids")) {
      card.channel_ids.clear();
      for (const auto &id : csv(options.at("--channel-ids"), "--channel-ids"))
        card.channel_ids.push_back(integer(id, "--channel-ids", true));
    } else {
      require(static_cast<uint64_t>(card.shape.channel_count) <= card.channel_ids.max_size(),
              "channel count exceeds the default ID array size range");
      card.channel_ids.resize(static_cast<size_t>(card.shape.channel_count));
      std::iota(card.channel_ids.begin(), card.channel_ids.end(), int64_t{0});
    }
    if (options.count("--units")) {
      card.feature_units = options.at("--units");
    } else {
      card.feature_units.clear();
      // Each coordinate contributes eight letters plus its comma, except the first.
      require(static_cast<uint64_t>(card.shape.input_width) <= card.feature_units.max_size() / 9,
              "feature count exceeds the default unit string size range");
      card.feature_units.reserve(static_cast<size_t>(card.shape.input_width) * 9 - 1);
      for (int64_t f = 0; f < card.shape.input_width; ++f) {
        if (f) card.feature_units += ',';
        card.feature_units += "unitless";
      }
    }
    card.seeds.clear();
    for (const auto &value : csv(get("--seeds", "101,202,303"), "--seeds"))
      card.seeds.push_back(seed(value));
    card.tasks.clear();
    for (const auto &task : csv(get("--tasks", "reversal,level,amplitude,lag_sign"), "--tasks")) {
      if (task == "reversal") card.tasks.push_back(ev::Task::reversal);
      else if (task == "level") card.tasks.push_back(ev::Task::level);
      else if (task == "amplitude") card.tasks.push_back(ev::Task::amplitude);
      else if (task == "lag_sign") card.tasks.push_back(ev::Task::lag_sign);
      else throw std::runtime_error("[embedding reconstruction] unknown task: " + task);
    }

    rpb::ReconstructionOptions training;
    training.config_path = get("--rpb-config", "");
    if (options.count("--rpb-config")) require(!training.config_path.empty(), "--rpb-config must not be empty");
    training.pretraining_updates = number("--rpb-steps", training.pretraining_updates);
    // Unless explicitly overridden, the control receives the same update budget.
    training.metadata_updates = number("--metadata-steps", training.pretraining_updates);
    training.batch_size = number("--batch-size", training.batch_size);
    require(training.batch_size > 0, "--batch-size must be positive");
    const auto registration = rpb::make_reconstruction_provider(training, card);
    card.provider_recipe = registration.recipe;
    ev::validate_reconstruction_card(card);
    ev::run_reconstruction_evaluation(run, registration.factory);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
#endif
}

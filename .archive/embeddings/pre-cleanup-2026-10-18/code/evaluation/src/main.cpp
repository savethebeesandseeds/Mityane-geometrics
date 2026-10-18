// SPDX-License-Identifier: MIT
#include "embedding/shared/feature_evaluation.h"
#include "reconstruction_cli.h"
#include "embedding/encoders/mtf_jepa_mae_vicreg/evaluation_adapter.h"
#ifndef EMBEDDING_WITH_RPB
#define EMBEDDING_WITH_RPB 0
#endif
#if EMBEDDING_WITH_RPB
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#endif
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
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
namespace ev = embedding::evaluation;
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error("[embedding evaluation] " + message);
}
std::vector<std::string> csv(const std::string &text) {
  require(!text.empty() && text.back() != ',', "empty comma-separated field");
  std::vector<std::string> out;
  size_t begin = 0;
  while (begin < text.size()) {
    const auto end = text.find(',', begin);
    auto field = text.substr(begin, end == std::string::npos ? end : end - begin);
    require(!field.empty(), "empty comma-separated field");
    out.push_back(std::move(field));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return out;
}
int64_t integer(const std::string &text) {
  require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
          "expected a nonnegative integer");
  return std::stoll(text);
}
ev::DimensionTier tier(const std::string &value) {
  if (value == "native") return ev::DimensionTier::native;
  if (value == "matched_global") return ev::DimensionTier::matched_global;
  if (value == "matched_channels") return ev::DimensionTier::matched_channels;
  throw std::runtime_error("unknown dimension tier: " + value);
}
void usage() {
  std::cout <<
    "embedding_evaluate [evaluate] --output NEW_DIRECTORY [--encoders baseline,rpb,rpb_mixer]\n"
    "embedding_evaluate reconstruct --help (held-out compact decoder diagnostics)\n"
    "  [--seeds 101,202,303] [--tasks reversal,level,amplitude,lag_sign]\n"
    "  [--channels 3] [--history 32] [--features 3] [--channel-ids 0,1,2]\n"
    "  [--units unitless,unitless,unitless] [--matched-global-width 12]\n"
    "  [--matched-channel-width 36] [--train-pairs 32]\n"
    "  [--validation-pairs 16] [--test-pairs 32] [--threads 1]\n"
    "  [--baseline-config FILE] [--baseline-steps 0]\n"
    "  [--rpb-config FILE] [--rpb-steps 0 | --rpb-checkpoint FILE]\n"
    "  [--rpb-mixer-config FILE] [--rpb-mixer-steps 0 | --rpb-mixer-checkpoint FILE]\n"
    "  [--compare LEFT,RIGHT,native|matched_global|matched_channels] (repeatable)\n"
    "  [--card-id controlled-pairs-v2]\n"
    "  [--stress-sweep none|fixed-readout-v1] (frozen testing-only missingness)\n"
    "The card owns geometry and comparison widths. Encoder configs must agree.\n"
    "Development only: fixed ridge and separate nonlinear probe; no acceptance.\n";
#if !EMBEDDING_WITH_RPB
  std::cout << "This minimum executable registers baseline only; build evaluation for RPB.\n";
#endif
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "reconstruct")
      return run_reconstruction_cli(argc, argv);
    const int begin = argc > 1 && std::string(argv[1]) == "evaluate" ? 2 : 1;
    if (argc <= begin || std::string(argv[begin]) == "--help") {
      usage();
      return argc <= begin ? 1 : 0;
    }
    const std::set<std::string> permitted{
      "--output", "--encoders", "--seeds", "--tasks", "--channels", "--history",
      "--features", "--channel-ids", "--units", "--matched-global-width",
      "--matched-channel-width", "--train-pairs", "--validation-pairs",
      "--test-pairs", "--threads", "--baseline-config", "--baseline-steps",
      "--rpb-config", "--rpb-steps", "--rpb-checkpoint", "--rpb-mixer-config",
      "--rpb-mixer-steps", "--rpb-mixer-checkpoint", "--compare", "--card-id",
      "--stress-sweep"};
    std::map<std::string, std::string> options;
    std::vector<std::string> explicit_pairs;
    for (int i = begin; i < argc; i += 2) {
      const std::string key = argv[i];
      require(i + 1 < argc && permitted.count(key), "unknown option or missing value: " + key);
      if (key == "--compare") explicit_pairs.emplace_back(argv[i + 1]);
      else require(options.emplace(key, argv[i + 1]).second, "duplicate option: " + key);
    }
    auto get = [&](const std::string &key, const std::string &fallback) {
      const auto found = options.find(key);
      return found == options.end() ? fallback : found->second;
    };
    const auto names = csv(get("--encoders", EMBEDDING_WITH_RPB ? "baseline,rpb" : "baseline"));
    std::set<std::string> selected;
    for (const auto &name : names) {
      require(name == "baseline" || (EMBEDDING_WITH_RPB && (name == "rpb" || name == "rpb_mixer")),
              "unregistered encoder: " + name);
      require(selected.insert(name).second, "duplicate encoder: " + name);
    }
    for (const auto &key : {"--baseline-config", "--baseline-steps"})
      require(!options.count(key) || selected.count("baseline"), "baseline options require baseline selection");
    for (const auto &key : {"--rpb-config", "--rpb-steps", "--rpb-checkpoint"})
      require(!options.count(key) || selected.count("rpb"), "RPB options require RPB selection");
    for (const auto &key : {"--rpb-mixer-config", "--rpb-mixer-steps", "--rpb-mixer-checkpoint"})
      require(!options.count(key) || selected.count("rpb_mixer"), "mixer options require rpb_mixer selection");

    ev::EvaluationRun run;
    auto &card = run.card;
    run.output_directory = get("--output", "");
    require(!run.output_directory.empty(), "--output is required");
    run.source_fingerprint = EVALUATION_SOURCE_ID;
    run.git_head = EVALUATION_GIT_HEAD;
    run.git_dirty = EVALUATION_GIT_DIRTY;
    const auto stress_sweep = get("--stress-sweep", "none");
    require(stress_sweep == "none" || stress_sweep == "fixed-readout-v1",
            "--stress-sweep must be none or fixed-readout-v1");
    run.stress_sweep = stress_sweep == "fixed-readout-v1";
    card.id = get("--card-id", card.id);
    card.shape.channel_count = integer(get("--channels", "3"));
    card.shape.history_length = integer(get("--history", "32"));
    card.shape.input_width = integer(get("--features", "3"));
    card.channel_ids.resize(static_cast<size_t>(card.shape.channel_count));
    std::iota(card.channel_ids.begin(), card.channel_ids.end(), int64_t{0});
    if (options.count("--channel-ids")) {
      card.channel_ids.clear();
      for (const auto &id : csv(options.at("--channel-ids"))) card.channel_ids.push_back(integer(id));
    }
    card.feature_units.clear();
    for (int64_t f = 0; f < card.shape.input_width; ++f) {
      if (f) card.feature_units += ',';
      card.feature_units += "unitless";
    }
    card.feature_units = get("--units", card.feature_units);
    card.matched_global_width = integer(get("--matched-global-width", "12"));
    card.matched_channel_width = integer(get("--matched-channel-width", "36"));
    card.train_pairs = integer(get("--train-pairs", "32"));
    card.validation_pairs = integer(get("--validation-pairs", "16"));
    card.test_pairs = integer(get("--test-pairs", "32"));
    card.threads = integer(get("--threads", "1"));
    card.seeds.clear();
    for (const auto &seed : csv(get("--seeds", "101,202,303"))) {
      require(seed.find_first_not_of("0123456789") == std::string::npos,
              "expected nonnegative seeds");
      card.seeds.push_back(std::stoull(seed));
    }
    card.tasks.clear();
    for (const auto &task : csv(get("--tasks", "reversal,level,amplitude,lag_sign"))) {
      if (task == "reversal") card.tasks.push_back(ev::Task::reversal);
      else if (task == "level") card.tasks.push_back(ev::Task::level);
      else if (task == "amplitude") card.tasks.push_back(ev::Task::amplitude);
      else if (task == "lag_sign") card.tasks.push_back(ev::Task::lag_sign);
      else throw std::runtime_error("unknown task: " + task);
    }

    const auto baseline_steps = integer(get("--baseline-steps", "0"));
    const auto rpb_steps = integer(get("--rpb-steps", "0"));
    const bool trained_rpb = rpb_steps > 0 || options.count("--rpb-checkpoint");
    const auto mixer_steps = integer(get("--rpb-mixer-steps", "0"));
    const bool trained_mixer = mixer_steps > 0 || options.count("--rpb-mixer-checkpoint");
    if (!explicit_pairs.empty()) {
      for (const auto &pair : explicit_pairs) {
        const auto fields = csv(pair);
        require(fields.size() == 3, "--compare requires LEFT,RIGHT,TIER");
        card.comparisons.push_back({"pair-" + std::to_string(card.comparisons.size() + 1),
                                   fields[0], fields[1], tier(fields[2])});
      }
    } else {
      if (selected.count("rpb_mixer")) {
        const std::string left = trained_mixer ? "rpb_mixer_trained" : "rpb_mixer_untrained";
        card.comparisons.push_back({"rpb_mixer_reference_global", left + "_contextual_global",
            trained_mixer ? "rpb_mixer_untrained_contextual_global" : "raw",
            ev::DimensionTier::matched_global});
      }
      if (selected.count("rpb") && trained_rpb)
        card.comparisons.push_back({"rpb_training_global", "rpb_trained_global",
                                   "rpb_untrained_global", ev::DimensionTier::matched_global});
      if (selected.count("rpb") && selected.count("baseline")) {
        const std::string left = trained_rpb ? "rpb_trained" : "rpb_untrained";
        const std::string right = baseline_steps > 0 ? "fresh_frozen_baseline" : "untrained_baseline";
        card.comparisons.push_back({"rpb_vs_baseline_global", left + "_global",
                                   right + "_global", ev::DimensionTier::matched_global});
        card.comparisons.push_back({"rpb_vs_baseline_channel_concatenation",
          left + "_channel_concatenation", right + "_channel_concatenation",
          ev::DimensionTier::matched_channels});
      } else if (selected.count("baseline")) {
        card.comparisons.push_back({"baseline_reference_global",
          baseline_steps > 0 ? "fresh_frozen_baseline_global" : "untrained_baseline_global",
          baseline_steps > 0 ? "untrained_baseline_global" : "raw",
          ev::DimensionTier::matched_global});
      } else if (selected.count("rpb") && !trained_rpb) {
        card.comparisons.push_back({"rpb_reference_global", "rpb_untrained_global",
                                   "raw", ev::DimensionTier::matched_global});
      }
    }
    ev::validate_evaluation_card(card);
    std::vector<ev::FeatureProviderFactory> factories;
    if (selected.count("baseline"))
      factories.push_back(embedding::encoders::mtf_jepa_mae_vicreg::make_evaluation_provider(
          {get("--baseline-config", ""), baseline_steps}));
#if EMBEDDING_WITH_RPB
    if (selected.count("rpb"))
      factories.push_back(embedding::encoders::raw_patch_bottleneck_mae::make_evaluation_provider(
          {get("--rpb-config", ""), get("--rpb-checkpoint", ""), rpb_steps}));
    if (selected.count("rpb_mixer"))
      factories.push_back(embedding::encoders::raw_patch_bottleneck_mae::make_evaluation_provider(
          {get("--rpb-mixer-config", options.count("--rpb-mixer-checkpoint") ? "" :
               "code/encoders/raw_patch_bottleneck_mae/config/channel_mixer.conf"),
           get("--rpb-mixer-checkpoint", ""), mixer_steps, "rpb_mixer", true}));
#endif
    ev::run_feature_evaluation(run, factories);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

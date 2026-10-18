// SPDX-License-Identifier: MIT
#include "embedding/shared/feature_stress.h"
#include "shared_test_support.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>

namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using test::check;
using test::close;
struct SavedCase { torch::Tensor values, mask, base_mask, erased; };
SavedCase load_case(const fs::path &directory, const std::string &name) {
  torch::serialize::InputArchive archive;
  archive.load_from((directory / "stress" / (name + "-corrupted.pt")).string(), torch::kCPU);
  SavedCase result;
  archive.read("observed", result.values, true); archive.read("feature_mask", result.mask, true);
  archive.read("base_feature_mask", result.base_mask, true); archive.read("requested_erasure_mask", result.erased, true);
  return result;
}
std::string read_file(const fs::path &path) {
  std::ifstream input(path); std::ostringstream output; output << input.rdbuf(); return output.str();
}
ev::FeatureMap extract(const embedding::Batch &batch) {
  const auto safe = torch::where(batch.feature_mask, batch.data, torch::zeros_like(batch.data));
  const auto valid = batch.feature_mask.flatten(1).any(1);
  const auto values = safe.flatten(1).sum(1).unsqueeze(-1);
  return {{"signal", {values, valid, "legal observed values only"}},
      {"control", {torch::zeros_like(values), torch::ones_like(valid), "always-valid metadata control"}},
      {"unsupported", {torch::zeros_like(values), torch::ones_like(valid), "unsupported ordinary fit"}}};
}
ev::StressPredictions predict(const ev::FeatureSurface &surface) {
  const auto p = surface.values.select(1, 0).gt(0).to(torch::kInt64);
  return {surface.valid.clone(), p, p.clone()};
}
ev::StressReadouts readouts(const ev::ControlledProtocol &protocol) {
  const auto features = extract(protocol.testing.observed);
  ev::StressReadouts result;
  for (const auto &name : {"signal", "control"}) {
    const auto native = predict(features.at(name));
    result.emplace(std::make_pair(name, ev::DimensionTier::native),
        ev::FrozenStressReadout{"measured", "", 1, native.valid, native.ridge, native.tiny, predict});
  }
  auto native = predict(features.at("signal"));
  native.valid = native.valid.clone();
  // One paired source supports the matched tier: point score remains legal,
  // while the source-group interval must be null.
  const auto &first_source = protocol.testing.source_ids.front();
  for (int64_t row = 0; row < native.valid.size(0); ++row)
    native.valid[row] = protocol.testing.source_ids.at(row) == first_source;
  result.emplace(std::make_pair("signal", ev::DimensionTier::matched_global),
      ev::FrozenStressReadout{"measured", "", 1, native.valid, native.ridge, native.tiny,
        [ids = protocol.testing.source_ids, first_source](const ev::FeatureSurface &surface) {
          auto result = predict(surface);
          for (int64_t row = 0; row < result.valid.size(0); ++row)
            if (ids.at(row) != first_source) result.valid[row] = false;
          return result;
        }});
  result.emplace(std::make_pair("unsupported", ev::DimensionTier::native),
      ev::FrozenStressReadout{"unsupported_fit", "no ordinary train fit", 0,
          features.at("unsupported").valid, {}, {},
          [](const ev::FeatureSurface &) -> ev::StressPredictions {
            throw std::runtime_error("unsupported callback must never run");
          }});
  return result;
}
void primitives_and_frozen_contract() {
  const auto root = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("feature-stress-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(root);
  ev::EvaluationCard card; card.shape = {2, 32, 1, torch::kFloat64, torch::kCPU};
  card.channel_ids = {202, 101}; card.feature_units = "volts";
  card.comparisons = {{"signal_vs_control", "signal", "control", ev::DimensionTier::native}};
  auto protocol = ev::make_controlled_protocol(ev::Task::level, card.shape, 4, 2, 3, 202, .2);
  protocol.testing.observed.data = protocol.testing.observed.data.masked_fill(
      protocol.testing.observed.feature_mask.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const auto original_values = protocol.testing.observed.data.clone();
  const auto original_mask = protocol.testing.observed.feature_mask.clone();
  const std::map<std::string, ev::SurfaceDescription> descriptions{
      {"signal", {ev::SurfaceKind::global, "at least one observed coordinate", {}}},
      {"control", {ev::SurfaceKind::control, "always-valid metadata", {}}},
      {"unsupported", {ev::SurfaceKind::control, "unsupported fit retained", {}}}};
  const auto frozen = readouts(protocol);
  const auto first = root / "first", repeated = root / "repeated";
  fs::create_directory(first); fs::create_directory(repeated);
  int calls = 0;
  const ev::StressExtractor extractor = [&](const embedding::Batch &batch) {
    ++calls; auto result = extract(batch);
    // A provider may mutate its clone; saved canonical masks/storage must survive.
    batch.data.fill_(123456); batch.feature_mask.fill_(true);
    return result;
  };
  torch::manual_seed(811); const auto expected_rng = torch::rand({8});
  torch::manual_seed(811);
  const auto report = ev::run_fixed_readout_stress(first.string(), card, protocol, descriptions, frozen, extractor);
  close(torch::rand({8}), expected_rng, "stress must not consume Torch RNG", 0, 0);
  const auto repeated_report = ev::run_fixed_readout_stress(
      repeated.string(), card, protocol, descriptions, frozen, extractor);
  check(report == repeated_report && calls == 22, "stress replay/order/frozen callback contract");
  close(protocol.testing.observed.feature_mask, original_mask, "source support mutated", 0, 0);
  close(torch::where(original_mask, protocol.testing.observed.data, torch::zeros_like(original_values)),
      torch::where(original_mask, original_values, torch::zeros_like(original_values)), "source observations mutated", 0, 0);
  close(torch::isnan(protocol.testing.observed.data), torch::isnan(original_values), "source hidden storage mutated", 0, 0);
  const std::vector<std::string> random{"random_dropout_010", "random_dropout_030",
      "random_dropout_060", "random_dropout_090"};
  const std::vector<std::string> contiguous{"contiguous_025", "contiguous_050", "contiguous_075"};
  const std::vector<std::string> all{"intact", "random_dropout_010", "random_dropout_030",
      "random_dropout_060", "random_dropout_090", "contiguous_025", "contiguous_050", "contiguous_075",
      "channel_id_202_absent", "channel_id_101_absent", "all_absent"};
  for (const auto &name : all) {
    const auto a = load_case(first, name), b = load_case(repeated, name);
    close(a.values, b.values, name + " deterministic values", 0, 0);
    close(a.mask, b.mask, name + " deterministic mask", 0, 0);
    check(torch::isfinite(a.values).all().item<bool>() &&
        a.values.masked_select(a.mask.logical_not()).eq(0).all().item<bool>(),
        name + " hidden NaN/erased storage must be zero");
    check(!a.mask.logical_and(original_mask.logical_not()).any().item<bool>(),
        name + " stress added observation support");
    close(a.values.masked_select(a.mask), original_values.masked_select(a.mask),
          name + " retained values changed", 0, 0);
    for (int64_t i = 0; i < a.erased.size(0); ++i)
      for (int64_t j = i + 1; j < a.erased.size(0); ++j)
        if (protocol.testing.source_ids.at(i) == protocol.testing.source_ids.at(j))
          close(a.erased[i], a.erased[j], name + " paired variants have unequal erasures", 0, 0);
    const auto manifest = read_file(first / "stress" / (name + "-manifest.json"));
    check(manifest.find("checksums_fnv1a64") != std::string::npos &&
        manifest.find(protocol.testing.source_ids.front()) != std::string::npos,
        name + " missing checksum/source manifest");
  }
  const double rates[]{.1,.3,.6,.9};
  for (size_t i = 0; i < random.size(); ++i) {
    const auto saved = load_case(first, random[i]);
    const auto pure = ev::make_coordinate_deletion_view(protocol.testing.observed,
        protocol.testing.source_ids, protocol.task, protocol.seed, rates[i]);
    close(saved.erased, pure.requested_erasure, "historical TEST coordinate stream changed", 0, 0);
    close(saved.mask, pure.observations.feature_mask, "historical TEST support changed", 0, 0);
    close(saved.values, pure.observations.data, "historical TEST hidden storage changed", 0, 0);
  }
  for (const auto &family : {random, contiguous})
    for (size_t i = 1; i < family.size(); ++i) {
      const auto less = load_case(first, family[i - 1]), more = load_case(first, family[i]);
      check(!less.erased.logical_and(more.erased.logical_not()).any().item<bool>(),
          "severity masks must be nested");
    }
  const auto absent202 = load_case(first, "channel_id_202_absent");
  check(!absent202.mask.select(1, 0).any().item<bool>(), "semantic channel202 was not erased");
  close(absent202.mask.select(1, 1), original_mask.select(1, 1), "other channel erased", 0, 0);
  const auto outage_manifest = read_file(first / "stress" / "channel_id_202_absent-manifest.json");
  const auto total_rows = std::to_string(original_mask.size(0));
  const auto first_retained = std::to_string(absent202.mask[0][1].sum().item<int64_t>());
  check(outage_manifest.find("\"channel_ids\":[202,101]") != std::string::npos &&
      outage_manifest.find("\"observed_channel_rows\":[0," + total_rows + "]") != std::string::npos &&
      outage_manifest.find("\"absent_channel_rows\":[" + total_rows + ",0]") != std::string::npos &&
      outage_manifest.find("\"observed_coordinate_counts_by_channel\":[0," + first_retained + "]") != std::string::npos,
      "semantic-channel support accounting disagrees with saved mask");
  const auto absent = load_case(first, "all_absent");
  check(!absent.mask.any().item<bool>() && absent.values.eq(0).all().item<bool>(), "all-absent corruption");
  torch::serialize::InputArchive signal, control;
  signal.load_from((first / "stress" / "all_absent-signal-native-predictions.pt").string(), torch::kCPU);
  control.load_from((first / "stress" / "all_absent-control-native-predictions.pt").string(), torch::kCPU);
  torch::Tensor signal_valid, control_valid;
  signal.read("prediction_valid", signal_valid, true); control.read("prediction_valid", control_valid, true);
  check(!signal_valid.any().item<bool>() && control_valid.all().item<bool>(),
      "all-absent signal/control coverage was conflated");
  check(report.find("\"source_groups\":1,\"estimate\":null,\"lower\":null,\"upper\":null") != std::string::npos &&
      report.find("\"status\":\"unsupported_fit\",\"reason\":\"no ordinary train fit\"") != std::string::npos &&
      report.find("\"full_population_correct_rate\":0") != std::string::npos,
      "unsupported fit/one-source/full-population score missing");
  const auto card_text = ev::fixed_readout_stress_card_json(card);
  check(card_text.find("\"cases\":[{\"id\":\"intact\"") != std::string::npos &&
      card_text.find("\"id\":\"all_absent\"") != std::string::npos, "frozen stress card case order");

  auto relabeled = protocol; relabeled.testing.labels = 1 - protocol.testing.labels;
  const auto label_directory = root / "relabeled"; fs::create_directory(label_directory);
  ev::run_fixed_readout_stress(label_directory.string(), card, relabeled, descriptions, frozen, extract);
  close(load_case(label_directory, random.front()).erased, load_case(first, random.front()).erased,
      "labels must not change corruption", 0, 0);
  auto renamed = protocol;
  for (auto &id : renamed.testing.source_ids) id += "/distinct-identity";
  const auto renamed_directory = root / "renamed"; fs::create_directory(renamed_directory);
  ev::run_fixed_readout_stress(renamed_directory.string(), card, renamed, descriptions, readouts(renamed), extract);
  check(!torch::equal(load_case(renamed_directory, random[1]).erased, load_case(first, random[1]).erased),
      "distinct source identity reused the same random mask");
  bool rejected = false;
  try { ev::run_fixed_readout_stress(first.string(), card, protocol, descriptions, frozen, extract); }
  catch (const std::exception &) { rejected = true; }
  check(rejected, "existing stress artifacts were replaced");
  std::cout << "feature stress tests passed; artifacts=" << root << '\n';
}
} // namespace
int main() {
  try { torch::set_num_threads(1); primitives_and_frozen_contract(); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

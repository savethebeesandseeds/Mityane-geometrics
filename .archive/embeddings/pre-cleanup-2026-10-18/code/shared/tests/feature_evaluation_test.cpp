// SPDX-License-Identifier: MIT
#include "embedding/shared/feature_evaluation.h"
#include "embedding/shared/feature_stress.h"
#include "shared_test_support.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>

namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
namespace {
template <typename T, typename = void> struct has_labels : std::false_type {};
template <typename T> struct has_labels<T, std::void_t<decltype(std::declval<T>().labels)>> : std::true_type {};
template <typename T, typename = void> struct has_clean : std::false_type {};
template <typename T> struct has_clean<T, std::void_t<decltype(std::declval<T>().clean)>> : std::true_type {};
template <typename T, typename = void> struct has_validation : std::false_type {};
template <typename T> struct has_validation<T, std::void_t<decltype(std::declval<T>().validation)>> : std::true_type {};
template <typename T, typename = void> struct has_testing : std::false_type {};
template <typename T> struct has_testing<T, std::void_t<decltype(std::declval<T>().testing)>> : std::true_type {};
static_assert(!has_labels<ev::ProviderFitInput>::value && !has_clean<ev::ProviderFitInput>::value &&
              !has_validation<ev::ProviderFitInput>::value && !has_testing<ev::ProviderFitInput>::value,
              "provider fit input exposes hidden or held-out data");

template <typename F> void rejects(F &&function, const std::string &message) {
  bool rejected = false;
  try { function(); } catch (const std::exception &) { rejected = true; }
  test::check(rejected, message);
}
std::string read(const fs::path &path) {
  std::ifstream input(path);
  test::check(bool(input), "missing artifact: " + path.string());
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}
// Locate a complete report object without depending on an external JSON library.
std::string object(const std::string &json, const std::string &marker) {
  const auto position = json.find(marker);
  test::check(position != std::string::npos, "missing report marker: " + marker);
  const auto begin = json.rfind('{', position);
  int depth = 0;
  bool quoted = false, escape = false;
  for (size_t i = begin; i < json.size(); ++i) {
    const auto c = json[i];
    if (quoted) {
      if (escape) escape = false;
      else if (c == '\\') escape = true;
      else if (c == '"') quoted = false;
    } else if (c == '"') quoted = true;
    else if (c == '{') ++depth;
    else if (c == '}' && --depth == 0) return json.substr(begin, i - begin + 1);
  }
  throw std::runtime_error("unterminated report object");
}
ev::EvaluationCard card() {
  ev::EvaluationCard value;
  value.id = "shared-dummy-provider-test";
  value.shape = {1, 32, 1, torch::kFloat64, torch::kCPU};
  value.channel_ids = {17};
  value.feature_units = "unitless";
  value.sampling_interval = .5;
  value.train_pairs = 8; value.validation_pairs = 4; value.test_pairs = 6;
  value.matched_global_width = 2; value.matched_channel_width = 2;
  value.seeds = {44}; value.tasks = {ev::Task::level};
  value.comparisons = {{"declared_pair", "alpha", "beta", ev::DimensionTier::matched_global}};
  return value;
}
torch::Tensor features(const embedding::Batch &batch) {
  const auto mask = batch.feature_mask.to(torch::kFloat64).flatten(1);
  const auto x = torch::where(batch.feature_mask, batch.data, torch::zeros_like(batch.data)).flatten(1);
  const auto count = mask.sum(1).clamp_min(1);
  const auto mean = x.sum(1) / count;
  const auto second = x.pow(2).sum(1) / count;
  return torch::stack({mean, second}, 1);
}
struct FitAudit {
  int fit_calls{0}, extract_calls{0}, save_calls{0};
  torch::Tensor mean;
};
ev::FeatureProviderFactory dummy(const fs::path &output, const ev::EvaluationCard &configuration,
                                const std::shared_ptr<FitAudit> &audit,
                                bool observed_support = false, bool stress_enabled = false) {
  return [output, configuration, audit, observed_support, stress_enabled](const ev::ProviderFitInput &input) {
    ++audit->fit_calls;
    test::check(fs::exists(output / "evaluation-card.json"), "provider fitted before card was frozen");
    if (stress_enabled)
      test::check(fs::exists(output / "stress-card.json"), "provider fitted before stress recipe was frozen");
    const auto expected = ev::make_controlled_protocol(configuration.tasks[0], configuration.shape,
        configuration.train_pairs, configuration.validation_pairs, configuration.test_pairs, configuration.seeds[0]);
    test::check(input.training_source_ids == expected.training.source_ids &&
                input.training_observations.data.size(0) == configuration.train_pairs * 2,
                "provider did not receive the training partition only");
    test::close(input.training_observations.data, expected.training.observed.data, "provider training observations", 0, 0);
    test::check(input.channel_ids == configuration.channel_ids && input.feature_units == configuration.feature_units &&
                input.endpoint == 15.5 && input.sampling_interval == .5 && input.seed == 44,
                "provider fit metadata does not follow card");
    const auto training = features(input.training_observations);
    audit->mean = training.mean(0).clone();
    const auto mean = audit->mean.clone();
    ev::FeatureProvider provider;
    provider.provenance = "dummy training-only frozen state";
    provider.audit_fields = {{"quoted-value", "a \"quoted\"\nstring"}, {"training-only", "true"}};
    const auto support = observed_support ? "at least one observed coordinate" : "all observations";
    provider.surfaces = {
      {"alpha", {ev::SurfaceKind::global, support, {}}},
      {"beta", {ev::SurfaceKind::global, support, {}}},
      {"arbitrary_vector", {ev::SurfaceKind::channel_concatenation, "all declared channels observed", {17}}},
      {"constant", {ev::SurfaceKind::global, support, {}}},
      {"one_class", {ev::SurfaceKind::global, "positive observed mean only", {}}}};
    provider.extract = [mean, audit, observed_support, stress_enabled](const embedding::Batch &batch) {
      ++audit->extract_calls;
      test::check(torch::equal(mean, audit->mean), "provider's frozen training mean changed");
      if (stress_enabled && audit->extract_calls > 3)
        test::check(batch.data.masked_select(batch.feature_mask.logical_not()).eq(0).all().item<bool>(),
                    "stress provider received hidden nonzero storage");
      const auto x = features(batch);
      const auto valid = observed_support ? batch.feature_mask.flatten(1).any(1) :
                                           torch::ones({x.size(0)}, torch::kBool);
      return ev::FeatureMap{
          {"alpha", {x, valid, "arbitrary global surface"}},
          {"beta", {x - mean, valid, "frozen training centering"}},
          {"arbitrary_vector", {x, valid, "one-channel concatenation"}},
          {"constant", {torch::ones_like(x), valid, "constant rank-zero features"}},
          {"one_class", {x, x.select(1, 0).gt(0).logical_and(valid), "one supported training class"}}};
    };
    provider.save_assets = [mean, audit](const std::string &directory) {
      ++audit->save_calls;
      torch::serialize::OutputArchive output;
      output.write("train_mean", mean, true);
      embedding::archive::save_archive((fs::path(directory) / "dummy-fit.pt").string(), output);
    };
    return provider;
  };
}
ev::FeatureProviderFactory invalid_provider();
torch::Tensor archived_tensor(const fs::path &path, const std::string &key) {
  torch::serialize::InputArchive input;
  input.load_from(path.string(), torch::kCPU);
  torch::Tensor value;
  input.read(key, value, true);
  return value;
}
void same_archive_fields(const fs::path &left, const fs::path &right,
                         const std::vector<std::string> &fields) {
  torch::serialize::InputArchive a, b;
  a.load_from(left.string(), torch::kCPU); b.load_from(right.string(), torch::kCPU);
  for (const auto &field : fields) {
    torch::Tensor x, y;
    const bool has_x = a.try_read(field, x, true), has_y = b.try_read(field, y, true);
    test::check(has_x == has_y, "stress changed fitted archive schema: " + field);
    if (has_x)
      test::check(torch::equal(x, y) && x.scalar_type() == y.scalar_type(),
                  "stress changed saved fitted tensor: " + field);
  }
}
void stress_integration(const fs::path &temporary) {
  const auto configuration = card();
  const auto off = temporary / "stress-off", on = temporary / "stress-on";
  auto ordinary_audit = std::make_shared<FitAudit>(), stress_audit = std::make_shared<FitAudit>();
  ev::run_feature_evaluation({configuration, off.string(), "fixed-source", "fixed-head", "true"},
      {dummy(off, configuration, ordinary_audit, true), invalid_provider()});
  ev::run_feature_evaluation({configuration, on.string(), "fixed-source", "fixed-head", "true", true},
      {dummy(on, configuration, stress_audit, true, true), invalid_provider()});
  test::check(ordinary_audit->fit_calls == 1 && ordinary_audit->extract_calls == 3 &&
              ordinary_audit->save_calls == 1 && stress_audit->fit_calls == 1 &&
              stress_audit->extract_calls == 3 + 9 + configuration.shape.channel_count &&
              stress_audit->save_calls == 1,
              "stress refitted/re-saved a provider or repeated ordinary extraction");
  test::check(torch::equal(ordinary_audit->mean, stress_audit->mean),
              "stress changed train-only provider statistics");
  test::check(read(off / "report.json") == read(on / "report.json") &&
              read(off / "evaluation-card.json") == read(on / "evaluation-card.json"),
              "opt-in stress changed the ordinary report/card");
  test::check(!fs::exists(off / "stress-card.json") && !fs::exists(off / "stress-report.json") &&
              read(on / "stress-card.json") == ev::fixed_readout_stress_card_json(configuration),
              "stress artifacts leaked into the default path or recipe serialization changed");
  const auto before = off / "seed-44-level", after = on / "seed-44-level", stress = after / "stress";
  test::check(read(before / "manifest.json") == read(after / "manifest.json") &&
              read(before / "provider-0" / "provider-audit.json") == read(after / "provider-0" / "provider-audit.json"),
              "stress changed ordinary data or provider provenance");
  same_archive_fields(before / "provider-0" / "dummy-fit.pt", after / "provider-0" / "dummy-fit.pt", {"train_mean"});
  same_archive_fields(before / "raw-control-scaler.pt", after / "raw-control-scaler.pt", {"mean", "scale", "counts"});
  for (const auto &entry : fs::directory_iterator(before)) {
    const auto name = entry.path().filename().string();
    if (name.size() < 7 || name.substr(name.size() - 7) != "-fit.pt") continue;
    same_archive_fields(entry.path(), after / name, {
        "feature_mean", "feature_scale", "fitted_rows", "pca_mean", "pca_components", "pca_singular_values",
        "pca_numerical_rank", "ridge_mean", "ridge_scale", "ridge_weights", "ridge_intercept",
        "tiny_mean", "tiny_scale", "tiny_w1", "tiny_b1", "tiny_w2", "tiny_b2",
        "shuffled_mean", "shuffled_scale", "shuffled_weights", "shuffled_intercept"});
  }
  const auto report = read(stress / "report.json");
  const auto aggregate = read(on / "stress-report.json");
  test::check(aggregate.find("\"protocol\":\"fixed-readout-stress-v1\"") != std::string::npos &&
              aggregate.find("\"source_fingerprint\":\"fixed-source\"") != std::string::npos &&
              aggregate.find("\"report_file\":\"seed-44-level/stress/report.json\"") != std::string::npos &&
              aggregate.find("\"results\":" + report) != std::string::npos,
              "aggregate stress report lost protocol/provenance/artifact pointers");
  int intact_readouts = 0;
  for (const auto &entry : fs::directory_iterator(stress)) {
    const auto name = entry.path().filename().string();
    if (name.rfind("intact-", 0) != 0 || name.size() < 15 ||
        name.substr(name.size() - 15) != "-predictions.pt") continue;
    ++intact_readouts;
    test::check(torch::equal(archived_tensor(entry.path(), "prediction_valid"),
                            archived_tensor(entry.path(), "ordinary_prediction_valid")) &&
                torch::equal(archived_tensor(entry.path(), "ridge"), archived_tensor(entry.path(), "ordinary_ridge")) &&
                torch::equal(archived_tensor(entry.path(), "tiny"), archived_tensor(entry.path(), "ordinary_tiny")),
                "intact stress predictions/validity differ from the ordinary frozen readout");
  }
  test::check(intact_readouts >= 6, "intact stress omitted measured readouts");
  test::check(torch::equal(archived_tensor(after / "alpha-testing.pt", "features"),
                          archived_tensor(stress / "intact-alpha-native-predictions.pt", "native_features_under_stress")),
              "provider mutation leaked into intact stress features");
  const auto absent = object(report, "\"case\":{\"id\":\"all_absent\"");
  const auto signal = object(absent, "\"surface\":\"alpha\",\"tier\":\"native\"");
  const auto raw = object(absent, "\"surface\":\"raw\",\"tier\":\"native\"");
  const auto mask = object(absent, "\"surface\":\"mask_metadata\",\"tier\":\"native\"");
  test::check(signal.find("\"coverage\":0") != std::string::npos &&
              signal.find("\"conditional_accuracy\":null") != std::string::npos &&
              signal.find("\"full_population_correct_rate\":0") != std::string::npos &&
              raw.find("\"conditional_accuracy\":null") != std::string::npos &&
              mask.find("\"coverage\":1") != std::string::npos &&
              mask.find("\"conditional_accuracy\":0.5") != std::string::npos,
              "all-absent signals fabricated coverage/correctness or erased metadata support");
  test::check(object(absent, "\"surface\":\"constant\",\"tier\":\"matched_global\"")
                  .find("\"status\":\"unsupported_compression\"") != std::string::npos &&
              object(absent, "\"surface\":\"irrelevant\",\"tier\":\"native\"")
                  .find("\"status\":\"unsupported_fit\"") != std::string::npos,
              "stress refitted or dropped an unsupported ordinary readout");
  const auto pair = object(absent, "\"comparison\":\"declared_pair\"");
  test::check(pair.find("\"common_population\":{\"total\":12,\"valid\":0") != std::string::npos &&
              pair.find("\"ridge_candidate_minus_comparator_grouped_interval\":{\"method\":") != std::string::npos &&
              pair.find("\"source_groups\":0,\"estimate\":null,\"lower\":null,\"upper\":null") != std::string::npos,
              "empty common stress population manufactured a conditional paired effect");
  test::check(archived_tensor(stress / "all_absent-alpha-native-predictions.pt", "prediction_valid")
                  .logical_not().all().item<bool>() &&
              archived_tensor(stress / "all_absent-mask_metadata-native-predictions.pt", "prediction_valid")
                  .all().item<bool>(),
              "archive validity disagrees with all-absent signal/control scores");

  // The served stressed predictions must be reproducible from ordinary saved
  // fit assets, including the outer PCA and each probe's own normalization.
  const auto predictions_path = stress / "contiguous_050-alpha-matched_global-predictions.pt";
  const auto fit_path = after / "alpha-matched_global-fit.pt";
  const auto valid = archived_tensor(predictions_path, "prediction_valid");
  const auto rows = valid.nonzero().flatten();
  auto x = archived_tensor(predictions_path, "native_features_under_stress");
  const auto normalize = [&rows](const torch::Tensor &values, const torch::Tensor &mean, const torch::Tensor &scale) {
    auto transformed = torch::zeros_like(values);
    transformed.index_copy_(0, rows, (values.index_select(0, rows) - mean) / scale);
    return transformed;
  };
  x = normalize(x, archived_tensor(fit_path, "feature_mean"), archived_tensor(fit_path, "feature_scale"));
  auto projected = torch::zeros({x.size(0), configuration.matched_global_width}, torch::kFloat64);
  projected.index_copy_(0, rows, (x.index_select(0, rows) - archived_tensor(fit_path, "pca_mean"))
                               .matmul(archived_tensor(fit_path, "pca_components")));
  const auto ridge_x = normalize(projected, archived_tensor(fit_path, "ridge_mean"), archived_tensor(fit_path, "ridge_scale"));
  const auto ridge = (ridge_x.matmul(archived_tensor(fit_path, "ridge_weights")) +
                      archived_tensor(fit_path, "ridge_intercept")).argmax(1);
  const auto tiny_x = normalize(projected, archived_tensor(fit_path, "tiny_mean"), archived_tensor(fit_path, "tiny_scale"));
  const auto tiny = (torch::tanh(tiny_x.matmul(archived_tensor(fit_path, "tiny_w1")) + archived_tensor(fit_path, "tiny_b1"))
                      .matmul(archived_tensor(fit_path, "tiny_w2")) + archived_tensor(fit_path, "tiny_b2")).argmax(1);
  test::check(rows.numel() == 12 && torch::equal(ridge, archived_tensor(predictions_path, "ridge")) &&
              torch::equal(tiny, archived_tensor(predictions_path, "tiny")),
              "stress served a refitted or different normalization/PCA/probe pipeline");
}
ev::FeatureProviderFactory invalid_provider() {
  return [](const ev::ProviderFitInput &input) {
    // Neither fitting nor extraction can mutate the engine's protocol tensors.
    input.training_observations.data.fill_(123456);
    ev::FeatureProvider provider;
    provider.provenance = "unrelated unsupported provider";
    provider.surfaces = {{"irrelevant", {ev::SurfaceKind::global, "no supported examples", {}}}};
    provider.extract = [](const embedding::Batch &batch) {
      batch.data.fill_(std::numeric_limits<double>::quiet_NaN());
      return ev::FeatureMap{{"irrelevant", {torch::full({batch.data.size(0), 2},
          std::numeric_limits<double>::quiet_NaN(), torch::kFloat64),
          torch::zeros({batch.data.size(0)}, torch::kBool), "invalid placeholders"}}};
    };
    return provider;
  };
}
void orchestration(const fs::path &temporary) {
  const auto configuration = card();
  auto audit = std::make_shared<FitAudit>();
  const auto first = temporary / "first", second = temporary / "extra-invalid";
  ev::run_feature_evaluation({configuration, first.string(), "test-source-sha", "test-head", "true"},
                            {dummy(first, configuration, audit)});
  test::check(audit->fit_calls == 1 && audit->extract_calls == 3, "provider refitted across held-out splits");
  const auto before = audit->mean.clone();
  auto other_audit = std::make_shared<FitAudit>();
  ev::run_feature_evaluation({configuration, second.string(), "test-source-sha", "test-head", "true"},
                            {dummy(second, configuration, other_audit), invalid_provider()});
  test::close(audit->mean, before, "held-out extraction changed fitted provider statistics", 0, 0);
  const auto report = read(first / "report.json"), extra = read(second / "report.json");
  test::check(object(report, "\"comparison\":\"declared_pair\"") ==
              object(extra, "\"comparison\":\"declared_pair\""),
              "unrelated invalid provider changed pair population, scores or uncertainty");
  const auto pair = object(report, "\"comparison\":\"declared_pair\"");
  test::check(pair.find("\"status\":\"measured\"") != std::string::npos &&
              pair.find("\"class_valid_rows\":[6,6]") != std::string::npos &&
              pair.find("\"complete_source_pairs\":6") != std::string::npos &&
              pair.find("\"ridge_candidate_minus_comparator_grouped_interval\"") != std::string::npos,
              "declared pair lost denominator/class/source support or effect interval");
  const auto alpha = object(report, "\"surface\":\"alpha\",\"kind\":\"global\",\"tier\":\"matched_global\"");
  test::check(alpha.find("\"status\":\"measured\"") != std::string::npos &&
              report.find("\"surface\":\"alpha\",\"kind\":\"global\",\"tier\":\"matched_channels\"") == std::string::npos,
              "global tier routing depends on feature-name suffix");
  test::check(report.find("\"surface\":\"arbitrary_vector\",\"kind\":\"channel_concatenation\",\"tier\":\"matched_channels\"") != std::string::npos &&
              report.find("\"surface\":\"arbitrary_vector\",\"kind\":\"channel_concatenation\",\"tier\":\"matched_global\"") == std::string::npos,
              "concatenation tier routing ignores typed declaration");
  const auto constant = object(report, "\"surface\":\"constant\",\"kind\":\"global\",\"tier\":\"matched_global\"");
  test::check(constant.find("\"status\":\"unsupported_compression\"") != std::string::npos &&
              constant.find("numerical training rank") != std::string::npos,
              "matching equal native width bypasses numerical-rank guard");
  test::check(object(report, "\"surface\":\"one_class\",\"kind\":\"global\",\"tier\":\"native\"")
                  .find("\"status\":\"unsupported_fit\"") != std::string::npos,
              "missing training class aborts or fits the run");
  const auto unsupported = object(extra, "\"surface\":\"irrelevant\",\"kind\":\"global\",\"tier\":\"native\"");
  test::check(unsupported.find("\"status\":\"unsupported_fit\"") != std::string::npos &&
              unsupported.find("\"class_valid_rows\":[0,0]") != std::string::npos &&
              unsupported.find("\"coverage\":0") != std::string::npos,
              "zero-valid surface disappeared or manufactured support");
  test::check(read(first / "evaluation-card.json") == ev::evaluation_card_json(configuration),
              "frozen card artifact differs from public serialization");
  test::check(report.find("\"source_fingerprint\":\"test-source-sha\"") != std::string::npos &&
              report.find("\"quoted-value\":\"a \\\"quoted\\\"\\u000astring\"") != std::string::npos &&
              report.find("\"incomplete_policy\"") != std::string::npos,
              "report lost source provenance, safely quoted audits or policy limits");
  const auto directory = first / "seed-44-level";
  const auto expected = ev::make_controlled_protocol(ev::Task::level, configuration.shape, 8, 4, 6, 44);
  torch::serialize::InputArchive archive;
  archive.load_from((directory / "controlled-training.pt").string(), torch::kCPU);
  torch::Tensor observed, mask, labels, ids;
  archive.read("observed", observed, true); archive.read("feature_mask", mask, true);
  archive.read("labels_scoring_only", labels, true); archive.read("source_ids_json", ids, true);
  test::check(observed.scalar_type() == torch::kFloat64 && torch::equal(mask, expected.training.observed.feature_mask) &&
              torch::equal(labels, expected.training.labels), "controlled archive changed precision/mask/labels");
  test::close(observed, expected.training.observed.data, "controlled observation archive", 0, 0);
  test::check(embedding::archive::tensor_text(ids).find(expected.training.source_ids[0]) != std::string::npos &&
              read(directory / "manifest.json").find("\"endpoint\":15.5") != std::string::npos,
              "controlled archive or manifest lost source/time metadata");
  torch::serialize::InputArchive fit;
  fit.load_from((directory / "alpha-native-fit.pt").string(), torch::kCPU);
  torch::Tensor mean, fitted_rows;
  fit.read("feature_mean", mean, true); fit.read("fitted_rows", fitted_rows, true);
  test::close(mean, features(expected.training.observed).mean(0), "probe fitted held-out observations", 0, 0);
  test::check(fitted_rows.item<int64_t>() == 16, "feature-fit asset has wrong training count");
  torch::serialize::InputArchive saved_provider;
  saved_provider.load_from((directory / "provider-0" / "dummy-fit.pt").string(), torch::kCPU);
  torch::Tensor saved_mean;
  saved_provider.read("train_mean", saved_mean, true);
  test::close(saved_mean, audit->mean, "provider assets did not preserve frozen fit", 0, 0);
  // Existing output directories are preserved rather than overwritten.
  rejects([&] { ev::run_feature_evaluation({configuration, first.string()}, {}); }, "existing run artifacts were overwritten");
}
void validation(const fs::path &temporary) {
  auto configuration = card();
  ev::validate_evaluation_card(configuration); // Single-channel non-lag cards are valid.
  auto bad = configuration; bad.version = 99;
  rejects([&] { ev::validate_evaluation_card(bad); }, "unimplemented card version was accepted");
  bad = configuration; bad.policy_version = "confirmation-v9";
  rejects([&] { ev::validate_evaluation_card(bad); }, "unimplemented policy version was accepted");
  bad = configuration; bad.tasks = {ev::Task::lag_sign};
  rejects([&] { ev::validate_evaluation_card(bad); }, "single-channel lag card was accepted");
  bad = configuration; bad.stage = "confirmation";
  rejects([&] { ev::validate_evaluation_card(bad); }, "incomplete confirmation policy was accepted");
  bad = configuration; bad.shape.dtype = torch::kFloat32;
  rejects([&] { ev::validate_evaluation_card(bad); }, "card dtype disagrees with float64 raw observations");
  bad = configuration; bad.channel_ids = {17,17};
  rejects([&] { ev::validate_evaluation_card(bad); }, "invalid channel schema was accepted");
  bad = configuration; bad.comparisons[0].id = "../escape";
  rejects([&] { ev::validate_evaluation_card(bad); }, "unsafe comparison filename was accepted");
  configuration.comparisons.clear();
  const auto malformed = [](int mode) -> ev::FeatureProviderFactory {
    return [mode](const ev::ProviderFitInput &) {
      ev::FeatureProvider provider;
      provider.provenance = "malformed dummy";
      const auto name = mode == 2 ? "../bad" : "arbitrary";
      provider.surfaces = {{name, {mode == 3 ? ev::SurfaceKind::channel_concatenation : ev::SurfaceKind::global,
                                   "declared support", mode == 3 ? std::vector<int64_t>{99} : std::vector<int64_t>{}}}};
      const auto calls = std::make_shared<int>(0);
      provider.extract = [mode, name, calls](const embedding::Batch &batch) {
        ++*calls;
        if (mode == 0 && *calls > 1) return ev::FeatureMap{};
        const auto width = mode == 1 && *calls > 1 ? 3 : 2;
        return ev::FeatureMap{{name, {torch::zeros({batch.data.size(0), width}, torch::kFloat64),
            torch::ones({batch.data.size(0)}, torch::kBool), "stable surface"}}};
      };
      return provider;
    };
  };
  for (int mode = 0; mode < 4; ++mode) {
    const auto output = temporary / ("malformed-" + std::to_string(mode));
    rejects([&] { ev::run_feature_evaluation({configuration, output.string()}, {malformed(mode)}); },
            "provider key/dimension/name/channel-order contract violation was accepted");
  }
}
} // namespace
int main() {
  try {
    const auto unique = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto temporary = fs::temp_directory_path() / ("embedding-feature-evaluation-test-" + unique);
    fs::create_directory(temporary);
    validation(temporary);
    orchestration(temporary);
    stress_integration(temporary);
    std::cout << "shared feature evaluation tests passed; artifacts=" << temporary << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

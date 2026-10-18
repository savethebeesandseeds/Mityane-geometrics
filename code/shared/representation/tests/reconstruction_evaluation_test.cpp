// SPDX-License-Identifier: MIT
#include "embedding/shared/reconstruction_evaluation.h"
#include "shared_test_support.h"

#include <chrono>
#include <cmath>
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
template <typename T, typename = void> struct has_testing : std::false_type {};
template <typename T> struct has_testing<T, std::void_t<decltype(std::declval<T>().testing)>> : std::true_type {};
static_assert(!has_labels<ev::ProviderFitInput>::value && !has_clean<ev::ProviderFitInput>::value &&
              !has_testing<ev::ProviderFitInput>::value, "reconstruction fit API exposes hidden/supervised data");
template <typename F> void rejects(F &&function, const std::string &message) {
  bool failed = false;
  try { function(); } catch (const std::exception &) { failed = true; }
  test::check(failed, message);
}
std::string read(const fs::path &path) {
  std::ifstream input(path);
  test::check(bool(input), "missing artifact: " + path.string());
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}
std::string object(const std::string &json, const std::string &marker, size_t offset = 0) {
  const auto position = json.find(marker, offset);
  test::check(position != std::string::npos, "report marker missing: " + marker);
  const auto begin = json.rfind('{', position);
  int depth = 0;
  bool quoted = false, escape = false;
  for (size_t i = begin; i < json.size(); ++i) {
    const char c = json[i];
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
torch::Tensor load_tensor(const fs::path &path, const std::string &key) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), torch::kCPU);
  torch::Tensor value;
  archive.read(key, value, true);
  return value;
}
ev::ReconstructionCard card() {
  ev::ReconstructionCard value;
  value.id = "shared-reconstruction-test";
  value.shape = {2, 24, 1, torch::kFloat64, torch::kCPU};
  value.channel_ids = {17, 23}; value.feature_units = "unitless";
  value.patch_length = 8;
  value.train_pairs = 4; value.validation_pairs = 4; value.test_pairs = 8;
  value.seeds = {44}; value.tasks = {ev::Task::level};
  value.bootstrap_replicates = 100;
  value.provider_recipe = "dummy: fixed three-coordinate sinusoid decoder; training-only coordinate metadata mean; no optimization";
  return value;
}
struct Audit {
  int fits{0}, encodes{0}, decodes{0}, metadata{0}, random{0};
  torch::Tensor train_mean;
};
torch::Tensor design(int64_t history) {
  const auto t = torch::arange(history, torch::kFloat64) * (2 * 3.14159265358979323846 / 16.0);
  return torch::stack({torch::ones_like(t), t.sin(), t.cos()}, 1);
}
ev::ReconstructionProviderFactory dummy(const fs::path &output, const ev::ReconstructionCard &configuration,
                                      const std::shared_ptr<Audit> &audit, int mode = 0) {
  return [output, configuration, audit, mode](const ev::ProviderFitInput &fit) {
    ++audit->fits;
    test::check(read(output / "reconstruction-card.json") == ev::reconstruction_card_json(configuration),
                "provider fitted before resolved card was frozen");
    const auto protocol = ev::make_controlled_protocol(configuration.tasks[0], configuration.shape,
        configuration.train_pairs, configuration.validation_pairs, configuration.test_pairs, configuration.seeds[0], 0);
    test::check(fit.training_source_ids == protocol.training.source_ids &&
                fit.training_observations.feature_mask.all().item<bool>(), "provider fit partition is not fully observed training only");
    test::close(fit.training_observations.data, protocol.training.observed.data, "provider sees wrong fit observations", 0, 0);
    audit->train_mean = fit.training_observations.data.mean(0).clone();
    const auto mean = audit->train_mean.clone(), basis = design(configuration.shape.history_length);
    const auto C = configuration.shape.channel_count;
    const auto H = configuration.shape.history_length, F = configuration.shape.input_width;
    const auto K = H / configuration.patch_length;
    // Deliberate writes must affect this provider's copy only, not metric fits or targets.
    fit.training_observations.data.fill_(std::numeric_limits<double>::quiet_NaN());
    ev::ReconstructionProvider provider;
    provider.name = "dummy";
    provider.export_semantics = "three independent sinusoid coordinates per channel; exact decoder input";
    provider.provenance = "training-only metadata mean, no hidden target access";
    provider.audit_fields = {{"metadata-fit", "training only"}, {"quoted", "\"value\"\n"}};
    provider.encode_visible = [basis, C, K, mode, audit](const embedding::Batch &batch) {
      ++audit->encodes;
      test::check(batch.data.masked_select(batch.feature_mask.logical_not()).eq(0).all().item<bool>(),
                  "hidden target storage reached encoder");
      test::check(batch.feature_mask[0].eq(batch.feature_mask).all().item<bool>(),
                  "query/visible masks differ between examples");
      auto latent = torch::zeros({batch.data.size(0), C, 3}, torch::kFloat64);
      auto valid = torch::ones({batch.data.size(0), C}, torch::kBool);
      for (int64_t row = 0; row < batch.data.size(0); ++row) for (int64_t channel = 0; channel < C; ++channel) {
        const auto mask = batch.feature_mask[row][channel].select(1, 0);
        const auto indices = mask.nonzero().flatten();
        const auto x = basis.index_select(0, indices);
        const auto y = batch.data[row][channel].select(1, 0).index_select(0, indices);
        const auto coefficients = at::linalg_solve(x.transpose(0, 1).matmul(x) + 1e-10 * torch::eye(3, torch::kFloat64),
                                                  x.transpose(0, 1).matmul(y));
        latent[row][channel].copy_(coefficients);
      }
      if (mode == 1) { valid.fill_(false); latent.zero_(); }
      if (mode == 2) { valid.select(1, 1).fill_(false); latent.select(1, 1).zero_(); }
      if (mode == 4 && audit->encodes > K)
        latent = torch::cat({latent, torch::zeros({batch.data.size(0), C, 1}, torch::kFloat64)}, 2);
      if (mode == 5) valid.select(1, 1).fill_(false); // Invalid exports must be zero.
      if (mode == 6) latent[0][0][0] = std::numeric_limits<double>::quiet_NaN();
      batch.data.fill_(9999);
      return ev::ReconstructionLatent{latent, valid};
    };
    provider.decode = [basis, H, F, mode, audit](const torch::Tensor &latent) {
      ++audit->decodes;
      auto prediction = latent.matmul(basis.transpose(0, 1)).reshape({latent.size(0), latent.size(1), H, F}).clone();
      if (mode == 3) prediction[0][0][0][0] = std::numeric_limits<double>::quiet_NaN();
      latent.fill_(7777); // Every intervention must own a distinct input clone.
      return prediction;
    };
    provider.metadata_predict = [mean, mode, audit](const torch::Tensor &visible) {
      ++audit->metadata;
      test::check(visible.scalar_type() == torch::kBool && visible.dim() == 4, "metadata decoder receives values instead of support");
      auto prediction = mean.unsqueeze(0).expand({visible.size(0), mean.size(0), mean.size(1), mean.size(2)}).clone();
      if (mode == 3) prediction.fill_(std::numeric_limits<double>::quiet_NaN());
      visible.fill_(true);
      return prediction;
    };
    provider.untrained_predict = [C, H, F, audit](const embedding::Batch &batch) {
      ++audit->random;
      test::check(batch.data.masked_select(batch.feature_mask.logical_not()).eq(0).all().item<bool>(),
                  "hidden target storage reached untrained encoder");
      auto prediction = torch::zeros({batch.data.size(0), C, H, F}, torch::kFloat64);
      auto valid = torch::ones({batch.data.size(0), C}, torch::kBool);
      valid.select(1, 1).fill_(false); // Independent validity must restrict this comparison only.
      batch.data.fill_(std::numeric_limits<double>::quiet_NaN());
      return ev::ReconstructionPrediction{prediction, valid};
    };
    provider.save_assets = [mean](const std::string &directory) {
      torch::serialize::OutputArchive archive;
      archive.write("train_coordinate_mean", mean, true);
      embedding::archive::save_archive((fs::path(directory) / "dummy-metadata-fit.pt").string(), archive);
    };
    return provider;
  };
}
void archive_and_effects(const fs::path &temporary) {
  const auto configuration = card();
  const auto first = temporary / "first", repeated = temporary / "repeated";
  auto audit = std::make_shared<Audit>(), again = std::make_shared<Audit>();
  ev::run_reconstruction_evaluation({configuration, first.string(), "test-source", "test-head", "dirty"}, dummy(first, configuration, audit));
  ev::run_reconstruction_evaluation({configuration, repeated.string(), "test-source", "test-head", "dirty"}, dummy(repeated, configuration, again));
  const auto report = read(first / "report.json");
  test::check(report == read(repeated / "report.json"), "fixed reconstruction card is not reproducible");
  test::check(audit->fits == 1 && audit->encodes == 6 && audit->decodes == 18 &&
              audit->metadata == 6 && audit->random == 6, "provider refit or repeated target schedule changed");
  const auto task = first / "seed-44-level";
  const auto expected = ev::make_controlled_protocol(ev::Task::level, configuration.shape, 4, 4, 8, 44, 0);
  test::close(load_tensor(task / "training-observations.pt", "observed"), expected.training.observed.data,
              "fit mutation changed canonical training archive", 0, 0);
  const auto observations = load_tensor(task / "testing-observations.pt", "observed");
  test::check(observations.scalar_type() == torch::kFloat64, "controlled target precision was reduced");
  test::close(observations, expected.testing.observed.data, "canonical testing archive", 0, 0);
  const ev::ObservationScaler metric_scaler(expected.training.observed);
  test::close(load_tensor(task / "metric-scaler.pt", "mean"), metric_scaler.mean, "metric mean is not training-only", 0, 0);
  test::close(load_tensor(task / "metric-scaler.pt", "scale"), metric_scaler.scale, "metric scale is not training-only", 0, 0);
  test::close(load_tensor(task / "provider-assets" / "dummy-metadata-fit.pt", "train_coordinate_mean"),
              audit->train_mean, "metadata fitted-state archive", 0, 0);
  const auto donor = load_tensor(task / "testing-donors.pt", "donor_rows");
  const auto block = load_tensor(task / "testing-donors.pt", "exchange_block_ids");
  const auto B = observations.size(0);
  auto visits = torch::zeros({B}, torch::kInt64);
  for (int64_t row = 0; row < B; ++row) {
    const auto other = donor[row].item<int64_t>();
    test::check(other >= 0 && other < B && expected.testing.source_ids[row] != expected.testing.source_ids[other] &&
                donor[other].item<int64_t>() == row && block[row].item<int64_t>() == block[other].item<int64_t>(),
                "donor includes own source or is not a disjoint reciprocal exchange");
    test::check(row % 2 == other % 2, "donor map used labels instead of original within-source row ordinal");
    visits[other] += 1;
  }
  test::check(visits.eq(1).all().item<bool>(), "donor mapping is not a permutation");
  const auto exchange_counts = torch::bincount(block);
  test::check(exchange_counts.eq(4).all().item<bool>(), "exchange bootstrap block does not contain exactly two whole sources");
  auto all_targets = torch::zeros_like(observations, torch::kInt64);
  for (int64_t trial = 0; trial < 3; ++trial) {
    const auto path = task / ("testing-trial-" + std::to_string(trial) + ".pt");
    const auto query = load_tensor(path, "target_mask"), visible = load_tensor(path, "visible_support");
    const auto values = load_tensor(path, "visible_values"), targets = load_tensor(path, "target_values");
    all_targets += query.to(torch::kInt64);
    test::check(query[0].eq(query).all().item<bool>() && torch::equal(visible, query.logical_not()) &&
                values.masked_select(query).eq(0).all().item<bool>(), "fixed target/visible schedule leaks or varies by row");
    test::close(values.masked_select(visible), observations.masked_select(visible), "visible observations changed", 0, 0);
    test::close(targets.masked_select(query), observations.masked_select(query), "target values changed", 0, 0);
    test::check(torch::equal(load_tensor(path, "donor_rows"), donor) && torch::equal(load_tensor(path, "exchange_block_ids"), block),
                "donors changed across target trials");
    const auto latent = load_tensor(path, "latent");
    test::check(latent.abs().max().item<double>() < 4, "decoder in-place mutation changed saved exact latent");
    test::check(load_tensor(path, "zeroed_prediction").eq(0).all().item<bool>(), "zeroed intervention received another intervention's values");
  }
  test::check(all_targets.eq(1).all().item<bool>(), "target trials did not query every coordinate exactly once");
  const auto own = load_tensor(task / "testing-intact-metrics.pt", "example_standardized_mae");
  const auto correct = load_tensor(task / "testing-shuffled-minus-intact.pt", "intact_example_standardized_mae");
  const auto shuffled = load_tensor(task / "testing-shuffled-minus-intact.pt", "candidate_example_standardized_mae");
  test::check(own.mean().item<double>() < .02 && (shuffled - correct).mean().item<double>() > .1,
              "known compact sinusoid decoder does not demonstrate held-out latent reliance");
  test::check(torch::equal(load_tensor(task / "testing-shuffled-minus-intact.pt", "candidate_target_counts"),
                          load_tensor(task / "testing-shuffled-minus-intact.pt", "intact_target_counts")),
              "shuffled comparison uses different query denominators");
  const auto untrained_counts = load_tensor(task / "testing-untrained-minus-intact.pt", "candidate_target_counts");
  test::check(untrained_counts.select(1, 0).eq(24).all().item<bool>() &&
              untrained_counts.select(1, 1).eq(0).all().item<bool>(), "untrained validity was discarded or expanded common coverage");
  test::check(load_tensor(task / "testing-intact-metrics.pt", "target_counts").eq(24).all().item<bool>(),
              "unrelated untrained invalidity shrank intact full coverage");
  const auto test_offset = report.find("\"split\":\"testing\"");
  const auto comparison = object(report, "\"comparison\":\"shuffled_minus_intact\"", test_offset);
  test::check(comparison.find("\"valid_source_groups\":8") != std::string::npos &&
              comparison.find("\"valid_independent_exchange_blocks\":4") != std::string::npos &&
              comparison.find("\"status\":\"measured\"") != std::string::npos,
              "paired uncertainty treats variants/trials/source groups as independent exchange units");
  test::check(read(first / "reconstruction-card.json") == ev::reconstruction_card_json(configuration) &&
              report.find("\"source_fingerprint\":\"test-source\"") != std::string::npos &&
              report.find("\"quoted\":\"\\\"value\\\"\\u000a\"") != std::string::npos,
              "card/report source identity or quoted provider audit is incorrect");
  rejects([&] { ev::run_reconstruction_evaluation({configuration, first.string()}, {}); },
          "existing reconstruction artifacts were overwritten");
}
void support_and_failures(const fs::path &temporary) {
  const auto configuration = card();
  for (const auto mode : {1, 2}) {
    const auto output = temporary / ("support-" + std::to_string(mode));
    auto audit = std::make_shared<Audit>();
    ev::run_reconstruction_evaluation({configuration, output.string()}, dummy(output, configuration, audit, mode));
    const auto report = read(output / "report.json");
    const auto intact = object(report, "\"case\":\"intact\"");
    const auto comparison = object(report, "\"comparison\":\"metadata_only_minus_intact\"");
    if (mode == 1) {
      test::check(intact.find("\"status\":\"unsupported_zero_support\"") != std::string::npos &&
                  intact.find("\"standardized_mae\":null") != std::string::npos &&
                  comparison.find("\"estimate\":null") != std::string::npos,
                  "all-invalid latents manufacture scores or reliance");
      test::check(object(report, "\"case\":\"metadata_only\"").find("\"example_coverage\":1") != std::string::npos,
                  "structural metadata control was mistaken for invalid learned coverage");
    } else {
      const auto path = output / "seed-44-level" / "testing-metadata_only-minus-intact.pt";
      const auto actual = load_tensor(path, "intact_target_counts"), control = load_tensor(path, "candidate_target_counts");
      test::check(torch::equal(actual, control) && actual.select(1, 1).eq(0).all().item<bool>() &&
                  actual.select(1, 0).eq(24).all().item<bool>(), "common-valid channel reduction differs by intervention");
    }
  }
  auto small = configuration;
  small.validation_pairs = 2; small.test_pairs = 2;
  const auto small_output = temporary / "one-exchange-block";
  auto audit = std::make_shared<Audit>();
  ev::run_reconstruction_evaluation({small, small_output.string()}, dummy(small_output, small, audit));
  test::check(object(read(small_output / "report.json"), "\"comparison\":\"shuffled_minus_intact\"")
                .find("\"status\":\"unsupported_insufficient_exchange_blocks\"") != std::string::npos,
              "one exchange block produces a confidence interval");
  for (const auto mode : {3, 4, 5, 6}) {
    const auto malformed = temporary / ("malformed-" + std::to_string(mode));
    rejects([&] { ev::run_reconstruction_evaluation({configuration, malformed.string()},
        dummy(malformed, configuration, std::make_shared<Audit>(), mode)); },
        "nonfinite target/vector, invalid nonzero export or cross-split width change was accepted");
  }
}
void validation() {
  auto value = card();
  ev::validate_reconstruction_card(value);
  auto bad = value; bad.version = 2;
  rejects([&] { ev::validate_reconstruction_card(bad); }, "unimplemented reconstruction card version accepted");
  bad = value; bad.policy_version = "future";
  rejects([&] { ev::validate_reconstruction_card(bad); }, "unimplemented policy accepted");
  bad = value; bad.stage = "confirmation";
  rejects([&] { ev::validate_reconstruction_card(bad); }, "confirmation accepted without acceptance contract");
  bad = value; bad.shape.dtype = torch::kFloat32;
  rejects([&] { ev::validate_reconstruction_card(bad); }, "raw precision declaration differs from generator");
  bad = value; bad.patch_length = 7;
  rejects([&] { ev::validate_reconstruction_card(bad); }, "partial target patch accepted");
  bad = value; bad.patch_length = 12;
  rejects([&] { ev::validate_reconstruction_card(bad); }, "fewer than two visible patches accepted");
  bad = value; bad.validation_pairs = 3;
  rejects([&] { ev::validate_reconstruction_card(bad); }, "odd donor group count accepted");
  bad = value; bad.provider_recipe.clear();
  rejects([&] { ev::validate_reconstruction_card(bad); }, "unresolved adapter recipe accepted");
  bad = value; bad.channel_ids = {17,17};
  rejects([&] { ev::validate_reconstruction_card(bad); }, "duplicate semantic channel ID accepted");
}
void large_finite_summary(const fs::path &temporary) {
  auto configuration = card();
  configuration.shape = {1, 8, 1, torch::kFloat64, torch::kCPU};
  configuration.channel_ids = {17};
  configuration.patch_length = 2;
  configuration.train_pairs = 4;
  configuration.validation_pairs = 32;
  configuration.test_pairs = 32;
  configuration.provider_recipe = "dummy large finite raw-unit prediction; numerical summary guard fixture";
  const auto output = temporary / "large-finite-summary";
  bool rejected = false;
  try {
    ev::run_reconstruction_evaluation({configuration, output.string()}, [](const ev::ProviderFitInput &) {
      ev::ReconstructionProvider provider;
      provider.name = "large-finite";
      provider.export_semantics = "finite one-coordinate latent";
      provider.provenance = "large finite prediction stress fixture";
      provider.encode_visible = [](const embedding::Batch &batch) {
        return ev::ReconstructionLatent{torch::zeros({batch.data.size(0), 1, 1}, torch::kFloat64),
                                       torch::ones({batch.data.size(0), 1}, torch::kBool)};
      };
      provider.decode = [](const torch::Tensor &latent) {
        return torch::full({latent.size(0), 1, 8, 1}, 1e307, torch::kFloat64);
      };
      provider.metadata_predict = [](const torch::Tensor &visible) {
        return torch::zeros(visible.sizes(), torch::kFloat64);
      };
      return provider;
    });
  } catch (const std::exception &error) {
    rejected = true;
    const std::string message = error.what();
    test::check(message.find("summary mean overflow") != std::string::npos ||
                message.find("exchange-block sum overflow") != std::string::npos ||
                message.find("effect total overflow") != std::string::npos ||
                message.find("bootstrap sum overflow") != std::string::npos,
                "large finite fixture failed before exercising aggregate overflow: " + message);
  }
  test::check(rejected && !fs::exists(output / "report.json"),
              "large finite row losses generated a nonfinite measured report");
}
} // namespace
int main() {
  try {
    const auto temporary = fs::temp_directory_path() /
        ("embedding-reconstruction-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(temporary);
    validation();
    archive_and_effects(temporary);
    support_and_failures(temporary);
    large_finite_summary(temporary);
    std::cout << "shared reconstruction tests passed; artifacts=" << temporary << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

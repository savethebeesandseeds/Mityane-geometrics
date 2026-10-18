// SPDX-License-Identifier: MIT
#include "embedding/shared/feature_harness.h"
#include "shared_test_support.h"
#include <iostream>
#include <limits>
#include <set>

namespace ev = embedding::evaluation;
namespace {
template <typename F> void rejects(F &&function, const std::string &message) {
  bool rejected = false;
  try { function(); } catch (const std::exception &) { rejected = true; }
  test::check(rejected, message);
}
void fitting() {
  auto x = torch::tensor({{1., 0., 1.}, {0., 1., 1.}, {-1., 0., -1.},
                          {0., -1., -1.}, {9000., 9000., 9000.}}, torch::kFloat64);
  x[4][0] = std::numeric_limits<double>::quiet_NaN();
  ev::FeatureSurface train{x, torch::tensor({true,true,true,true,false}, torch::kBool), "training only"};
  const ev::FeatureNormalizer fit(train);
  test::check(fit.fitted_rows == 4, "normalizer counts invalid placeholders");
  test::close(fit.mean, torch::zeros({3}, torch::kFloat64), "invalid rows change fitted mean");
  const auto normalized = fit.transform(train);
  test::finite(normalized.values, "invalid nonfinite placeholders were not excluded");
  test::check(normalized.values[4].eq(0).all().item<bool>(), "invalid feature transform is not zero");
  const ev::TrainPca pca(normalized, 2);
  test::check(pca.fitted_rows == 4 && pca.numerical_rank == 2, "PCA rank/support is incorrect");
  rejects([&] { ev::TrainPca invalid(normalized, 3); }, "PCA permits zero-rank padding");
  rejects([&] { ev::TrainPca invalid({x, torch::tensor({true,true,false,false,false}, torch::kBool), ""}, 2); },
          "PCA uses total rows instead of valid fitted rows");
  const auto labels = torch::tensor({1,1,0,0,1}, torch::kInt64);
  const ev::RidgeProbe probe(pca.transform(normalized), labels);
  const auto mean_before = fit.mean.clone(), components_before = pca.components.clone();
  const auto weights_before = probe.weights.clone();
  ev::FeatureSurface future{torch::full({3,3}, 1e8, torch::kFloat64), torch::ones({3}, torch::kBool), "test"};
  (void)probe.predict(pca.transform(fit.transform(future)));
  future.values.fill_(-1e9);
  (void)probe.predict(pca.transform(fit.transform(future)));
  test::close(fit.mean, mean_before, "test observations changed fitted feature scaling", 0, 0);
  test::close(pca.components, components_before, "test observations changed training PCA", 0, 0);
  test::close(probe.weights, weights_before, "test observations changed frozen probe", 0, 0);
  auto invalid = train; invalid.valid = torch::ones_like(train.valid);
  rejects([&] { ev::FeatureNormalizer fail(invalid); }, "observed nonfinite feature was accepted");
  rejects([&] { ev::FeatureNormalizer fail({torch::tensor({-1e308,1e308},torch::kFloat64).view({2,1}),
      torch::ones({2},torch::kBool),"finite inputs with overflowing fit"}); },
      "nonfinite fitted feature statistics were retained");
}
void protocols() {
  embedding::input_shape_t shape{3,32,3};
  for (const auto task : {ev::Task::reversal, ev::Task::level, ev::Task::amplitude, ev::Task::lag_sign}) {
    const auto protocol = ev::make_controlled_protocol(task, shape, 8, 4, 6, 44, 0);
    const auto repeated = ev::make_controlled_protocol(task, shape, 8, 4, 6, 44, 0);
    test::close(protocol.training.clean.data, repeated.training.clean.data, "controlled fixture reproducibility", 0, 0);
    test::check(protocol.testing.source_ids == repeated.testing.source_ids, "source assignments changed");
    for (const auto *split : {&protocol.training, &protocol.validation, &protocol.testing}) {
      const auto oracle = ev::raw_oracle(task, *split);
      const auto result = ev::score(oracle.predictions, split->labels, oracle.valid);
      test::check(result.supported && result.coverage == 1 && result.accuracy == 1,
                  ev::task_name(task) + " is not analytically solvable on raw observations");
    }
    auto overlap = protocol;
    overlap.validation.source_ids[0] = overlap.training.source_ids[0];
    overlap.validation.source_ids[1] = overlap.training.source_ids[0];
    rejects([&] { ev::validate_protocol(overlap); }, "paired source overlaps were accepted");
  }
  const auto order = ev::make_controlled_protocol(ev::Task::reversal, shape, 8,4,6,44,0.2);
  const auto level = ev::make_controlled_protocol(ev::Task::level, shape, 8,4,6,44,0.2);
  test::check(torch::equal(order.training.observed.feature_mask, level.training.observed.feature_mask),
              "signal/task family changes missingness stream");
  const auto a = order.training.clean.data[0], b = order.training.clean.data[1];
  test::close(a.mean(), b.mean(), "reversal pair mean");
  test::close(a.var(false), b.var(false), "reversal pair variance");
}
uint64_t bytes_hash(const unsigned char *bytes, int64_t count) {
  uint64_t hash = 14695981039346656037ULL;
  for (int64_t i = 0; i < count; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
  return hash;
}
uint64_t tensor_hash(const torch::Tensor &tensor) {
  const auto cpu = tensor.to(torch::kCPU).contiguous();
  return bytes_hash(static_cast<const unsigned char *>(cpu.const_data_ptr()),
                    cpu.numel() * cpu.element_size());
}
uint64_t sources_hash(const ev::ControlledDataset &split) {
  std::string text;
  for (size_t i = 0; i < split.source_ids.size(); ++i) {
    if (i) text += '\n';
    text += split.source_ids[i] + ":" + std::to_string(split.labels[i].item<int64_t>());
  }
  return bytes_hash(reinterpret_cast<const unsigned char *>(text.data()), text.size());
}
void legacy_generator_regression() {
  // Captured from the unchanged seed101/level archived controlled-pairs-v2
  // fixture (rpb-evaluation-7go3jq) before the partial-generation refactor.
  // Exact hashes cover the pinned managed-container generator/runtime, including
  // source partition/order, variant labels, numeric observations and missingness.
  const auto protocol = ev::make_controlled_protocol(ev::Task::level,
      {3,32,3,torch::kFloat64,torch::kCPU}, 32, 16, 32, 101);
  const uint64_t values[]{0xc8031c0b51619373ULL, 0x59ce893d26f5369fULL, 0x650d0fc2d7a3afd2ULL};
  const uint64_t masks[]{0xb47ecc27082cdb21ULL, 0x7e48fee91c1a7ee9ULL, 0x9b539671a50ec39dULL};
  const uint64_t sources[]{0xc62e8dbf6a425247ULL, 0x1c13de8f09735dd7ULL, 0x82ef3f2b33577f77ULL};
  size_t index = 0;
  for (const auto *split : {&protocol.training, &protocol.validation, &protocol.testing}) {
    test::check(tensor_hash(split->observed.data) == values[index] &&
        tensor_hash(split->observed.feature_mask) == masks[index] &&
        sources_hash(*split) == sources[index], "legacy generator numeric/support/source fixture changed");
    ++index;
  }
}
void development_and_final_generation() {
  const embedding::input_shape_t shape{3,32,3,torch::kFloat64,torch::kCPU};
  for (const auto task : {ev::Task::reversal, ev::Task::level, ev::Task::amplitude, ev::Task::lag_sign}) {
    const auto development = ev::make_controlled_development_protocol(task, shape, 8, 4, 901);
    const auto repeated = ev::make_controlled_development_protocol(task, shape, 8, 4, 901);
    const auto &absent = development.testing;
    test::check(!absent.clean.data.defined() && !absent.clean.feature_mask.defined() &&
        !absent.observed.data.defined() && !absent.observed.feature_mask.defined() &&
        !absent.labels.defined() && absent.source_ids.empty(),
        "development generation created testing tensors or source identities");
    rejects([&] { ev::validate_protocol(development); },
            "legacy three-way validation accepted a missing test split");
    const auto fresh_seed = ev::stream_seed(901, 0x6375727665746573ULL);
    const auto testing = ev::make_controlled_test_dataset(task, shape, 6, fresh_seed);
    const auto repeated_testing = ev::make_controlled_test_dataset(task, shape, 6, fresh_seed);
    ev::ControlledProtocol combined{task, 901, shape, development.training, development.validation, testing};
    ev::validate_protocol(combined); // Includes source disjointness and both labels per source.
    test::check(development.training.labels.size(0) == 16 && development.validation.labels.size(0) == 8 &&
        testing.labels.size(0) == 12, "partial generator changed requested pair counts");
    const ev::ControlledDataset *splits[]{&development.training, &development.validation, &testing};
    const ev::ControlledDataset *replays[]{&repeated.training, &repeated.validation, &repeated_testing};
    for (size_t i = 0; i < 3; ++i) {
      const auto &split = *splits[i], &replay = *replays[i];
      test::check(split.source_ids == replay.source_ids && torch::equal(split.labels, replay.labels),
                  "partial generator changed source IDs or paired labels on replay");
      test::close(split.clean.data, replay.clean.data, "partial clean replay", 0, 0);
      test::close(split.observed.data, replay.observed.data, "partial observed replay", 0, 0);
      test::check(torch::equal(split.observed.feature_mask, replay.observed.feature_mask) &&
          split.clean.feature_mask.all().item<bool>() && split.observed.data.scalar_type() == torch::kFloat64,
          "partial generator changed support or raw precision");
      test::check(split.observed.data.masked_select(split.observed.feature_mask.logical_not()).eq(0).all().item<bool>(),
                  "partial generator exposed hidden observation storage");
      for (int64_t row = 0; row < split.labels.size(0); row += 2) {
        test::check(split.source_ids[row] == split.source_ids[row + 1] &&
            split.labels[row].item<int64_t>() != split.labels[row + 1].item<int64_t>() &&
            torch::equal(split.observed.feature_mask[row], split.observed.feature_mask[row + 1]),
            "partial generation separated paired source labels or corruptions");
      }
    }
    // Generating the final cohort cannot mutate the already retained development
    // observations or draw a reserved third split inside the development helper.
    const auto after = ev::make_controlled_development_protocol(task, shape, 8, 4, 901);
    test::close(after.training.observed.data, development.training.observed.data,
                "final generation changed development observations", 0, 0);
    test::check(after.training.source_ids == development.training.source_ids,
                "final generation changed development source assignments");
  }
  rejects([&] { ev::make_controlled_development_protocol(ev::Task::level, shape, 0, 4, 901); },
          "empty development training accepted");
  rejects([&] { ev::make_controlled_development_protocol(ev::Task::level, shape, 8, 0, 901); },
          "empty development validation accepted");
  rejects([&] { ev::make_controlled_test_dataset(ev::Task::level, shape, 0, 901); },
          "empty final testing accepted");
  rejects([&] { ev::make_controlled_development_protocol(ev::Task::level, shape, 8, 4, 901, 1); },
          "invalid development missingness accepted");
  rejects([&] { ev::make_controlled_test_dataset(ev::Task::level, shape, 6, 901,
                                               std::numeric_limits<double>::quiet_NaN()); },
          "nonfinite final missingness accepted");
  rejects([&] { ev::make_controlled_protocol(ev::Task::level, shape, 8, 4, 0, 901); },
          "historical generator accepted an empty third split");
  rejects([&] { ev::make_controlled_development_protocol(ev::Task::level, shape,
          std::numeric_limits<int64_t>::max(), 4, 901); }, "development pair count overflow accepted");
}
void input_precision() {
  auto x = torch::tensor({1e12,1e12+1.,1e12+2.,1e12+3.}, torch::kFloat64).view({2,1,2,1});
  embedding::Batch train{x, torch::ones_like(x, torch::kBool)};
  const ev::ObservationScaler scaler(train);
  const auto normalized = scaler.transform(train);
  test::check(normalized.data.std(false).item<double>() > 0.9, "precision was lost before centering");
  auto absent = train; absent.feature_mask = torch::zeros_like(train.feature_mask);
  absent.data = torch::full_like(x, std::numeric_limits<double>::quiet_NaN());
  test::check(scaler.transform(absent).data.eq(0).all().item<bool>(), "masked NaN enters preparation");
  auto test_data = train; test_data.data = x + 1e5;
  const auto mean_before = scaler.mean.clone(); (void)scaler.transform(test_data);
  test::close(scaler.mean, mean_before, "future data changes observation scaler", 0, 0);
}
void coordinate_deletion_views() {
  auto values = torch::arange(4 * 2 * 32 * 3, torch::kFloat64).reshape({4,2,32,3});
  auto mask = torch::ones_like(values, torch::kBool);
  mask.select(2, 0).fill_(false);
  values = values.masked_fill(mask.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch source{values, mask};
  const std::vector<std::string> ids{"source-a", "source-a", "source-b", "source-b"};
  const auto original_values = values.clone(), original_mask = mask.clone();
  const auto seed = ev::stream_seed(3101, 0x63766c3164726f70ULL);
  const std::string space = "context-optimization-validation-v1";
  torch::manual_seed(717); const auto expected_rng = torch::rand({8});
  torch::manual_seed(717);
  const auto view = ev::make_coordinate_deletion_view(source, ids, ev::Task::lag_sign, seed, .3, space);
  test::close(torch::rand({8}), expected_rng, "pure coordinate deletion consumed Torch RNG", 0, 0);
  const auto repeated = ev::make_coordinate_deletion_view(source, ids, ev::Task::lag_sign, seed, .3, space);
  test::check(torch::equal(view.requested_erasure, repeated.requested_erasure) &&
      torch::equal(view.observations.data, repeated.observations.data), "coordinate view replay changed");
  for (int64_t row = 0; row < 4; row += 2)
    test::check(torch::equal(view.requested_erasure[row], view.requested_erasure[row + 1]),
        "coordinate view exposed paired variant identity to missingness");
  test::check(!torch::equal(view.requested_erasure[0], view.requested_erasure[2]),
      "different source identities reused one coordinate mask");
  test::check(!view.observations.feature_mask.logical_and(mask.logical_not()).any().item<bool>() &&
      torch::equal(view.observations.feature_mask, mask.logical_and(view.requested_erasure.logical_not())),
      "coordinate deletion added observations or erased outside the requested view");
  test::close(view.observations.data.masked_select(view.observations.feature_mask),
      values.masked_select(view.observations.feature_mask), "coordinate deletion changed retained values", 0, 0);
  test::check(view.observations.data.masked_select(view.observations.feature_mask.logical_not()).eq(0).all().item<bool>(),
      "coordinate view retained hidden NaN storage");
  test::check(view.requested_erasure.masked_select(mask.logical_not()).any().item<bool>(),
      "coordinate mask did not index originally absent coordinates");
  const auto permutation = torch::tensor({2,0,3,1}, torch::kInt64);
  const embedding::Batch reordered{values.index_select(0, permutation), mask.index_select(0, permutation)};
  const std::vector<std::string> reordered_ids{ids[2],ids[0],ids[3],ids[1]};
  const auto reordered_view = ev::make_coordinate_deletion_view(reordered, reordered_ids,
      ev::Task::lag_sign, seed, .3, space);
  test::check(torch::equal(reordered_view.requested_erasure, view.requested_erasure.index_select(0, permutation)),
      "source coordinate masks depended on row order");
  auto previous = ev::make_coordinate_deletion_view(source, ids, ev::Task::lag_sign, seed, 0, space);
  test::check(previous.observations.feature_mask.equal(mask) && !previous.requested_erasure.any().item<bool>(),
      "zero deletion changed support");
  for (const double rate : {.1, .3, .6, .9, 1.}) {
    const auto next = ev::make_coordinate_deletion_view(source, ids, ev::Task::lag_sign, seed, rate, space);
    test::check(!previous.requested_erasure.logical_and(next.requested_erasure.logical_not()).any().item<bool>(),
        "coordinate severities did not share nested uniforms");
    previous = next;
  }
  test::check(previous.requested_erasure.all().item<bool>() &&
      !previous.observations.feature_mask.any().item<bool>() && previous.observations.data.eq(0).all().item<bool>(),
      "complete deletion acquired observation support");
  test::check(!torch::equal(view.requested_erasure,
      ev::make_coordinate_deletion_view(source, ids, ev::Task::lag_sign, seed + 1, .3, space).requested_erasure) &&
      !torch::equal(view.requested_erasure,
      ev::make_coordinate_deletion_view(source, ids, ev::Task::lag_sign, seed, .3, space + "/other").requested_erasure),
      "declared seed or namespace did not affect coordinate erasures");
  test::close(mask, original_mask, "coordinate view mutated original support", 0, 0);
  test::close(torch::isnan(values), torch::isnan(original_values), "coordinate view mutated hidden storage", 0, 0);
  test::close(values.masked_select(mask), original_values.masked_select(mask), "coordinate view mutated observed storage", 0, 0);
  view.observations.data.fill_(1); view.observations.feature_mask.fill_(false);
  test::close(mask, original_mask, "view storage aliases original support", 0, 0);
  test::close(values.masked_select(mask), original_values.masked_select(mask), "view storage aliases original values", 0, 0);
  rejects([&] { ev::make_coordinate_deletion_view(source, {"too-few"}, ev::Task::lag_sign, seed, .3, space); },
      "coordinate view accepted unequal source row count");
  auto empty_id = ids; empty_id[0].clear();
  rejects([&] { ev::make_coordinate_deletion_view(source, empty_id, ev::Task::lag_sign, seed, .3, space); },
      "coordinate view accepted empty source identity");
  for (const double rate : {-.1, 1.1, std::numeric_limits<double>::quiet_NaN()})
    rejects([&] { ev::make_coordinate_deletion_view(source, ids, ev::Task::lag_sign, seed, rate, space); },
        "coordinate view accepted invalid rate");
  rejects([&] { ev::make_coordinate_deletion_view(source, ids, static_cast<ev::Task>(999), seed, .3, space); },
      "coordinate view accepted unknown task");
  rejects([&] { ev::make_coordinate_deletion_view(source, ids, ev::Task::lag_sign, seed, .3, ""); },
      "coordinate view accepted empty namespace");
}
void diagnostic_parity() {
  auto x=torch::tensor({{1.,2.,0.,3.,-1.,1.},{0.,1.,3.,-1.,2.,0.},
      {-2.,0.,1.,2.,0.,-1.},{3.,-1.,2.,0.,1.,2.},{1e9,1e9,1e9,1e9,1e9,1e9}},torch::kFloat64);
  const auto valid=torch::tensor({true,true,true,true,false},torch::kBool);
  const auto legacy=embedding::representation_diagnostics(x,valid);
  const auto bounded=ev::feature_diagnostics({x,valid,"wide fixture"});
  test::check(bounded.valid_rows==4 && bounded.dimensions==6,"bounded diagnostics count invalid rows");
  test::close(bounded.per_dimension_std,legacy.per_dimension_std,"population std changed in Gram diagnostics",0,0);
  test::check(std::abs(bounded.norm_mean-legacy.norm_mean)<1e-12 &&
      std::abs(bounded.covariance_effective_rank-legacy.covariance_effective_rank)<1e-8,
      "small-Gram covariance spectrum changed legacy rank/norm semantics");
  const auto clean=ev::feature_diagnostics({x.narrow(0,0,4),torch::ones({4},torch::kBool),"same valid rows"});
  test::check(std::abs(clean.covariance_effective_rank-bounded.covariance_effective_rank)<1e-12,
      "invalid rows changed bounded covariance rank");
}
void coverage() {
  const auto labels = torch::tensor({0,1,1,0}, torch::kInt64);
  const auto predictions = torch::tensor({0,0,1,0}, torch::kInt64);
  const auto result = ev::score(predictions, labels, torch::tensor({true,false,true,false}, torch::kBool));
  test::check(result.total == 4 && result.valid == 2 && result.correct == 2 &&
              result.coverage == .5 && result.accuracy == 1, "coverage silently drops abstentions");
  const auto empty = ev::score(predictions, labels, torch::zeros({4}, torch::kBool));
  test::check(!empty.supported && empty.valid == 0 && empty.coverage == 0, "empty score is interpreted as a measurement");
  const std::vector<std::string> groups{"source-a","source-a","source-b","source-b"};
  const auto interval = ev::grouped_accuracy_interval(predictions, labels, torch::ones({4},torch::kBool),groups,17);
  test::check(interval.supported && interval.source_groups == 2 && interval.estimate == .75,
              "uncertainty treats paired variants as independent source groups");
  const auto paired = ev::grouped_accuracy_interval(predictions, labels, torch::ones({4},torch::kBool),groups,17,1000,predictions);
  test::check(paired.estimate == 0 && paired.lower == 0 && paired.upper == 0,
              "paired identical predictors have nonzero accuracy difference");
  const auto singleton = ev::grouped_accuracy_interval(predictions, labels,torch::tensor({true,true,false,false},torch::kBool),groups,17);
  test::check(!singleton.supported && singleton.source_groups == 1,"singleton group uncertainty is interpreted as supported");
}
}
int main() {
  try {
    torch::set_num_threads(1);
    fitting(); protocols(); legacy_generator_regression(); development_and_final_generation();
    input_precision(); coordinate_deletion_views(); coverage(); diagnostic_parity();
    std::cout << "feature harness tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}

// SPDX-License-Identifier: MIT
#include "embedding/shared/fixed_feature_readouts.h"
#include "shared_test_support.h"
#include <ATen/Context.h>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
namespace {
template <class F> void rejects(F &&function, const std::string &message) {
  bool failed = false;
  try { function(); } catch (const std::exception &) { failed = true; }
  test::check(failed, message);
}
torch::Tensor tensor(const fs::path &path, const std::string &key) {
  torch::serialize::InputArchive input;
  input.load_from(path.string(), torch::kCPU);
  torch::Tensor value;
  input.read(key, value, true);
  return value;
}
std::string read(const fs::path &path) {
  std::ifstream input(path);
  test::check(bool(input), "missing metadata " + path.string());
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}
struct RuntimeState {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeState() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  void unchanged() const {
    test::check(threads == at::get_num_threads(), "fixed readouts changed ambient thread count");
    for (size_t i = 0; i < states.size(); ++i)
      test::check(torch::equal(states[i], generators[i].get_state()), "fixed readouts changed ambient Torch RNG");
  }
};
ev::FixedFeatureReadoutRun fixture(const fs::path &path) {
  const auto x = torch::tensor({{-1.0,-1.0},{1.0,-1.0},{-1.0,0.0},{1.0,0.0},
      {-1.0,1.0},{1.0,1.0},{-1.0,2.0},{1.0,2.0}}, torch::kFloat64);
  const auto train_valid = torch::tensor({true,true,true,true,true,true,false,false}, torch::kBool);
  const auto intact_valid = torch::ones({8}, torch::kBool);
  const auto deleted_valid = train_valid.clone();
  auto train = x.clone();
  // Invalid storage must not affect valid-row TRAIN statistics or predictions.
  train.narrow(0,6,2).fill_(1e100);
  ev::FixedFeatureReadoutRun run;
  run.output_directory = path.string(); run.master_seed = 9109;
  run.training_labels = torch::tensor({0,1,0,1,0,1,0,1}, torch::kInt64);
  run.validation_labels = run.training_labels.clone();
  run.training_source_ids = {"t0","t0","t1","t1","t2","t2","t3","t3"};
  run.validation_source_ids = {"v0","v0","v1","v1","v2","v2","v3","v3"};
  run.methods = {
      {"native_v4", {torch::zeros_like(train), train_valid.clone(), "dummy constant TRAIN"},
       {torch::zeros_like(x), intact_valid.clone(), "dummy constant intact"},
       {torch::zeros_like(x), deleted_valid.clone(), "dummy constant deleted"}, false, {}},
      {"native_v7", {train.clone(), train_valid.clone(), "dummy linear TRAIN"},
       {x.clone(), intact_valid.clone(), "dummy linear intact"},
       {x.clone(), deleted_valid.clone(), "dummy linear deleted"}, false, {}},
      {"pca_only", {train.clone(), train_valid.clone(), "caller-prepared dummy components TRAIN"},
       {x.clone(), intact_valid.clone(), "caller-prepared dummy intact"},
       {x.clone(), deleted_valid.clone(), "caller-prepared dummy deleted"}, true, {}},
      {"unsupported", {train.clone(), torch::zeros({8}, torch::kBool), "dummy no TRAIN support"},
       {x.clone(), intact_valid.clone(), "dummy intact"},
       {x.clone(), deleted_valid.clone(), "dummy deleted"}, false, {}}};
  return run;
}
void equal_fit(const fs::path &a, const fs::path &b) {
  for (const std::string key : {"feature_mean","feature_scale","outer_normalizer_applied","outer_fitted_rows","fitted_rows",
      "ridge_mean","ridge_scale","ridge_weights","ridge_intercept","tiny_mean","tiny_scale","tiny_w1","tiny_b1","tiny_w2","tiny_b2",
      "actual_probe_seed_decimal","training_source_ids_json","ridge_penalty","tiny_hidden","tiny_steps","tiny_learning_rate"})
    test::check(torch::equal(tensor(a,key),tensor(b,key)), "held-out labels changed fitted state: " + key);
}
void prediction_arithmetic(const fs::path &directory, const std::string &view) {
  const auto fit = directory/"fit.pt", prediction = directory/(view+"-predictions.pt");
  const auto x = tensor(prediction,"probe_input_features");
  const auto ridge = ((x-tensor(fit,"ridge_mean"))/tensor(fit,"ridge_scale")).matmul(tensor(fit,"ridge_weights")) + tensor(fit,"ridge_intercept");
  const auto hidden = ((x-tensor(fit,"tiny_mean"))/tensor(fit,"tiny_scale")).matmul(tensor(fit,"tiny_w1")) + tensor(fit,"tiny_b1");
  const auto tiny = torch::tanh(hidden).matmul(tensor(fit,"tiny_w2")) + tensor(fit,"tiny_b2");
  for (const auto &key : {"ridge_logits","tiny_hidden_preactivation","tiny_logits"})
    test::check(tensor(prediction,key).scalar_type() == torch::kFloat64, "saved double arithmetic witness required");
  test::close(tensor(prediction,"ridge_logits"),ridge,"saved ridge arithmetic",1e-12,1e-12);
  test::close(tensor(prediction,"tiny_hidden_preactivation"),hidden,"saved tiny preactivation",1e-12,1e-12);
  test::close(tensor(prediction,"tiny_logits"),tiny,"saved tiny arithmetic",1e-12,1e-12);
  test::check(torch::equal(tensor(prediction,"ridge"),tensor(prediction,"ridge_logits").argmax(1)), "saved ridge class differs from its own logits");
  test::check(torch::equal(tensor(prediction,"tiny_secondary"),tensor(prediction,"tiny_logits").argmax(1)), "saved tiny class differs from its own logits");
}
void suite(const fs::path &root) {
  auto run = fixture(root/"ordinary");
  const auto original_train = run.methods[1].training.values.clone();
  const auto original_intact = run.methods[1].validation_intact.values.clone();
  const auto original_deleted = run.methods[1].validation_deleted.valid.clone();
  const RuntimeState runtime;
  const auto report = ev::run_fixed_feature_readouts(run);
  runtime.unchanged();
  test::check(report == read(root/"ordinary/report.json").substr(0,report.size()), "returned report differs from saved report");
  test::check(report.find("\"ridge_fits\":9,\"tiny_fits\":9,\"validation_fits\":0") != std::string::npos,
      "fixed fit counts must exclude both VALIDATION views and unsupported method");
  test::check(report.find("\"status\":\"unsupported_fit\"") != std::string::npos &&
      !fs::exists(root/"ordinary/unsupported/rep-2701/fit.pt"), "unsupported TRAIN fit lost or constructed a head");
  test::check(torch::equal(run.methods[1].training.values,original_train) &&
      torch::equal(run.methods[1].validation_intact.values,original_intact) &&
      torch::equal(run.methods[1].validation_deleted.valid,original_deleted), "scoring mutated caller-owned tensors");
  const auto fit = root/"ordinary/native_v7/rep-2701/fit.pt";
  test::close(tensor(fit,"feature_mean"),torch::zeros({2},torch::kFloat64),"valid TRAIN mean excludes invalid storage",0,0);
  test::close(tensor(fit,"feature_scale"),torch::tensor({1.0,std::sqrt(2.0/3.0)},torch::kFloat64),"valid TRAIN population std",1e-12,1e-12);
  test::check(tensor(fit,"outer_fitted_rows").item<int64_t>() == 6 && tensor(fit,"fitted_rows").item<int64_t>() == 6,
      "outer and probe normalizers must fit valid TRAIN rows only");
  test::check(embedding::archive::tensor_text(tensor(fit,"actual_probe_seed_decimal")) == std::to_string(ev::stream_seed(2701,2)),
      "fixed head seed provenance differs from declared width-paired recipe");
  const auto pca_fit = root/"ordinary/pca_only/rep-2701/fit.pt";
  test::check(!tensor(pca_fit,"outer_normalizer_applied").item<bool>() && tensor(pca_fit,"outer_fitted_rows").item<int64_t>() == 0,
      "caller-prepared raw PCA received a redundant outer fit");
  test::close(tensor(pca_fit,"feature_mean"),torch::zeros({2},torch::kFloat64),"prepared identity mean",0,0);
  test::close(tensor(pca_fit,"feature_scale"),torch::ones({2},torch::kFloat64),"prepared identity scale",0,0);
  test::check(read(root/"ordinary/native_v7/fit-counts.json").find("\"ridge_fits\":3,\"tiny_fits\":3,\"validation_fits\":0") != std::string::npos,
      "one head per declared repetition must serve both views");
  const auto deleted = root/"ordinary/native_v7/rep-2701/validation-deleted-predictions.pt";
  const auto intact = root/"ordinary/native_v7/rep-2701/validation-intact-predictions.pt";
  test::check(ev::score(tensor(intact,"ridge"),run.validation_labels,tensor(intact,"valid")).accuracy == 1,
      "separable held-out linear fixture failed");
  test::check(ev::score(tensor(deleted,"ridge"),run.validation_labels,tensor(deleted,"valid")).valid == 6,
      "deleted-view support changed the declared denominator");
  const auto constant = tensor(root/"ordinary/native_v4/rep-2701/validation-deleted-predictions.pt","ridge");
  const auto effect = ev::grouped_accuracy_interval(tensor(deleted,"ridge"),run.validation_labels,tensor(deleted,"valid"),
      run.validation_source_ids,123,1000,constant);
  test::check(effect.supported && effect.source_groups == 3 && effect.estimate == .5 && effect.lower == .5 && effect.upper == .5,
      "paired source effect must use the common three groups, unaffected by unsupported method");
  for (const std::string method : {"native_v4","native_v7","pca_only"})
    for (const std::string rep : {"rep-2701","rep-2802","rep-2903"})
      for (const std::string view : {"training","validation-intact","validation-deleted"})
        prediction_arithmetic(root/"ordinary"/method/rep,view);
  // Flipping held-out labels changes scores but cannot change any fitted map,
  // probe or prediction. The same fixed training fits also serve deleted rows.
  auto changed = fixture(root/"heldout-label-swap");
  changed.validation_labels = 1-changed.validation_labels;
  ev::run_fixed_feature_readouts(changed); runtime.unchanged();
  for (const std::string method : {"native_v4","native_v7","pca_only"})
    for (const std::string rep : {"rep-2701","rep-2802","rep-2903"}) {
      const auto left = root/"ordinary"/method/rep, right = root/"heldout-label-swap"/method/rep;
      equal_fit(left/"fit.pt",right/"fit.pt");
      for (const std::string view : {"training","validation-intact","validation-deleted"})
        for (const std::string key : {"ridge","tiny_secondary","valid","probe_input_features","ridge_logits","tiny_hidden_preactivation","tiny_logits"})
          test::check(torch::equal(tensor(left/(view+"-predictions.pt"),key),tensor(right/(view+"-predictions.pt"),key)),
              "held-out labels influenced fitted prediction: " + method + '/' + view + '/' + key);
    }
  auto generic = fixture(root/"generic-pair");
  for (auto &method : generic.methods) {
    if (method.name == "native_v4") method.name = "reference";
    if (method.name == "native_v7") method.name = "candidate";
  }
  generic.comparison_reference = "reference"; generic.comparison_candidate = "candidate";
  const auto generic_report = ev::run_fixed_feature_readouts(generic); runtime.unchanged();
  test::check(generic_report.find("\"id\":\"candidate_minus_reference\"") != std::string::npos &&
      generic_report.find("native_v7_minus_native_v4") == std::string::npos,
      "generic comparison retains an encoder-specific pair name");
  for (const auto &names : std::vector<std::pair<std::string,std::string>>{{"native_v4","reference"},{"native_v7","candidate"}})
    for (const std::string rep : {"rep-2701","rep-2802","rep-2903"}) {
      const auto left = root/"ordinary"/names.first/rep, right = root/"generic-pair"/names.second/rep;
      equal_fit(left/"fit.pt",right/"fit.pt");
      for (const std::string view : {"training","validation-intact","validation-deleted"})
        for (const std::string key : {"ridge","tiny_secondary","valid","probe_input_features","ridge_logits","tiny_hidden_preactivation","tiny_logits"})
          test::check(torch::equal(tensor(left/(view+"-predictions.pt"),key),tensor(right/(view+"-predictions.pt"),key)),
              "generic pair naming changed fit/prediction arithmetic");
    }
  auto multiple = fixture(root/"multiple-pairs");
  multiple.comparisons = {{"native_v4","native_v7"},{"native_v4","pca_only"},{"pca_only","native_v7"}};
  const auto multiple_report = ev::run_fixed_feature_readouts(multiple); runtime.unchanged();
  test::check(multiple_report.find("\"ridge_fits\":9,\"tiny_fits\":9,\"validation_fits\":0") != std::string::npos,
      "multiple comparisons redundantly fitted heads");
  for (const std::string id : {"native_v7_minus_native_v4","pca_only_minus_native_v4","native_v7_minus_pca_only"}) {
    size_t position = 0, count = 0;
    while ((position = multiple_report.find("\"id\":\""+id+"\"", position)) != std::string::npos) { ++count; ++position; }
    test::check(count == 6, "each pair must retain three repetitions and both views");
  }
  for (const std::string method : {"native_v4","native_v7","pca_only"})
    for (const std::string rep : {"rep-2701","rep-2802","rep-2903"}) {
      const auto left = root/"ordinary"/method/rep, right = root/"multiple-pairs"/method/rep;
      equal_fit(left/"fit.pt",right/"fit.pt");
      for (const std::string view : {"training","validation-intact","validation-deleted"})
        for (const std::string key : {"ridge","tiny_secondary","valid","probe_input_features","ridge_logits","tiny_hidden_preactivation","tiny_logits"})
          test::check(torch::equal(tensor(left/(view+"-predictions.pt"),key),tensor(right/(view+"-predictions.pt"),key)),
              "adding comparisons changed a fitted prediction");
    }
  auto bad_multiple = fixture(root/"bad-multiple");
  bad_multiple.comparisons = {{"native_v4","native_v7"},{"native_v4","native_v7"}};
  rejects([&]{ ev::run_fixed_feature_readouts(bad_multiple); },"duplicate multiple pair admitted");
  bad_multiple.comparisons[1] = {"native_v4","missing"};
  rejects([&]{ ev::run_fixed_feature_readouts(bad_multiple); },"undeclared multiple pair admitted");
  bad_multiple.comparisons[1] = {"native_v4","native_v4"};
  rejects([&]{ ev::run_fixed_feature_readouts(bad_multiple); },"self multiple pair admitted");
  bad_multiple.comparisons.resize(1); bad_multiple.comparison_candidate = "native_v7";
  rejects([&]{ ev::run_fixed_feature_readouts(bad_multiple); },"mixed single/multiple fields admitted");
  test::check(!fs::exists(root/"bad-multiple"),"invalid multiple pair created output before validation");

  auto raw_unprepared = fixture(root/"raw-unprepared"); raw_unprepared.methods[1].name = "raw";
  ev::run_fixed_feature_readouts(raw_unprepared); runtime.unchanged();
  auto raw_prepared = fixture(root/"raw-prepared"); raw_prepared.methods[1].name = "raw";
  auto &raw = raw_prepared.methods[1];
  const ev::FeatureNormalizer shared_raw_outer(raw.training);
  raw.training = shared_raw_outer.transform(raw.training);
  raw.validation_intact = shared_raw_outer.transform(raw.validation_intact);
  raw.validation_deleted = shared_raw_outer.transform(raw.validation_deleted);
  raw.inputs_train_prepared = true;
  ev::run_fixed_feature_readouts(raw_prepared); runtime.unchanged();
  for (const std::string rep : {"rep-2701","rep-2802","rep-2903"}) {
    const auto left = root/"raw-unprepared/raw"/rep, right = root/"raw-prepared/raw"/rep;
    test::check(!tensor(right/"fit.pt","outer_normalizer_applied").item<bool>() &&
        tensor(right/"fit.pt","outer_fitted_rows").item<int64_t>() == 0,
        "caller-normalized raw redundantly fitted an outer map");
    for (const std::string key : {"fitted_rows","ridge_mean","ridge_scale","ridge_weights","ridge_intercept",
        "tiny_mean","tiny_scale","tiny_w1","tiny_b1","tiny_w2","tiny_b2","actual_probe_seed_decimal"})
      test::check(torch::equal(tensor(left/"fit.pt",key),tensor(right/"fit.pt",key)),
          "sharing the raw outer map changed a fitted probe");
    for (const std::string view : {"training","validation-intact","validation-deleted"})
      for (const std::string key : {"ridge","tiny_secondary","valid","probe_input_features","ridge_logits","tiny_hidden_preactivation","tiny_logits"})
        test::check(torch::equal(tensor(left/(view+"-predictions.pt"),key),tensor(right/(view+"-predictions.pt"),key)),
            "sharing the raw outer map changed a prediction");
  }
  auto missing_pair = fixture(root/"missing-pair"); missing_pair.comparison_reference = "native_v4";
  rejects([&]{ ev::run_fixed_feature_readouts(missing_pair); },"one-sided comparison admitted");
  test::check(!fs::exists(root/"missing-pair"),"invalid pair created output before validation");
  missing_pair.comparison_candidate = "missing";
  rejects([&]{ ev::run_fixed_feature_readouts(missing_pair); },"undeclared comparison admitted");
  missing_pair.comparison_candidate = "native_v4";
  rejects([&]{ ev::run_fixed_feature_readouts(missing_pair); },"self comparison admitted");
  rejects([&]{ ev::run_fixed_feature_readouts(run); },"existing result directory replaced");
  auto invalid = fixture(root/"leakage"); invalid.validation_source_ids[7] = invalid.training_source_ids[0];
  rejects([&]{ ev::run_fixed_feature_readouts(invalid); },"source leakage admitted");
  test::check(!fs::exists(root/"leakage"),"source leakage created output before validation");
  invalid = fixture(root/"unsafe"); invalid.methods[0].name = "../unsafe";
  rejects([&]{ ev::run_fixed_feature_readouts(invalid); },"unsafe name admitted");
  invalid = fixture(root/"duplicate"); invalid.methods[0].name = invalid.methods[1].name;
  rejects([&]{ ev::run_fixed_feature_readouts(invalid); },"duplicate method admitted");
  invalid = fixture(root/"shape"); invalid.methods[0].validation_deleted.values = torch::zeros({8,3},torch::kFloat64);
  rejects([&]{ ev::run_fixed_feature_readouts(invalid); },"split width changed");
  invalid = fixture(root/"nonfinite"); invalid.methods[1].validation_intact.values[0][0].fill_(std::numeric_limits<double>::quiet_NaN());
  rejects([&]{ ev::run_fixed_feature_readouts(invalid); },"valid nonfinite feature admitted");
  invalid = fixture(root/"native-prepared"); invalid.methods[1].inputs_train_prepared = true;
  rejects([&]{ ev::run_fixed_feature_readouts(invalid); },"native representation bypassed its declared outer normalizer");
  runtime.unchanged();
}
} // namespace
int main() {
  const auto root = fs::temp_directory_path()/
      ("fixed-feature-readouts-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    test::check(fs::create_directory(root),"exclusive readout fixture directory required");
    suite(root); fs::remove_all(root);
    std::cout << "Fixed feature readout tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\nRetained test fixture: " << root << '\n';
    return 1;
  }
}

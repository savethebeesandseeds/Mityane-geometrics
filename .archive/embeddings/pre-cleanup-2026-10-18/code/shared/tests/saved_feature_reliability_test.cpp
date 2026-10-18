// SPDX-License-Identifier: MIT
#include "embedding/shared/saved_feature_reliability.h"
#include "shared_test_support.h"
#include <ATen/Context.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unistd.h>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
namespace {
template<class F> void rejects(F &&function, const std::string &reason) {
  bool failed = false;
  try { function(); } catch (const std::exception &) { failed = true; }
  test::check(failed, reason);
}
void write(const fs::path &path, const std::string &value) {
  std::ofstream out(path); out << value; out.close(); test::check(bool(out), "fixture write failed");
}
torch::Tensor get(const fs::path &path, const std::string &key) {
  torch::serialize::InputArchive in; in.load_from(path.string(), torch::kCPU);
  torch::Tensor value; in.read(key, value, true); return value;
}
std::string sources(const std::vector<std::string> &ids) {
  std::ostringstream out; out << '[';
  for (size_t i = 0; i < ids.size(); ++i) { if (i) out << ','; out << '"' << ids[i] << '"'; }
  return out.str() + ']';
}
void capture_saved(ev::SavedFeatureAnalysis &value) {
  const auto &f = value.fit;
  const auto x = value.features.to(torch::kFloat64), mask = value.valid.unsqueeze(1);
  const auto outer = torch::where(mask, (x - f.feature_mean) / f.feature_scale, torch::zeros_like(x));
  const auto ridge = torch::where(mask, (outer - f.ridge_mean) / f.ridge_scale, torch::zeros_like(x));
  const auto tiny = torch::where(mask, (outer - f.tiny_mean) / f.tiny_scale, torch::zeros_like(x));
  const auto logits = ridge.matmul(f.ridge_weights) + f.ridge_intercept;
  const auto hidden = tiny.matmul(f.tiny_w1) + f.tiny_b1;
  const auto nonlinear = torch::tanh(hidden).matmul(f.tiny_w2) + f.tiny_b2;
  value.saved = {value.valid.clone(), logits.argmax(1), nonlinear.argmax(1), outer, logits, hidden, nonlinear};
}
ev::SavedFeatureAnalysis fixture(double stretch = 1) {
  ev::SavedFeatureAnalysis value;
  value.features = torch::tensor({{-1.,2.,5.},{1.,2.,5.},{-2.,4.,5.},{2.,4.,5.},
      {-3.,6.,5.},{3.,6.,5.},{-4.,8.,5.},{4.,8.,5.}}, torch::kFloat32);
  value.features.select(1, 0).mul_(stretch);
  value.valid = torch::ones({8}, torch::kBool);
  value.labels = torch::tensor({0,1,0,1,0,1,0,1}, torch::kInt64);
  value.source_ids = {"d","d","a","a","c","c","b","b"};
  auto &fit = value.fit;
  // Artificial retained maps and known head tensors; no fitting constructor.
  const auto x = value.features.to(torch::kFloat64);
  fit.feature_mean = x.mean(0); fit.feature_scale = (x - fit.feature_mean).pow(2).mean(0).sqrt().clamp_min(1e-8);
  const auto outer = (x - fit.feature_mean) / fit.feature_scale;
  fit.ridge_mean = outer.mean(0); fit.ridge_scale = (outer - fit.ridge_mean).pow(2).mean(0).sqrt().clamp_min(1e-8);
  fit.tiny_mean = fit.ridge_mean.clone(); fit.tiny_scale = fit.ridge_scale.clone();
  fit.ridge_weights = torch::tensor({{-.5,.5},{0.,0.},{0.,0.}}, torch::kFloat64);
  fit.ridge_intercept = torch::zeros({2}, torch::kFloat64);
  fit.tiny_w1 = torch::zeros({3,16}, torch::kFloat64); fit.tiny_w1[0][0] = .75;
  fit.tiny_b1 = torch::zeros({16}, torch::kFloat64);
  fit.tiny_w2 = torch::zeros({16,2}, torch::kFloat64); fit.tiny_w2[0][0] = -1.; fit.tiny_w2[0][1] = 1.;
  fit.tiny_b2 = torch::zeros({2}, torch::kFloat64); capture_saved(value); return value;
}
void pure_arithmetic() {
  auto value = fixture(); const auto before = value.features.clone(), mean = value.fit.feature_mean.clone();
  const auto result = ev::replay_saved_feature_analysis(value);
  test::check(result.pair_source_ids == std::vector<std::string>({"a","b","c","d"}), "source pairs must sort lexically without label lookup reordering");
  test::check(torch::equal(result.pair_row0, torch::tensor({2,6,4,0}, torch::kInt64)), "label0 row association");
  test::close(result.served_pair_distance, torch::tensor({4.,8.,6.,2.}, torch::kFloat64), "physical pair distances", 0, 0);
  test::close(result.served_pair_midpoint_norm, torch::tensor({std::sqrt(41.),std::sqrt(89.),std::sqrt(61.),std::sqrt(29.)}, torch::kFloat64), "pair midpoint norms", 1e-14, 1e-14);
  test::close(result.served_geometry.coordinate_mean, torch::tensor({0.,5.,5.}, torch::kFloat64), "served mean", 0, 0);
  test::close(result.served_geometry.population_sd, torch::tensor({std::sqrt(7.5),std::sqrt(5.),0.}, torch::kFloat64), "population SD", 1e-14, 1e-14);
  test::check(result.served_geometry.pooled_label_mean_distance == 5., "pooled label distance differs from individual source distance");
  test::close(result.served_geometry.centered_row_norm, torch::tensor({std::sqrt(10.),std::sqrt(10.),std::sqrt(5.),std::sqrt(5.),std::sqrt(10.),std::sqrt(10.),5.,5.}, torch::kFloat64), "centered row norms", 1e-14, 1e-14);
  test::check(result.json.find("\"constant_coordinates\":1") != std::string::npos && result.json.find("\"ridge_positive_order_fraction\":1") != std::string::npos, "constant and strict pair sign semantics");
  test::check(torch::equal(value.features, before) && torch::equal(value.fit.feature_mean, mean), "replay mutated supplied evidence");
  test::close(result.ridge_signed_margin, value.labels.to(torch::kFloat64).mul(2).sub(1) * (value.saved.ridge_logits.select(1,1)-value.saved.ridge_logits.select(1,0)), "true-class margin", 0, 0);
  // A retained near-tie may differ from another legal summation's argmax.
  auto tie = fixture(); tie.fit.ridge_weights.zero_(); capture_saved(tie);
  tie.saved.ridge_logits[1][1] = 1e-10; tie.saved.ridge = tie.saved.ridge_logits.argmax(1);
  const auto tied = ev::replay_saved_feature_analysis(tie);
  test::check(tie.saved.ridge[1].item<int64_t>() == 1 && tied.ridge_logits.argmax(1)[1].item<int64_t>() == 0,
              "near-tie fixture must separate saved and replayed class decisions");
  test::check(tied.ridge_signed_margin[1].item<double>() == 1e-10 && tied.ridge_pair_order_margin[3].item<double>() == 1e-10,
              "margins and strict pair signs must use retained saved logits");
  auto bad_class = tie; bad_class.saved.ridge = tie.saved.ridge.clone(); bad_class.saved.ridge[1] = 0;
  rejects([&] { ev::replay_saved_feature_analysis(bad_class); }, "saved own-logit class corruption was accepted");
  auto wrong_logit = fixture(); wrong_logit.saved.ridge_logits = wrong_logit.saved.ridge_logits + .01;
  rejects([&] { ev::replay_saved_feature_analysis(wrong_logit); }, "changed frozen head arithmetic was accepted");
  auto duplicate = fixture(); duplicate.source_ids[2] = "d";
  rejects([&] { ev::replay_saved_feature_analysis(duplicate); }, "duplicate source-label role was accepted");
  auto negative_scale = fixture(); negative_scale.fit.feature_scale[0] = -1;
  rejects([&] { ev::replay_saved_feature_analysis(negative_scale); }, "negative saved scale was accepted");
  auto nonfinite = fixture(); nonfinite.fit.ridge_weights[0][0] = std::numeric_limits<double>::infinity();
  rejects([&] { ev::replay_saved_feature_analysis(nonfinite); }, "nonfinite saved head was accepted");
  // Generic primitives preserve excluded support, while the production file
  // protocol separately requires complete retained TRAIN support.
  auto missing = fixture(); missing.valid.narrow(0,6,2).fill_(false); missing.features.narrow(0,6,2).fill_(1e30); capture_saved(missing);
  const auto partial = ev::replay_saved_feature_analysis(missing);
  test::check(partial.pair_valid.sum().item<int64_t>() == 3 && partial.served_geometry.coordinate_mean[1].item<double>() == 4., "invalid storage influenced valid-only geometry");
  auto absent = fixture(); absent.valid.zero_(); capture_saved(absent);
  const auto empty = ev::replay_saved_feature_analysis(absent);
  test::check(!empty.pair_valid.any().item<bool>() && empty.json.find("\"pooled_label_mean_distance\":null") != std::string::npos,
              "undefined population must remain null, not repaired or divided by epsilon");
}
ev::SavedTrainingTrace trace(int64_t steps = 512, int64_t batch = 8) {
  ev::SavedTrainingTrace out; out.attempted = out.completed = steps; out.batch_size = batch; out.sampled_rows = steps * batch;
  for (int64_t i = 1; i <= steps; ++i) out.rows.push_back({i,i,32,double(i)/512.,double(steps-i)/512.});
  return out;
}
std::string trace_metadata(const ev::SavedTrainingTrace &value, int64_t parameters) {
  std::ostringstream out; out << std::setprecision(17) << "{\"attempted\":" << value.attempted << ",\"completed\":" << value.completed
      << ",\"sampled_rows\":" << value.sampled_rows << ",\"parameter_count\":" << parameters << ",\"cuda_parameter_count\":" << parameters
      << ",\"training_device\":\"cuda\",\"last_input_cuda\":true,\"last_loss_cuda\":true,\"finite_gradients\":true,\"weights_changed\":true,\"losses\":[";
  for (size_t i = 0; i < value.rows.size(); ++i) { if (i) out << ','; const auto &r = value.rows[i];
    out << '[' << r.attempted << ',' << r.completed << ',' << r.target_cells << ',' << r.loss << ',' << r.gradient_norm << ']'; }
  return out.str() + "]}";
}
void trace_arithmetic() {
  const auto value = trace(); const auto json = ev::saved_training_trace_json(value);
  test::check(json.find("\"first_absolute_update\":1,\"last_absolute_update\":128") != std::string::npos &&
              json.find("\"first_absolute_update\":385,\"last_absolute_update\":512") != std::string::npos &&
              json.find("\"mean\":0.1259765625") != std::string::npos, "fixed absolute trace blocks and arithmetic mean");
  test::check(json.find("\"population_sd\":") != std::string::npos && json.find("\"minimum\":0.001953125") != std::string::npos,
              "trace distributions need full predeclared statistics");
  auto skip = value; skip.attempted++;
  rejects([&] { ev::saved_training_trace_json(skip); }, "extra attempted/skipped row was accepted");
  auto omitted = value; omitted.rows.erase(omitted.rows.begin()+100);
  rejects([&] { ev::saved_training_trace_json(omitted); }, "missing trace update was accepted");
  auto shifted = value; shifted.rows[128].completed = 128;
  rejects([&] { ev::saved_training_trace_json(shifted); }, "non-prefix trace was accepted");
  auto nan = value; nan.rows[0].loss = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { ev::saved_training_trace_json(nan); }, "nonfinite trace was accepted");
}
uint64_t seed(uint64_t value, uint64_t width) {
  value += 0x9e3779b97f4a7c15ULL * width;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL; value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
void fit_archive(const fs::path &path, const ev::SavedFeatureAnalysis &input, uint64_t rep) {
  torch::serialize::OutputArchive out; const auto &f = input.fit;
  for (const auto &[key, value] : std::vector<std::pair<std::string,torch::Tensor>>{
      {"feature_mean",f.feature_mean},{"feature_scale",f.feature_scale},{"ridge_mean",f.ridge_mean},{"ridge_scale",f.ridge_scale},
      {"ridge_weights",f.ridge_weights},{"ridge_intercept",f.ridge_intercept},{"tiny_mean",f.tiny_mean},{"tiny_scale",f.tiny_scale},
      {"tiny_w1",f.tiny_w1},{"tiny_b1",f.tiny_b1},{"tiny_w2",f.tiny_w2},{"tiny_b2",f.tiny_b2}}) out.write(key,value,true);
  out.write("outer_normalizer_applied",torch::tensor(true,torch::kBool),true);
  out.write("outer_fitted_rows",torch::tensor(int64_t(8)),true); out.write("fitted_rows",torch::tensor(int64_t(8)),true);
  out.write("actual_probe_seed_decimal",embedding::archive::text_tensor(std::to_string(seed(rep,3))),true);
  out.write("training_source_ids_json",embedding::archive::text_tensor(sources(input.source_ids)),true);
  out.write("ridge_penalty",torch::tensor(1.,torch::kFloat64),true); out.write("tiny_hidden",torch::tensor(int64_t(16)),true);
  out.write("tiny_steps",torch::tensor(int64_t(100)),true); out.write("tiny_learning_rate",torch::tensor(.01,torch::kFloat64),true);
  embedding::archive::save_archive(path.string(),out);
}
void prediction_archive(const fs::path &path, const ev::SavedFeatureAnalysis &input) {
  torch::serialize::OutputArchive out; const auto &s = input.saved;
  for (const auto &[key,value] : std::vector<std::pair<std::string,torch::Tensor>>{{"valid",s.valid},{"ridge",s.ridge},{"tiny_secondary",s.tiny},
      {"probe_input_features",s.probe_input},{"ridge_logits",s.ridge_logits},{"tiny_hidden_preactivation",s.tiny_hidden_preactivation},{"tiny_logits",s.tiny_logits}}) out.write(key,value,true);
  out.write("labels_scoring_only",input.labels,true); out.write("source_ids_json",embedding::archive::text_tensor(sources(input.source_ids)),true);
  embedding::archive::save_archive(path.string(),out);
}
void archive_pipeline(const fs::path &root) {
  fs::create_directories(root); ev::SavedFeatureReliabilityRun run;
  run.output_directory = (root/"analysis").string(); run.source_fingerprint = "synthetic source"; run.card_sha256 = "synthetic card";
  run.training_rows = 8; run.native_width = 3; run.expected_updates = 8; run.batch_size = 2; run.expected_parameter_count = 17;
  run.shape = {1,2,1,torch::kFloat64,torch::kCPU};
  for (int model = 0; model < 2; ++model) {
    const auto directory = root/(model ? "candidate" : "reference"); fs::create_directory(directory);
    const auto input = fixture(model ? 2 : 1);
    torch::serialize::OutputArchive observed, feature;
    observed.write("observations",torch::ones({8,1,2,1},torch::kFloat64),true); observed.write("feature_mask",torch::ones({8,1,2,1},torch::kBool),true);
    observed.write("labels_scoring_only",input.labels,true); observed.write("source_ids_json",embedding::archive::text_tensor(sources(input.source_ids)),true);
    embedding::archive::save_archive((directory/"observations.pt").string(),observed);
    feature.write("features",input.features,true); feature.write("valid",input.valid,true); feature.write("labels_scoring_only",input.labels,true);
    feature.write("source_ids_json",embedding::archive::text_tensor(sources(input.source_ids)),true); feature.write("provenance",embedding::archive::text_tensor("synthetic TRAIN export"),true);
    embedding::archive::save_archive((directory/"features.pt").string(),feature);
    write(directory/"progress.json",trace_metadata(trace(8,2),17));
    ev::SavedFeatureReliabilityInput item;
    item.id = model ? "candidate" : "reference"; item.display_tag = "synthetic"; item.master_seed = 42;
    item.controlled_training_path = (directory/"observations.pt").string(); item.native_training_path = (directory/"features.pt").string(); item.encoder_progress_path = (directory/"progress.json").string();
    for (size_t r=0;r<3;++r) {
      const uint64_t rep = std::array<uint64_t,3>{2701,2802,2903}[r];
      item.fit_paths[r] = (directory/ ("fit-"+std::to_string(rep)+".pt")).string();
      item.training_prediction_paths[r] = (directory/("prediction-"+std::to_string(rep)+".pt")).string();
      fit_archive(item.fit_paths[r],input,rep); prediction_archive(item.training_prediction_paths[r],input);
    }
    run.inputs.push_back(item);
  }
  run.comparisons = {{"paired","candidate","reference"}};
  const auto observed_before = get(run.inputs[0].controlled_training_path,"observations");
  const auto fit_before = get(run.inputs[1].fit_paths[0],"tiny_w1");
  const auto json = ev::run_saved_feature_reliability(run);
  test::check(json.find("\"head_refits\":0") != std::string::npos && json.find("\"heldout_analysis_payloads\":0") != std::string::npos,
              "saved-file pipeline changed the zero-fit/held-out scope");
  test::close(get(root/"analysis/paired/rep-2701.pt","served_distance_delta"),torch::tensor({4.,8.,6.,2.},torch::kFloat64),"paired geometry retains exact source populations",0,0);
  test::close(get(root/"analysis/paired/rep-2701.pt","outer_distance_delta"),torch::zeros({4},torch::kFloat64),"separate served vs saved-normalized scale",1e-14,1e-14);
  test::check(torch::equal(get(root/"analysis/reference/rep-2701/analysis.pt","served_centered_row_norm"),ev::replay_saved_feature_analysis(fixture()).served_geometry.centered_row_norm),"persisted geometry schema");
  test::check(torch::equal(observed_before,get(run.inputs[0].controlled_training_path,"observations")) && torch::equal(fit_before,get(run.inputs[1].fit_paths[0],"tiny_w1")),"file pipeline altered prior evidence");
  rejects([&] { ev::run_saved_feature_reliability(run); }, "existing output was overwritten");
  auto wrong_map = run; wrong_map.output_directory = (root/"wrong-source").string();
  auto invalid = fixture(); invalid.source_ids[0] = "changed"; fit_archive(run.inputs[0].fit_paths[0],invalid,2701);
  rejects([&] { ev::run_saved_feature_reliability(wrong_map); }, "mismatched fit TRAIN source order was accepted");
}
} // namespace
int main() try {
  const auto generator = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  const auto rng = generator.get_state().clone(); const int threads = at::get_num_threads();
  pure_arithmetic(); trace_arithmetic();
  const auto directory = fs::temp_directory_path() / ("saved-feature-reliability-"+std::to_string(::getpid())+'-'+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  archive_pipeline(directory);
  test::check(torch::equal(rng,generator.get_state()) && threads == at::get_num_threads(), "saved analysis changed ambient CPU RNG/thread state");
  std::cout << "Saved native reliability CPU fixtures passed\n" << EVALUATION_SOURCE_ID << '\n'; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }

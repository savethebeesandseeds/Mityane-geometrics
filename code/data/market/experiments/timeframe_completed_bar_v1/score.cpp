// Copyright (c) 2026 Waajacu. See LICENSE at repository root.
// Absolute ridge readout for the timeframe-channel screen.
// Each task is fit only on rows that past_window_targets marks eligible.
#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/family/family.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/preprocessing.h"
#include "embedding/data/observation_windows.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace cp = embedding::encoders::causal_patch_bottleneck_mae::family;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace rfamily = rpb::family;
namespace ow = embedding::data::observation_windows;
namespace ev = cuwacunu::evaluation;
namespace fs = std::filesystem;

namespace {
std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}
cp::RawHistory causal_raw(const ow::Dataset &data) {
  return {data.values, data.observed, data.channel_ids, data.timestamps, data.sampling_interval};
}
torch::Tensor encode_causal(cp::Loaded &live, const ow::Dataset &data, int64_t native_width) {
  std::vector<torch::Tensor> rows;
  torch::NoGradGuard guard;
  live.model->eval();
  for (int64_t first = 0; first < data.rows(); first += 8) {
    std::vector<int64_t> ids;
    for (int64_t i = first; i < std::min(first + 8, data.rows()); ++i) ids.push_back(i);
    const auto batch = ow::subset(data, ids);
    const auto encoded = live.model->encode_at(causal_raw(batch), batch.cutoffs, live.scaler);
    TORCH_CHECK(encoded.valid.all().item<bool>(), "[timeframe score] causal encode left a row invalid");
    rows.push_back(encoded.native.detach().to(torch::kCPU, torch::kFloat64));
  }
  const auto out = torch::cat(rows);
  TORCH_CHECK(out.size(0) == data.rows() && out.size(1) == native_width, "[timeframe score] causal width mismatch");
  return out;
}
torch::Tensor encode_raw(rfamily::Model &model, const rpb::FrozenScaler &scaler, const rpb::Config &config, const ow::Dataset &data) {
  std::vector<torch::Tensor> rows;
  torch::NoGradGuard guard;
  for (int64_t first = 0; first < data.rows(); first += 8) {
    std::vector<int64_t> ids;
    for (int64_t i = first; i < std::min(first + 8, data.rows()); ++i) ids.push_back(i);
    const auto batch = ow::subset(data, ids);
    const auto prepared = rpb::prepare_endpoint_history(batch.values, batch.observed, batch.channel_ids, batch.timestamps,
        batch.cutoffs, config);
    const auto encoded = rfamily::encode(model, scaler, prepared);
    TORCH_CHECK(encoded.sample_valid_mask.all().item<bool>(), "[timeframe score] raw encode left a row invalid");
    rows.push_back(encoded.z_contextual_global.detach().to(torch::kCPU, torch::kFloat64));
  }
  const auto out = torch::cat(rows);
  TORCH_CHECK(out.size(0) == data.rows() && out.size(1) == config.export_width, "[timeframe score] raw width mismatch");
  return out;
}
void same_ids(const ow::Dataset &features, const ow::Dataset &targets, const char *name) {
  TORCH_CHECK(features.rows() == targets.rows() && torch::equal(features.row_ids, targets.row_ids),
      "[timeframe score] feature and target row ids differ: ", name);
  TORCH_CHECK(features.role == targets.role, "[timeframe score] feature and target roles differ: ", name);
}
struct Fit {
  bool scored{false};
  double r2{0}, mae{0}, rmse{0}, coverage{0};
  int64_t train_eligible{0}, validation_eligible{0}, validation_rows{0};
};
Fit score_task(const torch::Tensor &train_x, const torch::Tensor &validation_x, const torch::Tensor &train_y,
    const torch::Tensor &validation_y, const torch::Tensor &train_valid, const torch::Tensor &validation_valid) {
  Fit fit;
  fit.validation_rows = validation_y.size(0);
  const auto train_index = train_valid.nonzero().flatten();
  const auto validation_index = validation_valid.nonzero().flatten();
  fit.train_eligible = train_index.numel();
  fit.validation_eligible = validation_index.numel();
  fit.coverage = double(fit.validation_eligible) / double(fit.validation_rows);
  if (fit.train_eligible < 2 || fit.validation_eligible < 2) return fit;
  const auto x = train_x.index_select(0, train_index);
  const auto y = train_y.index_select(0, train_index);
  const auto vx = validation_x.index_select(0, validation_index);
  const auto vy = validation_y.index_select(0, validation_index);
  if ((y - y.mean()).square().sum().item<double>() <= 0) return fit;
  const auto mean = x.mean(0);
  const auto scale = (x - mean).square().mean(0).sqrt().clamp_min(1e-8);
  const auto standardized = (x - mean) / scale;
  const auto validation_standardized = (vx - mean) / scale;
  const auto target_mean = y.mean();
  const auto target_scale = (y - target_mean).square().mean().sqrt().clamp_min(1e-8);
  const auto standardized_y = (y - target_mean) / target_scale;
  const auto feature_center = standardized.mean(0);
  const auto target_center = standardized_y.mean();
  const auto centered = standardized - feature_center;
  const auto weights = at::linalg_solve(
      centered.transpose(0, 1).matmul(centered) + torch::eye(standardized.size(1), standardized.options()),
      centered.transpose(0, 1).matmul((standardized_y - target_center).unsqueeze(1))).squeeze(1);
  const auto intercept = target_center - feature_center.dot(weights);
  const auto prediction = (validation_standardized.matmul(weights) + intercept) * target_scale + target_mean;
  const auto score = ev::regression_score(prediction, vy, torch::ones({vy.size(0)}, torch::kBool), torch::ones({vy.size(0)}, torch::kBool));
  TORCH_CHECK(score.supported && score.r2_supported, "[timeframe score] eligible task did not produce an R2");
  fit.scored = true;
  fit.r2 = score.r2;
  fit.mae = score.mae;
  fit.rmse = score.rmse;
  return fit;
}
}  // namespace

int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc == 19, "[timeframe score] nine flag/value pairs required");
    const std::set<std::string> allowed{"--family", "--settings", "--checkpoint", "--features-train", "--features-validation",
        "--targets-train", "--targets-validation", "--output", "--expected-updates"};
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) TORCH_CHECK(allowed.count(argv[i]) && args.emplace(argv[i], argv[i + 1]).second,
        "[timeframe score] unknown or duplicate option");
    for (const auto &key : allowed) TORCH_CHECK(args.count(key), "[timeframe score] missing ", key);
    const auto family = args.at("--family");
    TORCH_CHECK(family == "cpb" || family == "rpb", "[timeframe score] family must be cpb or rpb");
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    TORCH_CHECK(torch::cuda::is_available(), "[timeframe score] CUDA required");
    const fs::path out = args.at("--output");
    TORCH_CHECK(!fs::exists(out) && fs::is_directory(out.parent_path()), "[timeframe score] new output file required");
    const auto features_train = ow::read(args.at("--features-train"));
    const auto features_validation = ow::read(args.at("--features-validation"));
    const auto targets_train_data = ow::read(args.at("--targets-train"));
    const auto targets_validation_data = ow::read(args.at("--targets-validation"));
    TORCH_CHECK(features_train.role == "TRAIN" && features_validation.role == "VALIDATION", "[timeframe score] feature roles");
    same_ids(features_train, targets_train_data, "TRAIN");
    same_ids(features_validation, targets_validation_data, "VALIDATION");
    size_t used = 0;
    const auto expected = std::stoll(args.at("--expected-updates"), &used);
    TORCH_CHECK(used == args.at("--expected-updates").size() && expected >= 0, "[timeframe score] updates");
    torch::Tensor train_x, validation_x;
    int64_t width = 0;
    if (family == "cpb") {
      const auto config = cp::read_settings(args.at("--settings"));
      auto live = cp::load_checkpoint(args.at("--checkpoint"), config.level.device, &config);
      TORCH_CHECK(live.causal_updates == expected, "[timeframe score] causal update count");
      width = cp::dimensions(live.model->config()).native_width;
      train_x = encode_causal(live, features_train, width);
      validation_x = encode_causal(live, features_validation, width);
    } else {
      const auto config = rfamily::read_settings(args.at("--settings"));
      auto live = rfamily::load_inference(args.at("--checkpoint"), config.device, &config);
      TORCH_CHECK(live.training_updates == expected, "[timeframe score] raw update count");
      width = config.export_width;
      train_x = encode_raw(live.model, live.scaler, config, features_train);
      validation_x = encode_raw(live.model, live.scaler, config, features_validation);
    }
    TORCH_CHECK(torch::isfinite(train_x).all().item<bool>() && torch::isfinite(validation_x).all().item<bool>(),
        "[timeframe score] nonfinite features");
    const auto train_targets = ev::past_window_targets(targets_train_data.values.to(torch::kFloat32).to(torch::kFloat64),
        targets_train_data.observed);
    const auto validation_targets = ev::past_window_targets(
        targets_validation_data.values.to(torch::kFloat32).to(torch::kFloat64), targets_validation_data.observed);
    const auto names = ev::past_window_target_specs();
    std::ostringstream report;
    report << std::setprecision(17) << "{\"kind\":\"timeframe_completed_bar_readout_v1\",\"family\":" << quote(family)
           << ",\"updates\":" << expected << ",\"feature_width\":" << width
           << ",\"head\":\"per-task TRAIN ridge, eligible rows only\",\"promotes_reference\":false,\"tasks\":[";
    for (int64_t task = 0; task < 4; ++task) {
      const auto fit = score_task(train_x, validation_x, train_targets.values.select(1, task),
          validation_targets.values.select(1, task), train_targets.valid.select(1, task),
          validation_targets.valid.select(1, task));
      if (task) report << ',';
      report << "{\"task\":" << quote(names[static_cast<size_t>(task)].name) << ",\"scored\":" << (fit.scored ? "true" : "false")
             << ",\"r2\":";
      if (fit.scored) report << fit.r2; else report << "null";
      report << ",\"mae\":";
      if (fit.scored) report << fit.mae; else report << "null";
      report << ",\"rmse\":";
      if (fit.scored) report << fit.rmse; else report << "null";
      report << ",\"coverage\":" << fit.coverage << ",\"train_eligible\":" << fit.train_eligible
             << ",\"validation_eligible\":" << fit.validation_eligible << ",\"validation_rows\":" << fit.validation_rows << '}';
    }
    report << "]}";
    std::ofstream file(out);
    TORCH_CHECK(bool(file), "[timeframe score] open output");
    file << report.str() << '\n';
    file.close();
    TORCH_CHECK(bool(file), "[timeframe score] write output");
    std::cout << report.str() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
